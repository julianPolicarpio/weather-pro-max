# tests/test_citycast_features.py
import math
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

from citycast import db
from citycast.features import FEATURE_COLS, FeatureContext
from citycast.stations import Station

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"
D = date(2024, 1, 5)
T_PRE = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)      # 18:00 EST on D-1
T_SAME = datetime(2024, 1, 5, 19, 0, tzinfo=timezone.utc)     # 14:00 EST on D


def _mos(model, runtime, value, n_points=1):
    return {"station": "KNYC", "model": model, "runtime_utc": runtime, "target_date": D.isoformat(),
            "metric": "high", "value_f": value, "n_points": n_points, "fetched_at": F}


def _obs(valid, tmpf, six_max_c=None):
    return {"station": "KNYC", "valid_utc": valid, "source": "metar", "tmpf": tmpf,
            "tmpf_precise": tmpf, "dwpf": 20.0, "six_max_c": six_max_c, "raw": "", "fetched_at": F}


def _seed(c):
    clim = [{"station": "KNYC", "date": f"{y}-01-0{d}", "high_f": 38 + d, "low_f": 25, "final": 1,
             "issued_utc": None, "raw": "{}", "fetched_at": F} for y in (2021, 2022, 2023) for d in range(1, 9)]
    with c:
        db.upsert(c, "cli_daily", clim)
        db.upsert(c, "mos_runs", [_mos("MEX", "2024-01-04T12:00:00+00:00", 44.0),
                                  _mos("NBS", "2024-01-04T13:00:00+00:00", 46.0, n_points=8)])
        db.upsert(c, "obs", [_obs("2024-01-05T14:51:00+00:00", 41.0),
                             _obs("2024-01-05T17:51:00+00:00", 45.0, six_max_c=7.8)])


def _ctx(c, tmp_path):
    return FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")


def test_pre_day_row(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE)
        assert list(r)[:4] == ["station", "metric", "event_date", "asof_utc"]
        assert set(FEATURE_COLS) <= set(r)
        assert r["f_MEX"] == 44.0 and r["f_NBS"] == 46.0 and math.isnan(r["f_GFS"])
        assert r["anchor"] == 45.0 and r["spread_mos"] == 2.0
        assert r["age_MEX"] == 11.0
        assert math.isnan(r["run_ext"]) and r["lead_h"] == 30.0
        assert r["clim_mean"] > 0 and r["local_hour"] == 18


def test_partial_nbs_run_is_a_feature_but_not_the_anchor(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:   # a later NBS run that covers only 2 points of the day
            db.upsert(c, "mos_runs", [_mos("NBS", "2024-01-04T19:00:00+00:00", 40.0, n_points=2)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 5, 2, 0, tzinfo=timezone.utc))
        assert r["f_NBS"] == 40.0 and r["npts_NBS"] == 2.0      # newest published run is reported
        assert r["anchor"] == 44.0                                # ...but only full-coverage runs anchor


def test_publication_lag_is_per_model(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        # 16:30Z: NBS (13Z + 3 h) is published, MEX (12Z + 5 h = 17Z) is not
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 4, 16, 30, tzinfo=timezone.utc))
        assert r["f_NBS"] == 46.0 and math.isnan(r["f_MEX"]) and r["anchor"] == 46.0


def test_same_day_row_uses_visible_running_extremum(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        assert r["run_ext"] >= 45.0                       # 6-hour group 7.8 C = 46.04 F counts
        assert r["run_gap"] == r["run_ext"] - r["anchor"] and r["last_tmpf"] == 45.0


def test_no_anchor_returns_none(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        assert _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE) is None


def test_leakage_future_data_changes_nothing(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        before = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        with c:
            # a MOS run 30 min before t (inside the 1 h publication lag), an obs after t,
            # and a CLI report from the target's own year
            db.upsert(c, "mos_runs", [_mos("MEX", (T_SAME - timedelta(minutes=30)).isoformat(), 99.0)])
            db.upsert(c, "obs", [_obs("2024-01-05T20:51:00+00:00", 70.0, six_max_c=25.0)])
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2024-01-03", "high_f": 90,
                                        "low_f": 80, "final": 1, "issued_utc": None, "raw": "{}",
                                        "fetched_at": F}])
        after = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        same = lambda a, b: (math.isnan(a) and math.isnan(b)) or a == b   # noqa: E731
        assert all(same(before[k], after[k]) for k in FEATURE_COLS)


def test_truth_uses_final_reports_only(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "cli_daily", [
                {"station": "KNYC", "date": "2024-01-05", "high_f": 47, "low_f": 30, "final": 1,
                 "issued_utc": None, "raw": "{}", "fetched_at": F},
                {"station": "KNYC", "date": "2024-01-06", "high_f": 50, "low_f": 31, "final": 0,
                 "issued_utc": None, "raw": "{}", "fetched_at": F}])
        ctx = _ctx(c, tmp_path)
        assert ctx.truth("KNYC", "high", date(2024, 1, 5)) == 47.0
        assert math.isnan(ctx.truth("KNYC", "high", date(2024, 1, 6)))
