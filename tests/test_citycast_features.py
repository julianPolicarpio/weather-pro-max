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


def test_partial_nbs_run_is_a_feature_but_anchor_uses_latest_full_run(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:   # a later NBS run that covers only 2 points of the day
            db.upsert(c, "mos_runs", [_mos("NBS", "2024-01-04T19:00:00+00:00", 40.0, n_points=2)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 5, 2, 0, tzinfo=timezone.utc))
        assert r["f_NBS"] == 40.0 and r["npts_NBS"] == 2.0      # newest published run is reported
        assert r["anchor"] == 45.0                                # mean(full NBS 46, MEX 44)


def test_nbe_only_day_has_no_anchor(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "mos_runs", [_mos("NBE", "2024-01-04T12:00:00+00:00", 44.0, n_points=8)])
        assert _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE) is None


def test_spread_excludes_partial_runs(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "mos_runs", [_mos("NBS", "2024-01-04T19:00:00+00:00", 40.0, n_points=2)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 5, 2, 0, tzinfo=timezone.utc))
        assert r["spread_mos"] == 2.0


def test_non_utc_t_gives_identical_row(tmp_path):
    from zoneinfo import ZoneInfo
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        ctx = _ctx(c, tmp_path)
        a = ctx.row("KNYC", "high", D, T_SAME)
        b = ctx.row("KNYC", "high", D, T_SAME.astimezone(ZoneInfo("America/New_York")))
        assert a["asof_utc"] == b["asof_utc"]
        assert all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in FEATURE_COLS)


def test_null_mos_value_is_skipped(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "mos_runs", [_mos("MEX", "2024-01-04T13:00:00+00:00", None)])
        assert _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE)["f_MEX"] == 44.0


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
            # a MOS run 30 min before t (inside MEX's 5 h publication lag), an obs after t,
            # and a CLI report from the target's own year
            db.upsert(c, "mos_runs", [_mos("MEX", (T_SAME - timedelta(minutes=30)).isoformat(), 99.0)])
            db.upsert(c, "obs", [_obs("2024-01-05T20:51:00+00:00", 70.0, six_max_c=25.0)])
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2024-01-03", "high_f": 90,
                                        "low_f": 80, "final": 1, "issued_utc": None, "raw": "{}",
                                        "fetched_at": F}])
        after = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        same = lambda a, b: (math.isnan(a) and math.isnan(b)) or a == b   # noqa: E731
        assert all(same(before[k], after[k]) for k in FEATURE_COLS)


def test_boundary_mex_exactly_at_lag_is_used(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "mos_runs", [_mos("MEX", (T_SAME - timedelta(hours=5)).isoformat(), 50.0)])
        assert _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)["f_MEX"] == 50.0


def test_boundary_nbs_just_inside_lag_is_not_used(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "mos_runs", [_mos("NBS", (T_SAME - timedelta(hours=2, minutes=59)).isoformat(),
                                           99.0, n_points=8)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        assert r["f_NBS"] == 46.0


def test_pre_day_row_and_prev_obs_ignore_post_t_data(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:   # an obs on D-1 (climate day ends 05:00Z Jan 5) before t
            db.upsert(c, "obs", [_obs("2024-01-04T20:51:00+00:00", 30.0)])
        before = _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE)
        assert not math.isnan(before["prev_obs"])
        with c:   # post-t: D-1 obs, D obs, and a MEX run
            db.upsert(c, "obs", [_obs("2024-01-05T02:51:00+00:00", 60.0, six_max_c=20.0),
                                 _obs("2024-01-05T18:51:00+00:00", 80.0)])
            db.upsert(c, "mos_runs", [_mos("MEX", "2024-01-04T22:00:00+00:00", 99.0)])
        after = _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE)
        assert all((math.isnan(before[k]) and math.isnan(after[k])) or before[k] == after[k]
                   for k in FEATURE_COLS)


def test_six_hour_group_ending_after_t_is_ignored(tmp_path):
    t = datetime(2024, 1, 5, 17, 55, tzinfo=timezone.utc)   # 17:51 report valid, its 12-18Z window not over
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        before = _ctx(c, tmp_path).row("KNYC", "high", D, t)
        assert before["run_ext"] < 46.0                    # the 7.8 C group (46.04 F) is not admitted
        with c:
            db.upsert(c, "obs", [_obs("2024-01-05T17:51:00+00:00", 45.0, six_max_c=25.0)])
        assert _ctx(c, tmp_path).row("KNYC", "high", D, t)["run_ext"] == before["run_ext"]


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


def test_last_obs_excludes_a_reading_valid_exactly_at_t(tmp_path):
    """A METAR valid at t is published minutes after t (re-review mutation 3)."""
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "obs", [_obs(T_SAME.isoformat(), 99.0)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        assert r["last_tmpf"] == 45.0


def test_anchor_uses_full_run_published_exactly_at_lag(tmp_path):
    """mos_full's boundary is inclusive like mos's (re-review mutation 1b)."""
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "mos_runs", [_mos("MEX", (T_SAME - timedelta(hours=5)).isoformat(), 51.0)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        assert r is not None and r["anchor"] == 51.0
