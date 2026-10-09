import json
from contextlib import closing
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from citycast import db, walkforward
from citycast.features import FEATURE_COLS
from citycast.qmodel import PARAMS, QuantileModel
from citycast.walkforward import EMBARGO_D, choose_k, quarters, run


def test_quarters_only_complete_ones():
    q = quarters(date(2022, 7, 1), date(2023, 1, 15))
    assert q == [("2022Q3", date(2022, 7, 1), date(2022, 10, 1)),
                 ("2022Q4", date(2022, 10, 1), date(2023, 1, 1))]


def test_choose_k_widens_overconfident_forecasts():
    rng = np.random.default_rng(1)
    prev = []
    for i in range(400):
        y = 50 + rng.normal(0, 4)
        q = {a: 50 + v for a, v in zip((0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
                                        np.linspace(-2, 2, 13))}
        prev.append((q, float("nan"), round(y)))
    assert choose_k(prev, "high") > 1.0
    assert choose_k([], "high") == 1.0


def _frame(stations=("KNYC", "KLAX"), first="2022-01-01", last="2022-12-31", nan_mex_day=None):
    rng = np.random.default_rng(0)
    days = pd.date_range(first, last, freq="D")
    rows = []
    for st in stations:
        for d in days:
            y = round(60 + rng.normal(0, 3))
            for hour in ("06", "12", "18"):          # 3 as-of points/day: 1,080 train rows before the 2022-Q3 cutoff
                r = {c: rng.normal() for c in FEATURE_COLS}
                r.update(station=st, metric="high", event_date=d.strftime("%Y-%m-%d"),
                         asof_utc=f"{d.strftime('%Y-%m-%d')}T{hour}:00:00+00:00",
                         anchor=60.0, f_MEX=float("nan") if d.day == nan_mex_day else 60.0,
                         run_ext=float("nan"), y=y)
                rows.append(r)
    return pd.DataFrame(rows)


def _quarter_of(event_date: str) -> str:
    return f"{event_date[:4]}Q{(int(event_date[5:7]) - 1) // 3 + 1}"


NOW = lambda: "2026-10-07T00:00:00+00:00"


def test_run_writes_out_of_sample_rows_for_all_variants(tmp_path, monkeypatch):
    monkeypatch.setitem(PARAMS, "n_estimators", 20)         # structure under test, not accuracy
    df = _frame(nan_mex_day=5)
    with closing(db.connect(tmp_path / "h.db")) as c:
        n = run(c, df, "high", ("KLAX", "KNYC"), NOW, today=date(2023, 1, 1), progress=lambda m: None)
        got = c.execute("SELECT variant, COUNT(*), MIN(event_date) FROM wf_predictions GROUP BY variant").fetchall()
        assert n > 0
        assert {g[0] for g in got} == {"shared", "city", "mosnorm"}
        assert all(g[2] >= "2022-07-01" for g in got)          # only test quarters from the first fold are predicted
        stored = c.execute("SELECT variant, station, event_date, asof_utc, fold FROM wf_predictions").fetchall()
        assert n == len(stored)
        assert all(fold == _quarter_of(ev) for _, _, ev, _, fold in stored)
        keys = {v: {(s, e, a) for vv, s, e, a, _ in stored if vv == v} for v in ("shared", "mosnorm")}
        nan_keys = {(r.station, r.event_date, r.asof_utc) for r in df[df["f_MEX"].isna()].itertuples()}
        assert nan_keys & keys["shared"]                       # the NaN rows are really in the test windows
        assert keys["mosnorm"] == keys["shared"] - nan_keys
        pmf = json.loads(c.execute("SELECT pmf_json FROM wf_predictions LIMIT 1").fetchone()[0])
        assert abs(sum(pmf.values()) - 1) < 1e-6


class _SpyTruth(dict):
    """Records every event_date _prev looks up: exactly the dates its SQL let into the k history."""
    looked: list

    def get(self, key, default=None):
        self.looked.append(key[1])
        return super().get(key, default)


def _spy_walkforward(monkeypatch):
    fits, prevs, state = [], [], {}
    orig_fit, orig_prev = QuantileModel.fit, walkforward._prev

    def fit(self, train, ref):
        state["start"] = ref
        fits.append((ref, train["event_date"].max()))
        return orig_fit(self, train, ref)

    def prev(conn, station, metric, variant, before, truth):
        spy = _SpyTruth(truth)
        spy.looked = []
        out = orig_prev(conn, station, metric, variant, before, spy)
        prevs.append((state["start"], spy.looked, len(out)))
        return out

    monkeypatch.setattr(QuantileModel, "fit", fit)
    monkeypatch.setattr(walkforward, "_prev", prev)
    monkeypatch.setitem(PARAMS, "n_estimators", 5)
    return fits, prevs


def _two_year_frame():
    return _frame(("KNYC",), "2021-01-01", "2022-12-31")


def _assert_no_lookahead(fits, prevs):
    assert fits
    for start, newest in fits:                              # training stops a day before the fold
        assert newest < (start - timedelta(days=EMBARGO_D)).isoformat()
    for start, dates, _ in prevs:                           # k history stops a day before the fold
        assert all(d < (start - timedelta(days=EMBARGO_D)).isoformat() for d in dates)
    assert any(n > 0 for *_, n in prevs)                    # not vacuous: a later quarter had >=180 days of history


def test_training_and_k_history_stop_before_the_fold(tmp_path, monkeypatch):
    fits, prevs = _spy_walkforward(monkeypatch)
    msgs = []
    with closing(db.connect(tmp_path / "h.db")) as c:
        run(c, _two_year_frame(), "high", ("KNYC",), NOW, train_start=date(2021, 1, 1),
            first_fold=date(2022, 1, 1), today=date(2023, 4, 1), progress=msgs.append)
    _assert_no_lookahead(fits, prevs)
    assert any("2023Q1: skipped (no test rows)" in m for m in msgs)


def test_rerun_is_fresh_and_keeps_the_bounds(tmp_path, monkeypatch):
    fits, prevs = _spy_walkforward(monkeypatch)
    df, kw = _two_year_frame(), dict(train_start=date(2021, 1, 1), today=date(2023, 1, 1), progress=lambda m: None)

    def snapshot(c):
        return c.execute("SELECT station, event_date, asof_utc, variant, k FROM wf_predictions "
                         "ORDER BY 1, 2, 3, 4").fetchall()
    with closing(db.connect(tmp_path / "a.db")) as c:
        n1 = run(c, df, "high", ("KNYC",), NOW, first_fold=date(2022, 1, 1), **kw)
        first = snapshot(c)
        fits.clear()
        prevs.clear()
        run(c, df, "high", ("KNYC",), NOW, first_fold=date(2022, 1, 1), **kw)
        assert len(snapshot(c)) == len(first) == n1         # same config, same rows
        _assert_no_lookahead(fits, prevs)
        run(c, df, "high", ("KNYC",), NOW, first_fold=date(2022, 4, 1), **kw)   # fewer folds: 2022Q1 rows must go
        shortened = snapshot(c)
    with closing(db.connect(tmp_path / "b.db")) as c:
        run(c, df, "high", ("KNYC",), NOW, first_fold=date(2022, 4, 1), **kw)
        assert shortened == snapshot(c)
    assert len(shortened) < len(first)


def test_inconsistent_y_is_rejected(tmp_path):
    df = _frame(("KNYC",), "2022-01-01", "2022-01-03")
    df.loc[df.index[0], "y"] += 1                           # same (station, event_date), two different y
    with closing(db.connect(tmp_path / "h.db")) as c:
        with pytest.raises(ValueError, match="inconsistent y"):
            run(c, df, "high", ("KNYC",), NOW, today=date(2023, 1, 1), progress=lambda m: None)


def _wf_run_row(c, metric="high"):
    return c.execute("SELECT fingerprint, dataset_fingerprint, started_at, finished_at, rows FROM wf_runs "
                     "WHERE metric=?", (metric,)).fetchone()


def test_run_records_a_finished_run_tied_to_code_and_dataset(tmp_path, monkeypatch):
    from citycast.fingerprint import code_fingerprint
    monkeypatch.setitem(PARAMS, "n_estimators", 5)
    df = _frame()
    df.attrs["fingerprint"] = "dataset-fp"
    with closing(db.connect(tmp_path / "h.db")) as c:
        n = run(c, df, "high", ("KLAX", "KNYC"), NOW, today=date(2023, 1, 1), progress=lambda m: None)
        assert n > 0
        assert tuple(_wf_run_row(c)) == (code_fingerprint(), "dataset-fp", NOW(), NOW(), n)
        assert c.execute("SELECT COUNT(*) FROM wf_predictions WHERE metric='high'").fetchone()[0] == n


def test_crash_mid_run_leaves_the_run_unfinished(tmp_path, monkeypatch):
    monkeypatch.setitem(PARAMS, "n_estimators", 5)
    df, kw = _frame(), dict(today=date(2023, 1, 1))
    with closing(db.connect(tmp_path / "h.db")) as c:
        run(c, df, "high", ("KLAX", "KNYC"), NOW, progress=lambda m: None, **kw)
        assert _wf_run_row(c)[3] is not None                     # a finished run ...

        def crash(msg):
            if "predictions" in msg:                             # after the first fold is committed
                raise RuntimeError("killed")
        with pytest.raises(RuntimeError, match="killed"):
            run(c, df, "high", ("KLAX", "KNYC"), NOW, progress=crash, **kw)
        row = _wf_run_row(c)                                     # ... is reset by the next run's start
        assert row[3] is None and row[4] is None
        assert row[1] is None                                    # this frame carries no dataset fingerprint
        assert c.execute("SELECT COUNT(*) FROM wf_predictions").fetchone()[0] > 0


@pytest.mark.parametrize("metric, bound, dead", [("high", 61.4, lambda t: t < 61),    # rounds to 61
                                                 ("low", 58.6, lambda t: t > 59)])    # rounds to 59
def test_mosnorm_is_truncated_at_the_running_extremum_on_same_day_rows(tmp_path, monkeypatch, metric, bound, dead):
    from citycast.metrics import normal_pmf
    monkeypatch.setitem(PARAMS, "n_estimators", 5)
    df = _frame().assign(metric=metric)
    same_day = df["asof_utc"].str.endswith("T18:00:00+00:00")
    df.loc[same_day, "run_ext"] = bound
    with closing(db.connect(tmp_path / "h.db")) as c:
        run(c, df, metric, ("KLAX", "KNYC"), NOW, today=date(2023, 1, 1), progress=lambda m: None)
        rows = c.execute("SELECT asof_utc, run_ext, pmf_json FROM wf_predictions WHERE variant='mosnorm'").fetchall()
    assert {a[-14:-6] for a, _, _ in rows} == {"06:00:00", "12:00:00", "18:00:00"}
    assert any(dead(t) for t in normal_pmf(60.0))                # the bound really cuts the baseline
    for asof, run_ext, pj in rows:
        pmf = {int(t): p for t, p in json.loads(pj).items()}
        if asof.endswith("T18:00:00+00:00"):                    # same-day: no mass the bound has ruled out
            assert run_ext == bound and not any(dead(t) for t in pmf)
            assert abs(sum(pmf.values()) - 1) < 1e-9
        else:                                                   # pre-day: the plain MOS normal
            assert run_ext is None and pmf == normal_pmf(60.0)
