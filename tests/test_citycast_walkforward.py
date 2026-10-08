import json
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd

from citycast import db
from citycast.features import FEATURE_COLS
from citycast.walkforward import choose_k, quarters, run


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


def _frame():
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    rows = []
    for st in ("KNYC", "KLAX"):
        for d in days:
            y = round(60 + rng.normal(0, 3))
            for hour in ("06", "12", "18"):          # 3 as-of points/day: 1,086 rows before 2022-Q3
                r = {c: rng.normal() for c in FEATURE_COLS}
                r.update(station=st, metric="high", event_date=d.strftime("%Y-%m-%d"),
                         asof_utc=f"{d.strftime('%Y-%m-%d')}T{hour}:00:00+00:00",
                         anchor=60.0, f_MEX=60.0, run_ext=float("nan"), y=y)
                rows.append(r)
    return pd.DataFrame(rows)


def test_run_writes_out_of_sample_rows_for_all_variants(tmp_path):
    df = _frame()
    with closing(db.connect(tmp_path / "h.db")) as c:
        n = run(c, df, "high", ("KLAX", "KNYC"), lambda: "2026-10-07T00:00:00+00:00",
                today=date(2023, 1, 1), progress=lambda m: None)
        got = c.execute("SELECT variant, COUNT(*), MIN(event_date) FROM wf_predictions GROUP BY variant").fetchall()
        assert n > 0
        assert {g[0] for g in got} == {"shared", "city", "mosnorm"}
        assert all(g[2] >= "2022-07-01" for g in got)          # never predicts its training period
        pmf = json.loads(c.execute("SELECT pmf_json FROM wf_predictions LIMIT 1").fetchone()[0])
        assert abs(sum(pmf.values()) - 1) < 1e-6
