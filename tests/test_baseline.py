import dataclasses
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from highcast import db
from highcast.config import load_config
from highcast.features import build_features, uses_bias_correction
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend, LEVELS

def _frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.linspace(0, 4, n)) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "target_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "fc_ecmwf": y + rng.normal(0, 1.0, n),      # good model
        "fc_gfs": y + rng.normal(0, 3.0, n),        # worse model
        "fc_nws": np.nan,                            # entirely missing model
        "doy_sin": np.sin(2 * np.pi * np.arange(n) / 365),
        "doy_cos": np.cos(2 * np.pi * np.arange(n) / 365),
        "y": y,
    })

def test_weights_favor_better_model_and_handle_missing():
    m = BaselineBlend().fit(_frame())
    assert m.weights["fc_ecmwf"] > m.weights["fc_gfs"]
    assert "fc_nws" not in m.weights

def test_quantiles_are_monotone_and_centered():
    df = _frame()
    m = BaselineBlend().fit(df)
    qs = m.predict_quantiles(df.tail(5))
    for q in qs:
        vals = [q[l] for l in LEVELS]
        assert vals == sorted(vals)
        assert q[0.05] < q[0.50] < q[0.95]

def test_point_tracks_truth_reasonably():
    df = _frame()
    m = BaselineBlend().fit(df)
    mae = np.mean(np.abs(m.point(df) - df["y"]))
    assert mae < 2.0

def test_no_forecast_day_falls_back_to_climatology_not_nan():
    """A day with every fc_* NaN must yield finite quantiles (seasonal
    climatology), not NaN -- otherwise bucket_probs crashes downstream."""
    df = _frame()
    m = BaselineBlend().fit(df)
    gap = df.tail(1).copy()
    for c in [c for c in gap.columns if c.startswith("fc_")]:
        gap[c] = np.nan
    q = m.predict_quantiles(gap)[0]
    vals = [q[l] for l in LEVELS]
    assert all(np.isfinite(v) for v in vals)     # no NaN
    assert vals == sorted(vals)
    assert 60 < q[0.50] < 95                       # near the seasonal mean, sane


# ------------------------------------------------- only INDEPENDENT forecasts

CFG = load_config()
# mos_mex stops on day 55 so the `fc_anchor` coalesce is forced onto its SECOND
# branch (gfs) for the tail of the fixture -- otherwise a chain that never
# switches would only ever expose its first branch to the perturbation.
_MODELS = ("ecmwf", "gfs", "icon", "mos_mex")
_MOS_STOPS = 55
_DAYS = 80


def _seed(tmp_path, bumped: str | None = None, bump: float = 7.0):
    """A DB with `_MODELS` forecasting every day; `bumped` gets +bump degrees."""
    cfg = dataclasses.replace(CFG, db_path=tmp_path / "t.db")
    conn = db.connect(cfg.db_path)
    start = date(2026, 1, 1)
    for i in range(_DAYS):
        d = start + timedelta(days=i)
        hi, lo = 78.0 + 6 * math.sin(i / 9.0), 58.0 + 4 * math.cos(i / 7.0)
        db.upsert(conn, "cli_reports", [{
            "station": cfg.station.icao, "date": d.isoformat(), "high_f": hi,
            "low_f": lo, "raw": "{}", "fetched_at": "t"}])
        for j, m in enumerate(_MODELS):
            if m == "mos_mex" and i >= _MOS_STOPS:
                continue
            off = (j - 1.5) * 1.7 + (bump if m == bumped else 0.0)
            run = (d - timedelta(days=1)).isoformat()
            for key, base in (("high", hi), ("low", lo)):
                db.upsert(conn, "forecasts", [{
                    "station": cfg.station.icao, "target_date": d.isoformat(),
                    "model": m, "run_date": run, "max_temp_f": base + off,
                    "fetched_at": f"{run}T00:00:00+00:00", "metric": key}])
    return cfg, conn, [(start + timedelta(days=i)).isoformat()
                       for i in range(_DAYS)]


@pytest.mark.parametrize("metric_key", ["high", "low"])
@pytest.mark.parametrize("bumped", _MODELS)
def test_no_derived_column_enters_the_blend(tmp_path, metric_key, bumped):
    """INVARIANT: every blend input is an INDEPENDENT forecast -- no blend input
    may be a function of other blend inputs.

    `_blend_cols` already guards the fc_/fcc_ variant pair ("blending both
    `fc_gfs` and `fcc_gfs` would enter the same model twice and double-weight
    it"). The same failure arrives through a second door: `build_features`
    appends DERIVED `fc_*` columns, which the prefix scan cannot tell apart from
    a real model. `fc_anchor` (coalesce mos_mex -> gfs -> fc_mean) was one, and
    it took 8-17% of the blend weight on all four live markets -- re-entering
    whichever model the chain had selected that day, and on its third branch
    re-admitting `fc_mean`, which `_EXCLUDE` names explicitly.

    DETECTION IS BY PERTURBATION, NOT BY A NAME BLACKLIST. Bump one model's
    forecasts, rebuild the frame through the real pipeline, and every column
    that MOVED is a function of that model. Outside that model's own
    {fc_X, fcc_X} variant pair (which `_blend_cols` already collapses to one),
    a column that moves is derived and must not be a blend input. A future
    derived `fc_*` column is therefore caught with no edit to this test.
    """
    spec = get_metric(CFG, metric_key)
    cfg_a, conn_a, dates = _seed(tmp_path / "a")
    _cfg_b, conn_b, _ = _seed(tmp_path / "b", bumped=bumped)
    fa = build_features(conn_a, cfg_a, spec, dates)
    fb = build_features(conn_b, _cfg_b, spec, dates)

    moved = {c for c in fa.columns if not fa[c].equals(fb[c])}
    assert f"fc_{bumped}" in moved, "the perturbation did not reach the frame"

    weights = BaselineBlend(
        prefer_corrected=uses_bias_correction(metric_key)).fit(fa).weights
    derived = (moved - {f"fc_{bumped}", f"fcc_{bumped}"}) & set(weights)
    assert not derived, (
        f"{sorted(derived)} move when fc_{bumped} moves, so they are functions "
        f"of it, yet they enter the blend as independent models")


def test_the_blend_never_reads_an_aggregate_of_its_own_inputs():
    """The named half of the invariant above, pinned directly: `fc_mean` (and
    its corrected twin) is the mean OF the blend inputs and `fc_anchor` falls
    back to it, so neither may be weighted alongside them. Cheap and explicit,
    because the perturbation test can only see columns the fixture populates."""
    from highcast.models.baseline import _EXCLUDE
    assert {"fc_mean", "fcc_mean", "fc_spread", "fc_anchor"} <= _EXCLUDE
