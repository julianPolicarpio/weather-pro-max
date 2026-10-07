import numpy as np
import pandas as pd

from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.dayahead import DayAheadModel


def _frame(n=600, seed=5):
    """Synthetic frame whose target reaches well beyond the dense mass, so a
    plain GBM (bounded by training targets in its leaves) under-predicts the
    hot tail while a residual-target model can follow the anchor up."""
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    y[-20:] = np.linspace(92, 104, 20)            # a hot tail beyond the mass
    df = pd.DataFrame({c: np.nan for c in FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d")
    df["fc_ecmwf"] = y + rng.normal(0, 1, n)
    df["fc_gfs"] = y + rng.normal(0, 1, n)
    df["doy_sin"] = np.sin(2 * np.pi * np.arange(n) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * np.arange(n) / 365)
    df["year_index"] = 24.0
    df["heat_alert"] = 0.0
    df["y"] = np.round(y)
    df["fc_anchor"] = df["fc_gfs"]                # the level comes from here
    return df


def test_anchor_col_defaults_off_is_unchanged():
    df = _frame()
    a = DayAheadModel().fit(df.iloc[:500])
    b = DayAheadModel(anchor_col=None).fit(df.iloc[:500])
    qa = a.predict_quantiles(df.iloc[500:505])
    qb = b.predict_quantiles(df.iloc[500:505])
    assert qa == qb                      # anchor_col=None is the current model


def test_residual_target_tracks_the_anchor_into_the_hot_tail():
    df = _frame()
    train, test = df.iloc[:560], df.iloc[580:]   # test rows are the hot tail
    plain = DayAheadModel().fit(train)
    resid = DayAheadModel(anchor_col="fc_anchor").fit(train)
    yy = test["y"].to_numpy(dtype=float)
    mae_plain = np.mean([abs(q[0.5] - t) for q, t in
                         zip(plain.predict_quantiles(test), yy)])
    mae_resid = np.mean([abs(q[0.5] - t) for q, t in
                         zip(resid.predict_quantiles(test), yy)])
    # the residual model must follow the anchor upward where the GBM cannot
    assert mae_resid < mae_plain
    for q in resid.predict_quantiles(test):
        vals = [q[l] for l in LEVELS]
        assert vals == sorted(vals)      # still a valid (ordered) quantile set


def test_hybrid_keeps_bulk_and_lifts_tail():
    """Hybrid must equal the direct model below LO and follow the residual
    model above HI, so the dense regime is untouched and the tail is lifted."""
    from highcast.models.dayahead import HybridDayAheadModel
    df = _frame()
    train = df.iloc[:560]
    direct = DayAheadModel().fit(train)
    hyb = HybridDayAheadModel().fit(train)
    cool = df.iloc[500:520]          # predictions well below LO=82
    qd = direct.predict_quantiles(cool)
    qh = hyb.predict_quantiles(cool)
    for a, b in zip(qd, qh):
        if a[0.5] < 82.0:
            assert b == a            # untouched in the dense regime
    hot = df.iloc[580:]              # the hot tail
    yy = hot["y"].to_numpy(dtype=float)
    mae_d = np.mean([abs(q[0.5] - t) for q, t in zip(direct.predict_quantiles(hot), yy)])
    mae_h = np.mean([abs(q[0.5] - t) for q, t in zip(hyb.predict_quantiles(hot), yy)])
    assert mae_h < mae_d             # tail is lifted toward the anchor


def test_hybrid_falls_back_to_direct_without_anchor():
    from highcast.models.dayahead import HybridDayAheadModel
    df = _frame().drop(columns=["fc_anchor"])
    hyb = HybridDayAheadModel().fit(df.iloc[:500])
    direct = DayAheadModel().fit(df.iloc[:500])
    assert hyb.predict_quantiles(df.iloc[500:505]) == direct.predict_quantiles(df.iloc[500:505])


def test_hybrid_skips_the_second_fit_when_data_cannot_reach_the_blend_region():
    """LOW-style data (never near 82F) must not fit a residual model at all: a
    LOW blend is actively harmful (measured dMAE +4.72, +5..+9F cold-band bias)
    and the second fit would be pure waste."""
    from highcast.models.dayahead import HybridDayAheadModel
    df = _frame()
    df["y"] = df["y"] - 25.0          # push everything into LOW territory
    hyb = HybridDayAheadModel().fit(df.iloc[:500])
    assert hyb._ok is False
    direct = DayAheadModel().fit(df.iloc[:500])
    assert hyb.predict_quantiles(df.iloc[500:505]) == direct.predict_quantiles(df.iloc[500:505])
