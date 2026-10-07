import numpy as np
import pandas as pd
from highcast.models.intraday import INTRADAY_FEATURES, IntradayModel
from highcast.models.baseline import LEVELS


def _frame(n=800, seed=4):
    rng = np.random.default_rng(seed)
    y = np.round(75 + 8 * np.sin(np.arange(n)) + rng.normal(0, 2, n))
    hour = rng.choice([8, 10, 12, 14, 16, 18], n)
    # visible running max approaches y as the day progresses
    frac = (hour - 6) / 12.0
    run_hi = y - np.maximum(0, (1 - frac) * rng.uniform(2, 10, n))
    df = pd.DataFrame({c: np.nan for c in INTRADAY_FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    df["hour"] = hour.astype(float)
    df["run_lo"] = run_hi - 1.0
    df["run_hi"] = run_hi
    df["run_width"] = 1.0
    df["base_p50"] = y + rng.normal(0, 2, n)
    df["base_spread"] = 4.0
    df["heat_alert"] = 0.0
    df["y"] = y
    return df


def test_afternoon_is_sharper_than_morning():
    df = _frame()
    m = IntradayModel().fit(df.iloc[:600])
    test = df.iloc[600:]
    def spread(sub):
        qs = m.predict_quantiles(sub)
        return np.mean([q[0.9] - q[0.1] for q in qs])
    assert spread(test[test["hour"] >= 16]) < spread(test[test["hour"] <= 10])


def test_quantiles_floored_at_running_max():
    df = _frame()
    m = IntradayModel().fit(df.iloc[:600])
    sub = df.iloc[600:650]
    qs = m.predict_quantiles(sub)
    for q, lo in zip(qs, sub["run_lo"]):
        assert q[0.05] >= lo - 1e-9


def _low_frame(n=800, seed=4):
    rng = np.random.default_rng(seed)
    y = np.round(45 + 8 * np.sin(np.arange(n)) + rng.normal(0, 2, n))
    hour = rng.choice([8, 10, 12, 14, 16, 18], n)
    # visible running MIN upper bound approaches y from ABOVE as the day
    # progresses (mirror of the high frame, where the running max rises to y).
    frac = (hour - 6) / 12.0
    run_hi = y + np.maximum(0, (1 - frac) * rng.uniform(2, 10, n))
    df = pd.DataFrame({c: np.nan for c in INTRADAY_FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    df["hour"] = hour.astype(float)
    df["run_lo"] = run_hi - 1.0
    df["run_hi"] = run_hi
    df["run_width"] = 1.0
    df["base_p50"] = y + rng.normal(0, 2, n)
    df["base_spread"] = 4.0
    df["heat_alert"] = 0.0
    df["y"] = y
    return df


def test_quantiles_capped_at_running_min():
    df = _low_frame()
    m = IntradayModel(extremum="min").fit(df.iloc[:600])
    sub = df.iloc[600:650]
    qs = m.predict_quantiles(sub)
    for q, hi in zip(qs, sub["run_hi"]):
        assert q[0.95] <= hi + 1e-9


def test_residual_target_is_opt_in_and_respects_the_ratchet():
    """L1: the LOW nowcast can regress the undercut (remaining move from the
    observed running minimum) instead of the temperature. Measured win:
    MAE 0.806 -> 0.765 at the decision hour. Must stay ratchet-consistent."""
    import numpy as np
    import pandas as pd
    from highcast.models.intraday import IntradayModel, INTRADAY_FEATURES
    rng = np.random.default_rng(4)
    n = 400
    run = 60 + rng.normal(0, 3, n)              # observed running minimum
    under = rng.choice([0, 1, 1, 2, 3], size=n)  # the undercut, as measured
    df = pd.DataFrame({c: 0.0 for c in INTRADAY_FEATURES}, index=range(n))
    df["run_hi"] = run
    df["run_lo"] = run - 0.2
    df["cur_tmpf"] = run + rng.normal(0, 1, n)
    df["hour"] = 6.0
    df["y"] = run - under
    plain = IntradayModel(extremum="min").fit(df.iloc[:300])
    resid = IntradayModel(extremum="min", residual=True).fit(df.iloc[:300])
    test = df.iloc[300:]
    yy = test["y"].to_numpy(dtype=float)
    mae_p = np.mean([abs(q[0.5] - t) for q, t in zip(plain.predict_quantiles(test), yy)])
    mae_r = np.mean([abs(q[0.5] - t) for q, t in zip(resid.predict_quantiles(test), yy)])
    assert mae_r < mae_p                    # the residual target is easier to learn
    for q, b in zip(resid.predict_quantiles(test), test["run_hi"]):
        assert q[0.5] <= b + 1e-9           # a min can never exceed the observed min


def test_residual_mode_is_opt_in_and_metric_scoped():
    """The undercut residual is validated for the LOW only (MAE 0.806 -> 0.765).
    It must be off by default and off for the HIGH until measured."""
    from highcast.config import ModelConfig
    from highcast.models.intraday import IntradayModel
    mc = ModelConfig()
    assert mc.intraday_residual_for("low") is True
    assert mc.intraday_residual_for("high") is False
    assert IntradayModel(extremum="min").residual is False   # default off

