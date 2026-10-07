import numpy as np
import pandas as pd
from highcast import backtest
from highcast.models.baseline import BaselineBlend

def _frame(n=500, seed=2):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "target_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "fc_ecmwf": y + rng.normal(0, 1, n),
        "fc_nws": y + rng.normal(0, 1.5, n),
        "doy_sin": np.sin(2 * np.pi * np.arange(n) / 365),
        "doy_cos": np.cos(2 * np.pi * np.arange(n) / 365),
        "y": np.round(y),
    })

def test_walk_forward_only_predicts_after_min_train():
    df = _frame()
    recs = backtest.walk_forward(df, BaselineBlend, min_train=365, refit_every=60)
    assert len(recs) == len(df) - 365
    assert recs[0]["target_date"] == df["target_date"].iloc[365]
    assert all(r["crps"] >= 0 for r in recs)

def test_summarize_reports_calibration_and_baselines():
    df = _frame()
    recs = backtest.walk_forward(df, BaselineBlend, min_train=365, refit_every=60)
    s = backtest.summarize(recs, df)
    assert s["n"] == len(recs)
    assert s["mae"] < 2.5
    assert 0.05 <= s["cov_10"] <= 0.35      # rough calibration on synthetic data
    assert 0.65 <= s["cov_90"] <= 0.99
    assert "mae_nws" in s
