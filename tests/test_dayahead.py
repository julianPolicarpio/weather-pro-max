import numpy as np
import pandas as pd
from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.dayahead import DayAheadModel


def _frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    df = pd.DataFrame({c: np.nan for c in FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d")
    df["fc_ecmwf"] = y + rng.normal(0, 1, n)
    df["fc_gfs"] = y + rng.normal(0, 2, n)
    df["doy_sin"] = np.sin(2 * np.pi * np.arange(n) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * np.arange(n) / 365)
    df["year_index"] = 24.0
    df["heat_alert"] = 0.0
    df["y"] = np.round(y)
    return df


def test_fit_predict_quantiles_monotone_and_sane():
    df = _frame()
    m = DayAheadModel().fit(df.iloc[:500])
    qs = m.predict_quantiles(df.iloc[500:520])
    assert len(qs) == 20
    for q in qs:
        vals = [q[l] for l in LEVELS]
        assert vals == sorted(vals)
    med_err = np.mean([abs(q[0.5] - y) for q, y in zip(qs, df["y"].iloc[500:520])])
    assert med_err < 3.0


def test_recency_weights_defined():
    df = _frame()
    w = DayAheadModel.recency_weights(df)
    assert w.iloc[-1] > w.iloc[0]           # newer rows weigh more
    assert abs(w.iloc[-1] - 1.0) < 1e-6
