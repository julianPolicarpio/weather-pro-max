from datetime import date

import numpy as np
import pandas as pd

from citycast.features import FEATURE_COLS
from citycast.qmodel import (QUANTILES, QuantileModel, pmf_from_quantiles, recency_weights,
                             scale_quantiles)


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["anchor"] = rng.uniform(30, 90, n)
    df["station"] = rng.choice(["KNYC", "KLAX"], n)
    df["event_date"] = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    df["y"] = df["anchor"] + rng.normal(0, 2, n)
    return df


def test_recency_weights_halve_every_730_days():
    w = recency_weights(pd.Series(["2024-01-01", "2022-01-01"]), date(2024, 1, 1))
    assert w[0] == 1.0 and abs(w[1] - 0.5) < 0.01


def test_fit_predict_shapes_and_monotone():
    df = _frame()
    for variant in ("shared", "city"):
        m = QuantileModel(variant, ("KLAX", "KNYC")).fit(df, date(2025, 1, 1))
        q = m.predict_y_quantiles(df.head(5))
        assert q.shape == (5, len(QUANTILES))
        assert (np.diff(q, axis=1) > 0).all()
        assert abs(np.median(q[:, 6] - df.head(5)["anchor"].to_numpy())) < 3


def test_scale_quantiles_about_median():
    q = {0.1: 40.0, 0.5: 50.0, 0.9: 60.0}
    assert scale_quantiles(q, 0.5) == {0.1: 45.0, 0.5: 50.0, 0.9: 55.0}


def test_pmf_sums_to_one_and_truncates_high_below_bound():
    q = dict(zip(QUANTILES, np.linspace(40, 60, len(QUANTILES))))
    pmf = pmf_from_quantiles(q, 1.0, float("nan"), "high")
    assert abs(sum(pmf.values()) - 1) < 1e-9
    cut = pmf_from_quantiles(q, 1.0, 52.4, "high")
    assert min(k for k, v in cut.items() if v > 0) >= 52 and abs(sum(cut.values()) - 1) < 1e-9


def test_pmf_truncates_low_above_bound():
    q = dict(zip(QUANTILES, np.linspace(40, 60, len(QUANTILES))))
    cut = pmf_from_quantiles(q, 1.0, 47.6, "low")
    assert max(k for k, v in cut.items() if v > 0) <= 48 and abs(sum(cut.values()) - 1) < 1e-9


def test_pmf_rejects_unknown_metric():
    import pytest
    q = dict(zip(QUANTILES, np.linspace(40, 60, len(QUANTILES))))
    with pytest.raises(ValueError, match="metric"):
        pmf_from_quantiles(q, 1.0, float("nan"), "hgh")


def test_station_absent_from_training_keeps_fixed_codes():
    df = _frame()
    only_nyc = df[df["station"] == "KNYC"]
    m = QuantileModel("shared", ("KLAX", "KNYC")).fit(only_nyc, date(2025, 1, 1))
    X = m._X(df.head(4))
    assert list(X["station"].cat.categories) == ["KLAX", "KNYC"]
    assert m.predict_y_quantiles(df.head(4)).shape == (4, len(QUANTILES))
