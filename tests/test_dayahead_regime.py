import numpy as np
import pandas as pd

from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.calibrate import RegimeCalibrator
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


def test_off_mode_is_pit_passthrough():
    df = _frame()
    m = DayAheadModel(cal_mode="off").fit(df.iloc[:500])
    rows = df.iloc[500:510]
    out = m.predict_quantiles(rows)
    ref = [m._apply_pit(raw) for raw in m._raw_quantiles(m.models, rows)]
    assert out == ref                          # regime is a no-op in off mode


def test_regime_layer_wired_and_widens():
    df = _frame()
    m = DayAheadModel(cal_mode="width").fit(df.iloc[:500])
    rows = df.iloc[500:510]
    pit_only = [m._apply_pit(raw) for raw in m._raw_quantiles(m.models, rows)]
    # inject a corrector that widens everywhere (realized sd >> claimed 1.5)
    m.regime = RegimeCalibrator().fit(
        np.linspace(60.0, 95.0, 300),
        np.tile([-3.0, 3.0], 150),
        np.full(300, 1.5))
    widened = m.predict_quantiles(rows)
    for pit_q, wide_q in zip(pit_only, widened):
        assert abs(wide_q[0.5] - pit_q[0.5]) < 1e-6                    # median kept
        assert (wide_q[0.9] - wide_q[0.1]) > 1.3 * (pit_q[0.9] - pit_q[0.1])
        vals = [wide_q[l] for l in LEVELS]
        assert vals == sorted(vals)                                    # ordered


def test_calibration_config_defaults_off():
    from highcast.config import CalibrationConfig, load_config
    assert CalibrationConfig().mode == "off"
    assert load_config().calibration.mode == "off"    # klax.yaml default
