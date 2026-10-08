"""Residual quantile model (spec §3.4) and quantiles -> integer pmf."""
import math
from datetime import date

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from citycast.features import FEATURE_COLS
from highcast.buckets import bucket_probs, truncate_at_bound

QUANTILES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
HALF_LIFE_D = 730.0
MIN_GAP_F = 1e-3
CITY_COLS = [c for c in FEATURE_COLS if c not in ("lat", "lon", "elevation_m")]
SHARED_COLS = FEATURE_COLS + ["station"]


def recency_weights(event_dates: pd.Series, ref: date) -> np.ndarray:
    age = (pd.Timestamp(ref) - pd.to_datetime(event_dates)).dt.days.clip(lower=0).to_numpy()
    return 0.5 ** (age / HALF_LIFE_D)


class QuantileModel:
    def __init__(self, variant: str, station_categories: tuple[str, ...]):
        if variant not in ("shared", "city"):
            raise ValueError(variant)
        self.variant = variant
        self.cats = tuple(station_categories)
        self.cols = SHARED_COLS if variant == "shared" else CITY_COLS
        self.models: dict[float, LGBMRegressor] = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].copy()
        if "station" in X:
            X["station"] = pd.Categorical(X["station"], categories=list(self.cats))
        return X

    def fit(self, df: pd.DataFrame, ref: date) -> "QuantileModel":
        X = self._X(df)
        resid = (df["y"] - df["anchor"]).to_numpy()
        w = recency_weights(df["event_date"], ref)
        self.models = {a: LGBMRegressor(objective="quantile", alpha=a, **PARAMS).fit(X, resid, sample_weight=w)
                       for a in QUANTILES}
        return self

    def predict_y_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        q = np.column_stack([self.models[a].predict(X) for a in QUANTILES])
        q = np.sort(q, axis=1) + df["anchor"].to_numpy()[:, None]
        for j in range(1, q.shape[1]):                       # strictly increasing for the CDF
            q[:, j] = np.maximum(q[:, j], q[:, j - 1] + MIN_GAP_F)
        return q


def scale_quantiles(q: dict[float, float], k: float) -> dict[float, float]:
    m = q[0.5]
    return {a: m + k * (v - m) for a, v in q.items()}


def pmf_from_quantiles(q: dict[float, float], k: float, bound: float, metric: str) -> dict[int, float]:
    if metric not in ("high", "low"):   # a typo would otherwise truncate on the wrong side
        raise ValueError(f"metric must be 'high' or 'low', got {metric!r}")
    pmf = bucket_probs(scale_quantiles(q, k))
    if bound is not None and not math.isnan(bound):
        pmf = truncate_at_bound(pmf, bound, "max" if metric == "high" else "min")
    return pmf
