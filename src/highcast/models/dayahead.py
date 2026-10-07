import numpy as np
import pandas as pd

from highcast.buckets import cdf_from_quantiles
from highcast.features import FEATURES
from highcast.models import make_lgbm as _lgbm
from highcast.models.baseline import LEVELS
from highcast.models.calibrate import PITCalibrator, RegimeCalibrator

HALF_LIFE_DAYS = 730.0   # climate-trend handling: recent years dominate
CAL_HOLDOUT_DAYS = 90


class DayAheadModel:
    def __init__(self, cal_mode: str = "off", feature_cols=None,
                 anchor_col: str | None = None):
        """`anchor_col` switches the target to the MOS formulation: the trees
        regress the residual `y - anchor` and the anchor is added back at
        predict time. Gradient-boosted trees are piecewise-constant, so their
        predictions are bounded by the training targets in each leaf — the hot
        tail (few rows) gets dragged toward the dense 72-78F mass. Regressing a
        small, near-stationary residual moves the *level* into the anchor, which
        extrapolates freely. None = the plain direct-target model (default)."""
        self.cal_mode = cal_mode
        self.feature_cols = list(feature_cols) if feature_cols is not None else FEATURES
        self.anchor_col = anchor_col
        self.regime = RegimeCalibrator().fit([], [], [])   # null until fit()

    def _anchor(self, df: pd.DataFrame) -> np.ndarray:
        """Anchor level per row; zeros when running the direct-target model."""
        if self.anchor_col is None:
            return np.zeros(len(df), dtype=float)
        a = pd.to_numeric(df[self.anchor_col], errors="coerce").to_numpy(dtype=float)
        # A missing anchor would poison the residual; fall back to the training
        # mean level so the row still contributes rather than becoming NaN.
        return np.where(np.isnan(a), getattr(self, "_anchor_mean", 0.0), a)

    @staticmethod
    def recency_weights(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df["target_date"])
        age = (dates.max() - dates).dt.days.astype(float)
        return np.exp(-age * np.log(2) / HALF_LIFE_DAYS)

    def _fit_raw(self, df: pd.DataFrame) -> dict:
        X = df[self.feature_cols]
        y = df["y"].to_numpy(dtype=float) - self._anchor(df)   # residual if anchored
        w = self.recency_weights(df)
        return {q: _lgbm(q).fit(X, y, sample_weight=w) for q in LEVELS}

    def _raw_quantiles(self, models: dict, df: pd.DataFrame) -> list[dict]:
        preds = np.column_stack([models[q].predict(df[self.feature_cols]) for q in LEVELS])
        preds = np.sort(preds, axis=1)               # enforce non-crossing
        preds = preds + self._anchor(df)[:, None]    # add the level back
        return [dict(zip(LEVELS, row)) for row in preds]

    def fit(self, df: pd.DataFrame) -> "DayAheadModel":
        df = df[df["y"].notna()].sort_values("target_date").reset_index(drop=True)
        if self.anchor_col is not None:
            a = pd.to_numeric(df[self.anchor_col], errors="coerce")
            # training-set level, used only to rescue rows with a missing anchor
            self._anchor_mean = float(a.mean()) if a.notna().any() else 0.0
        cut = pd.to_datetime(df["target_date"]).max() - pd.Timedelta(
            days=CAL_HOLDOUT_DAYS)
        train = df[pd.to_datetime(df["target_date"]) <= cut]
        hold = df[pd.to_datetime(df["target_date"]) > cut]
        self.cal = PITCalibrator().fit([])
        self.regime = RegimeCalibrator().fit([], [], [])
        if len(train) >= 200 and len(hold) >= 30:
            tmp = self._fit_raw(train)
            raw_hold = self._raw_quantiles(tmp, hold)
            pits = [cdf_from_quantiles(qs)(y) for qs, y in
                    zip(raw_hold, hold["y"])]
            self.cal = PITCalibrator().fit(pits)
            # regime corrector: out-of-fold post-PIT quantiles on the holdout
            levels, resids, spreads = [], [], []
            for qs, y in zip(raw_hold, hold["y"].to_numpy()):
                adj = self._apply_pit(qs)
                levels.append(adj[0.5])
                resids.append(float(y) - adj[0.5])
                spreads.append((adj[0.9] - adj[0.1]) / 2.5631)
            self.regime = RegimeCalibrator().fit(levels, resids, spreads)
        self.models = self._fit_raw(df)              # refit on everything
        return self

    def _apply_pit(self, raw: dict) -> dict:
        cdf_levels = sorted(raw)                 # nominal levels
        vals = [raw[l] for l in cdf_levels]
        adj = {}
        for l in LEVELS:
            u = self.cal.level(l)
            adj[l] = float(np.interp(u, cdf_levels, vals))
        return dict(sorted(adj.items()))

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        return [self.regime.apply(self._apply_pit(raw), self.cal_mode)
                for raw in self._raw_quantiles(self.models, df)]


# Blend thresholds (°F). Below LO the direct model is strictly better (measured
# 2026-07-25: <80F MAE 1.48 direct vs 1.69 residual); at/above HI the direct
# model cannot extrapolate at all (true hot days >=88F: MAE 6.99, bias -6.99).
# The bands themselves came from the earlier seam analysis, not from tuning on
# the blend result — but they are not independently cross-validated, so treat
# them as physically-motivated defaults rather than optimised parameters.
HYBRID_LO, HYBRID_HI = 82.0, 88.0


class HybridDayAheadModel:
    """Direct-target model for the dense regime, residual(MOS)-target model for
    the hot tail, linearly blended by the direct model's own prediction.

    Rationale (measured): gradient-boosted trees are piecewise-constant, so the
    direct model is bounded by its training targets and collapses on hot days
    (>=88F: MAE 6.99, bias -6.99). Regressing `y - fc_anchor` fixes the tail
    (MAE 3.66) but costs accuracy on the 93% of days below 80F (1.69 vs 1.48).
    Blending keeps the bulk untouched and recovers most of the tail: overall
    dMAE -0.021 [95% CI -0.044, -0.0004], <80F unchanged, >=88F MAE 4.68.
    """

    def __init__(self, cal_mode: str = "off", feature_cols=None,
                 anchor_col: str = "fc_anchor",
                 lo: float = HYBRID_LO, hi: float = HYBRID_HI):
        self.anchor_col = anchor_col
        self.lo, self.hi = lo, hi
        self.direct = DayAheadModel(cal_mode=cal_mode, feature_cols=feature_cols)
        self.residual = DayAheadModel(cal_mode=cal_mode, feature_cols=feature_cols,
                                      anchor_col=anchor_col)

    def fit(self, df: pd.DataFrame) -> "HybridDayAheadModel":
        self.direct.fit(df)
        # HIGH-ONLY BY CONSTRUCTION. The blend thresholds (82/88F) are the HIGH's
        # hot tail, so on the LOW the weight is always 0 — but that was an
        # accident of the numbers, and a LOW blend is actively harmful: measured
        # 2026-07-26 on the LOW day-ahead (N=1664), anchoring on `fc_anchor`
        # (which falls back to GFS's low forecast, since mos_mex is HIGH-only)
        # gave dMAE +4.72 and a +5 to +9F bias in the cold bands, versus a direct
        # model that is already unbiased there (bias +0.14 / +0.13 / -0.01).
        # So: skip the second fit entirely unless the data can actually reach the
        # blend region. Saves a wasted model fit and removes the foot-gun.
        can_blend = (self.anchor_col in df
                     and float(df["y"].max() if "y" in df else 0) >= self.lo)
        if can_blend:
            self.residual.fit(df)
        self._ok = bool(can_blend)
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        qd = self.direct.predict_quantiles(df)
        if not self._ok:
            return qd
        qr = self.residual.predict_quantiles(df)
        out = []
        for a, b in zip(qd, qr):
            w = float(np.clip((a[0.5] - self.lo) / (self.hi - self.lo), 0.0, 1.0))
            out.append({l: (1.0 - w) * a[l] + w * b[l] for l in a} if w > 0 else a)
        return out
