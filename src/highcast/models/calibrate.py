import numpy as np


class PITCalibrator:
    """Map nominal quantile levels to the levels that actually achieve them.

    PIT value = F_pred(y). If forecasts were perfectly calibrated, PITs are
    uniform. cal.level(q) returns u such that P(PIT <= u) = q, i.e. the raw
    quantile curve should be read at level u to get true coverage q.
    """

    def fit(self, pits: list[float]) -> "PITCalibrator":
        self._sorted = np.sort(np.clip(np.asarray(pits, dtype=float), 0.0, 1.0))
        return self

    def level(self, nominal: float) -> float:
        if len(self._sorted) < 30:
            return nominal
        return float(np.clip(np.quantile(self._sorted, nominal), 0.01, 0.99))


class RegimeCalibrator:
    """Smooth, regime-conditional correction of a quantile set, keyed on the
    predicted median. Learns a mean-shift m(level) and spread-multiplier
    s(level) from out-of-fold (median, actual, pred_spread_sd) triples,
    kernel-smoothed over predicted level and shrunk toward the null
    (m=0, s=1) where data is sparse. mode 'off' is a no-op."""

    def __init__(self, bandwidth: float = 2.5, min_eff_n: float = 8.0):
        self.bw = bandwidth
        self.min_eff_n = min_eff_n
        self._fitted = False

    def fit(self, level, resid, pred_spread_sd) -> "RegimeCalibrator":
        self._level = np.asarray(level, dtype=float)
        self._resid = np.asarray(resid, dtype=float)
        self._pspread = np.asarray(pred_spread_sd, dtype=float)
        self._fitted = self._level.size >= 30
        return self

    def _local(self, x: float):
        d = (self._level - x) / self.bw
        w = np.exp(-0.5 * d * d)
        return w, float(w.sum())

    def mean_shift(self, x: float) -> float:
        if not self._fitted:
            return 0.0
        w, eff = self._local(x)
        if eff <= 0:
            return 0.0
        raw = float((w * self._resid).sum() / eff)
        return raw * (eff / (eff + self.min_eff_n))          # shrink toward 0

    def spread_mult(self, x: float) -> float:
        if not self._fitted:
            return 1.0
        w, eff = self._local(x)
        if eff <= 0:
            return 1.0
        med = float((w * self._resid).sum() / eff)
        realized = 1.4826 * float((w * np.abs(self._resid - med)).sum() / eff)
        predicted = float((w * self._pspread).sum() / eff)
        if predicted <= 1e-6:
            return 1.0
        raw = realized / predicted
        s = 1.0 + (raw - 1.0) * (eff / (eff + self.min_eff_n))  # shrink toward 1
        return float(max(s, 0.05))

    def apply(self, quantiles: dict, mode: str) -> dict:
        if mode == "off" or not self._fitted:
            return dict(quantiles)
        m0 = quantiles[0.5]
        s = self.spread_mult(m0)
        shift = self.mean_shift(m0) if mode == "center_width" else 0.0
        return {l: float(m0 + shift + s * (v - m0)) for l, v in quantiles.items()}
