import numpy as np
import pandas as pd

LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95
_FC_PREFIX = "fc_"
# EVERY NAME HERE IS DERIVED FROM THE BLEND'S OWN INPUTS, not an independent
# forecast. The prefix scan below cannot tell the two apart, so anything
# `build_features` computes FROM the guidance columns has to be named here.
#   fc_spread / fc_mean / fcc_mean  aggregates OF the fc_* columns
#   fc_anchor                       coalesce(fc_mos_mex -> fc_gfs -> fc_mean),
#                                   the level anchor for the residual day-ahead
#                                   target (features.py). Added 2026-08-02: it
#                                   had been taking 8-17% of the blend weight on
#                                   all four live markets, re-entering whichever
#                                   model the chain selected that day a SECOND
#                                   time -- and on its third branch re-admitting
#                                   fc_mean, which this set already forbids. On
#                                   the LOW it entered RAW (there is no
#                                   `fcc_anchor`), diluting the bias correction
#                                   `uses_bias_correction` measured as
#                                   significant for that metric.
#                                   Measured before shipping -- nested
#                                   walk-forward, all four markets, both layers:
#                                   `scripts/anchor_exclusion_eval.py`.
_EXCLUDE = {"fc_spread", "fc_mean", "fcc_mean", "fc_anchor"}
SEASON_WINDOW = 45  # +/- days of day-of-year for the residual pool
MIN_COL_ROWS = 30


class BaselineBlend:
    """Inverse-MSE weighted blend of the guidance columns, plus a seasonal
    residual pool for spread.

    THIS IS THE ONLY GUIDANCE CHANNEL THE INTRADAY NOWCAST HAS. The nowcast's
    feature list (`INTRADAY_FEATURES`) contains no `fc_*` column at all -- it sees
    guidance exclusively as `base_p50`/`base_spread` from this blend. Since
    `preferred_prediction` serves the nowcast for the LOW at every hour, this
    class is on the LOW's operational path and the day-ahead model is not. Any
    guidance improvement that does not reach here does not reach the LOW desk.
    """

    def __init__(self, prefer_corrected: bool = True):
        """`prefer_corrected` swaps in the bias-corrected `fcc_{model}` column in
        place of `fc_{model}` wherever one is populated. Kept as a flag so the
        two arms stay measurable (`scripts/baseline_corrected_eval.py`) rather
        than the change being unfalsifiable after the fact."""
        self.prefer_corrected = prefer_corrected

    def _blend_cols(self, df: pd.DataFrame) -> list[str]:
        """One column per guidance model, preferring the corrected variant.

        NOTE `"fcc_gfs".startswith("fc_")` is FALSE (third char is 'c', not '_'),
        so the corrected columns are invisible to the old prefix scan -- which is
        exactly why they had to be opted in explicitly here rather than being
        picked up for free. Selecting one variant PER MODEL also matters: blending
        both `fc_gfs` and `fcc_gfs` would enter the same model twice and
        double-weight it.

        THE SAME FAILURE HAS A SECOND DOOR, and this docstring used to guard only
        the first. A column can double-weight a model without being its `fcc_`
        twin: `build_features` appends DERIVED `fc_*` columns, and to a prefix
        scan a coalesce of three guidance columns is indistinguishable from a
        tenth forecast centre. `fc_anchor` was exactly that for as long as it
        existed. The invariant is therefore stated on `_EXCLUDE` rather than on
        the variant pair alone: NO BLEND INPUT MAY BE A FUNCTION OF OTHER BLEND
        INPUTS. `tests/test_baseline.py` enforces it by perturbation -- bump one
        model, rebuild the frame, and any other column that moves is derived --
        so a future derived `fc_*` column fails a test instead of silently
        collecting weight.
        """
        cols = []
        for col in df.columns:
            if not col.startswith(_FC_PREFIX) or col in _EXCLUDE:
                continue
            corrected = f"fcc_{col[len(_FC_PREFIX):]}"
            if (self.prefer_corrected and corrected in df.columns
                    and (df[corrected].notna() & df["y"].notna()).sum() >= MIN_COL_ROWS):
                cols.append(corrected)
            else:
                cols.append(col)
        return cols

    def fit(self, df: pd.DataFrame) -> "BaselineBlend":
        self.weights: dict[str, float] = {}
        for col in self._blend_cols(df):
            mask = df[col].notna() & df["y"].notna()
            if mask.sum() < MIN_COL_ROWS:
                continue
            mse = float(np.mean((df.loc[mask, col] - df.loc[mask, "y"]) ** 2))
            self.weights[col] = 1.0 / max(mse, 0.25)
        if not self.weights:
            raise ValueError("no forecast columns with enough data to fit")
        pts = self.point(df)
        ok = df["y"].notna() & ~np.isnan(pts)
        self._resid = (df.loc[ok, "y"] - pts[ok]).to_numpy()
        self._resid_doy = pd.to_datetime(
            df.loc[ok, "target_date"]).dt.dayofyear.to_numpy()
        # Seasonal climatology of the target, kept as a fallback point for any
        # day with NO usable forecast (all fc_* NaN) so the blend never emits a
        # NaN point that would crash bucket_probs downstream.
        yv = df["y"].to_numpy(dtype=float)
        oky = ~np.isnan(yv)
        self._clim_y = yv[oky]
        self._clim_doy = pd.to_datetime(
            df["target_date"]).dt.dayofyear.to_numpy()[oky]
        return self

    def _seasonal_clim(self, doy: int) -> float:
        dist = np.minimum(np.abs(self._clim_doy - doy),
                          365 - np.abs(self._clim_doy - doy))
        pool = self._clim_y[dist <= SEASON_WINDOW]
        pool = pool if len(pool) >= 30 else self._clim_y
        return float(np.mean(pool)) if len(pool) else float("nan")

    def point(self, df: pd.DataFrame) -> np.ndarray:
        num = np.zeros(len(df))
        den = np.zeros(len(df))
        for col, w in self.weights.items():
            v = df[col].to_numpy(dtype=float)
            has = ~np.isnan(v)
            num[has] += w * v[has]
            den[has] += w
        out = np.full(len(df), np.nan)
        out[den > 0] = num[den > 0] / den[den > 0]
        return out

    def _seasonal_resid(self, doy: int) -> np.ndarray:
        dist = np.minimum(np.abs(self._resid_doy - doy),
                          365 - np.abs(self._resid_doy - doy))
        pool = self._resid[dist <= SEASON_WINDOW]
        return pool if len(pool) >= 60 else self._resid

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        pts = self.point(df)
        doys = pd.to_datetime(df["target_date"]).dt.dayofyear.to_numpy()
        out = []
        for p, doy in zip(pts, doys):
            if np.isnan(p):                       # no forecast this day -> climatology
                p = self._seasonal_clim(int(doy))
            pool = self._seasonal_resid(int(doy))
            qs = np.quantile(pool, LEVELS)
            out.append({l: float(p + q) for l, q in zip(LEVELS, np.sort(qs))})
        return out
