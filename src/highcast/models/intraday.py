from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast import latentmax
from highcast.config import AppConfig
from highcast.features import (_alert_flag, build_features,
                               metar_point_features, uses_bias_correction)
from highcast.metric import MetricSpec, get_metric
from highcast.timeutil import climate_day_bounds
from highcast.models import make_lgbm as _lgbm
from highcast.models.baseline import LEVELS, BaselineBlend

INTRADAY_FEATURES = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                     "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                     "base_spread", "doy_sin", "doy_cos", "year_index",
                     "heat_alert"]

# ── WARMING-RATE FEATURES: KLAX HIGH ONLY (shipped 2026-07-31) ──────────────
#
# The nowcast had NO time-derivative of any kind, so it could not express "the
# temperature is climbing fast" -- which at a coastal station IS the marine-layer
# burn-off signal. The DAY-AHEAD HIGH model has had this as `morn_rate` since the
# start and `features.py` calls it a backtested win; the nowcast, which actually
# decides the bet at h11, did not get it.
#
# MEASURED, walk-forward, refit 45, matched-hour, parity-averaged bucket hit,
# test >= 2025-01-01, n=573 (`scripts/klax_high_env_eval.py`):
#
#     arm        MAE     HIT     dMAE     dHIT    97.5% CI on dHIT
#     BASE      1.104   51.7%      --       --
#     RATE      1.053   54.2%   -0.051    +2.5%   [+0.3%, +4.9%]   <- SHIPPED
#     ENV4      1.077   52.6%   -0.027    +1.0%   [-1.4%, +3.2%]
#     RATE+ENV  1.047   53.9%   -0.057    +2.3%   [-0.4%, +4.9%]
#     RAND      1.096   50.3%   -0.008    -1.4%   [-3.5%, +0.8%]
#
# Cleared all five pre-registered gates INCLUDING the Bonferroni-corrected
# significance test -- the first hypothesis in seven attempts here to do so.
# RATE ALONE BEATS RATE+ENV (+2.5 vs +2.3): the rejected environment columns
# dilute it, the same per-split feature-sampling dilution measured twice before.
#
# ⚠️ KLAX HIGH ONLY. `hour_features` returns these solely for that market. The
# other three markets are FROZEN and their scorecards depend on nothing changing;
# the LOW's own rate feature was measured INERT (0.00F on every day at its commit
# hour, because LAX sits flat at 4am under a marine layer), so this is not merely
# untested there -- it is known not to apply.
RATE_FEATURES = ["cur_rate_3h", "cur_rate_1h", "cur_rise_from_min"]


def features_for(station: str, metric: str) -> list[str]:
    """The feature columns a given market's nowcast consumes.

    Exists so the KLAX HIGH can carry the warming-rate columns without touching
    the three frozen markets. Any station/metric not named here gets exactly the
    shipped `INTRADAY_FEATURES`, so a new market cannot silently inherit a feature
    set measured somewhere else -- the same no-silent-fallback contract that
    `gating`'s constants already enforce.
    """
    if (station, metric) == ("KLAX", "high"):
        return [*INTRADAY_FEATURES, *RATE_FEATURES]
    return list(INTRADAY_FEATURES)

# Default checkpoint hours (station-local wall clock) for BACKTESTS and for the
# small-sample fallback. The live path no longer uses these to train: it trains
# at the single hour it is about to predict at, because predicting at an hour
# outside the trained set extrapolates the `hour` feature (LightGBM clamps to the
# nearest split and applies the wrong regime's residual). See cli.cmd_predict.
CHECKPOINT_HOURS = (8, 10, 12, 14, 16, 18, 20)

# Matched-hour training yields one row per day, so ~2,030 rows for KLAX's
# history. Below this, fall back to the wide frame rather than fit on noise.
MIN_MATCHED_ROWS = 400

# ❌ CLOSED NEGATIVE (2026-07-26): out-of-fold PIT calibration on the NOWCAST.
#
# The nowcast returns raw LightGBM quantiles with no calibration layer, unlike
# `DayAheadModel`, which has had out-of-fold PIT calibration since day one. That
# asymmetry looks like an obvious gap, and the tail coverage confirms the models
# under-disperse (one-sided P(y <= q), nominal -> actual, 2025+ holdout):
#     LOW  h6   0.10->0.125  0.50->0.446  0.90->0.833
#     HIGH h9   0.10->0.123  0.50->0.469  0.90->0.865
#
# It was implemented (PITCalibrator, 90-day out-of-fold holdout split by
# target_date, re-clamped after remapping) and MEASURED. It fixes tail coverage
# and it is still NOT worth shipping -- reproduce with
# `scripts/calibration_eval.py`:
#   * CRPS degraded at ALL 11 measured hour/metric combinations (LOW h6
#     0.500->0.511; HIGH h9 0.980->1.000). CRPS is the proper scoring rule.
#   * HIGH modal-bucket HIT collapsed: h11 52.4%->47.1%, h13 71.0%->65.9%.
#     Widening flattens the pmf and moves the argmax off the right bucket.
#   * LOW calibration gap got WORSE at every hour it is actually used
#     (h6 1.8%->7.5%): the modal probability was already well calibrated, so
#     widening flipped mild overconfidence into larger underconfidence.
#
# The lesson: there are TWO different miscalibrations here and one fix cannot
# serve both. The TAILS are too narrow, but the MODAL BUCKET -- the only number
# the staking gate consumes -- is already within ~2 points at the hours in use.
# The distribution's tightness is earning its CRPS. Do not retry without a
# tail-only correction that provably leaves the mode and CRPS alone.


def _asof_features(conn, cfg: AppConfig, d: date, asof_utc: str) -> dict:
    # The trailing columns are ENVIRONMENT EXTRAS: they reach the frame so an
    # experiment can name them via `IntradayModel(extra_features=[...])`, but
    # they enter no shipped model by default. See `features.metar_point_features`.
    row = conn.execute(
        """SELECT tmpf, tmpf_precise, dwpf, sknt, drct, skyc,
                  ceiling_ft, sky_lowest_ft, vis_sm, pres_tend_hpa, slp_hpa
           FROM obs
           WHERE station=? AND source='metar' AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (cfg.station.icao, asof_utc)).fetchone()
    feats = metar_point_features(row, "cur_")
    feats.update(_rate_features(conn, cfg, asof_utc, feats.get("cur_tmpf")))
    return feats


def _temp_at(conn, cfg: AppConfig, asof_utc: str) -> float:
    """Latest observed temperature at or before `asof_utc`, else NaN."""
    r = conn.execute(
        """SELECT tmpf, tmpf_precise FROM obs
           WHERE station=? AND source='metar' AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (cfg.station.icao, asof_utc)).fetchone()
    if r is None:
        return float("nan")
    t = r["tmpf_precise"] if r["tmpf_precise"] is not None else r["tmpf"]
    return float("nan") if t is None else float(t)


def _rate_features(conn, cfg: AppConfig, asof_utc: str, cur: float | None) -> dict:
    """WARMING-RATE candidates. In the FRAME only -- no shipped model reads them
    unless named via `IntradayModel(extra_features=[...])`.

    THE GAP. `INTRADAY_FEATURES` contains no rate, lag, delta or trend column, so
    the nowcast cannot express "the temperature is climbing fast" -- which at a
    coastal station IS the marine-layer burn-off signal, and is what nowcast.trade
    ships as RE-HEAT RISK. The DAY-AHEAD HIGH model already has this as `morn_rate`
    and `features.py` calls it a backtested win; the nowcast, which is what actually
    decides the bet at h11, does not get it.

    ⚠️ WHY HANDOFF'S "rate-of-change is inert" NEGATIVE DOES NOT APPLY HERE. That
    was measured at the LOW's commit hour, where the 2h change was 0.00F on EVERY
    day sampled -- LAX sits flat at 4am under a marine layer, so the feature was a
    constant and could discriminate nothing. At the HIGH's h11 the same quantity is
    highly variable (KLAX 08->11 local, n=400): mean +6.79F, sd 4.35F, range -6.1
    to +22.0, exactly zero on only 2% of days. A constant cannot help; this can.

    NaN, NEVER 0.0, when the lookback observation is missing. Zero would read as
    "no warming", which is a real and very different state from "unknown" -- and on
    a feature whose whole point is discriminating fast from slow warming, that
    substitution would inject a false signal on exactly the gappy days.
    """
    nan = float("nan")
    if cur is None or cur != cur:
        return {"cur_rate_3h": nan, "cur_rate_1h": nan,
                "cur_rise_from_min": nan}
    asof = datetime.fromisoformat(asof_utc)

    def back(hours):
        t = _temp_at(conn, cfg, (asof - timedelta(hours=hours)).isoformat())
        return nan if t != t else cur - t

    # How far it has already climbed off the day's floor. A different quantity
    # from the rate: the same +5F/3h means something different at 60F than at 75F.
    start, _ = climate_day_bounds(asof.date(), cfg.station.std_utc_offset)
    r = conn.execute(
        """SELECT MIN(COALESCE(tmpf_precise, tmpf)) AS m FROM obs
           WHERE station=? AND source='metar' AND valid_utc >= ? AND valid_utc <= ?""",
        (cfg.station.icao, start.isoformat(), asof_utc)).fetchone()
    day_min = nan if (r is None or r["m"] is None) else float(r["m"])
    return {"cur_rate_3h": back(3), "cur_rate_1h": back(1),
            "cur_rise_from_min": nan if day_min != day_min else cur - day_min}


def _split_hour(h) -> tuple[int, int]:
    """A checkpoint hour as (hour, minute). Accepts whole hours (13 -> 13:00)
    and half hours (13.5 -> 13:30) ONLY.

    WHY HALVES AND NOTHING FINER. `signals` is written every 5 minutes, so a
    13.75 checkpoint would build without complaint and look perfectly
    reasonable — but `gating`'s HOUR_ACCURACY, HOUR_CALIB, BAND_CALIB and
    CONFIDENCE_FLOOR are keyed by this number and every one of them FAILS
    CLOSED on a miss. The result would not be an error, it would be a market
    that silently never trades again. Refusing at the door is the loud version
    of that failure.

    Half hours exist because the KMIA HIGH ask cliffs at 14:00 (0.38 at 13:30,
    0.82 at 14:00 — the market's own price, so the step is real), and 13:30 is
    the latest checkpoint still on the cheap side of it.
    """
    f = float(h)
    if f < 0 or f >= 24 or (f * 2) % 1 != 0:
        raise ValueError(
            f"checkpoint hour {h!r} must be a whole or half hour in [0, 24) — "
            f"finer granularity has no gating table and would fail closed")
    return int(f), 30 if f % 1 else 0


def build_checkpoint_frame(conn, cfg: AppConfig, dates: list[str],
                           hours=CHECKPOINT_HOURS,
                           baseline: BaselineBlend | None = None,
                           require_target: bool = True,
                           metric: MetricSpec | None = None) -> pd.DataFrame:
    """Training path: baseline=None fits the blend on `dates` themselves.
    Live path: pass a baseline fitted on HISTORY (a 1-day frame with no
    outcome cannot fit one) and require_target=False.

    `metric` defaults to the high spec, reproducing the pre-low behavior
    exactly (build_features on the high metric == build_dayahead, and the
    "max" running extremum == visible_running_max). For the low, the
    run_lo/run_hi/run_width columns keep their names but now hold the
    running-MIN interval."""
    metric = metric or get_metric(cfg, "high")
    # Validate BEFORE any DB work: a bad hour is a caller bug, and surfacing it
    # as a ValueError beats surfacing it as an empty frame twenty seconds later.
    checkpoints = [_split_hour(h) for h in hours]
    day_df = build_features(conn, cfg, metric, dates, require_target=require_target)
    if day_df.empty:
        return pd.DataFrame(columns=["target_date", "y", *INTRADAY_FEATURES])
    # Bias-corrected guidance is LOW-only and measured — see
    # `features.uses_bias_correction`. The HIGH nowcast was WORSE with it at its
    # early hours (+0.050 MAE at hour 9, bucket hit 45.7% -> 43.2%).
    base = baseline or BaselineBlend(
        prefer_corrected=uses_bias_correction(metric.key)).fit(day_df)
    base_qs = base.predict_quantiles(day_df)
    tz = ZoneInfo(cfg.station.tz)
    rows = []
    for (_, day_row), qs in zip(day_df.iterrows(), base_qs):
        d = date.fromisoformat(day_row["target_date"])
        for h, (hh, mm) in zip(hours, checkpoints):
            asof = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz).astimezone(
                ZoneInfo("UTC")).isoformat()
            vis = latentmax.visible_running_extremum(
                conn, cfg, d, metric.extremum, asof_utc=asof)
            if vis is None:
                continue
            rec = {"target_date": day_row["target_date"], "hour": float(h),
                   "run_lo": vis[0], "run_hi": vis[1],
                   "run_width": vis[1] - vis[0],
                   "base_p50": qs[0.5], "base_spread": qs[0.9] - qs[0.1],
                   "doy_sin": day_row["doy_sin"], "doy_cos": day_row["doy_cos"],
                   "year_index": day_row["year_index"],
                   "heat_alert": day_row["heat_alert"], "y": day_row["y"]}
            # ── `heat_alert_asof`: THE SAME QUESTION, ASKED AT THE RIGHT TIME ──
            #
            # `heat_alert` above is INHERITED FROM THE DAY-AHEAD ROW, so it is
            # evaluated at the day-ahead cutoff. That is correct for a day-ahead
            # forecast and WRONG for a nowcast: Miami's Heat Advisories are
            # issued the same morning (onset typically 15:00-16:00Z), hours
            # BEFORE the h13 checkpoint but well AFTER the day-ahead cutoff.
            #
            # Measured 2026-08-19 over 2025-01-01..2026-08-16:
            #     KMIA high   as inherited  1/593      as-of  46/593
            #     KLAX high   as inherited 17/593      as-of  25/593
            # and KMIA's single inherited hit is 2026-02-23, a Red Flag Warning
            # -- the feature has never once fired on a Miami heat day. KLAX's 17
            # are January/March Santa Ana fire-weather products, so at that
            # station `heat_alert` is in practice a WIND flag, not a heat one.
            #
            # ⚠️ WRITTEN AS A SEPARATE COLUMN ON PURPOSE. `features._alert_flag`
            # carries an explicit warning that this feature changing silently is
            # "a model change disguised as a data change, with no gate and no way
            # to notice". So the incumbent column is left exactly as it was and
            # no shipped model reads this one unless it is named via
            # `IntradayModel(extra_features=[...])`.
            #
            # ✗ THE SWAP WAS GATED AND FAILED ON ALL FOUR MARKETS.
            # `scripts/heat_alert_asof_gate.py`, pre-registered, walk-forward
            # refit=45, test >= 2025-01-01, paired bootstrap at Bonferroni x4:
            #
            #     market      n     dHIT     CI (Bonf x4)     dMAE   vs RAND
            #     KLAX high  592   +0.2%   [+0.0%, +0.5%]   +0.000   beats
            #     KLAX low   592   +0.0%   [+0.0%, +0.0%]   +0.000   loses
            #     KMIA high  589   -0.1%   [-1.3%, +1.1%]   +0.000   loses
            #     KMIA low   586   +0.0%   [-0.7%, +0.7%]   -0.002   loses
            #
            # KLAX low is the harness's own sanity check: its 06:00 cutoff sees
            # the same alerts either way (78 rows vs 78), the arms are identical
            # and dHIT is exactly 0.000 -- while the RANDOM arm drifted +0.6% on
            # noise alone. That is the whole reason the control arm exists.
            #
            # SO: DO NOT SWAP THIS IN. Repairing WHEN the question is asked does
            # not improve bucket hit anywhere, even at KMIA HIGH where the flag
            # goes from 114 to 473 frame rows. The column stays as a MEASURED-
            # INERT frame column so the next person does not re-run this.
            #
            # ⚠️ The DESCRIPTIVE defect is still real and is NOT repaired by this
            # verdict: the shipped nowcast cannot see a Miami heat advisory. It
            # simply does not cost bucket hit. See HANDOFF 3.9.
            rec["heat_alert_asof"] = _alert_flag(
                conn, cfg, d, datetime.fromisoformat(asof))
            rec.update(_asof_features(conn, cfg, d, asof))
            # ── CANDIDATE: bound-vs-guidance disagreement ───────────────────
            # In the FRAME only; no shipped model reads it unless named via
            # `extra_features`.
            #
            # WHY A TREE CANNOT DERIVE THIS ITSELF. `run_hi` and `base_p50` are
            # both already inputs, but a tree CANNOT SUBTRACT -- splits threshold
            # one feature at a time, so it can never form their difference. This
            # is the identical argument that justified the `fcc_*` bias-corrected
            # columns, which measured as a significant win.
            #
            # WHY IT SHOULD MATTER, measured 2026-07-31 on the KLAX LOW at h4:
            # the ratchet bound is only informative once the minimum has actually
            # happened, and it has NOT on 36% of days. Split by when the minimum
            # lands, |error| of the bound vs guidance:
            #
            #     min lands      n     bound MAE   guidance MAE
            #     before h4    367       1.22          1.10
            #     h5-h7        144       2.35          1.24
            #     h19-h23       62       2.45          1.14
            #
            # Guidance error is FLAT; the bound's DOUBLES. The model weights the
            # bound identically in both regimes because nothing tells it which
            # regime it is in. This gap is that signal: large when the bound and
            # guidance disagree, which is exactly when the bound is misleading.
            # Live example, 2026-07-30: bound 69.88, guidance 67.52, gap +2.36,
            # actual 67 -- guidance was within 0.5F and the model overrode it.
            if rec.get("base_p50") is not None and vis is not None:
                bound = vis[1] if metric.extremum == "min" else vis[0]
                rec["cur_bound_gap"] = bound - rec["base_p50"]
            rows.append(rec)
    # Column order pins the SHIPPED set first so any consumer indexing
    # positionally is unaffected, then keeps whatever extras `_asof_features`
    # produced. An earlier version hard-pinned `columns=[..., *INTRADAY_FEATURES]`,
    # which silently DROPPED every candidate column -- an experiment naming one
    # via `extra_features` would have raised or, worse, trained on NaN. The
    # shipped model still consumes exactly `INTRADAY_FEATURES`; `IntradayModel`
    # selects its own columns, so widening the frame changes no prediction.
    base_cols = ["target_date", "y", *INTRADAY_FEATURES]
    extra_cols = [c for c in (rows[0].keys() if rows else [])
                  if c not in base_cols]
    return pd.DataFrame(rows, columns=[*base_cols, *extra_cols])


class IntradayModel:
    def __init__(self, extremum: str = "max", residual: bool = False,
                 extra_features: list | None = None):
        """`residual=True` regresses the REMAINING MOVE from the observed
        running extremum instead of the temperature itself, adding the bound
        back at predict time.

        Why: the ratchet bound is already known exactly at decision time, and
        the outstanding move is a small, tightly-distributed quantity. Measured
        for the LOW at LST 05:00 (N=2029): the settled CLI low equals the
        rounded running min on 36% of days and undercuts it by a median of 1F
        otherwise (0:36%, 1:38%, 2:14%, 3:6%). Predicting that residual is a far
        easier problem than predicting the temperature — the naive rule "the low
        IS the running min" already scores MAE 1.071F.
        """
        self.extremum = extremum
        self.residual = residual
        self._fallback = 0.0
        # EXTRA feature columns beyond INTRADAY_FEATURES. Default None reproduces
        # the shipped model exactly -- the column list is then byte-identical, so
        # KLAX cannot move. Used to test whether a signal the nowcast currently
        # lacks earns its place, without forking the model class.
        self.extra_features = list(extra_features or [])

    @property
    def _cols(self) -> list:
        return [*INTRADAY_FEATURES, *self.extra_features]

    def _bound(self, frame: pd.DataFrame) -> np.ndarray:
        """The ratchet bound: for a max the low end of the observed running max,
        for a min the high end of the observed running min."""
        col = "run_lo" if self.extremum == "max" else "run_hi"
        return frame[col].to_numpy(dtype=float)

    def _anchor(self, frame: pd.DataFrame) -> np.ndarray:
        b = self._bound(frame)
        return np.where(np.isnan(b), self._fallback, b)

    def _fit_raw(self, frame: pd.DataFrame) -> dict:
        X = frame[self._cols]
        y = frame["y"].to_numpy(dtype=float)
        if self.residual:
            y = y - self._anchor(frame)
        return {q: _lgbm(q).fit(X, y) for q in LEVELS}

    def _raw_quantiles(self, models: dict, frame: pd.DataFrame) -> list[dict]:
        """Uncalibrated quantiles, sorted and physically clamped. This is exactly
        what the model used to return."""
        preds = np.column_stack(
            [models[q].predict(frame[self._cols]) for q in LEVELS])
        if self.residual:
            preds = preds + self._anchor(frame)[:, None]   # restore the level
        preds = np.sort(preds, axis=1)              # enforce non-crossing
        return [dict(zip(LEVELS, self._clamp(row, b).astype(float)))
                for row, b in zip(preds, self._bounds(frame))]

    def _bounds(self, frame: pd.DataFrame) -> np.ndarray:
        col = "run_lo" if self.extremum == "max" else "run_hi"
        return frame[col].to_numpy(dtype=float)

    def _clamp(self, row: np.ndarray, bound: float) -> np.ndarray:
        """The RATCHET is physical truth and always wins: a max cannot be below
        the already-observed running max, a min cannot be above the observed
        running min. Verified on 2025+ holdouts that truth NEVER crosses it
        (P(violation) = 0.0% at every hour, both metrics), so clamping can only
        remove mass that is genuinely impossible."""
        if np.isnan(bound):
            return row
        return (np.maximum if self.extremum == "max" else np.minimum)(row, bound)

    def fit(self, frame: pd.DataFrame) -> "IntradayModel":
        # No sort/reset_index here: `target_date` is not required by this model
        # (only the abandoned calibration holdout needed it), and callers pass
        # bare feature frames without it -- see
        # test_intraday::test_residual_target_is_opt_in_and_respects_the_ratchet.
        frame = frame[frame["y"].notna()]
        if self.residual:
            b = self._bound(frame)
            self._fallback = float(np.nanmean(b)) if np.isfinite(b).any() else 0.0
        self.models = self._fit_raw(frame)
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> list[dict[float, float]]:
        return self._raw_quantiles(self.models, frame)
