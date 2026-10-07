"""Gating for the recommendation: is the model's pick trustworthy AND still
tradeable RIGHT NOW? Shared by the CLI `bet` decision-support command and (later)
the web display, so "what we'd show" and "what we'd bet" use one definition.

Two independent gates:

  committed      -- the model's most-confident 2F bucket is (a) confident
                    (prob >= min_confidence) and (b) STABLE: the same bucket has
                    been the pick across the last `stability_readings` intraday
                    predictions. This is what stops us acting on the 40-62%
                    oscillating phase before the day's value has formed.

  within_window  -- now is at/before the personal cutoff instant for the current
                    climate day (Manila `personal_cutoff`, ~LA late-morning). A
                    confident pick that only arrives after this is "too late to
                    be useful", so we gate it out live.

Metric-agnostic: everything keys off the stored per-degree pmf + the config, so
the identical gate serves the daily-high and daily-low markets.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.buckets import argmax_pair, pmf_from_buckets
from highcast.config import AppConfig
from highcast.timeutil import climate_day_bounds


def morning_obs_ready(conn, cfg: AppConfig, spec, target: str,
                      as_of: str | datetime | None = None) -> bool:
    """Does a day-ahead prediction for `target` have any morning observation
    behind it, or is it running the untrained no-morning regime?

    WHY THIS GATE EXISTS (2026-07-26, caught live on real money). The desk rolls
    to tomorrow once today's window closes. But the day-ahead model is trained
    only on rows that HAVE morning observations, and tomorrow's climate day has
    not started, so `features._morning_features` returns all-NaN for every
    `morn_*` column. The model then extrapolates into a regime it never saw:
    for 2026-07-26 it emitted 73.2F with a 15.3F p10-p90 band while all nine
    guidance models sat at 75.6-89.1 (mean 81.8). The staking gate compared that
    82% on "<=78" against the market's 22%, read a "+60% edge", and said PLACE.

    An edge that large against a liquid market is virtually always your own
    model breaking, not free money. The gate cannot tell a good number from a
    broken one, so it must not be handed a number from an unvalidated regime.

    Mirrors `features._morning_features`' window exactly: the high bounds the
    lookback at the climate-day start, the low uses a 24h lookback ending at its
    prior-evening cutoff. Fails CLOSED -- no observation, no trade.

    `as_of` MUST be the prediction's own `issued_at_utc`, not "now". The first
    version of this gate checked whether morning observations exist *currently*,
    which is the wrong instant and let the bug straight back through: at 03:52
    PDT on 2026-07-26 the morning METARs had arrived, so the gate opened -- but
    the day-ahead prediction being traded was issued at 18:45 PDT the previous
    evening, 9 hours before those observations existed, and was still the
    all-NaN 73.2F one. What matters is the data the model actually saw, so the
    window closes at min(cutoff, as_of).
    """
    from highcast.features import _cutoff_utc

    d = datetime.fromisoformat(target).date()
    cutoff = _cutoff_utc(cfg, d + timedelta(days=spec.cutoff_day_offset),
                         spec.cutoff_local_hour)
    if spec.cutoff_day_offset == 0:
        start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    else:
        start = cutoff - timedelta(hours=24)
    upto = cutoff
    if as_of is not None:
        issued = (datetime.fromisoformat(as_of) if isinstance(as_of, str)
                  else as_of)
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
        upto = min(cutoff, issued)
    if upto < start:
        return False
    row = conn.execute(
        """SELECT 1 FROM obs WHERE station=? AND source='metar'
           AND valid_utc >= ? AND valid_utc <= ? LIMIT 1""",
        (cfg.station.icao, start.isoformat(), upto.isoformat())).fetchone()
    return row is not None


# NOTE — a veto named `intraday_hour_trained` briefly lived here (2026-07-26).
# It blocked any nowcast issued outside CHECKPOINT_HOURS (8..20), to stop the
# live false LOW signal generated at local hour 3, where the `hour` feature was
# extrapolated and the model wrongly treated the minimum as already locked.
#
# It was REMOVED, for two reasons:
#   1. It was wrong at the boundary. The LOW's own decision hour is 06:00 PDT,
#      below CHECKPOINT_HOURS' minimum, so the veto blocked the LOW market
#      permanently -- a worse failure than the one it prevented.
#   2. The underlying defect is now fixed at the source: `cli.cmd_predict --live`
#      trains at the hour it predicts at, so no hour is extrapolated. Measured
#      gain at the very hour that produced the bad signal: LOW @3 MAE
#      1.719 -> 1.016 (dMAE -0.703 [-0.811,-0.593]).
#
# Do not reintroduce an hour-range veto without first checking it admits each
# metric's decision hour (LOW 06:00, HIGH 09:00-10:30 station-local).
#
# 2026-07-26 (later the same day): an hour gate IS back below, as
# `decision_hour_ready` -- but it is the opposite shape. The old veto banned a
# RANGE the model was untrained on. This one bans everything BEFORE the hour at
# which the model's answer was measured to stop moving, and the metric cutoffs in
# config moved with it so each market's own decision hour is still admitted. That
# check is the one the note above demands; `tests/test_decision_hour.py` asserts
# it so the pair can never drift back out of sync.


# Per-hour out-of-sample accuracy of the matched-hour nowcast, measured by
# `scripts/hour_curve.py` (holdout: train < 2025-01-01 <= test, N=569 days at
# every hour except 1, which has 194). `hit` is the fraction of days the model's
# MODAL Kalshi 2F bucket contained the settled CLI value -- i.e. "if I bet the
# bucket it told me, how often was it right?", which is what a bucket bettor
# actually consumes. `mae` is against the same settled values.
#
# Both curves are MONOTONE in the hour, for a physical reason: accuracy steps up
# when the extremum stops being a forecast and becomes an observation.
#   LOW  jumps 10 -> 11 (mae 0.560 -> 0.390, hit 77.3% -> 84.7%) because the 18Z
#        6-hour-minimum group lands at 17:53Z = 10:53 local. Exactly the payoff
#        HANDOFF-2026-07-26.md predicted for the `2sTTT` fix ("settlement
#        certainty from 11:00 PDT on") -- now quantified.
#   HIGH jumps 16 -> 17 (mae 0.452 -> 0.056) once the afternoon max is past.
#
# Single holdout, NOT walk-forward: these absolutes are not comparable to the
# walk-forward figures in HANDOFF.md. The across-hour comparison is the finding,
# and it is paired on matched test dates.
# Keyed by ICAO station first (2026-07-28, adding KMIA plumbing ahead of its
# data): every number below is LAX-specific -- measured by `scripts/hour_curve.py`
# against KLAX's own climate-day boundaries and CLI reports. A second station's
# curve is NOT a copy of this one with a different label; it needs its own
# `scripts/hour_curve.py` run once there is settled data to measure against. Until
# that exists there is deliberately no "KMIA" key here -- see the accessors below,
# which refuse to fall back to KLAX's numbers for an unmeasured station.
HOUR_ACCURACY: dict[str, dict[str, dict[int, tuple[float, float]]]] = {
    # KMIA — measured 2026-07-28 from `scripts/hour_curve.py both 2000
    # config/kmia.yaml`, single holdout train<2025-01-01, n=564-567/hour,
    # bucket hit PARITY-AVERAGED (see buckets.expected_bucket_hit). Hour 1 is
    # omitted deliberately: only 194 test days, versus 564+ everywhere else.
    #
    # THE HIGH IS FLAT ALL NIGHT — 53.2% at hour 2 and 55.6% at hour 11, with
    # nothing in between beating either. Miami's max is convective, so nothing
    # observable before late morning constrains it; there is no marine-layer
    # analogue to read off the overnight trace the way KLAX's LOW does.
    "KMIA": {
        # Re-measured 2026-07-31 with `scripts/hour_curve.py` AFTER fixing that
        # script's `features_for(icao, ...)` NameError -- it had crashed on every
        # run since the feature-set fix, so the numbers it was supposed to refresh
        # never were. KMIA's feature set is unchanged (`INTRADAY_FEATURES`), so
        # every move here is noise on a slightly larger test set (n 564->570).
        "low": {
            2: (1.448, 0.462), 3: (1.392, 0.499), 4: (1.299, 0.530), 5: (1.211, 0.567), 6: (1.123, 0.610), 7: (1.013, 0.663),
            8: (0.789, 0.795), 9: (0.798, 0.802), 10: (0.789, 0.803), 11: (0.766, 0.808), 12: (0.741, 0.811), 13: (0.708, 0.823),
            14: (0.659, 0.842), 15: (0.630, 0.849), 16: (0.606, 0.855), 17: (0.578, 0.861), 18: (0.559, 0.862), 19: (0.513, 0.868),
            20: (0.438, 0.887), 21: (0.384, 0.893), 22: (0.320, 0.899), 23: (0.233, 0.913),
        },
        "high": {
            2: (1.146, 0.531), 3: (1.157, 0.520), 4: (1.148, 0.516), 5: (1.152, 0.532), 6: (1.133, 0.536), 7: (1.143, 0.529),
            8: (1.150, 0.526), 9: (1.122, 0.531), 10: (1.109, 0.539), 11: (1.045, 0.561), 12: (0.910, 0.603), 13: (0.764, 0.676),
            14: (0.502, 0.797), 15: (0.438, 0.821), 16: (0.402, 0.834), 17: (0.383, 0.837), 18: (0.378, 0.836), 19: (0.271, 0.887),
            20: (0.084, 0.986), 21: (0.119, 0.986), 22: (0.159, 0.986), 23: (0.072, 0.987),
        },
    },
    "KLAX": {
        # hour: (mae, modal-bucket hit rate)
        # LOW re-measured 2026-07-26 after the MOS-min backfill, the trailing-bias
        # correction, and the NBM-bulletin channel. Cumulative change at the
        # pre-formation hours -- the ones with no observations to lean on, and the
        # only ones where "forecast" rather than "read the thermometer" applies:
        #     h2  MAE 1.113 -> 1.052   hit 52.4% -> 55.7%
        #     h3  MAE 1.016 -> 0.979   hit 57.5% -> 57.8%
        #     h4  MAE 0.901 -> 0.885   hit 60.1% -> 62.4%
        #     h5  MAE 0.794 -> 0.772   hit 65.6% -> 65.7%
        #     h6  MAE 0.744 -> 0.724   hit 67.0% -> 68.4%
        #     h9  MAE 0.710 -> 0.689   hit 68.9% -> 69.6%
        #     h10 MAE 0.560 -> 0.533   hit 77.3% -> 78.6%
        # The knee stays at 11 (84.7%): the 18Z 6-hour-min group still dominates
        # anything guidance can contribute, so MIN_DECISION_HOUR is unchanged.
        # Re-measured 2026-07-31 (see the KMIA note above -- same run, same fix).
        # THE HIGH IS THE ONE THAT MOVED: h11 hit 52.4% -> 53.4%, MAE 1.129 ->
        # 1.063, because KLAX HIGH is the only market carrying `RATE_FEATURES`
        # and this is the first `hour_curve` run that actually measured them.
        # That confirms the +2.5 pts the ship gate saw and the +1.6 the
        # walk-forward saw. The LOW moved at noise level, as expected.
        "low": {
            1: (1.689, 0.384), 2: (1.056, 0.551), 3: (0.974, 0.574), 4: (0.881, 0.614),
            5: (0.777, 0.652), 6: (0.727, 0.682), 7: (0.705, 0.689), 8: (0.689, 0.698),
            9: (0.683, 0.706), 10: (0.535, 0.775), 11: (0.386, 0.846), 12: (0.386, 0.846),
            13: (0.383, 0.846), 14: (0.381, 0.846), 15: (0.379, 0.847), 16: (0.372, 0.848),
            17: (0.375, 0.848), 18: (0.368, 0.849), 19: (0.365, 0.850), 20: (0.334, 0.859),
            21: (0.308, 0.868), 22: (0.281, 0.881), 23: (0.228, 0.902),
        },
        "high": {
            1: (2.016, 0.307), 2: (1.465, 0.420), 3: (1.485, 0.409), 4: (1.453, 0.429),
            5: (1.439, 0.436), 6: (1.424, 0.422), 7: (1.446, 0.429), 8: (1.392, 0.417),
            9: (1.358, 0.447), 10: (1.241, 0.490), 11: (1.063, 0.534), 12: (0.866, 0.603),
            13: (0.711, 0.695), 14: (0.646, 0.717), 15: (0.622, 0.738), 16: (0.425, 0.821),
            17: (0.053, 0.988), 18: (0.056, 0.987), 19: (0.046, 0.987), 20: (0.048, 0.988),
            21: (0.040, 0.987), 22: (0.026, 0.994), 23: (0.025, 0.995),
        },
    },
}

# The earliest station-local hour at which a nowcast may be acted on.
#
# ⚠️ THESE WERE MOVED EARLIER ON PURPOSE (2026-07-26, second revision). They were
# briefly set to each metric's ACCURACY knee -- low 11, high 16 -- which scored
# 84.7% / 83.3%. Those numbers were near-worthless as forecasts: measured against
# the CLI report's own `high_time`/`low_time`, at hour 16 the HIGH's max had
# ALREADY OCCURRED on 96% of days, and at hour 17 (98.8% "accuracy") on 98%. The
# desk was transcribing a published ASOS 6-hour group, not predicting anything,
# and a bucket the market has already read carries no edge either.
#
# The goal is a forecast issued BEFORE the extremum exists. So each hour below is
# the best accuracy available while the extremum is still mostly in the FUTURE
# ("future" = share of days the extremum had not yet occurred at that hour):
#
#   metric  hour  Manila   still future   bucket hit
#   LOW       4   19:00        85%          61.4%
#   HIGH     11   02:00        88%          53.4%
#
# (hit rates refreshed 2026-07-31 from the repaired `hour_curve.py`; the HIGH
# gained a point from `RATE_FEATURES`, the LOW moved at noise level. Neither
# moves the knee, so the hours below are unchanged.)
#
# For comparison, the hours that were previously gated to: LOW 11 -> 29% future,
# HIGH 16 -> 4% future. Those are scorekeeping hours, not forecasting hours.
#
# WHY NOT EVEN EARLIER: LOW hour 3 is 92% future but only 57.8%, and hour 1 falls
# off a cliff (37.1%, and n=194). HIGH hour 9 is 99% future but 45.7%. Hour 4 /
# hour 11 are where predictive content and accuracy are jointly best; shifting
# either by an hour trades ~3-5 points of accuracy for ~5-10 points of "still
# future". Revisit with `scripts/hour_curve.py` if the model improves at the
# early hours -- that is exactly the work this gate is meant to expose, not hide.
# Keyed by ICAO station first -- same reasoning as HOUR_ACCURACY above: these
# hours were derived FROM that station's curve, so they cannot be reused across
# stations. No "KMIA" entry until KMIA has its own measured curve.
# Calibration offset at each COMMIT hour: mean stated probability on the modal
# bucket MINUS its realised hit rate. Positive means the model OVERSTATES and the
# number must be shaded down before it is used as a probability.
#
#     p_hat = p_model - HOUR_CALIB[station][metric][hour]
#
# Measured 2026-07-30 by `scripts/calib_eval.py 30 2025-01-01` -- walk-forward,
# refit 30, matched-hour at each station's configured bandwidth, bucket hit
# PARITY-AVERAGED. The run takes ~30 minutes (~240 LightGBM refits), so its
# per-record output is kept at `data/calib_records-20260730.csv` (7,236 rows:
# station, metric, hour, target_date, model_pt, y, stated, hit). Load that rather
# than recomputing -- and note the script also writes /tmp, which does not survive.
#
# ⚠️ THESE ARE THE **RECENT-WINDOW** VALUES (test dates >= 2025-01-01) AND THAT
# CHOICE IS LOAD-BEARING. An expanding-window walk-forward starts at MIN_TRAIN=400
# rows and averages in early folds where the model held a fraction of the ~1,800
# production trains on today, so its full-window figure is PESSIMISTIC for the
# shipped model. The gap is not small:
#
#     market          full window    recent window
#     KLAX low  h4      +4.2%          +0.8%  (ns)
#     KLAX high h11     +8.9%          +7.4%
#     KMIA low  h5      +5.6%          +1.5%  (ns)
#     KMIA high h11    +10.6%          +3.9%
#
# Acting on the full-window numbers would have shaded every LOW pick ~3.5 points
# too hard, and would have "corrected" HANDOFF's +0.7% for the KLAX LOW -- which
# was RIGHT. The recent-window values also reproduce the independent single-holdout
# figures already recorded in MIN_DECISION_HOUR's own comments below (KMIA high
# +4.3% vs +3.9% measured, low +1.2% vs +1.5%), which is the cross-check that
# settles which regime to trust.
#
# WHAT THIS SAYS ABOUT THE MODEL: both LOWs are well calibrated (CIs contain zero)
# and both HIGHs are significantly overconfident. That is the structural asymmetry
# HANDOFF describes -- the HIGH is read with 88-95% of its answer still in the
# future, while the LOW's commit hour has the ratchet constraining it.
#
# NOT A MODEL CHANGE. Nothing in `features`, `models/` or `staking` reads this.
# It exists so display surfaces and the Pick-of-the-Day scorer can quote a
# probability we actually believe. Shading a number for display is not retuning.
#
# NO BAND TERM. An earlier draft applied a larger offset to KLAX HIGH readings
# >=80F. Measured, that band's calib is +8.6% [-0.2%,+17.9%] (n=81) against the
# <80 band's +8.9% [+6.8%,+11.1%] -- statistically identical, CI containing zero.
# The model is uniformly overconfident, not specially so up high. The >=80 band IS
# harder (hit 42.0% vs 50.6%) but that belongs on the display, not in this
# arithmetic.
HOUR_CALIB: dict[str, dict[str, dict[int, float]]] = {
    # ⚠️ RE-KEYED AND RE-MEASURED 2026-08-01, AT THE NEW COMMIT HOURS.
    # `HOUR_CALIB` is keyed by hour, so the decision-hour move orphaned every
    # entry. Left stale it returns None, which crashed `band_calib_eval.py`
    # outright -- the ordering dependency is real: this table must be refreshed
    # BEFORE the band table can be derived.
    #
    # THE HEADLINE: the overconfidence the calibration existed to correct has
    # LARGELY GONE, because the later hour is honest as well as more accurate.
    # Three of four markets are now slightly UNDER-confident:
    #
    #     market          old (old hour)   new (new hour)
    #     KLAX low          +0.0075 @h4      -0.0097 @h6
    #     KLAX high         +0.0671 @h11     +0.0145 @h13
    #     KMIA low          +0.0151 @h5      -0.0175 @h7
    #     KMIA high         +0.0394 @h11     -0.0281 @h14
    #
    # Measured by `scripts/calib_eval.py 30 2025-01-01`, RECENT window (the
    # shipped convention: the expanding-window FULL column is pessimistic about
    # the model production actually runs). n=567-574 per market.
    "KLAX": {
        # -0.0097, CI [-3.5%,+1.5%] -- NOT significant, and now slightly
        # UNDER-confident. A negative offset shades the probability UP, so it is
        # kept as the point estimate only because it is a near no-op; it must not
        # be quoted as evidence the LOW is under-stating itself. n=574.
        "low": {6: -0.0097},
        # +0.0145, CI [-1.1%,+4.0%] -- NO LONGER SIGNIFICANT. At h11 this was
        # +0.0671 [+3.8%,+9.7%], i.e. states 59.7% and delivers 53.0%. Moving to
        # h13 cut the overconfidence by a factor of ~4.6 and the CI now spans
        # zero: states 68.5%, delivers 67.9%. The hour move did what a
        # calibration layer was doing, at the source. n=574.
        "high": {13: 0.0145},
    },
    "KMIA": {
        # -0.0175, CI [-4.4%,+1.0%] -- NOT significant. n=567.
        "low": {7: -0.0175},
        # -0.0281, CI [-5.1%,-0.5%] -- significant, and NEGATIVE: at h14 the
        # model now UNDERSTATES its modal bucket (states 75.9%, delivers 79.8%).
        # This is the market the bot trades, and being under-confident is the
        # safe direction to be wrong in.
        # h13 measured 2026-08-05 from `data/calib_records-kmia-high-h13.csv`
        # (2,263 walk-forward records, n=574 in the recent window) when the
        # commit hour moved 14 -> 13. The h14 entry is KEPT: it is still a
        # true measurement of that hour, `hour_calib` is looked up per hour,
        # and the test bench compares the two.
        "high": {13: -0.0067, 14: -0.0281},
    },
}


# ── BAND CALIBRATION ────────────────────────────────────────────────────────
#
# WHY THE SCALAR ABOVE IS NOT ENOUGH, AND WHY IT COST REAL MONEY.
# `HOUR_CALIB` is one number per market. KLAX LOW's is +0.0075 -- essentially
# honest -- and for BETTING that average is a lie, because the curve under it is
# not flat (measured on the same 7,236 walk-forward records, recent window):
#
#     KLAX LOW h4    model says 60-70%  ->  hits 67.9%   UNDER-states by 2.9%
#                    model says 70-80%  ->  hits 67.4%   OVER-states  by 6.8%
#
# The two cancel to ~zero. But `staking.evaluate` only fires when the model
# claims an edge over the market, so it bets almost exclusively out of the TOP
# band -- the one band that overstates. Correcting by the average applies a
# ~0-point shave to a number needing ~7.
#
# THE 70-80% OVERSTATEMENT IS PRESENT IN ALL FOUR MARKETS AND SIGNIFICANT IN
# ALL FOUR -- KLAX low +6.8% [+1.6,+12.1], KLAX high +12.8% [+5.3,+20.4],
# KMIA high +9.5% [+2.2,+17.0], KMIA low +8.3% [+0.7,+16.0]. Four independent
# markets agreeing is why this is a curve and not an artefact.
#
# CAUGHT LIVE: on 2026-07-29/30 the app placed three bets at a stated 81% and
# lost all three. Reproduce with `scripts/band_calib_eval.py`.
#
# ⚠️ OUT-OF-SAMPLE RESULT, AND IT IS MIXED. `scripts/band_calib_oos.py` fits the
# bands on 2025-01-01..2025-10-15 and scores them on 2025-10-16..2026-07-28,
# against the shipped scalar. Weighted |stated - hit|:
#
#     market        RAW     SCALAR    BAND
#     KLAX high   0.0726    0.0525   0.0186   <- BAND much better
#     KLAX low    0.0307    0.0315   0.0187   <- BAND much better
#     KMIA high   0.0554    0.0264   0.0299   <- BAND slightly WORSE
#     KMIA low    0.0217    0.0064   0.0084   <- BAND slightly WORSE
#     pooled      0.0451    0.0292   0.0189
#
# BAND wins pooled and fixes KLAX HIGH's 7-point raw error decisively, but it is
# NOT uniformly better: on the two KMIA markets, where the scalar was already
# near-perfect, banding adds variance and costs 0.2-0.4 points. Brier is a tie
# throughout (0.1296 vs 0.1293). Shipped everywhere anyway, because the pooled
# gain is real, the KMIA losses are inside noise at n~280, and the >=0.80
# REFUSAL -- which is what actually blocks the losing bets -- is wanted in all
# four markets regardless of which offset is better. Do not quote this as
# "band calibration is better everywhere"; it is not.
#
# ⚠️ NOTHING IS MEASURED ABOVE 80%. The top band holds 2-15 records per market
# (KLAX LOW: n=2, which hit 50%), so no offset ships there and `band_calib`
# returns None -- the same no-silent-fallback contract `hour_calib` keeps. A
# caller that gets None MUST refuse, not fall back to the scalar: those three
# losing bets were all stated 81%, i.e. from exactly this unmeasured region.
#
# Each entry is (upper bound of the band, offset to SUBTRACT from p_model).
BAND_CALIB: dict[tuple[str, str, int], list[tuple[float, float]]] = {
    # ⚠️ RE-DERIVED 2026-08-01 AT THE NEW COMMIT HOURS, from
    # `data/calib_records-20260801.csv` (7,246 walk-forward records, verified
    # 0 duplicate keys) via `scripts/band_calib_eval.py`.
    #
    # WHY IT HAD TO BE: this table is keyed (station, metric, hour), so the
    # decision-hour move orphaned every entry and `band_calib()` returned None --
    # i.e. REFUSE -- for all four markets. That is the correct failure direction
    # and it is why the suite sat at 26 failures between the two changes, but it
    # leaves the desk inert, so the pair is atomic.
    #
    # THE RESULT THAT MATTERS FOR THE BOT: KMIA HIGH's >=80% band is now
    # MEASURED AND SHIPS -- n=244, stated 87.8%, hits 93.4%, overstatement
    # -5.6% CI [-7.9%,-3.2%]. At h11 that band held n<=15 and always returned
    # None, so the market could never be acted on at high confidence. At h14 it
    # is the best-evidenced band in the project, and it UNDER-states.
    #
    # KLAX HIGH's top band went the other way: +0.1088 at >=80% (n=76), i.e. it
    # still overstates badly up there. Consistent with the spec excluding KLAX
    # HIGH from the bot on two independent grounds.
    #
    # Each entry is (upper bound of the band, offset to SUBTRACT from p_model).
    # A band absent from a market's list is UNMEASURED -> band_calib returns
    # None -> the caller must refuse, never fall back to the raw number.
    ("KLAX", "high", 13): [
        (0.6, -0.0532),    # n=74
        (0.7, -0.0346),    # n=190
        (0.8, +0.0472),    # n=217
        (1.01, +0.1088),   # n=76  -- still badly overconfident up high
    ],
    ("KLAX", "low", 6): [
        (0.6, -0.0473),    # n=66
        (0.7, -0.0133),    # n=235
        (0.8, -0.0052),    # n=233
        # NO >=80% BAND: n=13 there, too thin to measure. Refuses by design.
    ],
    # h13 — THE COMMIT HOUR since 2026-08-05. Derived from
    # `data/calib_records-kmia-high-h13.csv` by the same arithmetic that
    # produced the h14 rows below, validated first by reproducing those rows
    # exactly from the cached 2026-08-01 CSV.
    #
    # The shape differs from h14 and the difference is informative: at h13 the
    # model OVER-states below 60% (+0.0373, n=156) and UNDER-states in the
    # 60-70% band (-0.0585, n=177). The top band is nearly honest (-0.0049,
    # n=78) rather than the large under-statement h14 shows -- consistent with
    # h13 being a genuine forecast (72.8% of maxima still ahead) where h14 is
    # half a reading (49.7%).
    ("KMIA", "high", 13): [
        (0.6, +0.0373),    # n=156 -- OVER-states down here; shave it
        (0.7, -0.0585),    # n=177
        (0.8, +0.0064),    # n=163
        (1.01, -0.0049),   # n=78
    ],
    # h14 KEPT after the commit hour moved to 13. Still a true measurement of
    # h14, still what the test bench compares against, and `band_calib` is
    # looked up per hour so a stale key cannot be read by accident.
    ("KMIA", "high", 14): [
        (0.7, +0.0187),    # n=87
        (0.8, -0.0079),    # n=187
        (1.01, -0.0560),   # n=244 -- significant UNDER-statement, ships
    ],
    ("KMIA", "low", 7): [
        (0.5, -0.0276),    # n=90
        (0.6, -0.0323),    # n=120
        (0.7, -0.0001),    # n=161
        (0.8, +0.0240),    # n=88
        (1.01, -0.0525),   # n=108
    ],
}


def band_calib(station: str, metric: str, hour: int,
               p_model: float) -> float | None:
    """Calibrated probability for `p_model`, or None where its band is unmeasured.

    None means REFUSE, never "use the raw number". It is returned in two cases,
    both of which are a claim the evidence does not support:
      * the market/hour has no band table at all
      * `p_model` sits above the highest measured band (>= 0.80 everywhere
        today, where every market has n <= 15 records)

    The result is clamped to (0, 1): a band offset larger than the probability
    itself would otherwise produce a negative "probability" that Kelly sizing
    would happily consume.
    """
    bands = BAND_CALIB.get((station, metric, hour))
    if bands is None:
        return None
    for upper, offset in bands:
        if p_model < upper:
            return min(0.999, max(0.001, p_model - offset))
    return None


# The CALIBRATED probability a market must clear before it may be acted on.
#
# THIS IS A POLICY TARGET, NOT A FITTED PARAMETER: act only when measurably right
# 4 days in 5. Per market the floor is the LOWEST stated-probability threshold
# whose realised hit rate has a 95% lower bound >= 0.80. A market with no such
# threshold is ABSENT here, and an absent market never fires.
#
# Measured 2026-07-31, walk-forward, n~570/arm:
#     KLAX HIGH h13  .50 69.5/65.6  .60 70.3/66.1  .70 71.4/65.8  .80 75.0/63.9
#     KMIA HIGH h14  .50 80.5/76.9  .60 82.1/78.6  .70 85.5/81.9  .80 93.4/89.6
#
# ⚠️ KLAX HIGH IS DELIBERATELY ABSENT. No threshold reached a 0.80 lower bound.
# It is independently excluded by price too: at h13 its favoured bucket traded
# 0.75-0.96 (median ~0.90) against a model that can justify only ~69c, and the
# market's favoured bucket was right 5 of 6. Do NOT lower the target to admit it.
#
# The LOWs are absent because Julian trades them by hand; this floor exists for
# the unattended path only.
# Minimum CALIBRATED probability the bot's chosen bucket must clear.
#
# 2026-08-02: KMIA HIGH lowered 0.70 -> 0.60, measured on 668 Jun-Sep
# walk-forward days at h14 (n=1336 day-ladders).
#
# ⚠️ MEASURED ON THE QUANTITY THE BOT ACTUALLY USES. Kalshi's ladder has ONE
# parity on any given day, so this scores each parity SEPARATELY and pools the
# day-ladders. An earlier cut used `expected_bucket_hit`, which AVERAGES the two
# parities -- correct for scoring the model, but kinder than the bot's reality,
# because a good odd-parity read rescues a bad even-parity one. That averaging
# reported the model as +14.9 points UNDER-confident at 50-60%. On the real
# quantity it is -4.2 points OVER-confident. The conclusion inverted:
#
#     stated band   n     mean stated   actual    gap
#       50-60%     168       54.8%      50.6%    -4.2%
#       60-70%     155       65.3%      61.3%    -4.0%
#       70-80%     154       75.4%      78.6%    +3.2%
#       90-100%    558       93.4%      96.6%    +3.1%
#
# WHY 0.60 AND NOT LOWER. The marginal band is what a floor admits, not the
# cohort average. At 0.60 the added days hit 61.3%; below it the 50-60% band
# hits 50.6% with the worst over-confidence on the board -- that is where the
# model stops being reliable. Holds out of sample: the >=0.60 cohort hit 86.9%
# on 2020-24 and 87.5% on 2025-26.
#
# WHAT THE FLOOR IS NOT. It is not what protects against a bad price. The edge
# rule (model > market + 3%) is the binding constraint on whether a trade is
# worth taking; this is a second gate that keeps the bot away from days where
# the model itself is unreliable regardless of price.
CONFIDENCE_FLOOR: dict[tuple[str, str, int], float] = {
    # h13 — THE COMMIT HOUR since 2026-08-05. Measured the way this table's
    # policy specifies: the lowest stated-probability threshold whose realised
    # hit rate has a 95% bootstrap lower bound >= 0.80.
    #
    #     >=0.75  n=148  hit 0.807  lb 0.760
    #     >=0.80  n= 78  hit 0.865  lb 0.801   <- the floor
    #     >=0.85  n= 35  hit 0.886  lb 0.800
    #
    # ⚠️ IT PASSES BY 0.001. That is a real pass on the stated rule and not a
    # judgement call, but it is thin, and it should be RE-MEASURED rather than
    # trusted when more h13 days settle. If it drops below the target the
    # correct response is to remove this line -- an absent market never fires
    # -- NOT to lower the 0.80 target to keep it.
    #
    # THE FLOOR IS MUCH STRICTER AT h13: 0.80 admits 78 of 574 days (13.6%),
    # where h14's 0.60 admitted far more. That is the honest trade for the
    # cheaper entry -- fewer days qualify, but the ones that do are priced
    # where a win is worth having.
    ("KMIA", "high", 13): 0.80,
    # h14 kept: still true of that hour, and looked up per hour.
    ("KMIA", "high", 14): 0.60,
}


def confidence_floor_ok(station: str, metric: str, hour: int,
                        p_hat: float) -> bool:
    """Does the CALIBRATED probability clear this market's measured floor?

    Fails closed on every unknown: an unmeasured market, the wrong hour, a
    missing or NaN probability. For an unattended caller "I could not tell" and
    "no" must be the same answer.

    `p_hat` must already be calibrated (`band_calib`), never the raw p_model --
    the raw number is what was 6.7 points overconfident at KLAX HIGH.
    """
    floor = CONFIDENCE_FLOOR.get((station, metric, hour))
    if floor is None:
        return False
    if p_hat is None:
        return False
    try:
        p = float(p_hat)
    except (TypeError, ValueError):
        return False
    if p != p:            # NaN
        return False
    return p >= floor


# ── MAXIMUM EDGE ─────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS. `orderbot` has always had a MINIMUM edge rule (calibrated
# must clear ask + fee + margin) and no ceiling. The project's own evidence says
# a very LARGE edge is usually a symptom of OUR error, not free money:
#
#   * HANDOFF-2026-08-03.md §0, on the 16 graded `app_bets`: edge >= +3% (the
#     rule that says BET) went 4/8; no-edge (SKIP) went 8/8. Edge correlates
#     -0.619 with WINNING; market price correlates +0.496. We lose precisely
#     where we think we found value.
#   * HANDOFF §4.2: a KLAX LOW bucket showed a +73% edge at 0.5c on
#     2026-08-01. Not an edge -- the bucket was already dead. The true minimum
#     had happened between hourly filings and was visible only in the 6-hour
#     `2sTTT` remark group; Kalshi traders watch 1-minute ASOS and already had
#     it priced. Our "edge" was staleness wearing a value bet's clothes.
#
# When our model disagrees with a liquid market by a wide margin, the base
# rate strongly favours OUR MODEL being broken (stale bound, missed obs,
# revised source data, a forecast product we don't consume) -- not the market
# missing something we alone can see.
#
# MEASURED (copy of `data/highcast.db`, 2026-08-02; never analysed against the
# live file -- see the order bot's data-safety note):
#
#   `signals`, ALL rows, edge = p_model - price (n=5238, every station /
#   metric / hour / committed-state pooled, most of which the bot never acts
#   on): p50 +0.06, p90 +0.70, p99 +0.81, max +0.83. Not the right population
#   -- it is dominated by early-hour noise before any pick has stabilised.
#
#   `signals`, KMIA HIGH ONLY, restricted to the bot's actual decision point
#   (station-local hour 14, the only hour `orderbot.decide` ever runs at):
#   n=60 readings across 5 distinct days (2026-07-28..08-01).
#       p50 +0.01   p90 +0.24   p99 +0.33   MAX EVER SEEN +0.35 (07-29)
#   Every legitimate KMIA HIGH h14 edge on record, across the whole capture
#   window, sits at or below +0.35.
#
#   `app_bets` (n=20, all 4 markets, all graded): edges >= +0.43 are 3 losses
#   and 1 win. The one win (+0.45, KMIA low) is the "entirely one 11c winner"
#   HANDOFF §0 flags as unreliable evidence on its own. The worst loss
#   (+0.72) is the §4.2 stale-bucket case above.
#
# So: ordinary KMIA HIGH edges at the real decision hour never exceeded +0.35
# in the data available; the well-documented BROKEN cases run +0.43 to +0.73
# and are disproportionately losses. 0.40 sits above every legitimate reading
# on record (room to spare) and below the known-bad cluster.
#
# ⚠️ THIS IS PROVISIONAL AND MUST BE READ AS SUCH. The right way to set this
# threshold is to measure how often edges of a given size actually resolve in
# our favour -- not to eyeball a gap in a small sample, which is what the line
# above just did. That measurement needs win/loss outcomes at a range of edge
# sizes for the market the bot trades, and KMIA HIGH has market price stored
# on only 6 distinct days total (`signals`: 2026-07-28..08-02; `app_bets`: 4
# of those, all graded) -- nowhere near enough to fit a win-rate-tuned ceiling
# for THIS market. 0.40 is a plausibility bound picked from the shape of the
# data above, not a fitted optimum. Re-derive it once `bot_orders.edge` (see
# `migrate.ensure_bot_orders_edge_column`) has accumulated enough decisions to
# ask the real question: at what edge size does the bot start losing money?
MAX_EDGE = 0.40


def metric_uses_nbs(metric: str) -> bool:
    """Does this metric's day-ahead model actually consume NBS columns?

    ⚠️ ASK THE FEATURE LIST, NEVER HARDCODE "low". Today the LOW carries seven
    NBS-derived columns and the HIGH carries NONE, so vetoing the HIGH on a
    missing NBS feed would be a FALSE VETO -- blocking a sound prediction over
    data it does not read. (My first version of the guard did exactly that.)
    Deriving it from `model_features` also means the guard follows automatically
    if the HIGH ever gains an nbs_* column.
    """
    from highcast import features
    return any("nbs" in c.lower() for c in features.model_features(metric))


def storm_warning_active(conn, cfg, target: str) -> bool:
    """Is a convective warning in force for this station's climate day?

    THE WEATHER GUARD. A temperature model has no precipitation input; a
    thunderstorm is a regime it cannot represent, and the failure is not subtle.

    CAUGHT LIVE 2026-08-01, KMIA:

        19:53Z  15:53 EDT   87.98F   -TSRA
        20:43Z  16:43 EDT   75.02F   +TSRA BR     <- 16F in 50 minutes

    The low fell from a settled-looking 82 to 74 and destroyed the >=81 pick.
    The model said 42% and the market 64% -- both wrong, because neither
    instrument prices an outflow boundary. The right response to "a storm is
    warned" is not a different number, it is NOT TRADING: the distribution the
    model emits is conditioned on a world without one.

    Reads only STORM_EVENTS. `heat_alert` reads only HEAT_EVENTS. The two share
    the `alerts` table and must never share a meaning -- see
    `features._alert_flag`, which had no event filter until the storm products
    were captured on 2026-08-02.

    Fails CLOSED via its caller: `decision_gates` wraps this in try/except and
    treats an exception as a refusal.
    """
    from datetime import date as _date

    from highcast.fetchers.alerts import STORM_EVENTS

    day = _date.fromisoformat(target)
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    marks = ",".join("?" * len(STORM_EVENTS))
    row = conn.execute(
        f"""SELECT 1 FROM alerts WHERE station=? AND event IN ({marks})
              AND onset_utc < ?
              AND (expires_utc IS NULL OR expires_utc = '' OR expires_utc > ?)
            LIMIT 1""",
        (cfg.station.icao, *sorted(STORM_EVENTS),
         end.isoformat(), start.isoformat())).fetchone()
    return row is not None


def nbs_guidance_ready(conn, cfg, target: str) -> bool:
    """Is there an NBS row for `target`, i.e. are the nbs_* features real?

    WHY THIS GUARD EXISTS -- CAUGHT LIVE 2026-07-31. The `nbm_nbs` fetch stopped
    returning rows after 2026-07-26 (KLAX) and 2026-07-28 (KMIA). The DAY-AHEAD
    LOW consumes SEVEN NBS-derived columns -- fc_nbm_nbs, fcc_nbm_nbs,
    nbs_sol_mean, nbs_sky_mean, nbs_cig_min, nbs_offshore, nbs_tsd_mean -- so
    from the first missing day it was running on NaN in a third of its inputs,
    a regime it was never trained on. The output was not subtly wrong:

        target      day-ahead LOW    every guidance source    settled
        2026-07-29     66.6 / 67.8      66.0 - 68.7             68
        2026-07-30     60.4 / 61.4      66.0 - 69.2             67
        2026-07-31     61.5             65.0 - 68.0             ?

    A 5-7F cold bias against unanimous guidance, which on the live ladder became
    a 35% probability on "<=60F" -- in July, at a station whose record low for
    the date is 57. `staking` was recommending 40 contracts of it at 1c.

    This is the SAME failure the `morn_*` check already catches for the HIGH:
    features silently going NaN and the model extrapolating. The HIGH had a
    guard and the LOW did not.

    Deliberately checks only for PRESENCE, not freshness. A stale NBS row is
    ordinary -- the feed is a few times a day, and the model trained on rows of
    exactly that age. A MISSING row is the regime change.
    """
    row = conn.execute(
        """SELECT 1 FROM forecasts
           WHERE station=? AND target_date=? AND model='nbm_nbs' LIMIT 1""",
        (cfg.station.icao, target)).fetchone()
    return row is not None


def guidance_lead_ready(conn, cfg, target: str) -> bool:
    """Is the newest guidance for `target` from YESTERDAY's run, not the day before?

    ⚠️ THIS GUARD IS PRECAUTIONARY, NOT EVIDENCED. READ BEFORE CITING IT.

    It was added 2026-07-31 on the belief that the evening `dayahead_tomorrow`
    run was seeing two-day-old guidance and extrapolating cold. **That belief was
    WITHDRAWN on 2026-08-01.** The reconstruction behind it asked
    `WHERE fetched_at <= issued_at` to decide what guidance had been visible --
    but `db.upsert` is INSERT OR REPLACE and the `forecasts` primary key EXCLUDES
    `fetched_at`, so every re-fetch overwrites that column with the latest write
    time. Rows that genuinely existed at 17:04 looked absent because their stamp
    had since been rewritten to 23:55. The apparent lead-2 was an artifact of the
    query, not a property of the feed.

    The COLD BIAS IS REAL -- it is measured against settled CLI values and owes
    nothing to `fetched_at` -- but its cause is the already-known all-NaN
    `morn_*` regime, which `morning_obs_ready` above detects. Split by that guard,
    KLAX HIGH's day-ahead error is:

        has morning obs   n=18   mean +0.31F   MAE 1.35
        no morning obs    n=13   mean -5.35F   MAE 5.35

    So this function is kept because refusing genuinely stale guidance is sound
    and it fails closed -- NOT because the lead-2 regime was ever observed here.
    Do not quote it as evidence for anything until it has been measured.

    Unlike `nbs_guidance_ready` this checks FRESHNESS, not presence: an NBS row a
    few hours old is ordinary, but guidance a whole extra run cycle old would be a
    different regime. Both fail closed.
    """
    row = conn.execute(
        """SELECT MAX(run_date) FROM forecasts
           WHERE station=? AND target_date=?""",
        (cfg.station.icao, target)).fetchone()
    newest = row[0] if row else None
    if newest is None:
        return False
    from datetime import date
    try:
        lead = (date.fromisoformat(target) - date.fromisoformat(newest)).days
    except (TypeError, ValueError):
        return False
    return lead <= 1


def hour_calib(station: str, metric: str, hour: int) -> float | None:
    """Calibration offset, or None when this station/metric/hour is unmeasured.

    NO cross-station fallback, for the same reason `HOUR_ACCURACY` has none: an
    offset measured at KLAX describes KLAX's model, and silently borrowing it
    would hand a confident-looking probability to a market nobody has measured.
    A caller that gets None must degrade visibly (the Pick-of-the-Day scorer
    reports UNSCORED), never quietly use the raw `p_model`.
    """
    return HOUR_CALIB.get(station, {}).get(metric, {}).get(hour)


MIN_DECISION_HOUR: dict[str, dict[str, int]] = {
    # Moved later 2026-07-31 by `scripts/decision_hour_move_eval.py`, a
    # pre-registered paired gate (Dhit CI excluding zero at Bonferroni x4, and
    # >=40% of the extremum still ahead so the answer stays a FORECAST).
    #
    # These hours are each station's MEDIAN PEAK-EXTREMUM hour, measured from
    # the CLI reports' own high_time/low_time (KMIA n=2366, median 13:58;
    # KLAX n=1959, median 13:05). Committing there means the extremum is a coin
    # flip between "already set" and "still to come".
    #
    # The move was made for HONESTY as much as accuracy: overall shortfall
    # (stated minus realised) went +4.8% -> -0.6% for KLAX HIGH and
    # +3.1% -> -1.9% for KMIA HIGH. That is also why no HIGH pre-trade screen
    # exists -- there is no longer a confidently-wrong regime for one to catch.
    # (Deliberately reworded from the plan's draft text, which used a term
    # this decision-path file is not allowed to contain even in a comment --
    # see the freeze-contract test in the daily-trade-quality-verdict module's
    # own test file.)
    # See docs/superpowers/specs/2026-07-31-accuracy-first-decision-hours-design.md
    "KLAX": {"low": 6, "high": 13},
    # ── KMIA HIGH MOVED 14 -> 13 on 2026-08-05, ON PRICE, NOT ACCURACY ──────
    # This is the first hour in this table chosen against what the market
    # CHARGES rather than against accuracy alone, and it fails the h14 gate
    # above on purpose. The gate has no price term; that is its blind spot.
    #
    #   walk-forward hit (n=574 matched days):  h13 68.0%   h14 79.3%
    #   mean ask on the modal bucket (7 days):  h13  0.46   h14  0.82
    #   margin (hit - ask, the break-even bar): h13  +22pt  h14  -2.7pt
    #
    # A contract at price P needs a hit rate of P to break even, so h14's
    # superior accuracy is bought at a price that already exceeds it. Measured
    # on the same 7 days, the desk's own gates (price <= ceiling AND edge >=
    # min_edge) admitted a trade on 5 of 7 days at h13 and 2 of 7 at h14 --
    # Julian's own observation that h14 "never passes the gates".
    #
    # THE ASK CLIFFS AT 14:00, it does not drift: the first signal in each
    # 15-minute slot prices the modal bucket 0.46 / 0.41 / 0.38 / 0.39 through
    # 13:00-13:45, then 0.82 at 14:00. The median peak-extremum hour is 13:58,
    # so by 14:00 the day's high has usually already happened and is readable
    # in the ASOS feed. The MARKET sets that price, not us, so the step is
    # independent evidence of a real information event.
    #
    # 13:30 WAS MEASURED AND REJECTED: h13.0 -> h13.5 is Dhit +0.9%,
    # CI [-0.8%,+2.6%] on the same 574 days -- the extra half hour buys
    # nothing. `models.intraday._split_hour` supports half hours if the cliff
    # ever moves; nothing calls it with one today.
    #
    # COST, STATED PLAINLY: 11.2 points of raw accuracy
    # (CI [-15.1%,-7.8%]), bought back with 36 cents of entry price. The
    # h14 figures stay in HOUR_CALIB / BAND_CALIB below -- they are still true
    # measurements of h14, and the test bench compares against them.
    # See docs and `scripts/decision_hour_move_eval.py` for the h14 rationale
    # this deliberately overrides.
    "KMIA": {"low": 7, "high": 13},
}


# Share of days on which the extremum had NOT yet occurred by each station-local
# hour -- i.e. how much of the number at that hour is genuinely a FORECAST rather
# than a reading of something already past. Measured from the CLI report's own
# `high_time` / `low_time` fields (n=1954 high, 1956 low).
#
# This is the axis `HOUR_ACCURACY` alone cannot show. Accuracy rises monotonically
# through the day for the trivial reason that the answer stops being unknown:
# the HIGH is 98.8% "accurate" at hour 17 with only 1.6% of maxima still ahead.
# A decision hour has to be read against BOTH numbers, which is why the desk shows
# them together.
#
# Shape note: the LOW's curve plateaus near 0.285 from hour 8 onward instead of
# decaying to ~0 like the HIGH's. That is not noise -- the KLAX climate day runs
# 01:00->01:00 local, so on ~28% of days the minimum falls in the late-evening
# tail after 20:00 rather than at dawn.
# Keyed by ICAO station first -- same reasoning as HOUR_ACCURACY above: these
# shares were measured from KLAX's own CLI `high_time`/`low_time` history, so
# they describe KLAX's climate-day shape (e.g. the LOW's late-evening tail
# below), not a curve every station shares. No "KMIA" entry until KMIA has its
# own measured shares.
EXTREMUM_STILL_FUTURE: dict[str, dict[str, dict[int, float]]] = {
    # KMIA — from the CLI report's own high_time/low_time, n=2363 days.
    "KMIA": {
        "high": {
            0: 1.000, 1: 0.997, 2: 0.993, 3: 0.991, 4: 0.991, 5: 0.991,
            6: 0.990, 7: 0.990, 8: 0.989, 9: 0.987, 10: 0.978, 11: 0.950,
            12: 0.878, 13: 0.728, 14: 0.497, 15: 0.225, 16: 0.070, 17: 0.014,
            18: 0.006, 19: 0.006, 20: 0.005, 21: 0.004, 22: 0.003, 23: 0.002,
        },
        "low": {
            0: 1.000, 1: 0.978, 2: 0.947, 3: 0.908, 4: 0.850, 5: 0.758,
            6: 0.598, 7: 0.417, 8: 0.319, 9: 0.306, 10: 0.296, 11: 0.289,
            12: 0.286, 13: 0.276, 14: 0.263, 15: 0.252, 16: 0.240, 17: 0.232,
            18: 0.222, 19: 0.208, 20: 0.195, 21: 0.183, 22: 0.171, 23: 0.152,
        },
    },
    "KLAX": {
        "high": {
            0: 0.999, 1: 0.999, 2: 0.999, 3: 0.998, 4: 0.998, 5: 0.998,
            6: 0.997, 7: 0.997, 8: 0.997, 9: 0.992, 10: 0.956, 11: 0.878,
            12: 0.739, 13: 0.517, 14: 0.294, 15: 0.112, 16: 0.038, 17: 0.016,
            18: 0.012, 19: 0.009, 20: 0.007, 21: 0.007, 22: 0.004, 23: 0.003,
        },
        "low": {
            0: 1.000, 1: 0.984, 2: 0.960, 3: 0.924, 4: 0.854, 5: 0.708,
            6: 0.499, 7: 0.352, 8: 0.303, 9: 0.291, 10: 0.288, 11: 0.287,
            12: 0.285, 13: 0.285, 14: 0.285, 15: 0.285, 16: 0.284, 17: 0.284,
            18: 0.283, 19: 0.280, 20: 0.274, 21: 0.262, 22: 0.247, 23: 0.220,
        },
    },
}


def timing_curve(station: str, metric: str) -> list[dict]:
    """Per-hour accuracy AND predictive content, for the desk's timing panel.

    Returns one row per measured hour: station-local hour, measured modal-bucket
    hit rate, MAE, the share of days the extremum is still ahead, and whether the
    hour is at/after `MIN_DECISION_HOUR`. Read-only; no DB, no model.

    Station-scoped, with NO fallback: an unmeasured station (anything but
    "KLAX" today) has no rows in `HOUR_ACCURACY`, so `acc` is `{}` and this
    naturally returns `[]` rather than borrowing KLAX's curve.
    """
    acc = HOUR_ACCURACY.get(station, {}).get(metric, {})
    fut = EXTREMUM_STILL_FUTURE.get(station, {}).get(metric, {})
    knee = min_decision_hour(station, metric)
    # knee is None for an unmeasured station -- `h >= None` raises TypeError,
    # and there is no safe hour to substitute, so every row reports not-ready.
    return [{"hour": h, "mae": acc[h][0], "hit": acc[h][1],
             "still_future": fut.get(h),
             "ready": False if knee is None else h >= knee}
            for h in sorted(acc)]


def hour_accuracy(station: str, metric: str, hour: int | None) -> dict | None:
    """Measured (mae, hit, is_decision_hour) for `metric` at station-local
    `hour`, for surfaces that want to label a read with how accurate it has
    actually been at this time of day. None when unmeasured (hour 0 has no
    training rows, `hour is None`, or the station itself is unmeasured).
    Read-only lookup — no model, no DB.

    NO fallback across stations: `HOUR_ACCURACY` only has a "KLAX" key today,
    so `.get(station, {})` on anything else is silently empty and this returns
    None -- it must NOT reach into KLAX's numbers for a different station,
    which would look like a real reading instead of the missing one it is.
    """
    if hour is None:
        return None
    row = HOUR_ACCURACY.get(station, {}).get(metric, {}).get(hour)
    if row is None:
        return None
    mae, hit = row
    knee = min_decision_hour(station, metric)
    # knee is None only when the station has no MIN_DECISION_HOUR entry -- we
    # cannot claim "at or after the decision hour" without one, so default to
    # not-ready rather than guessing an hour.
    return {"hour": hour, "mae": mae, "hit": hit,
            "ready": False if knee is None else hour >= knee}


def min_decision_hour(station: str, metric: str) -> int | None:
    """The earliest station-local hour at which a nowcast for `metric` may be
    acted on (see `MIN_DECISION_HOUR` above for how this hour was chosen).

    NO fallback to KLAX: plain nested `.get`, so an unmeasured station (any
    ICAO other than "KLAX" today) returns None. Serving LAX's decision hour to
    a Miami market would be silently, seriously wrong -- worse than an obvious
    error, since every downstream caller would treat it as a real answer.
    Callers (`decision_hour_ready`, `appbet.place`, etc.) must fail CLOSED on
    None, never substitute 0 or any other hour.
    """
    return MIN_DECISION_HOUR.get(station, {}).get(metric)


def extremum_still_future(station: str, metric: str, hour: int | None) -> float | None:
    """Share of days `metric`'s extremum had NOT yet occurred by station-local
    `hour` (see `EXTREMUM_STILL_FUTURE` above). None when `hour` is None or
    unmeasured for this station/metric.

    NO fallback to KLAX, for the same reason as `min_decision_hour`: this
    share describes one station's climate-day shape (KLAX's late-evening LOW
    tail, concretely), and presenting it as another station's number would be
    a wrong forecast wearing a measured one's clothes.
    """
    if hour is None:
        return None
    return EXTREMUM_STILL_FUTURE.get(station, {}).get(metric, {}).get(hour)


def decision_hour_ready(cfg: AppConfig, spec, as_of: str | datetime) -> bool:
    """Was this prediction issued at or after its metric's measured accuracy
    knee (`MIN_DECISION_HOUR`)?

    WHY: the live nowcast refits on ONLY the current local hour
    (`cli.cmd_predict --live`, `hours=(hour,)`), so every hour boundary fits a
    genuinely different model. Before the knee that makes the recommendation
    step around: on 2026-07-26 the LOW point went 68.82 (h3) -> 68.99 (h4) ->
    69.99 (h5) on IDENTICAL data, flipping the modal bucket across the 69/70
    line and issuing a PLACE on 70-71 after 68-69 had already been bought.
    Verified by holding the day's data fixed and varying only the hour.

    Keyed off the prediction's own `issued_at`, never "now" -- same rule as
    `morning_obs_ready`, for the same reason: what matters is the hour whose
    model produced this number, not the hour we happen to be reading it in.

    Fails CLOSED -- same philosophy as `morning_obs_ready` ("no observation,
    no trade"): if `cfg.station.icao` has no measured `MIN_DECISION_HOUR`
    entry, there is no knee to compare against, so this returns False rather
    than picking an arbitrary hour or borrowing KLAX's.
    """
    knee = min_decision_hour(cfg.station.icao, spec.key)
    if knee is None:
        return False
    return _local_hour(cfg, as_of) >= knee


def _local_hour(cfg: AppConfig, as_of: str | datetime) -> int:
    """Station-local wall-clock hour of `as_of`. Naive input is read as UTC,
    matching how `predictions.issued_at_utc` is stored."""
    ts = (datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(ZoneInfo(cfg.station.tz)).hour


def kind_at_hour(icao: str, metric: str, within_window: bool,
                 local_hour: int | None = None) -> str:
    """WHICH MODEL the desk reads at this hour — the rule, with no DB lookup.

    Split out of `preferred_prediction` on 2026-08-16 so that callers who need
    to know *which model produced a number* can ask without also fetching one.
    `notify.notify_divergence` is the reason: it has to compare a probability
    recorded at commit against a probability read now, and those are DIFFERENT
    MODELS whenever a HIGH is committed before the knee. Subtracting across
    them sent Julian a false "confidence collapsed" on KMIA 2026-07-29 — the
    committing model was unmoved at 64% and the bet won. See
    `scripts/divergence_kind_switch.py`.

    ⚠️ This expression must exist exactly once. `preferred_prediction` calls it.

    An unmeasured station has no knee to compare against; that degrades to the
    window-only behaviour documented on `preferred_prediction` for a missing
    `local_hour`, which is the correct safe default here -- not a crash and not
    KLAX's knee borrowed for a different station.
    """
    knee = min_decision_hour(icao, metric)
    past_knee = (local_hour is not None and knee is not None
                 and local_hour >= knee)
    return ("dayahead" if (within_window and metric == "high" and not past_knee)
            else "intraday")


def preferred_prediction(conn, icao: str, target: str, metric: str,
                         within_window: bool, local_hour: int | None = None):
    """THE single rule for which stored prediction any decision surface reads.

    **The right answer is opposite for the two markets**, because the extremum
    has formed at very different points relative to each decision hour. Both
    measured by walk-forward with a paired bootstrap (2026-07-25/26):

      HIGH @ 9am local (N=1364) — the max is still 2-5h in the FUTURE:
          day-ahead MAE 1.509 / CRPS 1.080   <- better
          intraday  MAE 1.620 / CRPS 1.178
          dMAE -0.111 [-0.166, -0.054]

      LOW @ 6am local (N=1333) — the min has largely ALREADY OCCURRED, so the
      nowcast's ratchet on the observed running minimum dominates:
          intraday  MAE 0.806 / CRPS 0.561   <- far better
          day-ahead MAE 1.469 / CRPS 1.058
          dMAE +0.663 [+0.596, +0.730] against the day-ahead

    So: day-ahead only for the HIGH while its window is open; the nowcast
    everywhere else. Outside the window the panel is informational and the
    nowcast is sharper for both metrics.

    2026-07-26 amendment — the HIGH's preference is now HOUR-BOUNDED, not
    window-bounded. The reasoning above ("the max is still 2-5h in the future")
    is only true early: `HOUR_ACCURACY` shows the nowcast overtaking the
    day-ahead decisively once the max is observed (MAE 1.397 at hour 9 vs 0.452
    at 16, 0.056 at 17). The HIGH window now closes at LA 18:00 to admit its
    measured decision hour, so a purely window-based preference would have kept
    serving the ~1.4-MAE day-ahead at hour 16 and thrown the entire gain away.
    Pass `local_hour` to get the hour-aware answer; omit it only where the hour
    is genuinely unknown, which falls back to the old window-only behaviour.

    Every surface (`signals.live_decision`, `/api/timing`'s model pick) MUST go
    through this so the dashboard cannot show two different picks at once.
    Returns `(pred, kind)`, or `(None, None)` when nothing is stored.
    """
    first = kind_at_hour(icao, metric, within_window, local_hour)
    order = (first, "intraday" if first == "dayahead" else "dayahead")
    for kind in order:
        pred = latest_prediction(conn, icao, target, kind, metric)
        if pred is not None:
            return pred, kind
    return None, None


def latest_prediction(conn, icao: str, target: str, kind: str, metric: str):
    row = conn.execute(
        """SELECT issued_at_utc, point_f, quantiles, buckets FROM predictions
           WHERE station=? AND target_date=? AND kind=? AND metric=?
           ORDER BY issued_at_utc DESC LIMIT 1""",
        (icao, target, kind, metric)).fetchone()
    if not row:
        return None
    return {"issued_at": row["issued_at_utc"], "point": row["point_f"],
            "buckets": json.loads(row["buckets"])}


def personal_cutoff_instant(cfg: AppConfig, day, metric: str = "high") -> datetime | None:
    """The unique UTC instant whose display_tz wall-clock == the metric's
    cutoff and which falls inside climate day `day`. HIGH: Manila 01:30 (~LA
    10:30am, pre-peak). LOW: Manila 21:00 (~LA 6am dawn, when the min locks) —
    the low forms in Julian's evening, so its window closes hours earlier."""
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    cutoff = (cfg.staking.low_personal_cutoff if metric == "low"
              else cfg.staking.personal_cutoff)
    hh, mm = (int(x) for x in cutoff.split(":"))
    tz = ZoneInfo(cfg.display_tz)
    for md in (day, day + timedelta(days=1)):
        cand = datetime(md.year, md.month, md.day, hh, mm,
                        tzinfo=tz).astimezone(timezone.utc)
        if start <= cand < end:
            return cand
    return None


def committed_state(conn, cfg: AppConfig, target: str, metric: str) -> dict:
    """Confidence + stability over the last `stability_readings` intraday preds
    for (target, metric). Returns {committed, stable, confident, label, prob,
    n_readings}."""
    n = cfg.staking.stability_readings
    rows = conn.execute(
        """SELECT buckets FROM predictions
           WHERE station=? AND target_date=? AND kind='intraday' AND metric=?
           ORDER BY issued_at_utc DESC LIMIT ?""",
        (cfg.station.icao, target, metric, n)).fetchall()
    picks = [argmax_pair(pmf_from_buckets(json.loads(r["buckets"]))) for r in rows]
    picks = [p for p in picks if p]
    if not picks:
        return {"committed": False, "stable": False, "confident": False,
                "label": None, "prob": None, "n_readings": 0}
    latest = picks[0]
    stable = len(picks) >= n and all(p["label"] == latest["label"] for p in picks)
    confident = latest["prob"] >= cfg.staking.min_confidence
    return {"committed": bool(stable and confident), "stable": bool(stable),
            "confident": bool(confident), "label": latest["label"],
            "prob": latest["prob"], "n_readings": len(picks)}


def gate_state(conn, cfg: AppConfig, target: str, metric: str,
               now: datetime) -> dict:
    """Full live gate for (target, metric) at `now` (UTC): the committed_state
    plus within_window and the cutoff instant, for the CLI/web to consume."""
    day = datetime.fromisoformat(target).date()
    cutoff = personal_cutoff_instant(cfg, day, metric)
    within = cutoff is not None and now <= cutoff
    cs = committed_state(conn, cfg, target, metric)
    return {**cs, "within_window": bool(within),
            "cutoff_utc": cutoff.isoformat() if cutoff else None,
            "cutoff_manila": (cutoff.astimezone(ZoneInfo(cfg.display_tz))
                              .strftime("%a %H:%M") if cutoff else None)}


@dataclass(frozen=True)
class GateVerdict:
    """Why the desk may or may not act. `refusals` names EVERY failing gate, not
    just the first: a desk shown one reason fixes it and is surprised by the
    next."""
    ok: bool
    refusals: tuple[str, ...]


def decision_gates(conn, cfg, *, metric: str, target: str, hour: int,
                   p_model: float, price: float | None = None) -> GateVerdict:
    """THE verdict. Every surface -- terminal, dashboard, and later the bot --
    must read this one function.

    WHY ONE FUNCTION. Scattered `if`s drift apart and the desk ends up giving
    advice that disagrees with itself. `board.evidence()` is assembled in Python
    for the same reason.

    EVERY GATE FAILS CLOSED. Missing data, an unmeasured market and an exception
    all produce a refusal. For an unattended caller "I could not tell" and "no"
    must be the same answer.

    `price` is OPTIONAL and the rail is skipped when it is None. Nothing in
    phase 1 passes a price; the rail is here so the aggregator does not need a
    second edit when the bot arrives.
    """
    from highcast import fees

    icao = cfg.station.icao
    refusals: list[str] = []

    # --- Gate W: severe weather -------------------------------------------
    try:
        if storm_warning_active(conn, cfg, target):
            refusals.append("storm_warning")
    except Exception:
        refusals.append("storm_warning")

    # --- Gate C: data freshness -------------------------------------------
    try:
        if metric_uses_nbs(metric) and not nbs_guidance_ready(conn, cfg, target):
            refusals.append("nbs_missing")
    except Exception:
        refusals.append("nbs_missing")

    try:
        if not guidance_lead_ready(conn, cfg, target):
            refusals.append("guidance_lead")
    except Exception:
        refusals.append("guidance_lead")

    # --- Gate B: calibrated confidence ------------------------------------
    p_hat = None
    try:
        p_hat = band_calib(icao, metric, hour, p_model)
    except Exception:
        p_hat = None
    if p_hat is None:
        # None means REFUSE, never "use the raw number" -- it is returned for an
        # unmeasured market/hour or for a probability above the highest measured
        # band, and all three losing bets were stated 81%, i.e. from there.
        refusals.append("unmeasured_band")
    elif not confidence_floor_ok(icao, metric, hour, p_hat):
        refusals.append("confidence_floor")

    # --- The price rail (optional) ----------------------------------------
    # A loss floor, NOT edge-hunting: never knowingly pay more than the bucket is
    # worth. At 1 contract Kalshi's taker fee rounds up to 2c at every price, so
    # the fee is a real part of the comparison.
    if price is not None:
        try:
            if p_hat is None:
                refusals.append("price_rail")
            else:
                fee = fees.taker_fee(1, float(price))
                if float(price) >= (p_hat - fee):
                    refusals.append("price_rail")
        except Exception:
            refusals.append("price_rail")

    return GateVerdict(ok=not refusals, refusals=tuple(refusals))
