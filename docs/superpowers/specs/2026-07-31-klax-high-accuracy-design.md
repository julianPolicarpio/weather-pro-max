# KLAX HIGH accuracy — warming rate and guidance freshness

**Date:** 2026-07-31
**Status:** ✅ **H1 SHIPPED** — cleared all five pre-registered gates.
H2 killed by a pre-check before implementation.

## RESULT

    arm        MAE     HIT     dMAE     dHIT    97.5% CI on dHIT
    BASE      1.104   51.7%      --       --
    RATE      1.053   54.2%   -0.051    +2.5%   [+0.3%, +4.9%]   <- SHIPPED
    ENV4      1.077   52.6%   -0.027    +1.0%   [-1.4%, +3.2%]
    RATE+ENV  1.047   53.9%   -0.057    +2.3%   [-0.4%, +4.9%]
    RAND      1.096   50.3%   -0.008    -1.4%   [-3.5%, +0.8%]

**KLAX HIGH bucket hit 51.7% -> 54.2%**, and the CI excludes zero even at the
Bonferroni-corrected threshold. This is the FIRST hypothesis in seven attempts in
this project to clear that bar — the previous six (convection, pooling,
parametric, rate-of-change at the LOW, the >=80 seam, the marine-layer deck) all
failed.

Corroborating rather than isolated: MAE improves too (-0.051), the noise arm still
HURTS (-1.4%), and RATE beats noise by 3.9 points.

**RATE ALONE BEATS RATE+ENV** (+2.5 vs +2.3). The rejected environment columns
dilute it — the same per-split feature-sampling dilution this project has now
measured three times. Per the gate, the better arm ships alone.

### Shipped as

`intraday.RATE_FEATURES` + `intraday.features_for(station, metric)`, which returns
the extra columns for **KLAX HIGH only**. Every other market gets exactly
`INTRADAY_FEATURES`. Verified byte-identical for all three frozen markets
(max |diff| 0.0). Wired into `cli.cmd_predict`'s live path.

### Audits passed

- **Leak-free**: seeding wild FUTURE observations (+999F, -999F) after the as-of
  instant moves no rate feature by any amount.
- **Frozen markets untouched**: KLAX LOW, KMIA LOW, KMIA HIGH all byte-identical.
- **Suite**: 538 passing, 12 deselected, zero failures.
- Missing history yields NaN, never 0.0 — asserted, because on a feature whose
  purpose is telling fast warming from slow, "no warming" is a real and different
  state from "unknown".

---

**Original status:** design, pre-registered before any result was seen
**Scope:** KLAX HIGH only. It was unfrozen 2026-07-31; KLAX LOW, KMIA LOW and
KMIA HIGH remain FROZEN and this design must not touch them.

---

## 1. The problem, measured

KLAX HIGH is the worst market on the board: **52.4%** bucket hit at its commit
hour and overconfident by **+7.4%**. Every loss in the test week has been a HIGH;
the LOWs are 5-0.

The cause is structural, measured 2026-07-31: at h11 the ratchet bound sits a mean
**2.88F** from the value that finally settles, against **1.63F** for the KLAX LOW
at h4. The HIGH is asked to forecast an entire afternoon against a 2F ladder.

**What was already ruled out (2026-07-31, earlier run).** Marine-layer deck height,
ceiling, visibility and pressure tendency, as LEVELS at h11. Deck height alone
scored **-1.3% dHIT against pure noise at -1.4%** -- indistinguishable from a
random column. Removing columns made it worse rather than better, which refuted
the dilution defence. See `klax_high_env_eval.py` and the negative result recorded
in the 2026-07-30 spec.

## 2. Research: what nowcast.trade does differently

Two things from their product surface bear on the HIGH:

> **[STATED]** "HRRR is the primary forecast -- it's the 3km short-range model
> that updates hourly, built specifically for same-day predictions. GFS, ECMWF and
> Best Match are confirmation models."

> **[OBSERVED]** 🔥 RE-HEAT RISK on a HIGH: "forecast says the temperature can
> still climb past the bracket."

We carry HRRR as one of nine channels with no special weighting, and we have no
re-heat concept at all. Both map onto gaps that are measurable here.

## 3. The two hypotheses

### H1 — the nowcast has NO time-derivative

`INTRADAY_FEATURES` contains no rate, lag, delta or trend column. The nowcast
cannot express "the temperature is climbing fast" -- which is the re-heat idea and,
at a coastal station, the marine-layer burn-off signal.

**The day-ahead HIGH model already has this and it is a documented win.**
`features.py` describes `morn_rate` as "the 6am->9am warming rate -- the
marine-layer burn-off signal (backtested win)". The nowcast, which is what actually
decides the bet at h11, does not get it.

**Why HANDOFF's rate-of-change negative result does NOT transfer.** That result was
measured at the **LOW's** commit hour, where the 2h change was **0.00F on every day
sampled** -- LAX sits flat at 4am under a marine layer, so the feature was a
constant and could discriminate nothing. At the HIGH's h11 the same quantity is
highly variable (measured, n=400 days, KLAX 08->11 local):

    mean +6.79F   sd 4.35F   range -6.1 .. +22.0   exactly zero on 2% of days

A feature that is constant cannot help; one with this spread can. The negative
result was correct for the LOW and says nothing about the HIGH.

### ❌ H2 — KILLED BY A PRE-CHECK, and the real diagnosis is worse

**Do not implement H2 as specified below.** §8 of this spec required checking, before
building anything, whether guidance actually MOVES between the 09:00 cutoff and the
h11 commit. It does not:

    KLAX HIGH, guidance blend at 09:00 vs 11:00, n=400 days
      mean move +0.000F   sd 0.000F   moved at all on 0/400 days (0%)

`fresh_minus_stale` would have been identically zero on every row -- the same
failure mode as the LOW's rate feature, where a constant could discriminate
nothing. The pre-check cost one query and saved building a dead column.

**The real diagnosis, and it is worse than the hypothesis.** At the h11 commit the
guidance the model uses is **11.1 hours old**. For target 2026-07-30:

    mos_mex   fetched 07-30 10:28 local   0.5h before commit
    ecmwf, gfs, icon, hrrr, nbm, nam, aifs, nws
              fetched 07-29 23:55 local  11.1h before commit

Every NWP channel -- including **HRRR, the model built for same-day forecasting and
refreshed HOURLY** -- was last pulled the previous night. The cause is not the 09:00
cutoff at all: **guidance is never fetched during the day.** The wake job refreshes
observations every 5 minutes but pulls NWP roughly three times daily, and same-day
runs only reach the database at 06:55Z the NEXT morning, after the target day is
over. The leakage guard correctly excludes them, which is exactly why the
09:00-vs-11:00 comparison shows zero movement.

So the model is doubly blind at commit: 2.88F of the answer is still unobserved AND
its guidance is 11 hours stale.

**Why it cannot be backtested.** Every historical same-day run carries a
`fetched_at` after its own target day. Using them would be pure hindsight -- the
same trap HANDOFF flags about the archive-API precipitation data, which serves
OBSERVED rather than forecast values. There is no honest historical arm.

**What H2 becomes:** an OPERATIONAL change (fetch NWP intraday so fresh runs exist
at commit), not a feature. Cheap to implement, impossible to validate until forward
data accumulates. Queued, not built.

---

### H2 as originally written (superseded -- kept for the reasoning)

The HIGH's feature cutoff is `cutoff_local_hour: 9`, so `base_p50` is built from
guidance known at **09:00 local**. The bet commits at **h11**. HRRR refreshes
hourly, and KLAX has **8 same-day model runs** available. So up to two hours of
fresher short-range guidance exists at commit and the nowcast never sees it.

This is the same design choice nowcast.trade made deliberately (HRRR-primary for
same-day), arrived at from our own cutoff rather than from copying them.

## 4. Architecture — additive only

**Both hypotheses ship as NEW candidate columns in the feature frame. Nothing
existing is modified.**

- `base_p50` is untouched, so the three FROZEN markets are provably unaffected.
  This is asserted mechanically, not assumed: with candidate columns present the
  shipped model's predictions are byte-identical (max |diff| 0.0), because
  `IntradayModel` selects exactly `INTRADAY_FEATURES` and the frame merely widens.
- New columns enter no model unless named via `IntradayModel(extra_features=[...])`.
- `INTRADAY_FEATURES` changes only if the gate passes.

### Arm A columns (H1)

| column | definition |
|---|---|
| `cur_rate_3h` | temp at as-of MINUS temp 3h earlier -- the `morn_rate` analogue |
| `cur_rate_1h` | temp at as-of MINUS temp 1h earlier -- short-term acceleration |
| `cur_rise_from_min` | temp at as-of MINUS the day's running minimum so far |

Three, not one: a warming trajectory has a level and a curvature, and "how far it
has already climbed" is a different quantity from "how fast it is climbing". Three
sits inside this project's own "3-4 columns, not everything" rule, which exists
because adding correlated columns has twice cost bucket hit by diluting LightGBM's
per-split feature sampling.

### Arm B columns (H2)

| column | definition |
|---|---|
| `base_p50_fresh` | the guidance blend re-evaluated AT THE AS-OF HOUR, not at 09:00 |
| `fresh_minus_stale` | `base_p50_fresh - base_p50` -- how far guidance moved since the cutoff |

`fresh_minus_stale` is the more interesting of the two: it isolates the UPDATE,
which is where the new information lives, rather than re-presenting a level the
model already has.

**LEAKAGE IS THE REAL RISK IN ARM B, and it is handled by construction.**
`_forecast_features` already takes an explicit cutoff instant and enforces
`run_date < cutoff_date OR (run_date = cutoff_date AND fetched_at <= cutoff)`. Arm
B passes the as-of instant and the existing guard does the rest -- no new leakage
logic is written. A test asserts that a run fetched AFTER the as-of instant is
excluded, and it must be verified to FAIL without the guard so it is not
tautological.

## 5. The pre-registered ship gate

**Written before any result was seen.** Two primary arms double the false-positive
risk, so the significance threshold is **Bonferroni-corrected: each arm must clear
a 97.5% CI excluding zero**, not 95%.

An arm ships only if ALL of:

1. `dHIT > 0` -- parity-averaged modal-bucket hit must improve. Bucket hit is what
   pays; MAE is not.
2. `dMAE <= 0` -- must not be worse on MAE.
3. The `RAND` control must not beat it on dHIT.
4. It must beat `RAND` by more than 0.5 points.
5. **97.5% CI on dHIT excludes zero.**

If both arms pass, ship both only if a COMBINED arm also passes; otherwise ship the
better one alone. Selecting the best of several arms and then testing it is exactly
how the previous run manufactured a false positive, so the correction is applied up
front rather than after seeing which arm won.

**Power check, so the bar is known to be winnable.** The previous run's paired CI
had SE ~1.07 points, so significance at the corrected level needs roughly
**dHIT > 2.4 points**. Real but achievable -- not the ~6 points a naive unpaired
test would demand.

**If both arms fail, that is the answer.** It gets written down as a negative
result and the model stays unchanged. This project is **1-for-6** on confident
pre-measurement hypotheses (convection, pooling, parametric, rate-of-change at the
LOW, the >=80 seam, and the marine-layer deck).

## 6. Regime

Identical to `hour_curve.py`, `calib_eval.py` and the previous env experiment, so
every number stays comparable to what is already measured:

- matched-hour training at the configured bandwidth
- walk-forward, refit 45, expanding training window
- parity-averaged modal-bucket hit (`buckets.expected_bucket_hit`)
- test window **>= 2025-01-01** (n~573). An expanding window averages in data-poor
  early folds and reads pessimistic for the shipped model -- measured, see
  `gating.HOUR_CALIB`.
- `+RANDOM` control with the same column COUNT as the largest real arm

## 7. Testing

- Rate columns are NaN when the lookback observation is missing, never 0.0 -- a
  missing history must not read as "no warming".
- `cur_rise_from_min` uses the running MIN of the climate day, not a fixed lookback.
- Arm B leakage: a run fetched after the as-of instant is excluded; asserted to
  FAIL without the guard.
- `base_p50` is unchanged by the presence of `base_p50_fresh`.
- The shipped feature lists keep their pinned counts (14 / 42 / 20 / 38) unless the
  gate passes, and no candidate column may appear in them otherwise.

## 8. Open, deliberately

- **This only addresses the h11 commit.** The structural fact -- 2.88F of the
  answer still unknown -- is not removed by any feature; it is the reason the HIGH
  is hard. A win here narrows the gap, it does not close it.
- **H2's ceiling is bounded by how much guidance actually moves in two hours.** If
  `fresh_minus_stale` is near zero on most days, the hypothesis is inert for the
  same reason the LOW's rate feature was, and that will show in its distribution
  before the model ever sees it. Check that first.
