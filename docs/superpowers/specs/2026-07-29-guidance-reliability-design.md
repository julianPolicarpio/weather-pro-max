# Guidance reliability: stop being confidently wrong

**Date:** 2026-07-29
**Status:** designed, not implemented
**Scope:** KMIA both metrics. KLAX frozen until its test week scores.

## The failure this fixes

2026-07-29, KLAX LOW. The model put **86.4% on ≤68** and committed 67–68 at
81.2%. Kalshi was at 58%. The temperature bottomed at **69.08°F at 01:53 PDT**,
sat flat at 69.98 under overcast, then rose to 71.06 by 05:53. The running
minimum rounds to 69, so the bet is LOSING on a 1°F boundary -- not yet settled
at the time of writing (the KLAX climate day runs to 01:00 PDT, and ~35% of July
lows fall in the 22:00-00:00 evening tail, so a late rescue is possible though
it would require tonight to be cooler than this morning was).

Note the spec does not depend on the outcome. The defect being fixed is the
CONFIDENCE (81% vs the market's 58% on a call the observation trace did not
support), and that is already established regardless of how the day settles.

Two separable defects. It was **wrong** (67–68 vs 69) and it was **confident
while wrong** (81% vs the market's 58%). Julian chose to fix the second: when
guidance and observations disagree, the model should widen rather than pick a
side. A read of 67–68 at ~55% would have failed the +5% edge gate and the desk
would have said don't place.

## Root cause

Not that guidance was wrong — guidance is often wrong. The defect is that **the
nowcast has no idea how reliable its guidance has been.**

Its 14 features (`INTRADAY_FEATURES`) include `base_p50`, the blended guidance
point, but nothing about that blend's recent track record. The DAY-AHEAD model
has `bias_mean`, `bias_abs_mean`, `bias_n` (see `features.BIAS_FEATURES`); the
intraday model has none of them.

So it consumed `base_p50 = 66.96` as trustworthy while guidance had run ~2°F
cool for three straight days (said ~67, delivered 70 / 69 / 69). The 14-day
trailing correction (`features.BIAS_WINDOW`) is a flat average, so a sharp
3-day shift is diluted 11:3 by stale history.

**The usual uncertainty signal pointed the wrong way.** All eight guidance
models agreed (65.8–68.1), so `fc_spread` was LOW — looking confident — while
they were collectively 2°F wrong. Any fix keyed on guidance dispersion would
have made this worse, not better. Agreement is not correctness.

## Design

### The signal — two features, not four

```
guidance_gap  = base_p50 - <the metric's ratchet bound>

# The bound is `IntradayModel._bound`, NOT the intuitive column name:
#   extremum == "max"  ->  run_lo   (low end of the observed running max)
#   extremum == "min"  ->  run_hi   (high end of the observed running min)
# So a LOW uses run_hi. Yesterday's KLAX LOW: 66.96 - 69.18 = -2.22.
# Getting this backwards silently feeds the model a gap measured from the wrong
# side of a 0.2F interval -- small, wrong, and invisible.
bias_abs_mean = existing trailing |error| magnitude, ported from BIAS_FEATURES
```

`guidance_gap` is how much further movement guidance expects versus where the
thermometer already sits. Yesterday's KLAX LOW: `66.96 - 69.18 = -2.22`, i.e. guidance wanted 2.2°F more
cooling and got none. It is guidance's OPINION of the undercut,
which is distinct from the undercut the model predicts — so it is a legitimate
input, not a leak of the target.

`bias_abs_mean` answers "has this guidance been reliable lately", which is
precisely the question the nowcast cannot currently ask.

**TWO columns, deliberately.** Measured twice in this project: adding correlated
columns actively hurts by diluting LightGBM's per-split feature sampling
(`nbm_nbs` alone cost 2.7 points of bucket hit). Do not add `guidance_gap`,
`bias_mean`, `bias_n`, `fc_spread` and a sky interaction "to see what sticks".

### Arm A — the feature

Add both via `IntradayModel(extra_features=[...])`, the hook added 2026-07-29.
Its default reproduces the shipped column list byte-identically, so this cannot
perturb a station that does not opt in.

### Arm B — the calibration fallback

If Arm A does not earn its place, use the existing `CalibrationConfig(mode)`
scaffolding (`off | width | center_width`), already per-station config: widen
the finished distribution when `|guidance_gap|` exceeds a threshold. One config
line to revert, and it cannot move the point forecast — only the spread.

Arm B is a fixed rule and cannot learn WHEN to widen, which is why it is the
fallback rather than the first choice.

## How it is judged — PRE-REGISTERED

**Primary: CRPS.** A proper scoring rule: it penalises both miscalibration AND
vagueness, so it cannot be gamed by being uncertain about everything. This is
the metric that matters when the goal is calibration rather than hit rate.

**Diagnostic: calibration error** (mean stated probability on the modal bucket
minus its realised hit rate) at predictive hours. Currently +2.0% for KMIA's LOW
at its commit hour and +2.9% for its HIGH.

**Guard: bucket hit must not fall by more than 1.0 point.** A model uncertain
about everything is perfectly calibrated and useless.

**SHIP GATE:** CRPS improves AND calibration error shrinks AND the hit guard
holds. Fixed before seeing any result. Today produced three losing hypotheses
(full pooling, parametric Student-t, convection features) and the only reason
the third was not shipped on a +0.3 mean is that the rule was set in advance.

**Scored on PREDICTIVE HOURS ONLY** (`gating.EXTREMUM_STILL_FUTURE >= 0.60`).
An arm that wins after the extremum has formed has won nothing.

## Testing

Reuse the two-arm pattern from `scripts/convection_eval.py`: same rows both
arms, identical date range, per-hour, restricted to predictive hours. The
existing `extra_features` hook means Arm A needs no model fork.

Unit tests: `guidance_gap` computes correctly for both metrics (`run_lo` for a max,
`run_hi` for a min — reuse `IntradayModel._bound` rather than reimplementing it), and is NaN-safe when no observation
exists yet.

## Scope and honesty

Measured on **KMIA both metrics**. KLAX stays frozen: any change to its features
would void a running scorecard.

**⚠️ THE FAILURE WAS ON KLAX AND THE FIX IS MEASURED ON KMIA.** Miami's
guidance-bias behaviour may differ — its low is convection-influenced, LAX's is
marine-layer. A win on Miami is suggestive for LAX, not proof. Re-measure on
KLAX after its week scores, and do not generalise on faith: this project has
twice had to walk back a metric-specific result that was assumed to transfer.

## Not doing

- Faster / exponentially-weighted `BIAS_WINDOW`. That targets accuracy, which
  Julian explicitly deprioritised. Revisit separately if the hit rate becomes
  the complaint.
- Anything keyed on guidance dispersion (`fc_spread`). It pointed the wrong way
  in the motivating case.
- Retuning the +5% edge gate. The gate behaved correctly; the probability fed
  into it was wrong.
