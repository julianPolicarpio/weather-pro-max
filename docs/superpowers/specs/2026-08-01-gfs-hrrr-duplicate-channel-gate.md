# PRE-REGISTRATION — the `gfs`/`hrrr` duplicate-channel gate

**Written 2026-08-01, BEFORE any arm was run.** Anything added after the first
result is marked as such.

---

## The defect

`fetchers/forecasts.MODELS` maps `"gfs" -> gfs_seamless` and `"hrrr" -> gfs_hrrr`.
Over CONUS, Open-Meteo's *seamless* product **is HRRR** for the first 48 hours.
So `fc_gfs` and `fc_hrrr` are the same series at every lead the models consume.

Evidence, gathered 2026-08-01:

| check | result |
|---|---|
| stored daily table, lead-0/lead-1 pairs since 2024-01-18 | identical on **3,733 of 3,735** |
| live API probe, 48-hour hourly series | **byte-identical** |
| historical lead-1 probe (2025-06-01..10) | `gfs_seamless` vs `gfs_hrrr` **240/240 identical** |
| same probe vs the real global GFS | `gfs_global` differs on **238 of 240** |

`BaselineBlend` weights by inverse MSE and therefore counts this one channel
twice:

| market | fc_gfs | fc_hrrr | **combined** |
|---|---|---|---|
| KLAX high | 12.7% | 13.5% | **26.1%** ← largest single source |
| KLAX low | 8.2% | 11.2% | 19.4% |
| KMIA low | 9.0% | 10.4% | 19.3% |
| KMIA high | 10.8% | 8.2% | 19.0% |

This is the exact failure `BaselineBlend._blend_cols` documents ("blending both
`fc_gfs` and `fcc_gfs` would enter the same model twice and double-weight it") —
in a form its guard cannot see, because it dedupes two *names* for one model, not
two models holding one series.

## Arms

All three are measured on identical rows, at each market's decision hour.

- **A — STATUS QUO.** `FC_MODELS` unchanged: `gfs` (=HRRR) and `hrrr` both present.
- **B — DEDUPE.** Drop `hrrr`. Isolates whether the *double-weighting itself*
  causes harm, using only data that already exists.
- **C — REPLACE (the intended fix).** `gfs` remapped to the real `gfs_global`,
  `hrrr` kept. Two genuinely independent channels. Requires backfilling
  `gfs_global` history, which the previous-runs endpoint supports.

B and C answer different questions and both are needed. If B ≈ A, the
double-weighting is harmless and only C matters. If B > A, the duplication is
actively costing accuracy.

## Primary outcome

Parity-averaged bucket hit at the decision hour, walk-forward (refit 30,
RECENT ≥ 2025-01-01), **paired on matched target dates**. Kalshi's ladder parity
is a coin flip per day, so both parities are scored and averaged
(`buckets.expected_bucket_hit`).

Secondary, reported but not gating: MAE and CRPS.

## Ship threshold — ALL must hold, per market, independently

1. Δhit (arm − A) > 0
2. the paired bootstrap CI on Δhit excludes zero at **Bonferroni ×4** (four
   markets), i.e. a 98.75 % interval
3. the arm does not make MAE worse by more than 0.05 °F (a hit-rate gain bought
   by degrading the point forecast is not a win)

A market that fails keeps the status quo. **C is preferred over B** where both
pass, because C recovers a real forecast channel rather than discarding one.

## Expected outcome, stated up front

**I expect B ≈ A and C to be a small improvement at best.** Inverse-MSE
weighting is fairly robust to a duplicated column — the duplicate splits the
weight it would otherwise get alone, so the *combined* influence is roughly what
one well-performing channel would earn anyway. The double-counting is real and
worth fixing for honesty and for feature-importance readings, but it is not
obviously an accuracy bug.

The measurement most likely to matter is C, and only because it ADDS information
(a genuine GFS) rather than because it removes a duplicate.

This project is 3-for-9 on confident pre-measurement hypotheses, and today alone
produced five nulls. A null here is the expected result and will be published as
one.

## What ships if it passes

`FC_MODELS` and `fetchers/forecasts.MODELS` change together, plus a backfill of
`gfs_global`. **This is a model change**: it alters `fc_gfs` for every future row
while history holds the old meaning, so the backfill must rewrite history
consistently or the column splices two series — the exact error the `FC_MODELS`
comment already warns about for the MOS bulletin.

## What ships if it fails

Nothing in the model. The finding is still recorded, because it invalidates the
"HRRR ranks 5th–7th at ≤22 %" refutation of nowcast.trade's "HRRR is primary"
claim — that ranking counted one channel twice, and combined it is the largest
input at KLAX HIGH.


---

# RESULT — 2026-08-01. NULL ON ALL FOUR MARKETS. NOTHING SHIPS.

Run: `scripts/gfs_hrrr_gate_eval.py`. Raw: `.superpowers/sdd/gfs-hrrr-gate.txt`.
Arms varied by DATA (a private DB copy each), so the production code path ran
unmodified. B deleted 3,735 `hrrr` rows; C rewrote 3,700 `gfs` rows with real
`gfs_global` values.

| market | n | A hit | B dedupe | C real GFS |
|---|---|---|---|---|
| KMIA HIGH h14 | 570 | 79.3% | −0.6% [−1.8, +0.4] | −0.1% [−1.4, +1.3] |
| KLAX HIGH h13 | 574 | 69.4% | −0.5% [−1.7, +0.9] | +0.4% [−1.0, +1.8] |
| KMIA LOW h7 | 567 | 66.9% | −0.1% [−1.1, +0.9] | +0.2% [−0.8, +1.1] |
| KLAX LOW h6 | 574 | 67.9% | +0.6% [−1.2, +2.6] | +0.6% [−1.1, +2.3] |

Every CI contains zero; no market clears condition 2. MAE moves ≤0.008 °F
everywhere, so condition 3 was never the binding one.

**The pre-registered expectation held.** It said "B ≈ A and C a small improvement
at best… the double-count is worth fixing for honesty and for feature-importance
readings; it is not obviously an accuracy bug." Arm B confirms the mechanism
directly: deleting HRRR entirely moves nothing, because inverse-MSE weighting
lets a duplicate split the weight it would otherwise earn alone, so the COMBINED
influence is near what one good channel gets.

**Arm C is the more informative null.** Adding a genuinely independent global GFS
— 3,700 rows of real new information — bought +0.4 to +0.6 points at best, none
significant. A well-fit ten-channel ensemble does not move for an eleventh
channel at the margin.

## What follows

1. **Do NOT remap `MODELS["gfs"]`.** The change is not justified by accuracy, and
   it would splice two meanings into one column mid-history — the exact hazard
   the `FC_MODELS` comment warns about for the MOS bulletin.
2. **The duplication is still real and still worth documenting.** Feature
   importances and blend-weight tables currently misattribute where the
   information comes from. That is a documentation-grade defect, not a model bug.
   Downgraded from "biggest untested distortion in the model surface" to
   cosmetic.
3. **The durable consequence is unchanged**: §8's refutation of nowcast.trade's
   "HRRR is primary" rested on HRRR ranking 5th–7th at ≤22%. Counting the channel
   once, HRRR-derived guidance is **26.1% at KLAX HIGH — the largest single
   input**. That refutation remains wrong regardless of this null.
4. **Backfilled `gfs_global` rows were removed** from the live database after the
   run. They served the experiment and are not a shipped channel; leaving them
   would put an unused model in `forecasts` that nothing reads and that a future
   blend change could silently pick up. Re-fetchable from the previous-runs
   endpoint (`temperature_2m_previous_day1`, models=`gfs_global`) if needed.
