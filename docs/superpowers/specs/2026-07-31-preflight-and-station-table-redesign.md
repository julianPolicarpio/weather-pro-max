# Preflight + Station Table redesign — 2026-07-31

Two requests, one session: make the Station Table readable, and give Julian a way
to know which prediction is worth trading "just like the preflight thing on
nowcast so that i dont waste money on unstable predictions".

They are answered separately below because only one of them needed a backtest.

---

## Part 1 — The preflight

### What the competitor's version actually is

Their "Weather Guard (preflight)" grades 18 cities A+ through C−. Per
`docs/nowcast-trade-reference.md` §4.5 it is a **static per-city climatological
grade**, labelled "Beta — grades are new", with no published validation. It
returns the same answer every day.

Copying that literally would produce something we already ship: the reliability
table, which ranks all four markets by measured accuracy and is likewise the same
every day. So the thing worth building is the one they do *not* have — a **daily**
signal that says whether *today's* instance of a market is one of its good days.

### The factor, and why it is not the model's own confidence in disguise

`bound_gap` = |ratchet bound − day-ahead guidance|. It is large exactly when the
two sources disagree about where the day is going.

**This is the same column that failed as a model feature, and that is not a
contradiction.** `scripts/klax_low_gap_eval.py` asked *does giving the tree this
column raise the hit rate* → no (GAP1 +1.0%, CI [−1.1%, +3.1%]). The preflight
asks *does this column tell me when to disbelieve the model* → yes. A tree
averaging a weak signal over every day gains little; a bettor who sits out the
worst quarter of days keeps his money. Both results stand.

### The outcome measured: shortfall, not accuracy

    shortfall = stated probability − realised hit

A hard day is fine to trade if the model *says* it is hard. What costs money is
the model staying confident while being wrong. Accuracy alone would flag the
difficult days; shortfall flags the **dishonest** ones, which is what a preflight
is for.

### Pre-registered gate (written before any result was seen)

Primary factor `bound_gap` on all four markets, split at the **within-market
median** — no threshold search, since hunting the cut that maximises the gap is
exactly how a band-aid gets manufactured. Significance Bonferroni-corrected to a
98.75% CI for four markets. Secondary factors (`stall_h`, `pres_tend`) reported
uncorrected and labelled exploratory, so a lucky one could not be promoted.

Ship only if: flagged stratum worse **and** its shortfall > 3 points **and** the
corrected CI excludes zero.

### Results — `scripts/preflight_eval.py`

| market | Δshortfall | 98.75% CI | gate |
|---|---|---|---|
| **KLAX LOW h4** | **+8.3%** | **[+1.5%, +15.5%]** | **SHIP** |
| KMIA LOW h5 | +7.5% | [−0.3%, +15.1%] | no — misses by 0.3pt |
| KMIA HIGH h11 | +2.3% | [−5.0%, +9.5%] | no |
| KLAX HIGH h11 | −3.6% | [−11.4%, +3.6%] | no — effect **reverses** |

`pres_tend` is dead on every market (largest |Δ| 2.1%, every CI spans zero).
`stall_h` misses even uncorrected on every market.

**The split is mechanistic, not curve-fitted.** Both LOWs show it; neither HIGH
does. The ratchet bound only becomes informative once the extremum has *happened*,
and at the HIGH's h11 commit 88% of it is still ahead on *every* day — so there
are no two regimes for the gap to tell apart. The effect appears exactly where the
physics says it should.

### Three ways it was attacked afterwards — `scripts/preflight_robustness.py`

**CHECK 1 — is it just restating the model's own confidence?** The one that
mattered: flagged days carry a lower stated probability (60.3% vs 66.1%), so if
the whole effect were "low-confidence days go wrong", the honest advice would be
the trivial "don't trade under 62%" and the gap would add nothing.

| stated-confidence bin | flagged hits | clean hits | Δshortfall |
|---|---|---|---|
| 32–60% | 46.0% (n=136) | 66.4% (n=55) | +16.0% |
| 60–68% | 61.5% (n=78) | 71.2% (n=113) | +9.7% |
| 68–80% | 66.7% (n=72) | 69.3% (n=119) | +2.2% |

Survives 3/3. In the lowest bin the model says ~50% either way while the truth is
46% against 66% — a 20-point spread it cannot see at all.

**CHECK 2 — out of sample in time.** Threshold fit on 2025-01-01..2025-10-15 and
applied untouched to 2025-10-16..2026-07-29: flagged 51.3% vs clean 70.9%,
Δshortfall +12.5%, 95% CI [+3.0%, +22.7%]. Holds, with a *larger* effect than
in-sample.

**CHECK 3 — threshold stability.** median 1.59 → 1.66, Q3 2.45 → 2.68 between
halves. Small enough to ship a fixed cut on, large enough that it must be
re-measured rather than trusted forever.

KLAX HIGH fails robustness too (1/3 bins, does not hold out of sample), and KMIA
HIGH is weakened (2/3) — consistent with their gate failures.

### Shipped operating points — KLAX LOW h4 only

| band | gap | n | stated | hit | shortfall |
|---|---|---|---|---|---|
| CLEAR | ≤ 1.62 | 287 | 66.1% | **69.5%** | −3.4% |
| FAIR | 1.62–2.53 | 143 | 63.4% | 64.7% | −1.2% |
| AVOID | > 2.53 | 143 | 57.2% | **46.2%** | **+11.0%** |

The damage is concentrated entirely in the top quartile. The median split cleared
the pre-registered gate, which is what makes the Q3 cut a legitimate operating
point rather than a scan.

**KMIA LOW does not ship despite being the strongest corroboration the result
has** — same direction, near-identical threshold (1.73 vs 1.62), same
Q4-concentrated damage, survives CHECK 1 3/3. It missed the corrected gate by
0.3pt. A gate that bends for a near miss is not a gate. Re-test when more settled
days are in.

### Implementation

- `src/highcast/preflight.py` — `BANDS` (KLAX LOW only, no cross-market
  fallback), `band_for`, `stored_gap`, `grade`. Four statuses: `graded`,
  `provisional` (gap not from the commit hour, the only hour the bands were
  fitted at), `pending`, `unmeasured`.
- `migrate.ensure_prediction_bound_gap` — `predictions.bound_gap`. The gap is
  **already computed** in the live frame and was being discarded; storing it
  avoids refitting a `BaselineBlend` on every dashboard request. Runs *after*
  `ensure_metric_column`, which may rebuild `predictions` with DDL that lacks the
  column.
- `cli.cmd_predict` writes it **after** `qs` is computed, so it cannot be an
  input.

**Freeze status: display only.** `tests/test_preflight.py` asserts no file in the
decision path (`staking`, `appbet`, `gating`, `notify`, `features`) mentions
`preflight`, and that `cur_bound_gap` is in no shipped feature list. If the app
began skipping bets on this verdict, its scorecard would stop measuring the model
and start measuring the filter.

---

## Part 2 — The Station Table redesign

No backtest needed: this changes how existing numbers are drawn, not what they
are.

### What was wrong

Julian: "its hard to understand pure text". Three specific failures:

1. **No hierarchy** — `temp / dewpt` carried the same weight as the number that
   settles the bet.
2. **No frame of reference** — "depression 5.9°F" requires the reader to supply
   the physics. Is that unusual here, at this hour, in July?
3. **Buried actionability** — "next group files 22:53, in 0.2h" is the single
   line that changes what you do next, and it sat mid-list.

### The redesign

Built on the existing "tower board" tokens; no new hue except a neutral band fill.

- **The verdict first**, then the settlement figure at 2.1rem tabular, then
  "can it still move" as a live/settled chip, then evidence, then provenance as a
  footnote.
- **The green arc** (the signature): every reading sits on a strip spanning this
  station's p10–p90 *for this hour and season*, with p25–p75 shaded and the
  reading as a tick. Straight out of the instrument-panel language the dashboard
  already speaks. It answers "is 5.9°F unusual?", which the bare number never
  could.
- **The reliability table became a ranked bar chart** where the bar is drawn to
  *measured hit* and a tick marks the *stated* probability — so the
  overconfidence is the literal gap between them. KLAX HIGH's bar reaches 52%
  with its claim tick at 60%.

**The arc says UNUSUAL, never GOOD or BAD.** Which direction favours the bet is
not measured, and colouring it by an assumed valence would put invented physics on
the dashboard — the error class this project has unwound twice.

`board.evidence()` is assembled in Python, not the browser, so the terminal and
the dashboard cannot drift into showing different verdicts for the same
observation.

### Cost

`climatology_bands` is one ~57k-row scan (`obs` is indexed on
`(station, valid_utc)` only, so an hour-of-day filter cannot use it), shared
across both metrics: **0.13s**, served per request like `/api/cities`.

---

## Scoreboard

This project is now **3-for-9** on confident pre-measurement hypotheses. The
preflight is the third to clear a pre-registered gate, after the `fcc_*`
bias-corrected columns and the KLAX HIGH warming-rate features.

## Not done / queued

- Re-measure `BANDS` when `HOUR_ACCURACY` is next refreshed; the thresholds drift.
- Re-test KMIA LOW's bands once more settled days exist — it missed by 0.3pt.
- `predictions.bound_gap` is NULL for every day before today, so the preflight
  reports `pending` until the first h4 prediction lands. It cannot be backfilled:
  there is no stored text to re-derive it from.
- Queue item 1 (parity-averaged KLAX `HOUR_ACCURACY`) is still open and the KLAX
  rows in the reliability table remain fixed-even while KMIA's are
  parity-averaged.

---

# Part 3 — Decision-layer work and the full audit (same day, later)

Triggered by Julian: "majority of the model predictions are a lose". The count
was wrong; the money was not.

## The record, corrected twice

My first tally said 8W–4L. That counted **evaluated candidates**; `app_bets`
holds both placed and skipped rows. The truth:

| | record |
|---|---|
| **placed** (model saw an edge ≥5%) | **2W–4L** |
| **skipped** ("no edge") | **6W–0L** |

Of the 6 placed, only 3 bought a contract at all — the $0.40 stake cap zeroed the
rest — and **all 3 real positions lost**.

## What the investigation found

1. **The stated probability is band-dependent and the app bets from the worst
   band.** KLAX LOW at 60–70% *under*-states by 2.9%; at 70–80% it *over*-states
   by 6.8%. The scalar `HOUR_CALIB` (+0.0075) is the average of those and is a
   lie for betting, because the edge gate only fires from the top band. The
   70–80% overstatement is present and significant in **all four markets**
   (+6.8% / +12.8% / +9.5% / +8.3%).
2. **Nothing is measured above 80%** (n = 2–15 per market; KLAX LOW's n=2 hit
   50%). All three losing bets were stated 81%.
3. **The gate selects for the model's errors** — it only fires on disagreement,
   and disagreement is when the model is wrong (0W–3L above +25 pts).
4. **The evening minimum.** KLAX's climate day is 01:00→01:00 local, so it holds
   two cooling periods. On 11.0% of days the low lands at h18+, seventeen hours
   after the h4 commit. That is what happened on 2026-07-30.

## Shipped

- `gating.BAND_CALIB` + `band_calib()` — per-band calibration; returns **None**
  above the highest measured band, which `staking.evaluate` turns into a refusal.
- `staking.evaluate(calib=…)` — correction applied **after** the pick is chosen
  so it cannot reorder the model's read; `Decision.p_hat` carries the corrected
  number while `p_model` stays **raw**.
- **Unbuyable refusal** — a "PLACE" that buys 0 contracts is no longer recorded
  as a position.
- `min_edge` 0.05 → **0.03** — the floor is the fee (2¢/contract at 1 contract
  means a 2-point edge nets exactly zero), per Julian's "prioritise accuracy even
  if the edge is small".
- **NBS fetch in the wake cycle** + `gating.nbs_guidance_ready` +
  `gating.metric_uses_nbs`.

## Out-of-sample check on band calibration — mixed, and shipped anyway

`scripts/band_calib_oos.py`, fit on 2025-01-01..10-15, scored on 10-16..2026-07-28:

| market | RAW | SCALAR | BAND |
|---|---|---|---|
| KLAX high | 0.0726 | 0.0525 | **0.0186** |
| KLAX low | 0.0307 | 0.0315 | **0.0187** |
| KMIA high | 0.0554 | **0.0264** | 0.0299 |
| KMIA low | 0.0217 | **0.0064** | 0.0084 |
| pooled | 0.0451 | 0.0292 | **0.0189** |

BAND wins pooled and fixes KLAX HIGH decisively, but is *slightly worse* on both
KMIA markets where the scalar was already near-perfect. Brier is a tie. Shipped
because the pooled gain is real and the ≥0.80 refusal is wanted everywhere.
**Not** "better everywhere" — do not quote it that way.

## The live bug the audit found

`nbm_nbs` was populated only by the manual `backfill` command, never the wake
cycle, so it ran out on 2026-07-26. The day-ahead LOW consumes seven NBS columns
and drifted 5–7°F cold — 61.5°F against unanimous guidance of 65–68, becoming 35%
on "≤60°F" in July with staking recommending 40 contracts at 1¢.

Fixed, verified end to end: the wake cycle ran the new code at 09:47Z and closed
the gap; the re-run day-ahead went **61.5 → 68.4°F** (KLAX) and **77.3 → 78.6°F**
(KMIA), and KMIA LOW's live pick changed from ≥81 to **79–80**.

## Bugs found in my own work during the audit

1. **`Decision.p_model` was being overwritten with the calibrated value** — it is
   written to `signals`, which holds thousands of raw historical rows, so the
   next calibration run would have corrected an already-corrected number. Fixed:
   `p_model` stays raw, `p_hat` carries the correction.
2. **The NBS veto was not scoped to metrics that read NBS.** The day-ahead HIGH
   consumes **zero** nbs columns, so the guard would have falsely blocked it.
   Fixed via `metric_uses_nbs`, derived from the feature list rather than
   hardcoded.
3. `board.render` printed `wind --` when only the direction was missing, while
   the evidence row above showed the speed.

## Verification

591 tests pass. Full walkthrough clean: all four live decisions refuse for
defensible reasons, dashboard renders with no console errors, CLI board renders,
appbet writes no rows and raises nothing.

## Honest limits

- **No model was changed.** Predictions are identical; the model surface was
  asserted clean. What changed is display, the decision layer, and a broken feed.
- **The counterfactual is not a win.** All changes applied to the 12 historical
  candidates: 6 placed → 1 placed, −0.66 → −0.42 units. It blocks 3 of 4 losses
  and 1 win, and the survivor lost. It makes the *reasons* defensible; it does
  not turn a losing week into a winning one.
- **n = 12.** No P&L rule here is measured to this project's usual standard, and
  only ~6 days of price history exist.
- The absurd-edge guard (`pick.MAX_PLAUSIBLE_EDGE`) remains dormant — offered,
  and Julian chose to investigate the model first.
