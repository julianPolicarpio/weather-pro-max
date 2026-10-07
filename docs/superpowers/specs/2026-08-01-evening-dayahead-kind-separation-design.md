# Separating the evening forecast from the day-ahead

**Status:** design, not built. Both open flags from the first draft are now resolved in-document (§4.5 migration decided; §5 withdrawn as incoherent and replaced).
**Date:** 2026-08-01
**Found by:** Julian, from the dashboard showing KLAX HIGH `≤77` while the nowcast said `82`.

---

## 1. The defect

**Two different forecast objects are stored under one `kind`.** They are produced
by two separate call sites in `cli.auto_cycle`, they see different data, they
have different accuracy, and every consumer treats them as one thing.

```python
if should_predict_dayahead(...):   cmd_predict(date=None,     ...)  # ≥09:00 local, target = today
if should_predict_tomorrow(...):   cmd_predict(date=tomorrow, ...)  # ≥17:00 local, target = tomorrow
```

Both write `kind='dayahead'`.

### Why they cannot be the same object

The HIGH's feature cutoff is **09:00 local on the target day**
(`metric.cutoff_local_hour=9, cutoff_day_offset=0`). The evening run fires at
~17:00 the *previous* day, so that cutoff is ~16 hours in its future. Every
`morn_*` feature — `morn_tmpf`, `morn_dwpf`, `morn_onshore`, `morn_sky` — is
NaN, because the observation it reads has not happened yet. The 09:00 refresh
sees all four.

Same code path. Different data availability. Different object.

### Measured

Stored predictions vs settled `cli_reports`, split by whether the run was issued
before the target day's 09:00 cutoff:

| | KLAX HIGH n | MAE | bias | | KLAX LOW n | MAE | bias |
|---|---|---|---|---|---|---|---|
| **evening** (issued pre-cutoff) | 19 | **4.42 °F** | **−4.05 °F** | | 6 | 3.08 °F | −3.08 °F |
| **morning refresh** | 14 | 1.45 °F | +0.19 °F | | 12 | 1.33 °F | −0.41 °F |

KMIA: LOW evening 2.51 / morning 2.06 (both cold). KMIA HIGH evening n=3 —
too few to read, and it is not used to justify anything here.

⚠️ CORRECTED 2026-08-01 during implementation. An earlier cut of this table
split on "issued before 09:00 local", which is the HIGH's cutoff. The LOW's is
20:00 on the PRIOR day (`cutoff_local_hour=20, cutoff_day_offset=-1`), so a LOW
run issued at 21:00 that evening is AFTER its own cutoff and its features are
complete. The correct split is "issued before its own feature cutoff". The HIGH
figures are unchanged — the two rules coincide there — but the LOW's evening
penalty is larger than first reported (3.08 not 2.66) and its morning bias
smaller (−0.41 not −0.76). The migration's safeguard caught this by refusing to
run on a row count that disagreed with the spec.

**KLAX HIGH is three times worse with a systematic 4 °F cold bias.**

### Why this invalidates the rule that serves it

`gating.preferred_prediction` prefers the day-ahead for the HIGH before the
commit hour, justified in its own docstring by a walk-forward:

> HIGH @ 9am local (N=1364) — day-ahead MAE 1.509 / intraday 1.620

That 1.509 matches the **morning** object (1.45) almost exactly and is nothing
like the evening object (4.42). The walk-forward rebuilds features from the full
historical record, so every day-ahead it scored had real `morn_*` values —
**the evening object has never been in any measurement in this project.** The
preference rule is applied to an object its evidence never covered.

This is a train/serve skew, not a modelling error: the model is fine, it is
being fed NaN and asked for an answer anyway.

---

## 2. Blast radius

**Contamination is close to half the record: 35 of 74 stored day-ahead rows
(KLAX 27, KMIA 8) are evening rows.** Any statistic
computed over `kind='dayahead'` mixes a 1.45 °F population with a 4.42 °F one.

Live harms, all currently active:

1. **The dashboard headline.** 00:00–09:00 local, KLAX HIGH shows the evening
   object. On 2026-08-01 that was 72.9 °F (`≤77`) against a nowcast of 81.1 °F
   (`82`) — a 9 °F gap — with a **69 % "edge"** computed off it.
2. **Both KLAX HIGH preview alerts.** `PREVIEW_HOURS["KLAX"]["high"] = (6, 8)`,
   and the refresh fires at h9. Every preview at 21:00 and 23:00 Manila has been
   built on the 4 °F-cold object.
3. **Any future consumer** that filters `kind='dayahead'`. There are ~18 such
   sites across `signals.py`, `scoring.py`, `gating.py`, `cli.py`, `web/app.py`.

Not harmed: `app_bets`. Bets commit at the decision hour (KLAX HIGH h13), where
`preferred_prediction` already returns the intraday. The paper ledger is sound.

---

## 3. Why the obvious fix is only a guard

`signals._regime_veto` **already detects this correctly** — it is what prints
*"every morn_\* feature was NaN — a regime the model was never trained on. Not
tradeable."* The tempting fix is to make `preferred_prediction` consult it.

That would work, and the veto is the right predicate. But it treats the
symptom: the two objects still share a label, so the next consumer to query
`kind='dayahead'` re-inherits the bug, and every accuracy statistic stays
contaminated. A guard that must be remembered at each of ~18 call sites is a
guard that will be forgotten at one.

**Fix the representation, keep the guard as a second line.**

---

## 4. Design

### 4.1 A distinct kind

The evening run writes `kind='evening'`. One call site changes.

Consequences, all of them the point:

- `preferred_prediction` never sees it as a day-ahead candidate. No veto needed
  in the hot path; the bug becomes unrepresentable rather than caught.
- Accuracy, calibration and hour curves computed over `kind='dayahead'` become
  single-population.
- The evening forecast keeps its real job — a view of tomorrow's market the
  night before — and consumers that genuinely want it ask for it **by name**.
  The dashboard's "tomorrow" card is the one such consumer and is updated to
  query `kind='evening'`.

### 4.2 What the pre-09:00 window shows instead

With no day-ahead stored for today until 09:00, `preferred_prediction` falls
through to the intraday — which is the correct answer and the sharper one
(KLAX HIGH intraday MAE 0.74 at h13 vs the day-ahead's 1.35).

Between 17:00 local and the first intraday of the target's climate day
(01:00 local at KLAX) neither exists, and `preferred_prediction` returns
`(None, None)`. Surfaces must render "no view yet" rather than a number. That
is a real state and is better than the 9 °F-wrong number it replaces.

### 4.3 The regime veto stays

`_regime_veto`'s `morning_obs_ready` check remains exactly as written. After
this change it should never fire on a `kind='dayahead'` row; a test asserts that
it still fires on an `'evening'` row, so the guard cannot rot into a no-op that
nobody notices.

### 4.4 KLAX HIGH preview hours

`(6, 8)` both precede the 09:00 refresh, so after this change they would preview
the *intraday* at hours the intraday is weakest, having previously previewed a
4 °F-cold day-ahead. Neither is defensible.

They were also chosen when the HIGH committed at **h11**; it now commits at
**h13** and they were never re-measured — the same staleness already found in
`PREVIEW_HOURS["KMIA"]["low"]` on 2026-08-01.

**v1 disables KLAX HIGH previews** (`"high": ()`), and re-deriving the right
hours is a separate, measured piece of work. Removing an alert that has been
systematically 4 °F cold needs no further justification; choosing its
replacement does.

### 4.5 Migration — DECIDED: relabel in place

35 existing rows are mislabelled (KLAX 27, KMIA 8). They are
**relabelled, not deleted and not left alone.**

Leaving them was rejected because the contaminated rows are the majority, so
every future backfilled statistic would silently mix a 1.45 °F population with a
4.42 °F one — the exact defect this spec exists to remove, preserved in the
record. Deleting them was rejected because they are the only evidence of the
evening object's behaviour, and §1's table cannot be reproduced without them.

**The predicate**, stated once and used verbatim by both the migration and the
invariant test in §6:

> A `kind='dayahead'` row is an EVENING row when it was **issued before its own
> feature cutoff**: `issued_at_utc < _cutoff_utc(target_date +
> metric.cutoff_day_offset, metric.cutoff_local_hour)`.
>
> The cutoff is **metric-specific** — 09:00 on the target day for the HIGH,
> 20:00 on the PRIOR day for the LOW. Using the HIGH's rule for both is the
> error that produced the wrong count of 38.

Safeguards, all required, in order:

1. Copy the database file first; print the copy's path before touching anything.
2. Count the matching rows and **assert the count is exactly 35** before the
   write. A different count means the predicate or the data has changed since
   this spec, and the migration must stop rather than proceed on a guess.
3. Perform the `UPDATE` in one transaction.
4. Re-run the §1 split afterwards and assert it now partitions on `kind` alone,
   with the same MAE/bias figures. A migration that changes the numbers has
   moved the wrong rows.
5. Idempotent: a second run matches zero rows and exits cleanly.

### 4.6 Components

| file | change |
|---|---|
| `src/highcast/cli.py` | the `should_predict_tomorrow` call site writes `kind='evening'`; `cmd_predict` gains a `kind` argument |
| `src/highcast/gating.py` | no change — it simply stops finding evening rows |
| `src/highcast/web/app.py` | the "tomorrow" card queries `kind='evening'`; surfaces handle `(None, None)` |
| `src/highcast/notify.py` | `PREVIEW_HOURS["KLAX"]["high"] = ()` |
| `scripts/migrate_evening_kind.py` (new) | one-shot, idempotent, backs up first, asserts the row count |

---

## 5. RESOLVED — nothing needs re-deriving, and here is why

An earlier draft of this spec flagged `HOUR_ACCURACY` and the walk-forward
behind `preferred_prediction` as possibly contaminated, and asked for them to be
re-run "with the evening rows excluded". **That instruction was incoherent and
is withdrawn: a walk-forward contains no evening rows to exclude.**

`features.build_features(conn, cfg, metric, dates, require_target, cutoff_local_hour)`
takes **no as-of or issue-time parameter**. The feature cutoff is derived purely
from the target date (`_cutoff_utc`, 09:00 local for the HIGH). A backtest
therefore reads observations that exist in the historical record and always
gets real values:

```
  2026-07-29: morn_* = 6 cols, NaN = 0   {morn_tmpf: 73.0, morn_dwpf: 63.0, morn_onshore: 1.7}
  2026-07-30: morn_* = 6 cols, NaN = 0   {morn_tmpf: 75.0, morn_dwpf: 63.0, morn_onshore: -0.0}
  2026-07-31: morn_* = 6 cols, NaN = 0   {morn_tmpf: 75.0, morn_dwpf: 64.0, morn_onshore: -3.8}
```

So **the walk-forward is structurally incapable of producing the evening
object.** Every day-ahead it has ever scored had real `morn_*`. Two consequences,
and they point in opposite directions:

1. **The measurements are clean.** `1.509` is a valid figure for the morning
   object, `HOUR_ACCURACY` is uncontaminated, and `preferred_prediction`'s
   preference rule stands **for the 09:00–13:00 window it actually governs**.
   Nothing needs recomputing.
2. **The evening object has never been measured by anything** — not by the
   walk-forward, which cannot construct it, and not by any gate. Its only
   characterisation in this project is §1's table, built from stored live rows
   after the fact. That is precisely why serving it as the headline was wrong,
   and why §4.1 removes it from the decision path rather than adding a threshold
   to it.

The live-versus-backtest gap is a **train/serve skew in the data pipeline, not
an error in the model or in the evidence**. The model is sound; it was handed
NaN and asked for an answer anyway.

### What remains genuinely open, and is NOT part of this change

At the decision hour the intraday beats the day-ahead on all four markets
(0.74/1.35, 0.61/2.25, 0.43/1.23, 0.49/2.05 MAE). That is a different question
from the one the 9am walk-forward answered, and it may eventually retire the
day-ahead preference for the 09:00–13:00 window too. It needs its own
pre-registered gate at the hours in question and is deliberately out of scope.
This spec makes the two objects distinguishable so that question can be asked
honestly; it does not pre-judge the answer.

---

## 6. Testing

- **The evening call site writes `kind='evening'`** — asserted at the call site,
  not by mocking `cmd_predict`.
- **`preferred_prediction` never returns an evening row.** Store one, store
  nothing else, assert `(None, None)` rather than the evening prediction.
- **The pre-09:00 window falls through to the intraday.** Store an evening row
  and an intraday row for the same target; assert the intraday is chosen at h6.
- **`_regime_veto` still fires on an evening row** — the guard must not rot into
  a no-op once the representation makes it redundant.
- **The tomorrow card still finds its forecast** after the rename.
- **The migration predicate selects exactly the rows §1 describes** — tested on a
  fixture DB with one known evening row and one known morning row, asserting it
  moves the first and not the second.
- **A freeze-style test that no `kind='dayahead'` row exists whose issue time
  precedes its target's cutoff.** This is the invariant the whole change buys;
  without it, a future call site can silently reintroduce the mix.

---

## 7. Honest limits

- **The n behind the headline numbers is small** — 19 evening and 14 morning
  KLAX HIGH days. The 4.42 vs 1.45 gap is large enough and one-directional
  enough (bias −4.05) that it is not a sampling artifact, but the exact
  magnitudes will move.
- **KMIA HIGH shows the opposite sign** (evening 0.70 vs morning 1.57) on n=3
  and n=5. That is noise, and it is recorded here so nobody later mistakes it
  for a counter-example that was hidden.
- **This fixes representation, not accuracy.** No prediction becomes better. The
  gain is that a known-bad prediction stops being served as the headline and
  stops contaminating statistics.
- **The evening forecast remains unmeasured**, and §5 shows it cannot be
  measured by the existing walk-forward at all — that machinery reconstructs the
  morning object by design. Characterising it would need a backtest that
  respects issue time, which does not exist here. After this change it is
  honestly labelled and out of the decision path; if it is ever to be acted on
  rather than glanced at, it needs both that machinery and a gate.
- **The day-ahead preference for 09:00–13:00 survives this change** (§5), but is
  independently questionable given the intraday's decision-hour advantage on all
  four markets. That is a separate pre-registered gate, not part of this work.
