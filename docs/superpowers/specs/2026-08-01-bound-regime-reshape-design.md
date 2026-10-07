# Bound-regime pmf reshape — KMIA LOW

**Status:** ⛔ **NULL RESULT — DO NOT BUILD.** The pre-registered gate was run
early, in prototype form (`scripts/bound_regime_probe.py`), before Tasks 2-6 were
written. It FAILED out of sample. See §8. Task 1's module `boundregime.py` exists
and is disabled (`gating.BOUND_REGIME = {}`) with no caller.
**Date:** 2026-08-01
**Scope:** KMIA LOW at h7 only. Jun–Sep only. The frozen model is not retrained.

---

## 1. The defect

The intraday model already consumes the observed running extremum as
`run_lo` / `run_hi`. What it lacks is any way to know **which of its two
information sources to trust today**, and its output distribution cannot
represent the shape the outcome actually has.

Two distinct failures, both measured:

### 1a. Regime blindness — already documented in the codebase

`models/intraday.py` computes a candidate feature `cur_bound_gap = bound −
base_p50` and deliberately does not ship it. Its own comment records the
measurement (KLAX LOW, h4):

| min lands | n | bound MAE | guidance MAE |
|---|---|---|---|
| before h4 | 367 | 1.22 | **1.10** |
| h5–h7 | 144 | **2.35** | 1.24 |
| h19–h23 | 62 | **2.45** | 1.14 |

> *"Guidance error is FLAT; the bound's DOUBLES. The model weights the bound
> identically in both regimes because nothing tells it which regime it is in."*

**2026-08-01 KMIA is that blind spot in reverse.** The minimum had already
landed (82.04 °F, flat three hours, sun up) — the regime where the bound is the
reliable side — and the model still emitted a point forecast of 80.0, tracking
guidance sitting 2–5 °F below an observed fact.

### 1b. The outcome distribution is zero-inflated; the model's is smooth

Let **Δ = bound − settled low**, the drop still to come after the decision hour.
Over 789 KMIA Jun–Sep days, Δ is a spike at zero plus a decaying tail:

```
  Δ ≈ 0.0 F   514  #######################################
  Δ ≈ 1.8 F   135  ##########
  Δ ≈ 3.6 F    95  #######
  Δ ≈ 5.4 F    31  ##
  Δ ≈ 7.2 F     9
  Δ ≈ 9.0 F     4
```

65 % of days never go below the bound at all. **The median Δ is 0.16 °F in
*every* regime cell** — all the signal is in the tail probability, none of it in
the centre. A point forecast cannot carry that, and a smooth unimodal pmf cannot
either: it must spread mass across the middle, which is where the outcome is
*least* likely.

Consequence on 2026-08-01, KMIA LOW:

Bound 82.14 °F, cell (81–83, gap 1.5–2.5), n=95:

| bucket | model | this layer | market ask |
|---|---|---|---|
| ≥81 | 42.4 % | **58.9 %** | 0.43 |
| 79–80 | **37.4 %** | **22.1 %** | 0.41 |
| 77–78 | 14.9 % | 15.8 % | 0.09 |
| 75–76 | 4.6 % | 2.1 % | 0.05 |
| 73–74 | 0.6 % | 1.1 % | 0.04 |
| ≤72 | 0.1 % | 0.0 % | 0.02 |

P(low ≤ 80): layer **41.1 %**, model 57.6 %, market ≈57 %.

The model puts 37.4 % on the bucket adjacent to an already-observed minimum,
where the regime-conditional rate is 22 %. Not because 79–80 is impossible —
because a drop that starts usually **overshoots straight past it**.

### 1c. Guidance disagreement is real signal, and the model over-reacts to it

The first version of this analysis was blind to today's forecast. It should not
have been — the disagreement predicts the drop:

| bound − guidance median | n | P(Δ ≥ 1.5 °F) |
|---|---|---|
| < 0.5 (guidance agrees) | 301 | **19.6 %** |
| 0.5–1.5 | 161 | 36.0 % |
| 1.5–2.5 (2026-08-01 ≈ +2.1) | 182 | 40.1 % |
| > 2.5 | 145 | **44.1 %** |

Correlation +0.281, n=789. Disagreement **doubles** the drop rate. So guidance
belongs in the regime key.

But it does not rescue the model's number: **even the most guidance-pessimistic
cell tops out at 44 %**, against the model's 57.6 %. The model does not ignore
guidance — it over-weights it in exactly the regime where the bound is reliable.

---

## 2. Two look-ahead traps, both found and closed

Recorded because both would have inflated the measured skill, and each was
present in an earlier draft of this analysis.

**2a. The bound must be evaluated at the decision hour exactly.**
An earlier cut used a fixed 12:00 Z. KMIA LOW commits at **h7 local = 11:00 Z
under EDT**. The bound is still falling between h7 and h8 on **40.2 %** of days
(161/400), so one hour of convenience was one hour of look-ahead. On the
2026-08-01 worked example it inflated P(≥81) from **58.9 % to 65.8 %** — nearly
seven points of skill the live system could never have had. Every fit and every
gate evaluates the bound with `asof = the decision hour`, never later.

**2b. Guidance must be restricted to `run_date < target_date`.**
`forecasts.run_date` is a DATE, not a timestamp, so a same-day row may be a 00 Z
run (available at 11 Z) or an 18 Z run (not). And **`fetched_at` cannot be used
as an availability proxy**: `db.upsert` is `INSERT OR REPLACE` and the
`forecasts` primary key excludes `fetched_at`, so it is overwritten on every
re-fetch — the artifact found on 2026-07-31.

The only strictly-safe filter is previous-day runs. Verified that the signal
survives it:

| filter | n | corr(gap, Δ) | gradient across gap bands |
|---|---|---|---|
| `run_date <= target` (unsafe) | 789 | +0.281 | 20.6 → 35.3 → 40.2 → 45.9 % |
| `run_date < target` (**used**) | 789 | +0.281 | 19.6 → 36.0 → 40.1 → 44.1 % |

Identical. The guidance dimension costs nothing to make safe.

---

## 3. Design

A post-hoc layer. The model stays frozen; nothing is retrained.

```
  predictions (frozen model pmf)
            │
            ▼
  ┌─────────────────────┐     bound  = visible_running_extremum(min) at h7, hi end
  │  boundregime.py     │ ◄── gap    = bound − guidance_median  (defined below)
  │  (pure, no network) │
  └─────────────────────┘
            │  cell → empirical Δ distribution → candidate pmf
            ▼
     blend: w·model + (1−w)·layer        w fitted per cell, DEFAULT w = 1.0
            │
            ▼
     reshaped pmf → buckets → the same downstream path as today
```

### 3.0 The two inputs, defined exactly

**`bound`** — the upper end of `latentmax.visible_running_extremum(conn, cfg,
day, "min", asof_utc=<the decision hour>)`. Upper end, not lower: it is the
tightest value consistent with everything observed, and the settled low cannot
exceed it.

**`guidance_median`** — the unweighted median of `forecasts.max_temp_f` (which
holds the LOW for `metric='low'`) over **every** model row for that
`target_date` with `run_date < target_date`. No model allow-list: naming one
would need its own justification, and the set has changed twice already this
month. Days with no such row fall through to the level-only cell (§3.3); in
Jun–Sep this has never happened — 0 of 789 days.

**`gap` = `bound − guidance_median`.** Positive means guidance is colder than
what has already been observed, which is the disagreement §1c shows is
predictive.

### 3.1 Regime key

Two dimensions, both required to be computable at the decision hour:

- **bound level**: `<79`, `79–81`, `81–83`, `≥83`. Required, because Δ is *not*
  translation-invariant — pooling across levels would import cool-morning
  behaviour into a warm-morning forecast. Measured at h7, Jun–Sep, n=789:

  | bound level | n | P(Δ ≥ 1.5 °F) |
  |---|---|---|
  | <79 | 266 | 22.2 % |
  | 79–81 | 229 | 34.9 % |
  | 81–83 | 193 | 40.4 % |
  | ≥83 | 101 | 36.6 % |

- **bound gap**: `<0.5`, `0.5–1.5`, `1.5–2.5`, `≥2.5` (§1c)

Observed cell occupancy at h7, Jun–Sep, using the §2b safe guidance filter.
Every day has previous-day guidance, so nothing is dropped and the rows
reconcile exactly with the level marginals above:

| level \ gap | <0.5 | 0.5–1.5 | 1.5–2.5 | ≥2.5 | row n |
|---|---|---|---|---|---|
| <79 | 215 | 37 | 13 | 1 | 266 |
| 79–81 | 74 | 87 | 49 | 19 | 229 |
| **81–83** | 10 | 33 | **95** | 55 | 193 |
| ≥83 | 2 | 4 | 25 | 70 | 101 |
| | | | | | **789** |

**Only 7 of 16 cells reach `MIN_CELL_N = 40`.** The other nine back off, and
that is the expected steady state rather than a defect — the occupied cells lie
on the diagonal, because a warm morning and a cold guidance forecast are
correlated. The backoff ladder in §3.3 is therefore load-bearing, not a corner
case, and §5 tests it as such.

### 3.2 Estimator

Within a cell, the empirical distribution of Δ. To score today's ladder:

```
P(bucket) = |{ d ∈ cell : round(bound_today − d) ∈ bucket }| / |cell|
```

Zero-inflation is represented natively because it is not modelled — it is
counted.

### 3.3 Blend, backoff, and fail-closed

`final = w · model + (1 − w) · layer`, with `w` fitted per cell on the walk-forward
window. **`w = 1.0` (pure model, layer disabled) is the default and the value
returned on every unknown.** If `w` fits to 1.0 everywhere, the honest answer is
"this did not help" and the gate will say so.

Backoff ladder, in order:

1. full cell (level × gap), if `n ≥ 40`
2. level-only cell, if `n ≥ 40`
3. **pure model** (`w = 1.0`)

Unconditional fail-closed to the pure model when:

- the bound is unavailable (no obs by the decision hour)
- no guidance row satisfies `run_date < target_date`
- the month is outside Jun–Sep (v1 scope)
- the station/metric/hour is anything other than KMIA / low / h7
- the reshaped pmf does not sum to 1 ± 1e-6

`MIN_CELL_N = 40` is pre-registered here and may not be lowered after seeing
results.

### 3.4 Components

| file | responsibility | depends on |
|---|---|---|
| `src/highcast/boundregime.py` (new) | `regime_key`, `fit`, `reshape`, `blend`. Pure — no network, no DB writes. | `latentmax`, `buckets` |
| `src/highcast/gating.py` | holds the fitted `BOUND_REGIME` table, same as `HOUR_CALIB` / `BAND_CALIB` | — |
| `scripts/bound_regime_eval.py` (new) | walk-forward evaluation, emits the gate verdict and the fitted table | `boundregime` |
| `tests/test_boundregime.py` (new) | rails below | — |

Deliberately **not** touched: `models/`, `signals.py`, `orderbot.py`. The layer
sits between the stored prediction and the bucket pmf, and only for one
(station, metric, hour).

---

## 4. Pre-registered gate

Binding. Written before the walk-forward is run.

**Primary metric:** mean log score of the six-bucket ladder pmf, KMIA LOW at h7,
Jun–Sep held-out days, layer vs frozen model. A proper scoring rule, because the
change is a distribution reshape — bucket hit rate is too coarse to detect
"moved mass out of the middle".

**Decision rule:** ship only if all three hold.

1. Log score improves, paired bootstrap over days, **p < 0.05**. One hypothesis,
   one market → no Bonferroni correction.
2. **Guardrail — accuracy.** Bucket hit rate at h7 (the layer's argmax bucket vs
   settled) is not worse than the frozen model's by more than **1.0 pt**.
3. **Guardrail — calibration.** The layer must not buy log score by becoming
   overconfident, which is the failure `HOUR_CALIB` exists to correct. Measured
   as **mean absolute calibration error of the top bucket**: bin held-out days by
   stated top-bucket probability in 10-pt bins, take |observed frequency −
   mean stated| per bin, weight by bin count. The layer's value must not exceed
   the frozen model's by more than **2.0 pt**. Bins with fewer than 10 days are
   excluded, and that exclusion is fixed here rather than chosen later.

**Discovery bias — the part that is easy to cheat.** This effect was *found* in
this data, by me, on 2026-08-01. An expanding-window walk-forward over the same
history is therefore contaminated: I already know the answer. So the gate is
evaluated on a **held-out period never inspected during discovery**:

- **fit on:** Jun–Sep, 2020 through 2024
- **evaluate on:** Jun–Sep 2025 and 2026 — untouched by any query in this design

If the held-out n is under 60 days, the result is reported as *indicative* and
the layer stays disabled until a further season accumulates. Not negotiable
after the fact.

**Negative control:** the identical pipeline is run with the regime key
**shuffled across days**. If the shuffled key also passes, the gate is measuring
overfitting and the real result is void.

---

## 5. Testing

- **Fail-closed on every unknown** — one test per condition in §3.3; each must
  return the input pmf *identically*, not approximately.
- **`w = 1.0` is an exact identity.** Asserted on the object, not on rounded
  probabilities.
- **Total probability is preserved** across every reshape.
- **No look-ahead, asserted structurally, not by comment:**
  - the bound is requested with `asof` equal to the decision hour — a test that
    fails if any caller passes a later time;
  - guidance selection uses `run_date < target_date` — a test that fails if the
    query is loosened to `<=`;
  - a test asserting `fetched_at` is **not** referenced in any availability
    filter, with a comment pointing at the `INSERT OR REPLACE` artifact.
- **Scope containment:** the layer is a no-op for KLAX, for `high`, and for any
  hour other than h7. This is a freeze-contract-style test, so widening scope
  requires deleting an assertion rather than happening by accident.
- **`MIN_CELL_N` is pinned at 40** by a test whose docstring records that it was
  pre-registered, so lowering it is a visible act.
- **A cell-occupancy regression test** on the fitted table, so a silent collapse
  in cell counts (e.g. a guidance feed dying) surfaces as a failure rather than
  as universal backoff to the pure model.

---

## 6. Honest limits

- **KMIA LOW, Jun–Sep, h7 only.** Nothing here transfers to the HIGH markets
  without its own gate — highs have the opposite shape, with the extremum
  landing late, so the bound is uninformative for most of the day. That includes
  KMIA HIGH, which is the market the order bot trades.
- **Today's cell is n=100.** The cell used for the worked example in §1b is the
  best-populated one on the board; corners have single digits.
- **Guidance median is unweighted**, and `gfs`/`hrrr` are a known duplicated
  channel — so the median is computed over a set where one model votes twice.
  Left as-is deliberately: fixing it is a separate, already-specified change,
  and folding it in here would confound the gate.
- **The effect was discovered in-sample.** §4's held-out period is the mitigation,
  not a cure. A layer that passes on 2025–26 and then fails live has told us
  something, and the `w = 1.0` default means failing live costs nothing.
- **This does not make the model right.** It compensates for a shape the model
  cannot express. The real fix is §1a's `cur_bound_gap` feature plus a
  distributional head, and that remains a separate, larger piece of work.


---

## 8. NULL RESULT — the gate failed, 2026-08-01

Julian asked "did you test that it's actually worth doing?" before Tasks 2-4 were
written. The honest answer was no: every number in §1 was in-sample. So the §4
gate was run early with throwaway code, which turned an unknown into a null for
the cost of an hour instead of three tasks of machinery.

**Held out: Jun-Sep 2025-26, 179 scored days.** Fit on Jun-Sep 2020-24. Model
pmfs regenerated by `backtest.walk_forward_at_hour` (expanding-window).

| arm | mean log-score Δ | bootstrap p | bucket hit | verdict |
|---|---|---|---|---|
| **real regime key** | **−0.0354** | 0.73 | 63.1% → **59.2%** (−3.9 pt) | **FAIL** |
| shuffled control | +0.0144 | 0.25 | 63.1% → 63.1% | FAIL |

**The shuffled key outperformed the real one.** That is the decisive line: a
randomly permuted regime key beating the real one means the key carries no
transferable information. The structure in §1 is a property of the sample it was
found in.

### Diagnostic: fixed weight on held-out days, no fitting

| model weight | mean log score | bucket hit |
|---|---|---|
| 0.00 (pure layer) | **−1.0976** | 59.2% |
| 0.50 | −0.9934 | 62.6% |
| 0.75 | −1.0001 | **64.8%** |
| 1.00 (pure model) | −1.0200 | 63.1% |

The **pure layer is the worst arm on the board**, so the empirical Δ distribution
does not transfer. A fixed mid-range weight looks marginally better than the
model — but that value is being read off the same 179 days it would be validated
on, which is this document's own discovery-bias failure repeated one level up.
The margins are also ~1.7 pt of hit rate on 179 days, roughly three days.
Pursuing it would need a third untouched period, and there isn't one.

### Why it probably failed

The intraday model **already consumes `run_lo`/`run_hi`** — it has the observed
bound. The residual information in the regime key was mostly noise, and
reshaping on it added variance without adding signal. §1's premise, that the
model "cannot represent" the zero-inflated shape, was correct about the shape and
wrong about the consequence: the model's smooth pmf was already closer to right
than the empirical reconstruction.

### What survives

- §2's two look-ahead traps are real and were worth finding: the bound must be
  read at the decision hour (40.2% of days it is still falling at h7→h8), and
  `fetched_at` can never be an availability filter.
- Task 1's primitives (`bound_for`, `guidance_median`, `regime_key`) are correct
  and tested. They are the substrate for any future attempt, and cost nothing
  disabled.
- The habit that produced this result: **run the gate before the machinery.**
  The plan had it at Task 5. It belongs at Task 0.
