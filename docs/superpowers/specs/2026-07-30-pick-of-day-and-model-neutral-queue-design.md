# Pick of the Day + the model-neutral queue — design

**Date:** 2026-07-30
**Status:** design, awaiting Julian's review
**Constraint:** the model is FROZEN (live testing since 2026-07-28). Nothing in
this spec may change a prediction, a probability, or when a bet commits.

> **SUPERSEDED 2026-08-02 — THE FREEZE IS FULLY LIFTED.** Everything below
> describes the partial unfreeze that held for two days. All four markets are now
> unfrozen; see
> `docs/superpowers/specs/2026-08-02-full-unfreeze-decision-record.md`. The
> section is kept because its *reasoning* — what a scorecard costs to spend, and
> the byte-identical check that proved the other three markets were untouched —
> is the pattern to reuse, not because its status line is still true.

## ⚠️ PARTIAL UNFREEZE: KLAX HIGH ONLY (2026-07-31, Julian's call)

**KLAX HIGH is unfrozen. KLAX LOW, KMIA LOW and KMIA HIGH remain FROZEN.**

Julian's instruction after KLAX HIGH lost twice in four days. Scope is deliberately
narrow, and the consequence is stated rather than glossed:

- **KLAX HIGH's test-week comparison ENDS here.** Any change to it invalidates the
  app-vs-Julian scorecard for that market. Its record to date (2W-1L, plus
  2026-07-30 pending) is what it is; nothing after this is comparable.
- **The other three markets' scorecards remain valid**, because nothing they
  consume changes. This is asserted mechanically, not assumed: with the candidate
  columns present in the frame the shipped model's predictions are **byte-identical**
  (max |diff| 0.0 over a 30-day block), because `IntradayModel` selects exactly
  `INTRADAY_FEATURES` and the frame merely got wider.

**Why KLAX HIGH and not something else.** It is the worst market on the board --
52.4% bucket hit at its commit hour, overconfident by 7.4 points, and the only
market to lose during the test week. Measured 2026-07-31, the cause is structural:
at h11 the ratchet bound sits a mean **2.88F** from the value that settles, versus
**1.63F** for the KLAX LOW at h4. The HIGH is asked to forecast the whole afternoon
against a 2F ladder.

**The mechanism it was blind to.** `cur_sky` derives from `skyc`, which stores the
LAST sky token -- the HIGHEST layer. LA's marine layer is a LOW deck whose depth
governs burn-off, and `BKN008 OVC200` was stored identically to `FEW200`. The model
could not see the one thing that decides a coastal max.

### ❌ NEGATIVE RESULT: the environment features do NOT earn their place

Measured 2026-07-31, `scripts/klax_high_env_eval.py`. Walk-forward, refit 45,
matched-hour, parity-averaged bucket hit, test >= 2025-01-01, **n=573**:

| arm | MAE | HIT | dHIT | 95% CI on dHIT |
|---|---|---|---|---|
| BASE | 1.104 | 51.7% | -- | -- |
| ENV4 (deck+ceiling+vis+tendency) | 1.077 | 52.6% | +1.0% | [-1.2%, +3.0%] |
| DECK2 (deck + tendency) | 1.088 | 51.7% | +0.0% | [-2.0%, +2.0%] |
| **DECK1 (deck alone)** | 1.103 | 50.3% | **-1.3%** | [-3.1%, +0.4%] |
| RAND (4 noise columns) | 1.096 | 50.3% | -1.4% | [-3.2%, +0.4%] |

**`cur_sky_lowest_ft` ALONE SCORES IDENTICALLY TO PURE NOISE** (-1.3% vs -1.4%).
The single column the hypothesis rested on -- and the one argued for most
confidently -- is indistinguishable from a random number.

**The subset test refutes the dilution defence.** Had correlated columns been the
problem, FEWER should have scored BETTER. The opposite happened
(+1.0 -> +0.0 -> -1.3 as columns were removed), so ENV4's +1.0% is best read as
the maximum of several arms tried rather than a signal. Selecting the best arm and
then testing it is precisely how a false positive is manufactured; gate rule 5
(CI excludes zero) caught it.

**A flaw in the gate, admitted.** The gate as originally pre-registered had four
rules and **omitted a significance requirement** -- so it returned SHIP on the
first run despite a CI spanning zero. Rule 5 was added after seeing that, which is
itself a deviation from pre-registration and is recorded as such rather than
presented as if the gate had always been right.

**What this does NOT license.** It does not show the marine layer is irrelevant --
it shows that the deck's HEIGHT AT h11, as a level, carries no measurable
information the model lacks. A different hypothesis remains untested: the deck's
RATE OF CHANGE over the morning (is it eroding?), which is a different quantity
from its level and is what "burn-off timing" actually means. Per HANDOFF's standing
rule, a retry must change the HYPOTHESIS, not the tuning.

**Scoreboard: this project is now 1-for-6 on confident pre-measurement
hypotheses.** The candidate columns remain in the frame -- no model consumes them,
so they cost nothing and the next experiment is a one-liner. The KLAX HIGH nowcast
is UNCHANGED.

## Audit log (2026-07-30, `/goal audit and backtest everything`)

Bugs found and fixed after the initial build:

1. **Divergence alert false positive.** The "confidence collapsed" floor trigger
   fired on any bet that COMMITTED below 60%, even with zero probability change —
   it read "53% → 53% (0%)". Caught live on the KMIA LOW. Fixed: the floor now
   requires confidence to have been ABOVE it at commit. Regression test added; one
   pre-existing test (missing-commit-probability fires on floor) was deliberately
   reversed, because a collapse cannot be claimed without knowing the start point.
2. **Churn leakage via local-hour proxy.** `churn_to_commit` excluded post-commit
   readings by `local_hour > commit_hour`, which cannot see date wrap — a
   prediction issued next-day at a small local hour leaked in (measured: 1 row per
   KLAX market). Fixed to a chronological cutoff at the END of the commit hour
   (the commit reading itself must count), never past `now`. Regression test added.

Audited clean (no bug): `fees.py` (published example, maker=25%, boundary raises,
fair-bet EV=0, dust rounding — the per-contract sawtooth is correct round-up-per-
order behaviour, not a defect); SLP/tendency parse (0 out-of-range over 106k rows,
decode exact vs raw on 500 samples, cross-checks altimeter); board group
admissibility (73 days/station, correct straddler rejected); stall clock (bounded);
`latentmax` invariants (0 violations / 3,596 states / 300 days); integrity sweep
(0 across obs/CLI/predictions/app_bets/picks); migration idempotency; `close_time`
additive change breaks no consumer. Model output unchanged: obs columns
byte-identical vs the pre-reparse backup on all 8 model-read columns.

---

## 1. Why, and what this is not

Julian asked for two things: a Pick-of-the-Day discipline so he knows which of
the day's predictions is most accurate and stable, and the queued improvements
that do not touch the model.

**Pick of the Day is a display and ranking layer.** It never gates
`staking.evaluate`, never writes to `app_bets`, and never changes the app's own
committed prediction. `app_bets` remains the immutable accuracy ledger with its
flat `DEFAULT_STAKE = 1.0` unit stake; Pick of the Day is advice about *Julian's*
money, computed alongside it.

**It protects against overpaying, not against losing.** Run against all eight
committed bets with the MEASURED calibration of S3.2, it publishes **2 picks**
(both KLAX LOW: 07-27 which won, 07-29 which is still live) and declines the other
six — including all four HIGHs and the 07-28 LOW at 80c, which won anyway.
Winning a badly-priced bet does not make it a good bet. Per HANDOFF's BIAS
DISCIPLINE the outcome is not the acceptance test; the arithmetic is.

**Read that result with the following in front of you.** The 07-29 KLAX HIGH,
which lost, scored **EV +0.046 against a MARGIN of 0.050**. The bar declined a
loser by four thousandths of an EV unit, on a threshold that S8 shows cannot be
backtested. n=2 published picks establishes nothing, and the single apparent
success is not evidence the constants are right. Anyone reading this table as
validation has misread it.

### What it is NOT copied from

nowcast.trade's Pick of the Day supplies the *shape* only. Their bar is
undisclosed ("the methodology is exclusive to NowCast and not disclosed"), so
there is no mechanism to copy. Their disclosed average entry of 71.6c requires a
**73.0%** true hit rate to break even after fees; our best commit-hour hit is
KLAX LOW at 62.4%. Buying at their price point loses 10.6c/contract on our best
market and 20.6c on KLAX HIGH. The discipline transfers; the number does not.

---

## 2. Fact-check results that shaped this design

Verified before writing. **Seven of my own earlier claims were wrong and one is
unverified** — two of the seven would have shipped as bugs. The seventh is the
worst kind: I claimed the measurement in S3.2 "corrected HANDOFF" on the KLAX LOW
calibration. It did not; I had measured the wrong window and HANDOFF was right.
See the retraction in S3.2. A confident correction of a trusted document, drawn
from a regime that did not match production, is exactly the failure this
fact-check pass exists to catch — and it nearly went into a constant.

| claim | verdict |
|---|---|
| Kalshi fee `0.07*C*P*(1-P)` | CONFIRMED, but `round up(...)` **per order**, and **no settlement fee** ("holding a winner to resolution costs $0"). Maker = 25% of taker. |
| breakeven at 80c = 81.1% | WRONG — **82.0%** at 1 contract. Every breakeven quoted earlier was optimistic by 0.3-1.0 pt. |
| only KLAX has measured `HOUR_ACCURACY` | WRONG — both stations measured. The KLAX-only comment in `gating.py` is stale. |
| KLAX values are pre-parity, KMIA parity-averaged | CONFIRMED (KLAX h4 stored 0.624; parity-averaged is 61.9%). |
| `EXTREMUM_STILL_FUTURE` 85/88/76/95 | CONFIRMED exactly (85.4 / 87.8 / 75.8 / 95.0). |
| market closes 15:59 MNL next day | **UNVERIFIED** — stored `market_json` carries only hi/label/lo/market_price/p_model/yes_bid/yes_ask. No `close_time`. Handoff prose only. |
| `min_confidence` / `stability_readings` vestigial | WRONG — both live in `gating.committed_state`. |
| we need a new churn metric | WRONG — stability already exists as `committed_state.stable`. |
| `committed_state` safe to consume | **NO** — its label comes from `buckets.argmax_pair`, documented as ODD-hardcoded and matching no reliable ladder. |
| `appbet` has no commit-hour upper bound | WRONG — **already fixed**, `LATE_FRACTION = 0.75` at `appbet.py:168`. |

Also verified independently:

- **Divergence fix works.** On the logged 07-29 KLAX LOW predictions the proposed
  rule fires at **20:00 Manila** vs the current rule's **22:00** — the claimed
  two-hour gain, reproduced exactly (81.2% -> 54.7% while 67-68 still ranked
  first). That day's ladder was ODD (67-68, 69-70).
- **`latentmax` holds its invariants.** 0 monotonicity and 0 min-above-max
  violations across 2,880 as-of states / 240 days / both stations. This is the
  invariant nowcast.trade violates (their NYC RUNNING HIGH went 78.1 -> 75.0 ->
  73.9 -> 78.1).
- **Six-hour group selection is correct.** Our first-principles admissibility
  rule reproduces the competitor doc's S17 observed posting times on both
  stations, *including which report to reject* — KLAX drops 12Z (files 04:53
  local), KMIA drops 06Z (files 01:53 local, their "overnight Min" warning).
  KMIA's 00Z files 19:53 local against their stated "Eastern ~7:50-7:55 PM".
- **Baseline suite: 385 passed, 12 deselected.** The handoff's "362" is stale.

### The affordability defect (new, found during this pass)

`staking.size()` returns `stake = min(0.25*kelly, 0.05) * 8.0 <= $0.40`, then
`contracts = int(stake // price)`. **Above 40c that is 0 whole contracts.** The
desk can therefore say PLACE and hand back a size that cannot be transacted.

At the current bankroll the only affordable picks sit at <=40c — precisely the
band where our own standing rule says a large claimed edge is usually the model
being broken, while the 60-88c range where the model is reliable is unbuyable.
**This is a bankroll decision only Julian can make.** Pick of the Day surfaces it
and never silently recommends an untransactable trade.

Note this does NOT affect `app_bets`, whose flat `DEFAULT_STAKE = 1.0` is
deliberate ("variable sizing would let a lucky big bet mask a poor hit rate").

---

## 3. Pick of the Day — architecture

New module `src/highcast/pick.py`. Pure scorer plus a thin persistence layer.
No model imports beyond reading `predictions`.

### 3.1 `fees.py` (new, small)

```
taker_fee(contracts, price) -> float   # ceil(0.07*C*P*(1-P) * 100)/100
maker_fee(contracts, price) -> float   # ceil(0.0175*C*P*(1-P) * 100)/100
breakeven(price, contracts=1) -> float # price + taker_fee/contracts
```

Rounding is up, per order, per Kalshi's published schedule. No settlement term.
Reproduces their calculator: 10 contracts at 65c -> 0.159 -> **$0.16**.

### 3.2 `gating.HOUR_CALIB` (new constant, MEASURED)

`{station: {metric: {hour: mean_stated - realised_hit}}}`, positive meaning the
model **overstates** its modal bucket. Measured by `scripts/calib_eval.py`
(walk-forward, refit 30, matched-hour at the configured bandwidth,
parity-averaged — same regime as `hour_curve.py`).

| market | n | stated | realised | **CALIB** | 95% CI |
|---|---|---|---|---|---|
| KLAX low h4 | 1,632 | 65.2% | 61.1% | **+0.0415** | [+2.4%, +5.8%] |
| KLAX high h11 | 1,633 | 58.5% | 50.3% | **+0.0824** | [+6.5%, +10.0%] |
| KMIA low h5 | 2,253 | 58.0% | 52.5% | **+0.0557** | [+4.0%, +7.1%] |
| KMIA high h11 | 2,257 | 60.6% | 50.5% | **+0.1010** | [+8.6%, +11.6%] |
| KLAX high h11, model >=80 | 126 | 54.1% | 41.3% | **+0.1290** | [+5.7%, +19.6%] |

**All four markets are overconfident at their commit hours; every CI excludes
zero.** Stated probabilities run 4-10 points hot.

### ⚠️ WHICH WINDOW — the figures above are the WRONG ONES

The table above is the FULL walk-forward window, and it is **pessimistic for the
model that actually ships.** The expanding window starts at 400 training rows and
refits every 30 days, so it averages in early folds where the model held a
fraction of the ~1,800 rows production trains on today. Restricted to the recent
window (test dates >= 2025-01-01, i.e. production-like data volume):

| market | CALIB full | **CALIB recent** | 95% CI (recent) | n | realised hit | stored `HOUR_ACCURACY` |
|---|---|---|---|---|---|---|
| KLAX low h4 | +4.2% | **+0.0075** | [-2.1,+3.5] **ns** | 572 | 62.3% | 62.4% ✓ |
| KLAX high h11 | +8.9% | **+0.0736** | [+4.4,+10.5] SIG | 572 | 51.5% | 52.4% |
| KMIA low h5 | +5.6% | **+0.0151** | [-1.6,+4.5] **ns** | 565 | 56.7% | 56.6% ✓ |
| KMIA high h11 | +10.6% | **+0.0394** | [+0.9,+6.8] SIG | 565 | 56.5% | 55.6% |

**Both LOWs are well calibrated; both HIGHs are significantly overconfident.**
That is the structural asymmetry HANDOFF describes -- a HIGH is read with 88-95%
of its extremum still ahead, a LOW with the ratchet already constraining it.

**Two independent cross-checks passed.** The recent-window values reproduce the
single-holdout figures already committed in `MIN_DECISION_HOUR`'s comments (KMIA
high +4.3% vs +3.9% measured, low +1.2% vs +1.5%), and the realised hit rates
reproduce the stored `HOUR_ACCURACY` values to within a tenth of a point on both
LOWs. The constants the desk quotes are correct.

**This vindicates HANDOFF, and an earlier draft of this spec was wrong to
"correct" it.** HANDOFF reports the KLAX LOW at h4 as "nearly perfectly
calibrated (+0.7%)"; the recent-window measurement is +0.8% with a CI containing
zero. The full-window +4.2% is a methodological artifact, not a discovery, and the
draft claim that HANDOFF understated the offset by 6x has been retracted.
`gating.py`'s own single-holdout comments (KMIA high +4.3%, low +1.2%) are the
same regime and should be read as the trustworthy ones.

**The scorer uses the RECENT column.** Reproduce with
`scripts/calib_eval.py 30 2025-01-01` (~30 minutes, ~240 LightGBM refits).
Per-record output is kept at **`data/calib_records-20260730.csv`** (7,236 rows) so
it never needs recomputing; the script's own `/tmp` copy does not survive a reboot.

The lesson generalises past this constant: a walk-forward average over an
expanding window measures *the average model across its own training history*,
not *the model you are about to run*. For anything used as a live correction, the
recent window is the honest estimator.

The >=80 band uses its **own** calib (+0.1290) **instead of** the overall +0.0824,
never in addition.

### 3.3 The score

For each of the four candidates at its own commit hour:

```
p_hat = p_model - HOUR_CALIB[station][metric][hour]     # recent-window value
ctr   = int(size(...).stake // price)                   # whole contracts, honest
ev    = fees.net_ev(p_hat, price, ctr)                  # per contract, net
```

Two details that are easy to get wrong:

- **No band term.** See the subsection below: an earlier draft both double-counted
  the calibs AND applied a band adjustment the data does not support. Both are gone.
- **The fee is amortised over `ctr`.** Kalshi rounds up **per order**, so a
  1-contract order at 58c pays the full $0.02 (3.4% of stake) while a 10-contract
  order at 50c pays $0.18 total, i.e. $0.018 each. `fees.fee_per_contract`
  divides the rounded order fee by the count and deliberately does not re-round;
  using `taker_fee(1, price)` unconditionally would overstate the drag on any
  multi-contract order.

`band_calib` uses the band's **own** measured calib (never the base-rate hit gap —
subtracting a base-rate difference from a probability conflates "this band is
harder" with "the model is overconfident here").

### ❌ `band_calib` IS DROPPED — the measurement killed it

An earlier draft applied a separate, larger calibration to the KLAX HIGH's >=80
readings (+12.9%, from the full-window run at n=126). The properly-conditioned
recent-window measurement refutes it:

| slice | n | stated | realised | CALIB | 95% CI |
|---|---|---|---|---|---|
| model >=80 | 81 | 50.5% | 42.0% | **+8.6%** | **[-0.2%, +17.9%]** |
| model <80 | 1,151 | 59.5% | 50.6% | **+8.9%** | [+6.8%, +11.1%] |

**+8.6% against +8.9% is the same number**, and the band's CI contains both zero
and the overall figure. The model is uniformly ~8% overconfident; it is NOT
specially overconfident when it prints high. The earlier +12.9% was noise.

So `p_hat = p_model - HOUR_CALIB[station][metric][hour]`, with no band term.

**And the honest consequence, which must not be buried.** The band adjustment was
load-bearing on the only verdict that flattered the bar:

| calib used | 07-29 KLAX HIGH p_hat | EV | verdict | actual |
|---|---|---|---|---|
| +12.9% (the dropped fudge) | 46.6% | +0.046 | NO PICK | lost |
| **+7.4% (measured, recent)** | **52.6%** | **+0.106** | **PICK** | **lost** |

Correctly specified, **the bar publishes the losing HIGH.** Its apparent success
on the eight-bet run was entirely the unsupported adjustment. Settled record
becomes 3 picks, 1W-1L-1 live — no better than betting everything (4W-2L). n=3
decides nothing, and this is recorded rather than smoothed over precisely because
BIAS DISCIPLINE exists to stop a flattering result being banked one unexamined
step at a time.

**What survives** is that the >=80 band is genuinely HARDER, though not more
overconfident: hit 42.0% vs 50.6%, and MAE 1.365-2.010 vs ~1.1. That becomes a
**display flag** on the pick face ("measured hit in this band is 42%, n=81"), not
an arithmetic adjustment to a probability.

**Verdict, in order (fail-closed):**

1. `ctr < 1` -> **UNBUYABLE**, naming the `max_stake_frac` that would fix it
2. `price > PRICE_CEILING` (0.75) -> **NO PICK (price)**
3. `ev <= MARGIN` (0.05) -> **NO PICK**
4. otherwise -> **PICK**

The ceiling is deliberate redundancy: if `p_hat` is still overconfident in an
unmeasured regime, a hard ceiling refuses expensive certainty regardless of what
the EV arithmetic says.

### 3.4 Stability

Recomputed on the **correct parity ladder** — the same parity as the candidate's
own bucket — never read from `committed_state`. Two numbers, both labelled
**UNVALIDATED**:

- `n_distinct_buckets` over the hours from first prediction to the commit hour
  (the existing `churn` definition, per-day instead of averaged)
- `stall_hours`: how long the running extremum has been unchanged

Julian's decision: score these provisionally. To make "provisional" mean
something, the pick log stores **both** `score_with_stability` and
`score_without`, so after N picks we can measure retroactively whether the
stability term improved selection. The provisional choice becomes the experiment.

**Leakage rule:** churn is computed only over hours **at or before** the commit
hour. Whole-day churn would include hours after commit and is a MODEL bug by the
freeze doc's own definition.

### 3.5 Persistence and the falsifiable promise

New table `picks`:

```
station, metric, target_date, kind ('preview'|'verdict'), issued_utc,
local_hour, verdict, bucket, p_model, p_hat, price, contracts, fee, ev,
score_with_stability, score_without, churn, stall_hours, reason,
actual, won, settled_utc
PRIMARY KEY (station, metric, target_date, kind)
```

Written via `db.upsert_ignore` — first write wins, same anti-cheat contract as
`app_bets`. **Every NO PICK is recorded too, with its reason.** Without the
declines logged we could never discover that the bar is refusing good trades —
which today's six-bet test suggests it may be (it passed on three winners).

Graded on settlement through the same path as `appbet.settle`. The panel shows
wins **and** losses.

**Pre-registered kill criterion:** if after 30 published picks the realised EV of
published picks is not positive, the feature comes out. Written into the module
docstring so it cannot be quietly forgotten.

### 3.6 Delivery

- **Preview** at the existing `PREVIEW_HOURS`, marked **PROVISIONAL**, carrying
  the latest available read. When the commit verdict later differs, the verdict
  message says so explicitly rather than silently replacing it.
- **Verdict** folded into the message `notify_pending` already sends at each
  commit hour — no sixth Discord event, no duplicate message about the same
  market, and it reaches Julian while asleep.
- **Dashboard panel** at tier 1, beside THE CALL.

No new alert stamps: preview reuses `alert_stamps`, verdict reuses
`app_bets.notified_utc`'s exactly-once contract.

---

## 4. The model-neutral queue

Ordered. Items 1-2 are prerequisites for items 3-4.

**This is too large for one implementation plan and will be phased**, each phase
reviewed against the spec before the next is dispatched:

- **Phase A (prereqs + fees):** items 1, 2, 3 — pure data/constants plus a small
  pure module. Nothing user-visible changes behaviour.
- **Phase B (Pick of the Day):** item 4 — the largest piece, and the only one
  that adds a table and an alert path.
- **Phase C (independent):** items 5-11 — each stands alone and can land in any
  order.

| # | item | class | status |
|---|---|---|---|
| 1 | Apply parity-averaged KLAX `HOUR_ACCURACY` (commit hours do NOT move) | data | **open** — needs a full 24-hour `hour_curve` rerun; the calib run only covers commit hours |
| 2 | Measure + store `HOUR_CALIB` for all four markets | data | ✅ **SHIPPED**, 8 tests; recent-window values in `gating.HOUR_CALIB` + `hour_calib()` with no cross-station fallback |
| 3 | `fees.py` + net-of-fee **display** beside gross (gate stays gross) | display | ✅ **SHIPPED**, 18 tests |
| 4 | Pick of the Day: `pick.py`, `picks` table, panel, preview + verdict, graded record | display | ⏸️ **BUILT, WRITER STILL DORMANT** — `pick.py` + `picks` table + 26 tests exist; single-pick selector (not the pair) picked by Julian's own choice. `pick.settle` IS now wired (wake cycle + `highcast settle`, 2026-08-02) so nothing published can go ungraded; `pick.record` is still called from nowhere, so `picks` stays empty and the grader is a no-op until Julian decides to publish. `notify`/`web` untouched. |
| 5 | Divergence fix: committed bucket lost >=20 pts or fell below 60% | alert | ✅ **SHIPPED**, 13 tests; fires 20:00 MNL vs 22:00 on the 07-29 fixture |
| 6 | Parse `SLP` + `5appp` into `obs`; `reparse-obs` | data | ✅ **SHIPPED**, 21 tests; 106,351 rows reparsed, decode validated at r~0.995-0.998 |
| 7 | Station Table: `board.py`, `highcast board` CLI (`--raw`, `--configs`, `--metric`), `/api/board`, dashboard panel | display | ✅ **SHIPPED**, 10 tests. NOTE: an earlier revision of this table marked item 7 shipped when only the CLI half existed — the panel came later. The claim was wrong when written. |
| 8 | Stall-duration display | display | ✅ **SHIPPED** (inside item 7) |
| 9 | Two-axis city survey (`persist`, `sd` beside `corr_hit`) | display | open |
| 10 | Retain `close_time` from the Kalshi payload | data | open |
| 11 | Codify the extremum invariant as a regression test | test | ✅ **SHIPPED**, 8 tests |

Suite: 385 before this work, **491 passing as of the latest full run** (503
collected, 12 deselected). No existing assertion changed. Also shipped, not in
the original 11-item list: `docs/nowcast-trade-reference.md`, a plain feature
reference for nowcast.trade (368 lines) requested separately.

### Explicitly OUT

- Advection veto, `guidance_gap`, convection features — model changes
- The **fee gate flip** (`min_edge` vs net) — decision-affecting
- `late_commit_guard` for KLAX — changes when a bet commits
- Fixing `committed_state`'s `argmax_pair` ladder — changes *which* prediction is
  displayed, which the freeze doc puts in the grey area. Documented, not fixed.
- `staking.size()` whole-contract handling — display-class and safe, but newly
  found rather than queued. Julian's call.
- Anything entering `model_features` / `INTRADAY_FEATURES`

### Off the queue after this pass

- `appbet` commit-hour upper bound — already fixed (`LATE_FRACTION`)
- S17 posting-time verification — verified correct, no work needed

---

## 5. Item 6: why `SLP` is worth parsing now

`obs` has `drct`, `skyc`, `sknt`, `dwpf` but **no pressure column at all**. The
raw METAR text is stored, and pressure is in it:

| station | raw rows | `SLPxxx` | `5appp` |
|---|---|---|---|
| KLAX | 57,674 | 48,772 (84.6%) | 16,178 (28.1%) |
| KMIA | 66,284 | 57,570 (86.9%) | 19,069 (28.8%) |

Hourly sea-level pressure across the whole five-year archive — better than the
3-hourly tendency group, because tendency can then be computed at any lag.
`highcast reparse-obs` already exists: no network, idempotent.

Adding a column and reparsing changes no prediction, provided nothing touches the
feature lists. This is the data-acquisition half of the advection work done
under the freeze, so that when the freeze lifts the measurement is a one-liner
rather than a fetch project.

**Decode correctly:** `SLP097` is **1009.7** hPa (3 digits, tenths, leading
9/10 implied). An earlier throwaway query in this session printed `100.97` — a
silent factor-of-ten. Analysis queries get shipped-code scrutiny.

---

## 6. Error handling

- Missing `HOUR_CALIB` for a station/metric -> verdict **UNSCORED**, never a
  silent fallback to another station's number. Same no-fallback contract
  `gating`'s three constants already enforce.
- No live price -> **NO PICK (no price)**. 37% of LOW days historically had no
  usable price.
- `market_price` outside its own bid/ask (observed: 70-71 at 0.25 with bid 0.28)
  -> prefer the ask for cost, and flag the row.
- Any exception in the pick path must not break the commit alert or `app_bets`.
  Pick of the Day is advisory; the ledger is load-bearing.

## 7. Testing

- `fees.py`: reproduces Kalshi's published example ($0.16); round-up boundary;
  maker = 25% of taker; symmetry about P=0.50.
- Verdict ordering: UNBUYABLE beats price-ceiling beats margin.
- **A pick is never written to `app_bets`** — the load-bearing test, mirroring
  `test_appbet`'s existing guarantees.
- Churn window ends at the commit hour (leakage guard), asserted to FAIL without
  the guard so it is not tautological.
- Parity: a pick on an odd-floor bucket scores on the odd ladder. Regression for
  the mistake made in this session's own analysis.
- Divergence: fires at 20:00 MNL on the 07-29 fixture, not 22:00.
- Extremum invariant: monotone and min<=max over a fixture day.
- UNSCORED when `HOUR_CALIB` lacks the station.

## 8. The EV bar CANNOT be backtested — measured, not assumed

Every price-bearing row in the database:

| table | rows | span |
|---|---|---|
| `signals` | 3,094 | 2026-07-23 .. 07-30 — **7 target dates per metric** |
| `app_bets` | 8 | 07-27 .. 07-29 |
| `alert_log` | 14 | 07-29 only |

There is no price history. Kalshi serves ~68 days of *settled* markets, but a
settlement price says nothing about the price at our commit hour, which is the
only price the bar consumes. **The price-dependent half of this design is
forward-validatable only.**

Therefore:

- `PRICE_CEILING = 0.75` and `MARGIN = 0.05` are **priors, not measurements.**
  They must be labelled as such wherever they appear, and revisited from the
  graded log rather than defended from theory.
- The backtestable inputs are all price-independent and all measured:
  `HOUR_ACCURACY` (n~2,000), `HOUR_CALIB` (n~1,600-2,000 per market), the KLAX
  HIGH >=80 band (n=126, walk-forward).
- The graded pick log and the kill criterion are not garnish — they are the only
  validation mechanism that exists for this feature. At ~1-2 picks/day, 30 picks
  is 3-4 weeks.

## 9. Open, deliberately

- **Market close time is unverified.** Item 10 retains `close_time` so an
  "act by" line can be sourced from data rather than prose.
- **Stability is unvalidated.** Scored provisionally by Julian's decision, logged
  both ways so it self-measures.
- **The bar may be too strict.** It declined three winners out of six. n=6
  decides nothing; the logged NO PICKs are how we find out.
