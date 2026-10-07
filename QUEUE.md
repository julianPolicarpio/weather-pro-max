# Queue — what is pending, and what each item is actually blocked on

Written 2026-08-02. Ordered by what would change a decision, not by effort.

The distinction that governs this whole list: **some items are builds, and some
are waits.** Confusing the two is how a busy week produces no progress. Items
marked ⏳ cannot be advanced by writing code — only by data arriving.

---

## 1. ⏳ THE EDGE SIGNAL POINTS THE WRONG WAY — DIAGNOSED 2026-08-12

> ### ✅ THE MECHANISM IS NOW KNOWN, AND IT IS NOT A TUNING PROBLEM
>
> `scripts/edge_signal_diagnosis.py` and `scripts/edge_signal_conditional.py`,
> 72 priced+settled day-markets, prices taken AT THE COMMIT HOUR.
>
> **THE MARKET IS BETTER CALIBRATED THAN OUR MODEL.** Multiclass Brier over the
> full ladder, n=72: market **0.3107**, model (RAW) **0.3881**, difference
> **+0.0774, 95% CI [+0.0030, +0.1537]** — excludes zero, but only just.
>
> ⚠️ Corrected 2026-08-15. The first run scored the PER-RUNG CALIBRATED pmf, an
> object the desk never builds. With the raw ladder the gap shrinks and the CI
> lower bound drops to +0.0030 — read this as marginal, not settled.
>
> Shrinkage `p = w·model + (1−w)·market` is better the lower w goes.
> **Brier-optimal w = 0.10, bootstrap CI [0.00, 0.30].** The data says: at the
> commit hour, the Kalshi price is a better probability than anything we produce.
>
> That single fact explains every symptom at once:
> * `corr(edge, won) = −0.555` at n=64, CI [−0.741,−0.315] — edge is
>   model-minus-market, and the model is the
>   worse of the two, so edge is mostly OUR ERROR;
> * PLACE 43% vs SKIP 91% — we trade exactly where our error is largest;
>   ⚠️ **re-measured 08-15 at n=63: 56.5% vs 82.5%, and on MARGIN (hit minus
>   ask) PLACE is AHEAD, +10.8% vs +8.5%. See the re-measurement below — the
>   hit-rate comparison ignores price and cannot support the weight put on it.**
> * why the 1-minute bound lifted hit rate 6 points and moved no money —
>   improving a model that is already worse than the price changes nothing.
>
> **Trading on model-minus-market disagreement is trading on our own error.**
> No threshold fixes that. `min_edge`, `MAX_EDGE`, and confidence gates all
> operate on a quantity whose sign is wrong.
>
> ### The anti-predictive signal, reproduced from first principles
>
> Model's top pick, binned by stated edge (n=28 picks):
>
> | edge band | n | model says | ACTUALLY hit |
> |---|---|---|---|
> | 0.03–0.10 | 8 | 71.7% | **87.5%** |
> | 0.10–0.20 | 8 | 68.4% | 75.0% |
> | 0.20–0.40 | 7 | 63.0% | 57.1% |
> | 0.40+ | 5 | 67.6% | **20.0%** |
>
> Monotone. `MAX_EDGE = 0.40` already removes the bottom row, which is the gate
> working as designed. ⚠️ **Do NOT lower it to 0.20 off this table** — that is
> n=7 vs n=8 and is exactly the threshold-fitting this item forbids.
>
> ### [Likely] WHY the market is better, and the one lever that could change it
>
> The market at commit sees 5-minute ASOS; we see hourly METARs plus a 6-hourly
> remark group (item 13, proven live on 2026-08-11). If the calibration gap is
> substantially an INFORMATION gap, the Synoptic token is not a hit-rate nicety —
> it is the only identified route to the model being worth trading on at all.
> Unproven: the 1-minute bound closed 6 points of hit rate and still did not
> overturn the market's Brier advantage.
>
> ### ⚠️ AN EARLIER VERSION OF THIS ANALYSIS SAID THE OPPOSITE — read why
>
> The first run concluded the MODEL was better calibrated (Brier 0.395 vs 0.493)
> and that its picks hit 71.7% against a market at 35.8%. **That was a lookahead
> bug**, now fixed in `highcast.pricehist`: the ladder was taken from the day's
> FIRST snapshot while the model predicted at the commit hour — a 16-hour gap for
> KMIA HIGH (01:02Z quote vs 17:00Z prediction), i.e. a quote from before the
> Miami climate day had even started.
>
> The tell was that it contradicted the live ledger. **When an analysis disagrees
> with the settled record, the analysis is wrong until proven otherwise.**
> Any conclusion in this file dated 2026-08-12 and sourced from a script that
> did not use `pricehist.ladders_at_commit` is void.



> ⚠️ **PREMISE CORRECTED 2026-08-06.** This item said "re-test at n≈60, arrives
> in 2–3 weeks". That was a CALENDAR estimate, never a power calculation, and
> the power calculation says n≈60 answers one of its two questions and cannot
> answer the other. Do not treat the milestone as a decision point. Detail at
> the bottom of this item.

### ✅ RE-MEASURED 2026-08-15 at n=63 — the gap HALVED, and the framing was wrong

`scripts/item1_recheck.py`. Eight more settled decisions.

| | then (n=36, 08-13) | **now (n=63)** |
|---|---|---|
| PLACE | 43% (n=14) | **56.5% (13/23)** |
| SKIP | 91% (n=22) | **82.5% (33/40)** |
| gap | −48 pt | **−26 pt** |
| | z=3.14, **p<0.001** | z=−2.24, **p=0.025** |

Still significant, but the gap has halved and p moved from overwhelming to
marginal. ⚠️ One legacy row (KLAX low 2026-07-27) has a NULL `reason` and is
EXCLUDED from the split rather than guessed into the favourable bucket.

**What is robust: `corr(edge, won) = −0.555`, 95% CI [−0.741, −0.315]** —
essentially unchanged from −0.571, CI comfortably excluding zero. THAT is the
finding that holds, not the PLACE/SKIP gap. `corr(ask, won)` strengthened to
+0.655 and the price ladder stayed monotone: **18% / 71% / 83% / 100%**.

### ⚠️ AND THE 43%-vs-91% LINE WAS ALWAYS DOING TOO MUCH WORK

This item's own reframe is that **the break-even bar is the ask**. Nobody had
applied it to the split:

| group | n | hit | mean ask | **margin** | net/contract |
|---|---|---|---|---|---|
| PLACE | 23 | 56.5% | 0.46 | **+10.8%** | +0.0909 |
| SKIP | 40 | 82.5% | 0.74 | **+8.5%** | +0.0691 |

**On MARGIN, PLACE is slightly AHEAD of SKIP.** Comparing hit rates on contracts
bought at 46¢ and 74¢ compares nothing — the desk buys cheap, and cheap things
hit less often, which is what cheap means. **[Likely] the hit-rate gap was
always partly an artefact of price selection rather than evidence of bad
selection.**

⚠️ **THIS DOES NOT RESCUE THE DESK.** Both margin CIs span zero
([−4.4%,+26.2%] and [−1.5%,+17.5%]), the money MDE is **0.220/contract** against
a realistic 0.02–0.05, and ~444 PLACE bets are still needed. The power
arithmetic below is unchanged. It also does not touch the §0 finding — the
market still beats us on Brier at every hour, which is separate and far better
powered.

**Everything below this line predates the re-measurement.** The 3/8 and 3/10
tables are earlier checkpoints, kept for the trajectory.

---

**This is the only item that decides whether any of the rest is worth doing.**

Measured on the 20 graded bets (post-settlement-correction, 2026-08-02):

| | won | net/contract | per bet |
|---|---|---|---|
| rule said **PLACE** | 3/8 | +$0.015 | **+$0.0019** |
| rule said **SKIP** | 11/12 | +$1.990 | **+$0.166** |

Correlation with winning: market price **+0.496**, our edge **−0.619**. We lose
precisely where we think we have found value. Strip the single 11¢ winner from
the PLACE set and the remaining seven are **−$0.124 per bet**.

**Blocked on:** price history. `signals.market_json` begins 2026-07-24 — **9
priced days for KMIA HIGH**. Kalshi does not sell historical prices, so no
backtest can be run over any earlier period. This is not a modelling gap that a
better feature closes; the data does not exist.

**The wait:** the wake cycle already records every decision's edge (shipped
2026-08-02 with `gating.MAX_EDGE`). At ~4 decisions/day, n≈60 arrives in 2–3
weeks. Re-run the PLACE/SKIP split and the correlations then.

### ✅ CHECKPOINT at n=28 (2026-08-05) — the sign held

Not the n≈60 re-test, but half way there and the finding did not soften:

| | won | mean ask | mean p_model |
|---|---|---|---|
| rule said **PLACE** | **3/10 (30%)** | 0.33 | 0.67 |
| rule said **SKIP** | **16/18 (89%)** | 0.77 | 0.70 |

Correlations: market price **+0.596**, p_model +0.163, our edge **−0.506**
(was −0.619 at n=16). Hit rate by what the market charged is monotone —
0–35¢ **20%**, 35–60¢ 57%, 60–85¢ 80%, 85¢+ **100%** — which is the same
statement from the other side: a cheap bucket is the market telling us it is
unlikely, and it is right.

Two further cuts worth keeping:

* **The model's 70–90% band claims 79% and delivers 55%.** That is the band the
  staking rule bets out of. The 90–100% band is fine (4/4, claims 96%).
* **6 of the 9 losses missed by exactly 1°F** — one bucket. This is not a wild
  model; it is an adjacent-bucket model being asked to be exact.

**Still do not** sweep MIN_EDGE on this sample. n=28 makes the sign credible;
it does not make any threshold fitted to it credible.

### ⚠️ THE POWER CALCULATION (2026-08-06) — read before waiting for n≈60

This item bundles two questions with wildly different sample requirements.

**A — is PLACE worse than SKIP on HIT RATE? ALREADY SETTLED.**

    PLACE 43% (n=14)   SKIP 91% (n=22)   z = 3.14   p < 0.001

n≈60 adds nothing. The answer is in hand today.

**B — does PLACE make or lose MONEY? UNREACHABLE ON THIS TIMELINE.**
Per-contract net has sd **0.404** (a bet returns +$1 or $0, so the noise is
enormous relative to any real edge). Smallest detectable effect at 95%/80%:

| settled bets | ~PLACE bets | detectable only if \|edge\| > |
|---|---|---|
| 36 (today) | 14 | 0.302 / contract |
| 60 | 23 | **0.236 / contract** |
| 200 | 77 | 0.129 / contract |

A realistic edge is 0.02–0.05/contract. Seeing 0.05 needs **512 PLACE bets ≈
1.0 years** at today's 1.4/day; seeing 0.02 needs **6.3 years**.

**So:** the n≈60 date will re-confirm what is already p<0.001 and leave the
money question exactly as open as it is now. Anything phrased "does this make
money" is not answerable by waiting. See item 7 — the only lever that changes
this arithmetic is BREADTH, not patience.

⚠️ **One commit hour DID move on 2026-08-05, and this item's "do not tune the
commit policy" line was knowingly overridden.** KMIA HIGH went h14 → h13. The
justification is not a threshold fitted to the 28 bets — it is the observation
that h14's mean ask (0.82) exceeds its own measured hit rate (79.3%), so the
market is structurally unprofitable there regardless of what the bets said,
while h13 asks 0.46 against 68.0%. See `gating.MIN_DECISION_HOUR`'s comment for
the full arithmetic and the accuracy cost. If the n≈60 re-test says there is no
edge anywhere, this move does not save it.

**Do not** tune the commit policy, sweep MIN_EDGE, or add model features hoping
to fix this in the meantime. At n=20 any improvement is indistinguishable from
noise, and a threshold swept on this sample fits it exactly.

**The outcome that must stay on the table:** it may come back "there is no edge
here". The days the model predicts best are the days the market also gets right,
and those settle at 94–97¢ where you risk 96 cents to make 3. That is not a
business, and finding it out cheaply is a success, not a failure.

---

## 2. ⏳ BANKROLL — $20 is the number, not $5.50

Live Kalshi balance is **$5.50** (read 2026-08-02, unarmed). At that size:

- `staking.max_stake_frac` 5% ⇒ **27¢ per market** ⇒ 0 contracts at most prices
- a $1 stake ⇒ **18% of bankroll**; 4 markets ⇒ **73% deployed daily**
- **5.5 consecutive losses ⇒ zero.** At the PLACE rule's realised 3/8 win rate,
  five straight losses lands about **1 time in 10**

There is no stake at $5.50 that is both safe and worth placing. Julian's stated
target — 80¢ to $1 per bet — fits the SHIPPED caps exactly at a **$20** bankroll:
5% of $20 = $1.00 per market, 4 × $1 = 20% deployed.

**Do not weaken a cap to make a small bankroll work.** The caps are priced
against losses this 20-bet sample does not contain. The correct output when the
computed stake buys under one contract is a visible `0 contracts (stake cap)`
row, not a rounded-up trade.

Blocked on item 1: no money goes in while the edge signal points backwards.

---

## 3. ⏳ CONVECTION — Task 0 PASSED, Task 1 DID NOT. Feed alive; do not ship features.

> **RESOLVED 2026-08-10.** The feed is wired into the wake cycle (both stations)
> and both gates have been run. Result: the signal is REAL but the desk already
> knows it by the time it commits. Detail below; the original spec follows for
> context.

### Task 0 — does the FORECAST flag the days the OBSERVATION flags? **PASS**

`scripts/convection_task0.py`, 728 days, base rate 14.1% of days fall ≥2 °F
after the commit hour.

| arm | flagged | fell \| flagged | fell \| not | separation | |
|---|---|---|---|---|---|
| pop ≥ 30 | 96 | 31.2% | 11.4% | **+19.9%** [+10.7,+29.5] | SIG |
| pop ≥ 50 *(pre-declared)* | 19 | 26.3% | 13.7% | +12.6% [−6.7,+34.2] | ns |
| cape ≥ 2000 | 269 | 23.0% | 8.7% | **+14.3%** [+8.7,+20.1] | SIG |
| cape ≥ 2500 *(pre-declared)* | 179 | 21.2% | 11.7% | **+9.6%** [+3.2,+16.1] | SIG |

Shuffled control: mean +0.0%, 95% of shuffles within [−14.4%, +12.6%]. The real
arms clear it.

⚠️ **Read the PRE-DECLARED thresholds, not the best ones.** `cape ≥ 2500` passes
at +9.6% — about **54%** of the +17.9 points observed thunder achieves, which
clears the spec's "meaningful fraction" bar. The +19.9% headline is `pop ≥ 30`,
a threshold chosen after seeing the table. The script prints every threshold so
this is checkable.

⚠️ `pop` gets WEAKER as the threshold rises (30→+19.9, 40→+14.1, 50→+12.6). If
higher precipitation probability meant more convection it should strengthen.
The usable signal looks like "any convective potential today", not "storms are
likely" — vaguer than the spec assumed. `cape` has the better shape (n=269,
tight CI).

### ✅ Task 1b — the LOW's commit hour, the gap Task 1 left. **ALSO NO** (2026-08-28)

`scripts/convection_kmia_low_gate.py`, pre-registered. Task 1 never scored KMIA
LOW h7 (its ≥60% still-ahead filter excluded it by construction), and the loss
decomposition says that is where ALL four ≥4 °F misses live. So it was run.

| h | n | A hit | B hit (+conv) | **C hit (+random)** | dHIT | CI 95% |
|---|---:|---:|---:|---:|---:|---|
| 5 | 236 | 50.6% | 52.3% | 51.9% | +1.7 | [−1.9, +5.3] |
| 6 | 236 | 53.4% | 54.9% | 54.7% | +1.5 | [−1.9, +4.9] |
| **7** | 236 | 59.3% | **61.2%** | **61.2%** | +1.9 | [−1.9, +5.7] |

**The noise arm scored IDENTICALLY to the convection arm at h7 (61.2% vs 61.2%).**
Four random columns bought exactly what four convection columns bought. The CI
also spans zero. Fails gates 2 and 3 → **DO NOT SHIP**.

⚠️ **This is why the noise control was added** — the shipped `convection_eval.py`
has none, and without it "+1.9 pts at the commit hour" reads as a modest win
rather than as feature-sampling jitter.

**Consequence:** the KMIA LOW storm tail is **blindspot B — unforecastable at the
commit hour**, by both channels. Forecast convection (this) and observed METAR
rain (Phase 0 bundle B3, on the HIGH) both failed. The answer is to **SIZE for
it, not to predict it**.

### Task 1 — do `conv_*` features improve the model? **NO, at the hours that matter**

`scripts/convection_eval.py`, single holdout split 2026-01-01.

Headline looks good — KMIA HIGH dMAE −0.042 / dHIT **+2.2 pts**, KMIA LOW dMAE
−0.026 / dHIT **+1.2 pts**, every predictive hour improving. **But those are
averages over hours the desk does not trade.**

Tested directly at the actual commit hours, with a bootstrap CI:

| market | h | n | A hit | B hit | dHIT | |
|---|---|---|---|---|---|---|
| KMIA high | 13 | 220 | 59.8% | 59.8% | **+0.0%** [−2.5,+2.7] | ns |
| KMIA low | 7 | 218 | 60.8% | 63.3% | +2.5% [−1.4,+6.4] | ns |

The HIGH's gains sit at h8–h10 (+3.9, +4.8, +4.1), five hours before commit. The
LOW's commit hour **is not scored at all** by the eval — its predictive-hour
filter needs ≥60% of the extremum still ahead and h7 has 42%, so h7 is excluded
by construction.

**Interpretation:** by the commit hour the ratchet and current temperature have
already absorbed what convection would have told us. The features help while the
day is still undetermined, which is not when we act.

### ✅ THE KMIA **LOW** h7 GAP IS NOW CLOSED — also NO (2026-08-28)

The note above said h7 "is not scored at all ... excluded by construction". That
mattered, because the 2026-08-28 loss decomposition found **every** ≥4 °F miss in
the whole ledger is KMIA LOW, all storm-driven (lows stamped 1:46 PM, 2:48 PM,
4:34 PM, 6:05 PM). So the one market carrying the catastrophic tail was the one
market convection had never been tested on.

`scripts/convection_kmia_low_gate.py` — pre-registered, PRIMARY = h7, single
holdout at 2026-01-01, n=236, **one** change from the shipped eval (name the
hours instead of the ≥60% filter; `PEAK=(11,17)` deliberately left alone):

| h | A hit | B (+conv) | **C (+4 random)** | dHIT | CI 95% |
|---|---|---|---|---|---|
| 5 | 50.6% | 52.3% | 51.9% | +1.7 | [−1.9, +5.3] |
| 6 | 53.4% | 54.9% | 54.7% | +1.5 | [−1.9, +4.9] |
| **7** | **59.3%** | **61.2%** | **61.2%** | **+1.9** | **[−1.9, +5.7]** |

⚠️ **The noise arm scored IDENTICALLY to the convection arm at h7 (both 61.2%).**
Four random columns bought the same "+1.9 points" as four convection columns. The
gain is LightGBM's per-split column sampling being jiggled, not information —
which is precisely why the control arm was added; the original eval had none and
would have reported +1.9 at the commit hour as promising.

**DO NOT SHIP.** Convection is dead at KMIA LOW h7 as well as KMIA HIGH h13.

**What this settles:** the KMIA-low storm tail is **blindspot B** — real, 10% of
Miami days, mechanically understood, and *not forecastable at the commit hour*
from anything we have. `BAND_CALIB` at (KMIA, low, 7) already absorbs the risk on
AVERAGE; what cannot be done is attributing it to a particular day. The remaining
lever is SIZING, not prediction — and sizing arms have died here before, so that
needs its own gate, not a config edit.

**DO NOT add `conv_*` to the shipped feature set.** Zero gain at the high's
decision hour, unproven at the low's, and this project has walked back a
metric-specific "win" before.

**DO NOT ship `conv_*` features.** **KEEP the feed running** — one API call an hour, and Task 0's signal is real
even if unusable at commit time.

**WALK-FORWARD, 2026-08-10 — the low's +2.5% was noise.** Re-run with the
harness the project uses for shipping decisions (expanding window, refit every
30, arms on identical rows, paired bootstrap):

| market | h | n | A hit | B hit | dHIT | |
|---|---|---|---|---|---|---|
| KMIA low | 7 | **526** | 62.9% | 62.5% | **−0.5%** [−2.9,+2.2] | ns |
| KMIA high | 13 | **529** | 62.9% | 62.7% | −0.3% [−2.0,+1.5] | ns |

low: 72 days improved, 73 worsened, 381 unchanged. high: 42 / 45 / 442. A coin
flip, and on most days the features change nothing at all.

⚠️ I predicted the walk-forward would not buy sample size. Wrong — `MIN_TRAIN`
counts frame rows across the bandwidth window, not days, so the test set runs
526 days from 2025-02-21, 2.4x the single split. The larger, better-specified
estimate lands at zero.

**QUESTION CLOSED.** Task 0 real, Task 1 dead at both commit hours, on the
largest sample available. Nothing further to run; do not re-run the walk-forward.


---

## 3b. ✅ CONVECTION FEED REVIVAL — DONE (verified 2026-08-28)

`forecast_conv` is **live**: 54,504 rows, latest valid 2026-08-28T23:00, fetched
2026-08-28T09:52Z, and `backfill._conv_fetch_due` is wired into the wake cycle —
the generalised cure the item asked for. Task 0 passed, Task 1 and Task 1b both
say the features do not help at the commit hour. The feed stays for monitoring
(`storm.py` uses observations, not this) and no model reads it.

### (original spec, kept for context) CONVECTION FEED REVIVAL

Spec: `docs/superpowers/specs/2026-08-02-convection-feed-revival-design.md`

`forecast_conv` (hourly pop/qpf/cape, 50,904 rows, KMIA) has been **dead since
2026-07-29** — the fetch script exists and was never wired into the wake cycle.
Same defect `nbm_nbs` had; that cure (`backfill._nbs_fetch_due`) was never
generalised, so the next hand-added feed died the same way.

Why it matters, measured 2026-08-02 on 2,383 KMIA days: on days with observed
thunder the low fell **≥2 °F after the commit hour 24.2%** of the time, versus
**6.3%** on quiet days — roughly 4× more likely to lose a whole bucket after we
have committed. The temperature model has **no precipitation input at all**, so
this is information it structurally lacks and the ladder may underprice.

**Gate is Task 0**, per HANDOFF §4.1: does the FORECAST identify the same days
the OBSERVATION does? Convection is the hardest thing in short-range
forecasting and it may not. A null result there is a valid, expected outcome.

---

## 4. ✅ STALENESS GUARD FOR ALL FEEDS — SHIPPED 2026-08-13

A feed dying silently has now happened **twice** (`nbm_nbs`, `forecast_conv`).
Both times nobody noticed for days. Fixing feed #2 without this means feed #3
dies the same way.

Report per feed how long since its last successful fetch and surface anything
past its cadence. Cheap: a line in the wake-cycle output, or a Discord warning
reusing the `alert_stamps` exactly-once pattern from `notify.notify_unstable`.

### ✅ BUILT. `src/highcast/staleness.py` + `notify.notify_stale_feeds`.

9 feeds checked per station, both surfaces: a line in the wake log EVERY cycle
(including when healthy — a guard that only speaks on failure gives no evidence
it is still watching) and a Discord alert. 15 tests; suite 1,062 → 1,077.

**BACKTESTED against the outage that motivated it.** Replaying `forecast_conv`'s
9-day death: the guard would have fired on **2026-08-01, three days in**, versus
nine days of nobody noticing.

⚠️ **Thresholds are the SOURCE's publication cadence, not the fetch cadence.**
`forecast_conv` is fetched every 55 min but republished ~6h, so healthy gaps of
39h (p90) are normal; a threshold set from the fetch interval would cry wolf
daily. Every limit is set generously against the measured p90, because a noisy
guard gets ignored and an ignored guard is worse than none.

⚠️ **CLI IS CHECKED SEMANTICALLY, NOT BY TIMESTAMP — and this was found the hard
way.** `backfill._yesterday_cli` returns 0 without writing once yesterday's row
is final, so `fetched_at` FREEZES during healthy operation: only 7 distinct
values since 2026-06-01, including a **379h gap that was not an outage**. A
timestamp check there would have fired for 14 consecutive days. It now asks how
far behind today the newest settlement row is, and one missing day is not an
outage until two are.

⚠️ **Four feeds are EXEMPT and must stay so**, each with a recorded reason:
`alerts` (event-driven — no alert means no rows), `obs_1min` (archive, no live
path), `twc_obs` (never populated), `syn5` (unconfigured, item 13). An exemption
is a claim that ABSENCE IS NORMAL; a test enforces that each carries a reason and
that nothing is both checked and exempt.

⚠️ **It does NOT cover a power cut.** This is "the app is running and a feed
stopped". **Nothing covers the machine being off at all** — the dead man's switch
was REMOVED 2026-08-28 at Julian's instruction, having never pinged once in its
life (`heartbeat_url` was empty in both yamls throughout). An unarmed guard reads
as coverage while providing none, which is the same defect as the two bugs below.
The gap is deliberate and explicit; re-add liveness monitoring ARMED or not at
all.

**Two of my own bugs were caught by its own tests**, both worth knowing:
the first CLI check measured minutes since local midnight (max 1,440) against a
2,880-minute threshold, so it **could never fire**; and the first alert stamp was
once-forever rather than once-per-day, which would have meant a feed that dies,
is fixed, and dies again never alerts twice.

---

## 5. ✅ REWIRE THE STORM WARNING ONTO OUR OWN DATA — SHIPPED 2026-08-28

`src/highcast/storm.py` + `notify_unstable`'s storm half. **The old guard had
never fired once**: `alert_stamps WHERE kind='unstable' AND key LIKE '%storm%'`
returned **0 rows** across a Miami August that thunderstorms almost daily, while
the HEAT half fired routinely. Errors #8 and #11 for the third time — a guard
that cannot fire looks exactly like coverage.

**Re-measured before rewiring**, 2,430 KMIA climate days, independently of the
figure this item quoted:

| | low breaks ≥2 °F | |
|---|---|---|
| after thunder ONSET | **28.6%** (199/695) | ← the decision-relevant test |
| quiet-day control | 5.3% (92/1733) | **5.4×** |
| after h7 commit, thunder day | 19.8% (138/697) | reproduces this item's 3.6× |

**Verified end-to-end on the real 2026-08-24 downdraft** (DB copy, `send`
stubbed): fires `⚠️ STORM · KMIA · 2026-08-24 · Observed convection (+TSRA BR) —
temperature already down 11.9°F`, and stays silent at 09:00Z the same morning.

⚠️ **A defect the live data caught:** the first version measured the temperature
swing inside the 60-minute detection window and reported that 13 °F collapse as
**1.98 °F**, because the pre-storm peak sat 17 minutes outside it. `WINDOW_MIN`
(detect) and `SWING_LOOKBACK_MIN` (measure the fall) are now separate constants
with a regression test — an alert about a cliff must not start measuring at the
bottom of it.

⚠️ **MONITOR, NOT PREDICTOR.** Convection has failed as a model feature three
times (Task 1, Task 1b, Phase 0 bundle B3). `tests/test_storm_from_own_obs.py`
(9) greps `staking.py`, `gating.py`, `features.py` and `models/intraday.py` and
fails if anything imports `highcast.storm`.

---

## 5b. (original spec, kept for context) REWIRE THE STORM WARNING

`notify.notify_unstable` (shipped 2026-08-02) currently fires on NWS alerts —
the **worst** of the four storm signals available:

| source | lead time | history |
|---|---|---|
| `forecast_conv` pop/cape | hours | 50,904 rows, dead — see item 3 |
| `obs.sky_cb` (cumulonimbus) | minutes | 7,050 obs, live |
| `obs.wx_thunder` | zero — it is happening | **678 KMIA days**, live |
| NWS storm alert | ~zero | **0 rows, cannot backfill** |

The archive path can never backfill storms: it requests only `HEAT_PHENOMENA`
and its VTEC map has no thunderstorm or tornado codes. Rewiring onto
`wx_thunder`/`sky_cb` costs little and works today with 678 days behind it.

---

## 6. 🔨 KLAX LOW ≥0.80 CALIBRATION BAND — blocking question ANSWERED 2026-08-28

> ### ✅ "Establish what — if anything — consumes the KLAX low calibrated probability"
>
> **It reaches a live decision on every KLAX low commit.** The chain:
> `signals.live_decision` → `staking.evaluate(calib=band_calib(KLAX, low, h6, p))`.
> When `band_calib` returns None the desk REFUSES, and that refusal is visible on
> the board — 2026-08-28 KLAX low graded **C** with
> `calibrated: WARN — 67–68 model 97% sits outside every measured band`, and the
> card read *"unmeasured confidence … refusing to bet a number that has never
> been checked"*.
>
> **So this item's own alternative hypothesis is false**: it is not "a
> calibration table nobody reads". `CONFIDENCE_FLOOR`'s missing KLAX entry is
> separately irrelevant — `confidence_floor_ok` is only consumed by the order
> bot, and `orderbot.TRADED_STATION = "KMIA"`, `TRADED_METRIC = "high"`.
>
> ⚠️ **STILL NOT SHIPPED, deliberately.** Extending the band upward would let
> bets through that are currently refused, and §0 measured that PLACE does worse
> than SKIP. The item's own instruction stands — *"use a recent window, and
> positive-control the test before trusting any null"* — and the n=671 / +4.3%
> figure predates the 07-31 decision-hour move, so it is orphaned like every
> other hour-keyed number was. **Re-measure before touching a live gate.**

## 6b. (context) KLAX LOW ≥0.80 CALIBRATION BAND — premise partly superseded

`BAND_CALIB[("KLAX","low",6)]` stops at 0.80 ("n=13, too thin"). A nested
walk-forward now puts **n=671** in that band at **+4.3% overconfident**.

⚠️ Before building: `gating.CONFIDENCE_FLOOR` has **no entry** for KLAX low, so
`confidence_floor_ok` refuses unconditionally there. Establish what — if
anything — consumes the KLAX low calibrated probability today. If the answer is
"nothing on the automated path", the finding is that the constant reaches no
decision, which is more useful than a calibration table nobody reads.

Also: expanding-window bands were measured to **overcorrect** on this market
(+2.1% over → −3.0% under). Use a recent window, and positive-control the test
before trusting any null.

### ✅ SUPERSEDED IN PART (2026-08-10) — the real gap was open-ended buckets

This item's premise was "extend the 2 °F band table upward". Measurement says
that would not have fixed the case that triggered it. `BAND_CALIB` describes the
MODAL 2 °F bucket; the refused pick was `≥69`, a CUMULATIVE bucket, and the two
are different objects. In 1,634 records KLAX low's modal 2 °F bucket never once
exceeded 90%, so no upward extension can reach a 96%.

`scripts/open_bucket_calibration.py` measures the right object — output in
`docs/results-open-bucket-calibration-2026-08-10.txt`, summary in
HANDOFF §2.16. Headline: at 0.90–0.97, KLAX LOW `≥T` is stated 0.944 / realised
0.943 on n=581. **Calibrated to a tenth of a point. The refusal was wrong.**

Two follow-ups, neither shipped:

- **11 🔨 Stop routing open-ended labels through `BAND_CALIB`.** In
  `staking.evaluate`, a label with `lo is None` or `hi is None` consults the
  modal-bucket table, gets None, and refuses. It should consult the open-bucket
  numbers instead. One-line predicate, but it CHANGES WHAT THE DESK PLACES —
  needs the money test, not just the hit-rate table, before it ships.
- **12 🔨 KMIA LOW `≥T` overstates by 3.3 points at 0.90–0.97** (0.941 stated,
  realised 0.909, CI [0.886,0.927], n=744). The only significantly miscalibrated
  open bucket of the eight. A 3¢ break-even shift on the market this project
  trades most.

⚠️ Both inherit §2.16's limits: thresholds anchored on the MODEL's median rather
than the market's, and hit rate measured, money not.

The original item below still stands as written for the modal-bucket table.

---

## 7. 🔨 BREADTH, NOT PATIENCE — panel restored 2026-08-28, cities still to add

> ### ✅ PART A DONE — `MARKETS BY CITY` is back on the desk
>
> Restored under *Markets by city — planning only*, sorted by climatological
> modal-bucket hit, current station marked. The 2026-08-05 removal reasoning
> ("nothing reads it, no bet was sized from it") was true and beside the point:
> this is a **planning** input, and choosing which city to add next is precisely
> what it is for. Its caveat is now the first thing on the panel —
> **predictable is not profitable** — together with the five per-market
> constants a new city needs.
>
> Current ranking (18 rows, climatological hit — NOT this desk's realised hit):
> MIA high 46% (MAE 1.30) · LAX low 40% (1.56) · ATL low 35% (2.07) ·
> SEA low 33% (1.97) · MIA low 34% (1.87) · LAX high 32% (2.09).
>
> ⚠️ **PART B — actually adding a city — is NOT done and is not small.** Each
> new market needs its own `HOUR_ACCURACY`, `MIN_DECISION_HOUR`, `HOUR_CALIB`,
> `BAND_CALIB` and `CONFIDENCE_FLOOR`, all five per-market and all fail-closed,
> i.e. the same walk-forward every existing market got. Shipping a city with any
> of them missing means the desk refuses every bet there — visibly, but
> pointlessly. **Do not start this without budgeting that run.**

## 7b. (original spec, kept for context) BREADTH, NOT PATIENCE

n scales with **markets traded per day**, not with waiting. At 1.4 PLACE bets
a day across two cities, the P&L question needs a year (item 1). Kalshi lists
far more than KLAX and KMIA; four cities instead of two roughly halves every
timeline in that table, and it does so starting immediately rather than after
a year of accumulation.

**Do this before any further "wait and see" item.** Every ⏳ in this file is
gated on sample size, and this is the only entry that attacks sample size
directly.

⚠️ **`MARKETS BY CITY` was deleted from the dashboard on 2026-08-05** as
"informs no decision" — that was wrong, and it was my call, not Julian's. It
ranked cities by predictability, which is exactly the input this item needs.
`/api/cities` still serves and `cities.survey()` is untouched; restoring the
panel is pasting the section back (see the removal comment in `index.html`).

**Prerequisite, and it is not small:** a new city needs its own
`HOUR_ACCURACY`, `MIN_DECISION_HOUR`, `HOUR_CALIB`, `BAND_CALIB` and
`CONFIDENCE_FLOOR` — all five are per-market and all fail closed. Budget the
same walk-forward run each existing market got, and expect some cities to come
back "no threshold qualifies", which is a valid answer.

---

## 8. ✅ SURFACE ENSEMBLE SPREAD AND BIMODALITY — SHIPPED 2026-08-28

`web/app._ensemble` + a line under the distribution chart. Live right now:
**KLAX high — 119 members span 85.0–100.1 °F (15.1 °F = 8 buckets)**, against
KMIA high at 4.6 °F = 2 buckets. Exactly the contrast this item described.

⚠️ **A LOOKAHEAD TRAP IN THE SOURCE, found while building it.** `forecast_ens`
holds rows whose `run_date` is AFTER `target_date` — **234 of 554 rows**, leads
from −5 to +2 days. "The latest row for this target" would have put a run made
after the day ended on the desk as a forecast: error #1's shape. Two filters are
now required and tested — `run_date <= target_date` AND `fetched_at <= now`.

⚠️ **MEASURED NOT TO PREDICT OUTCOMES**, 96 settled bets with a leak-free row:

| | corr with LOSING | 95% CI |
|---|---|---|
| `ens_std` | +0.034 | [−0.168, +0.233] |
| `ens_bimodality` | −0.038 | [−0.237, +0.164] |
| `p90 − p10` | +0.032 | [−0.170, +0.231] |

All straddle zero (high-spread 29% vs low-spread 19% on n=48 a side is noise),
consistent with the guidance-spread veto already being dead. **So it ships as
display only** — `tests/test_ensemble_display.py` (7) greps `staking.py`,
`gating.py`, `features.py` and `models/intraday.py` for `forecast_ens` and fails
if any of them reads it.

**What it IS good for is arithmetic, not prediction:** 119 members spanning
15 °F on a 2 °F ladder is eight rungs — no single-bucket bet can be confident
whatever the model's probability says.

## 8b. (original spec, kept for context) SURFACE ENSEMBLE SPREAD

Already fetched every wake into `forecast_ens` (119 members: `ens_std`,
`ens_p10/50/90`, `ens_bimodality`). **Nothing displays it and no model reads
it.** Measured 2026-08-06 for the same target day:

| market | spread | p10–p90 | bimodality |
|---|---|---|---|
| KMIA high | 1.8–2.3 °F | 84.0–88.4 | 0.45–0.60 |
| KLAX high | 5.3–6.1 °F | **78.0–92.0** | **1.08–1.19** |

KLAX HIGH's members disagree across 13 °F — seven 2 °F buckets — with
bimodality near 1.2, i.e. two competing scenarios rather than one fuzzy
answer. On that setup a single-bucket bet is close to a coin toss however good
the model is.

**Ship it as DISPLAY, not as a feature or a veto.** The guidance-spread veto
was measured on 2026-08-06 and is dead (item under "Not doing"); this is about
letting a human see that today is a two-scenario day before committing.

---

## 14. ⏳ SALVAGE ALERT — PRE-REGISTERED, DECIDABLE ~2026-09-10

`scripts/salvage_alert_gate.py`. **Do not read a verdict before n=100
out-of-sample day-markets** (`target_date >= 2026-08-17`). Progress prints on
every run; at ~28 settled day-markets/week that is roughly 3.5 weeks.

**The frozen rule:** fire when `max(market_price)` on a bucket that is NOT ours
is ≥ **0.60** while our bucket still trades ≥ **0.20**. Nothing else.

**Where it came from:** best of ~40 swept cells in
`scripts/divergence_alert_eval.py` — 15 fires, 11 losers, 4 winners, 73% vs 15%,
Fisher p=3.8e-5, 0.0015 after ×40 Bonferroni. **Surviving Bonferroni is not the
same as being pre-registered**: the sweep chose the rule's SHAPE too, and no
correction prices that in. Treated as a hypothesis only.

**Ships only if all four hold** — primary (diff>0, p<0.05, OOS days alone);
survives `stated` (sign holds in BOTH halves of the committed-probability
split); moves MONEY (`Σ yes_bid(ours) − won > 0`, exit priced at the bid);
agrees with the ledger.

⚠️ **Expected to FAIL on money, and that is the gate working.** It fires when
the market has already moved against us, i.e. when our bucket is cheap — the
winners abandoned pay 1.00 while the losers salvaged pay the bid. Also note
`staking.bankroll = 0`, so the money arm is counterfactual: a PASS does not
license real money by itself.

**On fail: delete the candidate. Do NOT re-sweep `(CONC, SALV)`** — that is how
a dead idea becomes an immortal one. Register a new gate or drop it.

`tests/test_salvage_gate.py` (16) drives every branch including SHIP, because a
gate that sits at "not yet decidable" for a month is otherwise a guard nobody
has ever seen fire — errors #8 and #11 twice over.

---

## 15. ⏳ CAN A COMMIT HOUR MOVE EARLIER ON PRICE? — decidable ~Feb 2027

> ### ✅ THE ACCURACY HALF IS ALREADY ANSWERED — NO (Phase 0, 2026-08-27)
>
> `scripts/earlier_high_eval.py` + `data/experiments/earlier_high/report.md`,
> run status **VERIFIED**. Forty station × hour × bundle cells, n≈595–601,
> coverage ~100%, h9–h13, both stations, four METAR bundles (B1 humidity,
> B2 wind, **B3 rain**, B4 station-specific).
>
> **"No station/hour/bundle combination cleared every pre-registered gate."**
> Every cell REJECT. And at KMIA h13 the replay arms are *significantly worse*
> than the shipped model — dHIT vs shipped −0.057 [−0.097, −0.015] (B1),
> −0.057 [−0.098, −0.014] (B3).
>
> **No feature set recovers the accuracy lost by committing earlier.** The only
> surviving reason to look at earlier hours is the cheaper ask, which Phase 0
> deliberately does not score — it has zero price terms. That, and only that, is
> what the price gate below is for.
>
> ⚠️ Phase 0's method is stronger than the price gate's: three-seed random
> control family with `noise_delta_hit_p975`, Holm-adjusted p, n≥400 and
> coverage≥0.80 floors, CRPS and calibration-gap guards, and full-history vs
> warm-season slice agreement. **If the two ever disagree, believe Phase 0.**


`scripts/decision_hour_price_gate.py`. Julian asked 2026-08-23 whether KLAX
could commit earlier (h13 HIGH = **04:00 Manila**).

**The old gate cannot answer it.** `decision_hour_move_eval.py` moved every hour
LATER on 2026-07-31 and its own comment names the blind spot: *no price term*.
That is why KMIA HIGH's h14 → h13 had to be argued outside it.

**Statistic:** `margin = 1{model's top rung contains the settled value} − that
rung's ask`, both from the SAME live ladder, paired across hours.
⚠️ Never mix a walk-forward `expected_bucket_hit` with a live ask — different
objects, the error #5 mistake in another costume.

**Four arms.** Arm 0 (representativeness vs `HOUR_ACCURACY`) can **VOID** the
result; arm 1 is paired margin with Bonferroni; arm 2 is ≥40% still-a-forecast;
arm 3 is a SHIPPING PRECONDITION — `BAND_CALIB`/`HOUR_CALIB` are hour-keyed with
no fallback and exist **only at the incumbent hour**, so moving to an unmeasured
hour makes `staking.evaluate` refuse every bet.

**Pilot result (exploratory, already seen — no verdict):**
* KLAX HIGH **arm 0 FAILS**: h12 observed 81% vs long-run 60.3%, binomial
  p=0.029. The window that produced the exciting "+14% margin at h12" cannot
  reproduce a curve measured on 1,959 days. **VOID.**
* Every candidate at every market has a margin CI spanning zero — best is KLAX
  HIGH h10 at +0.175, CI [−0.055, +0.410].
* KLAX LOW and KMIA LOW pass arm 0 cleanly and show nothing on arm 1.

⚠️ **N_MIN = 170 paired days, and that is measured not guessed.** Margin's
per-day SD is 0.46–0.53, so +0.10 needs ~168–221 paired days; +0.05 needs ~670.
At one day-market per day that is **~24 weeks**. A price question asked with 24
days of price history cannot be answered, and the +14% point estimate is fully
consistent with zero.

⚠️ **Do NOT pool the four markets to reach n faster** without first showing the
effect is common across them — a pooled +0.10 could be one market's +0.40 and
three zeros.

**Meanwhile, the 04:00 Manila problem has a cheaper fix than moving the hour:**
`orderbot_enabled: false` in both configs, so Julian must place manually. Arming
the bot solves the sleep problem at **zero** accuracy cost. Moving to h12 buys
one hour of sleep for ~9 points of hit rate; the hours that would actually help
(h9/h10 = midnight/01:00) are where the model is 45–49%.

---

## 9. ⏳ VALIDATE h13 — the price cliff must keep showing up

`MIN_DECISION_HOUR["KMIA"]["high"]` moved 14 → 13 on 2026-08-05 on a PRICE
argument, deliberately failing the accuracy gate (Δhit −11.2%, CI
[−15.1%,−7.8%]). It rests on one empirical claim: the ask **cliffs** at 14:00
(0.38 at 13:30 → 0.82 at 14:00) because the median peak is 13:58.

**What to watch, weekly:** does the 13:00 ask keep arriving near 0.46 rather
than drifting toward 0.82? If the cliff stops appearing in `signals`, the
justification is gone and the hour should move back — regardless of the win
record, which is n=1 (2026-08-05, 90–91 @0.73, WON).

**Also re-measure:** `CONFIDENCE_FLOOR[("KMIA","high",13)] = 0.80` passed its
0.80 lower-bound target by **0.001** (n=78, lb 0.801). If it drops, DELETE the
line — an absent market never fires — never lower the 0.80 target to keep it.

---

## 10. ⏳ CONFIDENCE GATE — built, OFF, and it should probably stay off

`StakingConfig.confidence_gate` (default `False`) makes `staking.evaluate`
refuse below a market's measured `CONFIDENCE_FLOOR`. Built 2026-08-06 on
request; shipped off because the only price-aware evidence inverts it:

| group | n | hit | mean ask | net/contract |
|---|---|---|---|---|
| all placed (today's rule) | 14 | 43% | 0.41 | +0.000 |
| confidence ≥ 0.80 | 5 | 60% | 0.60 | **−0.016** |
| confidence < 0.80 (vetoed) | 9 | 33% | 0.31 | **+0.009** |

It deletes the group that made money and keeps the one that lost it — the ask
climbs from 31¢ to 60¢ to meet the confidence and eats it. This is
`staking.evaluate`'s own documented reason for removing confidence gating in
the first place.

**Revisit only when** there is enough price history to measure the gate on P&L
rather than on hit rate — i.e. after item 7, not after item 1's date.

---

## 13. 🔨 WE ARE 6 HOURS BEHIND THE MARKET ON THE QUANTITY THAT SETTLES

**Found live 2026-08-11, and it is the first gap this project has that is a
disadvantage against the COUNTERPARTY rather than a modelling choice.**

KMIA high, 2026-08-11. Our hourly METARs peaked at 33.9 °C (93.02 °F, 18:53Z)
and fell away. `visible_running_extremum` read 92.92–93.12 all afternoon, which
settles 93 — bucket 92–93. Kalshi repriced **94–95 from 0.37 → 0.84 → 1.00
between 19:12Z and 20:24Z** on information that never reached us. The desk's
model had picked 94–95 at 57.9% (p50 93.7) and was overruled — by me, in
analysis, using the visible running max as though it constrained the answer.

### ✅ CONFIRMED 2026-08-12 — two independent sources, and the peak is nailed

| source | value |
|---|---|
| 23:53Z remark group `10344` | 34.4 °C = **93.92 °F** |
| CLI `CDUS42-CLIMIA` (08:33Z 08-12) | high **94**, `high_time` **2:39 PM** |

**The peak was at 18:39Z — fourteen minutes before the 18:53Z filing, and
0.9 °F above it.** Our hourly snapshot recorded 33.9 °C and the real maximum was
34.4 °C. That difference is one whole Kalshi rung.

Timeline of who knew what:

| | time | lag after the peak |
|---|---|---|
| actual peak | 18:39Z | — |
| market repriced 94–95 to 0.84 | 20:04Z | 1h25m |
| our six-hour group filed | 23:53Z | **5h14m** |

The desk's own pick (94–95 @ 31.5¢, model 57.9%) **WON**. It was overruled in
analysis — by me — on the grounds that `visible_running_extremum` read 92.92–93.12
and therefore "settles 93". That reasoning treats a FLOOR as an estimate. Julian
called it, against my stated read, and was right.

### The measurement — settled high minus the max VISIBLE to us, since 2025-01-01

Identical against both edges of the latent interval (it is narrower than 1 °F),
so this is not an artifact of which edge is read.

| | n | settles ABOVE what we see | by ≥2 °F |
|---|---|---|---|
| KMIA high @ 20:00Z (16:00 EDT) | 581 | **29.6%** | 6.5% |
| KLAX high @ 23:00Z (16:00 PDT) | 585 | **59.8%** | 12.3% |

### It is LATENCY, not blindness — which is why it is fixable

The peak lands in the 18Z–00Z window at both stations (KMIA median 13:58 EDT,
KLAX 13:05 PDT). The six-hour remark group covering that window does not file
until 23:53Z / 00:53Z. So the data DOES arrive — hours after the market has
already traded on it. The market reads 5-minute ASOS; we read hourly METARs plus
a 6-hourly group.

### ⚠️ THERE ARE TWO BLINDSPOTS AND THIS ITEM ONLY FIXES ONE (2026-08-15)

Proven by KMIA HIGH on 2026-08-14, which cost a bucket and which a live feed
would NOT have saved. The six-hour groups bracket the peak exactly:

    group filed 17:53Z  covers 12Z-18Z   max 93.92 F
    group filed 23:53Z  covers 18Z-00Z   max 96.08 F   <- the peak
    commit was 17:00Z, so the 96 happened AFTER it.

**A — LATENCY ON THE PAST.** The extreme has happened; we cannot see it yet.
This item fixes A.

**B — THE FUTURE.** The extreme has not happened at commit. NO observation feed
fixes B. It is forecasting, and the market beats us at 78 of 81 hours anyway.

The walk-forward already separated them and the split is monotone in
`gating.EXTREMUM_STILL_FUTURE` — the feed helps exactly in proportion to how
much of the extreme is already in the past:

| market | still future | dHIT from a live 1-min bound |
|---|---|---|
| KMIA low h7 | 0.417 | **+7.2%** ✓ |
| KLAX low h6 | 0.499 | **+6.4%** ✓ |
| KLAX high h13 | 0.517 | **+4.6%** ✓ |
| **KMIA high h13** | **0.728** | **−0.5% ns** ✗ |

⚠️ **Do not sell this item as fixing KMIA HIGH.** It is measured not to.

⚠️ **And do not "fix" B by refusing high-still-future markets.** That rule would
cut KMIA high, which is 12-of-13 — fitting a policy to one bad day against that
record is the overfitting item 1 forbids.

### Worth building, but it is MONITORING, not trading

The market's own repricing is a sensor for the observation we cannot see: it
crashed 93–94 to 0.01 at 18:07Z, **5h46m** before our group filed, and
`signals.market_json` already stores the ladder every five minutes. By
construction this can never beat the market — but grading, settlement alerts and
the dashboard could know the outcome hours early instead of waiting on the
remark group.

### ✅ AVAILABILITY CHECKED 2026-08-12 — and the answer splits in two

**Do not build a new fetcher. Both already exist.**

| path | code | wired? | state |
|---|---|---|---|
| IEM 1-minute ASOS | `latentmax.fetch_asos1min` | CLI only (`cli.py:901`) | works, **archive-only** |
| Synoptic 5-minute | `fetchers/fivemin.py` | **yes**, every wake (`backfill.py:291`) | **silent no-op, no token** |

`obs` currently holds **zero** rows from either source — 100% `metar`.

**1. IEM 1-minute is REAL but ~24–36h behind.** Confirmed live for both stations
back to 2025. Today: empty for both. 2026-08-11: KMIA stops at 08:10Z, KLAX
entirely empty. **It cannot serve the desk.** Coverage is also patchy (318–1439
of a possible 1440 minutes/day) and temperatures are **whole-degree F**, coarser
than the METAR T-group's 0.18 °F.

**2. But it proves the value is large.** 41 KMIA days, 2026-07-01..08-10, one
range request (48,109 rows):

| | |
|---|---|
| settled high already ATTAINED in 1-min data by 17:00Z (commit) | **61%** |
| attained after commit, same day | 29% |
| never attained — whole-degree F rounding | 10% |
| **median first-touch** | **16:33Z** — 27 min BEFORE commit, 7h20m before our 23:53Z group |

⚠️ "Attained by commit" is not "known to be final" — the max can still climb. The
correct reading is that the ratchet's FLOOR would be far tighter at decision
time, which is exactly the quantity I misused on 2026-08-11.

### The actual unblock: a Synoptic token

`fivemin.fetch_recent` already runs every wake and returns `[]` because
`synoptic_token: ""` in both `config/klax.yaml` and `config/kmia.yaml`.
**[Likely]** Synoptic serves this in real time and in Celsius with decimals,
which also fixes the whole-degree rounding that loses 10% of days above.

**Julian's action, not mine** (account creation): get a Synoptic Data API token
and paste it into both yamls. Nothing else needs to change for the live path.

### ✅ ARCHIVE INGESTED 2026-08-12 — into `obs_1min`, deliberately NOT `obs`

`scripts/ingest_asos1min.py`, resumable via `meta['asos1min_through:<ICAO>']`.
KLAX from 2021-01-01, KMIA from 2020-01-01 (the `cli_reports` floors — earlier
minutes are unusable because the trainer walks CLI dates).

⚠️ **THE TABLE CHOICE IS THE LOAD-BEARING DECISION HERE, not the fetch.**
`latentmax.visible_running_extremum` reads `asos1` rows WITHOUT filtering by
source, and its output feeds `run_lo` / `run_hi` / `run_width`, which are shipped
model features. The archive lags 24–36h, so it can only ever cover TRAINING
rows. Putting it in `obs` would give every training row a ~0.3 °F running-extremum
interval while every LIVE row still gets ~2 °F from hourly METARs — teaching the
model to trust a precision that does not exist at decision time. That is
over-trusting the ratchet exactly when it is loosest, which is the direction that
loses money, and it is the same mistake I made by hand on 2026-08-11.

The model's POINT features are safe either way — `models/intraday.py` and
`features.py` all pin `source='metar'`. Only the running extremum is exposed.

**MOVE IT INTO `obs` ONLY ONCE A LIVE SUB-HOURLY FEED EXISTS**, so both eras
carry comparable precision. The full argument is on the table definition in
`db.py`.

Coverage delivered: **KLAX 2,327,130 rows / 1,871 days (91% of span)**,
**KMIA 3,034,882 rows / 2,313 days (96%)**. Mean 1,244–1,312 of 1,440 minutes on
a covered day. Zero fetch failures across 100 chunks. `obs` is untouched — still
124,721 rows, 100% `metar`. Test suite green at **1,051**.
⚠️ The database grew **230 MB → 832 MB**. Anything that copies it (the
synthetic-clock guard in MEMORY.md requires a copy) is now a 3.6× heavier job.

### ⚠️ THE 1-MINUTE EXTREME IS NOT THE SETTLED VALUE — measured, do not skip

Caught on a five-day spot check before declaring the ingest done: the 1-minute
daily max read exactly +1 °F above the CLI high on four of five days. Over the
full archive:

| | n days | exact match | modal gap | mean bias |
|---|---|---|---|---|
| KLAX high | 1,540 | 29.5% | **+1 °F (56.4%)** | **+1.29 °F** |
| KMIA high | 2,015 | 31.7% | **+1 °F (59.8%)** | **+0.90 °F** |
| KLAX low | 1,540 | 68.6% | 0 (−1 °F on 24.4%) | −0.80 °F |
| KMIA low | 2,012 | 69.0% | 0 (−1 °F on 21.9%) | −0.58 °F |

**The 1-minute series overshoots highs and undershoots lows** — it is a WIDER
range than the one that settles. Mechanism: the official daily extreme comes
from the QC'd 5-minute observations, while this is 1,440 unQC'd samples per day,
and the max of more samples is larger.

**It also contains outright garbage.** ~7.5% of KLAX high days sit ≥3 °F above
settlement, with individual days at **+36, +46, +47, +52, +56 °F** and lows at
**−89, −136 °F**. Those are sensor spikes the official QC removes. A feature
built on a raw `MAX(tmpf)` will occasionally see +56 °F.

**This also revises the "61% attained by commit" figure upward-biased.** Because
the 1-minute max runs ~1 °F hot, it "reaches" the settled high earlier and more
often than a clean series would. Treat 61% as an optimistic ceiling.

### ✅ QC BUILT AND MEASURED 2026-08-12 — 30% → 98%

`src/highcast/asos1min.py` (despike → 5-minute rolling mean, on a gap-aware
minute grid), scored by `scripts/asos1min_qc.py`. 11 tests in
`tests/test_asos1min_qc.py`; **suite 1,051 → 1,062, all green.**

Exact match against the settled CLI value, same >=1000-covered-minute day filter
as the raw table above:

| variant | KLAX high | KLAX low | KMIA high | KMIA low |
|---|---|---|---|---|
| raw | 29.5% | 68.6% | 31.7% | 69.0% |
| despike only | 31.9% | 69.8% | 33.9% | 73.1% |
| 5-min mean only | 96.9% | 95.3% | **98.7%** | **95.5%** |
| **despike + 5-min mean** | **97.4%** | **95.9%** | 98.5% | 94.6% |

Mean bias collapses from +1.29 / −0.80 °F to **±0.05 °F** on every market.

**The attribution matters more than the headline.** The mechanism hypothesis was
right: the official extreme comes from 5-MINUTE observations, and that single
change does essentially all the work (+65 points on KLAX high). Despiking barely
moves exact-match — as predicted, since spikes are only ~7.5% of days — but it
is what kills the tail: `|gap| >= 3 °F` goes 7.7% → 0.6% on KLAX high and
2.7% → 0.4% on KLAX low.

⚠️ **Despike is a small NET LOSS on exact-match at KMIA** (high 98.7 → 98.5,
low 95.5 → 94.6) and a small gain at KLAX. Keep it on anyway, and the reason is
not the daily-extreme score: a feature is a RUNNING extreme, so one +46 °F spike
at 09:00 poisons `run_hi` for the rest of the day. The daily-extreme measurement
understates that damage because it only looks at the end state. Tail safety is
worth ~0.5 points of exact match.

⚠️ **Lows stay worse than highs** (94.6–95.9% vs 97.4–98.5%) and lean +1 °F,
i.e. the QC'd min sometimes sits ABOVE the CLI low — the true minimum is being
missed, probably falling in a coverage hole or getting smoothed out of a sharp
pre-dawn dip. Unfixed.

⚠️ **KMIA high still carries two days at +27 and +28 °F** after full QC. An
11-minute median cannot repair a multi-hour sensor fault. Any consumer needs a
plausibility bound against the METAR series, not just this pipeline.

### ✅ WALK-FORWARD DONE 2026-08-12 — the first idea in this project to survive

`scripts/asos1min_walkforward.py`. Paired arms on IDENTICAL rows (days without
1-minute coverage dropped from both), refit every 30 test days, 4,000-sample
paired bootstrap, 2025-01-01 onward.

Arm A = shipped METAR ratchet bound. Arm B = QC'd 1-minute bound with a **uniform
0.5 °F safety haircut** (see the gate section: KMIA low violated at 1.21% without
it; 0.5 °F takes all four markets to 0.00%).

| market | P(extremum still ahead) | A hit | B hit | dHIT | 98.75% CI | |
|---|---|---|---|---|---|---|
| KMIA low h7 | 0.417 | 65.2% | 72.4% | **+7.2%** | [+3.7, +10.6] | ✓ |
| KLAX low h6 | 0.499 | 69.6% | 76.0% | **+6.4%** | [+1.8, +11.2] | ✓ |
| KLAX high h13 | 0.517 | 67.8% | 72.4% | **+4.6%** | [+1.0, +8.5] | ✓ |
| KMIA high h13 | 0.728 | 68.8% | 68.3% | −0.5% | [−3.3, +2.3] | ✗ |

Three of four survive Bonferroni ×4. CIs are the corrected 98.75% ones,
computed from the bootstrap rather than scaled off the 95% interval.

### THE MECHANISM IS MONOTONE, which is why this is believable

dHIT tracks `gating.EXTREMUM_STILL_FUTURE` perfectly across all four markets:
the more often the extreme has ALREADY HAPPENED by commit, the more a precise
ratchet bound is worth. A bound carries no information about an extreme that has
not occurred yet. Nothing was fitted to produce that ordering.

⚠️ **The market that triggered this whole investigation is the one it does not
help.** KMIA high commits at 13:00 EDT against a 13:58 median peak, so 72.8% of
the time there is nothing yet to measure precisely. The 2026-08-11 case was the
27% — real, but not the average day.

### ⚠️ A LOOKAHEAD LEAK WAS FOUND AND FIXED — read this before trusting a rerun

`asos1min.qc_series` uses CENTRED windows (11-minute median, then 5-minute mean),
so a QC'd value at minute *t* depends on raw minutes to *t+7*. Slicing "strictly
before asof" is therefore NOT leak-free. The first run had that bug.

| market | leaky dHIT | clean dHIT | leak worth |
|---|---|---|---|
| KLAX high | +4.8% | +4.6% | 0.2 pt |
| KLAX low | +7.6% | +6.4% | 1.2 pt |
| KMIA low | +8.5% | +7.2% | 1.3 pt |
| KMIA high | +1.2% | **−0.5%** | 1.7 pt |

The leak did not create the effect, but it was worth up to 1.7 points and it
flipped KMIA high's sign. `LOOKAHEAD_GUARD_MIN = 8` in the script is the fix.
**Any future work on `obs_1min` inherits this trap** — centred QC plus an as-of
slice is a leak every time.

### ⚠️ MONEY TEST 2026-08-12 — DOES NOT MOVE MONEY. Unresolved, not refuted.

`scripts/asos1min_money.py`. Both arms trained on data before the priced window,
run through the FULL shipped desk (`band_calib`, `staking.evaluate`, min-edge,
MAX_EDGE, liquidity) against the REAL Kalshi ladders in `signals`, settled on CLI
with the real taker fee. Paired on day, so both arms face identical prices.

| market | days | A placed | A net/day | B placed | B net/day |
|---|---|---|---|---|---|
| KLAX high h13 | 12 | 5 | +0.1017 | 7 | **+0.1300** |
| KLAX low h6 | 11 | 2 | **+0.0845** | 2 | +0.0309 |
| KMIA high h13 | 13 | 5 | **+0.1154** | 8 | +0.0892 |
| KMIA low h7 | 11 | 6 | +0.1909 | 5 | **+0.2245** |

⚠️ **THE TABLE ABOVE IS VOID — it used the day's FIRST market snapshot, not the
commit-hour one (the `pricehist` lookahead bug).** Re-run at commit prices:

| | n | A placed | A net/day | B placed | B net/day |
|---|---|---|---|---|---|
| pooled | 43 | 20 | **+0.0507** | 21 | **+0.0537** |

**Difference +0.0030/contract, 95% CI [−0.0407, +0.0451], MDE 0.060.** Still no
detectable money effect — but note the corrected absolute returns are less than
HALF the contaminated ones (+0.051 vs +0.122/day), because the stale early quote
was handing both arms a fake edge. The A-vs-B comparison survived the bug (both
arms saw the same wrong prices); the absolute P&L did not.

**THE TEST CANNOT ANSWER THE QUESTION AND WAS KNOWN NOT TO IN ADVANCE.** Smallest
effect detectable at n=47 is **0.124/contract**; a realistic edge is 0.02–0.05.
It is underpowered by roughly 6×, and a CI spanning zero is the EXPECTED result.
This is item 1's power problem, unchanged: Kalshi prices start 2026-07-23 and
cannot be backfilled.

What it CAN do is detect a catastrophe, and there is none — arm B is not the
confidence-gate failure mode (that one inverted P&L outright). So:

- the four dead ideas were **REFUTED** — point estimates went the wrong way with
  a mechanism to explain it;
- this one is **UNRESOLVED** — flat point estimate, no mechanism for harm.
  Those are different verdicts and must not be filed together.

### ⚠️ BUT ONE REAL WARNING SIGN: arm B TRADES MORE, not better

Placements across 47 days: **A 18, B 22 (+22%)**, concentrated in the highs
(5→7 and 5→8). Mechanism: a tighter ratchet bound concentrates the pmf, raising
`p_model` on the top bucket, so more days clear `min_edge`. The desk converts the
improvement into MORE TRADES rather than BETTER ONES — and every extra trade pays
the taker fee.

That is the same shape as the confidence gate's failure (it changed WHAT was
bought, not how well). It did not invert P&L here, but at n=47 it would not have
to be large to hide.

### ❌ MIN_EDGE SENSITIVITY 2026-08-12 — retuning does NOT rescue it, and my
### "trades more, not better" hypothesis is REFUTED

`scripts/asos1min_money_sweep.py`. Deliberately NOT a tuning run — item 1's rule
against sweeping MIN_EDGE on this sample still holds. Two legitimate uses only:
dominance across the range, and turnover matching (threshold chosen from
PLACEMENT COUNTS, never from P&L).

| min_edge | A places | A net/day | B places | B net/day | B − A |
|---|---|---|---|---|---|
| **0.03** *(shipped)* | 18 | **+0.1223** | 22 | +0.1177 | −0.0047 |
| 0.05 | 17 | +0.1157 | 21 | +0.1121 | −0.0036 |
| 0.07 | 16 | +0.1264 | 18 | +0.0889 | −0.0374 |
| 0.10 | 16 | +0.1264 | 17 | +0.0789 | −0.0474 |
| 0.13 | 13 | +0.1247 | 14 | +0.0487 | −0.0760 |
| 0.16 | 11 | +0.0943 | 12 | +0.0472 | −0.0470 |
| 0.20 | 7 | +0.0634 | 9 | +0.0551 | −0.0083 |
| 0.25 | 4 | +0.0466 | 7 | +0.0464 | −0.0002 |

**Arm A is ahead at all 8 thresholds. Arm B is ahead at none.**

**TURNOVER-MATCHED:** arm B needs `min_edge` 0.07 to place 18 times, matching
arm A at 0.03. There: A **+0.1223** vs B **+0.0889**, difference
**−0.0334/contract**, 95% CI [−0.1177, +0.0515].

⚠️ **This refutes my own hypothesis from the money test.** I proposed the deficit
was extra turnover paying extra fees, and predicted matching turnover would close
it. It does the opposite — the gap WIDENS from −0.0047 to −0.0334. At equal trade
counts arm B trades WORSE, not merely more.

⚠️ **THE 8 ROWS ARE NOT 8 INDEPENDENT TESTS.** Same 47 days, nested placement
sets. A consistent sign across the row is roughly ONE observation. Everything
here is also inside the noise band (MDE 0.121/contract) and every CI spans zero.
Consistent direction, not significance.

### [Likely] WHY A BETTER FORECAST MAKES THIS DESK WORSE

Arm B places MORE at every single threshold — 7 vs 4 even at `min_edge` 0.25.
That means the tighter ratchet concentrates the pmf and shifts the whole STATED
EDGE distribution upward.

But this project has measured `corr(edge, won) = −0.571` over 48 settled bets:
**bigger stated edges lose more often.** So an improvement that inflates stated
edge pushes the desk deeper into its own known-bad region. The forecast gets
better and the trading gets worse, because the trading rule reads confidence as
opportunity.

If that is right, the binding constraint is **item 1 (the edge signal points the
wrong way), not model accuracy** — and no amount of forecast improvement pays
until the trading rule is fixed. This experiment is weak evidence for that, and
it is the most useful thing to come out of the whole 1-minute line of work.

**Do not ship the feature and a threshold change together as a package** — the
sweep shows there is no threshold at which the package wins.

### Next, and NOT yet done

1. **Still not deployable.** `obs_1min` lags 24-36h. Arm B measures what a LIVE
   sub-hourly feed would be worth, so this is an argument FOR the Synoptic token,
   not a feature that can ship today. Training on it while serving from METAR is
   exactly the train/serve skew this item exists to avoid.
2. If the token lands, re-verify the violation gate against the LIVE feed before
   wiring it to `_clamp` — the 0.5 °F haircut was calibrated on the archive, and
   Synoptic's 5-minute data has different resolution and QC.
3. Re-run this money test when the priced history reaches ~200 days (item 7's
   BREADTH is the only lever that gets there faster than a year).

⚠️ Re-examine the **cheap-bucket band** afterwards. It now reads **2 of 10 (20%)**
at 0–35¢, and the newest entry is this one — a genuine win the blind spot would
have talked us out of. Seven of the eight losses in that band are KLAX/KMIA
**LOW** picks, where the equivalent latency runs the other way. The band is not
one phenomenon and should not be read as one number.

| ask | record |
|---|---|
| 0–35¢ | 2/10 = 20% |
| 35–60¢ | 10/14 = 71% |
| 60–85¢ | 16/18 = 89% |
| 85¢+ | 14/14 = 100% |

KMIA high alone is now **12 of 13**.

---

## Not doing, and why

- **Auto-skip weatherguard** (Prompt 3 part 1). The near-boundary days it would
  skip still hit **72%**, and skipping is only correct if the market has not
  already discounted them — unprovable at 9 priced days. Shipped as a WARNING
  instead: it cannot lose money and Julian stays the decision-maker.
- **More model accuracy work.** Nested walk-forward says KMIA HIGH is MAE 0.551 /
  77.6% hit and calibration is within ~2 points. Accuracy is not the binding
  constraint; there is no defect left worth the implementation risk.
- **Near-boundary warning (all four markets).** Measured 2026-08-07 on 2,282
  walk-forward records. Raw, it looks strong — a point estimate within 0.35 °F
  of a rounding boundary hits worse in **every** market: KLAX high 63.2% vs
  69.3%, KLAX low 66.8% vs 68.8%, KMIA high 74.2% vs 80.9%, KMIA low 51.0% vs
  68.3%.
  **It is not independent information.** Hold the model's own `stated`
  probability fixed and the effect vanishes into noise in every band of every
  market — several bands flip sign (KMIA high goes +14.1%, +4.3%, +6.7%). Only
  KMIA LOW's raw difference is significant, and its mechanism is visible in the
  means: stated 0.545 on near-boundary days vs 0.661 elsewhere. The model
  already knows, and already says so. A warning would show the same fact twice
  under a new name.
  ⚠️ This is the FOURTH idea killed by the same test in one session (with arm
  B, the guidance-spread veto, and the confidence gate). The test is: *does it
  survive conditioning on `stated`, and does it move MONEY rather than hit
  rate?* Run it first, not last.
- **Guidance-spread veto on KMIA LOW.** Measured 2026-08-06, 914 leak-free days
  (day-ahead runs only, ≥5 models). Hit rate is FLAT across spread bands
  (68.2 / 64.2 / 66.9 / 68.6 / 68.3%) and at every threshold the days it would
  veto hit BETTER than the days it would keep — e.g. at range >5 °F, kept 63.8%
  vs vetoed 68.1%. The veto is backwards.
  The hypothesis was not stupid: spread does concentrate the model's warm
  blowouts (range >10 °F ⇒ 24.4% of days are >4 °F too warm, vs ~4% elsewhere,
  and the model has 33 blowouts >4 °F warm against ZERO >4 °F cold). But a
  bucket contract is BINARY — it pays the same whether you miss by 0.5 °F or
  7 °F — so error MAGNITUDE has no bearing on the payoff. The model's own
  stated probability separates hit rate far better anyway (42.6% at <0.50 vs
  84.1% at ≥0.70) and is already available for free.
- **Bias-correcting the HIGH's guidance.** Refuted on the codebase's own
  criterion: KMIA HIGH guidance error autocorrelation is r=0.22–0.40 versus the
  r=0.80–0.86 that justified `fcc_*` at KLAX LOW. The offset is static, and a
  tree absorbs a static offset by moving its split thresholds.
- **Repairing the nowcast's `heat_alert` timing.** Found 2026-08-19: the nowcast
  INHERITS `heat_alert` from the day-ahead row, so it is evaluated at the
  day-ahead cutoff. Miami advisories are issued the same morning — the feature
  fires on **1 of 593** KMIA days (and that one is a Red Flag Warning), against
  **46** if asked at the checkpoint. At KLAX the 17 hits are January/March Santa
  Ana products, so there `heat_alert` is really a WIND flag.
  `scripts/heat_alert_asof_gate.py` swapped the column contents and **failed on
  all four markets** (KMIA high dHIT −0.1%, CI [−1.3%,+1.1%]; best arm anywhere
  was KLAX high at +0.2% with the CI touching zero and MAE no better). The column
  survives as `heat_alert_asof`, measured-inert.
  ⚠️ Worth knowing WHY this one matters beyond the null: it is the first of these
  to die as a *correctness* fix rather than a hopeful feature. Being wrong about
  WHEN a question is asked cost nothing measurable — so do not assume the next
  timing bug is worth fixing for accuracy either. Fix it for honesty, and say so.
