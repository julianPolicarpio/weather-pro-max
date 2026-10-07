# Buying NO contracts instead of YES — DECISION RECORD: not built

**Date:** 2026-08-01
**Status:** investigated, measured, **NOT BUILT**. Revisit trigger below.
**Asked by:** Julian — "brainstorm buying no contracts on kalshi instead of
buying the yes", refined to **NO at 80–90¢** (i.e. selling buckets the market
prices at 10–20%).

---

## Why it looked promising

**The fee structure genuinely favours the extremes.** Kalshi's fee is
`0.07 × C × P × (1−P)`, so the *absolute* hurdle collapses away from the middle:

| price | fee | edge needed to break even |
|---|---|---|
| 0.50 | 1.75¢ | 1.75 points |
| 0.80 / 0.20 | 1.12¢ | 1.12 points |
| 0.95 / 0.05 | 0.33¢ | 0.33 points |

At 80–90¢ you keep a high win rate, get an 11–25% return per trade (versus 5% at
95¢), and pay ~1.2% in fees. The structure is sound.

**NO prices need no new capture.** Kalshi's identity gives them exactly:
`no_bid = 1 − yes_ask`, `no_ask = 1 − yes_bid`. We already store both sides on
100% of quotes, so every historical NO price is derivable.

**Availability is adequate.** A bucket sits in the 80–90¢ NO zone in 30–57% of
snapshots (KMIA LOW 57%, KLAX LOW 39%, KMIA HIGH 37%, KLAX HIGH 30%).

## Why it was not built

### 1. In the target zone, the model is CALIBRATED — nothing to sell

`scripts/tail_calibration_eval.py`, walk-forward at each market's decision hour,
every 2 °F bucket on both ladder parities, pooled n≈76,000 bucket-days:

| model says | n | stated | actual | gap | 95% CI on actual |
|---|---|---|---|---|---|
| 2–5% | 4,475 | 3.0% | **1.7%** | **+1.3** | [1.4, 2.1] ← excludes |
| 5–10% | 1,366 | 7.2% | 7.5% | −0.2 | [6.2, 9.0] |
| **10–15%** | 933 | 12.3% | 12.4% | **−0.1** | [10.5, 14.7] |
| **15–20%** | 715 | 17.3% | 15.5% | +1.8 | [13.1, 18.4] ← includes |
| 20–30% | 1,310 | 24.7% | 23.5% | +1.2 | [21.3, 25.9] ← includes |

The 10–20% zone — exactly where NO trades at 80–90¢ — shows gaps of −0.1 and
+1.8 with CIs containing the stated value. **No systematic overstatement exists
to harvest.**

Per market at 15–20%: KMIA HIGH +1.0 (ns), KLAX HIGH **−3.3** (wrong direction —
its NO would be OVERpriced), KLAX LOW −0.7 (ns), KMIA LOW **+8.8** (CI excludes).
That last is 1 significant result in 24 uncorrected tests — roughly the false
positive rate. Treat as hypothesis, not finding.

The only real overstatement is at **2–5%** (NO at 95–98¢), which is the zone
Julian explicitly moved away from, and where a 1.3-point edge against a 1¢
rounded fee leaves ~0.3% per trade.

### 2. THE DECISIVE POINT — the model and the market agree in the tail

Model vs de-vigged market at the decision hours, tail buckets only:

| | value |
|---|---|
| mean model prob | 6.4% |
| mean market prob | 6.8% |
| **model − market** | **−0.4 points** |

Even if the tail were systematically overpriced, **the model has no
differentiated view of it.** Harvesting a mispriced tail requires a better
estimate than the market's; ours is the same estimate. Selling tails regardless
is not an edge strategy — it is a high-win-rate, occasional-total-loss structure
with negative expectancy whenever the market is right.

This is the mirror of the existing finding. "Big claimed edge = model error" says
do not trade when you disagree loudly with the market. This says you also cannot
trade when you agree with it. The tail offers neither.

### 3. The price-vs-outcome test cannot be run yet

n = **27** tail bucket-days with both a price and a settled outcome (Kalshi
capture began 2026-07-24). Actual hit rate 3.7% is **one event in 27**; its
Wilson interval is ~[0.7%, 18%], which contains the market's 6.8%. Zero evidence
either way.

## Revisit trigger

Capture accumulates ~3.4 tail bucket-days per calendar day across four markets.
No new instrumentation is needed — the 2026-08-01 capture audit confirmed 100%
two-sided book coverage and predictions at every decision hour.

- **~50 more days (≈ mid-September 2026)** → n≈200, resolves a 5-point mispricing
- **~153 days** → n≈550, resolves 3 points

**Pre-registered gate for the revisit — BOTH must hold:**

1. `market_implied − realized` in the 10–20% zone exceeds the fee hurdle
   (≈1.1 points at those prices), CI excluding zero; **and**
2. the model deviates from the market enough to time it — i.e. |model − market|
   in the tail is materially non-zero, so there is a signal to act on rather
   than a blanket tail-sell.

Condition 2 is the one that failed here, and it is the one that matters. A
positive result on (1) alone justifies nothing.

## Scope decisions taken before measuring (recorded so they are not re-litigated)

- One NO position per day, the single best — keeps the $1/day cap meaningful.
- Alongside the YES logic, not replacing it. The KMIA HIGH ≥80% band measuring
  93.4% is the best-evidenced edge in the project and must not be abandoned.
- NO on unlikely buckets at 80–90¢, not NO on the favourite (which is a bet
  against consensus, where the ledger already shows the model loses).

## Files

`scripts/tail_calibration_eval.py` (new, kept — it answers a question no other
script asks, and the revisit will re-run it). Raw output:
`.superpowers/sdd/tail-calibration.txt`.
