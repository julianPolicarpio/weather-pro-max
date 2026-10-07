# FULL UNFREEZE — all four markets

**Date:** 2026-08-02
**Status:** decided (Julian's instruction: "unfreeze. everything")
**Supersedes:** the freeze declared 2026-07-28 and the KLAX-HIGH-only partial
unfreeze of 2026-07-31 (`2026-07-30-pick-of-day-and-model-neutral-queue-design.md`)

## What is now true

**KLAX HIGH, KLAX LOW, KMIA HIGH and KMIA LOW are all unfrozen.** A change may
alter a prediction, a probability, or when a bet commits, on any market, without
needing a per-market exemption first.

## What this costs, stated rather than glossed

**The app-vs-Julian scorecard ends for the remaining three markets.** KLAX HIGH's
comparison already ended on 2026-07-31. KLAX LOW, KMIA HIGH and KMIA LOW keep
their records to date, but nothing recorded after 2026-08-02 is comparable to
what came before, because the model underneath them is now allowed to move. That
was the entire purpose of the freeze and it is being spent deliberately.

The first change to land under this unfreeze is the `fc_anchor` exclusion from
`BaselineBlend` (same date), which alters `base_p50`/`base_spread` on all four
markets. See `scripts/anchor_exclusion_eval.py` for its before/after.

## What this does NOT do

An unfreeze is permission, not a queue of changes to apply. Nothing was flipped
as a side effect of this record. Three items were deferred *solely* because of
the freeze and are now unblocked, each still needing its own decision:

| item | where | state |
|---|---|---|
| `intraday_hour_bandwidth` k=1 | `config.py` | measured, k=1 dominates k=0 for the HIGH on all three axes; default still 0 |
| `late_commit_guard` | `config.py` | correct behaviour, narrows the commit window; default still False |
| Pick of the Day wiring | `pick.py`, `picks` table | built; the WRITER (`pick.record`) is still called from nowhere and remains the open decision. The GRADER (`pick.settle`) was wired into the wake cycle and `highcast settle` on 2026-08-02 — a no-op while `picks` is empty, so it publishes nothing and decides nothing; it only guarantees that whatever does get published cannot sit ungraded. `notify`/`web` untouched. |

The freeze also never gated the **trading-capability** contract, which is a
separate and still-active rule: trading code lives in exactly one file
(`kalshi_client.py`), `bankroll` stays 0, `app_bets` stay paper. See
`tests/test_freeze_contracts.py`. Unfreezing the MODEL does not touch any of it.

## The standing rules that survive

None of the project's measurement discipline changes. In particular: analysis
queries get the same scrutiny as shipped code; ladder parity is never
hardcoded; daily extremes come from `latentmax`, never from raw `obs`
snapshots; and a change to a live market still gets measured before it ships —
the unfreeze removes the scorecard objection, not the evidence bar.
