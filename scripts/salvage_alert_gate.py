"""SHOULD THE SALVAGE ALERT BECOME A TRIGGER? PRE-REGISTERED, OUT-OF-SAMPLE.

════════════════════════════════════════════════════════════════════════════
WRITTEN 2026-08-16, BEFORE ANY OUT-OF-SAMPLE DAY EXISTS. Nothing below may be
edited once the first OOS day settles. If a parameter here turns out to be
wrong, the honest move is to FAIL this gate and pre-register a new one — not to
adjust this file and re-run it.
════════════════════════════════════════════════════════════════════════════

THE CANDIDATE. Fire when the market has moved decisively against the committed
bucket WHILE that bucket is still worth selling:

    max(market_price) over buckets != ours  >=  0.60      (CONC)
    AND market_price(ours)                  >=  0.20      (SALV)

WHERE IT CAME FROM, and why it does not get to ship on that evidence.
`scripts/divergence_alert_eval.py` swept ~40 (persistence x concentration)
cells over 68 settled day-markets and this cell was the best one: 15 fires, 11
losers, 4 winners, P(lost | fired) 73% against 15% when silent, Fisher
p = 3.8e-5, still 0.0015 after x40 Bonferroni. Surviving Bonferroni is NOT the
same as being pre-registered — the sweep also chose the SHAPE of the rule, and
no correction prices that in. Four ideas in this project have died looking this
good on ~60 days, so the in-sample result is treated as a HYPOTHESIS ONLY.

⚠️ IN-SAMPLE DAYS ARE EXCLUDED FROM THE VERDICT. Every settled day-market with
`target_date <= 2026-08-16` was visible when the rule was chosen. They are
reported below for contrast and CONTRIBUTE NOTHING to the decision.

────────────────────────────────────────────────────────────────────────────
PRE-REGISTERED DECISION RULE
────────────────────────────────────────────────────────────────────────────

SAMPLE. All settled day-markets with `target_date >= 2026-08-17`, both
stations, both metrics, that have a committed bet and at least one stored
market snapshot after the commit. No other exclusions.

STOPPING RULE. Do not read a verdict before **N_MIN = 100** OOS day-markets.
Powered by simulation (3,000 draws, fire rate 15/68) against a SHRUNK effect of
55% vs 20% — not the 73% vs 15% observed, because a swept cell's effect size is
the most optimistic thing about it:

    n=60 -> 62%    n=80 -> 76%    n=100 -> 84%    n=120 -> 92%

At ~28 settled day-markets per week that is roughly 3.5 weeks. Peeking earlier
is fine and this script prints progress; ACTING on an early look is not.

PRIMARY OUTCOME. P(bet lost | rule fired) minus P(bet lost | rule silent),
two-sided Fisher exact. **alpha = 0.05, uncorrected** — this is ONE
pre-specified test of ONE frozen rule, which is the entire point of registering
it. No Bonferroni, and no second cell to compare against.

SHIP THRESHOLDS — ALL FOUR must hold. The first two are this gate; the last two
are the project's standing bar (HANDOFF §6), which no idea gets to skip:

  1. PRIMARY   difference > 0 and Fisher p < 0.05 on OOS days alone.

  2. SURVIVES `stated`  — the sign of the difference holds in BOTH halves of
     the committed-probability split (median of the OOS committed `p_model`).
     SIGN ONLY, not significance: each half is underpowered by construction and
     demanding p < 0.05 in both would be a power test wearing a validity
     costume. Four dead ideas were restatements of `stated`; this is the check
     that killed them.

  3. MOVES MONEY  — exiting at the fire must beat holding, priced at the BID
     the desk could actually have hit, summed over fires:

         delta = SUM over fired day-markets of ( yes_bid(ours) - won )

     A winner exited early costs (1 - bid); a loser exited early saves bid.
     Require delta > 0. ⚠️ Hit rate is not money: the alert can be a fine
     classifier and still lose, because it fires on days our bucket is cheap.

  4. AGREES WITH THE LEDGER — `won` is read from `app_bets`, the settled
     record, not from any replay of the model.

ON PASS: promote to a real alert kind (its own `salvage_alert_utc` stamp,
one fire per day-market, message already drafted in `notify.market_position`).
ON FAIL: delete the candidate. Do not re-sweep for a better (CONC, SALV) — that
is how a dead idea becomes an immortal one.

EXPECTED FAILURE MODE, STATED UP FRONT so it cannot be rationalised later:
threshold 3. The rule fires when the market has already moved against us, which
is exactly when our bucket is cheap, so the bid we could exit at is small while
the winners we abandon pay 1.00. Four fires on winners at ~0.25 would cost ~3.0
against 11 losers saving ~0.25 each = ~2.75. **[Likely] this fails on money
while passing on hit rate.** That would be the gate working.

SECOND EXPECTED FAILURE, honestly: with `staking.bankroll = 0` the desk holds
no position to exit, so threshold 3 is counterfactual arithmetic on paper bets.
It is still the right test — if it does not pay on paper it cannot pay live —
but a PASS here does not license real money by itself.

Writes no constant and changes no behaviour. Its output is read by a human.

Usage: PYTHONPATH=src .venv/bin/python scripts/salvage_alert_gate.py
"""
import json
from collections import defaultdict
from math import comb

from highcast.config import load_config
from highcast.db import connect

# ── FROZEN 2026-08-16. Do not edit. ──────────────────────────────────────────
CONC = 0.60          # market on some other bucket at or above this
SALV = 0.20          # our bucket still trading at or above this
OOS_START = "2026-08-17"
N_MIN = 100
ALPHA = 0.05
# ─────────────────────────────────────────────────────────────────────────────


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if not n or r1 in (0, n) or c1 in (0, n):
        return 1.0

    def p(x):
        return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)
    obs = p(a)
    lo = max(0, c1 - (n - r1))
    return sum(p(x) for x in range(lo, min(r1, c1) + 1)
               if p(x) <= obs * (1 + 1e-9))


def load():
    bets, snaps = {}, defaultdict(list)
    for icao in ("KLAX", "KMIA"):
        cfg = load_config(f"config/{icao.lower()}.yaml")
        conn = connect(cfg.db_path, migrate=False)
        for r in conn.execute(
                """SELECT metric, target_date, bucket, p_model, won, placed_utc
                     FROM app_bets WHERE station=? AND won IS NOT NULL
                       AND placed_utc IS NOT NULL""", (icao,)):
            bets[(icao, r["metric"], r["target_date"])] = dict(r)
        for r in conn.execute(
                """SELECT metric, target_date, created_utc, market_json
                     FROM signals WHERE station=? AND kind IN ('dayahead','intraday')
                       AND market_json IS NOT NULL AND market_json != ''
                   ORDER BY created_utc""", (icao,)):
            try:
                lad = json.loads(r["market_json"])
            except ValueError:
                continue
            if lad:
                snaps[(icao, r["metric"], r["target_date"])].append(
                    (r["created_utc"], lad))
    return bets, snaps


def fires(seq, bet):
    """First qualifying snapshot after the commit -> (when, our bid). None if
    the rule never fires. Exit is priced at the BID, never the mid: the mid is
    not a price anyone can sell into."""
    for when, lad in seq:
        if when < bet["placed_utc"]:
            continue
        ours = next((x for x in lad if x["label"] == bet["bucket"]), None)
        if ours is None or not isinstance(ours.get("market_price"), (int, float)):
            continue
        if ours["market_price"] < SALV:
            continue
        if any(isinstance(x.get("market_price"), (int, float))
               and x["label"] != bet["bucket"] and x["market_price"] >= CONC
               for x in lad):
            bid = ours.get("yes_bid")
            return when, (bid if isinstance(bid, (int, float))
                          else ours["market_price"])
    return None


def table(days, bets, fired):
    f = [k for k in days if fired.get(k)]
    q = [k for k in days if not fired.get(k)]
    fl = sum(1 for k in f if not bets[k]["won"])
    ql = sum(1 for k in q if not bets[k]["won"])
    pf = fl / len(f) if f else float("nan")
    pq = ql / len(q) if q else float("nan")
    return f, q, fl, ql, pf, pq, fisher(fl, len(f) - fl, ql, len(q) - ql)


def report(label, days, bets, fired, decisive):
    f, q, fl, ql, pf, pq, p = table(days, bets, fired)
    print(f"\n  {label}: n={len(days)}, {sum(1 for k in days if not bets[k]['won'])} lost")
    if not days:
        return None
    print(f"    fired {len(f)}  ({fl} lost / {len(f)-fl} won)   "
          f"silent {len(q)}  ({ql} lost / {len(q)-ql} won)")
    if not f or not q:
        print("    one cell empty — no comparison possible")
        return None
    print(f"    P(lost|fired) {pf:.0%}   P(lost|silent) {pq:.0%}   "
          f"diff {pf-pq:+.0%}   Fisher p {p:.4f}")
    if not decisive:
        print("    ⚠️ IN-SAMPLE — contributes NOTHING to the verdict.")
    return {"f": f, "q": q, "pf": pf, "pq": pq, "p": p}


def decide(oos, bets, fired):
    """The three testable thresholds. Returns a dict; prints nothing.

    ⚠️ EXTRACTED FOR TESTABILITY ONLY — no threshold, window or statistic here
    differs from the frozen block above, and none may. A pre-registered gate
    whose SHIP branch has never executed is a guard that cannot fire, which is
    the same defect as errors #8 and #11 (HANDOFF §0b): it looks like coverage
    until the day it matters. `tests/test_salvage_gate.py` drives every branch.
    """
    f, q, fl, ql, pf, pq, p = table(oos, bets, fired)
    primary = bool(f) and bool(q) and pf > pq and p < ALPHA

    ps = sorted(bets[k]["p_model"] for k in oos
                if bets[k]["p_model"] is not None)
    cut = ps[len(ps) // 2] if ps else None
    halves = {}
    for name, sel in (("below median", lambda k: bets[k]["p_model"] < cut),
                      ("at/above median", lambda k: bets[k]["p_model"] >= cut)):
        sub = [k for k in oos
               if cut is not None and bets[k]["p_model"] is not None and sel(k)]
        sf, sq, _, _, spf, spq, _ = table(sub, bets, fired)
        # An empty cell cannot demonstrate the sign HOLDS. Fail closed.
        halves[name] = {"pf": spf, "pq": spq,
                        "holds": bool(sf) and bool(sq) and spf > spq}
    stated = all(h["holds"] for h in halves.values())

    delta = sum(fired[k][1] - (1 if bets[k]["won"] else 0)
                for k in oos if fired.get(k))
    money = delta > 0

    return {"primary": primary, "stated": stated, "money": money,
            "ship": primary and stated and money, "p": p, "diff": pf - pq,
            "delta": delta, "n_fired": len(f), "halves": halves}


def main():
    print(__doc__.split("Usage:")[0])
    bets, snaps = load()
    days = sorted(k for k in bets if k in snaps)
    fired = {k: fires(snaps[k], bets[k]) for k in days}
    ins = [k for k in days if k[2] < OOS_START]
    oos = [k for k in days if k[2] >= OOS_START]

    print("=" * 78)
    print(f"RULE (frozen): market >= {CONC:.2f} on another bucket "
          f"AND ours >= {SALV:.2f}")
    print(f"OOS window: target_date >= {OOS_START}   |   N_MIN = {N_MIN}")
    print("=" * 78)

    report("IN-SAMPLE (the sweep that produced the hypothesis)", ins, bets,
           fired, decisive=False)
    res = report("OUT-OF-SAMPLE (the only evidence that counts)", oos, bets,
                 fired, decisive=True)

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    if len(oos) < N_MIN:
        pct = len(oos) / N_MIN
        print(f"  ⏳ NOT YET DECIDABLE — {len(oos)}/{N_MIN} out-of-sample "
              f"day-markets ({pct:.0%}).")
        print(f"     ~{max(0, (N_MIN - len(oos)) / 28):.1f} weeks to go at ~28/week.")
        print("     Any numbers printed above for the OOS window are a PEEK. The")
        print("     pre-registration says do not act on them, and an early look")
        print("     that happens to look good is exactly what the rule guards.")
    else:
        v = decide(oos, bets, fired)
        for name, h in v["halves"].items():
            print(f"  stated {name:<16} P(lost|fired) {h['pf']:.0%} vs "
                  f"{h['pq']:.0%}  -> {'holds' if h['holds'] else 'FLIPS'}")
        print(f"\n  1. primary   {'PASS' if v['primary'] else 'FAIL'}  "
              f"diff {v['diff']:+.0%}, p {v['p']:.4f}")
        print(f"  2. vs stated {'PASS' if v['stated'] else 'FAIL'}  "
              f"{'sign holds in both halves' if v['stated'] else 'the sign flips'}")
        print(f"  3. money     {'PASS' if v['money'] else 'FAIL'}  "
              f"exiting at the bid nets {v['delta']:+.2f} contracts over "
              f"{v['n_fired']} fires")
        print("  4. ledger    PASS  `won` read from app_bets throughout")
        print(f"\n  ==> {'SHIP' if v['ship'] else 'DO NOT SHIP'}")
        if not v["ship"]:
            print("      Delete the candidate. Do NOT re-sweep (CONC, SALV).")


# ⚠️ Guarded so `tests/test_salvage_gate.py` can import `decide`/`fires` without
# touching the live database. Importing a script must never run it.
if __name__ == "__main__":
    main()
