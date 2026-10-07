"""QUEUE ITEM 1, RE-MEASURED. The one number that decides whether any of the
rest is worth doing.

THE FINDING BEING RE-TESTED, as recorded at n=36 on 2026-08-13:

    PLACE 43% (n=14)   SKIP 91% (n=22)   z = 3.14   p < 0.001
    corr(edge, won) = -0.571     corr(ask, won) = +0.596

Eight more settled decisions have landed. The PLACE/SKIP gap has visibly
narrowed, and that gap is the empirical foundation for "the edge signal points
the wrong way" — so it is worth measuring properly rather than eyeballing.

⚠️ WHAT THIS CAN AND CANNOT SETTLE. Item 1's own power calculation:

    A — is PLACE worse than SKIP on HIT RATE?  answerable, this is that test
    B — does PLACE make or lose MONEY?         NOT answerable; per-contract net
                                               has sd 0.404, so ~512 PLACE bets
                                               are needed to see a 0.05 edge

So a money figure is printed for completeness and must not be read as a verdict.

⚠️ PLACE/SKIP IS NOT A RANDOM SPLIT. The rule chooses it, so this measures the
RULE'S DISCRIMINATION, not a treatment effect. And with `staking.bankroll = 0`
no PLACE was ever executed — every P&L number here is hypothetical.

Usage: PYTHONPATH=src .venv/bin/python scripts/item1_recheck.py
"""
import math

import numpy as np

from highcast.config import load_config
from highcast.db import connect
from highcast.fees import fee_per_contract

N_BOOT = 20000
rng = np.random.default_rng(20260815)


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided normal p
    pv = math.erfc(abs(z) / math.sqrt(2))
    return z, pv


print(__doc__.split("Usage:")[0])
cfg = load_config("config/kmia.yaml")
conn = connect(cfg.db_path, migrate=False)
rows = [dict(r) for r in conn.execute(
    """SELECT station, metric, target_date, bucket, p_model, price, actual, won,
              reason FROM app_bets
       WHERE won IS NOT NULL AND price IS NOT NULL AND p_model IS NOT NULL
       ORDER BY target_date""")]
# ⚠️ ONE LEGACY ROW HAS A NULL REASON (KLAX low 2026-07-27, p=0.88 @0.66, WON).
# Its edge of +22% makes PLACE the likely classification, but "likely" is not a
# record — it is EXCLUDED from the PLACE/SKIP split rather than guessed into the
# favourable bucket. It still counts in the correlations and price bands, which
# do not need the classification.
unclassified = [r for r in rows if r["reason"] is None]
for r in rows:
    r["place"] = bool(r["reason"]) and r["reason"].startswith("PLACE")
    r["classified"] = r["reason"] is not None
    r["edge"] = float(r["p_model"]) - float(r["price"])
    p = float(r["price"])
    r["net"] = ((1.0 - p) if r["won"] else -p) - fee_per_contract(1, p)

pl = [r for r in rows if r["classified"] and r["place"]]
sk = [r for r in rows if r["classified"] and not r["place"]]
if unclassified:
    print(f"\n  ⚠️ {len(unclassified)} settled row(s) excluded from the PLACE/SKIP "
          f"split — NULL reason, cannot be classified: "
          + ", ".join(f"{u['station']} {u['metric']} {u['target_date']}"
                      for u in unclassified))
print(f"\n{'='*72}\nA — HIT RATE, the question that IS answerable\n{'='*72}")
print(f"  {'group':<8}{'n':>5}{'won':>6}{'hit':>9}   95% CI")
for name, g in (("PLACE", pl), ("SKIP", sk)):
    k, n = sum(r["won"] for r in g), len(g)
    lo, hi = wilson(k, n)
    print(f"  {name:<8}{n:>5}{k:>6}{k/n:>9.1%}   [{lo:.1%}, {hi:.1%}]")
z, pv = two_prop_z(sum(r["won"] for r in pl), len(pl),
                   sum(r["won"] for r in sk), len(sk))
gap = sum(r["won"] for r in pl)/len(pl) - sum(r["won"] for r in sk)/len(sk)
print(f"\n  gap {gap:+.1%}   z = {z:.2f}   p = {pv:.4f}   "
      f"{'SIGNIFICANT' if pv < 0.05 else 'NOT significant at 0.05'}")
print(f"\n  THEN (2026-08-13, n=36):  PLACE 43% (n=14)  SKIP 91% (n=22)  "
      f"gap -48pt  z=3.14  p<0.001")
print(f"  NOW  (n={len(rows)}):            PLACE {sum(r['won'] for r in pl)/len(pl):.0%} "
      f"(n={len(pl)})  SKIP {sum(r['won'] for r in sk)/len(sk):.0%} (n={len(sk)})  "
      f"gap {gap:+.0f}pt  z={z:.2f}  p={pv:.3f}")

# ── correlations ────────────────────────────────────────────────────────────
print(f"\n{'='*72}\nCORRELATIONS (all settled decisions, n={len(rows)})\n{'='*72}")
won = np.array([float(r["won"]) for r in rows])
for label, key, then in (("edge (p_model - ask)", "edge", -0.571),
                         ("ask (what the market charged)", "price", +0.596),
                         ("p_model", "p_model", +0.163)):
    v = np.array([float(r[key]) for r in rows])
    c = float(np.corrcoef(v, won)[0, 1])
    boot = np.array([np.corrcoef(*np.array([[v[i], won[i]] for i in
                     rng.integers(0, len(v), len(v))]).T)[0, 1]
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:<32}{c:+.3f}   95% CI [{lo:+.3f},{hi:+.3f}]   "
          f"(was {then:+.3f})")

# ── the price ladder ────────────────────────────────────────────────────────
print(f"\n{'='*72}\nHIT RATE BY WHAT THE MARKET CHARGED\n{'='*72}")
bands = [(0.0, 0.35), (0.35, 0.60), (0.60, 0.85), (0.85, 1.01)]
prev = {"0-35c": "20%", "35-60c": "57%", "60-85c": "80%", "85c+": "100%"}
for lo, hi in bands:
    g = [r for r in rows if lo <= float(r["price"]) < hi]
    if not g:
        continue
    k, n = sum(r["won"] for r in g), len(g)
    wl, wh = wilson(k, n)
    key = f"{int(lo*100)}-{int(hi*100)}c" if hi < 1 else "85c+"
    print(f"  {key:<9}{k:>3}/{n:<4}{k/n:>7.0%}   [{wl:.0%},{wh:.0%}]"
          f"   (was {prev.get(key, '—')})")

# ── money, for completeness only ────────────────────────────────────────────
print(f"\n{'='*72}\nB — MONEY. UNDERPOWERED BY CONSTRUCTION; NOT A VERDICT\n{'='*72}")
for name, g in (("PLACE", pl), ("SKIP", sk)):
    if not g:
        continue
    a = np.array([r["net"] for r in g])
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean()
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:<8}n={len(a):<4} net/contract {a.mean():+.4f}   "
          f"95% CI [{lo:+.4f},{hi:+.4f}]   total {a.sum():+.2f}")
a = np.array([r["net"] for r in pl])
mde = 2.8 * a.std() / math.sqrt(len(a)) if len(a) > 1 else float("nan")
print(f"\n  smallest PLACE edge detectable at n={len(pl)}: {mde:.3f}/contract")
print(f"  a realistic edge is 0.02-0.05 -> still ~{int((2.8*a.std()/0.05)**2)} "
      f"PLACE bets needed. Item 1's arithmetic is unchanged.")
print("\n  ⚠️ bankroll is 0: no PLACE was executed. These are hypothetical.")
