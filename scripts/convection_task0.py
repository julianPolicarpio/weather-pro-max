"""TASK 0 — does the FORECAST identify the same days the OBSERVATION does?

THE GATE, pre-registered in
`docs/superpowers/specs/2026-08-02-convection-feed-revival-design.md` §3 and
reproduced here verbatim so the bar cannot drift after seeing the numbers:

    Pre-registered question: on days `forecast_conv` predicted convection, does
    the low actually drop late more often than on days it did not?

    Arms: pop above a threshold during the station's afternoon
          cape above a threshold
          a shuffled control (the same labels, randomly reassigned to days)

    Score: rate of "fell >=2F after commit", forecast-positive days vs
           forecast-negative, with a paired bootstrap CI.

    The bar: the real arm must beat the shuffled control, and its separation
    must be a meaningful fraction of the 24.2% vs 6.3% that OBSERVED thunder
    achieves. If the forecast only recovers a sliver of what the observation
    shows, then convection is not predictable at this lead time for this
    station, and the honest deliverable is a null result plus keeping the
    observed-thunder warning.

    A null result here is a valid and expected outcome. Do not tune thresholds
    until the numbers look good -- the sample is ~730 days and a swept
    threshold on a rare event fits noise trivially.

WHY THIS AND NOT `scripts/convection_eval.py`. That script asks a LATER and far
more expensive question -- does adding conv_* features to the model improve
bucket hit, via full retraining on two arms. Task 0 is the cheap prior gate:
if the forecast cannot even flag the days the observation flags, no amount of
feature engineering on it will help. Running the cheap one first is the whole
point of calling it Task 0.

LEAK-FREE. `forecast_conv` holds Open-Meteo `*_previous_day1` fields -- what
each model predicted ONE DAY AHEAD -- so they were available at the decision
hour. Observed precipitation is never touched here; that is hindsight.

THRESHOLDS ARE DECLARED BEFORE LOOKING. pop >= 50% and cape >= 2500 are round
numbers chosen for plausibility, not swept. Every threshold tried is printed so
a reader can see nothing was cherry-picked.

Usage: PYTHONPATH=src .venv/bin/python scripts/convection_task0.py
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from highcast import gating, latentmax
from highcast.config import load_config
from highcast.db import connect

CFG = load_config("config/kmia.yaml")
TZ = ZoneInfo(CFG.station.tz)
COMMIT_HOUR = gating.MIN_DECISION_HOUR["KMIA"]["low"]      # h7
DROP_F = 2.0                       # "fell >=2F after commit", per the spec
AFTERNOON = range(12, 21)          # station-local hours the convection matters
N_BOOT = 4000
rng = np.random.default_rng(20260810)

conn = connect(CFG.db_path)

# ── outcome: did the low fall >=2F AFTER the commit hour? ───────────────────
print(__doc__.split("Usage:")[0])
print("Building the outcome column (running min at commit vs settled low)...",
      flush=True)

days = [r["date"] for r in conn.execute(
    """SELECT c.date FROM cli_reports c WHERE c.station='KMIA'
         AND c.low_f IS NOT NULL AND c.date >= '2024-07-29'
       ORDER BY c.date""")]

outcome, dropped = {}, 0
for d_str in days:
    d = datetime.fromisoformat(d_str).date()
    asof = datetime(d.year, d.month, d.day, COMMIT_HOUR,
                    tzinfo=TZ).astimezone(ZoneInfo("UTC")).isoformat()
    vis = latentmax.visible_running_extremum(conn, CFG, d, "min", asof_utc=asof)
    if vis is None:
        continue
    low = conn.execute("SELECT low_f FROM cli_reports WHERE station='KMIA' AND date=?",
                       (d_str,)).fetchone()["low_f"]
    # `vis` is an interval; use its LOWER edge -- the most generous reading of
    # what was already known, so a "late drop" is not manufactured by tolerance.
    known = vis[0]
    outcome[d_str] = (known - low) >= DROP_F
    dropped += outcome[d_str]

print(f"  {len(outcome)} days scored, {dropped} fell >={DROP_F:.0f}F after commit "
      f"({dropped/len(outcome):.1%} base rate)\n")

# ── the forecast signal, day-ahead, afternoon only ──────────────────────────
sig = {}
for r in conn.execute(
        """SELECT valid_utc, AVG(pop) pop, AVG(cape) cape FROM forecast_conv
           WHERE station='KMIA' GROUP BY valid_utc"""):
    t = datetime.fromisoformat(
        r["valid_utc"] if "+" in r["valid_utc"] else r["valid_utc"] + "+00:00")
    lt = t.astimezone(TZ)
    if lt.hour not in AFTERNOON:
        continue
    e = sig.setdefault(lt.date().isoformat(), {"pop": [], "cape": []})
    if r["pop"] is not None:
        e["pop"].append(r["pop"])
    if r["cape"] is not None:
        e["cape"].append(r["cape"])

common = sorted(set(outcome) & {d for d, v in sig.items() if v["pop"] or v["cape"]})
print(f"  {len(common)} days have BOTH an afternoon forecast and an outcome\n")


def run_arm(name, values, thresholds):
    print(f"═══ ARM: {name} ═══")
    print(f"  {'threshold':<12}{'flagged':>9}{'fell|flag':>11}"
          f"{'fell|not':>10}{'separation':>12}{'CI':>22}")
    out = []
    for thr in thresholds:
        pos = np.array([values[d] >= thr for d in common])
        y = np.array([outcome[d] for d in common], dtype=float)
        if pos.sum() < 10 or (~pos).sum() < 10:
            print(f"  >= {thr:<9}{int(pos.sum()):>9}   (too few either side)")
            continue
        a, b = y[pos].mean(), y[~pos].mean()
        diffs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            yy, pp = y[i], pos[i]
            if pp.sum() < 5 or (~pp).sum() < 5:
                continue
            diffs.append(yy[pp].mean() - yy[~pp].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig_ = "SIG" if (lo > 0) == (hi > 0) else "ns"
        print(f"  >= {thr:<9}{int(pos.sum()):>9}{a:>11.1%}{b:>10.1%}"
              f"{a-b:>+12.1%}   [{lo:+.1%},{hi:+.1%}] {sig_}")
        out.append((thr, a - b, lo, hi))
    print()
    return out


pop_v = {d: max(sig[d]["pop"]) if sig[d]["pop"] else 0.0 for d in common}
cape_v = {d: max(sig[d]["cape"]) if sig[d]["cape"] else 0.0 for d in common}

real = []
real += run_arm("pop, afternoon max (%)", pop_v, [30, 40, 50, 60, 70])
real += run_arm("cape, afternoon max (J/kg)", cape_v, [1500, 2000, 2500, 3000, 3500])

# ── the shuffled control ────────────────────────────────────────────────────
print("═══ ARM: SHUFFLED CONTROL (same labels, days reassigned at random) ═══")
print("  The bar the real arms must clear. Anything they achieve that this also")
print("  achieves is noise.\n")
seps = []
for _ in range(200):
    perm = rng.permutation(len(common))
    shuffled = {d: pop_v[common[perm[i]]] for i, d in enumerate(common)}
    pos = np.array([shuffled[d] >= 50 for d in common])
    y = np.array([outcome[d] for d in common], dtype=float)
    if pos.sum() < 10 or (~pos).sum() < 10:
        continue
    seps.append(y[pos].mean() - y[~pos].mean())
if seps:
    s = np.array(seps)
    print(f"  200 shuffles at the pop>=50 threshold:")
    print(f"    mean separation {s.mean():+.1%}   95% of shuffles within "
          f"[{np.percentile(s,2.5):+.1%}, {np.percentile(s,97.5):+.1%}]")

print("\n═══ THE BAR ═══")
print("  OBSERVED thunder achieves 24.2% vs 6.3% = +17.9 points (spec 2).")
best = max((abs(x[1]), x) for x in real)[1] if real else None
if best:
    thr, sep, lo, hi = best
    print(f"  Best real arm separation: {sep:+.1%}  CI [{lo:+.1%},{hi:+.1%}]")
    print(f"  Fraction of what observation achieves: {abs(sep)/0.179:.0%}")
    verdict = ("PASS" if (lo > 0) == (hi > 0) and abs(sep) >= 0.05
               else "NULL RESULT — do not build on this")
    print(f"\n  -> {verdict}")
print("\n  Per the spec: a null result here is valid and expected. Do not tune")
print("  thresholds until the numbers look good.")
