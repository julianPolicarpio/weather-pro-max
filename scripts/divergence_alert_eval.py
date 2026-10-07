"""IS THE "MODEL MOVED" ALERT WORTH ITS NOISE? Measure before changing it.

WHY, 2026-08-16. Julian asked why the model "moves to the wrong bucket". On KMIA
HIGH 2026-08-15 the alert fired at 20:22Z saying the model now favoured 94-95 at
64%. The committed 92-93 was RIGHT (settled 93), and the market had 92-93 at 96c
at that moment. Tracing the day, the model's top bucket flipped between 92-93 and
94-95 EIGHT times while its p50 only ranged 92.4-94.4 -- 74% of the mass sat on a
single degree (93) hard against the 93/94 ladder boundary, so tiny refits move a
large block across the line.

⚠️ FREQUENCY IS NOT THE TEST. An alert that fires often is fine if it predicts
something. The question is whether firing is associated with the committed bet
LOSING. If it is not, the alert is noise at any rate, and tuning its threshold is
the fifth-idea-of-that-shape mistake this project keeps making.

WHAT IS MEASURED:

  CURRENT-LIVE   ground truth, not a replay -- `app_bets.diverged_utc` is
                 stamped by `notify.notify_divergence` when it actually fired.
  CURRENT-REPLAY the same rule re-run over stored `signals` snapshots. Present
                 only to show the replay reproduces the live rate; if these two
                 disagree badly the replay below cannot be trusted either.
  PROPOSED       the replay plus two conditions:
                   (a) PERSISTENCE -- the trigger holds for >= N consecutive
                       snapshots (kills the single-refit spike)
                   (b) MARKET NOT DECIDED -- top `market_price` < CONC at the
                       firing snapshot (say nothing once Kalshi is at 96c)

Every rule is scored the same way: P(bet lost | alert fired) against
P(bet lost | silent), with Fisher's exact two-sided p.

⚠️ N and CONC are SWEPT AND REPORTED, not chosen. Picking the best cell of a
sweep on ~68 day-markets is exactly the overfitting QUEUE item 1 forbids.

Usage: PYTHONPATH=src .venv/bin/python scripts/divergence_alert_eval.py
"""
import json
from collections import defaultdict
from math import comb

from highcast import notify
from highcast.config import load_config
from highcast.db import connect

PERSIST = (1, 2, 3, 4, 6)
CONC = (1.01, 0.95, 0.90, 0.85, 0.75)   # 1.01 = no market condition at all


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]. Small tables only."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)
    obs = p(a)
    lo = max(0, c1 - (n - r1))
    return sum(p(x) for x in range(lo, min(r1, c1) + 1)
               if p(x) <= obs * (1 + 1e-9))


def load(icao):
    cfg = load_config("config/kmia.yaml" if icao == "KMIA" else "config/klax.yaml")
    conn = connect(cfg.db_path, migrate=False)
    bets = {(icao, r["metric"], r["target_date"]): dict(r) for r in conn.execute(
        """SELECT metric, target_date, bucket, p_model, won, placed_utc,
                  diverged_utc FROM app_bets
           WHERE station=? AND won IS NOT NULL AND placed_utc IS NOT NULL""",
        (icao,))}
    snaps = defaultdict(list)
    for r in conn.execute(
            """SELECT metric, target_date, created_utc, kind, within_window,
                      market_json FROM signals
               WHERE station=? AND kind IN ('dayahead','intraday')
                 AND market_json IS NOT NULL AND market_json != ''
               ORDER BY created_utc""", (icao,)):
        # ⚠️ TAKE THE ROWS AS THE DESK WROTE THEM. An earlier version RECON-
        # STRUCTED the kind ("day-ahead for a HIGH inside its window") and threw
        # away rows that disagreed. That reconstruction is wrong twice over:
        # `preferred_prediction` returns whichever kind actually HAS a stored
        # prediction, and `MIN_DECISION_HOUR` moved 11 -> 13 on 2026-07-31, so
        # the same hour means different things across the sample. Error #16.
        pass
        try:
            lad = json.loads(r["market_json"])
        except Exception:
            continue
        if lad:
            snaps[(icao, r["metric"], r["target_date"])].append(
                (r["created_utc"], lad))
    return bets, snaps


def fires(lad_seq, bet, persist, conc):
    """First snapshot at/after the commit where the rule trips. None = silent."""
    run = 0
    for when, lad in lad_seq:
        if when < bet["placed_utc"]:
            continue                      # a bet cannot diverge before it exists
        priced = [x for x in lad if x.get("p_model") is not None]
        if not priced:
            run = 0
            continue
        top = max(priced, key=lambda x: x["p_model"])
        held = next((x for x in priced if x["label"] == bet["bucket"]), None)
        if notify.divergence_reason(bet.get("p_model"),
                                    (held or {}).get("p_model"),
                                    top["label"], bet["bucket"]) is None:
            run = 0
            continue
        run += 1
        if run < persist:
            continue
        asks = [x.get("market_price") for x in lad
                if x.get("market_price") is not None]
        if asks and max(asks) >= conc:
            continue                      # market already decided -- say nothing
        return when
    return None


def score(label, fired_keys, days, bets):
    f = [k for k in days if k in fired_keys]
    q = [k for k in days if k not in fired_keys]
    fl = sum(1 for k in f if not bets[k]["won"])
    ql = sum(1 for k in q if not bets[k]["won"])
    if not f:
        return f"  {label:<34}      0   (never fires)"
    pf, pq = fl / len(f), (ql / len(q) if q else float("nan"))
    p = fisher(fl, len(f) - fl, ql, len(q) - ql) if q else float("nan")
    return (f"  {label:<34}{len(f):>7}{fl:>10}{len(f)-fl:>11}"
            f"{pf:>13.0%}{pq:>14.0%}{pf-pq:>+9.0%}{p:>9.2f}")


HDR = (f"  {'rule':<34}{'fires':>7}{'losers':>10}{'winners':>11}"
       f"{'P(L|fire)':>13}{'P(L|quiet)':>14}{'diff':>9}{'p':>9}")

print(__doc__.split("Usage:")[0])
bets, snaps = {}, {}
for icao in ("KLAX", "KMIA"):
    b, s = load(icao)
    bets.update(b)
    snaps.update(s)

days = sorted(k for k in bets if k in snaps)
nlost = sum(1 for k in days if not bets[k]["won"])
print(f"\n  {len(days)} settled day-markets with post-commit snapshots; "
      f"{nlost} LOST ({nlost/len(days):.0%} base rate)")

print(f"\n{'='*97}\nDOES FIRING PREDICT LOSING?\n{'='*97}\n{HDR}")
live = {k for k in days if bets[k]["diverged_utc"]}
print(score("CURRENT (live, diverged_utc)", live, days, bets))
print(score("CURRENT (replayed)",
            {k for k in days if fires(snaps[k], bets[k], 1, 1.01)}, days, bets))
print()
for p in PERSIST:
    for cc in CONC:
        if p == 1 and cc > 1:
            continue                      # that IS the current rule, above
        keys = {k for k in days if fires(snaps[k], bets[k], p, cc)}
        tag = f"persist>={p}, price<{cc:.2f}" if cc <= 1 else f"persist>={p}"
        print(score(tag, keys, days, bets))
    print()

print("  P(L|fire) vs P(L|quiet) is the whole question. Equal columns mean the")
print("  alert carries no information about the outcome, and no threshold on it")
print("  will -- the honest fix would then be to DELETE it, not tune it.")


# ══════════════════════════════════════════════════════════════════════════════
# ADDENDUM, added after the sweep above inverted the hypothesis.
#
# The market-concentration filter did not remove noise, it removed SIGNAL: going
# from no filter to price<0.95 dropped 5 losing-day alerts and 0 winning-day
# alerts. The obvious reading is that on a losing day the market has often
# already concentrated -- on the bucket that is about to settle, which is not
# ours. "Kalshi is at 96c" is therefore the reason to SHOUT, not to stay quiet,
# provided the 96c is on somebody else's bucket.
#
# Two things tested here:
#   1. DIRECTIONAL concentration -- market >= CONC on a bucket != ours.
#   2. Whether firing survives conditioning on the committed probability. Four
#      ideas in this project have died as restatements of `stated`; if the alert
#      is one more, it is not worth a rule change either way.
# ══════════════════════════════════════════════════════════════════════════════

def fires_directional(lad_seq, bet, conc):
    """Market has concentrated at/above `conc` on a bucket that is NOT ours."""
    for when, lad in lad_seq:
        if when < bet["placed_utc"]:
            continue
        against = [x for x in lad if x.get("market_price") is not None
                   and x["label"] != bet["bucket"] and x["market_price"] >= conc]
        if against:
            return when
    return None


print(f"\n{'='*97}\nADDENDUM 1 -- concentration AGAINST us, no model condition\n{'='*97}\n{HDR}")
for cc in (0.75, 0.85, 0.90, 0.95):
    keys = {k for k in days if fires_directional(snaps[k], bets[k], cc)}
    print(score(f"market >= {cc:.2f} on another bucket", keys, days, bets))
print()
for cc in (0.85, 0.90, 0.95):
    keys = {k for k in days
            if fires(snaps[k], bets[k], 1, 1.01)
            and fires_directional(snaps[k], bets[k], cc)}
    print(score(f"diverged AND market >= {cc:.2f} vs us", keys, days, bets))

print(f"\n{'='*97}\nADDENDUM 2 -- does it survive the committed probability?\n{'='*97}")
ps = sorted(bets[k]["p_model"] for k in days if bets[k]["p_model"] is not None)
cut = ps[len(ps) // 2]
print(f"  split at the median committed p_model = {cut:.0%}\n")
for name, sel in (("p_model BELOW median", lambda k: bets[k]["p_model"] < cut),
                  ("p_model AT/ABOVE median", lambda k: bets[k]["p_model"] >= cut)):
    sub = [k for k in days if bets[k]["p_model"] is not None and sel(k)]
    nl = sum(1 for k in sub if not bets[k]["won"])
    print(f"  {name}: n={len(sub)}, {nl} lost ({nl/len(sub):.0%})")
    print(HDR)
    print(score("  live diverged_utc",
                {k for k in sub if bets[k]["diverged_utc"]}, sub, bets))
    print(score("  replayed divergence",
                {k for k in sub if fires(snaps[k], bets[k], 1, 1.01)}, sub, bets))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ADDENDUM 3 -- IS IT AN ALERT OR A POST-MORTEM?
#
# "Market >= 0.95 on a bucket that is not ours" caught 11 of 19 losing days with
# ZERO false positives. That is too clean to take at face value: a market that
# has concentrated at 95c has usually concentrated BECAUSE the outcome is nearly
# determined, in which case the alert is a death certificate, not a warning.
#
# The test is what our own bucket is still worth when it fires. If our contract
# is already at 3c there is nothing left to salvage and the alert is a
# post-mortem regardless of how well it separates. Reported: hours from the
# commit to the fire, hours of the climate day still left, and our bucket's
# price at that moment.
# ══════════════════════════════════════════════════════════════════════════════
from datetime import datetime, timedelta      # noqa: E402
from statistics import median                 # noqa: E402
from zoneinfo import ZoneInfo                 # noqa: E402

_TZ = {"KLAX": "America/Los_Angeles", "KMIA": "America/New_York"}
_STD = {"KLAX": 8, "KMIA": 5}                 # climate day ends at this UTC hour


def detail(lad_seq, bet, conc):
    for when, lad in lad_seq:
        if when < bet["placed_utc"]:
            continue
        against = [x for x in lad if x.get("market_price") is not None
                   and x["label"] != bet["bucket"] and x["market_price"] >= conc]
        if not against:
            continue
        held = next((x for x in lad if x["label"] == bet["bucket"]), None)
        return when, (held or {}).get("market_price")
    return None, None


for cc in (0.75, 0.90, 0.95):
    rows = []
    for k in days:
        icao, metric, target = k
        when, ours = detail(snaps[k], bets[k], cc)
        if when is None:
            continue
        t = datetime.fromisoformat(when)
        end = (datetime.fromisoformat(target + "T00:00:00+00:00")
               + timedelta(days=1, hours=_STD[icao]))
        rows.append((t - datetime.fromisoformat(bets[k]["placed_utc"]),
                     end - t, ours, bets[k]["won"], k))
    if not rows:
        continue
    print(f"\n  market >= {cc:.2f} against us -- n={len(rows)}, "
          f"{sum(1 for r in rows if not r[3])} lost")
    print(f"    median lag after commit : "
          f"{median(r[0].total_seconds() for r in rows)/3600:+.1f} h")
    print(f"    median climate-day left : "
          f"{median(r[1].total_seconds() for r in rows)/3600:.1f} h")
    priced = [r[2] for r in rows if r[2] is not None]
    if priced:
        print(f"    our bucket's price then  : median {median(priced):.2f}, "
              f"min {min(priced):.2f}, max {max(priced):.2f}, "
              f"{sum(1 for p in priced if p <= 0.10)}/{len(priced)} already <=10c")

print("\n  A median price near zero means the position was already gone when the")
print("  alert fired. Separation without lead time buys nothing.")


# ══════════════════════════════════════════════════════════════════════════════
# ADDENDUM 4 -- IS THERE A WARNING THAT ARRIVES WHILE THE POSITION IS STILL WORTH
# SOMETHING? Require our bucket to be salvageable (>= SALV) at the fire.
# ══════════════════════════════════════════════════════════════════════════════
def fires_salvageable(lad_seq, bet, conc, salv):
    for when, lad in lad_seq:
        if when < bet["placed_utc"]:
            continue
        held = next((x for x in lad if x["label"] == bet["bucket"]), None)
        if held is None or (held.get("market_price") or 0) < salv:
            continue
        if any(x.get("market_price") is not None and x["label"] != bet["bucket"]
               and x["market_price"] >= conc for x in lad):
            return when
    return None


print(f"\n{'='*97}\nADDENDUM 4 -- only alerts that arrive while we can still exit\n{'='*97}\n{HDR}")
for salv in (0.20, 0.30):
    for cc in (0.50, 0.60, 0.70):
        keys = {k for k in days if fires_salvageable(snaps[k], bets[k], cc, salv)}
        print(score(f"market >= {cc:.2f} vs us, ours >= {salv:.2f}",
                    keys, days, bets))
    print()

# The day that prompted all of this.
k = ("KMIA", "high", "2026-08-15")
if k in bets:
    b = bets[k]
    print(f"  KMIA high 2026-08-15 -- committed {b['bucket']} @ {b['p_model']:.0%}, "
          f"won={b['won']}, live alert at {b['diverged_utc']}")
    print(f"    current rule replayed : {fires(snaps[k], b, 1, 1.01)}")
    print(f"    persist>=3            : {fires(snaps[k], b, 3, 1.01)}")
    print(f"    market>=0.90 vs us    : {fires_directional(snaps[k], b, 0.90)}")
else:
    print("  KMIA high 2026-08-15 is not in the settled set yet.")
