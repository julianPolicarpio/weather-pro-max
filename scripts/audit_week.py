"""THE POST-WEEK AUDIT. Everything needed to decide what to fix next.

Run this when a test week ends. It answers, in order of how much they matter:

  1. COVERAGE     did the app actually bet every day it should have? A 4-1 record
                  means something very different if it only bet on 5 of 7 days,
                  and a missing day is a laptop-asleep story, not a model story.
  2. RECORD       wins/losses per (station, metric), grouped by MODEL FINGERPRINT
                  so two different configurations are never averaged together.
  3. EDGE         model% minus market price at the commit hour. THE question
                  accuracy cannot answer: being right 57% of the time earns
                  nothing if the market was already at 99%.
  4. CALIBRATION  stated probability vs realised hit rate. >0 = overconfident,
                  which manufactures fake edges through the +5% gate.
  5. LOSS FORENSICS  for each loss: how far off, and what the intraday trajectory
                  did. This is the part that tells you WHAT to fix rather than
                  just that something is wrong.

WHY THE GROUPING BY FINGERPRINT MATTERS. Every model knob in this project has
been changed at least once mid-project. Pooling a week of `k0|lg0` rows with a
week of `k1|lg1` rows and reporting one hit rate is how you conclude nothing
works. `app_bets.model_fingerprint` is stamped at placement for exactly this.

Reads only. Writes nothing, changes nothing.

Usage:  .venv/bin/python scripts/audit_week.py [start_date] [end_date]
"""
import json
import sys
from datetime import datetime

from highcast import db, gating
from highcast.buckets import expected_bucket_hit, pmf_from_buckets
from highcast.config import load_config

START = sys.argv[1] if len(sys.argv) > 1 else "2026-07-27"
END = sys.argv[2] if len(sys.argv) > 2 else "2099-12-31"
conn = db.connect(load_config().db_path)
print(f"AUDIT  {START} .. {END}\n" + "=" * 78)

bets = conn.execute(
    """SELECT * FROM app_bets WHERE target_date BETWEEN ? AND ?
       ORDER BY station, metric, target_date""", (START, END)).fetchall()
if not bets:
    print("no bets in range"); raise SystemExit

# ---------- 1. COVERAGE ----------
print("\n[1] COVERAGE — committed vs expected")
for st in sorted({b["station"] for b in bets}):
    # CALENDAR days in range up to today -- NOT settled CLI days. Using
    # cli_reports as the denominator undercounts: today's report does not exist
    # yet, so a day the app correctly bet on looks like it was not expected, and
    # coverage reads >100%. The question is "did it bet every day it could have",
    # and that is a calendar question.
    from datetime import date as _d, timedelta as _td
    a = _d.fromisoformat(START)
    b_end = min(_d.fromisoformat(END), _d.today())
    days = [(a + _td(days=i)).isoformat() for i in range((b_end - a).days + 1)]
    for mk in ("low", "high"):
        got = {b["target_date"] for b in bets
               if b["station"] == st and b["metric"] == mk}
        missing = [d for d in days if d not in got]
        print(f"  {st} {mk:<5} {len(got)}/{len(days)} days"
              + (f"   MISSING: {', '.join(missing)}" if missing else "   complete"))

# ---------- 2. RECORD, split by model fingerprint ----------
print("\n[2] RECORD — grouped by model fingerprint (never averaged across configs)")
groups: dict = {}
for b in bets:
    groups.setdefault(
        (b["station"], b["metric"], b["model_fingerprint"] or "unstamped"), []
    ).append(b)
for (st, mk, fp), rows in sorted(groups.items()):
    done = [r for r in rows if r["won"] is not None]
    w = sum(r["won"] for r in done)
    pend = len(rows) - len(done)
    rate = f"{w/len(done):.0%}" if done else "—"
    print(f"  {st} {mk:<5} [{fp}]  {w}-{len(done)-w} ({rate})"
          + (f"  +{pend} pending" if pend else ""))

# ---------- 3. EDGE at the commit hour ----------
print("\n[3] EDGE at commit — model% minus market price")
print("    (positive = the app thought it was underpriced; this is the money question)")
for b in bets:
    if b["p_model"] is None or b["price"] is None:
        continue
    edge = b["p_model"] - b["price"]
    res = "—" if b["won"] is None else ("WON " if b["won"] else "LOST")
    print(f"  {b['station']} {b['metric']:<5} {b['target_date']}  {b['bucket']:<7} "
          f"model {b['p_model']:>5.1%}  price {b['price']:>5.2f}  "
          f"edge {edge:>+6.1%}  {res}")

# ---------- 4. CALIBRATION ----------
print("\n[4] CALIBRATION — stated vs realised (>0 = overconfident)")
for (st, mk, fp), rows in sorted(groups.items()):
    done = [r for r in rows if r["won"] is not None and r["p_model"] is not None]
    if not done:
        continue
    stated = sum(r["p_model"] for r in done) / len(done)
    real = sum(r["won"] for r in done) / len(done)
    # n is tiny over one week, so say so rather than letting the number look solid.
    print(f"  {st} {mk:<5} [{fp}]  stated {stated:.1%}  realised {real:.1%}  "
          f"gap {stated-real:+.1%}   (n={len(done)} — inside noise below ~30)")

# ---------- 5. LOSS FORENSICS ----------
print("\n[5] LOSS FORENSICS — what to actually fix")
for b in bets:
    if b["won"] != 0 or b["actual"] is None:
        continue
    lo = int(str(b["bucket"]).split("–")[0].split("-")[0])
    off = b["actual"] - lo if b["actual"] > lo + 1 else b["actual"] - (lo + 1)
    # the intraday trajectory: did the model ever hold the right bucket?
    traj, ever_right = [], False
    for p in conn.execute(
            """SELECT issued_at_utc, buckets FROM predictions
               WHERE station=? AND metric=? AND target_date=? AND buckets IS NOT NULL
               ORDER BY issued_at_utc""",
            (b["station"], b["metric"], b["target_date"])):
        pmf = pmf_from_buckets(json.loads(p["buckets"]))
        agg: dict = {}
        for k, v in pmf.items():
            e = k - (k % 2)
            agg[e] = agg.get(e, 0.0) + v
        top = max(agg, key=agg.get)
        lab = f"{top}-{top+1}"
        if not traj or traj[-1][1] != lab:
            traj.append((datetime.fromisoformat(p["issued_at_utc"]), lab))
        if top <= b["actual"] <= top + 1:
            ever_right = True
    hit = gating.hour_accuracy(b["station"], b["metric"], b["placed_local_hour"])
    fut = gating.extremum_still_future(b["station"], b["metric"],
                                       b["placed_local_hour"])
    print(f"\n  {b['station']} {b['metric'].upper()} {b['target_date']}: "
          f"bet {b['bucket']}, actual {b['actual']} (off by {off:+d}F)")
    print(f"    committed h{b['placed_local_hour']} — that hour is right "
          f"{hit['hit']:.0%} of the time, {fut:.0%} still ahead"
          if hit and fut is not None else "    (hour unmeasured)")
    print(f"    model held the correct bucket at some point today: "
          f"{'YES — a timing/lock problem' if ever_right else 'NO — a model problem'}")
    print(f"    trajectory: {' -> '.join(l for _, l in traj[:8])}"
          + (" ..." if len(traj) > 8 else ""))
    print(f"    reason at commit: {(b['reason'] or '')[:110]}")

print("\n" + "=" * 78)
print("The 'model held the correct bucket' line is the key triage: YES means the")
print("information was there and the commit hour or the lock lost it; NO means the")
print("model never knew, which is a features/data problem. Those need opposite fixes.")
