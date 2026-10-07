"""THE CROSS-KIND DIVERGENCE BUG — and why my first trace of it was wrong.

⚠️ READ THE RETRACTION BEFORE THE NUMBERS.

FIRST TRACE, 2026-08-16, WRONG. I claimed: `gating.kind_at_hour` serves the
day-ahead model for a HIGH before `MIN_DECISION_HOUR`, so a HIGH committed
before the knee stores a day-ahead `p_model` that later checks compare against
the intraday nowcast. From that I reported 7 affected settled bets, 2 spurious
alerts and 1 suppressed one, and named KMIA 2026-07-29 as a false "confidence
collapsed" sent to Julian.

ALL OF THAT IS RETRACTED. `MIN_DECISION_HOUR` was MOVED FROM 11 TO 13 ON
2026-07-31 (its own comment says so, and I did not read it). Every HIGH commit
before h13 in the record is dated 07-27 to 07-31 — i.e. at the time, h11 WAS
the knee, the desk served intraday at commit and intraday afterwards, and no
cross-kind comparison happened on any of them. I applied today's constant to
last month's data. Same class as error #1: an analysis that disagreed with the
record, reported before it was reconciled with it.

WHAT IS ACTUALLY TRUE. Ground truth is the `kind` on the `signals` row the desk
wrote at the commit. Against that, exactly ONE settled bet has ever spanned a
model switch:

    KLAX low 2026-08-09, committed h19, 66-67 at 45% from the DAY-AHEAD model
    -- because no intraday prediction for that target existed until 05:13Z.
    At 05:13Z the nowcast arrived with 66-67 at 2% and 68-69 on top. The alert
    fired "overtaken", the bet LOST, and the alert was RIGHT.

So the mechanism is AVAILABILITY, not the hour: `preferred_prediction` walks an
order and returns the first kind that HAS a stored prediction. That is why the
one real case is a LOW, which the hour story says is immune. And it is why the
fix reads the committing kind from the record rather than recomputing it.

⚠️ NOTHING HAS BEEN MIS-ALERTED BY THIS YET. The one real switch fired
"overtaken", which compares top against held inside ONE ladder and was never
cross-model. Only "collapsed" and the floor trigger subtract across models.
This is a latent-bug guard. Do not cite it as a fix for observed damage.

WHAT SECTION 2 BELOW STILL MEASURES, and it is still worth knowing: how far
apart the two models are at h13 on the same day, same ladder, same running-max
truncation. That is a real property of the pair. It is NOT evidence that alerts
fired, because no bet is baselined on the day-ahead model any more.

Usage: PYTHONPATH=src .venv/bin/python scripts/divergence_kind_switch.py
"""
import json
from datetime import datetime, timezone
from statistics import median
from zoneinfo import ZoneInfo

from highcast import buckets, gating, latentmax
from highcast.config import load_config
from highcast.db import connect
from highcast.notify import DIVERGENCE_DROP, divergence_reason

TZ = {"KLAX": "America/Los_Angeles", "KMIA": "America/New_York"}


def _ladder(conn, icao, target, cut_utc):
    """The real Kalshi ladder as the desk saw it nearest the knee. Taken from a
    stored `signals` row rather than rebuilt: parity is a coin flip per day and
    must never be assumed (project rule)."""
    r = conn.execute(
        """SELECT market_json FROM signals WHERE station=? AND metric='high'
             AND target_date=? AND market_json IS NOT NULL AND market_json!=''
             AND created_utc<=? ORDER BY created_utc DESC LIMIT 1""",
        (icao, target, cut_utc)).fetchone()
    if not r:
        return None
    lad = json.loads(r["market_json"])
    return [x for x in lad if "lo" in x or "hi" in x] or None


def rungs(conn, cfg, icao, target, kind, cut_utc, lad, bound):
    """Reproduce `live_decision`: per-degree pmf -> truncate at the running max
    -> sum onto the ladder. Both kinds get the SAME bound at the SAME instant,
    so the comparison isolates the model swap and nothing else."""
    r = conn.execute(
        """SELECT buckets FROM predictions WHERE station=? AND target_date=?
             AND kind=? AND metric='high' AND issued_at_utc<=?
           ORDER BY issued_at_utc DESC LIMIT 1""",
        (icao, target, kind, cut_utc)).fetchone()
    if not r:
        return None
    pm = buckets.pmf_from_buckets(json.loads(r["buckets"]))
    if pm is None:
        return None
    pm = buckets.truncate_at_bound(pm, bound, "max")
    out = {x["label"]: buckets.pmf_mass(pm, x.get("lo"), x.get("hi"))
           for x in lad}
    return {k: v for k, v in out.items() if v is not None}


def both_at_knee(conn, cfg, icao, target, tz, knee):
    """(dayahead_rungs, intraday_rungs, bound) as of knee:05 local."""
    d = datetime.fromisoformat(target)
    cut = datetime(d.year, d.month, d.day, knee, 5,
                   tzinfo=tz).astimezone(timezone.utc).isoformat()
    lad = _ladder(conn, icao, target, cut)
    if not lad:
        return None, None, None
    # ⚠️ The SAME running-max bound for both models. `live_decision` truncates
    # before summing onto the ladder, and the truncation is large for a HIGH at
    # h13; comparing an untruncated pmf against the desk's truncated one would
    # manufacture a difference that has nothing to do with the model swap.
    iv = latentmax.visible_running_extremum(
        conn, cfg, d.date(), "max", asof_utc=cut)
    bound = iv[0] if iv else None
    return (rungs(conn, cfg, icao, target, "dayahead", cut, lad, bound),
            rungs(conn, cfg, icao, target, "intraday", cut, lad, bound),
            bound)


print(__doc__.split("Usage:")[0])
gaps, affected, fired_spuriously = [], [], []
for icao in ("KLAX", "KMIA"):
    cfg = load_config(f"config/{icao.lower()}.yaml")
    conn = connect(cfg.db_path, migrate=False)
    tz = ZoneInfo(TZ[icao])
    knee = gating.MIN_DECISION_HOUR[icao]["high"]

    # ── 1. WHO IS AFFECTED: settled HIGH bets committed before the knee ──────
    for b in conn.execute(
            """SELECT target_date, bucket, p_model, placed_local_hour, won,
                      diverged_utc FROM app_bets
               WHERE station=? AND metric='high' AND won IS NOT NULL""",
            (icao,)):
        if b["placed_local_hour"] is None or b["placed_local_hour"] >= knee:
            continue
        da, it, _ = both_at_knee(conn, cfg, icao, b["target_date"], tz, knee)
        if not da or not it:
            continue
        held_da, held_it = da.get(b["bucket"]), it.get(b["bucket"])
        if held_da is None or held_it is None:
            continue
        top_it = max(it, key=lambda k: it[k] or 0)
        top_da = max(da, key=lambda k: da[k] or 0)
        # BUGGY: committed p (day-ahead) vs the intraday model -- what ships.
        reason = divergence_reason(b["p_model"], held_it, top_it, b["bucket"])
        # LIKE FOR LIKE: the committing model compared against ITSELF. This is
        # the comparison the alert claims to be making, and the difference
        # between the two columns is the bug's actual cost.
        fixed = divergence_reason(b["p_model"], held_da, top_da, b["bucket"])
        affected.append((icao, b["target_date"], b["placed_local_hour"],
                         b["p_model"], held_da, held_it, top_it, b["bucket"],
                         reason, b["diverged_utc"], b["won"], fixed, top_da))
        if reason:
            fired_spuriously.append((icao, b["target_date"], reason))

    # ── 2. HOW BIG IS THE SWITCH, over every HIGH day with both models ───────
    for row in conn.execute(
            """SELECT DISTINCT target_date FROM predictions
               WHERE station=? AND metric='high'""", (icao,)):
        t = row["target_date"]
        da, it, _ = both_at_knee(conn, cfg, icao, t, tz, knee)
        if not da or not it:
            continue
        mode = max(da, key=lambda k: da[k] or 0)          # day-ahead's own pick
        if da.get(mode) is None or it.get(mode) is None:
            continue
        gaps.append((icao, t, mode, da[mode], it[mode], da[mode] - it[mode],
                     max(it, key=lambda k: it[k] or 0) != mode))

print(f"\n{'='*78}\n1. HIGH BETS COMMITTED BEFORE TODAY'S KNEE\n"
      f"   ⚠️ RETRACTED AS EVIDENCE -- all of these predate the 2026-07-31 move\n"
      f"   of MIN_DECISION_HOUR from 11 to 13, so at the time they were AT the\n"
      f"   knee and were served intraday. Kept only to show the retraction.\n"
      f"{'='*78}")
if not affected:
    print("  none")
for (icao, t, h, cp, hda, hit, top, buck, reason, div, won,
     fixed, top_da) in sorted(affected):
    verdict = ("SPURIOUS -- the committing model never moved"
               if reason and not fixed else
               "MISSED -- suppressed by the wrong model" if fixed and not reason
               else "same verdict either way" if reason == fixed
               else f"reason differs: {reason} vs {fixed}")
    print(f"  {icao} {t}  committed h{h} {buck} @ {cp:.0%}  won={bool(won)}")
    print(f"      at knee, day-ahead (the committing model): "
          f"{hda:.0%}, top = {top_da}")
    print(f"      at knee, intraday  (what the alert read) : "
          f"{hit:.0%}, top = {top}   shift {hda-hit:+.0%}")
    print(f"      ships: {str(reason or 'silent'):<10} "
          f"like-for-like: {str(fixed or 'silent'):<10} -> {verdict}")
    print(f"      live alert: {div or 'never fired'}")

n_spur = sum(1 for a in sorted(affected) if a[8] and not a[11])
n_miss = sum(1 for a in sorted(affected) if a[11] and not a[8])
print(f"\n  {len(affected)} affected settled bets: {n_spur} SPURIOUS, "
      f"{n_miss} MISSED, {len(affected)-n_spur-n_miss} unchanged.")

print(f"\n{'='*78}\n2. SIZE OF THE MODEL SWITCH — |dayahead - intraday| at the knee\n"
      f"   on the DAY-AHEAD's own top bucket, every HIGH day with both stored\n"
      f"{'='*78}")
for icao in ("KLAX", "KMIA"):
    g = [x for x in gaps if x[0] == icao]
    if not g:
        continue
    d = [abs(x[5]) for x in g]
    over = sum(1 for x in d if x >= DIVERGENCE_DROP)
    flip = sum(1 for x in g if x[6])
    print(f"  {icao} high  n={len(g)}   median |shift| {median(d):.1%}   "
          f"p90 {sorted(d)[int(.9*len(d))-1]:.1%}   max {max(d):.1%}")
    print(f"      >= DIVERGENCE_DROP ({DIVERGENCE_DROP:.0%}): {over}/{len(g)} "
          f"= {over/len(g):.0%}  -> would trip 'collapsed' on model change alone")
    print(f"      intraday's top bucket differs from day-ahead's: "
          f"{flip}/{len(g)} = {flip/len(g):.0%}  -> would trip 'overtaken'")
    either = sum(1 for x in g if abs(x[5]) >= DIVERGENCE_DROP or x[6])
    print(f"      EITHER trigger: {either}/{len(g)} = {either/len(g):.0%}")

print("\n  Read this as the FALSE-ALARM FLOOR for a pre-knee HIGH commit: the rate")
print("  at which the alert fires purely because the desk swapped models, with")
print("  no new weather and no change of mind by either forecaster.")
