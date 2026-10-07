"""How much does the settled CLI low undercut the visible running min AS OF
02:00Z (19:00 PDT) — i.e. with ~6h of the climate day still to run?

That is exactly the position we are in for 2026-07-25 right now. Reported
overall and for July-only days (the marine-layer regime that actually applies).
"""
import collections
from datetime import date, datetime, timedelta, timezone

from highcast.config import load_config
from highcast.db import connect
from highcast import latentmax
from highcast.timeutil import climate_day_bounds, round_half_up

cfg = load_config('config/klax.yaml')
conn = connect(cfg.db_path)

ASOF_HOUR_Z = 2   # 02:00Z == 19:00 PDT == 18:00 LST

rows = []
for r in conn.execute(
        "SELECT date, low_f FROM cli_reports WHERE station=? AND low_f IS NOT NULL ORDER BY date",
        (cfg.station.icao,)):
    d = date.fromisoformat(r["date"])
    start, end = climate_day_bounds(d, cfg.station.std_utc_offset)
    # 02:00Z on the day AFTER the climate-day start date (start is 08:00Z on d)
    asof = (start + timedelta(hours=18)).isoformat()   # 08:00Z + 18h = 02:00Z next day
    vis = latentmax.visible_running_extremum(conn, cfg, d, "min", asof_utc=asof)
    full = latentmax.visible_running_extremum(conn, cfg, d, "min")
    if vis is None or full is None:
        continue
    n = conn.execute(
        """SELECT COUNT(*) c FROM obs WHERE station=? AND source='metar'
           AND valid_utc>=? AND valid_utc<?""",
        (cfg.station.icao, start.isoformat(), end.isoformat())).fetchone()["c"]
    if n < 20:
        continue
    rows.append(dict(date=r["date"], month=d.month, y=r["low_f"],
                     bound_asof=round_half_up(vis[0]),
                     bound_full=round_half_up(full[0])))


def report(name, sub):
    if not sub:
        print(f"{name}: no data")
        return
    delta = collections.Counter(x["y"] - x["bound_asof"] for x in sub)
    late = collections.Counter(x["bound_full"] - x["bound_asof"] for x in sub)
    n = len(sub)
    print(f"\n{name}  (N={n})")
    print("  settled CLI low  MINUS  rounded running min at 02:00Z:")
    for k in sorted(delta):
        if delta[k] / n >= 0.002:
            print(f"    {k:+d} F   {delta[k]*100/n:5.1f}%   ({delta[k]})")
    print(f"    -> P(no further undercut, delta==0) = {delta[0]*100/n:.1f}%")
    print(f"    -> P(delta <= -1)                   = {sum(v for k,v in delta.items() if k<=-1)*100/n:.1f}%")
    print("  of which purely OBSERVED late cooling (running min still fell after 02:00Z):")
    for k in sorted(late):
        if late[k] / n >= 0.005:
            print(f"    {k:+d} F   {late[k]*100/n:5.1f}%")


report("ALL MONTHS", rows)
report("JULY ONLY", [x for x in rows if x["month"] == 7])
report("JUN-AUG", [x for x in rows if x["month"] in (6, 7, 8)])

# The specific 2026-07-25 situation: bound_asof == 70
sub = [x for x in rows if x["month"] in (6, 7, 8) and x["bound_asof"] >= 68]
report("JUN-AUG, warm nights (bound at 02:00Z >= 68F)", sub)
