"""Re-verify the headline claim using the SHIPPED code path (latentmax), not a
scratch reimplementation. Two arms over the full climate day:
  spot-only  -- what the ratchet saw before six_min_c existed
  shipped    -- visible_running_extremum as it now stands (6h groups + guard)
"""
import collections
import statistics as st
from datetime import date

from highcast.config import load_config
from highcast.db import connect
from highcast import latentmax
from highcast.timeutil import climate_day_bounds, round_half_up

cfg = load_config('config/klax.yaml')
conn = connect(cfg.db_path)

rows = []
for r in conn.execute(
        "SELECT date, low_f FROM cli_reports WHERE station=? AND low_f IS NOT NULL ORDER BY date",
        (cfg.station.icao,)):
    d = date.fromisoformat(r["date"])
    start, end = climate_day_bounds(d, cfg.station.std_utc_offset)
    obs = list(conn.execute(
        """SELECT tmpf_precise, tmpf FROM obs WHERE station=? AND source='metar'
           AND valid_utc>=? AND valid_utc<?""",
        (cfg.station.icao, start.isoformat(), end.isoformat())))
    if len(obs) < 20:
        continue
    vals = [(o["tmpf_precise"] if o["tmpf_precise"] is not None else o["tmpf"])
            for o in obs]
    vals = [v for v in vals if v is not None]
    if not vals:
        continue
    spot = min(vals)
    shipped = latentmax.visible_running_extremum(conn, cfg, d, "min")
    if shipped is None:
        continue
    rows.append(dict(date=r["date"], month=d.month, y=r["low_f"],
                     spot=spot, shipped=shipped[0]))

print(f"climate days: {len(rows)}")


def rep(name, sub):
    if not sub:
        return
    n = len(sub)
    print(f"\n{name} (N={n})")
    for label, key in (("spot-only (pre-fix)", "spot"), ("shipped (6h groups)", "shipped")):
        dd = collections.Counter(x["y"] - round_half_up(x[key]) for x in sub)
        exact = dd[0] / n * 100
        mae = st.mean(abs(x["y"] - round_half_up(x[key])) for x in sub)
        bias = st.mean(x["y"] - round_half_up(x[key]) for x in sub)
        print(f"  {label:22} exact {exact:5.1f}%  MAE {mae:.3f}  bias {bias:+.3f}")
    moved = sum(1 for x in sub
                if round_half_up(x["spot"]) != round_half_up(x["shipped"]))
    fixed = sum(1 for x in sub
                if round_half_up(x["spot"]) != x["y"] and round_half_up(x["shipped"]) == x["y"])
    broke = sum(1 for x in sub
                if round_half_up(x["spot"]) == x["y"] and round_half_up(x["shipped"]) != x["y"])
    print(f"  bound moved on {moved} day(s) ({moved/n*100:.1f}%): fixed {fixed}, broke {broke}")


rep("ALL", rows)
rep("JUN-AUG", [x for x in rows if x["month"] in (6, 7, 8)])
rep("2026", [x for x in rows if x["date"] >= "2026-01-01"])
