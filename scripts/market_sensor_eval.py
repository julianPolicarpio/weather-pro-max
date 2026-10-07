"""HOW EARLY DOES THE MARKET KNOW, versus when can WE see it? Measure first.

WHY. On 2026-08-14 KMIA's high peaked between hourly filings. The market crashed
93–94 from 0.74 to 0.01 at 18:07Z; our six-hour remark group did not file until
23:53Z — **5h46m later**. `signals.market_json` already stores the full ladder
every ~5 minutes, so that information was sitting in our own database, unread.

⚠️ THIS IS NOT A TRADING SIGNAL AND CANNOT BE ONE. By construction it is the
counterparty's own opinion arriving after they have already priced it. Reading
it can make us CONTEMPORANEOUS with the market, never ahead of it. Its value is
monitoring: grading, settlement alerts, and knowing in real time that we are
blind. Anyone tempted to feed it to the model should read QUEUE item 1 first —
the market already beats us at 78 of 81 hours, so a feature that copies it just
launders the market's view into ours.

WHAT IS MEASURED, per priced+settled day-market:

  t_market : first snapshot where the market's normalised top bucket >= CONF
             AND that bucket is the one that actually settled
  t_ours   : first moment `latentmax.visible_running_extremum` can see the
             settled value -- i.e. our own ratchet reaches it

  lead = t_ours - t_market       (positive = the market knew first)

Also reports how often a confident market is WRONG, because a sensor that is
early and wrong is worse than no sensor.

Usage: PYTHONPATH=src .venv/bin/python scripts/market_sensor_eval.py
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from highcast import latentmax
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds, round_half_up

CONF = 0.90          # "the market has decided"
STEP_MIN = 15        # resolution for the ratchet scan


def settled_rung(ladder, y):
    for i, x in enumerate(ladder):
        if ((x.get("lo") is None or y >= x["lo"])
                and (x.get("hi") is None or y <= x["hi"])):
            return i
    return None


print(__doc__.split("Usage:")[0])
rows = []
for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    cfg = load_config(cfgp)
    icao = cfg.station.icao
    conn = connect(cfg.db_path, migrate=False)
    for mkey in ("high", "low"):
        spec = get_metric(cfg, mkey)
        outcomes = {r["date"]: r[spec.target_col] for r in conn.execute(
            f"SELECT date,{spec.target_col} FROM cli_reports WHERE station=? "
            f"AND {spec.target_col} IS NOT NULL", (icao,))}
        snaps = {}
        for r in conn.execute(
                """SELECT target_date, created_utc, market_json FROM signals
                   WHERE station=? AND metric=? AND market_json IS NOT NULL
                     AND market_json!='' ORDER BY created_utc""", (icao, mkey)):
            snaps.setdefault(r["target_date"], []).append(r)

        for d, rs in sorted(snaps.items()):
            if d not in outcomes:
                continue
            y = int(outcomes[d])
            # ── when did the MARKET decide? ─────────────────────────────────
            t_mkt, mkt_right = None, None
            for r in rs:
                try:
                    lad = json.loads(r["market_json"])
                except Exception:
                    continue
                asks = np.array([x.get("yes_ask") if x.get("yes_ask") is not None
                                 else np.nan for x in lad], dtype=float)
                if np.isnan(asks).any() or asks.sum() <= 0:
                    continue
                p = asks / asks.sum()
                top = int(np.argmax(p))
                if p[top] < CONF:
                    continue
                k = settled_rung(lad, y)
                t_mkt = datetime.fromisoformat(r["created_utc"])
                mkt_right = (k is not None and top == k)
                break
            if t_mkt is None:
                continue
            # ── when could WE first see it? ─────────────────────────────────
            start, stop = climate_day_bounds(
                datetime.fromisoformat(d).date(), cfg.station.std_utc_offset)
            t_ours, t = None, start
            while t <= stop:
                vis = latentmax.visible_running_extremum(
                    conn, cfg, datetime.fromisoformat(d).date(), spec.extremum,
                    asof_utc=t.isoformat())
                if vis is not None:
                    bound = vis[0] if spec.extremum == "max" else vis[1]
                    if round_half_up(bound) == y:
                        t_ours = t
                        break
                t += timedelta(minutes=STEP_MIN)
            rows.append({"station": icao, "metric": mkey, "date": d,
                         "t_mkt": t_mkt, "t_ours": t_ours,
                         "mkt_right": mkt_right})

print(f"\n{'='*76}\nMARKET CONFIDENCE (>= {CONF:.0%}) vs OUR RATCHET — "
      f"n={len(rows)} day-markets\n{'='*76}")
right = [r for r in rows if r["mkt_right"]]
print(f"  when the market reaches {CONF:.0%} it is RIGHT on "
      f"{len(right)}/{len(rows)} = {len(right)/max(len(rows),1):.0%} of days")
wrong = [r for r in rows if not r["mkt_right"]]
if wrong:
    print(f"  ⚠️ WRONG on {len(wrong)}: "
          + ", ".join(f"{w['station']} {w['metric']} {w['date']}"
                      for w in wrong[:6]))

leads = [( (r["t_ours"] - r["t_mkt"]).total_seconds()/3600 )
         for r in right if r["t_ours"] is not None]
never = [r for r in right if r["t_ours"] is None]
if leads:
    a = np.array(leads)
    print(f"\n  LEAD TIME (hours the market knew before our ratchet could):")
    print(f"    n={len(a)}   median {np.median(a):+.1f}h   mean {a.mean():+.1f}h")
    print(f"    market first on {(a > 0).sum()}/{len(a)} days "
          f"({(a > 0).mean():.0%})")
    for q in (10, 25, 50, 75, 90):
        print(f"      p{q:<3} {np.percentile(a, q):+.1f}h")
print(f"  our ratchet NEVER reached the settled value on {len(never)} days "
      f"(the blindspot, outright)")

print(f"\n{'='*76}\nPER MARKET\n{'='*76}")
for st in ("KLAX", "KMIA"):
    for mk in ("high", "low"):
        sel = [r for r in right if r["station"] == st and r["metric"] == mk
               and r["t_ours"] is not None]
        if not sel:
            continue
        a = np.array([(r["t_ours"] - r["t_mkt"]).total_seconds()/3600
                      for r in sel])
        nv = len([r for r in right if r["station"] == st and r["metric"] == mk
                  and r["t_ours"] is None])
        print(f"  {st} {mk:<5} n={len(a):<3} median lead {np.median(a):+5.1f}h   "
              f"market first {(a > 0).mean():.0%}   never-seen days {nv}")
print("\n  A POSITIVE lead is the market knowing before our own observations can.")
print("  That is the blindspot, measured in hours, from data we already store.")
