"""Score the LOW predictions the system ACTUALLY RECORDED LIVE, against the
settled CLI low. This is a forward-tested record, not a re-simulation: every
row was written by a wake cycle at the time, with only the data then available.

For each climate day we take the LAST intraday LOW prediction issued at or
before the personal cutoff instant (Manila 21:00 == 06:00 PDT), which is the
number the desk would have been showing at the decision hour.
"""
import json
import sqlite3
from datetime import date

from highcast.config import load_config
from highcast.db import connect
from highcast.gating import personal_cutoff_instant
from highcast.buckets import pmf_from_buckets, argmax_pair
from highcast.metrics import crps_pmf
from highcast.timeutil import round_half_up

cfg = load_config('config/klax.yaml')
conn = connect(cfg.db_path)

truth = {r["date"]: r["low_f"] for r in conn.execute(
    "SELECT date, low_f FROM cli_reports WHERE station=? AND date>='2026-07-01'",
    (cfg.station.icao,))}

rows = []
for ds in sorted(truth):
    y = truth[ds]
    if y is None:
        continue
    cut = personal_cutoff_instant(cfg, date.fromisoformat(ds), "low")
    if cut is None:
        continue
    r = conn.execute(
        """SELECT issued_at_utc, point_f, buckets FROM predictions
           WHERE station=? AND target_date=? AND kind='intraday' AND metric='low'
             AND issued_at_utc <= ?
           ORDER BY issued_at_utc DESC LIMIT 1""",
        (cfg.station.icao, ds, cut.isoformat())).fetchone()
    if r is None:
        rows.append(dict(date=ds, y=y, point=None, cut=cut.isoformat()))
        continue
    pmf = pmf_from_buckets(json.loads(r["buckets"]))
    pick = argmax_pair(pmf)
    rows.append(dict(date=ds, y=y, point=r["point_f"],
                     issued=r["issued_at_utc"], cut=cut.isoformat(),
                     crps=crps_pmf({int(k): v for k, v in pmf.items()}, int(y)),
                     pick=pick["label"] if pick else None,
                     pick_p=pick["prob"] if pick else None))


print(f"{'date':12}{'cli':>4}{'point':>8}{'err':>7}{'round':>7}{'crps':>7}  issued(UTC)")
scored = [r for r in rows if r.get("point") is not None]
for r in rows:
    if r.get("point") is None:
        print(f"{r['date']:12}{r['y']:>4}{'--':>8}{'':>7}{'':>7}{'':>7}  (no intraday pred at/before cutoff {r['cut']})")
        continue
    err = r["point"] - r["y"]
    rerr = round_half_up(r["point"]) - r["y"]
    print(f"{r['date']:12}{r['y']:>4}{r['point']:>8.2f}{err:>+7.2f}{rerr:>+7.0f}{r['crps']:>7.3f}  {r['issued']}")

if scored:
    import statistics as st
    mae = st.mean(abs(r["point"] - r["y"]) for r in scored)
    bias = st.mean(r["point"] - r["y"] for r in scored)
    crps = st.mean(r["crps"] for r in scored)
    exact = sum(round_half_up(r["point"]) == r["y"] for r in scored) / len(scored)
    within1 = sum(abs(round_half_up(r["point"]) - r["y"]) <= 1 for r in scored) / len(scored)
    print(f"\nN={len(scored)}  MAE={mae:.3f}  bias={bias:+.3f}  CRPS={crps:.3f}  "
          f"rounded-exact={exact*100:.0f}%  within1={within1*100:.0f}%")
    print(f"days with no usable prediction at the cutoff: {len(rows)-len(scored)}")
