"""ARM C: full history, convection columns NaN before they exist.

Arms A/B (scripts/convection_eval.py) proved the features EARN their place on the
restricted range (+2.4 pts at the h11 commit hour) but that the restriction itself
costs 7.8 -- a net loss. This asks whether LightGBM's native NaN handling lets us
keep all 1,822 rows AND use convection on the ~450 where it exists.

If C beats the full-history baseline, ship it. If not, convection is a real signal
we cannot yet afford, and the answer is to wait for the archive to deepen.
"""
import numpy as np, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from highcast import gating
from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

cfg = load_config('config/kmia.yaml'); conn = connect(cfg.db_path)
tz = ZoneInfo(cfg.station.tz); spec = get_metric(cfg, 'high')
SPLIT, PEAK = '2026-01-01', (11, 17)
CONV = ["conv_pop_max", "conv_cape_max", "conv_qpf", "conv_pop_hour"]

per = {}
for r in conn.execute("SELECT valid_utc,pop,qpf,cape FROM forecast_conv WHERE station=?",
                      (cfg.station.icao,)):
    loc = datetime.fromisoformat(r['valid_utc']).replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)
    if not (PEAK[0] <= loc.hour <= PEAK[1]): continue
    b = per.setdefault(loc.date().isoformat(), {'p':[], 'q':[], 'c':[], 'ph':[]})
    if r['pop'] is not None: b['p'].append(r['pop']); b['ph'].append((r['pop'], loc.hour))
    if r['qpf'] is not None: b['q'].append(r['qpf'])
    if r['cape'] is not None: b['c'].append(r['cape'])
conv = pd.DataFrame([{'target_date': d,
    'conv_pop_max': max(v['p']) if v['p'] else np.nan,
    'conv_cape_max': max(v['c']) if v['c'] else np.nan,
    'conv_qpf': float(np.mean(v['q'])) if v['q'] else np.nan,
    'conv_pop_hour': max(v['ph'])[1] if v['ph'] else np.nan} for d, v in per.items()])

dates = [r['date'] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]
k = cfg.model.intraday_bandwidth_for('high')
print(f"{'h':>3}{'n':>6} | {'FULL mae':>9}{'hit':>7} | {'C mae':>8}{'hit':>7} | {'dMAE':>7}{'dHIT':>7}", flush=True)
res = []
for h in (9, 10, 11, 12, 13):
    trh = tuple(sorted({(h+d) % 24 for d in range(-k, k+1)}))
    fr = build_checkpoint_frame(conn, cfg, dates, hours=trh, metric=spec)
    fr = fr[fr['y'].notna()].merge(conv, on='target_date', how='left')
    tr = fr[fr['target_date'] < SPLIT]
    te = fr[(fr['target_date'] >= SPLIT) & (fr['hour'] == float(h))]
    if len(te) < 40: continue
    y = te['y'].to_numpy(dtype=float); out = {}
    for arm, extra in (('FULL', []), ('C', CONV)):
        m = IntradayModel(extremum=spec.extremum, residual=False, extra_features=extra).fit(tr)
        qs = m.predict_quantiles(te); pmfs = [bucket_probs(q) for q in qs]
        out[arm] = (float(np.abs(np.array([q[0.5] for q in qs]) - y).mean()),
                    float(np.mean([crps_pmf(p, int(v)) for p, v in zip(pmfs, y)])),
                    float(np.mean([expected_bucket_hit(p, int(v)) for p, v in zip(pmfs, y)])))
    a, c = out['FULL'], out['C']; res.append((h, a, c))
    print(f"{h:>3}{len(te):>6} | {a[0]:>9.3f}{a[2]:>6.1%} | {c[0]:>8.3f}{c[2]:>6.1%} | "
          f"{c[0]-a[0]:>+7.3f}{(c[2]-a[2])*100:>+6.1f}", flush=True)
if res:
    print(f"\nMEAN dMAE {np.mean([r[2][0]-r[1][0] for r in res]):+.3f}  "
          f"dHIT {np.mean([r[2][2]-r[1][2] for r in res])*100:+.1f} pts")
    print("negative dMAE / positive dHIT = ship arm C")
