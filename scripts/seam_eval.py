"""Is the KLAX HIGH nowcast biased CONDITIONAL ON ITS OWN OUTPUT?

Julian's observation (2026-07-30): the HIGH "overestimates". Live evidence at the
commit hour, n=8 settled days: mean bias +0.61F overall, but split by the MODEL'S
OWN reading it was +1.91F over on 5/5 days where the model printed >=80, and
-0.79F (1/4 over) below 80. n=5 in the loaded bin -- this script is the powered
version.

WHY BIN BY MODEL OUTPUT, NOT BY SETTLED BAND. HANDOFF reports bias -0.94 in the
80-84 *settled* band, i.e. the model UNDER-calls hot days. Both can be true:
binning on the outcome selects on the dependent variable and manufactures the
sign (if truth was high, the model probably under-called it). Only the model's
own reading is known at 02:00 Manila, so only that cut is actionable.

REGIME: identical to scripts/hour_curve.py -- matched-hour training at the
configured bandwidth, IntradayModel with the configured residual flag, test at
h11 alone. Difference: WALK-FORWARD (expanding window, periodic refit) instead
of a single 2025-01-01 holdout, so every fold predicts a day it never trained on
and n is ~3x larger.

KNOWN CAVEAT (HANDOFF open item 4): build_checkpoint_frame fits its BaselineBlend
on the whole frame, so `base_p50`/`base_spread` carry a mild global leak. It is
the same leak hour_curve has (which is why these numbers stay comparable to
HOUR_ACCURACY), and for a bias measurement it should pull the estimate TOWARD
zero -- so a seam that survives here is understated, not inflated.
"""
import sys

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

HOUR = int(sys.argv[1]) if len(sys.argv) > 1 else 11
METRIC = sys.argv[2] if len(sys.argv) > 2 else "high"
REFIT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
CONFIG = sys.argv[4] if len(sys.argv) > 4 else "config/klax.yaml"
MIN_TRAIN = 400          # same floor as cli.MIN_MATCHED_ROWS
N_BOOT = 2000
rng = np.random.default_rng(20260730)

cfg = load_config(CONFIG)
conn = connect(cfg.db_path)
spec = get_metric(cfg, METRIC)
resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
k = cfg.model.intraday_bandwidth_for(METRIC)
tr_hours = tuple(sorted({(HOUR + d) % 24 for d in range(-k, k + 1)}))

dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
    (cfg.station.icao,))]
print(f"station {cfg.station.icao}  metric {METRIC}  hour h{HOUR}  residual={resid}  "
      f"bandwidth k={k} -> train hours {tr_hours}  refit every {REFIT}d  "
      f"n_cli={len(dates)}", flush=True)

fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec)
te_all = fr[(fr["hour"] == float(HOUR)) & fr["y"].notna()].sort_values("target_date")
print(f"frame {len(fr)} rows; testable at h{HOUR}: {len(te_all)}", flush=True)

td = te_all["target_date"].to_numpy()
recs = []          # (date, model_point, truth, pmf)
start = None
for i, d in enumerate(td):                       # find first date with enough history
    if (fr["target_date"] < d).sum() >= MIN_TRAIN and fr[fr["target_date"] < d]["y"].notna().sum() >= MIN_TRAIN:
        start = i
        break
if start is None:
    sys.exit("never enough training rows")
print(f"walk-forward begins at {td[start]} (index {start})", flush=True)

blocks = list(range(start, len(td), REFIT))
for bi, b0 in enumerate(blocks):
    cut = td[b0]
    tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
    if len(tr) < MIN_TRAIN:
        continue
    model = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
    blk = te_all.iloc[b0:b0 + REFIT]
    qs = model.predict_quantiles(blk)
    for (_, row), q in zip(blk.iterrows(), qs):
        recs.append((row["target_date"], float(q[0.5]), float(row["y"]),
                     bucket_probs(q)))
    if bi % 10 == 0:
        print(f"  fold {bi+1}/{len(blocks)} cut={cut} train={len(tr)} "
              f"cum_test={len(recs)}", flush=True)

pt = np.array([r[1] for r in recs])
y = np.array([r[2] for r in recs])
err = pt - y
hit = np.array([expected_bucket_hit(r[3], int(r[2])) for r in recs])

def ci(a):
    if len(a) < 5:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return tuple(np.percentile(a[idx].mean(axis=1), [2.5, 97.5]))

print(f"\n=== WALK-FORWARD  {cfg.station.icao} {METRIC.upper()} nowcast @h{HOUR} ===")
print(f"n={len(err)}  {recs[0][0]} .. {recs[-1][0]}")
lo, hi = ci(err)
print(f"OVERALL   mean bias {err.mean():+.3f}F [{lo:+.3f},{hi:+.3f}]  "
      f"MAE {np.abs(err).mean():.3f}  over on {(err>0).mean():.1%}  "
      f"parity-avg HIT {hit.mean():.1%}")

BANDS = [(-99, 74), (74, 76), (76, 78), (78, 80), (80, 82), (82, 84), (84, 99)]
print(f"\nbinned by the MODEL'S OWN reading (actionable at decision time):")
print(f"{'model band':>12}{'n':>6}{'mean bias':>11}{'95% CI':>20}"
      f"{'%over':>7}{'MAE':>7}{'HIT':>8}")
for a, b in BANDS:
    m = (pt >= a) & (pt < b)
    if m.sum() == 0:
        continue
    e = err[m]
    l, h = ci(e)
    lab = f"<{b}" if a == -99 else (f">={a}" if b == 99 else f"{a}-{b}")
    star = "  <-- SIG" if (l > 0) == (h > 0) and not np.isnan(l) else ""
    print(f"{lab:>12}{m.sum():>6}{e.mean():>+11.3f}"
          f"{f'[{l:+.3f},{h:+.3f}]':>20}{(e>0).mean():>7.1%}"
          f"{np.abs(e).mean():>7.3f}{hit[m].mean():>8.1%}{star}")

print(f"\n--- the specific claim: model >= 80 ---")
m = pt >= 80
e = err[m]
l, h = ci(e)
print(f"n={m.sum()}  mean bias {e.mean():+.3f}F [{l:+.3f},{h:+.3f}]  "
      f"over on {(e>0).mean():.1%}  HIT {hit[m].mean():.1%}")
m2 = pt < 80
e2 = err[m2]
l2, h2 = ci(e2)
print(f"n={m2.sum()}  model <80: bias {e2.mean():+.3f}F [{l2:+.3f},{h2:+.3f}]  "
      f"over on {(e2>0).mean():.1%}  HIT {hit[m2].mean():.1%}")
d = e.mean() - e2.mean()
both = np.concatenate([e, -e2])
print(f"\ndifference in bias (>=80 minus <80): {d:+.3f}F")
print("positive mean bias = the model reads HIGHER than the day settles "
      "(fade it down)")

# Seasonal control: the HANDOFF's winter penalty could masquerade as a seam,
# since >=80 readings cluster in summer.
mon = np.array([int(r[0][5:7]) for r in recs])
print(f"\nseasonal control -- bias by month (is the seam just summer?):")
for mo in range(1, 13):
    mm = mon == mo
    if mm.sum() < 10:
        continue
    hi_mm = mm & (pt >= 80)
    extra = (f"   |  >=80: n={hi_mm.sum():>3} bias {err[hi_mm].mean():+.3f}"
             if hi_mm.sum() >= 5 else "")
    print(f"  month {mo:>2}  n={mm.sum():>4}  bias {err[mm].mean():+.3f}"
          f"  MAE {np.abs(err[mm]).mean():.3f}{extra}")
