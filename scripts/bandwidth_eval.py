"""24 SEPARATE MODELS vs ONE MODEL WITH `hour` AS A FEATURE.

Production trains a fresh booster on the single hour it is predicting
(`cli.cmd_predict --live` sets `hours=(now.hour,)`). Consequence, caught live on
2026-07-28 for KLAX's LOW with the running minimum pinned at 69.98F all night --
no new information whatsoever:

    LA h2  P(68-69) 89.5%   h3 69.6%   h4 73.8%   h5 55.1%   h6 20.3%

Five different answers to the same question, because five different models were
asked. WITHIN an hour the answer is frozen to the decimal; the jumps land exactly
on hour boundaries.

WHY THE CURRENT DESIGN WAS CHOSEN, AND WHY THAT REASONING WAS INCOMPLETE.
`scripts/hour_sweep.py` compared:

    PROD    train CHECKPOINT_HOURS = (8,10,12,14,16,18,20), test @h
    MATCH   train @h,                                       test @h

and MATCH won. But the bug that motivated it was a signal generated at local
hour 3, and hour 3 is outside [8,20] AND an odd hour when every training hour is
even -- so PROD was EXTRAPOLATING, not pooling badly. `hour` is already the first
entry in `INTRADAY_FEATURES`, so the pooled model was always hour-aware; its pool
was simply too narrow to contain the question being asked.

The option never tested is the obvious one: POOL ALL 24 HOURS, keep `hour` as a
feature. Every hour is then in-distribution, adjacent hours share structure (a
split at `hour <= 5` covers 4 and 5 together), and one model cannot contradict
itself -- the 89.5% -> 20.3% swing is not suppressed or alerted on, it is
impossible. Hour-specificity survives because the model splits on `hour` as
finely as the data supports rather than as coarsely as a constant decided.

THE THIRD METRIC IS THE POINT. MAE and HIT say whether pooling costs accuracy.
CHURN says whether it buys stability: the mean number of DISTINCT modal buckets a
single day passes through across the decision-relevant hours. A bettor reading
the desk experiences churn directly -- it is the "why does it keep changing"
complaint, quantified.

Scored on KMIA by default because KLAX is mid-test-week and frozen. Reads
nothing production writes; changes no shipped code.

Usage:  .venv/bin/python scripts/pooled_hours_eval.py [config] [lo_hour] [hi_hour]
"""
import sys

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml"
# BANDWIDTH SWEEP. k=0 is production (train on hour h alone); k=None is the fully
# pooled arm. We tested only those two extremes first and per-hour won -- but in
# local regression the optimal bandwidth is almost never at either end, so the
# interesting arms are the ones in between. k=1 means train on {h-1,h,h+1}, which
# triples the rows while keeping the dawn regime away from the evening one.
BANDWIDTHS = [0, 1, 2, 3, None]
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 2
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 14
SPLIT = "2025-01-01"
HOURS = tuple(range(24))

cfg = load_config(CONFIG)
conn = connect(cfg.db_path)
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]
print(f"station {cfg.station.icao}  n_cli={len(dates)}  split={SPLIT}  "
      f"churn window h{LO}-h{HI}", flush=True)


def score(pmfs, y):
    pt = np.array([sum(k * v for k, v in p.items()) / max(sum(p.values()), 1e-9)
                   for p in pmfs])
    return (float(np.abs(pt - y).mean()),
            float(np.mean([crps_pmf(p, int(v)) for p, v in zip(pmfs, y)])),
            float(np.mean([expected_bucket_hit(p, int(v)) for p, v in zip(pmfs, y)])))


def churn(per_hour_buckets, lo, hi):
    """Mean count of DISTINCT modal buckets a day passes through over [lo,hi].

    1.0 = the answer never moved all day. Higher = the desk kept changing its
    mind about a day whose weather was whatever it was always going to be.
    """
    days: dict = {}
    for h in range(lo, hi + 1):
        for d, lab in per_hour_buckets.get(h, {}).items():
            days.setdefault(d, []).append(lab)
    if not days:
        return float("nan")
    return float(np.mean([len(set(v)) for v in days.values()]))


def hours_for(h, k):
    """Training hours for bandwidth k, wrapping the clock (h=0 borrows h=23)."""
    if k is None:
        return HOURS
    return tuple((h + d) % 24 for d in range(-k, k + 1))


for metric in ("low", "high"):
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    print(f"\n=== {metric.upper()}  (residual={resid}) ===", flush=True)

    fr = build_checkpoint_frame(conn, cfg, dates, hours=HOURS, metric=spec)
    fr = fr[fr["y"].notna()]
    tr_all = fr[fr["target_date"] < SPLIT]
    te_all = fr[fr["target_date"] >= SPLIT]
    print(f"  rows: train {len(tr_all)}  test {len(te_all)}", flush=True)

    # only the decision-relevant hours -- the sweep is 5 bandwidths x N hours of
    # fits, so scoring all 24 would take hours for numbers nobody acts on.
    EVAL_HOURS = [2, 3, 4, 5, 6, 7, 11, 12, 13]
    res = {}
    for k in BANDWIDTHS:
        per_h, buckets = {}, {}
        for h in EVAL_HOURS:
            te = te_all[te_all["hour"] == float(h)]
            hrs = hours_for(h, k)
            tr = tr_all[tr_all["hour"].isin([float(x) for x in hrs])]
            if len(te) < 100 or len(tr) < 400:
                continue
            y = te["y"].to_numpy(dtype=float)
            m = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
            pmfs = [bucket_probs(q) for q in m.predict_quantiles(te)]
            per_h[h] = score(pmfs, y)
            buckets[h] = {dd: modal_bucket(pp, 0)[0]
                          for dd, pp in zip(te["target_date"].to_numpy(), pmfs)}
        if not per_h:
            continue
        mae = float(np.mean([v[0] for v in per_h.values()]))
        hit = float(np.mean([v[2] for v in per_h.values()]))
        ch = churn(buckets, min(EVAL_HOURS), max(EVAL_HOURS))
        res[k] = (mae, hit, ch, per_h)
        lab = "all" if k is None else f"+/-{k}"
        print(f"  k={lab:<4} trainhrs={'24' if k is None else 2*k+1:<3} "
              f"mae {mae:.3f}  hit {hit:.1%}  churn {ch:.2f}", flush=True)

    if 0 in res:
        b0 = res[0]
        print(f"  {'':4}vs production (k=0):", flush=True)
        for k, v in res.items():
            if k == 0:
                continue
            lab = "all" if k is None else f"+/-{k}"
            print(f"     k={lab:<4} dMAE {v[0]-b0[0]:+.3f}  dHIT "
                  f"{(v[1]-b0[1])*100:+.1f} pts  dCHURN {v[2]-b0[2]:+.2f}", flush=True)

print("\nnegative dMAE / positive dHIT / negative dCHURN = better than production")
