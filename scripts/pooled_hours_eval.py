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


for metric in ("low", "high"):
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    print(f"\n=== {metric.upper()}  (residual={resid}) ===", flush=True)

    # ONE frame containing every hour. The per-hour arm slices it; the pooled arm
    # trains on all of it. Same rows either way, so the comparison is paired.
    fr = build_checkpoint_frame(conn, cfg, dates, hours=HOURS, metric=spec)
    fr = fr[fr["y"].notna()]
    tr_all = fr[fr["target_date"] < SPLIT]
    te_all = fr[fr["target_date"] >= SPLIT]
    print(f"  rows: train {len(tr_all)}  test {len(te_all)}", flush=True)

    pooled = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr_all)

    rows = []
    buckets = {"match": {}, "pooled": {}}
    for h in HOURS:
        te = te_all[te_all["hour"] == float(h)]
        tr = tr_all[tr_all["hour"] == float(h)]
        if len(te) < 100 or len(tr) < 400:
            continue
        y = te["y"].to_numpy(dtype=float)
        d = te["target_date"].to_numpy()

        m_match = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
        out = {}
        for name, model in (("match", m_match), ("pooled", pooled)):
            pmfs = [bucket_probs(q) for q in model.predict_quantiles(te)]
            out[name] = score(pmfs, y)
            # even-ladder modal bucket per day, for the churn metric
            buckets[name][h] = {dd: modal_bucket(p, 0)[0]
                                for dd, p in zip(d, pmfs)}
        rows.append((h, len(te), out["match"], out["pooled"]))

    print(f"  {'h':>3}{'n':>6} | {'MATCH mae':>10}{'hit':>7} | "
          f"{'POOL mae':>9}{'hit':>7} | {'dMAE':>7}{'dHIT':>7}", flush=True)
    dmaes, dhits = [], []
    for h, n, (ma, ca, ha), (mp, cp_, hp) in rows:
        dmaes.append(mp - ma)
        dhits.append(hp - ha)
        print(f"  {h:>3}{n:>6} | {ma:>10.3f}{ha:>7.1%} | {mp:>9.3f}{hp:>7.1%} | "
              f"{mp - ma:>+7.3f}{(hp - ha) * 100:>+6.1f}", flush=True)
    if dmaes:
        print(f"  mean dMAE {np.mean(dmaes):+.3f}   mean dHIT "
              f"{np.mean(dhits) * 100:+.1f} pts   (negative dMAE = pooling BETTER)",
              flush=True)
    cm, cpool = churn(buckets["match"], LO, HI), churn(buckets["pooled"], LO, HI)
    print(f"  CHURN over h{LO}-h{HI}:  match {cm:.2f}  pooled {cpool:.2f} distinct "
          f"buckets/day   ({cpool - cm:+.2f})", flush=True)

print("\nnegative dMAE = the single pooled model is more accurate")
print("lower churn   = the desk changes its answer less on the same day")
