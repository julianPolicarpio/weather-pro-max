"""Does training the nowcast AT the hour it predicts beat the fixed default?

Production (`cli.cmd_predict --live`) trains on CHECKPOINT_HOURS (8..20) and then
predicts at `now.hour`, which is routinely OUTSIDE that range -- the 2026-07-26
false LOW signal was generated at local hour 3.

Arms, per (metric, hour):
  PROD    train CHECKPOINT_HOURS, test @h
  MATCH   train @h,               test @h

Single train/test holdout (train < SPLIT, test >= SPLIT) rather than full
walk-forward: this is a COMPARISON between two training regimes on identical
test rows, so one fit per arm is enough and it runs in minutes instead of hours.
Absolute numbers here are therefore not comparable to the walk-forward figures
in HANDOFF.md -- only the PROD-vs-MATCH deltas are.
"""
import sys

import numpy as np

from highcast.buckets import bucket_probs
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import CHECKPOINT_HOURS, IntradayModel, build_checkpoint_frame

SPLIT = "2025-01-01"
cfg = load_config('config/klax.yaml')
conn = connect(cfg.db_path)
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]

COMBOS = [("low", 3), ("low", 6), ("low", 20), ("high", 3), ("high", 9)]
rng = np.random.default_rng(0)
cache: dict = {}


def frame(metric, hours):
    key = (metric, tuple(hours))
    if key not in cache:
        spec = get_metric(cfg, metric)
        cache[key] = build_checkpoint_frame(conn, cfg, dates, hours=hours,
                                           metric=spec)
    return cache[key]


print(f"holdout: train < {SPLIT}, test >= {SPLIT}\n")
print(f"{'metric':7}{'hour':>5}{'n':>6} | {'PROD mae':>9}{'crps':>7} | "
      f"{'MATCH mae':>10}{'crps':>7} | {'dMAE (MATCH-PROD)':>28}{'dCRPS':>26}")

for metric, h in COMBOS:
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    test = frame(metric, (h,))
    test = test[(test["target_date"] >= SPLIT) & test["y"].notna()]
    if test.empty:
        print(f"{metric:7}{h:>5}  no test rows")
        continue
    out = {}
    for label, train_hours in (("PROD", CHECKPOINT_HOURS), ("MATCH", (h,))):
        tr = frame(metric, train_hours)
        tr = tr[(tr["target_date"] < SPLIT) & tr["y"].notna()]
        m = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
        qs = m.predict_quantiles(test)
        y = test["y"].to_numpy(dtype=float)
        pt = np.array([q[0.5] for q in qs])
        cr = np.array([crps_pmf(bucket_probs(q), int(v)) for q, v in zip(qs, y)])
        out[label] = (np.abs(pt - y), cr)
    (ap, cp), (am, cm) = out["PROD"], out["MATCH"]
    n = len(test)
    idx = rng.integers(0, n, size=(4000, n))
    dm, dc = am - ap, cm - cp
    ci_m = np.percentile(dm[idx].mean(axis=1), [2.5, 97.5])
    ci_c = np.percentile(dc[idx].mean(axis=1), [2.5, 97.5])
    sig_m = "SIG" if (ci_m[0] > 0) == (ci_m[1] > 0) else "ns"
    sig_c = "SIG" if (ci_c[0] > 0) == (ci_c[1] > 0) else "ns"
    print(f"{metric:7}{h:>5}{n:>6} | {ap.mean():>9.3f}{cp.mean():>7.3f} | "
          f"{am.mean():>10.3f}{cm.mean():>7.3f} | "
          f"{dm.mean():>+8.3f} [{ci_m[0]:+.3f},{ci_m[1]:+.3f}] {sig_m:>4}"
          f"{dc.mean():>+9.3f} [{ci_c[0]:+.3f},{ci_c[1]:+.3f}] {sig_c:>4}")
print("\nnegative delta = MATCH (train at the prediction hour) is BETTER")
