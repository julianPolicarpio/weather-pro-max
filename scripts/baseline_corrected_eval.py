"""Does bias-corrected guidance reach the INTRADAY nowcast and help it?

`scripts/bias_feature_eval.py` showed the trailing-bias correction helps the
DAY-AHEAD LOW (dMAE -0.091 SIG). But `preferred_prediction` serves the NOWCAST
for the LOW at every hour, and the nowcast has no `fc_*` feature -- it sees
guidance only as `base_p50`/`base_spread` from `BaselineBlend`. So the day-ahead
gain does not reach the LOW desk on its own.

`BaselineBlend(prefer_corrected=True)` swaps `fcc_{model}` in for `fc_{model}`.
This measures whether that propagates into nowcast accuracy AT THE HOURS THAT
MATTER -- including the pre-formation hours, which is the whole point: the
correction needs no observations, so unlike the ratchet it can help before dawn.

Per HANDOFF BIAS DISCIPLINE: out-of-sample against settled CLI, paired bootstrap
on matched test rows, and scored on modal-bucket hit rate as well as MAE.

Usage: .venv/bin/python scripts/baseline_corrected_eval.py [low|high] [hours csv]
"""
import sys

import numpy as np

from highcast import features
from highcast.buckets import bucket_probs
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

SPLIT = "2025-01-01"
metric = (sys.argv[1] if len(sys.argv) > 1 else "low").lower()
HOURS = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [3, 5, 6, 9, 11, 16, 20])

cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
spec = get_metric(cfg, metric)
resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
    (cfg.station.icao,))]
rng = np.random.default_rng(0)

day_df = features.build_features(conn, cfg, spec, dates)


def kalshi_lo(t: int) -> int:
    return t if t % 2 == 0 else t - 1


def arm(hour: int, prefer_corrected: bool):
    base = BaselineBlend(prefer_corrected=prefer_corrected).fit(day_df)
    fr = build_checkpoint_frame(conn, cfg, dates, hours=(hour,), baseline=base,
                                metric=spec)
    tr = fr[(fr["target_date"] < SPLIT) & fr["y"].notna()]
    te = fr[(fr["target_date"] >= SPLIT) & fr["y"].notna()]
    if te.empty or len(tr) < 400:
        return None
    m = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
    qs = m.predict_quantiles(te)
    y = te["y"].to_numpy(dtype=float)
    ae = np.abs(np.array([q[0.5] for q in qs]) - y)
    cr, hit = [], []
    for q, v in zip(qs, y):
        pmf = bucket_probs(q)
        cr.append(crps_pmf(pmf, int(v)))
        agg = {}
        for k, p in pmf.items():
            agg[kalshi_lo(int(k))] = agg.get(kalshi_lo(int(k)), 0.0) + p
        lo = max(agg, key=agg.get)
        hit.append(1.0 if lo <= int(v) <= lo + 1 else 0.0)
    return te["target_date"].to_numpy(), ae, np.array(cr), np.array(hit)


print(f"=== {metric.upper()} nowcast: BaselineBlend raw vs bias-corrected ===")
print(f"{'hour':>5}{'n':>6} | {'MAE raw':>9}{'MAE corr':>10}{'dMAE':>25} | "
      f"{'hit raw':>9}{'hit corr':>10}")
for h in HOURS:
    a = arm(h, False)
    b = arm(h, True)
    if a is None or b is None:
        print(f"{h:>5}  skipped")
        continue
    d_ae = b[1] - a[1]
    n = len(d_ae)
    idx = rng.integers(0, n, size=(3000, n))
    lo, hi = np.percentile(d_ae[idx].mean(axis=1), [2.5, 97.5])
    sig = "SIG" if (lo > 0) == (hi > 0) else "ns"
    print(f"{h:>5}{n:>6} | {a[1].mean():>9.3f}{b[1].mean():>10.3f}"
          f"{d_ae.mean():>+9.3f} [{lo:+.3f},{hi:+.3f}] {sig:>4} | "
          f"{a[3].mean():>8.1%}{b[3].mean():>9.1%}")

print("\nnegative dMAE = bias-corrected guidance is BETTER")
