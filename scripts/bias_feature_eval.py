"""Does trailing-bias-corrected guidance improve the DAY-AHEAD model?

The acceptance test HANDOFF-2026-07-26.md's BIAS DISCIPLINE section demands:
out-of-sample skill against settled CLI values, paired bootstrap, plus the
+RANDOM noise control. Nothing else counts.

Arms, all on identical test rows:
  BASE     FEATURES minus the new bias columns (i.e. production before this change)
  +BIAS    full FEATURES (fcc_* + bias_mean/bias_abs_mean/bias_n)
  +RANDOM  BASE plus the same NUMBER of pure-noise columns as +BIAS adds

+RANDOM is the control that matters. Adding any 13 columns changes LightGBM's
feature sampling and can shift MAE by chance; if +RANDOM also "improves", the
+BIAS gain is not evidence of signal. +BIAS must beat BASE *and* +RANDOM must not.

Walk-forward (refit every `--refit`) rather than a single holdout, because a
trailing-bias feature is exactly the kind of thing that can look good on one
split -- and because the handoff's standing complaint is that the hour work was
single-holdout only.

Usage: .venv/bin/python scripts/bias_feature_eval.py [low|high|both] [refit]
"""
import sys

import numpy as np

from highcast import features
from highcast.buckets import bucket_probs
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.dayahead import DayAheadModel

SPLIT = "2025-01-01"
BIAS_COLS = ([f"fcc_{m}" for m in features.FC_MODELS]
             + ["fcc_mean", "bias_mean", "bias_abs_mean", "bias_n"])

which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
REFIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
METRICS = ["low", "high"] if which == "both" else [which]

cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
rng = np.random.default_rng(0)


def kalshi_lo(t: int) -> int:
    return t if t % 2 == 0 else t - 1


def walk_forward(frame, cols, label):
    """Refit every REFIT rows, predict the block that follows. Returns
    (abs_err, crps, bucket_hit) aligned to the test rows."""
    frame = frame.reset_index(drop=True)
    test_idx = frame.index[frame["target_date"] >= SPLIT].tolist()
    ae, cr, hit = [], [], []
    start = test_idx[0]
    for blk in range(start, len(frame), REFIT):
        tr = frame.iloc[:blk]
        tr = tr[tr["y"].notna()]
        te = frame.iloc[blk:blk + REFIT]
        te = te[te["y"].notna()]
        if te.empty or len(tr) < 300:
            continue
        m = DayAheadModel(cal_mode=cfg.calibration.mode, feature_cols=cols)
        m.fit(tr)
        qs = m.predict_quantiles(te)
        y = te["y"].to_numpy(dtype=float)
        pt = np.array([q[0.5] for q in qs])
        ae.extend(np.abs(pt - y))
        for q, v in zip(qs, y):
            pmf = bucket_probs(q)
            cr.append(crps_pmf(pmf, int(v)))
            agg = {}
            for k, p in pmf.items():
                agg[kalshi_lo(int(k))] = agg.get(kalshi_lo(int(k)), 0.0) + p
            lo = max(agg, key=agg.get)
            hit.append(1.0 if lo <= int(v) <= lo + 1 else 0.0)
    return np.array(ae), np.array(cr), np.array(hit)


def boot(delta, n_boot=4000):
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    lo, hi = np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])
    return delta.mean(), lo, hi, "SIG" if (lo > 0) == (hi > 0) else "ns"


for metric in METRICS:
    spec = get_metric(cfg, metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    fr = features.build_features(conn, cfg, spec, dates)
    base_cols = [c for c in features.FEATURES if c not in BIAS_COLS]

    # +RANDOM: same column COUNT as +BIAS adds, pure noise, fixed seed.
    noise_cols = []
    for i in range(len(BIAS_COLS)):
        c = f"rand_{i}"
        fr[c] = rng.normal(size=len(fr))
        noise_cols.append(c)

    arms = {"BASE": base_cols,
            "+BIAS": features.FEATURES,
            "+RANDOM": base_cols + noise_cols}
    res = {}
    for label, cols in arms.items():
        res[label] = walk_forward(fr, cols, label)

    n = len(res["BASE"][0])
    print(f"\n=== {metric.upper()}  walk-forward refit={REFIT}  test>={SPLIT}  "
          f"N={n} ===")
    print(f"{'arm':9}{'MAE':>8}{'CRPS':>8}{'bucket hit':>12}")
    for label in arms:
        ae, cr, hit = res[label]
        print(f"{label:9}{ae.mean():>8.3f}{cr.mean():>8.3f}{hit.mean():>11.1%}")

    print(f"\n{'comparison':22}{'dMAE':>26}{'dCRPS':>26}")
    for label in ("+BIAS", "+RANDOM"):
        ae, cr, _ = res[label]
        b_ae, b_cr, _ = res["BASE"]
        k = min(len(ae), len(b_ae))
        dm = boot(ae[:k] - b_ae[:k])
        dc = boot(cr[:k] - b_cr[:k])
        print(f"{label + ' vs BASE':22}"
              f"{dm[0]:>+9.3f} [{dm[1]:+.3f},{dm[2]:+.3f}] {dm[3]:>4}"
              f"{dc[0]:>+9.3f} [{dc[1]:+.3f},{dc[2]:+.3f}] {dc[3]:>4}")
    bh = res["+BIAS"][2].mean() - res["BASE"][2].mean()
    rh = res["+RANDOM"][2].mean() - res["BASE"][2].mean()
    print(f"  bucket hit: +BIAS {bh:+.1%}   +RANDOM {rh:+.1%}")

print("\nnegative delta = that arm is BETTER than BASE")
print("SHIP RULE: +BIAS significantly better AND +RANDOM not significant")
