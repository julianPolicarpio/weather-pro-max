"""Do the NBS marine-layer / burn-off features improve the DAY-AHEAD forecast?

This targets the hours that matter for PREDICTION -- the day-ahead product is
issued before the extremum exists, which is the only kind of number worth calling
a forecast. Accuracy measured after the extremum has already occurred is
transcription, not skill.

Arms (identical test rows, walk-forward, paired bootstrap):
  BASE     the shipped feature set
  +AUX     BASE plus NBS_AUX_FEATURES (sol / sky / cig / offshore / tsd)
  +RANDOM  BASE plus the same NUMBER of pure-noise columns

+RANDOM is the control that decides it. This session has repeatedly seen extra
correlated columns make things WORSE by diluting LightGBM's per-split feature
sampling, so "it got better" is only meaningful if noise of the same width does
not.

Usage: .venv/bin/python scripts/aux_eval.py [low|high|both] [refit]
"""
import sys

import numpy as np

from highcast import features
from highcast.buckets import bucket_probs
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.dayahead import DayAheadModel, HybridDayAheadModel

SPLIT = "2025-01-01"
which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
REFIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
METRICS = ["low", "high"] if which == "both" else [which]

cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
rng = np.random.default_rng(0)
AUX = list(features.NBS_AUX_FEATURES)


def kalshi_lo(t):
    return t if t % 2 == 0 else t - 1


def run(fr, cols, metric):
    fr = fr.reset_index(drop=True)
    idx = fr.index[fr["target_date"] >= SPLIT].tolist()
    ae, cr, hit = [], [], []
    for blk in range(idx[0], len(fr), REFIT):
        tr = fr.iloc[:blk]; tr = tr[tr["y"].notna()]
        te = fr.iloc[blk:blk + REFIT]; te = te[te["y"].notna()]
        if te.empty or len(tr) < 300:
            continue
        use_hybrid = cfg.model.hybrid_anchor and metric == "high"
        m = (HybridDayAheadModel(cal_mode=cfg.calibration.mode, feature_cols=cols)
             if use_hybrid else
             DayAheadModel(cal_mode=cfg.calibration.mode, feature_cols=cols))
        m.fit(tr)
        for qs, yv in zip(m.predict_quantiles(te), te["y"].to_numpy(dtype=float)):
            ae.append(abs(qs[0.5] - yv))
            pmf = bucket_probs(qs)
            cr.append(crps_pmf(pmf, int(yv)))
            agg = {}
            for k, p in pmf.items():
                agg[kalshi_lo(int(k))] = agg.get(kalshi_lo(int(k)), 0.0) + p
            lo = max(agg, key=agg.get)
            hit.append(1.0 if lo <= int(yv) <= lo + 1 else 0.0)
    return np.array(ae), np.array(cr), np.array(hit)


for metric in METRICS:
    spec = get_metric(cfg, metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    fr = features.build_features(conn, cfg, spec, dates)
    shipped = features.model_features(metric)
    base = [c for c in shipped if c not in AUX]
    noise = []
    for i in range(len(AUX)):
        c = f"rand_{i}"
        fr[c] = rng.normal(size=len(fr))
        noise.append(c)
    arms = {"BASE": base, "+AUX": shipped, "+RANDOM": base + noise}
    res = {k: run(fr, v, metric) for k, v in arms.items()}

    b = res["BASE"]
    print(f"\n=== {metric.upper()}  day-ahead  walk-forward refit={REFIT}  "
          f"N={len(b[0])} ===")
    print(f"{'arm':10}{'MAE':>8}{'CRPS':>8}{'hit':>8}{'':>4}{'dMAE vs BASE':>26}")
    for k, (ae, cr, hit) in res.items():
        n = min(len(ae), len(b[0]))
        d = ae[:n] - b[0][:n]
        bs = d[rng.integers(0, n, size=(4000, n))].mean(axis=1)
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if (lo_ > 0) == (hi_ > 0) else "ns"
        tag = "" if k == "BASE" else f"{d.mean():>+8.3f} [{lo_:+.3f},{hi_:+.3f}] {sig}"
        print(f"{k:10}{ae.mean():>8.3f}{cr.mean():>8.3f}{hit.mean():>7.1%}"
              f"{'':>4}{tag:>26}")

print("\nSHIP RULE: +AUX significantly better AND +RANDOM not an improvement.")
