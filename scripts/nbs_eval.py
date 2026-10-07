"""Is nbm_nbs worth a feature slot, and does pruning dead columns pay for it?

Adding it naively COST the LOW 2.7 points of bucket hit (49.6% -> 46.9%) with MAE
flat -- the signature of feature-sampling dilution, not bad data. So test it
against a pruned set: `fc_nws` is 0.4% non-null (open item 5) and `fc_nbm` is
Open-Meteo NBM, ~all-NaN before 2024-10 and superseded by the bulletin channel.
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

SPLIT, REFIT = "2025-01-01", 120
cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
rng = np.random.default_rng(0)
DEAD = list(features.DEAD_FEATURES)
NBS = list(features.NBS_FEATURES)


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


for metric in ("low", "high"):
    spec = get_metric(cfg, metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    fr = features.build_features(conn, cfg, spec, dates)
    # Build arms from the RAW feature list, not model_features -- that function
    # now encodes the very decision this script exists to justify, so using it
    # would make every arm identical to the shipped one.
    full = [c for c in features.FEATURES
            if metric == "low" or c not in features.BIAS_FEATURES]
    arms = {
        "A no-nbs":       [c for c in full if c not in NBS],
        "B +nbs":         list(full),
        "C +nbs -dead":   [c for c in full if c not in DEAD],
        "D -dead only":   [c for c in full if c not in DEAD + NBS],
    }
    res = {k: run(fr, v, metric) for k, v in arms.items()}
    base = res["A no-nbs"]
    print(f"\n=== {metric.upper()}  N={len(base[0])} ===")
    print(f"{'arm':14}{'MAE':>8}{'CRPS':>8}{'hit':>8}{'ncols':>7}"
          f"{'dMAE vs A':>26}")
    for k, (ae, cr, hit) in res.items():
        n = min(len(ae), len(base[0]))
        d = ae[:n] - base[0][:n]
        bs = d[rng.integers(0, n, size=(3000, n))].mean(axis=1)
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if (lo_ > 0) == (hi_ > 0) else "ns"
        tag = "" if k == "A no-nbs" else f"{d.mean():>+8.3f} [{lo_:+.3f},{hi_:+.3f}] {sig}"
        print(f"{k:14}{ae.mean():>8.3f}{cr.mean():>8.3f}{hit.mean():>7.1%}"
              f"{len(arms[k]):>7}{tag:>26}")
