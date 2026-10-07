"""THE TOMORROW REGIME — how good is the evening-before forecast for the NEXT day?

The dashboard's "tomorrow" card runs the day-ahead model with every `morn_*`
feature NaN (verified live for 2026-07-26 HIGH: morn_tmpf/dwpf/onshore/sky/dtd/
rate all None, p10 66.0 / p90 81.3 — a 15F band). That regime is never
backtested: the model is trained on rows that all HAVE morning obs.

Measure it honestly by walk-forward on matched days:
  WITH    — the normal day-ahead frame (what the 9am refresh uses)
  WITHOUT — identical, but every morn_* column blanked at BOTH fit and predict
            time, i.e. a model actually TRAINED for the no-morning regime
  NAIVE   — trained WITH morning obs, predicted WITHOUT (what ships today)

NAIVE is the production tomorrow-card. If NAIVE is much worse than WITHOUT, the
fix is to train a dedicated no-morning variant for the tomorrow card.
"""
import sys
import numpy as np
import pandas as pd

from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, FEATURES
from highcast.metric import get_metric
from highcast.models.dayahead import DayAheadModel, HybridDayAheadModel
from highcast.buckets import bucket_probs
from highcast.metrics import crps_pmf

METRIC = sys.argv[1] if len(sys.argv) > 1 else "high"
REFIT = int(sys.argv[2]) if len(sys.argv) > 2 else 90

cfg = load_config('config/klax.yaml')
conn = connect(cfg.db_path)
spec = get_metric(cfg, METRIC)
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]
df = build_features(conn, cfg, spec, dates).sort_values("target_date").reset_index(drop=True)
MORN = [c for c in FEATURES if c.startswith("morn_")]
print(f"metric={METRIC}  rows={len(df)}  morn cols={MORN}  refit_every={REFIT}")

blank = df.copy()
for c in MORN:
    blank[c] = np.nan


def factory():
    return (HybridDayAheadModel() if cfg.model.hybrid_anchor else DayAheadModel())


MIN_TRAIN = 365


def run(fit_df, pred_df, label):
    recs, model = [], None
    for i in range(MIN_TRAIN, len(pred_df)):
        if model is None or (i - MIN_TRAIN) % REFIT == 0:
            try:
                model = factory().fit(fit_df.iloc[:i])
            except ValueError:
                model = None
        if model is None:
            continue
        row = pred_df.iloc[[i]]
        y = row["y"].iloc[0]
        if np.isnan(y):
            continue
        qs = model.predict_quantiles(row)[0]
        recs.append({"d": row["target_date"].iloc[0], "y": int(y),
                     "ae": abs(qs[0.5] - y),
                     "crps": crps_pmf(bucket_probs(qs), int(y)),
                     "width": qs[0.9] - qs[0.1],
                     "in90": float(qs[0.1] <= y <= qs[0.9])})
    print(f"  {label}: n={len(recs)}")
    return {r["d"]: r for r in recs}


arms = {
    "WITH morning obs (9am refresh)": run(df, df, "with"),
    "WITHOUT, trained for it": run(blank, blank, "without-trained"),
    "NAIVE (train WITH, predict WITHOUT) <- ships today": run(df, blank, "naive"),
}

common = sorted(set.intersection(*(set(v) for v in arms.values())))
print(f"\nmatched days: {len(common)}")
base = "WITH morning obs (9am refresh)"
rng = np.random.default_rng(0)
print(f"{'arm':52}{'MAE':>7}{'CRPS':>7}{'p10-90':>8}{'cov90':>7}   vs WITH")
for label, res in arms.items():
    ae = np.array([res[d]["ae"] for d in common])
    cr = np.array([res[d]["crps"] for d in common])
    wd = np.array([res[d]["width"] for d in common])
    c9 = np.array([res[d]["in90"] for d in common])
    line = f"{label:52}{ae.mean():>7.3f}{cr.mean():>7.3f}{wd.mean():>8.2f}{c9.mean():>7.2f}"
    if label != base:
        b = np.array([arms[base][d]["ae"] for d in common])
        bc = np.array([arms[base][d]["crps"] for d in common])
        dae, dcr = ae - b, cr - bc
        idx = rng.integers(0, len(common), size=(4000, len(common)))
        ci = np.percentile(dae[idx].mean(axis=1), [2.5, 97.5])
        cic = np.percentile(dcr[idx].mean(axis=1), [2.5, 97.5])
        line += (f"   dMAE {dae.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                 f"  dCRPS {dcr.mean():+.3f} [{cic[0]:+.3f},{cic[1]:+.3f}]")
    print(line)
