"""Does the trained day-ahead model beat a plain inverse-MSE guidance blend?

ANSWER (2026-07-26, walk-forward refit 30, paired on identical rows, bootstrap CI):

    metric  era        model MAE  blend MAE   d (model - blend)
    LOW     all            1.375      1.321   +0.054 SIG   (model WORSE)
    LOW     2021-2023      1.574      1.412   +0.163 SIG   (model much worse)
    LOW     2024           1.358      1.395   -0.037 ns
    LOW     2025+          1.130      1.157   -0.027 ns
    HIGH    all            1.485      1.696   -0.211 SIG
    HIGH    2021-2023      1.518      1.727   -0.209 SIG
    HIGH    2024           1.574      1.960   -0.386 SIG
    HIGH    2025+          1.385      1.487   -0.102 SIG

The HIGH model beats the blend in EVERY era. The LOW model beats it in NONE of the
recent ones -- and its full-window "loss" is an artifact of 2021-2023, when the LOW
had almost no guidance (only gfs reached back that far), so the trained model was
fitting mostly-NaN rows while BaselineBlend simply drops absent columns.

Re-run this after any change to the day-ahead feature sets. The era split is the
point: a full-window average here is dominated by a data regime that no longer
exists.

The full-history walk-forward says yes (MAE 1.321 vs 1.375). But that window
starts in 2021, when the LOW had almost no guidance -- only gfs reached back that
far, so the trained model spent years fitting mostly-NaN rows while BaselineBlend
degrades gracefully by simply dropping absent columns.

So: split the comparison by era. If the blend only wins on the data-poor years,
the model is fine and the headline is an artifact. If it wins on 2025+ too, the
trained day-ahead model is not earning its complexity for the LOW and that is a
real finding.

Paired on identical test rows, bootstrap CI. Read-only.
"""
import sys

import numpy as np

from highcast import backtest as bt, features
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.dayahead import DayAheadModel, HybridDayAheadModel

cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
rng = np.random.default_rng(0)

for metric in ("low", "high"):
    spec = get_metric(cfg, metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    df = features.build_features(conn, cfg, spec, dates)
    cols = features.model_features(metric)
    pc = features.uses_bias_correction(metric)
    klass = HybridDayAheadModel if cfg.model.hybrid_anchor else DayAheadModel

    recs_m = bt.walk_forward(
        df, lambda: klass(cal_mode=cfg.calibration.mode, feature_cols=cols),
        refit_every=30)
    recs_b = bt.walk_forward(
        df, lambda: BaselineBlend(prefer_corrected=pc), refit_every=30)

    by_m = {r["target_date"]: r for r in recs_m}
    by_b = {r["target_date"]: r for r in recs_b}
    common = sorted(set(by_m) & set(by_b))

    print(f"\n=== {metric.upper()}  matched n={len(common)} ===")
    print(f"{'era':16}{'n':>6}{'model MAE':>11}{'blend MAE':>11}"
          f"{'d (model-blend)':>28}")
    for label, lo, hi in (("all", "0000", "9999"),
                          ("2021-2023", "0000", "2024-01-01"),
                          ("2024", "2024-01-01", "2025-01-01"),
                          ("2025+", "2025-01-01", "9999")):
        sub = [d for d in common if lo <= d < hi]
        if len(sub) < 60:
            continue
        am = np.array([by_m[d]["abs_err"] for d in sub])
        ab = np.array([by_b[d]["abs_err"] for d in sub])
        diff = am - ab
        n = len(diff)
        bs = diff[rng.integers(0, n, size=(4000, n))].mean(axis=1)
        clo, chi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if (clo > 0) == (chi > 0) else "ns"
        print(f"{label:16}{n:>6}{am.mean():>11.3f}{ab.mean():>11.3f}"
              f"{diff.mean():>+11.3f} [{clo:+.3f},{chi:+.3f}] {sig}")
print("\npositive d = the trained model is WORSE than the plain guidance blend")
