"""Does out-of-fold PIT calibration fix the nowcast's miscalibration?

The nowcast shipped RAW LightGBM quantiles with no calibration layer at all, while
`DayAheadModel` has had out-of-fold PIT calibration since day one. LightGBM
quantile regression under-disperses, so both tails were pulled in at exactly the
hours that need a forecast rather than an observation.

Scored on what actually matters for a bucket bettor, in priority order:
  COVERAGE   one-sided P(y <= q_level) vs nominal -- the direct calibration test
  CAL GAP    |stated modal-bucket probability - realized hit rate|, the number
             the staking gate consumes. This is THE metric: a "62%" that is really
             50% is what produced live false PLACE signals.
  CRPS       proper scoring rule; must not degrade
  MAE / HIT  point + bucket accuracy; calibration should leave these ~unchanged
             (it rescales the distribution, it does not move the median much)

A calibrator that merely widens everything would improve coverage while wrecking
CRPS and the already-well-calibrated late hours -- so the late hours are reported
too, as the control.

Usage: .venv/bin/python scripts/calibration_eval.py [low|high|both]
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
from highcast.models.calibrate import PITCalibrator
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

SPLIT = "2025-01-01"
which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
METRICS = ["low", "high"] if which == "both" else [which]
HOURS = {"low": [3, 5, 6, 9, 11, 20], "high": [5, 9, 11, 13, 16]}

cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
rng = np.random.default_rng(0)


def kalshi_lo(t):
    return t if t % 2 == 0 else t - 1


def score(model, te, calibrated: bool):
    if not calibrated:
        # Neutralise the calibrator without touching the fitted trees: an empty
        # PITCalibrator's level() is the identity, reproducing the old behaviour
        # exactly. Same models both arms, so the ONLY difference is calibration.
        model = _uncal(model)
    qs = model.predict_quantiles(te)
    y = te["y"].to_numpy(dtype=float)
    pt = np.array([q[0.5] for q in qs])
    ae = np.abs(pt - y)
    cr, hit, stated, cov = [], [], [], {0.1: [], 0.5: [], 0.9: []}
    for q, v in zip(qs, y):
        pmf = bucket_probs(q)
        cr.append(crps_pmf(pmf, int(v)))
        agg = {}
        for k, p in pmf.items():
            agg[kalshi_lo(int(k))] = agg.get(kalshi_lo(int(k)), 0.0) + p
        lo = max(agg, key=agg.get)
        hit.append(1.0 if lo <= int(v) <= lo + 1 else 0.0)
        stated.append(agg[lo])
        for lv in cov:
            cov[lv].append(1.0 if v <= q[lv] else 0.0)
    return {"ae": ae, "crps": np.array(cr), "hit": np.array(hit),
            "stated": np.array(stated),
            "cov": {lv: float(np.mean(v)) for lv, v in cov.items()}}


class _uncal:
    """Wrap a fitted model so predict_quantiles runs with an identity PIT."""

    def __init__(self, m):
        self._m = m

    def predict_quantiles(self, frame):
        saved = self._m.cal
        self._m.cal = PITCalibrator().fit([])      # identity
        try:
            return self._m.predict_quantiles(frame)
        finally:
            self._m.cal = saved


for metric in METRICS:
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    day_df = features.build_features(conn, cfg, spec, dates)
    base = BaselineBlend(
        prefer_corrected=features.uses_bias_correction(metric)).fit(day_df)

    print(f"\n{'='*94}\n{metric.upper()} nowcast  RAW vs PIT-CALIBRATED  "
          f"(holdout >= {SPLIT})\n{'='*94}")
    print(f"{'hour':>4}{'n':>5} | {'cov10 raw>cal':>16}{'cov50 raw>cal':>16}"
          f"{'cov90 raw>cal':>16} | {'calgap raw>cal':>16}{'CRPS raw>cal':>15}"
          f"{'hit raw>cal':>14}")
    for h in HOURS[metric]:
        fr = build_checkpoint_frame(conn, cfg, dates, hours=(h,), baseline=base,
                                   metric=spec)
        tr = fr[(fr["target_date"] < SPLIT) & fr["y"].notna()]
        te = fr[(fr["target_date"] >= SPLIT) & fr["y"].notna()]
        if te.empty or len(tr) < 400:
            print(f"{h:>4}  skipped")
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
        raw, cal = score(m, te, False), score(m, te, True)
        gr = abs(raw["stated"].mean() - raw["hit"].mean())
        gc = abs(cal["stated"].mean() - cal["hit"].mean())
        print(f"{h:>4}{len(te):>5} | "
              f"{raw['cov'][0.1]:>7.3f}>{cal['cov'][0.1]:<8.3f}"
              f"{raw['cov'][0.5]:>7.3f}>{cal['cov'][0.5]:<8.3f}"
              f"{raw['cov'][0.9]:>7.3f}>{cal['cov'][0.9]:<8.3f} | "
              f"{gr:>7.1%}>{gc:<8.1%}"
              f"{raw['crps'].mean():>7.3f}>{cal['crps'].mean():<7.3f}"
              f"{raw['hit'].mean():>6.1%}>{cal['hit'].mean():<7.1%}")

print("\ncov* should approach its NOMINAL level (0.10 / 0.50 / 0.90).")
print("calgap = |stated modal-bucket prob - realized hit|; lower is honest.")
