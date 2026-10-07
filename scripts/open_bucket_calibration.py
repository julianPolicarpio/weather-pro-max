"""IS THE MODEL CALIBRATED ON OPEN-ENDED BUCKETS? Nothing has ever measured it.

WHY THIS EXISTS, 2026-08-10. The desk refused a KLAX LOW pick of `>=69` at a
stated 96%, reason: "unmeasured confidence -- sits above the highest band this
market has been calibrated on". True, but the diagnosis was subtler than it
looked: `BAND_CALIB`'s highest KLAX-low band tops out at 80% because only 13
recent records ever exceeded it -- AND, more importantly, the bands are built on
the MODAL 2F BUCKET. In 1,634 walk-forward records KLAX low's modal 2F bucket
never once exceeded 90%. Today's live pick was 96%.

The two numbers are different objects. `>=69` aggregates the whole upper tail,
so it reaches probabilities no single 2F bucket ever does. Extending the 2F band
table upward would NOT have licensed that 96%, because the table does not
describe cumulative buckets at all.

Kalshi lists an open-ended bucket at EACH END of every ladder, so this is not an
edge case -- it is two of the six rungs, every day, on every market, and none of
it has ever been calibrated.

WHAT IS MEASURED. Walk-forward at each market's commit hour, same harness as
`scripts/tail_calibration_eval.py` (which measures INDIVIDUAL 2F buckets -- a
different object, since a cumulative sum's errors can cancel or compound).
For every test day and both ladder parities:

    P(outcome >= T)  for the ladder's top open bucket
    P(outcome <= T)  for the bottom one

binned by the model's stated probability, against the realised rate.

THE THRESHOLD MOVES, SO SWEEP IT. Kalshi centres each day's ladder on consensus,
so the open bucket sits a few rungs from the middle and its position varies. The
walk-forward has no historical ladders, so instead of inventing one this sweeps
T across offsets from the model's own median and reports calibration BY STATED
PROBABILITY -- which is what the desk needs: given p, how often is it right?

PARITY IS AVERAGED, as everywhere else in this project.

⚠️ EXPECTED OUTCOME, stated before running: the tail is probably UNDER-stated.
`models/intraday.py` records measured under-dispersion (0.10 -> 0.125,
0.90 -> 0.833), and an under-dispersed model puts too little mass in the tail.
If so, a cumulative bucket that sums that tail would come back UNDER-confident,
meaning a 96% is really higher than 96% and the refusal was over-cautious.
Being wrong about that is the point of running it.

Usage: PYTHONPATH=src .venv/bin/python scripts/open_bucket_calibration.py
"""
import collections
import math

import numpy as np

from highcast.buckets import bucket_probs, ladder_lo
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)

REFIT, MIN_TRAIN, RECENT = 30, 400, "2025-01-01"
# How far from the model's median to place the open bucket's threshold.
#
# ⚠️ THE FIRST VERSION USED (4, 6, 8) AND MEASURED THE WRONG REGION. Those
# offsets put the threshold FAR out, so P(>=T) was always tiny -- every
# populated band came back below 0.50 and there was no data at all above 0.80,
# which is precisely where the question lives. The case that prompted this was
# KLAX low `>=69` at 96%, and 69 sat BELOW the model's median, capturing nearly
# all the mass.
#
# Kalshi centres the ladder on consensus but the open rungs are only ~2-3 rungs
# out, so a cumulative bucket routinely holds most of the distribution. Sweeping
# NEGATIVE offsets too spans the full range from ~1% to ~99%.
OFFSETS = (-6, -4, -2, 0, 2, 4, 6)
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def collect(cfg_path: str, metric_key: str):
    """(stated, hit, side) for the open-ended buckets, both parities, all days."""
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, metric_key)
    hour = MIN_DECISION_HOUR[icao][metric_key]
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric_key}", False))
    k = cfg.model.intraday_bandwidth_for(metric_key)
    tr_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
    feats = features_for(icao, metric_key)

    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (icao,))]
    day_df = build_features(conn, cfg, spec, dates, require_target=True)
    base = BaselineBlend(
        prefer_corrected=uses_bias_correction(spec.key)).fit(day_df)
    fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec,
                                baseline=base)
    te = fr[(fr["hour"] == float(hour)) & fr["y"].notna()].sort_values("target_date")
    te = te[te["target_date"] >= RECENT]
    td = te["target_date"].to_numpy()
    print(f"\n### {icao} {metric_key.upper()} h{hour}   test days={len(te)}",
          flush=True)

    out = []
    for b0 in range(0, len(td), REFIT):
        cut = td[b0]
        tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
        if len(tr) < MIN_TRAIN:
            continue
        blk = te.iloc[b0:b0 + REFIT]
        if blk.empty:
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=[c for c in feats
                                          if c not in _SHIPPED_BASE]).fit(tr)
        for (_, row), q in zip(blk.iterrows(), m.predict_quantiles(blk)):
            pmf = bucket_probs(q)
            y = int(row["y"])
            med = int(round(q[0.5]))
            for parity in (0, 1):
                for off in OFFSETS:
                    hi_t = ladder_lo(med + off, parity)
                    lo_t = ladder_lo(med - off, parity) + 1
                    p_hi = sum(p for d, p in pmf.items() if int(d) >= hi_t)
                    p_lo = sum(p for d, p in pmf.items() if int(d) <= lo_t)
                    out.append((p_hi, float(y >= hi_t), "upper"))
                    out.append((p_lo, float(y <= lo_t), "lower"))
    return out


BINS = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50),
        (0.50, 0.80), (0.80, 0.90), (0.90, 0.97), (0.97, 1.01)]

print(__doc__.split("Usage:")[0])
for cfg_path in ("config/klax.yaml", "config/kmia.yaml"):
    for metric in ("low", "high"):
        rows = collect(cfg_path, metric)
        for side in ("upper", "lower"):
            arr = np.array([(s, h) for s, h, sd in rows if sd == side])
            if not len(arr):
                continue
            print(f"  {side.upper()} open bucket")
            print(f"    {'stated band':<14}{'n':>6}{'stated':>9}{'realised':>10}"
                  f"{'gap':>9}   {'95% CI on realised'}")
            for lo, hi in BINS:
                sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
                if len(sel) < 30:
                    continue
                st, ht = sel[:, 0].mean(), sel[:, 1].mean()
                wl, wh = wilson(sel[:, 1].sum(), len(sel))
                flag = ""
                if not (wl <= st <= wh):
                    flag = "  <<< MISCALIBRATED" + (" (overstates)" if st > ht
                                                    else " (understates)")
                print(f"    {lo:.2f}-{hi:<9.2f}{len(sel):>6}{st:>9.3f}{ht:>10.3f}"
                      f"{st-ht:>+9.3f}   [{wl:.3f},{wh:.3f}]{flag}")
            print()
print("  gap > 0 = the model OVERSTATES the open bucket (its stated number is")
print("  too high).  gap < 0 = UNDERSTATES.  A band whose stated probability")
print("  falls inside the CI on the realised rate is calibrated.")
