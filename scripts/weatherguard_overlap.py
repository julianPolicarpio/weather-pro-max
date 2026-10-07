"""PROMPT-3 TASK 0: is a boundary-proximity guard a SECOND gate, or the SAME one?

Prompt 3 asks for a weatherguard that scores per-market confidence and skips
unstable predictions. Of its five candidate signals, boundary proximity is the
only one with measured discriminating power on these markets (2026-08-02,
`scripts/market_baseline.py`):

    KMIA HIGH  within 0.25F of a ladder boundary: 62.2% hit   mid-bucket: 85.8%
    KLAX LOW   within 0.25F of a ladder boundary: 52.9% hit   mid-bucket: 79.6%

BUT THE MODEL ALREADY KNOWS. Its stated probability tracks realised hit rate
along that same axis to within ~2.6 points -- near a boundary it says 59.6% and
hits 62.2%; mid-bucket it says 85.1% and hits 85.8%. And `gating.CONFIDENCE_FLOOR`
already refuses anything below 0.60 calibrated. So the guard and the floor may be
rejecting THE SAME DAYS FOR THE SAME REASON.

Prompt 2 names this failure explicitly: "If both filters end up rejecting the
same trades for the same reason, say so -- we do not need two of the same gate."
And HANDOFF §4.1's lesson is that the gate belongs at Task 0, not Task 5: the
bound-regime work put its gate fifth and wasted three tasks building something
that then failed.

So this script runs BEFORE any weatherguard is built, and it can conclude "do not
build it".

WHAT IT MEASURES, on nested walk-forward records at the commit hour:
  1. the 2x2 confusion between FLOOR-rejects and BOUNDARY-rejects
  2. of the days the boundary guard would reject, how many the floor ALREADY
     rejects -- the redundancy number
  3. the realised hit rate of the days the boundary guard rejects that the floor
     KEEPS. This is the only set that justifies the guard's existence. If those
     days win at the same rate as the days both gates keep, the guard adds
     nothing but variance reduction, and Prompt 3 is explicit that lowering
     variance by trading less "has proved nothing on its own".

⚠️ SCORED ON SINGLE-PARITY DAY-LADDERS, one record per (day, parity), because
that is what the bot faces. Parity-averaging lets a confident-and-right ladder
carry a confident-and-wrong one into the same band -- the averaging error that
once reported this model as +14.9 pts under-confident when it was -4.2 pts over.

⚠️ CALIBRATED, NOT RAW. The floor consumes `gating.band_calib` output; comparing
against `p_model` would test a gate nobody runs. Where `band_calib` returns None
it means REFUSE, and that is counted as a floor-reject, not skipped.

Usage:
    HIGHCAST_DB=/path/to/copy.db .venv/bin/python scripts/weatherguard_overlap.py \
        [config/kmia.yaml] [high|low]

Reads only, but `db.connect` runs DDL -- point HIGHCAST_DB at a COPY.
"""
import dataclasses
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from highcast.buckets import bucket_probs, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import CONFIDENCE_FLOOR, MIN_DECISION_HOUR, band_calib
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml"
METRIC = (sys.argv[2] if len(sys.argv) > 2 else "high").lower()
WF_START, FREEZE, STEP_DAYS, MIN_TRAIN = "2023-01-01", "2026-08-02", 90, 400
# The boundary threshold to test. 0.25F is the bin that measured 62%/53% hit.
BOUNDARY_F = 0.25


def folds(start, end, step):
    a, stop = date.fromisoformat(start), date.fromisoformat(end)
    while a < stop:
        b = min(a + timedelta(days=step), stop)
        yield a.isoformat(), b.isoformat()
        a = b


def main():
    cfg = load_config(CONFIG)
    db = os.environ.get("HIGHCAST_DB")
    if not db:
        print("HIGHCAST_DB is not set -- refusing to run against the live DB.")
        return 1
    cfg = dataclasses.replace(cfg, db_path=Path(db))
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, METRIC)
    station = cfg.station.icao
    hour = MIN_DECISION_HOUR.get(station, {}).get(METRIC)
    floor = CONFIDENCE_FLOOR.get((station, METRIC, hour))

    print(f"{station} {METRIC.upper()} h{hour}   boundary threshold {BOUNDARY_F}F")
    print(f"shipped CONFIDENCE_FLOOR: {floor if floor is not None else 'ABSENT'}")
    if floor is None:
        # `confidence_floor_ok` returns False for an unmeasured market, so the
        # floor rejects EVERY day and the overlap question is degenerate --
        # reporting "100% redundant" there would be an artifact, not a finding.
        print("  ⚠️ This market has no floor entry, so `confidence_floor_ok` "
              "refuses unconditionally. Overlap would be trivially total and "
              "tells us nothing. Run this on a market that has a floor.")
        return 0

    col = "high_f" if METRIC == "high" else "low_f"
    dates = [d for (d,) in conn.execute(
        f"SELECT date FROM cli_reports WHERE station=? AND {col} IS NOT NULL "
        "ORDER BY date", (station,))]
    k = cfg.model.intraday_bandwidth_for(METRIC)
    train_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
    resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
    extra = [c for c in features_for(station, METRIC) if c not in INTRADAY_FEATURES]

    fr = build_checkpoint_frame(conn, cfg, dates, hours=train_hours, metric=spec)
    fr = fr[fr["y"].notna()].copy()
    day_df = build_features(conn, cfg, spec, dates, require_target=True)

    recs = []   # (calibrated_or_None, boundary_dist, hit)
    for start, end in folds(WF_START, FREEZE, STEP_DAYS):
        tr_days = day_df[day_df["target_date"] < start]
        if len(tr_days) < MIN_TRAIN:
            continue
        base = BaselineBlend(
            prefer_corrected=uses_bias_correction(spec.key)).fit(tr_days)
        qs = base.predict_quantiles(day_df)
        f = fr.copy()
        f["base_p50"] = f["target_date"].map(
            dict(zip(day_df["target_date"], [q[0.5] for q in qs])))
        f["base_spread"] = f["target_date"].map(
            dict(zip(day_df["target_date"], [q[0.9] - q[0.1] for q in qs])))
        tr = f[f["target_date"] < start]
        te = f[(f["target_date"] >= start) & (f["target_date"] < end)
               & (f["hour"] == float(hour))]
        if len(tr) < MIN_TRAIN or te.empty:
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(tr)
        for (_, r), q in zip(te.iterrows(), m.predict_quantiles(te)):
            pmf = bucket_probs(q)
            pt = sum(kk * vv for kk, vv in pmf.items()) / max(sum(pmf.values()), 1e-9)
            y = int(r["y"])
            for parity in (0, 1):
                lo, stated = modal_bucket(pmf, parity)
                d = (pt - (parity + 1.5)) % 2.0
                dist = min(d, 2.0 - d)
                recs.append((band_calib(station, METRIC, hour, stated),
                             dist, 1.0 if lo <= y <= lo + 1 else 0.0))

    cal = np.array([np.nan if c is None else c for c, _, _ in recs])
    dist = np.array([d for _, d, _ in recs])
    hit = np.array([h for _, _, h in recs])
    n = len(recs)

    # `band_calib` -> None means REFUSE, so NaN counts as a floor reject.
    floor_rej = np.isnan(cal) | (cal < floor)
    bound_rej = dist < BOUNDARY_F

    print(f"\nday-ladders: {n}   overall hit {hit.mean():.1%}")
    print(f"\n  {'':>22}{'boundary KEEPS':>18}{'boundary REJECTS':>18}")
    for lab, fm in (("floor KEEPS", ~floor_rej), ("floor REJECTS", floor_rej)):
        a = int((fm & ~bound_rej).sum()); b = int((fm & bound_rej).sum())
        print(f"  {lab:>22}{a:>18}{b:>18}")

    only_bound = ~floor_rej & bound_rej
    both = floor_rej & bound_rej
    kept = ~floor_rej & ~bound_rej
    print(f"\n  boundary guard would reject {int(bound_rej.sum())} ladders; "
          f"the floor ALREADY rejects {int(both.sum())} of them "
          f"({both.sum()/max(bound_rej.sum(),1):.1%} redundant)")

    print(f"\n  THE SET THAT JUSTIFIES THE GUARD -- rejected by boundary, kept "
          f"by the floor:")
    if only_bound.sum() == 0:
        print("    n=0. The guard rejects nothing the floor does not already "
              "reject. IT IS THE SAME GATE -- do not build it.")
    else:
        print(f"    n={int(only_bound.sum())}   hit {hit[only_bound].mean():.1%}")
        print(f"    vs days BOTH gates keep: n={int(kept.sum())}   "
              f"hit {hit[kept].mean():.1%}")
        delta = hit[only_bound].mean() - hit[kept].mean()
        # Bootstrap the difference so a few dozen records cannot look decisive.
        rng = np.random.default_rng(20260802)
        a, b = hit[only_bound], hit[kept]
        boots = [rng.choice(a, len(a), replace=True).mean()
                 - rng.choice(b, len(b), replace=True).mean() for _ in range(2000)]
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        print(f"    difference {delta:+.1%}  95% CI [{lo_ci:+.1%}, {hi_ci:+.1%}]")
        if lo_ci < 0 < hi_ci:
            print("    -> NOT DISTINGUISHABLE. The days this guard uniquely "
                  "rejects win at the same rate as the ones it keeps, so it is "
                  "removing trades without removing losses. DO NOT BUILD IT.")
        elif hi_ci < 0:
            print("    -> the uniquely-rejected days really do lose more often. "
                  "The guard adds something the floor does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
