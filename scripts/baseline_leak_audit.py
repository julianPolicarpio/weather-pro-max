"""Does the nowcast's guidance channel see the future? KMIA HIGH @ h14.

PROMPT-2 AUDIT ITEM, run FIRST rather than last. `HANDOFF-2026-08-03.md` §4.1:
"the plan put the gate at Task 5. It belongs at Task 0." If the leak below is
large, every baseline number has to be recomputed honestly anyway -- so it is
cheaper to size it before measuring anything else.

THE SUSPECT. `build_checkpoint_frame(conn, cfg, dates)` with `baseline=None`
fits a `BaselineBlend` on `dates` -- ALL of them, train and test together
(`models/intraday.py`, the `base = baseline or BaselineBlend(...).fit(day_df)`
line). That blend then produces `base_p50` and `base_spread`, 2 of the nowcast's
14 features. `BaselineBlend.fit` consumes `df["y"]` twice:

    weights      inverse-MSE per guidance column, over every row incl. test
    _resid       seasonal residual pool -> `base_spread` reads its 10-90 width

So on the house method both features carry information from the test period.
`scripts/README.md` already flags this ("a known leakage smell, open item 4")
and argues it "inflates absolute figures equally across arms, so comparisons
stand and absolutes are optimistic". That argument is fine for the arm-vs-arm
deltas those scripts report. It is NOT fine here, because three ABSOLUTE numbers
are load-bearing:

    gating.HOUR_ACCURACY  -> MIN_DECISION_HOUR -> the commit hour (h14)
    gating.BAND_CALIB     -> the probability the bot actually trades
    gating.CONFIDENCE_FLOOR -> whether it trades at all

and the KMIA HIGH headline (walk-forward MAE 0.539, Jun-Sep 0.440 / hit 80.3%)
is quoted as the reason this market is worth trading at all.

⚠️ THERE IS ALSO A TRAIN/SERVE SKEW HERE, independent of the leak's size. The
LIVE path (`cli.cmd_predict --live`) fits its blend on HISTORY ONLY and passes
it in. So training-time `base_p50`/`base_spread` are computed one way and
serve-time the other -- the same class of defect as the evening/day-ahead kind
conflation in HANDOFF §2.1, which cost 4 degrees.

ARMS (identical in every other respect -- same rows, same features, same fit):
    LEAK    blend fitted on train+test, i.e. what every eval script does today
    HONEST  blend fitted on TRAIN ONLY, i.e. what production actually serves

Only `base_p50`/`base_spread` differ, so the frame is built once and those two
columns are recomputed for the honest arm rather than rebuilt from scratch.

Scored on the quantity the BOT faces (single-parity day-ladders, per HANDOFF
§2.6) as well as the parity-averaged one, because those two disagreed before.

Usage:  HIGHCAST_DB=/path/to/copy.db .venv/bin/python scripts/baseline_leak_audit.py

Reads only, but `db.connect` runs idempotent DDL, so point HIGHCAST_DB at a COPY
rather than racing the 5-minute wake cycle on the live file.
"""
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

CONFIG = "config/kmia.yaml"
STATION, METRIC, HOUR = "KMIA", "high", 14
# Two cut points. 2025-01-01 is the house convention, so the LEAK arm there is
# directly comparable to the figures already quoted. 2026-06-01 is the frozen
# window nothing has been tuned on -- small, and reported as such.
SPLITS = ["2025-01-01", "2026-06-01"]


def score(pmfs, ys):
    """MAE / bias / CRPS / hit / log loss, on both scoring conventions."""
    pt = np.array([sum(k * v for k, v in p.items()) / max(sum(p.values()), 1e-9)
                   for p in pmfs])
    err = pt - ys
    ehit = np.array([expected_bucket_hit(p, int(y)) for p, y in zip(pmfs, ys)])
    crps = np.array([crps_pmf(p, int(y)) for p, y in zip(pmfs, ys)])
    # SINGLE-PARITY DAY-LADDERS: one record per (day, parity).
    #
    # ⚠️ THE MEAN HIT RATE IS THE SAME NUMBER EITHER WAY, and reporting both as
    # if they were two findings would be a fake corroboration: mean over days of
    # (h0 + h1)/2 IS mean over (day, parity) records of h, identically. The
    # parity distinction from HANDOFF §2.6 bites when records are BANDED BY
    # STATED PROBABILITY -- averaging first assigns a day one blended stated
    # probability, which lets a confident-and-right parity carry a
    # confident-and-wrong one into the same band. So the split is reported on
    # the reliability table below, not on the headline hit rate.
    hits, stated, lls = [], [], []
    for p, y in zip(pmfs, ys):
        for parity in (0, 1):
            lo, mass = modal_bucket(p, parity)
            hits.append(1.0 if lo <= int(y) <= lo + 1 else 0.0)
            stated.append(mass)
            # log loss on the bucket that ACTUALLY contained y, not on the modal
            # one: scoring the modal bucket would reward confidence per se.
            true_lo = int(y) - ((int(y) - parity) % 2)
            p_true = sum(v for k, v in p.items() if true_lo <= k <= true_lo + 1)
            lls.append(-np.log(max(p_true, 1e-9)))
    return {
        "n": len(ys), "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
        "crps": float(crps.mean()), "ehit": float(ehit.mean()),
        "stated": float(np.mean(stated)), "logloss": float(np.mean(lls)),
        "calib_gap": float(np.mean(stated) - np.mean(hits)),
    }


def main():
    cfg = load_config(CONFIG)
    db = os.environ.get("HIGHCAST_DB")
    if db:
        cfg = dataclasses.replace(cfg, db_path=Path(db))
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, METRIC)

    dates = [d for (d,) in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? AND high_f IS NOT NULL "
        "ORDER BY date", (STATION,))]
    k = cfg.model.intraday_bandwidth_for(METRIC)
    train_hours = tuple(sorted({(HOUR + d) % 24 for d in range(-k, k + 1)}))
    print(f"{STATION} {METRIC.upper()} h{HOUR}  bandwidth k={k} -> train hours "
          f"{train_hours}   days={len(dates)}")

    # ONE build. `baseline=None` reproduces the house method exactly -> LEAK arm.
    fr = build_checkpoint_frame(conn, cfg, dates, hours=train_hours, metric=spec)
    fr = fr[fr["y"].notna()].copy()
    print(f"frame rows: {len(fr)}")

    day_df = build_features(conn, cfg, spec, dates, require_target=True)
    extra = [c for c in features_for(STATION, METRIC) if c not in INTRADAY_FEATURES]
    resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
    print(f"extra features: {extra or 'none'}   residual={resid}\n")

    for split in SPLITS:
        # HONEST arm: blend fitted on the training days only, exactly as the live
        # path does, then applied to every day.
        tr_days = day_df[day_df["target_date"] < split]
        base = BaselineBlend(
            prefer_corrected=uses_bias_correction(spec.key)).fit(tr_days)
        qs = base.predict_quantiles(day_df)
        p50 = dict(zip(day_df["target_date"], [q[0.5] for q in qs]))
        spread = dict(zip(day_df["target_date"],
                          [q[0.9] - q[0.1] for q in qs]))

        honest = fr.copy()
        honest["base_p50"] = honest["target_date"].map(p50)
        honest["base_spread"] = honest["target_date"].map(spread)

        print(f"=== split {split}  (train < split <= test, test at h{HOUR}) ===")
        print(f"{'arm':>7}{'n':>6}{'MAE':>8}{'bias':>8}{'CRPS':>8}"
              f"{'hit':>10}{'stated':>9}{'gap':>8}{'logloss':>9}")
        rows = {}
        for name, frame in (("LEAK", fr), ("HONEST", honest)):
            tr = frame[frame["target_date"] < split]
            te = frame[(frame["target_date"] >= split)
                       & (frame["hour"] == float(HOUR))]
            if len(tr) < 400 or te.empty:
                print(f"{name:>7}  skipped (train {len(tr)}, test {len(te)})")
                continue
            m = IntradayModel(extremum=spec.extremum, residual=resid,
                              extra_features=extra).fit(tr)
            pmfs = [bucket_probs(q) for q in m.predict_quantiles(te)]
            s = score(pmfs, te["y"].to_numpy(dtype=float))
            rows[name] = s
            print(f"{name:>7}{s['n']:>6}{s['mae']:>8.3f}{s['bias']:>+8.3f}"
                  f"{s['crps']:>8.3f}{s['ehit']:>9.1%}{s['stated']:>9.1%}"
                  f"{s['calib_gap']:>+8.1%}{s['logloss']:>9.3f}")
        if len(rows) == 2:
            a, b = rows["LEAK"], rows["HONEST"]
            print(f"{'DELTA':>7}{'':>6}{b['mae'] - a['mae']:>+8.3f}"
                  f"{b['bias'] - a['bias']:>+8.3f}{b['crps'] - a['crps']:>+8.3f}"
                  f"{b['ehit'] - a['ehit']:>+9.1%}{b['stated'] - a['stated']:>+9.1%}"
                  f"{b['calib_gap'] - a['calib_gap']:>+8.1%}"
                  f"{b['logloss'] - a['logloss']:>+9.3f}")
            print("  (DELTA = HONEST - LEAK; positive MAE/CRPS/logloss means the "
                  "leak was flattering us)")
        print()


if __name__ == "__main__":
    sys.exit(main())
