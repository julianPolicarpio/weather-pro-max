"""Measured CALIBRATION offsets at the commit hour, per station x metric.

WHY: the Pick-of-the-Day EV bar needs p_hat = a probability we believe, not the
model's stated one. HANDOFF says the KLAX HIGH is overconfident by 5.8 points at
its commit hour (states 56.8%, delivers 51.1%) and the LOW is +0.7. Those two
numbers were about to be HARD-CODED into the scorer from a prose document, and
KMIA has no such number at all -- which would leave two of the four markets
unscorable. This measures all four properly.

ALSO CHECKS THE BAND PENALTY METHODOLOGY. The earlier seam run found KLAX HIGH
readings >=80 hit only 41.3% vs 51.0% below. Subtracting that 9.7-point BASE-RATE
gap from p_model would be wrong -- it conflates "this band is harder" with "the
model is overconfident in this band". The right quantity is the band's own
calib (stated - realised). This reports both so the scorer uses the correct one.

Regime is identical to seam_eval.py / hour_curve.py: matched-hour training at the
configured bandwidth, IntradayModel with the configured residual flag, walk-forward
with periodic refit, parity-averaged bucket hit.
"""
import sys

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                       build_checkpoint_frame, features_for)

REFIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
# Test dates before this are still WALKED (they train the later folds) but are
# reported separately. Production fits on ALL available history, so a calib
# averaged over early folds -- where the expanding window held ~400 rows against
# production's ~1,800 -- is PESSIMISTIC for the model that actually runs.
# `gating.py`'s single-holdout comments (KMIA high +4.3%, low +1.2%) versus this
# script's full-window figures (+10.1%, +5.6%) is exactly that gap.
RECENT_FROM = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
MIN_TRAIN = 400
N_BOOT = 2000
CSV_OUT = "/tmp/calib_records.csv"
# ⚠️ TRUNCATED ONCE AT STARTUP, then appended per market.
# This file used to be opened in "a" mode with no truncation, so consecutive
# runs PILED UP: after three runs it held three copies of every record, from
# three DIFFERENT models (the 2026-07-30 run, a broken pre-RATE re-run, and the
# corrected one). A band table derived from that mixture is meaningless, and the
# duplication is invisible unless you count distinct keys. Caught 2026-07-31.
_CSV_STARTED = False
# HEADER, written once at the top of the dump. `band_calib_eval.py` reads this
# file with csv.DictReader, which takes the FIRST LINE as its field names -- so a
# headerless dump silently eats its first record AND mislabels every column.
# Until 2026-08-01 the header had to be prepended BY HAND between running the two
# scripts: an undocumented manual step, and the kind that gets forgotten exactly
# once. Kept on ONE line as a named constant so the contract test can read it.
CSV_HEADER = "station,metric,hour,target_date,model_pt,y,stated,hit"
rng = np.random.default_rng(20260730)


def stated_prob(pmf):
    """Stated probability on the modal bucket, parity-averaged -- same estimator
    hour_curve uses, so `stated` here is comparable to `hit`."""
    return float(np.mean([modal_bucket(pmf, par)[1] for par in (0, 1)]))


def ci(a):
    if len(a) < 5:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return tuple(np.percentile(a[idx].mean(axis=1), [2.5, 97.5]))


print(f"refit every {REFIT}d  min_train={MIN_TRAIN}\n", flush=True)
results = {}

for config in ("config/klax.yaml", "config/kmia.yaml"):
    cfg = load_config(config)
    conn = connect(cfg.db_path)
    icao = cfg.station.icao
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (icao,))]
    for metric in ("low", "high"):
        h = MIN_DECISION_HOUR[icao][metric]
        spec = get_metric(cfg, metric)
        resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
        k = cfg.model.intraday_bandwidth_for(metric)
        tr_hours = tuple(sorted({(h + d) % 24 for d in range(-k, k + 1)}))
        fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec)
        te_all = fr[(fr["hour"] == float(h)) & fr["y"].notna()].sort_values("target_date")
        td = te_all["target_date"].to_numpy()
        start = next((i for i, d in enumerate(td)
                      if fr[(fr["target_date"] < d) & fr["y"].notna()].shape[0] >= MIN_TRAIN),
                     None)
        if start is None:
            print(f"{icao} {metric}: never enough training rows"); continue
        recs = []
        for b0 in range(start, len(td), REFIT):
            cut = td[b0]
            tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
            if len(tr) < MIN_TRAIN:
                continue
            # ⚠️ MUST USE THE SHIPPED FEATURE SET FOR THIS MARKET.
            # This line used to be a bare `IntradayModel(...)`, which pins the
            # model to `INTRADAY_FEATURES` and therefore measures a model NOBODY
            # IS RUNNING. KLAX HIGH has carried `RATE_FEATURES` since
            # 2026-07-31, so the stale version re-measured the PRE-rate model
            # and would have written a wrong number into `HOUR_ACCURACY` --
            # which `BAND_CALIB`, the reliability table and the preflight bands
            # all read. Caught 2026-07-31 before it shipped.
            _extra = [c for c in features_for(icao, spec.key)
                      if c not in INTRADAY_FEATURES]
            model = IntradayModel(extremum=spec.extremum, residual=resid,
                                  extra_features=_extra).fit(tr)
            blk = te_all.iloc[b0:b0 + REFIT]
            for (_, row), q in zip(blk.iterrows(), model.predict_quantiles(blk)):
                pmf = bucket_probs(q)
                recs.append((row["target_date"], float(q[0.5]), float(row["y"]),
                             stated_prob(pmf), expected_bucket_hit(pmf, int(row["y"]))))
        dates = np.array([r[0] for r in recs])
        pt = np.array([r[1] for r in recs])
        stated = np.array([r[3] for r in recs])
        hit = np.array([r[4] for r in recs])
        cal = stated - hit
        # NO `global` HERE. `_CSV_STARTED` is module scope (line 49) and so is
        # this loop, so the declaration was both unnecessary and illegal --
        # "name assigned to before global declaration" is a SyntaxError, which
        # made the entire script unrunnable from the moment the append-mode fix
        # landed on 2026-07-31. Same class as the `icao` NameError that had been
        # sitting in `hour_curve.py`: a fix that was written but never executed.
        with open(CSV_OUT, "a" if _CSV_STARTED else "w") as fh:
            if not _CSV_STARTED:
                fh.write(CSV_HEADER + "\n")
            _CSV_STARTED = True
            for r in recs:
                fh.write(f"{icao},{metric},{h},{r[0]},{r[1]:.4f},{r[2]:.2f},"
                         f"{r[3]:.6f},{r[4]:.6f}\n")
        l, u = ci(cal)
        rec_m = dates >= RECENT_FROM
        rl, ru = ci(cal[rec_m]) if rec_m.sum() >= 5 else (float("nan"),) * 2
        results[(icao, metric)] = dict(
            h=h, n=len(recs), stated=stated.mean(), hit=hit.mean(),
            cal=cal.mean(), ci=(l, u), resid=resid, k=k,
            n_recent=int(rec_m.sum()),
            cal_recent=float(cal[rec_m].mean()) if rec_m.sum() else float("nan"),
            hit_recent=float(hit[rec_m].mean()) if rec_m.sum() else float("nan"),
            ci_recent=(rl, ru))
        print(f"{icao} {metric:4s} h{h:<2d} resid={resid} k={k}", flush=True)
        print(f"      FULL   n={len(recs):<5} stated {stated.mean():.1%}  "
              f"realised {hit.mean():.1%}  CALIB {cal.mean():+.1%} "
              f"[{l:+.1%},{u:+.1%}]", flush=True)
        print(f"      RECENT n={rec_m.sum():<5} (>= {RECENT_FROM}) "
              f"realised {hit[rec_m].mean():.1%}  CALIB {cal[rec_m].mean():+.1%} "
              f"[{rl:+.1%},{ru:+.1%}]   <-- matches production's data volume",
              flush=True)
        # band split -- only meaningful for the KLAX HIGH >=80 seam
        if icao == "KLAX" and metric == "high":
            for lab, m in (("model >=80", pt >= 80), ("model <80", pt < 80)):
                if m.sum() < 5:
                    continue
                bl, bu = ci(cal[m])
                print(f"      {lab:11s} n={m.sum():>4}  stated {stated[m].mean():.1%}  "
                      f"realised {hit[m].mean():.1%}  CALIB {cal[m].mean():+.1%} "
                      f"[{bl:+.1%},{bu:+.1%}]", flush=True)

print("\n=== what the scorer should use ===")
print("p_hat = p_model - CALIB(station, metric, commit_hour)")
print("CALIB > 0 means the model OVERSTATES its modal bucket and must be shaded down.")
print("Use the RECENT column: production trains on all history, so the early")
print("data-poor folds in the FULL column understate the shipped model.")
print(f"{'market':18s}{'CALIB full':>12}{'CALIB recent':>14}{'n recent':>10}{'hit recent':>12}")
for (icao, metric), d in sorted(results.items()):
    print(f"{icao+' '+metric+' h'+str(d['h']):18s}{d['cal']:>+12.4f}"
          f"{d['cal_recent']:>+14.4f}{d['n_recent']:>10}{d['hit_recent']:>12.1%}")
print(f"\nper-record data persisted to {CSV_OUT} -- do NOT recompute, load it.")
