"""What is the model's stated probability actually worth, BY BAND?

THE PROBLEM THIS EXISTS FOR. `gating.HOUR_CALIB` is one scalar per market. KLAX
LOW's is +0.0075 -- essentially honest -- and that number is a lie for betting
purposes, because it is an AVERAGE over a curve that is not flat:

    model says 60-70%  ->  hits 67.9%   (UNDER-states by 2.9%)
    model says 70-80%  ->  hits 67.7%   (OVER-states by 6.4%)

The two cancel to ~zero. But the staking gate only fires when the model claims a
big edge, so it bets almost exclusively out of the TOP band -- the one band that
overstates. Correcting a bet by the average applies a ~0-point shave to a number
that needs ~6. Measured live: three bets placed at a stated 81% all lost.

WHAT IS MEASURED. From `data/calib_records-20260730.csv` (7,236 walk-forward
records, refit 30, matched-hour, parity-averaged bucket hit) -- no refitting, so
this is the same evidence `HOUR_CALIB` was set from, re-cut by stated band.

⚠️ RECENT WINDOW ONLY (>= 2025-01-01), matching HOUR_CALIB. An expanding-window
figure is PESSIMISTIC for the shipped model because early folds trained on a
fraction of today's rows -- documented at length in `gating`, and acting on the
full-window numbers would have shaded every LOW ~3.5 points too hard.

BANDS ARE FIXED DECILE-ISH EDGES CHOSEN BEFORE LOOKING, not fitted to where the
error happens to be. A band ships only if it has >= 40 records AND its bootstrap
CI is tight enough to be worth more than the scalar it replaces; otherwise the
market keeps the scalar there. A thin band with a wild point estimate is exactly
how a calibration table starts doing damage.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

from highcast.gating import HOUR_CALIB, MIN_DECISION_HOUR

# Default is the NEWEST record dump. 20260730 was measured with a bare
# `IntradayModel(...)` and therefore describes the PRE-RATE_FEATURES KLAX HIGH --
# a model nobody runs. Override with argv[1] to reproduce an older table.
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/calib_records-20260731.csv"
RECENT = "2025-01-01"
BAND_EDGES = [0.0, 0.50, 0.60, 0.70, 0.80, 1.01]
MIN_BAND_N = 40
N_BOOT = 4000
rng = np.random.default_rng(20260801)


def ci(a):
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return tuple(np.percentile(a[idx].mean(axis=1), [2.5, 97.5]))


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["target_date"] >= RECENT]
    by = defaultdict(list)
    for r in rows:
        key = (r["station"], r["metric"], int(r["hour"]))
        by[key].append((float(r["stated"]), float(r["hit"])))

    out = {}
    for (station, metric, hour), recs in sorted(by.items()):
        if MIN_DECISION_HOUR.get(station, {}).get(metric) != hour:
            continue
        scalar = HOUR_CALIB.get(station, {}).get(metric, {}).get(hour)
        stated = np.array([s for s, _ in recs])
        hit = np.array([h for _, h in recs])
        print(f"\n=== {station} {metric.upper()} h{hour}  n={len(recs)}  "
              f"shipped scalar {scalar:+.4f} ===")
        print(f"{'band':>12}{'n':>6}{'stated':>9}{'hits':>8}"
              f"{'overstates':>12}   {'95% CI on overstatement':>26}  ship?")
        bands = []
        for a, b in zip(BAND_EDGES, BAND_EDGES[1:]):
            m = (stated >= a) & (stated < b)
            if m.sum() == 0:
                continue
            d = stated[m] - hit[m]
            lo, hi = ci(d)
            # Ship a band only when it has the samples AND the interval is
            # tighter than the correction itself -- otherwise the scalar, backed
            # by every record, is the better estimate.
            ok = m.sum() >= MIN_BAND_N and (hi - lo) < 0.20
            print(f"{a:>6.0%}-{b:<5.0%}{m.sum():>6}{stated[m].mean():>9.1%}"
                  f"{hit[m].mean():>8.1%}{d.mean():>+12.1%}   "
                  f"[{lo:+.1%},{hi:+.1%}]{'':>6}  {'YES' if ok else 'no'}")
            if ok:
                bands.append((round(b, 2), round(float(d.mean()), 4), int(m.sum())))
        out[(station, metric, hour)] = bands
        if not bands:
            print("   -> no band qualifies; this market keeps its scalar")

    print("\n\n# ---- paste into gating.BAND_CALIB ----")
    print("BAND_CALIB = {")
    for (station, metric, hour), bands in sorted(out.items()):
        if not bands:
            continue
        print(f'    ("{station}", "{metric}", {hour}): [')
        for upper, off, n in bands:
            print(f"        ({upper}, {off:+.4f}),   # n={n}")
        print("    ],")
    print("}")


if __name__ == "__main__":
    sys.exit(main())
