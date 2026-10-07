"""Does band calibration still help on days it was NOT fitted on?

The bands in `gating.BAND_CALIB` were measured on the whole recent window, so
their in-sample residual is ~0 BY CONSTRUCTION. That proves nothing. This fits
the bands on the FIRST HALF and scores them on the SECOND, against two baselines:

    RAW      no correction at all
    SCALAR   the shipped single-number `HOUR_CALIB`
    BAND     the curve

Scored on absolute calibration error |stated - hit| aggregated over bands, and
on Brier score. If BAND does not beat SCALAR out of sample, the extra machinery
is not earning its place and the scalar should stay.
"""
import csv
from collections import defaultdict

import numpy as np

from highcast.gating import HOUR_CALIB, MIN_DECISION_HOUR

CSV = "data/calib_records-20260730.csv"
RECENT = "2025-01-01"
EDGES = [0.0, 0.50, 0.60, 0.70, 0.80, 1.01]
rng = np.random.default_rng(20260801)


def fit_bands(stated, hit):
    out = []
    for a, b in zip(EDGES, EDGES[1:]):
        m = (stated >= a) & (stated < b)
        out.append((b, float((stated[m] - hit[m]).mean()) if m.sum() >= 20 else None,
                    int(m.sum())))
    return out


def apply_bands(bands, p):
    for upper, off, _ in bands:
        if p < upper:
            return None if off is None else min(0.999, max(0.001, p - off))
    return None


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["target_date"] >= RECENT]
    by = defaultdict(list)
    for r in rows:
        by[(r["station"], r["metric"], int(r["hour"]))].append(
            (r["target_date"], float(r["stated"]), float(r["hit"])))

    agg = {"RAW": [], "SCALAR": [], "BAND": []}
    for key, recs in sorted(by.items()):
        station, metric, hour = key
        if MIN_DECISION_HOUR.get(station, {}).get(metric) != hour:
            continue
        recs.sort()
        half = len(recs) // 2
        tr, te = recs[:half], recs[half:]
        bands = fit_bands(np.array([s for _, s, _ in tr]),
                          np.array([h for _, _, h in tr]))
        scalar = HOUR_CALIB.get(station, {}).get(metric, {}).get(hour) or 0.0
        s_te = np.array([s for _, s, _ in te])
        h_te = np.array([h for _, _, h in te])
        variants = {
            "RAW": s_te,
            "SCALAR": np.clip(s_te - scalar, 0.001, 0.999),
            "BAND": np.array([apply_bands(bands, p) if apply_bands(bands, p)
                              is not None else np.nan for p in s_te]),
        }
        print(f"\n=== {station} {metric.upper()} h{hour}  "
              f"train {tr[0][0]}..{tr[-1][0]}  test {te[0][0]}..{te[-1][0]} "
              f"(n={len(te)}) ===")
        print(f"{'variant':>8}{'n scored':>10}{'|calib err|':>13}{'Brier':>9}")
        for name, p in variants.items():
            ok = ~np.isnan(p)
            # aggregate calibration error over the SAME fixed bands
            errs, wts = [], []
            for a, b in zip(EDGES, EDGES[1:]):
                m = ok & (p >= a) & (p < b)
                if m.sum() >= 15:
                    errs.append(abs(p[m].mean() - h_te[m].mean()))
                    wts.append(m.sum())
            ce = float(np.average(errs, weights=wts)) if errs else float("nan")
            brier = float(np.mean((p[ok] - h_te[ok]) ** 2))
            agg[name].append((ce, brier, int(ok.sum())))
            print(f"{name:>8}{ok.sum():>10}{ce:>13.4f}{brier:>9.4f}")
        skipped = int(np.isnan(variants['BAND']).sum())
        if skipped:
            print(f"          BAND declined to score {skipped} record(s) whose "
                  f"band had <20 training rows — those become REFUSALS live.")

    print("\n\n=== POOLED OVER THE FOUR MARKETS (out of sample) ===")
    print(f"{'variant':>8}{'|calib err|':>13}{'Brier':>9}")
    for name, v in agg.items():
        ce = float(np.mean([x[0] for x in v]))
        br = float(np.mean([x[1] for x in v]))
        print(f"{name:>8}{ce:>13.4f}{br:>9.4f}")
    best = min(agg, key=lambda k: np.mean([x[0] for x in agg[k]]))
    print(f"\n  lowest out-of-sample calibration error: {best}")
    if best != "BAND":
        print("  ⚠️ BAND does not win out of sample — the scalar should stay.")


if __name__ == "__main__":
    main()
