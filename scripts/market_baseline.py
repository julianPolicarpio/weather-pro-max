"""PROMPT-2 BASELINE for any one market, nested walk-forward + frozen window.

WHAT SERVES A MARKET AT ITS COMMIT HOUR. The commit hour is read from
`gating.MIN_DECISION_HOUR`, never passed in -- at that hour `preferred_prediction`
returns the NOWCAST for both metrics (the HIGH is past its knee there by
definition, and the LOW never prefers the day-ahead at all). So the model
measured is always `IntradayModel`, configured from the station's own yaml
(`intraday_residual_*`, `intraday_hour_bandwidth`), plus the `BaselineBlend` that
supplies its `base_p50`/`base_spread`. The day-ahead / hybrid / PIT / regime
stack is NOT on any market's trading path at its commit hour -- it drives the
pre-knee display only.

EVERY PER-MARKET SETTING IS READ, NEVER ASSUMED. The two markets measured so far
differ on five axes at once -- residual target, bias-corrected guidance, NBS aux
features, hour bandwidth, and commit hour -- so a script that hardcoded any of
them would silently measure a model nobody runs. That is the same failure
`hour_curve.py` documents (a bare `IntradayModel(...)` pinned to the stale
feature set) and `orderbot`'s duplicated `CONFIDENCE_FLOOR` (a stale copy that
kept binding after the original moved).

WHY NESTED. Every tuned constant this market runs on -- the commit hour (from
HOUR_ACCURACY), the band-calibration offsets, the confidence floor -- was fitted
on the same 2020-2026 span that any naive holdout would score on. A plain
train/test split therefore reports the tuning back to itself. Here:

  * the model is refit per fold on days strictly before the fold
  * the `BaselineBlend` is fitted per fold on TRAIN ONLY (production behaviour;
    the house method fits it on train+test -- see `baseline_leak_audit.py`,
    which measured that leak at dMAE +0.001, i.e. immaterial, but it costs
    nothing to do correctly)
  * the BAND CALIBRATION is fitted ONLINE, on the pooled out-of-sample records
    of PRIOR folds only, and a band with too few records REFUSES rather than
    falling back to the raw number -- matching `gating.band_calib`'s contract

FROZEN WINDOW. 2026-06-01 onward has not been used to tune anything. n is ~61
days per station, which can catch something badly broken and cannot establish an
edge. It is reported separately and never pooled with the walk-forward.

WHAT THIS CANNOT MEASURE. Prompt 2's third baseline metric -- backtested P&L and
per-trade edge under the commit policy -- needs the market price at our decision
time. `signals.market_json` starts 2026-07-24; no Kalshi price series exists
behind that. So P&L, the timing test, and any tuning of MIN_EDGE or the max-edge
investigation threshold are out of reach. The priced-day count is printed at the
end of every run rather than the metric being estimated on it.

RELIABILITY IS SCORED SINGLE-PARITY. Kalshi runs one ladder parity per day;
`expected_bucket_hit` averages the two, which is right for scoring the model and
kinder than the bot's reality, because a confident-and-right parity can carry a
confident-and-wrong one into the same band. HANDOFF §2.6 is where that averaging
reported the model +14.9 pts UNDER-confident when the real quantity was -4.2 pts
OVER. Both conventions are printed so the gap stays visible.

Usage:
    HIGHCAST_DB=/path/to/copy.db .venv/bin/python scripts/market_baseline.py \
        [config/kmia.yaml] [high|low]

Defaults reproduce the KMIA HIGH figures this file was first written to produce
(walk-forward MAE 0.551 / hit 77.6%, frozen MAE 0.450 / hit 81.1%).

Reads only, but `db.connect` runs idempotent DDL -- point HIGHCAST_DB at a COPY
rather than racing the 5-minute wake cycle on the live file.

⚠️ MEASUREMENT ONLY. KLAX is frozen mid-test-week; running this against it reads
the model and changes nothing.
"""
import dataclasses
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import BAND_CALIB, CONFIDENCE_FLOOR, MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml"
METRIC = (sys.argv[2] if len(sys.argv) > 2 else "high").lower()
WF_START = "2023-01-01"      # first fold's test period begins here
FREEZE = "2026-06-01"        # walk-forward stops here; the rest is the frozen window
STEP_DAYS = 90               # refit cadence
MIN_TRAIN = 400              # matches models.intraday.MIN_MATCHED_ROWS
# Band edges for the reliability curve, matching `gating.BAND_CALIB`'s own.
EDGES = [0.0, 0.50, 0.60, 0.70, 0.80, 1.01]
MIN_BAND_N = 20              # below this a band REFUSES, per band_calib's contract


def folds(start: str, end: str, step: int):
    a = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while a < stop:
        b = min(a + timedelta(days=step), stop)
        yield a.isoformat(), b.isoformat()
        a = b


def ladder_records(pmf, y, pt):
    """One record per (day, parity) -- what the bot actually faces.

    Returns (stated, hit, p_true, boundary_dist) per parity:
      stated   mass on the model's MODAL bucket (the number the gate reads)
      hit      did the modal bucket contain y
      p_true   mass on the bucket that actually contained y (for log loss --
               scoring the modal bucket instead would reward confidence itself)
      boundary_dist  how far the point forecast sits from the nearest ladder
               edge, 0.0 = on the boundary, 1.0 = dead centre. Prompt 2: a
               0.4F gain deep inside a bucket changes no trade; the same gain
               near an edge flips the traded bucket.

    ⚠️ THE BOUNDARY IS AT A HALF-DEGREE, NOT AT AN INTEGER. The settled value is
    an INTEGER, so on a parity-p ladder the buckets are [lo, lo+1], [lo+2, lo+3]
    and the outcome only changes bucket between the integers lo+1 and lo+2 --
    i.e. at lo+1.5 in continuous point-forecast space. Measuring distance to the
    integer edge `lo` instead (the obvious-looking formula) puts the "boundary"
    half a degree from where a trade actually flips, and then bins days by
    whether `pt` sits near an integer -- which is a proxy for how concentrated
    the pmf is, not for boundary risk. That produced a table showing days NEAR
    the boundary as MORE accurate, which is the confound, not a finding.
    """
    out = []
    for parity in (0, 1):
        lo, mass = modal_bucket(pmf, parity)
        true_lo = int(y) - ((int(y) - parity) % 2)
        p_true = sum(v for k, v in pmf.items() if true_lo <= k <= true_lo + 1)
        d = (pt - (parity + 1.5)) % 2.0
        dist = min(d, 2.0 - d)
        out.append((mass, 1.0 if lo <= int(y) <= lo + 1 else 0.0, p_true, dist))
    return out


def fit_bands(recs):
    """(upper, offset) per band from out-of-sample (stated, hit) pairs.

    A band with < MIN_BAND_N records gets offset None = REFUSE. `gating.band_calib`
    returns None in exactly this case and the docstring is emphatic that None
    means refuse, never "use the raw number" -- so the honest simulation of that
    policy has to be able to refuse too.
    """
    stated = np.array([r[0] for r in recs])
    hit = np.array([r[1] for r in recs])
    bands = []
    for a, b in zip(EDGES, EDGES[1:]):
        m = (stated >= a) & (stated < b)
        bands.append((b, float((stated[m] - hit[m]).mean())
                      if m.sum() >= MIN_BAND_N else None, int(m.sum())))
    return bands


def apply_bands(bands, p):
    for upper, off, _ in bands:
        if p < upper:
            return None if off is None else min(0.999, max(0.001, p - off))
    return None


def reliability(recs, label):
    print(f"\n  reliability ({label}) -- single-parity day-ladders")
    print(f"  {'band':>12}{'n':>7}{'stated':>9}{'actual':>9}{'gap':>8}")
    stated = np.array([r[0] for r in recs])
    hit = np.array([r[1] for r in recs])
    for a, b in zip(EDGES, EDGES[1:]):
        m = (stated >= a) & (stated < b)
        if not m.any():
            continue
        s, h = stated[m].mean(), hit[m].mean()
        flag = "  <- n too thin to trust" if m.sum() < MIN_BAND_N else ""
        print(f"  {a:.2f}-{min(b,1.0):.2f}{m.sum():>7}{s:>9.1%}{h:>9.1%}"
              f"{s - h:>+8.1%}{flag}")
    print(f"  {'ALL':>12}{len(stated):>7}{stated.mean():>9.1%}"
          f"{hit.mean():>9.1%}{stated.mean() - hit.mean():>+8.1%}")


def report(tag, days):
    """days: list of (target_date, pmf, y, pt)."""
    if not days:
        print(f"\n### {tag}: no rows")
        return []
    ys = np.array([d[2] for d in days], dtype=float)
    pts = np.array([d[3] for d in days], dtype=float)
    err = pts - ys
    crps = np.array([crps_pmf(p, int(y)) for _, p, y, _ in days])
    ehit = np.array([expected_bucket_hit(p, int(y)) for _, p, y, _ in days])
    recs = [r for _, p, y, pt in days for r in ladder_records(p, y, pt)]
    ll = np.array([-np.log(max(r[2], 1e-9)) for r in recs])
    hit = np.array([r[1] for r in recs])

    print(f"\n### {tag}   days={len(days)}  day-ladders={len(recs)}")
    print(f"  MAE {np.abs(err).mean():.3f} F   bias {err.mean():+.3f} F   "
          f"CRPS {crps.mean():.3f}   log loss {ll.mean():.3f}")
    print(f"  bucket hit: {ehit.mean():.1%} (parity-averaged)   "
          f"{hit.mean():.1%} (single-parity)")
    reliability(recs, tag)

    print(f"\n  accuracy by distance from the point forecast to the nearest "
          f"ladder boundary\n  (0.00 = sitting on a boundary, 1.00 = mid-bucket; "
          f"the two parities are complementary,\n   so each day contributes one "
          f"record at d and one at 1-d, and MAE repeats across\n   the mirrored "
          f"bins by construction -- read HIT and STATED, not MAE)")
    print(f"  {'dist (F)':>12}{'n':>7}{'MAE':>8}{'hit':>8}{'stated':>9}")
    dists = np.array([r[3] for r in recs])
    st = np.array([r[0] for r in recs])
    day_ae = np.repeat(np.abs(err), 2)
    for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (dists >= a) & (dists < b)
        if not m.any():
            continue
        print(f"  {a:.2f}-{min(b,1.0):.2f}{m.sum():>7}{day_ae[m].mean():>8.3f}"
              f"{hit[m].mean():>8.1%}{st[m].mean():>9.1%}")
    return recs


def main():
    cfg = load_config(CONFIG)
    db = os.environ.get("HIGHCAST_DB")
    if db:
        cfg = dataclasses.replace(cfg, db_path=Path(db))
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, METRIC)
    station = cfg.station.icao

    # The commit hour is READ, never passed in: a literal here would drift from
    # the gate the way `orderbot`'s duplicated CONFIDENCE_FLOOR did.
    hour = MIN_DECISION_HOUR.get(station, {}).get(METRIC)
    if hour is None:
        print(f"{station} {METRIC}: no measured decision hour in "
              f"gating.MIN_DECISION_HOUR -- refusing to invent one.")
        return 1

    # The settled column follows the metric. Reading high_f for a LOW run would
    # silently score against the wrong target on every row.
    col = "high_f" if METRIC == "high" else "low_f"
    dates = [d for (d,) in conn.execute(
        f"SELECT date FROM cli_reports WHERE station=? AND {col} IS NOT NULL "
        "ORDER BY date", (station,))]
    # PER-METRIC as of 2026-08-02. Reading the station-wide field here would
    # score KLAX LOW at k=0 while production runs it at k=1 -- measuring a model
    # nobody runs, the exact failure `hour_curve.py`'s own comment warns about.
    k = cfg.model.intraday_bandwidth_for(METRIC)
    train_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
    resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
    extra = [c for c in features_for(station, METRIC) if c not in INTRADAY_FEATURES]
    corrected = uses_bias_correction(spec.key)

    print(f"{station} {METRIC.upper()} h{hour}  |  nowcast serves at the commit "
          f"hour (knee={hour}, extremum={spec.extremum})")
    print(f"bandwidth k={k} -> train hours {train_hours}   residual={resid}   "
          f"bias-corrected guidance={corrected}   extra features: {extra or 'none'}")
    print(f"days with a settled CLI {METRIC}: {len(dates)}  "
          f"({dates[0]} -> {dates[-1]})")
    bands = BAND_CALIB.get((station, METRIC, hour))
    print(f"shipped BAND_CALIB: {len(bands) if bands else 0} bands   "
          f"shipped CONFIDENCE_FLOOR: "
          f"{CONFIDENCE_FLOOR.get((station, METRIC, hour), 'absent')}")

    fr = build_checkpoint_frame(conn, cfg, dates, hours=train_hours, metric=spec)
    fr = fr[fr["y"].notna()].copy()
    day_df = build_features(conn, cfg, spec, dates, require_target=True)

    # ── guidance channels, diagnostic only (Prompt 2: never the target) ──────
    # Both variants are printed when the market uses bias correction, because
    # `fcc_*` is what the blend actually consumes there and `fc_*` alone would
    # describe an input this market does not read.
    prefixes = ("fc_", "fcc_") if corrected else ("fc_",)
    print(f"\n### guidance channels vs settled CLI {METRIC} (day-ahead-legal runs)")
    print(f"  {'channel':>14}{'n':>7}{'MAE':>8}{'bias':>8}{'lag1 r':>9}")
    skip = {"fc_spread", "fc_mean", "fc_anchor", "fcc_mean"}
    y = day_df["y"].to_numpy(dtype=float)
    for col in sorted(c for c in day_df.columns
                      if any(c.startswith(p) for p in prefixes) and c not in skip):
        v = day_df[col].to_numpy(dtype=float)
        m = ~np.isnan(v) & ~np.isnan(y)
        if m.sum() < 30:
            continue
        e = v[m] - y[m]
        # lag-1 autocorrelation of the ERROR is the criterion `features.py` uses
        # to justify bias correction at all: a tree absorbs a CONSTANT offset by
        # moving its split thresholds, but not a DRIFTING one. r ~0.8 justified
        # `fcc_*`; r ~0.2-0.4 says the offset is static and needs no column.
        r = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if m.sum() > 2 else float("nan")
        print(f"  {col:>14}{m.sum():>7}{np.abs(e).mean():>8.3f}"
              f"{e.mean():>+8.3f}{r:>9.2f}")

    # ── nested walk-forward ─────────────────────────────────────────────────
    wf_days, prior_recs, calibrated = [], [], []
    n_refuse = 0
    for start, end in folds(WF_START, FREEZE, STEP_DAYS):
        tr_days = day_df[day_df["target_date"] < start]
        if len(tr_days) < MIN_TRAIN:
            continue
        base = BaselineBlend(
            prefer_corrected=uses_bias_correction(spec.key)).fit(tr_days)
        qs = base.predict_quantiles(day_df)
        p50 = dict(zip(day_df["target_date"], [q[0.5] for q in qs]))
        spr = dict(zip(day_df["target_date"], [q[0.9] - q[0.1] for q in qs]))
        f = fr.copy()
        f["base_p50"] = f["target_date"].map(p50)
        f["base_spread"] = f["target_date"].map(spr)

        tr = f[f["target_date"] < start]
        te = f[(f["target_date"] >= start) & (f["target_date"] < end)
               & (f["hour"] == float(hour))]
        if len(tr) < MIN_TRAIN or te.empty:
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(tr)
        pmfs = [bucket_probs(q) for q in m.predict_quantiles(te)]

        # ONLINE band calibration: fitted on PRIOR folds' out-of-sample records
        # only. This is both honest and the form production could actually run.
        bands = fit_bands(prior_recs) if len(prior_recs) >= MIN_BAND_N else None
        rows = []
        for (_, r), pmf in zip(te.iterrows(), pmfs):
            pt = sum(kk * vv for kk, vv in pmf.items()) / max(sum(pmf.values()), 1e-9)
            rows.append((r["target_date"], pmf, float(r["y"]), pt))
            for stated, hit, _p_true, _d in ladder_records(pmf, r["y"], pt):
                if bands is None:
                    n_refuse += 1
                    continue
                c = apply_bands(bands, stated)
                if c is None:
                    n_refuse += 1
                else:
                    calibrated.append((c, hit))
        wf_days.extend(rows)
        prior_recs.extend([r for _, p, y, pt in rows
                           for r in ladder_records(p, y, pt)])

    report(f"WALK-FORWARD {WF_START} -> {FREEZE} (nested, {STEP_DAYS}d refit)",
           wf_days)

    if calibrated:
        c = np.array([x[0] for x in calibrated])
        h = np.array([x[1] for x in calibrated])
        print(f"\n  after ONLINE band calibration: n={len(c)} "
              f"(refused {n_refuse}), stated {c.mean():.1%}, "
              f"actual {h.mean():.1%}, gap {c.mean() - h.mean():+.1%}")
        for a, b in zip(EDGES, EDGES[1:]):
            msk = (c >= a) & (c < b)
            if msk.sum():
                print(f"    {a:.2f}-{min(b,1.0):.2f}  n={msk.sum():>5}  "
                      f"stated {c[msk].mean():.1%}  actual {h[msk].mean():.1%}  "
                      f"gap {c[msk].mean() - h[msk].mean():+.1%}")

    # ── frozen window ───────────────────────────────────────────────────────
    tr_days = day_df[day_df["target_date"] < FREEZE]
    base = BaselineBlend(
        prefer_corrected=uses_bias_correction(spec.key)).fit(tr_days)
    qs = base.predict_quantiles(day_df)
    f = fr.copy()
    f["base_p50"] = f["target_date"].map(
        dict(zip(day_df["target_date"], [q[0.5] for q in qs])))
    f["base_spread"] = f["target_date"].map(
        dict(zip(day_df["target_date"], [q[0.9] - q[0.1] for q in qs])))
    tr = f[f["target_date"] < FREEZE]
    te = f[(f["target_date"] >= FREEZE) & (f["hour"] == float(hour))]
    if not te.empty and len(tr) >= MIN_TRAIN:
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(tr)
        rows = []
        for (_, r), q in zip(te.iterrows(), m.predict_quantiles(te)):
            pmf = bucket_probs(q)
            pt = sum(kk * vv for kk, vv in pmf.items()) / max(sum(pmf.values()), 1e-9)
            rows.append((r["target_date"], pmf, float(r["y"]), pt))
        report(f"FROZEN {FREEZE} -> present (never tuned on)", rows)

    print("\n### BLOCKED: P&L / per-trade edge / timing")
    n_priced = conn.execute(
        "SELECT COUNT(DISTINCT target_date) FROM signals WHERE station=? "
        "AND metric=? AND market_json IS NOT NULL AND market_json != ''",
        (station, METRIC)).fetchone()[0]
    print(f"  {station} {METRIC.upper()} days with any stored market price: "
          f"{n_priced}. No Kalshi price history exists before 2026-07-24, so "
          f"backtested P&L, per-trade edge, the timing test and any tuning of "
          f"MIN_EDGE or the max-edge threshold cannot be computed. Reported as "
          f"blocked rather than estimated on n={n_priced}.")


if __name__ == "__main__":
    sys.exit(main())
