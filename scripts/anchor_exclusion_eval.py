"""A/B: does excluding the DERIVED `fc_anchor` column from `BaselineBlend` help?

THE DEFECT. `BaselineBlend._blend_cols` selects every `fc_*` column except
`_EXCLUDE = {fc_spread, fc_mean, fcc_mean}`. `features.build_features` appends
`fc_anchor` = coalesce(fc_mos_mex -> fc_gfs -> fc_mean) as a level anchor for
the residual day-ahead target. It starts with `fc_` and was never excluded, so
the blend counts it as a tenth INDEPENDENT guidance model. It is not one: it is
a function of three columns that are already blend inputs (or, for `fc_mean`,
deliberately excluded FROM being one). Measured weight share of `fc_anchor`:
KLAX high 16.2%, KLAX low 11.3%, KMIA high 16.5%, KMIA low 8.3%.

Two further wrinkles the weight table alone hides:

  * ON KLAX THE CHAIN NEVER SWITCHES. `fc_anchor == fc_mos_mex` on 2036/2036
    days for both metrics, so KLAX is pure double-weighting of mos_mex
    (high: 16.2% + 16.2% = 32.4%). On KMIA it DOES switch -- mos_mex on 2202
    days, fc_mean on 241 -- so which model is doubled changes by day, and on the
    fc_mean days an explicitly-excluded aggregate re-enters through the back
    door.
  * ON THE LOW THE DOUBLE IS UNCORRECTED. The LOW runs `prefer_corrected=True`,
    so every real channel enters as `fcc_*`. There is no `fcc_anchor`, so
    `fc_anchor` enters RAW -- injecting 8-11% of un-bias-corrected mos_mex into
    a blend that is otherwise fully corrected, partially undoing a correction
    `uses_bias_correction` measured as significant for the LOW (-0.091 dMAE).

WHY THIS NEEDS MEASURING AND NOT JUST FIXING. mos_mex is the single best channel
on both HIGHs, so double-weighting it may be accidentally BENEFICIAL. The blend
is inverse-MSE weighted, which does not know the two columns are the same model,
so "correct" and "better" are separate questions here. `base_p50`/`base_spread`
are live nowcast features on ALL FOUR markets and KLAX is frozen mid-test-week,
so the bar is evidence, not tidiness.

WHAT IS MEASURED. Two layers, both nested walk-forward with the blend fitted on
TRAIN ONLY (production behaviour), arms paired on identical days/folds:

  LAYER A -- the blend's own output vs the settled CLI value. This is the direct
    effect and has the most days behind it. It is NOT what trades.
  LAYER B -- the nowcast at each market's commit hour (`MIN_DECISION_HOUR`),
    refit per fold, reading the arm's `base_p50`/`base_spread` as features.
    This is the traded quantity; a Layer A move can be absorbed or amplified
    here, so Layer B is the one that decides shipping.

Deltas are PAIRED per day (same day, both arms) with a 10k-resample bootstrap
CI, because the arms share ~90% of their inputs and an unpaired CI on two ~1.5F
MAEs would be far too wide to resolve a ~0.03F effect.

Usage:  HIGHCAST_DB=/path/to/copy.db .venv/bin/python scripts/anchor_exclusion_eval.py

Reads only, but `db.connect` runs idempotent DDL -- point HIGHCAST_DB at a COPY
rather than racing the 5-minute wake cycle on the live file.
"""
import dataclasses
import os
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import highcast.models.baseline as baseline_mod
from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel,
                                      build_checkpoint_frame, features_for)

MARKETS = [("KLAX", "config/klax.yaml", "high"), ("KLAX", "config/klax.yaml", "low"),
           ("KMIA", "config/kmia.yaml", "high"), ("KMIA", "config/kmia.yaml", "low")]
WF_START = "2023-01-01"
FREEZE = "2026-06-01"
STEP_DAYS = 90
MIN_TRAIN = 400
BOOT = 10_000
ARMS = ("with_anchor", "no_anchor")


@contextmanager
def arm(name):
    """Swap the module-level `_EXCLUDE` -- i.e. exercise the EXACT shipped change
    rather than a subclass that could diverge from it."""
    old = baseline_mod._EXCLUDE
    if name == "no_anchor":
        baseline_mod._EXCLUDE = old | {"fc_anchor"}
    try:
        yield
    finally:
        baseline_mod._EXCLUDE = old


def folds(start, end, step):
    a, stop = date.fromisoformat(start), date.fromisoformat(end)
    while a < stop:
        b = min(a + timedelta(days=step), stop)
        yield a.isoformat(), b.isoformat()
        a = b


def ladder_records(pmf, y, pt):
    """One record per (day, parity) -- what the bot actually faces. Same
    convention as `scripts/kmia_high_baseline.py`, including scoring log loss on
    the bucket that CONTAINED y rather than on the modal bucket (which would
    reward confidence itself)."""
    out = []
    for parity in (0, 1):
        lo, mass = modal_bucket(pmf, parity)
        true_lo = int(y) - ((int(y) - parity) % 2)
        p_true = sum(v for k, v in pmf.items() if true_lo <= k <= true_lo + 1)
        out.append((mass, 1.0 if lo <= int(y) <= lo + 1 else 0.0, p_true))
    return out


def day_metrics(rows):
    """rows: [(target_date, pmf, y, pt)] -> per-day arrays, so deltas can pair."""
    ys = np.array([r[2] for r in rows], dtype=float)
    pts = np.array([r[3] for r in rows], dtype=float)
    ae = np.abs(pts - ys)
    crps = np.array([crps_pmf(p, int(y)) for _, p, y, _ in rows])
    ehit = np.array([expected_bucket_hit(p, int(y)) for _, p, y, _ in rows])
    ll, hit = [], []
    for _, p, y, pt in rows:
        recs = ladder_records(p, y, pt)
        ll.append(np.mean([-np.log(max(r[2], 1e-9)) for r in recs]))
        hit.append(np.mean([r[1] for r in recs]))
    return {"MAE": ae, "CRPS": crps, "log loss": np.array(ll),
            "bucket hit": ehit, "hit(1par)": np.array(hit),
            "bias": pts - ys}


def paired(a, b, rng):
    """b - a with a paired bootstrap CI. Positive = the `no_anchor` arm is
    LARGER on that metric (worse for MAE/CRPS/log loss, better for hit)."""
    d = b - a
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compare(tag, per_arm, rng):
    a, b = per_arm["with_anchor"], per_arm["no_anchor"]
    n = len(a["MAE"])
    print(f"\n  {tag}  (n={n} days, paired)")
    print(f"  {'metric':>11}{'with_anchor':>14}{'no_anchor':>12}"
          f"{'delta':>10}{'95% CI (paired)':>24}   verdict")
    verdicts = {}
    for m in ("MAE", "CRPS", "log loss", "bucket hit", "hit(1par)"):
        d, lo, hi = paired(a[m], b[m], rng)
        sig = not (lo <= 0.0 <= hi)
        better_is_low = m in ("MAE", "CRPS", "log loss")
        if not sig:
            v = "ns"
        elif (d < 0) == better_is_low:
            v = "BETTER without anchor"
        else:
            v = "WORSE without anchor"
        verdicts[m] = (d, lo, hi, v)
        print(f"  {m:>11}{a[m].mean():>14.4f}{b[m].mean():>12.4f}"
              f"{d:>+10.4f}   [{lo:>+8.4f},{hi:>+8.4f}]   {v}")
    return verdicts


def blend_rows(base, df):
    """Layer A rows straight off the blend, no downstream model."""
    qs = base.predict_quantiles(df)
    out = []
    for (_, r), q in zip(df.iterrows(), qs):
        if np.isnan(r["y"]):
            continue
        pmf = bucket_probs(q)
        pt = sum(k * v for k, v in pmf.items()) / max(sum(pmf.values()), 1e-9)
        out.append((r["target_date"], pmf, float(r["y"]), pt))
    return out


def run_market(station, cfgname, mk, rng):
    cfg = load_config(cfgname)
    if os.environ.get("HIGHCAST_DB"):
        cfg = dataclasses.replace(cfg, db_path=Path(os.environ["HIGHCAST_DB"]))
    conn = connect(cfg.db_path)
    spec = get_metric(cfg, mk)
    hour = MIN_DECISION_HOUR[station][mk]
    k = cfg.model.intraday_bandwidth_for(mk)
    train_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
    resid = bool(getattr(cfg.model, f"intraday_residual_{mk}", False))
    extra = [c for c in features_for(station, mk) if c not in INTRADAY_FEATURES]
    pc = uses_bias_correction(spec.key)

    dates = [d for (d,) in conn.execute(
        f"SELECT date FROM cli_reports WHERE station=? AND {spec.target_col} "
        "IS NOT NULL ORDER BY date", (station,))]

    print(f"\n{'='*78}\n### {station} {mk.upper()}   commit hour h{hour}   "
          f"bandwidth k={k} -> hours {train_hours}\n    residual={resid}   "
          f"prefer_corrected={pc}   extra features: {extra or 'none'}\n"
          f"    days with a settled CLI value: {len(dates)} "
          f"({dates[0]} -> {dates[-1]})")

    day_df = build_features(conn, cfg, spec, dates, require_target=True)
    fr = build_checkpoint_frame(conn, cfg, dates, hours=train_hours, metric=spec)
    fr = fr[fr["y"].notna()].copy()

    # weight share of the derived column, fitted on everything (diagnostic only)
    for a in ARMS:
        with arm(a):
            w = BaselineBlend(prefer_corrected=pc).fit(day_df).weights
        tot = sum(w.values())
        if a == "with_anchor":
            share = w.get("fc_anchor", 0.0) / tot
            print(f"    fc_anchor weight share (full-sample fit): {share:.1%} "
                  f"of {len(w)} blend inputs")
        else:
            assert "fc_anchor" not in w
            print(f"    no_anchor arm: {len(w)} blend inputs, fc_anchor absent")

    layerA = {a: [] for a in ARMS}
    layerB = {a: [] for a in ARMS}
    n_folds = 0
    for start, end in folds(WF_START, FREEZE, STEP_DAYS):
        tr_days = day_df[day_df["target_date"] < start]
        if len(tr_days) < MIN_TRAIN:
            continue
        te_days = day_df[(day_df["target_date"] >= start)
                         & (day_df["target_date"] < end)]
        fold_ok = True
        pend = {}
        for a in ARMS:
            with arm(a):
                base = BaselineBlend(prefer_corrected=pc).fit(tr_days)
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
                fold_ok = False
                break
            m = IntradayModel(extremum=spec.extremum, residual=resid,
                              extra_features=extra).fit(tr)
            rows = []
            for (_, r), q in zip(te.iterrows(), m.predict_quantiles(te)):
                pmf = bucket_probs(q)
                pt = sum(kk * vv for kk, vv in pmf.items()) / max(sum(pmf.values()), 1e-9)
                rows.append((r["target_date"], pmf, float(r["y"]), pt))
            pend[a] = (blend_rows(base, te_days), rows)
        if not fold_ok:
            continue
        n_folds += 1
        for a in ARMS:
            layerA[a].extend(pend[a][0])
            layerB[a].extend(pend[a][1])

    out = {}
    for label, store in (("LAYER A -- blend output (base_p50) vs settled CLI", layerA),
                         (f"LAYER B -- NOWCAST at the commit hour h{hour} (what trades)",
                          layerB)):
        keys = [set(r[0] for r in store[a]) for a in ARMS]
        assert keys[0] == keys[1], "arms disagree on days -- pairing is invalid"
        if not store["with_anchor"]:
            print(f"\n  {label}: no rows")
            continue
        per_arm = {a: day_metrics(sorted(store[a], key=lambda r: r[0])) for a in ARMS}
        print(f"\n  {label}")
        out[label] = compare(f"walk-forward {WF_START} -> {FREEZE} "
                             f"({n_folds} folds, {STEP_DAYS}d refit)", per_arm, rng)
        for a in ARMS:
            print(f"      {a:>12}  bias {per_arm[a]['bias'].mean():+.3f} F")
    return out


def main():
    rng = np.random.default_rng(20260802)
    summary = {}
    for station, cfgname, mk in MARKETS:
        summary[f"{station} {mk}"] = run_market(station, cfgname, mk, rng)

    print(f"\n\n{'='*78}\n### SUMMARY -- delta = (no_anchor) - (with_anchor); "
          f"negative is better for MAE/CRPS/log loss\n{'='*78}")
    for layer in ("LAYER A", "LAYER B"):
        print(f"\n{layer}")
        print(f"  {'market':>11}{'dMAE':>10}{'dCRPS':>10}{'dLogLoss':>11}"
              f"{'dBucketHit':>13}   flags")
        for mkt, res in summary.items():
            key = next((k for k in res if k.startswith(layer)), None)
            if key is None:
                print(f"  {mkt:>11}   (no rows)")
                continue
            r = res[key]
            flags = ", ".join(f"{m}:{v[3]}" for m, v in r.items()
                              if v[3] != "ns") or "all ns"
            print(f"  {mkt:>11}{r['MAE'][0]:>+10.4f}{r['CRPS'][0]:>+10.4f}"
                  f"{r['log loss'][0]:>+11.4f}{r['bucket hit'][0]:>+13.4f}   {flags}")


if __name__ == "__main__":
    sys.exit(main())
