"""KLAX intraday_hour_bandwidth: is k=1 better than the shipped k=0?

BACKGROUND. `ModelConfig.intraday_hour_bandwidth` (config.py:112) controls the
nowcast's matched-hour training window: k=0 trains only at the prediction
hour, k=1 trains at {h-1, h, h+1}. KMIA ships k=1 on a measurement quoted in
`config.py`'s own comment (HIGH: MAE 1.070->1.052, hit 55.1%->55.7%, churn
1.81->1.73 -- "the only unambiguous, trade-off-free improvement found in this
whole investigation"). KLAX stayed at k=0 ONLY because it was frozen
mid-test-week; that freeze is lifted, so this script re-measures k=1 for KLAX,
both metrics, from scratch. THE KMIA RESULT IS NOT ASSUMED TO TRANSFER -- the
two markets differ on five axes at once (residual target, bias-corrected
guidance, NBS aux features, hour bandwidth, commit hour), and a prior
measurement on this project found KMIA HIGH and KLAX LOW behave *oppositely*
under calibration changes.

MEASUREMENT ONLY. Does not import or edit anything in `config/klax.yaml` or
`src/`. `db.connect` runs idempotent DDL and a launchd wake cycle writes to
the live DB every 5 minutes -- run this only against a COPY:

    cp data/highcast.db "$TMPDIR/klax_bw.db"
    HIGHCAST_DB="$TMPDIR/klax_bw.db" .venv/bin/python scripts/klax_bandwidth_eval.py

STRUCTURE, reused from `scripts/market_baseline.py` (read first): nested
walk-forward, expanding-window folds (WF_START 2023-01-01, STEP_DAYS 90,
MIN_TRAIN 400 matching `models.intraday.MIN_MATCHED_ROWS`), `BaselineBlend`
fit on TRAIN ONLY per fold, `IntradayModel` refit per fold, scored at the
market's own commit hour read from `gating.MIN_DECISION_HOUR` (never
hardcoded: KLAX low=6, high=13). Every per-market setting is READ, not
assumed: `cfg.model.intraday_residual_{metric}`, `features_for("KLAX",
metric)` (KLAX HIGH carries RATE_FEATURES -- omitting them would measure a
model nobody runs), `uses_bias_correction(spec.key)` for the guidance the
`BaselineBlend` feeds the nowcast.

TWO ARMS, ONE FOLD SET. k=0 and k=1 share the identical fold boundaries and
identical test rows (target_date in [start,end), hour == the commit hour);
ONLY the training-hour window widens for k=1. A fold is included for BOTH
arms only if BOTH arms clear MIN_TRAIN in that fold -- k=1 always has at
least as many matched rows as k=0 (its window is a superset), so this in
practice just applies k=0's own threshold, but checking it explicitly
guarantees the two arms are ALWAYS scored on the exact same set of dates
(required for paired statistics; an unpaired comparison on two per-day
metrics as similar as these would be needlessly noisy -- see
`anchor_exclusion_eval.py`'s own reasoning for the same choice).

CHURN. Reproducing the KMIA precedent (`scripts/bandwidth_eval.py`'s
docstring: "the mean number of DISTINCT modal buckets a single day passes
through across the decision-relevant hours") inside a single-commit-hour
nested walk-forward needs one more model per extra checkpoint hour, because
churn is fundamentally a multi-hour quantity and the main axis only ever
tests hour h. This script fixes CHURN_HOURS = {h-1, h, h+1} -- literally the
k=1 arm's own training-hour window, which is also the minimal range that lets
k=0 disagree with itself at all -- and for EACH checkpoint c in that range,
for EACH arm, fits a FRESH IntradayModel trained on hours_for(c, k) (i.e. c's
own bandwidth-k window, exactly matching how production would serve c if it
shipped that k) and predicts at c. Per (arm, target_date), churn = count of
distinct single-parity modal buckets across whichever checkpoints that day
had a qualifying fit in THAT fold (again gated so both arms see the same
checkpoint/day set). c=h reuses the main axis's already-fitted model instead
of refitting. This is the one place this script's method is NOT a mechanical
copy of `market_baseline.py` -- flagged here rather than left implicit.

THE GATE (pre-registered, applied below, not editorialized after the fact):
k=1 ships for a metric only if it is BETTER-OR-NEUTRAL, by a paired bootstrap
95% CI (n_boot=2000, resampling matched target_dates), on ALL THREE of MAE,
parity-averaged bucket hit, and churn -- the same standard the KMIA change was
held to. Any axis significantly WORSE alongside another significantly BETTER
is INCONCLUSIVE, not "ship it". Flat on all three is NEUTRAL: no reason to
switch. Only worse-or-neutral on all three is REJECT. A null result on either
metric is an expected, valid outcome, not a script bug.

POSITIVE CONTROL (pre-registered, mandatory, run BEFORE any null is trusted).
A defect on this project once survived for weeks because an A/B returned "not
significant" while measuring the wrong layer entirely -- a null result proves
nothing unless the comparison is shown able to detect a real effect first.
`apply_synthetic_improvement(rows, delta_f)` shifts each day's point forecast
AND its full pmf by `delta_f` degrees TOWARD the truth, capped so a day
already within `delta_f` of correct cannot be pushed past it into a new
error. This is deliberately NOT a bare constant offset applied to every day
alike: whether this arm's existing errors happen to be biased mostly one way
or split roughly evenly, a plain "+C to every day" shift is not guaranteed
to be an improvement in aggregate (it helps the days already erring in the
opposite direction and hurts the days already erring in C's direction), so
it is unsuitable for a check whose whole point is an unambiguous, provably
positive signal. Capping toward truth keeps the shift strictly non-harmful
per day and heterogeneous in size (a day with 0.1F of error only gets 0.1F
of help even if delta_f is larger), which is why `resolving_power` below
must use the REAL comparison's own noise instead of bisecting on this
function -- this injection has no day that ever gets worse, so it has no
natural noise floor to overcome and would report a falsely tiny MDE.

⚠️ THE PMF IS KEYED BY INTEGER DEGREE. A first version of
`apply_synthetic_improvement` moved each key by
`int(round(k + step))`, which no-ops for any |step| < 0.5 -- it rounds
straight back to `k`. That silently zeroed the bucket-hit half of the sanity
check (dHIT pinned at exactly 0.0 with a zero-width CI) while the MAE half
still worked off the untouched continuous `pt`, which is exactly the
"measuring the wrong layer" failure this whole control exists to catch, this
time inside the control itself. Fixed by linearly interpolating each bin's
mass across its two integer neighbours for the fractional part of the shift.

Two DIFFERENT checks, not one, because they answer different questions and
conflating them is itself a way to fool yourself:
  (1) SANITY CHECK (`positive_control`), on the k=0 arm's own scored rows,
      paired to the real comparison's dates: shift each day's point forecast
      AND pmf by delta_f=0.30F TOWARD the truth (~16x the KMIA HIGH
      precedent's dMAE of 0.018 -- a size chosen to be undebatable) and
      confirm the identical paired_boot/classify path used below calls it
      BETTER on both MAE and bucket hit. If it does not, the PIPELINE is
      broken and any null verdict for that metric is reported UNTRUSTWORTHY.
      This check is intentionally an idealized, always-same-direction
      injection (no day's error ever grows) -- good for proving the
      mechanism is alive, but for exactly that reason it has no natural
      noise floor and must NOT be bisected on to claim a minimum detectable
      effect (an earlier version of this script did, and it reported an
      MDE of ~0.000F -- a real bug, not a good result: caught by noticing a
      deterministic, one-directional signal has no reason to ever fail to
      clear significance, however small).
  (2) RESOLVING POWER (`resolving_power`), computed from the REAL, already
      -paired (k1-k0) MAE array for this metric: demean it to isolate this
      comparison's actual cross-day noise (which, unlike (1), can and does
      go either direction night to night), add back a candidate constant
      shift, and bisect for the smallest shift whose 95% CI excludes zero.
      That is the honest "smallest difference this comparison can resolve
      at the n it has" -- printed next to the real dMAE so a NEUTRAL
      verdict can be read as "no effect" only when the observed |dMAE| is
      itself below this floor.

Usage:
    HIGHCAST_DB=/path/to/copy.db .venv/bin/python scripts/klax_bandwidth_eval.py [config/klax.yaml]
"""
import dataclasses
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

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

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/klax.yaml"
WF_START = "2023-01-01"
STEP_DAYS = 90
MIN_TRAIN = 400            # matches models.intraday.MIN_MATCHED_ROWS
N_BOOT = 2000
SEED = 20260802


def folds(start: str, end: str, step: int):
    a = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while a < stop:
        b = min(a + timedelta(days=step), stop)
        yield a.isoformat(), b.isoformat()
        a = b


def hours_for(c: int, k: int) -> tuple:
    """Training hours for bandwidth k around checkpoint c, wrapping the
    clock -- identical rule to config.py's own docstring for
    `intraday_hour_bandwidth`."""
    return tuple(sorted({(c + d) % 24 for d in range(-k, k + 1)}))


# ── shared scoring helpers (ported from market_baseline.py -- not importable,
# they are local helpers there too) ─────────────────────────────────────────

def ladder_records(pmf, y, pt):
    """One record per (day, parity). See market_baseline.py for the full
    rationale (boundary sits at a half-degree, not an integer, etc.) --
    reproduced verbatim here since it is a local helper there, not exported."""
    out = []
    for parity in (0, 1):
        lo, mass = modal_bucket(pmf, parity)
        true_lo = int(y) - ((int(y) - parity) % 2)
        p_true = sum(v for k, v in pmf.items() if true_lo <= k <= true_lo + 1)
        d = (pt - (parity + 1.5)) % 2.0
        dist = min(d, 2.0 - d)
        out.append((mass, 1.0 if lo <= int(y) <= lo + 1 else 0.0, p_true, dist))
    return out


def score_days(rows):
    """rows: list of (target_date, pmf, y, pt) -> dict of per-day arrays,
    aligned index-for-index with `rows`, plus the pooled per-ladder arrays
    for log loss / single-parity hit."""
    ys = np.array([r[2] for r in rows], dtype=float)
    pts = np.array([r[3] for r in rows], dtype=float)
    ae = np.abs(pts - ys)
    crps = np.array([crps_pmf(p, int(y)) for _, p, y, _ in rows])
    ehit = np.array([expected_bucket_hit(p, int(y)) for _, p, y, _ in rows])
    ll_per_day, hit1_per_day = [], []
    for _, p, y, pt in rows:
        recs = ladder_records(p, y, pt)
        ll_per_day.append(np.mean([-np.log(max(r[2], 1e-9)) for r in recs]))
        hit1_per_day.append(np.mean([r[1] for r in recs]))
    return {
        "dates": [r[0] for r in rows],
        "MAE": ae, "bias": pts - ys, "CRPS": crps,
        "hit": ehit, "logloss": np.array(ll_per_day),
        "hit1par": np.array(hit1_per_day),
    }


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """(b - a) mean with a paired bootstrap 95% CI -- same day resampled for
    both arms each draw. Positive delta = arm b is larger on that metric."""
    rng = np.random.default_rng(seed)
    d = b - a
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def classify(delta: float, lo: float, hi: float, better_is_negative: bool) -> str:
    """BETTER / NEUTRAL / WORSE for k=1 relative to k=0 on one axis, from a
    paired-bootstrap CI on (k1 - k0). `better_is_negative` is True for
    MAE/churn (lower is better) and False for bucket hit (higher is better)."""
    if lo <= 0.0 <= hi:
        return "NEUTRAL"
    improved = (delta < 0) if better_is_negative else (delta > 0)
    return "BETTER" if improved else "WORSE"


def gate_verdict(mae_v: str, hit_v: str, churn_v: str) -> str:
    vs = (mae_v, hit_v, churn_v)
    if "WORSE" in vs and "BETTER" in vs:
        return "INCONCLUSIVE (wins on one axis, loses on another -- not a ship)"
    if "WORSE" in vs:
        return "REJECT -- k=0 stays"
    if "BETTER" in vs:
        return "SHIP -- better-or-neutral on MAE, HIT and churn"
    return "NEUTRAL -- flat on all three, no reason to switch"


def churn_stat(labels_by_date: dict) -> tuple:
    if not labels_by_date:
        return float("nan"), 0
    counts = [len(set(v)) for v in labels_by_date.values()]
    return float(np.mean(counts)), len(counts)


# ── POSITIVE CONTROL (see module docstring for the full rationale) ─────────

SANITY_DELTA_F = 0.30   # ~16x the KMIA HIGH precedent's dMAE (0.018)


def apply_synthetic_improvement(rows: list, delta_f: float) -> list:
    """rows: list of (target_date, pmf, y, pt). Returns a NEW list with each
    day's point forecast and pmf shifted `delta_f` degrees TOWARD `y`, capped
    at that day's own |error| so a near-perfect day cannot be pushed past
    zero into a new one. See module docstring for why this beats a bare
    constant offset as an injected effect.

    ⚠️ THE PMF IS KEYED BY INTEGER DEGREE (`buckets.bucket_probs`'s own
    contract), so a sub-one-degree shift is not a relabelling of keys -- it
    is a redistribution of mass BETWEEN adjacent integer bins. An earlier
    version of this function did `int(round(k + step))`, which for any
    |step| < 0.5 rounds straight back to `k` and silently no-ops on bucket
    hit while still moving the continuous `pt` used for MAE -- caught by
    running this exact function against a synthetic pmf and finding dHIT
    pinned at exactly 0.0 with a zero-width CI, the very "wrong layer"
    failure this positive control exists to catch. Fixed here by splitting
    `step` into its integer and fractional part and linearly interpolating
    each bin's mass across its two neighbours, which is the standard way to
    shift a discretized distribution by a non-integer amount and is exact in
    the limit of infinitesimal steps.
    """
    out = []
    for td, pmf, y, pt in rows:
        err = pt - y
        step = 0.0 if err == 0 else -np.sign(err) * min(abs(err), delta_f)
        lo_shift = int(np.floor(step))
        frac = step - lo_shift          # in [0, 1)
        shifted: dict = {}
        for k, v in pmf.items():
            shifted[k + lo_shift] = shifted.get(k + lo_shift, 0.0) + v * (1 - frac)
            shifted[k + lo_shift + 1] = shifted.get(k + lo_shift + 1, 0.0) + v * frac
        out.append((td, shifted, y, pt + step))
    return out


def positive_control(rows: list, seed=SEED) -> dict:
    """Sanity check ONLY: does the scoring pipeline detect a deliberately
    injected, unambiguous effect on `rows` (already paired to the real
    comparison's test dates)? Returns {mae_v, hit_v, trustworthy}; also
    prints a short report.

    ⚠️ NOT the minimum-detectable-effect (MDE) calculation -- see
    `resolving_power` below and the module docstring for why an idealized,
    always-same-direction injection (every day's error strictly shrinks,
    never grows) has no natural noise floor and would report an
    artificially tiny, meaningless MDE if bisected on directly. This
    function only answers "is the mechanism alive at all", at one
    deliberately large, unambiguous size.
    """
    real = score_days(rows)
    pert = score_days(apply_synthetic_improvement(rows, SANITY_DELTA_F))
    mae_d, mae_lo, mae_hi = paired_boot(real["MAE"], pert["MAE"], seed=seed)
    hit_d, hit_lo, hit_hi = paired_boot(real["hit"], pert["hit"], seed=seed)
    mae_v = classify(mae_d, mae_lo, mae_hi, better_is_negative=True)
    hit_v = classify(hit_d, hit_lo, hit_hi, better_is_negative=False)

    print(f"\n  POSITIVE CONTROL (n={len(rows)} days, same dates as the real "
          f"comparison below):")
    print(f"    sanity injection delta_f={SANITY_DELTA_F:.2f}F toward truth:")
    print(f"      dMAE  {mae_d:+.3f}  95% CI [{mae_lo:+.3f}, {mae_hi:+.3f}]"
          f"   -> {mae_v}")
    print(f"      dHIT  {hit_d*100:+.1f} pts  95% CI [{hit_lo*100:+.1f}, "
          f"{hit_hi*100:+.1f}]   -> {hit_v}")
    trustworthy = mae_v == "BETTER" and hit_v == "BETTER"
    if not trustworthy:
        print("      *** PIPELINE DID NOT DETECT ITS OWN PLANTED EFFECT ***")
        print("      Any NEUTRAL/null verdict below for this metric is "
              "UNTRUSTWORTHY -- the comparison\n      itself has no "
              "demonstrated power, independent of what the real k=0-vs-k=1 "
              "numbers say.")

    return {"mae_v": mae_v, "hit_v": hit_v, "trustworthy": trustworthy}


def resolving_power(d_real: np.ndarray, seed=SEED) -> float:
    """The smallest constant MAE shift, in degrees F, whose paired-bootstrap
    95% CI would exclude zero -- using THIS comparison's own observed
    day-to-day noise, not an idealized injected signal.

    `d_real` is the REAL, already-paired (k1 - k0) MAE array. It is demeaned
    (`centered`) to isolate pure noise, then a candidate constant shift `c`
    is added back and bisected: the bootstrap CI of `centered - c` is exactly
    what a comparison with this same noise structure and a TRUE mean effect
    of `-c` would have produced. This is the honest complement to
    `positive_control`: that function proves the mechanism can see an
    obvious, idealized effect; this one says how small a REAL, noisy effect
    -- one that can go either direction on any given day, the way two
    different model fits actually would -- could have been told apart from
    chance at the n this metric's fold set has.
    """
    centered = d_real - d_real.mean()
    zeros = np.zeros_like(centered)
    hi_b = max(0.05, 6 * float(np.std(centered)) + 1e-6)
    for _ in range(6):
        d, ci_lo, ci_hi = paired_boot(zeros, centered - hi_b, seed=seed)
        if classify(d, ci_lo, ci_hi, better_is_negative=True) == "BETTER":
            break
        hi_b *= 2
    else:
        return float("nan")   # not resolvable even at a huge hypothetical shift
    lo_b = 0.0
    for _ in range(20):
        mid = (lo_b + hi_b) / 2
        d, ci_lo, ci_hi = paired_boot(zeros, centered - mid, seed=seed)
        if classify(d, ci_lo, ci_hi, better_is_negative=True) == "BETTER":
            hi_b = mid
        else:
            lo_b = mid
    return hi_b


def run_metric(conn, cfg, station: str, metric: str, dates: list, end: str):
    spec = get_metric(cfg, metric)
    hour = MIN_DECISION_HOUR.get(station, {}).get(metric)
    if hour is None:
        print(f"{station} {metric}: no MIN_DECISION_HOUR entry -- refusing to invent one.")
        return

    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    extra = [c for c in features_for(station, metric) if c not in INTRADAY_FEATURES]
    corrected = uses_bias_correction(spec.key)

    churn_hours = hours_for(hour, 1)   # {h-1, h, h+1} -- see module docstring
    superset = sorted({hh for c in churn_hours for k in (0, 1) for hh in hours_for(c, k)})

    print(f"\n{'='*78}\n{station} {metric.upper()}  commit hour={hour}  "
          f"residual={resid}  bias-corrected guidance={corrected}\n"
          f"extra features: {extra or 'none'}\n"
          f"k=0 train hours @h={hours_for(hour, 0)}   "
          f"k=1 train hours @h={hours_for(hour, 1)}\n"
          f"churn checkpoints: {churn_hours}   frame hours built: {superset}\n"
          f"days with settled CLI {metric}: {len(dates)} ({dates[0]} -> {dates[-1]})")

    day_df = build_features(conn, cfg, spec, dates, require_target=True)
    fr = build_checkpoint_frame(conn, cfg, dates, hours=tuple(superset), metric=spec)
    fr = fr[fr["y"].notna()].copy()

    main_rows = {0: [], 1: []}
    churn_labels = {0: {}, 1: {}}
    n_folds_used = 0

    for start, fend in folds(WF_START, end, STEP_DAYS):
        tr_days = day_df[day_df["target_date"] < start]
        if len(tr_days) < MIN_TRAIN:
            continue
        base = BaselineBlend(prefer_corrected=corrected).fit(tr_days)
        qs = base.predict_quantiles(day_df)
        p50 = dict(zip(day_df["target_date"], [q[0.5] for q in qs]))
        spr = dict(zip(day_df["target_date"], [q[0.9] - q[0.1] for q in qs]))
        f = fr.copy()
        f["base_p50"] = f["target_date"].map(p50)
        f["base_spread"] = f["target_date"].map(spr)

        te_main = f[(f["target_date"] >= start) & (f["target_date"] < fend)
                    & (f["hour"] == float(hour))]
        if te_main.empty:
            continue

        # ── main axis: both arms must clear MIN_TRAIN in THIS fold or the
        # fold is dropped for both -- keeps the two arms paired on identical
        # test dates throughout (see module docstring). ──────────────────
        arm_tr = {}
        ok = True
        for k in (0, 1):
            hrs = hours_for(hour, k)
            trk = f[(f["target_date"] < start)
                    & (f["hour"].isin([float(x) for x in hrs]))]
            if len(trk) < MIN_TRAIN:
                ok = False
            arm_tr[k] = trk
        if not ok:
            continue
        n_folds_used += 1

        fitted_main = {}
        for k in (0, 1):
            m = IntradayModel(extremum=spec.extremum, residual=resid,
                              extra_features=extra).fit(arm_tr[k])
            fitted_main[k] = m
            pmfs = [bucket_probs(q) for q in m.predict_quantiles(te_main)]
            for (_, r), pmf in zip(te_main.iterrows(), pmfs):
                pt = sum(kk * vv for kk, vv in pmf.items()) / max(sum(pmf.values()), 1e-9)
                main_rows[k].append((r["target_date"], pmf, float(r["y"]), pt))
                lo, _ = modal_bucket(pmf, 0)
                churn_labels[k].setdefault(r["target_date"], []).append(lo)

        # ── churn axis: the other checkpoints in {h-1, h+1}. Each arm needs
        # its OWN fresh fit at c's own bandwidth-k window; both arms must
        # clear MIN_TRAIN for a given checkpoint or it is dropped for both,
        # same pairing discipline as the main axis. ───────────────────────
        for c in churn_hours:
            if c == hour:
                continue  # already folded into churn_labels above
            te_c = f[(f["target_date"] >= start) & (f["target_date"] < fend)
                     & (f["hour"] == float(c))]
            if te_c.empty:
                continue
            c_tr, c_ok = {}, True
            for k in (0, 1):
                hrs = hours_for(c, k)
                trk = f[(f["target_date"] < start)
                        & (f["hour"].isin([float(x) for x in hrs]))]
                if len(trk) < MIN_TRAIN:
                    c_ok = False
                c_tr[k] = trk
            if not c_ok:
                continue
            for k in (0, 1):
                m = IntradayModel(extremum=spec.extremum, residual=resid,
                                  extra_features=extra).fit(c_tr[k])
                pmfs = [bucket_probs(q) for q in m.predict_quantiles(te_c)]
                for (_, r), pmf in zip(te_c.iterrows(), pmfs):
                    lo, _ = modal_bucket(pmf, 0)
                    churn_labels[k].setdefault(r["target_date"], []).append(lo)

    print(f"folds used (both arms qualified): {n_folds_used}")

    scored = {k: score_days(main_rows[k]) for k in (0, 1)}
    n0, n1 = len(main_rows[0]), len(main_rows[1])
    if n0 == 0 or n1 == 0:
        print(f"  no scoreable rows for one or both arms (n0={n0}, n1={n1}) -- skipping metric")
        return

    d0, d1 = scored[0]["dates"], scored[1]["dates"]
    if d0 != d1:
        print(f"  WARNING: arm test dates diverged (n0={len(d0)} n1={len(d1)}); "
              f"pairing on the intersection")
        common = sorted(set(d0) & set(d1))
        idx0 = {dd: i for i, dd in enumerate(d0)}
        idx1 = {dd: i for i, dd in enumerate(d1)}
        sel0 = [idx0[dd] for dd in common]
        sel1 = [idx1[dd] for dd in common]
    else:
        sel0 = sel1 = list(range(len(d0)))

    # ── positive control, BEFORE the real comparison is trusted ────────────
    pc = positive_control([main_rows[0][i] for i in sel0])

    print(f"\n  {'metric':>10}{'k=0 (shipped)':>16}{'k=1':>12}{'delta':>10}")
    for m, better_neg in (("MAE", True), ("bias", None), ("CRPS", True),
                          ("hit", False), ("logloss", True)):
        a0 = scored[0][m][sel0]
        a1 = scored[1][m][sel1]
        delta = float(a1.mean() - a0.mean())
        label = {"MAE": "MAE", "bias": "bias", "CRPS": "CRPS",
                 "hit": "bucket hit", "logloss": "log loss"}[m]
        if m in ("hit",):
            print(f"  {label:>10}{a0.mean():>15.1%}{a1.mean():>11.1%}{delta*100:>+9.1f}pt")
        else:
            print(f"  {label:>10}{a0.mean():>16.3f}{a1.mean():>12.3f}{delta:>+10.3f}")

    mae_d, mae_lo, mae_hi = paired_boot(scored[0]["MAE"][sel0], scored[1]["MAE"][sel1])
    hit_d, hit_lo, hit_hi = paired_boot(scored[0]["hit"][sel0], scored[1]["hit"][sel1])
    mae_v = classify(mae_d, mae_lo, mae_hi, better_is_negative=True)
    hit_v = classify(hit_d, hit_lo, hit_hi, better_is_negative=False)

    print(f"\n  paired bootstrap (n_boot={N_BOOT}, n_days={len(sel0)}):")
    print(f"    dMAE  {mae_d:+.3f}  95% CI [{mae_lo:+.3f}, {mae_hi:+.3f}]   -> {mae_v}")
    print(f"    dHIT  {hit_d*100:+.1f} pts  95% CI [{hit_lo*100:+.1f}, {hit_hi*100:+.1f}]   -> {hit_v}")
    mde = resolving_power(scored[1]["MAE"][sel1] - scored[0]["MAE"][sel0])
    if mde == mde:  # not NaN
        within = "" if abs(mae_d) >= mde else " -- SMALLER THAN THE RESOLVABLE FLOOR"
        print(f"    resolving power at this n (from this comparison's OWN "
              f"noise, see module docstring): {mde:.3f}F; observed "
              f"|dMAE|={abs(mae_d):.3f}F{within}")
    else:
        print(f"    resolving power at this n: could not bound (huge)")

    # ── churn ────────────────────────────────────────────────────────────
    c0, n_days0 = churn_stat(churn_labels[0])
    c1, n_days1 = churn_stat(churn_labels[1])
    common_days = sorted(set(churn_labels[0]) & set(churn_labels[1]))
    ch0 = np.array([len(set(churn_labels[0][d])) for d in common_days], dtype=float)
    ch1 = np.array([len(set(churn_labels[1][d])) for d in common_days], dtype=float)
    if len(common_days) >= 20:
        churn_d, churn_lo, churn_hi = paired_boot(ch0, ch1)
        churn_v = classify(churn_d, churn_lo, churn_hi, better_is_negative=True)
    else:
        churn_d = float(ch1.mean() - ch0.mean()) if len(common_days) else float("nan")
        churn_lo = churn_hi = float("nan")
        churn_v = "NEUTRAL (n too thin to test)"

    print(f"\n  CHURN (distinct modal buckets/day, checkpoints {churn_hours}, single-parity):")
    print(f"    k=0  {c0:.3f}  (n={n_days0} days)")
    print(f"    k=1  {c1:.3f}  (n={n_days1} days)")
    print(f"    delta {churn_d:+.3f}  95% CI [{churn_lo:+.3f}, {churn_hi:+.3f}]"
          f"  (paired n={len(common_days)})   -> {churn_v}")

    verdict = gate_verdict(mae_v, hit_v, churn_v)
    if not pc["trustworthy"] and verdict.startswith("NEUTRAL"):
        verdict += " -- BUT POSITIVE CONTROL FAILED, treat this null as UNTRUSTED"
    print(f"\n  GATE VERDICT ({metric.upper()}): {verdict}")
    return {"mae": mae_v, "hit": hit_v, "churn": churn_v, "verdict": verdict,
            "positive_control_trustworthy": pc["trustworthy"],
            "resolving_power_mae_f": mde}


def main():
    cfg = load_config(CONFIG)
    db = os.environ.get("HIGHCAST_DB")
    if not db:
        print("HIGHCAST_DB is not set -- refusing to run against the live DB "
              "(a launchd wake cycle writes to it every 5 minutes). "
              "cp data/highcast.db to a scratch copy and set HIGHCAST_DB to that path.")
        return 1
    cfg = dataclasses.replace(cfg, db_path=Path(db))
    conn = connect(cfg.db_path)
    station = cfg.station.icao
    if station != "KLAX":
        print(f"WARNING: config resolves to station {station}, not KLAX -- "
              f"this script was written to answer the KLAX question specifically.")

    results = {}
    for metric in ("low", "high"):
        col = "high_f" if metric == "high" else "low_f"
        dates = [d for (d,) in conn.execute(
            f"SELECT date FROM cli_reports WHERE station=? AND {col} IS NOT NULL "
            "ORDER BY date", (station,))]
        if not dates:
            print(f"{station} {metric}: no settled CLI rows -- skipping")
            continue
        end = (date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat()
        results[metric] = run_metric(conn, cfg, station, metric, dates, end)

    print(f"\n{'='*78}\nSUMMARY")
    for metric, r in results.items():
        if r is None:
            print(f"  {station} {metric.upper():<5}: no result (see above)")
        else:
            print(f"  {station} {metric.upper():<5}: MAE={r['mae']:<8} "
                  f"HIT={r['hit']:<8} CHURN={r['churn']:<8} -> {r['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
