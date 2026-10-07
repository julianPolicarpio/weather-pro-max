"""Per-hour accuracy curve for the matched-hour nowcast: WHICH HOUR IS BEST?

`hour_sweep.py` answered "does training at the prediction hour beat the fixed
default?" (yes). It sampled only 5 hand-picked hours, chosen AFTER seeing the
hour-3 bug -- HANDOFF-2026-07-26.md flags that as a biased sample and asks for
"a full per-hour curve (0-23)". This is that curve.

Production now trains at the hour it predicts (`cli.cmd_predict --live`), so
every hour here uses the MATCH regime: train @h on dates < SPLIT, test @h on
dates >= SPLIT. There is no PROD arm -- this is not a regime comparison, it is
"given the shipped model, at which hour is its answer most accurate?"

Scored three ways, because MAE is not what a bucket bettor consumes:
  MAE / CRPS      point + distribution error against the settled CLI value
  HIT             does the model's MODAL 2F bucket contain the settled value?
  CALIB           mean stated probability on that bucket minus its realized hit
                  rate (>0 = overconfident)

HIT IS AVERAGED OVER BOTH LADDER PARITIES. This file used to score against a
fixed EVEN ladder (68-69, 70-71). That was wrong: Kalshi centres each day's
ladder on the consensus forecast, so the parity is a coin flip -- measured over
68 settled days, KXLOWTLAX ran 36 odd / 32 even. Fixed-even scoring was right
about half the time and silently wrong the rest, and because this script is the
source of `gating.HOUR_ACCURACY` (hence `MIN_DECISION_HOUR`), the error reached
the commit hours and not just the display. See `buckets.expected_bucket_hit`.
Note `buckets.argmax_pair` is hardcoded ODD and matches no ladder reliably --
do not reuse it here.

Single holdout, not walk-forward: absolute numbers are not comparable to the
walk-forward figures in HANDOFF.md. The across-hour COMPARISON is the output,
and it is paired on matched test dates (an hour is only compared on days both
hours could be scored).

Usage:  .venv/bin/python scripts/hour_curve.py [low|high|both] [n_boot] [config]
"""
import sys

import numpy as np

from highcast.buckets import (bucket_probs, expected_bucket_hit,
                              modal_bucket)
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import (INTRADAY_FEATURES, IntradayModel, build_checkpoint_frame, features_for)

SPLIT = "2025-01-01"
HOURS = tuple(range(24))
which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
METRICS = ["low", "high"] if which == "both" else [which]

CONFIG = sys.argv[3] if len(sys.argv) > 3 else "config/klax.yaml"
cfg = load_config(CONFIG)

# Reference hour for the paired dMAE column, read from `gating` rather than
# copied -- a stale literal silently compares every hour against the wrong
# reference. An UNMEASURED station (which KMIA is, by design, until this script
# has told us its hours) has no knee at all, so it falls back to LAX's hours
# PURELY as a comparison anchor for the dMAE column. That is a display choice
# and nothing else: this script never writes a decision hour, and the KMIA row
# in `gating.MIN_DECISION_HOUR` stays absent until a human puts measured
# numbers there.
from highcast.gating import MIN_DECISION_HOUR  # noqa: E402

DECISION_HOUR = (MIN_DECISION_HOUR.get(cfg.station.icao)
                 or {"low": 4, "high": 11})
conn = connect(cfg.db_path)
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]
rng = np.random.default_rng(0)


def stated_prob(pmf: dict[int, float]) -> float:
    """Mean probability the model puts on its own modal bucket, over BOTH
    ladders -- the companion of `expected_bucket_hit`, so `calib` compares
    like with like."""
    return float(np.mean([modal_bucket(pmf, par)[1] for par in (0, 1)]))


for metric in METRICS:
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    print(f"\n=== {metric.upper()}  (residual={resid})  holdout: train < {SPLIT} "
          f"<= test  ===")
    print(f"{'hour':>4}{'n':>6} | {'MAE':>7}{'CRPS':>7} | {'HIT':>7}{'stated':>8}"
          f"{'calib':>7} | {'vs h' + str(DECISION_HOUR[metric]) + ' dMAE':>22}")

    per_hour: dict[int, dict] = {}
    for h in HOURS:
        # TRAIN on the configured bandwidth, TEST at hour h alone. The curve has
        # to describe the model that actually runs -- KMIA ships
        # `intraday_hour_bandwidth: 1`, and KLAX now overrides per metric
        # (`intraday_hour_bandwidth_low: 1`), so reading the station-wide field
        # would score a model nobody is using and hand back commit hours chosen
        # for it. PER-METRIC: use this loop's `metric`, not the bare field.
        _k = cfg.model.intraday_bandwidth_for(metric)
        _tr_hours = tuple(sorted({(h + d) % 24 for d in range(-_k, _k + 1)}))
        fr = build_checkpoint_frame(conn, cfg, dates, hours=_tr_hours, metric=spec)
        tr = fr[(fr["target_date"] < SPLIT) & fr["y"].notna()]
        te = fr[(fr["target_date"] >= SPLIT) & fr["y"].notna()
                & (fr["hour"] == float(h))]
        if te.empty or len(tr) < 400:
            print(f"{h:>4}{len(te):>6} | skipped (train rows {len(tr)})")
            continue
        # ⚠️ SHIPPED FEATURE SET. A bare `IntradayModel(...)` pins this to
        # `INTRADAY_FEATURES` and measures a model NOBODY RUNS -- KLAX HIGH has
        # carried `RATE_FEATURES` since 2026-07-31. This script produces
        # `gating.HOUR_ACCURACY`, which the reliability table, the preflight
        # bands and `BAND_CALIB` all read, so a stale feature set here poisons
        # all three. Caught 2026-07-31. See `tests/test_eval_scripts.py`.
        _extra = [c for c in features_for(cfg.station.icao, spec.key)
                  if c not in INTRADAY_FEATURES]
        model = IntradayModel(extremum=spec.extremum, residual=resid,
                              extra_features=_extra).fit(tr)
        qs = model.predict_quantiles(te)
        y = te["y"].to_numpy(dtype=float)
        pt = np.array([q[0.5] for q in qs])
        pmfs = [bucket_probs(q) for q in qs]
        ae = np.abs(pt - y)
        cr = np.array([crps_pmf(p, int(v)) for p, v in zip(pmfs, y)])
        # PARITY-AVERAGED. Kalshi's ladder parity is a coin flip per day (see
        # buckets.expected_bucket_hit); the previous fixed-even scoring was
        # right about half the time and silently wrong the rest.
        hit = np.array([expected_bucket_hit(p, int(v)) for p, v in zip(pmfs, y)])
        stated = np.array([stated_prob(p) for p in pmfs])
        per_hour[h] = {"dates": te["target_date"].to_numpy(), "ae": ae, "cr": cr,
                       "hit": hit, "stated": stated}

    dh = DECISION_HOUR[metric]
    ref = per_hour.get(dh)
    for h, d in sorted(per_hour.items()):
        cmp_txt = ""
        if ref is not None and h != dh:
            # Pair on matched test dates so the delta is apples-to-apples.
            common = np.intersect1d(d["dates"], ref["dates"])
            if len(common) >= 100:
                mi = np.isin(d["dates"], common)
                ri = np.isin(ref["dates"], common)
                diff = d["ae"][mi] - ref["ae"][ri]
                n = len(diff)
                idx = rng.integers(0, n, size=(N_BOOT, n))
                ci = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
                sig = "SIG" if (ci[0] > 0) == (ci[1] > 0) else "ns"
                cmp_txt = f"{diff.mean():>+7.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {sig:>3}"
        elif h == dh:
            cmp_txt = "  (reference)"
        print(f"{h:>4}{len(d['ae']):>6} | {d['ae'].mean():>7.3f}{d['cr'].mean():>7.3f} | "
              f"{d['hit'].mean():>6.1%}{d['stated'].mean():>8.1%}"
              f"{d['stated'].mean() - d['hit'].mean():>+7.1%} | {cmp_txt}")

    if per_hour:
        best_mae = min(per_hour, key=lambda h: per_hour[h]["ae"].mean())
        best_hit = max(per_hour, key=lambda h: per_hour[h]["hit"].mean())
        print(f"  lowest MAE at hour {best_mae} "
              f"({per_hour[best_mae]['ae'].mean():.3f}); "
              f"highest bucket hit rate at hour {best_hit} "
              f"({per_hour[best_hit]['hit'].mean():.1%})")

print("\nnegative dMAE = that hour is MORE accurate than the decision hour")
print("positive calib = the modal bucket's stated probability is overconfident")
