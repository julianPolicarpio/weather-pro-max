"""WHY IS THE EDGE SIGNAL ANTI-PREDICTIVE? Diagnose before attempting a fix.

THE FACT. Over 48 settled bets, `corr(edge, won) = -0.571`; PLACE hits 43% and
SKIP hits 91%. The desk loses precisely where it believes it has found value.

THE QUESTION THIS SCRIPT ANSWERS, and it is NOT "what threshold fixes it":

    Is the market better calibrated than our model?

If it is, then "edge" is mostly OUR ERROR, and a negative correlation between
edge and winning is a mathematical consequence rather than a bug. You cannot
patch that with a threshold; the only sound response is to shrink toward the
market, or to stop trading on disagreement at all.

⚠️ WHY THIS IS BETTER POWERED THAN THE MONEY TEST. The money test has 18
placements. This scores the ENTIRE LADDER every day with a proper scoring rule
(multiclass Brier + log loss), so each day contributes the full distribution
rather than one binary outcome. Rungs within a day are NOT independent -- they
sum to one -- so the effective n is still the ~47 days, but a proper score
extracts far more information per day than win/loss does.

⚠️ MARKET PRICES ARE NOT PROBABILITIES until they are normalised. Kalshi asks
across a ladder sum to more than 1 (the spread is the house's). Comparing a
normalised model pmf to un-normalised asks would hand the model a fake win, so
the asks are divided by their sum and the raw overround is reported.

⚠️ NO FITTING HAPPENS HERE. The one number estimated -- the shrinkage weight
that would have been optimal -- is reported WITH its bootstrap CI precisely so
that its uncertainty is visible, and it is not applied to anything.

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_signal_diagnosis.py
"""
import json
from datetime import date

import numpy as np

from highcast import gating, pricehist
from highcast.buckets import bucket_probs, pmf_mass
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)

N_BOOT = 4000
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]
rng = np.random.default_rng(20260812)
EPS = 1e-6


def brier(p: np.ndarray, k: int) -> float:
    t = np.zeros_like(p)
    t[k] = 1.0
    return float(((p - t) ** 2).sum())


def logloss(p: np.ndarray, k: int) -> float:
    return float(-np.log(max(p[k], EPS)))


print(__doc__.split("Usage:")[0])
rows = []
for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    cfg = load_config(cfgp)
    icao = cfg.station.icao
    conn = connect(cfg.db_path, migrate=False)
    for mkey in ("high", "low"):
        spec = get_metric(cfg, mkey)
        hour = MIN_DECISION_HOUR[icao][mkey]
        resid = bool(getattr(cfg.model, f"intraday_residual_{mkey}", False))
        k = cfg.model.intraday_bandwidth_for(mkey)
        tr_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
        extra = [c for c in features_for(icao, mkey) if c not in _SHIPPED_BASE]

        dates = [r["date"] for r in conn.execute(
            "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (icao,))]
        day_df = build_features(conn, cfg, spec, dates, require_target=True)
        base = BaselineBlend(
            prefer_corrected=uses_bias_correction(spec.key)).fit(day_df)
        fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec,
                                    baseline=base)
        fr = fr[fr["y"].notna()].copy()

        # ⚠️ MUST be the snapshot at/after the COMMIT hour. Using the day's
        # first snapshot paired a 17:00Z prediction with a 01:02Z quote and
        # handed the model 16 hours of hindsight. See `highcast.pricehist`.
        priced = pricehist.ladders_at_commit(conn, icao, mkey, hour,
                                             cfg.station.tz)
        outcomes = {r["date"]: r[spec.target_col] for r in conn.execute(
            f"SELECT date, {spec.target_col} FROM cli_reports WHERE station=?"
            f" AND {spec.target_col} IS NOT NULL", (icao,))}

        te = fr[fr["hour"] == float(hour)]
        days = sorted(d for d in priced if d in outcomes
                      and d in set(te["target_date"]))
        if not days:
            continue
        cut = days[0]
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(fr[fr["target_date"] < cut])
        blk = te[te["target_date"].isin(days)].sort_values("target_date")
        for (_, row), q in zip(blk.iterrows(), m.predict_quantiles(blk)):
            d = row["target_date"]
            y, lad = int(outcomes[d]), priced[d]
            asks = np.array([x.get("yes_ask") if x.get("yes_ask") is not None
                             else np.nan for x in lad], dtype=float)
            if np.isnan(asks).any() or asks.sum() <= 0:
                continue
            # which rung actually settled?
            won = [((x.get("lo") is None or y >= x["lo"]) and
                    (x.get("hi") is None or y <= x["hi"])) for x in lad]
            if sum(won) != 1:
                continue                       # ladder did not cover the outcome
            kidx = won.index(True)
            pmf = bucket_probs(q)
            raw = np.array([pmf_mass(pmf, x.get("lo"), x.get("hi")) for x in lad])
            pm = raw / raw.sum() if raw.sum() > 0 else None
            if pm is None:
                continue
            # calibrated model probability, as the desk would use it
            pc = np.array([gating.band_calib(icao, mkey, hour, float(v)) or v
                           for v in pm])
            pc = pc / pc.sum()
            overround = float(asks.sum())
            pk = asks / overround
            rows.append({
                "station": icao, "metric": mkey, "date": d, "k": kidx,
                "overround": overround, "pm": pm, "pc": pc, "pk": pk,
            })

print(f"\n{'='*78}\nMODEL vs MARKET on the FULL LADDER — n={len(rows)} "
      f"priced+settled days\n{'='*78}")
ov = np.array([r["overround"] for r in rows])
print(f"  mean Kalshi overround (sum of asks): {ov.mean():.3f}  "
      f"-> the house edge before anyone is right about anything\n")

# "model (calibrated)" is printed for reference ONLY. It is NOT the comparison
# below, because per-rung calibration is not a thing the desk does.
for label, key in (("model (raw)  <- the comparison", "pm"),
                   ("model (per-rung calib, ref only)", "pc"),
                   ("MARKET (normalised asks)", "pk")):
    b = np.array([brier(r[key], r["k"]) for r in rows])
    l = np.array([logloss(r[key], r["k"]) for r in rows])
    print(f"  {label:<26} Brier {b.mean():.4f}   log loss {l.mean():.4f}")

# ⚠️ RAW, NOT CALIBRATED. `gating.band_calib` is applied by `staking.evaluate`
# to the SELECTED bucket ONLY, never to every rung before the argmax — its own
# comment says why: per-band offsets applied across the ladder can REORDER the
# buckets. The first version of this script scored the per-rung-calibrated,
# renormalised pmf and reported the difference as +0.0995. That number described
# an object the desk never constructs. The raw pmf IS the model's ladder.
bm = np.array([brier(r["pm"], r["k"]) for r in rows])
bk = np.array([brier(r["pk"], r["k"]) for r in rows])
d = bm - bk                     # >0 means the MARKET is better
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
lo, hi = np.percentile(boot, [2.5, 97.5])
verdict = ("MARKET is better calibrated" if lo > 0 else
           "MODEL is better calibrated" if hi < 0 else
           "indistinguishable at this n")
print(f"\n  Brier difference (model - market) {d.mean():+.4f}  "
      f"95% CI [{lo:+.4f},{hi:+.4f}]")
print(f"  -> {verdict}")

# ── how much would shrinking toward the market have helped? ─────────────────
print(f"\n{'='*78}\nSHRINKAGE CURVE  p = w*model + (1-w)*market\n{'='*78}")
print(f"  {'w':>5}{'Brier':>10}{'log loss':>11}   (w=1 is today's desk, w=0 is "
      f"'just trade the market')")
best_w, best_b = None, None
for w in np.arange(0.0, 1.01, 0.1):
    p = [w * r["pm"] + (1 - w) * r["pk"] for r in rows]
    b = np.mean([brier(pi, r["k"]) for pi, r in zip(p, rows)])
    l = np.mean([logloss(pi, r["k"]) for pi, r in zip(p, rows)])
    print(f"  {w:>5.1f}{b:>10.4f}{l:>11.4f}")
    if best_b is None or b < best_b:
        best_b, best_w = b, w


def opt_w(idx):
    ws = np.arange(0.0, 1.01, 0.05)
    scores = [np.mean([brier(w * rows[i]["pc"] + (1 - w) * rows[i]["pk"],
                             rows[i]["k"]) for i in idx]) for w in ws]
    return ws[int(np.argmin(scores))]


bw = np.array([opt_w(rng.integers(0, len(rows), len(rows))) for _ in range(400)])
print(f"\n  Brier-optimal w on this sample: {best_w:.2f}")
print(f"  bootstrap 95% CI on that w: [{np.percentile(bw, 2.5):.2f}, "
      f"{np.percentile(bw, 97.5):.2f}]")
print("\n  ⚠️ THE CI IS THE POINT. A shrinkage weight is one scalar, which is")
print("  far cheaper to estimate than a threshold — but if that interval is")
print("  wide, the honest reading is still 'not enough data to tune', and no")
print("  value from this run should be written into the desk.")
