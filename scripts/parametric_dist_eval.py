"""19 INDEPENDENT QUANTILE BOOSTERS vs ONE SMOOTH PARAMETRIC DISTRIBUTION.

THE GOAL THIS IS JUDGED AGAINST: accurate predictions BEFORE the extremum has
happened. So every headline here is restricted to PREDICTIVE hours -- those where
`gating.EXTREMUM_STILL_FUTURE` says the extreme is still ahead on most days.
Scoring all 24 hours flatters any model, because accuracy climbs through the day
for the trivial reason that the answer stops being unknown. An arm that wins only
after the extremum has formed has won nothing.

WHY THIS ARM EXISTS. Production fits NINETEEN separate LightGBM models, one per
quantile level (LEVELS 0.05..0.95). Nothing couples them. Small data changes can
reshuffle them independently, which reshapes the implied PMF and flips the modal
2F bucket -- the "why does it keep changing" churn, arriving from the PROBABILITY
side rather than the point-forecast side.

A parametric fit cannot do that. Estimate a location and a scale (and one global
shape), and the distribution is smooth by construction: three numbers, no
internal kinks, no crossing quantiles. The idea came from surveying comparable
work -- Oalkhadra/prediction-market-trading trades the same Kalshi buckets using
a "Student-t predictive distribution per bucket" rather than a quantile ensemble.

    mu     LightGBM regression on the frame            -> location
    sigma  LightGBM on log|residual| (out-of-fold)     -> scale, varies by day
    nu     one global df from standardised residuals   -> tail weight

Student-t rather than Gaussian because temperature busts have fat tails: the
marine-layer/convection failures that matter are exactly the ones a Gaussian
underweights, and mispricing a tail is how a bucket bettor loses.

Run on KMIA. KLAX is mid-test-week and frozen; this reads nothing production
writes and changes no shipped code.

RESULT 2026-07-29: NEGATIVE. DO NOT RE-RUN THIS HYPOTHESIS.
Judged on predictive hours only, KMIA, holdout 2025-01-01:

    LOW   dMAE +0.149  dCRPS +0.101  dHIT -4.1 pts  dCHURN +0.01
    HIGH  dMAE +0.113  dCRPS +0.093  dHIT -2.2 pts  dCHURN +0.18

The 19 quantile boosters win outright, and the parametric arm did not even buy
the stability that motivated it -- churn ROSE for the HIGH. Note `nu` pinned at
its 30.0 clamp at every hour: the standardised residuals came out Gaussian-ish,
so the fat-tail premise was not supported at this scale. A joint location-scale
fit (NGBoost-style) rather than this two-stage approximation might close some of
the gap, but -4.1 points on the LOW is not a tuning-sized deficit.

Usage:  .venv/bin/python scripts/parametric_dist_eval.py [config]
"""
import sys

import numpy as np
from scipy import stats

from highcast import gating
from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml"
SPLIT = "2025-01-01"
HOURS = tuple(range(24))
# An hour counts as PREDICTIVE when the extremum is still ahead on at least this
# share of days. 0.60 keeps the hours a human could actually act on and drops the
# ones where the model is reading a thermometer.
PREDICTIVE_MIN = 0.60
GRID = np.arange(-40, 140)          # integer degF support for the PMF

cfg = load_config(CONFIG)
conn = connect(cfg.db_path)
dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? ORDER BY date", (cfg.station.icao,))]
print(f"station {cfg.station.icao}  n_cli={len(dates)}  split={SPLIT}", flush=True)


def lgbm(x, y, obj="regression"):
    import lightgbm as lgb
    return lgb.train({"objective": obj, "learning_rate": 0.05, "num_leaves": 31,
                      "min_data_in_leaf": 40, "feature_fraction": 0.9,
                      "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1,
                      "seed": 0},
                     lgb.Dataset(x, label=y), num_boost_round=300)


def t_pmf(mu, sigma, nu):
    """PMF over integer degF from a Student-t, by differencing its CDF.

    Integrating the CDF across each integer's [t-0.5, t+0.5] cell is the right
    discretisation: the CLI value is a ROUNDED integer, so the probability of
    reporting t is the mass within half a degree of t -- not the density at t.
    """
    sigma = max(float(sigma), 0.05)
    hi = stats.t.cdf((GRID + 0.5 - mu) / sigma, df=nu)
    lo = stats.t.cdf((GRID - 0.5 - mu) / sigma, df=nu)
    p = np.clip(hi - lo, 0.0, None)
    s = p.sum()
    return {int(k): float(v / s) for k, v in zip(GRID, p) if v / s > 1e-6} if s > 0 else {}


for metric in ("low", "high"):
    spec = get_metric(cfg, metric)
    resid = bool(getattr(cfg.model, f"intraday_residual_{metric}", False))
    fut = gating.EXTREMUM_STILL_FUTURE.get(cfg.station.icao, {}).get(metric, {})
    pred_hours = [h for h in HOURS if fut.get(h, 0.0) >= PREDICTIVE_MIN]
    print(f"\n=== {metric.upper()}  predictive hours (still-future >= "
          f"{PREDICTIVE_MIN:.0%}): {pred_hours} ===", flush=True)

    fr = build_checkpoint_frame(conn, cfg, dates, hours=HOURS, metric=spec)
    fr = fr[fr["y"].notna()]
    feat = [c for c in fr.columns if c not in ("target_date", "y")]

    rows = []
    bk = {"quant": {}, "param": {}}
    for h in HOURS:
        if h not in pred_hours:
            continue
        tr = fr[(fr["target_date"] < SPLIT) & (fr["hour"] == float(h))]
        te = fr[(fr["target_date"] >= SPLIT) & (fr["hour"] == float(h))]
        if len(te) < 100 or len(tr) < 400:
            continue
        y = te["y"].to_numpy(dtype=float)
        d = te["target_date"].to_numpy()
        xtr, xte = tr[feat].to_numpy(dtype=float), te[feat].to_numpy(dtype=float)

        # --- ARM 1: production, 19 independent quantile boosters ---
        qm = IntradayModel(extremum=spec.extremum, residual=resid).fit(tr)
        p_quant = [bucket_probs(q) for q in qm.predict_quantiles(te)]

        # --- ARM 2: parametric Student-t ---
        # sigma is fit on OUT-OF-FOLD residuals: training mu's residuals are
        # shrunk by its own fit, so an in-sample sigma is optimistically small
        # and the tails come out too thin exactly where they matter.
        cut = len(tr) // 2
        mu_a = lgbm(xtr[:cut], tr["y"].to_numpy(dtype=float)[:cut])
        mu_b = lgbm(xtr[cut:], tr["y"].to_numpy(dtype=float)[cut:])
        oof = np.concatenate([mu_b.predict(xtr[:cut]), mu_a.predict(xtr[cut:])])
        res = tr["y"].to_numpy(dtype=float) - oof
        mu_full = lgbm(xtr, tr["y"].to_numpy(dtype=float))
        sig_m = lgbm(xtr, np.log(np.abs(res) + 0.25))
        mu_te = mu_full.predict(xte)
        sig_te = np.exp(sig_m.predict(xte)) * np.sqrt(np.pi / 2)   # |N| -> sd
        z = res / np.maximum(np.exp(sig_m.predict(xtr)) * np.sqrt(np.pi / 2), 0.05)
        nu = max(2.5, min(30.0, float(stats.t.fit(z, floc=0, fscale=1)[0])))
        p_param = [t_pmf(m, s, nu) for m, s in zip(mu_te, sig_te)]

        out = {}
        for name, pmfs in (("quant", p_quant), ("param", p_param)):
            pt = np.array([sum(k * v for k, v in p.items()) for p in pmfs])
            out[name] = (float(np.abs(pt - y).mean()),
                         float(np.mean([crps_pmf(p, int(v)) for p, v in zip(pmfs, y)])),
                         float(np.mean([expected_bucket_hit(p, int(v))
                                        for p, v in zip(pmfs, y)])))
            bk[name][h] = {dd: modal_bucket(p, 0)[0] for dd, p in zip(d, pmfs)}
        rows.append((h, len(te), fut.get(h), out["quant"], out["param"], nu))

    print(f"  {'h':>3}{'n':>6}{'fut':>6} | {'QUANT mae':>10}{'crps':>7}{'hit':>7} | "
          f"{'PARAM mae':>10}{'crps':>7}{'hit':>7} | {'nu':>5}", flush=True)
    for h, n, f, (ma, ca, ha), (mp, cp, hp) in [(r[0], r[1], r[2], r[3], r[4]) for r in rows]:
        nu = [r[5] for r in rows if r[0] == h][0]
        print(f"  {h:>3}{n:>6}{(f or 0):>6.0%} | {ma:>10.3f}{ca:>7.3f}{ha:>7.1%} | "
              f"{mp:>10.3f}{cp:>7.3f}{hp:>7.1%} | {nu:>5.1f}", flush=True)
    if rows:
        dm = np.mean([r[4][0] - r[3][0] for r in rows])
        dc = np.mean([r[4][1] - r[3][1] for r in rows])
        dh = np.mean([r[4][2] - r[3][2] for r in rows]) * 100
        print(f"  PREDICTIVE-HOUR MEAN  dMAE {dm:+.3f}  dCRPS {dc:+.3f}  "
              f"dHIT {dh:+.1f} pts   (negative dMAE/dCRPS = parametric BETTER)",
              flush=True)
        lo, hi = min(r[0] for r in rows), max(r[0] for r in rows)

        def churn(b):
            days = {}
            for hh in range(lo, hi + 1):
                for dd, lab in b.get(hh, {}).items():
                    days.setdefault(dd, []).append(lab)
            return float(np.mean([len(set(v)) for v in days.values()])) if days else float("nan")
        cq, cpm = churn(bk["quant"]), churn(bk["param"])
        print(f"  CHURN h{lo}-h{hi}:  quantile {cq:.2f}  parametric {cpm:.2f} "
              f"distinct buckets/day  ({cpm - cq:+.2f})", flush=True)

print("\nJudged ONLY on hours where the extremum is still ahead -- accuracy after")
print("it has formed is transcription, not prediction.")
