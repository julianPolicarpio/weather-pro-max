"""WHERE does the model stop being better than the market? Condition on the edge.

CONTEXT. `scripts/edge_signal_diagnosis.py` found the MODEL beats the market on
the full ladder (Brier 0.395 vs 0.493, CI excluding zero). That refutes the
obvious explanation for `corr(edge, won) = -0.571` -- we are not simply the noisy
party. So the defect cannot be overall accuracy; it has to be in WHICH RUNG the
desk selects.

The desk picks the model's top bucket and trades it when `p_model - ask` clears
`min_edge`. That selection conditions on DISAGREEMENT. A model can be better on
average across a ladder and still be worse precisely where it disagrees most --
those are not contradictory, and only the second one gets traded.

WHAT IS MEASURED. Every rung of every priced day, binned by the size of the
disagreement `p_model - p_market`, with model and market scored against the
actual outcome inside each bin. Positive-edge rungs (the ones the desk would BUY)
are reported separately from negative-edge rungs, because only the first kind
ever becomes a trade.

⚠️ ONE ERROR FROM THE PREVIOUS SCRIPT IS CORRECTED HERE. That run applied
`gating.band_calib` to EVERY rung and renormalised, then reported that the
calibrated model scored worse than the raw one. That comparison was invalid:
`staking.evaluate` deliberately applies the calibration to the SELECTED bucket
only, and its own comment says why -- per-band offsets applied before the argmax
can REORDER the buckets and hand back a different pick. The "calibration makes it
worse" line from that run should be ignored; it measured a thing the desk does
not do. This script scores the RAW pmf.

⚠️ Rungs within a day are not independent. Bootstrap resampling is BY DAY, not
by rung, so the CIs do not pretend 6 rungs are 6 observations.

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_signal_conditional.py
"""
import json

import numpy as np

from highcast import pricehist
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
BINS = [(-1.01, -0.10), (-0.10, -0.03), (-0.03, 0.03), (0.03, 0.10),
        (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]
rng = np.random.default_rng(20260812)

print(__doc__.split("Usage:")[0])
recs = []          # one per RUNG, carrying its day id for clustered bootstrap
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
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra
                          ).fit(fr[fr["target_date"] < days[0]])
        blk = te[te["target_date"].isin(days)].sort_values("target_date")
        for (_, row), q in zip(blk.iterrows(), m.predict_quantiles(blk)):
            d = row["target_date"]
            y, lad = int(outcomes[d]), priced[d]
            asks = np.array([x.get("yes_ask") if x.get("yes_ask") is not None
                             else np.nan for x in lad], dtype=float)
            if np.isnan(asks).any() or asks.sum() <= 0:
                continue
            pmf = bucket_probs(q)
            raw = np.array([pmf_mass(pmf, x.get("lo"), x.get("hi")) for x in lad])
            if raw.sum() <= 0:
                continue
            pm = raw / raw.sum()
            pk = asks / asks.sum()
            day_id = f"{icao}:{mkey}:{d}"
            for i, x in enumerate(lad):
                hit = float((x.get("lo") is None or y >= x["lo"]) and
                            (x.get("hi") is None or y <= x["hi"]))
                recs.append({"day": day_id, "edge": float(pm[i] - pk[i]),
                             "pm": float(pm[i]), "pk": float(pk[i]),
                             "hit": hit, "ask": float(asks[i])})

days_all = sorted({r["day"] for r in recs})
by_day = {d: [r for r in recs if r["day"] == d] for d in days_all}
print(f"\n{'='*80}\nEVERY RUNG, BINNED BY DISAGREEMENT — {len(recs)} rungs over "
      f"{len(days_all)} day-markets\n{'='*80}")
print(f"  {'edge band':<16}{'rungs':>7}{'realised':>10}{'model says':>12}"
      f"{'market says':>13}{'model err':>11}{'mkt err':>9}{'winner':>9}")
for lo, hi in BINS:
    sel = [r for r in recs if lo <= r["edge"] < hi]
    if len(sel) < 15:
        continue
    real = np.mean([r["hit"] for r in sel])
    pm = np.mean([r["pm"] for r in sel])
    pk = np.mean([r["pk"] for r in sel])
    em = np.mean([(r["pm"] - r["hit"]) ** 2 for r in sel])
    ek = np.mean([(r["pk"] - r["hit"]) ** 2 for r in sel])
    win = "model" if em < ek else "MARKET"
    print(f"  {lo:+.2f}..{hi:+.2f}    {len(sel):>7}{real:>10.1%}{pm:>12.1%}"
          f"{pk:>13.1%}{em:>11.4f}{ek:>9.4f}{win:>9}")

# ── the rungs the desk would actually BUY ───────────────────────────────────
print(f"\n{'='*80}\nTHE TRADEABLE SUBSET: rungs where the model is the top pick "
      f"AND edge >= 0.03\n{'='*80}")
picks = []
for d, rs in by_day.items():
    top = max(rs, key=lambda r: r["pm"])
    if top["edge"] >= 0.03:
        picks.append((d, top))
if picks:
    hit = np.mean([p[1]["hit"] for p in picks])
    stated = np.mean([p[1]["pm"] for p in picks])
    mkt = np.mean([p[1]["pk"] for p in picks])
    print(f"  {len(picks)} picks   model says {stated:.1%}   market says {mkt:.1%}"
          f"   ACTUALLY hit {hit:.1%}")
    print(f"  -> the model overstates its own pick by "
          f"{stated - hit:+.1%}; the market by {mkt - hit:+.1%}")
    for lo, hi in [(0.03, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]:
        s = [p for p in picks if lo <= p[1]["edge"] < hi]
        if not s:
            continue
        print(f"     edge {lo:.2f}-{hi:.2f}: n={len(s):<3} stated "
              f"{np.mean([x[1]['pm'] for x in s]):.1%}  hit "
              f"{np.mean([x[1]['hit'] for x in s]):.1%}")
print("\n  ⚠️ n is small in every row above. Read the DIRECTION, and note that")
print("  the whole point of this project's standing rule is that a threshold")
print("  fitted to these counts would be fitting noise.")
