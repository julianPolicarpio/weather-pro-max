"""MIN_EDGE SENSITIVITY for the 1-minute bound — NOT a tuning run.

⚠️ WHY THIS IS NOT ALLOWED TO PICK A THRESHOLD. QUEUE item 1 carries a standing
rule: "Do not sweep MIN_EDGE on this sample. n=28 makes the sign credible; it
does not make any threshold fitted to it credible." The sample is now 47 priced
days and the rule is unchanged -- the smallest effect detectable here is
0.124/contract, so ANY threshold that looks best on these days is fitting noise
of exactly that magnitude. Reading a winner off the table below and shipping it
would be the single most overfit change this project could make.

What the sweep is legitimately for:

  1. DOMINANCE. If arm B beats arm A at every threshold, that is a far stronger
     statement than one point estimate, and it cannot be produced by picking a
     lucky cut. If the curves cross, the difference is noise and the honest
     answer is that there is nothing here.

  2. TURNOVER MATCHING, which uses NO outcome information and is therefore not
     fitting. The money test found arm B places 22% more often (18 -> 22) because
     a tighter ratchet bound concentrates the pmf and more days clear min_edge.
     That confounds "trades better" with "trades more". Raising B's threshold
     until it places as often as A removes the confound. The threshold is chosen
     from PLACEMENT COUNTS ONLY -- never from P&L.

Everything else (arms, training cut, real ladders, real fees, lookahead guard)
is identical to `scripts/asos1min_money.py`.

Usage: PYTHONPATH=src .venv/bin/python scripts/asos1min_money_sweep.py
"""
import dataclasses
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from highcast import asos1min, gating, pricehist
from highcast.buckets import bucket_probs, pmf_mass
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.fees import fee_per_contract
from highcast.gating import MIN_DECISION_HOUR
from highcast.latentmax import METAR_TOL
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)
from highcast.staking import evaluate
from highcast.timeutil import climate_day_bounds

SAFETY_F, MIN_WINDOW_SAMPLES, LOOKAHEAD_GUARD_MIN = 0.5, 60, 8
SHIPPED_MIN_EDGE = 0.03
GRID = [0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25]
N_BOOT = 4000
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]
rng = np.random.default_rng(20260812)


def qc_bound(cfg, grid, vals, d, hour, extremum):
    start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    asof = datetime(d.year, d.month, d.day, hour,
                    tzinfo=ZoneInfo(cfg.station.tz)).astimezone(ZoneInfo("UTC"))
    a = int(start.timestamp()) // 60
    b = int(asof.timestamp()) // 60 - LOOKAHEAD_GUARD_MIN
    i0, i1 = np.searchsorted(grid, a), np.searchsorted(grid, b)
    if i1 <= i0:
        return None
    seg = vals[i0:i1]
    if int(np.count_nonzero(~np.isnan(seg))) < MIN_WINDOW_SAMPLES:
        return None
    v = np.nanmax(seg) if extremum == "max" else np.nanmin(seg)
    return None if np.isnan(v) else float(v)


def net_per_contract(dec, y, ladder):
    if not dec.place or dec.price is None:
        return 0.0
    rung = next((r for r in ladder if r["label"] == dec.bucket), None)
    if rung is None:
        return 0.0
    lo, hi = rung.get("lo"), rung.get("hi")
    won = ((lo is None or y >= lo) and (hi is None or y <= hi))
    p = float(dec.price)
    fee = fee_per_contract(1, p)
    return (1.0 - p - fee) if won else (-p - fee)


print(__doc__.split("Usage:")[0])
# arm -> min_edge -> list of (net, placed) across every market/day
book = {"A": {e: [] for e in GRID}, "B": {e: [] for e in GRID}}

for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    cfg = load_config(cfgp)
    icao = cfg.station.icao
    conn = connect(cfg.db_path, migrate=False)
    valid, temps = asos1min.load_station(conn, icao)
    grid, vals1m = asos1min.qc_series(valid, temps)

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
        fr["b1"] = [qc_bound(cfg, grid, vals1m, date.fromisoformat(td), int(hh),
                             spec.extremum)
                    for td, hh in zip(fr["target_date"], fr["hour"])]
        fr = fr[fr["b1"].notna()].copy()
        frB = fr.copy()
        if spec.extremum == "max":
            frB["run_lo"] = fr["b1"] - METAR_TOL - SAFETY_F
            frB["run_hi"] = fr["b1"] + METAR_TOL
        else:
            frB["run_lo"] = fr["b1"] - METAR_TOL
            frB["run_hi"] = fr["b1"] + METAR_TOL + SAFETY_F
        frB["run_width"] = frB["run_hi"] - frB["run_lo"]

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
        mA = IntradayModel(extremum=spec.extremum, residual=resid,
                           extra_features=extra).fit(fr[fr["target_date"] < cut])
        mB = IntradayModel(extremum=spec.extremum, residual=resid,
                           extra_features=extra).fit(frB[frB["target_date"] < cut])
        blkA = te[te["target_date"].isin(days)].sort_values("target_date")
        blkB = frB.loc[blkA.index]

        for (_, row), qa, qb in zip(blkA.iterrows(),
                                    mA.predict_quantiles(blkA),
                                    mB.predict_quantiles(blkB)):
            d, y, lad = row["target_date"], int(outcomes[row["target_date"]]), \
                priced[row["target_date"]]
            for tag, q in (("A", qa), ("B", qb)):
                pmf = bucket_probs(q)
                ranges = [{**x, "p_model": pmf_mass(pmf, x.get("lo"), x.get("hi")),
                           "market_price": x.get("yes_ask")} for x in lad]
                for e in GRID:
                    sc = dataclasses.replace(cfg.staking, min_edge=e)
                    dec = evaluate(ranges, sc, within_window=True,
                                   calib=lambda p: gating.band_calib(icao, mkey,
                                                                     hour, p),
                                   extremum=spec.extremum)
                    book[tag][e].append((net_per_contract(dec, y, lad),
                                         bool(dec.place)))

n_days = len(book["A"][GRID[0]])
print(f"\n{'='*78}\nSENSITIVITY CURVE — n={n_days} priced+settled market-days\n{'='*78}")
print(f"  {'min_edge':>9} | {'A places':>9}{'A net/day':>12} | "
      f"{'B places':>9}{'B net/day':>12} | {'B - A':>9}")
stats = {}
for e in GRID:
    a = np.array([x[0] for x in book["A"][e]])
    b = np.array([x[0] for x in book["B"][e]])
    pa = sum(1 for x in book["A"][e] if x[1])
    pb = sum(1 for x in book["B"][e] if x[1])
    stats[e] = (a, b, pa, pb)
    star = "  <-- shipped" if e == SHIPPED_MIN_EDGE else ""
    print(f"  {e:>9.2f} | {pa:>9}{a.mean():>+12.4f} | {pb:>9}{b.mean():>+12.4f} | "
          f"{b.mean()-a.mean():>+9.4f}{star}")

wins = sum(1 for e in GRID if stats[e][1].mean() > stats[e][0].mean())
print(f"\n  arm B ahead at {wins} of {len(GRID)} thresholds")
if wins == len(GRID):
    print("  -> B DOMINATES across the whole range")
elif wins == 0:
    print("  -> A DOMINATES across the whole range: retuning min_edge does NOT")
    print("     rescue B, so the 'trades more, not better' story does not hold")
else:
    print("  -> the curves CROSS; the difference is threshold-dependent noise")
print("  ⚠️ These thresholds are NOT 8 independent tests. The placement sets are")
print("     nested on the same 47 days, so a consistent sign across the row is")
print("     roughly ONE observation, not eight. Do not read it as 8-for-8.")

# ── turnover matching: chosen from PLACEMENT COUNTS ONLY, never from P&L ─────
target = stats[SHIPPED_MIN_EDGE][2]
matched = min(GRID, key=lambda e: (abs(stats[e][3] - target), e))
a0 = stats[SHIPPED_MIN_EDGE][0]
bm = stats[matched][1]
print(f"\n{'='*78}\nTURNOVER-MATCHED COMPARISON\n{'='*78}")
print(f"  arm A at the shipped min_edge {SHIPPED_MIN_EDGE:.2f}: {target} placements")
print(f"  arm B needs min_edge {matched:.2f} to place {stats[matched][3]} times")
print(f"    A net/day {a0.mean():+.4f}    B net/day {bm.mean():+.4f}")
d = bm - a0
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"    difference {d.mean():+.4f}/contract   95% CI [{lo:+.4f},{hi:+.4f}]")
print(f"    MDE at this n (~80% power): {2.8*d.std()/np.sqrt(len(d)):.3f}/contract")
print("\n  At matched turnover the question is purely 'does it trade BETTER'.")
print("  ⚠️ The threshold above was picked to equalise PLACEMENTS, not returns.")
print("  ⚠️ Do NOT ship any number from the sensitivity table — see the header.")
