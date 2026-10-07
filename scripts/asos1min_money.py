"""DOES THE 1-MINUTE BOUND MOVE MONEY? The bar four ideas have already failed.

⚠️ READ THE POWER CALCULATION BEFORE THE RESULT. QUEUE item 1 measured
per-contract net at sd 0.404, so the smallest effect detectable at 60 settled
bets is 0.236/contract and a realistic 0.05 edge needs ~512. The priced history
here is ~15-20 days PER MARKET (Kalshi prices begin 2026-07-23 and cannot be
backfilled -- Kalshi does not sell them). This test is UNDERPOWERED BY
CONSTRUCTION and cannot return a positive verdict.

It is still worth running for exactly one reason: it can detect a CATASTROPHE.
A feature that lifts hit rate 7 points while destroying P&L is precisely the
shape of the four ideas that died here -- the confidence gate raised hit rate
and inverted returns, because it bought at 60c what it used to buy at 31c. If
the 1-minute bound does the same, this will show it even at n=15.

WHAT IS SIMULATED. For each priced+settled day, both arms produce a pmf from
models trained ONLY on prior days, the REAL Kalshi ladder from `signals` is
attached, and the full shipped desk runs on it: `gating.band_calib` +
`hour_calib`, then `staking.evaluate` (min-edge, MAX_EDGE ceiling, liquidity).
Net per contract is settled against the CLI outcome with the real taker fee.

⚠️ Paired on DAY, so both arms face identical prices. Any difference is the
feature, not the market.

⚠️ Same lookahead guard as the walk-forward (`LOOKAHEAD_GUARD_MIN`). Centred QC
plus an as-of slice leaks; that bug cost up to 1.7 points of dHIT before it was
caught.

Usage: PYTHONPATH=src .venv/bin/python scripts/asos1min_money.py
"""
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


def net_per_contract(dec, y: int, ladder: list[dict]) -> float:
    """Realised net on ONE contract, or 0.0 when the desk refuses."""
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
rows_all = []
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

        # one ladder per priced day: the LAST snapshot at/after the commit hour
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
            print(f"\n### {icao} {mkey.upper()}: no priced+settled days with "
                  f"1-minute coverage"); continue

        # train once on everything strictly before the first priced day: the
        # priced window is 3 weeks, so a refit inside it would train on <20 new
        # rows and add noise, not adaptivity.
        cut = days[0]
        trA, trB = fr[fr["target_date"] < cut], frB[frB["target_date"] < cut]
        mA = IntradayModel(extremum=spec.extremum, residual=resid,
                           extra_features=extra).fit(trA)
        mB = IntradayModel(extremum=spec.extremum, residual=resid,
                           extra_features=extra).fit(trB)
        blkA = te[te["target_date"].isin(days)].sort_values("target_date")
        blkB = frB.loc[blkA.index]
        qA, qB = mA.predict_quantiles(blkA), mB.predict_quantiles(blkB)

        recs = []
        for (_, row), qa, qb in zip(blkA.iterrows(), qA, qB):
            d = row["target_date"]
            y = int(outcomes[d])
            lad = priced[d]
            per_arm = {}
            for tag, q in (("A", qa), ("B", qb)):
                pmf = bucket_probs(q)
                ranges = []
                for x in lad:
                    ask = x.get("yes_ask")
                    ranges.append({**x, "p_model": pmf_mass(pmf, x.get("lo"),
                                                            x.get("hi")),
                                   "market_price": ask})
                dec = evaluate(
                    ranges, cfg.staking, within_window=True,
                    calib=lambda p: gating.band_calib(icao, mkey, hour, p),
                    extremum=spec.extremum)
                per_arm[tag] = (dec, net_per_contract(dec, y, lad))
            recs.append((d, per_arm["A"], per_arm["B"]))
            rows_all.append((icao, mkey, d, per_arm["A"][1], per_arm["B"][1],
                             per_arm["A"][0].place, per_arm["B"][0].place))

        nA = np.array([r[1][1] for r in recs])
        nB = np.array([r[2][1] for r in recs])
        pA = sum(1 for r in recs if r[1][0].place)
        pB = sum(1 for r in recs if r[2][0].place)
        print(f"\n### {icao} {mkey.upper()} h{hour}   {len(recs)} priced+settled days")
        print(f"    arm A (METAR):  placed {pA:>2}   net/day {nA.mean():+.4f}"
              f"   total {nA.sum():+.3f}")
        print(f"    arm B (1-min):  placed {pB:>2}   net/day {nB.mean():+.4f}"
              f"   total {nB.sum():+.3f}")
        diff = nB - nA
        changed = [r for r in recs if abs(r[2][1] - r[1][1]) > 1e-9]
        print(f"    differed on {len(changed)} of {len(recs)} days")
        for d, a, b in changed:
            print(f"      {d}  A {a[0].bucket if a[0].place else 'NO BET':<8}"
                  f"{a[1]:+.3f}   ->   B {b[0].bucket if b[0].place else 'NO BET':<8}"
                  f"{b[1]:+.3f}")

print("\n" + "=" * 74)
if rows_all:
    A = np.array([r[3] for r in rows_all])
    B = np.array([r[4] for r in rows_all])
    d = B - A
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"POOLED across all 4 markets, n={len(d)} priced+settled days")
    print(f"   arm A net/day {A.mean():+.4f}   arm B net/day {B.mean():+.4f}")
    print(f"   difference {d.mean():+.4f}/contract   95% CI [{lo:+.4f},{hi:+.4f}]")
    print(f"   sd of per-day net: A {A.std():.3f}  B {B.std():.3f}")
    mde = 2.8 * d.std() / np.sqrt(len(d))
    print(f"   smallest effect this n could detect (~80% power): {mde:.3f}/contract")
    print("\n   ⚠️ Compare that MDE to a realistic 0.02-0.05 edge before reading")
    print("   anything into the point estimate. A CI spanning zero here is the")
    print("   EXPECTED result and is not evidence the feature does not work.")
