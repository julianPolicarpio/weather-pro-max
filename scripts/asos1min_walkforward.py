"""DOES A 1-MINUTE RATCHET BOUND BEAT THE METAR ONE? Walk-forward, paired.

The case that motivated this: KMIA 2026-08-11 peaked at 18:39Z, fourteen minutes
before the 18:53Z filing and 0.9F above it. `visible_running_extremum` read
92.92-93.12 all afternoon; the truth was 93.92 and it settled 94. One rung.

⚠️ WHAT IS AND IS NOT BEING TESTED. The METAR interval is already NARROW -- the
T-group gives 0.1F precision, so a typical shipped interval is ~0.2F wide. The
1-minute bound is not narrower. Its advantage is that its CENTRE is right: it
sees the peak between filings. So this measures ACCURACY OF THE BOUND, not its
precision, and anyone reading `run_width` as the mechanism has the wrong model.

SAFETY MARGIN 0.5F, applied UNIFORMLY. `scripts/asos1min_ratchet_gate.py` found
KLAX high/low and KMIA high violate at 0.00% with no margin, but KMIA low
violates at 1.21% (6/497, worst 0.50F) because the QC'd running min occasionally
dips below the settled low. 0.5F takes every market to 0.00%. Applied everywhere
rather than only where needed -- a per-market margin fitted on ~500 days is
fitting noise, and the uniform version costs the three clean markets almost
nothing.

⚠️ THIS IS NOT DEPLOYABLE TODAY and the result must not be read as if it were.
`obs_1min` lags 24-36h, so arm B is a measurement of what a LIVE sub-hourly feed
would be worth (QUEUE item 13: `fetchers/fivemin.py` needs a Synoptic token).
Training on it while serving from METAR is the train/serve skew this whole line
of work exists to avoid.

⚠️ ARMS SEE IDENTICAL ROWS. Days without 1-minute coverage are dropped from BOTH
arms, and both are refit on the same training rows at every step. Anything less
and the comparison measures sample composition instead of the feature.

THE BAR: dHIT with a paired-bootstrap CI excluding zero. And per the standing
rule that has killed four ideas in this project -- hit rate is necessary, not
sufficient. A win here still has to move MONEY before it ships.

Usage: PYTHONPATH=src .venv/bin/python scripts/asos1min_walkforward.py
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from highcast import asos1min, gating
from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.latentmax import METAR_TOL
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)
from highcast.timeutil import climate_day_bounds

REFIT, MIN_TRAIN, RECENT, N_BOOT = 30, 400, "2025-01-01", 4000
SAFETY_F = 0.5
MIN_WINDOW_SAMPLES = 60
# ⚠️ LOOKAHEAD GUARD. `asos1min.qc_series` uses CENTRED windows -- an 11-minute
# median, then a 5-minute mean over the despiked series -- so the QC'd value at
# minute t depends on raw minutes as late as t + 5 + 2 = t+7. Slicing "strictly
# before asof" is therefore NOT leak-free: the last few QC'd values in the slice
# were computed with minutes the desk could not have seen.
#
# The first version of this script had exactly that bug and reported dHIT up to
# +8.5%. Seven minutes at the boundary of a 5-7 hour running extreme should
# barely matter, but "should barely matter" is not a measurement, and this was
# the first large positive effect in a project with four consecutive nulls.
#
# 8 > 7, so the guard is strictly safe.
LOOKAHEAD_GUARD_MIN = 8
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]
rng = np.random.default_rng(20260812)


def qc_bound(cfg, grid, vals, d: date, hour: int, extremum: str):
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


print(__doc__.split("Usage:")[0])
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

        # ── arm B bound for EVERY frame row, at that row's own hour ──────────
        bounds = []
        for td, hh in zip(fr["target_date"], fr["hour"]):
            bounds.append(qc_bound(cfg, grid, vals1m, date.fromisoformat(td),
                                   int(hh), spec.extremum))
        fr["b1"] = bounds
        before = len(fr)
        fr = fr[fr["b1"].notna()].copy()          # BOTH arms lose these rows
        print(f"\n### {icao} {mkey.upper()} h{hour}: {len(fr)}/{before} frame rows "
              f"have 1-minute coverage ({len(fr)/before:.0%})", flush=True)

        frB = fr.copy()
        if spec.extremum == "max":
            frB["run_lo"] = fr["b1"] - METAR_TOL - SAFETY_F   # the clamping side
            frB["run_hi"] = fr["b1"] + METAR_TOL
        else:
            frB["run_lo"] = fr["b1"] - METAR_TOL
            frB["run_hi"] = fr["b1"] + METAR_TOL + SAFETY_F   # the clamping side
        frB["run_width"] = frB["run_hi"] - frB["run_lo"]

        teA = fr[fr["hour"] == float(hour)].sort_values("target_date")
        teA = teA[teA["target_date"] >= RECENT]
        teB = frB.set_index(fr.index).loc[teA.index]
        td = teA["target_date"].to_numpy()
        if len(teA) < 60:
            print("   too few test days — skip"); continue

        per_day = {}
        for b0 in range(0, len(td), REFIT):
            cut = td[b0]
            trA = fr[(fr["target_date"] < cut)]
            trB = frB[(frB["target_date"] < cut)]
            if len(trA) < MIN_TRAIN:
                continue
            blkA, blkB = teA.iloc[b0:b0 + REFIT], teB.iloc[b0:b0 + REFIT]
            if blkA.empty:
                continue
            hits = {}
            for tag, tr, blk in (("A", trA, blkA), ("B", trB, blkB)):
                m = IntradayModel(extremum=spec.extremum, residual=resid,
                                  extra_features=extra).fit(tr)
                hits[tag] = [expected_bucket_hit(bucket_probs(q), int(r["y"]))
                             for (_, r), q in zip(blk.iterrows(),
                                                  m.predict_quantiles(blk))]
            for (_, r), a, bb in zip(blkA.iterrows(), hits["A"], hits["B"]):
                per_day[r["target_date"]] = (a, bb)

        if len(per_day) < 40:
            print(f"   only {len(per_day)} scored days — skip"); continue
        A = np.array([v[0] for v in per_day.values()])
        B = np.array([v[1] for v in per_day.values()])
        d = B - A
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                         for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # Bonferroni x4 markets: alpha 0.05 -> 0.0125 two-sided -> 98.75% CI.
        # Computed, not eyeballed off the 95% one.
        blo, bhi = np.percentile(boot, [0.625, 99.375])
        sig = "SIGNIFICANT" if (lo > 0) == (hi > 0) else "ns"
        bsig = "SURVIVES Bonferroni" if (blo > 0) == (bhi > 0) else "fails Bonferroni"
        sf = gating.EXTREMUM_STILL_FUTURE[icao][mkey][hour]
        np.savez(f"/tmp/wf1min_{icao}_{mkey}.npz", A=A, B=B,
                 days=np.array(list(per_day.keys())))
        print(f"   n={len(d)} matched test days   "
              f"P(extremum still ahead at commit) = {sf:.3f}")
        print(f"   A (METAR bound) hit {A.mean():.1%}   "
              f"B (1-min bound) hit {B.mean():.1%}")
        print(f"   dHIT {d.mean():+.1%}   95% CI [{lo:+.1%},{hi:+.1%}]   {sig}")
        print(f"                       98.75% CI [{blo:+.1%},{bhi:+.1%}]   {bsig}")
        print(f"   days improved {int((d > 0).sum())}  worsened {int((d < 0).sum())}"
              f"  unchanged {int((d == 0).sum())}", flush=True)

print("\nBonferroni x4 markets: read a single-market CI at the 98.75% level")
print("before calling anything here a result.")
