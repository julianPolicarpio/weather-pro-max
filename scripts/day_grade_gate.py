"""PRE-REGISTERED GATE for a day-quality grade. Measure before building.

THE ASK: grade each day so we know which nowcasts are trustworthy — stable, no
external factors, no blindspots.

⚠️ THIS IS THE FIFTH IDEA OF THIS SHAPE. Four have died here, all the same way:

    aggressive sizing arm    wrong on the live ledger
    guidance-spread veto     BACKWARDS — vetoed days hit BETTER
    confidence gate          inverts P&L
    near-boundary warning    vanishes once you condition on `stated`

The last one is the trap. A factor can separate hit rate beautifully and still
be worthless, because the MODEL ALREADY KNOWS. `stated` alone separates 42.6%
(<0.50) from 84.1% (>=0.70) at KMIA low h7. Any grade that merely rediscovers
that has added nothing and costs a gate.

THE BAR, declared before running:

    A factor earns a place in the grade ONLY if it separates hit rate WITHIN a
    stated-probability band. Raw separation does not count. If every factor
    vanishes given `stated`, the honest deliverable is "the model's own number
    IS the grade" — and that is a real answer, not a failure.

CANDIDATE FACTORS, all computable at commit time and all deterministic facts
about the day's information state rather than fitted predictors:

    ratchet_pinned   the visible interval already sits inside ONE ladder bucket
                     -> the answer is arithmetic, not forecast
    still_future     gating.EXTREMUM_STILL_FUTURE for this market/hour
    missing_groups   six-hour remark groups we were entitled to and never got
    guidance_spread  disagreement across the day-ahead models
    stability        how far p50 moved over the 3 hours before commit
    heat_alert       an active NWS advisory

Usage: PYTHONPATH=src .venv/bin/python scripts/day_grade_gate.py
"""
import collections
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from highcast import gating, latentmax
from highcast.buckets import bucket_probs, expected_bucket_hit, ladder_lo
from highcast.config import load_config
from highcast.db import connect
from highcast.features import build_features, uses_bias_correction
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.models.baseline import BaselineBlend
from highcast.models.intraday import (IntradayModel, build_checkpoint_frame,
                                      features_for)

REFIT, MIN_TRAIN, RECENT = 30, 400, "2025-01-01"
STATED_BANDS = [(0.0, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]
_SHIPPED_BASE = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf", "cur_dwpf",
                 "cur_onshore", "cur_sky", "base_p50", "base_spread", "doy_sin",
                 "doy_cos", "year_index", "heat_alert"]


def collect(cfg_path, mkey):
    cfg = load_config(cfg_path)
    icao = cfg.station.icao
    conn = connect(cfg.db_path, migrate=False)
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
    te = fr[(fr["hour"] == float(hour)) & fr["y"].notna()].sort_values("target_date")
    te = te[te["target_date"] >= RECENT]
    td = te["target_date"].to_numpy()
    print(f"\n### {icao} {mkey.upper()} h{hour}  test days={len(te)}", flush=True)

    out = []
    for b0 in range(0, len(td), REFIT):
        cut = td[b0]
        tr = fr[(fr["target_date"] < cut) & fr["y"].notna()]
        if len(tr) < MIN_TRAIN:
            continue
        blk = te.iloc[b0:b0 + REFIT]
        if blk.empty:
            continue
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(tr)
        for (_, row), q in zip(blk.iterrows(), m.predict_quantiles(blk)):
            pmf = bucket_probs(q)
            y = int(row["y"])
            hit = expected_bucket_hit(pmf, y)
            stated = max(
                sum(p for t, p in pmf.items()
                    if ladder_lo(int(t), par) == lo)
                for par in (0, 1)
                for lo in {ladder_lo(int(t), par) for t in pmf})
            d = date.fromisoformat(row["target_date"])
            asof = datetime(d.year, d.month, d.day, hour,
                            tzinfo=ZoneInfo(cfg.station.tz)).astimezone(
                                ZoneInfo("UTC")).isoformat()
            # ── the candidate factors ───────────────────────────────────────
            lo_b, hi_b = row["run_lo"], row["run_hi"]
            # pinned: the visible interval fits inside a single 2F bucket
            pinned = bool(int(np.floor(lo_b / 2)) == int(np.floor(hi_b / 2)))
            # ⚠️ COUNT MISSING GROUPS, DO NOT SUBTRACT MINUTES. The first
            # version used `day_minutes - covered_minutes`, which goes NEGATIVE
            # because a six-hour window overruns the as-of horizon (1080 vs 720
            # is normal). Every coverage row it produced was void, including a
            # spurious "+17.4% separation" that looked like the best result in
            # the run.
            cov = latentmax.six_hour_coverage(conn, cfg, d, spec.extremum,
                                              asof_utc=asof) or {}
            gap_min = float(len(cov.get("missing_groups") or []))
            out.append({
                "hit": hit, "stated": float(stated),
                "still_future": gating.EXTREMUM_STILL_FUTURE[icao][mkey][hour],
                "pinned": pinned,
                "coverage_gap": float(gap_min),
                "spread": float(row.get("base_spread", np.nan)),
                "heat_alert": bool(row.get("heat_alert", 0)),
            })
    return out


def report(rows, name, factor, split):
    """Hit rate by factor, RAW and then WITHIN each stated band."""
    print(f"\n  --- {factor} ---")
    a = [r for r in rows if split(r)]
    b = [r for r in rows if not split(r)]
    if min(len(a), len(b)) < 20:
        print(f"    too few either side ({len(a)}/{len(b)}) — skip")
        return
    print(f"    RAW      yes n={len(a):<4} hit {np.mean([r['hit'] for r in a]):.1%}"
          f"   no n={len(b):<4} hit {np.mean([r['hit'] for r in b]):.1%}"
          f"   delta {np.mean([r['hit'] for r in a])-np.mean([r['hit'] for r in b]):+.1%}")
    print(f"    {'stated band':<14}{'yes n':>7}{'yes hit':>9}{'no n':>7}"
          f"{'no hit':>9}{'delta':>9}")
    survives = False
    for lo, hi in STATED_BANDS:
        aa = [r for r in a if lo <= r["stated"] < hi]
        bb = [r for r in b if lo <= r["stated"] < hi]
        if min(len(aa), len(bb)) < 15:
            continue
        ha, hb = np.mean([r["hit"] for r in aa]), np.mean([r["hit"] for r in bb])
        flag = ""
        if abs(ha - hb) >= 0.05:
            flag = "  <-- separates"
            survives = True
        print(f"    {lo:.2f}-{hi:<9.2f}{len(aa):>7}{ha:>9.1%}{len(bb):>7}"
              f"{hb:>9.1%}{ha-hb:>+9.1%}{flag}")
    print(f"    -> {'SURVIVES conditioning on stated' if survives else 'VANISHES given stated — adds nothing'}")


print(__doc__.split("Usage:")[0])
allrows = []
for cfgp in ("config/klax.yaml", "config/kmia.yaml"):
    for mk in ("high", "low"):
        rows = collect(cfgp, mk)
        allrows += rows
        if len(rows) < 100:
            continue
        report(rows, f"{cfgp} {mk}", "ratchet_pinned", lambda r: r["pinned"])
        med = np.nanmedian([r["spread"] for r in rows])
        report(rows, "", f"guidance spread > median ({med:.1f}F)",
               lambda r: r["spread"] > med)
        report(rows, "", "a six-hour group went missing",
               lambda r: r["coverage_gap"] > 0)
        report(rows, "", "heat_alert active", lambda r: r["heat_alert"])

print("\n" + "=" * 74)
print("A factor that only separates in the RAW row is the near-boundary warning")
print("all over again: the model already knows, and the grade would show the")
print("same fact twice under a new name.")
