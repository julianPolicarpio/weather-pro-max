"""PRE-REGISTERED SAFETY GATE, before any walk-forward of 1-minute run_* features.

`IntradayModel._clamp` treats the ratchet bound as PHYSICAL TRUTH and forces
every quantile past it. Its docstring earns that right with a measurement:

    "Verified on 2025+ holdouts that truth NEVER crosses it (P(violation) =
     0.0% at every hour, both metrics), so clamping can only remove mass that
     is genuinely impossible."

A bound from `obs_1min` does NOT automatically inherit that. The QC'd DAILY
extreme already lands above the settled CLI value on a minority of days -- KMIA
high still shows +27F and +28F after full QC, and 0.2-0.8% of days sit at +1 to
+4F. Every one of those, used as a clamp, would force the model above the truth
and guarantee a loss on that day.

THE GATE, declared before running:

    The 1-minute bound must violate at a rate comparable to the shipped METAR
    bound (0.0%). Anything above ~0.5% at a commit hour means the clamp is
    actively harmful and the feature cannot ship as a bound, regardless of what
    a hit-rate walk-forward says afterwards.

Violation is defined against the SETTLEMENT grid, not the raw float: `y` is an
integer CLI value, so the true extreme lies in [y-0.5, y+0.5). A max bound is
impossible only once it exceeds y+0.5.

Both the raw and a margin-subtracted bound are scored, because if a fixed haircut
rescues it that is a cheap fix worth knowing about.

Usage: PYTHONPATH=src .venv/bin/python scripts/asos1min_ratchet_gate.py
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from highcast import asos1min, latentmax
from highcast.config import load_config
from highcast.db import connect
from highcast.gating import MIN_DECISION_HOUR
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds

MARGINS = (0.0, 0.5, 1.0)
MIN_WINDOW_SAMPLES = 60      # minutes of real data before asof, else no bound


def qc_running_bound(cfg, grid, vals, d: date, hour: int, extremum: str):
    """The QC'd 1-minute running extreme from climate-day start to `hour` local.

    Leak-free by construction: the slice ends AT asof, so no later minute can
    reach it.
    """
    start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    tz = ZoneInfo(cfg.station.tz)
    asof = datetime(d.year, d.month, d.day, hour, tzinfo=tz).astimezone(
        ZoneInfo("UTC"))
    a = int(start.timestamp()) // 60
    b = int(asof.timestamp()) // 60
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
    grid, vals = asos1min.qc_series(valid, temps)
    print(f"\n{'='*76}\n{icao}\n{'='*76}")

    for mkey in ("high", "low"):
        spec = get_metric(cfg, mkey)
        hour = MIN_DECISION_HOUR[icao][mkey]
        col = spec.target_col
        rows = list(conn.execute(
            f"""SELECT date, {col} y FROM cli_reports WHERE station=?
                AND {col} IS NOT NULL AND date>='2025-01-01' ORDER BY date""",
            (icao,)))
        stats = {m: [0, 0, 0.0] for m in MARGINS}     # margin -> [viol, n, worst]
        shipped = [0, 0, 0.0]
        for r in rows:
            d = date.fromisoformat(r["date"])
            y = float(r["y"])
            # shipped METAR bound, exactly as build_checkpoint_frame builds it
            asof = datetime(d.year, d.month, d.day, hour,
                            tzinfo=ZoneInfo(cfg.station.tz)).astimezone(
                                ZoneInfo("UTC")).isoformat()
            vis = latentmax.visible_running_extremum(conn, cfg, d, spec.extremum,
                                                     asof_utc=asof)
            b1 = qc_running_bound(cfg, grid, vals, d, hour, spec.extremum)
            if vis is None or b1 is None:
                continue
            sb = vis[0] if spec.extremum == "max" else vis[1]
            over = (sb - (y + 0.5)) if spec.extremum == "max" else ((y - 0.5) - sb)
            shipped[1] += 1
            if over > 0:
                shipped[0] += 1
                shipped[2] = max(shipped[2], over)
            for m in MARGINS:
                bb = b1 - m if spec.extremum == "max" else b1 + m
                over = ((bb - (y + 0.5)) if spec.extremum == "max"
                        else ((y - 0.5) - bb))
                stats[m][1] += 1
                if over > 0:
                    stats[m][0] += 1
                    stats[m][2] = max(stats[m][2], over)

        n = shipped[1]
        if not n:
            print(f"\n  {mkey.upper()} h{hour}: no paired days\n"); continue
        print(f"\n  {mkey.upper()} h{hour}   n={n} days with BOTH bounds "
              f"(2025-01-01 onward)")
        print(f"    {'bound':<28}{'violations':>12}{'rate':>9}{'worst':>9}")
        print(f"    {'shipped (METAR interval)':<28}{shipped[0]:>12}"
              f"{shipped[0]/n:>9.2%}{shipped[2]:>9.2f}")
        for m in MARGINS:
            v, nn, worst = stats[m]
            tag = f"1-min QC, margin {m:.1f}F"
            verdict = "  OK" if v / nn <= 0.005 else "  <<< FAILS THE GATE"
            print(f"    {tag:<28}{v:>12}{v/nn:>9.2%}{worst:>9.2f}{verdict}")

print("\nA bound that violates is not a conservative feature — `_clamp` forces")
print("every quantile past it, so each violation is a guaranteed loss that day.")
