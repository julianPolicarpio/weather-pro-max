"""DOES A CONVECTION SIGNAL IMPROVE KMIA'S HIGH? Two arms, same rows.

Miami's high is convection-limited and the model has NO precipitation input --
14 features of which `cur_sky` is the only storm-adjacent one. 2026-07-28: 89F by
10:53 ET, model said 92-93 at noon, a 14:44 outflow cut it 9F, true high 91F.

LEAK-FREE. `forecast_conv` holds Open-Meteo `*_previous_day1` fields -- what each
CONUS model predicted ONE DAY AHEAD -- so they are available at the decision hour.
The archive endpoint's OBSERVED precipitation would encode whether the storm
actually happened; that is hindsight and must never be used here.

TWO ARMS ON IDENTICAL ROWS, because convection history starts 2024-07-29 while
production trains on 2020+. Data volume demonstrably matters (it is why k=+/-1
beat k=0), so comparing against the full-history model would confound "features
help" with "less data hurts".

    ARM A   the shipped 14 features, date range restricted to CONV_START+
    ARM B   ARM A + conv_* features, SAME range

Scored on PREDICTIVE HOURS ONLY -- an arm that wins after the extremum has formed
has won nothing.

⚠️ Miami rains almost daily (all 9 days sampled 07-20..28 had 10-13 rain hours),
so "will it rain" barely discriminates. The signal must be TIMING and INTENSITY
relative to peak heating -- hence pop_hour alongside the maxima.

Usage: .venv/bin/python scripts/convection_eval.py [config] [metric]
"""
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast import gating
from highcast.buckets import bucket_probs, expected_bucket_hit
from highcast.config import load_config
from highcast.db import connect
from highcast.metric import get_metric
from highcast.metrics import crps_pmf
from highcast.models.intraday import IntradayModel, build_checkpoint_frame

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml"
METRIC = sys.argv[2] if len(sys.argv) > 2 else "high"
CONV_START = "2024-10-09"      # ncep_nbm_conus floor; the other two reach 2024-07-29
SPLIT = "2026-01-01"           # ~15 months train / ~7 months test inside the range
PREDICTIVE_MIN = 0.60
# Local hours the HIGH actually forms, from KMIA's still-future curve (h13 73%,
# h14 50%, h15 22%, h16 7%). The window the storm has to hit to cap the day.
PEAK = (11, 17)

cfg = load_config(CONFIG)
conn = connect(cfg.db_path)
tz = ZoneInfo(cfg.station.tz)
spec = get_metric(cfg, METRIC)
resid = bool(getattr(cfg.model, f"intraday_residual_{METRIC}", False))
print(f"{cfg.station.icao} {METRIC.upper()}  conv from {CONV_START}  split {SPLIT}",
      flush=True)

# --- build per-target-date convection features from the hourly forecast table ---
rows = conn.execute(
    """SELECT valid_utc, model, pop, qpf, cape FROM forecast_conv
       WHERE station=? ORDER BY valid_utc""", (cfg.station.icao,)).fetchall()
per_day: dict[str, dict] = {}
for r in rows:
    t = datetime.fromisoformat(r["valid_utc"]).replace(tzinfo=ZoneInfo("UTC"))
    loc = t.astimezone(tz)
    if not (PEAK[0] <= loc.hour <= PEAK[1]):
        continue
    d = loc.date().isoformat()
    b = per_day.setdefault(d, {"pop": [], "qpf": [], "cape": [], "pop_h": []})
    if r["pop"] is not None:
        b["pop"].append(r["pop"]); b["pop_h"].append((r["pop"], loc.hour))
    if r["qpf"] is not None:
        b["qpf"].append(r["qpf"])
    if r["cape"] is not None:
        b["cape"].append(r["cape"])
conv = pd.DataFrame([{
    "target_date": d,
    # max across models AND hours: any model firing a storm in the window is the
    # risk, and taking the max is the conservative read of "could be capped".
    "conv_pop_max": max(v["pop"]) if v["pop"] else np.nan,
    "conv_cape_max": max(v["cape"]) if v["cape"] else np.nan,
    "conv_qpf": float(np.mean(v["qpf"])) if v["qpf"] else np.nan,
    # hour of peak PoP -- an EARLY storm caps the day, a late one does not, and a
    # daily aggregate throws that distinction away entirely.
    "conv_pop_hour": (max(v["pop_h"])[1] if v["pop_h"] else np.nan),
} for d, v in per_day.items()])
print(f"conv feature rows: {len(conv)}  "
      f"{conv['target_date'].min()} .. {conv['target_date'].max()}", flush=True)
print(conv[["conv_pop_max", "conv_cape_max", "conv_qpf"]].describe().loc[
    ["mean", "std", "min", "max"]].to_string(), flush=True)

dates = [r["date"] for r in conn.execute(
    "SELECT date FROM cli_reports WHERE station=? AND date>=? ORDER BY date",
    (cfg.station.icao, CONV_START))]
fut = gating.EXTREMUM_STILL_FUTURE.get(cfg.station.icao, {}).get(METRIC, {})
pred_hours = [h for h in range(24) if fut.get(h, 0.0) >= PREDICTIVE_MIN]
print(f"predictive hours: {pred_hours}", flush=True)

k = cfg.model.intraday_bandwidth_for(METRIC)
print(f"{'h':>3}{'n':>6} | {'A mae':>7}{'A hit':>7} | {'B mae':>7}{'B hit':>7} | "
      f"{'dMAE':>7}{'dHIT':>7}", flush=True)
res = []
for h in pred_hours:
    tr_hours = tuple(sorted({(h + d) % 24 for d in range(-k, k + 1)}))
    fr = build_checkpoint_frame(conn, cfg, dates, hours=tr_hours, metric=spec)
    fr = fr[fr["y"].notna()].merge(conv, on="target_date", how="left")
    tr = fr[fr["target_date"] < SPLIT]
    te = fr[(fr["target_date"] >= SPLIT) & (fr["hour"] == float(h))]
    if len(te) < 40 or len(tr) < 200:
        print(f"{h:>3}{len(te):>6} | skipped (train {len(tr)})", flush=True); continue
    y = te["y"].to_numpy(dtype=float)
    out = {}
    CONV = ["conv_pop_max", "conv_cape_max", "conv_qpf", "conv_pop_hour"]
    for arm, extra in (("A", []), ("B", CONV)):
        t2, e2 = tr, te
        m = IntradayModel(extremum=spec.extremum, residual=resid,
                          extra_features=extra).fit(t2)
        pmfs = [bucket_probs(q) for q in m.predict_quantiles(e2)]
        pt = np.array([q[0.5] for q in m.predict_quantiles(e2)])
        out[arm] = (float(np.abs(pt - y).mean()),
                    float(np.mean([crps_pmf(p, int(v)) for p, v in zip(pmfs, y)])),
                    float(np.mean([expected_bucket_hit(p, int(v))
                                   for p, v in zip(pmfs, y)])))
    a, b = out["A"], out["B"]
    res.append((h, len(te), a, b))
    print(f"{h:>3}{len(te):>6} | {a[0]:>7.3f}{a[2]:>6.1%} | {b[0]:>7.3f}{b[2]:>6.1%} | "
          f"{b[0]-a[0]:>+7.3f}{(b[2]-a[2])*100:>+6.1f}", flush=True)
if res:
    print(f"\nPREDICTIVE-HOUR MEAN  dMAE {np.mean([r[3][0]-r[2][0] for r in res]):+.3f}"
          f"  dCRPS {np.mean([r[3][1]-r[2][1] for r in res]):+.3f}"
          f"  dHIT {np.mean([r[3][2]-r[2][2] for r in res])*100:+.1f} pts")
    print("negative dMAE / positive dHIT = convection features HELP")
