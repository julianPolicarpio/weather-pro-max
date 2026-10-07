"""Hourly LA temperature: how well can we predict it, and by what lead time?

For the KXTEMPLAXH "Hourly Directional Los Angeles Temperature" market. Strikes
sit 1 F apart (…73.99, 74.99…) and the market is directional, so the metric that
matters is not MAE but P(correct SIDE of the strike).

⚠️ GROUND-TRUTH CAVEAT. KXTEMPLAXH resolves against THE WEATHER COMPANY for
coordinates KLAX, explicitly NOT the NWS. This script trains and scores against
our METAR/ASOS observations, which is a DIFFERENT SOURCE. Kalshi's own rules warn
of "rounding and conversion differences". At 1 F strike granularity that
difference is the entire bet, so treat every number here as an upper bound on
real performance until the METAR-vs-TWC gap has been measured directly.

Arms, by lead time:
  PERSIST    temperature now == temperature at the target hour (the naive rule
             any model must beat; at short lead it is very strong)
  CLIMO      persistence plus the mean diurnal change for (hour, lead, season)
  MODEL      LightGBM on lags, diurnal position, dewpoint/wind/sky, and lead

Usage: .venv/bin/python scripts/hourly_eval.py [max_lead]
"""
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast.config import load_config
from highcast.db import connect
from highcast.models import make_lgbm as _lgbm

MAX_LEAD = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SPLIT = "2025-01-01"
cfg = load_config("config/klax.yaml")
conn = connect(cfg.db_path)
tz = ZoneInfo(cfg.station.tz)

rows = conn.execute(
    """SELECT valid_utc, tmpf_precise, tmpf, dwpf, sknt, drct, skyc FROM obs
       WHERE station=? AND source='metar' AND tmpf_precise IS NOT NULL
       ORDER BY valid_utc""", (cfg.station.icao,)).fetchall()
print(f"raw METAR obs: {len(rows)}")

# Snap each observation to its nearest wall-clock hour. ASOS files at :53, so the
# 23:53Z report is the observation for hour 00Z -- the same synoptic convention
# `latentmax` already relies on. Without this every ob sits 7 minutes before the
# hour the market settles on.
SKY = {"CLR": 0.0, "SKC": 0.0, "FEW": 1.0, "SCT": 2.0, "BKN": 3.0, "OVC": 4.0}
recs = {}
for r in rows:
    t = datetime.fromisoformat(r["valid_utc"])
    snapped = (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    recs[snapped] = {
        "t": float(r["tmpf_precise"]),
        "dwpf": float(r["dwpf"]) if r["dwpf"] is not None else np.nan,
        "sknt": float(r["sknt"]) if r["sknt"] is not None else np.nan,
        "drct": float(r["drct"]) if r["drct"] is not None else np.nan,
        "sky": SKY.get(r["skyc"], np.nan),
    }
print(f"hour-snapped: {len(recs)} distinct hours")

hours = sorted(recs)
idx = {h: i for i, h in enumerate(hours)}
data = []
for h in hours:
    # Require a contiguous 3-hour history and the target, all on the hour.
    prev = [h - timedelta(hours=k) for k in (1, 2, 3)]
    if any(p not in recs for p in prev):
        continue
    base = recs[h]
    lag = [recs[p]["t"] for p in prev]
    loc = h.astimezone(tz)
    for lead in range(1, MAX_LEAD + 1):
        tgt = h + timedelta(hours=lead)
        if tgt not in recs:
            continue
        tl = tgt.astimezone(tz)
        doy = tl.timetuple().tm_yday
        data.append({
            "asof": h.isoformat(), "target_date": tl.date().isoformat(),
            "lead": float(lead),
            "cur": base["t"], "d1": base["t"] - lag[0], "d2": base["t"] - lag[1],
            "d3": base["t"] - lag[2],
            "dwpf": base["dwpf"], "dtd": base["t"] - base["dwpf"],
            "sknt": base["sknt"], "sky": base["sky"],
            "onshore": (np.cos(np.radians(base["drct"] - 250.0))
                        if not np.isnan(base["drct"]) else np.nan),
            "tgt_hour": float(tl.hour), "asof_hour": float(loc.hour),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "y": recs[tgt]["t"],
        })

df = pd.DataFrame(data)
print(f"training rows: {len(df)}  ({df['target_date'].min()} .. {df['target_date'].max()})\n")

FEATS = ["lead", "cur", "d1", "d2", "d3", "dwpf", "dtd", "sknt", "sky",
         "onshore", "tgt_hour", "asof_hour", "doy_sin", "doy_cos"]
tr = df[df["target_date"] < SPLIT]
te = df[df["target_date"] >= SPLIT]

# CLIMO: mean change from `cur` to target, keyed on (target hour, lead, month).
tr = tr.copy()
tr["mon"] = pd.to_datetime(tr["target_date"]).dt.month
climo = tr.assign(delta=tr["y"] - tr["cur"]).groupby(
    ["tgt_hour", "lead", "mon"])["delta"].mean()

model = _lgbm(0.5).fit(tr[FEATS], tr["y"].to_numpy(dtype=float))

te = te.copy()
te["mon"] = pd.to_datetime(te["target_date"]).dt.month
te["p_persist"] = te["cur"]
te["p_climo"] = te["cur"] + [
    climo.get((r.tgt_hour, r.lead, r.mon), 0.0) for r in te.itertuples()]
te["p_model"] = model.predict(te[FEATS])

print(f"{'lead':>5}{'n':>7} | " + "".join(
    f"{a+' MAE':>12}{a+' side':>12}" for a in ("persist", "climo", "model")))
for lead in range(1, MAX_LEAD + 1):
    s = te[te["lead"] == lead]
    if s.empty:
        continue
    out = [f"{lead:>5}{len(s):>7} | "]
    y = s["y"].to_numpy()
    cur = s["cur"].to_numpy()
    # STRIKE LADDER, chosen WITHOUT looking at the answer. An earlier version set
    # the strike to floor(y)-0.01, i.e. derived from the truth -- which makes
    # `y > strike` always true, so any over-forecast scored as correct and
    # persistence beat the model on "accuracy" while losing badly on MAE. The
    # ladder must be anchored on information available at trade time; Kalshi
    # lists ~10 strikes bracketing the current temperature, so mirror that.
    ladder = np.arange(-3, 4)
    for col in ("p_persist", "p_climo", "p_model"):
        pred = s[col].to_numpy()
        agree = []
        for k in ladder:
            strike = np.floor(cur) + k + 0.99
            agree.append((pred > strike) == (y > strike))
        side_ok = float(np.mean(np.concatenate(agree)))
        out.append(f"{np.abs(pred - y).mean():>12.3f}{side_ok:>11.1%}")
    print("".join(out))

print("\nside = P(model on the correct side of a near-the-money 1F strike).")
print("Scored against METAR, NOT The Weather Company — see the caveat above.")
