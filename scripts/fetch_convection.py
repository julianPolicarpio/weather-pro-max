"""Pull day-ahead FORECAST convection fields for KMIA into a new table.

Leak-free by construction: `*_previous_day1` is what the model predicted ONE DAY
AHEAD, so it is information genuinely available at the decision hour -- unlike the
archive's observed precipitation, which is hindsight and would leak.

WHY THIS MATTERS FOR MIAMI. The high is convection-limited: on 2026-07-28 the
temperature reached 89F by 10:53 ET, a thunderstorm outflow dropped it to 80F at
14:44, and the day topped out at 91. The model predicted 92-93 at noon because it
has NO precipitation input at all -- 14 features, of which `cur_sky` (a cloud-cover
code) is the only storm-adjacent one. This is the LAX marine layer all over again,
except LAX has nbs_sol/sky/cig features for its version and Miami has nothing.
"""
import sys
from datetime import datetime, timezone
import httpx
from highcast import db
from highcast.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/kmia.yaml")
conn = db.connect(cfg.db_path)
conn.executescript("""
CREATE TABLE IF NOT EXISTS forecast_conv (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, model TEXT NOT NULL,
  pop REAL, qpf REAL, cape REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc, model));
""")
MODELS = ["ncep_nbm_conus", "gfs_seamless", "gfs_hrrr"]
now = datetime.now(timezone.utc).isoformat()
total = 0
for m in MODELS:
    r = httpx.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
        "latitude": cfg.station.lat, "longitude": cfg.station.lon, "models": m,
        "hourly": "precipitation_probability_previous_day1,"
                  "precipitation_previous_day1,cape_previous_day1",
        "past_days": 730, "forecast_days": 1, "timezone": "UTC"}, timeout=180)
    h = r.json().get("hourly", {})
    if not h:
        print(f"{m}: no data ({r.status_code})"); continue
    def col(name):
        for k in h:
            if name in k:
                return h[k]
        return [None] * len(h["time"])
    pops, qpfs, capes = col("precipitation_probability"), None, col("cape")
    # `precipitation` also substring-matches precipitation_probability, so pick the
    # key that is NOT the probability one rather than trusting substring order.
    for k in h:
        if "precipitation" in k and "probability" not in k:
            qpfs = h[k]
    rows = [{"station": cfg.station.icao, "valid_utc": t, "model": m,
             "pop": p, "qpf": q, "cape": c, "fetched_at": now}
            for t, p, q, c in zip(h["time"], pops, qpfs or [None]*len(h["time"]), capes)
            if p is not None or q is not None or c is not None]
    n = db.upsert(conn, "forecast_conv", rows)
    total += len(rows)
    print(f"{m}: {len(rows)} hourly rows  ({h['time'][0][:10]} .. {h['time'][-1][:10]})")
print("total:", total)
