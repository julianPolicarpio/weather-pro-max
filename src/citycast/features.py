# src/citycast/features.py
"""Feature rows for (station, metric, climate day, as-of instant) — spec §3.3.

Everything here is strictly as-of t: MOS runs need runtime_utc <= t - 1 h, climatology uses
only years before the target's year, observations go through latentmax with asof_utc=t.
"""
import bisect
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from citycast.asof import lead_hours, same_day
from citycast.stations import app_config
from highcast.latentmax import visible_running_extremum
from highcast.timeutil import climate_day_bounds

MOS_MODELS = ("MEX", "GFS", "NBS", "NBE")
# Publication lag per model (spec §3.3, revised 2026-10-08): conservative defaults until Task 0
# measures first-seen times on IEM; a too-short lag leaks unpublished forecasts into training.
PUB_LAG_H = {"GFS": 5.0, "MEX": 5.0, "NBS": 3.0, "NBE": 3.0}
FULL_DAY_MIN_PTS = 6    # an NBS/NBE run counts toward the anchor only if it covers >= 18 h of the day
CLIM_WINDOW_D = 7
CLIM_MIN_N = 10
NAN = float("nan")

FEATURE_COLS = [*(f"f_{m}" for m in MOS_MODELS), *(f"age_{m}" for m in MOS_MODELS),
                *(f"npts_{m}" for m in MOS_MODELS),
                "anchor", "spread_mos", "clim_mean", "clim_sd", "prev_obs", "run_ext", "run_gap",
                "last_tmpf", "last_dwpf", "doy_sin", "doy_cos", "lead_h", "local_hour",
                "lat", "lon", "elevation_m"]


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class FeatureContext:
    def __init__(self, conn, stations: dict, db_path, mos_since: date | None = None):
        self.conn, self.stations = conn, stations
        self.cfg = {i: app_config(s, db_path) for i, s in stations.items()}
        self._mos: dict[tuple, tuple[list[str], list[float]]] = {}
        sql = ("SELECT station, model, metric, target_date, runtime_utc, value_f, n_points FROM mos_runs "
               + ("WHERE target_date >= ? " if mos_since else "") + "ORDER BY runtime_utc")
        for r in conn.execute(sql, (mos_since.isoformat(),) if mos_since else ()):
            rts, vals = self._mos.setdefault((r[0], r[1], r[2], r[3]), ([], []))
            rts.append(r[4])
            vals.append((float(r[5]), int(r[6])))
        self._cli: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._truth: dict[tuple[str, str, str], float] = {}
        for icao in stations:
            for metric, col in (("high", "high_f"), ("low", "low_f")):
                rows = conn.execute(f"SELECT date, {col} FROM cli_daily WHERE station=? AND final=1 "
                                    f"AND {col} IS NOT NULL", (icao,)).fetchall()
                ds = [date.fromisoformat(d) for d, _ in rows]
                self._cli[(icao, metric)] = (np.array([d.year for d in ds]),
                                             np.array([d.timetuple().tm_yday for d in ds]),
                                             np.array([float(v) for _, v in rows]))
                for d, v in rows:
                    self._truth[(icao, metric, d)] = float(v)
        self._clim_cache: dict[tuple, tuple[float, float]] = {}

    def truth(self, icao: str, metric: str, day: date) -> float:
        return self._truth.get((icao, metric, day.isoformat()), NAN)

    def mos(self, icao, model, metric, day: date, t: datetime) -> tuple[float, float, float]:
        """(value, age_h, n_points) of the latest run published by t; NaNs when none."""
        found = self._mos.get((icao, model, metric, day.isoformat()))
        if not found:
            return NAN, NAN, NAN
        rts, vals = found
        i = bisect.bisect_right(rts, (t - timedelta(hours=PUB_LAG_H[model])).isoformat()) - 1
        if i < 0:
            return NAN, NAN, NAN
        value, npts = vals[i]
        return value, (t - datetime.fromisoformat(rts[i])).total_seconds() / 3600, float(npts)

    def clim(self, icao, metric, day: date) -> tuple[float, float]:
        key = (icao, metric, day)
        if key not in self._clim_cache:
            years, doys, vals = self._cli[(icao, metric)]
            doy = day.timetuple().tm_yday
            dist = np.abs(doys - doy)
            dist = np.minimum(dist, 366 - dist)
            v = vals[(years < day.year) & (dist <= CLIM_WINDOW_D)]
            self._clim_cache[key] = ((float(v.mean()), float(v.std(ddof=1)))
                                     if len(v) >= CLIM_MIN_N else (NAN, NAN))
        return self._clim_cache[key]

    def bound(self, icao, metric, day: date, t: datetime) -> float:
        extremum = "max" if metric == "high" else "min"
        iv = visible_running_extremum(self.conn, self.cfg[icao], day, extremum, asof_utc=t.isoformat())
        if iv is None:
            return NAN
        return float(iv[0] if metric == "high" else iv[1])

    def last_obs(self, icao, day: date, t: datetime) -> tuple[float, float]:
        start = climate_day_bounds(day, self.stations[icao].std_utc_offset)[0]
        r = self.conn.execute("SELECT tmpf, dwpf FROM obs WHERE station=? AND valid_utc<=? "
                              "AND valid_utc>=? AND tmpf IS NOT NULL ORDER BY valid_utc DESC LIMIT 1",
                              (icao, t.isoformat(), start.isoformat())).fetchone()
        if r is None:
            return NAN, NAN
        return float(r[0]), (NAN if r[1] is None else float(r[1]))

    def row(self, icao: str, metric: str, day: date, t: datetime) -> dict | None:
        st = self.stations[icao]
        r: dict = {"station": icao, "metric": metric, "event_date": day.isoformat(),
                   "asof_utc": t.isoformat()}
        for m in MOS_MODELS:
            r[f"f_{m}"], r[f"age_{m}"], r[f"npts_{m}"] = self.mos(icao, m, metric, day, t)

        def full(m):   # MEX/GFS carry one n_x value; NBS/NBE must cover most of the day
            v = r[f"f_{m}"]
            ok = m in ("MEX", "GFS") or (not _isnan(r[f"npts_{m}"]) and r[f"npts_{m}"] >= FULL_DAY_MIN_PTS)
            return v if ok and not _isnan(v) else NAN
        pair = [full(m) for m in ("NBS", "MEX") if not _isnan(full(m))]
        if pair:
            anchor = float(np.mean(pair))
        elif not _isnan(full("GFS")):
            anchor = full("GFS")
        else:
            anchor = full("NBE")
        if _isnan(anchor):
            return None
        avail = [r[f"f_{m}"] for m in MOS_MODELS if not _isnan(r[f"f_{m}"])]
        r["anchor"], r["spread_mos"] = anchor, float(max(avail) - min(avail))
        r["clim_mean"], r["clim_sd"] = self.clim(icao, metric, day)
        r["prev_obs"] = self.bound(icao, metric, day - timedelta(days=1), t)
        if same_day(st, day, t):
            b = self.bound(icao, metric, day, t)
            r["run_ext"] = b
            r["run_gap"] = NAN if _isnan(b) else b - anchor
            r["last_tmpf"], r["last_dwpf"] = self.last_obs(icao, day, t)
        else:
            r["run_ext"] = r["run_gap"] = r["last_tmpf"] = r["last_dwpf"] = NAN
        doy = day.timetuple().tm_yday
        r["doy_sin"], r["doy_cos"] = math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)
        r["lead_h"] = lead_hours(st, day, t)
        r["local_hour"] = t.astimezone(ZoneInfo(st.tz)).hour
        r["lat"], r["lon"], r["elevation_m"] = st.lat, st.lon, st.elevation_m
        return r
