"""Offline, leakage-safe METAR feature builder for the Phase 0 experiment."""

import math
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.timeutil import climate_day_bounds


CURRENT_LOOKBACK = timedelta(hours=2)
TREND_LOOKBACK = timedelta(hours=3)
LAG_TOLERANCE = timedelta(minutes=30)
RAIN_LOOKBACK = timedelta(hours=3)
KLAX_CLEARING_CEILING_FT = 3000.0
_ONSHORE_DEGREES = {"KLAX": 250.0, "KMIA": 90.0}
_GUST = re.compile(r"\b(?:\d{3}|VRB)\d{2,3}G(\d{2,3})KT\b")
_RAIN = re.compile(r"(?:^|\s)(?:[-+]|VC)?(?:MI|BC|PR|DR|BL|SH|TS|FZ)?(?:DZ|RA)(?:DZ|RA|SN|SG|PL|GR|GS|BR|FG|FU|HZ|DU|SA|PY|SQ|FC|TS)*(?=\s|$)")
_CLEAR_SKY = {"CLR", "SKC", "FEW", "SCT"}


def checkpoint_utc(target_date: str, hour: int, station_tz: str) -> datetime:
    d = datetime.fromisoformat(target_date).date()
    return datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(station_tz)).astimezone(timezone.utc)


def wind_components(speed_kt: float | None, direction_deg: float | None) -> tuple[float, float]:
    if speed_kt is None or direction_deg is None:
        return math.nan, math.nan
    r = math.radians(float(direction_deg))
    return -float(speed_kt) * math.sin(r), -float(speed_kt) * math.cos(r)


def wind_direction_encoding(direction_deg: float | None) -> tuple[float, float]:
    if direction_deg is None:
        return math.nan, math.nan
    r = math.radians(float(direction_deg))
    return math.sin(r), math.cos(r)


def rain_state(raw: str | None) -> str:
    if raw is None:
        return "missing"
    return "rain" if _RAIN.search(raw.split(" RMK", 1)[0]) else "none"


def _n(row, key):
    return math.nan if row is None or row[key] is None else float(row[key])


def _temp(row):
    return _n(row, "tmpf_precise") if row is not None and row["tmpf_precise"] is not None else _n(row, "tmpf")


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _diff(a, b):
    return math.nan if math.isnan(a) or math.isnan(b) else a - b


def _rh(temp_f, dew_f):
    if math.isnan(temp_f) or math.isnan(dew_f):
        return math.nan
    t, d = (temp_f - 32) * 5 / 9, (dew_f - 32) * 5 / 9
    return 100 * math.exp(17.625 * d / (243.04 + d) - 17.625 * t / (243.04 + t))


def _onshore(speed, direction, station):
    if math.isnan(speed) or math.isnan(direction):
        return math.nan
    if station not in _ONSHORE_DEGREES:
        raise ValueError(f"Lane A needs a declared onshore direction for {station}")
    return speed * math.cos(math.radians(direction - _ONSHORE_DEGREES[station]))


def _gust(raw):
    if raw is None:
        return math.nan
    m = _GUST.search(raw.split(" RMK", 1)[0])
    return float(m.group(1)) if m else math.nan


def _prior(rows, asof, hours):
    requested = timedelta(hours=hours)
    if requested > TREND_LOOKBACK:
        return None
    expected = asof - requested
    earliest = expected - LAG_TOLERANCE
    return next((r for r in reversed(rows)
                 if earliest.isoformat() <= r["valid_utc"] <= expected.isoformat()), None)


def _cloud_state(row):
    """low/high/clear/unknown; null ceiling is clear only with a clear-sky code."""
    if row is None or row["skyc"] is None:
        return "unknown"
    ceiling, sky = _n(row, "ceiling_ft"), row["skyc"]
    if sky in _CLEAR_SKY and math.isnan(ceiling):
        return "clear"
    if not math.isnan(ceiling):
        return "low" if ceiling < KLAX_CLEARING_CEILING_FT else "high"
    return "unknown"


_UNITS = {
    "cur_rh_pct": "percent", "cur_dewpoint_depression_f": "degF", "rh_change_1h": "percentage_points", "dewpoint_change_3h": "degF",
    "wind_u_kt": "kt", "wind_v_kt": "kt", "onshore_change_1h": "kt", "onshore_change_3h": "kt", "wind_speed_change_1h": "kt", "wind_direction_sin": "unitless", "wind_direction_cos": "unitless", "gust_excess_kt": "kt",
    "wx_rain_now": "boolean", "wx_rain_seen_3h": "boolean", "hours_since_rain": "hours", "visibility_change_1h": "statute_miles", "ceiling_change_1h": "ft", "lowest_cloud_change_1h": "ft", "cloud_clearing_flag": "boolean", "temp_recovery_since_rain_f": "degF",
    "marine_clear_utc": "iso8601", "low_cloud_persistent": "boolean", "onshore_transition": "kt", "temp_acceleration_after_clear_fph": "degF_per_hour",
    "rain_onset_utc": "iso8601", "rain_end_utc": "iso8601", "min_temp_after_rain_f": "degF", "warming_rate_since_rain_end_fph": "degF_per_hour", "persistent_cloud_flag": "boolean", "gust_front_wind_shift_kt": "kt",
}


def _metadata(bundle, values, lookback):
    return {"source": "obs:metar", "allowable_lookback": lookback,
            "units": {k: _UNITS[k] for k in values},
            "missing": {k: _missing(v) for k, v in values.items()}}


def build_lane_a_features(conn, cfg, target_date: str, hour: int) -> dict:
    """Return registered B1--B3 plus station-specific B4 from target-day METARs.

    All rows are explicitly station/source/as-of isolated.  Target-day bounds
    prevent prior-day/month events becoming current checkpoint features.
    """
    station, tz = cfg.station.icao, cfg.station.tz
    asof = checkpoint_utc(target_date, hour, tz)
    day_start, _ = climate_day_bounds(date.fromisoformat(target_date), cfg.station.std_utc_offset)
    rows = list(conn.execute(
        """SELECT station, source, valid_utc, tmpf, tmpf_precise, dwpf, sknt, drct,
                  skyc, raw, ceiling_ft, sky_lowest_ft, vis_sm
           FROM obs WHERE station=? AND source='metar' AND valid_utc >= ? AND valid_utc <= ?
           ORDER BY valid_utc""",
        (station, day_start.isoformat(), asof.isoformat()),
    ))
    current = rows[-1] if rows and rows[-1]["valid_utc"] >= (asof - CURRENT_LOOKBACK).isoformat() else None
    p1, p3 = _prior(rows, asof, 1), _prior(rows, asof, 3)
    temp, dew, speed, direction = _temp(current), _n(current, "dwpf"), _n(current, "sknt"), _n(current, "drct")
    rh, onshore = _rh(temp, dew), _onshore(speed, direction, station)
    p1_temp, p1_dew, p1_speed, p1_direction = _temp(p1), _n(p1, "dwpf"), _n(p1, "sknt"), _n(p1, "drct")
    p3_dew, p3_speed, p3_direction = _n(p3, "dwpf"), _n(p3, "sknt"), _n(p3, "drct")
    p1_rh, p1_onshore, p3_onshore = _rh(p1_temp, p1_dew), _onshore(p1_speed, p1_direction, station), _onshore(p3_speed, p3_direction, station)
    u, v = wind_components(None if math.isnan(speed) else speed, None if math.isnan(direction) else direction)
    d_sin, d_cos = wind_direction_encoding(None if math.isnan(direction) else direction)
    gust = _gust(current["raw"]) if current else math.nan

    b1 = {"cur_rh_pct": rh, "cur_dewpoint_depression_f": _diff(temp, dew), "rh_change_1h": _diff(rh, p1_rh), "dewpoint_change_3h": _diff(dew, p3_dew)}
    b2 = {"wind_u_kt": u, "wind_v_kt": v, "onshore_change_1h": _diff(onshore, p1_onshore), "onshore_change_3h": _diff(onshore, p3_onshore), "wind_speed_change_1h": _diff(speed, p1_speed), "wind_direction_sin": d_sin, "wind_direction_cos": d_cos, "gust_excess_kt": _diff(gust, speed)}

    recent = [r for r in rows if r["valid_utc"] >= (asof - RAIN_LOOKBACK).isoformat()]
    states = [rain_state(r["raw"]) for r in recent]
    now_state = rain_state(current["raw"]) if current else "missing"
    rain_now = 1.0 if now_state == "rain" else (0.0 if now_state == "none" else math.nan)
    rain_seen = 1.0 if "rain" in states else (math.nan if not states or "missing" in states else 0.0)
    rain_indices = [i for i, r in enumerate(rows) if rain_state(r["raw"]) == "rain"]
    last_rain_i = rain_indices[-1] if rain_indices else None
    last_rain = rows[last_rain_i] if last_rain_i is not None else None
    hours_since = ((asof - datetime.fromisoformat(last_rain["valid_utc"])).total_seconds() / 3600 if last_rain else math.nan)
    p1_state, cur_state = _cloud_state(p1), _cloud_state(current)
    clearing = (1.0 if p1_state == "low" and cur_state in {"high", "clear"} else (0.0 if p1_state != "unknown" and cur_state != "unknown" else math.nan))
    b3 = {"wx_rain_now": rain_now, "wx_rain_seen_3h": rain_seen, "hours_since_rain": hours_since,
          "visibility_change_1h": _diff(_n(current, "vis_sm"), _n(p1, "vis_sm")),
          "ceiling_change_1h": _diff(_n(current, "ceiling_ft"), _n(p1, "ceiling_ft")),
          "lowest_cloud_change_1h": _diff(_n(current, "sky_lowest_ft"), _n(p1, "sky_lowest_ft")),
          "cloud_clearing_flag": clearing, "temp_recovery_since_rain_f": _diff(temp, _temp(last_rain)) if last_rain else math.nan}

    if station == "KLAX":
        states_day = [_cloud_state(r) for r in rows]
        clear_i = next((i for i in range(1, len(rows)) if states_day[i - 1] == "low" and states_day[i] in {"high", "clear"}), None)
        known_cloud = [s for s in states_day if s != "unknown"]
        persistent = 1.0 if len(known_cloud) >= 2 and known_cloud.count("low") >= 2 else (0.0 if len(known_cloud) >= 2 else math.nan)
        first, last = (rows[0], current) if len(rows) >= 2 and current else (None, None)
        transition = _diff(_onshore(_n(last, "sknt"), _n(last, "drct"), station), _onshore(_n(first, "sknt"), _n(first, "drct"), station)) if first else math.nan
        accel = math.nan
        if clear_i is not None and current is not None and clear_i + 1 < len(rows):
            before, clear = rows[clear_i - 1], rows[clear_i]
            t_before, t_clear = datetime.fromisoformat(before["valid_utc"]), datetime.fromisoformat(clear["valid_utc"])
            pre = _diff(_temp(clear), _temp(before)) / ((t_clear - t_before).total_seconds() / 3600)
            current_at = datetime.fromisoformat(current["valid_utc"])
            post = _diff(temp, _temp(clear)) / ((current_at - t_clear).total_seconds() / 3600)
            accel = _diff(post, pre)
        b4 = {"marine_clear_utc": rows[clear_i]["valid_utc"] if clear_i is not None else None,
              "low_cloud_persistent": persistent, "onshore_transition": transition,
              "temp_acceleration_after_clear_fph": accel}
        b4_lookback = "target-day event window; clear threshold 3000 ft"
    elif station == "KMIA":
        onset_i = None
        if last_rain_i is not None:
            onset_i = last_rain_i
            while onset_i > 0 and rain_state(rows[onset_i - 1]["raw"]) == "rain":
                onset_i -= 1
        end_i = next((i for i in range(last_rain_i + 1, len(rows)) if rain_state(rows[i]["raw"]) == "none"), None) if last_rain_i is not None else None
        after = rows[end_i:] if end_i is not None else []
        after_temps = [_temp(r) for r in after if not math.isnan(_temp(r))]
        end_at = datetime.fromisoformat(rows[end_i]["valid_utc"]) if end_i is not None else None
        current_at = datetime.fromisoformat(current["valid_utc"]) if current else None
        warming = _diff(temp, _temp(rows[end_i])) / ((current_at - end_at).total_seconds() / 3600) if end_at and current_at and current_at > end_at else math.nan
        cloud_after = [_cloud_state(r) for r in rows[onset_i:] if _cloud_state(r) != "unknown"] if onset_i is not None else []
        persistent = 1.0 if len(cloud_after) >= 2 and cloud_after.count("low") >= 2 else (0.0 if len(cloud_after) >= 2 else math.nan)
        onset = rows[onset_i] if onset_i is not None else None
        onset_u, onset_v = wind_components(None if math.isnan(_n(onset, "sknt")) else _n(onset, "sknt"), None if math.isnan(_n(onset, "drct")) else _n(onset, "drct"))
        shift = math.hypot(u - onset_u, v - onset_v) if current and not any(math.isnan(x) for x in (u, v, onset_u, onset_v)) else math.nan
        b4 = {"rain_onset_utc": onset["valid_utc"] if onset else None, "rain_end_utc": rows[end_i]["valid_utc"] if end_i is not None else None,
              "min_temp_after_rain_f": min(after_temps) if after_temps else math.nan,
              "warming_rate_since_rain_end_fph": warming, "persistent_cloud_flag": persistent,
              "gust_front_wind_shift_kt": shift}
        b4_lookback = "target-day rain-event window"
    else:
        raise ValueError(f"Lane A supports only KMIA and KLAX, got {station}")

    return {"station": station, "target_date": target_date, "hour": hour, "asof_utc": asof.isoformat(),
            "B1": b1, "B2": b2, "B3": b3, "B4": b4,
            "metadata": {"B1": _metadata("B1", b1, "current <=2h; 1h/3h lag ±30m"), "B2": _metadata("B2", b2, "current <=2h; 1h/3h lag ±30m"), "B3": _metadata("B3", b3, "current <=2h; rain/changes <=3h; target-day recovery"), "B4": _metadata("B4", b4, b4_lookback)},
            "lineage": {"metar_rows": [{"station": r["station"], "source": r["source"], "valid_utc": r["valid_utc"]} for r in rows]}}
