from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import c_to_f, norm_valid, utcnow_iso

SYNOPTIC = "https://api.synopticdata.com/v2/stations/timeseries"


def parse_rows(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    stations = payload.get("STATION") or []
    if not stations:
        return []
    obs = stations[0].get("OBSERVATIONS", {})
    times = obs.get("date_time") or []
    temps = obs.get("air_temp_set_1") or []
    rows = []
    for t, c in zip(times, temps):
        if c is None:
            continue
        rows.append({
            "station": icao, "valid_utc": norm_valid(t),
            "source": "syn5", "tmpf": round(c_to_f(float(c)), 2),
            "tmpf_precise": None, "dwpf": None, "sknt": None, "drct": None,
            "skyc": None, "six_max_c": None, "six_min_c": None, "raw": None,
            "fetched_at": fetched_at,
        })
    return rows


def fetch_recent(cfg: AppConfig, minutes: int = 360) -> list[dict]:
    if not cfg.synoptic_token:
        return []
    payload = get_json(SYNOPTIC, {
        "stid": cfg.station.icao, "recent": minutes, "vars": "air_temp",
        "token": cfg.synoptic_token,
    })
    return parse_rows(payload, cfg.station.icao, utcnow_iso())


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "obs", rows)
