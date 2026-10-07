import time

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import norm_valid, utcnow_iso

NWS_ALERTS = "https://api.weather.gov/alerts/active"
IEM_VTEC = "https://mesonet.agron.iastate.edu/json/vtec_events_bywfo.py"

HEAT_EVENTS = {"Excessive Heat Warning", "Extreme Heat Warning",
               "Excessive Heat Watch", "Extreme Heat Watch",
               "Heat Advisory", "Red Flag Warning"}

# CONVECTIVE PRODUCTS, captured 2026-08-02. NOT part of the `heat_alert` model
# feature -- see the warning below.
#
# WHY. On 2026-08-01 a thunderstorm cut KMIA 16F in 50 minutes (91.04 -> 75.02,
# +TSRA at 20:43Z) and destroyed the >=81 low. A temperature model structurally
# cannot see that coming; an NWS warning can. `api.weather.gov/alerts/active`
# has returned these all along and `parse_active` was throwing them away.
STORM_EVENTS = {"Severe Thunderstorm Warning", "Severe Thunderstorm Watch",
                "Tornado Warning", "Tornado Watch"}

# ⚠️ EVERYTHING WE STORE. `features._alert_flag` MUST keep filtering to
# HEAT_EVENTS, because it selects from `alerts` and, before this change, every
# row in that table was a heat row -- so it had no event filter at all. Widening
# the fetch without scoping the reader would have made `heat_alert` fire on
# thunderstorms, silently changing a live feature on all four markets. Same
# shape as the evening/day-ahead kinds sharing one label: two different things
# in one table have to stay distinguishable at the point of USE.
TRACKED_EVENTS = HEAT_EVENTS | STORM_EVENTS
# VTEC phenomena codes for the same events
HEAT_PHENOMENA = {"EH", "XH", "HT", "FW"}
# (phenomena, significance) -> event name. Significance matters: EH/A is a
# WATCH not a Warning, and FW/A is a Fire Weather Watch (not a heat product),
# so combos absent from this map are skipped.
_VTEC_EVENT = {
    ("EH", "W"): "Excessive Heat Warning", ("EH", "A"): "Excessive Heat Watch",
    ("XH", "W"): "Extreme Heat Warning", ("XH", "A"): "Extreme Heat Watch",
    ("HT", "Y"): "Heat Advisory", ("FW", "W"): "Red Flag Warning",
}


def parse_active(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    # Live api.weather.gov onset/expires carry local UTC offsets (e.g.
    # "-07:00"), not "Z"/UTC. Normalize through timeutil.norm_valid so
    # onset_utc/expires_utc are always canonical "+00:00" strings and sort
    # lexicographically alongside parse_archive's rows.
    rows = []
    for feat in payload.get("features", []):
        p = feat.get("properties", {})
        if p.get("event") not in TRACKED_EVENTS:
            continue
        onset = p.get("onset") or p.get("effective")
        expires = p.get("expires")
        rows.append({"station": icao, "event": p["event"],
                     "onset_utc": norm_valid(onset) if onset else onset,
                     "expires_utc": norm_valid(expires) if expires else None,
                     "fetched_at": fetched_at})
    return [r for r in rows if r["onset_utc"]]


def fetch_active(cfg: AppConfig) -> list[dict]:
    st = cfg.station
    payload = get_json(NWS_ALERTS, {"point": f"{st.lat},{st.lon}"})
    return parse_active(payload, st.icao, utcnow_iso())


def parse_archive(payload: dict, cfg: AppConfig, fetched_at: str) -> list[dict]:
    # REALITY CHECK (differs from the brief): the real IEM VTEC archive is
    # /json/vtec_events_bywfo.py, not /api/1/vtec/events.json (404s; the
    # /api/1 spec only has point-in-time/SBW VTEC services). Confirmed live
    # for wfo=LOX year=2024: the list is keyed "events" (not "data"), each
    # row is one (event, UGC zone) pair with a scalar "ugc" (so exact
    # nws_zones matching works), ISO "issue"/"expire" times, and a
    # "significance" code (W/A/Y) needed to tell Warning from Watch (and to
    # drop FW/A Fire Weather Watches, which are not Red Flag Warnings).
    zones = set(cfg.station.nws_zones)
    rows = []
    for ev in payload.get("events", []):
        event = _VTEC_EVENT.get((ev.get("phenomena"), ev.get("significance")))
        if event is None:
            continue
        if zones and ev.get("ugc") not in zones:
            continue
        onset = ev.get("issue")
        if not onset:
            continue
        expires = ev.get("expire")
        rows.append({"station": cfg.station.icao, "event": event,
                     "onset_utc": norm_valid(onset),
                     "expires_utc": norm_valid(expires) if expires else None,
                     "fetched_at": fetched_at})
    return rows


def fetch_archive_year(cfg: AppConfig, year: int) -> list[dict]:
    # One request per phenomena code: the unfiltered whole-WFO-year query
    # is slow enough to time out, while phenomena-filtered ones return fast.
    rows: list[dict] = []
    for phenomena in sorted(HEAT_PHENOMENA):
        payload = get_json(IEM_VTEC, {"wfo": cfg.station.wfo, "year": year,
                                      "phenomena": phenomena})
        rows.extend(parse_archive(payload, cfg, utcnow_iso()))
        time.sleep(0.2)  # be polite to the IEM server
    return rows


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "alerts", rows)
