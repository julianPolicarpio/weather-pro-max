from highcast.fetchers import alerts

# Shape observed live from api.weather.gov/alerts/active: onset/expires carry
# a local UTC offset (not "Z"), e.g. "-07:00" for America/Los_Angeles PDT.
ACTIVE = {"features": [
    {"properties": {"event": "Heat Advisory",
                    "onset": "2026-07-19T17:00:00-07:00",
                    "expires": "2026-07-20T03:00:00-07:00"}},
    {"properties": {"event": "Small Craft Advisory",
                    "onset": "2026-07-19T17:00:00-07:00",
                    "expires": "2026-07-20T03:00:00-07:00"}},
]}

def test_parse_active_keeps_heat_events_only():
    rows = alerts.parse_active(ACTIVE, "KLAX", "t0")
    assert len(rows) == 1
    assert rows[0]["event"] == "Heat Advisory"
    # -07:00 local offset normalized to canonical UTC "+00:00" form.
    assert rows[0]["onset_utc"] == "2026-07-20T00:00:00+00:00"
    assert rows[0]["expires_utc"] == "2026-07-20T10:00:00+00:00"


def test_parse_active_missing_expires_is_none():
    payload = {"features": [
        {"properties": {"event": "Heat Advisory",
                        "onset": "2026-07-19T17:00:00-07:00"}},
    ]}
    rows = alerts.parse_active(payload, "KLAX", "t0")
    assert rows[0]["expires_utc"] is None


# Shape observed live from
# https://mesonet.agron.iastate.edu/json/vtec_events_bywfo.py?wfo=LOX&year=2024
# &phenomena=EH : list keyed "events", one row per (event, ugc zone), with
# "issue"/"expire" ISO times and a "significance" code (W/A/Y) that
# distinguishes Warning vs Watch vs Advisory.
ARCHIVE = {"events": [
    {"phenomena": "EH", "significance": "W", "ugc": "CAZ366",
     "issue": "2024-09-05T18:00:00Z", "expire": "2024-09-10T03:00:00Z"},
    {"phenomena": "EH", "significance": "A", "ugc": "CAZ366",
     "issue": "2024-09-03T17:00:00Z", "expire": "2024-09-05T18:00:00Z"},
    {"phenomena": "FW", "significance": "A", "ugc": "CAZ366",  # Fire Wx Watch
     "issue": "2024-11-05T19:00:00Z", "expire": "2024-11-06T19:00:00Z"},
    {"phenomena": "HT", "significance": "Y", "ugc": "CAZ379",  # other zone
     "issue": "2024-07-01T18:00:00Z", "expire": "2024-07-02T03:00:00Z"},
]}


def _cfg_with_zone(zone):
    from highcast.config import AppConfig, StationConfig
    from pathlib import Path
    st = StationConfig(key="k", name="n", icao="KLAX", iem_id="LAX",
                       iem_network="CA_ASOS", wfo="LOX", lat=0.0, lon=0.0,
                       tz="America/Los_Angeles", std_utc_offset=-8,
                       nws_zones=[zone])
    return AppConfig(station=st, display_tz="UTC", synoptic_token="",
                     db_path=Path("x.db"))


def test_parse_archive_maps_significance_and_filters_zone():
    rows = alerts.parse_archive(ARCHIVE, _cfg_with_zone("CAZ366"), "t0")
    assert [r["event"] for r in rows] == ["Excessive Heat Warning",
                                         "Excessive Heat Watch"]
    # Fire Weather WATCH (FW/A) is not a Red Flag Warning; HT row is another zone
    # Z-suffixed IEM times normalized to canonical "+00:00" form.
    assert rows[0]["onset_utc"] == "2024-09-05T18:00:00+00:00"
    assert rows[0]["expires_utc"] == "2024-09-10T03:00:00+00:00"
    assert all(r["station"] == "KLAX" and r["fetched_at"] == "t0" for r in rows)
    assert all(r["event"] in alerts.HEAT_EVENTS for r in rows)


def test_parse_archive_missing_expire_is_none():
    payload = {"events": [
        {"phenomena": "EH", "significance": "W", "ugc": "CAZ366",
         "issue": "2024-09-05T18:00:00Z", "expire": None},
    ]}
    rows = alerts.parse_archive(payload, _cfg_with_zone("CAZ366"), "t0")
    assert rows[0]["expires_utc"] is None


import pytest
from highcast.config import load_config

@pytest.mark.live
def test_live_vtec_2024_has_heat_events():
    rows = alerts.fetch_archive_year(load_config(), 2024)
    assert rows, "adapt parse_archive to the real VTEC payload (print it)"
    assert all(r["event"] in alerts.HEAT_EVENTS for r in rows)
