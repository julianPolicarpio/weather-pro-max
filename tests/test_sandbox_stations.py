from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sandbox_helpers import make_market

from sandbox.stations import (STATIONS, climate_day_start_utc, event_started,
                              metric_from_series, station_from_rules)

UTC = timezone.utc

# The 24 settlement codes measured live on 2026-10-04 from every open series'
# rules_primary. If Kalshi adds a city, this list and STATIONS grow together.
LIVE_CODES = ["AUS", "MDW", "DEN", "LAX", "MIA", "NYC", "PHL", "ATL", "BOS", "DFW",
              "DCA", "EWR", "HOU", "LAS", "MSP", "MSY", "OKC", "PHX", "SAN", "SAT",
              "SDF", "SEA", "SFO", "TTN"]


def test_station_from_rules_reads_the_cli_code():
    rules = ("If the maximum temperature recorded at New York City (CLINYC) for "
             "Oct 4, 2026, is greater than 70° fahrenheit")
    assert station_from_rules(rules) == "KNYC"
    assert station_from_rules("recorded at Chicago (CLIMDW) for") == "KMDW"


def test_station_from_rules_none_when_absent():
    assert station_from_rules("no code here") is None
    assert station_from_rules(None) is None


def test_metric_from_series():
    assert metric_from_series("KXHIGHNY") == "high"
    assert metric_from_series("KXHIGHTATL") == "high"
    assert metric_from_series("KXLOWTNYC") == "low"
    assert metric_from_series("KXRAINNYC") is None


def test_every_live_station_has_a_valid_timezone():
    for code in LIVE_CODES:
        tz, offset = STATIONS[f"K{code}"]
        ZoneInfo(tz)  # raises if the zone name is wrong
        assert offset in (-5, -6, -7, -8)
    assert len(STATIONS) == len(LIVE_CODES)


# ---- F2: the climate day starts at local STANDARD midnight -------------------



def test_climate_day_start_is_local_standard_midnight_in_utc():
    assert climate_day_start_utc("KNYC", "2026-10-04") == datetime(2026, 10, 4, 5, tzinfo=UTC)
    assert climate_day_start_utc("KLAX", "2026-10-04") == datetime(2026, 10, 4, 8, tzinfo=UTC)
    assert climate_day_start_utc("KPHX", "2026-10-04") == datetime(2026, 10, 4, 7, tzinfo=UTC)
    # winter: same 05:00Z, because the climate day ignores daylight saving time
    assert climate_day_start_utc("KNYC", "2027-01-04") == datetime(2027, 1, 4, 5, tzinfo=UTC)


def test_climate_day_start_of_an_unknown_station_is_none():
    assert climate_day_start_utc("KXXX", "2026-10-04") is None
    assert climate_day_start_utc(None, "2026-10-04") is None


def test_event_started_boundary_and_fail_closed():
    m = make_market("X")                                  # KNYC, 2026-10-04
    assert not event_started(m, "2026-10-04T04:59:59+00:00")
    assert event_started(m, "2026-10-04T05:00:00+00:00")
    # an unknown or missing station cannot be shown to be pre-day: treat it as started
    assert event_started(replace(m, station="KXXX"), "2026-10-01T00:00:00+00:00")
    assert event_started(replace(m, station=None), "2026-10-01T00:00:00+00:00")
