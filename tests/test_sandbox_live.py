import pytest

from sandbox.kalshi_public import PacedGet, list_temperature_series, open_markets
from sandbox.stations import STATIONS, station_from_rules


@pytest.mark.live
def test_every_open_temperature_series_maps_to_a_known_station():
    get = PacedGet()
    seen = 0
    for s in list_temperature_series(get):
        markets = open_markets(get, s["ticker"]).get("markets") or []
        if not markets:
            continue
        seen += 1
        st = station_from_rules(markets[0].get("rules_primary"))
        assert st in STATIONS, (s["ticker"], st)
    assert seen > 0
