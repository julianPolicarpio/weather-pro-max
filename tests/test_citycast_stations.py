from pathlib import Path

import pytest

from citycast.stations import Station, app_config, load_stations, series_index
from sandbox.stations import STATIONS


def test_registry_matches_sandbox_stations():
    st = load_stations()
    assert set(st) == set(STATIONS) and len(st) == 24
    for icao, s in st.items():
        assert (s.tz, s.std_utc_offset) == STATIONS[icao]
        assert s.iem_id == icao[1:]
        assert s.high_series.startswith("KXHIGH") and s.low_series.startswith("KXLOWT")


def test_primary_series_are_48_distinct_and_index_has_no_duplicates():
    st = load_stations()
    primaries = {s.high_series for s in st.values()} | {s.low_series for s in st.values()}
    assert len(primaries) == 48
    idx = series_index(st)
    assert idx[st["KNYC"].high_series] == ("KNYC", "high")
    assert idx[st["KNYC"].low_series] == ("KNYC", "low")


def _st(icao="KNYC", **kw):
    base = dict(icao=icao, iem_id=icao[1:], iem_network="NY_ASOS", wfo="OKX", lat=40.78,
                lon=-73.97, elevation_m=40.0, tz="America/New_York", std_utc_offset=-5,
                high_series="KXHIGHNY", low_series="KXLOWTNYC")
    return Station(**{**base, **kw})


def test_series_index_includes_aliases_and_rejects_double_mapping():
    a = _st(high_aliases=("KXHIGHOLDNY",))
    assert series_index({"KNYC": a})["KXHIGHOLDNY"] == ("KNYC", "high")
    b = _st("KXXX", high_series="KXHIGHNY", low_series="KXLOWTXXX")
    with pytest.raises(ValueError, match="KXHIGHNY"):
        series_index({"KNYC": a, "KXXX": b})


def test_app_config_carries_station_fields(tmp_path):
    cfg = app_config(_st(), tmp_path / "h.db")
    assert cfg.station.icao == "KNYC" and cfg.station.iem_id == "NYC"
    assert cfg.station.std_utc_offset == -5 and cfg.db_path == tmp_path / "h.db"
