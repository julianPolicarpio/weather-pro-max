from types import SimpleNamespace

import math
import pytest

from highcast import db
from highcast.weather_features import build_lane_a_features, checkpoint_utc, rain_state, wind_components

B1 = {"cur_rh_pct", "cur_dewpoint_depression_f", "rh_change_1h", "dewpoint_change_3h"}
B2 = {"wind_u_kt", "wind_v_kt", "onshore_change_1h", "onshore_change_3h", "wind_speed_change_1h", "wind_direction_sin", "wind_direction_cos", "gust_excess_kt"}
B3 = {"wx_rain_now", "wx_rain_seen_3h", "hours_since_rain", "visibility_change_1h", "ceiling_change_1h", "lowest_cloud_change_1h", "cloud_clearing_flag", "temp_recovery_since_rain_f"}
KLAX_B4 = {"marine_clear_utc", "low_cloud_persistent", "onshore_transition", "temp_acceleration_after_clear_fph"}
KMIA_B4 = {"rain_onset_utc", "rain_end_utc", "min_temp_after_rain_f", "warming_rate_since_rain_end_fph", "persistent_cloud_flag", "gust_front_wind_shift_kt"}


def _cfg(icao, tz):
    return SimpleNamespace(station=SimpleNamespace(
        icao=icao, tz=tz, std_utc_offset={"KMIA": -5, "KLAX": -8}[icao]
    ))


def _put(conn, station, valid_utc, *, source="metar", temp=80.0, dew=70.0, speed=10.0, direction=90.0, skyc="SCT", raw="METAR 10SM SCT030", ceiling=3000, lowest=3000, visibility=10.0):
    conn.execute(
        """INSERT INTO obs (station, valid_utc, source, tmpf, tmpf_precise, dwpf, sknt, drct, skyc, raw, fetched_at, ceiling_ft, sky_lowest_ft, vis_sm)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fixture', ?, ?, ?)""",
        (station, valid_utc, source, temp, temp, dew, speed, direction, skyc, raw, ceiling, lowest, visibility),
    )
    conn.commit()


@pytest.mark.parametrize(("station_tz", "target_date", "hour", "want"), [
    ("America/Los_Angeles", "2026-07-01", 9, "2026-07-01T16:00:00+00:00"),
    ("America/Los_Angeles", "2026-01-01", 9, "2026-01-01T17:00:00+00:00"),
    ("America/New_York", "2026-07-01", 9, "2026-07-01T13:00:00+00:00"),
    ("America/New_York", "2026-01-01", 9, "2026-01-01T14:00:00+00:00"),
])
def test_checkpoint_utc_uses_station_dst_rules(station_tz, target_date, hour, want):
    assert checkpoint_utc(target_date, hour, station_tz).isoformat() == want


def test_registered_bundle_keys_and_units_are_metadata_only(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    _put(conn, "KMIA", "2026-07-01T12:50:00+00:00")
    got = build_lane_a_features(conn, _cfg("KMIA", "America/New_York"), "2026-07-01", 9)
    assert set(got["B1"]) == B1
    assert set(got["B2"]) == B2
    assert set(got["B3"]) == B3
    assert set(got["B4"]) == KMIA_B4
    assert "units" not in got
    for bundle, keys in (("B1", B1), ("B2", B2), ("B3", B3), ("B4", KMIA_B4)):
        assert set(got["metadata"][bundle]["units"]) == keys
        assert set(got["metadata"][bundle]["missing"]) == keys


def test_builder_is_station_source_and_asof_isolated_with_row_lineage(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KMIA", "America/New_York")
    _put(conn, "KMIA", "2026-07-01T12:50:00+00:00", temp=80.0)
    _put(conn, "KLAX", "2026-07-01T12:55:00+00:00", temp=45.0)
    _put(conn, "KMIA", "2026-07-01T12:55:00+00:00", source="syn5", temp=45.0)
    _put(conn, "KMIA", "2026-07-01T13:01:00+00:00", temp=99.0)
    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    assert got["asof_utc"] == "2026-07-01T13:00:00+00:00"
    assert got["lineage"]["metar_rows"] == [{"station": "KMIA", "source": "metar", "valid_utc": "2026-07-01T12:50:00+00:00"}]
    assert got["B2"]["wind_u_kt"] == pytest.approx(-10.0)


def test_future_observations_cannot_change_trends_or_timing_features(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KMIA", "America/New_York")
    _put(conn, "KMIA", "2026-07-01T10:00:00+00:00", temp=76.0)
    _put(conn, "KMIA", "2026-07-01T12:00:00+00:00", temp=80.0)
    before = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    _put(conn, "KMIA", "2026-07-01T13:01:00+00:00", temp=55.0, raw="METAR 1SM +TSRA BKN005")
    after = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    assert after["B1"]["rh_change_1h"] == before["B1"]["rh_change_1h"]
    assert after["B3"]["wx_rain_seen_3h"] == before["B3"]["wx_rain_seen_3h"]
    assert after["B4"]["rain_onset_utc"] == before["B4"]["rain_onset_utc"]


def test_stale_or_absent_observations_stay_missing_not_dry_or_false(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    _put(conn, "KMIA", "2026-06-30T12:50:00+00:00", raw="METAR 10SM SCT030")
    got = build_lane_a_features(conn, _cfg("KMIA", "America/New_York"), "2026-07-01", 9)
    assert math.isnan(got["B1"]["cur_rh_pct"])
    assert math.isnan(got["B2"]["wind_u_kt"])
    assert math.isnan(got["B3"]["wx_rain_now"])
    assert math.isnan(got["B3"]["wx_rain_seen_3h"])
    assert math.isnan(got["B3"]["cloud_clearing_flag"])
    assert got["B4"]["rain_onset_utc"] is None
    assert got["metadata"]["B4"]["missing"]["persistent_cloud_flag"] is True
    assert got["lineage"]["metar_rows"] == []


@pytest.mark.parametrize(("raw", "want"), [(None, "missing"), ("METAR KMIA 011200Z 09010KT 10SM SCT030", "none"), ("METAR KMIA 011200Z 09010KT 2SM -RA BKN020", "rain"), ("METAR KMIA 011200Z 09010KT 10SM SCT030 RMK RAB05E10", "none")])
def test_rain_state_distinguishes_missing_no_token_and_present_rain(raw, want):
    assert rain_state(raw) == want


def test_gust_missingness_and_circular_wind_encoding(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KLAX", "America/Los_Angeles")
    _put(conn, "KLAX", "2026-07-01T15:50:00+00:00", speed=12.0, direction=359.0, raw="METAR KLAX 011550Z 35912KT 10SM SCT030")
    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    assert math.isnan(got["B2"]["gust_excess_kt"])
    assert got["metadata"]["B2"]["missing"]["gust_excess_kt"] is True
    u, v = wind_components(10.0, 90.0)
    assert u == pytest.approx(-10.0)
    assert v == pytest.approx(0.0, abs=1e-12)
    assert got["B2"]["wind_direction_sin"] == pytest.approx(math.sin(math.radians(359)))
    assert got["B2"]["wind_direction_cos"] == pytest.approx(math.cos(math.radians(359)))


def test_kmia_uses_the_first_row_of_a_multirow_target_day_rain_event(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KMIA", "America/New_York")
    _put(conn, "KMIA", "2026-06-30T13:00:00+00:00", raw="METAR 2SM RA BKN008")
    _put(conn, "KMIA", "2026-07-01T10:00:00+00:00", temp=74.0, raw="METAR 2SM RA BKN008", skyc="BKN", ceiling=800)
    _put(conn, "KMIA", "2026-07-01T10:30:00+00:00", temp=73.0, raw="METAR 2SM -RA BKN008", skyc="BKN", ceiling=800)
    _put(conn, "KMIA", "2026-07-01T11:00:00+00:00", temp=76.0, raw="METAR 6SM BKN020", skyc="BKN", ceiling=2000)
    _put(conn, "KMIA", "2026-07-01T12:00:00+00:00", temp=80.0, raw="METAR 10SM SCT030", skyc="SCT", ceiling=None)
    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    assert got["B4"]["rain_onset_utc"] == "2026-07-01T10:00:00+00:00"
    assert got["B4"]["rain_end_utc"] == "2026-07-01T11:00:00+00:00"
    assert got["B4"]["min_temp_after_rain_f"] == pytest.approx(76.0)
    assert got["B4"]["warming_rate_since_rain_end_fph"] == pytest.approx(4.0)
    assert got["B4"]["persistent_cloud_flag"] == 1.0


def test_klax_b4_uses_target_day_clear_and_persistence_with_acceleration(tmp_path):
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KLAX", "America/Los_Angeles")
    _put(conn, "KLAX", "2026-06-01T16:00:00+00:00", temp=90.0, skyc="SCT", ceiling=None, raw="METAR 10SM SCT030")
    _put(conn, "KLAX", "2026-07-01T14:00:00+00:00", temp=60.0, speed=5.0, direction=250.0, skyc="BKN", ceiling=800, raw="METAR 5SM BKN008")
    _put(conn, "KLAX", "2026-07-01T15:00:00+00:00", temp=62.0, speed=8.0, direction=250.0, skyc="BKN", ceiling=1200, raw="METAR 6SM BKN012")
    _put(conn, "KLAX", "2026-07-01T15:30:00+00:00", temp=64.0, speed=10.0, direction=250.0, skyc="SCT", ceiling=None, raw="METAR 10SM SCT030")
    _put(conn, "KLAX", "2026-07-01T15:50:00+00:00", temp=68.0, speed=12.0, direction=250.0, skyc="SCT", ceiling=None, raw="METAR 10SM SCT030")
    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)
    assert set(got["B4"]) == KLAX_B4
    assert got["B4"]["marine_clear_utc"] == "2026-07-01T15:30:00+00:00"
    assert got["B4"]["low_cloud_persistent"] == 1.0
    assert got["B4"]["onshore_transition"] == pytest.approx(7.0)
    assert got["B4"]["temp_acceleration_after_clear_fph"] == pytest.approx(8.0)


def test_dst_civil_midnight_row_before_climate_day_is_excluded_from_kmia_events(tmp_path):
    # Break caught: using IANA civil midnight instead of this project's fixed
    # local-standard climate-day boundary during DST.
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KMIA", "America/New_York")
    _put(conn, "KMIA", "2026-07-01T04:30:00+00:00", raw="METAR 2SM RA BKN008", skyc="BKN", ceiling=800)
    _put(conn, "KMIA", "2026-07-01T12:50:00+00:00", raw="METAR 10SM SCT030", skyc="SCT", ceiling=None)

    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)

    assert got["B4"]["rain_onset_utc"] is None
    assert got["lineage"]["metar_rows"] == [{
        "station": "KMIA", "source": "metar", "valid_utc": "2026-07-01T12:50:00+00:00",
    }]


def test_sparse_eight_hour_old_rows_cannot_supply_one_or_three_hour_trends(tmp_path):
    # Break caught: treating any earlier target-day row as an approximate lag.
    conn = db.connect(tmp_path / "features.db")
    cfg = _cfg("KMIA", "America/New_York")
    _put(conn, "KMIA", "2026-07-01T05:00:00+00:00", temp=60.0, dew=40.0)
    _put(conn, "KMIA", "2026-07-01T12:50:00+00:00", temp=80.0, dew=70.0)

    got = build_lane_a_features(conn, cfg, "2026-07-01", 9)

    assert math.isnan(got["B1"]["rh_change_1h"])
    assert math.isnan(got["B1"]["dewpoint_change_3h"])
    assert math.isnan(got["B2"]["onshore_change_1h"])
