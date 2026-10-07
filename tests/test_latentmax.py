from datetime import date
from highcast import db as hdb, latentmax
from highcast.config import load_config

CFG = load_config()
ICAO = CFG.station.icao

def _ob(valid, source, **kw):
    base = {"station": ICAO, "valid_utc": valid, "source": source,
            "tmpf": None, "tmpf_precise": None, "dwpf": None, "sknt": None,
            "drct": None, "skyc": None, "six_max_c": None, "six_min_c": None,
            "raw": "", "fetched_at": "x"}
    return base | kw

def test_visible_running_max_combines_sources(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        # climate day 2025-07-16 starts 08:00Z; both obs inside it
        _ob("2025-07-16T18:53:00+00:00", "metar", tmpf_precise=75.02, tmpf=75.2),
        # syn5 stores c_to_f(22 C) = 71.6; whole-C reading -> true F in {71, 72}
        _ob("2025-07-16T20:05:00+00:00", "syn5", tmpf=71.6),
    ])
    lo, hi = latentmax.visible_running_max(conn, CFG, date(2025, 7, 16))
    # metar precise 75.02 dominates both bounds (+/- 0.1 tolerance band)
    assert 74.8 <= lo <= 75.1
    assert 75.0 <= hi <= 75.3

def test_syn5_dominates_when_hotter(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        _ob("2025-07-16T18:53:00+00:00", "metar", tmpf_precise=75.02),
        # c_to_f(26 C) = 78.8; whole-C reading -> true F in {78, 79}
        _ob("2025-07-16T21:05:00+00:00", "syn5", tmpf=78.8),
    ])
    lo, hi = latentmax.visible_running_max(conn, CFG, date(2025, 7, 16))
    assert lo == 78.0 and hi == 79.0

def test_no_obs_returns_none(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    assert latentmax.visible_running_max(conn, CFG, date(2025, 7, 16)) is None


def test_visible_running_extremum_min_and_max_wrapper_unchanged(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        # climate day 2025-07-16 starts 08:00Z; all three obs inside it
        _ob("2025-07-16T18:53:00+00:00", "metar", tmpf_precise=75.02),
        _ob("2025-07-16T19:30:00+00:00", "metar", tmpf_precise=68.02),
        # lowest reading: 60.02 +/- 0.1 tolerance band
        _ob("2025-07-16T20:05:00+00:00", "metar", tmpf_precise=60.02),
    ])
    day = date(2025, 7, 16)

    lo, hi = latentmax.visible_running_extremum(conn, CFG, day, "min")
    # lowest reading (60.02 +/- 0.1) dominates both bounds -> running-MINIMUM
    # interval is that reading's own interval
    assert 59.8 <= lo <= 60.0
    assert 60.0 <= hi <= 60.2

    # visible_running_max wrapper is unaffected: highest reading still wins
    max_lo, max_hi = latentmax.visible_running_max(conn, CFG, day)
    assert 74.8 <= max_lo <= 75.1
    assert 75.0 <= max_hi <= 75.3

    # cross-check: visible_running_extremum(..., "max") matches the wrapper exactly
    assert latentmax.visible_running_extremum(conn, CFG, day, "max") == (max_lo, max_hi)


def test_six_hour_window_maps_report_to_its_synoptic_window():
    """A report at HH:53Z belongs to synoptic hour HH+1 and summarises the
    6 hours ending there."""
    from datetime import datetime, timezone
    ws, we = latentmax._six_hour_window("2025-07-16T11:53:00+00:00")
    assert ws == datetime(2025, 7, 16, 6, tzinfo=timezone.utc)
    assert we == datetime(2025, 7, 16, 12, tzinfo=timezone.utc)
    # an exactly-on-the-hour report is already at its synoptic hour
    ws, we = latentmax._six_hour_window("2025-07-16T18:00:00+00:00")
    assert ws == datetime(2025, 7, 16, 12, tzinfo=timezone.utc)
    assert we == datetime(2025, 7, 16, 18, tzinfo=timezone.utc)


def test_six_min_group_straddling_climate_day_start_is_ignored(tmp_path):
    """The 12Z report's 6-hr min covers 06-12Z, but the KLAX climate day
    starts at 08:00Z — two hours of that window belong to YESTERDAY. Using it
    would import a temperature from the previous climate day. The 18Z report
    (12-18Z) lies wholly inside and must be used."""
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        # 12Z synoptic report: window 06-12Z STRADDLES the 08:00Z day start
        _ob("2025-07-16T11:53:00+00:00", "metar", tmpf_precise=70.0, six_min_c=5.0),
        # 18Z synoptic report: window 12-18Z is wholly inside the climate day
        _ob("2025-07-16T17:53:00+00:00", "metar", tmpf_precise=70.0, six_min_c=15.0),
    ])
    lo, hi = latentmax.visible_running_extremum(conn, CFG, date(2025, 7, 16), "min")
    # c_to_f(15.0) = 59.0 wins; the straddling c_to_f(5.0) = 41.0 is ignored
    assert 58.8 <= lo <= 59.0
    assert 59.0 <= hi <= 59.2


def test_six_max_group_straddling_climate_day_start_is_ignored(tmp_path):
    """Mirror of the min guard: the 12Z report's 6-hr MAX also covers 06-12Z
    and must not be folded into the running max."""
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        _ob("2025-07-16T11:53:00+00:00", "metar", tmpf_precise=70.0, six_max_c=40.0),
        _ob("2025-07-16T17:53:00+00:00", "metar", tmpf_precise=70.0, six_max_c=30.0),
    ])
    lo, hi = latentmax.visible_running_extremum(conn, CFG, date(2025, 7, 16), "max")
    # c_to_f(30.0) = 86.0 wins; the straddling c_to_f(40.0) = 104.0 is ignored
    assert 85.8 <= lo <= 86.0
    assert 86.0 <= hi <= 86.2


def test_six_min_group_tightens_bound_below_every_spot_reading(tmp_path):
    """The whole point: the 6-hr min sees continuously-sampled temperature
    between the hourly spot readings, so it can legitimately sit below all
    of them."""
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        _ob("2025-07-16T17:53:00+00:00", "metar", tmpf_precise=70.0, six_min_c=15.0),
        _ob("2025-07-16T23:53:00+00:00", "metar", tmpf_precise=68.0),
    ])
    day = date(2025, 7, 16)
    lo, hi = latentmax.visible_running_extremum(conn, CFG, day, "min")
    assert 58.8 <= lo <= 59.0 and 59.0 <= hi <= 59.2   # 15.0 C = 59.0 F
    # the max ratchet is untouched by a six_min_c
    max_lo, max_hi = latentmax.visible_running_extremum(conn, CFG, day, "max")
    assert 69.8 <= max_lo <= 70.0 and 70.0 <= max_hi <= 70.2


import pytest

@pytest.mark.live
def test_live_fetch_asos1min():
    rows = latentmax.fetch_asos1min(CFG, date(2025, 7, 1), date(2025, 7, 2))
    assert len(rows) >= 1000   # ~1-min cadence over 2 days
    assert all(r["source"] == "asos1" for r in rows)
    assert all(r["station"] == ICAO for r in rows)
    assert any(r["tmpf"] is not None for r in rows)
    # canonical timestamp form: YYYY-MM-DDTHH:MM:SS+00:00
    assert rows[0]["valid_utc"].endswith("+00:00")
    assert len(rows[0]["valid_utc"]) == 25
