"""CONVECTION FETCH, wired into the wake cycle — QUEUE item 3, task 1.

WHY NOW. `forecast_conv` has been dead since 2026-07-29: the fetch script
existed and was never wired in, the same defect `nbm_nbs` had. On 2026-08-06 it
cost a bucket. KMIA settled a low of 75F at 01:51 local with 0.41" of rain; our
best observation was 78.1F at the 01:53 METAR two minutes later, and the model —
which has NO precipitation input at all — predicted 78-79.

The station did not file a 2sTTT six-hour min group in that report (it happens
on ~3% of synoptic reports), so nothing in the METAR stream ever carried the
event. The market priced it and we did not.

WHAT THIS SHIPS, AND WHAT IT DOES NOT. This collects data and nothing more.
Per QUEUE item 3 the gate on USING it is Task 0 — does the FORECAST identify
the same days the OBSERVATION does? — and that is unmeasured. No feature, no
veto, no display reads `forecast_conv` yet, and this file does not change that.

⚠️ BOTH STATIONS. The old hand-run script was KMIA-only, which is why
`forecast_conv` holds zero KLAX rows. LAX has its own convective days.
"""
from datetime import datetime, timedelta, timezone

import pytest

from highcast import backfill, db as hdb
from highcast.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/kmia.yaml")
    object.__setattr__(c, "db_path", tmp_path / "t.db")
    return c


@pytest.fixture
def conn(cfg):
    return hdb.connect(cfg.db_path)


def test_first_ever_fetch_is_due(conn, cfg):
    """No rows at all ⇒ fetch. This is the state the feed has been stuck in
    since 2026-07-29."""
    assert backfill._conv_fetch_due(conn, cfg) is True


def test_a_recent_fetch_is_not_due(conn, cfg):
    """The wake cycle runs every 5 minutes; day-ahead convective guidance does
    not change that fast. Without this the feed would be re-requested 288 times
    a day for nothing."""
    hdb.upsert(conn, "forecast_conv", [{
        "station": cfg.station.icao, "valid_utc": "2026-08-06T18:00",
        "model": "gfs_hrrr", "pop": 40.0, "qpf": 0.1, "cape": 1500.0,
        "fetched_at": datetime.now(timezone.utc).isoformat()}])
    assert backfill._conv_fetch_due(conn, cfg) is False


def test_a_stale_fetch_is_due_again(conn, cfg):
    old = (datetime.now(timezone.utc)
           - timedelta(minutes=backfill.CONV_FETCH_MIN_GAP_MIN + 5)).isoformat()
    hdb.upsert(conn, "forecast_conv", [{
        "station": cfg.station.icao, "valid_utc": "2026-08-06T18:00",
        "model": "gfs_hrrr", "pop": 40.0, "qpf": 0.1, "cape": 1500.0,
        "fetched_at": old}])
    assert backfill._conv_fetch_due(conn, cfg) is True


def test_due_is_per_station(conn, cfg):
    """KLAX must not be starved because KMIA just fetched — the same reasoning
    `_nbs_fetch_due` carries. `forecast_conv` currently holds ZERO KLAX rows
    precisely because the hand-run script was KMIA-only."""
    hdb.upsert(conn, "forecast_conv", [{
        "station": "KMIA", "valid_utc": "2026-08-06T18:00", "model": "gfs_hrrr",
        "pop": 40.0, "qpf": 0.1, "cape": 1500.0,
        "fetched_at": datetime.now(timezone.utc).isoformat()}])
    klax = load_config("config/klax.yaml")
    object.__setattr__(klax, "db_path", cfg.db_path)
    assert backfill._conv_fetch_due(conn, klax) is True


def test_both_due_checks_are_the_same_code(conn, cfg):
    """THE SPEC ASKED FOR THIS BY NAME (§4.1): "GENERALISE, do not copy. Two
    near-identical due-checks is how the third one gets written wrong."

    The first version of this feature copied `_nbs_fetch_due`. This test is the
    thing that stops feed #3 doing it again — both callers must route through
    `_feed_fetch_due`, so there is one implementation to get right.
    """
    import inspect
    assert hasattr(backfill, "_feed_fetch_due")
    for fn in (backfill._conv_fetch_due, backfill._nbs_fetch_due):
        src = inspect.getsource(fn)
        assert "_feed_fetch_due(" in src, fn.__name__
        # the arithmetic must live in ONE place, not be re-derived per caller
        assert "total_seconds()" not in src, fn.__name__


def test_shared_due_check_filters_by_model_when_asked(conn, cfg):
    """`forecasts` holds many models in one table; `forecast_conv` holds one
    feed. The shared helper has to serve both without the caller re-writing
    the query."""
    from datetime import datetime, timezone
    hdb.upsert(conn, "forecasts", [{
        "station": cfg.station.icao, "target_date": "2026-08-07",
        "model": "gfs", "run_date": "2026-08-06", "max_temp_f": 88.0,
        "metric": "high", "fetched_at": datetime.now(timezone.utc).isoformat()}])
    # a FRESH row for a DIFFERENT model must not satisfy nbm_nbs's check
    assert backfill._nbs_fetch_due(conn, cfg) is True


def test_unparseable_stamp_fails_open(conn, cfg):
    """Fails OPEN, like `_nbs_fetch_due`: one extra HTTP call costs nothing,
    and this feed going quiet is what cost a bucket on 2026-08-06."""
    hdb.upsert(conn, "forecast_conv", [{
        "station": cfg.station.icao, "valid_utc": "2026-08-06T18:00",
        "model": "gfs_hrrr", "pop": 40.0, "qpf": 0.1, "cape": 1500.0,
        "fetched_at": "not-a-timestamp"}])
    assert backfill._conv_fetch_due(conn, cfg) is True


def test_parse_picks_precipitation_not_the_probability_column(cfg):
    """`precipitation` substring-matches `precipitation_probability`. The
    hand-run script had to guard this; the guard has to survive the move."""
    hourly = {
        "time": ["2026-08-06T12:00", "2026-08-06T13:00"],
        "precipitation_probability_previous_day1": [55.0, 60.0],
        "precipitation_previous_day1": [0.2, 0.4],
        "cape_previous_day1": [1800.0, 2100.0],
    }
    rows = backfill._parse_conv(hourly, "KMIA", "gfs_hrrr", "2026-08-06T00:00Z")
    assert [r["pop"] for r in rows] == [55.0, 60.0]
    assert [r["qpf"] for r in rows] == [0.2, 0.4]
    assert [r["cape"] for r in rows] == [1800.0, 2100.0]


def test_parse_drops_all_null_hours(cfg):
    hourly = {"time": ["2026-08-06T12:00", "2026-08-06T13:00"],
              "precipitation_probability_previous_day1": [None, 60.0],
              "precipitation_previous_day1": [None, 0.4],
              "cape_previous_day1": [None, 2100.0]}
    rows = backfill._parse_conv(hourly, "KMIA", "gfs_hrrr", "2026-08-06T00:00Z")
    assert len(rows) == 1
    assert rows[0]["valid_utc"] == "2026-08-06T13:00"


def test_parse_survives_an_empty_payload(cfg):
    assert backfill._parse_conv({}, "KMIA", "gfs_hrrr", "x") == []
