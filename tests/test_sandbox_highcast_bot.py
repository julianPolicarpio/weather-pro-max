import json
import logging
import sqlite3

import pytest
from sandbox_helpers import three_buckets

from sandbox.bots import HighcastBot
from sandbox.models import Snapshot

T = "2026-10-04T11:00:00+00:00"


def make_highcast_db(path):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT,
                 kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, metric TEXT)""")
    rows = [
        ("2026-10-04T10:00:00+00:00", {"67": 0.2, "68": 0.2, "69": 0.3, "70": 0.2, "71": 0.1}),
        ("2026-10-04T12:00:00+00:00", {"71": 1.0}),   # AFTER the snapshot: must be ignored
    ]
    for issued, b in rows:
        c.execute("INSERT INTO predictions VALUES ('KLAX','2026-10-04',?, 'intraday', NULL, NULL, ?, 'high')",
                  (issued, json.dumps(b)))
    c.commit()
    c.close()


def lax_buckets():
    return three_buckets(series="KXHIGHLAX", station="KLAX", event="KXHIGHLAX-26OCT04")


def test_reads_latest_prediction_at_or_before_the_snapshot(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    markets = lax_buckets()
    pmf = HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    t69, mid, t70 = (m.ticker for m in markets)
    assert pmf[t69] == pytest.approx(0.4)
    assert pmf[mid] == pytest.approx(0.5)
    assert pmf[t70] == pytest.approx(0.1)


def test_other_stations_and_missing_db_give_nothing(tmp_path):
    ny = three_buckets()
    assert HighcastBot(str(tmp_path / "x.db")).pmf_for_event(ny, Snapshot(1, T, tuple(ny))) is None
    lax = lax_buckets()
    assert HighcastBot(str(tmp_path / "missing.db")).pmf_for_event(lax, Snapshot(1, T, tuple(lax))) is None


def test_never_writes_to_the_highcast_db(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    before = db.read_bytes()
    markets = lax_buckets()
    HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)))
    assert db.read_bytes() == before


def test_connects_read_only(tmp_path, monkeypatch):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    real_connect = sqlite3.connect
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("sandbox.bots.sqlite3.connect", spy)
    markets = lax_buckets()
    HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert "mode=ro" in args[0]
    assert kwargs.get("uri") is True
    assert kwargs.get("timeout") == 30   # ride out a write lock instead of failing at 5 s


def test_db_error_is_logged(tmp_path, caplog):
    db = tmp_path / "highcast.db"
    db.write_bytes(b"not a database")
    markets = lax_buckets()
    with caplog.at_level(logging.WARNING, logger="sandbox.bots"):
        pmf = HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    assert pmf is None
    assert any("KLAX" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_malformed_buckets_json_is_logged(tmp_path, caplog):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO predictions VALUES ('KLAX','2026-10-04','2026-10-04T10:30:00+00:00',"
              "'intraday', NULL, NULL, 'not json', 'high')")
    c.commit()
    c.close()
    markets = lax_buckets()
    with caplog.at_level(logging.WARNING, logger="sandbox.bots"):
        pmf = HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    assert pmf is None
    assert any("KLAX" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_metric_and_boundary_filters(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    c = sqlite3.connect(db)   # a LOW row issued at exactly the snapshot time
    c.execute("INSERT INTO predictions VALUES ('KLAX','2026-10-04',?, 'intraday', NULL, NULL, ?, 'low')",
              (T, json.dumps({"60": 1.0})))
    c.commit()
    c.close()
    bot = HighcastBot(str(db))

    highs = lax_buckets()
    pmf = bot.pmf_for_event(highs, Snapshot(1, T, tuple(highs)))   # metric filter: low row ignored
    t69, mid, t70 = (m.ticker for m in highs)
    assert pmf[t69] == pytest.approx(0.4)
    assert pmf[mid] == pytest.approx(0.5)
    assert pmf[t70] == pytest.approx(0.1)

    lows = three_buckets(series="KXLOWTLAX", station="KLAX", event="KXLOWTLAX-26OCT04", metric="low")
    pmf = bot.pmf_for_event(lows, Snapshot(1, T, tuple(lows)))     # issued_at == T is included (<=)
    l69, lmid, l70 = (m.ticker for m in lows)
    assert pmf[l69] == pytest.approx(1.0)
    assert pmf[lmid] == pytest.approx(0.0)
    assert pmf[l70] == pytest.approx(0.0)


def _insert(db, kind, issued, buckets, metric="high"):
    c = sqlite3.connect(db)
    try:
        c.execute("INSERT INTO predictions VALUES ('KLAX','2026-10-04',?,?,NULL,NULL,?,?)",
                  (issued, kind, json.dumps(buckets), metric))
        c.commit()
    finally:
        c.close()


def _empty_highcast_db(path):
    c = sqlite3.connect(path)
    try:
        c.execute("""CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT,
                     kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, metric TEXT)""")
        c.commit()
    finally:
        c.close()


# ---- F5: only the dayahead and intraday kinds are read ---------------------------

def test_a_newer_evening_row_is_ignored_for_the_dayahead_row(tmp_path):
    db = tmp_path / "highcast.db"
    _empty_highcast_db(db)
    _insert(db, "dayahead", "2026-10-04T09:00:00+00:00", {"69": 1.0})
    _insert(db, "evening", "2026-10-04T10:30:00+00:00", {"71": 1.0})
    markets = lax_buckets()
    pmf = HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    t69, mid, t70 = (m.ticker for m in markets)
    assert pmf[mid] == pytest.approx(1.0) and pmf[t70] == pytest.approx(0.0)


def test_only_an_evening_row_gives_no_pmf_and_no_candidates(tmp_path):
    db = tmp_path / "highcast.db"
    _empty_highcast_db(db)
    _insert(db, "evening", "2026-10-04T10:30:00+00:00", {"71": 1.0})
    markets = lax_buckets()
    s = Snapshot(1, T, tuple(markets))
    assert HighcastBot(str(db)).pmf_for_event(markets, s) is None
    assert HighcastBot(str(db)).decide(s).candidates == []


# ---- F2 exemption + F4: same-day events, decision time vs snapshot time ----------

def test_highcast_still_evaluates_a_same_day_event(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    markets = lax_buckets()                    # KLAX climate day 2026-10-04 began 08:00Z
    d = HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)), T)
    assert len(d.candidates) == 3


def test_a_prediction_issued_after_the_snapshot_but_before_the_decision_is_ignored(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)                       # rows at 10:00 and 12:00; snapshot at 11:00
    markets = lax_buckets()
    decided = "2026-10-04T12:05:00+00:00"
    d = HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)), decided)
    by = {c.ticker: c for c in d.candidates}
    t69, mid, t70 = (m.ticker for m in markets)
    assert by[t70].p_bot == pytest.approx(0.1)        # the 12:00 row ({"71": 1.0}) was not used
    assert {c.decided_at for c in d.candidates} == {decided}


# ---- F16: the forecast behind the pmf is logged ----------------------------------

def test_highcast_candidates_record_the_prediction(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    markets = lax_buckets()
    d = HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)))
    assert {c.forecast_json for c in d.candidates} == {json.dumps(
        {"issued_at_utc": "2026-10-04T10:00:00+00:00", "kind": "intraday", "source": "highcast"},
        sort_keys=True)}


@pytest.mark.parametrize('metric,now,expected', [
    ('high', '2026-10-04T16:00:00+00:00', 'dayahead'),
    ('high', '2026-10-05T00:00:00+00:00', 'intraday'),
    ('low', '2026-10-04T16:00:00+00:00', 'intraday'),
])
def test_desk_kind_uses_decision_clock_not_snapshot(tmp_path, metric, now, expected):
    db = tmp_path / 'highcast.db'
    _empty_highcast_db(db)
    _insert(db, 'dayahead', '2026-10-04T09:00:00+00:00', {'69': 1.0}, metric)
    _insert(db, 'intraday', '2026-10-04T10:00:00+00:00', {'71': 1.0}, metric)
    markets = three_buckets(station='KLAX', metric=metric)
    decision = HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)), now)
    assert {json.loads(c.forecast_json)['kind'] for c in decision.candidates} == {expected}


def test_future_preferred_kind_falls_back_without_lookahead(tmp_path):
    db = tmp_path / 'highcast.db'
    _empty_highcast_db(db)
    _insert(db, 'intraday', '2026-10-04T10:00:00+00:00', {'69': 1.0})
    _insert(db, 'dayahead', '2026-10-04T12:00:00+00:00', {'71': 1.0})
    markets = lax_buckets()
    d = HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)),
                                  '2026-10-04T16:00:00+00:00')
    assert {json.loads(c.forecast_json)['kind'] for c in d.candidates} == {'intraday'}


def test_preday_evening_only_is_excluded(tmp_path):
    db = tmp_path / 'highcast.db'
    _empty_highcast_db(db)
    _insert(db, 'evening', '2026-10-03T10:00:00+00:00', {'71': 1.0})
    markets = lax_buckets()
    stamp = '2026-10-03T23:00:00+00:00'
    assert HighcastBot(str(db)).decide(Snapshot(1, stamp, tuple(markets)), stamp).candidates == []
