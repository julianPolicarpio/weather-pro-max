import sqlite3
from contextlib import closing

import pytest
from sandbox_helpers import three_buckets

from sandbox.db import connect, get_meta, set_meta
from sandbox.models import Candidate
from sandbox.recorder import (latest_snapshot_id, load_snapshot, record_candidates,
                              record_snapshot)

T1 = "2026-10-04T10:00:00+00:00"
T2 = "2026-10-04T10:15:00+00:00"
BETWEEN = "2026-10-04T10:07:00+00:00"


@pytest.fixture
def conn():
    with closing(connect(":memory:")) as c:
        yield c


def test_snapshot_roundtrip(conn):
    sid = record_snapshot(conn, three_buckets(), T1)
    snap = load_snapshot(conn, sid)
    assert snap.taken_at == T1
    assert [m.label for m in snap.markets] == ["≤68", "69–70", "≥71"]
    assert snap.markets[0].station == "KNYC"
    assert list(snap.events()) == ["KXHIGHNY-26OCT04"]


def test_latest_snapshot_never_looks_ahead(conn):
    s1 = record_snapshot(conn, three_buckets(), T1)
    s2 = record_snapshot(conn, three_buckets(asks=(0.5, 0.5, 0.5)), T2)
    assert latest_snapshot_id(conn, BETWEEN) == s1
    assert latest_snapshot_id(conn, T2) == s2
    assert latest_snapshot_id(conn, "2026-10-04T09:00:00+00:00") is None


def test_quotes_are_per_snapshot(conn):
    s1 = record_snapshot(conn, three_buckets(), T1)
    s2 = record_snapshot(conn, three_buckets(asks=(0.5, 0.5, 0.5)), T2)
    assert load_snapshot(conn, s1).markets[0].yes_ask == 0.20
    assert load_snapshot(conn, s2).markets[0].yes_ask == 0.5


def test_meta_roundtrip(conn):
    assert get_meta(conn, "k") is None
    set_meta(conn, "k", "v")
    assert get_meta(conn, "k") == "v"


def test_record_candidates_stores_rows(conn):
    sid = record_snapshot(conn, three_buckets(), T1)
    c = Candidate(bot="mos", ticker="KXHIGHNY-26OCT04-T69", series="KXHIGHNY",
                  station="KNYC", metric="high", event_date="2026-10-04",
                  snapshot_id=sid, decided_at=T1, p_bot=0.37, pmf_json="{}",
                  p_market=0.2, yes_bid=0.15, yes_ask=0.20, spread=0.05,
                  ask_size=3.0, minutes_to_close=1140.0, fee_per_contract=0.02,
                  edge_after_fee=0.15, placed=True, reason="edge")
    record_candidates(conn, [c])
    row = conn.execute("SELECT bot, placed, reason, result FROM candidates").fetchone()
    assert tuple(row) == ("mos", 1, "edge", None)


def test_failed_snapshot_write_leaves_nothing_behind(conn):
    ms = three_buckets()
    with pytest.raises(sqlite3.IntegrityError):
        record_snapshot(conn, ms + [ms[0]], T1)      # duplicate ticker -> quotes PK
    set_meta(conn, "k", "v")                          # an unrelated commit
    assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 0


# ---- F16: candidates.forecast_json, also added to a database made before it ------

def test_record_candidates_stores_the_forecast():
    with closing(connect(":memory:")) as conn:
        sid = record_snapshot(conn, three_buckets(), T1)
        c = Candidate(bot="mos", ticker="KXHIGHNY-26OCT04-T69", series="KXHIGHNY",
                      station="KNYC", metric="high", event_date="2026-10-04",
                      snapshot_id=sid, decided_at=T1, p_bot=0.37, pmf_json="{}",
                      p_market=0.2, yes_bid=0.15, yes_ask=0.20, spread=0.05,
                      ask_size=3.0, minutes_to_close=1140.0, fee_per_contract=0.02,
                      edge_after_fee=0.15, placed=True, reason="edge",
                      forecast_json='{"mu": 69.5}')
        record_candidates(conn, [c])
        assert conn.execute("SELECT forecast_json FROM candidates").fetchone()[0] == '{"mu": 69.5}'


def test_connect_adds_forecast_json_to_an_older_database(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    try:
        old.execute("""CREATE TABLE candidates (
          id INTEGER PRIMARY KEY AUTOINCREMENT, bot TEXT NOT NULL, ticker TEXT NOT NULL,
          series TEXT NOT NULL, station TEXT, metric TEXT, event_date TEXT NOT NULL,
          snapshot_id INTEGER NOT NULL, decided_at TEXT NOT NULL, p_bot REAL NOT NULL,
          pmf_json TEXT NOT NULL, p_market REAL, yes_bid REAL, yes_ask REAL, spread REAL,
          ask_size REAL, minutes_to_close REAL, fee_per_contract REAL, edge_after_fee REAL,
          gates_json TEXT, regime_json TEXT, placed INTEGER NOT NULL, reason TEXT NOT NULL,
          result TEXT, asif_profit REAL)""")
        old.commit()
    finally:
        old.close()
    with closing(connect(db)) as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(candidates)")]
    assert "forecast_json" in cols
    with closing(connect(db)) as conn:       # idempotent on the next open
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(candidates)")]
    assert cols.count("forecast_json") == 1


# ---- M1: sandbox connections wait out a writer instead of failing at 5 s --------

def test_connect_uses_a_30_second_busy_timeout(monkeypatch):
    real = sqlite3.connect
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("sandbox.db.sqlite3.connect", spy)
    with closing(connect(":memory:")):
        pass
    assert seen and seen[0].get("timeout") == 30
