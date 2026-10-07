"""A READ must not run DDL. `db.connect(..., migrate=False)`.

THE BUG THIS FIXES, from `data/serve.err`:

    File "highcast/web/app.py", line 407, in timing
        conn = db.connect(c.db_path)
    File "highcast/db.py", in connect
        conn.executescript(SCHEMA)
    sqlite3.OperationalError: database is locked

`connect()` runs `executescript(SCHEMA)` plus eight `ensure_*` migrations on
EVERY call, and the dashboard calls it per request — 16 call sites, on a
60-second poll. So a page load fires a burst of schema writes at a file the
5-minute wake cycle is also writing, and `PRAGMA busy_timeout` (then 4000 ms) only bought
four seconds before the read fails outright.

The dashboard has no business writing DDL at all. It is a reader; the wake
cycle and the CLI own the schema. `migrate=False` gives readers a connection
that opens and returns.

⚠️ THE DEFAULT STAYS TRUE. Every existing caller — wake cycle, CLI, tests,
migrations — keeps byte-identical behaviour, because those are the paths that
legitimately create and upgrade the schema. Only the web read path opts out,
and it opts out AFTER `create_app` has already opened one migrating connection
at startup, so the schema is guaranteed present before any request is served.
"""
import sqlite3

import pytest

from highcast import db


def test_migrate_false_creates_no_tables(tmp_path):
    """On a fresh file, a non-migrating connect must leave it empty rather
    than half-built — the honest failure is 'no such table', not a silently
    partial schema."""
    p = tmp_path / "fresh.db"
    conn = db.connect(p, migrate=False)
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables == [], tables


def test_migrate_false_still_reads_an_existing_db(tmp_path):
    """The actual use: schema already built by someone else, reader just reads."""
    p = tmp_path / "t.db"
    db.connect(p).close()                       # writer builds it
    conn = db.connect(p, migrate=False)         # reader opens it
    assert conn.execute("SELECT COUNT(*) c FROM obs").fetchone()["c"] == 0


def test_migrate_false_sets_the_busy_timeout(tmp_path):
    """Skipping DDL must not also skip the lock-wait — a reader still shares
    the file with the wake cycle's writes."""
    p = tmp_path / "t.db"
    db.connect(p).close()
    conn = db.connect(p, migrate=False)
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS


def test_a_reader_can_open_a_read_only_database_file(tmp_path):
    """A reader must work against a file it cannot write.

    ⚠️ NOTE ON WHAT THIS DOES *NOT* PROVE, because the first version of this
    test asserted it and was wrong: the MIGRATING connect also succeeds here.
    `SCHEMA` is all `CREATE TABLE IF NOT EXISTS` and every `ensure_*` checks
    before altering, so against a current schema the DDL is a no-op and
    SQLite never writes a page.

    That is precisely why the original bug was confusing. The lock in
    `serve.err` was NOT contention over a write — it was `executescript()`
    ending and reopening a transaction while the wake cycle held the file.
    A no-op schema script still has to take the lock to discover it is a
    no-op. Skipping the script is what removes the contention window;
    `test_migrate_false_creates_no_tables` is the proof that it is skipped.
    """
    p = tmp_path / "t.db"
    db.connect(p).close()
    p.chmod(0o444)
    try:
        conn = db.connect(p, migrate=False)
        assert conn.execute("SELECT COUNT(*) c FROM obs").fetchone()["c"] == 0
    finally:
        p.chmod(0o644)


def test_default_still_migrates(tmp_path):
    """The regression guard: every existing caller depends on connect()
    building the schema, so the default must not change."""
    p = tmp_path / "t.db"
    conn = db.connect(p)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "obs" in tables and "app_bets" in tables
    assert "paper_account_txns" in tables


def test_web_requests_do_not_migrate(tmp_path, monkeypatch):
    """End to end: serving a request must not run the migration suite.

    This is the assertion that would have caught the original bug — the schema
    calls were never wrong, they were just on a path that had no business
    making them."""
    from fastapi.testclient import TestClient
    from highcast.config import load_config
    from highcast.web.app import create_app

    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    db.connect(cfg.db_path).close()
    client = TestClient(create_app(cfg))       # startup may migrate: allowed

    from highcast import migrate as migrate_mod
    calls = []
    for name in ("ensure_metric_column", "ensure_six_min_column",
                 "ensure_paper_account_tables"):
        monkeypatch.setattr(migrate_mod, name,
                            lambda conn, _n=name: calls.append(_n))
    client.get("/api/clock")
    client.get("/api/history")
    assert calls == [], calls
