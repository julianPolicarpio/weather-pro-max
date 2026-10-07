from contextlib import closing

from citycast import db

TABLES = {"stations", "cli_daily", "mos_runs", "obs", "kalshi_markets",
          "kalshi_candles", "backfill_log", "meta"}


def test_connect_creates_schema_wal_and_timeout(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert TABLES <= names
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_obs_columns_match_highcast_obs(tmp_path):
    """latentmax.visible_running_extremum must run unchanged against history.db."""
    from highcast import db as hdb
    with closing(db.connect(tmp_path / "h.db")) as c, \
            closing(hdb.connect(tmp_path / "hc.db")) as hc:
        cols = lambda conn: {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
        assert cols(c) == cols(hc)


def test_upsert_replaces_on_primary_key(tmp_path):
    row = {"station": "KNYC", "date": "2024-01-01", "high_f": 40, "low_f": 30, "final": 1,
           "issued_utc": None, "raw": "{}", "fetched_at": "2026-10-07T00:00:00+00:00"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            assert db.upsert(c, "cli_daily", [row]) == 1
            db.upsert(c, "cli_daily", [{**row, "high_f": 41}])
        assert [tuple(r) for r in c.execute("SELECT high_f FROM cli_daily")] == [(41,)]
        assert db.upsert(c, "cli_daily", []) == 0


def test_cli_init_creates_db(tmp_path, capsys):
    from citycast.cli import main
    p = tmp_path / "sub" / "h.db"
    assert main(["--db", str(p), "init"]) == 0
    assert p.exists() and '"status": "ok"' in capsys.readouterr().out
