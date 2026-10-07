"""THE STANDING CLAIM: this project has never placed an order.

⚠️ `bot_orders` LOGS DECISIONS, NOT EXECUTIONS. A verification sweep on
2026-08-07 asserted `COUNT(*) == 0` and failed — on a row that was the bot
REFUSING to trade:

    2026-08-01 KMIA high  place=0 placed=0 dry_run=1 contracts=0
    reason: "calibrated 66.7% below the measured floor 70%"

The claim is about `placed`, and reading `COUNT(*)` as "orders placed" gets it
exactly backwards: a healthy, well-behaved bot writes MORE rows here, not
fewer, because every refusal is recorded. A future reader checking "have we
traded?" must look at `placed`.
"""
import sqlite3

from highcast import db as hdb


def _conn(tmp_path):
    return hdb.connect(tmp_path / "t.db")


def test_a_refusal_row_is_not_an_order(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        """INSERT INTO bot_orders (station, metric, target_date, decided_utc,
                                   local_hour, place, reason, bucket, contracts,
                                   dry_run, env, placed)
           VALUES ('KMIA','high','2026-08-01','2026-08-01T18:02:13+00:00',14,
                   0,'calibrated 66.7% below the measured floor 70%','94–95',
                   0,1,'production',0)""")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM bot_orders").fetchone()["c"] == 1
    # ...and yet nothing was placed. THIS is the query that answers the claim.
    assert conn.execute(
        "SELECT COALESCE(SUM(placed),0) s FROM bot_orders").fetchone()["s"] == 0


def test_the_live_database_has_placed_nothing():
    """Runs against the REAL database when present, because the claim is about
    the real account. Skipped on a machine without one."""
    import pathlib
    p = pathlib.Path("data/highcast.db")
    if not p.exists():
        import pytest
        pytest.skip("no live database on this machine")
    conn = hdb.connect(p, migrate=False)
    placed = conn.execute(
        "SELECT COALESCE(SUM(placed),0) s FROM bot_orders").fetchone()["s"]
    assert placed == 0, f"{placed} order(s) have been placed"
    live = conn.execute(
        "SELECT COUNT(*) c FROM bot_orders WHERE dry_run = 0").fetchone()["c"]
    assert live == 0, f"{live} non-dry-run decision(s) recorded"
