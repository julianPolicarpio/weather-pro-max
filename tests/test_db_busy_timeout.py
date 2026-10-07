"""A writer must wait out a long reader in another connection, not fail.

WHY, from data/launchd.err and data/launchd-kmia.err (2026-10-03/04): two
5-minute wake jobs and the dashboard share data/highcast.db in rollback-journal
mode. A commit needs every other connection's SHARED lock released, and the
dashboard alone holds reads for up to ~2.5 s per request (/api/demo-account)
while a wake job's feature reads run concurrently. With busy_timeout=4000 a
commit gave up after 4 s ("database is locked"), and while it waited every new
read in the other processes failed the same way. Reproduced 2026-10-04: a
6-second reader in another process makes a 4000 ms writer fail at 4.2 s and a
30000 ms writer succeed.

The reader here holds its lock for 5 s, longer than the old 4 s limit.
"""
import sqlite3
import threading
import time

from highcast import db


def test_writer_waits_out_a_long_reader(tmp_path):
    path = tmp_path / "highcast.db"
    writer = db.connect(path)
    holding = threading.Event()

    def long_read():
        reader = sqlite3.connect(path, isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM obs").fetchone()   # takes SHARED
        holding.set()
        time.sleep(5)
        reader.execute("COMMIT")
        reader.close()

    t = threading.Thread(target=long_read)
    t.start()
    holding.wait()
    try:
        db.set_meta(writer, "busy-timeout-test", "1")   # must not raise "database is locked"
    finally:
        t.join()
    assert db.get_meta(writer, "busy-timeout-test") == "1"
    writer.close()


def test_busy_timeout_is_thirty_seconds(tmp_path):
    conn = db.connect(tmp_path / "highcast.db", migrate=False)
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS == 30_000
    conn.close()
