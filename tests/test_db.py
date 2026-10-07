import json

from highcast import db

def test_schema_and_upsert(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rows = [
        {"station": "KLAX", "date": "2026-07-16", "high_f": 84, "low_f": 64,
         "raw": "{}", "fetched_at": "2026-07-17T00:00:00+00:00"},
    ]
    assert db.upsert(conn, "cli_reports", rows) == 1
    # upsert is idempotent and updates in place
    rows[0]["high_f"] = 85
    db.upsert(conn, "cli_reports", rows)
    got = conn.execute("SELECT high_f FROM cli_reports").fetchall()
    assert len(got) == 1 and got[0]["high_f"] == 85

def test_meta_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    assert db.get_meta(conn, "grid_url") is None
    db.set_meta(conn, "grid_url", "https://x")
    assert db.get_meta(conn, "grid_url") == "https://x"

def test_notes_insert_select_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    conn.execute(
        """INSERT INTO notes (station, target_date, created_utc, sky, context)
           VALUES (?, ?, ?, ?, ?)""",
        ("KLAX", "2026-07-18", "2026-07-18T20:00:00+00:00",
         "Marine layer never cleared — grey till 3pm",
         json.dumps({"phase": "peak_heating"})))
    conn.commit()
    row = conn.execute("SELECT * FROM notes WHERE station='KLAX'").fetchone()
    assert row["target_date"] == "2026-07-18"
    assert row["created_utc"] == "2026-07-18T20:00:00+00:00"
    assert row["sky"] == "Marine layer never cleared — grey till 3pm"
    assert json.loads(row["context"])["phase"] == "peak_heating"

def test_busy_timeout_pragma_set(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    assert row[0] == db.BUSY_TIMEOUT_MS
