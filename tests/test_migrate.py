import sqlite3
from highcast import db, migrate

def test_new_db_has_metric_columns(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    fcols = {r[1] for r in conn.execute("PRAGMA table_info(forecasts)")}
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    assert "metric" in fcols and "metric" in pcols

def test_new_db_has_six_min_column(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
    assert "six_min_c" in cols


def test_six_min_column_added_to_pre_existing_obs_table(tmp_path):
    """An obs table created before six_min_c existed must gain the column on
    connect, without losing its rows. Re-running connect must be a no-op."""
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE obs (station TEXT NOT NULL, valid_utc TEXT NOT NULL, "
              "source TEXT NOT NULL, tmpf REAL, tmpf_precise REAL, dwpf REAL, "
              "sknt REAL, drct REAL, skyc TEXT, six_max_c REAL, raw TEXT, "
              "fetched_at TEXT NOT NULL, PRIMARY KEY (station, valid_utc, source))")
    c.execute("INSERT INTO obs (station, valid_utc, source, six_max_c, fetched_at) "
              "VALUES ('KLAX','2026-07-01T17:53:00+00:00','metar',24.4,'x')")
    c.commit(); c.close()
    conn = db.connect(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
    assert "six_min_c" in cols
    row = conn.execute("SELECT six_max_c, six_min_c FROM obs").fetchone()
    assert row["six_max_c"] == 24.4 and row["six_min_c"] is None
    conn.close()
    migrate.ensure_six_min_column(db.connect(p))   # safe to re-run


def test_migration_backfills_existing_rows_to_high(tmp_path):
    # simulate an OLD db without the metric column
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE forecasts (station TEXT, target_date TEXT, model TEXT, run_date TEXT, max_temp_f REAL, fetched_at TEXT, PRIMARY KEY(station,target_date,model,run_date))")
    c.execute("INSERT INTO forecasts VALUES ('KLAX','2026-07-01','gfs','2026-06-30',84.0,'x')")
    c.execute("CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT, kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, PRIMARY KEY(station,target_date,issued_at_utc,kind))")
    c.execute("INSERT INTO predictions VALUES ('KLAX','2026-07-01','2026-07-01T00:00:00+00:00','dayahead',84.0,'{}','{}')")
    c.commit(); c.close()
    conn = db.connect(p)   # connect must migrate
    assert conn.execute("SELECT metric FROM forecasts").fetchone()[0] == "high"
    assert conn.execute("SELECT metric FROM predictions").fetchone()[0] == "high"


def test_new_db_has_bot_orders_edge_column(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bot_orders)")}
    assert "edge" in cols


def test_bot_orders_edge_column_added_to_pre_existing_table(tmp_path):
    """A `bot_orders` table created before the `edge` column existed must gain
    it on connect, without losing its rows. Re-running the migration must be a
    no-op."""
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE bot_orders (station TEXT NOT NULL, metric TEXT NOT NULL, "
              "target_date TEXT NOT NULL, decided_utc TEXT NOT NULL, local_hour INTEGER, "
              "place INTEGER NOT NULL, reason TEXT, bucket TEXT, limit_price REAL, "
              "contracts INTEGER, calibrated REAL, ask REAL, dry_run INTEGER NOT NULL, "
              "env TEXT, placed INTEGER, response TEXT, "
              "PRIMARY KEY (station, metric, target_date))")
    c.execute("INSERT INTO bot_orders (station, metric, target_date, decided_utc, "
              "place, calibrated, ask, dry_run) VALUES "
              "('KMIA','high','2026-08-01','2026-08-01T18:00:00+00:00',0,0.667,0.97,1)")
    c.commit(); c.close()
    conn = db.connect(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bot_orders)")}
    assert "edge" in cols
    row = conn.execute("SELECT calibrated, ask, edge FROM bot_orders").fetchone()
    assert row["calibrated"] == 0.667 and row["ask"] == 0.97 and row["edge"] is None
    conn.close()
    migrate.ensure_bot_orders_edge_column(db.connect(p))   # safe to re-run


def test_pk_rebuilt_to_include_metric_so_high_low_coexist(tmp_path):
    """A pre-metric DB (PK without metric) must be rebuilt so a low forecast
    can't overwrite the high one sharing (station,target,model,run)."""
    import sqlite3
    from highcast import db
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE forecasts (station TEXT, target_date TEXT, model TEXT, "
              "run_date TEXT, max_temp_f REAL, fetched_at TEXT, "
              "PRIMARY KEY(station,target_date,model,run_date))")
    c.execute("INSERT INTO forecasts VALUES ('KLAX','2026-07-01','ecmwf','2026-06-30',84.0,'t')")
    c.commit(); c.close()
    conn = db.connect(p)                       # connect migrates + rebuilds PK
    # same (station,target,model,run) but metric='low' must NOT clobber the high
    db.upsert(conn, "forecasts", [{"station": "KLAX", "target_date": "2026-07-01",
              "model": "ecmwf", "run_date": "2026-06-30", "max_temp_f": 61.0,
              "fetched_at": "t", "metric": "low"}])
    rows = {r["metric"]: r["max_temp_f"] for r in conn.execute(
        "SELECT metric, max_temp_f FROM forecasts WHERE model='ecmwf'")}
    assert rows == {"high": 84.0, "low": 61.0}   # both survive


def test_new_db_has_paper_account_txns_table(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_account_txns)")}
    assert {"mode", "station", "metric", "target_date", "txn_type", "txn_utc",
           "bucket", "price", "contracts", "fee", "cash_delta",
           "balance_after", "rule", "won", "reason"} <= cols


def test_ensure_paper_account_tables_is_safe_to_rerun(tmp_path):
    p = tmp_path / "t.db"
    conn = db.connect(p)
    conn.execute(
        """INSERT INTO paper_account_txns
             (mode, station, metric, target_date, txn_type, txn_utc, price,
              contracts, balance_after)
           VALUES ('unit_replay','KLAX','high','2026-08-01','OPEN','t',0.5,1,5.0)""")
    conn.commit()
    migrate.ensure_paper_account_tables(conn)          # must not touch data
    migrate.ensure_paper_account_tables(db.connect(p))  # or error on re-run
    n = conn.execute("SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"]
    assert n == 1


def test_ensure_paper_account_tables_never_touches_app_bets(tmp_path):
    """This migration must be additive only -- `app_bets` is this project's
    only evidence and must stay byte-identical across any schema change."""
    conn = db.connect(tmp_path / "t.db")
    db.upsert(conn, "app_bets", [{
        "station": "KLAX", "metric": "high", "target_date": "2026-08-01",
        "placed_utc": "t", "placed_manila": "00:00", "placed_local_hour": 0,
        "bucket": "80-81", "p_model": 0.5, "price": 0.5, "stake": 1.0,
        "payout_if_win": 2.0, "actual": None, "won": None, "settled_utc": None,
        "reason": None, "notified_utc": None}])
    before = [dict(r) for r in conn.execute("SELECT * FROM app_bets")]
    migrate.ensure_paper_account_tables(conn)
    after = [dict(r) for r in conn.execute("SELECT * FROM app_bets")]
    assert before == after
