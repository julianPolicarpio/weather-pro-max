"""data/history.db — the multi-city archive behind citycast (spec 2026-10-07 §2.3)."""
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path("data/history.db")
BUSY_TIMEOUT_MS = 30_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stations (icao TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cli_daily (station TEXT, date TEXT, high_f INTEGER, low_f INTEGER,
  final INTEGER NOT NULL, issued_utc TEXT, raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, date));
CREATE TABLE IF NOT EXISTS mos_runs (station TEXT, model TEXT, runtime_utc TEXT, target_date TEXT,
  metric TEXT, value_f REAL, n_points INTEGER NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, model, runtime_utc, target_date, metric));
CREATE TABLE IF NOT EXISTS obs (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, source TEXT NOT NULL, tmpf REAL, tmpf_precise REAL,
  dwpf REAL, sknt REAL, drct REAL, skyc TEXT, six_max_c REAL, raw TEXT, fetched_at TEXT NOT NULL,
  six_min_c REAL, slp_hpa REAL, pres_tend_hpa REAL, pres_tend_code INTEGER, vis_sm REAL,
  ceiling_ft INTEGER, sky_lowest_ft INTEGER, n_sky_layers INTEGER, sky_cb INTEGER, wx_present TEXT,
  wx_thunder INTEGER, PRIMARY KEY (station, valid_utc, source));
CREATE INDEX IF NOT EXISTS idx_obs_valid ON obs (station, valid_utc);
CREATE TABLE IF NOT EXISTS kalshi_markets (ticker TEXT PRIMARY KEY, series TEXT NOT NULL,
  event_ticker TEXT NOT NULL, station TEXT NOT NULL, metric TEXT NOT NULL, event_date TEXT NOT NULL,
  lo INTEGER, hi INTEGER, open_time TEXT, close_time TEXT NOT NULL, result TEXT,
  expiration_value TEXT, raw TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_km_event ON kalshi_markets (station, metric, event_date);
CREATE TABLE IF NOT EXISTS kalshi_candles (ticker TEXT, end_utc TEXT, yes_bid REAL, yes_ask REAL,
  volume REAL, open_interest REAL, PRIMARY KEY (ticker, end_utc));
CREATE TABLE IF NOT EXISTS backfill_log (source TEXT, key TEXT, done_at TEXT NOT NULL,
  rows INTEGER NOT NULL, PRIMARY KEY (source, key));
CREATE TABLE IF NOT EXISTS wf_predictions (station TEXT, metric TEXT, event_date TEXT, asof_utc TEXT,
  variant TEXT, fold TEXT, anchor REAL, run_ext REAL, quantiles_json TEXT NOT NULL,
  pmf_json TEXT NOT NULL, k REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (station, metric, event_date, asof_utc, variant));
CREATE TABLE IF NOT EXISTS model_choice (station TEXT, metric TEXT, variant TEXT NOT NULL,
  crps_sel REAL, mos_crps_sel REAL, gate_ci_hi REAL, sel_pass INTEGER NOT NULL,
  crps_rep REAL, mos_crps_rep REAL, rep_days INTEGER NOT NULL, gate_pass INTEGER NOT NULL,
  decided_at TEXT NOT NULL, PRIMARY KEY (station, metric));
CREATE TABLE IF NOT EXISTS eval_log (run_at TEXT, metric TEXT, fingerprint TEXT NOT NULL,
  report_scored INTEGER NOT NULL, PRIMARY KEY (run_at, metric));
CREATE TABLE IF NOT EXISTS live_predictions (station TEXT, metric TEXT, event_date TEXT,
  issued_utc TEXT, variant TEXT NOT NULL, anchor REAL, run_ext REAL, quantiles_json TEXT NOT NULL,
  pmf_json TEXT NOT NULL, k REAL NOT NULL, PRIMARY KEY (station, metric, event_date, issued_utc));
"""


def connect(path=DEFAULT_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE every row (columns from the first row). Never commits."""
    if not rows:
        return 0
    cols = list(rows[0])
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({','.join('?' * len(cols))})")
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)
