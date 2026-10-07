"""sandbox.db schema. Own file; never the highcast database."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS markets (
  ticker TEXT PRIMARY KEY, series TEXT NOT NULL, event_ticker TEXT NOT NULL,
  event_date TEXT NOT NULL, lo INTEGER, hi INTEGER, label TEXT NOT NULL,
  metric TEXT, station TEXT, close_time TEXT, result TEXT, settled_at TEXT);
CREATE INDEX IF NOT EXISTS markets_event ON markets(event_ticker);
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT, taken_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS snapshots_taken ON snapshots(taken_at);
CREATE TABLE IF NOT EXISTS quotes (
  snapshot_id INTEGER NOT NULL, ticker TEXT NOT NULL,
  yes_bid REAL, yes_ask REAL, yes_bid_size REAL, yes_ask_size REAL,
  PRIMARY KEY (snapshot_id, ticker));
CREATE TABLE IF NOT EXISTS accounts (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL, start_cash REAL NOT NULL,
  label TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, ticker TEXT NOT NULL,
  side TEXT NOT NULL, contracts_requested INTEGER NOT NULL, price_cap REAL NOT NULL,
  snapshot_id INTEGER, decided_at TEXT NOT NULL, status TEXT NOT NULL,
  filled_contracts INTEGER NOT NULL, cost REAL NOT NULL, fee REAL NOT NULL,
  levels_json TEXT NOT NULL, book_json TEXT, p_bot REAL, p_market REAL, payout REAL);
CREATE INDEX IF NOT EXISTS orders_account ON orders(account);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bot TEXT NOT NULL, ticker TEXT NOT NULL,
  series TEXT NOT NULL, station TEXT, metric TEXT, event_date TEXT NOT NULL,
  snapshot_id INTEGER NOT NULL, decided_at TEXT NOT NULL, p_bot REAL NOT NULL,
  pmf_json TEXT NOT NULL, p_market REAL, yes_bid REAL, yes_ask REAL, spread REAL,
  ask_size REAL, minutes_to_close REAL, fee_per_contract REAL, edge_after_fee REAL,
  gates_json TEXT, regime_json TEXT, placed INTEGER NOT NULL, reason TEXT NOT NULL,
  result TEXT, asif_profit REAL, forecast_json TEXT);
CREATE INDEX IF NOT EXISTS candidates_ticker ON candidates(ticker);
"""


# Columns added after the first live database was created: (table, column, type).
_ADDED_COLUMNS = (("candidates", "forecast_json", "TEXT"),)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, kind in _ADDED_COLUMNS:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    # timeout=30: the cycle, the dashboard and `report` share this file; wait out a
    # writer rather than failing at sqlite's 5 s default.
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing DB for displays without schema, journal or account writes."""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


NOT_INITIALISED = "sandbox DB not initialised — run `sandbox init` first"


class NotInitialised(Exception):
    """A read-only surface met a missing file, a missing schema or missing accounts."""


def require_initialised(conn: sqlite3.Connection, accounts: tuple[str, ...] = ()) -> None:
    """Raise NotInitialised unless the schema exists and every named account is open.
    Reads only: a display must explain an empty database, never create one."""
    try:
        have = {r[0] for r in conn.execute("SELECT name FROM accounts")}
    except sqlite3.OperationalError as e:
        if "no such table" not in str(e):   # locked, I/O, corrupt: a real error, not "run init"
            raise
        raise NotInitialised(NOT_INITIALISED) from e   # a blank or foreign file
    if not set(accounts) <= have:
        raise NotInitialised(NOT_INITIALISED)
