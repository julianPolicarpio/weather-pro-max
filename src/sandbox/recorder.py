"""Append-only snapshots and the candidate log."""
import sqlite3
from dataclasses import astuple, fields

from sandbox.models import Candidate, Market, Snapshot


def upsert_markets(conn: sqlite3.Connection, markets: list[Market]) -> None:
    conn.executemany(
        """INSERT INTO markets(ticker, series, event_ticker, event_date, lo, hi, label,
                               metric, station, close_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET close_time=excluded.close_time""",
        [(m.ticker, m.series, m.event_ticker, m.event_date, m.lo, m.hi, m.label,
          m.metric, m.station, m.close_time) for m in markets])


def record_snapshot(conn: sqlite3.Connection, markets: list[Market], taken_at: str) -> int:
    with conn:  # one transaction: a failed quote insert must not leave a snapshot behind
        upsert_markets(conn, markets)
        sid = conn.execute("INSERT INTO snapshots(taken_at) VALUES (?)", (taken_at,)).lastrowid
        conn.executemany(
            """INSERT INTO quotes(snapshot_id, ticker, yes_bid, yes_ask, yes_bid_size, yes_ask_size)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(sid, m.ticker, m.yes_bid, m.yes_ask, m.yes_bid_size, m.yes_ask_size)
             for m in markets])
    return sid


def latest_snapshot_id(conn: sqlite3.Connection, at_or_before: str) -> int | None:
    """Newest snapshot taken at or before the given time. Never one after it."""
    row = conn.execute("SELECT id FROM snapshots WHERE taken_at <= ? "
                       "ORDER BY taken_at DESC, id DESC LIMIT 1", (at_or_before,)).fetchone()
    return None if row is None else row["id"]


def row_to_market(r) -> Market:
    return Market(ticker=r["ticker"], series=r["series"], event_ticker=r["event_ticker"],
                  event_date=r["event_date"], lo=r["lo"], hi=r["hi"], label=r["label"],
                  metric=r["metric"], station=r["station"], yes_bid=r["yes_bid"],
                  yes_ask=r["yes_ask"], yes_bid_size=r["yes_bid_size"],
                  yes_ask_size=r["yes_ask_size"], close_time=r["close_time"])


def load_snapshot(conn: sqlite3.Connection, snapshot_id: int) -> Snapshot:
    taken = conn.execute("SELECT taken_at FROM snapshots WHERE id=?",
                         (snapshot_id,)).fetchone()["taken_at"]
    rows = conn.execute(
        """SELECT m.*, q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size
           FROM quotes q JOIN markets m ON m.ticker = q.ticker
           WHERE q.snapshot_id = ?
           ORDER BY m.event_ticker, COALESCE(m.hi, 1000)""", (snapshot_id,)).fetchall()
    return Snapshot(id=snapshot_id, taken_at=taken,
                    markets=tuple(row_to_market(r) for r in rows))


_CANDIDATE_COLS = [f.name for f in fields(Candidate)]


def record_candidates(conn: sqlite3.Connection, candidates: list[Candidate]) -> None:
    if not candidates:
        return
    cols = ", ".join(_CANDIDATE_COLS)
    marks = ", ".join("?" for _ in _CANDIDATE_COLS)
    with conn:
        conn.executemany(f"INSERT INTO candidates({cols}) VALUES ({marks})",
                         [astuple(c) for c in candidates])
