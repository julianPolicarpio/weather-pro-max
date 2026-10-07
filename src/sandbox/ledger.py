"""Paper accounts. Cash is always derived from orders, never stored."""
import json
import sqlite3

from sandbox.book import NO_FILL, Fill
from sandbox.models import Intent

START_CASH = 5.0


def open_account(conn: sqlite3.Connection, name: str, kind: str, label: str = "") -> None:
    conn.execute("INSERT OR IGNORE INTO accounts(name, kind, start_cash, label) "
                 "VALUES (?, ?, ?, ?)", (name, kind, START_CASH, label))
    conn.commit()


def cash(conn: sqlite3.Connection, account: str) -> float:
    start = conn.execute("SELECT start_cash FROM accounts WHERE name=?", (account,)).fetchone()
    if start is None:
        raise KeyError(account)
    spent, paid = conn.execute(
        "SELECT COALESCE(SUM(cost + fee), 0), COALESCE(SUM(payout), 0) "
        "FROM orders WHERE account=?", (account,)).fetchone()
    return round(start["start_cash"] - spent + paid, 4)


def holds(conn: sqlite3.Connection, account: str, ticker: str) -> bool:
    return conn.execute("SELECT 1 FROM orders WHERE account=? AND ticker=? "
                        "AND filled_contracts > 0 LIMIT 1", (account, ticker)).fetchone() is not None


def place(conn: sqlite3.Connection, intent: Intent, fill: Fill, decided_at: str,
          book_json: str | None) -> tuple[int, str]:
    status, f = fill.status, fill
    if f.contracts > 0 and f.cost + f.fee > cash(conn, intent.account) + 1e-9:
        status, f = "insufficient_cash", NO_FILL
    cur = conn.execute(
        """INSERT INTO orders(account, ticker, side, contracts_requested, price_cap,
                              snapshot_id, decided_at, status, filled_contracts, cost, fee,
                              levels_json, book_json, p_bot, p_market)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (intent.account, intent.ticker, intent.side, intent.contracts, intent.price_cap,
         intent.snapshot_id, decided_at, status, f.contracts, f.cost, f.fee,
         json.dumps(f.levels), book_json, intent.p_bot, intent.p_market))
    conn.commit()
    return cur.lastrowid, status
