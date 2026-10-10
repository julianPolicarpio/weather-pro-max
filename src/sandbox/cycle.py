"""One scheduled run: discover -> snapshot; once per UTC hour also settle and decide.

Time comes from a `clock` read at each stage, never one value captured up front:
the snapshot is stamped when discovery STARTS (every quote in it was observed at or
after that, so its age is never understated), staleness is re-checked before each bot
and before each intent, and an order is stamped when it is filled.
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone

from highcast.fees import taker_fee
from sandbox.book import parse_book, walk
from sandbox.db import get_meta, set_meta
from sandbox.kalshi_public import fetch_book_payload, list_temperature_series, open_markets
from sandbox.ledger import cash, holds, open_account, place
from sandbox.markets import parse_series_payload
from sandbox.models import Intent, Market, Snapshot
from sandbox.recorder import (latest_snapshot_id, load_snapshot, record_candidates,
                              record_snapshot)
from sandbox.settle import settle_due
from sandbox.scoreboard import advance_confirmation

log = logging.getLogger(__name__)

STALE_AFTER_S = 600
ACCOUNTS: tuple[tuple[str, str, str], ...] = (
    ("favorite", "bot", "control: follow the crowd after fees"),
    ("mos", "bot", "uncalibrated: MOS with a fixed 3.0°F spread"),
    ("highcast", "bot", "KLAX/KMIA only"),
    ("citycast", "bot", "24 cities: citycast model, gated pairs only"),
    ("gate", "bot", "decision gate: winner of the held-out test (paper)"),
    ("manual", "manual", "Julian's hand bets"),
)
_FILLED = ("filled", "partial")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_accounts(conn: sqlite3.Connection) -> None:
    for name, kind, label in ACCOUNTS:
        open_account(conn, name, kind, label)


def discover(get) -> list[Market]:
    markets: list[Market] = []
    for s in list_temperature_series(get):
        markets.extend(parse_series_payload(s["ticker"], open_markets(get, s["ticker"])))
    return markets


def execute_intent(conn: sqlite3.Connection, get, intent: Intent, now_iso: str) -> str:
    try:
        payload = fetch_book_payload(get, intent.ticker)
    except Exception as e:  # noqa: BLE001 - an unreachable book is a no-fill
        log.warning("book fetch failed for %s: %r", intent.ticker, e)
        payload = {}
    fill = walk(parse_book(payload), intent.side, intent.contracts, intent.price_cap)
    book_json = json.dumps(payload.get("orderbook_fp")) if payload else None
    _, status = place(conn, intent, fill, now_iso, book_json)
    return status


def _age_s(taken_at: str, now_iso: str) -> float:
    return (datetime.fromisoformat(now_iso) - datetime.fromisoformat(taken_at)).total_seconds()


def _fresh(snapshot: Snapshot, now_iso: str) -> bool:
    return 0 <= _age_s(snapshot.taken_at, now_iso) <= STALE_AFTER_S


def run_bot(conn: sqlite3.Connection, get, bot, snapshot: Snapshot, clock) -> tuple[int, int, bool]:
    """Decide, then fund intents highest priority first. Returns (filled orders,
    candidates logged, went stale part-way)."""
    decision = bot.decide(snapshot, clock())
    by_ticker = {c.ticker: c for c in decision.candidates}
    intents = sorted(decision.intents, key=lambda i: (-i.priority, i.ticker))
    placed = 0
    stale = False
    executed: set[str] = set()
    try:
        for k, intent in enumerate(intents):
            cand = by_ticker.get(intent.ticker)
            t = clock()
            if not _fresh(snapshot, t):
                stale = True
                for later in intents[k:]:
                    c = by_ticker.get(later.ticker)
                    if c is not None:
                        c.placed, c.reason = False, "stale_snapshot"
                break
            if holds(conn, intent.account, intent.ticker):
                if cand:
                    cand.placed, cand.reason = False, "already_held"
                continue
            need = intent.contracts * intent.price_cap + taker_fee(intent.contracts, intent.price_cap)
            if cash(conn, intent.account) + 1e-9 < need:
                # no order row, no book fetch: a cheaper intent later in the list may still fit
                if cand:
                    cand.placed, cand.reason = False, "insufficient_cash"
                continue
            status = execute_intent(conn, get, intent, t)
            executed.add(intent.ticker)
            if status in _FILLED:
                placed += 1
            elif cand:
                cand.placed = False
                cand.reason = "insufficient_cash" if status == "insufficient_cash" else "no_fill"
    finally:  # the log of what the bot considered survives a failure part-way through
        for c in decision.candidates:
            if c.placed and c.ticker not in executed:
                c.placed, c.reason = False, "bot_error"   # never ran: must not read as a bet
        record_candidates(conn, decision.candidates)
    return placed, len(decision.candidates), stale


def decide_all(conn: sqlite3.Connection, get, bots, clock) -> dict:
    now = clock()
    stale = {"status": "stale_snapshot", "orders": 0, "candidates": 0, "bot_errors": [],
             "stale_bots": []}
    sid = latest_snapshot_id(conn, now)
    if sid is None:
        return stale
    snapshot = load_snapshot(conn, sid)
    if not _fresh(snapshot, now):
        return stale
    orders = candidates = 0
    bot_errors: list[str] = []
    stale_bots: list[str] = []
    for bot in bots:
        if not _fresh(snapshot, clock()):
            stale_bots.append(bot.name)
            continue
        try:
            o, c, went_stale = run_bot(conn, get, bot, snapshot, clock)
        except Exception:  # noqa: BLE001 - one bad bot must not abort the cycle
            log.exception("bot %s failed", bot.name)
            bot_errors.append(bot.name)
            continue
        if went_stale:
            stale_bots.append(bot.name)
        orders += o
        candidates += c
    return {"status": "ok", "orders": orders, "candidates": candidates,
            "bot_errors": bot_errors, "stale_bots": stale_bots}


def _summary(status: str, snapshot_id: int | None = None, markets: int = 0) -> dict:
    return {"status": status, "snapshot_id": snapshot_id, "markets": markets,
            "settled": 0, "orders": 0, "candidates": 0, "decided": False,
            "bot_errors": [], "stale_bots": [], "confirmation_errors": [],
            "confirmation_transitions": []}


def run_cycle(conn: sqlite3.Connection, get, bots, clock=utc_now_iso, *,
              force_decide: bool = False) -> dict:
    started = clock()   # the snapshot's taken_at: every quote was observed at or after it
    try:
        markets = discover(get)
        if not markets:  # Kalshi always lists open temperature markets; none means a bad read
            raise ValueError("no open temperature markets discovered")
    except Exception as e:  # noqa: BLE001 - skip the whole cycle; the hour stays open
        log.warning("discovery failed: %r", e)
        return {**_summary("fetch_failed"), "error": repr(e)}
    sid = record_snapshot(conn, markets, started)
    summary = _summary("ok", sid, len(markets))
    hour = started[:13]
    if force_decide or get_meta(conn, "last_decide_hour") != hour:
        summary["settled"] = settle_due(conn, get, clock())
        if get_meta(conn, "last_confirmation_hour") != hour:
            for row in conn.execute("SELECT name FROM accounts ORDER BY name").fetchall():
                name = row["name"]
                try:   # a confirmation bug must never stop trading: log it, flag it, decide on
                    transition = advance_confirmation(conn, name, clock())
                except Exception:  # noqa: BLE001 - one bad account must not abort the cycle
                    log.exception("confirmation failed for %s", name)
                    conn.rollback()
                    summary["confirmation_errors"].append(name)
                    continue
                if transition is not None:
                    summary["confirmation_transitions"].append(
                        {"account": name, "from": transition[0], "to": transition[1]})
            set_meta(conn, "last_confirmation_hour", hour)
        result = decide_all(conn, get, bots, clock)
        summary.update(status=result["status"], orders=result["orders"],
                       candidates=result["candidates"], decided=True,
                       bot_errors=result["bot_errors"], stale_bots=result["stale_bots"])
        set_meta(conn, "last_decide_hour", hour)
    return summary
