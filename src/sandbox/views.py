# src/sandbox/views.py
"""Data for the Weather Pro Max pages (spec an internal design note §3-§6).
Pure: a sqlite connection and snapshots in, plain dicts out. No FastAPI imports, no writes."""
import sqlite3
from datetime import date, datetime

from sandbox.cycle import ACCOUNTS
from sandbox.ledger import cash
from sandbox.markets import usable_ask, usable_bid
from sandbox.models import Snapshot
from sandbox.pricing import devig
from sandbox.recorder import load_snapshot

ACTIVE_BOTS = ("favorite", "citycast", "highcast", "mos")
DEFAULT_BOT = "citycast"
STALE_MIN = 30
LIVE_RAIL_N = 5
AWAIT_RAIL_N = 2
MOVES_N = 8
MOVE_MIN_PTS = 0.5      # smaller changes are rounding noise and would show as "+0"
AWAITING = "settles when the NWS report posts"

CITY = {"KATL": "Atlanta", "KAUS": "Austin", "KBOS": "Boston", "KDCA": "Washington DC", "KDEN": "Denver",
        "KDFW": "Dallas", "KEWR": "Newark", "KHOU": "Houston", "KLAS": "Las Vegas", "KLAX": "Los Angeles",
        "KMDW": "Chicago", "KMIA": "Miami", "KMSP": "Minneapolis", "KMSY": "New Orleans", "KNYC": "New York",
        "KOKC": "Oklahoma City", "KPHL": "Philadelphia", "KPHX": "Phoenix", "KSAN": "San Diego",
        "KSAT": "San Antonio", "KSDF": "Louisville", "KSEA": "Seattle", "KSFO": "San Francisco",
        "KTTN": "Trenton"}

_POS_SQL = """SELECT o.id, o.ticker, o.side, o.filled_contracts AS n, o.cost, o.fee, o.payout,
                     m.station, m.metric, m.event_date, m.lo, m.hi, m.close_time, m.settled_at
              FROM orders o JOIN markets m ON m.ticker = o.ticker
              WHERE o.account = ? AND o.filled_contracts > 0"""


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def city(station: str | None) -> str:
    return CITY.get(station or "", station or "Unknown")


def bucket_text(lo: int | None, hi: int | None) -> str:
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤{hi}°"
    if hi is None:
        return f"≥{lo}°"
    return f"{lo}–{hi}°"


def _short_day(event_date: str) -> str:
    d = date.fromisoformat(event_date)
    return f"{d:%b} {d.day}"


def title(station, metric, lo, hi, event_date) -> str:
    return f"{city(station)} {metric or ''} {bucket_text(lo, hi)} · {_short_day(event_date)}".replace("  ", " ")


def until(close_iso: str | None, now_iso: str) -> str:
    if not close_iso:
        return "close time unknown"
    s = (_ts(close_iso) - _ts(now_iso)).total_seconds()
    if s <= 0:
        return "closed"
    if s < 3600:
        return f"closes in {max(1, int(s // 60))}m"
    if s < 48 * 3600:
        return f"closes in {int(s // 3600)}h"
    return f"closes in {int(s // 86400)}d"


def day_text(day_iso: str) -> str:
    d = date.fromisoformat(day_iso)
    return f"{d:%A}, {d:%b} {d.day}"


def previous_snapshot(conn: sqlite3.Connection, snap: Snapshot | None) -> Snapshot | None:
    """Newest snapshot before `snap` that recorded quotes."""
    if snap is None:
        return None
    row = conn.execute("SELECT s.id FROM snapshots s WHERE s.id < ? AND EXISTS "
                       "(SELECT 1 FROM quotes q WHERE q.snapshot_id = s.id) ORDER BY s.id DESC LIMIT 1",
                       (snap.id,)).fetchone()
    return load_snapshot(conn, row[0]) if row else None


def freshness(snap: Snapshot | None, now_iso: str) -> dict:
    if snap is None:
        return {"state": "none", "minutes": None}
    minutes = max(0, int((_ts(now_iso) - _ts(snap.taken_at)).total_seconds() // 60))
    return {"state": "stale" if minutes > STALE_MIN else "fresh", "minutes": minutes}


def current_day(snap: Snapshot | None, now_iso: str) -> str | None:
    """Earliest climate day that still has an open market; the latest day once all are closed."""
    if snap is None or not snap.markets:
        return None
    now = _ts(now_iso)
    open_days = [m.event_date for m in snap.markets if m.close_time and _ts(m.close_time) > now]
    return min(open_days) if open_days else max(m.event_date for m in snap.markets)


def bot_summaries(conn: sqlite3.Connection, show_all: bool = False) -> list[dict]:
    have = {r[0] for r in conn.execute("SELECT name FROM accounts")}
    out = []
    for name, _kind, label in ACCOUNTS:
        if name not in have or (not show_all and name not in ACTIVE_BOTS):
            continue
        open_n, at_risk = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost + fee), 0) FROM orders "
            "WHERE account=? AND filled_contracts > 0 AND payout IS NULL", (name,)).fetchone()
        settled_n, profit = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(payout - cost - fee), 0) FROM orders "
            "WHERE account=? AND filled_contracts > 0 AND payout IS NOT NULL", (name,)).fetchone()
        out.append({"name": name, "label": label, "cash": round(cash(conn, name), 2), "open": open_n,
                    "at_risk": round(at_risk, 2), "settled": settled_n, "profit": round(profit, 2)})
    return out


def held_tickers(conn: sqlite3.Connection, account: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT ticker FROM orders WHERE account=? AND filled_contracts > 0 "
                                       "AND payout IS NULL", (account,))}


def _mark(m, side: str) -> float | None:
    if m is None:
        return None
    if side == "yes":
        return usable_bid(m)
    ask = usable_ask(m)
    return None if ask is None else 1.0 - ask


def open_positions(conn: sqlite3.Connection, account: str, snap: Snapshot | None, now_iso: str) -> dict:
    by_ticker = {m.ticker: m for m in snap.markets} if snap else {}
    now = _ts(now_iso)
    live, awaiting, at_risk = [], [], 0.0
    for r in conn.execute(_POS_SQL + " AND o.payout IS NULL ORDER BY o.id", (account,)):
        avg = r["cost"] / r["n"]
        row = {"ticker": r["ticker"], "title": title(r["station"], r["metric"], r["lo"], r["hi"], r["event_date"]),
               "side": r["side"], "n": r["n"], "paid_c": round(avg * 100), "close_time": r["close_time"]}
        at_risk += r["cost"] + r["fee"]
        if r["close_time"] and _ts(r["close_time"]) > now:
            mark = _mark(by_ticker.get(r["ticker"]), r["side"])
            row.update(mark_c=None if mark is None else round(mark * 100),
                       unrealized_c=None if mark is None else round((mark - avg) * r["n"] * 100),
                       status=until(r["close_time"], now_iso))
            live.append(row)
        else:
            row.update(mark_c=None, unrealized_c=None, status=AWAITING)
            awaiting.append(row)
    live.sort(key=lambda x: x["close_time"])
    awaiting.sort(key=lambda x: x["close_time"] or "", reverse=True)
    return {"live": live, "awaiting": awaiting, "count": len(live) + len(awaiting), "at_risk": round(at_risk, 2)}


def settled_positions(conn: sqlite3.Connection, account: str) -> list[dict]:
    rows = conn.execute(_POS_SQL + " AND o.payout IS NOT NULL ORDER BY m.settled_at DESC, o.id DESC", (account,))
    return [{"ticker": r["ticker"], "title": title(r["station"], r["metric"], r["lo"], r["hi"], r["event_date"]),
             "side": r["side"], "n": r["n"], "paid_c": round(r["cost"] / r["n"] * 100),
             "payout": round(r["payout"], 2), "realized": round(r["payout"] - r["cost"] - r["fee"], 2),
             "settled_at": r["settled_at"]} for r in rows]


def _event_probs(snap: Snapshot | None) -> dict[str, dict[str, float]]:
    """event_ticker -> {ticker: devigged p}. Keys are the event's priced rungs."""
    if snap is None:
        return {}
    return {event: devig(ms) for event, ms in snap.events().items()}


def _comparable(probs: dict[str, float], prev_probs: dict[str, float] | None) -> bool:
    """Devigged probabilities only compare when both snapshots priced the identical set of rungs."""
    return prev_probs is not None and set(probs) == set(prev_probs)


def market_cards(snap, prev, held: set[str], day: str, metric: str | None = None,
                 station: str | None = None, now_iso: str | None = None) -> list[dict]:
    if snap is None:
        return []
    prev_p = _event_probs(prev)
    cards = []
    for event, ms in snap.events().items():
        m0 = ms[0]
        if m0.event_date != day or (metric and m0.metric != metric) or (station and m0.station != station):
            continue
        probs = devig(ms)
        prev_probs = prev_p.get(event)
        comparable = _comparable(probs, prev_probs)
        rungs, moves = [], []
        for m in ms:
            p = probs.get(m.ticker)
            if comparable and p is not None:
                moves.append((p - prev_probs[m.ticker]) * 100)
            bid, ask = usable_bid(m), usable_ask(m)
            rungs.append({"ticker": m.ticker, "label": bucket_text(m.lo, m.hi), "p": p,
                          "bid_c": None if bid is None else round(bid * 100),
                          "ask_c": None if ask is None else round(ask * 100), "held": m.ticker in held})
        biggest = max(moves, key=abs) if moves else None
        word = "Highest" if m0.metric == "high" else "Lowest"
        cards.append({"event": event, "station": m0.station, "city": city(m0.station), "metric": m0.metric,
                      "closes": until(m0.close_time, now_iso) if now_iso else "",
                      "question": f"{word} temperature in {city(m0.station)} on {_short_day(day)}?",
                      "rungs": rungs, "any_held": any(r["held"] for r in rungs),
                      "max_move": biggest if biggest is not None and abs(biggest) >= MOVE_MIN_PTS else None})
    cards.sort(key=lambda c: (c["city"], c["metric"] or ""))
    return cards


def biggest_moves(snap, prev, day: str, n: int = MOVES_N) -> list[dict]:
    if snap is None or prev is None:
        return []
    prev_p = _event_probs(prev)
    out = []
    for event, ms in snap.events().items():
        if ms[0].event_date != day:
            continue
        probs = devig(ms)
        prev_probs = prev_p.get(event)
        if not _comparable(probs, prev_probs):
            continue
        for m in ms:
            p = probs.get(m.ticker)
            if p is None or abs(p - prev_probs[m.ticker]) * 100 < MOVE_MIN_PTS:
                continue
            q = prev_probs[m.ticker]
            out.append({"ticker": m.ticker, "p": p, "change": (p - q) * 100,
                        "title": f"{city(m.station)} {m.metric or ''} {bucket_text(m.lo, m.hi)}".replace("  ", " ")})
    out.sort(key=lambda x: (-abs(x["change"]), -x["p"], x["ticker"]))
    return out[:n]


def out_of_cash(cash_now: float, snap: Snapshot | None) -> bool:
    """True when the account cannot afford even the cheapest contract in the latest snapshot."""
    if cash_now < 0.01:
        return True
    asks = [a for m in snap.markets if (a := usable_ask(m)) is not None] if snap else []
    return bool(asks) and cash_now < min(asks)
