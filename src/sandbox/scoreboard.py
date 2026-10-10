"""Per-account evidence and the unlock verdict (spec §1)."""
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from sandbox.db import get_meta, set_meta
from sandbox.stations import climate_day_start_utc
from sandbox.ledger import cash
from sandbox.pricing import devig, multiclass_brier
from sandbox.recorder import row_to_market

log = logging.getLogger(__name__)

N_REQUIRED = 100
N_EVENTS_REQUIRED = 30
BOOT_N = 2000
SEED = 20261004
MANILA = ZoneInfo("Asia/Manila")

# The gate returns an independent P(win) per contract, not one distribution per event, so its
# Brier-vs-market is scored per order as a binary question, not as a multiclass event.
BINARY_SCORED_ACCOUNTS = frozenset({"gate"})


def bootstrap_mean_ci(values, n: int = BOOT_N, seed: int = SEED) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 6), round(float(hi), 6)


def cluster_bootstrap_ratio_ci(sums, counts, n: int = BOOT_N, seed: int = SEED) -> tuple[float, float]:
    """95% CI of sum(sums)/sum(counts), resampling whole clusters with replacement."""
    s = np.asarray(sums, dtype=float)
    c = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    ratios = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return round(float(lo), 6), round(float(hi), 6)


@dataclass(frozen=True)
class Scorecard:
    account: str
    cash: float
    settled_bets: int
    profit: float
    profit_ci: tuple[float, float] | None
    brier_diff: float | None
    brier_ci: tuple[float, float] | None
    profit_pass: bool
    score_pass: bool
    brier_n: int = 0         # distinct (snapshot, event) questions actually scored
    n_events: int = 0        # distinct events among the settled bets
    brier_events: int = 0    # distinct events among the scored Brier samples

    @property
    def unlocked(self) -> bool:
        return (self.settled_bets >= N_REQUIRED and self.brier_n >= N_REQUIRED
                and self.n_events >= N_EVENTS_REQUIRED
                and self.brier_events >= N_EVENTS_REQUIRED
                and self.profit_pass and self.score_pass)


def _winner(conn: sqlite3.Connection, event_ticker: str) -> str | None:
    rows = conn.execute("SELECT ticker, result FROM markets WHERE event_ticker=?",
                        (event_ticker,)).fetchall()
    if not rows or any(r["result"] is None for r in rows):
        return None
    winners = [r["ticker"] for r in rows if r["result"] == "yes"]
    return winners[0] if len(winners) == 1 else None


def _market_probs(conn: sqlite3.Connection, snapshot_id: int, event_ticker: str) -> dict[str, float]:
    rows = conn.execute(
        """SELECT m.*, q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size
           FROM quotes q JOIN markets m ON m.ticker = q.ticker
           WHERE q.snapshot_id = ? AND m.event_ticker = ?""",
        (snapshot_id, event_ticker)).fetchall()
    return devig([row_to_market(r) for r in rows])


def _cluster_ci(by_event: dict[str, list[float]]) -> tuple[float, float] | None:
    """Bets on one event share one outcome, so resample whole events, not single bets."""
    if len(by_event) < 2:
        return None
    return cluster_bootstrap_ratio_ci([sum(v) for v in by_event.values()],
                                      [len(v) for v in by_event.values()])


def report(conn: sqlite3.Connection, account: str, after_order_id: int | None = None,
           segment: str | None = None, new_events_only: bool = False) -> Scorecard:
    """Read-only sample evidence. Cash remains the whole account's current balance.

    `new_events_only` (needs `after_order_id`) is for confirmation windows: it also drops
    every order on an event this account already had ANY order on at or before
    `after_order_id`, so a window counts only city-days that are new to it. Which events
    are dropped never depends on an outcome."""
    if segment not in (None, "pre_day", "same_day"):
        raise ValueError(f"unknown segment: {segment}")
    if new_events_only and after_order_id is None:
        raise ValueError("new_events_only needs after_order_id")
    marker = after_order_id or 0
    sql = """SELECT o.*, m.event_ticker, m.station, m.event_date FROM orders o JOIN markets m ON m.ticker = o.ticker
           WHERE o.account = ? AND o.filled_contracts > 0 AND o.payout IS NOT NULL
           AND o.id > ?"""
    args: list = [account, marker]
    if new_events_only:
        sql += """ AND m.event_ticker NOT IN (SELECT m2.event_ticker FROM orders o2
                   JOIN markets m2 ON m2.ticker = o2.ticker WHERE o2.account = ? AND o2.id <= ?)"""
        args += [account, marker]
    orders = conn.execute(sql + " ORDER BY o.id", args).fetchall()
    if segment is not None:
        def order_segment(order):
            start = climate_day_start_utc(order["station"], order["event_date"])
            return ("pre_day" if start is not None
                    and datetime.fromisoformat(order["decided_at"]) < start else "same_day")
        orders = [o for o in orders if order_segment(o) == segment]
    pnl: list[float] = []
    pnl_by_event: dict[str, list[float]] = {}
    for o in orders:
        p = o["payout"] - o["cost"] - o["fee"]
        pnl.append(p)
        pnl_by_event.setdefault(o["event_ticker"], []).append(p)
    diffs: list[float] = []
    diffs_by_event: dict[str, list[float]] = {}
    scored: set[tuple[int, str]] = set()
    binary = account in BINARY_SCORED_ACCOUNTS
    for o in orders:
        key = (o["snapshot_id"], o["ticker"] if binary else o["event_ticker"])
        if o["snapshot_id"] is None or key in scored:
            continue
        cand = conn.execute("SELECT pmf_json FROM candidates WHERE bot=? AND ticker=? "
                            "AND snapshot_id=? ORDER BY id LIMIT 1",
                            (account, o["ticker"], o["snapshot_id"])).fetchone()
        winner = _winner(conn, o["event_ticker"])
        market = _market_probs(conn, o["snapshot_id"], o["event_ticker"])
        if cand is None or winner is None or not market:
            continue
        bot = json.loads(cand["pmf_json"])
        if binary:
            if o["ticker"] not in bot or o["ticker"] not in market:
                continue
            y = 1.0 if winner == o["ticker"] else 0.0
            diff = (bot[o["ticker"]] - y) ** 2 - (market[o["ticker"]] - y) ** 2
            scored.add(key)
            diffs.append(diff)
            diffs_by_event.setdefault(o["event_ticker"], []).append(diff)
            continue
        if not set(bot) <= set(market):
            # the market priced fewer buckets than the bot: scoring would charge the
            # market a full miss on a bucket it never quoted, a free win for the bot
            continue
        diff = multiclass_brier(bot, winner) - multiclass_brier(market, winner)
        scored.add(key)
        diffs.append(diff)
        diffs_by_event.setdefault(o["event_ticker"], []).append(diff)
    pci = _cluster_ci(pnl_by_event)
    bci = _cluster_ci(diffs_by_event)
    return Scorecard(
        account=account, cash=cash(conn, account), settled_bets=len(pnl),
        profit=round(sum(pnl), 4), profit_ci=pci,
        brier_diff=round(float(np.mean(diffs)), 6) if diffs else None, brier_ci=bci,
        profit_pass=pci is not None and pci[0] > 0,
        score_pass=bci is not None and bci[1] < 0,
        brier_n=len(diffs), n_events=len(pnl_by_event), brier_events=len(diffs_by_event))


def _confirmation(conn: sqlite3.Connection, account: str) -> dict:
    raw = get_meta(conn, f"confirm:{account}")
    state = json.loads(raw) if raw else {
        "state": "LOCKED", "attempt_start_id": 0, "provisional_at": None,
        "marker_id": None, "resolved_at": None}
    state.setdefault("failed_attempts", 0)   # absent in rows written before the counter
    return state


def confirmation_status(conn: sqlite3.Connection, account: str) -> dict:
    """Read-only persisted state (with `failed_attempts`) plus the fresh_* counts of the
    window being graded, zero unless PROVISIONAL. Timestamp/marker keys match the stored
    confirmation JSON."""
    status = _confirmation(conn, account)
    status.update(fresh_settled_bets=0, fresh_n_events=0, fresh_brier_n=0, fresh_brier_events=0)
    if status["state"] == "PROVISIONAL":
        fresh = report(conn, account, after_order_id=status["marker_id"], new_events_only=True)
        status.update(fresh_settled_bets=fresh.settled_bets, fresh_n_events=fresh.n_events,
                      fresh_brier_n=fresh.brier_n, fresh_brier_events=fresh.brier_events)
    return status


def manila_stamp(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(MANILA).strftime("%Y-%m-%d %H:%M Manila")


def _attempts(k: int) -> str:
    return f"{k} failed attempt" + ("" if k == 1 else "s")


def verdict_text(status: dict) -> str:
    """The one wording of the persisted verdict, shared by `sandbox report` and the dashboard."""
    state, failed = status["state"], status.get("failed_attempts", 0)
    if state == "PROVISIONAL":
        after = f", after {_attempts(failed)}" if failed > 0 else ""
        return (f"PROVISIONAL (fresh {status['fresh_settled_bets']}/{N_REQUIRED} settled, "
                f"{status['fresh_n_events']}/{N_EVENTS_REQUIRED} events, "
                f"{status['fresh_brier_n']}/{N_REQUIRED} scored, "
                f"{status['fresh_brier_events']}/{N_EVENTS_REQUIRED} scored events{after})")
    if state == "LOCKED" and status.get("resolved_at"):
        # a stored failure always counts at least one attempt, even if the counter is absent
        return (f"LOCKED — confirmation failed {manila_stamp(status['resolved_at'])} "
                f"({_attempts(max(failed, 1))})")
    if state == "UNLOCKED" and failed > 0:
        return f"UNLOCKED (after {_attempts(failed)})"
    return state


def advance_confirmation(conn: sqlite3.Connection, account: str,
                         now_iso: str) -> tuple[str, str] | None:
    """Only confirmation writer. Each attempt and confirmation uses later order IDs and only
    city-days new to its window; this two-stage heuristic does not control error over
    unlimited attempts. Returns the persisted transition (from, to), or None.

    The PROVISIONAL window is graded once, when it holds every floor the unlock rule needs:
    settled bets, events, scored Brier samples and scored events. None of those counts
    depends on who won, so waiting for them cannot be steered by outcomes."""
    state = _confirmation(conn, account)
    before = state["state"]
    if before == "UNLOCKED":
        return None
    provisional = before == "PROVISIONAL"
    marker = state["marker_id"] if provisional else state["attempt_start_id"]
    sample = report(conn, account, after_order_id=marker, new_events_only=True)
    if provisional:
        if (sample.settled_bets < N_REQUIRED or sample.n_events < N_EVENTS_REQUIRED
                or sample.brier_n < N_REQUIRED or sample.brier_events < N_EVENTS_REQUIRED):
            return None
        state["resolved_at"] = now_iso
        if sample.unlocked:
            state["state"] = "UNLOCKED"
        else:
            state["state"] = "LOCKED"
            state["failed_attempts"] += 1
            state["attempt_start_id"] = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM orders").fetchone()[0]
    elif sample.unlocked:
        state.update(state="PROVISIONAL", provisional_at=now_iso, resolved_at=None,
                     marker_id=conn.execute("SELECT COALESCE(MAX(id), 0) FROM orders").fetchone()[0])
    else:
        return None
    set_meta(conn, f"confirm:{account}", json.dumps(state, sort_keys=True))
    log.info("confirmation %s: %s -> %s (fresh settled %d, events %d, scored %d, scored events %d,"
             " failed attempts %d)", account, before, state["state"], sample.settled_bets,
             sample.n_events, sample.brier_n, sample.brier_events, state["failed_attempts"])
    return before, state["state"]
