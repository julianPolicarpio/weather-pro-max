"""Settle on Kalshi's own result: $1 per winning contract, no settlement fee."""
import logging
import sqlite3

from sandbox.kalshi_public import settled_markets

log = logging.getLogger(__name__)

_RESULT = "(SELECT result FROM markets WHERE markets.ticker = {t}.ticker)"


def settle_due(conn: sqlite3.Connection, get, now_iso: str) -> int:
    """Fetch results for every series with a closed, unsettled market, then write them
    all in ONE short transaction after the fetches: no write lock is held across the
    network, so the dashboard and `report` never wait on Kalshi."""
    due = [r["series"] for r in conn.execute(
        "SELECT DISTINCT series FROM markets WHERE result IS NULL "
        "AND close_time IS NOT NULL AND close_time <= ?", (now_iso,))]
    results: list[tuple[str, str]] = []
    for series in due:
        try:
            # Kalshi returns settled markets newest-first (verified 2026-10-04);
            # limit=200 covers ~1 month while settle runs hourly.
            payload = settled_markets(get, series)
            for m in payload.get("markets") or []:
                if m.get("status") not in ("finalized", "settled"):
                    continue
                if m.get("result") not in ("yes", "no"):
                    log.warning("settled market %s has unusable result %r; left unsettled",
                                m.get("ticker"), m.get("result"))
                    continue
                if m.get("ticker"):
                    results.append((m["ticker"], m["result"]))
        except Exception as e:  # noqa: BLE001 - try again next hour
            log.warning("settle fetch failed for %s: %r", series, e)
    newly = 0
    o = _RESULT.format(t="orders")
    c = _RESULT.format(t="candidates")
    with conn:
        for ticker, result in results:
            newly += conn.execute(
                "UPDATE markets SET result=?, settled_at=? WHERE ticker=? AND result IS NULL",
                (result, now_iso, ticker)).rowcount
        conn.execute(f"""UPDATE orders
                         SET payout = CASE WHEN {o} = orders.side THEN orders.filled_contracts * 1.0
                                           ELSE 0.0 END
                         WHERE payout IS NULL AND filled_contracts > 0 AND {o} IS NOT NULL""")
        conn.execute(f"""UPDATE candidates
                         SET result = {c},
                             asif_profit = CASE
                               WHEN yes_ask IS NULL OR fee_per_contract IS NULL THEN NULL
                               ELSE ROUND((CASE WHEN {c} = 'yes' THEN 1.0 ELSE 0.0 END)
                                          - yes_ask - fee_per_contract, 4) END
                         WHERE result IS NULL AND {c} IS NOT NULL""")
    return newly
