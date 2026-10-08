"""Kalshi hourly candles -> kalshi_candles (close bid/ask per active hour; spec §1)."""
from datetime import datetime, timedelta, timezone

from citycast.backfill import Chunk
from citycast.http import HttpError
from citycast.sources.kalshi_markets import KALSHI, historical_cutoff

MAX_WINDOW_H = 5000


def _num(x) -> float | None:
    return None if x in (None, "") else float(x)


def _close(d) -> float | None:
    return _num(d.get("close")) if isinstance(d, dict) else None


def candle_rows(ticker: str, payload: dict) -> list[dict]:
    return [{"ticker": ticker,
             "end_utc": datetime.fromtimestamp(int(c["end_period_ts"]), timezone.utc).isoformat(),
             "yes_bid": _close(c.get("yes_bid")), "yes_ask": _close(c.get("yes_ask")),
             "volume": _num(c.get("volume")), "open_interest": _num(c.get("open_interest"))}
            for c in payload.get("candlesticks") or []]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def fetch_candles(paced, ticker: str, series: str, open_iso: str | None, close_iso: str,
                  cutoff_iso: str) -> dict:
    end = _ts(close_iso)
    start = _ts(open_iso) if open_iso else end - 7 * 86400
    paths = [f"/historical/markets/{ticker}/candlesticks",
             f"/series/{series}/markets/{ticker}/candlesticks"]
    if close_iso >= cutoff_iso:
        paths.reverse()
    merged: list[dict] = []
    for i, path in enumerate(paths):
        try:
            a = start
            while a < end:
                b = min(end, a + MAX_WINDOW_H * 3600)
                j = paced.get_json(KALSHI + path, {"start_ts": a, "end_ts": b, "period_interval": 60})
                merged.extend(j.get("candlesticks") or [])
                a = b
            return {"candlesticks": merged}
        except HttpError as e:
            if e.status != 404 or i == len(paths) - 1:
                raise
            merged = []
    return {"candlesticks": merged}


def chunks(paced, conn, now_iso, since: str | None = None) -> list[Chunk]:
    now = now_iso()
    sql = ("SELECT ticker, series, open_time, close_time FROM kalshi_markets "
           "WHERE NOT (lo IS NULL AND hi IS NULL) AND close_time < ?")
    args: list = [now]
    if since:
        sql += " AND close_time >= ?"
        args.append(since)
    rows = conn.execute(sql + " ORDER BY close_time", args).fetchall()
    if not rows:
        return []
    cutoff = historical_cutoff(paced)
    out = []
    for r in rows:
        def run(r=r):
            payload = fetch_candles(paced, r["ticker"], r["series"], r["open_time"],
                                    r["close_time"], cutoff)
            return [("kalshi_candles", candle_rows(r["ticker"], payload))]
        out.append(Chunk("kalshi-candles", r["ticker"], False, run))
    return out
