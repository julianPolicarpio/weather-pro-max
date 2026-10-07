# src/citycast/sources/kalshi_markets.py
"""Kalshi settled markets (archive + recent) -> kalshi_markets (spec §1 'Legacy market shape')."""
import json
import re
from datetime import datetime

from citycast.backfill import Chunk
from highcast.fetchers.kalshi import _lo_hi

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
_DATE = re.compile(r"(\d{2}[A-Z]{3}\d{2})")
_GT = re.compile(r"strictly greater than (-?\d+)")
_LT = re.compile(r"strictly less than (-?\d+)")
_BETWEEN = re.compile(r"between (-?\d+)\s*(?:-|–|to|and)\s*(-?\d+)")


def event_date(event_ticker: str) -> str | None:
    m = _DATE.search(event_ticker or "")
    if not m:
        return None
    return datetime.strptime(m.group(1), "%y%b%d").date().isoformat()


def interval(m: dict) -> tuple[int | None, int | None] | None:
    if m.get("floor_strike") is not None or m.get("cap_strike") is not None:
        return _lo_hi(m.get("floor_strike"), m.get("cap_strike"))
    rules = m.get("rules_primary") or ""
    if g := _GT.search(rules):
        return int(g.group(1)) + 1, None
    if lt := _LT.search(rules):
        return None, int(lt.group(1)) - 1
    if b := _BETWEEN.search(rules):
        return int(b.group(1)), int(b.group(2))
    return None


def _utc(ts: str | None) -> str | None:
    if not ts:
        return None
    return ts[:-1] + "+00:00" if ts.endswith("Z") else ts


def market_rows(markets: list[dict], series: str, index: dict) -> list[dict]:
    icao, metric = index[series]
    rows = []
    for m in markets:
        if m.get("market_type") not in (None, "binary"):
            continue
        d = event_date(m.get("event_ticker", ""))
        if d is None or not m.get("ticker") or not m.get("close_time"):
            continue
        lo, hi = interval(m) or (None, None)
        rows.append({"ticker": m["ticker"], "series": series, "event_ticker": m["event_ticker"],
                     "station": icao, "metric": metric, "event_date": d, "lo": lo, "hi": hi,
                     "open_time": _utc(m.get("open_time")), "close_time": _utc(m["close_time"]),
                     "result": m.get("result"), "expiration_value": m.get("expiration_value"),
                     "raw": json.dumps(m, sort_keys=True)})
    return rows


def fetch_series_markets(paced, series: str) -> list[dict]:
    """Archive first, then recent settled; a ticker in both keeps the recent copy."""
    out: dict[str, dict] = {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cursor = None
        while True:
            params = {"series_ticker": series, "limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor
            j = paced.get_json(KALSHI + path, params)
            ms = j.get("markets") or []
            for m in ms:
                out[m["ticker"]] = m
            cursor = j.get("cursor")
            if not cursor or not ms:
                break
    return list(out.values())


def historical_cutoff(paced) -> str:
    return _utc(paced.get_json(KALSHI + "/historical/cutoff")["market_settled_ts"])


def chunks(paced, index: dict) -> list[Chunk]:
    out = []
    for series in sorted(index):
        def run(series=series):
            return [("kalshi_markets", market_rows(fetch_series_markets(paced, series), series, index))]
        out.append(Chunk("kalshi-markets", series, True, run))
    return out
