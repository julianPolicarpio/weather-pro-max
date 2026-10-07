"""Unauthenticated, paced reads from Kalshi's public market-data API.

Kalshi's authenticated Basic tier allows 20 reads/s (200 tokens/s, 10 per
request). The unauthenticated limit is undocumented, so we stay at 5/s.
Every function takes `get` so tests inject a fake and never touch the network.
"""
import time

from highcast.http import get_json
from sandbox.stations import metric_from_series

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MIN_INTERVAL_S = 0.2


class PacedGet:
    """get(path, params) -> JSON dict, never faster than one call per interval."""

    def __init__(self, fetch=get_json, min_interval: float = MIN_INTERVAL_S,
                 clock=time.monotonic, sleep=time.sleep):
        self.fetch = fetch
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self._last: float | None = None

    def __call__(self, path: str, params: dict | None = None) -> dict:
        if self._last is not None:
            wait = self._last + self.min_interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()
        return self.fetch(BASE + path, params)


def list_temperature_series(get) -> list[dict]:
    payload = get("/series", {"category": "Climate and Weather", "limit": 500})
    return [s for s in payload.get("series") or []
            if metric_from_series(s.get("ticker", "")) is not None]


def open_markets(get, series: str) -> dict:
    return get("/markets", {"series_ticker": series, "status": "open", "limit": 200})


def settled_markets(get, series: str) -> dict:
    # Verified 2026-10-04: status=settled returns markets whose status is
    # "finalized" with result "yes"/"no".
    return get("/markets", {"series_ticker": series, "status": "settled", "limit": 200})


def fetch_book_payload(get, ticker: str) -> dict:
    return get(f"/markets/{ticker}/orderbook")
