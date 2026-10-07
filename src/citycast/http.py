# src/citycast/http.py
"""Paced, throttle-aware GET for IEM and Kalshi (spec §2.4)."""
import json
import time
from urllib.parse import urlparse

import httpx

IEM_HOST = "mesonet.agron.iastate.edu"
KALSHI_HOST = "api.elections.kalshi.com"
DEFAULT_INTERVALS = {IEM_HOST: 1.0, KALSHI_HOST: 0.2}
BACKOFFS_S = (5, 15, 60)


class HttpError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} {url}")
        self.status = status


class RateLimited(HttpError):
    pass


def _httpx_fetch(url: str, params: dict | None) -> tuple[int, str]:
    r = httpx.get(url, params=params, timeout=120, follow_redirects=True,
                  headers={"User-Agent": "citycast/0.1 (research archive)"})
    return r.status_code, r.text


def _throttled(status: int, text: str) -> bool:
    return status == 429 or "too many requests" in text[:300].lower()


class Paced:
    def __init__(self, host_interval: dict[str, float] | None = None, fetch=_httpx_fetch,
                 sleep=time.sleep, clock=time.monotonic, backoffs=BACKOFFS_S):
        self.host_interval = dict(DEFAULT_INTERVALS if host_interval is None else host_interval)
        self.fetch, self.sleep, self.clock, self.backoffs = fetch, sleep, clock, tuple(backoffs)
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            delay = self.host_interval.get(host, 1.0) - (self.clock() - last)
            if delay > 0:
                self.sleep(delay)
        self._last[host] = self.clock()

    def get_text(self, url: str, params: dict | None = None) -> str:
        host = urlparse(url).hostname or ""
        for attempt in range(len(self.backoffs) + 1):
            self._wait(host)
            status, text = self.fetch(url, params)
            if _throttled(status, text):
                if attempt < len(self.backoffs):
                    self.sleep(self.backoffs[attempt])
                    continue
                raise RateLimited(status, url)
            if status >= 400:
                raise HttpError(status, url)
            return text
        raise RateLimited(429, url)   # unreachable; keeps type checkers honest

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.get_text(url, params))
