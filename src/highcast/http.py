import time

import httpx

HEADERS = {"User-Agent": "highcast personal weather project (contact: local)"}


def _get(url: str, params: dict | None, tries: int = 3) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = httpx.get(url, params=params, timeout=30, headers=HEADERS,
                          follow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retry any transport/status error
            last = e
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def get_json(url: str, params: dict | None = None):
    return _get(url, params).json()


def get_text(url: str, params: dict | None = None) -> str:
    return _get(url, params).text
