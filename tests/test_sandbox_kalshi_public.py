from sandbox.kalshi_public import (BASE, PacedGet, fetch_book_payload,
                                   list_temperature_series, open_markets,
                                   settled_markets)


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.t += s


def test_paced_get_waits_between_calls_and_prefixes_base():
    calls = []
    fc = FakeClock()
    get = PacedGet(fetch=lambda url, params: calls.append((url, params)) or {"ok": 1},
                   min_interval=0.2, clock=fc.clock, sleep=fc.sleep)
    assert get("/series", {"a": 1}) == {"ok": 1}
    get("/markets")
    assert calls == [(BASE + "/series", {"a": 1}), (BASE + "/markets", None)]
    assert fc.slept == [0.2]  # second call waited the full interval


def test_list_temperature_series_filters_to_high_and_low():
    def get(path, params=None):
        assert path == "/series"
        assert params == {"category": "Climate and Weather", "limit": 500}
        return {"series": [{"ticker": "KXHIGHNY"}, {"ticker": "KXRAINNYC"},
                           {"ticker": "KXLOWTNYC"}]}
    assert [s["ticker"] for s in list_temperature_series(get)] == ["KXHIGHNY", "KXLOWTNYC"]


def test_market_and_book_paths():
    seen = []

    def get(path, params=None):
        seen.append((path, params))
        return {}
    open_markets(get, "KXHIGHNY")
    settled_markets(get, "KXHIGHNY")
    fetch_book_payload(get, "KXHIGHNY-26OCT04-T70")
    assert seen == [
        ("/markets", {"series_ticker": "KXHIGHNY", "status": "open", "limit": 200}),
        ("/markets", {"series_ticker": "KXHIGHNY", "status": "settled", "limit": 200}),
        ("/markets/KXHIGHNY-26OCT04-T70/orderbook", None),
    ]
