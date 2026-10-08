from contextlib import closing

from citycast import db
from citycast.http import HttpError
from citycast.sources.kalshi_candles import candle_rows, chunks, fetch_candles

PAYLOAD = {"candlesticks": [
    {"end_period_ts": 1736056800, "volume": "10.00", "open_interest": "46.00",
     "yes_bid": {"close": "0.3000"}, "yes_ask": {"close": "0.3200"}, "price": {"close": None}},
    {"end_period_ts": 1736060400, "volume": "0.00", "open_interest": "46.00",
     "yes_bid": {"close": None}, "yes_ask": {"close": "0.3300"}},
]}


def test_candle_rows_keep_close_bid_ask():
    rows = candle_rows("T", PAYLOAD)
    assert rows[0] == {"ticker": "T", "end_utc": "2025-01-05T06:00:00+00:00", "yes_bid": 0.30,
                       "yes_ask": 0.32, "volume": 10.0, "open_interest": 46.0}
    assert rows[1]["yes_bid"] is None and rows[1]["yes_ask"] == 0.33


def test_fetch_uses_historical_before_cutoff_and_falls_back_on_404():
    calls = []

    class FakePaced:
        def get_json(self, url, params):
            calls.append(url.split("/trade-api/v2")[1])
            if "/historical/" in url:
                raise HttpError(404, url)
            return PAYLOAD
    got = fetch_candles(FakePaced(), "T", "KXHIGHLAX", "2025-01-04T20:00:00+00:00",
                        "2025-01-06T07:59:00+00:00", "2026-08-08T00:00:00+00:00")
    assert calls == ["/historical/markets/T/candlesticks", "/series/KXHIGHLAX/markets/T/candlesticks"]
    assert len(got["candlesticks"]) == 2


def test_chunks_only_closed_usable_markets(tmp_path):
    base = {"series": "KXHIGHLAX", "event_ticker": "E", "station": "KLAX", "metric": "high",
            "event_date": "2025-01-05", "open_time": "2025-01-04T20:00:00+00:00", "result": "no",
            "expiration_value": "", "raw": "{}"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [
                {**base, "ticker": "OK", "lo": 66, "hi": 67, "close_time": "2025-01-06T07:59:00+00:00"},
                {**base, "ticker": "UNUSABLE", "lo": None, "hi": None, "close_time": "2025-01-06T07:59:00+00:00"},
                {**base, "ticker": "OPEN", "lo": 66, "hi": 67, "close_time": "2099-01-01T00:00:00+00:00"},
            ])

        class FakePaced:
            def get_json(self, url, params=None):
                if url.endswith("/historical/cutoff"):
                    return {"market_settled_ts": "2026-08-08T00:00:00Z"}
                return PAYLOAD
        cs = chunks(FakePaced(), c, lambda: "2026-10-07T00:00:00+00:00")
        assert [ch.key for ch in cs] == ["OK"] and not cs[0].current
        table, rows = cs[0].run()[0]
        assert table == "kalshi_candles" and len(rows) == 2


def test_chunks_since_filters_by_close_time(tmp_path):
    base = {"series": "KXHIGHLAX", "event_ticker": "E", "station": "KLAX", "metric": "high",
            "event_date": "2025-01-05", "open_time": "2025-01-04T20:00:00+00:00", "result": "no",
            "expiration_value": "", "raw": "{}", "lo": 66, "hi": 67}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [
                {**base, "ticker": "OLD", "close_time": "2025-01-06T07:59:00+00:00"},
                {**base, "ticker": "NEW", "close_time": "2026-09-10T07:59:00+00:00"},
            ])

        class FakePaced:
            def get_json(self, url, params=None):
                return {"market_settled_ts": "2026-08-08T00:00:00Z"}
        now = lambda: "2026-10-07T00:00:00+00:00"  # noqa: E731
        assert [ch.key for ch in chunks(FakePaced(), c, now, since="2026-09-01")] == ["NEW"]
        assert [ch.key for ch in chunks(FakePaced(), c, now)] == ["OLD", "NEW"]
