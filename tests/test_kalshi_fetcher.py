import pytest

from highcast.config import load_config
from highcast.fetchers import kalshi
from highcast.metric import get_metric

# Trimmed real payload captured live 2026-07-19 from:
#   GET https://api.elections.kalshi.com/trade-api/v2/markets
#       ?series_ticker=KXHIGHLAX&status=open&limit=50
# Kept only the fields fetch_ranges/parse_markets actually use, plus enough of
# each market shape to prove the mapping: a top open tail (only floor_strike),
# a bottom open tail (only cap_strike), a between-bucket (both strikes), and
# one market from a *different* event_ticker (JUL18) that must be filtered out.
MARKETS_PAYLOAD = {
    "markets": [
        {
            "ticker": "KXHIGHLAX-26JUL19-T80",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": 80,
            "cap_strike": None,
            "yes_sub_title": "81° or above",
            "no_sub_title": "81° or above",
            "last_price_dollars": "0.0500",
            "yes_bid_dollars": "0.0500",
            "yes_ask_dollars": "0.0600",
            "status": "active",
        },
        {
            "ticker": "KXHIGHLAX-26JUL19-T73",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": None,
            "cap_strike": 73,
            "yes_sub_title": "72° or below",
            "no_sub_title": "72° or below",
            "last_price_dollars": "0.0100",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0100",
            "status": "active",
        },
        {
            "ticker": "KXHIGHLAX-26JUL19-B79.5",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": 79,
            "cap_strike": 80,
            "yes_sub_title": "79° to 80°",
            "no_sub_title": "79° to 80°",
            "last_price_dollars": "0.2400",
            "yes_bid_dollars": "0.2300",
            "yes_ask_dollars": "0.2400",
            "status": "active",
        },
        {
            "ticker": "KXHIGHLAX-26JUL19-B77.5",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": 77,
            "cap_strike": 78,
            "yes_sub_title": "77° to 78°",
            "no_sub_title": "77° to 78°",
            "last_price_dollars": "0.5700",
            "yes_bid_dollars": "0.5500",
            "yes_ask_dollars": "0.5600",
            "status": "active",
        },
        {
            "ticker": "KXHIGHLAX-26JUL19-B75.5",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": 75,
            "cap_strike": 76,
            "yes_sub_title": "75° to 76°",
            "no_sub_title": "75° to 76°",
            "last_price_dollars": "0.1800",
            "yes_bid_dollars": "0.1700",
            "yes_ask_dollars": "0.1900",
            "status": "active",
        },
        {
            "ticker": "KXHIGHLAX-26JUL19-B73.5",
            "event_ticker": "KXHIGHLAX-26JUL19",
            "market_type": "binary",
            "floor_strike": 73,
            "cap_strike": 74,
            "yes_sub_title": "73° to 74°",
            "no_sub_title": "73° to 74°",
            # no last_price on this thin market -- must fall back to bid/ask mid
            "last_price_dollars": None,
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0100",
            "status": "active",
        },
        {
            # different event (tomorrow's date) -- must be excluded from today's
            # event_ticker filter
            "ticker": "KXHIGHLAX-26JUL18-T79",
            "event_ticker": "KXHIGHLAX-26JUL18",
            "market_type": "binary",
            "floor_strike": 79,
            "cap_strike": None,
            "yes_sub_title": "80° or above",
            "no_sub_title": "80° or above",
            "last_price_dollars": "0.0100",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0100",
            "status": "active",
        },
    ]
}


# Small hand-built KXLOWTLAX (low-temp) payload -- same shape as the high
# series, just a different series prefix and event date, to prove
# parse_markets is series-agnostic. One between bucket + one top open tail
# + one bottom open tail.
LOW_MARKETS_PAYLOAD = {
    "markets": [
        {
            "ticker": "KXLOWTLAX-26JUL23-T60",
            "event_ticker": "KXLOWTLAX-26JUL23",
            "market_type": "binary",
            "floor_strike": 60,
            "cap_strike": None,
            "yes_sub_title": "61° or above",
            "no_sub_title": "61° or above",
            "last_price_dollars": "0.0300",
            "yes_bid_dollars": "0.0200",
            "yes_ask_dollars": "0.0400",
            "status": "active",
        },
        {
            "ticker": "KXLOWTLAX-26JUL23-T55",
            "event_ticker": "KXLOWTLAX-26JUL23",
            "market_type": "binary",
            "floor_strike": None,
            "cap_strike": 55,
            "yes_sub_title": "54° or below",
            "no_sub_title": "54° or below",
            "last_price_dollars": "0.0200",
            "yes_bid_dollars": "0.0100",
            "yes_ask_dollars": "0.0300",
            "status": "active",
        },
        {
            "ticker": "KXLOWTLAX-26JUL23-B57.5",
            "event_ticker": "KXLOWTLAX-26JUL23",
            "market_type": "binary",
            "floor_strike": 57,
            "cap_strike": 58,
            "yes_sub_title": "57° to 58°",
            "no_sub_title": "57° to 58°",
            "last_price_dollars": "0.4100",
            "yes_bid_dollars": "0.4000",
            "yes_ask_dollars": "0.4200",
            "status": "active",
        },
    ]
}


def test_parse_markets_is_series_agnostic_for_low_event():
    rows = kalshi.parse_markets(
        LOW_MARKETS_PAYLOAD, kalshi.event_ticker("KXLOWTLAX", "2026-07-23")
    )
    assert len(rows) == 3
    assert [r["label"] for r in rows] == ["≤54", "57–58", "≥61"]
    top = next(r for r in rows if r["hi"] is None)
    assert top["lo"] == 61
    bottom = next(r for r in rows if r["lo"] is None)
    assert bottom["hi"] == 54
    mid = next(r for r in rows if r["lo"] == 57)
    assert mid["hi"] == 58
    assert mid["last_price"] == pytest.approx(0.41)


def test_event_ticker_for_date_matches_kalshi_format():
    # Verified live: KXHIGHLAX-26JUL19 for 2026-07-19, KXHIGHLAX-26JUL18 for
    # 2026-07-18 -- 2-digit year + 3-letter uppercase month + 2-digit day.
    assert kalshi.event_ticker("KXHIGHLAX", "2026-07-19") == "KXHIGHLAX-26JUL19"
    assert kalshi.event_ticker("KXHIGHLAX", "2026-07-18") == "KXHIGHLAX-26JUL18"
    # zero-padded day
    assert kalshi.event_ticker("KXHIGHLAX", "2026-01-05") == "KXHIGHLAX-26JAN05"


def test_parse_markets_filters_to_event_and_sorts_ascending():
    rows = kalshi.parse_markets(MARKETS_PAYLOAD, "KXHIGHLAX-26JUL19")
    assert len(rows) == 6
    assert [r["label"] for r in rows] == [
        "≤72", "73–74", "75–76", "77–78", "79–80", "≥81",
    ]


def test_parse_markets_top_tail_mapping():
    # floor_strike=80, subtitle "81° or above" -> strict greater-than-80 ->
    # the open range starts at 81 (floor_strike + 1), hi is open (None).
    rows = kalshi.parse_markets(MARKETS_PAYLOAD, "KXHIGHLAX-26JUL19")
    top = next(r for r in rows if r["hi"] is None)
    assert top["lo"] == 81
    assert top["hi"] is None
    assert top["label"] == "≥81"
    assert top["last_price"] == pytest.approx(0.05)
    assert top["yes_bid"] == pytest.approx(0.05)
    assert top["yes_ask"] == pytest.approx(0.06)


def test_parse_markets_bottom_tail_mapping():
    # cap_strike=73, subtitle "72° or below" -> lo is open (None), hi = 72
    # (cap_strike - 1).
    rows = kalshi.parse_markets(MARKETS_PAYLOAD, "KXHIGHLAX-26JUL19")
    bottom = next(r for r in rows if r["lo"] is None)
    assert bottom["hi"] == 72
    assert bottom["label"] == "≤72"


def test_parse_markets_between_bucket_mapping():
    # floor_strike=79, cap_strike=80, subtitle "79° to 80°" -> lo/hi map
    # directly onto the strikes, no offset.
    rows = kalshi.parse_markets(MARKETS_PAYLOAD, "KXHIGHLAX-26JUL19")
    mid = next(r for r in rows if r["lo"] == 79)
    assert mid["hi"] == 80
    assert mid["label"] == "79–80"
    assert mid["last_price"] == pytest.approx(0.24)


def test_parse_markets_falls_back_to_bid_ask_mid_when_last_price_missing():
    rows = kalshi.parse_markets(MARKETS_PAYLOAD, "KXHIGHLAX-26JUL19")
    thin = next(r for r in rows if r["lo"] == 73 and r["hi"] == 74)
    assert thin["last_price"] is None
    assert thin["yes_bid"] == pytest.approx(0.0)
    assert thin["yes_ask"] == pytest.approx(0.01)


def test_resolve_price_prefers_last_price():
    r = {"last_price": 0.24, "yes_bid": 0.23, "yes_ask": 0.25}
    assert kalshi.resolve_price(r) == pytest.approx(0.24)


def test_resolve_price_falls_back_to_bid_ask_mid():
    r = {"last_price": None, "yes_bid": 0.0, "yes_ask": 0.01}
    assert kalshi.resolve_price(r) == pytest.approx(0.005)


def test_resolve_price_falls_back_to_single_side_when_only_one_present():
    assert kalshi.resolve_price({"last_price": None, "yes_bid": 0.05, "yes_ask": None}) == pytest.approx(0.05)
    assert kalshi.resolve_price({"last_price": None, "yes_bid": None, "yes_ask": None}) is None


def test_fetch_ranges_returns_none_when_series_blank(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg.station, "kalshi_series", "")
    assert kalshi.fetch_ranges(cfg, "2026-07-19") is None


def test_fetch_ranges_returns_none_on_network_error(monkeypatch):
    cfg = load_config()

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(kalshi, "get_json", boom)
    assert kalshi.fetch_ranges(cfg, "2026-07-19") is None


def test_fetch_ranges_returns_none_on_bad_shape(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(kalshi, "get_json", lambda *a, **k: {"unexpected": "shape"})
    # missing "markets" key entirely -> parse_markets sees no markets -> None
    assert kalshi.fetch_ranges(cfg, "2026-07-19") is None


def test_fetch_ranges_uses_parsed_payload(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(kalshi, "get_json", lambda *a, **k: MARKETS_PAYLOAD)
    ranges = kalshi.fetch_ranges(cfg, "2026-07-19")
    assert ranges is not None
    assert len(ranges) == 6


def test_fetch_ranges_uses_low_series_when_low_metric_passed(monkeypatch):
    cfg = load_config()
    captured = {}

    def fake_get_json(url, params, *a, **k):
        captured["series_ticker"] = params.get("series_ticker")
        return LOW_MARKETS_PAYLOAD

    monkeypatch.setattr(kalshi, "get_json", fake_get_json)

    low = get_metric(cfg, "low")
    ranges = kalshi.fetch_ranges(cfg, "2026-07-23", low)
    assert captured["series_ticker"] == "KXLOWTLAX"
    assert captured["series_ticker"] == cfg.station.kalshi_low_series
    assert ranges is not None
    assert len(ranges) == 3


def test_fetch_ranges_without_metric_still_uses_high_series(monkeypatch):
    cfg = load_config()
    captured = {}

    def fake_get_json(url, params, *a, **k):
        captured["series_ticker"] = params.get("series_ticker")
        return MARKETS_PAYLOAD

    monkeypatch.setattr(kalshi, "get_json", fake_get_json)

    ranges = kalshi.fetch_ranges(cfg, "2026-07-19")
    assert captured["series_ticker"] == cfg.station.kalshi_series
    assert captured["series_ticker"] == "KXHIGHLAX"
    assert ranges is not None
    assert len(ranges) == 6


@pytest.mark.live
def test_live_fetch_ranges_today_has_real_prices():
    cfg = load_config()
    from datetime import date

    ranges = kalshi.fetch_ranges(cfg, date.today().isoformat())
    assert ranges is not None
    assert len(ranges) >= 4
    assert all(r["label"] for r in ranges)
    assert any(r["last_price"] is not None or r["yes_bid"] is not None for r in ranges)
