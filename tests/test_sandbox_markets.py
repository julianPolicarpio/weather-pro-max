from sandbox_helpers import EVENT, make_market, ny_event_payload

from sandbox.markets import (normalize_ts, parse_event_date, parse_series_payload,
                             usable_ask, usable_bid)


def test_parse_event_date():
    assert parse_event_date("KXHIGHNY-26OCT04") == "2026-10-04"
    assert parse_event_date("KXLOWTSFO-26JAN31") == "2026-01-31"


def test_normalize_ts():
    assert normalize_ts("2026-10-05T05:00:00Z") == "2026-10-05T05:00:00+00:00"
    assert normalize_ts("2026-10-05T05:00:00+00:00") == "2026-10-05T05:00:00+00:00"
    assert normalize_ts(None) is None


def test_parse_series_payload_builds_markets_with_station_and_sizes():
    ms = parse_series_payload("KXHIGHNY", ny_event_payload())
    assert [m.label for m in ms] == ["≤68", "69–70", "≥71"]
    first = ms[0]
    assert first.ticker == f"{EVENT}-T69"
    assert first.event_date == "2026-10-04"
    assert (first.lo, first.hi) == (None, 68)
    assert first.metric == "high" and first.station == "KNYC"
    assert (first.yes_bid, first.yes_ask) == (0.15, 0.20)
    assert (first.yes_bid_size, first.yes_ask_size) == (10.0, 3.0)
    assert first.close_time == "2026-10-05T05:00:00+00:00"


def test_usable_prices_drop_empty_sides():
    assert usable_ask(make_market("A", ask=1.0)) is None   # no seller
    assert usable_bid(make_market("A", bid=0.0)) is None   # no buyer
    assert usable_ask(make_market("A", ask=0.24)) == 0.24


def test_parse_series_payload_dedupes_repeated_markets():
    payload = ny_event_payload()
    payload["markets"].append(dict(payload["markets"][0]))   # same market listed twice
    ms = parse_series_payload("KXHIGHNY", payload)
    assert [m.label for m in ms] == ["≤68", "69–70", "≥71"]
