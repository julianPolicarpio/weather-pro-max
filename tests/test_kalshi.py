from highcast.fetchers import kalshi


def test_a_live_two_sided_book_beats_a_stale_last_trade():
    """MEASURED 2026-07-31: 16.1% of stored quotes (4,251 of 26,460) sat OUTSIDE
    their own bid/ask because `last_price` was preferred unconditionally. The
    median deviation is only 1c and it did NOT explain the losing bets — but the
    worst was 52c, and an unattended bot must never act on a stale print."""
    r = {"last_price": 0.63, "yes_bid": 0.10, "yes_ask": 0.11}
    assert kalshi.resolve_price(r) == 0.105


def test_last_price_is_used_when_there_is_no_book():
    r = {"last_price": 0.42, "yes_bid": None, "yes_ask": None}
    assert kalshi.resolve_price(r) == 0.42


def test_a_crossed_book_is_rejected_in_favour_of_the_last_trade():
    """ask < bid is nonsense; trusting its midpoint would invent a price."""
    r = {"last_price": 0.50, "yes_bid": 0.60, "yes_ask": 0.40}
    assert kalshi.resolve_price(r) == 0.50


def test_a_single_sided_book_is_used_when_nothing_else_exists():
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": 0.33, "yes_ask": None}) == 0.33
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": None, "yes_ask": 0.37}) == 0.37


def test_no_information_at_all_returns_none():
    assert kalshi.resolve_price(
        {"last_price": None, "yes_bid": None, "yes_ask": None}) is None
