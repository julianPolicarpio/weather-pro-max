"""Kalshi fee arithmetic. The load-bearing test is `test_reproduces_kalshi_example`
-- everything else in Pick-of-the-Day rests on that formula being right."""
import math

import pytest

from highcast import fees


def test_reproduces_kalshi_published_example():
    """10 contracts at 65c -> 0.15925 raw -> $0.16 after rounding UP.

    This is the number Kalshi's own calculator shows. It is what pins the
    rounding DIRECTION: 0.15925 rounds to 0.16 up but 0.16 either way under
    round-half-up, so the discriminating part is that we never round DOWN.
    """
    assert fees.taker_fee(10, 0.65) == pytest.approx(0.16)


def test_never_rounds_down():
    """A raw fee just above a cent boundary must bill the higher cent."""
    # 1 contract at 0.58 -> 0.07*0.58*0.42 = 0.017052 -> must be 0.02, not 0.01
    assert fees.taker_fee(1, 0.58) == pytest.approx(0.02)
    # 1 contract at 0.90 -> 0.0063 -> 0.01, not 0.00
    assert fees.taker_fee(1, 0.90) == pytest.approx(0.01)


def test_float_dust_does_not_bill_an_extra_cent():
    """The reason `_round_up_cents` rounds to 9 places before ceiling.

    A fee that is EXACTLY one cent must bill one cent. Without the guard, binary
    representation can put the product a hair above 0.01 and `ceil` would charge
    two. Constructed so the raw value is exactly 0.01: rate*C*P*(1-P) with
    C chosen to land on the boundary.
    """
    # 0.07 * 4 * 0.5 * 0.5 = 0.07 exactly -> 7 cents, not 8
    assert fees.taker_fee(4, 0.5) == pytest.approx(0.07)
    assert fees._round_up_cents(0.01) == pytest.approx(0.01)
    assert fees._round_up_cents(0.0100000000000001) == pytest.approx(0.01)


def test_max_fee_is_1_75_cents_per_contract_at_fifty():
    """Kalshi's documented maximum. Exact, so it is a clean invariant."""
    assert fees._raw(fees.TAKER_RATE, 1, 0.50) == pytest.approx(0.0175)


def test_fee_is_symmetric_about_fifty_cents():
    for p in (0.10, 0.25, 0.40, 0.49):
        assert fees._raw(fees.TAKER_RATE, 1, p) == pytest.approx(
            fees._raw(fees.TAKER_RATE, 1, 1.0 - p))


def test_maker_is_a_quarter_of_taker_before_rounding():
    for p in (0.20, 0.50, 0.80):
        assert fees._raw(fees.MAKER_RATE, 100, p) == pytest.approx(
            0.25 * fees._raw(fees.TAKER_RATE, 100, p))


def test_rounding_is_per_order_so_per_contract_cost_falls_with_size():
    """The detail that matters at Julian's scale.

    A 1-contract order at 58c pays the whole $0.02. A 2-contract order pays $0.04
    total, i.e. $0.02 each -- but at 10 contracts the round-up is amortised and
    the per-contract cost drops below it. Asserting the DIRECTION rather than
    exact values keeps this honest if the rate ever changes.
    """
    one = fees.fee_per_contract(1, 0.58)
    ten = fees.fee_per_contract(10, 0.58)
    assert one > ten
    # and the order-level fee is never cheaper for fewer contracts
    assert fees.taker_fee(10, 0.58) > fees.taker_fee(1, 0.58)


def test_fee_per_contract_does_not_re_round():
    """10 at 50c: raw 0.175 -> order $0.18 -> $0.018 each, NOT $0.02.

    The per-contract figure is deliberately a fraction of a cent. Re-rounding it
    would bill $0.02 each and overstate the drag by 11%.
    """
    assert fees.taker_fee(10, 0.50) == pytest.approx(0.18)
    assert fees.fee_per_contract(10, 0.50) == pytest.approx(0.018)
    assert fees.fee_per_contract(10, 0.50) < 0.02


def test_breakeven_matches_the_hand_computed_table():
    """These are the figures quoted to Julian; pin them so they cannot drift.

    Each is 0.3-1.0 point above the unrounded arithmetic -- the whole point of
    modelling the round-up.
    """
    assert fees.breakeven(0.80) == pytest.approx(0.82)
    assert fees.breakeven(0.87) == pytest.approx(0.88)
    assert fees.breakeven(0.90) == pytest.approx(0.91)
    assert fees.breakeven(0.95) == pytest.approx(0.96)
    assert fees.breakeven(0.40) == pytest.approx(0.42)


def test_breakeven_at_their_average_entry():
    """nowcast.trade's disclosed 71.6c average entry.

    73.0% UNROUNDED, but 73.6% at 1 contract once the per-order round-up is
    charged -- and 1 contract is Julian's actual size. Our best commit-hour hit is
    KLAX LOW at ~62%, so their price point is firmly negative EV on our markets.
    That is why the Pick-of-the-Day bar is built from our measured numbers rather
    than their disclosed one.
    """
    assert fees.breakeven(0.716, contracts=1) == pytest.approx(0.736, abs=1e-9)
    # unrounded, i.e. the large-order limit
    raw_be = 0.716 + fees._raw(fees.TAKER_RATE, 1, 0.716)
    assert raw_be == pytest.approx(0.7302, abs=1e-4)
    # either way it is above our best measured commit-hour hit rate
    assert fees.breakeven(0.716) > 0.624


def test_no_settlement_fee_means_breakeven_is_price_plus_entry_only():
    p = 0.62
    assert fees.breakeven(p) == pytest.approx(p + fees.fee_per_contract(1, p))


def test_net_ev_sign_flips_at_breakeven():
    p = 0.62
    be = fees.breakeven(p)
    assert fees.net_ev(be + 0.01, p) > 0
    assert fees.net_ev(be - 0.01, p) < 0
    assert fees.net_ev(be, p) == pytest.approx(0.0, abs=1e-12)


def test_maker_beats_taker_for_the_same_order():
    assert fees.maker_fee(100, 0.5) < fees.taker_fee(100, 0.5)
    assert fees.breakeven(0.5, 100, maker=True) < fees.breakeven(0.5, 100)


def test_zero_or_negative_contracts_is_free_not_an_error():
    """`staking.size` legitimately returns 0 contracts when the stake cap cannot
    afford one, and callers compute EV before checking. Zero must not raise."""
    assert fees.taker_fee(0, 0.5) == 0.0
    assert fees.fee_per_contract(0, 0.5) == 0.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_price_outside_the_open_unit_interval_raises(bad):
    """A price of exactly 0 or 1 is a settled market, not a tradeable one, and
    `market_price` has been observed outside its own bid/ask in this project --
    so silently returning a fee for a nonsense price would hide a data bug."""
    with pytest.raises(ValueError):
        fees.taker_fee(1, bad)
