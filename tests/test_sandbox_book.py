import pytest
from sandbox_helpers import book_payload

from sandbox.book import NO_FILL, parse_book, walk

# NO bids 0.38 x2, 0.36 x5, 0.30 x10  ->  YES asks 0.62 x2, 0.64 x5, 0.70 x10
BOOK = parse_book(book_payload(no=[(0.38, 2), (0.36, 5), (0.30, 10)]))


def test_parse_book_reads_bids():
    assert BOOK.no == ((0.38, 2.0), (0.36, 5.0), (0.30, 10.0))
    assert BOOK.yes == ()


def test_buy_yes_walks_no_bids_cheapest_first():
    f = walk(BOOK, "yes", 3, 0.65)
    assert f.status == "filled" and f.contracts == 3
    assert f.levels == ((0.62, 2.0), (0.64, 1.0))
    assert f.cost == pytest.approx(1.88)
    # per level, rounded up: 0.07*2*0.62*0.38=0.032984 -> 0.04 ; 0.07*0.64*0.36=0.016128 -> 0.02
    assert f.fee == pytest.approx(0.06)
    assert f.avg_price == pytest.approx(1.88 / 3)


def test_price_cap_makes_a_partial_fill():
    f = walk(BOOK, "yes", 3, 0.62)
    assert f.status == "partial" and f.contracts == 2
    assert f.cost == pytest.approx(1.24) and f.fee == pytest.approx(0.04)


def test_nothing_under_the_cap_is_no_fill():
    assert walk(BOOK, "yes", 1, 0.60) == NO_FILL


def test_buy_no_walks_yes_bids():
    book = parse_book(book_payload(yes=[(0.55, 4)]))
    f = walk(book, "no", 1, 0.50)
    assert f.levels == ((0.45, 1.0),)
    assert f.cost == pytest.approx(0.45) and f.fee == pytest.approx(0.02)


def test_fractional_book_sizes_still_fill_whole_contracts():
    book = parse_book(book_payload(no=[(0.40, 1.5), (0.39, 3)]))
    f = walk(book, "yes", 2, 0.61)
    assert f.contracts == 2
    assert f.levels == ((0.60, 1.5), (0.61, 0.5))
    assert f.cost == pytest.approx(1.205)
    assert f.fee == pytest.approx(0.04)  # 0.0252 -> 0.03 ; 0.0083 -> 0.01
    g = walk(book, "yes", 2, 0.60)        # only 1.5 available under the cap
    assert g.status == "partial" and g.contracts == 1 and g.levels == ((0.60, 1.0),)


def test_empty_payload_is_no_fill():
    assert walk(parse_book({}), "yes", 1, 0.99) == NO_FILL


def test_walk_sorts_bids_regardless_of_input_order():
    book = parse_book(book_payload(no=[(0.30, 10), (0.36, 5), (0.38, 2)]))
    f = walk(book, "yes", 3, 0.65)
    assert f.levels == ((0.62, 2.0), (0.64, 1.0))
    assert f.cost == pytest.approx(1.88)
    assert f.fee == pytest.approx(0.06)


def test_walk_rejects_an_unknown_side():
    with pytest.raises(ValueError):
        walk(BOOK, "YES", 1, 0.65)
