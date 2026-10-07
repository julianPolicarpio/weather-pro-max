from contextlib import closing

import pytest

from sandbox.book import NO_FILL, Fill
from sandbox.db import connect
from sandbox.ledger import START_CASH, cash, holds, open_account, place
from sandbox.models import Intent

T = "2026-10-04T10:00:00+00:00"


def intent(ticker="A", account="mos"):
    return Intent(account=account, ticker=ticker, side="yes", contracts=3,
                  price_cap=0.65, snapshot_id=1, p_bot=0.7, p_market=0.6)


@pytest.fixture
def conn():
    with closing(connect(":memory:")) as c:
        yield c


def test_new_account_starts_with_five_dollars(conn):
    open_account(conn, "mos", "bot", "label")
    open_account(conn, "mos", "bot", "label")  # idempotent
    assert START_CASH == 5.0
    assert cash(conn, "mos") == pytest.approx(5.0)


def test_fill_debits_cost_and_fee(conn):
    open_account(conn, "mos", "bot")
    fill = Fill("filled", 3, ((0.62, 2.0), (0.64, 1.0)), 1.88, 0.06)
    oid, status = place(conn, intent(), fill, T, '{"no_dollars": []}')
    assert status == "filled" and oid == 1
    assert cash(conn, "mos") == pytest.approx(3.06)
    assert holds(conn, "mos", "A") and not holds(conn, "mos", "B")


def test_insufficient_cash_records_a_rejected_order(conn):
    open_account(conn, "mos", "bot")
    big = Fill("filled", 10, ((0.62, 10.0),), 6.20, 0.17)
    _, status = place(conn, intent(), big, T, None)
    assert status == "insufficient_cash"
    assert cash(conn, "mos") == pytest.approx(5.0)
    assert not holds(conn, "mos", "A")


def test_no_fill_is_recorded_but_costs_nothing(conn):
    open_account(conn, "mos", "bot")
    _, status = place(conn, intent(), NO_FILL, T, None)
    assert status == "none"
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    assert cash(conn, "mos") == pytest.approx(5.0)
