import logging
from dataclasses import replace

import pytest
from sandbox_helpers import three_buckets

from sandbox.book import NO_FILL, Fill
from sandbox.db import connect
from sandbox.ledger import cash, open_account, place
from sandbox.models import Candidate, Intent
from sandbox.recorder import record_candidates, record_snapshot
from sandbox.settle import settle_due

T = "2026-10-04T10:00:00+00:00"
AFTER = "2026-10-05T12:00:00+00:00"      # markets close 2026-10-05T05:00:00+00:00
A, B, C = (m.ticker for m in three_buckets())
_OPEN = []


@pytest.fixture(autouse=True)
def _close_connections():
    """An unclosed sqlite3 connection raises ResourceWarning whenever it is finalized,
    which under -W error lands on whichever unrelated test happens to be running."""
    yield
    while _OPEN:
        _OPEN.pop().close()


def candidate(ticker, placed, ask, fpc, sid):
    return Candidate(bot="mos", ticker=ticker, series="KXHIGHNY", station="KNYC",
                     metric="high", event_date="2026-10-04", snapshot_id=sid,
                     decided_at=T, p_bot=0.4, pmf_json="{}", p_market=0.3,
                     yes_bid=None, yes_ask=ask, spread=None, ask_size=None,
                     minutes_to_close=None, fee_per_contract=fpc, edge_after_fee=0.1,
                     placed=placed, reason="edge" if placed else "below_min_edge")


def setup():
    conn = connect(":memory:")
    sid = record_snapshot(conn, three_buckets(), T)
    open_account(conn, "mos", "bot")
    place(conn, Intent("mos", A, "yes", 1, 0.20, sid, 0.4, 0.3),
          Fill("filled", 1, ((0.20, 1.0),), 0.20, 0.02), T, None)
    record_candidates(conn, [candidate(A, True, 0.20, 0.02, sid),
                             candidate(B, False, 0.30, 0.02, sid)])
    _OPEN.append(conn)
    return conn


def get_settled(path, params=None):
    assert path == "/markets" and params["status"] == "settled"
    return {"markets": [
        {"ticker": A, "status": "finalized", "result": "yes"},
        {"ticker": B, "status": "finalized", "result": "no"},
        {"ticker": C, "status": "finalized", "result": "no"},
    ]}


def test_settle_pays_winners_once():
    conn = setup()
    assert settle_due(conn, get_settled, AFTER) == 3
    assert cash(conn, "mos") == pytest.approx(5.0 - 0.22 + 1.0)
    assert settle_due(conn, get_settled, AFTER) == 0          # idempotent
    assert cash(conn, "mos") == pytest.approx(5.78)


def test_skipped_candidates_are_labelled_too():
    conn = setup()
    settle_due(conn, get_settled, AFTER)
    rows = {r["ticker"]: (r["result"], r["asif_profit"])
            for r in conn.execute("SELECT ticker, result, asif_profit FROM candidates")}
    assert rows[A] == ("yes", pytest.approx(1 - 0.20 - 0.02))
    assert rows[B] == ("no", pytest.approx(-0.30 - 0.02))


def test_settlement_only_touches_outcome_fields():
    conn = setup()
    cols = "bot, ticker, snapshot_id, p_bot, pmf_json, yes_ask, fee_per_contract, placed, reason"
    before = [tuple(r) for r in conn.execute(f"SELECT {cols} FROM candidates ORDER BY id")]
    settle_due(conn, get_settled, AFTER)
    after = [tuple(r) for r in conn.execute(f"SELECT {cols} FROM candidates ORDER BY id")]
    assert before == after


def test_nothing_is_due_before_close():
    conn = setup()
    calls = []
    assert settle_due(conn, lambda p, q=None: calls.append(p) or {}, T) == 0
    assert calls == []


def snapshot_id(conn):
    return conn.execute("SELECT id FROM snapshots").fetchone()["id"]


def payout(conn, ticker):
    return conn.execute("SELECT payout FROM orders WHERE ticker=?", (ticker,)).fetchone()["payout"]


def market_result(conn, ticker):
    return conn.execute("SELECT result FROM markets WHERE ticker=?", (ticker,)).fetchone()["result"]


def test_losing_order_pays_zero():
    conn = setup()
    place(conn, Intent("mos", B, "yes", 1, 0.30, snapshot_id(conn), 0.4, 0.3),
          Fill("filled", 1, ((0.30, 1.0),), 0.30, 0.02), T, None)
    settle_due(conn, get_settled, AFTER)
    assert payout(conn, B) == 0.0                       # settled and lost, not left NULL
    assert payout(conn, A) == 1.0
    assert cash(conn, "mos") == pytest.approx(5 - 0.22 - 0.32 + 1.0)


def test_unfilled_order_keeps_null_payout():
    conn = setup()
    place(conn, Intent("mos", C, "yes", 1, 0.45, snapshot_id(conn), 0.4, 0.3), NO_FILL, T, None)
    settle_due(conn, get_settled, AFTER)
    assert market_result(conn, C) == "no"               # the market did settle
    assert payout(conn, C) is None


def test_fetch_failure_is_logged_and_retried(caplog):
    conn = setup()

    def broken(path, params=None):
        raise RuntimeError("network down")

    with caplog.at_level(logging.WARNING, logger="sandbox.settle"):
        assert settle_due(conn, broken, AFTER) == 0
    assert [r.levelno for r in caplog.records if r.name == "sandbox.settle"] == [logging.WARNING]
    assert conn.execute("SELECT COUNT(*) FROM markets WHERE result IS NOT NULL").fetchone()[0] == 0
    assert settle_due(conn, get_settled, AFTER) == 3    # nothing was lost; next run settles


def test_non_final_rows_write_nothing(caplog):
    conn = setup()
    rows = lambda path, params=None: {"markets": [
        {"ticker": A, "status": "active", "result": ""},
        {"ticker": B, "status": "determined", "result": "yes"}]}    # result present, not final
    with caplog.at_level(logging.WARNING, logger="sandbox.settle"):
        assert settle_due(conn, rows, AFTER) == 0
    assert market_result(conn, A) is None and market_result(conn, B) is None
    assert [r for r in caplog.records if r.name == "sandbox.settle"] == []  # not final: no noise


def test_candidate_without_ask_gets_null_asif():
    conn = setup()
    record_candidates(conn, [candidate(C, False, None, None, snapshot_id(conn))])
    settle_due(conn, get_settled, AFTER)
    row = conn.execute("SELECT result, asif_profit FROM candidates WHERE ticker=?", (C,)).fetchone()
    assert row["result"] == "no"
    assert row["asif_profit"] is None


def test_entry_without_ticker_is_ignored():
    conn = setup()
    bad = {"status": "finalized", "result": "yes"}
    assert settle_due(conn, lambda path, params=None: {"markets": [bad]}, AFTER) == 0
    assert market_result(conn, A) is None
    # a malformed entry must not cost the valid rows that follow it in the same series
    mixed = lambda path, params=None: {"markets": [
        bad, {"ticker": A, "status": "finalized", "result": "yes"}]}
    assert settle_due(conn, mixed, AFTER) == 1
    assert market_result(conn, A) == "yes"


def test_non_dict_payload_is_logged_and_skipped(caplog):
    conn = setup()
    with caplog.at_level(logging.WARNING, logger="sandbox.settle"):
        assert settle_due(conn, lambda path, params=None: ["not", "a", "dict"], AFTER) == 0
    assert [r.levelno for r in caplog.records if r.name == "sandbox.settle"] == [logging.WARNING]
    assert settle_due(conn, get_settled, AFTER) == 3    # the series is retried next run


def test_final_row_with_unusable_result_is_warned_about(caplog):
    conn = setup()
    voided = lambda path, params=None: {"markets": [{"ticker": A, "status": "finalized",
                                                     "result": "void"}]}
    with caplog.at_level(logging.WARNING, logger="sandbox.settle"):
        assert settle_due(conn, voided, AFTER) == 0
    messages = [r.getMessage() for r in caplog.records if r.name == "sandbox.settle"]
    assert len(messages) == 1 and A in messages[0] and "void" in messages[0]
    assert market_result(conn, A) is None


# ---- M1: no write transaction is held across a network fetch -----------------------

def test_no_transaction_is_open_while_fetching():
    conn = setup()
    lax = three_buckets(series="KXHIGHLAX", station="KLAX", event="KXHIGHLAX-26OCT04")
    lax = [replace(m, ticker=m.ticker.replace("NY", "LAX")) for m in lax]
    record_snapshot(conn, lax, T)
    open_during_fetch = []

    def get(path, params=None):
        open_during_fetch.append(conn.in_transaction)
        series = params["series_ticker"]
        ms = three_buckets() if series == "KXHIGHNY" else lax
        return {"markets": [{"ticker": m.ticker, "status": "finalized",
                             "result": "yes" if i == 0 else "no"} for i, m in enumerate(ms)]}

    assert settle_due(conn, get, AFTER) == 6
    assert open_during_fetch == [False, False]
    assert not conn.in_transaction
    assert payout(conn, A) == 1.0
