# tests/test_sandbox_views.py
from contextlib import closing

import pytest
from sandbox_helpers import EVENT, make_market, three_buckets

from sandbox import views
from sandbox.cycle import ensure_accounts
from sandbox.db import connect
from sandbox.ledger import cash
from sandbox.models import Snapshot
from sandbox.recorder import load_snapshot, record_snapshot

NOW = "2026-10-04T12:00:00+00:00"
T69, B69, T70 = f"{EVENT}-T69", f"{EVENT}-B69.5", f"{EVENT}-T70"
OLD = "KXHIGHNY-26OCT03-T69"


@pytest.fixture
def conn(tmp_path):
    with closing(connect(tmp_path / "s.db")) as c:
        ensure_accounts(c)
        yield c


def _order(conn, account, ticker, side="yes", n=1, cost=0.20, fee=0.01, payout=None):
    conn.execute(
        "INSERT INTO orders(account, ticker, side, contracts_requested, price_cap, snapshot_id, decided_at, "
        "status, filled_contracts, cost, fee, levels_json, payout) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (account, ticker, side, n, 0.5, None, NOW, "filled", n, cost, fee, "[]", payout))
    conn.commit()


def _snap(conn, markets, taken=NOW):
    return load_snapshot(conn, record_snapshot(conn, markets, taken))


def _old_market():
    return make_market(OLD, None, 68, event="KXHIGHNY-26OCT03", close="2026-10-04T05:00:00+00:00")


def test_text_helpers():
    assert views.city("KNYC") == "New York" and views.city("KXXX") == "KXXX" and views.city(None) == "Unknown"
    assert views.bucket_text(None, 68) == "≤68°" and views.bucket_text(69, 70) == "69–70°"
    assert views.bucket_text(71, None) == "≥71°"
    assert views.title("KMIA", "high", 91, 92, "2026-10-11") == "Miami high 91–92° · Oct 11"
    assert views.until("2026-10-05T05:00:00+00:00", NOW) == "closes in 17h"
    assert views.until("2026-10-04T12:30:00+00:00", NOW) == "closes in 30m"
    assert views.until("2026-10-04T05:00:00+00:00", NOW) == "closed"
    assert views.until(None, NOW) == "close time unknown"
    assert views.day_text("2026-10-11") == "Sunday, Oct 11"


def test_freshness_states():
    assert views.freshness(None, NOW) == {"state": "none", "minutes": None}
    assert views.freshness(Snapshot(1, "2026-10-04T11:50:00+00:00", ()), NOW) == {"state": "fresh", "minutes": 10}
    assert views.freshness(Snapshot(1, "2026-10-04T11:30:00+00:00", ()), NOW)["state"] == "fresh"
    assert views.freshness(Snapshot(1, "2026-10-04T11:29:00+00:00", ()), NOW) == {"state": "stale", "minutes": 31}


def test_previous_snapshot_and_current_day(conn):
    first = _snap(conn, three_buckets(), "2026-10-04T11:00:00+00:00")
    second = _snap(conn, three_buckets() + [_old_market()])
    assert views.previous_snapshot(conn, second).id == first.id
    assert views.previous_snapshot(conn, first) is None and views.previous_snapshot(conn, None) is None
    assert views.current_day(second, NOW) == "2026-10-04"
    assert views.current_day(second, "2026-10-06T00:00:00+00:00") == "2026-10-04"   # all closed: latest day
    assert views.current_day(None, NOW) is None


def test_open_positions_split_live_and_awaiting_with_marks(conn):
    snap = _snap(conn, three_buckets() + [_old_market()])
    _order(conn, "citycast", B69, "yes", cost=0.20, fee=0.01)
    _order(conn, "citycast", T69, "no", cost=0.70, fee=0.01)
    _order(conn, "citycast", OLD, "yes", cost=0.30, fee=0.01)
    _order(conn, "citycast", T70, "yes", cost=0.30, fee=0.02, payout=1.0)     # settled: not open
    _order(conn, "mos", B69, "yes")                                          # another account
    pos = views.open_positions(conn, "citycast", snap, NOW)
    assert pos["count"] == 3 and pos["at_risk"] == 1.23
    live, awaiting = pos["live"], pos["awaiting"]
    assert [r["ticker"] for r in live] == [B69, T69] and [r["ticker"] for r in awaiting] == [OLD]
    assert (live[0]["paid_c"], live[0]["mark_c"], live[0]["unrealized_c"]) == (20, 25, 5)      # YES mark = bid
    assert (live[1]["paid_c"], live[1]["mark_c"], live[1]["unrealized_c"]) == (70, 80, 10)     # NO mark = 1 - ask
    assert live[0]["status"] == "closes in 17h" and live[0]["title"] == "New York high 69–70° · Oct 4"
    assert awaiting[0]["mark_c"] is None and awaiting[0]["unrealized_c"] is None
    assert awaiting[0]["status"] == "settles when the NWS report posts"


def test_live_position_without_a_quote_has_no_mark(conn):
    snap = _snap(conn, three_buckets()[:2])
    conn.execute("INSERT INTO markets(ticker, series, event_ticker, event_date, lo, hi, label, metric, station, "
                 "close_time) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (T70, "KXHIGHNY", EVENT, "2026-10-04", 71, None, "≥71", "high", "KNYC", "2026-10-05T05:00:00+00:00"))
    conn.commit()
    _order(conn, "citycast", T70)
    (row,) = views.open_positions(conn, "citycast", snap, NOW)["live"]
    assert row["mark_c"] is None and row["unrealized_c"] is None and row["status"] == "closes in 17h"


def test_settled_positions_bot_summaries_and_held(conn):
    _snap(conn, three_buckets() + [_old_market()])
    _order(conn, "citycast", B69, cost=0.20, fee=0.01)
    _order(conn, "citycast", T70, cost=0.30, fee=0.02, payout=1.0)
    (s,) = views.settled_positions(conn, "citycast")
    assert (s["ticker"], s["paid_c"], s["payout"], s["realized"]) == (T70, 30, 1.0, 0.68)
    bots = views.bot_summaries(conn)
    assert [b["name"] for b in bots] == ["favorite", "mos", "highcast", "citycast"]
    assert [b["name"] for b in views.bot_summaries(conn, show_all=True)][-2:] == ["gate", "manual"]
    cc = next(b for b in bots if b["name"] == "citycast")
    assert (cc["open"], cc["at_risk"], cc["settled"], cc["profit"], cc["cash"]) == (1, 0.21, 1, 0.68, 5.47)
    assert views.held_tickers(conn, "citycast") == {B69} and views.held_tickers(conn, "mos") == set()


def _moving_pair(conn):
    prev = _snap(conn, three_buckets(), "2026-10-04T11:00:00+00:00")
    cur = _snap(conn, three_buckets(bids=(0.05, 0.35, 0.40), asks=(0.10, 0.42, 0.45)))
    return prev, cur


def test_biggest_moves_order_and_threshold(conn):
    prev, cur = _moving_pair(conn)
    moves = views.biggest_moves(cur, prev, "2026-10-04")
    assert [m["ticker"] for m in moves] == [B69, T69, T70]
    assert round(moves[0]["change"], 1) == 12.1 and round(moves[1]["change"], 1) == -11.5
    assert moves[0]["title"] == "New York high 69–70°"
    assert views.biggest_moves(cur, prev, "2026-10-04", n=1)[0]["ticker"] == B69
    assert views.biggest_moves(cur, None, "2026-10-04") == []
    assert views.biggest_moves(cur, prev, "2026-10-05") == []
    same = _snap(conn, three_buckets(bids=(0.05, 0.35, 0.40), asks=(0.10, 0.42, 0.45)), "2026-10-04T12:15:00+00:00")
    assert views.biggest_moves(same, cur, "2026-10-04") == []          # below MOVE_MIN_PTS


def test_market_cards_filters_held_and_moves(conn):
    prev, cur = _moving_pair(conn)
    (card,) = views.market_cards(cur, prev, {B69}, "2026-10-04", now_iso=NOW)
    assert card["city"] == "New York" and card["metric"] == "high" and card["closes"] == "closes in 17h"
    assert card["question"] == "Highest temperature in New York on Oct 4?"
    assert [r["label"] for r in card["rungs"]] == ["≤68°", "69–70°", "≥71°"]
    assert [r["held"] for r in card["rungs"]] == [False, True, False] and card["any_held"]
    assert (card["rungs"][1]["bid_c"], card["rungs"][1]["ask_c"]) == (35, 42)
    assert round(card["max_move"], 1) == 12.1
    assert views.market_cards(cur, prev, set(), "2026-10-04", metric="low", now_iso=NOW) == []
    assert views.market_cards(cur, prev, set(), "2026-10-04", station="KMIA", now_iso=NOW) == []
    assert views.market_cards(cur, prev, set(), "2026-10-05", now_iso=NOW) == []
    assert views.market_cards(None, None, set(), "2026-10-04") == []
    (no_prev,) = views.market_cards(cur, None, set(), "2026-10-04", now_iso=NOW)
    assert no_prev["max_move"] is None


def test_out_of_cash(conn):
    snap = _snap(conn, three_buckets())                 # cheapest usable ask 0.20
    assert views.out_of_cash(0.0, snap) and views.out_of_cash(0.15, snap)
    assert not views.out_of_cash(0.20, snap) and not views.out_of_cash(1.0, snap)
    assert views.out_of_cash(0.005, None) and not views.out_of_cash(0.5, None)


def _ladder(event, tickers, mids, station="KNYC", series="KXHIGHNY"):
    """Two-rung ladder with exactly representable mids (binary fractions), so changes compare exactly."""
    lo_hi = ((None, 68), (69, None))
    return [make_market(t, lo, hi, v - 0.0625, v + 0.0625, event=event, series=series, station=station)
            for t, (lo, hi), v in zip(tickers, lo_hi, mids)]


def test_rung_losing_its_quote_makes_no_phantom_move(conn):
    prev = _snap(conn, three_buckets(), "2026-10-04T11:00:00+00:00")
    # T70 loses its quote: devig renormalises the other two rungs, which is not a price move.
    cur = _snap(conn, three_buckets(bids=(0.15, 0.25, None), asks=(0.20, 0.30, None)))
    assert views.biggest_moves(cur, prev, "2026-10-04") == []
    (card,) = views.market_cards(cur, prev, set(), "2026-10-04", now_iso=NOW)
    assert card["max_move"] is None
    # ... and the reverse: a quote appearing between snapshots is equally not a move
    assert views.biggest_moves(prev, cur, "2026-10-04") == []
    (card,) = views.market_cards(prev, cur, set(), "2026-10-04", now_iso=NOW)
    assert card["max_move"] is None


def test_rung_loss_in_one_event_leaves_other_events_moves(conn):
    other = "KXHIGHMIA-26OCT04"
    a, b = f"{other}-A", f"{other}-B"
    prev = _snap(conn, three_buckets() + _ladder(other, [a, b], (0.25, 0.75), "KMIA", "KXHIGHMIA"),
                 "2026-10-04T11:00:00+00:00")
    cur = _snap(conn, three_buckets(bids=(0.15, 0.25, None), asks=(0.20, 0.30, None))
                + _ladder(other, [a, b], (0.5, 0.5), "KMIA", "KXHIGHMIA"))
    assert {m["ticker"] for m in views.biggest_moves(cur, prev, "2026-10-04")} == {a, b}


def test_move_ties_break_by_current_p_then_ticker(conn):
    other = "KXHIGHMIA-26OCT04"
    a1, a2, b1, b2 = f"{EVENT}-A1", f"{EVENT}-A2", f"{other}-B1", f"{other}-B2"
    prev = _snap(conn, _ladder(EVENT, [a1, a2], (0.25, 0.75))
                 + _ladder(other, [b1, b2], (0.125, 0.875), "KMIA", "KXHIGHMIA"), "2026-10-04T11:00:00+00:00")
    cur = _snap(conn, _ladder(EVENT, [a1, a2], (0.5, 0.5))
                + _ladder(other, [b1, b2], (0.375, 0.625), "KMIA", "KXHIGHMIA"))
    moves = views.biggest_moves(cur, prev, "2026-10-04")
    assert [abs(m["change"]) for m in moves] == [25.0] * 4
    assert [m["ticker"] for m in moves] == [b2, a1, a2, b1]      # p .625, .5, .5 (ticker order), .375


def test_card_max_move_below_threshold_is_none(conn):
    prev = _snap(conn, three_buckets(), "2026-10-04T11:00:00+00:00")
    cur = _snap(conn, three_buckets(bids=(0.15, 0.25, 0.40), asks=(0.20, 0.30, 0.4501)))
    (card,) = views.market_cards(cur, prev, set(), "2026-10-04", now_iso=NOW)
    assert views.biggest_moves(cur, prev, "2026-10-04") == [] and card["max_move"] is None


def test_empty_snapshot(conn):
    empty = Snapshot(1, NOW, ())
    assert views.current_day(empty, NOW) is None
    assert views.market_cards(empty, empty, set(), "2026-10-04", now_iso=NOW) == []
    assert views.biggest_moves(empty, empty, "2026-10-04") == []
    _snap(conn, three_buckets())
    _order(conn, "citycast", B69)
    (row,) = views.open_positions(conn, "citycast", None, NOW)["live"]
    assert row["mark_c"] is None and row["unrealized_c"] is None and row["status"] == "closes in 17h"


def _fill(conn, account, ticker, requested, filled, cost, fee, status):
    conn.execute(
        "INSERT INTO orders(account, ticker, side, contracts_requested, price_cap, snapshot_id, decided_at, "
        "status, filled_contracts, cost, fee, levels_json, payout) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
        (account, ticker, "yes", requested, 0.5, None, NOW, status, filled, cost, fee, "[]"))
    conn.commit()


def test_partial_and_zero_fills(conn):
    snap = _snap(conn, three_buckets())
    _fill(conn, "citycast", B69, 3, 2, 0.40, 0.02, "partial")
    _fill(conn, "citycast", T70, 2, 0, 0.0, 0.0, "insufficient_cash")
    pos = views.open_positions(conn, "citycast", snap, NOW)
    (row,) = pos["live"]
    assert pos["awaiting"] == [] and pos["count"] == 1 and pos["at_risk"] == 0.42
    assert (row["ticker"], row["n"], row["paid_c"], row["mark_c"], row["unrealized_c"]) == (B69, 2, 20, 25, 10)
    assert views.settled_positions(conn, "citycast") == []
    assert views.held_tickers(conn, "citycast") == {B69}
    cc = next(b for b in views.bot_summaries(conn) if b["name"] == "citycast")
    assert (cc["open"], cc["at_risk"], cc["cash"]) == (1, 0.42, round(cash(conn, "citycast"), 2)) and cc["cash"] == 4.58
