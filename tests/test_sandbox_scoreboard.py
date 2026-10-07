import json
from contextlib import closing

import pytest
from sandbox_helpers import make_market

from sandbox.book import NO_FILL, Fill
from sandbox.db import connect
from sandbox.ledger import open_account, place
from sandbox.models import Candidate, Intent
from sandbox.recorder import record_candidates, record_snapshot
from sandbox.scoreboard import (
    N_EVENTS_REQUIRED,
    N_REQUIRED,
    Scorecard,
    bootstrap_mean_ci,
    cluster_bootstrap_ratio_ci,
    report,
)

T = "2026-10-04T10:00:00+00:00"


def test_bootstrap_of_a_constant_is_that_constant():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_cluster_bootstrap_of_a_constant_ratio_is_that_ratio():
    # every cluster has ratio 1.0, so every resample does too
    assert cluster_bootstrap_ratio_ci([1.0, 2.0, 3.0], [1, 2, 3]) == (1.0, 1.0)


def card(n, p, s, brier_n=None, n_events=30, brier_events=30):
    return Scorecard("mos", 5.0, n, 0.0, None, None, None, p, s,
                     n if brier_n is None else brier_n, n_events, brier_events)


def test_unlock_needs_all_five():
    assert N_REQUIRED == 100
    assert N_EVENTS_REQUIRED == 30
    assert card(100, True, True).unlocked
    assert not card(99, True, True).unlocked
    assert not card(100, False, True).unlocked
    assert not card(100, True, False).unlocked
    assert not card(100, True, True, brier_n=50).unlocked   # too few scored events
    assert not card(99, True, True, brier_n=100).unlocked   # too few settled bets
    assert card(100, True, True, n_events=30).unlocked
    assert not card(100, True, True, n_events=29).unlocked  # too few distinct events
    assert not card(100, True, True, brier_events=29).unlocked  # too few SCORED events


# ---- building settled events directly in the database ----------------------

def _pair(i, a_quote=(0.48, 0.52), b_quote=(0.48, 0.52)):
    ev = f"EV{i}"
    return [make_market(f"{ev}-A", hi=70, bid=a_quote[0], ask=a_quote[1], event=ev),
            make_market(f"{ev}-B", lo=71, bid=b_quote[0], ask=b_quote[1], event=ev)]


def _candidate(ticker, sid, pmf):
    return Candidate(
        bot="mos", ticker=ticker, series="KXHIGHNY", station="KNYC", metric="high",
        event_date="2026-10-04", snapshot_id=sid, decided_at=T, p_bot=pmf[ticker],
        pmf_json=json.dumps(pmf), p_market=0.5, yes_bid=0.48, yes_ask=0.52, spread=0.04,
        ask_size=3.0, minutes_to_close=None, fee_per_contract=0.02, edge_after_fee=0.0,
        placed=True, reason="edge")


def _bet(conn, ticker, sid, pmf, candidate=True, price=0.50):
    """One filled YES contract at `price` + 0.02 fee, plus the candidate row that explains it."""
    _, status = place(conn, Intent("mos", ticker, "yes", 1, price, sid, pmf[ticker], 0.5),
                      Fill("filled", 1, ((price, 1.0),), price, 0.02), T, None)
    assert status == "filled"
    if candidate:
        record_candidates(conn, [_candidate(ticker, sid, pmf)])


def _settle(conn, tickers, winner):
    for t in tickers:
        conn.execute("UPDATE markets SET result=? WHERE ticker=?",
                     ("yes" if t == winner else "no", t))
        conn.execute("UPDATE orders SET payout=? WHERE ticker=?", (1.0 if t == winner else 0.0, t))
    conn.commit()


def _open_rich_account(conn):
    open_account(conn, "mos", "bot")
    # place() refuses a fill the account cannot afford and 120 losing bets cost far more
    # than the $5 start, so give the test account room.
    conn.execute("UPDATE accounts SET start_cash = 1000.0 WHERE name = 'mos'")
    conn.commit()


def _settled_events(conn, k, p_a, a_wins, bets=1, snapshots=1, start=None, buckets=1,
                    b_price=0.50):
    """K NEW events. Each has `snapshots` snapshots with A and B both quoted 0.48/0.52 (market
    0.5/0.5) and `bets` filled YES bets on A at each, settled so A wins or loses. `a_wins` is
    one bool for every event or a per-event sequence of bools.

    Event numbers continue after the events already in the database, so a second call never
    rewrites the tickers, outcomes or payouts a first call settled: tests of window isolation
    depend on that. `start` pins the first event number instead, to bet on an event again
    deliberately. `buckets=2` also fills a YES bet on B (at `b_price`) next to every bet on A,
    two filled buckets for one (snapshot, event)."""
    wins = list(a_wins) if isinstance(a_wins, (list, tuple)) else [a_wins] * k
    _open_rich_account(conn)
    if start is None:
        start = 1 + conn.execute("SELECT COALESCE(MAX(CAST(substr(event_ticker, 3) AS INTEGER)), -1) "
                                 "FROM markets WHERE event_ticker LIKE 'EV%'").fetchone()[0]
    for j in range(k):
        i = start + j
        a, b = f"EV{i}-A", f"EV{i}-B"
        for _ in range(snapshots):
            sid = record_snapshot(conn, _pair(i), T)
            pmf = {a: p_a, b: 1 - p_a}
            for _ in range(bets):
                _bet(conn, a, sid, pmf)
                if buckets == 2:
                    _bet(conn, b, sid, pmf, price=b_price)
        _settle(conn, (a, b), a if wins[j] else b)


# ---- report -----------------------------------------------------------------

def test_report_profit_and_brier_against_the_market_at_bet_time():
    with closing(connect(":memory:")) as conn:
        ms = [make_market("A", bid=0.20, ask=0.24), make_market("B", bid=0.50, ask=0.56),
              make_market("C", bid=0.0, ask=0.30)]
        sid = record_snapshot(conn, ms, T)
        open_account(conn, "mos", "bot")
        place(conn, Intent("mos", "B", "yes", 1, 0.56, sid, 0.8, 0.5),
              Fill("filled", 1, ((0.56, 1.0),), 0.56, 0.02), T, None)
        pmf = {"A": 0.1, "B": 0.8, "C": 0.1}
        record_candidates(conn, [Candidate(
            bot="mos", ticker="B", series="KXHIGHNY", station="KNYC", metric="high",
            event_date="2026-10-04", snapshot_id=sid, decided_at=T, p_bot=0.8,
            pmf_json=json.dumps(pmf), p_market=0.5, yes_bid=0.5, yes_ask=0.56, spread=0.06,
            ask_size=3.0, minutes_to_close=None, fee_per_contract=0.02, edge_after_fee=0.22,
            placed=True, reason="edge")])
        conn.execute("UPDATE markets SET result = CASE ticker WHEN 'B' THEN 'yes' ELSE 'no' END")
        conn.execute("UPDATE orders SET payout = 1.0")
        conn.commit()
        sc = report(conn, "mos")
    assert sc.settled_bets == 1
    assert sc.brier_n == 1
    assert sc.n_events == 1
    assert sc.profit == pytest.approx(0.42)
    # bot brier .06 ; market (.2095, .5048, .2857) brier .3708 -> diff -.3108
    assert sc.brier_diff == pytest.approx(-0.3108, abs=1e-3)
    assert sc.profit_ci is None and not sc.profit_pass   # one event: nothing to resample
    assert sc.brier_ci is None and not sc.score_pass
    assert not sc.unlocked
    assert sc.cash == pytest.approx(5.42)


def test_report_on_an_empty_account():
    with closing(connect(":memory:")) as conn:
        open_account(conn, "favorite", "bot")
        sc = report(conn, "favorite")
    assert (sc.settled_bets, sc.profit, sc.brier_diff, sc.unlocked) == (0, 0.0, None, False)
    assert sc.brier_n == 0
    assert sc.n_events == 0


def test_report_unlocks_when_both_tests_pass():
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 120, p_a=0.9, a_wins=True)
        sc = report(conn, "mos")
    assert sc.settled_bets == 120
    assert sc.brier_n == 120
    assert sc.n_events == 120
    assert sc.profit_pass       # +0.48 a bet, every event
    assert sc.score_pass        # bot .02 vs market .50 on every event
    assert sc.unlocked


def test_report_stays_locked_when_bets_lose_and_score_worse():
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 120, p_a=0.9, a_wins=False)
        sc = report(conn, "mos")
    assert sc.settled_bets == 120
    assert not sc.profit_pass   # -0.52 a bet
    assert not sc.score_pass    # bot 1.62 vs market .50
    assert not sc.unlocked


def test_brier_counts_one_diff_per_event_snapshot():
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0), T)
        pmf = {a: 0.7, b: 0.3}
        _bet(conn, a, sid, pmf)
        _bet(conn, b, sid, pmf)         # same event, same snapshot: same question, same answer
        _settle(conn, (a, b), a)
        sc = report(conn, "mos")
    assert sc.settled_bets == 2
    assert sc.brier_n == 1


def test_market_baseline_uses_the_bets_own_snapshot():
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        first = record_snapshot(conn, _pair(0, (0.48, 0.52), (0.48, 0.52)), T)    # market .5/.5
        record_snapshot(conn, _pair(0, (0.88, 0.92), (0.08, 0.12)),
                        "2026-10-04T11:00:00+00:00")                               # market .9/.1
        _bet(conn, a, first, {a: 0.7, b: 0.3})
        _settle(conn, (a, b), a)
        sc = report(conn, "mos")
    # bot .18 ; market at the bet (.5/.5) .50 -> -.32.  The later .9/.1 market (.02) would give +.16.
    assert sc.brier_n == 1
    assert sc.brier_diff == pytest.approx(-0.32, abs=1e-6)


def test_profit_ci_resamples_events_not_bets():
    # 6 events, 2 bets each; both bets in an event win or lose together (+.48 x2 / -.52 x2).
    # Resampling the 12 bets as if independent would overstate how much evidence that is.
    outcomes = [True, False, True, False, True, False]
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 6, p_a=0.9, a_wins=outcomes, bets=2)
        sc = report(conn, "mos")
        pnl = [r["payout"] - r["cost"] - r["fee"] for r in conn.execute(
            "SELECT payout, cost, fee FROM orders ORDER BY id")]
    assert sc.settled_bets == 12 and sc.brier_n == 6
    lo, hi = sc.profit_ci
    iid_lo, iid_hi = bootstrap_mean_ci(pnl)
    assert hi - lo > 1.2 * (iid_hi - iid_lo)
    assert lo < 0 < hi                     # a coin-flip record is not evidence of an edge
    assert not sc.profit_pass
    assert sc.brier_ci[0] < 0 < sc.brier_ci[1]
    assert not sc.score_pass


def test_brier_ci_resamples_events_not_snapshots():
    # 6 events, each scored at 2 snapshots with the same answer: 12 diffs but only 6 outcomes.
    outcomes = [True, False, True, False, True, False]
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 6, p_a=0.9, a_wins=outcomes, snapshots=2)
        sc = report(conn, "mos")
    assert sc.brier_n == 12
    diffs = [-0.48 if w else 1.12 for w in outcomes for _ in range(2)]   # bot .02/1.62 vs market .5
    iid_lo, iid_hi = bootstrap_mean_ci(diffs)
    lo, hi = sc.brier_ci
    assert hi - lo > 1.2 * (iid_hi - iid_lo)


# ---- what counts toward the gate --------------------------------------------

def test_unsettled_sibling_excludes_the_event_from_brier():
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0), T)
        _bet(conn, a, sid, {a: 0.9, b: 0.1})
        conn.execute("UPDATE markets SET result='yes' WHERE ticker=?", (a,))   # B stays NULL
        conn.execute("UPDATE orders SET payout=1.0 WHERE ticker=?", (a,))
        conn.commit()
        sc = report(conn, "mos")
    assert sc.settled_bets == 1       # the bet itself is settled
    assert sc.brier_n == 0            # but the event's winner is not known for sure yet
    assert sc.brier_diff is None


def test_bet_without_candidate_is_not_scored():
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0), T)
        _bet(conn, a, sid, {a: 0.9, b: 0.1}, candidate=False)    # no pmf was ever logged
        _settle(conn, (a, b), a)
        sc = report(conn, "mos")
    assert sc.settled_bets == 1
    assert sc.brier_n == 0
    assert sc.brier_diff is None


def test_unquoted_ladder_is_not_scored():
    # No usable quote means no market probabilities. Scoring against {} would hand the bot
    # a free "beat the market" diff (multiclass_brier({}, w) is 1.0).
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0, (None, None), (None, None)), T)
        _bet(conn, a, sid, {a: 0.9, b: 0.1})
        _settle(conn, (a, b), a)
        sc = report(conn, "mos")
    assert sc.settled_bets == 1
    assert sc.brier_n == 0
    assert sc.brier_diff is None


def test_open_bet_is_excluded():
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0), T)
        _bet(conn, a, sid, {a: 0.9, b: 0.1})
        # the event is decided, but this order's payout was never recorded
        conn.execute("UPDATE markets SET result = CASE ticker WHEN ? THEN 'yes' ELSE 'no' END", (a,))
        conn.commit()
        sc = report(conn, "mos")
    assert sc.settled_bets == 0
    assert sc.profit == 0.0
    assert sc.brier_n == 0
    assert sc.n_events == 0


def test_unfilled_order_is_excluded():
    # a rejected / unfilled order is a row in `orders`, not a bet; even if something wrote a
    # payout on it, it must not count toward the 100 bets (or dilute the profit sample)
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        a, b = "EV0-A", "EV0-B"
        sid = record_snapshot(conn, _pair(0), T)
        _, status = place(conn, Intent("mos", a, "yes", 1, 0.50, sid, 0.9, 0.5), NO_FILL, T, None)
        assert status == "none"
        _settle(conn, (a, b), a)
        conn.execute("UPDATE orders SET payout = 0.0")
        conn.commit()
        sc = report(conn, "mos")
    assert sc.settled_bets == 0
    assert sc.n_events == 0


def test_too_few_events_stays_locked():
    # 120 settled bets and 120 scored questions, both tests passing, but only 20 distinct
    # events (6 snapshots each, one bet per snapshot): the bets are not independent evidence.
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 20, p_a=0.9, a_wins=True, snapshots=6)
        sc = report(conn, "mos")
    assert sc.settled_bets == 120
    assert sc.brier_n == 120
    assert sc.profit_pass and sc.score_pass
    assert not sc.unlocked
    assert sc.n_events == 20



# ---- F8: the event floor applies to the scored Brier samples too ------------------

def test_brier_events_counts_distinct_scored_events():
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 6, p_a=0.9, a_wins=True, snapshots=2)
        sc = report(conn, "mos")
    assert sc.brier_n == 12 and sc.brier_events == 6


def test_settled_events_without_scores_do_not_unlock():
    # 20 events scored at 6 snapshots each (120 Brier samples) plus 20 more settled events
    # whose bets were never scored: 40 settled events, but only 20 behind the score test.
    with closing(connect(":memory:")) as conn:
        _settled_events(conn, 20, p_a=0.9, a_wins=True, snapshots=6)
        for i in range(20, 40):
            a, b = f"EV{i}-A", f"EV{i}-B"
            sid = record_snapshot(conn, _pair(i), T)
            _bet(conn, a, sid, {a: 0.9, b: 0.1}, candidate=False)
            _settle(conn, (a, b), a)
        sc = report(conn, "mos")
    assert sc.settled_bets == 140 and sc.n_events == 40
    assert sc.brier_n == 120 and sc.brier_events == 20
    assert sc.profit_pass and sc.score_pass
    assert not sc.unlocked


# ---- M2: the market must have priced every bucket the bot priced -----------------

def test_a_ladder_missing_a_bot_bucket_is_not_scored():
    # C has no usable quote at the bet's snapshot, so the market gives it no probability.
    # Scoring would charge the market a full miss whenever C wins, which the bot never pays.
    with closing(connect(":memory:")) as conn:
        _open_rich_account(conn)
        ms = [make_market("A", hi=68, bid=0.48, ask=0.52, event="EV0"),
              make_market("B", lo=69, hi=70, bid=0.48, ask=0.52, event="EV0"),
              make_market("C", lo=71, bid=None, ask=None, event="EV0")]
        sid = record_snapshot(conn, ms, T)
        pmf = {"A": 0.3, "B": 0.3, "C": 0.4}
        _bet(conn, "A", sid, pmf)
        _settle(conn, ("A", "B", "C"), "C")
        sc = report(conn, "mos")
    assert sc.settled_bets == 1
    assert sc.brier_n == 0 and sc.brier_events == 0 and sc.brier_diff is None


def test_settled_events_helper_continues_after_a_pinned_start():
    """A default call after `start=` must not reuse (and rewrite) the pinned events."""
    with closing(connect(':memory:')) as c:
        _settled_events(c, 3, .9, True)                      # EV0..EV2
        _settled_events(c, 2, .9, True, start=20)            # EV20, EV21
        before = c.execute("SELECT id, payout FROM orders ORDER BY id").fetchall()
        _settled_events(c, 5, .9, False)                     # must start at EV22
        after = c.execute("SELECT id, payout FROM orders WHERE id <= ? ORDER BY id",
                          (before[-1][0],)).fetchall()
        assert [tuple(r) for r in after] == [tuple(r) for r in before]
        assert c.execute("SELECT COUNT(*) FROM markets WHERE event_ticker='EV22'").fetchone()[0]
