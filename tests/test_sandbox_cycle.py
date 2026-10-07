from contextlib import closing

from sandbox_helpers import FakeClock, book_payload, ny_event_payload, three_buckets

import sandbox.cycle as cycle_mod
from sandbox.bots import Decision
from sandbox.cycle import decide_all, ensure_accounts, run_cycle
from sandbox.db import connect, get_meta
from sandbox.ledger import cash
from sandbox.models import Candidate, Intent
from sandbox.recorder import record_snapshot

NOW = "2026-10-04T10:00:00+00:00"
SAME_HOUR = "2026-10-04T10:15:00+00:00"
NEXT_HOUR = "2026-10-04T11:00:00+00:00"
T69 = "KXHIGHNY-26OCT04-T69"
B69 = "KXHIGHNY-26OCT04-B69.5"
T70 = "KXHIGHNY-26OCT04-T70"


def at(t):
    """A frozen clock: every stage of the cycle sees the same time."""
    return lambda: t


def fake_get(path, params=None):
    if path == "/series":
        return {"series": [{"ticker": "KXHIGHNY"}, {"ticker": "KXRAINNYC"}]}
    if path == "/markets" and params["status"] == "open":
        return ny_event_payload() if params["series_ticker"] == "KXHIGHNY" else {"markets": []}
    if path == "/markets" and params["status"] == "settled":
        return {"markets": []}
    if path.endswith("/orderbook"):
        return book_payload(no=[(0.80, 5)])          # YES ask 0.20 x5
    raise AssertionError(path)


class SpyBot:
    name = "favorite"

    def __init__(self):
        self.seen = []

    def decide(self, snapshot, now_iso=None):
        self.seen.append(snapshot.taken_at)
        return Decision([Intent("favorite", T69, "yes", 1, 0.20, snapshot.id, 0.3, 0.3)], [])


class BoomBot:
    name = "boom"

    def decide(self, snapshot, now_iso=None):
        raise RuntimeError("corrupt row")


def test_cycle_records_a_snapshot_and_bets_on_the_hour():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        bot = SpyBot()
        out = run_cycle(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "ok" and out["markets"] == 3 and out["orders"] == 1
        assert out["bot_errors"] == []
        assert bot.seen == [NOW]                                   # never newer than now
        order = conn.execute("SELECT status, cost, fee FROM orders").fetchone()
        assert tuple(order) == ("filled", 0.2, 0.02)
        assert cash(conn, "favorite") == 4.78


def test_decisions_run_once_per_utc_hour():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        bot = SpyBot()
        run_cycle(conn, fake_get, [bot], at(NOW))
        run_cycle(conn, fake_get, [bot], at(SAME_HOUR))
        assert bot.seen == [NOW]
        run_cycle(conn, fake_get, [bot], at(NEXT_HOUR))
        assert bot.seen == [NOW, NEXT_HOUR]
        # second hour: already holds T69 -> no second order
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_failed_discovery_records_nothing():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)

        def broken(path, params=None):
            raise RuntimeError("kalshi down")
        out = run_cycle(conn, broken, [SpyBot()], at(NOW))
        assert out["status"] == "fetch_failed"
        assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0


def test_stale_snapshot_makes_bots_sit_out():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        record_snapshot(conn, three_buckets(), "2026-10-04T09:30:00+00:00")
        bot = SpyBot()
        out = decide_all(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "stale_snapshot" and bot.seen == []


def test_already_held_and_no_fill_reasons_reach_the_candidate_log():
    from sandbox.bots import ForecastBot, normal_bucket_probs

    class Fixed(ForecastBot):
        name = "mos"

        def pmf_for_event(self, markets, snapshot):
            return normal_bucket_probs(markets, 69.5, 3.0)

    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        run_cycle(conn, fake_get, [Fixed()], at(NOW))
        run_cycle(conn, fake_get, [Fixed()], at(NEXT_HOUR))
        reasons = [r["reason"] for r in conn.execute(
            "SELECT reason FROM candidates WHERE ticker=? ORDER BY id", (T69,))]
        assert reasons == ["edge", "already_held"]


def test_one_failing_bot_does_not_stop_the_others():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        spy = SpyBot()
        out = run_cycle(conn, fake_get, [BoomBot(), spy], at(NOW))
        assert out["status"] == "ok" and out["decided"] is True
        assert out["bot_errors"] == ["boom"]
        assert spy.seen == [NOW] and out["orders"] == 1
        assert conn.execute("SELECT COUNT(*) FROM orders WHERE account='favorite'"
                            ).fetchone()[0] == 1


def test_decide_all_never_sees_a_future_snapshot():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        record_snapshot(conn, three_buckets(), "2026-10-04T09:58:00+00:00")
        record_snapshot(conn, three_buckets(), "2026-10-04T10:30:00+00:00")   # after NOW
        bot = SpyBot()
        out = decide_all(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "ok"
        assert bot.seen == ["2026-10-04T09:58:00+00:00"]
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        record_snapshot(conn, three_buckets(), "2026-10-04T10:30:00+00:00")   # only the future
        bot = SpyBot()
        out = decide_all(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "stale_snapshot" and bot.seen == []


def test_decide_all_rejects_a_future_snapshot_even_if_the_lookup_returns_one(monkeypatch):
    # Defence in depth: a negative age must not count as fresh.
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        sid = record_snapshot(conn, three_buckets(), "2026-10-04T10:30:00+00:00")
        monkeypatch.setattr(cycle_mod, "latest_snapshot_id", lambda c, at: sid)
        bot = SpyBot()
        out = decide_all(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "stale_snapshot" and bot.seen == []


def test_recent_snapshot_still_decides():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        record_snapshot(conn, three_buckets(), "2026-10-04T09:55:00+00:00")   # 300 s old
        bot = SpyBot()
        out = decide_all(conn, fake_get, [bot], at(NOW))
        assert out["status"] == "ok" and bot.seen == ["2026-10-04T09:55:00+00:00"]


def empty_get(path, params=None):
    if path == "/series":
        return {"series": []}
    raise AssertionError(path)


def test_empty_discovery_is_a_fetch_failure():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        bot = SpyBot()
        out = run_cycle(conn, empty_get, [bot], at(NOW))
        assert out["status"] == "fetch_failed"
        assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0
        # the hour stays open: a healthy retry in the same hour still decides
        out = run_cycle(conn, fake_get, [bot], at(SAME_HOUR))
        assert out["status"] == "ok" and out["decided"] is True
        assert bot.seen == [SAME_HOUR]


def test_fetch_failed_summary_has_the_same_keys():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        ok = run_cycle(conn, fake_get, [SpyBot()], at(NOW))
        failed = run_cycle(conn, empty_get, [SpyBot()], at(NEXT_HOUR))
        assert failed["status"] == "fetch_failed"
        assert set(ok) <= set(failed)
        assert {k: failed[k] for k in ok if k != "status"} == {
            "snapshot_id": None, "markets": 0, "settled": 0, "orders": 0,
            "candidates": 0, "decided": False, "bot_errors": [], "stale_bots": [],
            "confirmation_errors": [], "confirmation_transitions": []}


def _candidate(ticker, snapshot_id, bot="twoshot"):
    return Candidate(bot=bot, ticker=ticker, series="KXHIGHNY", station="KNYC",
                     metric="high", event_date="2026-10-04", snapshot_id=snapshot_id,
                     decided_at=NOW, p_bot=0.3, pmf_json="{}", p_market=0.3, yes_bid=0.15,
                     yes_ask=0.20, spread=0.05, ask_size=3.0, minutes_to_close=1140.0,
                     fee_per_contract=0.02, edge_after_fee=0.08, placed=True, reason="edge")


class TwoShot:
    """T69 first (higher priority), then B69.5; one placed candidate per intent."""
    name = "twoshot"

    def decide(self, snapshot, now_iso=None):
        specs = [(T69, 1.0), (B69, 0.5)]
        return Decision(
            [Intent("twoshot", t, "yes", 1, 0.20, snapshot.id, 0.3, 0.3, priority=pr)
             for t, pr in specs],
            [_candidate(t, snapshot.id) for t, _ in specs])


def _flaky_execute(monkeypatch):
    real = cycle_mod.execute_intent

    def flaky(conn, get, intent, now_iso):
        if intent.ticker == B69:
            raise RuntimeError("book exploded")
        return real(conn, get, intent, now_iso)

    monkeypatch.setattr(cycle_mod, "execute_intent", flaky)


def _open_twoshot(conn):
    from sandbox.ledger import open_account
    open_account(conn, "twoshot", "bot")


def test_candidates_are_recorded_even_if_a_later_intent_raises(monkeypatch):
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _open_twoshot(conn)
        record_snapshot(conn, three_buckets(), NOW)
        _flaky_execute(monkeypatch)
        out = decide_all(conn, fake_get, [TwoShot()], at(NOW))
        assert out["bot_errors"] == ["twoshot"]
        rows = conn.execute("SELECT ticker FROM candidates WHERE bot='twoshot' ORDER BY id"
                            ).fetchall()
        assert [r["ticker"] for r in rows] == [T69, B69]


# ---- F7: a candidate whose intent never ran cannot stay "placed" -------------------

def test_unexecuted_placed_candidates_become_bot_error(monkeypatch):
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _open_twoshot(conn)
        record_snapshot(conn, three_buckets(), NOW)
        _flaky_execute(monkeypatch)
        decide_all(conn, fake_get, [TwoShot()], at(NOW))
        rows = {r["ticker"]: (r["placed"], r["reason"]) for r in conn.execute(
            "SELECT ticker, placed, reason FROM candidates WHERE bot='twoshot'")}
        assert rows == {T69: (1, "edge"), B69: (0, "bot_error")}


# ---- F3: intents are funded by priority; short cash is skipped quietly -------------

class Kalshi:
    """fake_get plus a call log, per-ticker books and hooks run when a path is hit."""

    def __init__(self, books=None, hooks=None):
        self.calls: list[str] = []
        self.books = books or {}
        self.hooks = hooks or {}

    def __call__(self, path, params=None):
        self.calls.append(path)
        for key, fn in list(self.hooks.items()):
            if key in path:
                fn()
        if path.endswith("/orderbook"):
            ticker = path.split("/")[2]
            if ticker in self.books:
                return self.books[ticker]
        return fake_get(path, params)

    def book_calls(self):
        return [c for c in self.calls if c.endswith("/orderbook")]


class IntentsBot:
    """Returns one intent + one placed candidate per (ticker, price_cap, priority)."""

    def __init__(self, name, specs, on_decide=None):
        self.name, self.specs, self.on_decide = name, specs, on_decide
        self.seen = []

    def decide(self, snapshot, now_iso=None):
        self.seen.append(now_iso)
        if self.on_decide:
            self.on_decide()
        return Decision(
            [Intent(self.name, t, "yes", 1, cap, snapshot.id, 0.5, 0.3, priority=pr)
             for t, cap, pr in self.specs],
            [_candidate(t, snapshot.id, bot=self.name) for t, _, _ in self.specs])


def _set_start_cash(conn, account, amount):
    conn.execute("UPDATE accounts SET start_cash=? WHERE name=?", (amount, account))
    conn.commit()


def _orders(conn):
    return [(r["ticker"], r["status"]) for r in conn.execute(
        "SELECT ticker, status FROM orders ORDER BY id")]


def _reasons(conn, bot):
    return {r["ticker"]: (r["placed"], r["reason"]) for r in conn.execute(
        "SELECT ticker, placed, reason FROM candidates WHERE bot=?", (bot,))}


def test_a_higher_priority_intent_is_funded_before_alphabetical_order():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _set_start_cash(conn, "favorite", 0.22)          # room for exactly one .20 + .02 fill
        bot = IntentsBot("favorite", [(T69, 0.20, 0.1), (T70, 0.20, 0.5)])
        run_cycle(conn, Kalshi(), [bot], at(NOW))
        assert _orders(conn) == [(T70, "filled")]
        assert _reasons(conn, "favorite") == {T70: (1, "edge"), T69: (0, "insufficient_cash")}


def test_short_cash_writes_no_order_row_and_fetches_no_book():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _set_start_cash(conn, "favorite", 0.10)
        kalshi = Kalshi()
        run_cycle(conn, kalshi, [IntentsBot("favorite", [(T69, 0.20, 0.1)])], at(NOW))
        assert _orders(conn) == []
        assert kalshi.book_calls() == []
        assert _reasons(conn, "favorite") == {T69: (0, "insufficient_cash")}


def test_a_cheaper_later_intent_still_fills():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _set_start_cash(conn, "favorite", 0.15)
        kalshi = Kalshi(books={T69: book_payload(no=[(0.90, 5)])})      # YES ask .10
        bot = IntentsBot("favorite", [(T70, 0.45, 0.5), (T69, 0.10, 0.1)])
        run_cycle(conn, kalshi, [bot], at(NOW))
        assert _orders(conn) == [(T69, "filled")]
        assert kalshi.book_calls() == [f"/markets/{T69}/orderbook"]
        assert _reasons(conn, "favorite") == {T70: (0, "insufficient_cash"), T69: (1, "edge")}


def test_an_hour_at_zero_cash_writes_no_orders():
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        _set_start_cash(conn, "favorite", 0.0)
        kalshi = Kalshi()
        out = run_cycle(conn, kalshi, [SpyBot()], at(NOW))
        assert out["status"] == "ok" and out["orders"] == 0
        assert _orders(conn) == [] and kalshi.book_calls() == []


# ---- F4: each stage reads the real clock -------------------------------------------

def test_snapshot_taken_at_is_the_first_tick():
    clock = FakeClock(NOW)
    kalshi = Kalshi(hooks={"/series": lambda: clock.advance(minutes=2)})
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        run_cycle(conn, kalshi, [], clock)
        assert conn.execute("SELECT taken_at FROM snapshots").fetchone()[0] == NOW


def test_orders_and_candidates_are_stamped_when_they_happen():
    from sandbox.bots import ForecastBot, normal_bucket_probs
    clock = FakeClock(NOW)

    class SlowMos(ForecastBot):
        name = "mos"

        def pmf_for_event(self, markets, snapshot):
            clock.advance(minutes=3)                       # a slow forecast fetch
            return normal_bucket_probs(markets, 69.5, 3.0)

    kalshi = Kalshi(hooks={"/series": lambda: clock.advance(minutes=1)})
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        out = run_cycle(conn, kalshi, [SlowMos()], clock)
        assert out["status"] == "ok" and out["orders"] == 1
        assert conn.execute("SELECT taken_at FROM snapshots").fetchone()[0] == NOW
        assert conn.execute("SELECT decided_at FROM orders").fetchone()[0] == \
            "2026-10-04T10:04:00+00:00"                   # the fill tick
        cands = conn.execute("SELECT DISTINCT decided_at, minutes_to_close FROM candidates "
                             "WHERE ticker=?", (T69,)).fetchall()
        assert [tuple(c) for c in cands] == [("2026-10-04T10:01:00+00:00", 1139.0)]


def test_slow_discovery_makes_the_hour_stale():
    clock = FakeClock(NOW)
    kalshi = Kalshi(hooks={"/series": lambda: clock.advance(minutes=11)})
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        spy = SpyBot()
        out = run_cycle(conn, kalshi, [spy], clock)
        assert out["status"] == "stale_snapshot" and out["orders"] == 0
        assert spy.seen == []
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_a_slow_bot_makes_the_next_bot_stale():
    clock = FakeClock(NOW)
    slow = IntentsBot("highcast", [], on_decide=lambda: clock.advance(minutes=11))
    spy = SpyBot()
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        out = run_cycle(conn, Kalshi(), [slow, spy], clock)
        assert out["status"] == "ok"
        assert out["stale_bots"] == ["favorite"] and spy.seen == []
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_staleness_mid_intents_stops_the_remaining_intents():
    clock = FakeClock(NOW)
    state = {"books": 0}

    def slow_book():
        state["books"] += 1
        if state["books"] == 1:
            clock.advance(minutes=11)                      # the first book fetch hangs

    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        bot = IntentsBot("favorite", [(T69, 0.20, 0.9), (B69, 0.20, 0.5), (T70, 0.20, 0.1)])
        out = run_cycle(conn, Kalshi(hooks={"/orderbook": slow_book}), [bot], clock)
        assert _orders(conn) == [(T69, "filled")]          # no order row for the stale ones
        assert _reasons(conn, "favorite") == {T69: (1, "edge"), B69: (0, "stale_snapshot"),
                                              T70: (0, "stale_snapshot")}
        assert out["stale_bots"] == ["favorite"]
        assert conn.execute("SELECT decided_at FROM orders").fetchone()[0] == NOW


def test_confirmation_after_settle_before_decide_once_per_hour_even_when_forced(monkeypatch):
    calls = []
    monkeypatch.setattr(cycle_mod, 'settle_due', lambda *args: calls.append('settle') or 0)
    monkeypatch.setattr(cycle_mod, 'advance_confirmation',
                        lambda conn, account, now: calls.append(account))
    def decide(*args):
        calls.append('decide')
        return {'status': 'ok', 'orders': 0, 'candidates': 0,
                'bot_errors': [], 'stale_bots': []}
    monkeypatch.setattr(cycle_mod, 'decide_all', decide)
    with closing(connect(':memory:')) as conn:
        ensure_accounts(conn)
        run_cycle(conn, fake_get, [], at(NOW))
        assert calls == ['settle', 'favorite', 'highcast', 'manual', 'mos', 'decide']
        calls.clear()
        run_cycle(conn, fake_get, [], at(SAME_HOUR), force_decide=True)
        assert calls == ['settle', 'decide']
        calls.clear()
        run_cycle(conn, fake_get, [], at(NEXT_HOUR))
        assert calls == ['settle', 'favorite', 'highcast', 'manual', 'mos', 'decide']


# ---- M2: a confirmation error never stops trading ------------------------------------

def test_a_confirmation_error_never_stops_trading(monkeypatch, caplog):
    seen = []

    def advance(conn, account, now):
        seen.append(account)
        if account == 'highcast':
            raise RuntimeError('corrupt confirmation row')

    monkeypatch.setattr(cycle_mod, 'advance_confirmation', advance)
    with closing(connect(':memory:')) as conn:
        ensure_accounts(conn)
        spy = SpyBot()
        out = run_cycle(conn, fake_get, [spy], at(NOW))
        assert seen == ['favorite', 'highcast', 'manual', 'mos']      # the rest still advance
        assert out['confirmation_errors'] == ['highcast']
        assert out['status'] == 'ok' and out['decided'] is True
        assert out['orders'] == 1 and spy.seen == [NOW]               # decide_all still ran
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
        assert get_meta(conn, 'last_confirmation_hour') == NOW[:13]   # the hour guard is set
        assert get_meta(conn, 'last_decide_hour') == NOW[:13]
        assert 'highcast' in caplog.text and 'corrupt confirmation row' in caplog.text
        seen.clear()
        run_cycle(conn, fake_get, [spy], at(SAME_HOUR), force_decide=True)
        assert seen == []                                             # not retried this hour


def test_clean_cycle_reports_no_confirmation_errors_or_transitions():
    with closing(connect(':memory:')) as conn:
        ensure_accounts(conn)
        out = run_cycle(conn, fake_get, [SpyBot()], at(NOW))
        assert out['confirmation_errors'] == [] and out['confirmation_transitions'] == []


def test_cycle_summary_lists_confirmation_transitions():
    from test_sandbox_scoreboard import _settled_events
    with closing(connect(':memory:')) as conn:
        ensure_accounts(conn)
        _settled_events(conn, 100, .9, True)
        out = run_cycle(conn, fake_get, [], at(NOW))
        assert out['confirmation_transitions'] == [
            {'account': 'mos', 'from': 'LOCKED', 'to': 'PROVISIONAL'}]
        assert out['confirmation_errors'] == []
        out = run_cycle(conn, fake_get, [], at(NEXT_HOUR))
        assert out['confirmation_transitions'] == []
