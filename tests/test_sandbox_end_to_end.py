"""Two simulated days of hourly decisions against a fake Kalshi: discovery, bets, books,
settlement, payouts, re-funding from payouts, and the report — checked against the
fake world's own prices and winners, not against the ledger's own arithmetic."""
import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta

import pytest
from sandbox_helpers import FakeClock, book_payload, raw_market

from highcast.fees import taker_fee
from sandbox.bots import FavoriteBot, HighcastBot, MosBot
from sandbox.cycle import ensure_accounts, run_cycle
from sandbox.db import connect
from sandbox.ledger import START_CASH, cash
from sandbox.scoreboard import report
from sandbox.stations import climate_day_start_utc

# (series, CLI code) for six cities; KLAX and KMIA are the highcast stations
CITIES = [("KXHIGHNY", "NYC"), ("KXHIGHLAX", "LAX"), ("KXHIGHMIA", "MIA"),
          ("KXHIGHDEN", "DEN"), ("KXHIGHCHI", "MDW"), ("KXHIGHAUS", "AUS")]
D1, D2, D3 = "2026-10-05", "2026-10-06", "2026-10-07"
# bucket suffix -> (floor, cap, bid, ask); the favorite is always T70 at .60
LADDER = {"T69": (None, 69, 0.15, 0.20), "B69.5": (69, 70, 0.25, 0.30),
          "T70": (70, None, 0.55, 0.60)}
# MOS mu per (code, date): 72 makes T70 an edge, 66 makes T69 an edge, 69.5 neither
MU = {"NYC": 72.0, "LAX": 66.0, "MIA": 72.0, "DEN": 69.5, "MDW": 66.0, "AUS": 72.0}
# winning bucket per (code, date)
WINNER = {(c, d): ("T70" if i % 2 == 0 else "T69")
          for d in (D1, D2, D3) for i, (_, c) in enumerate(CITIES)}


def event_ticker(series, d):
    return f"{series}-{datetime.fromisoformat(d).strftime('%y%b%d').upper()}"


def close_time(d):
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat() + "T10:00:00Z"


class World:
    """The fake Kalshi. Set `open_dates` / `settled_dates` before each cycle."""

    def __init__(self):
        self.open_dates: list[str] = []
        self.settled_dates: list[str] = []
        self.calls: list[str] = []

    def tickers(self, series, d):
        ev = event_ticker(series, d)
        return [(f"{ev}-{suffix}", suffix) for suffix in LADDER]

    def __call__(self, path, params=None):
        self.calls.append(path)
        if path == "/series":
            return {"series": [{"ticker": s} for s, _ in CITIES] + [{"ticker": "KXRAINNYC"}]}
        if path == "/markets":
            series = params["series_ticker"]
            code = dict(CITIES).get(series)
            if code is None:
                return {"markets": []}
            if params["status"] == "open":
                out = []
                for d in self.open_dates:
                    for ticker, suffix in self.tickers(series, d):
                        floor, cap, bid, ask = LADDER[suffix]
                        out.append(raw_market(ticker, event=event_ticker(series, d), floor=floor,
                                              cap=cap, bid=f"{bid:.4f}", ask=f"{ask:.4f}",
                                              close=close_time(d), code=code))
                return {"markets": out}
            return {"markets": [
                {"ticker": ticker, "status": "finalized",
                 "result": "yes" if suffix == WINNER[(code, d)] else "no"}
                for d in self.settled_dates for ticker, suffix in self.tickers(series, d)]}
        if path.endswith("/orderbook"):
            suffix = path.split("/")[2].rsplit("-", 1)[1]
            ask = LADDER[suffix][3]
            return book_payload(no=[(round(1 - ask, 4), 5)])
        raise AssertionError(path)


class FakeMos:
    model = "MEX"

    def value(self, icao, metric, event_date):
        return MU[icao[1:]]


def make_highcast_db(path):
    c = sqlite3.connect(path)
    try:
        c.execute("""CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT,
                     kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, metric TEXT)""")
        for station in ("KLAX", "KMIA"):
            for d in (D1, D2, D3):
                c.execute("INSERT INTO predictions VALUES (?,?,?,?,NULL,NULL,?,'high')",
                          (station, d, "2026-10-04T00:00:00+00:00", "dayahead",
                           json.dumps({"66": 0.2, "67": 0.5, "70": 0.3})))
        c.commit()
    finally:
        c.close()


# (cycle time, open event dates, settled event dates)
SCHEDULE = [
    ("2026-10-04T12:00:00+00:00", [D1], []),
    ("2026-10-05T12:00:00+00:00", [D1, D2], []),
    ("2026-10-06T12:00:00+00:00", [D2, D3], [D1]),
    ("2026-10-07T12:00:00+00:00", [D3], [D1, D2]),
]


@pytest.fixture
def run(tmp_path):
    hc = tmp_path / "highcast.db"
    make_highcast_db(hc)
    world = World()
    summaries = []
    with closing(connect(":memory:")) as conn:
        ensure_accounts(conn)
        for when, open_dates, settled in SCHEDULE:
            world.open_dates, world.settled_dates = open_dates, settled
            bots = [FavoriteBot(), HighcastBot(str(hc)), MosBot(FakeMos())]
            summaries.append(run_cycle(conn, world, bots, FakeClock(when, step_s=2)))
        yield conn, summaries


def _ticker_info():
    """ticker -> (code, event date, suffix) for every market the world ever listed."""
    out = {}
    for series, code in CITIES:
        for d in (D1, D2, D3):
            for suffix in LADDER:
                out[f"{event_ticker(series, d)}-{suffix}"] = (code, d, suffix)
    return out


def test_every_cycle_ran_clean(run):
    _, summaries = run
    for s in summaries:
        assert s["status"] == "ok" and s["decided"] is True
        assert s["bot_errors"] == [] and s["stale_bots"] == []
    assert summaries[2]["settled"] == 18 and summaries[3]["settled"] == 18


def test_cash_matches_the_worlds_prices_and_winners(run):
    conn, _ = run
    info = _ticker_info()
    settled = {D1, D2}
    for account in ("favorite", "mos", "highcast", "manual"):
        spent = paid = 0.0
        for o in conn.execute("SELECT * FROM orders WHERE account=? AND filled_contracts > 0",
                              (account,)):
            code, d, suffix = info[o["ticker"]]
            ask = LADDER[suffix][3]
            n = o["filled_contracts"]
            spent += n * ask + taker_fee(n, ask)                  # what the world charged
            if d in settled:
                paid += n * (1.0 if suffix == WINNER[(code, d)] else 0.0)
        assert cash(conn, account) == pytest.approx(START_CASH - spent + paid, abs=1e-9)
        assert cash(conn, account) >= 0
        assert report(conn, account).cash == pytest.approx(cash(conn, account))


def test_settled_bets_count_filled_and_settled_orders(run):
    conn, _ = run
    info = _ticker_info()
    for account in ("favorite", "mos", "highcast"):
        filled = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM orders WHERE account=? AND filled_contracts > 0", (account,))]
        expected = sum(1 for t in filled if info[t][1] in (D1, D2))
        assert expected > 0
        assert report(conn, account).settled_bets == expected


def test_every_candidate_on_a_settled_market_has_its_result(run):
    conn, _ = run
    info = _ticker_info()
    rows = conn.execute("SELECT ticker, result, asif_profit, yes_ask FROM candidates").fetchall()
    on_settled = [r for r in rows if info[r["ticker"]][1] in (D1, D2)]
    assert on_settled
    for r in on_settled:
        code, d, suffix = info[r["ticker"]]
        assert r["result"] == ("yes" if suffix == WINNER[(code, d)] else "no")
        assert r["asif_profit"] is not None
    assert all(r["result"] is None for r in rows if info[r["ticker"]][1] == D3)


def test_no_ticker_is_filled_twice(run):
    conn, _ = run
    dupes = conn.execute("SELECT account, ticker FROM orders WHERE filled_contracts > 0 "
                         "GROUP BY account, ticker HAVING COUNT(*) > 1").fetchall()
    assert dupes == []


def test_pre_day_bots_only_bet_before_the_climate_day(run):
    conn, _ = run
    info = _ticker_info()
    for o in conn.execute("SELECT o.ticker, o.decided_at, m.station FROM orders o "
                          "JOIN markets m ON m.ticker = o.ticker "
                          "WHERE o.account IN ('favorite', 'mos')"):
        start = climate_day_start_utc(o["station"], info[o["ticker"]][1])
        assert datetime.fromisoformat(o["decided_at"]) < start


def test_payouts_refund_an_account_that_ran_dry(run):
    conn, _ = run
    # favorite pays .62 a bet: six on day 1 ($3.72), two more fit on day 2 ($1.24, leaving
    # $0.04), then D1's three winners pay $3 and it can bet again on D3.
    by_day = {r["d"]: r["n"] for r in conn.execute(
        "SELECT substr(decided_at, 1, 10) AS d, COUNT(*) AS n FROM orders "
        "WHERE account='favorite' AND filled_contracts > 0 GROUP BY d")}
    assert by_day == {"2026-10-04": 6, "2026-10-05": 2, "2026-10-06": 4}
    # short cash was skipped quietly: never an order row it could not pay for
    assert conn.execute("SELECT COUNT(*) FROM orders WHERE status='insufficient_cash'"
                        ).fetchone()[0] == 0
