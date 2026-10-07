"""Synthetic payloads shaped exactly like the live Kalshi responses verified on
2026-10-04 (markets: *_dollars price strings, *_size_fp size strings, Z close
times; orderbook: orderbook_fp.{yes,no}_dollars = [[price, qty], ...] bids)."""
from datetime import datetime, timedelta

from sandbox.models import Market

EVENT = "KXHIGHNY-26OCT04"


def raw_market(ticker, event=EVENT, floor=None, cap=None, bid="0.2000",
               ask="0.2400", bid_size="10.00", ask_size="3.00",
               close="2026-10-05T05:00:00Z", code="NYC"):
    return {
        "ticker": ticker, "event_ticker": event, "market_type": "binary",
        "floor_strike": floor, "cap_strike": cap,
        "yes_bid_dollars": bid, "yes_ask_dollars": ask,
        "yes_bid_size_fp": bid_size, "yes_ask_size_fp": ask_size,
        "close_time": close,
        "rules_primary": (f"If the maximum temperature recorded at Somewhere (CLI{code}) "
                          "for Oct 4, 2026, is greater than 70° fahrenheit"),
    }


def ny_event_payload():
    """Three buckets: ≤68 (cap 69), 69–70 (floor 69 cap 70), ≥71 (floor 70)."""
    return {"markets": [
        raw_market(f"{EVENT}-T69", cap=69, bid="0.1500", ask="0.2000"),
        raw_market(f"{EVENT}-B69.5", floor=69, cap=70, bid="0.2500", ask="0.3000"),
        raw_market(f"{EVENT}-T70", floor=70, bid="0.4000", ask="0.4500"),
    ]}


def make_market(ticker, lo=None, hi=None, bid=0.20, ask=0.24, event=EVENT,
                series="KXHIGHNY", metric="high", station="KNYC",
                close="2026-10-05T05:00:00+00:00", bid_size=10.0, ask_size=3.0):
    label = f"≤{hi}" if lo is None else (f"≥{lo}" if hi is None else f"{lo}–{hi}")
    return Market(ticker=ticker, series=series, event_ticker=event,
                  event_date="2026-10-04", lo=lo, hi=hi, label=label,
                  metric=metric, station=station, yes_bid=bid, yes_ask=ask,
                  yes_bid_size=bid_size, yes_ask_size=ask_size, close_time=close)


def three_buckets(bids=(0.15, 0.25, 0.40), asks=(0.20, 0.30, 0.45), **kw):
    return [
        make_market(f"{EVENT}-T69", None, 68, bids[0], asks[0], **kw),
        make_market(f"{EVENT}-B69.5", 69, 70, bids[1], asks[1], **kw),
        make_market(f"{EVENT}-T70", 71, None, bids[2], asks[2], **kw),
    ]


def book_payload(no=(), yes=()):
    fmt = lambda levels: [[f"{p:.4f}", f"{q:.2f}"] for p, q in levels]
    return {"orderbook_fp": {"no_dollars": fmt(no), "yes_dollars": fmt(yes)}}


class FakeClock:
    """An advancing UTC clock for run_cycle(clock=...). Each call returns the current
    time as an ISO string, then moves on by `step_s`; tests can also `advance()` it
    (for example from inside a fake `get`, to make discovery or a book fetch slow)."""

    def __init__(self, start: str, step_s: float = 0.0):
        self.t = datetime.fromisoformat(start)
        self.step = timedelta(seconds=step_s)
        self.calls = 0

    def __call__(self) -> str:
        now = self.t.isoformat()
        self.t += self.step
        self.calls += 1
        return now

    def peek(self) -> str:
        return self.t.isoformat()

    def advance(self, **kw) -> None:
        self.t += timedelta(**kw)
