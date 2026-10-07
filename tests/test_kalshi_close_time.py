"""`parse_markets` retains `close_time`.

Why it matters: without it, "how long do I have to act" is unanswerable from our
own data, and HANDOFF's "closes 15:59 MNL the next day" sat unverified for weeks.
Measured 2026-07-30, every market stays open ~11 hours past the LAST of the four
commit hours — so the four markets need not be taken in sequence.
"""
import datetime as dt
from zoneinfo import ZoneInfo

from highcast.fetchers.kalshi import parse_markets

EV = "KXLOWTLAX-26JUL30"


def _payload(**over):
    m = {"event_ticker": EV, "market_type": "binary",
         "floor_strike": 68, "cap_strike": 69,
         "yes_bid_dollars": 0.40, "yes_ask_dollars": 0.44,
         "last_price_dollars": 0.42,
         "close_time": "2026-07-31T08:00:00Z"}
    m.update(over)
    return {"markets": [m]}


def test_close_time_is_retained():
    r = parse_markets(_payload(), EV)[0]
    assert r["close_time"] == "2026-07-31T08:00:00Z"


def test_absent_close_time_is_none_not_an_error():
    """Older fixtures and any API shape change must degrade to unknown, never
    crash and never be mistaken for 'closed'."""
    p = _payload()
    del p["markets"][0]["close_time"]
    assert parse_markets(p, EV)[0]["close_time"] is None


def test_the_measured_close_leaves_room_past_every_commit_hour():
    """The load-bearing fact. KLAX LOW commits 19:00 MNL; KLAX HIGH — the LAST of
    the four — commits 02:00 MNL the next day. This market closes 16:00 MNL after
    that, so a best-of-four decision is still actionable when Julian wakes."""
    mnl = ZoneInfo("Asia/Manila")
    close = dt.datetime.fromisoformat(
        parse_markets(_payload(), EV)[0]["close_time"].replace("Z", "+00:00"))
    assert close.astimezone(mnl).strftime("%H:%M") == "16:00"
    last_commit = dt.datetime(2026, 7, 31, 2, 0, tzinfo=mnl)
    assert (close - last_commit).total_seconds() / 3600 > 10


def test_other_fields_are_unchanged():
    """Adding a key must not disturb the existing contract."""
    r = parse_markets(_payload(), EV)[0]
    # NB the label uses an EN DASH, not a hyphen — `pick.settle` and any other
    # consumer that splits a bucket label must normalise it (as they do).
    assert (r["label"], r["lo"], r["hi"]) == ("68–69", 68, 69)
    assert (r["yes_bid"], r["yes_ask"], r["last_price"]) == (0.40, 0.44, 0.42)
