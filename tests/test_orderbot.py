"""The order bot's rails. Every one of these prevents a real loss.

DESIGN: an internal design note.

THE RAIL THAT MATTERS MOST IS THE PRICE CEILING. Exiting at 99c changes TIMING,
not expected value; what decides profit is entry price against true hit rate:

    buy at 0.70 -> needs 73.5%      buy at 0.90 -> needs 92.9%
    buy at 0.80 -> needs 83.7%      buy at 0.95 -> needs 98.0%

KMIA HIGH measures 84.2% in August, so 0.81 is the highest price that can profit.
In a simulated week KLAX HIGH went SIX FOR SIX and still had EV -20.5c/contract,
because its average entry was 0.91 against 72.6% accuracy. A high win rate above
85c is a losing system that feels like a winning one. The ceiling is the one rail
that stops the bot doing that.

Everything here fails CLOSED: a missing price, an exception, an unparseable
response all mean DO NOT ORDER.
"""
import datetime as dt

import pytest

from highcast import orderbot


def _market(price=0.75, bid=0.74, ask=0.75, label="94–95", p_model=0.90):
    return [{"label": label, "lo": 94, "hi": 95, "p_model": p_model,
             "market_price": price, "yes_bid": bid, "yes_ask": ask}]


def _ctx(**over):
    """A day on which every rail passes, so a test can knock exactly one out."""
    base = dict(station="KMIA", metric="high", hour=14,
                target="2026-08-02", ranges=_market(),
                gates_ok=True, refusals=(), calibrated=0.90,
                orders_today=0, budget=1.00)
    base.update(over)
    return base


# ── THE PRICE CEILING ───────────────────────────────────────────────────────

def test_a_price_above_the_ceiling_is_refused():
    """0.81 is the break-even at 84.2% accuracy. The ceiling is 0.80."""
    d = orderbot.decide(**_ctx(ranges=_market(price=0.85, bid=0.84, ask=0.85)))
    assert d.place is False
    assert "ceiling" in d.reason.lower()


def test_a_price_at_the_ceiling_is_allowed():
    d = orderbot.decide(**_ctx(ranges=_market(price=0.80, bid=0.79, ask=0.80)))
    assert d.place is True


def test_the_six_for_six_trap_is_refused():
    """The KLAX HIGH week: six wins out of six, EV -20.5c/contract, average entry
    0.91. Exactly the shape this rail exists to stop, even though every single
    trade in it 'worked'."""
    d = orderbot.decide(**_ctx(ranges=_market(price=0.91, bid=0.90, ask=0.91)))
    assert d.place is False
    assert "ceiling" in d.reason.lower()


# ── SPEND AND FREQUENCY ─────────────────────────────────────────────────────

def test_at_most_one_order_per_day():
    """The wake cycle runs every 5 minutes. Without this the bot would re-order
    on every tick for the whole commit hour."""
    d = orderbot.decide(**_ctx(orders_today=1))
    assert d.place is False
    assert "already" in d.reason.lower()


def test_it_never_spends_more_than_the_budget():
    d = orderbot.decide(**_ctx(ranges=_market(price=0.75, bid=0.74, ask=0.75)))
    assert d.place is True
    assert d.contracts * d.limit_price <= 1.00 + 1e-9


def test_a_price_that_buys_no_contract_does_not_place():
    """$1 at 0.80 buys one contract. If the budget cannot buy even one, placing a
    zero-contract order is not a position -- it is a logging artifact that would
    pollute the scorecard."""
    d = orderbot.decide(**_ctx(budget=0.10,
                               ranges=_market(price=0.80, bid=0.79, ask=0.80)))
    assert d.place is False
    assert "contract" in d.reason.lower()


# ── THE GATES ───────────────────────────────────────────────────────────────

def test_a_gate_refusal_stops_the_order():
    d = orderbot.decide(**_ctx(gates_ok=False, refusals=("guidance_lead",)))
    assert d.place is False
    assert "guidance_lead" in d.reason


def test_below_the_confidence_floor_is_refused():
    """The floor applies to the CALIBRATED number, and is read from `gating`.

    Was written against 0.65 when the floor was 0.70. The floor was re-measured
    to 0.60 on 2026-08-02 (on the single-parity ladder the bot actually faces),
    so 0.65 now CLEARS it and the edge rule refuses instead — a pass for the
    wrong reason. Uses a value below the live floor rather than a literal, so
    re-measuring it again cannot silently turn this into an edge-rule test.
    """
    below = orderbot.confidence_floor() - 0.05
    d = orderbot.decide(**_ctx(calibrated=below))
    assert d.place is False
    assert "floor" in d.reason.lower()


def test_an_unmeasured_calibration_is_refused():
    """`band_calib` returns None for an unmeasured band. None means REFUSE and
    must never be read as 'use the raw number'."""
    d = orderbot.decide(**_ctx(calibrated=None))
    assert d.place is False


# ── SCOPE ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("station,metric", [
    ("KLAX", "high"), ("KLAX", "low"), ("KMIA", "low")])
def test_only_kmia_high_is_traded(station, metric):
    """The other three have no admissible confidence floor. KLAX HIGH in
    particular is excluded by two independent lines of evidence."""
    d = orderbot.decide(**_ctx(station=station, metric=metric))
    assert d.place is False
    assert "kmia" in d.reason.lower() or "market" in d.reason.lower()


def test_the_wrong_hour_is_refused():
    """h14 is where every number behind this bot was measured."""
    d = orderbot.decide(**_ctx(hour=11))
    assert d.place is False
    assert "hour" in d.reason.lower()


# ── FAIL CLOSED ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ranges", [
    [],                                                   # no market
    [{"label": "94–95", "p_model": 0.9}],                 # no price at all
    [{"label": "94–95", "p_model": 0.9, "yes_bid": None, "yes_ask": None}],
])
def test_a_missing_or_empty_market_is_refused(ranges):
    d = orderbot.decide(**_ctx(ranges=ranges))
    assert d.place is False


def test_a_crossed_book_is_refused():
    """ask < bid is nonsense; ordering into it would fill at an invented price."""
    d = orderbot.decide(**_ctx(ranges=_market(price=0.75, bid=0.80, ask=0.70)))
    assert d.place is False


def test_the_limit_price_never_exceeds_the_ask():
    """A limit above the ask is a market order wearing a disguise."""
    d = orderbot.decide(**_ctx(ranges=_market(price=0.75, bid=0.70, ask=0.75)))
    assert d.place is True
    assert d.limit_price <= 0.75 + 1e-9


# ── THE EDGE RULE ───────────────────────────────────────────────────────────
#
# Added 2026-08-02. The bot had a confidence floor and a price ceiling and
# NOTHING that compared them: a calibrated 0.72 against an 0.80 ask passed every
# rail and is -10% after fees. Every guard was about not trading; none was about
# the trade being worth it.

def _mkt(ask, p_model=0.95, label="94–95"):
    return [{"label": label, "lo": 94, "hi": 95, "p_model": p_model,
             "ticker": "KXHIGHMIA-26AUG02-B94.5",
             "market_price": ask, "yes_bid": ask - 0.01, "yes_ask": ask}]


def _decide(ask, calibrated):
    return orderbot.decide(
        station="KMIA", metric="high", hour=14, target="2026-08-02",
        ranges=_mkt(ask), gates_ok=True, refusals=(), calibrated=calibrated,
        orders_today=0, budget=1.00)


def test_a_calibrated_probability_below_the_ask_is_refused():
    """THE case the bot used to take. 0.72 calibrated at an 0.80 ask cleared the
    old 0.70 floor and the old 0.80 ceiling, and loses money every time."""
    i = _decide(ask=0.80, calibrated=0.72)
    assert i.place is False
    assert "no edge" in i.reason


def test_the_margin_is_fee_aware():
    """Kalshi charges roundup(0.07*C*P*(1-P)) ON TOP. At 0.80 that is 2c, so
    0.83 calibrated is break-even-ish and must still be refused; 0.86 clears."""
    assert orderbot.taker_fee(0.80) == 0.02
    assert _decide(ask=0.80, calibrated=0.83).place is False
    assert _decide(ask=0.80, calibrated=0.86).place is True


def test_a_high_confidence_day_can_buy_up_to_the_new_ceiling():
    """Julian's stated tolerance: 70-84c is fine, 90c is not. A 96% day at 0.84
    is exactly the trade the ceiling was raised to allow."""
    assert orderbot.PRICE_CEILING == 0.84
    i = _decide(ask=0.84, calibrated=0.96)
    assert i.place is True and i.limit_price == 0.84


def test_the_ceiling_still_refuses_ninety_cents():
    i = _decide(ask=0.90, calibrated=0.99)
    assert i.place is False
    assert "ceiling" in i.reason


def test_the_floor_is_read_from_gating_not_copied():
    """⚠️ It was a bare 0.70 here until 2026-08-02, so re-measuring it in
    `gating` (0.70 -> 0.60) silently did nothing to the bot. A constant with two
    homes has one that is wrong."""
    from highcast import gating
    assert orderbot.confidence_floor() == gating.CONFIDENCE_FLOOR[
        ("KMIA", "high", 14)]
    # a 0.62 day now clears the floor (it did not under the stale 0.70) and is
    # then judged on EDGE, which is the rail that should decide it
    i = _decide(ask=0.40, calibrated=0.62)
    assert i.place is True, i.reason


# ── THE MAXIMUM EDGE ────────────────────────────────────────────────────────
#
# Added 2026-08-02. MIN_EDGE stops a mediocre disagreement from being bought;
# nothing stopped an IMPLAUSIBLE one. HANDOFF-2026-08-03.md §0: on the 16
# graded `app_bets`, edge correlates -0.619 with WINNING -- we lose precisely
# where we think we found value. §4.2: a KLAX LOW bucket showed a +73% edge at
# 0.5c because the bucket was already dead (a stale bound, not a real gap). A
# calibrated number this far from a liquid market is usually OUR error.

# HANDOFF §2.5's four reference cases, restated here so the MAX_EDGE addition
# is checked against them directly rather than trusted not to have moved them.
def test_handoff_2_5_reference_cases_are_unchanged():
    assert _decide(ask=0.90, calibrated=0.99).place is False        # ceiling
    assert _decide(ask=0.84, calibrated=0.96).place is True         # PLACE
    assert _decide(ask=0.84, calibrated=0.86).place is False        # no edge
    i = _decide(ask=0.40, calibrated=0.62)
    assert i.place is True and i.contracts == 2                     # PLACE, 2


def test_an_ordinary_edge_still_places():
    """0.22 -- HANDOFF §2.5's own 0.40-ask/0.62-calib case -- is well inside
    MAX_EDGE (0.40) and must clear exactly as before."""
    i = _decide(ask=0.40, calibrated=0.62)
    assert i.place is True
    assert i.edge == pytest.approx(0.22)


def test_an_egregious_edge_refuses_and_records_the_bucket():
    """0.99 calibrated against a 0.10 ask is an 89-point disagreement -- the
    shape of the §4.2 stale-bucket case, not a value bet. Must refuse with a
    reason distinguishable from the ordinary 'no edge' (too-small) refusal,
    and must NOT lose the bucket the way the pre-§2.7 refusals did."""
    i = _decide(ask=0.10, calibrated=0.99)
    assert i.place is False
    assert "edge_implausible" in i.reason
    assert "no edge" not in i.reason  # distinguishable from the MIN_EDGE refusal
    assert i.bucket == "94–95"
    assert i.bucket is not None
    assert i.edge == pytest.approx(0.89)


def test_the_investigation_checklist_is_in_the_refusal_message():
    """Whoever resolves the alert needs to know what to check, not just that
    something tripped."""
    i = _decide(ask=0.10, calibrated=0.99)
    for item in ("latest model run", "source data", "ingested the obs",
                 "forecast product", "stale"):
        assert item in i.reason, f"missing checklist item: {item!r}"


def test_max_edge_has_no_local_copy_in_orderbot():
    """⚠️ `orderbot.CONFIDENCE_FLOOR` was once a bare duplicate of
    `gating.CONFIDENCE_FLOOR`, and re-measuring the original was a silent
    NO-OP because the stale copy still bound (HANDOFF §2.5). MAX_EDGE must
    have exactly one home."""
    assert not hasattr(orderbot, "MAX_EDGE")


def test_max_edge_is_read_from_gating_live(monkeypatch):
    """The behavioural half of the guard above: prove the value is actually
    READ at decide-time, not just absent from orderbot's namespace. Lowering
    `gating.MAX_EDGE` below a previously-fine edge must flip the verdict."""
    from highcast import gating
    assert orderbot.decide(
        station="KMIA", metric="high", hour=14, target="2026-08-02",
        ranges=_mkt(ask=0.40, p_model=0.62), gates_ok=True, refusals=(),
        calibrated=0.62, orders_today=0, budget=1.00).place is True

    monkeypatch.setattr(gating, "MAX_EDGE", 0.05)
    i = orderbot.decide(
        station="KMIA", metric="high", hour=14, target="2026-08-02",
        ranges=_mkt(ask=0.40, p_model=0.62), gates_ok=True, refusals=(),
        calibrated=0.62, orders_today=0, budget=1.00)
    assert i.place is False
    assert "edge_implausible" in i.reason
