"""THE CONFIDENCE GATE on the desk's PLACE — built, tested, and DEFAULT OFF.

WHAT IT DOES. When `StakingConfig.confidence_gate` is true and a measured
floor is supplied, `staking.evaluate` refuses to PLACE below that floor. The
floor is the market's own `gating.CONFIDENCE_FLOOR` entry — never a number
invented here, and never one borrowed from another market.

⚠️ WHY IT SHIPS OFF, AND WHY YOU SHOULD THINK BEFORE TURNING IT ON.

`evaluate`'s own docstring already argued against this: confidence only
arrives once the outcome has formed, and by then the market has priced it, so
gating on confidence structurally removes the edge rather than the losses.

Measured on the 14 settled PLACE bets — the only price-aware evidence that
exists, since Kalshi sells no historical prices:

    group                    n    hit   mean ask   net/contract
    all placed (today)      14    43%      0.41         +0.000
    confidence >= 0.80       5    60%      0.60         -0.016
    confidence <  0.80       9    33%      0.31         +0.009   <- vetoed

The high-confidence group HITS MORE AND LOSES MONEY; the group this gate
would delete is the one that made money. The ask rises from 31c to 60c to
meet the confidence, and eats it.

The 574-day walk-forward says the opposite about HIT RATE (stated 0.70+ hits
84.1%, stated <0.50 hits 42.6%) — and hit rate is not what pays. Both facts
are true; they are about different quantities.

So: n=14 cannot settle this, and n=574 cannot see price at all. The gate
exists so the question can be answered later with one config line, not so it
can be switched on today.
"""
import dataclasses

import pytest

from highcast import gating, staking
from highcast.config import StakingConfig

CFG = StakingConfig(bankroll=100.0, min_edge=0.03)


def _ranges(p_model, price, label="80–81"):
    return [{"label": label, "p_model": p_model, "market_price": price,
             "bid": price - 0.02, "ask": price + 0.02}]


def test_gate_is_off_by_default(monkeypatch):
    """The shipped default must not change behaviour. A 60%-confidence bet
    with a real edge still places, exactly as it did before this existed."""
    d = staking.evaluate(_ranges(0.60, 0.40), CFG)
    assert d.place is True, d.reason
    assert CFG.confidence_gate is False


def test_gate_on_refuses_below_the_floor():
    cfg = dataclasses.replace(CFG, confidence_gate=True)
    d = staking.evaluate(_ranges(0.60, 0.40), cfg, floor=0.80)
    assert d.place is False
    assert "confidence" in d.reason.lower()
    assert "0.80" in d.reason or "80%" in d.reason


def test_gate_on_allows_at_or_above_the_floor():
    cfg = dataclasses.replace(CFG, confidence_gate=True)
    d = staking.evaluate(_ranges(0.85, 0.50), cfg, floor=0.80)
    assert d.place is True, d.reason


def test_gate_on_with_no_measured_floor_does_not_veto():
    """FAIL-OPEN HERE, DELIBERATELY, AND IT IS THE OPPOSITE OF THE BOT.

    `gating.confidence_floor_ok` fails CLOSED on an unmeasured market, which is
    right for an unattended order bot. The desk is decision support for a human
    who can see the number and disagree. Failing closed here would silently
    stop the desk recommending anything on KLAX high, KLAX low and KMIA low —
    three of four markets, none of which has ever qualified for a floor — and
    that is a far larger behaviour change than 'add a confidence gate'.

    A market with no floor therefore keeps today's edge-only rule, and the
    absence is visible rather than acted on.
    """
    cfg = dataclasses.replace(CFG, confidence_gate=True)
    d = staking.evaluate(_ranges(0.55, 0.30), cfg, floor=None)
    assert d.place is True, d.reason


def test_the_floor_must_come_from_the_measured_table():
    """Only KMIA HIGH has ever qualified. This pins that, so a future edit that
    invents floors for the other three has to change a test that says why."""
    measured = {k for k in gating.CONFIDENCE_FLOOR}
    assert ("KMIA", "high", 13) in measured
    for absent in (("KLAX", "high", 13), ("KLAX", "low", 6), ("KMIA", "low", 7)):
        assert absent not in measured, absent


def test_gate_runs_after_the_edge_check_not_before():
    """Ordering matters for the REASON the desk shows. A no-edge bet must still
    say 'no edge' — the more actionable message — rather than being masked by a
    confidence refusal it would also have failed."""
    cfg = dataclasses.replace(CFG, confidence_gate=True)
    d = staking.evaluate(_ranges(0.50, 0.49), cfg, floor=0.80)
    assert d.place is False
    assert "edge" in d.reason.lower()


@pytest.mark.parametrize("p,expected", [(0.799, False), (0.80, True), (0.81, True)])
def test_floor_is_inclusive(p, expected):
    cfg = dataclasses.replace(CFG, confidence_gate=True)
    d = staking.evaluate(_ranges(p, 0.50), cfg, floor=0.80)
    assert d.place is expected, d.reason
