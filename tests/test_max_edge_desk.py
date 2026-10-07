"""MAX_EDGE ON THE DESK — refuse an edge too large to be real.

THE INCONSISTENCY THIS FIXES. `gating.MAX_EDGE = 0.40` has guarded the ORDER
BOT since 2026-08-02 on the grounds that a very large edge is usually OUR error
rather than free money. The desk had no such ceiling, so it would happily
recommend a bet the bot in the same repo refuses. One project, two answers.

THE EVIDENCE, measured 2026-08-10 on the desk's own record:

    corr(edge, won) over all 48 settled bets:  -0.571
    PLACE bets with edge > 0.40:  6 of 18, and 5 of those 6 LOST

    ceiling   kept  hit   net/bet
    none        18  50%   +0.065
    <= 0.50     15  60%   +0.099
    <= 0.40     12  67%   +0.114
    <= 0.30     11  73%   +0.148

⚠️ THE P&L IMPROVEMENT IS NOT SIGNIFICANT AT n=18. Paired bootstrap at the 0.40
cap: D net/bet +0.049, CI [-0.095, +0.195]. What carries this is not those 18
rows, it is the -0.571 correlation on 48 and the bot's own prior derivation.

⚠️ 0.40 IS NOT FITTED HERE. The table above says 0.25 or 0.30 looks better —
that is exactly the threshold-swept-on-a-tiny-sample error QUEUE item 1
forbids. The shipped value is reused unchanged so this is applying a measured
constant to a second code path, not inventing one.

CONTRAST WITH THE CONFIDENCE GATE (`test_confidence_gate.py`), which ships OFF:
its point estimate was NEGATIVE — it deleted the group that made money. This
one is positive at every threshold tested. Same discipline, opposite verdict.
"""
import dataclasses

import pytest

from highcast import gating, staking
from highcast.config import StakingConfig

CFG = StakingConfig(bankroll=100.0, min_edge=0.03)


def _r(p_model, price, label="80–81"):
    return [{"label": label, "p_model": p_model, "market_price": price}]


def test_the_default_matches_the_bot():
    """One constant, two callers. If these drift, the desk and the bot disagree
    about the same bet — the shadowing bug `gating` already warns about."""
    assert CFG.max_edge == gating.MAX_EDGE == 0.40


def test_an_implausible_edge_is_refused():
    """+72% on a 1c bucket — the 2026-08-01 KLAX low. The bucket was already
    dead and the market knew; our 'edge' was staleness."""
    d = staking.evaluate(_r(0.73, 0.01), CFG)
    assert d.place is False
    assert "40%" in d.reason or "0.40" in d.reason, d.reason
    assert "our own error" in d.reason.lower() or "stale" in d.reason.lower(), d.reason


def test_an_ordinary_edge_still_places():
    d = staking.evaluate(_r(0.66, 0.545), CFG)
    assert d.place is True, d.reason


def test_the_boundary_is_inclusive():
    """Exactly at the ceiling is allowed; only ABOVE it is refused."""
    assert staking.evaluate(_r(0.90, 0.50), CFG).place is True     # edge 0.40
    assert staking.evaluate(_r(0.905, 0.50), CFG).place is False   # edge 0.405


def test_it_runs_after_the_min_edge_check():
    """A no-edge bet must still say 'no edge' — that is the more common and
    more actionable message, and a bet cannot be both too small and too big."""
    d = staking.evaluate(_r(0.50, 0.49), CFG)
    assert d.place is False
    assert "no edge" in d.reason.lower(), d.reason


def test_it_can_be_disabled():
    """0 or None means no ceiling, for anyone re-measuring the threshold."""
    cfg = dataclasses.replace(CFG, max_edge=0.0)
    assert staking.evaluate(_r(0.73, 0.01), cfg).place is True


def test_the_refusal_carries_the_numbers():
    """A veto that erases the evidence makes `signals` unreconstructable —
    the same argument `staking.blocked` makes for carrying them through."""
    d = staking.evaluate(_r(0.73, 0.01), CFG)
    assert d.bucket == "80–81"
    assert d.p_model == 0.73
    assert d.price == 0.01
    assert d.edge is not None


def test_the_six_historical_bets_it_would_have_stopped():
    """Pinned from the real record: every PLACE with edge > 0.40, and how it
    actually settled. 5 of 6 lost."""
    cases = [(0.73, 0.01, False), (0.72, 0.16, False), (0.62, 0.12, False),
             (0.81, 0.34, False), (0.56, 0.11, True), (0.81, 0.38, False)]
    stopped = won = 0
    for p, price, did_win in cases:
        d = staking.evaluate(_r(p, price), CFG)
        if not d.place:
            stopped += 1
            won += did_win
    assert stopped == 6, "all six should be refused"
    assert won == 1, "and only one of them actually won"
