"""THE RATCHET GATE — refuse a bucket the day has already made impossible.

⚠️ THE FIRST VERSION OF THIS FILE HAD THE DIRECTION BACKWARDS, and the tests
asserted the bug. Recording it because it is the easiest possible mistake here
and the next reader will be tempted by the same intuition.

A settlement LOW is the MINIMUM over the whole climate day. Once 84.0F has been
observed, more observations can only push the minimum DOWN:

    running min 84.0
      bucket 82–83   REACHABLE — the low only has to fall 1–2F further
      bucket 85–86   DEAD      — we have already seen 84.0

So for a MIN the ratchet is a CEILING and the dead buckets are the ones ABOVE
it. For a MAX it is a FLOOR and the dead ones are BELOW. Getting that backwards
refuses exactly the live buckets and admits the dead ones.

WHAT THIS IS. Arithmetic on data the desk already holds, not a probability
judgement: `latentmax.visible_running_extremum` returns the interval of
settlement values still consistent with everything observed, and a bucket
entirely outside it cannot win. Nothing here touches `features`, `models/` or
the probability — the model may believe what it likes; the desk declines to
stake on an outcome that is closed.

⚠️ HONEST SCOPE. This would NOT have caught the 2026-08-10 KMIA low (82–83
against a running min of 84.0) — that bucket was live, and the desk's problem
there was overconfidence, not impossibility. This gate catches a narrower and
rarer error. It is cheap and it cannot be wrong, which is the whole argument
for it.

⚠️ USE THE INTERVAL, NOT THE MIDPOINT. `visible_running_extremum` returns
(lo, hi) because a T-group is precise to 0.1F and a 6-hour group can sit
outside every spot reading. Refuse only when NO value in the bucket survives.
"""
import pytest

from highcast import staking
from highcast.config import StakingConfig

CFG = StakingConfig(bankroll=100.0, min_edge=0.03)


def _r(label, p_model=0.6, price=0.2):
    return [{"label": label, "p_model": p_model, "market_price": price}]


# ── LOW: the ratchet is a CEILING; dead buckets sit ABOVE it ────────────────

def test_a_low_bucket_above_the_running_min_is_dead():
    """We have already observed 84.0. The day's minimum can never settle at
    85–86 — that is arithmetic, not an opinion about the weather."""
    d = staking.evaluate(_r("85–86"), CFG, extremum="min", ratchet=(83.92, 84.12))
    assert d.place is False
    assert "84" in d.reason and "already" in d.reason.lower(), d.reason


def test_a_low_bucket_below_the_running_min_is_LIVE():
    """THE CASE THE FIRST VERSION GOT WRONG. 82–83 against a running min of
    84.0 is entirely reachable — a minimum falls."""
    d = staking.evaluate(_r("82–83"), CFG, extremum="min", ratchet=(83.92, 84.12))
    assert d.place is True, d.reason


def test_the_bucket_containing_the_ratchet_is_live():
    d = staking.evaluate(_r("84–85"), CFG, extremum="min", ratchet=(83.92, 84.12))
    assert d.place is True, d.reason


def test_the_interval_edge_keeps_a_bucket_alive():
    """With the ratchet at (83.92, 84.12), a bucket starting at 84 survives:
    the true minimum may be 84.12, which is inside 84–85."""
    d = staking.evaluate(_r("84–85"), CFG, extremum="min", ratchet=(83.92, 84.12))
    assert d.place is True, d.reason


# ── HIGH: the ratchet is a FLOOR; dead buckets sit BELOW it ─────────────────

def test_a_high_bucket_below_the_running_max_is_dead():
    """Running max interval (90.94, 91.14) means 88–89 can never settle.

    The message must quote 90.9 — the LOWER edge — because that is the weakest
    claim the evidence supports: the high is *at least* 90.9. Quoting 91.1
    would assert more than is known."""
    d = staking.evaluate(_r("88–89"), CFG, extremum="max", ratchet=(90.94, 91.14))
    assert d.place is False
    assert "90.9" in d.reason, d.reason
    assert "at least" in d.reason, d.reason


def test_a_high_bucket_above_the_running_max_is_live():
    d = staking.evaluate(_r("92–93"), CFG, extremum="max", ratchet=(90.94, 91.14))
    assert d.place is True, d.reason


def test_the_two_metrics_are_mirror_images():
    """The same bucket-vs-ratchet pair must give OPPOSITE answers for min and
    max. If both agree, the direction has been collapsed again."""
    lo = staking.evaluate(_r("85–86"), CFG, extremum="min", ratchet=(83.9, 84.1))
    hi = staking.evaluate(_r("85–86"), CFG, extremum="max", ratchet=(83.9, 84.1))
    assert lo.place is False and hi.place is True, (lo.reason, hi.reason)


# ── it must be inert whenever it cannot judge ──────────────────────────────

def test_no_ratchet_changes_nothing():
    """Every existing caller passes no ratchet and must be byte-identical."""
    a = staking.evaluate(_r("85–86"), CFG)
    b = staking.evaluate(_r("85–86"), CFG, extremum="min", ratchet=None)
    assert a.place == b.place and a.reason == b.reason


def test_an_open_ended_bucket_is_never_refused_by_this_gate():
    """`≥81` and `≤75` are real Kalshi labels. An unbounded side cannot be
    proven dead, so fall through rather than invent a refusal."""
    for label in ("≥81", "≤75"):
        d = staking.evaluate(_r(label), CFG, extremum="min", ratchet=(83.92, 84.12))
        assert d.place is True, (label, d.reason)


def test_an_unparseable_bucket_falls_through():
    d = staking.evaluate(_r("weird"), CFG, extremum="min", ratchet=(83.92, 84.12))
    assert d.place is True, d.reason


def test_it_runs_before_the_edge_check():
    """A dead bucket should say it is dead, not 'no edge' — impossibility is
    the more useful message and it is not a matter of degree."""
    d = staking.evaluate(_r("85–86", p_model=0.30, price=0.29), CFG,
                         extremum="min", ratchet=(83.92, 84.12))
    assert d.place is False
    assert "edge" not in d.reason.lower(), d.reason
