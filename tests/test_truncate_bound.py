"""Outcomes the ratchet has already ruled out must carry NO probability.

CAUGHT BY JULIAN, 2026-08-01. With KLAX's running low already at 68.10, the
nowcast still assigned 1.1% to ">=70". A daily MINIMUM can only fall, so that
bucket cannot occur — the mass is spent on a dead outcome and every LIVE bucket
is diluted by exactly that much.

The load-bearing test is `test_a_bucket_still_reachable_by_ROUNDING_survives`:
settlement rounds, so a running min of 68.10 can still settle at 68. Truncating
against the raw bound instead of the rounded one would delete the single most
likely outcome on the board — a fix worse than the bug.
"""
import pytest

from highcast.buckets import dead_mass, truncate_at_bound

PMF = {66: 0.10, 67: 0.15, 68: 0.50, 69: 0.20, 70: 0.05}


def test_impossible_buckets_are_removed_for_a_min():
    out = truncate_at_bound(PMF, 68.10, "min")
    assert 69 not in out and 70 not in out
    assert set(out) == {66, 67, 68}


def test_impossible_buckets_are_removed_for_a_max():
    out = truncate_at_bound(PMF, 68.10, "max")
    assert 66 not in out and 67 not in out
    assert set(out) == {68, 69, 70}


def test_a_bucket_still_reachable_by_ROUNDING_survives():
    """THE ONE THAT MATTERS. A running min of 68.10 can still settle at 68 —
    anything in [67.50, 68.49] rounds there. Comparing against the raw bound
    would kill 68, which holds half the mass here."""
    out = truncate_at_bound(PMF, 68.10, "min")
    assert 68 in out
    assert out[68] == pytest.approx(0.50 / 0.75)


def test_the_survivors_are_renormalised():
    out = truncate_at_bound(PMF, 68.10, "min")
    assert sum(out.values()) == pytest.approx(1.0)


def test_live_buckets_gain_exactly_the_dead_mass():
    """The point of renormalising: the surviving buckets keep their RELATIVE
    shape and absorb what the dead ones were holding."""
    out = truncate_at_bound(PMF, 68.10, "min")
    assert out[66] / out[67] == pytest.approx(PMF[66] / PMF[67])
    assert out[68] > PMF[68]


def test_dead_mass_reports_what_was_dropped():
    assert dead_mass(PMF, 68.10, "min") == pytest.approx(0.25)
    assert dead_mass(PMF, 68.10, "max") == pytest.approx(0.25)


def test_no_bound_means_no_truncation():
    """A rolled-forward target has no observations yet — nothing is ruled out."""
    assert truncate_at_bound(PMF, None, "min") == PMF
    assert dead_mass(PMF, None, "min") == 0.0


def test_none_pmf_is_passed_through():
    assert truncate_at_bound(None, 68.0, "min") is None
    assert dead_mass(None, 68.0, "min") == 0.0


def test_the_input_is_not_mutated():
    before = dict(PMF)
    truncate_at_bound(PMF, 68.10, "min")
    assert PMF == before


def test_a_bound_that_would_kill_everything_returns_the_original():
    """Better a distribution the ratchet contradicts than an empty one: a
    caller that must choose a bucket cannot act on {}, and a silent empty dict
    surfaces downstream as a mysterious 'no prediction'."""
    out = truncate_at_bound(PMF, -100.0, "min")
    assert out == PMF


def test_truncation_can_change_the_modal_bucket():
    """This is a DECISION-PATH change, not cosmetic: removing dead mass can
    promote a different bucket to the top, which is the whole point."""
    from highcast.buckets import argmax_pair
    skewed = {66: 0.05, 67: 0.10, 68: 0.25, 69: 0.30, 70: 0.30}
    before = argmax_pair(skewed)["label"]
    after = argmax_pair(truncate_at_bound(skewed, 68.10, "min"))["label"]
    assert before != after
