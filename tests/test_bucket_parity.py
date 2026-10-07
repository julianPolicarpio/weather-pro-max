"""Kalshi's 2F ladder parity is a coin flip, and scoring must not assume one.

THE BUG THIS PINS (found 2026-07-28). `hour_curve.kalshi_bucket` and
`cities._kalshi_lo` both hardcoded an EVEN ladder (68-69, 70-71). Measured over
68 settled days per series, Kalshi's actual parity splits roughly evenly --
KXLOWTLAX 36 odd / 32 even, KXHIGHMIA 35 odd / 33 even -- because each day's
ladder is centred on the consensus forecast. Fixed-even scoring is therefore
right about half the time, and since `hour_curve` is the source of
`gating.HOUR_ACCURACY` and hence `MIN_DECISION_HOUR`, the error reached the
COMMIT HOURS, not just the display.
"""
import pytest

from highcast.buckets import (expected_bucket_hit, ladder_lo, modal_bucket,
                              pmf_mass)


@pytest.mark.parametrize("t,parity,expected", [
    (68, 0, 68), (69, 0, 68), (70, 0, 70), (71, 0, 70),   # even ladder
    (68, 1, 67), (69, 1, 69), (70, 1, 69), (71, 1, 71),   # odd ladder
    (73, 1, 73), (74, 1, 73),                             # live KXLOWTMIA ladder
    (88, 0, 88), (89, 0, 88),                             # live KXHIGHMIA ladder
])
def test_ladder_lo_matches_the_real_kalshi_ladders(t, parity, expected):
    assert ladder_lo(t, parity) == expected


def test_ladder_lo_handles_negative_temperatures():
    """Kalshi lists down to -30F; a naive `t - 1 if odd` breaks on negatives
    under Python's floor semantics if written with // instead of %."""
    for t in range(-30, 5):
        for par in (0, 1):
            lo = ladder_lo(t, par)
            assert lo % 2 == par and lo <= t <= lo + 1


def test_modal_bucket_differs_by_parity():
    """The same pmf picks DIFFERENT buckets on the two ladders -- which is the
    entire reason a fixed-parity scorer is wrong."""
    pmf = {68: 0.05, 69: 0.90, 70: 0.05}
    assert modal_bucket(pmf, 0)[0] == 68        # even ladder -> 68-69
    assert modal_bucket(pmf, 1)[0] == 69        # odd  ladder -> 69-70


def test_expected_hit_is_the_average_of_the_two_ladders():
    pmf = {68: 0.05, 69: 0.90, 70: 0.05}
    assert expected_bucket_hit(pmf, 69) == 1.0   # inside 68-69 AND 69-70
    assert expected_bucket_hit(pmf, 70) == 0.5   # inside 69-70 only
    assert expected_bucket_hit(pmf, 68) == 0.5   # inside 68-69 only
    assert expected_bucket_hit(pmf, 75) == 0.0   # neither


def test_a_fixed_even_scorer_would_have_overstated_this_day():
    """Concrete instance of the bug: on an ODD day the even scorer scores a
    miss as a hit. Averaging reports 0.5 instead of a confident 1.0."""
    pmf = {69: 0.6, 70: 0.4}
    # even ladder modal = 68-69; odd ladder modal = 69-70. Truth 70.
    assert modal_bucket(pmf, 0)[0] == 68 and modal_bucket(pmf, 1)[0] == 69
    assert expected_bucket_hit(pmf, 70) == 0.5


def test_the_live_path_is_unaffected_because_it_reads_the_real_ladder():
    """`pmf_mass` sums over floor/cap taken from the Kalshi API, so live
    probabilities never guess a parity. This is why committed bets stayed
    correct while every offline hit-rate figure was wrong."""
    pmf = {68: 0.1, 69: 0.6, 70: 0.3}
    assert pmf_mass(pmf, 68, 69) == 0.7      # even-ladder market row
    assert pmf_mass(pmf, 69, 70) == 0.9      # odd-ladder market row
