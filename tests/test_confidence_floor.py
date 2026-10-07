"""The bot may act only when the model is measurably right 4 days in 5.

THE FLOOR IS A POLICY TARGET, NOT A FITTED PARAMETER. Per market it is the
LOWEST stated-probability threshold whose realised hit rate has a 95% lower
bound >= 0.80. A market with no such threshold gets NO floor and never fires.

Measured 2026-07-31, n~570/arm, walk-forward:

    KLAX HIGH h13  .50 69.5/65.6  .60 70.3/66.1  .70 71.4/65.8  .80 75.0/63.9
                   -> no floor clears; the bot never trades KLAX HIGH
    KMIA HIGH h14  .50 80.5/76.9  .60 82.1/78.6  .70 85.5/81.9  .80 93.4/89.6
                   -> floor 0.70, fires on 75.3% of days at 85.5%

KLAX HIGH is excluded by its own calibration AND, independently, by price: at
h13 the favoured bucket traded 0.75-0.96 (median ~0.90) while the model can
justify only ~69c, and the market's favoured bucket was right 5 of 6. Two
independent lines of evidence agreeing is the strongest result in the spec.
DO NOT lower the target to admit KLAX HIGH.
"""
import pytest

from highcast import gating


def test_kmia_high_has_the_measured_floor():
    """0.70 -> 0.60 on 2026-08-02. Pinned so a change is a visible edit.

    ⚠️ RE-MEASURED ON THE QUANTITY THE BOT ACTUALLY USES. Kalshi's ladder has
    ONE parity per day, so the new figure scores each parity SEPARATELY and
    pools the day-ladders (n=1336 over 668 Jun-Sep walk-forward days at h14).
    An earlier cut used `expected_bucket_hit`, which AVERAGES both parities --
    right for scoring the model, but kinder than the bot's reality, because a
    good odd-parity read rescues a bad even-parity one. Averaged, the model
    looked +14.9 pt UNDER-confident at 50-60%; on the real quantity it is
    -4.2 pt OVER-confident. The conclusion inverted.

        band      n     stated   actual    gap
        50-60%   168    54.8%    50.6%   -4.2%
        60-70%   155    65.3%    61.3%   -4.0%
        70-80%   154    75.4%    78.6%   +3.2%

    0.60 and no lower: a floor admits its MARGINAL band, not the cohort
    average. At 0.60 the added days hit 61.3%; below it the 50-60% band hits
    50.6% with the worst over-confidence measured. Holds out of sample --
    the >=0.60 cohort hit 86.9% on 2020-24 and 87.5% on 2025-26.
    """
    assert gating.CONFIDENCE_FLOOR[("KMIA", "high", 14)] == 0.60


def test_klax_high_has_no_floor_and_therefore_never_fires():
    """Not an oversight. No threshold reached a 0.80 lower bound."""
    assert ("KLAX", "high", 13) not in gating.CONFIDENCE_FLOOR
    assert gating.confidence_floor_ok("KLAX", "high", 13, 0.99) is False


def test_a_probability_at_or_above_the_floor_passes():
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.60) is True
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.93) is True


def test_a_probability_below_the_floor_is_refused():
    assert gating.confidence_floor_ok("KMIA", "high", 14, 0.599) is False


def test_an_unmeasured_market_fails_closed():
    assert gating.confidence_floor_ok("KMIA", "low", 7, 0.99) is False
    assert gating.confidence_floor_ok("NOPE", "high", 14, 0.99) is False


def test_the_wrong_hour_fails_closed():
    """The floor is keyed by hour. A caller at the old h11 must not inherit the
    h14 floor — the two hours have completely different accuracy."""
    assert gating.confidence_floor_ok("KMIA", "high", 11, 0.99) is False


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_a_missing_probability_fails_closed(bad):
    assert gating.confidence_floor_ok("KMIA", "high", 14, bad) is False
