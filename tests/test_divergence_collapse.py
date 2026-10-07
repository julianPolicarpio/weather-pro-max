"""The divergence alert watches the COMMITTED bucket's confidence, not just which
bucket ranks first.

Regression for the 2026-07-29 KLAX LOW: committed 67-68 at 81.2%, fell to 54.7%
an hour later while STILL nominally leading, and the old rule stayed silent for
two more hours. The fixture below is that day's real probability trace, taken from
the `predictions` table.
"""
import pytest

from highcast.notify import (DIVERGENCE_DROP, DIVERGENCE_FLOOR,
                             build_divergence, divergence_reason)

# Real trace, KLAX LOW target 2026-07-29, P(67-68) by station-local hour.
TRACE_0729 = {4: 0.812, 5: 0.547, 6: 0.559, 7: 0.315, 8: 0.252}
COMMITTED_P = TRACE_0729[4]
COMMITTED = "67-68"


def _top_label_at(hour):
    """Which bucket led at each hour. 67-68 held the lead through h6; >=69 took it
    at h7. This is what makes the old rule silent at h5 and h6."""
    return COMMITTED if hour <= 6 else "69-70"


def test_old_rule_alone_would_stay_silent_at_hour_5():
    """Guard against regressing to 'a different bucket ranks first'."""
    assert _top_label_at(5) == COMMITTED  # nothing has overtaken it


def test_fires_at_hour_5_on_confidence_collapse():
    """26.5 points gone from the commit value -> must fire."""
    reason = divergence_reason(COMMITTED_P, TRACE_0729[5], _top_label_at(5),
                               COMMITTED)
    assert reason == "collapsed"


def test_does_not_fire_at_the_commit_hour_itself():
    reason = divergence_reason(COMMITTED_P, TRACE_0729[4], _top_label_at(4),
                               COMMITTED)
    assert reason is None


def test_the_whole_0729_trace_fires_two_hours_earlier_than_the_old_rule():
    first_new = next(h for h in sorted(TRACE_0729)
                     if divergence_reason(COMMITTED_P, TRACE_0729[h],
                                          _top_label_at(h), COMMITTED))
    first_old = next(h for h in sorted(TRACE_0729)
                     if _top_label_at(h) != COMMITTED)
    assert first_new == 5
    assert first_old == 7
    assert first_old - first_new == 2


def test_drop_trigger_is_independent_of_the_floor():
    """A bucket that commits at 95% and falls to 70% must fire on the DROP even
    though 70% is comfortably above the 60% floor."""
    assert divergence_reason(0.95, 0.70, COMMITTED, COMMITTED) == "collapsed"
    assert 0.70 > DIVERGENCE_FLOOR


def test_floor_trigger_is_independent_of_the_drop():
    """A bucket committed at 65% that slips to 58% has lost only 7 points but is
    now below the floor -- a sub-60% pick is worth knowing about regardless."""
    assert divergence_reason(0.65, 0.58, COMMITTED, COMMITTED) == "collapsed"
    assert (0.65 - 0.58) < DIVERGENCE_DROP


def test_a_modest_wobble_does_not_fire():
    """Churn is inherent (see the pooling and parametric negative results). The
    alert must not fire on ordinary hour-to-hour movement or it becomes noise."""
    assert divergence_reason(0.81, 0.75, COMMITTED, COMMITTED) is None


def test_overtaken_still_fires_even_with_healthy_confidence():
    """Both triggers remain live. If another bucket leads, that is news even if
    ours has barely moved."""
    assert divergence_reason(0.81, 0.79, "69-70", COMMITTED) == "overtaken"


def test_missing_live_probability_stays_silent_rather_than_guessing():
    assert divergence_reason(0.81, None, COMMITTED, COMMITTED) is None


def test_missing_commit_probability_stays_silent():
    """CHANGED 2026-07-30. This once asserted the floor fires even without a
    commit probability. The KMIA-LOW false alarm showed why that is wrong: a
    'collapse' alert claims confidence FELL, and that claim cannot be
    substantiated without knowing where it started. If committed_p is None we do
    not know it fell, so we do not cry collapse. All live rows carry p_model; this
    only affects legacy rows, where a false 'collapsed' is worse than silence."""
    assert divergence_reason(None, 0.40, COMMITTED, COMMITTED) is None
    assert divergence_reason(None, 0.75, COMMITTED, COMMITTED) is None


class _Cfg:
    class station:
        icao = "KLAX"


def test_collapse_message_does_not_claim_the_model_favours_another_bucket():
    """The wording bug this guards: with reason='collapsed' the leading bucket IS
    the committed one, so 'model now favours 67-68' would read as agreement while
    reporting a 26-point loss."""
    bet = {"metric": "low", "target_date": "2026-07-29", "bucket": COMMITTED,
           "p_model": COMMITTED_P}
    msg = build_divergence(_Cfg(), bet, COMMITTED, TRACE_0729[5],
                           reason="collapsed", held_p=TRACE_0729[5])
    embed = msg["embeds"][0]
    assert "confidence collapsed" in embed["title"]
    assert "Model now favours" not in str(embed["fields"])
    assert "81% → 55%" in str(embed["fields"])
    # and it must be labelled a warning, not a fix
    assert "early warning" in embed["footer"]["text"]


def test_overtaken_message_keeps_the_original_wording():
    bet = {"metric": "low", "target_date": "2026-07-29", "bucket": COMMITTED,
           "p_model": COMMITTED_P}
    msg = build_divergence(_Cfg(), bet, "69-70", 0.62, reason="overtaken")
    embed = msg["embeds"][0]
    assert "model moved" in embed["title"]
    assert "Model now favours" in str(embed["fields"])


@pytest.mark.parametrize("held_p", [0.0, 0.01, 0.30])
def test_a_fully_collapsed_bucket_always_fires(held_p):
    assert divergence_reason(COMMITTED_P, held_p, COMMITTED, COMMITTED) \
        == "collapsed"


def test_a_bet_that_committed_below_the_floor_does_not_fire_on_the_floor(tmp_path=None):
    """Caught live 2026-07-30: the KMIA LOW committed at 53%, already under the
    60% floor. The old floor rule fired 'confidence collapsed' immediately, on a
    probability that never moved — the alert read '53% -> 53% (0%)'. A pick that
    was always weak has not collapsed. The floor must require confidence to have
    been ABOVE it at commit."""
    # committed weak, unchanged: no fire
    assert divergence_reason(0.53, 0.53, "80-81", "80-81") is None
    # committed weak, drifts a little lower but still no 20pt drop: no fire
    assert divergence_reason(0.53, 0.49, "80-81", "80-81") is None
    # committed weak but then loses 20+ points: the DROP trigger still fires
    assert divergence_reason(0.53, 0.30, "80-81", "80-81") == "collapsed"
    # committed ABOVE the floor and falls through it: fires, as intended
    assert divergence_reason(0.65, 0.58, "80-81", "80-81") == "collapsed"
    # committed above floor, still above: no fire
    assert divergence_reason(0.72, 0.63, "80-81", "80-81") is None
