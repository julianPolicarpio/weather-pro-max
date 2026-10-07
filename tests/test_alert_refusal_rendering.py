"""A REFUSAL MUST NOT LOOK LIKE A BET.

WHAT WAS WRONG. `build_message` rendered every `app_bets` row identically —
Bucket, Model, Kalshi says, **Edge**, **Stake → payout if it wins**, title
"Committed at HH:MM Manila" — and only the Reason field at the bottom said
whether the desk actually wanted the trade. Measured 2026-08-10 on the live
`alert_log`: **26 of 44 commit alerts (59%) were refusals** presented this way.

The two worst:

    2026-08-10  KLAX low 66–67  model 45%  kalshi  2%  edge +43%  "too late to place"
    2026-08-06  KLAX low 66–67  model 62%  kalshi 12%  edge +50%  no-edge refusal

Both headline a huge edge on a bucket the market has priced as nearly dead, and
both render `USD 1.00 → USD 50.00`. HANDOFF §4.2 already named that pattern:
"Not an edge — the bucket was already dead… our 'edge' was staleness wearing a
value bet's clothes." The alert was formatting the project's best-documented
failure mode as an opportunity.

And the refusals were RIGHT — most carry won=1. The desk declined, the market
was correct, and the alert advertised the decision as a missed windfall.

THE RULE: if `reason` does not start with PLACE, there is no stake, no payout,
and nothing was committed. Say so first, not last.
"""
import pytest

from highcast import notify
from highcast.config import load_config

CFG = load_config("config/klax.yaml")


def _bet(reason, **kw):
    base = dict(metric="low", target_date="2026-08-09", bucket="66–67",
                p_model=0.45, price=0.02, stake=1.0, payout_if_win=50.0,
                placed_manila="10:31", won=None, reason=reason)
    base.update(kw)
    return base


REFUSAL = "past the tradeable window — too late to place"
NO_EDGE = "no edge: 66–67 model 45% vs market 2% = +43% < 3% required — do not place"
PLACED = "PLACE 66–67: 45% vs market 2% (edge +43%) -> USD 1.00, 50 contracts"


def _text(payload):
    return str(payload)


@pytest.mark.parametrize("reason", [REFUSAL, NO_EDGE,
                                    "market too thin: spread 21¢ > 10¢ max",
                                    "unmeasured confidence: sits above the band"])
def test_a_refusal_never_shows_a_payout(reason):
    """THE MOST MISLEADING ELEMENT. '$1.00 → $50.00' is what the eye lands on,
    and for a refused bet there is no stake and no payout — the figure is
    fabricating an opportunity that was explicitly declined."""
    body = _text(notify.build_message(CFG, _bet(reason), {"reason": reason}))
    assert "payout" not in body.lower(), body
    assert "50.00" not in body, body


@pytest.mark.parametrize("reason", [REFUSAL, NO_EDGE])
def test_a_refusal_does_not_claim_to_be_committed(reason):
    body = _text(notify.build_message(CFG, _bet(reason), {"reason": reason}))
    assert "Committed at" not in body, body
    assert "NO BET" in body, body


def test_the_reason_comes_first_on_a_refusal(reason=NO_EDGE):
    """It was the last field, under the numbers. On a refusal it IS the
    message."""
    fields = notify.build_message(CFG, _bet(reason),
                                  {"reason": reason})["embeds"][0]["fields"]
    assert fields[0]["name"].lower().startswith("why"), fields[0]


def test_a_big_edge_on_a_dead_bucket_is_labelled_not_celebrated():
    """+43% against a 2¢ market is the project's best-documented failure mode
    (HANDOFF §4.2), not a windfall. It must not render as a bare '+43%'."""
    body = _text(notify.build_message(CFG, _bet(NO_EDGE), {"reason": NO_EDGE}))
    assert "+43%" not in body or "stale" in body.lower() or "dead" in body.lower(), body


def test_a_real_place_is_UNCHANGED():
    """The regression that matters: a genuine commit must keep every field it
    had, payout included."""
    body = _text(notify.build_message(
        CFG, _bet(PLACED, stake=1.0, payout_if_win=50.0), {"reason": PLACED}))
    assert "Committed at" in body
    assert "payout" in body.lower()
    assert "50.00" in body
    assert "NO BET" not in body


def test_placed_detection_matches_the_project_definition():
    """`_is_place` in paper_account is the one definition: reason starts with
    PLACE, and NULL counts as SKIP. Diverging here would make the alert
    disagree with the ledger about what happened."""
    assert notify._alert_is_place(PLACED) is True
    assert notify._alert_is_place(NO_EDGE) is False
    assert notify._alert_is_place(None) is False
    assert notify._alert_is_place("") is False
