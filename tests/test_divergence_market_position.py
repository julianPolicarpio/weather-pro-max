"""The divergence alert must say where the MARKET stands, not just the model.

WHY, measured 2026-08-16 in `scripts/divergence_alert_eval.py` over 68 settled
day-markets. Julian asked why the model "moves to the wrong bucket" after KMIA
HIGH 2026-08-15, where the alert announced 94-95 at 64% while Kalshi had the
committed 92-93 at 96% and 92-93 duly settled.

The obvious fixes were measured and both were WRONG:

  * persistence (fire only after N consecutive disagreeing snapshots) changed
    essentially nothing -- N = 1, 2, 3, 4, 6 gave 30/30/30/30/29 fires with 16
    losers throughout;
  * suppressing the alert once the market had concentrated did not help, and
    started hurting below 0.90 (separation +45% -> +38% -> +25%, p 0.06).

⚠️ Those two figures were RESTATED after error #16: the first run filtered
replay snapshots by a reconstructed model kind and dropped a third of them. The
conclusions held; the numbers did not. The live column was never affected.

The trigger separates well as it stands (P(lost | fired) 50% vs 8% silent,
Fisher p = 2.8e-4), so it was left alone. What changed is the MESSAGE: it now
reports the market leader, our bucket's price, and whether the position can
still be exited. These tests pin that wording, because the wording is the fix.
"""
from highcast.notify import build_divergence, market_position


class _Cfg:
    class station:
        icao = "KMIA"
        tz = "America/New_York"
    discord_webhook_url = "https://example.invalid/hook"


BET = {"metric": "high", "target_date": "2026-08-15", "bucket": "92-93",
       "p_model": 0.51}


def _ladder(pairs):
    return [{"label": lab, "market_price": px} for lab, px in pairs]


def test_no_ladder_yields_no_position_and_no_extra_field():
    """A decoration must never be able to break the alert it decorates."""
    assert market_position("92-93", None) is None
    assert market_position("92-93", []) is None
    assert market_position("92-93", [{"label": "92-93"}]) is None  # unpriced
    msg = build_divergence(_Cfg(), BET, "94-95", 0.64)
    names = [f["name"] for f in msg["embeds"][0]["fields"]]
    assert "Market position" not in names


def test_market_still_backing_us_says_the_model_is_the_outlier():
    """The 2026-08-15 case. Kalshi at 96% on OUR bucket; the model is wrong."""
    pos = market_position("92-93", _ladder([("92-93", 0.96), ("94-95", 0.03)]))
    assert pos["leads"] is True
    assert "outlier" in pos["verdict"]
    msg = build_divergence(_Cfg(), BET, "94-95", 0.64,
                           market=_ladder([("92-93", 0.96), ("94-95", 0.03)]))
    field = next(f for f in msg["embeds"][0]["fields"]
                 if f["name"] == "Market position")
    assert "leader **92-93** 96%" in field["value"]
    assert "yours 92-93 96%" in field["value"]


def test_position_already_gone_is_labelled_a_post_mortem():
    """Measured: when the market hits 0.95 against us, ours is 1c in 14/14."""
    pos = market_position("92-93", _ladder([("94-95", 0.97), ("92-93", 0.01)]))
    assert pos["leads"] is False
    assert "post-mortem" in pos["verdict"]
    assert "1%" in pos["verdict"]


def test_market_against_us_while_still_exitable_flags_the_window():
    """The one shape with salvage value: 15 fires, 11 losers, 73% vs 15%,
    p = 3.8e-5 (0.0015 after x40 Bonferroni). Reported in the message and still
    deliberately NOT a trigger -- it is a cell chosen from a sweep."""
    pos = market_position("92-93", _ladder([("94-95", 0.68), ("92-93", 0.27)]))
    assert pos["leads"] is False
    assert "still worth something" in pos["verdict"]
    assert "27%" in pos["verdict"]


def test_undecided_market_is_reported_as_undecided():
    pos = market_position("92-93", _ladder([("94-95", 0.41), ("92-93", 0.35)]))
    assert "not made up its mind" in pos["verdict"]


def test_leader_can_differ_from_the_models_top_bucket():
    """`now_bucket` is the MODEL's pick; the market may lead somewhere else. The
    alert has to show both or it cannot be acted on."""
    market = _ladder([("92-93", 0.30), ("94-95", 0.22), ("96-97", 0.44)])
    msg = build_divergence(_Cfg(), BET, "94-95", 0.64, market=market)
    field = next(f for f in msg["embeds"][0]["fields"]
                 if f["name"] == "Market position")
    assert "leader **96-97** 44%" in field["value"]
    assert market_position("92-93", market)["leader"] == "96-97"


def test_collapsed_wording_keeps_the_market_position():
    """Both reasons carry the field; `collapsed` is the one where the committed
    bucket still leads the model, so the market's view matters most."""
    msg = build_divergence(_Cfg(), BET, "92-93", 0.25, reason="collapsed",
                           held_p=0.25,
                           market=_ladder([("94-95", 0.91), ("92-93", 0.05)]))
    field = next(f for f in msg["embeds"][0]["fields"]
                 if f["name"] == "Market position")
    assert "post-mortem" in field["value"]
