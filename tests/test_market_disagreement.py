"""The commit alert must show where the counterparty outbids us most.

WHY, 2026-08-19. KMIA high 2026-08-18 settled 99–100. At commit the model gave
that rung **2.6%** while Kalshi paid **36.5%**, and nothing on the desk showed
it — the day grade said a clean B, correctly, because every INPUT was fine. The
failure was the distribution, and no plumbing check watches distributions.

⚠️ THIS IS A DISPLAY, NOT A GATE, and the tests pin that too. Measured over 79
settled day-markets, corr(D, LOST) = +0.571 CI [+0.400,+0.703] — but
corr(D, our ask) = −0.798, and within price bands D separates +60% below 40c
and −8% at 60–80c. Nearly everything it knows the ask already knew. A D veto
would be a cheap-contract veto, and QUEUE item 1 measured the cheap bets AHEAD
on margin. The trigger question is pre-registered separately in
`scripts/salvage_alert_gate.py`.
"""
from highcast.marketsensor import disagreement
from highcast.notify import build_message


class _Cfg:
    class station:
        icao = "KMIA"
        tz = "America/New_York"

    class staking:
        currency = "USD"
    discord_webhook_url = "https://example.invalid/hook"


# The real 2026-08-18 KMIA high ladder at the 17:02Z commit.
LADDER_0818 = [
    {"label": "≤92", "p_model": 0.0, "yes_ask": 0.005},
    {"label": "93–94", "p_model": 0.0, "yes_ask": 0.005},
    {"label": "95–96", "p_model": 0.4332, "yes_ask": 0.065},
    {"label": "97–98", "p_model": 0.5367, "yes_ask": 0.58},
    {"label": "99–100", "p_model": 0.026, "yes_ask": 0.365},
    {"label": "≥101", "p_model": 0.0041, "yes_ask": 0.005},
]
BET = {"metric": "high", "target_date": "2026-08-18", "bucket": "97–98",
       "p_model": 0.5367, "price": 0.58, "placed_manila": "01:02",
       "stake": 0.0, "payout_if_win": 0.0, "won": None,
       "reason": "no edge: -8% < 3% required — do not place"}


def test_finds_the_rung_the_market_outbids_us_on():
    d = disagreement(LADDER_0818)
    assert d["label"] == "99–100"          # the rung that actually settled
    assert round(d["model"], 3) == 0.026
    assert 0.30 < d["gap"] < 0.40


def test_asks_are_normalised_for_the_overround():
    """Raw asks sum to >1; comparing them to a pmf would overstate every rung."""
    d = disagreement(LADDER_0818)
    # ⚠️ The denominator is the WHOLE ladder, not just the rungs that clear
    # MIN_MARKET. The floor decides which rung is SELECTED; the normalisation
    # has to span every rung or the result is not a probability.
    total = sum(x["yes_ask"] for x in LADDER_0818)
    assert total > 1.0                            # the overround is real
    assert d["market"] < 0.365                    # normalised down from the raw ask
    assert abs(d["market"] - 0.365 / total) < 1e-9


def test_dead_rungs_are_not_reported_as_disagreements():
    """⚠️ The dashboard's first render said "≤74 — Kalshi 1% vs our 0% (+1%)":
    a disagreement between two numbers that are both nothing. `MIN_MARKET`
    filters rungs the market is not paying for. Display hygiene only — swept
    over the same 79 day-markets it moves corr(D, LOST) +0.571 -> +0.569 and
    leaves the 11 days with D >= 30% identical."""
    lad = [{"label": "≤74", "p_model": 0.0, "yes_ask": 0.01},
           {"label": "75–76", "p_model": 0.90, "yes_ask": 0.95}]
    assert disagreement(lad)["label"] == "75–76"


def test_falls_back_rather_than_going_silent_when_every_rung_is_thin():
    """A ladder where nothing clears the floor still returns its best rung —
    silence would be indistinguishable from "no data"."""
    lad = [{"label": "a", "p_model": 0.10, "yes_ask": 0.02},
           {"label": "b", "p_model": 0.01, "yes_ask": 0.03}]
    assert disagreement(lad)["label"] == "b"


def test_no_ladder_or_unpriced_ladder_yields_nothing():
    assert disagreement(None) is None
    assert disagreement([]) is None
    assert disagreement([{"label": "x", "p_model": 0.5}]) is None      # no ask
    assert disagreement([{"label": "x", "yes_ask": 0.5}]) is None      # no model
    assert disagreement([{"label": "x", "p_model": 0.5, "yes_ask": 0}]) is None


def test_commit_alert_shows_the_disagreement():
    msg = build_message(_Cfg(), BET, {"disagreement": disagreement(LADDER_0818)})
    f = next(x for x in msg["embeds"][0]["fields"]
             if x["name"] == "Market outbids us most on")
    assert "99–100" in f["value"] and "3%" in f["value"]


def test_field_is_absent_when_the_market_agrees_with_us():
    """No field rather than a misleading 0% — the top disagreement being our own
    bucket means the market is on our side."""
    lad = [{"label": "97–98", "p_model": 0.30, "yes_ask": 0.90},
           {"label": "99–100", "p_model": 0.70, "yes_ask": 0.10}]
    msg = build_message(_Cfg(), BET, {"disagreement": disagreement(lad)})
    assert all(x["name"] != "Market outbids us most on"
               for x in msg["embeds"][0]["fields"])


def test_a_rounding_to_zero_gap_is_not_reported_as_a_disagreement():
    """"Kalshi 95% vs our 95% (+0%)" asserts a disagreement that is not there.
    The cut is the DISPLAY PRECISION, not a measured threshold."""
    lad = [{"label": "81–82", "p_model": 0.9481, "yes_ask": 0.95},
           {"label": "83–84", "p_model": 0.05, "yes_ask": 0.048}]
    d = disagreement(lad)
    assert d["gap"] < 0.005
    msg = build_message(_Cfg(), BET, {"disagreement": d})
    assert all(x["name"] != "Market outbids us most on"
               for x in msg["embeds"][0]["fields"])


def test_alert_still_builds_with_no_disagreement_key():
    """A decoration must never be able to break the alert it decorates."""
    msg = build_message(_Cfg(), BET, {})
    assert msg["embeds"][0]["title"].startswith("NO BET")


def test_display_only_no_veto_wiring():
    """⚠️ PINS THE DESIGN. If someone later routes `disagreement` into a
    staking or gating decision, this fails and they must re-read the docstring:
    corr(D, ask) = -0.798, and a D veto is a cheap-contract veto."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "highcast"
    for mod in ("staking.py", "gating.py"):
        assert "disagreement" not in (src / mod).read_text(), (
            f"{mod} references `disagreement` — it is a DISPLAY, not a gate. "
            "The trigger question is pre-registered in salvage_alert_gate.py.")
