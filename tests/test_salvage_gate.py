"""The pre-registered salvage gate must be able to return every verdict.

WHY THIS FILE EXISTS. `scripts/salvage_alert_gate.py` will sit at "NOT YET
DECIDABLE" for ~3.5 weeks, so its SHIP and DO-NOT-SHIP branches never execute
in practice until the one day they decide something. That is the exact shape of
errors #8 and #11 (HANDOFF §0b): a guard that looks like coverage and cannot
fire. Both were caught only by a test asserting the guard GOES OFF.

So these drive `decide()` through every threshold, each failing in isolation,
plus the firing rule itself. Nothing here re-derives or re-tunes the gate — the
frozen constants are imported, never restated, so a future edit to CONC/SALV
breaks nothing here and is caught by `test_constants_are_frozen`.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "salvage_alert_gate",
    Path(__file__).resolve().parents[1] / "scripts" / "salvage_alert_gate.py")
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)          # must NOT touch the live DB


def bet(won, p=0.70, placed="2026-08-20T15:00:00+00:00"):
    return {"bucket": "80–81", "p_model": p, "won": 1 if won else 0,
            "placed_utc": placed}


def lad(ours, other, bid=None, label="80–81"):
    return [{"label": label, "market_price": ours,
             "yes_bid": ours if bid is None else bid},
            {"label": "82–83", "market_price": other, "yes_bid": other}]


def snap(ours, other, bid=None, when="2026-08-20T18:00:00+00:00"):
    return [(when, lad(ours, other, bid))]


# ── the firing rule ──────────────────────────────────────────────────────────

def test_fires_only_when_both_conditions_hold():
    b = bet(False)
    assert gate.fires(snap(0.30, 0.65), b) is not None          # both hold
    assert gate.fires(snap(0.30, 0.55), b) is None              # market < CONC
    assert gate.fires(snap(0.10, 0.65), b) is None              # ours < SALV


def test_fires_at_the_exact_thresholds():
    """Boundaries are `>=` in the registration; pin them so a later `>` is a
    test failure rather than a silent redefinition of the frozen rule."""
    b = bet(False)
    assert gate.fires(snap(gate.SALV, gate.CONC), b) is not None


def test_exit_is_priced_at_the_bid_not_the_mid():
    """The mid is not a price anyone can sell into."""
    when, px = gate.fires(snap(0.30, 0.70, bid=0.24), bet(False))
    assert px == pytest.approx(0.24)


def test_missing_bid_falls_back_to_the_mid():
    seq = [("2026-08-20T18:00:00+00:00",
            [{"label": "80–81", "market_price": 0.30},
             {"label": "82–83", "market_price": 0.70}])]
    assert gate.fires(seq, bet(False))[1] == pytest.approx(0.30)


def test_snapshots_before_the_commit_are_ignored():
    seq = snap(0.30, 0.70, when="2026-08-20T09:00:00+00:00")
    assert gate.fires(seq, bet(False)) is None


def test_our_bucket_absent_from_the_ladder_does_not_fire_or_raise():
    seq = [("2026-08-20T18:00:00+00:00",
            [{"label": "90–91", "market_price": 0.95, "yes_bid": 0.94}])]
    assert gate.fires(seq, bet(False)) is None


# ── the verdict, one branch at a time ────────────────────────────────────────

def _world(fire_losers, fire_winners, quiet_losers, quiet_winners, bid=0.90):
    """Build (oos, bets, fired) with the requested 2x2 and a chosen exit bid.

    `p_model` alternates across the median so both `stated` halves are
    populated; the split is on the median of these values.
    """
    bets, fired, oos = {}, {}, []
    n = 0
    for count, won, does_fire in ((fire_losers, False, True),
                                  (fire_winners, True, True),
                                  (quiet_losers, False, False),
                                  (quiet_winners, True, False)):
        for _ in range(count):
            k = ("KLAX", "high", f"2026-09-{n+1:02d}")
            bets[k] = bet(won, p=0.60 if n % 2 else 0.80)
            fired[k] = ("2026-09-01T18:00:00+00:00", bid) if does_fire else None
            oos.append(k)
            n += 1
    return oos, bets, fired


def test_ship_when_every_threshold_passes():
    """⚠️ THE BRANCH THAT OTHERWISE NEVER RUNS."""
    oos, bets, fired = _world(11, 1, 2, 40, bid=0.95)
    v = gate.decide(oos, bets, fired)
    assert v["primary"] and v["stated"] and v["money"]
    assert v["ship"] is True


def test_no_ship_when_the_primary_test_is_not_significant():
    oos, bets, fired = _world(3, 3, 8, 16, bid=0.95)
    v = gate.decide(oos, bets, fired)
    assert v["primary"] is False
    assert v["ship"] is False


def test_no_ship_when_the_effect_points_the_wrong_way():
    """Firing must predict LOSING. Predicting winning is not a success."""
    oos, bets, fired = _world(1, 11, 40, 2, bid=0.95)
    v = gate.decide(oos, bets, fired)
    assert v["diff"] < 0 and v["primary"] is False


def test_no_ship_when_money_fails_though_hit_rate_passes():
    """⚠️ THE PRE-REGISTERED EXPECTED FAILURE. A great classifier that fires
    when our bucket is cheap loses money: the winners abandoned pay 1.00 and
    the losers salvaged pay the bid."""
    oos, bets, fired = _world(11, 1, 2, 40, bid=0.05)
    v = gate.decide(oos, bets, fired)
    assert v["primary"] is True, "hit rate is fine..."
    assert v["money"] is False, "...and it still must not ship"
    assert v["ship"] is False
    assert v["delta"] == pytest.approx(11 * 0.05 + (0.05 - 1))


def test_money_arm_counts_abandoned_winners_against_the_rule():
    oos, bets, fired = _world(2, 1, 0, 0, bid=0.40)
    v = gate.decide(oos, bets, fired)
    assert v["delta"] == pytest.approx(2 * 0.40 + (0.40 - 1.0))


def test_no_ship_when_the_sign_flips_inside_a_stated_half():
    """The check that killed four earlier ideas: a rule that only restates the
    model's own probability shows the effect in one half and not the other."""
    bets, fired, oos = {}, {}, []
    # low-confidence half: firing predicts losing, strongly
    for i in range(20):
        k = ("KLAX", "high", f"2026-09-{i+1:02d}")
        bets[k] = bet(i >= 8, p=0.55)
        fired[k] = ("x", 0.95) if i < 8 else None
    # high-confidence half: firing predicts WINNING -- the sign flips
    for i in range(20):
        k = ("KMIA", "high", f"2026-09-{i+1:02d}")
        bets[k] = bet(i < 8, p=0.85)
        fired[k] = ("x", 0.95) if i < 8 else None
    oos = list(bets)
    v = gate.decide(oos, bets, fired)
    assert v["stated"] is False
    assert v["ship"] is False


def test_an_empty_stated_half_fails_closed():
    """No data cannot demonstrate the sign HOLDS."""
    oos, bets, fired = _world(11, 1, 2, 40, bid=0.95)
    for k in oos:                      # collapse both halves onto one value
        bets[k]["p_model"] = 0.70
    v = gate.decide(oos, bets, fired)
    assert v["stated"] is False, "a degenerate split must not pass"


def test_never_firing_cannot_ship():
    oos, bets, fired = _world(0, 0, 19, 49)
    v = gate.decide(oos, bets, fired)
    assert v["primary"] is False and v["ship"] is False


# ── the registration itself ──────────────────────────────────────────────────

def test_constants_are_frozen():
    """⚠️ These were pre-registered on 2026-08-16. Changing one invalidates the
    gate; it does not improve it. If this test fails, the correct response is to
    revert the constant and register a NEW gate, not to update this line."""
    assert (gate.CONC, gate.SALV) == (0.60, 0.20)
    assert gate.OOS_START == "2026-08-17"
    assert (gate.N_MIN, gate.ALPHA) == (100, 0.05)


def test_fisher_matches_the_registered_headline():
    """The in-sample 2x2 the hypothesis came from: 11/4 fired vs 8/45 silent."""
    assert gate.fisher(11, 4, 8, 45) == pytest.approx(3.83e-5, rel=0.02)
