"""THE AGGRESSIVE-SIZING EXPERIMENT — three arms, one set of days.

Julian asked for the stake cap to come off when the model is very confident:
"remove the cap if its super confident bet 4 dollar or 5". This file pins the
three arms that question turned into, all of them PAPER ONLY:

  A  confidence_only     — what he asked for, literally: confident ⇒ bet big.
  B  confidence_x_return — the same instinct with the PAYOFF factored in, so a
                           95c pick that pays 5c does not get the same $5 as a
                           45c pick that pays 55c.
  C  control             — the shipped quarter-Kelly + 5% cap, unchanged.

WHY B EXISTS AT ALL. On the settled record the market price is what predicts
the outcome (+0.596 correlation) and the model's own edge signal is
ANTI-predictive (-0.506). The confident days are precisely the expensive days,
so arm A spends nearly the whole balance to win a few cents. Arm B is the test
of whether Julian's instinct survives once price is in the arithmetic.

NOTHING HERE TOUCHES CONFIG. `max_stake_frac` and `kelly_fraction` keep their
shipped values; the aggressive fractions live in this experiment's own
constants, and `test_arms_never_mutate_staking_config` is what holds that line.
"""
import dataclasses

import pytest

from highcast import db, paper_account
from highcast.config import load_config

from tests.test_paper_account import _load_archive

KLAX = load_config("config/klax.yaml")
KMIA = load_config("config/kmia.yaml")


@pytest.fixture
def archive_conn():
    conn = db.connect(":memory:")
    _load_archive(conn)
    return conn


@pytest.fixture
def cfgs():
    return {"KLAX": dataclasses.replace(KLAX, db_path=":memory:"),
            "KMIA": dataclasses.replace(KMIA, db_path=":memory:")}


def test_all_three_arms_run_on_the_same_days(archive_conn, cfgs):
    """The arms are only comparable if they see identical bets. A day one arm
    skips must still appear in its row list, as a no-trade — otherwise an arm
    could 'win' by quietly trading fewer days."""
    out = paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    arms = {a["arm"]: a for a in out["arms"]}
    assert set(arms) == {"confidence_only", "confidence_x_return", "control"}
    seen = {a["arm"]: [r["target_date"] for r in a["rows"]] for a in out["arms"]}
    assert len(set(map(tuple, seen.values()))) == 1, seen


def test_confidence_only_bets_big_on_a_confident_day(archive_conn, cfgs):
    """Arm A is the literal request: when confidence clears the floor, commit
    the aggressive fraction of the balance, cap or no cap."""
    out = paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    a = next(x for x in out["arms"] if x["arm"] == "confidence_only")
    traded = [r for r in a["rows"] if r["trade"]]
    assert traded, "arm A never traded — the confidence gate cannot be this tight"
    assert max(r["stake_frac"] for r in traded) > KMIA.staking.max_stake_frac * 2


def test_confidence_x_return_stands_down_on_a_tiny_payoff(archive_conn, cfgs):
    """Arm B's whole reason to exist: an expensive bucket is refused for the
    payoff, not for the confidence."""
    rows = paper_account._arm_rows_for_test(archive_conn, cfgs)
    dear = {"price": 0.95, "p_model": 0.99}
    d = paper_account._arm_decision("confidence_x_return", dear, 5.50, KMIA)
    assert d["trade"] is False
    assert "payoff" in d["reason"].lower() or "return" in d["reason"].lower()
    cheap = paper_account._arm_decision(
        "confidence_x_return", {"price": 0.45, "p_model": 0.90}, 5.50, KMIA)
    assert cheap["trade"] is True
    assert rows  # the fixture archive really does contain days to score


def test_control_arm_matches_the_shipped_daily_plan(archive_conn, cfgs):
    """Arm C must BE the shipped rule, not a re-implementation of it: same
    closing balance as `daily_plan_sim` over the same days."""
    out = paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    c = next(x for x in out["arms"] if x["arm"] == "control")
    plan = paper_account.daily_plan_sim(archive_conn, cfgs, 5.50)
    assert c["closing_balance"] == pytest.approx(plan["closing_balance"], abs=1e-6)


def test_ruin_is_absorbing(archive_conn, cfgs):
    """A wiped-out arm stops trading and records the date it died. Ruin is not
    a drawdown you trade back from — at $0 there is no next bet."""
    out = paper_account.sizing_arms(archive_conn, cfgs, 0.60)
    for a in out["arms"]:
        if a["ruin_date"] is None:
            continue
        after = [r for r in a["rows"] if r["target_date"] > a["ruin_date"]]
        assert all(not r["trade"] for r in after), a["arm"]


def test_arms_never_mutate_staking_config(archive_conn, cfgs):
    """The experiment lives in the sim. If it ever edits the shipped staking
    config, a live order could be sized off an experimental fraction."""
    before = (KMIA.staking.max_stake_frac, KMIA.staking.kelly_fraction,
              KLAX.staking.max_stake_frac, KLAX.staking.kelly_fraction)
    paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    after = (KMIA.staking.max_stake_frac, KMIA.staking.kelly_fraction,
             KLAX.staking.max_stake_frac, KLAX.staking.kelly_fraction)
    assert before == after
    assert cfgs["KMIA"].staking.max_stake_frac == KMIA.staking.max_stake_frac


def test_report_carries_the_four_asked_for_numbers(archive_conn, cfgs):
    out = paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    for a in out["arms"]:
        for k in ("closing_balance", "net_per_contract", "max_stake_frac_seen",
                  "ruin_date"):
            assert k in a, (a["arm"], k)


def test_an_open_position_is_not_counted_as_a_loss(archive_conn, cfgs):
    """Cash leaves the balance at OPEN and only comes back at SETTLE, so an arm
    holding a live bet looks poorer than it is. Comparing arms on cash alone
    would have reported arm A down $5.10 on a day it was 3-for-3 — the fair
    figure is equity at cost."""
    out = paper_account.sizing_arms(archive_conn, cfgs, 5.50)
    for a in out["arms"]:
        assert a["equity_at_cost"] >= a["closing_balance"] - 1e-9, a["arm"]
        if a["open_contracts"]:
            assert a["equity_at_cost"] > a["closing_balance"], a["arm"]
            assert a["net_settled"] > a["net"], a["arm"]
        else:
            assert a["equity_at_cost"] == pytest.approx(a["closing_balance"])
