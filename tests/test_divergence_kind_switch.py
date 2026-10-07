"""A divergence alert must never subtract one model's probability from another's.

THE BUG. `app_bets.p_model` can be a DAY-AHEAD probability while every later
divergence check reads the INTRADAY nowcast, and `divergence_reason` was handed
both — so its two probability triggers measured the model swap rather than a
change of mind.

⚠️ THE CAUSE IS AVAILABILITY, NOT THE HOUR. My first trace said "a HIGH
committed before MIN_DECISION_HOUR is priced day-ahead" and was WRONG (error
#16). `preferred_prediction` walks an order and returns the first kind that HAS
a stored prediction, so a bet committed before the nowcast has run for that
target gets a day-ahead number whatever its metric or hour. Two facts killed the
hour story:

  * `MIN_DECISION_HOUR` moved 11 -> 13 on 2026-07-31, and every HIGH commit
    before h13 in the record predates that move — so at the time they were AT
    the knee and were served intraday. No cross-kind comparison occurred.
  * the ONE settled bet that has ever spanned a switch is a LOW (KLAX
    2026-08-09, committed h19 at 45% off the day-ahead because no intraday
    prediction for that target existed until 05:13Z) — and the hour story says
    LOWs are immune.

⚠️ NOTHING HAS EVER BEEN MIS-ALERTED BY THIS. On that one real switch the alert
fired "overtaken", which compares top against held INSIDE ONE LADDER and was
never cross-model; the bet lost and the alert was right. These tests guard a
LATENT bug. Do not let the file's existence imply damage was observed.

THE FIX keeps reading the currently preferred model — after the knee it is the
measured-sharper one — but re-baselines to THAT model's own first reading on the
committed bucket, and reads the committing kind from the `signals` record rather
than reconstructing it from the hour.
"""
import json

import pytest

from highcast import gating
from highcast.db import connect
from highcast.notify import (DIVERGENCE_FLOOR, build_divergence,
                             divergence_baseline, divergence_reason)


class _Cfg:
    class station:
        icao = "KMIA"
        tz = "America/New_York"
    discord_webhook_url = "https://example.invalid/hook"


# Shaped after the one real switch (KLAX low 2026-08-09) but kept synthetic:
# a LOW, committed outside its window, priced off the day-ahead because no
# nowcast existed yet.
BET = {"metric": "low", "target_date": "2026-07-29", "bucket": "66–67",
       "p_model": 0.64, "placed_local_hour": 19,
       "placed_utc": "2026-07-29T15:02:00+00:00"}


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _signal(c, kind, created, rungs, target="2026-07-29", metric="low"):
    c.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
                                committed, within_window, place, market_json)
           VALUES (?,?,?,?,?,0,1,0,?)""",
        ("KMIA", target, metric, kind, created,
         json.dumps([{"label": k, "p_model": v} for k, v in rungs.items()])))
    c.commit()


# ── the rule itself, in one place ────────────────────────────────────────────

def test_kind_at_hour_is_the_single_source_of_the_model_choice():
    knee = gating.MIN_DECISION_HOUR["KMIA"]["high"]
    assert gating.kind_at_hour("KMIA", "high", True, knee - 1) == "dayahead"
    assert gating.kind_at_hour("KMIA", "high", True, knee) == "intraday"
    # outside the window the nowcast serves regardless of hour
    assert gating.kind_at_hour("KMIA", "high", False, knee - 1) == "intraday"
    # LOWS ARE NEVER DAY-AHEAD -- this is why they were immune to the bug
    for h in range(0, 24):
        assert gating.kind_at_hour("KMIA", "low", True, h) == "intraday"


def test_preferred_prediction_still_agrees_with_the_extracted_rule(conn):
    """The refactor must not have changed which model the desk serves."""
    conn.execute(
        """INSERT INTO predictions (station, target_date, issued_at_utc, kind,
             point_f, quantiles, buckets, metric)
           VALUES (?,?,?,?,?,?,?,?)""",
        ("KMIA", "2026-07-29", "2026-07-29T14:00:00+00:00", "dayahead",
         94.0, "{}", '{"94": 1.0}', "high"))
    conn.execute(
        """INSERT INTO predictions (station, target_date, issued_at_utc, kind,
             point_f, quantiles, buckets, metric)
           VALUES (?,?,?,?,?,?,?,?)""",
        ("KMIA", "2026-07-29", "2026-07-29T14:00:00+00:00", "intraday",
         94.0, "{}", '{"94": 1.0}', "high"))
    conn.commit()
    for hour, want in ((11, "dayahead"), (13, "intraday")):
        _, kind = gating.preferred_prediction(
            conn, "KMIA", "2026-07-29", "high", True, local_hour=hour)
        assert kind == want


# ── which model committed: GROUND TRUTH, not the hour ────────────────────────

COMMIT_ROW = "2026-07-29T15:00:00+00:00"    # just before BET["placed_utc"]


def test_committing_kind_comes_from_the_record_not_the_hour(conn):
    """⚠️ THE CORRECTION. The hour rule says a LOW is always intraday, so a
    reconstruction would see no switch here. The record says day-ahead — which
    is what actually happened on KLAX low 2026-08-09."""
    assert gating.kind_at_hour("KMIA", "low", True, 19) == "intraday"
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.55})
    base, switched = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert switched is True, "the record must win over the hour rule"
    assert base == pytest.approx(0.55)


def test_shadow_rows_never_count_as_the_committing_decision(conn):
    """A `shadow` row is written alongside the real one and would otherwise win
    the nearest-at-or-before race on an identical timestamp."""
    _signal(conn, "intraday", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "shadow", COMMIT_ROW, {"66–67": 0.11})
    assert divergence_baseline(conn, _Cfg(), BET, "intraday") == (0.64, False)


def test_falls_back_to_the_hour_rule_when_nothing_was_logged(conn):
    """Rows predating the signals log still have to resolve to something."""
    assert divergence_baseline(conn, _Cfg(), BET, "intraday") == (0.64, False)
    high = dict(BET, metric="high", placed_local_hour=11)
    base, switched = divergence_baseline(conn, _Cfg(), high, "intraday")
    assert switched is True and base is None


# ── the baseline ─────────────────────────────────────────────────────────────

def test_no_switch_leaves_the_committed_probability_untouched(conn):
    _signal(conn, "intraday", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.20})
    assert divergence_baseline(conn, _Cfg(), BET, "intraday") == (0.64, False)


def test_switch_rebaselines_to_the_new_models_first_reading(conn):
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.55})
    _signal(conn, "intraday", "2026-07-29T19:02:00+00:00", {"66–67": 0.31})
    base, switched = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert switched is True
    assert base == pytest.approx(0.55)   # the FIRST intraday reading, not 0.64


def test_rebaseline_ignores_readings_from_before_the_commit(conn):
    """A baseline taken before the bet existed would import a stale number."""
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "intraday", "2026-07-29T09:00:00+00:00", {"66–67": 0.05})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.55})
    base, _ = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert base == pytest.approx(0.55)


def test_no_stored_reading_yet_suppresses_rather_than_falling_back(conn):
    """⚠️ Must NOT fall back to the committed number — that is the bug."""
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    base, switched = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert (base, switched) == (None, True)
    # and a None baseline silences both probability triggers...
    assert divergence_reason(None, 0.30, "66–67", "66–67") is None
    # ...while leaving "overtaken" alive, since it never was cross-model.
    assert divergence_reason(None, 0.30, "68–69", "66–67") == "overtaken"


def test_malformed_market_json_does_not_raise(conn):
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
                                committed, within_window, place, market_json)
           VALUES (?,?,?,?,?,0,1,0,?)""",
        ("KMIA", "2026-07-29", "low", "intraday",
         "2026-07-29T17:02:00+00:00", "{not json"))
    conn.commit()
    assert divergence_baseline(conn, _Cfg(), BET, "intraday") == (None, True)


# ── the trigger, across a switch ─────────────────────────────────────────────

def test_a_cross_model_floor_trip_no_longer_fires(conn):
    """Committed 64% off the day-ahead; the nowcast's FIRST reading is 55%.
    The old code compared 64% against 55% and tripped the floor (64% >= 60%,
    55% < 60%) on a number the committing model never produced."""
    assert BET["p_model"] >= DIVERGENCE_FLOOR and 0.55 < DIVERGENCE_FLOOR
    assert divergence_reason(BET["p_model"], 0.55, "66–67", "66–67") \
        == "collapsed", "precondition: the old comparison DID fire"

    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.64})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.55})
    base, switched = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert switched is True
    assert divergence_reason(base, 0.55, "66–67", "66–67") is None


def test_overtaken_still_fires_across_a_switch(conn):
    """⚠️ THE ONE REAL CASE (KLAX low 2026-08-09) fired `overtaken`, and it was
    RIGHT — the bet lost. `overtaken` compares top against held inside a single
    ladder, so the fix must not touch it."""
    _signal(conn, "dayahead", COMMIT_ROW, {"66–67": 0.45})
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00",
            {"66–67": 0.02, "68–69": 0.80})
    base, switched = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert switched is True
    assert divergence_reason(base, 0.02, "68–69", "66–67") == "overtaken"


def test_a_real_collapse_within_the_new_model_still_fires(conn):
    """The fix must not have bought silence by disabling the trigger."""
    _signal(conn, "intraday", "2026-07-29T17:02:00+00:00", {"66–67": 0.72})
    base, _ = divergence_baseline(conn, _Cfg(), BET, "intraday")
    assert divergence_reason(base, 0.41, "66–67", "66–67") == "collapsed"


# ── the message ──────────────────────────────────────────────────────────────

def test_message_quotes_the_baseline_and_discloses_the_swap():
    msg = build_divergence(_Cfg(), BET, "66–67", 0.41, reason="collapsed",
                           held_p=0.41, baseline_p=0.72,
                           baseline_kind="intraday")
    fields = {f["name"]: f["value"] for f in msg["embeds"][0]["fields"]}
    # the drop is measured from the baseline, NOT from the committed 64%
    assert "72% → 41%" in fields["Confidence on your bucket"]
    assert "31%" in fields["Confidence on your bucket"]
    swap = fields["Which model is speaking"]
    assert "64%" in swap and "nowcast" in swap and "72%" in swap


def test_unswitched_message_is_unchanged_and_has_no_swap_field():
    msg = build_divergence(_Cfg(), BET, "66–67", 0.30, reason="collapsed",
                           held_p=0.30)
    fields = {f["name"]: f["value"] for f in msg["embeds"][0]["fields"]}
    assert "Which model is speaking" not in fields
    assert "64% → 30%" in fields["Confidence on your bucket"]
