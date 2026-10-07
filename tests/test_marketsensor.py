"""The market-as-sensor module.

⚠️ THE MOST IMPORTANT TEST IN THIS FILE IS THE LAST ONE. This module reads the
counterparty's opinion, and the standing danger is that someone wires it into
the model and calls the market's own view an edge. QUEUE item 1: the market
already beats our model at 78 of 81 hours.
"""
import dataclasses
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from highcast import db as hdb
from highcast import marketsensor
from highcast.config import load_config
from highcast.metric import get_metric

CFG = load_config()
HIGH = get_metric(CFG, "high")
LOW = get_metric(CFG, "low")
NOW = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)


def _ladder(*rows):
    """(label, lo, hi, ask) tuples -> ladder dicts."""
    return [{"label": l, "lo": lo, "hi": hi, "yes_ask": a} for l, lo, hi, a in rows]


# ── implied(): has the market decided? ──────────────────────────────────────

def test_undecided_market_returns_none():
    assert marketsensor.implied(_ladder(
        ("90-91", 90, 91, 0.45), ("92-93", 92, 93, 0.55))) is None


def test_decided_market_returns_the_rung():
    imp = marketsensor.implied(_ladder(
        ("90-91", 90, 91, 0.02), ("92-93", 92, 93, 0.99)))
    assert imp["label"] == "92-93" and imp["prob"] > 0.9


def test_asks_are_normalised_before_thresholding():
    """⚠️ A Kalshi ladder's asks sum to ~1.09 — the house spread. Comparing raw
    asks to 0.90 would call the market decided ~9% too eagerly. Here the raw ask
    is 0.90 but the normalised probability is below it, so this must NOT count."""
    lad = _ladder(("90-91", 90, 91, 0.90), ("92-93", 92, 93, 0.19))
    assert lad[0]["yes_ask"] >= marketsensor.CONF          # raw would pass
    assert marketsensor.implied(lad) is None               # normalised does not


def test_a_missing_ask_makes_the_ladder_unusable():
    assert marketsensor.implied(_ladder(
        ("90-91", 90, 91, None), ("92-93", 92, 93, 0.99))) is None


# ── divergence(): are we blind right now? ───────────────────────────────────

def _obs(conn, valid_utc, tmpf, six_max_c=None):
    hdb.upsert(conn, "obs", [{
        "station": CFG.station.icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": tmpf, "tmpf_precise": tmpf, "dwpf": 50.0, "sknt": 5.0,
        "drct": 260.0, "skyc": "CLR", "six_max_c": six_max_c, "six_min_c": None,
        "raw": "x", "fetched_at": valid_utc}])


def test_divergence_fires_when_the_market_is_above_what_we_can_see(tmp_path):
    """THE 2026-08-14 CASE. The market priced 95-96 while our hourly snapshots
    had never exceeded 93.9 — the peak happened between filings."""
    conn = hdb.connect(tmp_path / "t.db")
    day = date(2026, 8, 14)
    _obs(conn, "2026-08-14T17:53:00+00:00", 93.92)
    lad = _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99))
    d = marketsensor.divergence(conn, CFG, HIGH, day, lad,
                                "2026-08-14T19:00:00+00:00")
    assert d is not None
    assert d["seen"] == 94 and d["gap_f"] == 1
    assert d["label"] == "95-96"


def test_no_divergence_when_our_ratchet_already_agrees(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    day = date(2026, 8, 14)
    _obs(conn, "2026-08-14T17:53:00+00:00", 95.4)
    lad = _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99))
    assert marketsensor.divergence(conn, CFG, HIGH, day, lad,
                                   "2026-08-14T19:00:00+00:00") is None


def test_no_divergence_from_an_undecided_market(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _obs(conn, "2026-08-14T17:53:00+00:00", 90.0)
    lad = _ladder(("93-94", 93, 94, 0.45), ("95-96", 95, 96, 0.55))
    assert marketsensor.divergence(conn, CFG, HIGH, date(2026, 8, 14), lad,
                                   "2026-08-14T19:00:00+00:00") is None


def test_an_open_bucket_has_no_edge_to_compare(tmp_path):
    """`≥96` has no lower bound to measure a gap against; returning None beats
    inventing one."""
    conn = hdb.connect(tmp_path / "t.db")
    _obs(conn, "2026-08-14T17:53:00+00:00", 90.0)
    lad = _ladder(("93-94", 93, 94, 0.01), ("open", None, None, 0.99))
    assert marketsensor.divergence(conn, CFG, HIGH, date(2026, 8, 14), lad,
                                   "2026-08-14T19:00:00+00:00") is None


def test_divergence_direction_is_mirrored_for_a_LOW(tmp_path):
    """A minimum only falls, so blindness means the market is BELOW what we see.
    Using the max logic here would silently never fire on the low."""
    conn = hdb.connect(tmp_path / "t.db")
    day = date(2026, 8, 14)
    _obs(conn, "2026-08-14T13:53:00+00:00", 70.0)
    lad = _ladder(("67-68", 67, 68, 0.99), ("69-70", 69, 70, 0.01))
    d = marketsensor.divergence(conn, CFG, LOW, day, lad,
                                "2026-08-14T15:00:00+00:00")
    assert d is not None and d["seen"] == 70 and d["gap_f"] == 2


# ── grade_conflicts(): the false-WIN guard ──────────────────────────────────

def _bet(conn, target_date, metric, bucket, actual, won):
    hdb.upsert(conn, "app_bets", [{
        "station": CFG.station.icao, "metric": metric,
        "target_date": target_date, "placed_utc": f"{target_date}T20:00:00+00:00",
        "placed_manila": "04:00", "placed_local_hour": 13, "bucket": bucket,
        "p_model": 0.8, "price": 0.5, "stake": 1.0, "payout_if_win": 2.0,
        "actual": actual, "won": won, "reason": "test"}])


def _signal(conn, target_date, metric, ladder):
    hdb.upsert(conn, "signals", [{
        "station": CFG.station.icao, "metric": metric,
        "target_date": target_date, "kind": "intraday",
        "created_utc": f"{target_date}T23:00:00+00:00", "committed": 1,
        "within_window": 1, "place": 0, "market_json": json.dumps(ladder)}])


def test_no_conflict_when_the_market_agrees_with_the_grade(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _bet(conn, "2026-08-14", "high", "95-96", 96, 1)
    _signal(conn, "2026-08-14", "high",
            _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99)))
    assert marketsensor.grade_conflicts(conn, CFG) == []


def test_a_false_WIN_is_caught(tmp_path):
    """⚠️ THE FAILURE THIS EXISTS FOR. MEMORY.md: a preliminary CLI mis-graded a
    bet TWICE. On 2026-07-30 KLAX's report said "low 70" hours before the climate
    day closed; the day bottomed at 67, the ">=69" bet was booked a WIN, and an
    audit confirmed nothing because the audit and the grade read the SAME
    corrupted row. The market is an independent witness."""
    conn = hdb.connect(tmp_path / "t.db")
    _bet(conn, "2026-08-14", "high", "93-94", 94, 1)      # booked as a WIN
    _signal(conn, "2026-08-14", "high",                    # market says 96
            _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99)))
    bad = marketsensor.grade_conflicts(conn, CFG)
    assert len(bad) == 1
    assert bad[0]["recorded"] == 94 and bad[0]["market_says"] == "95-96"
    assert bad[0]["won"] is True, "a false WIN must be flagged as such"


def test_ungraded_bets_are_not_conflicts(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _bet(conn, "2026-08-14", "high", "93-94", None, None)
    _signal(conn, "2026-08-14", "high",
            _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99)))
    assert marketsensor.grade_conflicts(conn, CFG) == []


def test_an_undecided_market_never_disputes_a_grade(tmp_path):
    """The market only gets a vote once it is confident. Otherwise a coin-flip
    ladder would contradict every grade it happened to disagree with."""
    conn = hdb.connect(tmp_path / "t.db")
    _bet(conn, "2026-08-14", "high", "93-94", 94, 1)
    _signal(conn, "2026-08-14", "high",
            _ladder(("93-94", 93, 94, 0.45), ("95-96", 95, 96, 0.55)))
    assert marketsensor.grade_conflicts(conn, CFG) == []


# ── the wake line stays quiet unless there is news ──────────────────────────

def test_wake_line_is_silent_on_a_normal_day(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    assert marketsensor.wake_line(conn, CFG, NOW) is None


def test_wake_line_reports_a_grade_conflict_loudly(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _bet(conn, "2026-08-14", "high", "93-94", 94, 1)
    _signal(conn, "2026-08-14", "high",
            _ladder(("93-94", 93, 94, 0.01), ("95-96", 95, 96, 0.99)))
    line = marketsensor.wake_line(conn, CFG, NOW)
    assert line and "GRADE CONFLICT" in line and "booked as a WIN" in line


# ── the guard that matters ──────────────────────────────────────────────────

def test_the_sensor_is_never_wired_into_the_model():
    """⚠️ STANDING RULE. This module reads the COUNTERPARTY'S opinion. Feeding it
    to the model would launder the market's own view into ours and call it an
    edge — and the market already beats our model at 78 of 81 hours, so the
    'improvement' would be circular by construction.

    If this test fails, someone has imported the sensor into the prediction
    path. That is not a refactor; it is a change of what the model is."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "highcast"
    forbidden = ["models/intraday.py", "models/baseline.py", "features.py",
                 "staking.py", "signals.py", "gating.py"]
    for rel in forbidden:
        text = (src / rel).read_text()
        assert "marketsensor" not in text, (
            f"{rel} imports marketsensor — the market's opinion must not reach "
            f"the model or the staking rule. See QUEUE item 1.")
