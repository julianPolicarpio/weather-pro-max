"""The day grade — input quality, NOT expected accuracy.

⚠️ THE TWO MOST IMPORTANT TESTS ARE AT THE BOTTOM. This module is the fifth
artefact of a shape that has failed four times, and it only survives because it
makes NO predictive claim. The tests that matter are the ones stopping it from
acquiring one.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from highcast import daygrade
from highcast import db as hdb
from highcast.config import load_config

CFG = load_config()
DAY = date(2026, 8, 14)
NOW = datetime(2026, 8, 14, 23, 0, tzinfo=timezone.utc)   # 16:00 PDT, past h13


def _clean(tmp_path, now=NOW, day=DAY):
    """A database where every input check passes."""
    conn = hdb.connect(tmp_path / "t.db")
    icao, ts, d = CFG.station.icao, now.isoformat(), day.isoformat()
    hdb.upsert(conn, "obs", [{"station": icao, "valid_utc": ts, "source": "metar",
                              "tmpf": 70.0, "fetched_at": ts}])
    for t, extra in (("forecast_hourly", {"valid_utc": ts, "model": "m", "temp_f": 70.0}),
                     ("forecast_intraday", {"target_date": d, "model": "m",
                                            "metric": "high", "run_date": d}),
                     ("forecasts", {"target_date": d, "model": "m", "run_date": d,
                                    "metric": "high", "max_temp_f": 70.0}),
                     ("forecast_nbs", {"target_date": d, "run_date": d, "metric": "high"}),
                     ("forecast_conv", {"valid_utc": ts, "model": "m"}),
                     ("forecast_ens", {"target_date": d, "run_date": d, "metric": "high"}),
                     ("forecast_aloft", {"target_date": d, "run_date": d, "metric": "high"})):
        hdb.upsert(conn, t, [{"station": icao, "fetched_at": ts, **extra}])
    hdb.upsert(conn, "cli_reports", [{
        "station": icao, "date": (day - timedelta(days=1)).isoformat(),
        "high_f": 80, "low_f": 60, "raw": "{}", "fetched_at": ts}])
    # The six-hour remark group the coverage check is ENTITLED to by h13. KLAX's
    # climate day is 08:00Z-08:00Z, so the 12Z-18Z window (filed 17:53Z) is the
    # only one wholly inside the day and already due at the commit hour.
    hdb.upsert(conn, "obs", [{
        "station": icao, "valid_utc": f"{d}T17:53:00+00:00", "source": "metar",
        "tmpf": 79.0, "tmpf_precise": 79.0, "six_max_c": 26.1, "six_min_c": 15.0,
        "fetched_at": ts}])
    # A market snapshot so the blindspot check can actually run. An UNDECIDED
    # ladder is the quiet case: `divergence` returns None and the check is ok.
    hdb.upsert(conn, "signals", [{
        "station": icao, "metric": "high", "target_date": d, "kind": "intraday",
        "created_utc": ts, "committed": 1, "within_window": 1, "place": 0,
        "market_json": json.dumps([
            {"label": "79-80", "lo": 79, "hi": 80, "yes_ask": 0.5},
            {"label": "81-82", "lo": 81, "hi": 82, "yes_ask": 0.55}])}])
    hdb.upsert(conn, "signals", [{
        "station": icao, "metric": "low", "target_date": d, "kind": "intraday",
        "created_utc": ts, "committed": 1, "within_window": 1, "place": 0,
        "market_json": json.dumps([
            {"label": "59-60", "lo": 59, "hi": 60, "yes_ask": 0.5},
            {"label": "61-62", "lo": 61, "hi": 62, "yes_ask": 0.55}])}])
    for m, p in (("high", 0.75), ("low", 0.75)):
        hdb.upsert(conn, "app_bets", [{
            "station": icao, "metric": m, "target_date": d,
            "placed_utc": ts, "placed_manila": "04:00", "placed_local_hour": 13,
            "bucket": "79-80", "p_model": p, "price": 0.5, "stake": 1.0,
            "payout_if_win": 2.0, "reason": "test"}])
    return conn


def test_a_clean_day_grades_A(tmp_path):
    conn = _clean(tmp_path)
    g = daygrade.grade(conn, CFG, "high", DAY, NOW)
    assert g.letter == "A", g.summary
    assert all(c.level == "ok" for c in g.checks), g.summary


def test_a_dead_feed_is_an_F(tmp_path):
    """If the inputs are stale the number should not be read at all."""
    conn = _clean(tmp_path)
    conn.execute("DELETE FROM forecast_conv")
    g = daygrade.grade(conn, CFG, "high", DAY, NOW)
    assert g.letter == "F"
    assert any(c.name == "feeds" and c.level == "fail" for c in g.checks)


def test_a_missing_commit_after_the_hour_is_an_F(tmp_path):
    conn = _clean(tmp_path)
    conn.execute("DELETE FROM app_bets")
    g = daygrade.grade(conn, CFG, "high", DAY, NOW)
    assert g.letter == "F"
    assert any(c.name == "commit" and c.level == "fail" for c in g.checks)


def test_before_the_commit_hour_a_missing_bet_is_only_INFO(tmp_path):
    """Not-yet-due is not a fault. Grading it F would make every morning look
    broken and train the reader to ignore the grade."""
    early = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)   # 05:00 PDT
    conn = _clean(tmp_path, now=early)
    conn.execute("DELETE FROM app_bets")
    g = daygrade.grade(conn, CFG, "high", DAY, early)
    c = [x for x in g.checks if x.name == "commit"][0]
    assert c.level == "info" and "not due yet" in c.detail
    assert g.letter in ("A", "B")


def test_an_uncalibrated_probability_is_a_C(tmp_path):
    """QUEUE items 11/12 and the live 2026-08-10 refusal: `BAND_CALIB` for KLAX
    LOW tops out at 0.80 because only 13 recent records exceeded it, so a 97%
    there sits outside every measured band and the desk refuses it.

    ⚠️ Uses LOW deliberately. KLAX HIGH h13's band table runs to 1.01, so 0.97
    IS calibrated there — the first version of this test asserted C on `high`
    and failed, which is the table telling the truth, not a bug."""
    conn = _clean(tmp_path)
    conn.execute("UPDATE app_bets SET p_model=0.97 WHERE metric='low'")
    g = daygrade.grade(conn, CFG, "low", DAY, NOW)
    assert g.letter == "C"
    assert any(c.name == "calibrated" and c.level == "warn" for c in g.checks)


def test_an_active_advisory_is_INFO_not_a_downgrade_to_C(tmp_path):
    """⚠️ An advisory is an EXTERNAL FACT, not a broken input. Treating it as a
    fault would be the guidance-spread veto again — that one vetoed days which
    then hit BETTER than the days it kept."""
    conn = _clean(tmp_path)
    hdb.upsert(conn, "alerts", [{
        "station": CFG.station.icao, "event": "Heat Advisory",
        "onset_utc": (NOW - timedelta(hours=1)).isoformat(),
        "expires_utc": (NOW + timedelta(hours=4)).isoformat(),
        "fetched_at": NOW.isoformat()}])
    g = daygrade.grade(conn, CFG, "high", DAY, NOW)
    c = [x for x in g.checks if x.name == "advisory"][0]
    assert c.level == "info" and "Heat Advisory" in c.detail
    assert g.letter == "B", "an advisory must not make the inputs look degraded"


def test_the_advisory_check_actually_runs(tmp_path):
    """⚠️ THE BUG THIS PINS. The first version queried `effective_utc`, which
    does not exist — the column is `onset_utc` — and a bare `except: ok`
    swallowed the error, so this check reported CLEAN on every day forever.
    Same failure class as the CLI staleness check that could never fire.

    An alert that is active RIGHT NOW must be seen; if this ever returns ok
    with a live alert in the table, the query has silently broken again."""
    conn = _clean(tmp_path)
    hdb.upsert(conn, "alerts", [{
        "station": CFG.station.icao, "event": "Excessive Heat Warning",
        "onset_utc": (NOW - timedelta(hours=2)).isoformat(),
        "expires_utc": (NOW + timedelta(hours=2)).isoformat(),
        "fetched_at": NOW.isoformat()}])
    c = [x for x in daygrade.grade(conn, CFG, "high", DAY, NOW).checks
         if x.name == "advisory"][0]
    assert c.level == "info", "a live alert must not report as clean"


def test_an_expired_advisory_does_not_count(tmp_path):
    conn = _clean(tmp_path)
    hdb.upsert(conn, "alerts", [{
        "station": CFG.station.icao, "event": "Heat Advisory",
        "onset_utc": (NOW - timedelta(hours=9)).isoformat(),
        "expires_utc": (NOW - timedelta(hours=1)).isoformat(),
        "fetched_at": NOW.isoformat()}])
    c = [x for x in daygrade.grade(conn, CFG, "high", DAY, NOW).checks
         if x.name == "advisory"][0]
    assert c.level == "ok"


def test_grade_never_raises_on_a_bare_database(tmp_path):
    """It runs inside the wake cycle. A monitor that throws takes down the thing
    it was meant to watch."""
    conn = hdb.connect(tmp_path / "t.db")
    for g in daygrade.grade_all(conn, CFG, DAY, NOW):
        assert g.letter in ("A", "B", "C", "F")


# ── the guards that keep this from becoming a trading signal ────────────────

def test_the_grade_is_never_read_by_the_model_or_the_desk():
    """⚠️ STANDING RULE. The grade describes INPUT QUALITY. It was measured NOT
    to predict accuracy — `scripts/day_grade_gate.py` found every candidate
    factor flips sign between markets. Wiring it into staking would be the
    confidence gate a second time: that one filtered to 'good' days, the ask
    climbed 31c -> 60c to meet the filter, and it inverted P&L.

    `stated` is the accuracy grade. This is the plumbing grade."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "highcast"
    for rel in ("staking.py", "signals.py", "gating.py", "features.py",
                "models/intraday.py", "appbet.py"):
        assert "daygrade" not in (src / rel).read_text(), (
            f"{rel} imports daygrade — the grade must never gate a trade. "
            f"See the module docstring and QUEUE item 1.")


def test_the_disclaimer_travels_with_the_payload(tmp_path):
    """Anything rendering this must carry the caveat. A letter grade on a
    dashboard invites 'A means it will hit', which is exactly wrong."""
    conn = _clean(tmp_path)
    d = daygrade.grade(conn, CFG, "high", DAY, NOW).as_dict()
    assert "does NOT predict accuracy" in d["disclaimer"]
    assert d["letter"] == "A" and d["meaning"]


def test_six_hour_groups_not_due_yet_is_INFO_not_WARN(tmp_path):
    """⚠️ THE BUG JULIAN FOUND. The check evaluated as of the COMMIT HOUR even
    when the clock was hours short of it, so a group whose filing time had not
    ARRIVED was reported as "never filed". On 2026-08-15 at 09:11Z it claimed
    KMIA high was missing the 12:00Z group — two hours in the future.

    That reads as a fault when it is a forecast of one, and it contradicted the
    `commit` check beside it, which correctly says "not due yet". Same family as
    the guards that could not fire: a check whose verdict does not match reality
    trains the reader to ignore it."""
    early = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)   # 05:00 PDT
    conn = _clean(tmp_path, now=early)
    conn.execute("DELETE FROM obs WHERE six_max_c IS NOT NULL")   # no groups yet
    c = [x for x in daygrade.grade(conn, CFG, "high", DAY, early).checks
         if x.name == "six_hour_groups"][0]
    assert c.level != "warn", (
        f"a group that is not due yet must not be a fault (got {c.level}: "
        f"{c.detail})")


def test_six_hour_groups_genuinely_overdue_still_WARNs(tmp_path):
    """The other half: once the filing time HAS passed and nothing arrived, that
    is a real 6-hour blind window and must still be flagged. A fix that silenced
    both cases would be worse than the bug."""
    conn = _clean(tmp_path)                    # NOW = 23:00Z, past h13
    conn.execute("DELETE FROM obs WHERE six_max_c IS NOT NULL")
    c = [x for x in daygrade.grade(conn, CFG, "high", DAY, NOW).checks
         if x.name == "six_hour_groups"][0]
    assert c.level == "warn" and "overdue" in c.detail


def test_after_the_commit_hour_the_verdict_freezes(tmp_path):
    """The grade describes the information the DECISION was made with. Groups
    arriving after the commit hour must not retroactively clean up the grade,
    and later windows must not retroactively dirty it."""
    conn = _clean(tmp_path)
    at_commit = daygrade.grade(conn, CFG, "high", DAY, NOW)
    much_later = daygrade.grade(conn, CFG, "high", DAY,
                                NOW + timedelta(hours=6))
    a = [x for x in at_commit.checks if x.name == "six_hour_groups"][0]
    b = [x for x in much_later.checks if x.name == "six_hour_groups"][0]
    assert (a.level, a.detail) == (b.level, b.detail)
