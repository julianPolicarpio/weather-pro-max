"""A COMMIT HOUR PASSED AND NO BET WAS RECORDED — say so.

WHAT HAPPENED, 2026-08-08/09. A power outage took the laptop down for 11.6h
then 9.6h. Four commits were lost — KLAX high on 08-08, and KLAX high, KMIA
high and KMIA low on 08-09 — and NOTHING said so. No alert, no log line, no
dashboard flag. Julian found out by asking two days later.

The existing staleness thinking (QUEUE item 4) is about FEEDS dying. This is a
different failure: every feed can be perfectly healthy while the process simply
was not running at 17:00Z, and the only trace is a row that does not exist.
Absence has no timestamp, so nothing downstream can notice it.

⚠️ THE TARGET DATE IS STATION-LOCAL, AND THIS IS EASY TO GET WRONG. I got it
wrong while diagnosing the outage: at 05:10Z, KMIA's local date is 2026-08-10
but KLAX's is still 2026-08-09, because LA is seven hours behind UTC. Comparing
a station's clock HOUR against its commit hour without asking which DAY that
hour belongs to reports phantom misses for a day that has not started. The
first two tests below pin exactly that.

A REFUSAL COUNTS AS COMMITTED. `app_bets` records the decision, not the trade —
"no edge" and "past the tradeable window" are both rows. The thing being
detected is the wake cycle never having run, not the desk declining.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from highcast import appbet, db as hdb
from highcast.config import load_config

KLAX = load_config("config/klax.yaml")
KMIA = load_config("config/kmia.yaml")
CFGS = {"KLAX": KLAX, "KMIA": KMIA}


@pytest.fixture
def conn(tmp_path):
    return hdb.connect(tmp_path / "t.db")


def _bet(conn, station, metric, target, hour=13, reason="no edge: do not place"):
    conn.execute(
        """INSERT INTO app_bets (station, metric, target_date, placed_utc,
                                 placed_manila, placed_local_hour, bucket,
                                 p_model, price, stake, payout_if_win, reason)
           VALUES (?,?,?,?, '00:00', ?, '80–81', 0.7, 0.5, 0, 0, ?)""",
        (station, metric, target, f"{target}T12:00:00+00:00", hour, reason))
    conn.commit()


def _at(utc: str) -> dt.datetime:
    return dt.datetime.fromisoformat(utc).replace(tzinfo=dt.timezone.utc)


# ── the target-date trap ─────────────────────────────────────────────────────

def test_uses_each_stations_own_local_date(conn):
    """At 05:10Z on 2026-08-10, KMIA is on 08-10 (01:10 EDT) while KLAX is
    still on 08-09 (22:10 PDT). A check that used the UTC date would ask KLAX
    about a day it has not begun."""
    now = _at("2026-08-10T05:10:00")
    got = {(m["station"], m["target_date"]) for m in
           appbet.missing_commits(conn, CFGS, now, lookback_days=0)}
    for station, target in got:
        assert target == ("2026-08-10" if station == "KMIA" else "2026-08-09"), got


def test_does_not_fire_before_the_commit_hour(conn):
    """KMIA at 01:10 local has neither commit hour (h7, h13) behind it. Firing
    here would shout on nearly every wake and train the reader to ignore it."""
    now = _at("2026-08-10T05:10:00")
    kmia = [m for m in appbet.missing_commits(conn, CFGS, now, lookback_days=0)
            if m["station"] == "KMIA"]
    assert kmia == [], kmia


def test_fires_once_the_hour_has_passed_with_no_row(conn):
    now = _at("2026-08-10T05:10:00")          # KLAX local 08-09 22:10, h13 long past
    klax = [m for m in appbet.missing_commits(conn, CFGS, now, lookback_days=0)
            if m["station"] == "KLAX" and m["metric"] == "high"]
    assert len(klax) == 1, klax
    assert klax[0]["target_date"] == "2026-08-09"
    assert klax[0]["commit_hour"] == 13


# ── what counts as committed ─────────────────────────────────────────────────

def test_a_refusal_counts_as_committed(conn):
    """`app_bets` records the DECISION. "no edge" is the desk working, not the
    desk missing — only an absent row means the cycle never ran."""
    _bet(conn, "KLAX", "high", "2026-08-09", reason="no edge: 80–81 ... do not place")
    now = _at("2026-08-10T05:10:00")
    klax = [m for m in appbet.missing_commits(conn, CFGS, now, lookback_days=0)
            if m["station"] == "KLAX" and m["metric"] == "high"]
    assert klax == [], klax


def test_a_late_commit_counts_too(conn):
    """2026-08-09 KLAX low was written at h19 instead of h6 with "past the
    tradeable window". Degraded, but the cycle DID run — that is a different
    problem and must not be reported as a missing commit."""
    _bet(conn, "KLAX", "low", "2026-08-09", hour=19,
         reason="past the tradeable window — too late to place")
    now = _at("2026-08-10T05:10:00")
    low = [m for m in appbet.missing_commits(conn, CFGS, now, lookback_days=0)
           if m["station"] == "KLAX" and m["metric"] == "low"]
    assert low == [], low


# ── actionability ────────────────────────────────────────────────────────────

def test_it_says_whether_the_window_is_still_open(conn):
    """An overdue commit inside the tradeable window is URGENT — Julian could
    still act. One past the window is history. Collapsing the two would bury
    the actionable case."""
    now = _at("2026-08-10T05:10:00")
    got = appbet.missing_commits(conn, CFGS, now, lookback_days=0)
    assert got, "the fixture has no bets, so something must be overdue"
    for m in got:
        assert "still_actionable" in m, m


def test_looks_back_but_not_forever(conn):
    """Yesterday's miss is worth knowing once. A miss from last month would
    shout on every wake forever."""
    now = _at("2026-08-10T05:10:00")
    got = appbet.missing_commits(conn, CFGS, now, lookback_days=1)
    oldest = min((m["target_date"] for m in got), default="9999")
    assert oldest >= "2026-08-08", got


def test_clean_history_reports_nothing(conn):
    now = _at("2026-08-10T05:10:00")
    for d in ("2026-08-08", "2026-08-09"):
        for s, mets in (("KLAX", ("low", "high")), ("KMIA", ("low", "high"))):
            for m in mets:
                _bet(conn, s, m, d)
    for d in ("2026-08-10",):
        for m in ("low", "high"):
            _bet(conn, "KMIA", m, d)
    assert appbet.missing_commits(conn, CFGS, now, lookback_days=1) == []
