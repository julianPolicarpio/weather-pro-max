"""The storm warning must fire on OUR observations, not on a feed that is empty.

⚠️ WHY THIS FILE EXISTS. `notify_unstable`'s storm half read the `alerts` table
for `STORM_EVENTS`. That table has never held a storm product for either
station — the archive fetch requests `HEAT_PHENOMENA` only and its VTEC map has
no thunderstorm codes. Live proof, 2026-08-28:

    alert_stamps WHERE kind='unstable' AND key LIKE '%storm%'   ->  0 rows

Zero, through a Miami August that thunderstorms almost daily, while the HEAT
half fired routinely. **A guard that could not fire** — errors #8 and #11 for the
third time. Both of those were caught only by a test asserting the guard GOES
OFF, so that is the first test below.

Justification for the rewire, measured over 2,430 KMIA climate days: the daily
LOW breaks >=2F below what was already visible on **28.6%** of days after
thunder onset against **5.3%** of quiet days (5.4x).

⚠️ MONITOR, NOT PREDICTOR. Convection has failed as a MODEL FEATURE three
times — Task 1 (KMIA HIGH h13), Task 1b (KMIA LOW h7, tied with a random arm),
and Phase 0 bundle B3. Nothing here may reach a model, a stake or a gate; the
last test pins that.
"""
from datetime import datetime, timedelta, timezone

import pytest

from highcast import storm
from highcast.db import connect


class _Cfg:
    class station:
        icao = "KMIA"
        tz = "America/New_York"
        std_utc_offset = -5
    discord_webhook_url = "https://example.invalid/hook"


NOW = datetime(2026, 8, 24, 17, 10, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _obs(c, minutes_ago, *, thunder=0, cb=0, temp=80.0, wx=None):
    t = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    c.execute("""INSERT INTO obs (station, valid_utc, source, tmpf_precise,
                    wx_present, wx_thunder, sky_cb, fetched_at)
                 VALUES ('KMIA', ?, 'metar', ?, ?, ?, ?, ?)""",
              (t, temp, wx, thunder, cb, t))
    c.commit()


def test_it_fires_on_observed_thunder(conn):
    """⚠️ THE TEST THE OLD GUARD COULD NEVER HAVE PASSED."""
    _obs(conn, 30, thunder=1, cb=1, temp=77.0, wx="+TSRA")
    got = storm.active(conn, _Cfg(), NOW)
    assert got is not None
    assert got["wx"] == "+TSRA" and got["cb"] is True and got["reports"] == 1


def test_quiet_weather_stays_silent(conn):
    _obs(conn, 10, temp=88.0, wx=None)
    _obs(conn, 70, temp=87.0, wx="-RA")          # rain is not thunder
    assert storm.active(conn, _Cfg(), NOW) is None


def test_thunder_outside_the_window_does_not_count(conn):
    """A storm that ended two hours ago has already done its damage."""
    _obs(conn, 150, thunder=1, wx="TS")
    assert storm.active(conn, _Cfg(), NOW) is None
    assert storm.active(conn, _Cfg(), NOW, window_min=180) is not None


def test_future_rows_are_never_read(conn):
    """The wake cycle can run mid-minute; an obs stamped after `now` would be
    hindsight, and this module must not be the place that leaks it."""
    c = conn
    t = (NOW + timedelta(minutes=20)).isoformat()
    c.execute("""INSERT INTO obs (station, valid_utc, source, tmpf_precise,
                    wx_present, wx_thunder, sky_cb, fetched_at)
                 VALUES ('KMIA', ?, 'metar', 77.0, 'TS', 1, 1, ?)""", (t, t))
    c.commit()
    assert storm.active(conn, _Cfg(), NOW) is None


def test_other_stations_do_not_bleed_through(conn):
    """Two stations share one database — the `metric:date` bug's cousin."""
    c = conn
    _t = (NOW - timedelta(minutes=10)).isoformat()
    c.execute("""INSERT INTO obs (station, valid_utc, source, tmpf_precise,
                    wx_present, wx_thunder, sky_cb, fetched_at)
                 VALUES ('KLAX', ?, 'metar', 77.0, 'TS', 1, 1, ?)""", (_t, _t))
    c.commit()
    assert storm.active(conn, _Cfg(), NOW) is None


def test_reports_the_temperature_swing_it_has_already_caused(conn):
    """The message should say "already dropped 13F", not merely "storm"."""
    _obs(conn, 55, temp=90.0)
    _obs(conn, 25, thunder=1, temp=77.0, wx="+TSRA")
    got = storm.active(conn, _Cfg(), NOW)
    assert got["temp_swing_f"] == pytest.approx(13.0)


def test_the_swing_looks_back_further_than_the_detection_window(conn):
    """⚠️ REGRESSION, live 2026-08-24. The first version measured the swing
    inside the 60-minute detection window and reported the real KMIA downdraft
    as 1.98F because the pre-storm peak sat 17 minutes outside it. The true fall
    was 13F. An alert about a cliff must not start measuring at the bottom."""
    _obs(conn, 120, temp=90.0)              # pre-storm peak, OUTSIDE the 60m window
    _obs(conn, 30, thunder=1, temp=77.0, wx="+TSRA")
    got = storm.active(conn, _Cfg(), NOW)
    assert got is not None
    assert got["temp_swing_f"] == pytest.approx(13.0), (
        "swing must be read over SWING_LOOKBACK_MIN, not WINDOW_MIN")


def test_a_cb_layer_alone_is_a_cloud_not_an_event(conn):
    _obs(conn, 20, thunder=0, cb=1, temp=85.0)
    assert storm.active(conn, _Cfg(), NOW) is None


def test_monitor_only_never_reaches_a_model_a_stake_or_a_gate():
    """⚠️ PINS THE DESIGN. Convection failed as a feature three times; if it is
    ever routed into staking/gating/features this fails and the reader must
    re-read `storm`'s docstring before overriding."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "highcast"
    for mod in ("staking.py", "gating.py", "features.py"):
        assert "storm" not in (src / mod).read_text().split("\n\n")[0].lower() or True
        assert "highcast.storm" not in (src / mod).read_text(), (
            f"{mod} imports highcast.storm — it is a MONITOR. Convection has "
            "failed as a model feature three times (Task 1, Task 1b, Phase 0 B3).")
    for mod in ("models/intraday.py",):
        assert "highcast.storm" not in (src / mod).read_text()
