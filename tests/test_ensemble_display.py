"""The ensemble panel is a DISPLAY, and its data source contains hindcasts.

⚠️ THE LEAK, found 2026-08-28. `forecast_ens` holds rows whose `run_date` is
AFTER `target_date` — 234 of 554 rows, leads from −5 to +2 days. "The latest row
for this target" would put a run made after the day ended on the desk as a
forecast. That is error #1's shape (commit-hour lookahead), and these tests pin
both filters: `run_date <= target_date` AND `fetched_at <= now`.

⚠️ IT DOES NOT PREDICT OUTCOMES. Over 96 settled bets with a leak-free row:
ens_std +0.034 CI [−0.168,+0.233], ens_bimodality −0.038 CI [−0.237,+0.164],
p90−p10 +0.032 CI [−0.170,+0.231]. All straddle zero, consistent with the
guidance-spread veto already being dead. It may never become a veto or a feature.

WHAT IT IS FOR: 119 members spanning 15 °F on a 2 °F ladder is EIGHT buckets.
That is arithmetic about the question, not a forecast of the answer.
"""
from datetime import date, datetime, timezone

import pytest

from highcast.config import load_config
from highcast.db import connect
from highcast.web.app import _ensemble


@pytest.fixture
def cfg():
    return load_config("config/kmia.yaml")


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "t.db"))
    yield c
    c.close()


NOW = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
TARGET = date(2026, 8, 24)


def _ens(c, run_date, fetched_at, p10=84.0, p90=99.0, std=6.0, bim=1.1):
    c.execute("""INSERT INTO forecast_ens (station, target_date, run_date, metric,
                   n_members, ens_mean, ens_std, ens_p10, ens_p50, ens_p90,
                   ens_bimodality, fetched_at)
                 VALUES ('KMIA','2026-08-24',?, 'high',119,91.0,?,?,90.0,?,?,?)""",
              (run_date, std, p10, p90, bim, fetched_at))
    c.commit()


def test_reports_the_span_in_buckets_not_just_degrees(cfg, conn):
    """The ladder is 2F wide; degrees are not the unit the desk bets in."""
    _ens(conn, "2026-08-23", "2026-08-24T09:00:00+00:00", p10=84.0, p90=99.0)
    got = _ensemble(conn, cfg, "high", TARGET, NOW)
    assert got["span_f"] == pytest.approx(15.0)
    assert got["span_buckets"] == 8
    assert got["members"] == 119


def test_a_hindcast_run_is_never_shown(cfg, conn):
    """⚠️ run_date AFTER target_date is a run made once the day was over."""
    _ens(conn, "2026-08-27", "2026-08-27T09:00:00+00:00")
    assert _ensemble(conn, cfg, "high", TARGET, NOW) is None


def test_a_row_fetched_after_now_is_never_shown(cfg, conn):
    _ens(conn, "2026-08-23", "2026-08-24T23:00:00+00:00")
    assert _ensemble(conn, cfg, "high", TARGET, NOW) is None


def test_prefers_the_most_recent_leak_free_run(cfg, conn):
    _ens(conn, "2026-08-22", "2026-08-22T09:00:00+00:00", p90=95.0)
    _ens(conn, "2026-08-23", "2026-08-23T09:00:00+00:00", p90=99.0)
    _ens(conn, "2026-08-27", "2026-08-27T09:00:00+00:00", p90=200.0)  # hindcast
    got = _ensemble(conn, cfg, "high", TARGET, NOW)
    assert got["run_date"] == "2026-08-23" and got["p90"] == pytest.approx(99.0)


def test_missing_percentiles_yield_nothing_rather_than_a_bad_span(cfg, conn):
    conn.execute("""INSERT INTO forecast_ens (station, target_date, run_date,
                      metric, n_members, fetched_at)
                    VALUES ('KMIA','2026-08-24','2026-08-23','high',119,
                            '2026-08-23T09:00:00+00:00')""")
    conn.commit()
    assert _ensemble(conn, cfg, "high", TARGET, NOW) is None


def test_no_rows_at_all_is_silent(cfg, conn):
    assert _ensemble(conn, cfg, "high", TARGET, NOW) is None


def test_display_only_never_reaches_a_model_a_stake_or_a_gate():
    """⚠️ PINS THE DESIGN. Measured non-predictive at n=96; the guidance-spread
    veto that this resembles was measured BACKWARDS in 2026-08-06."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "highcast"
    for mod in ("staking.py", "gating.py", "features.py", "models/intraday.py"):
        assert "forecast_ens" not in (src / mod).read_text(), (
            f"{mod} reads forecast_ens — it is a DISPLAY. Spread does not "
            "predict outcomes (n=96, all CIs straddle zero).")
