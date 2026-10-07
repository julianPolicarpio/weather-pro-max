import json
from datetime import datetime, timezone

from highcast import db, gating
from highcast.config import load_config

CFG = load_config()


def _seed_pred(conn, target, issued, buckets, metric="high", kind="intraday"):
    db.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": target,
        "issued_at_utc": issued, "kind": kind, "point_f": 77.0,
        "quantiles": json.dumps({}),
        "buckets": json.dumps(buckets), "metric": metric}])


CONFIDENT = {"77": 0.5, "78": 0.4, "79": 0.1}   # 77-78 bucket = 0.9
WEAK = {"77": 0.4, "78": 0.3, "79": 0.3}        # 77-78 bucket = 0.7
OTHER = {"79": 0.5, "80": 0.4, "78": 0.1}       # 79-80 bucket = 0.9


def test_personal_cutoff_in_climate_day():
    from zoneinfo import ZoneInfo
    from datetime import date
    from highcast.timeutil import climate_day_bounds
    day = date(2026, 7, 22)
    cut = gating.personal_cutoff_instant(CFG, day)
    assert cut is not None
    start, end = climate_day_bounds(day, CFG.station.std_utc_offset)
    assert start <= cut < end
    # Manila wall clock is exactly the configured cutoff. Read it FROM config so
    # this test checks the wiring rather than pinning a policy number that moves
    # whenever the measured decision hour does (it moved 01:30 -> 09:00 on
    # 2026-07-26; see gating.MIN_DECISION_HOUR).
    assert (cut.astimezone(ZoneInfo(CFG.display_tz)).strftime("%H:%M")
            == CFG.staking.personal_cutoff)


def test_committed_requires_confidence_and_stability(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    for i, ts in enumerate(["2026-07-22T13:00:00+00:00",
                            "2026-07-22T13:05:00+00:00",
                            "2026-07-22T13:10:00+00:00"]):
        _seed_pred(conn, "2026-07-22", ts, CONFIDENT)
    cs = gating.committed_state(conn, CFG, "2026-07-22", "high")
    assert cs["committed"] is True
    assert cs["label"] == "77-78" and cs["confident"] and cs["stable"]


def test_not_committed_when_too_few_readings(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    _seed_pred(conn, "2026-07-22", "2026-07-22T13:00:00+00:00", CONFIDENT)
    cs = gating.committed_state(conn, CFG, "2026-07-22", "high")
    assert cs["committed"] is False and cs["stable"] is False


def test_not_committed_when_oscillating(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    _seed_pred(conn, "2026-07-22", "2026-07-22T13:00:00+00:00", CONFIDENT)
    _seed_pred(conn, "2026-07-22", "2026-07-22T13:05:00+00:00", OTHER)
    _seed_pred(conn, "2026-07-22", "2026-07-22T13:10:00+00:00", CONFIDENT)
    cs = gating.committed_state(conn, CFG, "2026-07-22", "high")
    assert cs["stable"] is False and cs["committed"] is False


def test_not_committed_when_not_confident(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    for ts in ["2026-07-22T13:00:00+00:00", "2026-07-22T13:05:00+00:00",
               "2026-07-22T13:10:00+00:00"]:
        _seed_pred(conn, "2026-07-22", ts, WEAK)
    cs = gating.committed_state(conn, CFG, "2026-07-22", "high")
    assert cs["stable"] is True and cs["confident"] is False
    assert cs["committed"] is False


def test_gate_state_within_window_flag(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    for ts in ["2026-07-22T13:00:00+00:00", "2026-07-22T13:05:00+00:00",
               "2026-07-22T13:10:00+00:00"]:
        _seed_pred(conn, "2026-07-22", ts, CONFIDENT)
    # The HIGH cutoff is Manila 09:00 = LA 18:00 (it moved out from LA 10:30 on
    # 2026-07-26 to admit the nowcast's measured knee at LA 16:00).
    # 13:00Z on 2026-07-22 is LA ~06:00 -> open
    early = datetime(2026, 7, 22, 13, 0, tzinfo=timezone.utc)
    assert gating.gate_state(conn, CFG, "2026-07-22", "high", early)["within_window"] is True
    # 23:00Z is LA ~16:00, the decision hour itself -> still open
    knee = datetime(2026, 7, 22, 23, 0, tzinfo=timezone.utc)
    assert gating.gate_state(conn, CFG, "2026-07-22", "high", knee)["within_window"] is True
    # 02:00Z next day is LA ~19:00, past the cutoff -> closed
    late = datetime(2026, 7, 23, 2, 0, tzinfo=timezone.utc)
    assert gating.gate_state(conn, CFG, "2026-07-22", "high", late)["within_window"] is False


def test_low_metric_isolated_from_high(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    for ts in ["2026-07-22T13:00:00+00:00", "2026-07-22T13:05:00+00:00",
               "2026-07-22T13:10:00+00:00"]:
        _seed_pred(conn, "2026-07-22", ts, CONFIDENT, metric="high")
    # no low predictions -> low gate is not committed even though high is
    assert gating.committed_state(conn, CFG, "2026-07-22", "low")["n_readings"] == 0
    assert gating.committed_state(conn, CFG, "2026-07-22", "high")["committed"] is True


def test_low_window_closes_earlier_than_high():
    """The LOW's window still closes first, but for a measured reason rather than
    "dawn": its answer becomes knowable earlier in the day. The 18Z 6-hour-min
    group lands at LA 10:53 and the nowcast's bucket hit rate steps 77.3% ->
    84.7% right after it (LA 11:00), whereas the HIGH's max is not observed until
    LA 16:00-17:00. So LOW closes at LA 13:00 and HIGH at LA 18:00."""
    from zoneinfo import ZoneInfo
    from datetime import date
    day = date(2026, 7, 22)
    lo = gating.personal_cutoff_instant(CFG, day, "low")
    hi = gating.personal_cutoff_instant(CFG, day, "high")
    assert (lo.astimezone(ZoneInfo(CFG.display_tz)).strftime("%H:%M")
            == CFG.staking.low_personal_cutoff)
    assert lo < hi   # low window closes earlier in absolute time
    # Each cutoff must leave its own decision hour actionable.
    for metric, cut in (("low", lo), ("high", hi)):
        local_hour = cut.astimezone(ZoneInfo(CFG.station.tz)).hour
        assert local_hour > gating.MIN_DECISION_HOUR[CFG.station.icao][metric], metric
