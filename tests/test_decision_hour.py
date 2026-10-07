"""The nowcast must not be acted on before its measured accuracy knee.

WHY (2026-07-26, on real money). The live path refits on ONLY the current local
hour, so every hour boundary fits a different model. Julian bought 68-69 on the
LOW, then the desk issued PLACE on 70-71 an hour later. Holding the day's data
fixed and varying only the hour reproduced it exactly:

    hour 3  point 68.82   68-69 74%   <- recommended 68-69
    hour 4  point 68.99   68-69 65%
    hour 5  point 69.99   70-71 59%   <- recommended 70-71, same data

`scripts/hour_curve.py` measured the cost: the LOW's modal bucket is right 67.0%
of the time at hour 6 but 84.7% from hour 11 (when the 18Z 6-hour-min group
lands); the HIGH goes 45.7% at hour 9 to 83.3% at hour 16.

The second test here is the load-bearing one. An hour veto that does not admit
its own metric's decision hour blocks the market PERMANENTLY -- that is exactly
what the removed `intraday_hour_trained` veto did to the LOW, and gating.py
carries a standing warning about it. The gate and the config cutoffs are two
separate knobs that must agree, so assert they do.
"""
import dataclasses
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from highcast import gating
from highcast.config import StakingConfig, load_config
from highcast.gating import (MIN_DECISION_HOUR, decision_hour_ready,
                             hour_accuracy, personal_cutoff_instant,
                             preferred_prediction)
from highcast.metric import get_metric

CFG = load_config()
SPECS = {"low": get_metric(CFG, "low"), "high": get_metric(CFG, "high")}


def _at(local_hour: int, day="2026-07-26") -> str:
    """An issued_at_utc whose station-local wall clock is `local_hour`."""
    tz = ZoneInfo(CFG.station.tz)
    d = datetime.fromisoformat(day)
    return (datetime(d.year, d.month, d.day, local_hour, 30, tzinfo=tz)
            .astimezone(timezone.utc).isoformat())


@pytest.mark.parametrize("metric", ["low", "high"])
def test_before_the_knee_is_not_ready(metric):
    knee = MIN_DECISION_HOUR[CFG.station.icao][metric]
    for h in range(1, knee):
        assert decision_hour_ready(CFG, SPECS[metric], _at(h)) is False, h


@pytest.mark.parametrize("metric", ["low", "high"])
def test_from_the_knee_onward_is_ready(metric):
    for h in range(MIN_DECISION_HOUR[CFG.station.icao][metric], 24):
        assert decision_hour_ready(CFG, SPECS[metric], _at(h)) is True, h


@pytest.mark.parametrize("metric", ["low", "high"])
def test_the_window_admits_the_decision_hour(metric):
    """THE invariant. If the personal cutoff lands before the accuracy knee, the
    gate and the window never overlap and the market can never be traded --
    the failure mode gating.py warns about in the removed-veto note."""
    day = datetime.fromisoformat("2026-07-26").date()
    cutoff = personal_cutoff_instant(CFG, day, metric)
    assert cutoff is not None, f"no cutoff instant inside the climate day for {metric}"
    cutoff_local_hour = cutoff.astimezone(ZoneInfo(CFG.station.tz)).hour
    knee = MIN_DECISION_HOUR[CFG.station.icao][metric]
    assert cutoff_local_hour > knee, (
        f"{metric}: cutoff is {cutoff_local_hour}:00 station-local but the "
        f"decision hour is {knee}:00 — the window closes before the model is "
        f"trustworthy, so this market is permanently blocked")


@pytest.mark.parametrize("metric", ["low", "high"])
def test_the_knee_is_the_measured_best_of_the_early_hours(metric):
    """The knee must not be an arbitrary constant: it has to beat every earlier
    hour on measured bucket hit rate, or it is not a knee."""
    icao = CFG.station.icao
    knee_hit = gating.HOUR_ACCURACY[icao][metric][MIN_DECISION_HOUR[icao][metric]][1]
    for h, (_, hit) in gating.HOUR_ACCURACY[icao][metric].items():
        if h < MIN_DECISION_HOUR[icao][metric]:
            assert hit < knee_hit, f"{metric} hour {h} ({hit:.1%}) beats the knee"


def test_hour_accuracy_reports_readiness_and_handles_unmeasured():
    icao = CFG.station.icao
    # Knee is hour 6 for the LOW (moved 4 -> 6 on 2026-07-31, see
    # gating.MIN_DECISION_HOUR), so 3 is early and 6 is ready. It was briefly 11 —
    # an hour at which the min had already occurred on 71% of days, i.e.
    # scorekeeping rather than forecasting.
    assert hour_accuracy(icao, "low", 3)["ready"] is False
    assert hour_accuracy(icao, "low", 6)["ready"] is True
    assert hour_accuracy(icao, "low", 11)["ready"] is True
    # Re-measured after the MOS-min backfill (was 0.670). Pinning the value keeps
    # the dashboard's "67% at 6:00" row honest — a stale table would keep quoting
    # a number the model no longer produces.
    # 0.684 -> 0.682 when HOUR_ACCURACY was re-measured 2026-07-31 against a
    # slightly larger holdout. KLAX LOW's feature set did not change; this is noise.
    assert hour_accuracy(icao, "low", 6)["hit"] == pytest.approx(0.682)
    assert hour_accuracy(icao, "low", 11)["hit"] > hour_accuracy(icao, "low", 6)["hit"]
    assert hour_accuracy(icao, "low", 0) is None      # hour 0 has no training rows
    assert hour_accuracy(icao, "nonsense", 11) is None
    # Unmeasured STATION (not just unmeasured metric) must also return None,
    # never fall back to this station's numbers. KMIA no longer serves as this
    # example — it got its own measured HOUR_ACCURACY on 2026-07-28 — so this
    # uses a station with genuinely no entry in the table.
    assert hour_accuracy("KJFK", "low", 4) is None


def test_naive_issued_at_is_read_as_utc():
    """`predictions.issued_at_utc` is sometimes stored without an offset. Reading
    a naive stamp as station-local instead of UTC would shift the gate by 7-8h."""
    # 13:30Z = 06:30 LA -> ready (knee moved 4 -> 6 on 2026-07-31, see gating.MIN_DECISION_HOUR)
    naive = "2026-07-26T13:30:00"
    assert decision_hour_ready(CFG, SPECS["low"], naive) is True
    naive_early = "2026-07-26T09:30:00"  # 09:30Z = 02:30 LA -> not ready
    assert decision_hour_ready(CFG, SPECS["low"], naive_early) is False
    # If the naive stamp were read as station-LOCAL instead of UTC, 09:30 would
    # look like 09:30 LA and wrongly pass. That 7-8h shift is the bug this pins.


def _seed(conn, cfg, target, kind, metric, point):
    db_buckets = '{"%d": 1.0}' % int(point)
    conn.execute(
        """INSERT INTO predictions (station, target_date, issued_at_utc, kind,
             point_f, quantiles, buckets, metric) VALUES (?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, target, _at(12), kind, point, "{}", db_buckets, metric))
    conn.commit()


def test_high_prefers_the_nowcast_once_past_its_knee(tmp_path):
    """The HIGH window now closes at LA 18:00, past the nowcast's knee at 16.
    A window-only preference would keep serving the ~1.4-MAE day-ahead there and
    discard the nowcast's measured 0.452."""
    from highcast import db
    cfg = dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                              staking=StakingConfig(bankroll=8.0))
    conn = db.connect(cfg.db_path)
    _seed(conn, cfg, "2026-07-26", "dayahead", "high", 80.0)
    _seed(conn, cfg, "2026-07-26", "intraday", "high", 82.0)
    args = (conn, cfg.station.icao, "2026-07-26", "high", True)
    assert preferred_prediction(*args, local_hour=9)[1] == "dayahead"
    assert preferred_prediction(*args, local_hour=16)[1] == "intraday"
    # Omitting the hour keeps the old window-only behaviour for callers (the
    # roll-to-tomorrow path) where the hour is not meaningful.
    assert preferred_prediction(*args)[1] == "dayahead"


def test_low_always_prefers_the_nowcast(tmp_path):
    """Unchanged: the LOW's min has largely already occurred, so the nowcast
    wins at every hour (dMAE +0.663 against the day-ahead)."""
    from highcast import db
    cfg = dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                              staking=StakingConfig(bankroll=8.0))
    conn = db.connect(cfg.db_path)
    _seed(conn, cfg, "2026-07-26", "dayahead", "low", 70.0)
    _seed(conn, cfg, "2026-07-26", "intraday", "low", 68.0)
    for h in (3, 6, 11, 20):
        assert preferred_prediction(conn, cfg.station.icao, "2026-07-26", "low",
                                    True, local_hour=h)[1] == "intraday"


@pytest.mark.parametrize("kind,expect_accuracy", [("dayahead", False),
                                                 ("intraday", True)])
def test_hour_accuracy_is_reported_for_the_nowcast_only(tmp_path, monkeypatch,
                                                        kind, expect_accuracy):
    """Regression: /api/bet labelled the evening-issued day-ahead card with the
    NOWCAST's hour curve, rendering "99% at 18:00" (a post-max nowcast number) as
    a PASSING gate on the 73.2F prediction that the morning-obs veto was blocking
    at the same moment — a green tick on the most broken read of the day.
    HOUR_ACCURACY is a per-hour curve of the nowcast; the day-ahead has one daily
    cutoff and its issued_at hour indexes nothing."""
    from fastapi.testclient import TestClient

    from highcast import db
    from highcast.fetchers import kalshi
    from highcast.web.app import create_app

    monkeypatch.setattr(kalshi, "fetch_ranges", lambda *a, **k: [])  # no network
    cfg = dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                              staking=StakingConfig(bankroll=8.0))
    conn = db.connect(cfg.db_path)
    today = datetime.now(ZoneInfo(CFG.station.tz)).date().isoformat()
    _seed(conn, cfg, today, kind, "high", 80.0)
    conn.close()

    client = TestClient(create_app(cfg))
    body = client.get("/api/bet?metric=high").json()
    assert body["kind"] == kind
    assert (body["hour_accuracy"] is not None) is expect_accuracy


def test_decision_hours_are_the_measured_peak_extremum_hours():
    """Pins the commit hours so a future edit is deliberate, not drift.

    Three of the four came from `scripts/decision_hour_move_eval.py` on
    2026-07-31, a pre-registered paired gate. They are each station's MEDIAN
    peak-extremum hour, which is why roughly half the extremum is still ahead
    at commit — the property that keeps the answer a forecast rather than a
    transcription.

    KMIA HIGH IS THE EXCEPTION and its 14 -> 13 move on 2026-08-05 FAILS that
    gate on purpose: Dhit -11.2%, CI [-15.1%,-7.8%] over 574 matched days. It
    was moved on PRICE, which the gate has no term for. At h14 the modal
    bucket's mean ask is 0.82 against a 79.3% hit rate — below the break-even
    bar, since a contract at price P needs a hit rate of P — while h13 asks
    0.46 against 68.0%. Measured on the same days, the desk's own gates
    admitted a trade on 5 of 7 days at h13 and 2 of 7 at h14.

    A future editor should read that as: this hour is defensible on the margin
    argument and indefensible on accuracy alone, and it is being watched. If
    the price cliff at 14:00 stops showing up in `signals`, the reason for h13
    is gone and it should move back.
    """
    assert gating.MIN_DECISION_HOUR["KLAX"] == {"low": 6, "high": 13}
    assert gating.MIN_DECISION_HOUR["KMIA"] == {"low": 7, "high": 13}


def test_every_decision_hour_still_leaves_the_extremum_mostly_unknown():
    """THE reason these hours were chosen. If a future edit pushes a commit hour
    past the point where the extremum has usually happened, the desk is
    transcribing a published number, not forecasting — and a bucket the market
    has already read carries no edge either."""
    for icao, per_metric in gating.MIN_DECISION_HOUR.items():
        for metric, hour in per_metric.items():
            ahead = gating.EXTREMUM_STILL_FUTURE[icao][metric][hour]
            assert ahead >= 0.40, (
                f"{icao} {metric} h{hour}: only {ahead:.1%} of the extremum is "
                f"still ahead — that is scorekeeping, not forecasting")
