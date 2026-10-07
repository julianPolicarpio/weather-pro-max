"""WHEN ARE WE BLIND? Six-hour-group coverage of a climate day.

THE INCIDENT, 2026-08-06. KMIA settled a low of 75F at 01:51 local. Our best
observation was 78.1F at the 01:53 METAR — two minutes later — and the model
predicted 78-79 on exactly that evidence. The event was invisible.

The 1sTTT/2sTTT six-hour remark groups exist precisely to catch extremes that
fall between hourly filings. Two things defeated them that day:

1. The 05:53Z report filed NO 2sTTT group at all (~3% of synoptic reports omit
   it), so the six hours it would have summarised are simply unreported.
2. Even had it been filed, `visible_running_extremum` would have REFUSED it:
   its window (23:53Z-05:53Z) straddles the 05:00Z climate-day start, and an
   admissible group must lie wholly inside the day. That rule is correct —
   otherwise you import the previous day's extreme — but it means a window at
   each end of every climate day is covered by hourly spot readings alone.

This module does NOT change what is admissible and does NOT touch the running
extremum. It reports where the gaps are, so "we were blind" is something the
desk can say at commit time instead of something discovered in a settlement
three days later.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from highcast import db as hdb, latentmax
from highcast.config import load_config

DAY = dt.date(2026, 8, 6)


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/kmia.yaml")
    object.__setattr__(c, "db_path", tmp_path / "t.db")
    return c


def _seed(cfg, six_at_hours=(5, 11, 17, 23), spot_every_hour=True):
    """METARs at :53. `six_at_hours` are the UTC hours whose report carries a
    2sTTT group (KMIA's synoptic reports are 05/11/17/23 :53Z)."""
    conn = hdb.connect(cfg.db_path)
    rows = []
    base = dt.datetime(2026, 8, 5, 0, 53, tzinfo=dt.timezone.utc)
    for i in range(72):
        t = base + dt.timedelta(hours=i)
        if not spot_every_hour and t.hour not in six_at_hours:
            continue
        rows.append({
            "station": cfg.station.icao, "valid_utc": t.isoformat(),
            "source": "metar", "tmpf": 80.0, "tmpf_precise": 80.0,
            "dwpf": 70.0, "sknt": 5.0, "drct": 90.0, "skyc": "FEW",
            "six_max_c": None,
            "six_min_c": 25.0 if t.hour in six_at_hours else None,
            "raw": "METAR", "fetched_at": "x"})
    hdb.upsert(conn, "obs", rows)
    return conn


def test_a_normal_day_still_has_gaps_at_both_ends(cfg):
    """THE STRUCTURAL FINDING, and it is not a bug — it falls out of the
    admissibility rule. Even when every synoptic report files its group, the
    first and last stretches of the climate day are covered by spot readings
    only, because the groups spanning them straddle the boundaries."""
    conn = _seed(cfg)
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min")
    assert cov["gaps"], "a climate day cannot be fully covered by 6h groups"
    total = sum(g["minutes"] for g in cov["gaps"])
    assert total > 0
    # the day opens at 05:00Z; the first admissible group ends 05:53Z
    assert any(g["start"].endswith("05:00:00+00:00") for g in cov["gaps"]), cov["gaps"]


def test_a_missing_group_widens_the_gap(cfg):
    """The 2026-08-06 case: drop the 11:53Z group and the blind window grows by
    six hours."""
    full = latentmax.six_hour_coverage(_seed(cfg), cfg, DAY, "min")
    cfg2 = load_config("config/kmia.yaml")
    object.__setattr__(cfg2, "db_path", cfg.db_path.parent / "t2.db")
    holed = latentmax.six_hour_coverage(
        _seed(cfg2, six_at_hours=(5, 17, 23)), cfg2, DAY, "min")
    assert (sum(g["minutes"] for g in holed["gaps"])
            > sum(g["minutes"] for g in full["gaps"]) + 300)
    assert holed["missing_groups"], "the absent report must be named"


def test_it_names_the_unreported_SYNOPTIC_WINDOW_not_a_report(cfg):
    """Named by window end (12:00Z), not by the report that should have
    carried it (11:53Z). Several reports can fall inside one window — the
    routine filing plus any SPECIs — so the window is the thing that is or is
    not closed, and the report is an implementation detail of who closes it."""
    conn = _seed(cfg, six_at_hours=(5, 17, 23))
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min")
    assert any(m.endswith("12:00:00+00:00") for m in cov["missing_groups"]), \
        cov["missing_groups"]


def test_a_speci_is_not_reported_as_a_missing_group(cfg):
    """A SPECI fires whenever conditions jump. One at 17:20Z rounds to the same
    18:00Z window as the routine 17:53Z filing, and the first version of this
    code named four of them as missing groups on 2026-08-06 when only the
    11:53Z routine report actually was. SPECIs are never expected to carry a
    six-hour group."""
    conn = _seed(cfg)
    hdb.upsert(conn, "obs", [{
        "station": cfg.station.icao,
        "valid_utc": dt.datetime(2026, 8, 6, 17, 20,
                                 tzinfo=dt.timezone.utc).isoformat(),
        "source": "metar", "tmpf": 84.0, "tmpf_precise": 84.0, "dwpf": 70.0,
        "sknt": 5.0, "drct": 90.0, "skyc": "FEW", "six_max_c": None,
        "six_min_c": None, "raw": "SPECI", "fetched_at": "x"}])
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min")
    assert not any("17:20" in m for m in cov["missing_groups"]), cov["missing_groups"]


def test_max_and_min_are_asked_separately(cfg):
    """A report can carry 1sTTT and not 2sTTT. The fixture seeds only
    `six_min_c`, so the MAX must show total blindness while the MIN does not."""
    conn = _seed(cfg)
    mn = latentmax.six_hour_coverage(conn, cfg, DAY, "min")
    mx = latentmax.six_hour_coverage(conn, cfg, DAY, "max")
    assert sum(g["minutes"] for g in mx["gaps"]) > \
        sum(g["minutes"] for g in mn["gaps"])


def test_a_window_that_has_not_closed_yet_is_not_missing(cfg):
    """THE LIVE-USE BUG. The wake cycle runs every 5 minutes, so most of the
    time it is asked mid-day. At 09:00Z the 12:00Z and 18:00Z windows have not
    happened yet — reporting them as unreported would cry wolf on every wake
    for most of the day, which is how a real warning gets ignored."""
    conn = _seed(cfg, six_at_hours=(5,))          # only the 06:00Z window filed
    asof = dt.datetime(2026, 8, 6, 9, 0, tzinfo=dt.timezone.utc).isoformat()
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min", asof_utc=asof)
    assert cov["missing_groups"] == [], cov["missing_groups"]
    assert cov["pending_groups"], "windows still to come must be visible as pending"


def test_a_window_that_HAS_closed_is_still_reported(cfg):
    conn = _seed(cfg, six_at_hours=(5,))
    asof = dt.datetime(2026, 8, 6, 13, 0, tzinfo=dt.timezone.utc).isoformat()
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min", asof_utc=asof)
    assert any(m.endswith("12:00:00+00:00") for m in cov["missing_groups"])


def test_blind_windows_stop_at_now(cfg):
    """Blind minutes must count elapsed time only — the rest of the day has
    not had a chance to be covered."""
    conn = _seed(cfg)
    asof = dt.datetime(2026, 8, 6, 8, 0, tzinfo=dt.timezone.utc).isoformat()
    cov = latentmax.six_hour_coverage(conn, cfg, DAY, "min", asof_utc=asof)
    assert cov["day_minutes"] == 180        # 05:00Z -> 08:00Z elapsed
    for g in cov["gaps"]:
        assert g["end"] <= asof, g


def test_it_never_changes_the_running_extremum(cfg):
    """Report only. The admissibility rule is correct and this must not
    relax it."""
    conn = _seed(cfg)
    before = latentmax.visible_running_extremum(conn, cfg, DAY, "min")
    latentmax.six_hour_coverage(conn, cfg, DAY, "min")
    after = latentmax.visible_running_extremum(conn, cfg, DAY, "min")
    assert before == after
