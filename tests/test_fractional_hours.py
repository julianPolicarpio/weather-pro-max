"""Checkpoints at HALF hours — the enabling change for a 13:30 commit.

WHY THIS IS NEEDED. On KMIA HIGH the ask cliffs at 14:00 station-local: the
first signal in each 15-minute slot prices the modal bucket at 0.46 / 0.41 /
0.38 / 0.39 through 13:00-13:45 and then 0.82 at 14:00. The market sets that
price, not us, so the step is independent evidence of a real information event
— the median peak-extremum hour is 13:58, i.e. by 14:00 the day's high has
usually already happened and is visible in the ASOS feed to everyone.

Committing at 13:30 rather than 13:00 therefore buys 30 more minutes of
observation at the same price. `signals` already records prices every 5
minutes, so the PRICE side needs nothing. The ACCURACY side is what is
integer-hour: every checkpoint was built with `datetime(y, m, d, h)`.

The model needs no change — `hour` is already a float column in the frame — so
this is only about letting the frame be BUILT at 13.5.

⚠️ THIS MOVES NO COMMIT HOUR. `gating.MIN_DECISION_HOUR` stays untouched until
a walk-forward at 13.5 has filled HOUR_ACCURACY, HOUR_CALIB, BAND_CALIB and
CONFIDENCE_FLOOR at that key. All four fail closed on a miss, so moving the
constant alone leaves the market permanently inert — the failure BAND_CALIB's
own comment records ("the suite sat at 26 failures between the two changes").
"""
import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from highcast import db as hdb, latentmax
from highcast.config import load_config
from highcast.metric import get_metric
from highcast.models.intraday import _split_hour, build_checkpoint_frame

DAY = dt.date(2026, 7, 20)


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/kmia.yaml")
    object.__setattr__(c, "db_path", tmp_path / "t.db")
    return c


@pytest.fixture
def conn(cfg):
    """One climate day of METARs at :23 and :53, warming monotonically, so the
    running max at 13:30 is strictly between 13:00 and 14:00 — the property
    that proves 13.5 is a distinct as-of instant and not a rounded duplicate."""
    c = hdb.connect(cfg.db_path)
    tz = ZoneInfo(cfg.station.tz)
    rows = []
    for h in range(24):
        for minute in (23, 53):
            local = dt.datetime(DAY.year, DAY.month, DAY.day, h, minute, tzinfo=tz)
            temp = 70.0 + h + (0.5 if minute == 53 else 0.0)
            rows.append({
                "station": cfg.station.icao,
                "valid_utc": local.astimezone(dt.timezone.utc).isoformat(),
                "source": "metar", "tmpf": temp, "tmpf_precise": temp,
                "dwpf": temp - 10, "sknt": 8.0, "drct": 120.0, "skyc": "FEW",
                "six_max_c": None, "six_min_c": None,
                "raw": f"METAR {cfg.station.icao} X RMK AO2", "fetched_at": "x"})
    hdb.upsert(c, "obs", rows)
    return c


# ── the mapping itself ───────────────────────────────────────────────────────

@pytest.mark.parametrize("h,expected", [
    (13, (13, 0)), (13.0, (13, 0)), (13.5, (13, 30)),
    (0, (0, 0)), (0.5, (0, 30)), (23.5, (23, 30)),
])
def test_split_hour_maps_to_a_clock_time(h, expected):
    assert _split_hour(h) == expected


@pytest.mark.parametrize("bad", [13.75, 13.25, 13.1, -0.5, 24.0, 24.5])
def test_quarter_hours_and_out_of_range_are_refused(bad):
    """Quarter hours would BUILD happily — `signals` fires every 5 minutes — but
    no gating table is keyed for 13.75 and every one of them fails closed. A
    market that silently never trades is the worst outcome available here, so
    refuse loudly at the door instead."""
    with pytest.raises(ValueError, match="whole or half"):
        _split_hour(bad)


def test_the_frame_rejects_a_bad_hour_before_touching_the_db(cfg):
    """Validation happens before `build_features`, so a caller bug surfaces as
    a ValueError immediately rather than as an empty frame much later."""
    with pytest.raises(ValueError, match="whole or half"):
        build_checkpoint_frame(None, cfg, [DAY.isoformat()], hours=(13.75,),
                               metric=get_metric(cfg, "high"))


# ── the as-of instant is genuinely 30 minutes later ──────────────────────────

def test_half_hour_asof_sees_more_of_the_day_than_the_hour(conn, cfg):
    """What the checkpoint ultimately reads is the running extremum at its
    as-of instant. On a warming day 13:30 must have seen the 13:23 reading
    that 13:00 has not — if any `int(h)` crept into the as-of construction
    these two would be identical and the whole exercise a no-op."""
    tz = ZoneInfo(cfg.station.tz)

    def run_hi_at(h, m):
        asof = dt.datetime(DAY.year, DAY.month, DAY.day, h, m,
                           tzinfo=tz).astimezone(dt.timezone.utc).isoformat()
        return latentmax.visible_running_extremum(conn, cfg, DAY, "max",
                                                  asof_utc=asof)[1]

    at13, at1330, at14 = run_hi_at(13, 0), run_hi_at(13, 30), run_hi_at(14, 0)
    assert at13 < at1330 < at14, (at13, at1330, at14)


def test_whole_hours_are_byte_identical_to_before(conn, cfg):
    """The regression that matters: every shipped caller passes ints, and this
    change must be invisible to them."""
    tz = ZoneInfo(cfg.station.tz)

    def asof(h, m):
        return dt.datetime(DAY.year, DAY.month, DAY.day, h, m,
                           tzinfo=tz).astimezone(dt.timezone.utc).isoformat()

    for h in (12, 13, 14):
        hh, mm = _split_hour(h)
        assert (hh, mm) == (h, 0)
        assert latentmax.visible_running_extremum(
            conn, cfg, DAY, "max", asof_utc=asof(hh, mm)) == \
            latentmax.visible_running_extremum(
                conn, cfg, DAY, "max", asof_utc=asof(h, 0))
