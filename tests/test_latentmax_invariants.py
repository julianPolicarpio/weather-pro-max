"""Invariants a running extremum cannot violate.

Motivated by a competitor's live failure, not a hypothetical: nowcast.trade's NYC
HIGH time series showed RUNNING HIGH going 78.1 -> 75.0 -> 73.9 -> 78.1 down the
table, and their FAQ says "the high only rises". A running maximum that falls is
either mislabelled or computed over an inconsistent window, and everything graded
downstream of it inherits the error.

Swept over our own archive on 2026-07-30: 0 monotonicity and 0 min-above-max
violations across 2,880 (station, day, as-of) states over 240 sampled days on both
stations. These tests pin that behaviour so a future change to `latentmax` or to
the six-hour-group admissibility rule cannot silently break it.

`scripts/` has no equivalent sweep script; the DB-wide version is a one-liner over
`visible_running_extremum` and is documented in the spec rather than shipped, since
these unit tests cover the logic and the sweep needs the production database.
"""
import datetime as dt

import pytest
import yaml

from highcast import db as hdb
from highcast import latentmax
from highcast.config import load_config

DAY = dt.date(2026, 7, 15)
# KLAX climate day: 08:00Z -> 08:00Z, DST-invariant via std_utc_offset = -8.
START = dt.datetime(2026, 7, 15, 8, 0, tzinfo=dt.timezone.utc)


def _cfg(tmp_path):
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _seed(cfg, spec):
    """spec: list of (hours_from_start, tmpf, six_max_c, six_min_c)."""
    conn = hdb.connect(cfg.db_path)
    rows = []
    for hours, temp, smax, smin in spec:
        v = START + dt.timedelta(hours=hours, minutes=53)
        rows.append({"station": cfg.station.icao, "valid_utc": v.isoformat(),
                     "source": "metar", "tmpf": temp, "tmpf_precise": temp,
                     "dwpf": temp - 12, "sknt": 8.0, "drct": 250.0,
                     "skyc": "FEW", "six_max_c": smax, "six_min_c": smin,
                     "raw": "METAR X RMK AO2", "fetched_at": "x"})
    hdb.upsert(conn, "obs", rows)
    return conn


def _walk(conn, cfg, extremum, upto=24):
    """The running extremum at each hour of the climate day."""
    out = []
    for h in range(1, upto):
        asof = START + dt.timedelta(hours=h, minutes=59)
        iv = latentmax.visible_running_extremum(conn, cfg, DAY, extremum,
                                                asof.isoformat())
        if iv is not None:
            out.append((h, iv))
    return out


def test_running_max_never_falls(tmp_path):
    """The invariant nowcast.trade violated."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(h, 60.0 + h, None, None) for h in range(1, 20)])
    seq = _walk(conn, cfg, "max")
    assert len(seq) > 5
    for (_, a), (_, b) in zip(seq, seq[1:]):
        assert b[0] >= a[0] - 1e-9, "running max lower bound fell"
        assert b[1] >= a[1] - 1e-9, "running max upper bound fell"


def test_running_min_never_rises(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(h, 80.0 - h, None, None) for h in range(1, 20)])
    seq = _walk(conn, cfg, "min")
    assert len(seq) > 5
    for (_, a), (_, b) in zip(seq, seq[1:]):
        assert b[0] <= a[0] + 1e-9, "running min lower bound rose"
        assert b[1] <= a[1] + 1e-9, "running min upper bound rose"


def test_min_interval_never_sits_above_the_max_interval(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(h, 70.0 + (h % 5), None, None) for h in range(1, 20)])
    for h, mn in _walk(conn, cfg, "min"):
        mx = latentmax.visible_running_extremum(
            conn, cfg, DAY, "max",
            (START + dt.timedelta(hours=h, minutes=59)).isoformat())
        assert mn[0] <= mx[1] + 1e-9


def test_a_six_hour_group_may_exceed_every_spot_reading(tmp_path):
    """This is WHY the invariant is non-trivial. The 6-hourly groups come from
    continuous sampling, so a legitimate group can sit outside every hourly
    snapshot -- which is the whole reason `MAX(tmpf_precise)` understates the day
    and the project rules forbid it. Monotonicity must survive that jump.

    Modelled on KMIA 2026-07-28: snapshots peaked at 89.06F while the 17:53Z group
    carried 32.8C = 91.04F, and Kalshi settled 91.
    """
    cfg = _cfg(tmp_path)
    # spot readings all <= 89F; the 00Z group (files at 23:53Z = hour 15 from an
    # 08:00Z start) reports 32.8C = 91.04F over the 18-00Z window.
    spec = [(h, 85.0 + (h % 4), None, None) for h in range(1, 15)]
    spec.append((15, 88.0, 32.8, 20.0))
    conn = _seed(cfg, spec)
    seq = _walk(conn, cfg, "max", upto=20)
    for (_, a), (_, b) in zip(seq, seq[1:]):
        assert b[1] >= a[1] - 1e-9
    # and the group actually moved the bound above every spot reading
    assert seq[-1][1][1] > 89.0


def test_the_straddling_report_is_rejected(tmp_path):
    """KLAX's climate day starts 08:00Z, so the 12Z report's 06-12Z window
    straddles it and must be ignored. Folding it in would import temperatures from
    the PREVIOUS climate day -- the boundary bug §19 of the competitor doc
    suspects in their own series.

    The 12Z report files at 11:53Z, i.e. hour 3.88 from an 08:00Z start.
    """
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(1, 70.0, None, None),
                       # 12Z report at 11:53Z carrying an absurdly low 6h min
                       (3, 70.0, None, -20.0),
                       (5, 71.0, None, None)])
    iv = latentmax.visible_running_extremum(
        conn, cfg, DAY, "min",
        (START + dt.timedelta(hours=6)).isoformat())
    # -20C = -4F. If the straddling group were admitted the bound would collapse.
    assert iv[0] > 60.0, "the 12Z straddling group was wrongly admitted"


def test_extremum_is_none_before_any_observation(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert latentmax.visible_running_extremum(
        conn, cfg, DAY, "min", START.isoformat()) is None


@pytest.mark.parametrize("extremum", ["min", "max"])
def test_bounds_are_ordered_within_the_interval(tmp_path, extremum):
    """lo <= hi always -- the interval is a range on the true value, and an
    inverted one would break every consumer silently."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(h, 70.0 + (h % 6), 25.0, 18.0) for h in range(1, 18)])
    for _, iv in _walk(conn, cfg, extremum):
        assert iv[0] <= iv[1] + 1e-9
