"""Warming-rate candidate features for the KLAX HIGH nowcast.

Pre-registered 2026-07-31. These live in the FRAME only; no shipped model reads
them unless an experiment names them via `IntradayModel(extra_features=[...])`.

The load-bearing test is `test_missing_history_is_nan_not_zero`: on a feature
whose entire purpose is telling fast warming from slow, substituting 0.0 for a
missing lookback would inject "no warming" as a real signal on exactly the gappy
days.
"""
import datetime as dt

import pytest
import yaml

from highcast import db as hdb
from highcast.config import load_config
from highcast.models.intraday import _rate_features

START = dt.datetime(2026, 7, 15, 8, 0, tzinfo=dt.timezone.utc)  # KLAX day start


def _cfg(tmp_path):
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _seed(cfg, offsets_and_temps):
    conn = hdb.connect(cfg.db_path)
    rows = []
    for hours, temp in offsets_and_temps:
        v = START + dt.timedelta(hours=hours)
        rows.append({"station": cfg.station.icao, "valid_utc": v.isoformat(),
                     "source": "metar", "tmpf": temp, "tmpf_precise": temp,
                     "dwpf": temp - 12, "sknt": 8.0, "drct": 250.0,
                     "skyc": "FEW", "six_max_c": None, "six_min_c": None,
                     "raw": "METAR X RMK AO2", "fetched_at": "x"})
    hdb.upsert(conn, "obs", rows)
    return conn


def test_rate_measures_the_actual_warming(tmp_path):
    cfg = _cfg(tmp_path)
    # 60F at the day start, climbing 2F/hour to 70F at hour 5
    conn = _seed(cfg, [(h, 60.0 + 2.0 * h) for h in range(6)])
    asof = (START + dt.timedelta(hours=5)).isoformat()
    got = _rate_features(conn, cfg, asof, 70.0)
    assert got["cur_rate_3h"] == pytest.approx(6.0)   # 70 - 64
    assert got["cur_rate_1h"] == pytest.approx(2.0)   # 70 - 68
    assert got["cur_rise_from_min"] == pytest.approx(10.0)  # 70 - 60


def test_missing_history_is_nan_not_zero(tmp_path):
    """A gap must read UNKNOWN, never 'no warming'. Zero is a real state that
    means something else entirely, and this feature exists to tell them apart."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(5, 70.0)])          # only ONE observation, no history
    asof = (START + dt.timedelta(hours=5)).isoformat()
    got = _rate_features(conn, cfg, asof, 70.0)
    assert got["cur_rate_3h"] != got["cur_rate_3h"], "3h gap became a number"
    assert got["cur_rate_1h"] != got["cur_rate_1h"], "1h gap became a number"
    # rise_from_min IS defined: the single reading is itself the day's min so far
    assert got["cur_rise_from_min"] == pytest.approx(0.0)


def test_cooling_gives_a_negative_rate(tmp_path):
    """The range measured on real data runs -6.1 to +22.0; the sign must survive."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(h, 80.0 - 2.0 * h) for h in range(6)])
    asof = (START + dt.timedelta(hours=5)).isoformat()
    got = _rate_features(conn, cfg, asof, 70.0)
    assert got["cur_rate_3h"] == pytest.approx(-6.0)
    assert got["cur_rise_from_min"] == pytest.approx(0.0)  # now IS the min


def test_rise_from_min_uses_the_climate_day_not_a_lookback(tmp_path):
    """The floor is the CLIMATE DAY's minimum. A fixed lookback would miss a dawn
    minimum that fell outside the window -- the same clipping error that
    under-reported a stall by five hours earlier in this project."""
    cfg = _cfg(tmp_path)
    # min of 55F right at the day start, then warm all day
    conn = _seed(cfg, [(0, 55.0)] + [(h, 75.0) for h in range(1, 12)])
    asof = (START + dt.timedelta(hours=11)).isoformat()
    got = _rate_features(conn, cfg, asof, 75.0)
    assert got["cur_rise_from_min"] == pytest.approx(20.0)


def test_no_current_temperature_yields_all_nan(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, [(1, 60.0)])
    got = _rate_features(conn, cfg, (START + dt.timedelta(hours=2)).isoformat(),
                         float("nan"))
    assert all(v != v for v in got.values())


def test_rate_features_are_not_in_any_shipped_list():
    """They are candidates. Until the pre-registered gate passes they must reach
    no model."""
    from highcast.features import FEATURES, model_features
    from highcast.models.intraday import INTRADAY_FEATURES
    for lst in (INTRADAY_FEATURES, FEATURES, model_features("high"),
                model_features("low")):
        assert not [c for c in lst if "rate_3h" in c or "rate_1h" in c
                    or "rise_from_min" in c]
