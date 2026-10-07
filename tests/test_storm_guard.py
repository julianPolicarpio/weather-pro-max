"""The weather guard: don't trade a market a thunderstorm is about to rewrite.

CAUGHT LIVE 2026-08-01, KMIA — 87.98F at 19:53Z, +TSRA at 20:43Z, 75.02F. The
low fell 16F in 50 minutes and destroyed the >=81 pick. The model said 42% and
the market said 64%; BOTH were wrong, because a temperature model has no
precipitation input and the ladder does not price an outflow boundary.

The right response to a warned storm is not a different number — it is not
trading. The distribution the model emits is conditioned on a world without one.
"""
import datetime as dt

import yaml

from highcast import db as hdb
from highcast import gating
from highcast.config import load_config


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _alert(conn, icao, event, onset, expires):
    conn.execute("""INSERT INTO alerts (station,event,onset_utc,expires_utc,fetched_at)
                    VALUES (?,?,?,?,?)""",
                 (icao, event, onset, expires, "2026-08-01T00:00:00+00:00"))
    conn.commit()


def test_a_severe_thunderstorm_warning_trips_the_guard(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _alert(conn, "KMIA", "Severe Thunderstorm Warning",
           "2026-08-01T20:00:00+00:00", "2026-08-01T22:00:00+00:00")
    assert gating.storm_warning_active(conn, cfg, "2026-08-01") is True


def test_a_heat_advisory_does_NOT_trip_the_storm_guard(tmp_path):
    """The two products share the `alerts` table and must never share a meaning.
    Miami is under a heat advisory constantly; treating that as a storm would
    refuse every day and the guard would be deleted within a week."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _alert(conn, "KMIA", "Heat Advisory",
           "2026-08-01T12:00:00+00:00", "2026-08-02T00:00:00+00:00")
    assert gating.storm_warning_active(conn, cfg, "2026-08-01") is False


def test_a_storm_on_a_DIFFERENT_climate_day_does_not_trip_it(tmp_path):
    """KMIA's climate day is 05:00Z->05:00Z. A warning that expired before it
    opened belongs to yesterday's market."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _alert(conn, "KMIA", "Severe Thunderstorm Warning",
           "2026-07-31T18:00:00+00:00", "2026-07-31T20:00:00+00:00")
    assert gating.storm_warning_active(conn, cfg, "2026-08-01") is False


def test_a_storm_at_another_station_does_not_trip_it(tmp_path):
    """Two stations share one database — anything keyed without the station is
    a latent cross-station bug."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _alert(conn, "KLAX", "Severe Thunderstorm Warning",
           "2026-08-01T20:00:00+00:00", "2026-08-01T22:00:00+00:00")
    assert gating.storm_warning_active(conn, cfg, "2026-08-01") is False


def test_decision_gates_refuses_while_a_storm_is_warned(tmp_path):
    """End to end through the one aggregator every surface reads."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _alert(conn, "KMIA", "Severe Thunderstorm Warning",
           "2026-08-01T20:00:00+00:00", "2026-08-01T22:00:00+00:00")
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-01",
                              hour=14, p_model=0.95)
    assert v.ok is False
    assert "storm_warning" in v.refusals


def test_the_guard_is_quiet_when_no_storm_is_warned(tmp_path):
    """The positive control. Without it, a guard broken to always-refuse would
    pass every test above and silently stop the bot forever."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-01",
                              hour=14, p_model=0.95)
    assert "storm_warning" not in v.refusals
