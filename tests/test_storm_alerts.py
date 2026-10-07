"""Storm alerts are captured, and they must NOT touch the `heat_alert` feature.

WHY THIS EXISTS. On 2026-08-01 a thunderstorm cut KMIA 16F in 50 minutes
(91.04 -> 75.02, +TSRA at 20:43Z) and destroyed the >=81 low. `api.weather.gov`
returns Severe Thunderstorm Warnings and the fetcher was discarding them.

THE TRAP THAT CAME WITH FIXING IT. `features._alert_flag` had NO event filter —
it did not need one, because every row in `alerts` was a heat row. Widening the
fetch without scoping the reader would have made `heat_alert` fire on
thunderstorms: a live model feature on all four markets, changed silently, with
no gate. These tests pin both halves.
"""
import datetime as dt

import yaml

from highcast import db as hdb
from highcast import features
from highcast.config import load_config
from highcast.fetchers import alerts


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _payload(*events):
    return {"features": [{"properties": {
        "event": e, "onset": "2026-08-01T12:00:00+00:00",
        "expires": "2026-08-02T00:00:00+00:00"}} for e in events]}


def test_storm_products_are_now_captured():
    rows = alerts.parse_active(
        _payload("Severe Thunderstorm Warning", "Tornado Warning"),
        "KMIA", "2026-08-01T12:00:00+00:00")
    assert {r["event"] for r in rows} == {"Severe Thunderstorm Warning",
                                          "Tornado Warning"}


def test_heat_products_are_still_captured():
    rows = alerts.parse_active(_payload("Heat Advisory"), "KMIA", "x")
    assert [r["event"] for r in rows] == ["Heat Advisory"]


def test_unrelated_products_are_still_discarded():
    """The filter must stay a filter. Capturing everything NWS emits would put
    Small Craft Advisories and Air Quality Alerts into a table two model
    features read from."""
    rows = alerts.parse_active(
        _payload("Small Craft Advisory", "Air Quality Alert", "Flood Watch"),
        "KMIA", "x")
    assert rows == []


def test_the_tracked_set_is_exactly_heat_plus_storm():
    assert alerts.TRACKED_EVENTS == alerts.HEAT_EVENTS | alerts.STORM_EVENTS
    assert not (alerts.HEAT_EVENTS & alerts.STORM_EVENTS), "sets must be disjoint"


# ── THE FEATURE MUST NOT MOVE ───────────────────────────────────────────────

def _insert(conn, icao, event, onset, expires):
    conn.execute("""INSERT INTO alerts (station,event,onset_utc,expires_utc,fetched_at)
                    VALUES (?,?,?,?,?)""",
                 (icao, event, onset, expires, "2026-08-01T00:00:00+00:00"))
    conn.commit()


def test_a_storm_warning_does_NOT_set_the_heat_alert_feature(tmp_path):
    """THE test. Before the event filter this returned 1.0 and would have
    changed a live model input on every station the moment a storm was warned."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _insert(conn, "KMIA", "Severe Thunderstorm Warning",
            "2026-08-01T12:00:00+00:00", "2026-08-02T00:00:00+00:00")
    cutoff = dt.datetime(2026, 8, 1, 18, 0, tzinfo=dt.timezone.utc)
    assert features._alert_flag(conn, cfg, dt.date(2026, 8, 1), cutoff) == 0.0


def test_a_heat_advisory_still_sets_the_heat_alert_feature(tmp_path):
    """The positive control. Without it the test above would pass against a
    feature that had been broken to always return 0."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _insert(conn, "KMIA", "Heat Advisory",
            "2026-08-01T12:00:00+00:00", "2026-08-02T00:00:00+00:00")
    cutoff = dt.datetime(2026, 8, 1, 18, 0, tzinfo=dt.timezone.utc)
    assert features._alert_flag(conn, cfg, dt.date(2026, 8, 1), cutoff) == 1.0


def test_a_storm_alongside_a_heat_advisory_still_reads_as_heat(tmp_path):
    """Both present: the feature means 'heat', so it stays 1.0 — the storm row
    neither adds nor removes anything."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    for e in ("Heat Advisory", "Severe Thunderstorm Warning"):
        _insert(conn, "KMIA", e, "2026-08-01T12:00:00+00:00",
                "2026-08-02T00:00:00+00:00")
    cutoff = dt.datetime(2026, 8, 1, 18, 0, tzinfo=dt.timezone.utc)
    assert features._alert_flag(conn, cfg, dt.date(2026, 8, 1), cutoff) == 1.0
