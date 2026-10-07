from datetime import datetime, timedelta, timezone

from highcast import backfill, db
from highcast.config import load_config
from highcast.fetchers import alerts, cli_reports, fivemin, forecasts, metar


def test_gap_start_none_when_fresh():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    last = (now - timedelta(minutes=30)).isoformat()
    assert backfill.gap_start(last, now, max_gap_hours=2) is None


def test_gap_start_returns_last_when_stale():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    last = (now - timedelta(hours=30)).isoformat()
    start = backfill.gap_start(last, now, max_gap_hours=2)
    assert start is not None and start.date() == (now - timedelta(hours=30)).date()


def test_gap_start_handles_empty_db():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    start = backfill.gap_start(None, now, max_gap_hours=2)
    assert start == (now - timedelta(days=7))   # bootstrap: last week


def test_reparse_obs_backfills_six_min_and_is_idempotent(tmp_path):
    """The stored raw METARs predate the 2sTTT parser, so their six_min_c is
    NULL. reparse_obs must re-derive it in place (no duplicate rows) and
    report zero changes on a second run."""
    cfg = load_config()
    conn = db.connect(tmp_path / "t.db")
    raw = ("METAR KLAX 251153Z 12005KT 10SM FEW250 21/19 A2986 "
           "RMK AO2 SLP109 T02110189 10228 20211 58010")
    # store the row as the OLD parser would have: six_min_c missing
    db.upsert(conn, "obs", [{
        "station": cfg.station.icao, "valid_utc": "2026-07-25T11:53:00+00:00",
        "source": "metar", "tmpf": 69.8, "tmpf_precise": 69.98, "dwpf": 66.2,
        "sknt": 5.0, "drct": 120.0, "skyc": "FEW", "six_max_c": 22.8,
        "six_min_c": None, "raw": raw, "fetched_at": "x"}])

    first = backfill.reparse_obs(conn, cfg, progress=lambda *_: None)
    assert first["changed"] == 1
    row = conn.execute("SELECT six_min_c FROM obs").fetchone()
    assert row["six_min_c"] == 21.1
    assert conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0] == 1   # updated, not inserted

    second = backfill.reparse_obs(conn, cfg, progress=lambda *_: None)
    assert second["changed"] == 0
    assert second["scanned"] == first["scanned"] == 1


def test_fetch_snapshot_isolates_a_failing_source(tmp_path, monkeypatch):
    """A transient failure in one source must not abort the whole cycle."""
    cfg = load_config()
    conn = db.connect(tmp_path / "t.db")

    # Every network fetch is stubbed to a no-op except nws, which raises the
    # same DNS error seen in the launchd log.
    monkeypatch.setattr(metar, "fetch_recent", lambda c: [])
    monkeypatch.setattr(fivemin, "fetch_recent", lambda c: [])
    monkeypatch.setattr(forecasts, "fetch_openmeteo", lambda c, m=None: [])
    monkeypatch.setattr(forecasts, "fetch_mos", lambda c, metric=None: [])
    monkeypatch.setattr(alerts, "fetch_active", lambda c: [])
    monkeypatch.setattr(cli_reports, "fetch_year", lambda c, y: [])

    def boom(*a, **k):
        raise ConnectionError("nodename nor servname provided, or not known")

    monkeypatch.setattr(forecasts, "fetch_nws", boom)

    counts = backfill.fetch_snapshot(conn, cfg, progress=lambda *_: None)

    # The failure was captured, not raised (nws fetched per metric now)...
    assert "nws_high" in counts["_errors"] and "nws_low" in counts["_errors"]
    assert counts["nws_high"] == 0 and counts["nws_low"] == 0
    # ...and every other source (both metrics for openmeteo) still completed.
    # mos is per-metric now too: the LOW gained its own MOS channel on
    # 2026-07-26 (the min was being discarded by the parser).
    for src in ("metar", "syn5", "openmeteo_high", "openmeteo_low",
                "mos_high", "mos_low", "alerts", "cli"):
        assert src in counts
