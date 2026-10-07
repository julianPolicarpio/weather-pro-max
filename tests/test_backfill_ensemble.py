from highcast import backfill, db as hdb
from highcast.config import load_config
from highcast.fetchers import forecasts


def _rows():
    return [{"station": "KLAX", "target_date": "2026-07-24",
             "run_date": "2026-07-24", "metric": "high", "n_members": 119,
             "ens_mean": 80.1, "ens_std": 2.3, "ens_p10": 77.0, "ens_p50": 80.0,
             "ens_p90": 84.0, "ens_frac_ge_82": 0.4, "ens_bimodality": 1.1,
             "fetched_at": "2026-07-24T16:00:00+00:00"}]


def test_backfill_ensemble_stores(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    monkeypatch.setattr(forecasts, "fetch_openmeteo_ensemble",
                        lambda cfg, past_days=93, forecast_days=1: _rows())
    out = backfill.backfill_ensemble(conn, cfg)
    assert out["ensemble_backfill"] == 1
    assert conn.execute("SELECT COUNT(*) FROM forecast_ens").fetchone()[0] == 1


def test_fetch_snapshot_includes_ensemble(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    # neutralize the network fetchers this test doesn't exercise
    for name in ("fetch_openmeteo", "fetch_nws", "fetch_mos"):
        monkeypatch.setattr(forecasts, name, lambda *a, **k: [])
    monkeypatch.setattr(forecasts, "fetch_openmeteo_ensemble",
                        lambda cfg, past_days=2, forecast_days=2: _rows())
    from highcast.fetchers import metar, fivemin, alerts, cli_reports
    monkeypatch.setattr(metar, "fetch_recent", lambda cfg: [])
    monkeypatch.setattr(fivemin, "fetch_recent", lambda cfg: [])
    monkeypatch.setattr(alerts, "fetch_active", lambda cfg: [])
    monkeypatch.setattr(cli_reports, "fetch_year", lambda cfg, y: [])
    counts = backfill.fetch_snapshot(conn, cfg, progress=lambda *a: None)
    assert counts.get("ensemble") == 1
    assert conn.execute("SELECT COUNT(*) FROM forecast_ens").fetchone()[0] == 1
