from highcast import db as hdb
from highcast.fetchers import forecasts


def test_forecast_ens_table_exists_and_store_roundtrips(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    # table exists after connect (CREATE TABLE IF NOT EXISTS in SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(forecast_ens)")}
    assert {"station", "target_date", "run_date", "metric", "n_members",
            "ens_std", "ens_p10", "ens_p90", "ens_frac_ge_82",
            "ens_bimodality", "fetched_at"} <= cols
    n = forecasts.store_ens(conn, [{
        "station": "KLAX", "target_date": "2026-07-24", "run_date": "2026-07-24",
        "metric": "high", "n_members": 119, "ens_mean": 80.1, "ens_std": 2.3,
        "ens_p10": 77.0, "ens_p50": 80.0, "ens_p90": 84.0,
        "ens_frac_ge_82": 0.4, "ens_bimodality": 1.1,
        "fetched_at": "2026-07-24T16:00:00+00:00"}])
    assert n == 1
    row = conn.execute("SELECT ens_std, ens_frac_ge_82 FROM forecast_ens").fetchone()
    assert row["ens_std"] == 2.3 and row["ens_frac_ge_82"] == 0.4
