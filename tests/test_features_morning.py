from highcast import db as hdb
from highcast.config import load_config
from highcast.features import FEATURES, build_dayahead


def test_morning_features_list_includes_dtd_and_rate():
    assert "morn_dtd" in FEATURES and "morn_rate" in FEATURES


def _seed(conn, icao):
    hdb.upsert(conn, "cli_reports", [
        {"station": icao, "date": "2026-06-15", "high_f": 75, "low_f": 60,
         "raw": "", "fetched_at": "2026-06-16T00:00:00+00:00"}])

    def ob(ts, t, td):
        return {"station": icao, "valid_utc": ts, "source": "metar",
                "tmpf": t, "tmpf_precise": t, "dwpf": td, "sknt": 5.0,
                "drct": 250.0, "skyc": "BKN", "six_max_c": None, "raw": "",
                "fetched_at": ts}
    # 9am cutoff in June is PDT (UTC-7) -> 16:00 UTC; 3h before -> 13:00 UTC
    hdb.upsert(conn, "obs", [
        ob("2026-06-15T12:50:00+00:00", 62.0, 58.0),   # ~6am PDT (cutoff - 3h)
        ob("2026-06-15T15:50:00+00:00", 68.0, 55.0),   # ~9am PDT (cutoff)
    ])


def test_build_features_computes_dtd_and_rate(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    _seed(conn, cfg.station.icao)
    df = build_dayahead(conn, cfg, ["2026-06-15"])
    row = df[df["target_date"] == "2026-06-15"].iloc[0]
    assert abs(row["morn_tmpf"] - 68.0) < 1e-6
    assert abs(row["morn_dtd"] - 13.0) < 1e-6      # 68 - 55
    assert abs(row["morn_rate"] - 6.0) < 1e-6      # 68 - 62
