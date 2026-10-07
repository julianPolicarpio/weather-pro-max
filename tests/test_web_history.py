import json

from fastapi.testclient import TestClient

from highcast import db as hdb
from highcast.config import load_config
from highcast.web.app import create_app


def _client(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    return TestClient(create_app(cfg)), conn, cfg.station.icao


def _pred(icao, date, issued, point, metric):
    return {"station": icao, "target_date": date, "issued_at_utc": issued,
            "kind": "dayahead", "point_f": point,
            "quantiles": json.dumps({"0.5": point}),
            "buckets": json.dumps({str(int(point)): 1.0}), "metric": metric}


def test_history_one_high_prediction_per_date(tmp_path):
    client, conn, icao = _client(tmp_path)
    hdb.upsert(conn, "cli_reports", [
        {"station": icao, "date": "2026-06-01", "high_f": 80, "low_f": 60,
         "raw": "", "fetched_at": "2026-06-02T00:00:00+00:00"},
        {"station": icao, "date": "2026-06-02", "high_f": 82, "low_f": 61,
         "raw": "", "fetched_at": "2026-06-03T00:00:00+00:00"},
    ])
    hdb.upsert(conn, "predictions", [
        _pred(icao, "2026-06-01", "2026-06-01T00:00:00+00:00", 76.0, "high"),  # earlier high
        _pred(icao, "2026-06-01", "2026-06-01T16:00:00+00:00", 81.0, "high"),  # latest high -> chosen
        _pred(icao, "2026-06-01", "2026-06-01T04:00:00+00:00", 68.0, "low"),   # low -> excluded
        _pred(icao, "2026-06-02", "2026-06-02T04:00:00+00:00", 61.0, "low"),   # 06-02: only a low
    ])
    body = client.get("/api/history").json()
    by_date = {d["date"]: d for d in body["days"]}
    # latest HIGH dayahead chosen; the earlier high and the low are excluded
    assert by_date["2026-06-01"]["predicted"] == 81.0
    # a date with only a LOW dayahead shows no prediction (not the low value)
    assert by_date["2026-06-02"]["predicted"] is None
    # exactly one row per date (no duplicate join rows)
    dates = [d["date"] for d in body["days"]]
    assert len(dates) == len(set(dates))
    # MAE is over the single correct high prediction only: |81 - 80| = 1.0
    assert abs(body["mae"] - 1.0) < 1e-9
