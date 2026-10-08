import json
from contextlib import closing
from datetime import date, datetime, timezone

from citycast import db, live
from citycast.features import FeatureContext
from test_citycast_features import NYC, _seed
from test_citycast_walkforward import _frame


def test_train_then_predict_writes_live_rows_for_gated_station(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            c.execute("INSERT INTO model_choice VALUES ('KNYC','high','shared',1.0,1.2,-0.1,1,1.0,1.2,90,1,'x')")
        df = _frame()
        path = live.train_final(c, df, "high", ("KLAX", "KNYC"), date(2023, 1, 1))
        assert path.exists()
        ctx = FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")
        now = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        n = live.predict(c, ctx, "high", now)
        row = c.execute("SELECT event_date, issued_utc, variant, pmf_json FROM live_predictions").fetchone()
        assert n == 1 and row[0] == "2024-01-05" and row[1] == now.isoformat() and row[2] == "shared"
        assert abs(sum(json.loads(row[3]).values()) - 1) < 1e-6
