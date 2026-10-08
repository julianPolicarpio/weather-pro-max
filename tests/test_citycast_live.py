import json
import math
import pickle
from contextlib import closing
from datetime import date, datetime, timezone

import pytest

from citycast import db, live
from citycast.features import FEATURE_COLS, FeatureContext
from test_citycast_features import NYC, _seed
from test_citycast_walkforward import _frame

TRAINED = date(2023, 12, 30)
NOW = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)


def _choice(c, variant="shared", gate=1, metric="high", station="KNYC"):
    with c:
        c.execute("INSERT OR REPLACE INTO model_choice VALUES (?,?,?,1.0,1.2,-0.1,1,1.0,1.2,90,?,'x')",
                  (station, metric, variant, gate))


class _FakeCtx:
    """Feature rows for any day, with a chosen running extremum, so the pmf logic is tested alone."""
    stations = {"KNYC": NYC}

    def __init__(self, run_ext=float("nan"), metric="high"):
        self.run_ext, self.metric = run_ext, metric

    def row(self, icao, metric, day, t):
        r = {c: 0.0 for c in FEATURE_COLS}
        r.update(station=icao, metric=metric, event_date=day.isoformat(), asof_utc=t.isoformat(),
                 anchor=41.0, run_ext=self.run_ext)
        return r


def test_train_then_predict_writes_live_rows_for_gated_station(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        _choice(c)
        df = _frame()
        path = live.train_final(c, df, "high", ("KLAX", "KNYC"), TRAINED)
        assert path.exists()
        ctx = FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")
        n = live.predict(c, ctx, "high", NOW)
        row = c.execute("SELECT event_date, issued_utc, variant, pmf_json FROM live_predictions").fetchone()
        assert n == 1 and row[0] == "2024-01-05" and row[1] == NOW.isoformat() and row[2] == "shared"
        assert abs(sum(json.loads(row[3]).values()) - 1) < 1e-6


def test_gate_zero_station_gets_no_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c, gate=0)
        live.train_final(c, _frame(), "high", ("KLAX", "KNYC"), TRAINED)
        assert live.predict(c, _FakeCtx(), "high", NOW) == 0


def test_gate_is_reread_at_predict_time(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c)
        live.train_final(c, _frame(), "high", ("KLAX", "KNYC"), TRAINED)
        with c:
            c.execute("UPDATE model_choice SET gate_pass=0")
        assert live.predict(c, _FakeCtx(), "high", NOW) == 0
        _choice(c, gate=1, variant="city")          # variant changed since training: also blocked
        assert live.predict(c, _FakeCtx(), "high", NOW) == 0
        with c:
            c.execute("DELETE FROM model_choice")   # row gone
        assert live.predict(c, _FakeCtx(), "high", NOW) == 0


def test_city_variant_used_when_chosen_and_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c, variant="city")
        live.train_final(c, _frame(), "high", ("KLAX", "KNYC"), TRAINED)
        bundle = pickle.loads((tmp_path / "models" / "high.pkl").read_bytes())
        assert bundle["choice"]["KNYC"][0] == "city" and bundle["choice"]["KNYC"][2] == 1
        assert live.predict(c, _FakeCtx(), "high", NOW) == 2
        assert {r[0] for r in c.execute("SELECT variant FROM live_predictions")} == {"city"}


def test_city_choice_with_too_few_rows_falls_back_ungated(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c, variant="city")
        df = _frame()
        df = df[(df["station"] == "KLAX") | (df["event_date"] < "2022-01-03")]   # KNYC: far below MIN_CITY_ROWS
        live.train_final(c, df, "high", ("KLAX", "KNYC"), TRAINED)
        bundle = pickle.loads((tmp_path / "models" / "high.pkl").read_bytes())
        assert bundle["choice"]["KNYC"][0] == "shared" and bundle["choice"]["KNYC"][2] == 0
        assert live.predict(c, _FakeCtx(), "high", NOW) == 0


def test_low_prediction_has_no_mass_above_rounded_running_min(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c, metric="low")
        df = _frame()
        df["metric"] = "low"
        live.train_final(c, df, "low", ("KLAX", "KNYC"), TRAINED)
        live.predict(c, _FakeCtx(run_ext=41.3), "low", NOW)
        row = c.execute("SELECT pmf_json FROM live_predictions WHERE event_date='2024-01-04'").fetchone()
        pmf = {int(k): v for k, v in json.loads(row[0]).items()}
        assert all(v == 0 for k, v in pmf.items() if k >= 42)
        assert pmf[41] > 0 and math.isclose(sum(pmf.values()), 1.0, abs_tol=1e-6)


def test_both_event_days_written_in_station_local_time(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c)
        live.train_final(c, _frame(), "high", ("KLAX", "KNYC"), TRAINED)
        # 2024-01-05T03:00Z is still 2024-01-04 22:00 in New York
        assert live.predict(c, _FakeCtx(), "high", datetime(2024, 1, 5, 3, 0, tzinfo=timezone.utc)) == 2
        days = sorted(r[0] for r in c.execute("SELECT event_date FROM live_predictions"))
        assert days == ["2024-01-04", "2024-01-05"]


@pytest.mark.parametrize("trained, ok", [(date(2023, 11, 25), True), (date(2023, 11, 24), False)])
def test_stale_bundle_boundary(tmp_path, monkeypatch, trained, ok):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    now = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)   # 2023-11-25 is exactly 40 days earlier
    with closing(db.connect(tmp_path / "h.db")) as c:
        _choice(c)
        live.train_final(c, _frame(), "high", ("KLAX", "KNYC"), trained)
        if ok:
            assert live.predict(c, _FakeCtx(), "high", now) == 2
        else:
            with pytest.raises(RuntimeError, match="retrain"):
                live.predict(c, _FakeCtx(), "high", now)
