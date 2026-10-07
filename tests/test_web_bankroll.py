import shutil

from fastapi.testclient import TestClient

from highcast import db as hdb
from highcast.config import load_config, set_bankroll
from highcast.fetchers import kalshi
from highcast.web.app import create_app


def _client(tmp_path):
    cfg_path = tmp_path / "klax.yaml"
    shutil.copy("config/klax.yaml", cfg_path)
    set_bankroll(cfg_path, 0)  # normalize: tests must not depend on the live bankroll value
    cfg = load_config(cfg_path)
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    hdb.connect(cfg.db_path)  # initialize schema
    return TestClient(create_app(cfg, config_path=str(cfg_path))), cfg_path


def test_get_bankroll_defaults_paper(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/bankroll")
    assert r.status_code == 200
    b = r.json()
    assert b["bankroll"] == 0.0 and b["paper"] is True and b["currency"] == "USD"


def test_post_bankroll_updates_live_and_file(tmp_path):
    client, cfg_path = _client(tmp_path)
    r = client.post("/api/bankroll", json={"bankroll": 250})
    assert r.status_code == 200
    body = r.json()
    assert body["bankroll"] == 250.0 and body["paper"] is False
    # reflected in-process (live app.state swap)
    assert client.get("/api/bankroll").json()["bankroll"] == 250.0
    # and persisted to the file
    assert load_config(cfg_path).staking.bankroll == 250.0


def test_post_bankroll_reflected_in_bet(tmp_path, monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_ranges",
                        lambda cfg, target_date, metric=None: None)
    client, _ = _client(tmp_path)
    client.post("/api/bankroll", json={"bankroll": 500})
    bet = client.get("/api/bet").json()
    assert bet["bankroll"] == 500.0


def test_post_bankroll_rejects_negative(tmp_path):
    client, cfg_path = _client(tmp_path)
    r = client.post("/api/bankroll", json={"bankroll": -5})
    assert r.status_code == 400
    assert client.get("/api/bankroll").json()["bankroll"] == 0.0
    assert load_config(cfg_path).staking.bankroll == 0.0


def test_post_bankroll_zero_returns_paper(tmp_path):
    client, _ = _client(tmp_path)
    client.post("/api/bankroll", json={"bankroll": 100})
    r = client.post("/api/bankroll", json={"bankroll": 0})
    assert r.status_code == 200 and r.json()["paper"] is True


def test_index_has_bankroll_control(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="bankroll-control"' in r.text
    assert 'id="bankroll-input"' in r.text
