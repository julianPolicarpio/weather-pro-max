"""The DEMO ACCOUNT panel's endpoint — `GET /api/demo-account`.

Two properties matter more than the numbers (those are already pinned by
`tests/test_paper_account.py`, which is the reconciliation gate):

1. THE DASHBOARD NEVER WRITES. The panel is hit on every page load and by a
   poll timer; `paper_account.refresh` is called with `persist=False` so a
   browser refresh can never race the CLI's own ledger rebuild.

2. THE TWO LENSES STAY SEPARATE. `unit_replay` and `daily_plan` are different
   objects with different meanings (see `paper_account`'s module docstring) —
   the endpoint must hand the UI both, unmerged, so the panel can never
   present one number that is secretly a blend of the two.
"""
import dataclasses

from fastapi.testclient import TestClient

from highcast import db as hdb
from highcast.config import load_config
from highcast.web.app import create_app

from tests.test_paper_account import ARCHIVE_ROWS, _load_archive


def _no_credentials(monkeypatch):
    """The seed is a LIVE Kalshi balance read when credentials exist — and on
    this machine they do, in a file, so clearing the env var is not enough.
    Stubbing `load_credentials` is what keeps the suite off the network and
    off the real account; it exercises the documented no-credentials path
    (`seed_opening_balance` falls back rather than raising)."""
    from highcast import kalshi_client
    monkeypatch.setattr(kalshi_client, "load_credentials",
                        lambda *a, **kw: (None, None))


def _client(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    _load_archive(conn)
    return TestClient(create_app(cfg)), conn


def test_demo_account_returns_both_lenses(tmp_path, monkeypatch):
    _no_credentials(monkeypatch)
    client, _ = _client(tmp_path)
    body = client.get("/api/demo-account").json()
    assert body["seed"]["source"] == "fallback"
    assert body["unit_replay"]["n_graded"] == 20
    assert body["unit_replay"]["wins"] == 14
    # The daily plan is a DIFFERENT lens over the same archive — it replays one
    # market only, so its bet count must not equal the unit replay's.
    assert body["daily_plan"]["n_bets_seen"] != len(ARCHIVE_ROWS)
    assert "curve" in body["unit_replay"] and "curve" in body["daily_plan"]


def test_demo_account_never_writes_to_the_db(tmp_path, monkeypatch):
    """A page load must not persist ledger rows. See `persist=False`."""
    _no_credentials(monkeypatch)
    client, conn = _client(tmp_path)
    client.get("/api/demo-account")
    client.get("/api/demo-account?refresh=1")
    assert conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"] == 22


def test_demo_account_is_cached_between_calls(tmp_path, monkeypatch):
    """The seed is a live Kalshi read; the replay scans the whole archive.
    Neither should run again on every poll tick."""
    _no_credentials(monkeypatch)
    client, _ = _client(tmp_path)
    calls = []
    from highcast.web import app as app_module
    real = app_module.paper_account.refresh

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(app_module.paper_account, "refresh", counted)
    client.get("/api/demo-account")
    client.get("/api/demo-account")
    assert len(calls) == 1
    client.get("/api/demo-account?refresh=1")
    assert len(calls) == 2


def test_index_has_demo_account_panel(tmp_path):
    text = _client(tmp_path)[0].get("/").text
    assert 'id="demo-account"' in text
    assert 'id="demo-balance"' in text
    assert 'id="demo-ledger"' in text
    # The lens switch must be a real tablist, not two styled divs — the panel
    # is keyboard-reachable or it is not accessible.
    assert 'role="tablist"' in text
    assert 'id="demo-lens-daily"' in text
    assert 'id="demo-lens-unit"' in text
