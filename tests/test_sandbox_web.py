from contextlib import closing

from fastapi.testclient import TestClient
from sandbox_helpers import book_payload, three_buckets

from sandbox.cycle import ensure_accounts, utc_now_iso
from sandbox.db import connect
from sandbox.recorder import record_snapshot
from sandbox.web import create_app

T69 = "KXHIGHNY-26OCT04-T69"


def client(tmp_path, get=None, snapshot=True):
    db = str(tmp_path / "sandbox.db")
    with closing(connect(db)) as conn:
        ensure_accounts(conn)
        if snapshot:
            record_snapshot(conn, three_buckets(), utc_now_iso())
    get = get or (lambda path, params=None: book_payload(no=[(0.80, 5)]))
    return TestClient(create_app(db, get=get)), db


def rows(db, sql):
    with closing(connect(db)) as conn:
        return [tuple(r) for r in conn.execute(sql).fetchall()]


def test_markets_page_is_unbranded_and_lists_buckets(tmp_path):
    c, _ = client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "Weather Desk Sandbox" in r.text and T69 in r.text
    assert "Kalshi" not in r.text


def test_preview_shows_the_fill_without_writing(tmp_path):
    c, db = client(tmp_path)
    r = c.get("/api/preview", params={"ticker": T69, "side": "yes", "contracts": 2, "price_cap": 0.25})
    assert r.json() == {"status": "filled", "contracts": 2, "cost": 0.4, "fee": 0.03,
                        "levels": [[0.2, 2.0]]}
    assert rows(db, "SELECT COUNT(*) FROM orders") == [(0,)]


def test_manual_bet_goes_to_the_manual_account(tmp_path):
    c, db = client(tmp_path)
    r = c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    assert r.json()["status"] == "filled"
    assert rows(db, "SELECT account, filled_contracts FROM orders") == [("manual", 1)]


def test_bad_bet_is_rejected(tmp_path):
    c, _ = client(tmp_path)
    assert c.post("/api/bet", json={"ticker": T69, "side": "maybe", "contracts": 1,
                                    "price_cap": 0.25}).status_code == 400
    assert c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 0,
                                    "price_cap": 0.25}).status_code == 400


def test_scoreboard_and_log_pages(tmp_path):
    c, _ = client(tmp_path)
    c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    s = c.get("/scoreboard")
    assert s.status_code == 200 and "LOCKED" in s.text and "uncalibrated" in s.text
    assert "scored 0/100" in s.text and "events 0/30" in s.text
    assert "scored-events 0/30" in s.text
    lg = c.get("/log")
    assert lg.status_code == 200 and T69 in lg.text


BET = {"ticker": "NOT-A-MARKET", "side": "yes", "contracts": 1, "price_cap": 0.25}


def test_unknown_ticker_is_rejected(tmp_path):
    calls = []

    def get(path, params=None):
        calls.append(path)
        return book_payload(no=[(0.80, 5)])

    c, db = client(tmp_path, get=get)
    r = c.post("/api/bet", json=BET)
    assert r.status_code == 400 and "unknown ticker" in r.text
    r = c.get("/api/preview", params=BET)
    assert r.status_code == 400 and "unknown ticker" in r.text
    assert rows(db, "SELECT COUNT(*) FROM orders") == [(0,)]
    assert calls == []  # rejected before any book was fetched


def test_no_snapshot_rejects_every_ticker(tmp_path):
    c, db = client(tmp_path, snapshot=False)
    assert c.post("/api/bet", json={**BET, "ticker": T69}).status_code == 400
    assert c.get("/api/preview", params={**BET, "ticker": T69}).status_code == 400
    assert rows(db, "SELECT COUNT(*) FROM orders") == [(0,)]


def test_preview_book_failure_is_json_502(tmp_path):
    def get(path, params=None):
        raise OSError("network down")

    c, _ = client(tmp_path, get=get)
    r = c.get("/api/preview", params={**BET, "ticker": T69})
    assert r.status_code == 502
    assert r.json() == {"error": "order book unavailable"}


# ---- M1: GET pages never write -----------------------------------------------------

def test_get_pages_do_not_create_accounts(tmp_path):
    db = str(tmp_path / "sandbox.db")
    with closing(connect(db)) as conn:                 # schema only, no accounts
        record_snapshot(conn, three_buckets(), utc_now_iso())
    c = TestClient(create_app(db, get=lambda path, params=None: book_payload(no=[(0.80, 5)])))
    for page in ("/", "/bet", "/scoreboard", "/log"):
        assert c.get(page).status_code == 200, page
    assert rows(db, "SELECT COUNT(*) FROM accounts") == [(0,)]
    # the POST still opens the accounts it needs
    r = c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    assert r.json()["status"] == "filled"
    assert rows(db, "SELECT COUNT(*) FROM accounts") == [(4,)]


# ---- F12: Confirm is single-use and only for the values that were previewed ------
# (no JS harness here: this pins that the guards are on the page)

def test_bet_page_disables_confirm_on_click_and_on_edit(tmp_path):
    c, _ = client(tmp_path)
    html = c.get("/bet").text
    assert "f.addEventListener('input', invalidate)" in html
    assert "f.addEventListener('change', invalidate)" in html
    click = html[html.index("confirmBtn.onclick"):]
    assert click.index("invalidate()") < click.index("fetch('/api/bet'")


def _confirmation(db, state, marker=0, resolved=None, failed=None):
    import json
    from sandbox.db import set_meta
    stored = {'state': state, 'attempt_start_id': 0, 'marker_id': marker,
              'provisional_at': '2026-10-04T10:00:00+00:00', 'resolved_at': resolved}
    if failed is not None:
        stored['failed_attempts'] = failed
    with closing(connect(db)) as conn:
        set_meta(conn, 'confirm:mos', json.dumps(stored))


def test_scoreboard_persisted_state_overrides_sample(tmp_path):
    from test_sandbox_scoreboard import _settled_events
    c, db = client(tmp_path)
    # Empty lifetime evidence must not revoke a persisted unlock.
    _confirmation(db, 'UNLOCKED')
    assert 'banner good' in c.get('/scoreboard').text
    with closing(connect(db)) as conn:
        _settled_events(conn, 100, .9, True)
    # A passing lifetime sample must not override persisted LOCKED.
    _confirmation(db, 'LOCKED', resolved='2026-10-04T10:00:00+00:00', failed=2)
    html = c.get('/scoreboard').text
    assert 'UNLOCKED' not in html
    assert 'LOCKED — confirmation failed 2026-10-04 18:00 Manila (2 failed attempts)' in html
    assert 'Oct 04 18:00 MNL' not in html
    assert 'profit test ✓' in html


def test_scoreboard_provisional_fresh_marker_counts(tmp_path):
    from test_sandbox_scoreboard import _settled_events
    c, db = client(tmp_path)
    with closing(connect(db)) as conn:
        _settled_events(conn, 4, .9, True)
    _confirmation(db, 'PROVISIONAL', marker=2)
    html = c.get('/scoreboard').text
    assert ('PROVISIONAL (fresh 2/100 settled, 2/30 events, 2/100 scored, 2/30 scored events)'
            in html)
    assert 'banner good' not in html


def test_scoreboard_unlocked_after_failures_says_so(tmp_path):
    c, db = client(tmp_path)
    _confirmation(db, 'UNLOCKED', resolved='2026-10-04T10:00:00+00:00', failed=3)
    html = c.get('/scoreboard').text
    assert 'UNLOCKED (after 3 failed attempts)' in html and 'banner good' in html
    _confirmation(db, 'UNLOCKED', resolved='2026-10-04T10:00:00+00:00', failed=0)
    assert 'UNLOCKED (after' not in c.get('/scoreboard').text


def test_scoreboard_timing_rows_use_segment_evidence(tmp_path):
    from test_sandbox_scoreboard import _settled_events
    c, db = client(tmp_path)
    with closing(connect(db)) as conn:
        _settled_events(conn, 3, .9, True)
        conn.execute("UPDATE orders SET decided_at='2026-10-04T04:59:59+00:00' WHERE id=1")
        conn.commit()
    html = c.get('/scoreboard').text
    assert 'Informational timing split' in html
    assert 'Pre-day</th><td>1</td><td>$0.48<' in html
    assert 'Same-day</th><td>2</td><td>$0.96<' in html
    assert '$0.4800' not in html and '$0.9600' not in html
    assert 'No settled sample' in html and 'Not available' in html


def test_every_get_uses_readonly_without_database_changes(tmp_path, monkeypatch):
    from pathlib import Path
    from sandbox import web
    from sandbox.db import connect_readonly
    c, db = client(tmp_path)
    _confirmation(db, 'PROVISIONAL')
    calls = []
    def readonly(path):
        conn = connect_readonly(path)
        calls.append(path)
        return conn
    def forbidden(path):
        raise AssertionError('GET opened a writable database')
    monkeypatch.setattr(web, 'connect_readonly', readonly, raising=False)
    monkeypatch.setattr(web, 'connect', forbidden)
    before = Path(db).read_bytes()
    for page in ('/', '/bet', '/scoreboard', '/log'):
        assert c.get(page).status_code == 200
    assert c.get('/api/preview', params={**BET, 'ticker': T69}).status_code == 200
    assert len(calls) == 5
    assert Path(db).read_bytes() == before


def test_empty_scoreboard_explains_missing_accounts(tmp_path):
    db = str(tmp_path / 'sandbox.db')
    with closing(connect(db)):
        pass
    c = TestClient(create_app(db))
    assert 'No initialized accounts' in c.get('/scoreboard').text


# ---- M5: an uninitialised database is a 503 that says what to do ----------------------

NOT_INITIALISED = "sandbox DB not initialised — run `sandbox init` first"


def _assert_503_explains(c):
    for page in ('/', '/bet', '/scoreboard', '/log'):
        r = c.get(page)
        assert r.status_code == 503, page
        assert NOT_INITIALISED in r.text and 'Weather Desk Sandbox' in r.text, page
        assert 'Traceback' not in r.text
    r = c.get('/api/preview', params={**BET, 'ticker': T69})
    assert r.status_code == 503 and r.json()['error'] == NOT_INITIALISED


def test_every_get_route_explains_a_missing_database_and_creates_nothing(tmp_path):
    db = tmp_path / 'absent.db'
    _assert_503_explains(TestClient(create_app(str(db))))
    assert not db.exists() and list(tmp_path.iterdir()) == []


def test_every_get_route_explains_a_database_without_the_schema(tmp_path):
    import sqlite3
    db = tmp_path / 'blank.db'
    sqlite3.connect(db).close()
    _assert_503_explains(TestClient(create_app(str(db))))
    assert db.read_bytes() == b'' and list(tmp_path.iterdir()) == [db]


def test_wide_tables_scroll_inside_their_own_box(tmp_path):
    """At phone width the log (11 columns) and market ladders overflowed the page
    (browser check 2026-10-04: log 663px, markets 413px on a 375px screen)."""
    c, _ = client(tmp_path)
    c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    for path in ("/", "/log"):
        html = c.get(path).text
        assert html.count("<table") >= 1
        assert html.count('class="table-scroll"') == html.count("<table"), path
