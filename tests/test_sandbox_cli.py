import fcntl
import json
import logging
from contextlib import closing

from sandbox import cli
from sandbox.cli import main
from sandbox.cycle import utc_now_iso
from sandbox.db import connect


def test_init_creates_all_five_accounts(tmp_path, capsys):
    db = tmp_path / "s.db"
    assert main(["--db", str(db), "init"]) == 0
    with closing(connect(db)) as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM accounts ORDER BY name")]
    assert names == ["citycast", "favorite", "highcast", "manual", "mos"]


def test_report_prints_account_and_two_segment_lines(tmp_path, capsys):
    db = tmp_path / "s.db"
    main(["--db", str(db), "init"])
    capsys.readouterr()
    assert main(["--db", str(db), "report"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 15 and all("LOCKED" in ln for ln in lines[::3])
    assert all("pre_day" in ln for ln in lines[1::3])
    assert all("same_day" in ln for ln in lines[2::3])


def test_report_prints_every_unlock_gate(tmp_path, capsys):
    db = tmp_path / "s.db"
    main(["--db", str(db), "init"])
    capsys.readouterr()
    main(["--db", str(db), "report"])
    first = capsys.readouterr().out.splitlines()[0]
    assert first == ("favorite  LOCKED settled 0/100 scored 0/100 events 0/30 "
                     "scored-events 0/30 "
                     "cash $5.00 profit $+0.00 profit_test=fail score_test=fail")


def _cycle_summary(**over):
    return {"status": "ok", "snapshot_id": 1, "markets": 3, "settled": 0, "orders": 0,
            "candidates": 0, "decided": True, "bot_errors": [], "stale_bots": [],
            "confirmation_errors": [], "confirmation_transitions": [], **over}


def test_cycle_prints_json_and_exits_zero_when_clean(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    summary = _cycle_summary()
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: summary)
    assert main(["--db", str(db), "cycle"]) == 0
    assert json.loads(capsys.readouterr().out) == summary


def test_cycle_exits_one_when_a_bot_errored(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: _cycle_summary(bot_errors=["mos"]))
    assert main(["--db", str(db), "cycle"]) == 1
    assert json.loads(capsys.readouterr().out)["bot_errors"] == ["mos"]


def test_cycle_exits_one_when_a_confirmation_errored(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    monkeypatch.setattr(cli, "run_cycle",
                        lambda *a, **k: _cycle_summary(confirmation_errors=["mos"]))
    assert main(["--db", str(db), "cycle"]) == 1
    assert json.loads(capsys.readouterr().out)["confirmation_errors"] == ["mos"]


def test_cycle_exits_one_when_fetch_failed(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: _cycle_summary(
        status="fetch_failed", decided=False, error="boom"))
    assert main(["--db", str(db), "cycle"]) == 1


def test_cycle_passes_force_decide_through(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    seen = {}

    def fake(conn, get, bots, clock, *, force_decide=False):
        seen["force_decide"] = force_decide
        return _cycle_summary()

    monkeypatch.setattr(cli, "run_cycle", fake)
    main(["--db", str(db), "cycle", "--force-decide"])
    assert seen["force_decide"] is True


# ---- F4: the CLI hands run_cycle a clock, not one frozen time ----------------------

def test_cycle_passes_the_clock_function_and_the_bot_order(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    seen = {}

    def fake(conn, get, bots, clock, *, force_decide=False):
        seen.update(clock=clock, bots=[b.name for b in bots])
        return _cycle_summary()

    monkeypatch.setattr(cli, "run_cycle", fake)
    main(["--db", str(db), "cycle"])
    assert seen["clock"] is utc_now_iso
    assert seen["bots"] == ["favorite", "highcast", "citycast", "mos"]


def test_cycle_exits_one_on_a_stale_run(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: _cycle_summary(stale_bots=["mos"]))
    assert main(["--db", str(db), "cycle"]) == 1
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: _cycle_summary(status="stale_snapshot"))
    assert main(["--db", str(db), "cycle"]) == 1


# ---- F10: httpx's per-request INFO lines stay out of the launchd .err log ----------

def test_httpx_logging_is_quieted(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.NOTSET)
    main(["--db", str(db), "init"])
    assert logging.getLogger("httpx").level == logging.WARNING


# ---- F11: one cycle at a time --------------------------------------------------------

def test_cycle_is_locked_out_while_another_cycle_holds_the_lock(tmp_path, capsys, monkeypatch):
    db = tmp_path / "s.db"
    called = []
    monkeypatch.setattr(cli, "run_cycle", lambda *a, **k: called.append(1) or _cycle_summary())
    with open(f"{db}.cycle.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert main(["--db", str(db), "cycle"]) == 0
        assert json.loads(capsys.readouterr().out) == {"status": "locked_out"}
        assert called == []
    assert main(["--db", str(db), "cycle"]) == 0            # released: the next cycle runs
    assert called == [1]


def test_serve_binds_loopback_only_on_8713(tmp_path, monkeypatch):
    import uvicorn
    db = tmp_path / "s.db"
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert main(["--db", str(db), "serve"]) == 0
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8713


# ---- M5: an uninitialised database is explained, not a traceback ---------------------

NOT_INITIALISED = "sandbox DB not initialised — run `sandbox init` first"


def test_report_on_a_missing_database_explains_and_creates_nothing(tmp_path, capsys):
    db = tmp_path / "missing.db"
    assert main(["--db", str(db), "report"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and NOT_INITIALISED in cap.err and "Traceback" not in cap.err
    assert not db.exists() and list(tmp_path.iterdir()) == []


def test_report_on_a_database_without_accounts_explains_and_writes_nothing(tmp_path, capsys):
    import sqlite3
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()                      # a file with no tables at all
    bare = tmp_path / "bare.db"
    with closing(connect(bare)):                        # schema, but no account rows
        pass
    before = {p: p.read_bytes() for p in (empty, bare)}
    for db in (empty, bare):
        assert main(["--db", str(db), "report"]) == 1
        cap = capsys.readouterr()
        assert cap.out == "" and NOT_INITIALISED in cap.err and "Traceback" not in cap.err
    assert {p: p.read_bytes() for p in (empty, bare)} == before


# ---- M6: segment lines read like the account line ------------------------------------

def test_report_segment_lines_use_bracketed_intervals_and_na(tmp_path, capsys):
    from sandbox.cycle import ensure_accounts
    from test_sandbox_scoreboard import _settled_events
    db = tmp_path / "s.db"
    with closing(connect(db)) as conn:
        ensure_accounts(conn)
        _settled_events(conn, 3, .9, True)
        conn.execute("UPDATE orders SET decided_at='2026-10-04T04:59:59+00:00' WHERE id=1")
        conn.commit()
    assert main(["--db", str(db), "report"]) == 0
    lines = capsys.readouterr().out.splitlines()
    mos = lines.index(next(ln for ln in lines if ln.startswith("mos ")))
    assert lines[mos + 1] == ("  pre_day: settled 1 profit $+0.48 profit CI n/a "
                              "Brier diff -0.4800 Brier CI n/a")
    assert lines[mos + 2] == ("  same_day: settled 2 profit $+0.96 profit CI [0.48, 0.48] "
                              "Brier diff -0.4800 Brier CI [-0.4800, -0.4800]")
    favorite = lines.index(next(ln for ln in lines if ln.startswith("favorite")))
    assert lines[favorite + 1] == ("  pre_day: settled 0 profit $+0.00 profit CI n/a "
                                   "Brier diff n/a Brier CI n/a")
    assert not any("(" in ln for ln in lines if ln.startswith("  "))   # no raw tuples
