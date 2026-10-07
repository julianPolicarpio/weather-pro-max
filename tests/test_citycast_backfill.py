# tests/test_citycast_backfill.py
from contextlib import closing

from citycast import db
from citycast.backfill import Chunk, run_chunks

NOW = lambda: "2026-10-07T00:00:00+00:00"   # noqa: E731


def _row(day, high):
    return {"station": "KNYC", "date": day, "high_f": high, "low_f": 1, "final": 1,
            "issued_utc": None, "raw": "{}", "fetched_at": NOW()}


def test_runs_logs_and_skips_done_chunks(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return [("cli_daily", [_row("2024-01-01", 40)])]
    with closing(db.connect(tmp_path / "h.db")) as c:
        s1 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], now_iso=NOW, progress=lambda m: None)
        s2 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], now_iso=NOW, progress=lambda m: None)
        s3 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], refresh=True, now_iso=NOW,
                        progress=lambda m: None)
        assert (s1["done"], s2["skipped"], s3["done"], len(calls)) == (1, 1, 1, 2)
        assert c.execute("SELECT rows FROM backfill_log").fetchone()[0] == 1


def test_current_chunk_always_runs_and_is_never_logged(tmp_path):
    run = lambda: [("cli_daily", [_row("2026-10-01", 70)])]   # noqa: E731
    with closing(db.connect(tmp_path / "h.db")) as c:
        for _ in range(2):
            s = run_chunks(c, [Chunk("cli", "KNYC:2026", True, run)], now_iso=NOW, progress=lambda m: None)
            assert s["done"] == 1
        assert c.execute("SELECT COUNT(*) FROM backfill_log").fetchone()[0] == 0


def test_error_is_reported_not_logged_and_does_not_stop_others(tmp_path):
    def bad():
        raise RuntimeError("boom")
    good = lambda: [("cli_daily", [_row("2024-01-02", 41)])]   # noqa: E731
    with closing(db.connect(tmp_path / "h.db")) as c:
        s = run_chunks(c, [Chunk("cli", "A", False, bad), Chunk("cli", "B", False, good)],
                       now_iso=NOW, progress=lambda m: None)
        assert s["done"] == 1 and s["errors"] == [
            {"source": "cli", "key": "A", "error": "RuntimeError: boom"}]
        assert [r[0] for r in c.execute("SELECT key FROM backfill_log")] == ["B"]


def test_empty_result_is_not_logged_so_it_is_retried(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        s = run_chunks(c, [Chunk("cli", "EMPTY", False, lambda: [("cli_daily", [])])],
                       now_iso=NOW, progress=lambda m: None)
        assert s["done"] == 1
        assert c.execute("SELECT COUNT(*) FROM backfill_log").fetchone()[0] == 0


def test_write_error_fails_only_that_chunk_and_rolls_back(tmp_path):
    bad_rows = lambda: [("cli_daily", [_row("2024-01-03", 1)]), ("no_such_table", [{"x": 1}])]  # noqa: E731
    good = lambda: [("cli_daily", [_row("2024-01-04", 2)])]   # noqa: E731
    with closing(db.connect(tmp_path / "h.db")) as c:
        s = run_chunks(c, [Chunk("cli", "BAD", False, bad_rows), Chunk("cli", "OK", False, good)],
                       now_iso=NOW, progress=lambda m: None)
        assert s["done"] == 1 and s["errors"][0]["key"] == "BAD"
        assert [r[0] for r in c.execute("SELECT date FROM cli_daily")] == ["2024-01-04"]


def _mk_chunks(n, sleep_s=0.0, bad=()):
    import time as _t

    def mk(i):
        def run():
            _t.sleep(sleep_s)
            if i in bad:
                raise RuntimeError("boom")
            return [("cli_daily", [_row(f"2024-02-{i + 1:02d}", i)])]
        return Chunk("cli", f"K{i}", False, run)
    return [mk(i) for i in range(n)]


def test_workers_match_sequential_and_upsert_on_main_thread(tmp_path, monkeypatch):
    import threading
    main = threading.get_ident()
    threads = []
    real = db.upsert

    def spy(conn, table, rows):
        threads.append(threading.get_ident())
        return real(conn, table, rows)
    monkeypatch.setattr(db, "upsert", spy)
    out = {}
    for w in (1, 2):
        with closing(db.connect(tmp_path / f"h{w}.db")) as c:
            s = run_chunks(c, _mk_chunks(20, 0.05, bad=(3,)), now_iso=NOW, progress=lambda m: None,
                           workers=w)
            out[w] = (s, c.execute("SELECT date, high_f FROM cli_daily ORDER BY date").fetchall(),
                      c.execute("SELECT key FROM backfill_log ORDER BY key").fetchall())
    assert out[1][0] == out[2][0]
    assert out[2][0]["done"] == 19 and [e["key"] for e in out[2][0]["errors"]] == ["K3"]
    assert [tuple(r) for r in out[1][1]] == [tuple(r) for r in out[2][1]]
    assert [tuple(r) for r in out[1][2]] == [tuple(r) for r in out[2][2]]
    assert "K3" not in [r[0] for r in out[2][2]]
    assert set(threads) == {main}


def test_workers_skip_logged_chunks(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        run_chunks(c, _mk_chunks(6), now_iso=NOW, progress=lambda m: None, workers=2)
        s = run_chunks(c, _mk_chunks(6), now_iso=NOW, progress=lambda m: None, workers=2)
        assert (s["done"], s["skipped"]) == (0, 6)
