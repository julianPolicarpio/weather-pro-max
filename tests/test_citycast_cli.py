import json

from citycast import cli


def test_backfill_dispatches_source_and_exit_code(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run(conn, chunks, **kw):
        seen["n"] = len(chunks)
        return {"done": len(chunks), "skipped": 0, "rows": 0, "errors": []}
    monkeypatch.setattr(cli, "run_chunks", fake_run)
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: [object(), object()])
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli", "--since", "2024-01-01"]) == 0
    assert seen["n"] == 2 and json.loads(capsys.readouterr().out.splitlines()[-1])["done"] == 2


def test_backfill_exit_1_on_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_chunks", lambda *a, **k: {"done": 0, "skipped": 0, "rows": 0,
                                                            "errors": [{"x": 1}]})
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: [])
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "obs"]) == 1


def test_unknown_station_rejected(tmp_path):
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli", "--station", "KXXX"]) == 2


def _spy(monkeypatch):
    seen = {"workers": [], "since": []}

    def fake_run(conn, chunks, **kw):
        seen["workers"].append(kw.get("workers", 1))
        return {"done": 0, "skipped": 0, "rows": 0, "errors": []}

    def fake_candles(paced, conn, now_iso, since=None):
        seen["since"].append(since)
        return []
    monkeypatch.setattr(cli, "run_chunks", fake_run)
    monkeypatch.setattr(cli.kalshi_candles, "chunks", fake_candles)
    monkeypatch.setattr(cli.cli_reports, "chunks", lambda *a, **k: [])
    return seen


def test_candles_run_with_two_workers_and_no_since_means_all(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "kalshi-candles"]) == 0
    assert seen["workers"] == [2] and seen["since"] == [None]


def test_candles_since_passed_when_given(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    cli.main(["--db", str(tmp_path / "h.db"), "backfill", "kalshi-candles", "--since", "2026-09-01"])
    assert seen["since"] == ["2026-09-01"]


def test_other_sources_use_one_worker(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli"])
    assert seen["workers"] == [1]
