import json
from contextlib import closing
from pathlib import Path

import pytest

from citycast import cli, db
from test_citycast_evaluate import _mark_runs
from test_citycast_evaluate import _seed as _seed_eval


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


def test_candles_run_with_candle_workers_and_no_since_means_all(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "kalshi-candles"]) == 0
    assert seen["workers"] == [cli.CANDLE_WORKERS] == [3] and seen["since"] == [None]


def test_candles_since_passed_when_given(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    cli.main(["--db", str(tmp_path / "h.db"), "backfill", "kalshi-candles", "--since", "2026-09-01"])
    assert seen["since"] == ["2026-09-01"]


def test_other_sources_use_one_worker(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli"])
    assert seen["workers"] == [1]


def test_backfill_passes_utc_today_and_two_day_margin(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    seen = {}

    def fake_chunks(source, paced, conn, stations, since, today, **kw):
        seen.update(today=today, complete_before=kw.get("complete_before"))
        return []
    monkeypatch.setattr(cli, "_chunks_for", fake_chunks)
    monkeypatch.setattr(cli, "run_chunks", lambda *a, **k: {"done": 0, "skipped": 0, "rows": 0,
                                                            "errors": []})
    cli.main(["--db", str(tmp_path / "h.db"), "backfill", "obs"])
    assert seen["today"] == datetime.now(timezone.utc).date()
    assert seen["complete_before"] == seen["today"] - timedelta(days=cli.COMPLETE_MARGIN_D) \
        and cli.COMPLETE_MARGIN_D == 2


def _stub_refresh(monkeypatch, errors):
    monkeypatch.setattr(cli, "Paced", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: [])
    monkeypatch.setattr(cli, "run_chunks", lambda *a, **k: {"done": 0, "skipped": 0, "rows": 0,
                                                            "errors": errors})


def test_predict_blocked_by_refresh_errors(tmp_path, monkeypatch, capsys):
    _stub_refresh(monkeypatch, [{"source": "obs", "key": "k", "error": "boom"}])
    monkeypatch.setattr(cli.live, "predict", lambda *a, **k: pytest.fail("must not predict on stale data"))
    dbp = tmp_path / "h.db"
    assert cli.main(["--db", str(dbp), "predict"]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["refresh_errors"] and out["refresh_errors"][0]["error"] == "boom"
    with closing(db.connect(dbp)) as c:
        assert c.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0] == 0


def test_predict_proceeds_when_refresh_clean(tmp_path, monkeypatch):
    _stub_refresh(monkeypatch, [])
    monkeypatch.setattr(cli.live, "predict", lambda *a, **k: 0)
    assert cli.main(["--db", str(tmp_path / "h.db"), "predict"]) == 0


def test_evaluate_writes_report_to_patched_dir(tmp_path, monkeypatch, capsys):
    dbp = tmp_path / "h.db"
    with closing(db.connect(dbp)) as c:
        _seed_eval(c, good="city")
        _mark_runs(c)
    monkeypatch.setattr(cli, "REPORT_DIR", tmp_path / "reports")
    assert cli.main(["--db", str(dbp), "evaluate"]) == 0
    path = Path(json.loads(capsys.readouterr().out.splitlines()[-1])["report"])
    assert path.parent == tmp_path / "reports" and "KNYC" in path.read_text()
    with closing(db.connect(dbp)) as c:
        assert tuple(c.execute("SELECT variant, gate_pass FROM model_choice WHERE metric='high'").fetchone()) \
            == ("city", 1)
