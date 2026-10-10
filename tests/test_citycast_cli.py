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


def _fake_refresh(monkeypatch, fail_counts, stations=("KSEA", "KDFW")):
    """One chunk per station; each fails `fail_counts[icao]` times, then succeeds. Returns the call log."""
    from citycast.backfill import Chunk
    calls = []

    def make(icao):
        def run():
            calls.append(icao)
            if calls.count(icao) <= fail_counts.get(icao, 0):
                raise TimeoutError("iem timeout")
            return []
        return Chunk("cli", f"{icao}:recent", True, run)
    chunks = [make(i) for i in stations]
    monkeypatch.setattr(cli, "RETRY_PAUSE_S", 0)
    real = cli.load_stations()
    monkeypatch.setattr(cli, "load_stations", lambda: {i: real[i] for i in stations})
    monkeypatch.setattr(cli, "Paced", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: chunks)
    monkeypatch.setattr(cli.mos, "recent_chunks", lambda *a, **k: [])
    monkeypatch.setattr(cli.obs, "recent_chunks", lambda *a, **k: [])
    monkeypatch.setattr(cli, "run_chunks", _fake_run_chunks)
    return calls


def _fake_run_chunks(conn, chunks, **kw):
    out = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
    for ch in chunks:
        try:
            ch.run()
            out["done"] += 1
        except Exception as e:  # noqa: BLE001
            out["errors"].append({"source": ch.source, "key": ch.key, "error": str(e)})
    return out


def test_predict_retries_failed_chunk_once_and_does_not_skip(tmp_path, monkeypatch, capsys):
    calls = _fake_refresh(monkeypatch, {"KSEA": 1})
    seen = []
    monkeypatch.setattr(cli.live, "predict", lambda conn, ctx, m, now, skip=frozenset(): seen.append(skip) or 1)
    assert cli.main(["--db", str(tmp_path / "h.db"), "predict"]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert calls == ["KSEA", "KDFW", "KSEA"]     # only the failed chunk is re-run
    assert out["skipped_stations"] == [] and out["refresh_errors"] == [] and out["high"] == out["low"] == 1
    assert seen == [frozenset(), frozenset()]


def test_predict_station_failing_twice_is_skipped_exit_3(tmp_path, monkeypatch, capsys):
    calls = _fake_refresh(monkeypatch, {"KSEA": 2})
    seen = []
    monkeypatch.setattr(cli.live, "predict", lambda conn, ctx, m, now, skip=frozenset(): seen.append(skip) or 1)
    assert cli.main(["--db", str(tmp_path / "h.db"), "predict"]) == 3
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert calls.count("KSEA") == 2 and calls.count("KDFW") == 1
    assert out["skipped_stations"] == ["KSEA"]
    assert [e["key"] for e in out["refresh_errors"]] == ["KSEA:recent"]
    assert seen == [frozenset({"KSEA"})] * 2


def test_predict_all_stations_failing_exit_1_writes_nothing(tmp_path, monkeypatch, capsys):
    _fake_refresh(monkeypatch, {"KSEA": 9, "KDFW": 9})
    monkeypatch.setattr(cli.live, "predict", lambda *a, **k: pytest.fail("nothing may be predicted"))
    dbp = tmp_path / "h.db"
    assert cli.main(["--db", str(dbp), "predict"]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["skipped_stations"] == ["KDFW", "KSEA"] and len(out["refresh_errors"]) == 2
    with closing(db.connect(dbp)) as c:
        assert c.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0] == 0


def test_predict_proceeds_when_refresh_clean(tmp_path, monkeypatch):
    _stub_refresh(monkeypatch, [])
    monkeypatch.setattr(cli.live, "predict", lambda *a, **k: 0)
    assert cli.main(["--db", str(tmp_path / "h.db"), "predict"]) == 0


def test_predict_refresh_passes_utc_today_two_day_margin_and_recent_chunks(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    seen = {"sources": []}

    def fake_chunks(source, paced, conn, stations, since, today, **kw):
        seen["sources"].append(source)
        seen.update(today=today, complete_before=kw.get("complete_before"))
        return []

    def fake_recent(name):
        def recent(paced, stations, now, now_iso, **kw):
            seen[name] = now
            return []
        return recent
    monkeypatch.setattr(cli, "Paced", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_chunks_for", fake_chunks)
    monkeypatch.setattr(cli.mos, "recent_chunks", fake_recent("mos_now"))
    monkeypatch.setattr(cli.obs, "recent_chunks", fake_recent("obs_now"))
    monkeypatch.setattr(cli, "run_chunks", lambda *a, **k: {"done": 0, "skipped": 0, "rows": 0, "errors": []})
    monkeypatch.setattr(cli.live, "predict", lambda *a, **k: 0)
    assert cli.main(["--db", str(tmp_path / "h.db"), "predict"]) == 0
    today = datetime.now(timezone.utc).date()
    assert seen["sources"] == ["cli"]               # MOS and obs refresh through never-logged recent chunks
    assert seen["today"] == today
    assert seen["complete_before"] == today - timedelta(days=cli.COMPLETE_MARGIN_D)
    for name in ("mos_now", "obs_now"):
        assert seen[name].utcoffset() == timedelta(0)
        assert abs(datetime.now(timezone.utc) - seen[name]) < timedelta(minutes=1)


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


# ---- plan 3: decisions / gate commands ----

def _seed_rows(c, n=60):
    rows = []
    for split_date in ("2026-02-10", "2026-05-10", "2026-08-10"):
        for i in range(n):
            won = int(i % 3 == 0)
            rows.append({"ticker": f"{split_date}-{i}", "asof_utc": f"{split_date}T12:00:00+00:00",
                         "event_ticker": "E", "station": "KNYC", "metric": "high",
                         "event_date": f"{split_date[:8]}{1 + i % 28:02d}", "lo": 1, "hi": 2,
                         "p_model": 0.5, "model_median": 1, "model_iqr": 1, "ask": 0.30, "bid": 0.28,
                         "mid": 0.29, "spread": 0.02, "market_p": 0.29, "fee": 0.02, "edge": 0.18,
                         "hours_to_close": 10, "lead_h": 10, "run_gap_to_lo": None, "run_gap_to_hi": None,
                         "month": 5, "won": won, "profit_if_bet": won - 0.32})
    with c:
        db.upsert(c, "decision_rows", rows)


def _db(tmp_path, n_rows=60, **meta_rows):
    dbp = tmp_path / "h.db"
    with closing(db.connect(dbp)) as c:
        _seed_rows(c, n_rows)
        with c:
            for k, v in meta_rows.items():
                c.execute("INSERT INTO meta VALUES (?, ?)", (k, v))
    return dbp


def _last(capsys):
    return json.loads(capsys.readouterr().out.splitlines()[-1])


def test_decisions_build_reports_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.decisions, "build", lambda conn, stations: 7)
    assert cli.main(["--db", str(tmp_path / "h.db"), "decisions", "build"]) == 0
    assert _last(capsys) == {"rows": 7}


def test_decisions_export_writes_train_and_validation_never_test(tmp_path, capsys):
    dbp = _db(tmp_path)
    out = tmp_path / "laya"
    assert cli.main(["--db", str(dbp), "decisions", "export", "--out", str(out)]) == 0
    res = _last(capsys)
    train = (out / "train.jsonl").read_text().splitlines()
    val = (out / "validation.jsonl").read_text().splitlines()
    assert len(train) == res["train"] > 0 and len(val) == res["validation"] == 60
    assert sorted(p.name for p in out.iterdir()) == ["train.jsonl", "validation.jsonl"]


class _FakeB2:
    def __init__(self, cats):
        self.cats = cats

    def fit(self, df):
        return self

    def predict(self, df):
        return [0.4] * len(df)


def test_gate_score_b2_stores_train_and_validation_never_test(tmp_path, monkeypatch, capsys):
    dbp = _db(tmp_path)
    monkeypatch.setattr(cli.meta, "MetaLabeler", _FakeB2)
    saved = []
    monkeypatch.setattr(cli.meta, "save", lambda m, path: saved.append(path))
    monkeypatch.setattr(cli, "B2_PATH", tmp_path / "gate-b2.pkl")
    assert cli.main(["--db", str(dbp), "gate", "score", "--contender", "B2"]) == 0
    assert _last(capsys) == {"scored": "B2"} and saved == [tmp_path / "gate-b2.pkl"]
    with closing(db.connect(dbp)) as c:
        dates = [r[0] for r in c.execute("SELECT DISTINCT substr(ticker, 1, 7) FROM gate_scores "
                                         "WHERE contender='B2' ORDER BY 1")]
    assert dates == ["2026-02", "2026-05"]


REF = "me/laya@abcdef1"
REF_B = "me/laya@1234567"


@pytest.mark.parametrize("contender,ref,expected,key", [("L0", [], None, "L0"), ("L1", ["--model-ref", REF], REF, f"L1@{REF}")])
def test_gate_score_laya_scores_validation_only(tmp_path, monkeypatch, capsys, contender, ref, expected, key):
    dbp = _db(tmp_path)
    seen = {}

    def fake_router(model_ref, device=None):
        seen["ref"] = model_ref
        return lambda states: [0.4] * len(states)
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", fake_router)
    assert cli.main(["--db", str(dbp), "gate", "score", "--contender", contender, *ref]) == 0
    assert seen["ref"] == expected
    with closing(db.connect(dbp)) as c:
        n, dates = c.execute("SELECT COUNT(*), MIN(substr(ticker, 1, 7)) FROM gate_scores "
                             "WHERE contender=?", (key,)).fetchone()
        assert (n, dates) == (60, "2026-05")
        assert c.execute("SELECT COUNT(*) FROM gate_scores WHERE contender!=?", (key,)).fetchone()[0] == 0


def test_gate_score_l1_is_cached_per_model_ref_not_reused_across_refs(tmp_path, monkeypatch):
    dbp = _db(tmp_path)
    loaded, predicted = [], []

    def fake_router(model_ref, device=None):
        loaded.append(model_ref)
        return lambda states: predicted.append((model_ref, len(states))) or [0.4] * len(states)
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", fake_router)
    for ref in (REF, REF_B, REF):
        assert cli.main(["--db", str(dbp), "gate", "score", "--contender", "L1", "--model-ref", ref]) == 0
    assert loaded == [REF, REF_B, REF]
    assert predicted == [(REF, 60), (REF_B, 60)]       # B is scored afresh; the repeat of A is a cache hit
    with closing(db.connect(dbp)) as c:
        assert sorted(r[0] for r in c.execute("SELECT DISTINCT contender FROM gate_scores")) \
            == sorted([f"L1@{REF}", f"L1@{REF_B}"])


@pytest.mark.parametrize("ref", ["me/laya", "me/laya@main", "me/laya@", "me/laya@abc"])
def test_gate_score_l1_rejects_an_unpinned_model_ref(tmp_path, monkeypatch, capsys, ref):
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", lambda *a, **k: pytest.fail("no model may load"))
    with pytest.raises(SystemExit) as e:
        cli.main(["--db", str(tmp_path / "h.db"), "gate", "score", "--contender", "L1", "--model-ref", ref])
    assert e.value.code == 2 and "repo@" in capsys.readouterr().err


def test_gate_score_l1_requires_model_ref(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["--db", str(tmp_path / "h.db"), "gate", "score", "--contender", "L1"])
    assert e.value.code == 2


def test_gate_score_requires_contender(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["--db", str(tmp_path / "h.db"), "gate", "score"])
    assert e.value.code == 2


_LAYA_CMDS = [["gate", "score", "--contender", "L0"], ["gate", "final-report"]]


@pytest.mark.parametrize("argv", _LAYA_CMDS)
def test_laya_commands_explain_the_missing_laya_install(tmp_path, monkeypatch, capsys, argv):
    dbp = _db(tmp_path, laya_l1_ref=json.dumps(REF))
    monkeypatch.setattr(cli.meta, "load_model", lambda path: _FakeB2(()))

    def no_laya(model_ref, device=None):
        raise ModuleNotFoundError("No module named 'laya'", name="laya")
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", no_laya)
    monkeypatch.setattr(cli.laya_io, "score", lambda *a, **k: pytest.fail("nothing may be scored"))
    assert cli.main(["--db", str(dbp), *argv]) != 0
    err = capsys.readouterr().err
    assert "run this from ~/projects/laya-weather-gate/.venv/bin/citycast" in err
    assert "laya is not installed in this environment" in err


@pytest.mark.parametrize("argv", _LAYA_CMDS)
def test_other_missing_modules_are_not_reported_as_a_missing_laya(tmp_path, monkeypatch, capsys, argv):
    dbp = _db(tmp_path, laya_l1_ref=json.dumps(REF))
    monkeypatch.setattr(cli.meta, "load_model", lambda path: _FakeB2(()))

    def no_torch(model_ref, device=None):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", no_torch)
    monkeypatch.setattr(cli.laya_io, "score", lambda *a, **k: pytest.fail("nothing may be scored"))
    with pytest.raises(ModuleNotFoundError) as e:
        cli.main(["--db", str(dbp), *argv])
    assert e.value.name == "torch"
    assert "laya-weather-gate" not in capsys.readouterr().err


def _no_scoring(monkeypatch):
    monkeypatch.setattr(cli.laya_io, "score", lambda *a, **k: pytest.fail("test split must not be scored"))
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", lambda *a, **k: pytest.fail("no model may load"))
    monkeypatch.setattr(cli.meta, "load_model", lambda path: pytest.fail("no model may load"))


def test_final_report_refuses_second_run_before_any_scoring(tmp_path, monkeypatch):
    dbp = _db(tmp_path, final_report_run_at="2026-10-08T00:00:00+00:00", laya_l1_ref=json.dumps("me/laya@abcdef1"))
    _no_scoring(monkeypatch)
    with pytest.raises(RuntimeError, match="already run"):
        cli.main(["--db", str(dbp), "gate", "final-report"])


@pytest.mark.parametrize("stored", [None, "null", '""', "me/laya@abcdef1", "7", '["me/laya@abcdef1"]', '{"ref": 1}'])
def test_final_report_needs_l1_ref_before_any_scoring(tmp_path, monkeypatch, stored):
    """Missing, empty, not JSON (a raw string set by hand) or not a string: the same clear error."""
    dbp = _db(tmp_path, **({} if stored is None else {"laya_l1_ref": stored}))
    _no_scoring(monkeypatch)
    with pytest.raises(RuntimeError, match="laya_l1_ref not set"):
        cli.main(["--db", str(dbp), "gate", "final-report"])


@pytest.mark.parametrize("ref", ["me/laya", "me/laya@main", "me/laya@abc"])
def test_final_report_rejects_an_unpinned_l1_ref_before_any_scoring(tmp_path, monkeypatch, ref):
    dbp = _db(tmp_path, laya_l1_ref=json.dumps(ref))
    _no_scoring(monkeypatch)
    with pytest.raises(RuntimeError, match=r"laya_l1_ref.*repo@"):
        cli.main(["--db", str(dbp), "gate", "final-report"])


def _stub_final(monkeypatch, tmp_path, scored):
    monkeypatch.setattr(cli.meta, "load_model", lambda path: _FakeB2(()))
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", lambda ref, device=None: (lambda s: s))

    def fake_score(df, fn, contender, conn):
        scored.append(contender)
        return df["won"].to_numpy().astype(float) if contender.startswith("L1") else [0.5] * len(df)
    monkeypatch.setattr(cli.laya_io, "score", fake_score)
    monkeypatch.setattr(cli, "REPORT_DIR", tmp_path / "reports")


def test_final_report_runs_end_to_end_and_pins_the_winner_and_model(tmp_path, monkeypatch, capsys):
    dbp = _db(tmp_path, n_rows=180, laya_l1_ref=json.dumps(REF))   # 60 winners per split: oracle L1 is tunable
    scored, refs = [], []
    _stub_final(monkeypatch, tmp_path, scored)
    monkeypatch.setattr(cli.laya_io, "router_predict_fn", lambda ref, device=None: refs.append(ref) or (lambda s: s))
    assert cli.main(["--db", str(dbp), "gate", "final-report"]) == 0
    res = _last(capsys)
    assert refs == [None, REF] and res["winner"] == "L1" and res["margin"] is not None
    assert scored == ["L0", "L0", f"L1@{REF}", f"L1@{REF}"]            # validation then test, per contender
    assert len(list((tmp_path / "reports").glob("gate-final-*.md"))) == 1
    with closing(db.connect(dbp)) as c:
        choice = json.loads(c.execute("SELECT value FROM meta WHERE key='gate_choice'").fetchone()[0])
    assert choice["winner"] == "L1" and choice["model_ref"] == REF and choice["margin"] == res["margin"]
    with pytest.raises(RuntimeError, match="already run"):      # and now the test split is spent
        cli.main(["--db", str(dbp), "gate", "final-report"])


def test_final_report_force_gets_past_ensure_not_run(tmp_path, monkeypatch, capsys):
    dbp = _db(tmp_path, n_rows=180, final_report_run_at="2026-10-08T00:00:00+00:00",
              laya_l1_ref=json.dumps(REF))
    scored = []
    _stub_final(monkeypatch, tmp_path, scored)
    assert cli.main(["--db", str(dbp), "gate", "final-report", "--force"]) == 0
    assert scored and _last(capsys)["winner"] == "L1"
    with closing(db.connect(dbp)) as c:
        assert c.execute("SELECT value FROM meta WHERE key='final_report_run_at'").fetchone()[0] \
            != "2026-10-08T00:00:00+00:00"
