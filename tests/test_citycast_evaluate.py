# tests/test_citycast_evaluate.py
import json
from contextlib import closing

import pytest

from citycast import db
from citycast.evaluate import choose_and_gate, evaluate_all, market_brier, pre_day_diffs, scored

F = "2026-10-07T00:00:00+00:00"


def _pred(variant, d, pmf, asof="12:00", run_ext=None):
    return {"station": "KNYC", "metric": "high", "event_date": d, "asof_utc": f"{d}T{asof}:00+00:00",
            "variant": variant, "fold": "x", "anchor": 50.0, "run_ext": run_ext, "quantiles_json": "{}",
            "pmf_json": json.dumps(pmf), "k": 1.0, "created_at": F}


def _seed(c, good="city", rep_days=60, rep_good=True):
    """60 selection-period days where `good` beats everything, then `rep_days` report-period days
    where the chosen variant beats mosnorm (rep_good) or loses to it."""
    preds, cli = [], []
    days = [(f"2025-0{1 + i // 28}-{1 + i % 28:02d}", True) for i in range(60)]
    days += [(f"2026-0{4 + i // 28}-{1 + i % 28:02d}", rep_good) for i in range(rep_days)]
    for d, wins in days:
        cli.append({"station": "KNYC", "date": d, "high_f": 50, "low_f": 30, "final": 1,
                    "issued_utc": None, "raw": "{}", "fetched_at": F})
        sharp, wide = {50: 0.9, 51: 0.1}, {47: 0.5, 53: 0.5}
        preds += [_pred(good, d, sharp if wins else wide),
                  _pred("shared" if good == "city" else "city", d, {49: 0.5, 51: 0.5}),
                  _pred("mosnorm", d, wide if wins else sharp)]
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "wf_predictions", preds)


def _mark_runs(c, **over):
    """A finished walk-forward record for both metrics that matches wf_predictions and the current code."""
    from citycast.evaluate import code_fingerprint
    fp = code_fingerprint()
    with c:
        for metric in ("high", "low"):
            n = c.execute("SELECT COUNT(*) FROM wf_predictions WHERE metric=?", (metric,)).fetchone()[0]
            db.upsert(c, "wf_runs", [{"metric": metric, "fingerprint": fp, "dataset_fingerprint": fp,
                                      "started_at": F, "finished_at": F, "rows": n, **over}])


def test_selects_better_variant_and_passes_gate(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        out = choose_and_gate(c, "high")
        assert out[0]["variant"] == "city" and out[0]["gate_pass"] == 1
        assert c.execute("SELECT COUNT(*) FROM model_choice").fetchone()[0] == 0   # computes, never writes
        _mark_runs(c)
        evaluate_all(c, tmp_path / "r.md", lambda: F)
        row = c.execute("SELECT variant, gate_pass, decided_at FROM model_choice WHERE station='KNYC'").fetchone()
        assert tuple(row) == ("city", 1, F)
        log = c.execute("SELECT metric, report_scored FROM eval_log ORDER BY metric").fetchall()
        assert [tuple(r) for r in log] == [("high", 1), ("low", 0)]


def test_gate_fails_when_model_loses_in_report_period(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_good=False)
        out = choose_and_gate(c, "high")
        assert out[0]["sel_pass"] == 1 and out[0]["gate_pass"] == 0
        assert out[0]["crps_rep"] > out[0]["mos_crps_rep"]


def test_gate_fails_with_too_few_report_days(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=59)
        out = choose_and_gate(c, "high")
        assert out[0]["sel_pass"] == 1 and out[0]["rep_days"] == 59 and out[0]["gate_pass"] == 0


def test_variant_choice_ignores_report_period(tmp_path):
    # city wins the selection period; in the report period shared is far better than city.
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=0)
        rows = []
        for i in range(60):
            d = f"2026-0{4 + i // 28}-{1 + i % 28:02d}"
            rows += [_pred("city", d, {40: 1.0}), _pred("shared", d, {50: 1.0}), _pred("mosnorm", d, {45: 1.0})]
        with c:
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": r["event_date"], "high_f": 50, "low_f": 30,
                                        "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F}
                                       for r in rows[::3]])
            db.upsert(c, "wf_predictions", rows)
        out = choose_and_gate(c, "high")
        assert out[0]["variant"] == "city"                      # chosen on selection only
        assert out[0]["gate_pass"] == 0                         # and then fails confirmation


def test_report_warns_when_code_changed_after_report_was_scored(tmp_path):
    from citycast.evaluate import code_fingerprint
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        assert "WARNING" not in evaluate_all(c, tmp_path / "r.md", lambda: F).read_text()
        assert "WARNING" not in evaluate_all(c, tmp_path / "r.md", lambda: "2026-10-08T00:00:00+00:00").read_text()
        with c:
            c.execute("UPDATE eval_log SET fingerprint='0000'")       # first scoring ran on other code
        assert "WARNING" in evaluate_all(c, tmp_path / "r.md", lambda: "2026-10-09T00:00:00+00:00").read_text()
        assert len(code_fingerprint()) == 64


def test_canonical_events_picks_one_event_per_city_day(tmp_path):
    from citycast.evaluate import canonical_events
    base = {"series": "KXHIGHCHI", "station": "KMDW", "metric": "high", "event_date": "2024-02-28",
            "lo": 1, "hi": 2, "open_time": None, "close_time": "2024-02-29T05:59:00+00:00",
            "result": "no", "expiration_value": "", "raw": "{}"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [{**base, "ticker": "A1", "event_ticker": "EV-B"},
                                            {**base, "ticker": "A2", "event_ticker": "EV-B"},
                                            {**base, "ticker": "B1", "event_ticker": "EV-A"}])
        assert canonical_events(c) == {"EV-B"}                   # more markets wins


def test_market_brier_uses_fresh_candle_only(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        with c:
            db.upsert(c, "kalshi_markets", [{
                "ticker": "M1", "series": "KXHIGHNY", "event_ticker": "E1", "station": "KNYC",
                "metric": "high", "event_date": "2025-01-01", "lo": 50, "hi": 51, "open_time": None,
                "close_time": "2025-01-02T04:59:00+00:00", "result": "yes", "expiration_value": "50",
                "raw": "{}"}])
            db.upsert(c, "kalshi_candles", [
                {"ticker": "M1", "end_utc": "2025-01-01T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.50,
                 "volume": 1.0, "open_interest": 1.0},
                {"ticker": "M1", "end_utc": "2025-01-01T08:00:00+00:00", "yes_bid": 0.10, "yes_ask": 0.20,
                 "volume": 1.0, "open_interest": 1.0}])
        res = market_brier(c, "high", {"KNYC": "city"})
        mean, lo, hi, n = res["selection"]["KNYC"]
        # model p = 1.0 (all mass on 50-51), market mid 0.45 from the 11:00 candle; won = 1
        assert n == 1 and abs(mean - (0.0 - 0.3025)) < 1e-9
        assert res["report"] == {}                              # the market is in the selection period
        # the choice argument picks the variant: shared puts 0.5 on 50-51
        mean, *_ = market_brier(c, "high", {"KNYC": "shared"})["selection"]["KNYC"]
        assert abs(mean - (0.25 - 0.3025)) < 1e-9


STALE = "INSERT INTO model_choice VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"


def test_evaluate_all_replaces_stale_choices_of_both_metrics(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        with c:
            c.execute(STALE, ("KSFO", "high", "city", 1.0, 2.0, -0.1, 1, 1.0, 2.0, 70, 1, F))
            c.execute(STALE, ("KSFO", "low", "city", 1.0, 2.0, -0.1, 1, 1.0, 2.0, 70, 1, F))
        evaluate_all(c, tmp_path / "r.md", lambda: F)
        assert {(r[0], r[1]) for r in c.execute("SELECT station, metric FROM model_choice")} == {("KNYC", "high")}


def test_evaluate_all_commits_nothing_when_a_computation_fails(tmp_path, monkeypatch):
    import citycast.evaluate as ev
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        with c:
            c.execute(STALE, ("KSFO", "high", "city", 1.0, 2.0, -0.1, 1, 1.0, 2.0, 70, 1, F))

        def boom(*a, **k):
            raise MemoryError("market_brier ran out of memory")
        monkeypatch.setattr(ev, "market_brier", boom)
        with pytest.raises(MemoryError):
            evaluate_all(c, tmp_path / "r.md", lambda: F)
        assert [tuple(r) for r in c.execute("SELECT station, metric FROM model_choice")] == [("KSFO", "high")]
        assert c.execute("SELECT COUNT(*) FROM eval_log").fetchone()[0] == 0     # report period not spent
        assert not (tmp_path / "r.md").exists()


def test_evaluate_all_checks_both_metrics_before_scoring(tmp_path, monkeypatch):
    import citycast.evaluate as ev
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        with c:
            c.execute("UPDATE wf_runs SET finished_at=NULL WHERE metric='low'")
        monkeypatch.setattr(ev, "scored", lambda *a, **k: pytest.fail("scored before the inputs were checked"))
        with pytest.raises(RuntimeError, match="low: the walk-forward run .* never finished"):
            evaluate_all(c, tmp_path / "r.md", lambda: F)
        assert c.execute("SELECT COUNT(*) FROM model_choice").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM eval_log").fetchone()[0] == 0


def test_evaluate_all_hashes_the_code_once_and_logs_that_value(tmp_path, monkeypatch):
    import citycast.evaluate as ev
    first = "A" * 64
    calls = []

    def hashes():                                                # the code "changes" after the first hash
        calls.append(1)
        return first if len(calls) == 1 else "B" * 64
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c, fingerprint=first, dataset_fingerprint=first)
        monkeypatch.setattr(ev, "code_fingerprint", hashes)
        text = evaluate_all(c, tmp_path / "r.md", lambda: F).read_text()
        assert "WARNING" not in text                              # nothing earlier was scored under other code
        assert {r[0] for r in c.execute("SELECT fingerprint FROM eval_log")} == {first}
        assert len(calls) == 1                                    # check_inputs, warning and log share it


def test_evaluate_all_uses_the_fingerprint_it_is_given_without_hashing(tmp_path, monkeypatch):
    import citycast.evaluate as ev
    given = "C" * 64
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c, fingerprint=given, dataset_fingerprint=given)
        monkeypatch.setattr(ev, "code_fingerprint", lambda: pytest.fail("evaluate_all hashed the code itself"))
        evaluate_all(c, tmp_path / "r.md", lambda: F, fingerprint=given)
        assert {r[0] for r in c.execute("SELECT fingerprint FROM eval_log")} == {given}


def test_evaluate_command_passes_its_start_of_command_hash(tmp_path, monkeypatch, capsys):
    from citycast import cli
    seen, calls = [], []

    def hashes():
        calls.append(1)
        return f"hash-{len(calls)}"
    monkeypatch.setattr(cli, "code_fingerprint", hashes)
    monkeypatch.setattr(cli.evaluate, "evaluate_all",
                        lambda conn, path, now_iso, fingerprint=None: seen.append(fingerprint) or path)
    assert cli.main(["--db", str(tmp_path / "h.db"), "evaluate"]) == 0
    assert seen == ["hash-1"] and len(calls) == 1


def test_scored_streams_without_pmfs_unless_asked(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        df = scored(c, "high")
        assert list(df.columns) == ["station", "metric", "event_date", "asof_utc", "variant", "rps", "y", "run_ext"]
        assert "pmf" not in df.columns
        assert len(df) == 360 and set(df["variant"]) == {"city", "shared", "mosnorm"}
        assert df["run_ext"].isna().all()
        sub = scored(c, "high", variants=("city", "mosnorm"), keep_pmf=True)
        assert "pmf" in sub.columns and len(sub) == 240 and set(sub["variant"]) == {"city", "mosnorm"}
        assert sub["pmf"].iloc[0] in ({50: 0.9, 51: 0.1}, {47: 0.5, 53: 0.5})
        assert scored(c, "high", station="KSFO").empty
        empty = scored(c, "low")                                 # no rows: still filterable by date
        assert empty[empty["event_date"] < "2026-04-01"].empty


def test_fingerprint_covers_all_modelling_files():
    import citycast.evaluate as ev
    from citycast import fingerprint
    assert ev.code_fingerprint is fingerprint.code_fingerprint
    assert not hasattr(ev, "FINGERPRINT_MODULES")
    files = fingerprint.FINGERPRINT_FILES
    assert {"src/citycast/metrics.py", "src/highcast/buckets.py", "src/highcast/latentmax.py",
            "src/highcast/timeutil.py", "src/citycast/stations.py", "config/citycast_stations.yaml"} <= set(files)
    assert files[-5:] == ("src/highcast/timeutil.py", "src/highcast/quantize.py", "src/highcast/config.py",
                          "src/highcast/metric.py", "config/citycast_stations.yaml")   # latentmax's imports
    for rel in files:
        assert (fingerprint.REPO_ROOT / rel).is_file(), rel


def test_fingerprint_changes_when_a_highcast_helper_changes(tmp_path, monkeypatch):
    import shutil

    from citycast import fingerprint
    for rel in fingerprint.FINGERPRINT_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fingerprint.REPO_ROOT / rel, tmp_path / rel)
    before = fingerprint.code_fingerprint()
    monkeypatch.setattr(fingerprint, "REPO_ROOT", tmp_path)
    assert fingerprint.code_fingerprint() == before                  # same bytes, same hash
    with open(tmp_path / "src/highcast/buckets.py", "a") as f:
        f.write("\n# edited on main\n")
    assert fingerprint.code_fingerprint() != before


@pytest.mark.parametrize("change, reason", [
    ("DELETE FROM wf_runs WHERE metric='high'", "no walk-forward run"),
    ("UPDATE wf_runs SET finished_at=NULL WHERE metric='high'", "never finished"),
    ("UPDATE wf_runs SET rows=rows+1 WHERE metric='high'", "holds 360 rows but the walk-forward run wrote 361"),
    ("DELETE FROM wf_predictions WHERE rowid IN (SELECT rowid FROM wf_predictions LIMIT 1)", "holds 359 rows"),
    ("UPDATE wf_runs SET fingerprint='0000' WHERE metric='high'", "walk-forward ran under different code"),
    ("UPDATE wf_runs SET dataset_fingerprint='0000' WHERE metric='high'", "dataset was built under different code"),
    ("UPDATE wf_runs SET dataset_fingerprint=NULL WHERE metric='high'", "dataset was built under different code"),
])
def test_check_inputs_refuses_inconsistent_inputs(tmp_path, change, reason):
    from citycast.evaluate import check_inputs, code_fingerprint
    fp = code_fingerprint()
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        check_inputs(c, "high", fp)                              # consistent: passes
        check_inputs(c, "low", fp)                               # an empty finished run is consistent too
        with c:
            c.execute(change)
        with pytest.raises(RuntimeError, match=reason):
            check_inputs(c, "high", fp)


def test_check_inputs_compares_against_the_fingerprint_it_is_given(tmp_path, monkeypatch):
    import citycast.evaluate as ev
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c, fingerprint="A" * 64, dataset_fingerprint="A" * 64)
        monkeypatch.setattr(ev, "code_fingerprint", lambda: pytest.fail("check_inputs hashed the code itself"))
        ev.check_inputs(c, "high", "A" * 64)
        with pytest.raises(RuntimeError, match="walk-forward ran under different code"):
            ev.check_inputs(c, "high", "B" * 64)


def test_warning_survives_reverting_to_original_code(tmp_path):
    from citycast.evaluate import code_fingerprint
    cur = code_fingerprint()
    with closing(db.connect(tmp_path / "h.db")) as c:
        _mark_runs(c)
        with c:
            c.executemany("INSERT INTO eval_log VALUES (?,?,?,?)", [
                ("2026-10-01T00:00:00", "high", cur, 1),
                ("2026-10-02T00:00:00", "high", "0000", 1),
                ("2026-10-03T00:00:00", "high", cur, 1)])
        assert "WARNING" in evaluate_all(c, tmp_path / "r.md", lambda: "2026-10-04T00:00:00").read_text()


def test_market_brier_uses_kalshi_result_not_cli_bracket(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        with c:
            db.upsert(c, "kalshi_markets", [{
                "ticker": "M1", "series": "KXHIGHNY", "event_ticker": "E1", "station": "KNYC",
                "metric": "high", "event_date": "2025-01-01", "lo": 50, "hi": 51, "open_time": None,
                "close_time": "2025-01-02T04:59:00+00:00", "result": "no", "expiration_value": "50",
                "raw": "{}"}])
            db.upsert(c, "kalshi_candles", [
                {"ticker": "M1", "end_utc": "2025-01-01T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.50,
                 "volume": 1.0, "open_interest": 1.0}])
        mean, _, _, n = market_brier(c, "high", {"KNYC": "city"})["selection"]["KNYC"]
        # CLI truth 50 is in [50,51] but Kalshi settled "no": won=0, model p=1 -> 1.0 - 0.45**2
        assert n == 1 and abs(mean - (1.0 - 0.2025)) < 1e-9


def test_report_labels_selection_columns_in_sample(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        _mark_runs(c)
        text = evaluate_all(c, tmp_path / "r.md", lambda: F).read_text()
        assert "in-sample for the variant choice" in text and "| KNYC | city |" in text


def _seed_pre_and_same_day(c):
    """city loses to mosnorm before the day's first observation (06:00, run_ext NULL) and wins by more
    once it has the running max (18:00, run_ext 50). shared is worse than both, so city is chosen."""
    preds, cli = [], []
    days = [f"2025-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(60)]
    days += [f"2026-0{4 + i // 28}-{1 + i % 28:02d}" for i in range(60)]
    for d in days:
        cli.append({"station": "KNYC", "date": d, "high_f": 50, "low_f": 30, "final": 1,
                    "issued_utc": None, "raw": "{}", "fetched_at": F})
        preds += [_pred("city", d, {47: 0.5, 53: 0.5}, "06:00"), _pred("mosnorm", d, {50: 0.9, 51: 0.1}, "06:00"),
                  _pred("shared", d, {45: 1.0}, "06:00"),
                  _pred("city", d, {50: 1.0}, "18:00", 50.0), _pred("mosnorm", d, {40: 1.0}, "18:00", 50.0),
                  _pred("shared", d, {45: 1.0}, "18:00", 50.0)]
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "wf_predictions", preds)


def test_pre_day_split_isolates_rows_before_the_first_observation(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed_pre_and_same_day(c)
        choices = choose_and_gate(c, "high")
        assert choices[0]["variant"] == "city" and choices[0]["gate_pass"] == 1   # carried by same-day rows
        split = pre_day_diffs(scored(c, "high"), choices)["KNYC"]
        # pre-day: city RPS 1.5 (mass 0.5 at 47 and 53, y = 50) minus mosnorm 0.01
        assert split == {"selection": (pytest.approx(1.49), 60), "report": (pytest.approx(1.49), 60)}
        _mark_runs(c)
        text = evaluate_all(c, tmp_path / "r.md", lambda: F).read_text()
        assert "pre-day Δ sel (days)" in text and "| PASS | +1.4900 (60) | +1.4900 (60) |" in text


def test_pre_day_split_is_empty_without_pre_day_rows(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        with c:
            c.execute("UPDATE wf_predictions SET run_ext=50.0")          # every row has seen an observation
        df = scored(c, "high")
        assert pre_day_diffs(df, choose_and_gate(c, "high")) == {"KNYC": {"selection": None, "report": None}}
