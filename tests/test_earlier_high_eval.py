"""Contracts for the isolated Phase 0 replay ledger."""

import gzip
import hashlib
import json
import math
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from earlier_high_eval import (
    append_feature_snapshot,
    append_outcome,
    append_prediction,
    append_run,
    append_source_event,
    canonical_json,
    ensure_replay_schema,
    replay_miniature,
    sha256_json,
    verify_ledger,
)
import earlier_high_eval as replay_eval


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_replay_schema(conn)
    return conn


def _run(conn, run_id="run-1", mode="backtest"):
    return append_run(conn, {
        "run_id": run_id, "experiment": "earlier_high_remaining_heat_v1",
        "spec_version": "phase-0", "started_utc": "2026-08-26T00:00:00+00:00",
        "mode": mode, "code_fingerprint": "code", "config_sha256": "cfg",
        "schema_version": 1, "manifest_json": {"stations": ["KMIA"]}, "status": "started",
    })


def _source(conn, event_id="source-1", **changes):
    record = {
        "source_event_id": event_id, "run_id": "run-1", "station": "KMIA", "source": "metar",
        "fetched_utc": "2030-01-02T00:00:00+00:00", "valid_utc": "2026-08-26T12:00:00+00:00",
        "parser_version": "metar-v1", "raw_body": {"raw": "METAR KMIA 261200Z"},
        "parsed_input": {"raw": "METAR KMIA 261200Z"},
    }
    record.update(changes)
    return append_source_event(conn, record)


def _snapshot(conn, snapshot_id="snapshot-1", **changes):
    record = {
        "feature_snapshot_id": snapshot_id, "run_id": "run-1", "station": "KMIA",
        "target_date": "2026-08-26", "checkpoint_hour": 9, "asof_utc": "2026-08-26T13:00:00+00:00",
        "bundle": "B1", "feature_schema_version": 1, "feature_code_sha256": "feature-code",
        "features_json": {"cur_rh_pct": 70.0}, "missing_json": {"cur_rh_pct": False},
        "lineage_json": {"source_event_ids": ["source-1"]}, "visible_floor_f": 85.0,
        "source_max_known_utc": "2026-08-26T12:00:00+00:00", "created_utc": "2026-08-26T13:00:01+00:00",
    }
    record.update(changes)
    return append_feature_snapshot(conn, record)


def _prediction(conn, prediction_id="prediction-1", **changes):
    record = {
        "replay_prediction_id": prediction_id, "run_id": "run-1", "feature_snapshot_id": "snapshot-1",
        "station": "KMIA", "target_date": "2026-08-26", "checkpoint_hour": 9,
        "issued_utc": "2026-08-26T13:00:02+00:00", "bundle": "B1", "fold_id": "fold-1",
        "model_fingerprint": "model", "train_cutoff_date": "2026-08-25", "train_row_count": 42,
        "point_f": 87.0, "quantiles_json": {"p50": 87.0}, "buckets_json": {"86-87": 0.5},
        "created_utc": "2026-08-26T13:00:03+00:00",
    }
    record.update(changes)
    return append_prediction(conn, record)


def test_canonical_json_is_utf8_sorted_compact_and_rejects_nan():
    # Break caught: nondeterministic/non-UTF-8 serialization changes fingerprints.
    assert canonical_json({"z": None, "a": ["é", 1]}) == b'{"a":["\xc3\xa9",1],"z":null}'
    assert sha256_json({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    with pytest.raises(ValueError, match="NaN|finite"):
        canonical_json({"bad": math.nan})


def test_duplicate_logical_writes_are_rejected_and_rows_are_immutable():
    # Break caught: a rerun can silently overwrite or duplicate a prediction.
    conn = _conn()
    _run(conn)
    _source(conn)
    _snapshot(conn)
    _prediction(conn)
    append_outcome(conn, {"outcome_id": "outcome-1", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 88, "recorded_utc": "2026-08-27T00:00:00+00:00",
                          "raw_sha256": "outcome-raw", "is_final": False})
    with pytest.raises(sqlite3.IntegrityError):
        _prediction(conn, "prediction-2")
    for table in ("replay_runs", "replay_run_events", "replay_source_events",
                  "replay_feature_snapshots", "replay_predictions", "replay_outcomes"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")


def test_update_is_rejected_for_every_replay_table():
    # Break caught: an in-place correction destroys original audit evidence.
    conn = _conn()
    _run(conn)
    _source(conn)
    _snapshot(conn)
    _prediction(conn)
    append_outcome(conn, {"outcome_id": "outcome-1", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 88, "recorded_utc": "2026-08-27T00:00:00+00:00",
                          "raw_sha256": "outcome-raw", "is_final": False})
    for table in ("replay_runs", "replay_run_events", "replay_source_events",
                  "replay_feature_snapshots", "replay_predictions", "replay_outcomes"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(f"UPDATE {table} SET rowid=rowid")


def test_outcome_correction_appends_and_preserves_superseded_outcome():
    # Break caught: corrected CLI values erase a previously recorded outcome.
    conn = _conn()
    append_outcome(conn, {"outcome_id": "first", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 88, "recorded_utc": "2026-08-27T00:00:00+00:00",
                          "raw_sha256": "a", "is_final": False})
    append_outcome(conn, {"outcome_id": "corrected", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 89, "recorded_utc": "2026-08-27T01:00:00+00:00",
                          "raw_sha256": "b", "is_final": True, "supersedes_id": "first"})
    rows = conn.execute("SELECT outcome_id, cli_high_f, supersedes_id FROM replay_outcomes ORDER BY recorded_utc").fetchall()
    assert [tuple(row) for row in rows] == [("first", 88, None), ("corrected", 89, "first")]


def test_supersedes_requires_the_same_source_or_outcome_logical_identity():
    # Break caught: a correction chain can link unrelated stations/sources/days.
    conn = _conn()
    _run(conn)
    _source(conn, "source-original")
    with pytest.raises(ValueError, match="supersedes.*logical identity"):
        _source(conn, "source-wrong-station", station="KLAX", supersedes_id="source-original")
    append_outcome(conn, {"outcome_id": "outcome-original", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 88, "recorded_utc": "2026-08-27T00:00:00+00:00",
                          "raw_sha256": "a", "is_final": False})
    with pytest.raises(ValueError, match="supersedes.*logical identity"):
        append_outcome(conn, {"outcome_id": "outcome-wrong-date", "station": "KMIA", "target_date": "2026-08-27",
                              "cli_high_f": 89, "recorded_utc": "2026-08-28T00:00:00+00:00",
                              "raw_sha256": "b", "is_final": True, "supersedes_id": "outcome-original"})


def test_snapshot_rejects_future_known_lineage_but_archive_metar_uses_valid_time(tmp_path):
    # Break caught: archive download time is treated as the time an observation became knowable.
    conn = _conn()
    _run(conn)
    _source(conn, raw_root=tmp_path / "raw")
    row = conn.execute("SELECT known_utc, known_utc_basis FROM replay_source_events").fetchone()
    assert tuple(row) == ("2026-08-26T12:00:00+00:00", "observation_valid_time")
    with pytest.raises(ValueError, match="future-known|known_utc"):
        _snapshot(conn, asof_utc="2026-08-26T11:59:59+00:00")


def test_forecast_known_time_is_later_of_availability_and_fetch_time():
    # Break caught: forecast data fetched after a checkpoint leaks into it.
    conn = _conn()
    _run(conn)
    _source(conn, source="forecast_guidance", model_run_utc="2026-08-26T10:00:00+00:00",
            fetched_utc="2026-08-26T14:00:00+00:00", raw_body="forecast")
    row = conn.execute("SELECT known_utc, known_utc_basis FROM replay_source_events").fetchone()
    assert tuple(row) == ("2026-08-26T14:00:00+00:00", "forecast_available_and_fetched")


def test_prediction_requires_an_immutable_snapshot_and_no_trade_shadow_is_retained():
    # Break caught: a prediction can exist without its reproducible feature vector, or shadow rows are dropped.
    conn = _conn()
    _run(conn, mode="forward_shadow")
    with pytest.raises(sqlite3.IntegrityError):
        _prediction(conn, feature_snapshot_id="missing")
    _source(conn)
    _snapshot(conn)
    _prediction(conn)
    row = conn.execute("SELECT mode, count(*) FROM replay_runs JOIN replay_predictions USING (run_id) GROUP BY mode").fetchone()
    assert tuple(row) == ("forward_shadow", 1)


def test_no_trade_forward_shadow_later_joins_final_cli_truth():
    # Break caught: forward-shadow rows are omitted until a trade/PASS exists.
    conn = _conn()
    _run(conn, mode="forward_shadow")
    _source(conn)
    _snapshot(conn)
    _prediction(conn)
    before = conn.execute(
        "SELECT p.replay_prediction_id, o.outcome_id FROM replay_predictions p LEFT JOIN replay_outcomes o "
        "ON o.station=p.station AND o.target_date=p.target_date WHERE p.replay_prediction_id='prediction-1'"
    ).fetchone()
    assert tuple(before) == ("prediction-1", None)
    append_outcome(conn, {"outcome_id": "later-final", "station": "KMIA", "target_date": "2026-08-26",
                          "cli_high_f": 88, "product_utc": "2026-08-27T00:00:00+00:00",
                          "recorded_utc": "2026-08-27T01:00:00+00:00", "raw_sha256": "final", "is_final": True})
    after = conn.execute(
        "SELECT p.replay_prediction_id, o.cli_high_f FROM replay_predictions p JOIN replay_outcomes o "
        "ON o.station=p.station AND o.target_date=p.target_date WHERE p.replay_prediction_id='prediction-1'"
    ).fetchone()
    assert tuple(after) == ("prediction-1", 88)


def test_verify_ledger_reports_missing_and_tampered_raw_without_repair(tmp_path):
    # Break caught: integrity checks hide or repair raw evidence instead of reporting it.
    db_path, raw_root = tmp_path / "ledger.db", tmp_path / "raw"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    _run(conn)
    _source(conn, raw_root=raw_root)
    conn.commit()
    conn.close()
    raw_file = next(raw_root.rglob("*.json.gz"))
    raw_file.unlink()
    missing = verify_ledger(db_path, raw_root)
    assert missing["ok"] is False and missing["missing_raw_paths"] == [str(raw_file.relative_to(raw_root))]
    assert not raw_file.exists()
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_file, "wb") as handle:
        handle.write(b"tampered")
    tampered = verify_ledger(db_path, raw_root)
    assert tampered["ok"] is False and tampered["raw_sha256_mismatches"] == [str(raw_file.relative_to(raw_root))]
    assert gzip.open(raw_file, "rb").read() == b"tampered"


def test_retained_raw_promotion_never_clobbers_evidence_or_strands_it_on_failed_insert(tmp_path):
    # Break caught: raw promotion happens before an INSERT that can fail.
    db_path, raw_root = tmp_path / "ledger.db", tmp_path / "raw"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    _run(conn)
    _source(conn, "same-event", raw_root=raw_root, raw_body=b"original")
    conn.commit()
    raw_file = next(raw_root.rglob("*.json.gz"))
    with pytest.raises(sqlite3.IntegrityError):
        _source(conn, "same-event", raw_root=raw_root, raw_body=b"replacement")
    assert gzip.open(raw_file, "rb").read() == b"original"
    with pytest.raises(sqlite3.IntegrityError):
        _source(conn, "orphan-event", run_id="missing-run", raw_root=raw_root, raw_body=b"orphan")
    assert not list(raw_root.rglob("*orphan-event*"))
    assert not list(raw_root.rglob("*.tmp"))
    conn.close()


def test_retained_raw_uses_atomic_rename_after_preflight_without_duplicate_clobber(tmp_path, monkeypatch):
    # Break caught: publication substitutes another primitive or renames over existing evidence.
    db_path, raw_root = tmp_path / "ledger.db", tmp_path / "raw"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    _run(conn)
    real_replace, replacements = replay_eval.os.replace, []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(replay_eval.os, "replace", recording_replace)
    _source(conn, "rename-event", raw_root=raw_root, raw_body=b"original")
    conn.commit()
    assert len(replacements) == 1
    assert replacements[0][0].parent == replacements[0][1].parent
    assert replacements[0][1].suffixes == [".json", ".gz"]
    with pytest.raises(sqlite3.IntegrityError):
        _source(conn, "rename-event", raw_root=raw_root, raw_body=b"replacement")
    assert len(replacements) == 1
    assert gzip.open(replacements[0][1], "rb").read() == b"original"
    conn.close()


def test_verifier_reports_orphan_raw_after_enclosing_transaction_rollback(tmp_path):
    # Break caught: raw publication survives caller rollback but verifier ignores it.
    db_path, raw_root = tmp_path / "ledger.db", tmp_path / "raw"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    conn.commit()
    _run(conn)
    _source(conn, "rolled-back-event", raw_root=raw_root, raw_body=b"orphan")
    raw_file = next(raw_root.rglob("*.json.gz"))
    conn.rollback()
    conn.close()
    report = verify_ledger(db_path, raw_root)
    assert report["ok"] is False
    assert report["orphan_raw_paths"] == [str(raw_file.relative_to(raw_root))]


def test_unretained_raw_uses_explicit_canonical_parsed_input_not_caller_digest():
    # Break caught: a no-raw event can claim an arbitrary, unverifiable digest.
    conn = _conn()
    _run(conn)
    _source(conn, raw_body=None, raw_sha256="untrusted", parsed_input={"parsed": {"temp_f": 80.0}})
    row = conn.execute("SELECT raw_path, raw_sha256 FROM replay_source_events").fetchone()
    assert tuple(row) == (None, sha256_json({"parsed": {"temp_f": 80.0}}))
    with pytest.raises(ValueError, match="parsed_input"):
        append_source_event(conn, {
            "source_event_id": "missing-parsed", "run_id": "run-1", "station": "KMIA", "source": "metar",
            "fetched_utc": "2030-01-02T00:00:00+00:00", "valid_utc": "2026-08-26T12:00:00+00:00",
            "parser_version": "metar-v1", "raw_sha256": "untrusted",
        })


def test_verifier_independently_rejects_bad_lineage_and_wrong_supersession(tmp_path):
    # Break caught: direct/corrupt ledger rows bypass append-time temporal and identity checks.
    db_path, raw_root = tmp_path / "ledger.db", tmp_path / "raw"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    _run(conn)
    _source(conn)
    _source(conn, "klax-source", station="KLAX")
    _source(conn, "future-source", source="forecast_guidance", model_run_utc="2026-08-26T14:00:00+00:00",
            fetched_utc="2026-08-26T14:00:00+00:00", raw_body="forecast")
    conn.execute(
        """INSERT INTO replay_feature_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("bad-lineage", "run-1", "KMIA", "2026-08-26", 9, "2026-08-26T13:00:00+00:00", "B2", 1,
         "feature", '{"x":1}', '{"x":false}', '{"source_event_ids":["future-source"]}', 80.0,
         "2026-08-26T14:00:00+00:00", "2026-08-26T13:00:00+00:00"),
    )
    conn.execute(
        """INSERT INTO replay_feature_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("cross-station-lineage", "run-1", "KMIA", "2026-08-26", 11,
         "2026-08-26T13:00:00+00:00", "B2", 1, "feature", '{"x":1}',
         '{"x":false}', '{"source_event_ids":["klax-source"]}', 80.0,
         "2026-08-26T12:00:00+00:00", "2026-08-26T13:00:00+00:00"),
    )
    conn.execute(
        """INSERT INTO replay_feature_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("noncanonical-lineage", "run-1", "KMIA", "2026-08-26", 10, "2026-08-26T13:00:00+00:00", "B2", 1,
         "feature", '{"x":1}', '{"x":false}', '{"source_event_ids":["source-1"] }', 80.0,
         "2026-08-26T12:00:00+00:00", "2026-08-26T13:00:00+00:00"),
    )
    conn.execute(
        """INSERT INTO replay_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("outcome-a", "KMIA", "2026-08-26", 88, None, "2026-08-27T00:00:00+00:00", "a", 0, None),
    )
    conn.execute(
        """INSERT INTO replay_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("outcome-b", "KLAX", "2026-08-26", 70, None, "2026-08-27T01:00:00+00:00", "b", 1, "outcome-a"),
    )
    conn.execute(
        """INSERT INTO replay_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("duplicate-final-a", "KMIA", "2026-08-27", 88, "2026-08-28T06:00:00+00:00",
         "2026-08-28T06:01:00+00:00", "c", 1, None),
    )
    conn.execute(
        """INSERT INTO replay_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("duplicate-final-b", "KMIA", "2026-08-27", 89, "2026-08-28T07:00:00+00:00",
         "2026-08-28T07:01:00+00:00", "d", 1, None),
    )
    conn.commit()
    conn.close()
    report = verify_ledger(db_path, raw_root)
    assert report["ok"] is False
    assert {error["feature_snapshot_id"] for error in report["lineage_errors"]} == {
        "bad-lineage", "noncanonical-lineage", "cross-station-lineage",
    }
    assert report["supersession_errors"] == [{"child_id": "outcome-b", "kind": "outcome", "reason": "logical_identity_mismatch", "supersedes_id": "outcome-a"}]
    assert {item["table"] for item in report["duplicate_logical_events"]} == {"replay_outcomes"}


def test_verifier_missing_database_is_read_only_and_creates_nothing(tmp_path):
    # Break caught: an integrity check creates an empty ledger while reporting an error.
    missing_db = tmp_path / "does-not-exist.db"
    report = verify_ledger(missing_db, tmp_path / "raw")
    assert report["ok"] is False and report["missing_database"] == str(missing_db)
    assert not missing_db.exists()


def test_miniature_replay_is_byte_for_byte_deterministic():
    # Break caught: fixture replay output varies due to ordering or ambient state.
    fixture = Path(__file__).parent / "fixtures" / "earlier_high_miniature.json"
    first = replay_miniature(fixture)
    assert first == replay_miniature(fixture)
    artifact = json.loads(first)
    assert artifact["feature_snapshots"][0]["features_json"] == {"cur_rh_pct": 70.0}
    assert artifact["predictions"][0]["feature_snapshot_id"] == "mini-snapshot"
    assert artifact["predictions"][0]["model_fingerprint"] == "mini-model"
    assert artifact["forward_shadow"] == [{"final_outcome_id": "mini-outcome", "no_trade": True, "run_id": "mini-run"}]
    changed = json.loads(fixture.read_text(encoding="utf-8"))
    changed["snapshots"][0]["features_json"]["cur_rh_pct"] = -999.0
    changed_path = fixture.parent / "changed-miniature.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        assert replay_miniature(changed_path) != first
    finally:
        changed_path.unlink()
    changed = json.loads(fixture.read_text(encoding="utf-8"))
    changed["predictions"][0]["model_fingerprint"] = "materially-different-model"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        assert replay_miniature(changed_path) != first
    finally:
        changed_path.unlink()


# Phase 0 Task 3 evaluation and artifact contracts.  Each test names the
# production break it catches and derives expected values independently.


def _result_row(**changes):
    row = {
        "station": "KMIA", "hour": 9, "bundle": "B1", "n": 500,
        "coverage": 0.9, "bucket_hit": 0.55, "delta_hit": 0.03,
        "delta_hit_ci": [0.01, 0.05], "crps": 0.8, "delta_crps": 0.0,
        "noise_delta_hit_p975": 0.02, "verdict": "PASS",
        "safeguards": {"calibration_gap_increase": 0.01},
        "slices": {"full_history": {"delta_hit": 0.01},
                   "warm_season": {"delta_hit": 0.02}},
        "holm_adjusted_p": 0.04, "random_family_contaminated": False,
        "shipped_incumbent_comparison": {
            "name": "SHIPPED_DIRECT_HIGH_INTRADAY", "n": 500,
            "delta_hit": 0.025, "delta_hit_ci": [0.005, 0.045],
            "delta_crps": 0.0, "holm_adjusted_p": 0.04,
            "calibration_gap_increase": 0.01,
            "slices": {"full_history": {"delta_hit": 0.01},
                       "warm_season": {"delta_hit": 0.02}},
        },
        "exclusions": [],
    }
    row.update(changes)
    return row


def test_settlement_floor_remaining_heat_reports_negative_without_clipping():
    # Break caught: raw observations or Python round() replace the conservative
    # lower edge plus the project's half-up settlement quantization.
    assert replay_eval.settlement_floor_from_visible((84.49, 84.69)) == 84
    assert replay_eval.settlement_floor_from_visible((84.50, 84.70)) == 85
    record = replay_eval.remaining_heat_record(
        "KMIA", "2026-08-26", 9, cli_high_f=84,
        visible=(84.50, 84.70),
    )
    assert record == {
        "station": "KMIA", "date": "2026-08-26", "hour": 9,
        "cli_high_f": 84, "visible_interval_f": [84.5, 84.7],
        "visible_floor_f": 85, "remaining_heat_f": -1,
        "invariant_ok": False,
    }


def test_reconstructed_quantiles_are_floored_and_non_crossing():
    # Break caught: a negative/crossed remaining-heat prediction puts mass below
    # an already observed settlement floor.
    assert replay_eval.reconstruct_high_quantiles(
        80.0, {0.1: -2.0, 0.5: 3.0, 0.9: 1.0}
    ) == {0.1: 80.0, 0.5: 81.0, 0.9: 83.0}


def test_arm_matching_uses_only_identical_dates_in_sorted_order():
    # Break caught: an incumbent receives easier or more numerous scoring days
    # than its candidate/noise controls.
    arms = {
        "B0": [{"target_date": "2025-01-01", "v": 1},
               {"target_date": "2025-01-02", "v": 2},
               {"target_date": "2025-01-03", "v": 3}],
        "B1": [{"target_date": "2025-01-02", "v": 20},
               {"target_date": "2025-01-03", "v": 30},
               {"target_date": "2025-01-04", "v": 40}],
        "noise": [{"target_date": "2025-01-03", "v": 300},
                  {"target_date": "2025-01-02", "v": 200}],
    }
    matched = replay_eval.match_arm_records(arms)
    assert matched["dates"] == ["2025-01-02", "2025-01-03"]
    assert [row["v"] for row in matched["arms"]["B0"]] == [2, 3]
    assert [row["v"] for row in matched["arms"]["B1"]] == [20, 30]
    assert [row["v"] for row in matched["arms"]["noise"]] == [200, 300]


def test_bootstrap_and_same_width_random_controls_are_deterministic():
    # Break caught: row order/global RNG state changes the registered comparison.
    delta = [0.5, 0.0, -0.5, 1.0]
    first = replay_eval.paired_bootstrap(delta, n_boot=200, seed=1729)
    second = replay_eval.paired_bootstrap(delta, n_boot=200, seed=1729)
    assert first == second
    keys = ["KMIA|2025-01-01|9", "KMIA|2025-01-02|9"]
    controls = replay_eval.deterministic_random_controls(keys, 3, [1729, 2718, 3141])
    assert controls == replay_eval.deterministic_random_controls(keys, 3, [1729, 2718, 3141])
    assert sorted(controls) == [1729, 2718, 3141]
    assert all(len(rows) == 2 and len(rows[0]) == 3 for rows in controls.values())
    assert controls[1729] != controls[2718]


def test_holm_adjustment_matches_known_vector():
    # Break caught: multiplicity adjustment sorts in the wrong direction or
    # omits the monotonic cumulative maximum.
    got = replay_eval.holm_adjust([0.01, 0.04, 0.03])
    assert got == pytest.approx([0.03, 0.06, 0.06])


def test_run_refuses_absent_stale_and_requested_unsafe_manifest(tmp_path):
    # Break caught: scoring starts without a current audit or admits a
    # FORWARD_ONLY/REJECTED source into a historical arm.
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="audit.*before run|manifest.*absent"):
        replay_eval.validate_manifest_for_run(
            path, ["KMIA"], [9], ["B0", "B1"], "feature-sha", {"KMIA": "cfg-sha"}
        )
    manifest = {
        "audit_complete": True, "stations": ["KMIA"], "hours": [9],
        "feature_code_sha256": "old", "config_sha256": {"KMIA": "cfg-sha"},
        "features": [{"bundle": "B0", "status": "HISTORICAL_SAFE"},
                     {"bundle": "B1", "status": "HISTORICAL_SAFE"}],
        "station_scoring_safe": {"KMIA": True},
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        replay_eval.validate_manifest_for_run(
            path, ["KMIA"], [9], ["B0", "B1"], "feature-sha", {"KMIA": "cfg-sha"}
        )
    manifest["feature_code_sha256"] = "feature-sha"
    manifest["features"][1]["status"] = "FORWARD_ONLY"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe|FORWARD_ONLY"):
        replay_eval.validate_manifest_for_run(
            path, ["KMIA"], [9], ["B0", "B1"], "feature-sha", {"KMIA": "cfg-sha"}
        )


def test_station_hour_verdicts_are_independent_and_exact():
    # Break caught: one station/hour PASS authorizes a neighbour, or a small
    # sample can be mislabeled REJECT/PASS instead of INSUFFICIENT_DATA.
    passed = replay_eval.verdict_for_result(_result_row())
    insufficient = replay_eval.verdict_for_result(_result_row(station="KLAX", hour=10, n=399))
    rejected = replay_eval.verdict_for_result(
        _result_row(station="KMIA", hour=10, delta_hit_ci=[-0.01, 0.05])
    )
    assert (passed, insufficient, rejected) == ("PASS", "INSUFFICIENT_DATA", "REJECT")


def test_verdict_fails_closed_and_calibration_improvement_is_not_a_regression():
    # Break caught: absent Holm/slice/incumbent evidence passes by default, or a
    # calibration improvement larger than two points is rejected via abs().
    assert replay_eval.verdict_for_result(
        _result_row(safeguards={"calibration_gap_increase": -0.03})
    ) == "PASS"
    missing_holm = _result_row()
    del missing_holm["holm_adjusted_p"]
    assert replay_eval.verdict_for_result(missing_holm) == "REJECT"
    missing_slice = _result_row(slices={"warm_season": {"delta_hit": 0.02}})
    assert replay_eval.verdict_for_result(missing_slice) == "REJECT"
    missing_shipped = _result_row()
    del missing_shipped["shipped_incumbent_comparison"]
    assert replay_eval.verdict_for_result(missing_shipped) == "REJECT"


def test_declared_40_comparison_family_does_not_shrink_when_an_arm_errors():
    # Break caught: dropping ERROR rows changes Holm's multiplier and can turn a
    # pre-registered 40-arm family into a passing one-comparison family.
    passing = _result_row(bootstrap_p_two_sided=0.001, _bootstrap_samples=[0.01] * 20)
    failed = _result_row(station="KLAX", verdict="ERROR", bootstrap_p_two_sided=None,
                         _bootstrap_samples=[], error="fit failed")
    rows = [passing, failed]
    replay_eval._apply_multiplicity(rows, declared_family_size=40)
    assert passing["holm_adjusted_p"] == pytest.approx(0.04)
    assert failed["holm_adjusted_p"] == 1.0
    assert failed["verdict"] == "ERROR"


def test_shared_cli_finality_filters_truth_and_retains_product_evidence():
    # Break caught: preliminary high_f rows enter targets and outcomes lose the
    # actual product issue time/canonical raw identity.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE cli_reports (station TEXT, date TEXT, high_f REAL, raw TEXT, fetched_at TEXT)")
    final_raw = {"valid": "2026-01-01", "high": 84,
                 "product": "202601020600-KMFL-CDUS42-CLIMIA"}
    preliminary_raw = {"valid": "2026-01-02", "high": 85,
                       "product": "202601030400-KMFL-CDUS42-CLIMIA"}
    conn.executemany("INSERT INTO cli_reports VALUES (?,?,?,?,?)", [
        ("KMIA", "2026-01-01", 84, json.dumps(final_raw), "2026-01-02T06:01:00+00:00"),
        ("KMIA", "2026-01-02", 85, json.dumps(preliminary_raw), "2026-01-03T04:01:00+00:00"),
    ])
    cfg = SimpleNamespace(station=SimpleNamespace(icao="KMIA", std_utc_offset=-5))
    rows, excluded = replay_eval.load_final_cli_truth(conn, cfg)
    assert [row["date"] for row in rows] == ["2026-01-01"]
    assert excluded == [{"date": "2026-01-02", "reason": "NON_FINAL_CLI_PRODUCT",
                         "product": preliminary_raw["product"]}]
    assert rows[0]["product_utc"] == "2026-01-02T06:00:00+00:00"
    assert rows[0]["raw_sha256"] == sha256_json(final_raw)
    assert rows[0]["raw_payload"] == final_raw


def test_fold_baseline_inputs_are_refit_from_each_expanding_cutoff():
    # Break caught: base_p50/base_spread stay frozen to one pre-2025 blend instead
    # of changing with the outcomes available inside each expanding fold.
    import pandas as pd

    dates = [(datetime(2020, 1, 1) + timedelta(days=index)).date().isoformat()
             for index in range(70)]
    day_frame = pd.DataFrame({
        "target_date": dates,
        "fc_gfs": [70.0] * 70,
        "y": [70.0] * 35 + [90.0] * 35,
    })
    fold = pd.DataFrame({"target_date": [dates[-1]], "base_p50": [-999.0],
                         "base_spread": [-999.0]})
    early = replay_eval.refit_fold_baseline_inputs(day_frame, fold, dates[35], "high")
    late = replay_eval.refit_fold_baseline_inputs(day_frame, fold, dates[-1], "high")
    assert early.loc[0, "base_p50"] == pytest.approx(70.0)
    assert early.loc[0, "base_spread"] == pytest.approx(0.0)
    assert late.loc[0, "base_spread"] > 10.0


def test_shipped_comparator_is_the_exact_direct_high_intraday_model():
    # Break caught: the evaluator calls a remaining-heat B0 arm the incumbent and
    # never instantiates the shipped direct-HIGH IntradayModel convention.
    from highcast.models.intraday import IntradayModel, features_for

    cfg = SimpleNamespace(model=SimpleNamespace(intraday_residual_for=lambda metric: False))
    model = replay_eval.shipped_direct_high_model(cfg, "KLAX")
    assert isinstance(model, IntradayModel)
    assert model.residual is False
    assert model._cols == features_for("KLAX", "high")


def test_full_history_walk_forward_starts_before_primary_test_period_after_declared_minimum():
    # Break caught: full_history is merely the >=2025 primary sample under a new
    # name instead of genuine earlier leakage-safe OOS predictions.
    import pandas as pd

    dates = [(datetime(2023, 1, 1) + timedelta(days=index)).date().isoformat()
             for index in range(760)]
    frame = pd.DataFrame({"target_date": dates, "hour": [9.0] * len(dates),
                          "remaining_heat_f": [1.0] * len(dates)})
    blocks = replay_eval.walk_forward_blocks(
        frame, target_hour=9, training_hours=(9,), refit_every=45,
        minimum_training_rows=400,
    )
    assert blocks[0]["cutoff"] == dates[400]
    assert blocks[0]["cutoff"] < "2025-01-01"
    assert sum(len(block["dates"]) for block in blocks) == 360


def test_negative_target_is_unresolved_and_makes_full_history_unsafe():
    # Break caught: naming the two disagreeing sources self-certifies a negative
    # target as explained without resolving either product's provenance.
    target = replay_eval.remaining_heat_record("KMIA", "2021-07-03", 13, 94, (96.0, 96.2))
    record = replay_eval.unresolved_negative_provenance(
        target, {"product": "202107040600-KMFL-CDUS42-CLIMIA", "raw_sha256": "cli"},
        {"metar_rows": [{"station": "KMIA", "source": "metar",
                         "valid_utc": "2021-07-03T17:53:00+00:00"}]},
    )
    assert record["explained"] is False
    assert record["provenance_status"] == "UNRESOLVED_CLI_METAR_CONTRADICTION"
    assert record["full_history_safe"] is False


def test_fingerprinted_kmia_single_report_qc_contradiction_is_explained_and_excluded():
    # Break caught: the known KMIA anomaly is either not excluded, or is
    # explained without retaining the exact CLI and offending-METAR evidence.
    target = replay_eval.remaining_heat_record("KMIA", "2021-07-03", 13, 94, (96.0, 96.2))
    record = replay_eval.resolve_negative_provenance(
        target,
        {"product": "202107040840-KMFL-CDUS42-CLIMIA",
         "raw_sha256": "115076c962758dd72636127b845dd013ac453d4ef80657001411ef08e1a40912",
         "raw_payload": {"high": 94, "high_time": "200 PM"}},
        {"metar_rows": [{
            "station": "KMIA", "source": "metar", "valid_utc": "2021-07-03T16:53:00+00:00",
            "raw_sha256": "e7b429e9b8e3ab946a864773ea612735e12d55c2bf6b42f954adf25268414de2",
            "tmpf_precise": 96.08, "six_max_c": None,
        }]},
    )
    assert record["explained"] is True
    assert record["provenance_status"] == "METAR_SINGLE_REPORT_QC_CONTRADICTION"
    assert record["full_history_safe"] is True
    assert record["exclusion"]["reason"] == "METAR_SINGLE_REPORT_QC_CONTRADICTION"
    assert record["resolution_evidence"]["cli"]["high_time"] == "200 PM"
    assert record["resolution_evidence"]["offending_metar"]["raw_sha256"] == (
        "e7b429e9b8e3ab946a864773ea612735e12d55c2bf6b42f954adf25268414de2"
    )


@pytest.mark.parametrize("changed", [
    ("product", "202107040600-KMFL-CDUS42-CLIMIA"),
    ("cli_raw_sha256", "not-the-final-cli-raw-hash"),
    ("metar_raw_sha256", "not-the-offending-metar-raw-hash"),
    ("station", "KLAX"),
    ("date", "2021-07-04"),
    ("cli_high_f", 95),
    ("visible_floor_f", 95),
    ("remaining_heat_f", -1),
    ("invariant_ok", True),
])
def test_negative_provenance_resolution_fails_closed_when_any_fingerprint_key_changes(changed):
    # Break caught: a broad date/station exception can suppress an unrelated
    # negative target after the CLI product or METAR source evidence changes.
    field, value = changed
    target = replay_eval.remaining_heat_record("KMIA", "2021-07-03", 13, 94, (96.0, 96.2))
    cli_truth = {
        "product": "202107040840-KMFL-CDUS42-CLIMIA",
        "raw_sha256": "115076c962758dd72636127b845dd013ac453d4ef80657001411ef08e1a40912",
        "raw_payload": {"high": 94, "high_time": "200 PM"},
    }
    lineage = {"metar_rows": [{
        "station": "KMIA", "source": "metar", "valid_utc": "2021-07-03T16:53:00+00:00",
        "raw_sha256": "e7b429e9b8e3ab946a864773ea612735e12d55c2bf6b42f954adf25268414de2",
        "tmpf_precise": 96.08, "six_max_c": None,
    }]}
    if field == "product":
        cli_truth["product"] = value
    elif field == "cli_raw_sha256":
        cli_truth["raw_sha256"] = value
    elif field == "metar_raw_sha256":
        lineage["metar_rows"][0]["raw_sha256"] = value
    else:
        target[field] = value

    record = replay_eval.resolve_negative_provenance(target, cli_truth, lineage)

    assert record["explained"] is False
    assert record["provenance_status"] == "UNRESOLVED_CLI_METAR_CONTRADICTION"
    assert record["full_history_safe"] is False


def test_offline_boundary_audit_rejects_trading_imports_and_live_capture_readers(tmp_path):
    # Break caught: the Phase 0 experiment gains an import path to order
    # placement, or a live module starts consuming capture-only Lane B fields.
    safe = tmp_path / "safe.py"
    safe.write_text("from highcast import buckets\n", encoding="utf-8")
    assert replay_eval.audit_offline_boundaries([safe], live_paths=[]) == []
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text("from highcast import orderbot\n", encoding="utf-8")
    live = tmp_path / "live.py"
    live.write_text("x = row['future_rain_prob_max']\n", encoding="utf-8")
    errors = replay_eval.audit_offline_boundaries([unsafe], live_paths=[live])
    assert {error["kind"] for error in errors} == {"forbidden_import", "capture_only_live_reader"}


def test_results_artifact_schema_requires_machine_readable_gate_evidence():
    # Break caught: a human report says PASS while the machine artifact omits a
    # gate input, slice, safeguard, or exclusion trail.
    doc = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "rows": [_result_row()]}
    assert replay_eval.validate_results_document(doc) is doc
    broken = json.loads(json.dumps(doc))
    del broken["rows"][0]["noise_delta_hit_p975"]
    with pytest.raises(ValueError, match="noise_delta_hit_p975"):
        replay_eval.validate_results_document(broken)


def test_run_cli_checks_audit_before_creating_a_ledger(tmp_path, capsys):
    # Break caught: the run command creates partial state before discovering
    # that its prerequisite audit manifest does not exist.
    rc = replay_eval.main([
        "run", "--stations", "KMIA", "--hours", "9", "--bundles", "B0,B1",
        "--test-start", "2025-01-01", "--refit-every", "45", "--bootstrap", "100",
        "--out", str(tmp_path),
    ])
    assert rc != 0
    assert "audit" in capsys.readouterr().err.lower()
    assert not (tmp_path / "replay.db").exists()


def test_station_row_kill_gate_does_not_use_the_sparsest_candidate_bundle():
    # Break caught: sparse rain-event B4 values kill an otherwise feasible
    # station by being substituted for the station/hour matched truth-row count.
    criteria = replay_eval.station_kill_criteria(
        test_rows_by_hour={9: 596, 10: 596, 11: 596, 12: 596, 13: 596},
        bundle_coverages={
            "B1": [0.99] * 5, "B2": [0.99] * 5, "B3": [1.0] * 5,
            "B4": [0.17, 0.18, 0.20, 0.22, 0.24],
        },
        unexplained_negative_count=0, audited_target_count=2980,
    )
    assert criteria == {
        "negative_rate_over_0_5pct": False,
        "fewer_than_400_at_three_or_more_hours": False,
        "all_lane_a_bundles_below_80pct": False,
        "random_controls_repeatedly_clear_threshold": None,
    }


def test_evaluation_ledger_retains_canonical_checkpoint_and_metar_row_lineage(tmp_path):
    # Break caught: the evaluator hashes then discards its complete canonical
    # checkpoint payload and snapshots retain no underlying METAR row identities.
    reproducibility = replay_eval.reproducibility_document(["KMIA"])
    writer = replay_eval._LedgerWriter(
        tmp_path / "replay.db", tmp_path / "raw",
        {"experiment": "test", "spec_version": "phase-0", "reproducibility": reproducibility},
        "run-test", "code", "config",
    )
    row = {
        "target_date": "2025-01-01", "hour": 9.0,
        "asof_utc": "2025-01-01T14:00:00+00:00", "visible_floor_f": 80.0,
        "y": 84.0, "remaining_heat_f": 4.0, "negative_remaining_heat": False,
        "x": 1.0, "lane_lineage": {"metar_rows": [
            {"station": "KMIA", "source": "metar",
             "valid_utc": "2025-01-01T13:53:00+00:00"},
        ]},
        "cli_product_utc": "2025-01-02T06:00:00+00:00",
        "cli_raw_sha256": "f" * 64,
        "train_cutoff_date": "2025-01-01",
    }
    prediction = {"point_f": 84.0, "quantiles": {0.5: 84.0}, "pmf": {84: 1.0}}
    try:
        import pandas as pd
        retained = writer.retain_fold_training(
            "KMIA", "fold", "B0", "remaining_heat_B0", ["x"], pd.DataFrame([row]),
            "2025-01-01", reproducibility,
        )
        writer.write("KMIA", row, "B0", ["x"], prediction, "fold", 500,
                     "model", "feature", retained)
        writer.commit()
    finally:
        writer.close()
    conn = sqlite3.connect(tmp_path / "replay.db")
    try:
        raw_path, raw_sha = conn.execute(
            "SELECT raw_path, raw_sha256 FROM replay_source_events "
            "WHERE source='canonical_legacy_checkpoint'"
        ).fetchone()
        quantiles = json.loads(conn.execute(
            "SELECT quantiles_json FROM replay_predictions"
        ).fetchone()[0])
        lineage = json.loads(conn.execute(
            "SELECT lineage_json FROM replay_feature_snapshots"
        ).fetchone()[0])
    finally:
        conn.close()
    assert raw_path is not None
    assert len(raw_sha) == 64
    assert quantiles == {"0.5": 84.0}
    raw_file = tmp_path / "raw" / raw_path
    payload = json.loads(gzip.open(raw_file, "rb").read())
    assert payload["metar_lineage"] == row["lane_lineage"]
    assert lineage["metar_rows"] == row["lane_lineage"]["metar_rows"]
    assert replay_eval.verify_ledger(tmp_path / "replay.db", tmp_path / "raw")["ok"] is False
    # The run has not emitted its append-only completion event yet; raw integrity
    # itself must nevertheless be clean.
    integrity = replay_eval.verify_ledger(tmp_path / "replay.db", tmp_path / "raw")
    assert integrity["missing_raw_paths"] == []
    assert integrity["raw_sha256_mismatches"] == []


def test_snapshot_lineage_enforces_run_and_station_at_append_time():
    # Break caught: a KMIA/run-1 snapshot can cite KLAX or run-2 evidence.
    conn = _conn()
    _run(conn)
    _run(conn, run_id="run-2")
    _source(conn, event_id="wrong-run", run_id="run-2")
    with pytest.raises(ValueError, match="run_id"):
        _snapshot(conn, lineage_json={"source_event_ids": ["wrong-run"]})
    _source(conn, event_id="wrong-station", station="KLAX")
    with pytest.raises(ValueError, match="station"):
        _snapshot(conn, lineage_json={"source_event_ids": ["wrong-station"]})


def test_final_outcome_is_reused_and_conflicting_revision_must_supersede():
    # Break caught: every rerun appends another independent final truth for the
    # same station/date because its outcome id embeds the run id.
    conn = _conn()
    first = {"outcome_id": "first", "station": "KMIA", "target_date": "2026-08-26",
             "cli_high_f": 88, "product_utc": "2026-08-27T06:00:00+00:00",
             "recorded_utc": "2026-08-27T06:01:00+00:00", "raw_sha256": "a",
             "is_final": True}
    assert append_outcome(conn, first) == "first"
    assert append_outcome(conn, {**first, "outcome_id": "rerun-copy"}) == "first"
    assert conn.execute("SELECT count(*) FROM replay_outcomes").fetchone()[0] == 1
    with pytest.raises(ValueError, match="supersede|active outcome"):
        append_outcome(conn, {**first, "outcome_id": "conflict", "cli_high_f": 89,
                              "raw_sha256": "b"})
    assert append_outcome(conn, {**first, "outcome_id": "revision", "cli_high_f": 89,
                                 "raw_sha256": "b", "supersedes_id": "first"}) == "revision"


def test_append_only_run_completion_is_required_by_integrity_verification(tmp_path):
    # Break caught: an immutable_started run with completed_utc NULL is certified
    # healthy even though no append-only terminal lifecycle event exists.
    db_path = tmp_path / "replay.db"
    conn = sqlite3.connect(db_path)
    ensure_replay_schema(conn)
    _run(conn)
    conn.commit()
    assert verify_ledger(db_path, tmp_path / "raw")["run_lifecycle_errors"] == [
        {"run_id": "run-1", "reason": "missing_terminal_event"}
    ]
    replay_eval.append_run_event(conn, {
        "run_event_id": "run-1-complete", "run_id": "run-1",
        "event_type": "completed", "event_utc": "2026-08-26T01:00:00+00:00",
        "status": "completed", "error_text": None,
    })
    conn.commit()
    conn.close()
    report = verify_ledger(db_path, tmp_path / "raw")
    assert report["run_lifecycle_errors"] == []
    assert report["ok"] is True


def test_reproducibility_fingerprint_closes_code_model_and_dependency_inputs():
    # Break caught: predictions keep the same fingerprint when imported model,
    # PMF, baseline, feature, or dependency behavior changes.
    document = replay_eval.reproducibility_document(["KMIA"])
    required_files = {
        "scripts/earlier_high_eval.py", "src/highcast/weather_features.py",
        "src/highcast/models/intraday.py", "src/highcast/models/baseline.py",
        "src/highcast/models/__init__.py", "src/highcast/features.py",
        "src/highcast/buckets.py", "src/highcast/metrics.py",
        "src/highcast/latentmax.py", "src/highcast/timeutil.py",
    }
    assert required_files.issubset(document["code_files_sha256"])
    assert {"python", "numpy", "pandas", "lightgbm", "scikit-learn"}.issubset(
        document["dependency_versions"]
    )
    assert document["model_spec"]["quantile_levels"] == [round(0.05 * i, 2) for i in range(1, 20)]
    assert document["model_spec"]["lgbm_parameters"]["n_estimators"] == 300
    first = replay_eval._model_fingerprint(["x"], "2025-01-01", 400, "closure-a", "remaining_heat")
    second = replay_eval._model_fingerprint(["x"], "2025-01-01", 400, "closure-b", "remaining_heat")
    assert first != second


def test_fold_training_payload_is_retained_linked_and_bound_to_prediction_fingerprint(tmp_path):
    # Break caught: fold fitting rows are discarded, so a prediction links only
    # to its own checkpoint and cannot be replayed into the fitted matrix.
    import pandas as pd

    reproducibility = replay_eval.reproducibility_document(["KMIA"])
    writer = replay_eval._LedgerWriter(
        tmp_path / "replay.db", tmp_path / "raw",
        {"experiment": "test", "spec_version": "phase-0", "reproducibility": reproducibility},
        "run-training", "code", "config",
    )
    training = pd.DataFrame([
        {"target_date": "2024-12-30", "hour": 8.0,
         "asof_utc": "2024-12-30T13:00:00+00:00", "x": 1.0,
         "y": 83.0, "remaining_heat_f": 3.0},
        {"target_date": "2024-12-31", "hour": 9.0,
         "asof_utc": "2024-12-31T14:00:00+00:00", "x": 2.0,
         "y": 85.0, "remaining_heat_f": 4.0},
    ])
    prediction_row = {
        "target_date": "2025-01-01", "hour": 9.0,
        "asof_utc": "2025-01-01T14:00:00+00:00", "visible_floor_f": 80.0,
        "x": 3.0, "y": 86.0, "remaining_heat_f": 6.0,
        "negative_remaining_heat": False, "lane_lineage": {"metar_rows": []},
        "cli_product_utc": "2025-01-02T06:00:00+00:00",
        "cli_raw_sha256": "f" * 64, "train_cutoff_date": "2025-01-01",
    }
    prediction = {"point_f": 86.0, "quantiles": {0.5: 86.0}, "pmf": {86: 1.0}}
    try:
        retained = writer.retain_fold_training(
            "KMIA", "fold-1", "B1", "remaining_heat_B1", ["x"], training,
            "2025-01-01", reproducibility,
        )
        fingerprint = replay_eval._model_fingerprint(
            ["x"], "2025-01-01", 2, reproducibility,
            "remaining_heat_B1", retained["training_content_sha256"],
        )
        writer.write("KMIA", prediction_row, "B1", ["x"], prediction, "fold-1", 2,
                     fingerprint, "feature", retained)
        writer.finish(False)
    finally:
        writer.close()

    conn = sqlite3.connect(tmp_path / "replay.db")
    try:
        source_id, raw_path = conn.execute(
            "SELECT source_event_id, raw_path FROM replay_source_events "
            "WHERE source='canonical_fold_training'"
        ).fetchone()
        lineage = json.loads(conn.execute(
            "SELECT lineage_json FROM replay_feature_snapshots"
        ).fetchone()[0])
    finally:
        conn.close()
    payload = json.loads(gzip.open(tmp_path / "raw" / raw_path, "rb").read())
    digest = payload.pop("training_content_sha256")
    assert digest == retained["training_content_sha256"] == sha256_json(payload)
    assert payload["rows"] == [
        {"identity": {"asof_utc": "2024-12-30T13:00:00+00:00", "checkpoint_hour": 8.0,
                      "station": "KMIA", "target_date": "2024-12-30"},
         "inputs": {"x": 1.0}, "target": 3.0, "weight_inputs": {"target_date": "2024-12-30"}},
        {"identity": {"asof_utc": "2024-12-31T14:00:00+00:00", "checkpoint_hour": 9.0,
                      "station": "KMIA", "target_date": "2024-12-31"},
         "inputs": {"x": 2.0}, "target": 4.0, "weight_inputs": {"target_date": "2024-12-31"}},
    ]
    assert source_id in lineage["source_event_ids"]
    assert lineage["training_corpus"] == retained
    assert replay_eval.verify_ledger(tmp_path / "replay.db", tmp_path / "raw")["ok"] is True


def _write_shared_training_ledger(tmp_path, prediction_count=5):
    """Test ledger with many predictions bound to one immutable fit corpus."""
    import pandas as pd

    db_path, raw_root = tmp_path / "replay.db", tmp_path / "raw"
    reproducibility = replay_eval.reproducibility_document(["KMIA"])
    manifest = {
        "experiment": "test", "spec_version": "phase-0",
        "stations": ["KMIA"], "hours": [9], "test_start": "2025-01-01",
        "reproducibility": reproducibility,
    }
    writer = replay_eval._LedgerWriter(
        db_path, raw_root, manifest, "run-training-cache",
        reproducibility["closure_sha256"], "config",
    )
    training = pd.DataFrame([
        {"target_date": "2024-12-30", "hour": 9.0,
         "asof_utc": "2024-12-30T14:00:00+00:00", "x": 1.0,
         "y": 83.0, "remaining_heat_f": 3.0},
        {"target_date": "2024-12-31", "hour": 9.0,
         "asof_utc": "2024-12-31T14:00:00+00:00", "x": 2.0,
         "y": 85.0, "remaining_heat_f": 4.0},
    ])
    retained = writer.retain_fold_training(
        "KMIA", "fold-shared", "B1", "remaining_heat_B1", ["x"], training,
        "2025-01-01", reproducibility,
    )
    fingerprint = replay_eval._model_fingerprint(
        ["x"], "2025-01-01", len(training), reproducibility,
        "remaining_heat_B1", retained["training_content_sha256"],
    )
    prediction = {"point_f": 86.0, "quantiles": {0.5: 86.0}, "pmf": {86: 1.0}}
    try:
        for index in range(prediction_count):
            target_date = f"2025-01-{index + 1:02d}"
            row = {
                "target_date": target_date, "hour": 9.0,
                "asof_utc": f"{target_date}T14:00:00+00:00", "visible_floor_f": 80.0,
                "x": float(index), "y": 86.0, "remaining_heat_f": 6.0,
                "negative_remaining_heat": False, "lane_lineage": {"metar_rows": []},
                "cli_product_utc": f"{target_date}T23:00:00+00:00",
                "cli_raw_sha256": f"{index + 1:064x}",
                "train_cutoff_date": "2025-01-01",
            }
            writer.write(
                "KMIA", row, "B1", ["x"], prediction, "fold-shared", len(training),
                fingerprint, "feature", retained,
            )
        writer.finish(False)
    finally:
        writer.close()
    return db_path, raw_root, manifest, retained, fingerprint


def test_training_verifier_opens_one_gzip_for_many_predictions_sharing_one_source(
        tmp_path, monkeypatch):
    # Break caught: verifier decompresses/reparses the same fold corpus once per
    # prediction, making verification scale with predictions rather than fits.
    db_path, raw_root, _, _, _ = _write_shared_training_ledger(
        tmp_path, prediction_count=7,
    )
    real_open, opened = replay_eval.gzip.open, []

    def counting_open(*args, **kwargs):
        opened.append(Path(args[0]))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(replay_eval.gzip, "open", counting_open)
    report = replay_eval.verify_ledger(db_path, raw_root)
    assert report["ok"] is True
    assert len(opened) == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("station", "station"),
        ("run", "run"),
        ("arm", "arm"),
        ("train_row_count", "row count"),
        ("model_fingerprint", "fingerprint"),
    ],
)
def test_cached_training_corpus_still_validates_every_prediction_identity(
        tmp_path, monkeypatch, mutation, reason):
    # Break caught: caching a valid corpus accidentally trusts a later prediction
    # without rechecking its run/station/arm/count/model binding.
    db_path, raw_root, manifest, retained, fingerprint = _write_shared_training_ledger(
        tmp_path, prediction_count=1,
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    run_id, station, arm = "run-training-cache", "KMIA", "B1"
    if mutation == "run":
        run_id = "run-cross"
        append_run(conn, {
            "run_id": run_id, "experiment": "test", "spec_version": "phase-0",
            "started_utc": "2026-08-26T00:00:00+00:00", "mode": "backtest",
            "code_fingerprint": "code", "config_sha256": "config", "schema_version": 1,
            "manifest_json": manifest, "status": "started",
        })
        replay_eval.append_run_event(conn, {
            "run_event_id": "run-cross-complete", "run_id": run_id,
            "event_type": "completed", "event_utc": "2026-08-26T01:00:00+00:00",
            "status": "completed", "error_text": None,
        })
    if mutation == "station":
        station = "KLAX"
    if mutation == "arm":
        arm = "B2"
    source = conn.execute(
        "SELECT known_utc FROM replay_source_events WHERE source_event_id=?",
        (retained["source_event_id"],),
    ).fetchone()
    lineage = {
        "source_event_ids": [retained["source_event_id"]],
        "metar_rows": [], "training_corpus": retained,
    }
    conn.execute(
        """INSERT INTO replay_feature_snapshots VALUES
           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("snapshot-mutated", run_id, station, "2025-01-02", 9.0,
         "2025-01-02T14:00:00+00:00", arm, 1, "feature", '{"x":3.0}',
         '{"x":false}', canonical_json(lineage).decode("utf-8"), 80.0,
         source["known_utc"], "2025-01-02T14:00:01+00:00"),
    )
    conn.execute(
        """INSERT INTO replay_predictions VALUES
           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("prediction-mutated", run_id, "snapshot-mutated", station, "2025-01-02", 9.0,
         "2025-01-02T14:00:02+00:00", arm, "fold-shared",
         "wrong" if mutation == "model_fingerprint" else fingerprint,
         "2025-01-01", 999 if mutation == "train_row_count" else 2, 86.0,
         '{"0.5":86.0}', '{"86":1.0}', "2025-01-02T14:00:03+00:00"),
    )
    conn.commit()
    conn.close()
    real_open, opened = replay_eval.gzip.open, []

    def counting_open(*args, **kwargs):
        opened.append(Path(args[0]))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(replay_eval.gzip, "open", counting_open)
    report = replay_eval.verify_ledger(db_path, raw_root)
    assert report["ok"] is False
    assert any(reason in item["reason"] for item in report["training_lineage_errors"])
    assert len(opened) == 1


def test_model_fingerprint_changes_for_training_content_and_declares_actual_fit_semantics():
    # Break caught: different fitted matrices or shipped unweighted intraday
    # semantics can share a fingerprint that falsely claims DayAhead weighting.
    document = replay_eval.reproducibility_document(["KMIA"])
    first = replay_eval._model_fingerprint(
        ["x"], "2025-01-01", 400, document, "remaining_heat_B1", "a" * 64,
    )
    changed_training = replay_eval._model_fingerprint(
        ["x"], "2025-01-01", 400, document, "remaining_heat_B1", "b" * 64,
    )
    shipped = replay_eval.model_fingerprint_payload(
        ["x"], "2025-01-01", 400, document, "shipped_direct_high_intraday", "a" * 64,
    )
    assert first != changed_training
    assert shipped["training_content_sha256"] == "a" * 64
    assert shipped["factory_code_closure_sha256"]
    assert shipped["dependency_versions_sha256"]
    assert shipped["quantile_pmf_bucket_code_sha256"]
    assert document["model_spec"]["shipped_direct_high"]["sample_weights"] is None
    assert "unweighted" in document["model_spec"]["shipped_direct_high"]["fit_semantics"]


def test_integrity_failure_invalidates_machine_results_and_pass_recommendations():
    # Break caught: results.json publishes a PASS/recommendation before a failed
    # raw or lineage verification can mark the run corrupt.
    document = {
        "schema_version": 1, "run_id": "run-corrupt", "rows": [_result_row()],
        "recommendations": [{"station": "KMIA", "hour": 9, "bundle": "B1"}],
        "candidate_errors": [],
    }
    finalized = replay_eval.finalize_verified_results(document, {
        "ok": False, "missing_raw_paths": ["canonical/fold.json.gz"],
    })
    assert finalized["status"] == "CORRUPT"
    assert finalized["integrity_ok"] is False
    assert finalized["recommendations"] == []
    assert {row["verdict"] for row in finalized["rows"]} == {"ERROR"}
    assert finalized["corruption_reasons"] == ["missing_raw_paths"]


def test_feature_manifest_has_exact_b0_b5_fields_and_klax_3000ft_threshold():
    # Break caught: placeholders and field_specific metadata cannot reconstruct a
    # feature, and KLAX B4 omits the pre-registered clearing threshold.
    manifest = replay_eval._feature_manifest(["KLAX"], {"KLAX": "2000-01-01"})
    required = {"source_table", "formula", "units", "allowable_lookback",
                "missingness_policy", "earliest_trustworthy_date"}
    assert all(required.issubset(item) for item in manifest)
    assert not any("existing" in str(item[key]).lower() or "field_specific" in str(item[key])
                   for item in manifest for key in required if item[key] is not None)
    b5 = [item for item in manifest if item["bundle"] == "B5"]
    assert len({item["formula"] for item in b5}) == len(b5)
    assert len({item["units"] for item in b5}) > 3
    klax_b4 = [item for item in manifest if item["bundle"] == "B4"]
    assert all("3000" in f"{item['formula']} {item['allowable_lookback']}" for item in klax_b4)


def test_hot_day_threshold_weights_each_training_outcome_date_once():
    # Break caught: a day with more checkpoint rows receives more percentile
    # weight than a day with fewer available checkpoint rows.
    import pandas as pd

    frame = pd.DataFrame({
        "target_date": ["2024-01-01"] * 10 + ["2024-01-02"],
        "y": [50.0] * 10 + [100.0],
    })
    assert replay_eval.hot_day_threshold(frame, "2025-01-01") == pytest.approx(90.0)


def test_failed_empty_arm_does_not_emit_nonfinite_percentile_warning():
    # Break caught: an independently failed candidate/noise arm feeds -Infinity
    # to numpy.percentile, polluting the command output before it can report ERROR.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row = replay_eval._evaluate_comparison(
            "KMIA", 9, "B1", [], [], {1729: [], 2718: [], 3141: []},
            0.0, 100, [{"reason": "candidate failed"}], 90.0,
        )
    assert caught == []
    assert row["noise_delta_hit_p975"] is None
