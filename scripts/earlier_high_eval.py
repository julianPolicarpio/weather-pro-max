#!/usr/bin/env python3
"""Offline append-only replay ledger for the Earlier HIGH Phase 0 experiment.

This module intentionally owns an experiment-local schema.  It never imports the
production database schema or writes to a live application path.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MINIMUM_TRAINING_ROWS = 400
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("data/experiments/earlier_high")
DEFAULT_RAW_ROOT = Path("data/replay/raw")
RANDOM_SEEDS = (1729, 2718, 3141)
ALLOWED_STATIONS = ("KMIA", "KLAX")
ALLOWED_HOURS = (9, 10, 11, 12, 13)
ALLOWED_BUNDLES = ("B0", "B1", "B2", "B3", "B4")
MAX_CANONICAL_FOLD_TRAINING_ROWS = 10_000
NEGATIVE_PROVENANCE_RESOLUTION_REGISTRY = (
    {
        "resolution_id": "KMIA_2021-07-03_H13_METAR_SINGLE_REPORT_QC_CONTRADICTION",
        "provenance_status": "METAR_SINGLE_REPORT_QC_CONTRADICTION",
        "target": {
            "station": "KMIA", "date": "2021-07-03", "hour": 13,
            "cli_high_f": 94, "visible_floor_f": 96, "remaining_heat_f": -2,
            "invariant_ok": False,
        },
        "cli": {
            "product": "202107040840-KMFL-CDUS42-CLIMIA",
            "raw_sha256": "115076c962758dd72636127b845dd013ac453d4ef80657001411ef08e1a40912",
        },
        "offending_metar": {
            "station": "KMIA", "source": "metar", "valid_utc": "2021-07-03T16:53:00+00:00",
            "raw_sha256": "e7b429e9b8e3ab946a864773ea612735e12d55c2bf6b42f954adf25268414de2",
        },
    },
)
RESULT_REQUIRED = (
    "station", "hour", "bundle", "n", "coverage", "bucket_hit", "delta_hit",
    "delta_hit_ci", "crps", "delta_crps", "noise_delta_hit_p975", "verdict",
    "safeguards", "slices", "exclusions", "holm_adjusted_p",
    "random_family_contaminated", "shipped_incumbent_comparison",
)
CAPTURE_ONLY_FIELDS = {
    "future_rain_prob_max", "future_rain_sum_mm", "future_cloud_low_mean",
    "future_shortwave_sum", "future_rh_mean", "future_gust_max",
    "future_onshore_mean", "first_solar_threshold_hour", "first_rain_hour",
}
REPLAY_TABLES = (
    "replay_runs",
    "replay_run_events",
    "replay_source_events",
    "replay_feature_snapshots",
    "replay_predictions",
    "replay_outcomes",
)


def settlement_floor_from_visible(visible: Sequence[float]) -> int:
    """Settlement-safe HIGH floor from the conservative interval edge.

    The lower edge is the weakest claim supported by the observation interval;
    NWS settlement uses half-up rounding rather than Python's bankers' rounding.
    """
    if visible is None or len(visible) != 2:
        raise ValueError("visible running maximum must be a two-value interval")
    from highcast.timeutil import round_half_up
    return int(round_half_up(float(visible[0])))


def remaining_heat_record(station: str, target_date: str, hour: int,
                          cli_high_f: int | float, visible: Sequence[float]) -> dict[str, Any]:
    floor = settlement_floor_from_visible(visible)
    remaining = int(cli_high_f) - floor
    return {
        "station": station, "date": target_date, "hour": int(hour),
        "cli_high_f": int(cli_high_f),
        "visible_interval_f": [float(visible[0]), float(visible[1])],
        "visible_floor_f": floor, "remaining_heat_f": remaining,
        "invariant_ok": remaining >= 0,
    }


def reconstruct_high_quantiles(visible_floor_f: float,
                               remaining_quantiles: Mapping[float, float]) -> dict[float, float]:
    """Restore the HIGH level, floor at the ratchet, and remove crossings."""
    levels = sorted(float(level) for level in remaining_quantiles)
    restored = [float(visible_floor_f) + max(0.0, float(remaining_quantiles[level]))
                for level in levels]
    restored.sort()
    return {level: value for level, value in zip(levels, restored)}


def match_arm_records(arms: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Restrict every arm to its common target dates, preserving sorted order."""
    if not arms:
        return {"dates": [], "arms": {}}
    indexed = {
        label: {str(row["target_date"]): dict(row) for row in records}
        for label, records in arms.items()
    }
    date_sets = [set(rows) for rows in indexed.values()]
    dates = sorted(set.intersection(*date_sets)) if date_sets else []
    return {"dates": dates,
            "arms": {label: [rows[target_date] for target_date in dates]
                     for label, rows in indexed.items()}}


def paired_bootstrap(deltas: Sequence[float], n_boot: int, seed: int,
                     interval: tuple[float, float] = (2.5, 97.5)) -> dict[str, Any]:
    """Deterministic paired mean bootstrap with a finite-sample two-sided p."""
    import numpy as np
    values = np.asarray(list(deltas), dtype=float)
    if len(values) == 0:
        return {"mean": None, "ci": [None, None], "p_two_sided": None,
                "samples": []}
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    samples = values[indexes].mean(axis=1)
    negative = (float(np.count_nonzero(samples <= 0)) + 1.0) / (len(samples) + 1.0)
    positive = (float(np.count_nonzero(samples >= 0)) + 1.0) / (len(samples) + 1.0)
    return {
        "mean": float(values.mean()),
        "ci": [float(np.percentile(samples, interval[0])),
               float(np.percentile(samples, interval[1]))],
        "p_two_sided": float(min(1.0, 2.0 * min(negative, positive))),
        "samples": [float(value) for value in samples],
    }


def deterministic_random_controls(row_keys: Sequence[str], column_count: int,
                                  seeds: Sequence[int] = RANDOM_SEEDS) -> dict[int, list[list[float]]]:
    """Noise columns stable under process order and unrelated global RNG use."""
    import numpy as np
    controls: dict[int, list[list[float]]] = {}
    for seed in seeds:
        rows = []
        for key in row_keys:
            digest = hashlib.sha256(f"{int(seed)}|{key}".encode("utf-8")).digest()
            row_seed = int.from_bytes(digest[:8], "big", signed=False)
            rows.append([float(value) for value in
                         np.random.default_rng(row_seed).standard_normal(int(column_count))])
        controls[int(seed)] = rows
    return controls


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down family-wise adjusted p-values in input order."""
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def validate_manifest_for_run(path: str | Path, stations: Sequence[str], hours: Sequence[int],
                              bundles: Sequence[str], feature_code_sha256: str,
                              config_sha256: Mapping[str, str],
                              reproducibility_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError("audit manifest absent; audit must complete before run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("audit_complete"):
        raise ValueError("audit manifest is incomplete; audit must complete before run")
    if manifest.get("feature_code_sha256") != feature_code_sha256:
        raise ValueError("audit manifest is stale relative to feature code")
    if (reproducibility_sha256 is not None
            and manifest.get("reproducibility", {}).get("closure_sha256") != reproducibility_sha256):
        raise ValueError("audit manifest is stale relative to experiment code/model/dependencies")
    recorded_configs = manifest.get("config_sha256", {})
    if any(recorded_configs.get(station) != config_sha256.get(station) for station in stations):
        raise ValueError("audit manifest is stale relative to station config")
    if not set(stations).issubset(set(manifest.get("stations", []))):
        raise ValueError("audit manifest does not cover every requested station")
    if not set(map(int, hours)).issubset(set(map(int, manifest.get("hours", [])))):
        raise ValueError("audit manifest does not cover every requested hour")
    statuses: dict[str, set[str]] = {}
    for feature in manifest.get("features", []):
        statuses.setdefault(str(feature.get("bundle")), set()).add(str(feature.get("status")))
    unsafe = {bundle: sorted(statuses.get(bundle, {"MISSING"}))
              for bundle in bundles
              if statuses.get(bundle, {"MISSING"}) != {"HISTORICAL_SAFE"}}
    if unsafe:
        raise ValueError(f"unsafe historical bundle selection: {unsafe}")
    unsafe_stations = [station for station in stations
                       if not manifest.get("station_scoring_safe", {}).get(station, False)]
    if unsafe_stations:
        raise ValueError(f"unsafe station invariant audit: {unsafe_stations}")
    return manifest


def verdict_for_result(row: Mapping[str, Any]) -> str:
    """Apply the exact pre-registered gate to one station/hour/bundle only."""
    n = int(row.get("n", 0) or 0)
    coverage = float(row.get("coverage", 0.0) or 0.0)
    if n < 400 or coverage < 0.80:
        return "INSUFFICIENT_DATA"
    def finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def comparison_passes(comparison: Mapping[str, Any], require_noise: bool) -> bool:
        interval = comparison.get("delta_hit_ci")
        safeguards = comparison.get("safeguards", comparison)
        slices = comparison.get("slices")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return False
        required_numbers = (
            comparison.get("delta_hit"), interval[0], interval[1], comparison.get("delta_crps"),
            comparison.get("holm_adjusted_p"), safeguards.get("calibration_gap_increase")
            if isinstance(safeguards, Mapping) else None,
        )
        if not all(finite_number(value) for value in required_numbers):
            return False
        if not isinstance(slices, Mapping):
            return False
        full = slices.get("full_history")
        warm = slices.get("warm_season")
        if not isinstance(full, Mapping) or not isinstance(warm, Mapping):
            return False
        full_delta, warm_delta = full.get("delta_hit"), warm.get("delta_hit")
        if not finite_number(full_delta) or not finite_number(warm_delta):
            return False
        checks = [
            float(comparison["delta_hit"]) > 0,
            float(interval[0]) > 0 and float(interval[1]) >= float(interval[0]),
            float(comparison["delta_crps"]) <= 0.01,
            float(safeguards["calibration_gap_increase"]) <= 0.02,
            not (float(full_delta) < 0 and float(warm_delta) < 0),
            float(comparison["holm_adjusted_p"]) <= 0.05,
        ]
        if require_noise:
            noise = comparison.get("noise_delta_hit_p975")
            checks.append(finite_number(noise) and float(comparison["delta_hit"]) > float(noise))
        return all(checks)

    if "random_family_contaminated" not in row or not isinstance(row["random_family_contaminated"], bool):
        return "REJECT"
    shipped = row.get("shipped_incumbent_comparison")
    if not isinstance(shipped, Mapping) or int(shipped.get("n", 0) or 0) < 400:
        return "REJECT"
    bundle_comparison = {
        "delta_hit": row.get("delta_hit"), "delta_hit_ci": row.get("delta_hit_ci"),
        "delta_crps": row.get("delta_crps"), "holm_adjusted_p": row.get("holm_adjusted_p"),
        "noise_delta_hit_p975": row.get("noise_delta_hit_p975"),
        "safeguards": row.get("safeguards"), "slices": row.get("slices"),
    }
    return "PASS" if (
        comparison_passes(bundle_comparison, require_noise=True)
        and comparison_passes(shipped, require_noise=False)
        and not row["random_family_contaminated"]
    ) else "REJECT"


def station_kill_criteria(test_rows_by_hour: Mapping[int, int],
                          bundle_coverages: Mapping[str, Sequence[float]],
                          unexplained_negative_count: int,
                          audited_target_count: int) -> dict[str, Any]:
    """Station-level feasibility gates, distinct from per-bundle coverage."""
    return {
        "negative_rate_over_0_5pct": (
            int(unexplained_negative_count) / max(1, int(audited_target_count))
        ) > 0.005,
        "fewer_than_400_at_three_or_more_hours": sum(
            int(count) < 400 for count in test_rows_by_hour.values()
        ) >= 3,
        "all_lane_a_bundles_below_80pct": all(
            all(float(coverage) < 0.80 for coverage in coverages)
            for coverages in bundle_coverages.values()
        ) if bundle_coverages else True,
        "random_controls_repeatedly_clear_threshold": None,
    }


def audit_offline_boundaries(experiment_paths: Sequence[str | Path],
                             live_paths: Sequence[str | Path]) -> list[dict[str, str]]:
    """Static safety audit for imports and capture-only reads at the boundary."""
    errors: list[dict[str, str]] = []
    forbidden = {"trading", "orderbot", "orders", "broker", "alpaca"}
    for path_value in experiment_paths:
        path = Path(path_value)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                leaf = module.split(".")[-1]
                if leaf in forbidden:
                    errors.append({"kind": "forbidden_import", "path": str(path),
                                   "detail": module})
    for path_value in live_paths:
        path = Path(path_value)
        text_value = path.read_text(encoding="utf-8")
        for field in sorted(CAPTURE_ONLY_FIELDS):
            if field in text_value:
                errors.append({"kind": "capture_only_live_reader", "path": str(path),
                               "detail": field})
    return errors


def validate_results_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != 1 or not isinstance(document.get("rows"), list):
        raise ValueError("results document requires schema_version=1 and rows")
    for index, row in enumerate(document["rows"]):
        for key in RESULT_REQUIRED:
            if key not in row:
                raise ValueError(f"results row {index} missing {key}")
        if row["verdict"] not in {"PASS", "REJECT", "INSUFFICIENT_DATA", "ERROR"}:
            raise ValueError(f"results row {index} has invalid verdict")
        shipped = row["shipped_incumbent_comparison"]
        if not isinstance(shipped, Mapping):
            raise ValueError(f"results row {index} shipped_incumbent_comparison must be an object")
        for key in ("name", "n", "delta_hit", "delta_hit_ci", "delta_crps",
                    "calibration_gap_increase", "slices", "holm_adjusted_p"):
            if key not in shipped:
                raise ValueError(f"results row {index} shipped comparator missing {key}")
        if row["verdict"] == "PASS" and verdict_for_result(row) != "PASS":
            raise ValueError(f"results row {index} claims PASS without complete gate evidence")
    return document


def finalize_verified_results(document: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only results document that may be published after verification.

    Ledger evidence remains append-only when verification fails, but a failed
    hash/lineage check revokes every machine-readable authorization derived from
    it.  In particular, a stale PASS must not survive merely because the CLI
    command will eventually return a nonzero status.
    """
    finalized = json.loads(canonical_json(_jsonable(document)))
    finalized["integrity_ok"] = bool(integrity.get("ok"))
    if integrity.get("ok"):
        finalized["status"] = "VERIFIED"
        finalized["corruption_reasons"] = []
        return validate_results_document(finalized)
    finalized["status"] = "CORRUPT"
    finalized["corruption_reasons"] = sorted(
        key for key, value in integrity.items()
        if key not in {"ok", "missing_database"} and bool(value)
    )
    if integrity.get("missing_database"):
        finalized["corruption_reasons"].append("missing_database")
    finalized["recommendations"] = []
    for row in finalized["rows"]:
        if row.get("verdict") == "PASS":
            row["verdict"] = "ERROR"
    return validate_results_document(finalized)


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON accepts finite numbers only (NaN/Infinity rejected)")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)


def canonical_json(value: Any) -> bytes:
    """Return the ledger's canonical UTF-8 JSON representation."""
    _reject_nonfinite(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        # JSON columns always retain canonical JSON rather than caller formatting.
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return canonical_json(value).decode("utf-8")
    return canonical_json(value).decode("utf-8")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must have an explicit UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: str) -> str:
    return _parse_utc(value).isoformat()


def _insert(conn: sqlite3.Connection, table: str, record: Mapping[str, Any], columns: tuple[str, ...]) -> None:
    values = [record.get(column) for column in columns]
    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})", values,
    )


def ensure_replay_schema(conn: sqlite3.Connection) -> None:
    """Install the isolated replay tables and their append-only enforcement."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS replay_runs (
          run_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, spec_version TEXT NOT NULL,
          started_utc TEXT NOT NULL, completed_utc TEXT, mode TEXT NOT NULL CHECK (mode IN ('backtest','forward_shadow')),
          code_fingerprint TEXT NOT NULL, config_sha256 TEXT NOT NULL, schema_version INTEGER NOT NULL,
          manifest_json TEXT NOT NULL, status TEXT NOT NULL, error_text TEXT
        );
        CREATE TABLE IF NOT EXISTS replay_run_events (
          run_event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
          event_type TEXT NOT NULL CHECK (event_type IN ('started','completed','failed')),
          event_utc TEXT NOT NULL, status TEXT NOT NULL, error_text TEXT,
          FOREIGN KEY (run_id) REFERENCES replay_runs(run_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_run_event_type
          ON replay_run_events (run_id, event_type);
        CREATE TABLE IF NOT EXISTS replay_source_events (
          source_event_id TEXT PRIMARY KEY, run_id TEXT, station TEXT NOT NULL, source TEXT NOT NULL,
          requested_utc TEXT, fetched_utc TEXT NOT NULL, known_utc TEXT NOT NULL, known_utc_basis TEXT NOT NULL,
          valid_from_utc TEXT, valid_to_utc TEXT, request_json TEXT, http_status INTEGER, raw_path TEXT,
          raw_sha256 TEXT NOT NULL, parser_version TEXT NOT NULL, supersedes_id TEXT,
          FOREIGN KEY (run_id) REFERENCES replay_runs(run_id),
          FOREIGN KEY (supersedes_id) REFERENCES replay_source_events(source_event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_replay_source_asof ON replay_source_events (station, source, fetched_utc);
        CREATE TABLE IF NOT EXISTS replay_feature_snapshots (
          feature_snapshot_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, station TEXT NOT NULL, target_date TEXT NOT NULL,
          checkpoint_hour REAL NOT NULL, asof_utc TEXT NOT NULL, bundle TEXT NOT NULL,
          feature_schema_version INTEGER NOT NULL, feature_code_sha256 TEXT NOT NULL, features_json TEXT NOT NULL,
          missing_json TEXT NOT NULL, lineage_json TEXT NOT NULL, visible_floor_f REAL NOT NULL,
          source_max_known_utc TEXT NOT NULL, created_utc TEXT NOT NULL,
          FOREIGN KEY (run_id) REFERENCES replay_runs(run_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_feature_identity ON replay_feature_snapshots
          (run_id, station, target_date, checkpoint_hour, bundle);
        CREATE TABLE IF NOT EXISTS replay_predictions (
          replay_prediction_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, feature_snapshot_id TEXT NOT NULL,
          station TEXT NOT NULL, target_date TEXT NOT NULL, checkpoint_hour REAL NOT NULL, issued_utc TEXT NOT NULL,
          bundle TEXT NOT NULL, fold_id TEXT NOT NULL, model_fingerprint TEXT NOT NULL, train_cutoff_date TEXT NOT NULL,
          train_row_count INTEGER NOT NULL, point_f REAL NOT NULL, quantiles_json TEXT NOT NULL,
          buckets_json TEXT NOT NULL, created_utc TEXT NOT NULL,
          FOREIGN KEY (run_id) REFERENCES replay_runs(run_id),
          FOREIGN KEY (feature_snapshot_id) REFERENCES replay_feature_snapshots(feature_snapshot_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_prediction_identity ON replay_predictions
          (run_id, station, target_date, checkpoint_hour, bundle, fold_id);
        CREATE TABLE IF NOT EXISTS replay_outcomes (
          outcome_id TEXT PRIMARY KEY, station TEXT NOT NULL, target_date TEXT NOT NULL, cli_high_f INTEGER NOT NULL,
          product_utc TEXT, recorded_utc TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
          is_final INTEGER NOT NULL CHECK (is_final IN (0,1)), supersedes_id TEXT,
          FOREIGN KEY (supersedes_id) REFERENCES replay_outcomes(outcome_id)
        );
        CREATE INDEX IF NOT EXISTS idx_replay_outcome_day ON replay_outcomes (station, target_date, recorded_utc);
        """
    )
    for table in REPLAY_TABLES:
        conn.executescript(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'append-only replay ledger: UPDATE rejected'); END;
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT, 'append-only replay ledger: DELETE rejected'); END;
            """
        )


def append_run_event(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    prepared = dict(record)
    event_type = str(prepared.get("event_type", ""))
    if event_type not in {"started", "completed", "failed"}:
        raise ValueError("run event_type must be started, completed, or failed")
    if conn.execute("SELECT 1 FROM replay_runs WHERE run_id=?", (prepared["run_id"],)).fetchone() is None:
        raise ValueError("run event references a missing replay run")
    if event_type in {"completed", "failed"} and conn.execute(
        "SELECT 1 FROM replay_run_events WHERE run_id=? AND event_type IN ('completed','failed')",
        (prepared["run_id"],),
    ).fetchone():
        raise ValueError("replay run already has an immutable terminal event")
    prepared["event_utc"] = _utc_text(str(prepared["event_utc"]))
    _insert(conn, "replay_run_events", prepared, (
        "run_event_id", "run_id", "event_type", "event_utc", "status", "error_text",
    ))
    return str(prepared["run_event_id"])


def append_run(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    prepared = dict(record)
    prepared["manifest_json"] = _json_text(prepared["manifest_json"])
    _insert(conn, "replay_runs", prepared, (
        "run_id", "experiment", "spec_version", "started_utc", "completed_utc", "mode", "code_fingerprint",
        "config_sha256", "schema_version", "manifest_json", "status", "error_text",
    ))
    append_run_event(conn, {
        "run_event_id": f"{prepared['run_id']}:started", "run_id": prepared["run_id"],
        "event_type": "started", "event_utc": prepared["started_utc"],
        "status": "started", "error_text": None,
    })
    return str(prepared["run_id"])


def _raw_bytes(raw_body: Any) -> bytes:
    if isinstance(raw_body, bytes):
        return raw_body
    if isinstance(raw_body, str):
        return raw_body.encode("utf-8")
    return canonical_json(raw_body)


def _raw_location(raw_root: Path, source: str, fetched_utc: str, source_event_id: str) -> tuple[str, Path]:
    moment = _parse_utc(fetched_utc)
    relative = Path(source) / f"{moment:%Y}" / f"{moment:%m}" / f"{moment:%d}" / f"{source_event_id}.json.gz"
    return relative.as_posix(), raw_root / relative


def _prepare_gzip_raw(raw_root: Path, source: str, fetched_utc: str, source_event_id: str, raw_body: Any) -> tuple[str, str, Path, Path]:
    """Write and fsync an unpromoted sibling gzip file for atomic rename."""
    relative, destination = _raw_location(raw_root, source, fetched_utc, source_event_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{source_event_id}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw_file:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as compressed:
                compressed.write(_raw_bytes(raw_body))
            raw_file.flush()
            os.fsync(raw_file.fileno())
        temp_path = Path(temp_name)
        return relative, sha256_bytes(temp_path.read_bytes()), temp_path, destination
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _source_known_time(record: Mapping[str, Any]) -> tuple[str, str, str | None, str | None]:
    source = str(record["source"]).lower()
    if "metar" in source:
        valid_utc = record.get("valid_utc") or record.get("valid_to_utc")
        if not valid_utc:
            raise ValueError("METAR source event requires valid_utc")
        known_utc = _utc_text(valid_utc)
        supplied = record.get("known_utc")
        if supplied is not None and _utc_text(supplied) != known_utc:
            raise ValueError("METAR known_utc must equal observation valid_utc")
        return known_utc, "observation_valid_time", record.get("valid_from_utc"), record.get("valid_to_utc") or known_utc
    if source in {"forecast", "forecast_guidance", "openmeteo", "open-meteo"} or "forecast" in source:
        available = record.get("model_run_utc") or record.get("available_utc")
        if not available:
            raise ValueError("forecast source event requires model_run_utc or available_utc")
        known_dt = max(_parse_utc(str(available)), _parse_utc(str(record["fetched_utc"])))
        known_utc = known_dt.isoformat()
        supplied = record.get("known_utc")
        if supplied is not None and _utc_text(supplied) != known_utc:
            raise ValueError("forecast known_utc must be later of availability and fetch time")
        return known_utc, "forecast_available_and_fetched", record.get("valid_from_utc"), record.get("valid_to_utc")
    if not record.get("known_utc") or not record.get("known_utc_basis"):
        raise ValueError("source event requires explicit known_utc and known_utc_basis")
    return _utc_text(str(record["known_utc"])), str(record["known_utc_basis"]), record.get("valid_from_utc"), record.get("valid_to_utc")


def _same_optional_timestamp(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return _utc_text(left) == _utc_text(right)


def _validate_source_supersedes(conn: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    supersedes_id = record.get("supersedes_id")
    if not supersedes_id:
        return
    parent = conn.execute(
        "SELECT station, source, valid_from_utc, valid_to_utc FROM replay_source_events WHERE source_event_id=?",
        (supersedes_id,),
    ).fetchone()
    if parent is None:
        raise ValueError("supersedes_id references a missing source event")
    same_identity = (
        parent[0] == record["station"] and parent[1] == record["source"]
        and _same_optional_timestamp(parent[2], record.get("valid_from_utc"))
        and _same_optional_timestamp(parent[3], record.get("valid_to_utc"))
    )
    if not same_identity:
        raise ValueError("supersedes_id must retain the same source logical identity")


def _promote_atomic_rename(temp_path: Path, destination: Path) -> None:
    """Publish a prepared same-directory raw file with one atomic rename."""
    os.replace(temp_path, destination)


def append_source_event(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    prepared = dict(record)
    known_utc, basis, valid_from, valid_to = _source_known_time(prepared)
    prepared["fetched_utc"] = _utc_text(str(prepared["fetched_utc"]))
    prepared["known_utc"], prepared["known_utc_basis"] = known_utc, basis
    prepared["valid_from_utc"], prepared["valid_to_utc"] = valid_from, valid_to
    if prepared.get("request_json") is not None:
        prepared["request_json"] = _json_text(prepared["request_json"])
    _validate_source_supersedes(conn, prepared)
    raw_body = prepared.pop("raw_body", None)
    raw_root = prepared.pop("raw_root", None)
    has_parsed_input = "parsed_input" in prepared
    parsed_input = prepared.pop("parsed_input", None)
    temp_path: Path | None = None
    destination: Path | None = None
    if raw_body is not None and raw_root is not None:
        raw_root_path = Path(raw_root)
        relative, target = _raw_location(
            raw_root_path, str(prepared["source"]), prepared["fetched_utc"], str(prepared["source_event_id"])
        )
        if conn.execute("SELECT 1 FROM replay_source_events WHERE source_event_id=?", (prepared["source_event_id"],)).fetchone():
            raise sqlite3.IntegrityError("duplicate source_event_id")
        if target.exists():
            raise FileExistsError(f"raw evidence already exists for source event: {relative}")
        prepared["raw_path"], prepared["raw_sha256"], temp_path, destination = _prepare_gzip_raw(
            raw_root_path, str(prepared["source"]), prepared["fetched_utc"], str(prepared["source_event_id"]), raw_body,
        )
    elif raw_root is None:
        if not has_parsed_input:
            raise ValueError("source event without retained raw requires explicit parsed_input")
        prepared["raw_path"] = None
        prepared["raw_sha256"] = sha256_json(parsed_input)
    else:
        raise ValueError("raw_root requires raw_body for retained raw evidence")
    columns = (
        "source_event_id", "run_id", "station", "source", "requested_utc", "fetched_utc", "known_utc",
        "known_utc_basis", "valid_from_utc", "valid_to_utc", "request_json", "http_status", "raw_path",
        "raw_sha256", "parser_version", "supersedes_id",
    )
    conn.execute("SAVEPOINT append_source_event")
    try:
        _insert(conn, "replay_source_events", prepared, columns)
        if temp_path is not None and destination is not None:
            _promote_atomic_rename(temp_path, destination)
    except Exception:
        conn.execute("ROLLBACK TO append_source_event")
        conn.execute("RELEASE append_source_event")
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        raise
    else:
        conn.execute("RELEASE append_source_event")
    return str(prepared["source_event_id"])


def _lineage_source_ids(lineage: Any) -> list[str]:
    if isinstance(lineage, Mapping):
        values = lineage.get("source_event_ids", [])
    else:
        values = lineage
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("lineage_json must contain a source_event_ids string list")
    return values


def append_feature_snapshot(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    prepared = dict(record)
    lineage_value = prepared["lineage_json"]
    if isinstance(lineage_value, str):
        lineage_value = json.loads(lineage_value)
    source_ids = _lineage_source_ids(lineage_value)
    asof = _parse_utc(str(prepared["asof_utc"]))
    source_rows = []
    for source_id in source_ids:
        row = conn.execute(
            "SELECT run_id, station, known_utc FROM replay_source_events WHERE source_event_id=?",
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"lineage references missing source event: {source_id}")
        if row[0] != prepared["run_id"]:
            raise ValueError("lineage source event run_id must match feature snapshot run_id")
        if row[1] != prepared["station"]:
            raise ValueError("lineage source event station must match feature snapshot station")
        known = _parse_utc(row[2])
        if known > asof:
            raise ValueError("future-known source lineage cannot appear in a feature snapshot")
        source_rows.append(known)
    metar_rows = lineage_value.get("metar_rows", []) if isinstance(lineage_value, Mapping) else []
    if not isinstance(metar_rows, list):
        raise ValueError("lineage_json metar_rows must be a list")
    for metar in metar_rows:
        if not isinstance(metar, Mapping):
            raise ValueError("lineage_json metar_rows entries must be objects")
        if metar.get("station") != prepared["station"]:
            raise ValueError("METAR row lineage station must match feature snapshot station")
        if metar.get("source") != "metar":
            raise ValueError("METAR row lineage source must be metar")
        if _parse_utc(str(metar.get("valid_utc"))) > asof:
            raise ValueError("future METAR row lineage cannot appear in a feature snapshot")
    if source_rows:
        maximum = max(source_rows).isoformat()
        if _utc_text(str(prepared["source_max_known_utc"])) != maximum:
            raise ValueError("source_max_known_utc does not match source lineage")
    prepared["features_json"] = _json_text(prepared["features_json"])
    prepared["missing_json"] = _json_text(prepared["missing_json"])
    prepared["lineage_json"] = _json_text(lineage_value)
    _insert(conn, "replay_feature_snapshots", prepared, (
        "feature_snapshot_id", "run_id", "station", "target_date", "checkpoint_hour", "asof_utc", "bundle",
        "feature_schema_version", "feature_code_sha256", "features_json", "missing_json", "lineage_json",
        "visible_floor_f", "source_max_known_utc", "created_utc",
    ))
    return str(prepared["feature_snapshot_id"])


def append_prediction(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    prepared = dict(record)
    snapshot = conn.execute(
        "SELECT run_id, station, target_date, checkpoint_hour, bundle FROM replay_feature_snapshots WHERE feature_snapshot_id=?",
        (prepared["feature_snapshot_id"],),
    ).fetchone()
    if snapshot is None:
        # Keep SQLite's FK failure as the authoritative constraint too.
        raise sqlite3.IntegrityError("prediction requires an immutable feature snapshot")
    expected = (prepared["run_id"], prepared["station"], prepared["target_date"], float(prepared["checkpoint_hour"]), prepared["bundle"])
    actual = (snapshot[0], snapshot[1], snapshot[2], float(snapshot[3]), snapshot[4])
    if actual != expected:
        raise ValueError("prediction identity must match its immutable feature snapshot")
    prepared["quantiles_json"] = _json_text(prepared["quantiles_json"])
    prepared["buckets_json"] = _json_text(prepared["buckets_json"])
    _insert(conn, "replay_predictions", prepared, (
        "replay_prediction_id", "run_id", "feature_snapshot_id", "station", "target_date", "checkpoint_hour",
        "issued_utc", "bundle", "fold_id", "model_fingerprint", "train_cutoff_date", "train_row_count",
        "point_f", "quantiles_json", "buckets_json", "created_utc",
    ))
    return str(prepared["replay_prediction_id"])


def append_outcome(conn: sqlite3.Connection, record: Mapping[str, Any]) -> str:
    active = conn.execute(
        """SELECT current.outcome_id, current.cli_high_f, current.product_utc,
                  current.raw_sha256, current.is_final
             FROM replay_outcomes current
             LEFT JOIN replay_outcomes revision ON revision.supersedes_id=current.outcome_id
            WHERE current.station=? AND current.target_date=?
              AND revision.outcome_id IS NULL
            ORDER BY current.recorded_utc""",
        (record["station"], record["target_date"]),
    ).fetchall()
    supersedes_id = record.get("supersedes_id")
    if supersedes_id:
        parent = conn.execute(
            "SELECT station, target_date FROM replay_outcomes WHERE outcome_id=?", (supersedes_id,)
        ).fetchone()
        if parent is None:
            raise ValueError("supersedes_id references a missing outcome")
        if (parent[0], parent[1]) != (record["station"], record["target_date"]):
            raise ValueError("supersedes_id must retain the same outcome logical identity")
        if supersedes_id not in {row[0] for row in active}:
            raise ValueError("supersedes_id must reference the active outcome revision")
    elif active:
        if len(active) == 1:
            prior = active[0]
            same_product = _same_optional_timestamp(prior[2], record.get("product_utc"))
            same_evidence = (
                int(prior[1]) == int(record["cli_high_f"])
                and same_product
                and prior[3] == record["raw_sha256"]
                and int(prior[4]) == int(bool(record["is_final"]))
            )
            if same_evidence:
                return str(prior[0])
        raise ValueError("logical identity already has an active outcome; append an explicit superseding revision")
    _insert(conn, "replay_outcomes", record, (
        "outcome_id", "station", "target_date", "cli_high_f", "product_utc", "recorded_utc", "raw_sha256",
        "is_final", "supersedes_id",
    ))
    return str(record["outcome_id"])


def _empty_verification_report() -> dict[str, Any]:
    return {
        "ok": False, "missing_database": None, "missing_tables": [], "foreign_key_errors": [],
        "missing_raw_paths": [], "raw_sha256_mismatches": [], "unsafe_raw_paths": [],
        "orphan_raw_paths": [],
        "duplicate_logical_events": [], "schema_version_errors": [], "immutable_snapshot_errors": [],
        "lineage_errors": [], "training_lineage_errors": [], "supersession_errors": [],
        "run_lifecycle_errors": [],
    }


def _lineage_errors(conn: sqlite3.Connection) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for snapshot in conn.execute(
        "SELECT feature_snapshot_id, run_id, station, asof_utc, lineage_json, source_max_known_utc "
        "FROM replay_feature_snapshots ORDER BY feature_snapshot_id"
    ):
        snapshot_id = snapshot["feature_snapshot_id"]
        try:
            lineage = json.loads(snapshot["lineage_json"])
            if canonical_json(lineage).decode("utf-8") != snapshot["lineage_json"]:
                raise ValueError("lineage_json is not canonical")
            source_ids = _lineage_source_ids(lineage)
            asof = _parse_utc(snapshot["asof_utc"])
            known_times = []
            for source_id in source_ids:
                event = conn.execute(
                    "SELECT run_id, station, known_utc FROM replay_source_events WHERE source_event_id=?",
                    (source_id,),
                ).fetchone()
                if event is None:
                    raise ValueError("lineage references missing source event")
                if event[0] != snapshot["run_id"]:
                    raise ValueError("lineage source event run_id mismatch")
                if event[1] != snapshot["station"]:
                    raise ValueError("lineage source event station mismatch")
                known = _parse_utc(event[2])
                if known > asof:
                    raise ValueError("lineage references future-known source event")
                known_times.append(known)
            metar_rows = lineage.get("metar_rows", [])
            if not isinstance(metar_rows, list):
                raise ValueError("lineage metar_rows is not a list")
            for metar in metar_rows:
                if not isinstance(metar, Mapping):
                    raise ValueError("lineage METAR row is not an object")
                if metar.get("station") != snapshot["station"]:
                    raise ValueError("lineage METAR row station mismatch")
                if metar.get("source") != "metar":
                    raise ValueError("lineage METAR row source mismatch")
                if _parse_utc(str(metar.get("valid_utc"))) > asof:
                    raise ValueError("lineage METAR row is future-known")
            if known_times and _parse_utc(snapshot["source_max_known_utc"]) != max(known_times):
                raise ValueError("source_max_known_utc does not match source lineage")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"feature_snapshot_id": snapshot_id, "reason": str(exc)})
    return errors


def _training_lineage_errors(conn: sqlite3.Connection, raw_root: Path) -> list[dict[str, str]]:
    """Prove each prediction is bound to a retained, replayable fit matrix."""
    errors: list[dict[str, str]] = []
    corpus_cache: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    reproducibility_by_run: dict[str, tuple[Mapping[str, Any] | None, str | None]] = {}
    for run in conn.execute("SELECT run_id, manifest_json FROM replay_runs"):
        try:
            manifest = json.loads(run["manifest_json"])
            reproducibility = manifest.get("reproducibility")
            if not isinstance(reproducibility, Mapping):
                raise ValueError("run manifest lacks reproducibility closure for training replay")
            reproducibility_by_run[str(run["run_id"])] = (reproducibility, None)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            reproducibility_by_run[str(run["run_id"])] = (None, str(exc))

    def validated_corpus(source_id: str) -> dict[str, Any]:
        """Validate source-intrinsic bytes once; prediction bindings stay outside."""
        cached = corpus_cache.get(source_id)
        if cached is not None:
            metadata, reason = cached
            if reason is not None:
                raise ValueError(reason)
            assert metadata is not None
            return metadata
        try:
            source = conn.execute(
                "SELECT run_id, station, source, raw_path FROM replay_source_events "
                "WHERE source_event_id=?",
                (source_id,),
            ).fetchone()
            if source is None:
                raise ValueError("fold-training lineage references a missing source")
            if source["source"] != "canonical_fold_training" or source["raw_path"] is None:
                raise ValueError(
                    "fold-training lineage does not reference retained canonical training evidence"
                )
            candidate = (raw_root / str(source["raw_path"])).resolve()
            if raw_root not in candidate.parents or not candidate.is_file():
                raise ValueError("fold-training raw payload is unavailable")
            with gzip.open(candidate, "rb") as handle:
                payload = json.loads(handle.read())
            if not isinstance(payload, dict):
                raise ValueError("canonical fold-training payload is not an object")
            stored_digest = payload.pop("training_content_sha256", None)
            if not isinstance(stored_digest, str) or sha256_json(payload) != stored_digest:
                raise ValueError("fold-training content digest does not match canonical payload")
            if payload.get("schema") != "phase0-canonical-fold-training-v1":
                raise ValueError("unknown canonical fold-training payload schema")
            rows = payload.get("rows")
            if not isinstance(rows, list) or payload.get("row_count") != len(rows):
                raise ValueError("fold-training row count does not match retained rows")
            payload_station = payload.get("station")
            for row in rows:
                identity = row.get("identity") if isinstance(row, Mapping) else None
                if not isinstance(identity, Mapping) or identity.get("station") != payload_station:
                    raise ValueError("fold-training row identity does not match payload station")
            metadata = {
                "source_run_id": source["run_id"], "source_station": source["station"],
                "training_content_sha256": stored_digest,
                "payload_station": payload_station, "arm": payload.get("arm"),
                "fold_id": payload.get("fold_id"),
                "train_cutoff_date": payload.get("train_cutoff_date"),
                "row_count": len(rows), "matrix_columns": payload.get("matrix_columns", []),
                "model_kind": payload.get("model_kind"),
                "model_spec_sha256": payload.get("model_spec_sha256"),
            }
            reproducibility, manifest_error = reproducibility_by_run.get(
                str(source["run_id"]), (None, "fold-training source references a missing run manifest")
            )
            if manifest_error is not None or reproducibility is None:
                raise ValueError(str(manifest_error))
            if metadata["model_spec_sha256"] != reproducibility.get("model_spec_sha256"):
                raise ValueError("fold-training model specification does not match run closure")
            metadata["expected_model_fingerprint"] = _model_fingerprint(
                metadata["matrix_columns"], str(metadata["train_cutoff_date"]),
                metadata["row_count"], reproducibility, str(metadata["model_kind"]),
                str(metadata["training_content_sha256"]),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc)
            corpus_cache[source_id] = (None, reason)
            raise ValueError(reason) from exc
        corpus_cache[source_id] = (metadata, None)
        return metadata

    predictions = conn.execute(
        """SELECT p.replay_prediction_id, p.run_id, p.station, p.bundle,
                  p.fold_id, p.model_fingerprint, p.train_cutoff_date, p.train_row_count,
                  s.lineage_json
             FROM replay_predictions p
             JOIN replay_feature_snapshots s ON s.feature_snapshot_id=p.feature_snapshot_id
             ORDER BY p.replay_prediction_id"""
    )
    for prediction in predictions:
        prediction_id = prediction["replay_prediction_id"]
        try:
            lineage = json.loads(prediction["lineage_json"])
            corpus = lineage.get("training_corpus")
            if not isinstance(corpus, Mapping):
                raise ValueError("missing canonical fold-training lineage")
            required = {"source_event_id", "training_content_sha256", "model_spec_sha256"}
            if not required.issubset(corpus):
                raise ValueError("incomplete canonical fold-training lineage")
            source_id = str(corpus["source_event_id"])
            if source_id not in _lineage_source_ids(lineage):
                raise ValueError("fold-training source is not linked from source_event_ids")
            metadata = validated_corpus(source_id)
            if metadata["source_run_id"] != prediction["run_id"]:
                raise ValueError("fold-training lineage references a missing or cross-run source")
            if metadata["source_station"] != prediction["station"]:
                raise ValueError("fold-training lineage references a cross-station source")
            if metadata["training_content_sha256"] != corpus["training_content_sha256"]:
                raise ValueError("fold-training content digest does not match snapshot lineage")
            if metadata["payload_station"] != prediction["station"]:
                raise ValueError("fold-training payload station does not match prediction")
            if metadata["arm"] != prediction["bundle"]:
                raise ValueError("fold-training payload arm does not match prediction")
            if metadata["fold_id"] != prediction["fold_id"]:
                raise ValueError("fold-training payload fold does not match prediction")
            if str(metadata["train_cutoff_date"]) != str(prediction["train_cutoff_date"]):
                raise ValueError("fold-training cutoff does not match prediction")
            if int(prediction["train_row_count"]) != metadata["row_count"]:
                raise ValueError("prediction train row count does not match retained training corpus")
            if metadata["model_spec_sha256"] != corpus["model_spec_sha256"]:
                raise ValueError("fold-training model specification does not match snapshot lineage")
            if metadata["expected_model_fingerprint"] != prediction["model_fingerprint"]:
                raise ValueError("prediction model fingerprint does not bind retained training content")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"replay_prediction_id": prediction_id, "reason": str(exc)})
    return errors


def _supersession_errors(conn: sqlite3.Connection) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    source_rows = conn.execute(
        """SELECT child.source_event_id AS child_id, child.supersedes_id, child.station AS child_station,
                  child.source AS child_source, child.valid_from_utc AS child_from, child.valid_to_utc AS child_to,
                  parent.station AS parent_station, parent.source AS parent_source,
                  parent.valid_from_utc AS parent_from, parent.valid_to_utc AS parent_to
           FROM replay_source_events child JOIN replay_source_events parent
             ON parent.source_event_id=child.supersedes_id
           WHERE child.supersedes_id IS NOT NULL ORDER BY child.source_event_id"""
    )
    for row in source_rows:
        try:
            same = (
                row["child_station"] == row["parent_station"] and row["child_source"] == row["parent_source"]
                and _same_optional_timestamp(row["child_from"], row["parent_from"])
                and _same_optional_timestamp(row["child_to"], row["parent_to"])
            )
        except ValueError:
            same = False
        if not same:
            errors.append({"kind": "source", "child_id": row["child_id"], "supersedes_id": row["supersedes_id"],
                           "reason": "logical_identity_mismatch"})
    outcome_rows = conn.execute(
        """SELECT child.outcome_id AS child_id, child.supersedes_id, child.station AS child_station,
                  child.target_date AS child_date, parent.station AS parent_station, parent.target_date AS parent_date
           FROM replay_outcomes child JOIN replay_outcomes parent ON parent.outcome_id=child.supersedes_id
           WHERE child.supersedes_id IS NOT NULL ORDER BY child.outcome_id"""
    )
    for row in outcome_rows:
        if (row["child_station"], row["child_date"]) != (row["parent_station"], row["parent_date"]):
            errors.append({"kind": "outcome", "child_id": row["child_id"], "supersedes_id": row["supersedes_id"],
                           "reason": "logical_identity_mismatch"})
    return errors


def _run_lifecycle_errors(conn: sqlite3.Connection) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for run in conn.execute("SELECT run_id, started_utc FROM replay_runs ORDER BY run_id"):
        events = list(conn.execute(
            "SELECT event_type, event_utc FROM replay_run_events WHERE run_id=? ORDER BY event_utc",
            (run["run_id"],),
        ))
        starts = [event for event in events if event["event_type"] == "started"]
        terminals = [event for event in events if event["event_type"] in {"completed", "failed"}]
        if len(starts) != 1:
            errors.append({"run_id": run["run_id"], "reason": "missing_or_duplicate_started_event"})
            continue
        if _parse_utc(starts[0]["event_utc"]) != _parse_utc(run["started_utc"]):
            errors.append({"run_id": run["run_id"], "reason": "started_event_timestamp_mismatch"})
        if len(terminals) == 0:
            errors.append({"run_id": run["run_id"], "reason": "missing_terminal_event"})
        elif len(terminals) > 1:
            errors.append({"run_id": run["run_id"], "reason": "duplicate_terminal_events"})
        elif _parse_utc(terminals[0]["event_utc"]) < _parse_utc(starts[0]["event_utc"]):
            errors.append({"run_id": run["run_id"], "reason": "terminal_event_precedes_start"})
    return errors


def verify_ledger(db_path: str | Path, raw_root: str | Path) -> dict[str, Any]:
    """Inspect an existing ledger read-only, without repairing any evidence."""
    database = Path(db_path)
    report = _empty_verification_report()
    if not database.is_file():
        report["missing_database"] = str(database)
        return report
    conn = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    raw_root_path = Path(raw_root).resolve()
    try:
        existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        report["missing_tables"] = [table for table in REPLAY_TABLES if table not in existing]
        report["foreign_key_errors"] = [dict(row) for row in conn.execute("PRAGMA foreign_key_check")]
        if not report["missing_tables"]:
            for row in conn.execute("SELECT raw_path, raw_sha256 FROM replay_source_events WHERE raw_path IS NOT NULL"):
                relative = str(row["raw_path"])
                candidate = (raw_root_path / relative).resolve()
                if raw_root_path not in candidate.parents:
                    report["unsafe_raw_paths"].append(relative)
                elif not candidate.is_file():
                    report["missing_raw_paths"].append(relative)
                elif sha256_bytes(candidate.read_bytes()) != row["raw_sha256"]:
                    report["raw_sha256_mismatches"].append(relative)
            referenced_raw_paths = {
                str(row[0]) for row in conn.execute("SELECT raw_path FROM replay_source_events WHERE raw_path IS NOT NULL")
            }
            if raw_root_path.is_dir():
                report["orphan_raw_paths"] = sorted(
                    path.relative_to(raw_root_path).as_posix()
                    for path in raw_root_path.rglob("*.json.gz")
                    if path.relative_to(raw_root_path).as_posix() not in referenced_raw_paths
                )
            duplicate_queries = (
                ("replay_feature_snapshots", "run_id, station, target_date, checkpoint_hour, bundle"),
                ("replay_predictions", "run_id, station, target_date, checkpoint_hour, bundle, fold_id"),
            )
            for table, columns in duplicate_queries:
                for row in conn.execute(f"SELECT {columns}, COUNT(*) AS count FROM {table} GROUP BY {columns} HAVING COUNT(*) > 1"):
                    report["duplicate_logical_events"].append({"table": table, **dict(row)})
            for row in conn.execute(
                """SELECT current.station, current.target_date, COUNT(*) AS count
                     FROM replay_outcomes current
                     LEFT JOIN replay_outcomes revision ON revision.supersedes_id=current.outcome_id
                    WHERE current.is_final=1 AND revision.outcome_id IS NULL
                    GROUP BY current.station, current.target_date HAVING COUNT(*) > 1"""
            ):
                report["duplicate_logical_events"].append({"table": "replay_outcomes", **dict(row)})
            report["schema_version_errors"] = [dict(row) for row in conn.execute(
                "SELECT run_id, schema_version FROM replay_runs WHERE schema_version != ?", (SCHEMA_VERSION,)
            )]
            report["immutable_snapshot_errors"] = [row[0] for row in conn.execute(
                "SELECT replay_prediction_id FROM replay_predictions p LEFT JOIN replay_feature_snapshots s "
                "ON s.feature_snapshot_id=p.feature_snapshot_id WHERE s.feature_snapshot_id IS NULL"
            )]
            report["lineage_errors"] = _lineage_errors(conn)
            report["training_lineage_errors"] = _training_lineage_errors(conn, raw_root_path)
            report["supersession_errors"] = _supersession_errors(conn)
            report["run_lifecycle_errors"] = _run_lifecycle_errors(conn)
        for key in ("missing_raw_paths", "raw_sha256_mismatches", "unsafe_raw_paths", "orphan_raw_paths"):
            report[key].sort()
        report["ok"] = not any(value for key, value in report.items() if key not in {"ok", "missing_database"})
        return report
    finally:
        conn.close()


def replay_miniature(fixture_path: str | Path) -> bytes:
    """Replay the fixed miniature fixture and return its canonical summary bytes."""
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="earlier-high-miniature-") as directory:
        conn = sqlite3.connect(Path(directory) / "miniature.db")
        conn.row_factory = sqlite3.Row
        try:
            ensure_replay_schema(conn)
            for record in fixture.get("runs", []):
                append_run(conn, record)
            for record in fixture.get("source_events", []):
                append_source_event(conn, record)
            for record in fixture.get("snapshots", []):
                append_feature_snapshot(conn, record)
            for record in fixture.get("predictions", []):
                append_prediction(conn, record)
            for record in fixture.get("outcomes", []):
                append_outcome(conn, record)
            def full_rows(table: str, order: str, json_columns: tuple[str, ...] = ()) -> list[dict[str, Any]]:
                rows = []
                for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                    value = dict(row)
                    for column in json_columns:
                        value[column] = json.loads(value[column])
                    rows.append(value)
                return rows

            shadows = []
            for shadow in fixture.get("forward_shadow", []):
                run = conn.execute("SELECT mode FROM replay_runs WHERE run_id=?", (shadow["run_id"],)).fetchone()
                if run is None or run["mode"] != "forward_shadow":
                    raise ValueError("forward_shadow fixture entry requires a forward-shadow replay run")
                outcome = conn.execute("SELECT outcome_id FROM replay_outcomes WHERE outcome_id=?", (shadow["final_outcome_id"],)).fetchone()
                if outcome is None:
                    raise ValueError("forward_shadow fixture entry requires its later final outcome")
                shadows.append({"run_id": shadow["run_id"], "no_trade": bool(shadow["no_trade"]),
                                "final_outcome_id": shadow["final_outcome_id"]})
            return canonical_json({
                "schema_version": fixture["schema_version"],
                "feature_snapshots": full_rows(
                    "replay_feature_snapshots", "feature_snapshot_id", ("features_json", "missing_json", "lineage_json")
                ),
                "predictions": full_rows("replay_predictions", "replay_prediction_id", ("quantiles_json", "buckets_json")),
                "outcomes": full_rows("replay_outcomes", "outcome_id"),
                "forward_shadow": sorted(shadows, key=lambda row: row["run_id"]),
            })
        finally:
            conn.close()


def _parse_csv(value: str, cast=str) -> list[Any]:
    parsed = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("comma-separated argument cannot be empty")
    return parsed


def _validate_scope(stations: Sequence[str], hours: Sequence[int],
                    bundles: Sequence[str] | None = None) -> None:
    bad_stations = sorted(set(stations) - set(ALLOWED_STATIONS))
    bad_hours = sorted(set(map(int, hours)) - set(ALLOWED_HOURS))
    bad_bundles = sorted(set(bundles or ()) - set(ALLOWED_BUNDLES))
    if bad_stations or bad_hours or bad_bundles:
        raise ValueError(f"outside Phase 0 scope: stations={bad_stations}, hours={bad_hours}, bundles={bad_bundles}")


def _config_path(station: str) -> Path:
    return PROJECT_ROOT / "config" / f"{station.lower()}.yaml"


def _sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _current_fingerprints(stations: Sequence[str]) -> tuple[str, dict[str, str]]:
    feature_path = PROJECT_ROOT / "src/highcast/weather_features.py"
    return _sha256_file(feature_path), {
        station: _sha256_file(_config_path(station)) for station in stations
    }


def reproducibility_document(stations: Sequence[str]) -> dict[str, Any]:
    """Hash the experiment's complete imported code/model/dependency closure."""
    from importlib.metadata import PackageNotFoundError, version
    from highcast.models import make_lgbm
    from highcast.models.baseline import LEVELS

    relative_paths = (
        "scripts/earlier_high_eval.py", "src/highcast/weather_features.py",
        "src/highcast/models/intraday.py", "src/highcast/models/baseline.py",
        "src/highcast/models/__init__.py", "src/highcast/models/dayahead.py",
        "src/highcast/models/calibrate.py",
        "src/highcast/features.py", "src/highcast/buckets.py", "src/highcast/metrics.py",
        "src/highcast/latentmax.py", "src/highcast/timeutil.py", "src/highcast/metric.py",
        "src/highcast/config.py", "src/highcast/quantize.py", "src/highcast/http.py",
        "src/highcast/db.py", "src/highcast/fetchers/cli_reports.py",
        "src/highcast/fetchers/alerts.py",
    )
    dependencies: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "pandas", "lightgbm", "scikit-learn", "PyYAML", "httpx"):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "MISSING"
    lgbm_params = make_lgbm(0.5).get_params()
    lgbm_params.pop("alpha", None)
    model_spec = {
        "quantile_levels": [float(level) for level in LEVELS],
        "lgbm_parameters": _jsonable(lgbm_params),
        "remaining_heat": {
            "factory": "scripts.earlier_high_eval._RemainingHeatModel",
            "target": "remaining_heat_f = cli_high_f - settlement_floor_from_visible",
            "fit_n_jobs_override": 1,
            "sample_weights": "highcast.models.dayahead.DayAheadModel.recency_weights",
            "fit_semantics": "one n_jobs=1 LightGBM quantile model per registered level",
            "reconstruction": "visible_floor + max(0,predicted_remaining), then sorted",
        },
        "shipped_direct_high": {
            "class": "highcast.models.intraday.IntradayModel", "target": "y",
            "residual": "cfg.model.intraday_residual_for('high')",
            "columns": "features_for(station,'high')", "training_hours": "intraday_bandwidth_for('high')",
            "sample_weights": None,
            "fit_semantics": "IntradayModel.fit filters y and fits unweighted LightGBM quantile models",
        },
        "baseline_inputs": {
            "class": "BaselineBlend", "prefer_corrected": "uses_bias_correction('high')",
            "fit_scope": "each expanding fold, target_date < cutoff",
        },
        "pmf": "highcast.buckets.bucket_probs", "bucket_hit": "expected_bucket_hit",
        "crps": "highcast.metrics.crps_pmf",
    }
    document = {
        "code_files_sha256": {path: _sha256_file(PROJECT_ROOT / path) for path in relative_paths},
        "config_files_sha256": {
            f"config/{station.lower()}.yaml": _sha256_file(_config_path(station))
            for station in stations
        },
        "dependency_versions": dependencies, "model_spec": model_spec,
    }
    document["closure_sha256"] = sha256_json(document)
    document["model_spec_sha256"] = sha256_json(model_spec)
    return document


def _cli_product_utc(product: str) -> str:
    try:
        return datetime.strptime(product[:12], "%Y%m%d%H%M").replace(tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"final CLI product has no parseable publication timestamp: {product!r}") from exc


def load_final_cli_truth(conn: sqlite3.Connection, cfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only authoritative final CLI highs with their exact product evidence."""
    from highcast.fetchers.cli_reports import cli_row_is_final

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in conn.execute(
        "SELECT date, high_f, raw, fetched_at FROM cli_reports "
        "WHERE station=? AND high_f IS NOT NULL ORDER BY date",
        (cfg.station.icao,),
    ):
        try:
            raw_payload = json.loads(source["raw"]) if source["raw"] else None
        except (TypeError, json.JSONDecodeError):
            raw_payload = None
        product = raw_payload.get("product") if isinstance(raw_payload, Mapping) else None
        target_date = str(source["date"])
        if not cli_row_is_final(product, target_date, cfg.station.std_utc_offset):
            excluded.append({"date": target_date, "reason": "NON_FINAL_CLI_PRODUCT",
                             "product": product})
            continue
        if not isinstance(raw_payload, Mapping):
            raise ValueError(f"final CLI row lacks canonical raw evidence: {cfg.station.icao} {target_date}")
        rows.append({
            "date": target_date, "high_f": int(source["high_f"]), "product": str(product),
            "product_utc": _cli_product_utc(str(product)),
            "fetched_at": source["fetched_at"], "raw_payload": dict(raw_payload),
            "raw_sha256": sha256_json(raw_payload),
        })
    return rows, excluded


def unresolved_negative_provenance(target: Mapping[str, Any], cli_truth: Mapping[str, Any],
                                   lane_lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Record an unresolved truth/observation conflict without self-certifying it."""
    return {
        **dict(target), "explained": False,
        "provenance_status": "UNRESOLVED_CLI_METAR_CONTRADICTION",
        "full_history_safe": False,
        "cli_evidence": {key: _jsonable(cli_truth.get(key)) for key in
                         ("product", "product_utc", "fetched_at", "raw_sha256")},
        "metar_lineage": _jsonable(lane_lineage),
    }


def resolve_negative_provenance(target: Mapping[str, Any], cli_truth: Mapping[str, Any],
                                lane_lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve only registry-pinned negative truth/observation contradictions."""
    unresolved = unresolved_negative_provenance(target, cli_truth, lane_lineage)
    metar_rows = lane_lineage.get("metar_rows", [])
    if not isinstance(metar_rows, list):
        return unresolved
    for rule in NEGATIVE_PROVENANCE_RESOLUTION_REGISTRY:
        target_matches = all(target.get(key) == value for key, value in rule["target"].items())
        cli_matches = all(cli_truth.get(key) == value for key, value in rule["cli"].items())
        offending = next((row for row in metar_rows if isinstance(row, Mapping) and all(
            row.get(key) == value for key, value in rule["offending_metar"].items()
        )), None)
        if not (target_matches and cli_matches and offending is not None):
            continue
        cli_payload = cli_truth.get("raw_payload")
        payload = cli_payload if isinstance(cli_payload, Mapping) else {}
        evidence = {
            "cli": {
                **dict(rule["cli"]), "high_f": payload.get("high"),
                "high_time": payload.get("high_time"),
            },
            "offending_metar": {
                **dict(rule["offending_metar"]), "tmpf_precise": offending.get("tmpf_precise"),
                "six_max_c": offending.get("six_max_c"),
            },
        }
        return {
            **unresolved, "explained": True, "provenance_status": rule["provenance_status"],
            "full_history_safe": True, "resolution_id": rule["resolution_id"],
            "resolution_evidence": _jsonable(evidence),
            "exclusion": {
                "station": target["station"], "date": target["date"], "hour": target["hour"],
                "reason": rule["provenance_status"], "resolution_id": rule["resolution_id"],
            },
        }
    return unresolved


def _metar_lineage_with_raw_evidence(conn: sqlite3.Connection,
                                     lane_lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Add raw fingerprints only to the METAR lineage needed for a conflict record."""
    rows: list[dict[str, Any]] = []
    for lineage_row in lane_lineage.get("metar_rows", []):
        if not isinstance(lineage_row, Mapping):
            continue
        identity = {key: lineage_row.get(key) for key in ("station", "source", "valid_utc")}
        source = conn.execute(
            "SELECT raw, tmpf_precise, six_max_c FROM obs "
            "WHERE station=? AND source=? AND valid_utc=?",
            (identity["station"], identity["source"], identity["valid_utc"]),
        ).fetchone()
        evidence = dict(identity)
        if source is not None:
            raw = source["raw"]
            evidence.update({
                "raw_sha256": sha256_bytes(str(raw).encode("utf-8")) if raw is not None else None,
                "tmpf_precise": source["tmpf_precise"], "six_max_c": source["six_max_c"],
            })
        rows.append(evidence)
    return {**dict(lane_lineage), "metar_rows": rows}


def _read_only_connection(path: str | Path) -> sqlite3.Connection:
    resolved = (PROJECT_ROOT / Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _jsonable(value: Any) -> Any:
    """Canonical-JSON-safe conversion for numpy/pandas/NaN experiment values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_write(path: str | Path, value: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: str | Path, value: Any) -> None:
    _atomic_write(path, canonical_json(_jsonable(value)) + b"\n")


def _append_progress(path: str | Path, event: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(canonical_json(_jsonable(event)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


_BUNDLE_FIELDS = {
    "B1": {
        "cur_rh_pct": ("100*exp(17.625*dew_c/(243.04+dew_c)-17.625*temp_c/(243.04+temp_c)); C=(F-32)*5/9", "pct"),
        "cur_dewpoint_depression_f": ("as-of temperature - dewpoint", "degF"),
        "rh_change_1h": ("current RH - RH of latest row in [checkpoint-1h-30m,checkpoint-1h]", "percentage_points"),
        "dewpoint_change_3h": ("current dewpoint - dewpoint of latest row in [checkpoint-3h-30m,checkpoint-3h]", "degF"),
    },
    "B2": {
        "wind_u_kt": ("-speed*sin(direction)", "kt"),
        "wind_v_kt": ("-speed*cos(direction)", "kt"),
        "onshore_change_1h": ("current speed*cos(direction-station_onshore_deg) minus matched 1h component", "kt"),
        "onshore_change_3h": ("current speed*cos(direction-station_onshore_deg) minus matched 3h component", "kt"),
        "wind_speed_change_1h": ("current sknt - matched 1h sknt", "kt"),
        "wind_direction_sin": ("sin(direction)", "unitless"),
        "wind_direction_cos": ("cos(direction)", "unitless"),
        "gust_excess_kt": ("gust parsed from pre-RMK METAR token (ddd|VRB)sssGgggKT minus sknt", "kt"),
    },
    "B3": {
        "wx_rain_now": ("as-of raw-METAR rain token state", "boolean"),
        "wx_rain_seen_3h": ("any rain token in prior 3h; unknown remains NaN", "boolean"),
        "hours_since_rain": ("checkpoint - last target-day rain report", "hours"),
        "visibility_change_1h": ("visibility change over 1h", "statute_miles"),
        "ceiling_change_1h": ("ceiling change over 1h", "ft"),
        "lowest_cloud_change_1h": ("lowest cloud change over 1h", "ft"),
        "cloud_clearing_flag": ("low deck to high/clear transition", "boolean"),
        "temp_recovery_since_rain_f": ("temperature - temperature at last rain", "degF"),
    },
}
_B4_FIELDS = {
    "KLAX": {
        "marine_clear_hour_local": ("station-local wall hour of marine_clear_utc", "local_hour"),
        "low_cloud_persistent": ("at least two target-day low-cloud reports", "boolean"),
        "onshore_transition": ("target-day onshore component change", "kt"),
        "temp_acceleration_after_clear_fph": ("post-clear minus pre-clear warming rate", "degF_per_hour"),
    },
    "KMIA": {
        "rain_onset_hour_local": ("station-local wall hour of rain_onset_utc", "local_hour"),
        "rain_end_hour_local": ("station-local wall hour of rain_end_utc", "local_hour"),
        "min_temp_after_rain_f": ("minimum temperature since rain ended", "degF"),
        "warming_rate_since_rain_end_fph": ("warming since rain end divided by elapsed hours", "degF_per_hour"),
        "persistent_cloud_flag": ("at least two low-cloud reports since rain onset", "boolean"),
        "gust_front_wind_shift_kt": ("vector wind change since rain onset", "kt"),
    },
}
_B5_FIELDS = (
    "future_rain_prob_max", "future_rain_sum_mm", "future_cloud_low_mean",
    "future_shortwave_sum", "future_rh_mean", "future_gust_max",
    "future_onshore_mean", "first_solar_threshold_hour", "first_rain_hour",
)
_LANE_A_ONSHORE_DEG = {"KLAX": 250.0, "KMIA": 90.0}

_B0_FIELDS = {
    "hour": ("checkpoint argument converted in station IANA timezone", "local_hour", "none", "required"),
    "run_lo": ("latentmax.visible_running_extremum(...,'max',asof_utc)[0]", "degF", "target climate day through checkpoint", "row excluded when visible extremum is absent"),
    "run_hi": ("latentmax.visible_running_extremum(...,'max',asof_utc)[1]", "degF", "target climate day through checkpoint", "row excluded when visible extremum is absent"),
    "run_width": ("run_hi - run_lo", "degF", "same as run_lo/run_hi", "row excluded when visible extremum is absent"),
    "cur_tmpf": ("latest METAR COALESCE(tmpf_precise,tmpf) at or before checkpoint", "degF", "latest prior METAR; no future row", "NaN when no value"),
    "cur_dwpf": ("latest METAR dwpf at or before checkpoint", "degF", "latest prior METAR; no future row", "NaN when no value"),
    "cur_onshore": ("sknt*cos(radians(drct-250.0)) from latest METAR", "kt", "latest prior METAR; no future row", "NaN when speed or direction missing"),
    "cur_sky": ("SKY_ORD[skyc] where CLR/SKC=0, FEW=1, SCT=2, BKN=3, OVC=4", "ordinal", "latest prior METAR; no future row", "NaN for missing/unknown sky code"),
    "base_p50": ("p50 from BaselineBlend inverse-MSE guidance blend plus seasonal residual quantile, refit on fold dates < cutoff", "degF", "expanding outcomes < fold cutoff; target guidance constrained by its day-ahead cutoff", "seasonal target climatology fallback when all guidance is missing"),
    "base_spread": ("BaselineBlend p90 - p10 from fold-local seasonal residual pool", "degF", "expanding outcomes < fold cutoff; +/-45 day-of-year residual pool", "global residual pool fallback when seasonal pool has <60 rows"),
    "doy_sin": ("sin(2*pi*day_of_year/365.25)", "unitless", "target date only", "required for valid target date"),
    "doy_cos": ("cos(2*pi*day_of_year/365.25)", "unitless", "target date only", "required for valid target date"),
    "year_index": ("target year - 2000", "years", "target date only", "required for valid target date"),
    "heat_alert": ("1 when a configured heat event has onset<=day-ahead cutoff and expiry>=climate-day start; else 0", "boolean", "alerts active for target climate day and issued by day-ahead cutoff", "absence is 0 after explicit station/event/as-of query"),
    "cur_rate_3h": ("cur_tmpf - latest COALESCE(tmpf_precise,tmpf) at checkpoint-3h", "degF_per_3h", "3 hours", "NaN when either observation missing"),
    "cur_rate_1h": ("cur_tmpf - latest COALESCE(tmpf_precise,tmpf) at checkpoint-1h", "degF_per_hour", "1 hour", "NaN when either observation missing"),
    "cur_rise_from_min": ("cur_tmpf - MIN(COALESCE(tmpf_precise,tmpf)) on climate day through checkpoint", "degF", "target climate day through checkpoint", "NaN when current or daily minimum missing"),
}

_B0_SOURCE = {
    "hour": "checkpoint request",
    "run_lo": "obs via latentmax (tmpf/tmpf_precise and six_max_c)",
    "run_hi": "obs via latentmax (tmpf/tmpf_precise and six_max_c)",
    "run_width": "derived from run_lo/run_hi",
    "cur_tmpf": "obs (source='metar')", "cur_dwpf": "obs (source='metar')",
    "cur_onshore": "obs (source='metar')", "cur_sky": "obs (source='metar')",
    "base_p50": "forecasts + cli_reports through BaselineBlend",
    "base_spread": "forecasts + cli_reports through BaselineBlend",
    "doy_sin": "target_date", "doy_cos": "target_date", "year_index": "target_date",
    "heat_alert": "alerts", "cur_rate_3h": "obs (source='metar')",
    "cur_rate_1h": "obs (source='metar')", "cur_rise_from_min": "obs (source='metar')",
}

_B5_MANIFEST = {
    "future_rain_prob_max": ("MAX(precipitation_probability) over local hourly valid times checkpoint..18:00 inclusive", "percent"),
    "future_rain_sum_mm": ("SUM(rain_mm) over local hourly valid times checkpoint..18:00 inclusive", "mm"),
    "future_cloud_low_mean": ("MEAN(cloud_cover_low) over local hourly valid times checkpoint..18:00 inclusive", "percent"),
    "future_shortwave_sum": ("SUM(shortwave_radiation_wm2*1h) over local hourly valid times checkpoint..18:00 inclusive", "Wh_per_m2"),
    "future_rh_mean": ("MEAN(relative_humidity_2m) over local hourly valid times checkpoint..18:00 inclusive", "percent"),
    "future_gust_max": ("MAX(wind_gust_kt) over local hourly valid times checkpoint..18:00 inclusive", "kt"),
    "future_onshore_mean": ("MEAN(wind_speed_kt*cos(radians(wind_direction_deg-station_onshore_deg))) over local hours checkpoint..18:00 inclusive", "kt"),
    "first_solar_threshold_hour": ("first local valid hour checkpoint..18:00 with shortwave_radiation_wm2>=200", "local_hour"),
    "first_rain_hour": ("first local valid hour checkpoint..18:00 with rain_mm>0", "local_hour"),
}


def _feature_manifest(stations: Sequence[str], earliest: Mapping[str, str | None]) -> list[dict[str, Any]]:
    from highcast.models.intraday import features_for
    features: list[dict[str, Any]] = []
    for station in stations:
        first = earliest.get(station)
        for name in features_for(station, "high"):
            formula, units, lookback, missingness = _B0_FIELDS[name]
            features.append({
                "station_scope": station, "bundle": "B0", "feature": name,
                "source_table": _B0_SOURCE[name], "formula": formula,
                "units": units, "allowable_lookback": lookback,
                "missingness_policy": missingness,
                "earliest_trustworthy_date": first, "status": "HISTORICAL_SAFE",
            })
        for bundle, fields in _BUNDLE_FIELDS.items():
            lookback = ("current <=2h; 1h/3h lag +/-30m" if bundle in {"B1", "B2"}
                        else "current <=2h; rain/change <=3h; target-day recovery")
            for name, (formula, units) in fields.items():
                formula = formula.replace(
                    "station_onshore_deg", f"{_LANE_A_ONSHORE_DEG[station]:g}deg"
                )
                features.append({
                    "station_scope": station, "bundle": bundle, "feature": name,
                    "source_table": "obs (source='metar')", "formula": formula,
                    "units": units, "allowable_lookback": lookback,
                    "missingness_policy": "NaN/JSON null; no imputation; missing weather is not dry",
                    "earliest_trustworthy_date": first, "status": "HISTORICAL_SAFE",
                })
        for name, (formula, units) in _B4_FIELDS[station].items():
            threshold = ("target-day event window; cloud-state low iff ceiling <3000 ft; "
                         "high/clear iff ceiling >=3000 ft or clear-sky code"
                         if station == "KLAX" else "target-day rain-event window")
            features.append({
                "station_scope": station, "bundle": "B4", "feature": name,
                "source_table": "obs (source='metar')", "formula": formula,
                "units": units, "allowable_lookback": threshold,
                "missingness_policy": "NaN/JSON null when event or required report is absent",
                "earliest_trustworthy_date": first, "status": "HISTORICAL_SAFE",
            })
        for name in _B5_FIELDS:
            formula, units = _B5_MANIFEST[name]
            formula = formula.replace(
                "station_onshore_deg", f"{_LANE_A_ONSHORE_DEG[station]:g}deg"
            )
            features.append({
                "station_scope": station, "bundle": "B5", "feature": name,
                "source_table": "Open-Meteo exact single-run capture (not historically retained)",
                "formula": formula, "units": units,
                "allowable_lookback": "exact run_utc and fetched_at <= checkpoint; valid hours checkpoint through declared peak window",
                "missingness_policy": "JSON null when exact run/field absent; FORWARD_ONLY; never stitched/backfilled",
                "earliest_trustworthy_date": None, "status": "FORWARD_ONLY",
            })
    return features


def _numeric_bundle(station: str, station_tz: str, bundle: str,
                    payload: Mapping[str, Any]) -> dict[str, float]:
    from zoneinfo import ZoneInfo
    fields = _BUNDLE_FIELDS.get(bundle)
    if fields is not None:
        return {name: float(payload[name]) if payload.get(name) is not None else math.nan
                for name in fields}
    if bundle != "B4":
        raise ValueError(f"no numeric Lane A mapping for {bundle}")
    result: dict[str, float] = {}
    for output_name in _B4_FIELDS[station]:
        if output_name.endswith("_hour_local"):
            source_name = output_name.replace("_hour_local", "_utc")
            raw = payload.get(source_name)
            if raw is None:
                result[output_name] = math.nan
            else:
                local = datetime.fromisoformat(str(raw)).astimezone(ZoneInfo(station_tz))
                result[output_name] = local.hour + local.minute / 60.0 + local.second / 3600.0
        else:
            raw = payload.get(output_name)
            result[output_name] = float(raw) if raw is not None else math.nan
    return result


def _probe_lane_b(stations: Sequence[str]) -> dict[str, Any]:
    """Bounded availability probe; retrospective fetched-at evidence remains zero."""
    from highcast.config import load_config
    representatives = ("2024-01-15", "2025-07-15", "2026-01-15")
    probes: list[dict[str, Any]] = []
    fields = "temperature_2m_previous_day1,relative_humidity_2m_previous_day1,precipitation_previous_day1,cloud_cover_low_previous_day1,shortwave_radiation_previous_day1,wind_gusts_10m_previous_day1"
    for station in stations:
        cfg = load_config(_config_path(station))
        for target_date in representatives:
            query = urllib.parse.urlencode({
                "latitude": cfg.station.lat, "longitude": cfg.station.lon,
                "start_date": target_date, "end_date": target_date,
                "hourly": fields, "timezone": "UTC",
            })
            url = f"https://previous-runs-api.open-meteo.com/v1/forecast?{query}"
            record = {"station": station, "representative_date": target_date,
                      "endpoint": "previous-runs", "http_status": None,
                      "available": False, "exact_issue_time_matched_rows": 0}
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    body = response.read(2048)
                    record["http_status"] = int(response.status)
                    record["available"] = response.status == 200 and bool(body)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["reason_exact_rows_zero"] = (
                "payload fetched during Phase 0 lacks historical fetched_at<=checkpoint proof; "
                "no stitched/observed backfill is admitted"
            )
            probes.append(record)
    return {
        "conclusion": "FORWARD_ONLY", "exact_issue_time_matched_rows": 0,
        "required_rows_per_station_hour": 400, "network_failure_blocks_lane_a": False,
        "probes": probes,
    }


def run_audit(stations: Sequence[str], hours: Sequence[int], out: str | Path) -> dict[str, Any]:
    from highcast import latentmax
    from highcast.config import load_config
    from highcast.weather_features import build_lane_a_features, checkpoint_utc
    import numpy as np

    _validate_scope(stations, hours)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    feature_sha, config_shas = _current_fingerprints(stations)
    reproducibility = reproducibility_document(stations)
    generated = datetime.now(timezone.utc).isoformat()
    station_audits: dict[str, Any] = {}
    earliest: dict[str, str | None] = {}
    schema_by_station: dict[str, list[str]] = {}
    for station in stations:
        cfg = load_config(_config_path(station))
        conn = _read_only_connection(cfg.db_path)
        try:
            schema_by_station[station] = [row[1] for row in conn.execute("PRAGMA table_info(obs)")]
            cli_rows, non_final_cli = load_final_cli_truth(conn, cfg)
            earliest[station] = cli_rows[0]["date"] if cli_rows else None
            negatives: list[dict[str, Any]] = []
            exclusions: list[dict[str, Any]] = [
                {"station": station, **item} for item in non_final_cli
            ]
            by_hour: dict[str, Any] = {}
            values_for_corr: dict[tuple[int, str, str], list[tuple[float, float]]] = {}
            missing_counts: dict[tuple[int, str, str], int] = {}
            totals: dict[tuple[int, str], int] = {}
            usable_counts: dict[tuple[int, str], int] = {}
            test_usable: dict[tuple[int, str], int] = {}
            test_rows_by_hour: dict[int, int] = {int(hour): 0 for hour in hours}
            for row in cli_rows:
                target_date = str(row["date"])
                for hour in hours:
                    asof = checkpoint_utc(target_date, int(hour), cfg.station.tz)
                    visible = latentmax.visible_running_extremum(
                        conn, cfg, date.fromisoformat(target_date), "max", asof_utc=asof.isoformat()
                    )
                    if visible is None:
                        exclusions.append({"station": station, "date": target_date, "hour": int(hour),
                                           "reason": "MISSING_VISIBLE_EXTREMUM"})
                        continue
                    lane = build_lane_a_features(conn, cfg, target_date, int(hour))
                    target = remaining_heat_record(station, target_date, int(hour), row["high_f"], visible)
                    if not target["invariant_ok"]:
                        provenance = resolve_negative_provenance(
                            target, row, _metar_lineage_with_raw_evidence(conn, lane["lineage"])
                        )
                        exclusions.append({**provenance, "reason": provenance["provenance_status"]})
                        if not provenance["explained"]:
                            negatives.append(provenance)
                        continue
                    if target_date >= "2025-01-01":
                        test_rows_by_hour[int(hour)] += 1
                    for bundle in ("B1", "B2", "B3", "B4"):
                        numeric = _numeric_bundle(station, cfg.station.tz, bundle, lane[bundle])
                        totals[(int(hour), bundle)] = totals.get((int(hour), bundle), 0) + 1
                        usable = any(math.isfinite(value) for value in numeric.values()) and bool(lane["lineage"]["metar_rows"])
                        usable_counts[(int(hour), bundle)] = usable_counts.get((int(hour), bundle), 0) + int(usable)
                        if target_date >= "2025-01-01" and usable:
                            test_usable[(int(hour), bundle)] = test_usable.get((int(hour), bundle), 0) + 1
                        for name, value in numeric.items():
                            key = (int(hour), bundle, name)
                            if math.isfinite(value):
                                values_for_corr.setdefault(key, []).append((value, float(target["remaining_heat_f"])))
                            else:
                                missing_counts[key] = missing_counts.get(key, 0) + 1
            for hour in hours:
                bundles_out = {}
                for bundle in ("B1", "B2", "B3", "B4"):
                    total = totals.get((int(hour), bundle), 0)
                    fields = _BUNDLE_FIELDS.get(bundle, _B4_FIELDS[station])
                    missingness, correlations = {}, {}
                    for name in fields:
                        pairs = values_for_corr.get((int(hour), bundle, name), [])
                        missingness[name] = (missing_counts.get((int(hour), bundle, name), 0) / total
                                             if total else None)
                        if len(pairs) >= 3:
                            x = np.asarray([pair[0] for pair in pairs], dtype=float)
                            y = np.asarray([pair[1] for pair in pairs], dtype=float)
                            correlations[name] = (float(np.corrcoef(x, y)[0, 1])
                                                  if np.std(x) > 0 and np.std(y) > 0 else None)
                        else:
                            correlations[name] = None
                    bundles_out[bundle] = {
                        "n": total, "usable_n": usable_counts.get((int(hour), bundle), 0),
                        "coverage": usable_counts.get((int(hour), bundle), 0) / total if total else 0.0,
                        "test_usable_n": test_usable.get((int(hour), bundle), 0),
                        "missingness": missingness, "remaining_heat_correlations": correlations,
                    }
                by_hour[str(int(hour))] = bundles_out
            unexplained = [item for item in negatives if not item.get("explained")]
            criteria = station_kill_criteria(
                test_rows_by_hour,
                {bundle: [by_hour[str(int(hour))][bundle]["coverage"] for hour in hours]
                 for bundle in ("B1", "B2", "B3", "B4")},
                len(unexplained), len(cli_rows) * len(hours),
            )
            station_audits[station] = {
                "cli_rows": len(cli_rows), "non_final_cli_rows": non_final_cli,
                "hours": by_hour,
                "negative_remaining_heat": negatives, "excluded": exclusions,
                "unexplained_negative_count": len(unexplained),
                "test_rows_by_hour": test_rows_by_hour,
                "kill_criteria": criteria,
                "scoring_safe": not unexplained and not any(
                    value for value in criteria.values() if value is not None
                ),
            }
        finally:
            conn.close()
    lane_b = _probe_lane_b(stations)
    live_paths = [path for path in (PROJECT_ROOT / "src/highcast").rglob("*.py")
                  if path.name != "weather_features.py"]
    boundary_errors = audit_offline_boundaries(
        [Path(__file__), PROJECT_ROOT / "src/highcast/weather_features.py"], live_paths
    )
    features = _feature_manifest(stations, earliest)
    manifest = {
        "experiment": "earlier_high_remaining_heat_v1", "spec_version": "phase-0",
        "generated_utc": generated, "audit_complete": True,
        "stations": list(stations), "metric": "high", "hours": [int(hour) for hour in hours],
        "train_window": "expanding", "minimum_training_rows": MINIMUM_TRAINING_ROWS,
        "full_history_oos_start": "first target date with minimum_training_rows prior eligible checkpoint rows",
        "test_start": "2025-01-01", "refit_every_days": 45,
        "primary_metric": "parity_averaged_bucket_hit", "bundles": list(ALLOWED_BUNDLES),
        "random_seeds": list(RANDOM_SEEDS),
        "multiplicity": {
            "declared_arm_identities": 40,
            "bundle_control_family": "40 candidate-vs-B0 paired comparisons, errors retained at p=1",
            "shipped_incumbent_family": "the same 40 candidate-vs-SHIPPED_DIRECT_HIGH identities, errors retained at p=1",
            "method": "Holm step-down with rank-specific paired-bootstrap intervals",
        },
        "data_cutoffs": {
            "observations": "station = ? AND valid_utc <= checkpoint_utc",
            "forecasts": "run_utc <= checkpoint_utc AND fetched_at <= checkpoint_utc",
            "truth": "cli_reports.high_f only when highcast.fetchers.cli_reports.cli_row_is_final(raw.product,date,std_utc_offset)",
        },
        "feature_code_sha256": feature_sha, "config_sha256": config_shas,
        "code_fingerprint": reproducibility["closure_sha256"],
        "model_spec_sha256": reproducibility["model_spec_sha256"],
        "reproducibility": reproducibility,
        "schema_columns": schema_by_station, "features": features,
        "station_scoring_safe": {station: audit["scoring_safe"] for station, audit in station_audits.items()},
        "audit": station_audits, "lane_b": lane_b,
        "offline_boundary_errors": boundary_errors,
        "legacy_lineage_limitations": [
            "historical provider HTTP bodies are unavailable; gzip files are explicitly labeled canonical legacy checkpoint inputs and retain METAR row identities"
        ],
    }
    if boundary_errors:
        manifest["audit_complete"] = False
    _write_json(out_path / "manifest.json", manifest)
    return manifest


class _RemainingHeatModel:
    """Experiment-only LightGBM quantile model for nonnegative remaining heat."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, frame):
        import numpy as np
        from highcast.models import make_lgbm
        from highcast.models.baseline import LEVELS
        from highcast.models.dayahead import DayAheadModel
        usable = frame[frame["remaining_heat_f"].notna()].copy()
        if len(usable) < 400:
            raise ValueError(f"insufficient training rows: {len(usable)} < 400")
        target = usable["remaining_heat_f"].to_numpy(dtype=float)
        if np.any(target < 0):
            raise ValueError("negative remaining-heat target reached model fit")
        weights = DayAheadModel.recency_weights(usable)
        inputs = usable[self.columns]

        def fit_level(level: float):
            # The factory's statistical parameters remain untouched.  One
            # thread per booster lets independent registered quantiles run in
            # parallel without nested OpenMP oversubscription; predictions are
            # byte-stable because LightGBM's seed/data/order are unchanged.
            model = make_lgbm(level)
            model.set_params(n_jobs=1)
            return level, model.fit(inputs, target, sample_weight=weights)

        workers = min(len(LEVELS), max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            self.models = dict(executor.map(fit_level, LEVELS))
        return self

    def predict_quantiles(self, frame) -> list[dict[float, float]]:
        import numpy as np
        from highcast.models.baseline import LEVELS
        remaining = np.column_stack([
            self.models[level].predict(frame[self.columns]) for level in LEVELS
        ])
        output = []
        for raw, floor in zip(remaining, frame["visible_floor_f"].to_numpy(dtype=float)):
            output.append(reconstruct_high_quantiles(
                float(floor), {level: float(value) for level, value in zip(LEVELS, raw)}
            ))
        return output


def refit_fold_baseline_inputs(day_frame, fold_frame, cutoff: str, metric_key: str):
    """Recompute baseline-derived inputs from outcomes available to one fold."""
    from highcast.features import uses_bias_correction
    from highcast.models.baseline import BaselineBlend

    training_days = day_frame[(day_frame["target_date"] < cutoff) & day_frame["y"].notna()].copy()
    if len(training_days) < 30:
        raise ValueError(f"insufficient fold-local BaselineBlend rows: {len(training_days)} < 30")
    baseline = BaselineBlend(prefer_corrected=uses_bias_correction(metric_key)).fit(training_days)
    requested_dates = set(str(value) for value in fold_frame["target_date"])
    prediction_days = day_frame[day_frame["target_date"].isin(requested_dates)].copy()
    quantiles = baseline.predict_quantiles(prediction_days)
    p50 = {str(row.target_date): float(qs[0.5])
           for row, qs in zip(prediction_days.itertuples(), quantiles)}
    spread = {str(row.target_date): float(qs[0.9] - qs[0.1])
              for row, qs in zip(prediction_days.itertuples(), quantiles)}
    result = fold_frame.copy()
    result["base_p50"] = result["target_date"].map(p50)
    result["base_spread"] = result["target_date"].map(spread)
    return result


def shipped_direct_high_model(cfg, station: str):
    """Instantiate the exact shipped same-hour direct-HIGH model convention."""
    from highcast.models.intraday import INTRADAY_FEATURES, IntradayModel, features_for

    columns = features_for(station, "high")
    extra = [column for column in columns if column not in INTRADAY_FEATURES]
    residual = cfg.model.intraday_residual_for("high")
    return IntradayModel(extremum="max", residual=residual, extra_features=extra)


def walk_forward_blocks(frame, target_hour: int, training_hours: Sequence[int],
                        refit_every: int, minimum_training_rows: int) -> list[dict[str, Any]]:
    """Declare genuine expanding OOS blocks as soon as the minimum train set exists."""
    eligible = frame[
        (frame["hour"] == float(target_hour)) & frame["remaining_heat_f"].notna()
    ]
    dates = sorted(str(value) for value in eligible["target_date"].unique())
    blocks: list[dict[str, Any]] = []
    start = None
    training_counts: dict[str, int] = {}
    hours = {float(value) for value in training_hours}
    for index, cutoff in enumerate(dates):
        count = int((
            (frame["target_date"] < cutoff) & frame["hour"].isin(hours)
            & frame["remaining_heat_f"].notna()
        ).sum())
        training_counts[cutoff] = count
        if count >= int(minimum_training_rows):
            start = index
            break
    if start is None:
        return []
    oos_dates = dates[start:]
    for offset in range(0, len(oos_dates), int(refit_every)):
        block_dates = oos_dates[offset:offset + int(refit_every)]
        cutoff = block_dates[0]
        blocks.append({"cutoff": cutoff, "dates": block_dates,
                       "train_row_count": training_counts.get(cutoff) or int((
                           (frame["target_date"] < cutoff) & frame["hour"].isin(hours)
                           & frame["remaining_heat_f"].notna()
                       ).sum())})
    return blocks


def hot_day_threshold(frame, test_start: str) -> float:
    """Training-only 80th percentile with one settled outcome per target date."""
    import numpy as np
    training = frame[(frame["target_date"] < test_start) & frame["y"].notna()]
    outcomes = training.groupby("target_date", sort=True)["y"].first().to_numpy(dtype=float)
    return float(np.percentile(outcomes, 80)) if len(outcomes) else float("inf")


def _model_kind_spec(kind: str, model_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    if kind == "shipped_direct_high_intraday":
        return model_spec.get("shipped_direct_high", {})
    if kind.startswith("remaining_heat"):
        return model_spec.get("remaining_heat", {})
    raise ValueError(f"unknown model fingerprint kind: {kind}")


def model_fingerprint_payload(columns: Sequence[str], train_cutoff: str, train_count: int,
                              reproducibility: Mapping[str, Any] | str, kind: str,
                              training_content_sha256: str | None = None) -> dict[str, Any]:
    """Canonical provenance for one fitted model, before it is hashed.

    `training_content_sha256` is deliberately part of the fingerprint rather
    than a descriptive side field: a model trained on a different row order,
    target, feature value, or recency-date sequence must never share its ID.
    """
    if isinstance(reproducibility, Mapping):
        code_files = reproducibility.get("code_files_sha256", {})
        dependencies = reproducibility.get("dependency_versions", {})
        model_spec = reproducibility.get("model_spec", {})
        closure_sha256 = str(reproducibility.get("closure_sha256", sha256_json(reproducibility)))
        model_spec_sha256 = str(reproducibility.get("model_spec_sha256", sha256_json(model_spec)))
    else:
        # Compatibility for the narrow deterministic helper test. Production
        # evaluation always supplies the complete reproducibility document.
        code_files, dependencies, model_spec = {}, {}, {}
        closure_sha256, model_spec_sha256 = str(reproducibility), sha256_json({})
    if kind == "shipped_direct_high_intraday":
        factory_files = (
            "src/highcast/models/intraday.py", "src/highcast/models/__init__.py",
            "src/highcast/models/baseline.py",
        )
        factory = "highcast.models.intraday.IntradayModel"
    elif kind.startswith("remaining_heat"):
        factory_files = (
            "scripts/earlier_high_eval.py", "src/highcast/models/__init__.py",
            "src/highcast/models/dayahead.py", "src/highcast/models/baseline.py",
        )
        factory = "scripts.earlier_high_eval._RemainingHeatModel"
    else:
        raise ValueError(f"unknown model fingerprint kind: {kind}")
    scoring_files = (
        "src/highcast/models/baseline.py", "src/highcast/buckets.py",
        "src/highcast/metrics.py", "src/highcast/quantize.py", "src/highcast/timeutil.py",
    )
    return {
        "kind": kind, "factory": factory, "columns": list(columns),
        "train_cutoff": train_cutoff, "train_count": int(train_count),
        "training_content_sha256": training_content_sha256,
        "factory_code_closure_sha256": sha256_json({
            path: code_files.get(path) for path in factory_files
        }),
        "quantile_pmf_bucket_code_sha256": sha256_json({
            path: code_files.get(path) for path in scoring_files
        }),
        "dependency_versions_sha256": sha256_json(dependencies),
        "model_spec_sha256": model_spec_sha256,
        "model_fit_semantics": _model_kind_spec(kind, model_spec),
        "reproducibility_closure_sha256": closure_sha256,
    }


def _model_fingerprint(columns: Sequence[str], train_cutoff: str, train_count: int,
                       reproducibility: Mapping[str, Any] | str, kind: str,
                       training_content_sha256: str | None = None) -> str:
    return sha256_json(model_fingerprint_payload(
        columns, train_cutoff, train_count, reproducibility, kind, training_content_sha256,
    ))


def _build_evaluation_frame(conn: sqlite3.Connection, cfg,
                            truth_rows: Sequence[Mapping[str, Any]],
                            hours: Sequence[int]):
    import pandas as pd
    from highcast.features import build_features, uses_bias_correction
    from highcast.metric import get_metric
    from highcast.models.baseline import BaselineBlend
    from highcast.models.intraday import build_checkpoint_frame
    from highcast.weather_features import build_lane_a_features

    dates = [str(row["date"]) for row in truth_rows]
    truth_by_date = {str(row["date"]): row for row in truth_rows}
    spec = get_metric(cfg, "high")
    day_frame = build_features(conn, cfg, spec, dates, require_target=True)
    baseline_train = day_frame.sort_values("target_date").head(MINIMUM_TRAINING_ROWS)
    if len(baseline_train) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"{cfg.station.icao}: fewer than {MINIMUM_TRAINING_ROWS} final-CLI baseline rows"
        )
    baseline = BaselineBlend(prefer_corrected=uses_bias_correction(spec.key)).fit(baseline_train)
    bandwidth = cfg.model.intraday_bandwidth_for("high")
    training_hours = tuple(sorted({(int(hour) + delta) % 24
                                   for hour in hours for delta in range(-bandwidth, bandwidth + 1)}))
    frame = build_checkpoint_frame(
        conn, cfg, dates, hours=training_hours, baseline=baseline,
        require_target=True, metric=spec,
    ).sort_values(["target_date", "hour"]).reset_index(drop=True)
    augmented: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        value = row.to_dict()
        target_date = str(value["target_date"])
        hour = int(float(value["hour"]))
        floor = settlement_floor_from_visible((float(value["run_lo"]), float(value["run_hi"])))
        remaining = int(float(value["y"])) - floor
        lane = build_lane_a_features(conn, cfg, target_date, hour)
        value["visible_floor_f"] = float(floor)
        value["remaining_heat_f"] = float(remaining) if remaining >= 0 else math.nan
        value["negative_remaining_heat"] = remaining < 0
        value["asof_utc"] = lane["asof_utc"]
        value["lane_lineage"] = lane["lineage"]
        truth = truth_by_date[target_date]
        value["cli_product"] = truth["product"]
        value["cli_product_utc"] = truth["product_utc"]
        value["cli_fetched_at"] = truth["fetched_at"]
        value["cli_raw_sha256"] = truth["raw_sha256"]
        for bundle in ("B1", "B2", "B3", "B4"):
            numeric = _numeric_bundle(cfg.station.icao, cfg.station.tz, bundle, lane[bundle])
            value.update(numeric)
            value[f"{bundle}_usable"] = bool(lane["lineage"]["metar_rows"]) and any(
                math.isfinite(number) for number in numeric.values()
            )
        augmented.append(value)
    return pd.DataFrame(augmented), training_hours, day_frame


def _add_noise_columns(frame, station: str) -> dict[tuple[str, int], list[str]]:
    columns: dict[tuple[str, int], list[str]] = {}
    keys = [f"{station}|{row.target_date}|{float(row.hour):g}" for row in frame.itertuples()]
    for bundle in ("B1", "B2", "B3", "B4"):
        count = len(_BUNDLE_FIELDS.get(bundle, _B4_FIELDS[station]))
        controls = deterministic_random_controls(keys, count, RANDOM_SEEDS)
        for seed, rows in controls.items():
            names = [f"__{bundle}_noise_{seed}_{index}" for index in range(count)]
            columns[(bundle, seed)] = names
            for index, name in enumerate(names):
                frame[name] = [row[index] for row in rows]
    return columns


def _prediction_record(target_date: str, y: float, quantiles: Mapping[float, float],
                       extra: Mapping[str, Any]) -> dict[str, Any]:
    from highcast.buckets import bucket_probs, expected_bucket_hit, modal_bucket
    from highcast.metrics import crps_pmf
    pmf = bucket_probs(dict(quantiles))
    actual = int(y)
    hit = expected_bucket_hit(pmf, actual)
    confidence = sum(modal_bucket(pmf, parity)[1] for parity in (0, 1)) / 2.0
    point = float(quantiles[0.5])
    return {
        "target_date": target_date, "y": actual, "point_f": point,
        "bucket_hit": float(hit), "crps": float(crps_pmf(pmf, actual)),
        "abs_error": abs(point - actual), "signed_bias": point - actual,
        "coverage_50": float(quantiles[0.25] <= actual <= quantiles[0.75]),
        "coverage_90": float(quantiles[0.05] <= actual <= quantiles[0.95]),
        "modal_confidence": float(confidence),
        "quantiles": {float(key): float(value) for key, value in quantiles.items()},
        "pmf": {int(key): float(value) for key, value in pmf.items()},
        **dict(extra),
    }


def _mean_or_none(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "bucket_hit": _mean_or_none(row["bucket_hit"] for row in records),
        "crps": _mean_or_none(row["crps"] for row in records),
        "mae": _mean_or_none(row["abs_error"] for row in records),
        "signed_bias": _mean_or_none(row["signed_bias"] for row in records),
        "coverage_50": _mean_or_none(row["coverage_50"] for row in records),
        "coverage_90": _mean_or_none(row["coverage_90"] for row in records),
        "calibration_gap": _mean_or_none(
            row["modal_confidence"] - row["bucket_hit"] for row in records
        ),
    }


def _paired_slice(candidate: Sequence[Mapping[str, Any]], incumbent: Sequence[Mapping[str, Any]],
                  predicate) -> dict[str, Any]:
    candidate_by_date = {str(row["target_date"]): row for row in candidate}
    incumbent_by_date = {str(row["target_date"]): row for row in incumbent}
    dates = [target_date for target_date in sorted(set(candidate_by_date) & set(incumbent_by_date))
             if predicate(candidate_by_date[target_date])]
    left = [candidate_by_date[target_date] for target_date in dates]
    right = [incumbent_by_date[target_date] for target_date in dates]
    candidate_summary, incumbent_summary = _summary(left), _summary(right)
    return {
        **candidate_summary,
        "delta_hit": ((candidate_summary["bucket_hit"] - incumbent_summary["bucket_hit"])
                      if candidate_summary["bucket_hit"] is not None else None),
        "delta_crps": ((candidate_summary["crps"] - incumbent_summary["crps"])
                       if candidate_summary["crps"] is not None else None),
    }


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    return _jsonable(value)


class _LedgerWriter:
    def __init__(self, db_path: Path, raw_root: Path, manifest: Mapping[str, Any],
                 run_id: str, code_fingerprint: str, config_fingerprint: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        ensure_replay_schema(self.conn)
        self.run_id = run_id
        self.raw_root = raw_root
        self.source_ids: dict[tuple[str, str, int], str] = {}
        self.training_sources: dict[tuple[str, str, str, str], dict[str, str]] = {}
        self.outcomes: set[tuple[str, str]] = set()
        self.predictions: set[tuple[str, str, int, str]] = set()
        self.terminal_event_written = False
        append_run(self.conn, {
            "run_id": run_id, "experiment": manifest["experiment"],
            "spec_version": manifest["spec_version"], "started_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "backtest", "code_fingerprint": code_fingerprint,
            "config_sha256": config_fingerprint, "schema_version": SCHEMA_VERSION,
            "manifest_json": manifest, "status": "immutable_started",
        })
        self.conn.commit()

    @staticmethod
    def _id(prefix: str, parts: Sequence[Any]) -> str:
        digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{digest}"

    def _source(self, station: str, row: Mapping[str, Any]) -> str:
        key = (station, str(row["target_date"]), int(float(row["hour"])))
        if key in self.source_ids:
            return self.source_ids[key]
        source_id = self._id("src", (self.run_id, *key))
        excluded = {"y", "remaining_heat_f", "negative_remaining_heat"}
        payload = {name: _row_value(row, name) for name in sorted(row)
                   if name not in excluded and name != "lane_lineage" and not name.startswith("cli_")}
        payload["metar_lineage"] = _jsonable(row.get("lane_lineage"))
        append_source_event(self.conn, {
            "source_event_id": source_id, "run_id": self.run_id, "station": station,
            "source": "canonical_legacy_checkpoint", "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "known_utc": str(row["asof_utc"]), "known_utc_basis": "offline_asof_reconstruction",
            "valid_from_utc": str(row["asof_utc"]), "valid_to_utc": str(row["asof_utc"]),
            # This is explicitly a canonical legacy checkpoint payload, not a
            # manufactured provider HTTP response. It is retained because a hash
            # alone cannot reconstruct the exact parsed model input.
            "parser_version": "phase0-canonical-legacy-checkpoint-v2",
            "raw_body": payload, "raw_root": self.raw_root,
        })
        self.source_ids[key] = source_id
        return source_id

    def retain_fold_training(self, station: str, fold_id: str, arm: str, model_kind: str,
                             columns: Sequence[str], training, train_cutoff_date: str,
                             reproducibility: Mapping[str, Any]) -> dict[str, str]:
        """Retain the exact ordered matrix consumed by one fold/arm fit.

        One gzip source event per fold/arm bounds event growth while preserving
        every row identity, feature value, target, and recency-weight date.
        """
        if len(training) > MAX_CANONICAL_FOLD_TRAINING_ROWS:
            raise ValueError(
                f"fold training corpus exceeds bounded limit {MAX_CANONICAL_FOLD_TRAINING_ROWS}"
            )
        target_column = "y" if model_kind == "shipped_direct_high_intraday" else "remaining_heat_f"
        if target_column not in training.columns:
            raise ValueError(f"fold training frame lacks model target {target_column!r}")
        missing_columns = [column for column in columns if column not in training.columns]
        if missing_columns:
            raise ValueError(f"fold training frame lacks model columns: {missing_columns}")
        identity_columns = ("target_date", "hour", "asof_utc")
        missing_identity = [column for column in identity_columns if column not in training.columns]
        if missing_identity:
            raise ValueError(f"fold training frame lacks row identity columns: {missing_identity}")
        if len(training) == 0:
            raise ValueError("fold training corpus cannot be empty")
        weight_inputs = ["target_date"] if model_kind.startswith("remaining_heat_") else []
        rows: list[dict[str, Any]] = []
        asofs: list[datetime] = []
        for _, training_row in training.iterrows():
            asof = _utc_text(str(training_row["asof_utc"]))
            asofs.append(_parse_utc(asof))
            rows.append({
                "identity": {
                    "station": station, "target_date": str(training_row["target_date"]),
                    "checkpoint_hour": float(training_row["hour"]), "asof_utc": asof,
                },
                "inputs": {column: _row_value(training_row, column) for column in columns},
                "target": _row_value(training_row, target_column),
                "weight_inputs": {
                    column: _row_value(training_row, column) for column in weight_inputs
                },
            })
        model_spec_sha256 = str(reproducibility["model_spec_sha256"])
        payload = {
            "schema": "phase0-canonical-fold-training-v1", "station": station,
            "fold_id": fold_id, "arm": arm, "model_kind": model_kind,
            "train_cutoff_date": str(train_cutoff_date), "matrix_columns": list(columns),
            "target_column": target_column, "weight_inputs": weight_inputs,
            "row_count": len(rows), "rows": rows,
            "model_spec_sha256": model_spec_sha256,
        }
        content_sha256 = sha256_json(payload)
        key = (station, fold_id, arm, content_sha256)
        retained = self.training_sources.get(key)
        if retained is not None:
            return retained
        source_id = self._id("train", (self.run_id, *key))
        append_source_event(self.conn, {
            "source_event_id": source_id, "run_id": self.run_id, "station": station,
            "source": "canonical_fold_training",
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "known_utc": max(asofs).isoformat(),
            "known_utc_basis": "offline_fold_training_reconstruction",
            "valid_from_utc": min(asofs).isoformat(), "valid_to_utc": max(asofs).isoformat(),
            "parser_version": "phase0-canonical-fold-training-v1",
            "raw_body": {**payload, "training_content_sha256": content_sha256},
            "raw_root": self.raw_root,
        })
        retained = {
            "source_event_id": source_id, "training_content_sha256": content_sha256,
            "model_spec_sha256": model_spec_sha256,
        }
        self.training_sources[key] = retained
        return retained

    def write(self, station: str, row: Mapping[str, Any], arm: str, columns: Sequence[str],
              prediction: Mapping[str, Any], fold_id: str, train_count: int,
              model_fingerprint: str, feature_code_sha256: str,
              training_corpus: Mapping[str, str]) -> None:
        identity = (station, str(row["target_date"]), int(float(row["hour"])), arm)
        if identity in self.predictions:
            return
        required_training = {"source_event_id", "training_content_sha256", "model_spec_sha256"}
        if not required_training.issubset(training_corpus):
            raise ValueError("prediction requires retained canonical fold-training provenance")
        source_id = self._source(station, row)
        snapshot_id = self._id("snap", (self.run_id, *identity))
        prediction_id = self._id("pred", (self.run_id, *identity, fold_id))
        features = {column: _row_value(row, column) for column in columns}
        missing = {column: features[column] is None for column in columns}
        append_feature_snapshot(self.conn, {
            "feature_snapshot_id": snapshot_id, "run_id": self.run_id, "station": station,
            "target_date": str(row["target_date"]), "checkpoint_hour": float(row["hour"]),
            "asof_utc": str(row["asof_utc"]), "bundle": arm, "feature_schema_version": 1,
            "feature_code_sha256": feature_code_sha256, "features_json": features,
            "missing_json": missing, "lineage_json": {
                "source_event_ids": [source_id, str(training_corpus["source_event_id"])],
                "metar_rows": _jsonable((row.get("lane_lineage") or {}).get("metar_rows", [])),
                "training_corpus": dict(training_corpus),
            },
            "visible_floor_f": float(row["visible_floor_f"]),
            "source_max_known_utc": str(row["asof_utc"]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        append_prediction(self.conn, {
            "replay_prediction_id": prediction_id, "run_id": self.run_id,
            "feature_snapshot_id": snapshot_id, "station": station,
            "target_date": str(row["target_date"]), "checkpoint_hour": float(row["hour"]),
            "issued_utc": str(row["asof_utc"]), "bundle": arm, "fold_id": fold_id,
            "model_fingerprint": model_fingerprint,
            "train_cutoff_date": str(row["train_cutoff_date"]), "train_row_count": int(train_count),
            "point_f": float(prediction["point_f"]),
            "quantiles_json": {str(key): value for key, value in prediction["quantiles"].items()},
            "buckets_json": {str(key): value for key, value in prediction["pmf"].items()},
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        outcome_key = (station, str(row["target_date"]))
        if outcome_key not in self.outcomes:
            append_outcome(self.conn, {
                "outcome_id": self._id("out", (*outcome_key, row["cli_product_utc"], row["cli_raw_sha256"])),
                "station": station,
                "target_date": str(row["target_date"]), "cli_high_f": int(float(row["y"])),
                "product_utc": str(row["cli_product_utc"]),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "raw_sha256": str(row["cli_raw_sha256"]),
                "is_final": 1,
            })
            self.outcomes.add(outcome_key)
        self.predictions.add(identity)

    def commit(self) -> None:
        self.conn.commit()

    def finish(self, failed: bool, errors: Sequence[Mapping[str, Any]] = ()) -> None:
        if self.terminal_event_written:
            return
        event_type = "failed" if failed else "completed"
        append_run_event(self.conn, {
            "run_event_id": self._id("run-event", (self.run_id, event_type)),
            "run_id": self.run_id, "event_type": event_type,
            "event_utc": datetime.now(timezone.utc).isoformat(),
            "status": event_type,
            "error_text": (canonical_json(_jsonable(list(errors))).decode("utf-8") if errors else None),
        })
        self.conn.commit()
        self.terminal_event_written = True

    def close(self) -> None:
        self.conn.close()


def _evaluate_comparison(station: str, hour: int, bundle: str,
                         incumbent: Sequence[Mapping[str, Any]],
                         candidate: Sequence[Mapping[str, Any]],
                         noise: Mapping[int, Sequence[Mapping[str, Any]]],
                         coverage: float, bootstrap: int,
                         excluded: Sequence[Mapping[str, Any]],
                         hot_threshold: float,
                         shipped_incumbent: Sequence[Mapping[str, Any]] | None = None,
                         full_history_incumbent: Sequence[Mapping[str, Any]] | None = None,
                         full_history_candidate: Sequence[Mapping[str, Any]] | None = None,
                         full_history_shipped: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    arms = {"B0": incumbent, bundle: candidate,
            **{f"noise_{seed}": records for seed, records in noise.items()}}
    if shipped_incumbent is not None:
        arms["SHIPPED_DIRECT_HIGH"] = shipped_incumbent
    matched = match_arm_records(arms)
    dates = matched["dates"]
    inc = matched["arms"]["B0"]
    cand = matched["arms"][bundle]
    inc_summary, cand_summary = _summary(inc), _summary(cand)
    delta_hit_values = [left["bucket_hit"] - right["bucket_hit"] for left, right in zip(cand, inc)]
    boot = paired_bootstrap(delta_hit_values, bootstrap, RANDOM_SEEDS[0])
    noise_deltas = []
    noise_significant = 0
    noise_metrics = {}
    for seed in RANDOM_SEEDS:
        records = matched["arms"][f"noise_{seed}"]
        values = [left["bucket_hit"] - right["bucket_hit"] for left, right in zip(records, inc)]
        noise_boot = paired_bootstrap(values, bootstrap, seed)
        noise_delta = noise_boot["mean"]
        if noise_delta is not None:
            noise_deltas.append(float(noise_delta))
        significant = bool(noise_delta is not None and noise_delta > 0
                           and noise_boot["ci"][0] is not None and noise_boot["ci"][0] > 0)
        noise_significant += int(significant)
        noise_metrics[str(seed)] = {"delta_hit": noise_boot["mean"], "delta_hit_ci": noise_boot["ci"],
                                    "clears_threshold": significant}
    noise_p975 = None
    if noise_deltas:
        import numpy as np
        noise_p975 = float(np.percentile(noise_deltas, 97.5))
    warm_months = ({4, 5, 6, 7, 8, 9, 10} if station == "KMIA" else {5, 6, 7, 8, 9, 10})
    full_inc_source = list(full_history_incumbent) if full_history_incumbent is not None else inc
    full_cand_source = list(full_history_candidate) if full_history_candidate is not None else cand
    full_matched = match_arm_records({"B0": full_inc_source, bundle: full_cand_source})
    full_inc = full_matched["arms"]["B0"]
    full_cand = full_matched["arms"][bundle]
    full = _paired_slice(full_cand, full_inc, lambda row: True)
    warm = _paired_slice(full_cand, full_inc,
                         lambda row: int(str(row["target_date"])[5:7]) in warm_months)
    slices = {
        "full_history": full,
        "warm_season": warm,
        "hot_day": _paired_slice(cand, inc, lambda row: float(row["y"]) >= hot_threshold),
    }
    if station == "KMIA":
        slices["observed_rain"] = _paired_slice(cand, inc, lambda row: row.get("rain_seen") == 1.0)
        slices["no_rain"] = _paired_slice(cand, inc, lambda row: row.get("rain_seen") == 0.0)
    else:
        slices["marine_layer_persistent"] = _paired_slice(
            cand, inc, lambda row: row.get("marine_persistent") == 1.0
        )
    baseline_gap = inc_summary["calibration_gap"]
    candidate_gap = cand_summary["calibration_gap"]
    gap_increase = (abs(candidate_gap) - abs(baseline_gap)
                    if candidate_gap is not None and baseline_gap is not None else None)
    shipped_comparison: dict[str, Any]
    if shipped_incumbent is not None:
        shipped = matched["arms"]["SHIPPED_DIRECT_HIGH"]
        shipped_summary = _summary(shipped)
        shipped_values = [left["bucket_hit"] - right["bucket_hit"]
                          for left, right in zip(cand, shipped)]
        shipped_boot = paired_bootstrap(shipped_values, bootstrap, RANDOM_SEEDS[0])
        shipped_gap = shipped_summary["calibration_gap"]
        shipped_full_source = (list(full_history_shipped)
                               if full_history_shipped is not None else shipped)
        shipped_full_match = match_arm_records({
            bundle: full_cand_source, "SHIPPED_DIRECT_HIGH": shipped_full_source,
        })
        shipped_full_cand = shipped_full_match["arms"][bundle]
        shipped_full_base = shipped_full_match["arms"]["SHIPPED_DIRECT_HIGH"]
        shipped_comparison = {
            "name": "SHIPPED_DIRECT_HIGH_INTRADAY", "n": len(shipped),
            "baseline": shipped_summary,
            "delta_hit": ((cand_summary["bucket_hit"] - shipped_summary["bucket_hit"])
                          if cand_summary["bucket_hit"] is not None else None),
            "delta_hit_ci": shipped_boot["ci"],
            "delta_crps": ((cand_summary["crps"] - shipped_summary["crps"])
                           if cand_summary["crps"] is not None else None),
            "calibration_gap_increase": (
                abs(candidate_gap) - abs(shipped_gap)
                if candidate_gap is not None and shipped_gap is not None else None
            ),
            "slices": {
                "full_history": _paired_slice(shipped_full_cand, shipped_full_base, lambda row: True),
                "warm_season": _paired_slice(
                    shipped_full_cand, shipped_full_base,
                    lambda row: int(str(row["target_date"])[5:7]) in warm_months,
                ),
            },
            "bootstrap_p_two_sided": shipped_boot["p_two_sided"],
            "holm_adjusted_p": None, "_bootstrap_samples": shipped_boot["samples"],
        }
    else:
        shipped_comparison = {
            "name": "SHIPPED_DIRECT_HIGH_INTRADAY", "n": 0, "baseline": _summary([]),
            "delta_hit": None, "delta_hit_ci": [None, None], "delta_crps": None,
            "calibration_gap_increase": None,
            "slices": {"full_history": {"delta_hit": None},
                       "warm_season": {"delta_hit": None}},
            "bootstrap_p_two_sided": None, "holm_adjusted_p": None,
            "_bootstrap_samples": [],
        }
    row = {
        "station": station, "hour": int(hour), "bundle": bundle,
        "n": len(dates), "coverage": float(coverage),
        "bucket_hit": cand_summary["bucket_hit"],
        "delta_hit": ((cand_summary["bucket_hit"] - inc_summary["bucket_hit"])
                      if cand_summary["bucket_hit"] is not None else None),
        "delta_hit_ci": boot["ci"], "crps": cand_summary["crps"],
        "delta_crps": ((cand_summary["crps"] - inc_summary["crps"])
                       if cand_summary["crps"] is not None else None),
        "noise_delta_hit_p975": noise_p975, "verdict": "REJECT",
        "safeguards": {
            "mae": cand_summary["mae"], "delta_mae": (
                cand_summary["mae"] - inc_summary["mae"] if cand_summary["mae"] is not None else None),
            "signed_bias": cand_summary["signed_bias"],
            "coverage_50": cand_summary["coverage_50"], "coverage_90": cand_summary["coverage_90"],
            "calibration_gap": candidate_gap, "baseline_calibration_gap": baseline_gap,
            "calibration_gap_increase": gap_increase,
            "baseline": inc_summary, "noise_controls": noise_metrics,
            "noise_significant_count": noise_significant,
            "invariants": {"predictions_below_visible_floor": 0, "quantile_crossings": 0},
        },
        "slices": slices, "exclusions": list(excluded),
        "matched_dates": dates, "bootstrap_p_two_sided": boot["p_two_sided"],
        "holm_adjusted_p": None, "random_family_contaminated": False,
        "shipped_incumbent_comparison": shipped_comparison,
        "_bootstrap_samples": boot["samples"],
    }
    return row


def _apply_multiplicity(rows: list[dict[str, Any]], declared_family_size: int) -> None:
    import numpy as np
    p_values = [float(row.get("bootstrap_p_two_sided") or 1.0) for row in rows]
    if len(rows) > int(declared_family_size):
        raise ValueError("result rows exceed the declared comparison family")
    adjusted = holm_adjust([*p_values, *([1.0] * (int(declared_family_size) - len(rows)))])[:len(rows)]
    order = sorted(range(len(rows)), key=lambda index: p_values[index])
    rank_by_index = {index: rank for rank, index in enumerate(order)}
    total = int(declared_family_size)
    for index, row in enumerate(rows):
        row["holm_adjusted_p"] = adjusted[index]
        samples = row.pop("_bootstrap_samples", [])
        if samples:
            alpha = 0.05 / max(1, total - rank_by_index[index])
            row["delta_hit_ci"] = [float(np.percentile(samples, 100 * alpha / 2)),
                                   float(np.percentile(samples, 100 * (1 - alpha / 2)))]
    shipped_p = [float((row.get("shipped_incumbent_comparison") or {}).get(
        "bootstrap_p_two_sided") or 1.0) for row in rows]
    shipped_adjusted = holm_adjust(
        [*shipped_p, *([1.0] * (int(declared_family_size) - len(rows)))]
    )[:len(rows)]
    shipped_order = sorted(range(len(rows)), key=lambda index: shipped_p[index])
    shipped_rank = {index: rank for rank, index in enumerate(shipped_order)}
    for index, row in enumerate(rows):
        comparison = row.get("shipped_incumbent_comparison")
        if not isinstance(comparison, dict):
            continue
        comparison["holm_adjusted_p"] = shipped_adjusted[index]
        samples = comparison.pop("_bootstrap_samples", [])
        if samples:
            alpha = 0.05 / max(1, total - shipped_rank[index])
            comparison["delta_hit_ci"] = [
                float(np.percentile(samples, 100 * alpha / 2)),
                float(np.percentile(samples, 100 * (1 - alpha / 2))),
            ]
    contaminated_bundles = {
        bundle for bundle in ("B1", "B2", "B3", "B4")
        if sum(int((row.get("safeguards") or {}).get("noise_significant_count", 0))
               for row in rows if row.get("bundle") == bundle) >= 2
    }
    for row in rows:
        row["random_family_contaminated"] = row.get("bundle") in contaminated_bundles
        if row.get("verdict") != "ERROR":
            row["verdict"] = verdict_for_result(row)


def _artifact_report(manifest: Mapping[str, Any], results: Mapping[str, Any],
                     hashes: Mapping[str, str], integrity: Mapping[str, Any],
                     miniature_bytes: int) -> str:
    lines = [
        "# Earlier HIGH Phase 0 review gate", "",
        f"Generated UTC: {results['generated_utc']}",
        f"Run status: `{results.get('status', 'UNVERIFIED')}`", "",
        "## Kill criteria", "",
    ]
    for station, audit in manifest["audit"].items():
        lines.append(f"- {station}: `{json.dumps(audit['kill_criteria'], sort_keys=True)}`; scoring_safe={audit['scoring_safe']}; negative rows={len(audit['negative_remaining_heat'])}.")
    lines.extend(["", "## Lane B conclusion", "",
                  f"`{manifest['lane_b']['conclusion']}` with {manifest['lane_b']['exact_issue_time_matched_rows']} exact historical matched rows; Lane A is independent of network probe failures.",
                  "", "## Verdicts", "",
                  "| Station | Hour | Bundle | n | Coverage | dHIT vs B0 | corrected CI | dHIT vs shipped direct-HIGH | shipped corrected CI | verdict |",
                  "|---|---:|---|---:|---:|---:|---|---:|---|---|"])
    for row in results["rows"]:
        interval = row["delta_hit_ci"]
        shipped = row["shipped_incumbent_comparison"]
        lines.append(f"| {row['station']} | {row['hour']} | {row['bundle']} | {row['n']} | {row['coverage']:.1%} | {row['delta_hit'] if row['delta_hit'] is not None else 'NA'} | {interval} | {shipped['delta_hit'] if shipped['delta_hit'] is not None else 'NA'} | {shipped['delta_hit_ci']} | {row['verdict']} |")
    lines.extend(["", "## Recommendation", ""])
    if results["recommendations"]:
        lines.append("Only arms clearing both the remaining-heat B0 bundle gate and the separately corrected shipped direct-HIGH incumbent gate appear below.")
        for rec in results["recommendations"]:
            lines.append(f"- {rec['station']} h{rec['hour']} {rec['bundle']}")
    else:
        lines.append("- No station/hour/bundle combination cleared every pre-registered gate.")
    lines.extend(["", "## Artifact integrity", ""])
    for name, digest in hashes.items():
        lines.append(f"- {name}: `{digest}`")
    lines.extend([f"- Ledger integrity: `{integrity.get('ok')}`", f"- Miniature replay bytes: `{miniature_bytes}`",
                  "", "## Test evidence", "",
                  "Focused and full pytest counts are recorded in `test-output.txt` after the external completion gate.", ""])
    return "\n".join(lines)


def run_evaluation(stations: Sequence[str], hours: Sequence[int], bundles: Sequence[str],
                   test_start: str, refit_every: int, bootstrap: int,
                   out: str | Path) -> tuple[dict[str, Any], bool]:
    from highcast.config import load_config
    from highcast.models.intraday import features_for
    import pandas as pd

    _validate_scope(stations, hours, bundles)
    out_path = Path(out)
    manifest_path = out_path / "manifest.json"
    feature_sha, config_shas = _current_fingerprints(stations)
    reproducibility = reproducibility_document(stations)
    manifest = validate_manifest_for_run(
        manifest_path, stations, hours, bundles, feature_sha, config_shas,
        reproducibility["closure_sha256"],
    )
    if test_start != manifest["test_start"] or int(refit_every) != int(manifest["refit_every_days"]):
        raise ValueError("run parameters are stale relative to the audited manifest")
    out_path.mkdir(parents=True, exist_ok=True)
    progress_path = out_path / "replay-progress.jsonl"
    run_id = f"phase0-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:10]}"
    writer = _LedgerWriter(
        out_path / "replay.db", PROJECT_ROOT / DEFAULT_RAW_ROOT, manifest, run_id,
        reproducibility["closure_sha256"], sha256_json(config_shas),
    )
    _append_progress(progress_path, {"event": "run_started", "run_id": run_id,
                                     "generated_utc": datetime.now(timezone.utc).isoformat()})
    candidate_bundles = [bundle for bundle in bundles if bundle != "B0"]
    declared_family_size = int(manifest["multiplicity"]["declared_arm_identities"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for station in stations:
            cfg = load_config(_config_path(station))
            conn = _read_only_connection(cfg.db_path)
            try:
                truth_rows, non_final_truth = load_final_cli_truth(conn, cfg)
                frame, training_hours, day_frame = _build_evaluation_frame(
                    conn, cfg, truth_rows, hours
                )
            except Exception as exc:
                errors.append({"station": station, "error": f"{type(exc).__name__}: {exc}",
                               "reason": "STATION_FRAME_ERROR"})
                _append_progress(progress_path, {"event": "station_error", "station": station,
                                                 "error": errors[-1]["error"]})
                continue
            finally:
                conn.close()
            noise_columns = _add_noise_columns(frame, station)
            base_columns = features_for(station, "high")
            hot_threshold = hot_day_threshold(frame, test_start)
            for hour in hours:
                all_eval = frame[(frame["hour"] == float(hour))
                                 & frame["remaining_heat_f"].notna()].copy()
                primary_eval = all_eval[all_eval["target_date"] >= test_start].copy()
                base_records: list[dict[str, Any]] = []
                shipped_records: list[dict[str, Any]] = []
                candidate_records = {bundle: [] for bundle in candidate_bundles}
                noise_records = {bundle: {seed: [] for seed in RANDOM_SEEDS} for bundle in candidate_bundles}
                comparison_errors = {bundle: [] for bundle in candidate_bundles}
                usable_totals = {bundle: int(primary_eval[f"{bundle}_usable"].sum())
                                 for bundle in candidate_bundles}
                hour_training = tuple(sorted({
                    (int(hour) + delta) % 24
                    for delta in range(-cfg.model.intraday_bandwidth_for("high"),
                                       cfg.model.intraday_bandwidth_for("high") + 1)
                }))
                blocks = walk_forward_blocks(
                    frame, int(hour), hour_training, int(refit_every), MINIMUM_TRAINING_ROWS
                )
                for block_spec in blocks:
                    block_dates = block_spec["dates"]
                    cutoff = block_spec["cutoff"]
                    raw_training = frame[
                        (frame["target_date"] < cutoff)
                        & frame["hour"].isin([float(value) for value in hour_training])
                        & frame["remaining_heat_f"].notna()
                    ].copy()
                    raw_block = all_eval[all_eval["target_date"].isin(block_dates)].copy()
                    fold_frame = refit_fold_baseline_inputs(
                        day_frame, pd.concat([raw_training, raw_block], ignore_index=True), cutoff, "high"
                    )
                    training = fold_frame[fold_frame["target_date"] < cutoff].copy()
                    block = fold_frame[fold_frame["target_date"].isin(block_dates)].copy()
                    fold_id = f"{station}-h{int(hour)}-{cutoff}"
                    try:
                        shipped_training_corpus = writer.retain_fold_training(
                            station, fold_id, "SHIPPED_DIRECT_HIGH",
                            "shipped_direct_high_intraday", base_columns, training, cutoff,
                            reproducibility,
                        )
                        base_training_corpus = writer.retain_fold_training(
                            station, fold_id, "B0", "remaining_heat_B0", base_columns,
                            training, cutoff, reproducibility,
                        )
                        shipped_model = shipped_direct_high_model(cfg, station).fit(training)
                        shipped_quantiles = shipped_model.predict_quantiles(block)
                        shipped_fp = _model_fingerprint(
                            base_columns, cutoff, len(training),
                            reproducibility, "shipped_direct_high_intraday",
                            shipped_training_corpus["training_content_sha256"],
                        )
                        base_model = _RemainingHeatModel(base_columns).fit(training)
                        base_quantiles = base_model.predict_quantiles(block)
                        base_fp = _model_fingerprint(
                            base_columns, cutoff, len(training),
                            reproducibility, "remaining_heat_B0",
                            base_training_corpus["training_content_sha256"],
                        )
                        for row_pair, shipped_q, base_q in zip(
                            block.iterrows(), shipped_quantiles, base_quantiles
                        ):
                            _, source_row = row_pair
                            extra = {
                                "rain_seen": _jsonable(source_row.get("wx_rain_seen_3h")),
                                "marine_persistent": _jsonable(source_row.get("low_cloud_persistent")),
                            }
                            shipped_record = _prediction_record(
                                str(source_row["target_date"]), source_row["y"], shipped_q, extra,
                            )
                            base_record = _prediction_record(
                                str(source_row["target_date"]), source_row["y"], base_q,
                                extra,
                            )
                            shipped_records.append(shipped_record)
                            base_records.append(base_record)
                            ledger_row = source_row.to_dict()
                            ledger_row["train_cutoff_date"] = cutoff
                            writer.write(station, ledger_row, "SHIPPED_DIRECT_HIGH", base_columns,
                                         shipped_record, fold_id, len(training), shipped_fp, feature_sha,
                                         shipped_training_corpus)
                            writer.write(station, ledger_row, "B0", base_columns, base_record, fold_id,
                                         len(training), base_fp, feature_sha, base_training_corpus)
                    except Exception as exc:
                        error = {"station": station, "hour": int(hour),
                                 "bundle": "B0_AND_SHIPPED_DIRECT_HIGH", "fold": fold_id,
                                 "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(error)
                        for bundle in candidate_bundles:
                            comparison_errors[bundle].append(error)
                        continue
                    for bundle in candidate_bundles:
                        bundle_columns = list(_BUNDLE_FIELDS.get(bundle, _B4_FIELDS[station]))
                        candidate_training = training[training[f"{bundle}_usable"]].copy()
                        candidate_block = block[block[f"{bundle}_usable"]].copy()
                        if len(candidate_training) < MINIMUM_TRAINING_ROWS or candidate_block.empty:
                            continue
                        try:
                            columns = [*base_columns, *bundle_columns]
                            candidate_training_corpus = writer.retain_fold_training(
                                station, fold_id, bundle, f"remaining_heat_{bundle}", columns,
                                candidate_training, cutoff, reproducibility,
                            )
                            candidate_model = _RemainingHeatModel(columns).fit(candidate_training)
                            predicted = candidate_model.predict_quantiles(candidate_block)
                            model_fp = _model_fingerprint(
                                columns, cutoff, len(candidate_training),
                                reproducibility, f"remaining_heat_{bundle}",
                                candidate_training_corpus["training_content_sha256"],
                            )
                            for (_, source_row), quantiles in zip(candidate_block.iterrows(), predicted):
                                record = _prediction_record(
                                    str(source_row["target_date"]), source_row["y"], quantiles,
                                    {"rain_seen": _jsonable(source_row.get("wx_rain_seen_3h")),
                                     "marine_persistent": _jsonable(source_row.get("low_cloud_persistent"))},
                                )
                                candidate_records[bundle].append(record)
                                ledger_row = source_row.to_dict(); ledger_row["train_cutoff_date"] = cutoff
                                writer.write(station, ledger_row, bundle, columns, record, fold_id,
                                             len(candidate_training), model_fp, feature_sha,
                                             candidate_training_corpus)
                            for seed in RANDOM_SEEDS:
                                noise_cols = noise_columns[(bundle, seed)]
                                noise_model_columns = [*base_columns, *noise_cols]
                                noise_arm = f"{bundle}_NOISE_{seed}"
                                noise_training_corpus = writer.retain_fold_training(
                                    station, fold_id, noise_arm, f"remaining_heat_{bundle}_noise_{seed}",
                                    noise_model_columns, candidate_training, cutoff, reproducibility,
                                )
                                noise_model = _RemainingHeatModel(noise_model_columns).fit(candidate_training)
                                noise_predictions = noise_model.predict_quantiles(candidate_block)
                                noise_fp = _model_fingerprint(
                                    noise_model_columns, cutoff, len(candidate_training),
                                    reproducibility, f"remaining_heat_{bundle}_noise_{seed}",
                                    noise_training_corpus["training_content_sha256"],
                                )
                                for (_, source_row), quantiles in zip(candidate_block.iterrows(), noise_predictions):
                                    record = _prediction_record(
                                        str(source_row["target_date"]), source_row["y"], quantiles,
                                        {"rain_seen": _jsonable(source_row.get("wx_rain_seen_3h")),
                                         "marine_persistent": _jsonable(source_row.get("low_cloud_persistent"))},
                                    )
                                    noise_records[bundle][seed].append(record)
                                    ledger_row = source_row.to_dict(); ledger_row["train_cutoff_date"] = cutoff
                                    writer.write(station, ledger_row, noise_arm,
                                                 noise_model_columns, record, fold_id,
                                                 len(candidate_training), noise_fp, feature_sha,
                                                 noise_training_corpus)
                        except Exception as exc:
                            error = {"station": station, "hour": int(hour), "bundle": bundle, "fold": fold_id,
                                     "error": f"{type(exc).__name__}: {exc}"}
                            comparison_errors[bundle].append(error)
                            errors.append(error)
                    writer.commit()
                    _append_progress(progress_path, {"event": "fold_complete", "run_id": run_id,
                                                     "station": station, "hour": int(hour), "fold": fold_id,
                                                     "train_rows": len(training), "test_rows": len(block)})
                for bundle in candidate_bundles:
                    excluded = [
                        {"date": str(row.target_date), "reason": "CANDIDATE_SOURCE_UNUSABLE"}
                        for row in primary_eval[~primary_eval[f"{bundle}_usable"]].itertuples()
                    ]
                    excluded.extend({"date": item["date"], "reason": item["reason"]}
                                    for item in non_final_truth if item["date"] >= test_start)
                    if comparison_errors[bundle]:
                        excluded.extend(comparison_errors[bundle])
                    primary_base = [record for record in base_records
                                    if record["target_date"] >= test_start]
                    primary_shipped = [record for record in shipped_records
                                       if record["target_date"] >= test_start]
                    primary_candidate = [record for record in candidate_records[bundle]
                                         if record["target_date"] >= test_start]
                    primary_noise = {
                        seed: [record for record in noise_records[bundle][seed]
                               if record["target_date"] >= test_start]
                        for seed in RANDOM_SEEDS
                    }
                    result = _evaluate_comparison(
                        station, int(hour), bundle, primary_base, primary_candidate,
                        primary_noise, usable_totals[bundle] / max(1, len(primary_eval)),
                        int(bootstrap), excluded, hot_threshold,
                        shipped_incumbent=primary_shipped,
                        full_history_incumbent=base_records,
                        full_history_candidate=candidate_records[bundle],
                        full_history_shipped=shipped_records,
                    )
                    if comparison_errors[bundle]:
                        result["verdict"] = "ERROR"
                    rows.append(result)
            _append_progress(progress_path, {"event": "station_complete", "run_id": run_id,
                                             "station": station, "frame_rows": len(frame),
                                             "training_hours": list(training_hours)})
        existing = {(row["station"], row["hour"], row["bundle"]) for row in rows}
        for station in stations:
            for hour in hours:
                for bundle in candidate_bundles:
                    if (station, int(hour), bundle) in existing:
                        continue
                    error = next((item for item in errors if item.get("station") == station),
                                 {"station": station, "reason": "MISSING_DECLARED_RESULT"})
                    row = _evaluate_comparison(
                        station, int(hour), bundle, [], [],
                        {seed: [] for seed in RANDOM_SEEDS}, 0.0, int(bootstrap),
                        [error], hot_threshold=float("inf"), shipped_incumbent=[],
                        full_history_incumbent=[], full_history_candidate=[],
                        full_history_shipped=[],
                    )
                    row["verdict"] = "ERROR"
                    rows.append(row)
        _apply_multiplicity(rows, declared_family_size)
        generated = datetime.now(timezone.utc).isoformat()

        def strip_private(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: strip_private(item) for key, item in value.items()
                        if not str(key).startswith("_")}
            if isinstance(value, list):
                return [strip_private(item) for item in value]
            return value

        clean_rows = strip_private(rows)
        document = {
            "schema_version": 1, "experiment": manifest["experiment"],
            "generated_utc": generated, "run_id": run_id,
            "declared_primary_comparisons": declared_family_size,
            "minimum_training_rows": MINIMUM_TRAINING_ROWS,
            "full_history_definition": "all expanding-window OOS dates after the declared minimum, including pre-test-start dates",
            "rows": clean_rows,
            "recommendations": [
                {"station": row["station"], "hour": row["hour"], "bundle": row["bundle"]}
                for row in clean_rows if row["verdict"] == "PASS"
            ],
            "candidate_errors": errors,
        }
        validate_results_document(document)
        miniature = replay_miniature(PROJECT_ROOT / "tests/fixtures/earlier_high_miniature.json")
        _atomic_write(out_path / "miniature-replay.json", miniature + b"\n")
        writer.finish(bool(errors), errors)
    except Exception as exc:
        failure = {"reason": "UNHANDLED_EVALUATION_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        try:
            writer.finish(True, [*errors, failure])
        finally:
            writer.close()
        raise
    else:
        writer.close()
    integrity = verify_ledger(out_path / "replay.db", PROJECT_ROOT / DEFAULT_RAW_ROOT)
    _write_json(out_path / "ledger-integrity.json", integrity)
    document = finalize_verified_results(document, integrity)
    _write_json(out_path / "results.json", document)
    _atomic_write(out_path / "test-output.txt", b"Pending focused and full pytest completion gate.\n")
    hashes = {
        "manifest_sha256": _sha256_file(out_path / "manifest.json"),
        "results_sha256": _sha256_file(out_path / "results.json"),
        "replay_sha256": _sha256_file(out_path / "replay.db"),
    }
    _atomic_write(out_path / "report.md", _artifact_report(
        manifest, document, hashes, integrity, len(miniature)
    ).encode("utf-8"))
    _append_progress(progress_path, {"event": "run_complete" if not errors else "run_complete_with_errors",
                                     "run_id": run_id, "generated_utc": generated,
                                     "result_rows": len(clean_rows), "errors": len(errors)})
    return document, bool(errors or not integrity.get("ok"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--stations", required=True)
    audit.add_argument("--hours", required=True)
    audit.add_argument("--out", default=str(DEFAULT_OUT))
    run = commands.add_parser("run")
    run.add_argument("--stations", required=True)
    run.add_argument("--hours", required=True)
    run.add_argument("--bundles", required=True)
    run.add_argument("--test-start", required=True)
    run.add_argument("--refit-every", required=True, type=int)
    run.add_argument("--bootstrap", required=True, type=int)
    run.add_argument("--out", default=str(DEFAULT_OUT))
    verify = commands.add_parser("verify-ledger")
    verify.add_argument("--db", required=True)
    verify.add_argument("--raw-root", required=True)
    miniature = commands.add_parser("replay-miniature")
    miniature.add_argument("--fixture", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            stations = _parse_csv(args.stations, str)
            hours = _parse_csv(args.hours, int)
            manifest = run_audit(stations, hours, args.out)
            print(canonical_json({
                "manifest": str(Path(args.out) / "manifest.json"),
                "audit_complete": manifest["audit_complete"],
                "station_scoring_safe": manifest["station_scoring_safe"],
                "lane_b": manifest["lane_b"]["conclusion"],
            }).decode("utf-8"))
            return 0 if manifest["audit_complete"] else 1
        if args.command == "run":
            stations = _parse_csv(args.stations, str)
            hours = _parse_csv(args.hours, int)
            bundles = _parse_csv(args.bundles, str)
            document, failed = run_evaluation(
                stations, hours, bundles, args.test_start, args.refit_every,
                args.bootstrap, args.out,
            )
            print(canonical_json({
                "results": str(Path(args.out) / "results.json"),
                "rows": len(document["rows"]), "recommendations": document["recommendations"],
                "candidate_errors": len(document["candidate_errors"]),
            }).decode("utf-8"))
            return 1 if failed else 0
        if args.command == "verify-ledger":
            report = verify_ledger(args.db, args.raw_root)
            print(canonical_json(report).decode("utf-8"))
            return 0 if report["ok"] else 1
        print(replay_miniature(args.fixture).decode("utf-8"))
        return 0
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
