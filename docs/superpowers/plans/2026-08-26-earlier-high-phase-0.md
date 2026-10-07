# Earlier HIGH Weather-Regime Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the complete offline Phase 0 feasibility evaluation, append-only replay evidence, and review-gate artifacts for KMIA/KLAX HIGH at h9-h13 without changing live behavior.

**Architecture:** Add one isolated Lane A feature module and one offline evaluation CLI. The CLI owns an experiment-only replay schema, canonical serialization/raw lineage, auditing, walk-forward scoring, artifact generation, integrity checking, and deterministic miniature replay. Existing `latentmax`, quantization, model, PMF, and bucket utilities remain authoritative.

**Tech Stack:** Python 3.14, SQLite, pandas/numpy, LightGBM through existing highcast models, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-earlier-high-weather-regime-design.md`

## Global Constraints

- Phase 0 only; stop at its review gate.
- Daily HIGH only, stations KMIA and KLAX, local checkpoints h9-h13.
- Every observation query includes `station = ?` and `valid_utc <= asof_utc`; no cross-station fallback.
- Visible maxima come only from `latentmax.visible_running_extremum`; use existing settlement quantization.
- Missing observations remain missing; missing gust is NaN and missing weather is not “no rain.”
- Lane B is historical only when exact run/fetch availability is proved; otherwise label `FORWARD_ONLY` and emit no retrospective score.
- Replay inputs, features, predictions, and outcomes are append-only; corrections append with `supersedes_id`.
- Do not edit `features_for`, `INTRADAY_FEATURES`, `RATE_FEATURES`, live CLI prediction code, gating, staking, bot, dashboard, or trading/order modules.
- The project is intentionally not Git; run no Git commands and claim no commits.
- Write tests before implementation and preserve RED/GREEN command evidence in task reports.

---

### Task 1: Leakage-safe Lane A feature builder

**Files:**
- Create: `src/highcast/weather_features.py`
- Create: `tests/test_weather_regime_features.py`

**Interfaces:**
- Produces `checkpoint_utc(target_date: str, hour: int, station_tz: str) -> datetime`.
- Produces `build_lane_a_features(conn, cfg, target_date: str, hour: int) -> dict` with named B1-B4 values, units, missingness, and lineage row identities.
- Produces `rain_state(raw: str | None) -> str` and circular wind helpers; no raw angular subtraction.

- [ ] Write focused tests that fail because the module is absent: DST conversion for both station zones; station/as-of SQL isolation; future rows cannot affect trends; rain missing/no-token/rain distinction; gust NaN; circular wind encoding; B3/B4 timing derived only from prior rows.
- [ ] Run `.venv/bin/pytest -q tests/test_weather_regime_features.py` and record the expected collection/import failure.
- [ ] Implement the smallest builder using METAR rows only, explicit `station/source/valid_utc` predicates, station-specific onshore direction from config/existing semantics, and feature metadata.
- [ ] Re-run the focused test file and record exact counts and elapsed time.

### Task 2: Append-only replay contracts and miniature fixture

**Files:**
- Create: `tests/test_earlier_high_eval.py`
- Create/extend: `scripts/earlier_high_eval.py`
- Create: `tests/fixtures/earlier_high_miniature.json`

**Interfaces:**
- Produces `canonical_json(value) -> bytes` and SHA-256 fingerprint helpers rejecting NaN.
- Produces `ensure_replay_schema(conn)` with the four spec tables and indexes, plus SQLite triggers that reject UPDATE/DELETE.
- Produces append functions for runs, source events, snapshots, predictions, outcomes; all use plain INSERT.
- Produces `verify_ledger(db_path, raw_root) -> dict` and `replay_miniature(fixture_path) -> bytes`.

- [ ] Write failing tests for canonical byte vectors, duplicate write rejection, update/delete rejection, superseding outcomes, future-known lineage rejection, METAR archive known-time semantics, immutable snapshot FK, tampered/missing raw detection without repair, no-trade forward shadow retention, and byte-for-byte miniature replay.
- [ ] Run the named replay tests and record RED failures caused by missing behavior.
- [ ] Implement the experiment-local schema, atomic gzip raw writer (`temp sibling -> flush/fsync -> rename`), append APIs, verifier, and fixed miniature replay.
- [ ] Re-run replay tests and record GREEN evidence.

### Task 3: Audit, walk-forward evaluation, and generated review artifacts

**Files:**
- Extend: `scripts/earlier_high_eval.py`
- Extend: `tests/test_earlier_high_eval.py`
- Generate: `data/experiments/earlier_high/manifest.json`
- Generate: `data/experiments/earlier_high/results.json`
- Generate: `data/experiments/earlier_high/report.md`
- Generate: `data/experiments/earlier_high/ledger-integrity.json`
- Generate: `data/experiments/earlier_high/miniature-replay.json`
- Generate: `data/experiments/earlier_high/test-output.txt`
- Append only: `data/experiments/earlier_high/replay-progress.jsonl`

**Interfaces:**
- CLI subcommands: `audit`, `run`, `verify-ledger`, `replay-miniature` with the spec arguments.
- `audit` writes a fingerprinted manifest containing feature status, source/units/formula/lookback/missingness/earliest date, coverage, invariant violations, Lane B probe findings, and unsafe-source refusal state.
- `run` requires a fresh safe manifest, evaluates B0-B4 independently per station/hour with matched dates, deterministic noise arms/bootstrap, Holm correction, and exact spec verdict gate; it records all raw lineage, snapshots, predictions, and outcomes.

- [ ] Write failing tests for remaining-heat invariant/floor, quantile non-crossing, matched dates, deterministic bootstrap/noise, Holm known vector, stale/unsafe manifest refusal, station-hour independent verdicts, capture-only/live import boundaries, and machine-readable artifact schemas.
- [ ] Run the named tests and record RED failures.
- [ ] Implement audit with existing CLI truth, `visible_running_extremum`, settlement quantization, observation coverage, descriptive correlations only, and bounded Open-Meteo exact-run probes whose unavailable/insufficient result becomes `FORWARD_ONLY`.
- [ ] Implement expanding-window same-hour evaluation reusing `features_for(station, "high")`, existing LightGBM quantile/sample-weight conventions and PMF/bucket utilities; never use market data.
- [ ] Implement deterministic paired metrics, parity averaging, random controls, Holm intervals/verdicts, robustness slices, exclusions, error continuation with final nonzero exit, and append-only run logging.
- [ ] Run `audit`, then `run` with the exact Phase 0 command from the spec and generate all artifacts.
- [ ] Run `verify-ledger` and `replay-miniature`; store their machine-readable reports.
- [ ] Run `.venv/bin/pytest -q tests/test_weather_regime_features.py tests/test_earlier_high_eval.py`, then `.venv/bin/pytest -q`; capture exact output in `test-output.txt` and summarize it in `report.md`.
- [ ] Re-read the spec stop gate, verify no forbidden live files changed, and stop for human review with exact station/hour/bundle recommendations.

