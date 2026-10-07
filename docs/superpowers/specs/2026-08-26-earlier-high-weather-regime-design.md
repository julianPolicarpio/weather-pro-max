# Earlier KMIA/KLAX HIGH Weather-Regime Predictor — Spec & Execution Contract

**Doc version:** 1.0 — 2026-08-26  
**Status:** Approved design; implementation has not started.  
**Scope:** Daily HIGH only, stations KMIA and KLAX. Accuracy first; no trading changes.  
**Canonical project:** `/Users/julian/projects/weather-temp-predictor`  
**Run instruction:** A Codex task may execute Phase 0 from this file. It must stop at
the Phase 0 review gate before changing any live model.

## 0. TL;DR

Build a leakage-safe experiment that improves HIGH forecasts at station-local
checkpoints h9, h10, h11, h12, and h13. Instead of asking a model to relearn the
whole daily high, predict the heating still unresolved after the visible running
maximum, using station-specific weather-regime features. Test observational
features now; capture exact forecast-run weather fields forward until their sample
is large enough.

**The core architectural insight:**

> Predict the remaining heating from the information state at each hour, not the
> final high from a flat bag of weather fields. Exact issue-time availability is
> part of every feature's definition; a physically plausible field without an
> honest historical as-of value is not a backtest feature.

This drives three decisions: matched-hour training remains mandatory, feature
bundles are gated separately by station and hour, and exact forecast-run weather
fields begin as capture-only unless Phase 0 proves adequate history.

## 1. Existing system facts that constrain this work

- Ground truth is the final NWS CLI daily HIGH, not a raw sensor maximum.
- Never derive the running high from raw `obs` snapshots. Reuse
  `latentmax.visible_running_extremum(...)`, which includes valid `six_max_c`
  groups and climate-day window guards.
- KLAX and KMIA share one SQLite database. Every persisted or cached record must
  include `station`; no cross-station fallback is allowed.
- The current live intraday path already trains at its prediction hour. This spec
  does not recreate "one model per hour" as if it were missing.
- Current HIGH decision hour is h13 for both stations. KMIA h13 was chosen partly
  for price rather than maximum predictive accuracy; this project evaluates
  prediction quality only and does not change that policy.
- KLAX HIGH already ships `cur_rate_3h`, `cur_rate_1h`, and
  `cur_rise_from_min`. Broad environment additions previously diluted it.
- Same-morning heat alerts, the prior convection bundle, generic environment
  features, and several calibration variants were measured and rejected. Do not
  rerun them unchanged.
- The repository is intentionally not a Git repository. Do not run Git commands
  and do not claim commits.

## 2. Goals and success criteria

| ID | Goal | Measurable success |
|---|---|---|
| G1 | Produce honest earlier HIGH forecasts | Leakage-safe walk-forward predictions at h9–h13 for both stations |
| G2 | Improve the 2°F bucket decision | Parity-averaged modal-bucket hit improves over the shipped model at the same station/hour |
| G3 | Preserve distribution quality | CRPS does not worsen materially; MAE and calibration are reported, not hidden |
| G4 | Identify weather-regime value | Each bundle is evaluated alone and against a random-noise control |
| G5 | Avoid cross-market generalization | A bundle ships only for the station and hours where it independently passes |
| G6 | Make failures reproducible | One command writes manifest, coverage, scores, paired deltas, confidence intervals, and verdicts |
| G7 | Preserve future replayability | Every evaluated or shadow prediction can be reconstructed from append-only raw-input lineage, feature snapshots, code/config fingerprints, and later CLI outcomes |

### Non-goals

- LOW-temperature models.
- Changing `MIN_DECISION_HOUR`, betting windows, staking, calibration tables,
  price rails, order placement, or dashboard advice.
- Treating Kalshi prices as truth or using agreement with the market as evidence.
- Adding a neural network, recurrent model, or external paid ML platform.
- Backfilling exact intraday model-run values from observed/reanalysis weather.
- Shipping all candidate features together.
- Retrying rejected heat-alert or convection experiments without a materially
  different feature definition and a pre-registered gate.

## 3. Recommended approach

### 3.1 Target

For station `s`, climate day `d`, and checkpoint `h`:

```text
visible = latentmax.visible_running_extremum(conn, cfg, d, "max", asof_utc)
visible_floor_f = settlement-safe visible maximum derived through existing helpers
remaining_heat_f = cli_high_f - visible_floor_f
```

The implementation must define `visible_floor_f` using existing quantization and
settlement semantics, not a new `round()` call. Phase 0 must audit the invariant
that `remaining_heat_f >= 0`. Any violations are reported by station/date and are
excluded until explained; they are not silently clipped.

Model quantiles of `remaining_heat_f`, then reconstruct:

```text
predicted_high_q = visible_floor_f + predicted_remaining_heat_q
predicted_high_q = max(predicted_high_q, visible_floor_f)
```

Enforce non-crossing quantiles after reconstruction. Convert to the existing 1°F
PMF and parity-averaged 2°F buckets through existing project utilities.

### 3.2 Why not add all fields to the shipped model?

LightGBM feature subsampling has already shown dilution when correlated or weak
columns are added. The unit of experimentation is therefore a small, named bundle.
The incumbent feature set is the control; only one candidate bundle is added per
arm. Combined bundles are tested only after two individual bundles pass.

### 3.3 Two data lanes

**Lane A — immediately backtestable observations**

Derived only from METAR rows at or before the checkpoint:

- temperature, dewpoint, dewpoint depression, calculated RH;
- 1h and 3h temperature trends;
- wind east/north components, station-specific onshore component, gust when
  present in raw METAR, and 1h/3h direction/component changes;
- present-weather precipitation flags parsed from raw METAR;
- visibility, lowest cloud layer, ceiling, sky state, and their trends;
- pressure and pressure-tendency fields already stored or derivable as-of.

Historical METAR retrieval must follow the project's existing IEM/archive path;
the live Aviation Weather Center API exposes only about 30 days and is not the
multi-year backfill source.

**Lane B — exact forecast-run fields, capture-forward by default**

- relative humidity/dewpoint forecasts;
- precipitation probability, precipitation/rain amount;
- cloud layers and visibility;
- wind speed, direction, and gusts;
- shortwave/direct radiation;
- optional CAPE/CIN only as a newly defined timing/recovery feature, not the
  rejected aggregate convection bundle.

Open-Meteo currently exposes these fields, but its Historical Forecast API
stitches initial hours from successive runs. That is not automatically the exact
forecast available at h9–h13. The Single Runs API preserves initialization time,
but most non-ECMWF run archives begin only in 2026. Phase 0 must prove exact
coverage before any Lane B field is allowed into a retrospective arm. Otherwise,
capture it with true `fetched_at` and mark the model experiment `FORWARD_ONLY`.

## 4. Candidate bundles

Every bundle must have a manifest listing source table, formula, units, allowable
lookback, missingness policy, and earliest trustworthy date.

### B0 — incumbent

Use `features_for(station, "high")` exactly. This is the same-hour baseline.

### B1 — thermodynamic state

```text
cur_rh_pct
cur_dewpoint_depression_f
rh_change_1h
dewpoint_change_3h
wet_bulb_proxy_f (only if calculated from as-of temperature/RH with a tested formula)
```

Do not include apparent temperature: the target is physical temperature and the
underlying temperature, humidity, wind, and radiation fields are already explicit.

### B2 — wind/advection transition

```text
wind_u_kt, wind_v_kt
onshore_change_1h, onshore_change_3h
wind_speed_change_1h
wind_direction_sin, wind_direction_cos
gust_excess_kt = gust_kt - sustained_kt
```

Never use raw angular subtraction. Missing gust means unknown (`NaN`), not zero.

### B3 — rain/cloud suppression and recovery

```text
wx_rain_now
wx_rain_seen_3h
hours_since_rain
visibility_change_1h
ceiling_change_1h
lowest_cloud_change_1h
cloud_clearing_flag
temp_recovery_since_rain_f
```

This is not "CAPE again." It measures whether observed rain/cloud suppression is
active and whether heating has resumed. If historical raw METAR parsing cannot
reliably distinguish missing weather from no rain, fail the bundle.

### B4 — station-specific timing

**KLAX:** marine-layer burn-off timing: first hour ceiling clears/rises above a
pre-registered threshold, low-cloud persistence, onshore-flow transition, and
temperature acceleration after clearing.

**KMIA:** rain-cooled recovery timing: rain onset/end, minimum temperature after
rain, warming rate since rain end, persistent cloud flag, and gust-front/wind
shift proxy.

Thresholds must be declared before scoring. They may be chosen from meteorological
meaning or training-only data, never from test-set bucket hit.

### B5 — forecast weather (forward-only unless Phase 0 clears history)

Aggregate only the future hours from checkpoint through the station's empirically
relevant peak window:

```text
future_rain_prob_max
future_rain_sum_mm
future_cloud_low_mean
future_shortwave_sum
future_rh_mean
future_gust_max
future_onshore_mean
first_solar_threshold_hour
first_rain_hour
```

Do not average over a broad fixed window when timing is the mechanism. Preserve
first/last threshold-crossing times.

## 5. Architecture and data flow

```text
CLI truth ──────────────────────────────────────────────┐
                                                       │
METAR archive ──> as-of parser ──> Lane A features ────┤
                                                       ├─> checkpoint frame
exact run capture ─> weather capture table ─> Lane B ──┤       │
                                                       │       v
visible_running_extremum ──> remaining-heat target ────┘  walk-forward arms
                                                               │
                                                               v
                                                  PMF / 2°F bucket scoring
                                                               │
                                                               v
                                                paired bootstrap + verdict
                                                               │
                                                               v
                                                append-only replay ledger
```

All work is offline/background. No request-time dashboard path changes in this
spec. Network probes are restricted to Phase 0 coverage checks and optional
capture wiring after the review gate.

## 6. Proposed data structures

Do not extend `forecast_hourly`; it has a deliberate capture-only contract for
temperature and freeze tests protecting that contract. Add a separate table only
if Phase 0 reaches the capture phase:

```sql
CREATE TABLE IF NOT EXISTS forecast_weather_hourly (
  station TEXT NOT NULL,
  valid_utc TEXT NOT NULL,
  model TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  run_utc TEXT,
  relative_humidity_2m REAL,
  dewpoint_f REAL,
  precipitation_probability REAL,
  precipitation_mm REAL,
  rain_mm REAL,
  cloud_cover REAL,
  cloud_cover_low REAL,
  visibility_m REAL,
  wind_speed_kt REAL,
  wind_direction_deg REAL,
  wind_gust_kt REAL,
  shortwave_wm2 REAL,
  direct_radiation_wm2 REAL,
  cape_jkg REAL,
  cin_jkg REAL,
  source_payload TEXT,
  PRIMARY KEY (station, valid_utc, model, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_fcweather_asof
  ON forecast_weather_hourly (station, fetched_at, valid_utc, model);
```

Experiment manifest example:

```json
{
  "experiment": "earlier_high_remaining_heat_v1",
  "generated_utc": "2026-08-26T10:00:00+00:00",
  "stations": ["KMIA", "KLAX"],
  "metric": "high",
  "hours": [9, 10, 11, 12, 13],
  "train_window": "expanding",
  "test_start": "2025-01-01",
  "refit_every_days": 45,
  "primary_metric": "parity_averaged_bucket_hit",
  "bundles": ["B0", "B1", "B2", "B3", "B4"],
  "random_seeds": [1729, 2718, 3141],
  "data_cutoffs": {
    "observations": "valid_utc <= checkpoint_utc",
    "forecasts": "run_utc <= checkpoint_utc AND fetched_at <= checkpoint_utc"
  }
}
```

Result rows must be machine-readable:

```json
{
  "station": "KMIA",
  "hour": 11,
  "bundle": "B3",
  "n": 571,
  "coverage": 0.91,
  "bucket_hit": 0.604,
  "delta_hit": 0.021,
  "delta_hit_ci": [-0.004, 0.046],
  "crps": 0.842,
  "delta_crps": -0.018,
  "noise_delta_hit_p975": 0.012,
  "verdict": "REJECT"
}
```

### 6.1 Durable replay and audit logging

The existing `predictions` table is not sufficient for future backtesting: it
stores forecast outputs but not the exact feature vector, model/config version,
or input lineage. `app_bets.features_json` is also unsuitable because it covers
only committed trading cases and belongs to a frozen decision ledger.

Create a separate, append-only replay ledger. It must serve two uses:

1. **Backward test replay:** prove exactly which rows, feature definitions, and
   model version produced an historical experimental score.
2. **Forward test accumulation:** store every scheduled shadow forecast—including
   days with no trade or no eventual PASS—then join the final CLI outcome later.

No `UPDATE` or `INSERT OR REPLACE` is allowed for input, feature, or prediction
events. Corrections are new rows linked with `supersedes_id`; the original event
remains queryable.

```sql
CREATE TABLE IF NOT EXISTS replay_runs (
  run_id TEXT PRIMARY KEY,
  experiment TEXT NOT NULL,
  spec_version TEXT NOT NULL,
  started_utc TEXT NOT NULL,
  completed_utc TEXT,
  mode TEXT NOT NULL CHECK (mode IN ('backtest','forward_shadow')),
  code_fingerprint TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  manifest_json TEXT NOT NULL,
  status TEXT NOT NULL,
  error_text TEXT
);

CREATE TABLE IF NOT EXISTS replay_source_events (
  source_event_id TEXT PRIMARY KEY,
  run_id TEXT,
  station TEXT NOT NULL,
  source TEXT NOT NULL,
  requested_utc TEXT,
  fetched_utc TEXT NOT NULL,
  known_utc TEXT NOT NULL,
  known_utc_basis TEXT NOT NULL,
  valid_from_utc TEXT,
  valid_to_utc TEXT,
  request_json TEXT,
  http_status INTEGER,
  raw_path TEXT,
  raw_sha256 TEXT NOT NULL,
  parser_version TEXT NOT NULL,
  supersedes_id TEXT,
  FOREIGN KEY (run_id) REFERENCES replay_runs(run_id),
  FOREIGN KEY (supersedes_id) REFERENCES replay_source_events(source_event_id)
);
CREATE INDEX IF NOT EXISTS idx_replay_source_asof
  ON replay_source_events (station, source, fetched_utc);

CREATE TABLE IF NOT EXISTS replay_feature_snapshots (
  feature_snapshot_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  station TEXT NOT NULL,
  target_date TEXT NOT NULL,
  checkpoint_hour REAL NOT NULL,
  asof_utc TEXT NOT NULL,
  bundle TEXT NOT NULL,
  feature_schema_version INTEGER NOT NULL,
  feature_code_sha256 TEXT NOT NULL,
  features_json TEXT NOT NULL,
  missing_json TEXT NOT NULL,
  lineage_json TEXT NOT NULL,
  visible_floor_f REAL NOT NULL,
  source_max_known_utc TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES replay_runs(run_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_feature_identity
  ON replay_feature_snapshots
     (run_id, station, target_date, checkpoint_hour, bundle);

CREATE TABLE IF NOT EXISTS replay_predictions (
  replay_prediction_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  feature_snapshot_id TEXT NOT NULL,
  station TEXT NOT NULL,
  target_date TEXT NOT NULL,
  checkpoint_hour REAL NOT NULL,
  issued_utc TEXT NOT NULL,
  bundle TEXT NOT NULL,
  fold_id TEXT NOT NULL,
  model_fingerprint TEXT NOT NULL,
  train_cutoff_date TEXT NOT NULL,
  train_row_count INTEGER NOT NULL,
  point_f REAL NOT NULL,
  quantiles_json TEXT NOT NULL,
  buckets_json TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES replay_runs(run_id),
  FOREIGN KEY (feature_snapshot_id)
    REFERENCES replay_feature_snapshots(feature_snapshot_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_prediction_identity
  ON replay_predictions
     (run_id, station, target_date, checkpoint_hour, bundle, fold_id);

CREATE TABLE IF NOT EXISTS replay_outcomes (
  outcome_id TEXT PRIMARY KEY,
  station TEXT NOT NULL,
  target_date TEXT NOT NULL,
  cli_high_f INTEGER NOT NULL,
  product_utc TEXT,
  recorded_utc TEXT NOT NULL,
  raw_sha256 TEXT NOT NULL,
  is_final INTEGER NOT NULL CHECK (is_final IN (0,1)),
  supersedes_id TEXT,
  FOREIGN KEY (supersedes_id) REFERENCES replay_outcomes(outcome_id)
);
CREATE INDEX IF NOT EXISTS idx_replay_outcome_day
  ON replay_outcomes (station, target_date, recorded_utc);
```

Raw provider responses are stored as gzip-compressed JSON/text outside SQLite:

```text
data/replay/raw/<source>/<YYYY>/<MM>/<DD>/<source_event_id>.json.gz
```

Write each raw file to a temporary sibling, `fsync`, then atomically rename it.
Record SHA-256 after the final bytes are written. A source event with no retained
raw body must still hash a canonical representation of the parsed input and set
`raw_path = NULL`; Phase 0 must report which legacy sources have this limitation.

Canonical JSON rules are required for reproducible hashes: UTF-8, sorted keys,
compact separators, explicit JSON `null` for missing values, and no NaN tokens.
Feature snapshots store units and feature-schema version in their manifest.

`known_utc` is source-specific and must be explicit. For a METAR observation it
is the report's `valid_utc`, even if an archive backfill downloaded it years later;
`known_utc_basis = 'observation_valid_time'`. For forecast guidance it is the
later of the model-run availability and the actual fetch time;
`known_utc_basis = 'forecast_available_and_fetched'`. For CLI outcomes it is the
product publication time, never the climate date. A feature snapshot may reference
only source events with `known_utc <= asof_utc`.

Retention policy: keep all replay rows and raw files for both stations. The volume
is small relative to model training data, and deleting old events would destroy the
forward-test sample. Provide an integrity command, not a cleanup command:

```bash
.venv/bin/python scripts/earlier_high_eval.py verify-ledger \
  --db data/highcast.db --raw-root data/replay/raw
```

It verifies foreign keys, raw-file existence, SHA-256, duplicate logical events,
schema versions, and that every prediction references one immutable feature
snapshot. It must never rewrite or repair records automatically.

## 7. Evaluation protocol

### 7.1 Dataset

- Stations scored independently.
- Test dates: default `>= 2025-01-01`; also report full-history and warm-season
  slices without using them to override the primary verdict.
- Hours: 9, 10, 11, 12, 13 local wall clock, converted with each station's IANA
  timezone. Never hardcode Manila offsets or assume DST.
- Same dates for incumbent and candidate within each station/hour/bundle.
- Report `n`, feature coverage, excluded dates, and reasons.

### 7.2 Walk-forward model

- Expanding-window, same refit cadence as comparable existing scripts (default
  45 days unless Phase 0 establishes a different current project standard).
- Reuse the existing LightGBM quantile factory and sample-weight conventions.
- No random hyperparameter sweep. Candidate model parameters equal incumbent
  parameters unless the remaining-heat target structurally requires one declared
  change.
- Generate three deterministic random-noise control arms with seeds in the
  manifest. The control has the same number of columns as the candidate bundle.

### 7.3 Metrics

Primary:

1. parity-averaged 2°F modal-bucket hit;
2. paired delta versus incumbent at the same station/hour.

Secondary safeguards:

- CRPS and paired delta;
- MAE and signed bias;
- 50% and 90% interval coverage;
- modal-probability calibration gap;
- hot-day slice (station-specific training-only threshold, then frozen);
- observed-rain and no-rain slices for KMIA;
- marine-layer persistence slice for KLAX;
- invariant and missingness counts.

### 7.4 Multiplicity and ship gate

Phase 0 tests 2 stations × 5 hours × up to 4 immediate bundles. Use a family-wise
correction across the declared primary comparisons. Holm correction is preferred
over a flat Bonferroni threshold; implement it once and test it.

A bundle/hour/station is a **PASS** only when all hold:

1. `delta_hit > 0` and the multiplicity-corrected 95% paired bootstrap interval
   excludes zero;
2. candidate `delta_hit` exceeds the 97.5th percentile of its random controls;
3. CRPS does not worsen by more than 0.01 absolute;
4. effective `n >= 400` and feature coverage `>= 80%` on the matched test rows;
5. no severe calibration regression (`abs(calibration_gap)` increase <= 0.02);
6. direction is not reversed in both the full-history and warm-season robustness
   slices.

Passing one hour does not authorize neighboring hours. Passing KMIA does not
authorize KLAX.

## 8. Failure handling and quality gates

| Failure | Required behavior |
|---|---|
| Feature timestamp unavailable | Mark `UNSAFE_ASOF`; exclude bundle, never impute from a later fetch |
| Exact forecast run cannot be reconstructed | Mark `FORWARD_ONLY`; create no historical score |
| Missing METAR report | Preserve NaN; never interpret missing as calm, clear, or no rain |
| Remaining-heat invariant violation | Emit station/date audit and stop that station's run |
| Too few matched rows | `INSUFFICIENT_DATA`, not PASS/REJECT |
| Candidate crashes | Record error and continue other independent arms; final command exits nonzero |
| Network unavailable | Phase 0 local Lane A evaluation still runs; coverage probe reports unavailable |
| Random control appears significant | Reject the candidate family as search/overfit contamination |
| Replay log write fails | Fail the prediction/evaluation closed; never emit an unlogged shadow forecast |
| Raw hash or lineage verification fails | Mark run `CORRUPT`, exclude it from scoring, preserve evidence |

## 9. Files and interfaces

Expected new files during the approved Phase 0 execution:

```text
scripts/earlier_high_eval.py
tests/test_earlier_high_eval.py
tests/test_weather_regime_features.py
src/highcast/weather_features.py
data/experiments/earlier_high/manifest.json       # generated, not hand-edited
data/experiments/earlier_high/results.json        # generated
data/experiments/earlier_high/report.md           # generated
data/replay/raw/...                               # append-only raw source archive
```

Only after a separate capture approval:

```text
src/highcast/fetchers/forecast_weather.py
tests/test_forecast_weather_capture.py
src/highcast/replay.py
tests/test_replay.py
src/highcast/db.py or src/highcast/migrate.py      # additive migration only
```

CLI for Phase 0:

```bash
.venv/bin/python scripts/earlier_high_eval.py audit \
  --stations KMIA,KLAX --hours 9,10,11,12,13

.venv/bin/python scripts/earlier_high_eval.py run \
  --stations KMIA,KLAX --hours 9,10,11,12,13 \
  --bundles B0,B1,B2,B3,B4 --test-start 2025-01-01 \
  --refit-every 45 --bootstrap 10000 \
  --out data/experiments/earlier_high
```

`audit` must complete before `run`. The run command refuses to start if the
manifest is absent, stale relative to feature code, or contains unsafe sources.

## 10. Testing requirements

Write tests before implementation behavior.

Minimum unit tests:

- checkpoint UTC conversion across DST for KMIA and KLAX;
- every SQL observation query has `station` and `valid_utc <= asof`;
- rain token parsing distinguishes missing report, no weather token, and rain;
- circular wind encoding and no raw angle subtraction;
- gust missingness remains NaN;
- trend features cannot read future observations;
- remaining-heat invariant and reconstruction floor;
- non-crossing reconstructed quantiles;
- candidate/incumbent matched-date scoring;
- deterministic bootstrap and random controls;
- Holm correction on known p-values;
- no imports from trading/order modules;
- no reader of capture-only forecast fields in live prediction paths.
- append-only enforcement: duplicate logical writes fail rather than replace;
- canonical JSON and stable SHA-256 test vectors;
- source lineage cannot reference a `known_utc` after `asof_utc`, with fixtures
  proving that a later archive-download time does not invalidate an older METAR;
- replay prediction requires an existing immutable feature snapshot;
- final CLI revisions append and supersede rather than overwrite;
- `verify-ledger` detects missing/tampered raw files and never repairs them;
- backward replay reproduces stored features and predictions byte-for-byte on a
  fixed miniature fixture;
- forward-shadow fixture stores days with no trade and later joins final CLI truth.

Required verification:

```bash
.venv/bin/pytest -q tests/test_weather_regime_features.py \
  tests/test_earlier_high_eval.py
.venv/bin/pytest -q
```

The full suite is the completion gate. Record exact pass/fail/deselected counts
and elapsed time in the generated report.

## 11. Phase 0 — coverage and feasibility spike

### Work

1. Audit current schema, observation coverage, feature availability, and the
   remaining-heat invariant for h9–h13.
2. Implement only the leakage-safe Lane A feature builder behind tests.
3. Produce descriptive coverage and correlation tables without scoring candidates.
4. Probe Open-Meteo Single Runs/Previous Runs for exact issue-time coverage of
   Lane B variables for both stations and representative dates in 2024–2026.
5. Write the manifest with each feature labeled `HISTORICAL_SAFE`,
   `FORWARD_ONLY`, or `REJECTED_SOURCE`.
6. Specify and test the canonical serialization, fingerprints, and append-only
   replay contracts on a miniature local fixture. Phase 0 may create the replay
   schema behind experiment code, but must not wire live scheduled predictions.
7. If Lane A clears feasibility, run the pre-registered B0–B4 evaluation and
   store its run, feature snapshots, predictions, and outcomes in the ledger.

### Kill criteria

Rethink the remaining-heating design if either station has:

- more than 0.5% unexplained negative remaining-heat targets;
- fewer than 400 matched test rows at three or more target hours;
- less than 80% usable coverage for all non-incumbent Lane A bundles;
- random controls that repeatedly clear the candidate threshold.

Lane B historical modeling is killed—not the whole project—if exact as-of run
coverage is below 400 matched rows per station/hour. Continue capture-forward.

### Phase 0 review gate

Codex must stop after producing:

- `manifest.json`;
- `results.json`;
- `report.md` with PASS/REJECT/INSUFFICIENT_DATA verdicts;
- a passing `verify-ledger` report and a replayed miniature fixture;
- full test output;
- a concise recommendation naming the exact station/hour/bundle combinations,
  if any, that justify an implementation phase.

Do not edit `features_for`, `INTRADAY_FEATURES`, `RATE_FEATURES`, live CLI
prediction code, gating, staking, bot, or dashboard files in Phase 0.

## 12. Later phases (not authorized by this file alone)

### Phase 1 — walking skeleton

For one passing station/hour/bundle only, wire the remaining-heating candidate
behind an explicit opt-in experiment flag. Generate a shadow prediction alongside
the incumbent; do not serve it to the decision desk.

### Phase 2 — forward capture

If Lane B is forward-only, add the separate capture table and fetcher, true
`fetched_at`, raw-response archiving, staleness monitoring, and no readers outside
experiment code. Schedule append-only h9–h13 shadow snapshots for both stations,
including days on which inputs are missing; log the missing-input verdict rather
than silently skipping the day.

### Phase 3 — shadow validation

Accumulate at least 100 settled out-of-sample station-days and apply a frozen
verdict. Do not resweep thresholds on the same window.

### Phase 4 — guarded live adoption

Requires a new approved spec. Adopt only exact passing station/hour combinations,
retain instant rollback, and regenerate all probability calibration artifacts that
consume the changed model.

## 13. Cost and licensing

Verified 2026-08-26; re-check before commercialization.

| Item | Expected development use | Cost/constraint |
|---|---:|---|
| Existing SQLite/CPU | Local walk-forward evaluation | $0 marginal; compute time is the main cost |
| Aviation Weather Center API | Small live probes | $0; 100 requests/minute, ~30-day database window |
| Open-Meteo free API | Coverage probes and personal non-commercial capture | $0; under 10,000 calls/day, 5,000/hour, 600/minute; no uptime guarantee |
| Open-Meteo data | Stored derived fields | CC BY 4.0 attribution required |
| Commercial Open-Meteo use | Not part of Phase 0 | Paid customer plan required; historical APIs require Professional or higher according to current pricing |

Batch both stations and variables in each request where supported. A normal wake
cycle should need at most one forecast-weather request per provider/model update,
not one request per variable. Prices and limits change; verify before committing.

## 14. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Hindsight leakage from stitched or later forecast runs | Critical | Exact `run_utc`/`fetched_at` contract; otherwise forward-only |
| Search-driven overfitting across 40 comparisons | High | Pre-registered bundles, Holm correction, random controls, fixed primary metric |
| Feature dilution | High | One small bundle per arm; combined arms only after individual PASS |
| METAR rain tokens are sparse/inconsistent | High | Coverage audit and explicit missing/no-rain states; kill B3 if ambiguous |
| Climate-day/rounding mismatch | Critical | Existing latentmax and quantization helpers only; invariant audit |
| Station-specific result silently generalized | Critical | Explicit station/hour manifests and no fallback |
| Replay ledger silently loses or mutates history | Critical | Append-only tables, raw SHA-256, versioned schemas, integrity command, no automatic repair |
| Logging failure creates an unmeasured selection bias | High | Fail closed and record missing-input/error events for every scheduled checkpoint |
| Earlier accuracy still worse than h13 | Medium | Compare honestly; the goal is improvement at the same earlier hour, not pretending earlier equals later |
| API terms or availability change | Medium | Provider abstraction, capture raw payload, date-stamped source review |
| Long full-suite runtime | Low | Focused tests during iteration, full suite once at the gate |
| Over-engineering forecast capture before signal exists | High | Lane A Phase 0 first; no capture schema until review |

## 15. Codex execution instructions

When a Codex task is told to execute this spec:

1. Read `AGENTS.md`, `HANDOFF-2026-08-12.md`, `HANDOFF.md`, and this file.
2. Confirm the project is not a Git repository; do not initialize Git.
3. Restate that only Phase 0 is authorized.
4. Create a written implementation plan using the project's planning workflow.
5. Use test-driven development.
6. Preserve all unrelated files and current live behavior.
7. Do not use market prices as labels, features, or correctness evidence.
8. Run focused tests, then the full suite.
9. Run `verify-ledger` and replay the fixed miniature fixture.
10. Produce the Phase 0 artifacts and stop for human review.

Suggested task prompt:

```text
Execute Phase 0 only from
docs/superpowers/specs/2026-08-26-earlier-high-weather-regime-design.md.
Follow AGENTS.md and the spec's leakage, station-isolation, TDD, and stop-gate
requirements. Do not change live prediction, gating, trading, bot, or dashboard
behavior. Produce the manifest, machine-readable results, report, and test evidence,
plus an append-only replay ledger, raw-input lineage, a passing ledger-integrity
report, and a byte-for-byte miniature replay. Then stop for review.
```

## 16. Open questions for the Phase 0 report

These are evidence questions, not blockers to starting:

1. Does remaining-heating outperform direct-high prediction at any earlier hour?
   Default: require it to beat the shipped same-hour incumbent.
2. Is B3 METAR rain history complete enough for KMIA? Default: kill below 80%
   unambiguous coverage.
3. Does KLAX benefit from a threshold-crossing timing feature where window means
   failed? Default: test B4 separately and do not combine it with B1/B2.
4. Can exact forecast runs provide 400 matched rows per station/hour? Default:
   assume no until the audit proves yes; capture forward.
5. Should any PASS become shadow output? Default: only after a separate human
   approval following Phase 0.

## 17. Current primary-source references

- Open-Meteo Historical Forecast API:
  https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Previous Runs API:
  https://open-meteo.com/en/docs/previous-runs-api
- Open-Meteo Single Runs API:
  https://open-meteo.com/en/docs/single-runs-api
- Open-Meteo GFS/HRRR variables:
  https://open-meteo.com/en/docs/gfs-api
- Open-Meteo pricing and licence:
  https://open-meteo.com/en/pricing
  https://open-meteo.com/en/license
- Aviation Weather Center Data API:
  https://aviationweather.gov/data/api/
