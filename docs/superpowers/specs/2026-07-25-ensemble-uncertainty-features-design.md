# Ensemble Uncertainty Features (HIGH) — Design

Updated 2026-07-25. Give the HIGH day-ahead model an **honest width/shape
signal** from Open-Meteo ensemble members, so its probability distribution is
trustworthy in the marine-layer transition regime (~80–84°F) where it currently
overshoots. Goal is **calibration for betting**, not point accuracy.

## Problem — the marine-layer seam

Investigation (2026-07-25, walk-forward over 1,663 forecast-days) found the HIGH
model is well-calibrated overall (bias −0.23°F, MAE 1.60) but **biased by
temperature regime**:

| predicted | N | bias (P−A) | MAE | overshoot % |
|---|---|---|---|---|
| < 80 | 1549 | −0.21 | 1.51 | 47% |
| **80–82** | 35 | **+0.43** | 2.05 | **63%** |
| **82–84** | 39 | **+0.48** | 2.01 | **62%** |
| 84+ | 40 | −2.54 | 4.39 | 40% |

LAX summer highs are effectively **bimodal**: the marine layer either holds
(mid-upper 70s) or burns off / offshore flow wins (mid-80s+). **~80–82°F is the
seam** between the two regimes, where an hour's difference in burn-off timing
swings the high 3–4°F. On seam days the model produces a unimodal PMF whose
median falls in the empty *trough* between the two real modes (observed
2026-07-24: point 77.5 but modal 2°F bucket 83–85 — the distribution collapsed
into a false single peak).

**Root cause is NOT missing data.** The HIGH day-ahead model already uses
marine-layer features at a 9am cutoff (`morn_tmpf`, `morn_dwpf`, `morn_onshore`,
`morn_sky`, plus `fc_spread`, `fc_mean` — see `features.FEATURES`). It sees the
layer is present at 9am; it cannot tell whether it burns off by 11am or holds
till 2pm. Its **only uncertainty signal is `fc_spread` = max−min of ~6–9
deterministic models**, smoothed by a global PIT calibration — a thin,
crude width estimate that does not capture the burn-off split.

## Goal

On a genuinely bimodal day, the model should output an **honestly wide (and when
warranted, bimodal) distribution** — mass at both ~78 and ~84 rather than a
false peak at 81 — so the edge-vs-market gate (`staking.evaluate`) stops seeing
phantom edge on the trough bucket. Concretely: **seam-conditional coverage moves
toward nominal and CRPS improves**, without regressing the well-behaved <80
regime.

## Confirmed facts (from the codebase)

- `models/dayahead.py`: `DayAheadModel` fits LightGBM quantile models
  (`make_lgbm`) over `features.FEATURES` with recency weights, then a
  `PITCalibrator`. **LightGBM handles NaN natively** (splits on missingness) →
  new features need no imputation.
- `features.py`: `FEATURES` list + `build_features(conn, cfg, metric, dates)`;
  HIGH cutoff is **09:00 on the target day** (`metric.cutoff_local_hour=9`,
  offset 0) — so features legitimately include target-day morning obs, no leak.
- `fetchers/forecasts.py`: `OPEN_METEO = ".../v1/forecast"`, `MODELS` maps our
  names → Open-Meteo model ids; `fetch_openmeteo`/`parse_openmeteo` write rows
  to the `forecasts` table. `climate_day_bounds` defines the daily-max window.
- `db.py`: `forecasts (station, target_date, model, run_date, max_temp_f,
  fetched_at, min_temp_f, metric)`, PK includes `metric` (post-corruption-fix).
- Open-Meteo serves free ensemble members at `ensemble-api.open-meteo.com/v1/
  ensemble` (GFS ~31, ECMWF ~51, ICON ~40) — flagged available-but-unused in the
  handoff.

## Key definitions

- **Ensemble member** — one run of a model with perturbed initial
  conditions/physics; ~120 members across the three ensembles give ~120 plausible
  daily-max values for a target day. Their spread *is* measured uncertainty.
- **Seam day** — a day whose HIGH prediction lands in ~80–84°F, the marine-layer
  regime boundary.
- **Summarize-at-fetch** — we store per-day *summary stats* of the members, not
  the ~120 raw values.

## Design

### Component 1 — Ensemble fetcher (`fetchers/forecasts.py`)
- `fetch_openmeteo_ensemble(cfg, metric)` mirroring `fetch_openmeteo`: GET the
  ensemble endpoint with `models=gfs025,ecmwf_ifs025,icon_seamless`,
  `hourly=temperature_2m`. Response has `temperature_2m_memberNN` columns per
  model.
- `parse_openmeteo_ensemble(payload, icao, run_date, fetched_at, metric)`: for
  each member column, take the daily max over `climate_day_bounds` → a vector of
  member maxes per target day → compute the summary (Component 2) → yield one
  summary row per (station, target_date, run_date, metric). Exact model ids /
  member counts are confirmed by the **spike** (below).

### Component 2 — Storage (summary, new table `forecast_ens`)
- New table `forecast_ens (station, target_date, run_date, metric, n_members,
  ens_mean, ens_std, ens_p10, ens_p50, ens_p90, ens_frac_ge_82, ens_bimodality,
  fetched_at)`, PK `(station, target_date, run_date, metric)`. A sibling to
  `forecasts` (not new columns on it) because the grain is per-day-summary, not
  per-model.
- **Summary stats** computed at fetch: mean, std, p10/p50/p90 (empirical
  quantiles), `frac_ge_82` (fraction of members ≥ 82°F — a direct "how much
  clear-regime mass" signal), and `ens_bimodality` (a dip/gap statistic — exact
  choice is an Open Item; start with a simple normalized gap between the two
  largest member clusters).
- **Rationale for summarize-at-fetch:** once the shape stats exist the raw
  members are never needed again; keeps the table tiny and features direct.

### Component 3 — Features (`features.py`)
- Extend `FEATURES` with `ens_std, ens_p10, ens_p90, ens_frac_ge_82,
  ens_bimodality`. `build_features` populates them from `forecast_ens` using the
  run available by the metric cutoff (same leakage discipline as `fc_*`), NaN
  where unavailable.
- `ens_mean`/`ens_p50` are **stored** (Component 2) for debugging/future use but
  deliberately **not featured** — the distribution's *center* is already covered
  by `fc_mean`; these features add *width and shape*, which is the whole point.
- They sit **beside** `fc_spread`/`fc_mean` — the model gains a real
  ensemble spread/shape in addition to the crude deterministic spread.

### Component 4 — Model integration
- No structural change. `DayAheadModel`/baseline read `FEATURES`; LightGBM's
  native NaN handling covers rows with no ensemble data. PIT calibrator and
  recency weighting unchanged.

### Component 5 — Backfill + the data-availability spike (FIRST)
- **Spike (implementation step 1):** fetch KLAX ensemble history and record how
  far back it goes. This gates expectations: if the archive is thin (≈1–2 yr),
  the backtest can only *measure* lift on the recent slice (the nbm/nam/aifs
  dilution), and the honest framing becomes "a live-coverage improvement," to be
  decided with eyes open before building the rest.
- Backfill writes `forecast_ens` rows for the available history with correct
  `run_date`/`fetched_at` = true issue time (mirror `parse_openmeteo_lead1`).

### Component 6 — Validation (the acceptance gate)
- Walk-forward backtest **with vs without** the `ens_*` features, scored on
  **calibration and CRPS, not MAE**:
  - **Coverage** of the 50%/90% intervals vs nominal, computed **overall AND
    conditioned on the 80–84 seam**.
  - **CRPS** overall and on seam days.
  - Every seam metric reported **with its N**; small moves treated cautiously
    (thin seam sample).
- **Ship rule:** adopt `ens_*` only if seam coverage/CRPS improve (or at least
  don't regress) and the <80 regime doesn't degrade. Otherwise keep the model
  as-is and record the finding.

## Data flow

```
Open-Meteo /ensemble → ~120 member daily-maxes (per target_date, run_date)
  → parse: summarize (std, p10/p50/p90, frac_ge_82, bimodality)
  → forecast_ens table
  → build_features adds ens_* columns (run available by 9am cutoff; NaN if none)
  → DayAheadModel (LightGBM quantiles, native NaN) trains on wider feature set
  → PIT calibration → predictive PMF honestly wide/bimodal at the seam
  → Kalshi 2°F buckets → edge-vs-market gate (UNCHANGED)
```

## Error handling & edge cases

- **Missing / partial members** (endpoint returns fewer columns, or NaNs): parse
  what's present; if `n_members` below a floor (e.g. < 10), write NaN summary
  (LightGBM treats as missing) rather than a noisy stat. Log the shortfall.
- **No ensemble history for a training row:** `ens_*` = NaN — expected and safe.
- **Endpoint failure / rate limit:** fetch is best-effort like the other
  forecast fetchers; a miss leaves that day's `ens_*` NaN, never crashes the
  cycle.
- **Leakage:** only use the run whose issue time ≤ the metric cutoff; backfill
  stamps issue time, not backfill time.

## Testing

- **Unit** (`tests/`):
  - `parse_openmeteo_ensemble`: member-column extraction → per-member daily max
    over the climate-day window → correct summary stats (mean/std/quantiles,
    `frac_ge_82`, bimodality) on a fixture payload with a known bimodal split.
  - NaN/short-response handling (< floor members → NaN summary row).
  - Leakage/cutoff: a run issued after the cutoff is excluded.
  - `build_features` populates `ens_*` from `forecast_ens`; NaN when absent;
    HIGH path byte-identical when ensemble table is empty.
- **Empirical gate:** the Component 6 backtest (with vs without), reported on
  seam-conditional coverage + CRPS.

## Non-goals (YAGNI)

- The **distribution-blend** variant (mixing the members' empirical CDF directly
  into the predictive CDF) — a possible graduation *only if* the features prove
  out.
- The **LOW** metric — the seam is a HIGH-market problem; low is separate.
- Storing **raw** members.
- Any change to the **edge/betting** logic, gates, sizing, or dashboard.
- An explicit fitted **mixture model** (Approach C) — real members supply
  bimodality without the overfitting risk.

## Open items for review

- **Ensemble archive depth** — resolved by the Component 5 spike; may reframe the
  win as live-coverage rather than backtest lift.
- **Bimodality statistic** — exact form (dip test vs normalized two-cluster gap);
  start simple, revisit if it earns its place.
- **`frac_ge_82` threshold** — 82°F is the seam's upper edge; may tune (or add a
  second threshold) once we see the data.
- **Exact ensemble model ids / member counts** — confirmed by the spike against
  the live endpoint.
