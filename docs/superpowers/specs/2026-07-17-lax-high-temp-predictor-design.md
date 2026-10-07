# KLAX Daily High Temperature Predictor — Design

**Date:** 2026-07-17
**Status:** Approved by Julian (design discussion, this session)
**Target:** Kalshi LAHIGH contract (rules: `LAXHIGH.pdf` in repo root)

## 1. Goal

Predict the daily maximum temperature at Los Angeles International Airport (KLAX)
as it will appear in the NWS Daily Climate Report (CLI) — the exact number the
Kalshi LAHIGH contract settles on. Output a calibrated probability for every 1°F
bucket, both day-ahead and continuously updated intraday, running locally as a
Python project with a small web dashboard.

### What the contract settles on (from LAXHIGH.pdf)

- Underlying: max temperature in the NWS **Daily Climate Report for Los Angeles
  Airport, CA** for the specified date. Source agency: NWS.
- Expiration 10:00 AM ET the following morning; determination delayed to 11 AM ET
  if the high is inconsistent with METAR 6-hr/24-hr max groups, or if the final
  report's high is lower than an earlier report's.
- Strikes are 1°F increments; "greater than X" is strict (X itself does not count),
  "between A and B" is inclusive of both ends.
- Post-expiration revisions to the CLI report do not change settlement.

Implications: the ground truth series is the CLI product for KLAX, full stop.
Training, verification, and live tracking must target that series, and the system
should replicate the METAR-consistency check Kalshi itself applies.

## 2. Scope

- **In scope (v1):** KLAX only. Day-ahead forecast + live intraday nowcast. Full
  1°F bucket probability distribution + point estimate. Local dashboard + CLI.
  Walk-forward backtesting.
- **Architected for later:** additional Kalshi temperature stations (KNYC
  Central Park, KMDW, KMIA, KAUS, KDEN, KHOU, KPHL) — station ID is a parameter
  throughout; no KLAX hardcoding outside config.
- **Out of scope (v1):** any trading integration or order placement, raw GRIB
  ensemble processing (GEFS/ECMWF-ENS member data), cloud hosting, notifications,
  scraping general news media (lagging, unstructured duplicate of NWS data).
- **Later enhancement:** featurizing the LOX Area Forecast Discussion (AFD) text
  (forecaster commentary on marine-layer burn-off, offshore flow) — archived and
  testable, but extra NLP engineering not needed for v1.

## 3. Architecture

Local Python project, three layers plus a scheduler:

```
fetchers ──► SQLite (obs, forecasts, cli_reports, predictions)
                │
        feature builder
                │
   ┌────────────┴────────────┐
   │ baseline blend           │ LightGBM quantile models
   │ (yardstick)              │ (day-ahead + intraday)
   └────────────┬────────────┘
          calibration layer
                │
      1°F bucket probabilities
                │
     FastAPI dashboard + CLI
```

- **Storage:** single SQLite file. Every row carries `fetched_at` so the system
  always knows what was knowable when (leakage-free training and honest backtests).
- **Scheduler:** macOS `launchd` jobs — forecasts ~4×/day; KLAX obs every 5 min
  during 10:00–20:00 PT and every 15 min otherwise; CLI report fetched each
  morning LA time (verifies yesterday and becomes new training data). All job
  times computed in America/Los_Angeles regardless of the Mac's local timezone.
- **Backfill on wake:** the user is in the Philippines (Asia/Manila, LA+15/16h),
  so the Mac is typically asleep during LA's afternoon. On wake/startup, all
  fetchers backfill gaps from archive endpoints (IEM) before serving predictions
  — the dashboard must be correct whenever it's opened, not only if fetching ran
  continuously.
- **Interfaces:** CLI commands `predict`, `backtest`, `status`, `fetch`;
  dashboard at `localhost`.

## 4. Data sources (all free, no keys except as noted)

| Data | Source | Notes |
|---|---|---|
| CLI daily highs (truth), historical + live | Iowa Environmental Mesonet (IEM) parsed CLI API | Backup: `api.weather.gov` CLI text product for the LOX office, parsed locally |
| KLAX METAR (incl. T-groups at 0.1°C and 6-hr max temp groups in remarks) | aviationweather.gov data API; IEM for archive | Highest-precision real-time reading; settlement cross-check |
| KLAX 5-min ASOS feed | IEM / NWS data API (not the webpage — it lags ~10 min) | Coarsely quantized (whole-degree °F→°C→°F double rounding, per trader reports — verify empirically); each reading is a 1–2°F interval of possible true values |
| KLAX 1-min ASOS | IEM/NCEI, published after the fact only | Not available live; used to train the "hidden spike" model (how far the true max exceeds the best observable running max) |
| ECMWF, GFS, HRRR, ICON point forecasts for KLAX | Open-Meteo forecast API (`models=` param) | Free for non-commercial use |
| Archived past model runs (training data) | Open-Meteo previous-runs / historical-forecast APIs | Makes leakage-free training possible; several years of depth |
| NBM + NWS official point forecast | api.weather.gov gridpoints for KLAX (33.9425, −118.4081) | NWS forecast is a strong LAX baseline (marine-layer aware) |
| MOS max-temp guidance (MAV/MEX) for KLAX | IEM MOS archive | Cheap extra feature with long history |
| NWS heat alerts for the LAX zone (Excessive Heat Warning, Heat Advisory, Red Flag Warning) | api.weather.gov alerts API (live); IEM VTEC archive (history) | Structured "heatwave news"; leakage-free trainable |

Exact endpoint URLs/params are confirmed in the implementation plan (fetcher task
includes a smoke test against each live endpoint before anything else is built).

## 5. Modeling

### 5.1 Baseline (built first, kept forever)

Weighted average of available model max-temp forecasts, weights fit by inverse
historical squared error. Distribution = blend value + empirical error
distribution (season-conditioned). This is the yardstick every ML change must
beat in backtests. If ML fails to beat it, the baseline ships.

### 5.2 Day-ahead model

LightGBM quantile regression, ~19 quantiles (5%…95%), target = CLI high.

Features:
- Per-model max-temp forecasts (NBM, ECMWF, GFS, HRRR, ICON, MOS, NWS official)
- Inter-model spread and blend-vs-NBM deltas (disagreement = uncertainty signal)
- Day-of-year (cyclic encoding)
- Marine-layer proxies from latest KLAX obs at prediction time: dewpoint,
  onshore wind component, sky cover, morning temperature; yesterday's CLI high
- Active NWS heat-alert flags for the LAX zone (Excessive Heat Warning, Heat
  Advisory, Red Flag Warning) — heat events are where raw model forecasts bust
  hardest; alert-day error behavior is learned from the VTEC archive
- Climate-trend handling (see 5.4)

### 5.3 Intraday model

Same machinery, conditioned on the day in progress, but observation handling is
**rounding-aware** — the observable feeds do not show the true temperature:

- **Latent running max:** the running max is estimated as an interval, not a
  point. METAR T-groups (0.1°C) are precise hourly anchors; 5-min feed values
  are inverted through their rounding pipeline into candidate true-°F sets;
  METAR 6-hr max groups bound the answer each 6 hours.
- **Hidden-spike component:** the official CLI high comes from continuous 1-min
  data that is not published live and can exceed anything observable intraday.
  The distribution of (CLI high − best observable running max), conditioned on
  time of day and conditions, is learned from historical 1-min/5-min/METAR
  archives vs. CLI highs, and added to the nowcast distribution.
- **Quantization verification:** the °F→°C→°F double-rounding behavior of the
  5-min feed (reported by market participants) is verified empirically against
  archived METAR/1-min/CLI data before the inversion logic is trusted.

Conditioning features: latent running-max interval, local time, current
temp/dewpoint/wind/sky, morning distribution parameters. Trained by replaying
historical observation days at multiple checkpoint times (e.g., hourly
08:00–20:00 PT). Output replaces the day-ahead distribution as the day unfolds;
by mid-afternoon it is mostly reporting observed reality plus the hidden-spike
tail and remaining-heating uncertainty.

### 5.4 Climate-trend handling

The LA basin is warming; naive training on old years runs cool. Concretely:
- Exponential recency weighting of training samples (recent ~2 years dominate)
- Year-index trend feature and anomaly-vs-rolling-climatology framing (rolling
  10-year window), **not** the static 1991–2020 NWS normals
- The hot tail of predicted distributions is never clipped to the historical
  observed range — record highs get nonzero probability

### 5.5 Calibration

Isotonic/conformal correction fit on held-out recent years, applied to model
quantiles before bucketing, so stated bucket probabilities match observed
frequencies. Bucket edges follow Kalshi conventions from the PDF: "greater
than X" strict, "between A–B" inclusive.

## 6. Backtesting and success criteria

Walk-forward protocol: train on all data before day D (respecting `fetched_at`
availability), predict D, roll forward across the evaluation year. No shuffled
CV — time order only.

Success criteria (on most recent full held-out year):
1. Point MAE ≤ NWS official forecast MAE and ≤ raw NBM MAE for KLAX
2. CRPS and per-strike Brier score better than the baseline blend
3. Calibration: reliability diagram within ±5 percentage points across deciles
4. Intraday: monotonic accuracy improvement over the day; nowcast at 14:00 PT
   materially sharper than the morning distribution

## 7. Live operation, error handling, edge cases

- **Source outage:** drop the missing feature (models handle missing natively),
  widen the distribution via the calibration layer's missing-source mode; never
  reuse stale data silently — dashboard shows per-source freshness.
- **Settlement-consistency flag:** replicate Kalshi's check — warn when the
  5-min-derived running max disagrees with METAR 6-hr max groups by ≥1°F.
- **Climate-day boundary:** ASOS daily max uses the local-standard-time
  midnight-to-midnight day (during DST the boundary is 01:00 PDT). Verify
  against historical CLI reports during implementation and encode the exact
  boundary in the obs aggregation. This is a settlement-relevant detail.
- **Two CLI reports per day:** a preliminary report (~4:35 PM local) and a
  final report after the climate day closes. Evening/overnight warmth can raise
  the high after the preliminary — the nowcast keeps a live tail until the
  climate day actually ends, and verification always uses the final report
  (matching the contract's 10:00 AM ET expiration value).
- **Data hygiene:** missing-sensor days and post-hoc CLI corrections are logged
  and excluded from training; all fetches logged with timestamps.

## 8. Dashboard (v1)

- Today: bucket-probability bar chart, day-ahead vs. live distribution side by
  side, running (latent) max vs. forecast, countdown of remaining heating hours
- **Manila information clock:** the LAX information cycle rendered in
  Asia/Manila time — model-run arrivals, LA climate-day start (15:00 PHT),
  morning-obs window (~21:00–24:00 PHT, historically the sharpest pre-outcome
  forecast), peak heating (03:00–07:00 PHT), preliminary CLI (~07:35 PHT),
  market close (14:59 PHT), settlement (22:00 PHT); DST-aware. Alongside:
  historical forecast sharpness (CRPS) by time of day, so it is visible when
  the distribution typically stops moving. This is information timing only —
  bet decisions and sizing are manual and out of scope.
- Inputs: table of each model's forecast + freshness
- History: daily point/probabilistic accuracy vs. NWS and baseline, reliability
  plot

## 9. Build order

1. Fetchers + SQLite schema + endpoint smoke tests; assemble historical dataset
2. Baseline blend + backtest harness (usable forecast exists at this point)
3. Day-ahead LightGBM model + calibration; beat baseline in backtest
4. Intraday nowcast model
5. Dashboard + launchd scheduling
6. (Later) more stations, notifications, ensemble features
