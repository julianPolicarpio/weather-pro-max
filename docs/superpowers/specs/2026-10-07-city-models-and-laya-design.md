# City models (all 24 cities) and the Laya decision gate — design

Approved by Julian 2026-10-07. Sub-projects **2a, 2b and 3** of the "all weather markets + Laya"
programme; sub-project 1 is the Weather Desk Sandbox
(`2026-10-04-weather-sandbox-design.md`). Every data source below was verified live on
2026-10-07; quirks found during verification are binding requirements, not notes.

## 0. Goal, non-goals, success

**Goal.** For every Kalshi daily temperature market we trade (24 cities × high/low = 48 series),
produce (2b) a calibrated forecast distribution for the official NWS value that is honestly
compared against the market's own price, and (3) a bet/skip decision layer — a LightGBM
meta-labeler and a fine-tuned Laya — that is proven on held-out history to decide better than
the current hand rule.

**Non-goals.** Real-money trading (locked by the sandbox unlock rule for every model), stake
sizing (always 1 contract), international series, `KXHIGHUS`, order-book reconstruction for past
days, replacing the live KLAX/KMIA desk (`highcast`), which keeps running unchanged.

**What "done" means.**
- 2a: `citycast coverage` reports ≥ 98 % of days present per station since each source's start
  (§2.5), every gap listed.
- 2b: walk-forward forecasts exist for every station/metric/day/as-of point; a per-station/metric
  report shows CRPS vs the MOS baseline and binary Brier vs the market; the `citycast` sandbox bot
  trades only station/metrics that pass the gate in §3.8.
- 3: the test-period comparison in §4.7 is run exactly once and recorded, whichever way it falls;
  the winner (Laya, or LightGBM if Laya fails) becomes the sandbox `gate` bot.

## 1. Verified sources and their quirks (binding)

| Source | Endpoint | Depth verified | Notes |
|---|---|---|---|
| NWS CLI (truth) | IEM `json/cli.py?station=K…&year=Y` | 2021→, all 24 | Use **final** reports only (`highcast.fetchers.cli_reports.cli_row_is_final`). |
| MOS MEX, GFS, NBS, NBE | IEM `cgi-bin/request/mos.py` (`model=MEX|GFS|NBS|NBE`, `sts`/`ets`, `format=csv`) | 2022→, all 24 | `MAV` is named **`GFS`** at IEM. IEM returns "Too many requests" when hammered: ≤ 1 request/s, exponential backoff on HTTP 429 or that text. |
| ASOS METAR | IEM `cgi-bin/request/asos.py` (as `highcast.fetchers.metar.fetch_archive`) | 2021→ | Must keep `six_max_c`/`six_min_c` (project rule: never derive an extreme from spot readings alone). |
| Kalshi settled markets, recent | `GET /markets?series_ticker=S&status=settled` | only since the historical cutoff | Cutoff from `GET /historical/cutoff` (`market_settled_ts`, 2026-08-08 on 2026-10-07); it moves. |
| Kalshi settled markets, archive | `GET /historical/markets?series_ticker=S&limit=1000&cursor=…` | KXHIGHNY/CHI 2021-08→, AUS/MIA 2023-05→, DEN/PHIL/HOU 2024-11→, LAX 2025-01→, other highs 2026-01/02→, lows 2025-12 or 2026-04→ | Querying the **current** series ticker returns legacy tickers (`HIGHNY-21AUG06-T86`, `HIGHAUS--23MAY11-B84.5`). EWR, SAN, SDF, TTN have no archive (live only). Houston highs live under both `KXHIGHHOU` (70 events) and `KXHIGHTHOU`. |
| Kalshi hourly prices | `GET /historical/markets/{ticker}/candlesticks?start_ts&end_ts&period_interval=60`; recent markets `GET /series/{S}/markets/{ticker}/candlesticks` | 2021→ | Candles exist only for hours with activity (4–37 per market seen): **forward-fill**, max staleness 2 h. Fields `yes_bid`/`yes_ask` are OHLC dicts of decimal strings; `price` may be all null. No historical order book: past fills = 1 contract at the ask. |
| Laya | HF `convaiinnovations/laya` | — | ModernBERT-large, 421 M params, 512-token context, Apache-2.0, `pip install laya`, `Router().predict(state, questions)`; `noul` question type returns P(yes). Official fine-tune notebook: `NandhaKishorM/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` (Kaggle 2×T4). |

**Legacy market shape.** 2021-era markets have `strike_type`/`floor_strike`/`cap_strike` = null and
were threshold contracts ("strictly greater than 86°F"), not a mutually exclusive ladder. Every
market's YES interval is therefore derived in this order: (1) strike fields
(`between` → [floor, cap]; `greater` → [floor+1, ∞); `less` → (−∞, cap−1]); (2) else the rules
text (`strictly greater than X` → [X+1, ∞); `strictly less than X` → (−∞, X−1]; `between A-B` /
`A-B°` → [A, B]); (3) else the market is unusable and counted in coverage. Station comes from the
series alias table (§2.2), never from the rules text (it names places, not stations). Event date
comes from the `YYMMMDD` token anywhere in `event_ticker` (regex `(\d{2}[A-Z]{3}\d{2})`; the
double dash in `HIGHAUS--23MAY11` is real). `expiration_value` (Kalshi's settled number) is stored
and cross-checked against CLI.

## 2. Sub-project 2a — history store

### 2.1 Placement
New package `src/citycast/` and console script `citycast` (added to `pyproject.toml`). New DB
`data/history.db` (WAL, `busy_timeout` 30 s). Rules: never write `data/highcast.db` or
`data/sandbox.db`; never import `highcast.kalshi_client`; reuse `highcast` parsers/helpers by
import (read-only use): `fetchers.cli_reports.parse_rows / cli_row_is_final`,
`fetchers.metar.parse_metar`,
`timeutil.climate_day_bounds`, `latentmax.visible_running_extremum`, `buckets.*`, `fees.taker_fee`.
Those helpers take a `highcast.config.AppConfig`; `citycast.stations.app_config(icao)` builds one
per station from the registry.

### 2.2 Station registry
`config/citycast_stations.yaml`, checked in, one entry per station (24):
`icao, iem_id, iem_network, wfo, lat, lon, elevation_m, tz, std_utc_offset, high_series, low_series,
high_aliases, low_aliases`. Generated once by `scripts/build_citycast_stations.py` from IEM station
metadata (`geojson/network.py`) and the live Kalshi series list, then reviewed by hand; tz and
std offset must equal `sandbox.stations.STATIONS`. Aliases: every series ticker whose archive
belongs to the station (e.g. KHOU high: `KXHIGHHOU`, `KXHIGHTHOU`). A unit test asserts 24
entries, 48 distinct primary series, and agreement with `sandbox.stations.STATIONS`.

### 2.3 Schema (`citycast/db.py`)
```sql
CREATE TABLE stations (icao TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE cli_daily (station TEXT, date TEXT, high_f INTEGER, low_f INTEGER, final INTEGER NOT NULL,
  issued_utc TEXT, raw TEXT, fetched_at TEXT NOT NULL, PRIMARY KEY (station, date));
CREATE TABLE mos_runs (station TEXT, model TEXT, runtime_utc TEXT, target_date TEXT, metric TEXT,
  value_f REAL, n_points INTEGER NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, model, runtime_utc, target_date, metric));
CREATE TABLE obs (  -- identical columns to highcast.db `obs`, so latentmax runs unchanged
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, source TEXT NOT NULL, tmpf REAL, tmpf_precise REAL,
  dwpf REAL, sknt REAL, drct REAL, skyc TEXT, six_max_c REAL, raw TEXT, fetched_at TEXT NOT NULL,
  six_min_c REAL, slp_hpa REAL, pres_tend_hpa REAL, pres_tend_code INTEGER, vis_sm REAL,
  ceiling_ft INTEGER, sky_lowest_ft INTEGER, n_sky_layers INTEGER, sky_cb INTEGER, wx_present TEXT,
  wx_thunder INTEGER, PRIMARY KEY (station, valid_utc, source));
CREATE TABLE kalshi_markets (ticker TEXT PRIMARY KEY, series TEXT NOT NULL, event_ticker TEXT NOT NULL,
  station TEXT NOT NULL, metric TEXT NOT NULL, event_date TEXT NOT NULL, lo INTEGER, hi INTEGER,
  open_time TEXT, close_time TEXT NOT NULL, result TEXT, expiration_value TEXT, raw TEXT NOT NULL);
CREATE TABLE kalshi_candles (ticker TEXT, end_utc TEXT, yes_bid REAL, yes_ask REAL, volume REAL,
  open_interest REAL, PRIMARY KEY (ticker, end_utc));
CREATE TABLE backfill_log (source TEXT, key TEXT, done_at TEXT NOT NULL, rows INTEGER NOT NULL,
  PRIMARY KEY (source, key));
```
`mos_runs.value_f` = the run's forecast daily max (`metric='high'`) or min (`'low'`) for the
station's climate day `target_date`. `citycast/mos.py` parses the IEM CSV itself because the
highcast parsers keep only the run's local date and the as-of rule needs `runtime_utc`; it follows
the same conventions: climate day = `(ftime_utc + std_utc_offset h).date()`; MEX/GFS take `n_x`
from local-afternoon rows for the max and local-morning rows for the min; NBS/NBE take the max/min
of `tmp` over the run's points inside the climate day, with `n_points` recording how many (a run
issued mid-day covers only the rest of the day). `kalshi_candles.yes_bid/yes_ask` = the candle's `close` values.

### 2.4 Backfill (`citycast backfill SOURCE [--station K…] [--since YYYY-MM-DD]`)
SOURCE ∈ `cli | mos | obs | kalshi-markets | kalshi-candles | all`. Default `--since 2021-01-01`.
- Chunking: CLI per station-year; MOS per station-model-half-year; obs per station-month;
  markets per series (both endpoints, deduped, live endpoint wins); candles per market over
  `[open_time, close_time]` (one request per market; split if > 5,000 h).
- Resumable: each chunk records `backfill_log(source, key)`; a logged chunk is skipped unless
  `--refresh`. The current/most recent chunk is never logged as done (it is still filling).
- Idempotent: all writes `INSERT OR REPLACE` in one transaction per chunk.
- Pacing: `citycast.http.Paced(host_interval={'mesonet.agron.iastate.edu': 1.0,
  'api.elections.kalshi.com': 0.2})`; on 429 or IEM's "Too many requests" text, back off 5 s,
  15 s, 60 s, then fail the chunk (logged as an error, not done) and continue.
- Progress line per chunk; summary JSON at the end; exit 1 if any chunk failed.

### 2.5 Coverage (`citycast coverage [--json]`)
Per station: CLI final days / calendar days since 2021-01-01; per MOS model: days with a run
issued before the climate day starts / days since that model's first row; obs: days with ≥ 20
METARs; Kalshi: events, markets, markets with a usable interval, markets with ≥ 1 candle,
CLI-vs-`expiration_value` disagreements (listed). **Acceptance:** CLI and MEX/NBS ≥ 98 % for
every station; Kalshi coverage reported (no threshold — it is what Kalshi has). Every
disagreement between CLI and `expiration_value` is listed for a human; truth = CLI final.

## 3. Sub-project 2b — city models

### 3.1 Prediction problem
For station s, metric m ∈ {high, low}, climate day D and as-of instant t: a pmf over integer °F
for the final CLI value, using only information available at t.

### 3.2 As-of grid
Local wall-clock hours on D−1 at 06, 09, 12, 15, 18, 21 and on D at 00, 03, 06, 09, 12, 15, 18, 21
(14 points), converted with the station's tz; points after the climate day ends are dropped.
`lead_h` = hours from t to climate-day end.

### 3.3 Features (all strictly as-of t; `citycast/features.py`)
For each MOS model in (MEX, GFS, NBS, NBE): `f_{model}` = value_f of the latest run with
`runtime_utc ≤ t − lag(model)` for (s, D, m); `age_{model}` = hours since that runtime;
`npts_{model}` = that run's `n_points` (how many forecast points it has inside the climate day);
NaN when none. **Publication lag per model** (`PUB_LAG_H`, revised 2026-10-08 after the 2a review):
until measured, the conservative defaults GFS 5 h, MEX 5 h, NBS 3 h, NBE 3 h; 2b Task 0 measures
first-seen times on IEM's archive for 24 h and may only replace a default with
`max(observed max lag) + 0.5 h` (never shorter than observed). A lag that is too short leaks a
forecast that was not yet published into training. **Anchor and spread use full-coverage runs only**
(revised 2026-10-08 after the Task 2 review on real data): for MEX and GFS every run counts; for NBS
a run counts when `npts ≥ 6` (≥ 18 h of the climate day), and the anchor uses the **latest published
full-coverage run**, not the latest run (after local noon the newest NBS run is always partial, which
made the anchor jump between bases). `anchor` = mean of full-coverage NBS and MEX, else MEX, else GFS
(NBE never reaches 6 points in the archive, so it is a feature only, never an anchor; rows with no
anchor are dropped). `spread_mos` = max − min over the full-coverage values. `f_*`, `age_*`, `npts_*`
still report each model's latest published run, partial or not, so the model sees remaining-hours
forecasts too. Measured on the 2a archive: NBS/NBE daily max from 3-hourly points runs low
(≈ −1.5 °F NBS, ≈ −4.8 °F NBE). `spread_mos` = max − min of available f_*. Climatology
`clim_mean`, `clim_sd` = mean/sd of CLI final values for s, m over ±7 days of D's day-of-year in
**years before D's year** only. Persistence `prev_obs` = visible extreme of the previous climate day
from `obs` with `asof_utc = t` (via `latentmax.visible_running_extremum`). Same-day (t ≥ climate
day start): `run_ext` = visible running extremum so far, `run_gap = run_ext − anchor`,
`last_tmpf`, `last_dwpf`; NaN pre-day. Calendar `doy_sin`, `doy_cos`, `lead_h`, `local_hour`.
Station `lat`, `lon`, `elevation_m`, and `station` as a categorical (shared model only).

### 3.4 Target and model
Target residual `r = y − anchor` (y = CLI final). LightGBM quantile regression (`objective=
'quantile'`), one model per α in Q = {0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
0.90, 0.95, 0.98}; separate models per metric. Params: `num_leaves=31, learning_rate=0.05,
n_estimators=400, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8`,
sample weights = recency with half-life 730 days (as `highcast.models.dayahead`). Predicted
quantiles of y = anchor + q̂_α, sorted ascending (crossing fix), then
`highcast.buckets.bucket_probs` → integer pmf; same-day rows truncated with
`buckets.truncate_at_bound(pmf, run_ext, extremum)`.

### 3.5 Shared vs per-city
Two variants per metric: **shared** (one model, all stations, `station` categorical) and **city**
(one model per station, no station features). Both always trained and stored.

### 3.6 Calibration
Per (station, metric, variant): a spread factor k ∈ {0.6, 0.7, …, 1.6} scaling every quantile's
distance from the median, chosen to minimise mean CRPS on that variant's **previous** walk-forward
folds only (k = 1.0 until 6 prior months exist). Applied before the pmf is built.

### 3.7 Walk-forward protocol
Quarterly refits: for each quarter Q from 2022-Q3 to the latest complete quarter, train on all
rows with D < start(Q) − 1 day, predict every row with D in Q. The one-day embargo (revised
2026-10-09) exists because Q's first event has as-of points on start(Q) − 1, when that day's
own high/low is not yet known; the k history (§3.6) uses the same cutoff. Each run first deletes
the metric's existing `wf_predictions`, so k never reads rows from an older configuration. Training data begins 2022-01-01 (MOS
depth). Output table in `history.db`:
```sql
CREATE TABLE wf_predictions (station TEXT, metric TEXT, event_date TEXT, asof_utc TEXT,
  variant TEXT, fold TEXT, anchor REAL, run_ext REAL, quantiles_json TEXT NOT NULL,
  pmf_json TEXT NOT NULL, k REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (station, metric, event_date, asof_utc, variant));
```
`variant` ∈ {shared, city, mosnorm}; `mosnorm` = the sandbox `mos` baseline
(Normal(f_MEX, σ = 3.0 °F)) for the same rows.

### 3.8 Evaluation, selection and gate (`citycast evaluate`)
- **CRPS** on the integer pmf vs CLI, per (station, metric, variant), clustered by event date.
- **Market Brier**: for each Kalshi market of a canonical event (§4.1) whose `result` is
  "yes"/"no", with a candle within 2 h before an as-of point while open, binary Brier of P_model(y ∈ [lo, hi]) vs the outcome, beside the market mid
  ((bid+ask)/2) at that hour; difference (model − market) with a cluster bootstrap by event
  (2,000 resamples) and its 95 % CI.
- **Selection period** = folds 2022-Q3 … 2026-Q1; **report period** = 2026-Q2 onward.
  Per (station, metric) choose the variant (shared/city) with lower mean CRPS in the selection
  period; record it in `model_choice(station, metric, variant, crps, decided_at)`.
- **Gate** for the sandbox (both must hold; revised 2026-10-09):
  1. *Selection test:* the chosen variant's CRPS minus mosnorm CRPS has a 95 % cluster-CI upper
     bound < 0 over the selection period.
  2. *Report confirmation:* over the report period the same difference has mean < 0 on at least
     60 distinct event dates (fewer → fail). A point estimate, not a CI: six months per station
     is too little power for a second CI test, and the job here is to catch a model that only
     looked good in-sample. Report-period numbers never choose the variant or k.
- **Design freeze and fingerprint.** Every `citycast evaluate` run logs a sha256 fingerprint of
  the modelling code (`asof, features, dataset, qmodel, metrics, walkforward, evaluate` modules) in
  `eval_log`. Once the report period has been scored, any later run under a different
  fingerprint puts a warning at the top of the report: the report period is no longer an
  untouched test and only the sandbox counts. Real data is not evaluated before `PUB_LAG_H` is
  set from the Task 0 measurement, so that planned change does not spend the report period.
  Known residual: design rules (e.g. the §3.3 anchor rule) were revised after inspecting the
  structure of data from all periods, though never model scores; the sandbox forward test is
  the only fully clean test.
- Report: `docs/reports/citycast-evaluation-<date>.md` generated by the command.

### 3.9 Live use
`citycast train --final` fits the chosen variants on all data (monthly, by hand for now);
`citycast predict` computes predictions for every open (station, metric, event) at the current
as-of and writes `live_predictions` (same columns as `wf_predictions` plus `issued_utc`).
Sandbox: new bot `CitycastBot(ForecastBot)` in `src/sandbox/bots.py`, account `citycast` ($5),
reads `data/history.db` read-only (`?mode=ro`), uses the latest `live_predictions` row with
`issued_utc ≤ snapshot.taken_at` for gated (station, metric) pairs only; `forecast_json` =
`{"source":"citycast","variant":…,"issued_utc":…}`. `citycast predict` runs on its own
(by hand, or a launchd job installed only with Julian's explicit yes); the sandbox never calls
it. With no fresh prediction (issued within 2 h of the snapshot) the bot skips that event.

## 4. Sub-project 3 — Laya decision gate

### 4.1 Decision rows (`citycast decisions build`)
Table `decision_rows` in `history.db`, one row per (market ticker, as-of point t) where the market
is open at t and a candle exists in (t − 2 h, t]:
`ticker, station, metric, event_date, asof_utc, lo, hi, p_model` (chosen variant's
wf_prediction mass on [lo, hi]), `model_median, model_iqr` (q0.80 − q0.20)`, ask, bid, mid, spread, market_p` (= mid; ask when
no bid), `fee` (`fee_per_contract(1, ask)`, as the sandbox),
`edge = p_model − ask − fee, hours_to_close, lead_h, run_gap_to_lo, run_gap_to_hi` (NaN pre-day),
`month, won` (**Kalshi's own `result`**: 1 for "yes", 0 for "no"), `profit_if_bet = won − ask − fee`.
Win/loss comes from what Kalshi paid, not the CLI: the 2a archive has 14 events where Kalshi settled
on a different value than the NWS final report (6 of them changed a payout); the city models keep
CLI final as weather truth. Markets whose `result` is not "yes"/"no" (8 cancelled events in the
archive) are excluded. Where one (station, metric, event_date) has more than one event (KMDW
2024-02-28), only the canonical event is used: the one with the most markets, ties broken by the
lexicographically smallest `event_ticker` (`evaluate.canonical_events`, also used in §3.8).
Rows with ask < 0.01 or > 0.99 are excluded. One sandbox-style constraint applies at evaluation
time, not build time: at most one bet per ticker (the first as-of point where the rule fires).

### 4.2 Splits (by event_date)
Train ≤ 2026-03-31; validation 2026-04-01 … 2026-06-30; **test 2026-07-01 … 2026-09-30, used once**.
Clustered statistics use the city-day (station, metric, event_date).

### 4.3 Contenders (each outputs P(win) per row; rule: bet iff P − ask − fee ≥ margin)
- B0 never bet. B1 hand rule: P = p_model, margin = `highcast.orderbot.MIN_EDGE` (0.03).
- B2 LightGBM binary classifier (`objective='binary'`) on the §4.1 features (excluding `won`,
  `profit_if_bet`), same params as §3.4 except `n_estimators=300`; margin tuned.
- L0 Laya zero-shot; L1 Laya fine-tuned. Margin for B2/L0/L1 chosen on validation from
  {0.00, 0.01, …, 0.10} maximising profit per decision with ≥ 50 bets.

### 4.4 Laya input
State = compact JSON, ≤ 200 tokens:
`{"city":"KLAX","metric":"high","bucket":"66-67","hours_to_close":14.5,"lead":"pre_day",
"model_p":0.41,"model_median":66.8,"model_iqr":3.1,"ask":0.33,"bid":0.30,"market_p":0.315,
"edge_after_fee":0.06,"running_gap":null,"month":1}`. Question:
`{"win":{"type":"noul","instructions":"Will this YES contract settle YES?"}}`; P(win) =
`answers.win.noul`.

### 4.5 Fine-tune
Export `citycast decisions export --split train|validation` → JSONL in the official notebook's
dataset format (pinned in Plan 3 Task 1 by reading the notebook; the format is not guessed).
Training set = all rows of a random 60 % of training city-days, capped at 100,000 rows (seeded);
B2 is reported both on all training rows and on the same subset. Fine-tune on Kaggle 2×T4 with the
official notebook (including its calibration-temperature step); push to a **private** HF repo
`julianPolicarpio/laya-weather-gate`; inference locally with `Router`, model pinned by revision.

### 4.6 Leakage controls (tests)
Every feature builder has a test that inserts data issued after t and asserts the row is
unchanged. Decision rows use only `wf_predictions` (never final-model predictions). The test
split is read only by `citycast decisions final-report`, which refuses to run twice unless
`--force` (records `meta.final_report_run_at`).

### 4.7 Success criterion ("we fine-tuned Laya and it decides better")
On the test split: L1 profit per decision − X > 0 with cluster-bootstrap 95 % CI lower bound > 0,
for X ∈ {B1, B2, L0}; and L1's P(win) Brier ≤ B2's. If all hold, Laya becomes the gate; if not,
the result is recorded and B2 becomes the gate if B2 beats B1 by the same test, else B1 stays.
The gate enters the sandbox as bot `gate` (account `gate`, $5): it takes the citycast bot's
candidate rows each decision hour and bets those its model approves. It outputs P(win) only and
never sets stake. Real money stays locked by the sandbox unlock rule.

## 5. Testing
`.venv/bin/python -m pytest`. Unit tests never touch the network (fake `get` returning recorded
fixtures under `tests/fixtures/citycast/`). Each plan task is test-first. Leakage tests per §4.6.
Full suite before reporting each task done.

## 6. Order and dependencies
2a (now; independent of the sandbox) → 2b (after `citycast coverage` passes) → 3 (after
`wf_predictions` exist). The sandbox cycle should run meanwhile; its live candidates are the
final out-of-sample check for both 2b and 3. Estimates [Guessing]: 2a ≈ 2–3 build sessions +
~6 h unattended backfill (≈ 90,000 candle requests at 5/s); 2b ≈ 3 sessions + hours of training
(48 quarters×metrics×(1+24) variants×13 quantiles); 3 ≈ 2 sessions + one Kaggle run.

## 7. Risks
- Kalshi's historical cutoff moves → always query both endpoints and dedupe.
- Thin market history for lows and newer cities → market comparison CIs will be wide; reported,
  not hidden.
- The model may lose to the market everywhere (as highcast does at KLAX/KMIA) → the gate in §3.8
  only requires beating MOS to *paper*-trade; money still needs the sandbox unlock rule.
- Laya may not beat LightGBM [Guessing: likely] → §4.7 makes that an accepted, recorded outcome.
- Candle sparsity → 2 h staleness limit; rows without a fresh candle are excluded, not filled.
