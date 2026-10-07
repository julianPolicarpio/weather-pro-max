# Low-Temperature Predictor — Design Handoff (IN PROGRESS)

**Date:** 2026-07-22 (design) / build started 2026-07-23
**Status:** BUILD IN PROGRESS, PAUSED by user at Task 2. Approach A locked (user
said "build the lowest temp feature"). Full implementation plan written:
`docs/superpowers/plans/2026-07-23-low-temp-predictor.md` (14 tasks). Resume via
superpowers:subagent-driven-development at Task 2.

## BUILD PROGRESS (updated 2026-07-23)

- **Feasibility CONFIRMED:** Open-Meteo multi-model min (`temperature_2m_min`) +
  NWS `minTemperature` fetch fine; `cli_reports.low_f` already backfilled (2026
  rows, 39–77°F). **MOS `n_n` is NULL/unavailable** → low runs on ecmwf/gfs/icon/
  hrrr/nws min only; feature frame tolerates the missing `fc_mos_mex` column.
- **Task 1 COMPLETE** (reviewed clean): `src/highcast/metric.py` — `MetricSpec`
  dataclass + `get_metric(cfg, key)` for HIGH/LOW; added `kalshi_low_series` to
  `StationConfig` + `config/klax.yaml` (= KXLOWTLAX). Nothing consumes it yet.
- **Task 2 (schema migration) NOT STARTED** — brief ready in scratchpad
  (`low-task-2-brief.md`); adds a `metric` column (default 'high') to
  `forecasts`+`predictions`, migrating the existing live DB. THIS IS THE RESUME
  POINT.
- Tasks 3–14 pending: min fetchers, metric features (mirrored cutoff), min
  extremum, models, KXLOWTLAX kalshi, cap-tail scoring, CLI --metric, backfill,
  empirical low timing, clock, dashboard High/Low toggle, end-to-end.
- **Ledger:** scratchpad `low-progress.md`. Task-1 minor findings logged there
  for the final whole-tree review.
- **High predictor VERIFIED INTACT** after Task 1 (2026-07-23): full suite 146
  passed/11 deselected, `highcast predict` works, dashboard HTTP 200, launchd
  wake+serve active, DB clean. "high" = default metric everywhere, so Task 1's
  additive changes did not touch the live high path.

## Goal

Add a full-parity **lowest-temperature-in-LA** predictor as a twin of the existing
high-temp predictor (`highcast`), targeting the Kalshi **KXLOWTLAX** market
("Lowest temperature in LA today?").

## Confirmed facts

- **Market exists:** series ticker **KXLOWTLAX**. Verified live via the Kalshi API.
- **Same structure as the highs:** 2°F buckets (e.g. `62–63`, `64–65`, `66–67`,
  `68–69`) with open tails (`≤61`, `≥70`). So the existing `kalshi.py` floor/cap →
  bucket parsing works as-is — only the series ticker differs.
- **Settlement:** the NWS Daily Climate Report **minimum** for LA, 10 AM ET
  expiration — same source/timing machinery as the high.
- **Ground truth already stored:** `cli_reports.low_f` has ~5.5 years of daily lows
  already in the DB (we've been fetching it all along). No target backfill needed.
- **Liquidity caveat:** the low market looks thinner than the highs — expect the
  live model-vs-market comparison to sometimes have `None` prices.

## Scope (chosen)

**Full parity** with the high predictor: min forecasts + day-ahead + intraday +
calibration + scorecard + live KXLOWTLAX comparison + dashboard.

## Architecture (LEADING candidate — NOT yet locked)

**Approach A: generalize with a `metric` parameter.** (User has engaged on A and is
designing on that basis, but the formal A/B/C choice was not explicitly confirmed —
see Open Items.)

Introduce one small abstraction, `MetricSpec`, capturing everything that differs
between high and low:

```python
@dataclass(frozen=True)
class MetricSpec:
    key: str            # "high" | "low"
    target_col: str     # "high_f" | "low_f"   (both already in cli_reports)
    openmeteo_var: str  # "temperature_2m_max" | "temperature_2m_min"
    nws_field: str      # "maxTemperature" | "minTemperature"
    mos_field: str      # "n_x" (day max) | "n_n" (night min)
    kalshi_series: str  # "KXHIGHLAX" | "KXLOWTLAX"
    extremum: str       # "max" | "min"
    tail_dir: str       # "floor" (>=T) for high | "cap" (<=T) for low
    # plus metric-specific TIMING logic (see mirrored-timing below)
```

Thread `metric` through the pipeline (defaulting to `"high"` everywhere so existing
behavior is byte-identical):

| Layer | Change |
|---|---|
| forecasts fetcher | request `metric.openmeteo_var` / `nws_field` / `mos_field`; tag rows with `metric.key`. The MOS `n_n` (night min) we currently discard becomes the low's input. |
| db schema | `forecasts` and `predictions` gain a `metric` column (migration defaults existing rows to `"high"`). `obs` + `cli_reports` unchanged. |
| features | `build_features(conn, cfg, metric, …)` uses `metric.target_col` as `y`, pulls `fc_*` where `metric = metric.key`. |
| latentmax | `visible_running_max` generalizes to running extremum (max **or** min). |
| models | identical LightGBM machinery, trained on the metric's frame. |
| scoring | same buckets; tail metric flips to `metric.tail_dir` — a **≤T cap** ("cold tail") for the low vs the high's ≥T floor. |
| kalshi | `fetch_ranges(cfg, metric, date)` uses `metric.kalshi_series`; bucket math identical. |
| cli / dashboard | `--metric low` flag; High/Low toggle passing `?metric=low` to the APIs. |

**Regression safety:** `"high"` is the default metric, so the live high predictor is
untouched; the 141 existing tests all run against the high and catch any regression.
New low tests added on top.

## The mirrored-timing subtlety (the one genuinely tricky part)

The low is NOT just "same pipeline, different target" — its temporal structure is
mirrored:

- The **high** sets in the afternoon → day-ahead cutoff = 9 AM LA (morning obs in,
  day not yet formed).
- The **low** sets at **dawn (~6 AM LA)** → a 9 AM cutoff would land *after* the low
  already happened. The low's day-ahead prediction must be issued the **evening
  before** (before the dawn it predicts); the intraday nowcast tracks the running
  **minimum** through the overnight into dawn.
- Driver physics differ: overnight **radiational cooling** — clear + calm nights →
  colder low; cloudy/windy → warmer. So evening/overnight cloud + wind + dewpoint
  are the key features (vs morning marine-layer burn-off for the high).

The `MetricSpec` must encode metric-specific timing (not a single hardcoded 9 AM).

## Dashboard (DECIDED)

- **Option 1: High/Low toggle** at the top. One recommendation header, one
  running-extremum display, one bucket board — all switch when the toggle flips.
- **Timing-aware default:** the toggle defaults to whichever metric is currently in
  its action window (Low in the user's evening, High around midnight Manila).
- **Per-metric "When to act" panel:** the low gets its OWN placement-window /
  phase / model-pick / sharpness-by-time panel, exactly like the high — switching
  with the toggle.

### Manila-time windows (user is in Asia/Manila, UTC+8)

| | Resolves (Manila) | Action window (Manila) |
|---|---|---|
| High | afternoon LA → user's early morning | ~midnight (9 AM-LA refresh) |
| Low | dawn LA ≈ user's **evening (~9–10 PM)** | user's **afternoon–evening**, before the dawn low forms |

Result: two natural daily check-ins — Low in the evening, High around midnight.

**Important:** the low's *exact* sweet-spot hour must be **measured empirically**
from historical data (the same sharpness-by-time-of-day analysis that produced the
high's window), NOT hardcoded — the low sets up differently and its optimal
placement time is a backtest question.

## Min forecast sources (to wire up)

- Open-Meteo: `temperature_2m_min`
- NWS gridpoint: `minTemperature`
- MOS: `n_n` (night min — currently discarded by the high's afternoon filter)
- **Backfill needed:** historical min forecasts (Open-Meteo min archive, MOS n_n
  archive) — the min equivalents of what the high's 5.5-year backfill pulled.

## Open items (resolve before/at build time)

1. **Lock the architecture** — A (generalize) vs B (parallel low modules) vs C
   (hybrid). Leaning A; user is designing on A but has not formally confirmed.
2. **Derive the low's cutoff + sharpness empirically** from the data, don't hardcode.
3. **Schema migration** — add `metric` column to `forecasts`/`predictions`,
   default existing rows to `"high"`.
4. **Backfill min forecasts** historically.

## Resolved (was open, now settled)

- ✅ KXLOWTLAX bucket structure confirmed live: whole-degree **2°F** buckets with
  open tails (`≤61`, `≥70`) — identical to the highs, so `kalshi.py` handles it
  as-is. (Rules PDF mentions 1-decimal precision, but the listed market is
  whole-degree.)

---

# Model Improvement Backlog

Applies to the CURRENT (high) model now, and to the low once built. Two cost
axes: **💵 money** (does it need a paid subscription?) and **🔧 effort** (dev
time). Only the Synoptic feed costs money; everything else is free to run.
**Validate every model change with a before/after backtest and ship only if it
helps** (same discipline used to reject the skyc change).

| Improvement | 💵 Money | 🔧 Effort | Expected value |
|---|---|---|---|
| Synoptic 5-min feed (intraday between-METAR data) | **PAID** | tiny (token only) | **highest** — closes the 78-vs-79 spike gap |
| Santa Ana / offshore-flow feature | Free | medium | real — sharpens the hot tail |
| Marine-layer burn-off feature (cloud/dewpoint trend) | Free | medium | real — decides model-vs-market splits |
| **ECMWF bias-correction** | Free | small | recovers a wasted skillful model (see below) |
| Evening-before masking fix (day-ahead-for-tomorrow) | Free | small–med | makes the pre-9am read usable |
| Hyperparameter tuning / better calibration | Free | small | marginal |
| **GEFS 31-member ensemble features** | Free | large (medium via Open-Meteo) | uncertainty/tail calibration (see below) |
| Forecaster-text (AFD) NLP | Free | large | v2 |

> **Status of the two model-input experiments below (ECMWF bias-correction and
> GEFS ensemble): FUTURE — do NOT build now.** Both are backtest-first: prototype,
> run a before/after single-split backtest, ship only if MAE/CRPS improve. The low
> predictor and any confirmed-help changes come first; these are deferred until the
> user says go.

## ECMWF bias-correction (detailed — the cheap high-value one)

**Problem:** ECMWF has a systematic **summer WARM bias at KLAX (~+9 to +14°F;
MAE ~6.9°F vs ~2.5°F for the other models)**, likely a coastal-vs-inland
grid-cell mismatch (reads a hotter inland point, misses the marine layer).
Because it's so biased, the inverse-MSE weighting **down-weights it to near-zero
— we discard a skillful model.**

**Fix:** learn ECMWF's seasonal mean error (e.g. per-month bias) and **subtract
it before the forecast enters features**. The corrected value regains a low
error, and — more importantly — ECMWF's informative **day-to-day pattern**
becomes usable instead of ignored.

**How to build/validate:** compute the per-month (or rolling) mean(fc_ecmwf −
actual) from history, store a de-biased `fc_ecmwf_corr`, add it as a feature (or
replace fc_ecmwf), then **single-split backtest before/after** (the same harness
used for skyc). Keep only if MAE/CRPS improve. Beware bias instability — if the
correction doesn't generalize, don't ship.

## BACKTEST RESULTS (2026-07-23) — all candidates rigorously tested, NONE shipped

Ran a proper harness (19 recency-weighted LGBM quantile models + PIT calibration,
same as production) with multi-window robustness checks. Verdicts:

| Candidate | Result | Ship? |
|---|---|---|
| **GEFS 31-member ensemble** | Open-Meteo archives return member keys but NULL data (previous-runs AND historical-forecast). No leakage-free history to train on. | **Can't test — deferred.** Correct live model = `ncep_gefs025` (31 members), but would need a "collect live members now, validate in months" path. |
| **ECMWF bias-correction** (per-month de-bias) | Single-split: MAE −0.0036 but CRPS **+0.0122** (worse). Model already down-weights ECMWF; a corrected copy adds noise. | **Reject.** |
| **Santa Ana dryness** (dewpoint depression) | Alone: MAE −0.017/CRPS −0.002 (marginal). Stacked with burn-off: CRPS worsens. Redundant with existing morn_dwpf/onshore. | **Reject.** |
| **Marine-layer burn-off** (sky/dewpoint morning trend) | Big win on the most-recent window (MAE −0.040, CRPS −0.010) but robustness check: won BOTH metrics on only **1/3** windows (CRPS flips + on the other two). MAE-favorable but CRPS-neutral — within noise. | **Reject** (borderline; MAE-favorable — could revisit with more data). |
| **Evening-before masking** (train with morn_*/yesterday_high randomly NaN) | EVENING scenario fixed dramatically (MAE 4.4→1.65, **3/3 windows**) BUT the primary 9am/FULL scenario DEGRADED **3/3** (~+0.03–0.09 MAE). A single masked model is a net loss. | **Reject as single model.** The clean fix = a SEPARATE evening-only model (primary untouched) — architecture change, and reverses the user's prior "leave the evening-before, rely on 9am" decision. NEEDS USER OK before building. |

**Takeaway:** the model is already well-tuned; hand-crafted features add only noise.
Backtest-first correctly prevented shipping degradations. Experiment scripts saved
in the session scratchpad (`feat_experiment.py`, `robust_burnoff.py`,
`mask_experiment.py`).

**The one available clean win (pending user OK):** a dedicated evening-before /
forecast-only model would fix the catastrophic ~4.4°F-MAE evening prediction down
to ~1.65°F with ZERO degradation to the validated 9am path — because it's a second
model, not a change to the primary. Deferred pending the user's decision (they had
previously chosen to leave the evening-before and rely on the 9am refresh).

## GEFS 31-member ensemble features (detailed — FUTURE, backtest-first)

**What it is:** GEFS runs the GFS model **31 times** with slightly perturbed
starting conditions ("members"). The spread across members is a direct,
physically-grounded uncertainty signal: all 31 agree → confident day; members
scatter → uncertain day. (Context: a Reddit trader described member-counting →
probability → edge-vs-price as a bot strategy — which is conceptually what
highcast already does, with more inputs and real calibration.)

**Proposed features:** ensemble mean, ensemble spread (std), and per-bucket
member fractions (share of members ≥ / ≤ key thresholds) for the KLAX point.
These could sharpen TAIL probabilities and day-specific uncertainty beyond the
current cross-model `fc_spread` proxy.

**Route:** prefer **Open-Meteo's Ensemble API** (already-used provider; serves
GEFS members per lat/lon) over raw GRIB downloads — turns a "large" effort into
"medium". Historical member data availability for training needs checking.

**Caveats (why naive member-counting is NOT the plan):**
- Raw ensembles are famously **under-dispersive/overconfident** — member
  fraction ≠ true probability; must pass through the existing calibration.
- GEFS members share the same **coastal-vs-inland grid-cell problem** found in
  ECMWF at KLAX — the members' absolute levels need bias handling; their
  *pattern/spread* is the value.
- Feed as **features into the existing LightGBM + calibration pipeline**, not as
  a replacement probability engine.

**Validate:** same discipline — before/after backtest (MAE/CRPS, esp. tail-hit
and calibration coverage); ship only on a clear win.

## Constraints carried from the main project

- NO GIT (user mandate) — do not init a repo or commit.
- Communicate all times in **Manila time** (LA/ET secondary).
- Keep the live high predictor working; tests are the regression guard.
