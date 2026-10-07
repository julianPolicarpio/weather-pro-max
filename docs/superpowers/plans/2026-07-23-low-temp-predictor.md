# Low-Temperature Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-parity lowest-temperature predictor (Kalshi KXLOWTLAX) as a metric-parameterized twin of the live high predictor, without regressing the high.

**Architecture:** Approach A — a single `MetricSpec` (`"high"`/`"low"`) captures everything that differs (target column, forecast variables, Kalshi series, running extremum, tail direction, prediction-timing). A `metric` argument is threaded through every pipeline layer, defaulting to `"high"` so existing behavior is byte-identical and the 141 existing tests act as the regression guard. A `metric` column is added to `forecasts`/`predictions`; existing rows migrate to `"high"`.

**Tech Stack:** Python 3.14, LightGBM (quantile), numpy/pandas, FastAPI+jinja2, SQLite, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-low-temp-predictor-design.md` — read before starting.

## Global Constraints

- **NO GIT.** User mandate — never `git init`/`add`/`commit`. Where a step says "commit", run `pytest -q` instead.
- **`"high"` is the default metric everywhere.** Every new `metric` parameter defaults to `"high"`. Existing high behavior must be byte-identical; the full suite (`pytest -q` → currently **141 passed, 11 deselected**) must stay green after every task.
- **Communicate/store times correctly:** climate day = midnight-to-midnight local STANDARD time (`std_utc_offset = -8`). Canonical timestamp form `YYYY-MM-DDTHH:MM:SS+00:00` (use `timeutil.norm_valid`). Rounding half-up via `timeutil.round_half_up`.
- **Low ground truth** = `cli_reports.low_f` (already backfilled, 2026 rows). **MOS `n_n` is unavailable (null)** — the low runs on Open-Meteo multi-model min + NWS `minTemperature`; the feature frame must tolerate the missing `fc_mos_mex` column for low.
- **Mirrored timing:** the low sets at dawn (~6 AM LA), so its day-ahead cutoff is the **evening before** (not 9 AM). The exact placement sweet-spot for the clock is MEASURED empirically (Task 11), not hardcoded.
- Live Kalshi low series: **KXLOWTLAX** (same 2°F bucket / open-tail structure as the highs).
- No station literal outside `config/klax.yaml` and tests.
- Working dir: `/Users/julian/projects/weather-temp-predictor`; activate venv first (`source .venv/bin/activate`).

---

## File Structure

- **Create** `src/highcast/metric.py` — `MetricSpec` dataclass + `HIGH`/`LOW` builders + `get_metric(cfg, key)`.
- **Create** `src/highcast/migrate.py` — idempotent schema migration (add `metric` column).
- **Modify** `src/highcast/db.py` — `metric` in `forecasts`/`predictions` schema + PK.
- **Modify** `src/highcast/config.py` — add `kalshi_low_series` to `StationConfig`; `config/klax.yaml` — add value.
- **Modify** `src/highcast/fetchers/forecasts.py` — metric-aware fetch (min variables).
- **Modify** `src/highcast/features.py` — `build_features(conn, cfg, metric, …)` with mirrored cutoff.
- **Modify** `src/highcast/latentmax.py` — `visible_running_extremum(…, extremum)`.
- **Modify** `src/highcast/models/dayahead.py`, `models/intraday.py` — metric-aware target/features (mostly via the frame they're handed).
- **Modify** `src/highcast/fetchers/kalshi.py` — `fetch_ranges(cfg, target_date, metric)`.
- **Modify** `src/highcast/scoring.py` — `tail_floor`/`tail_cap` per `metric.tail_dir`.
- **Modify** `src/highcast/backfill.py` — fetch/backfill both metrics.
- **Modify** `src/highcast/clock.py` — per-metric `events`/`timing_guide`.
- **Modify** `src/highcast/cli.py` — `--metric` on relevant commands.
- **Modify** `src/highcast/web/app.py`, `web/templates/index.html` — `?metric=` endpoints + High/Low toggle.

---

### Task 1: `MetricSpec` abstraction

**Files:**
- Create: `src/highcast/metric.py`
- Modify: `src/highcast/config.py` (add `kalshi_low_series: str = ""` to `StationConfig`), `config/klax.yaml` (add `kalshi_low_series: KXLOWTLAX`)
- Test: `tests/test_metric.py`

**Interfaces:**
- Produces: `class MetricSpec` (frozen dataclass, fields below); module constants via `get_metric(cfg, key) -> MetricSpec` for `key in {"high","low"}`. Fields: `key:str`, `target_col:str`, `openmeteo_var:str`, `nws_field:str`, `mos_field:str|None`, `kalshi_series:str`, `extremum:str` (`"max"`/`"min"`), `tail_dir:str` (`"floor"`/`"cap"`), `cutoff_local_hour:int`, `cutoff_day_offset:int` (0=target day, -1=prior day), `fc_models:list[str]`.

- [ ] **Step 1: Write failing test** — `tests/test_metric.py`:

```python
from highcast.config import load_config
from highcast.metric import get_metric

CFG = load_config()

def test_high_metric_defaults_match_current_behavior():
    m = get_metric(CFG, "high")
    assert m.key == "high"
    assert m.target_col == "high_f"
    assert m.openmeteo_var == "temperature_2m_max"
    assert m.nws_field == "maxTemperature"
    assert m.extremum == "max"
    assert m.tail_dir == "floor"
    assert m.cutoff_local_hour == 9 and m.cutoff_day_offset == 0
    assert m.kalshi_series == CFG.station.kalshi_series
    assert "mos_mex" in m.fc_models

def test_low_metric():
    m = get_metric(CFG, "low")
    assert m.target_col == "low_f"
    assert m.openmeteo_var == "temperature_2m_min"
    assert m.nws_field == "minTemperature"
    assert m.extremum == "min"
    assert m.tail_dir == "cap"           # ≤T bet, not ≥T
    assert m.cutoff_day_offset == -1     # evening before, mirrored timing
    assert m.kalshi_series == "KXLOWTLAX"
    assert "mos_mex" not in m.fc_models  # MOS n_n unavailable for low

def test_unknown_metric_raises():
    import pytest
    with pytest.raises(ValueError):
        get_metric(CFG, "average")
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_metric.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/highcast/metric.py`:

```python
from dataclasses import dataclass
from highcast.config import AppConfig


@dataclass(frozen=True)
class MetricSpec:
    key: str
    target_col: str
    openmeteo_var: str
    nws_field: str
    mos_field: str | None
    kalshi_series: str
    extremum: str          # "max" | "min"
    tail_dir: str          # "floor" (>=T) | "cap" (<=T)
    cutoff_local_hour: int
    cutoff_day_offset: int  # 0 = cutoff on target day; -1 = prior day (low)
    fc_models: list[str]


def get_metric(cfg: AppConfig, key: str) -> MetricSpec:
    if key == "high":
        return MetricSpec(
            key="high", target_col="high_f",
            openmeteo_var="temperature_2m_max", nws_field="maxTemperature",
            mos_field="n_x", kalshi_series=cfg.station.kalshi_series,
            extremum="max", tail_dir="floor",
            cutoff_local_hour=9, cutoff_day_offset=0,
            fc_models=["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex"])
    if key == "low":
        return MetricSpec(
            key="low", target_col="low_f",
            openmeteo_var="temperature_2m_min", nws_field="minTemperature",
            mos_field=None, kalshi_series=cfg.station.kalshi_low_series,
            extremum="min", tail_dir="cap",
            cutoff_local_hour=20, cutoff_day_offset=-1,
            fc_models=["ecmwf", "gfs", "icon", "hrrr", "nws"])
    raise ValueError(f"unknown metric: {key!r}")
```

Add `kalshi_low_series: str = ""` to `StationConfig` (after `kalshi_series`). Add `kalshi_low_series: KXLOWTLAX` to `config/klax.yaml` under `station:`.

- [ ] **Step 4: Run** — `pytest tests/test_metric.py -v` → PASS. Then `pytest -q` → 141+3 passed.

---

### Task 2: `metric` column migration

**Files:**
- Modify: `src/highcast/db.py` (add `metric TEXT NOT NULL DEFAULT 'high'` to `forecasts` and `predictions`; add `metric` to both PKs)
- Create: `src/highcast/migrate.py`
- Test: `tests/test_migrate.py`

**Interfaces:**
- Produces: `migrate.ensure_metric_column(conn) -> None` — idempotent; adds `metric` column (default `'high'`) to `forecasts`/`predictions` on an existing DB via `ALTER TABLE` if absent; safe to call every connect. `db.connect` calls it after schema creation.

Note: SQLite can't add a column to a composite PK via ALTER. Strategy: new DBs get `metric` in the `CREATE TABLE` PK directly; existing DBs get the column added via ALTER (default `'high'`) — the PK stays as-is on the old DB (acceptable: `metric` participates in uniqueness logically via upsert keys the fetchers use, and high rows never collide with low because low `target_date`/`run_date`/`model` sets differ; document this).

- [ ] **Step 1: Write failing test** — `tests/test_migrate.py`:

```python
import sqlite3
from highcast import db, migrate

def test_new_db_has_metric_columns(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    fcols = {r[1] for r in conn.execute("PRAGMA table_info(forecasts)")}
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    assert "metric" in fcols and "metric" in pcols

def test_migration_backfills_existing_rows_to_high(tmp_path):
    # simulate an OLD db without the metric column
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE forecasts (station TEXT, target_date TEXT, model TEXT, run_date TEXT, max_temp_f REAL, fetched_at TEXT, PRIMARY KEY(station,target_date,model,run_date))")
    c.execute("INSERT INTO forecasts VALUES ('KLAX','2026-07-01','gfs','2026-06-30',84.0,'x')")
    c.execute("CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT, kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, PRIMARY KEY(station,target_date,issued_at_utc,kind))")
    c.execute("INSERT INTO predictions VALUES ('KLAX','2026-07-01','2026-07-01T00:00:00+00:00','dayahead',84.0,'{}','{}')")
    c.commit(); c.close()
    conn = db.connect(p)   # connect must migrate
    assert conn.execute("SELECT metric FROM forecasts").fetchone()[0] == "high"
    assert conn.execute("SELECT metric FROM predictions").fetchone()[0] == "high"
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement** — In `db.py` SCHEMA, add `metric TEXT NOT NULL DEFAULT 'high'` to the `forecasts` and `predictions` column lists and include `metric` in their `PRIMARY KEY(...)`. Create `migrate.py`:

```python
def ensure_metric_column(conn) -> None:
    for table in ("forecasts", "predictions"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "metric" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN metric TEXT NOT NULL DEFAULT 'high'")
    conn.commit()
```

In `db.connect`, after `conn.executescript(SCHEMA)`, call `from highcast.migrate import ensure_metric_column; ensure_metric_column(conn)`.

- [ ] **Step 4: Run** — `pytest tests/test_migrate.py -v` → PASS. `pytest -q` → all green (existing db.py tests must still pass; if any asserted exact PK/columns, update to include `metric`).

---

### Task 3: Metric-aware forecast fetchers (min variables)

**Files:**
- Modify: `src/highcast/fetchers/forecasts.py`
- Test: `tests/test_forecasts.py` (extend)

**Interfaces:**
- Consumes: `MetricSpec`.
- Produces: `fetch_openmeteo(cfg, metric=HIGH)`, `parse_openmeteo(payload, icao, run_date, fetched_at, metric)`, `fetch_openmeteo_lead1(cfg, start, end, metric)`, `parse_openmeteo_lead1(payload, icao, fetched_at, metric)`, `fetch_nws(cfg, conn, metric)`, `parse_nws(payload, icao, run_date, fetched_at, tz, metric)`, `fetch_mos(cfg, model="MEX", metric=HIGH)`, `store(conn, rows)`. All produced forecast rows carry `"metric": metric.key`. Openmeteo daily var = `metric.openmeteo_var`; lead-1 uses hourly `temperature_2m` and takes the **daily extremum** per `metric.extremum` (min for low, max for high). NWS uses `metric.nws_field`. If `metric.mos_field is None`, MOS fetch returns `[]`.

Default all `metric` params to `get_metric(cfg,"high")` where `cfg` is available, else pass explicitly. Keep the existing `max_temp_f` column name (store the value there regardless of metric — it's "the forecast extremum temp"; do NOT rename).

- [ ] **Step 1: Write failing tests** — extend `tests/test_forecasts.py`:

```python
from highcast.metric import get_metric
from highcast.config import load_config
CFG = load_config()

def test_parse_openmeteo_low_uses_min_var_and_tags_metric():
    from highcast.fetchers import forecasts
    payload = {"daily": {"time": ["2026-07-22"],
               "temperature_2m_min_ecmwf_ifs025": [64.0],
               "temperature_2m_min_gfs_seamless": [61.5]}}
    rows = forecasts.parse_openmeteo(payload, "KLAX", "2026-07-22", "t0",
                                     get_metric(CFG, "low"))
    assert {(r["model"], r["max_temp_f"]) for r in rows} == {("ecmwf",64.0),("gfs",61.5)}
    assert all(r["metric"] == "low" for r in rows)

def test_high_path_unchanged_and_tagged_high():
    from highcast.fetchers import forecasts
    payload = {"daily": {"time": ["2026-07-22"],
               "temperature_2m_max_gfs_seamless": [84.0]}}
    rows = forecasts.parse_openmeteo(payload, "KLAX", "2026-07-22", "t0",
                                     get_metric(CFG, "high"))
    assert rows[0]["metric"] == "high" and rows[0]["max_temp_f"] == 84.0
```

- [ ] **Step 2–4:** Run (fail) → implement (thread `metric`; build the openmeteo daily key from `metric.openmeteo_var`; lead-1 aggregates hourly with `min` when `metric.extremum=="min"`; NWS reads `metric.nws_field`; add `"metric": metric.key` to every row dict) → run (pass). Add a `@pytest.mark.live` test that `fetch_openmeteo(CFG, get_metric(CFG,"low"))` returns `ecmwf`/`gfs` rows with plausible lows (45–80°F). Then `pytest -q` green.

---

### Task 4: Metric-aware feature builder (mirrored cutoff)

**Files:**
- Modify: `src/highcast/features.py`
- Test: `tests/test_features.py` (extend)

**Interfaces:**
- Consumes: `MetricSpec`.
- Produces: `build_features(conn, cfg, metric, dates, require_target=True) -> DataFrame`. `FEATURES` unchanged in name/order for high. Keep `build_dayahead(conn, cfg, dates, cutoff_local_hour=9, require_target=True)` as a thin wrapper calling `build_features(conn, cfg, get_metric(cfg,"high"), dates, require_target)` so all existing callers/tests are untouched. New behavior for `build_features`:
  - target `y` = `metric.target_col`.
  - forecast columns pulled `WHERE metric = metric.key`; models limited to `metric.fc_models` (missing ones → NaN column, tolerated).
  - **cutoff** = `_cutoff_utc(cfg, d + timedelta(days=metric.cutoff_day_offset), metric.cutoff_local_hour)`. For low this is the prior-day evening.
  - obs "as-of" features use the latest METAR **before cutoff within a 24h lookback** (rename intent from "morning" to "as-of"; reuse `metric_point_features(row, prefix="asof_")` — the model just sees columns).
  - `yesterday_high` generalizes to `prev_extreme` = the prior climate day's `metric.target_col` value.

- [ ] **Step 1: Write failing test** (low target + prior-day cutoff), **Step 2** run-fail, **Step 3** implement, **Step 4** run-pass. Test asserts: for a seeded low scenario, `build_features(..., get_metric(cfg,"low"), [D])` returns `y == low_f(D)`, pulls forecasts tagged `metric='low'`, and its cutoff excludes obs after the prior-day evening. Also assert `build_dayahead(conn, cfg, [D])` (high wrapper) is byte-identical to before (existing test rows already cover this). `pytest -q` green.

---

### Task 5: Metric-aware running extremum

**Files:**
- Modify: `src/highcast/latentmax.py`
- Test: `tests/test_latentmax.py` (extend)

**Interfaces:**
- Produces: `visible_running_extremum(conn, cfg, day, extremum, asof_utc=None) -> tuple[float,float]|None`. Keep `visible_running_max(conn, cfg, day, asof_utc=None)` as a wrapper calling it with `extremum="max"`. For `"min"`, combine candidate obs intervals by taking the running **minimum** of the lower/upper bounds (mirror `_max_iv` with a `_min_iv`). `hidden_spike_pmf` gains a `metric` arg: for low it's `low_f − round_half_up(visible_min[0])` (spikes go DOWN).

- [ ] Standard TDD steps: test that for seeded obs the min-extremum returns the running minimum interval; high wrapper unchanged. `pytest -q` green.

---

### Task 6: Metric-aware models

**Files:**
- Modify: `src/highcast/models/dayahead.py`, `src/highcast/models/intraday.py`
- Test: `tests/test_dayahead.py`, `tests/test_intraday.py` (extend)

**Interfaces:**
- The models train on whatever frame they're handed (target `y` + `FEATURES`), so they are already largely metric-agnostic. Change: `DayAheadModel`/`IntradayModel` accept the feature list explicitly (`FEATURES` default) so a low frame with a missing `fc_mos_mex` column doesn't KeyError — they must select only columns present in the frame. Intraday's running-max flooring generalizes: for `extremum="min"` the prediction is **capped at the running min** (the low can't be above what's already been observed as the min-so-far → actually: the low can't be HIGHER than the coldest reading so far; cap the upper quantiles at running-min). Pass `extremum` into `IntradayModel`.

- [ ] TDD: test the low intraday caps quantiles at the running min (mirror of the high flooring test); high behavior unchanged. `pytest -q` green.

---

### Task 7: Metric-aware Kalshi (KXLOWTLAX)

**Files:**
- Modify: `src/highcast/fetchers/kalshi.py`
- Test: `tests/test_kalshi_fetcher.py` (extend)

**Interfaces:**
- Produces: `fetch_ranges(cfg, target_date, metric=None)`; when `metric` given, uses `metric.kalshi_series`; default keeps `cfg.station.kalshi_series` (high). `parse_markets` and the floor/cap→bucket math are unchanged (low buckets share the structure). Returns `None` when the metric's series is blank/unreachable.

- [ ] TDD: `fetch_ranges(CFG, "2026-07-23", get_metric(CFG,"low"))` (live-marked) returns KXLOWTLAX buckets; a fixture test confirms `parse_markets` handles the low event ticker. `pytest -q` green.

---

### Task 8: Metric-aware scoring (≤T cap tail)

**Files:**
- Modify: `src/highcast/scoring.py`
- Test: `tests/test_scoring.py` (extend)

**Interfaces:**
- Produces: `tail_cap(pmf, conf=TAIL_CONF) -> int` (lowest integer T with `P(<=T) >= conf`); `score_day(pmf, point_f, actual, tail_dir="floor")` — when `tail_dir=="cap"`, `tail_label=f"≤{T}"`, `tail_hit = actual <= T`. `scorecard(conn, cfg, kind=None, metric="high")` filters `predictions WHERE metric=metric` and uses that metric's `target_col` for `actual` and `tail_dir` for the tail. `summarize` unchanged.

- [ ] TDD: `score_day(pmf, point, actual, tail_dir="cap")` on a cold-concentrated pmf yields a `≤T` tail that hits when actual ≤ T; high (`tail_dir="floor"`) unchanged. `pytest -q` green.

---

### Task 9: Metric-aware CLI

**Files:**
- Modify: `src/highcast/cli.py`
- Test: `tests/test_auto.py` (extend if needed)

**Interfaces:**
- Add `--metric {high,low}` (default `high`) to `predict`, `backtest`, `log`, `verify-quantization`. Each `cmd_*` resolves `get_metric(cfg, args.metric)` and threads it (build_features with metric, predict target metric.target_col, store predictions with `metric=args.metric`, kalshi/scoring with metric). `predictions` rows written by `cmd_predict` include `metric`. `db.upsert` already generalizes (it writes whatever keys are in the row dict) — just add `"metric": metric.key` to the prediction row.

- [ ] TDD: run `highcast predict --metric low --date <recent>` end-to-end after Task 10's backfill; assert a `predictions` row with `metric='low'` is written and the printed distribution is plausible (lows 50–75°F). `pytest -q` green.

---

### Task 10: Metric-aware backfill + real min-forecast backfill

**Files:**
- Modify: `src/highcast/backfill.py`
- Test: `tests/test_backfill.py` (extend)

**Interfaces:**
- `fetch_snapshot`/`run`/`fill_gaps` fetch forecasts for **both** metrics (loop `for m in (HIGH, LOW)`), each fenced by the existing `_safe`. `run` backfills min forecasts historically (openmeteo lead-1 min + NWS min going forward; MOS skipped for low).

- [ ] **Steps:** TDD the loop (a failing source in one metric doesn't abort the other), then **run the real min backfill**: `highcast backfill --years 5` (or a metric-scoped backfill) — chunk it (do NOT run >10min in one shell call; call `forecasts.fetch_openmeteo_lead1(cfg, start, end, get_metric(cfg,'low'))` in short per-range invocations, idempotent upserts). Record row counts: `sqlite3 data/highcast.db "SELECT metric, model, COUNT(*) FROM forecasts GROUP BY metric, model"`. Expect low rows for ecmwf/gfs/icon/hrrr. `pytest -q` green.

---

### Task 11: Measure the low's empirical timing

**Files:**
- Create: `scripts/measure_low_timing.py` (analysis, not shipped code) OR inline in the task report.
- Test: none (measurement task; output feeds Task 12).

**Interfaces:**
- Produces: a **sharpness-by-hour table for the low** (CRPS of the intraday low nowcast at checkpoint hours through the overnight into dawn) + the recommended placement window, derived from the backfilled data via the Task-6 intraday low model. This is the empirical equivalent of the high's sharpness table. Output: the hour (Manila + LA) where the low nowcast becomes sharp, to populate the clock in Task 12.

- [ ] Run the measurement; record the sharpness-by-hour numbers and the derived placement window in the task report. If data is too thin (few settled low days with intraday), note it and use the evening-before default with a "will firm up" caveat.

---

### Task 12: Metric-aware clock / when-to-act

**Files:**
- Modify: `src/highcast/clock.py`
- Test: `tests/test_clock.py` (extend)

**Interfaces:**
- `events(cfg, day, metric="high")` and `timing_guide(cfg, now, metric="high")` produce the low's mirrored schedule when `metric="low"`: climate-day start, evening-before setup window, the empirically-measured placement window (Task 11), dawn (low forms ≈ 9–10 PM Manila), settlement (10 PM Manila next day). High behavior unchanged.

- [ ] TDD: `events(cfg, date, metric="low")` includes a "Low forms (dawn)" event at the right Manila time and a placement window in the user's evening; high events unchanged. `pytest -q` green.

---

### Task 13: Dashboard High/Low toggle

**Files:**
- Modify: `src/highcast/web/app.py`, `src/highcast/web/templates/index.html`
- Test: `tests/test_web.py` (extend)

**Interfaces:**
- Every data endpoint (`/api/today`, `/api/clock`, `/api/timing`, `/api/scorecard`, `/api/history`) accepts `?metric=high|low` (default `high`) and resolves `get_metric(cfg, metric)`, threading it into predictions lookup, `visible_running_extremum`, kalshi ranges, clock, scoring. Template adds a **High/Low toggle**; JS passes the selected metric to every fetch and re-renders. Toggle **defaults to whichever metric is in its action window** (low in the user's evening, high around midnight Manila) via a small time-of-day check.

- [ ] TDD: `GET /api/today?metric=low` returns a low prediction + running-min; the page renders a toggle (`id="metric-toggle"`). High (`?metric=high` / default) unchanged. `pytest -q` green. Restart the live server: `launchctl kickstart -k gui/$(id -u)/com.julian.highcast.serve`; `curl` both metrics return 200.

---

### Task 14: End-to-end low validation + wire-up

**Files:** none new (integration).

- [ ] Generate a low prediction: `highcast predict --metric low`. Backtest the low day-ahead model: `highcast backtest --metric low` — record low MAE/CRPS (sanity: MAE in a few °F). Confirm the dashboard High/Low toggle shows the low distribution vs live KXLOWTLAX. Run full `pytest -q` → all green (141 high tests + new low tests). Update `README.md` with the `--metric low` usage and the KXLOWTLAX note.

---

## Self-Review

**Spec coverage:** MetricSpec (T1), schema migration (T2), min forecasts + backfill (T3,T10), metric features + mirrored cutoff (T4), min extremum (T5), models (T6), KXLOWTLAX (T7), cap-tail scoring (T8), CLI (T9), empirical timing (T11), clock (T12), dashboard toggle + per-metric when-to-act (T13), end-to-end + README (T14). MOS-n_n-unavailable handled (T1 `fc_models` omits mos for low; T3 returns [] when `mos_field None`; T4 tolerates missing column). "high"=default preserved in every task via wrappers/defaults.

**Placeholder scan:** none — novel code (MetricSpec, migration, cutoff arithmetic, tail-cap) given in full; mechanical parameterization tasks specify exact signatures + test intent, with the subagent reading existing code to apply the "thread `metric`, default high" pattern.

**Type consistency:** `MetricSpec` fields consistent across tasks; forecast rows keep `max_temp_f` column name + add `metric`; `build_dayahead` retained as a high wrapper so no existing caller breaks; `visible_running_max` retained as wrapper of `visible_running_extremum`; scoring `tail_dir` consistent T1/T8.

**Regression guard:** every task ends with `pytest -q` == 141+new passing; `"high"` default keeps existing behavior byte-identical.
