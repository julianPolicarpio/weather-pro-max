# Ensemble Capture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start capturing Open-Meteo ensemble-member summary stats for KLAX into a new `forecast_ens` table — every wake plus a one-time 93-day backfill — so ensemble history accumulates for a future model change. This plan changes **no predictions**.

**Architecture:** Mirror the existing `fetch_openmeteo`/`parse_openmeteo`/`store` pattern for the ensemble endpoint. The parse step turns ~119 hourly member series into one per-day summary row (std, quantiles, `frac_ge_82`, a bimodality gap stat) via each member's daily max over the NWS climate day. Storage is a new sibling table; capture is wired into `backfill.fetch_snapshot` (runs each wake) plus a one-time `backfill_ensemble`.

**Tech Stack:** Python 3, SQLite, `highcast.http.get_json`, NumPy; pytest.

## Global Constraints

- **CAPTURE ONLY — zero effect on live predictions.** Do NOT touch `features.FEATURES`, `models/`, `predictions`, `signals`, betting/staking, or the dashboard. If a task would change any of those, it is out of scope for this plan.
- **Not a git repo.** No commits. Each task ends by running tests: `.venv/bin/python -m pytest -q`. Reviewers read files directly.
- **Ensemble endpoint:** `https://ensemble-api.open-meteo.com/v1/ensemble`, `models=gfs025,ecmwf_ifs025,icon_seamless`, `hourly=temperature_2m`, `temperature_unit=fahrenheit`, `timezone=GMT`. **`past_days` ≤ 93** (hard API cap, confirmed by spike).
- **Members** = response `hourly` columns matching `^temperature_2m_member\d+_` (the bare per-model control column, e.g. `temperature_2m_ncep_gefs025`, is EXCLUDED).
- **Daily max per member** = max of that member's hourly temps within `climate_day_bounds(target_date, cfg.station.std_utc_offset)` (UTC bounds; hourly times parsed as UTC).
- **Summary floor:** if usable members < 10, write NaN (`None`) for all `ens_*` stats (never a noisy stat).
- **`frac_ge_82` threshold = 82.0°F** (the seam's upper edge).
- **Leakage discipline:** `run_date` = local today (`cfg.station.tz`), `fetched_at` = `utcnow_iso()` (issue time), mirroring `fetch_openmeteo`.
- Run tests with `.venv/bin/python -m pytest ...`.

---

### Task 1: `forecast_ens` table + `store_ens`

**Files:**
- Modify: `src/highcast/db.py` (add table to `SCHEMA`)
- Modify: `src/highcast/fetchers/forecasts.py` (add `store_ens`)
- Test: `tests/test_forecast_ens_store.py` (create)

**Interfaces:**
- Consumes: `db.connect`, `db.upsert`.
- Produces: table `forecast_ens (station, target_date, run_date, metric, n_members, ens_mean, ens_std, ens_p10, ens_p50, ens_p90, ens_frac_ge_82, ens_bimodality, fetched_at)` PK `(station, target_date, run_date, metric)`; `forecasts.store_ens(conn, rows: list[dict]) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forecast_ens_store.py`:

```python
from highcast import db as hdb
from highcast.fetchers import forecasts


def test_forecast_ens_table_exists_and_store_roundtrips(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    # table exists after connect (CREATE TABLE IF NOT EXISTS in SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(forecast_ens)")}
    assert {"station", "target_date", "run_date", "metric", "n_members",
            "ens_std", "ens_p10", "ens_p90", "ens_frac_ge_82",
            "ens_bimodality", "fetched_at"} <= cols
    n = forecasts.store_ens(conn, [{
        "station": "KLAX", "target_date": "2026-07-24", "run_date": "2026-07-24",
        "metric": "high", "n_members": 119, "ens_mean": 80.1, "ens_std": 2.3,
        "ens_p10": 77.0, "ens_p50": 80.0, "ens_p90": 84.0,
        "ens_frac_ge_82": 0.4, "ens_bimodality": 1.1,
        "fetched_at": "2026-07-24T16:00:00+00:00"}])
    assert n == 1
    row = conn.execute("SELECT ens_std, ens_frac_ge_82 FROM forecast_ens").fetchone()
    assert row["ens_std"] == 2.3 and row["ens_frac_ge_82"] == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forecast_ens_store.py -q`
Expected: FAIL — `no such table: forecast_ens` (and `AttributeError: store_ens`).

- [ ] **Step 3: Write minimal implementation**

In `src/highcast/db.py`, add this block to the `SCHEMA` string, immediately after the `forecasts` table's `CREATE TABLE ... PRIMARY KEY (...forecasts...);` (before the `alerts` table):

```sql
CREATE TABLE IF NOT EXISTS forecast_ens (
  station TEXT NOT NULL, target_date TEXT NOT NULL, run_date TEXT NOT NULL,
  metric TEXT NOT NULL DEFAULT 'high', n_members INTEGER,
  ens_mean REAL, ens_std REAL, ens_p10 REAL, ens_p50 REAL, ens_p90 REAL,
  ens_frac_ge_82 REAL, ens_bimodality REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, target_date, run_date, metric));
```

In `src/highcast/fetchers/forecasts.py`, add next to `store` (at the end of the file):

```python
def store_ens(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_ens", rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forecast_ens_store.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Suite checkpoint (no commit — not a git repo)**

Run: `.venv/bin/python -m pytest -q`
Expected: baseline (215 passed, 12 deselected) + 1 new, no regressions.

---

### Task 2: `parse_openmeteo_ensemble` + summary stats (the core)

**Files:**
- Modify: `src/highcast/fetchers/forecasts.py` (imports, constants, `_ens_summary`, `parse_openmeteo_ensemble`)
- Test: `tests/test_parse_ensemble.py` (create)

**Interfaces:**
- Consumes: `timeutil.climate_day_bounds`.
- Produces:
  - `parse_openmeteo_ensemble(payload: dict, icao: str, run_date: str, fetched_at: str, std_utc_offset: int, metric_key: str = "high") -> list[dict]` — one summary row per target_date present in the payload's hourly window.
  - Row keys exactly match the `forecast_ens` columns (Task 1).

- [ ] **Step 1: Write the failing test**

Create `tests/test_parse_ensemble.py`:

```python
import math

from highcast.fetchers.forecasts import parse_openmeteo_ensemble

# KLAX std_utc_offset is 8 (PST). climate_day_bounds(2026-07-24, 8) =
# 2026-07-24T08:00Z .. 2026-07-25T08:00Z. Build 24 hourly UTC stamps inside it.
def _hours():
    return [f"2026-07-24T{h:02d}:00" for h in range(8, 24)] + \
           [f"2026-07-25T{h:02d}:00" for h in range(0, 8)]


def _payload(member_series: dict):
    return {"hourly": {"time": _hours(),
                       # bare control column present but must be IGNORED:
                       "temperature_2m_ncep_gefs025": [70.0] * 24,
                       **member_series}}


def test_bimodal_split_summary():
    # 20 members: 10 peak at ~78 (layer holds), 10 peak at ~84 (clears).
    series = {}
    for i in range(1, 11):
        series[f"temperature_2m_member{i:02d}_ncep_gefs025"] = [78.0] * 24
    for i in range(11, 21):
        series[f"temperature_2m_member{i:02d}_ncep_gefs025"] = [84.0] * 24
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", 8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["n_members"] == 20
    assert row["ens_p10"] == 78.0 and row["ens_p90"] == 84.0
    assert math.isclose(row["ens_frac_ge_82"], 0.5)
    # a 6°F empty gap between two clumps, std 3 => bimodality ~2, clearly high
    assert row["ens_bimodality"] > 1.0
    assert set(row) >= {"station", "run_date", "metric", "fetched_at", "ens_mean"}


def test_unimodal_low_bimodality():
    series = {f"temperature_2m_member{i:02d}_ncep_gefs025":
              [77.0 + (i % 3)] * 24 for i in range(1, 21)}
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", 8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["ens_bimodality"] < 1.0          # tight cluster, no gap
    assert row["ens_frac_ge_82"] == 0.0


def test_below_floor_writes_nan():
    series = {f"temperature_2m_member{i:02d}_ncep_gefs025": [80.0] * 24
              for i in range(1, 6)}            # only 5 members < floor 10
    rows = parse_openmeteo_ensemble(_payload(series), "KLAX",
                                    "2026-07-24", "2026-07-24T16:00:00+00:00", 8)
    row = next(r for r in rows if r["target_date"] == "2026-07-24")
    assert row["n_members"] == 5
    assert row["ens_std"] is None and row["ens_frac_ge_82"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_parse_ensemble.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_openmeteo_ensemble'`.

- [ ] **Step 3: Write minimal implementation**

In `src/highcast/fetchers/forecasts.py`, add to the imports at the top:

```python
import re

import numpy as np

from highcast.timeutil import climate_day_bounds
```

(the module already imports `datetime, timezone` from datetime and `utcnow_iso`; keep those.)

Add these constants near `MODELS` (after `_BY_SUFFIX`):

```python
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODELS = "gfs025,ecmwf_ifs025,icon_seamless"
_ENS_MEMBER_RE = re.compile(r"^temperature_2m_member\d+_")
_ENS_FLOOR = 10          # min usable members for a summary; else NaN
_ENS_FRAC_THRESH = 82.0  # seam upper edge — fraction of members at/above this
_ENS_STAT_KEYS = ["ens_mean", "ens_std", "ens_p10", "ens_p50", "ens_p90",
                  "ens_frac_ge_82", "ens_bimodality"]
```

Add these two functions (near `parse_openmeteo`):

```python
def _ens_summary(member_maxes: list[float]) -> dict:
    """Summary stats of one day's ~119 member daily-maxes. Below the member
    floor, every stat is None (LightGBM will later treat these as missing)."""
    arr = np.array([m for m in member_maxes if m is not None], dtype=float)
    if arr.size < _ENS_FLOOR:
        return {"n_members": int(arr.size), **{k: None for k in _ENS_STAT_KEYS}}
    std = float(arr.std())
    gaps = np.diff(np.sort(arr))
    return {
        "n_members": int(arr.size),
        "ens_mean": float(arr.mean()),
        "ens_std": std,
        "ens_p10": float(np.percentile(arr, 10)),
        "ens_p50": float(np.percentile(arr, 50)),
        "ens_p90": float(np.percentile(arr, 90)),
        "ens_frac_ge_82": float((arr >= _ENS_FRAC_THRESH).mean()),
        # normalized largest gap between adjacent sorted members: big when the
        # members split into two clumps (the marine-layer seam), ~0 when tight.
        "ens_bimodality": float(gaps.max() / (std + 1e-6)) if gaps.size else 0.0,
    }


def parse_openmeteo_ensemble(payload: dict, icao: str, run_date: str,
                             fetched_at: str, std_utc_offset: int,
                             metric_key: str = "high") -> list[dict]:
    """One summary row per target_date in the payload. Each member's daily max
    is taken over the NWS climate day (`climate_day_bounds`)."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return []
    ts = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    member_cols = [k for k in hourly if _ENS_MEMBER_RE.match(k)]
    rows = []
    for d in sorted({t.date() for t in ts}):
        start, end = climate_day_bounds(d, std_utc_offset)
        idx = [i for i, t in enumerate(ts) if start <= t < end]
        if not idx:
            continue
        maxes = []
        for col in member_cols:
            vals = [hourly[col][i] for i in idx if hourly[col][i] is not None]
            if vals:
                maxes.append(max(vals))
        rows.append({"station": icao, "target_date": d.isoformat(),
                     "run_date": run_date, "metric": metric_key,
                     "fetched_at": fetched_at, **_ens_summary(maxes)})
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_parse_ensemble.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: all green, no regressions.

---

### Task 3: `fetch_openmeteo_ensemble` (network fetch)

**Files:**
- Modify: `src/highcast/fetchers/forecasts.py` (add `fetch_openmeteo_ensemble`)
- Test: `tests/test_fetch_ensemble.py` (create)

**Interfaces:**
- Consumes: `get_json` (already imported in the module), `parse_openmeteo_ensemble` (Task 2), `store_ens` (Task 1).
- Produces: `fetch_openmeteo_ensemble(cfg: AppConfig, past_days: int = 2, forecast_days: int = 2) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetch_ensemble.py`:

```python
from highcast.config import load_config
from highcast.fetchers import forecasts


def test_fetch_openmeteo_ensemble_parses_mocked_payload(monkeypatch):
    cfg = load_config()
    captured = {}

    def fake_get_json(url, params=None):
        captured["url"] = url
        captured["params"] = params
        hrs = [f"2026-07-24T{h:02d}:00" for h in range(8, 24)] + \
              [f"2026-07-25T{h:02d}:00" for h in range(0, 8)]
        series = {f"temperature_2m_member{i:02d}_ncep_gefs025": [79.0 + (i % 4)] * 24
                  for i in range(1, 21)}
        return {"hourly": {"time": hrs, **series}}

    monkeypatch.setattr(forecasts, "get_json", fake_get_json)
    rows = forecasts.fetch_openmeteo_ensemble(cfg, past_days=1, forecast_days=1)
    assert captured["url"] == forecasts.ENSEMBLE_URL
    assert captured["params"]["models"] == forecasts.ENSEMBLE_MODELS
    assert captured["params"]["past_days"] == 1
    assert captured["params"]["temperature_unit"] == "fahrenheit"
    assert captured["params"]["timezone"] == "GMT"
    assert any(r["target_date"] == "2026-07-24" and r["n_members"] == 20
               for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fetch_ensemble.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'fetch_openmeteo_ensemble'`.

- [ ] **Step 3: Write minimal implementation**

In `src/highcast/fetchers/forecasts.py`, add (near `fetch_openmeteo`):

```python
def fetch_openmeteo_ensemble(cfg: AppConfig, past_days: int = 2,
                             forecast_days: int = 2) -> list[dict]:
    """Fetch KLAX ensemble members and summarize per day. `past_days` <= 93
    (API cap). Capture-only: callers store into forecast_ens via store_ens."""
    st = cfg.station
    payload = get_json(ENSEMBLE_URL, {
        "latitude": st.lat, "longitude": st.lon,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
        "timezone": "GMT", "past_days": past_days, "forecast_days": forecast_days,
        "models": ENSEMBLE_MODELS,
    })
    run_date = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    return parse_openmeteo_ensemble(payload, st.icao, run_date, utcnow_iso(),
                                    st.std_utc_offset, "high")
```

(`datetime`, `ZoneInfo`, `utcnow_iso` are already imported at the top of the module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fetch_ensemble.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Suite checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

---

### Task 4: Wire capture into the wake cycle + one-time backfill

**Files:**
- Modify: `src/highcast/backfill.py` (add ensemble to `fetch_snapshot`; add `backfill_ensemble`)
- Modify: `src/highcast/cli.py` (`cmd_backfill` calls `backfill_ensemble` once)
- Test: `tests/test_backfill_ensemble.py` (create)

**Interfaces:**
- Consumes: `forecasts.fetch_openmeteo_ensemble`, `forecasts.store_ens` (Tasks 1–3); `_safe`.
- Produces: `backfill.backfill_ensemble(conn, cfg, progress=print) -> dict`; a new `"ensemble"` key in `fetch_snapshot` counts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill_ensemble.py`:

```python
from highcast import backfill, db as hdb
from highcast.config import load_config
from highcast.fetchers import forecasts


def _rows():
    return [{"station": "KLAX", "target_date": "2026-07-24",
             "run_date": "2026-07-24", "metric": "high", "n_members": 119,
             "ens_mean": 80.1, "ens_std": 2.3, "ens_p10": 77.0, "ens_p50": 80.0,
             "ens_p90": 84.0, "ens_frac_ge_82": 0.4, "ens_bimodality": 1.1,
             "fetched_at": "2026-07-24T16:00:00+00:00"}]


def test_backfill_ensemble_stores(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    monkeypatch.setattr(forecasts, "fetch_openmeteo_ensemble",
                        lambda cfg, past_days=93, forecast_days=1: _rows())
    out = backfill.backfill_ensemble(conn, cfg)
    assert out["ensemble_backfill"] == 1
    assert conn.execute("SELECT COUNT(*) FROM forecast_ens").fetchone()[0] == 1


def test_fetch_snapshot_includes_ensemble(tmp_path, monkeypatch):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    # neutralize the network fetchers this test doesn't exercise
    for name in ("fetch_openmeteo", "fetch_nws", "fetch_mos"):
        monkeypatch.setattr(forecasts, name, lambda *a, **k: [])
    monkeypatch.setattr(forecasts, "fetch_openmeteo_ensemble",
                        lambda cfg, past_days=2, forecast_days=2: _rows())
    from highcast.fetchers import metar, fivemin, alerts, cli_reports
    monkeypatch.setattr(metar, "fetch_recent", lambda cfg: [])
    monkeypatch.setattr(fivemin, "fetch_recent", lambda cfg: [])
    monkeypatch.setattr(alerts, "fetch_active", lambda cfg: [])
    monkeypatch.setattr(cli_reports, "fetch_year", lambda cfg, y: [])
    counts = backfill.fetch_snapshot(conn, cfg, progress=lambda *a: None)
    assert counts.get("ensemble") == 1
    assert conn.execute("SELECT COUNT(*) FROM forecast_ens").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backfill_ensemble.py -q`
Expected: FAIL — `AttributeError: module 'highcast.backfill' has no attribute 'backfill_ensemble'` (and `counts.get("ensemble")` is None).

- [ ] **Step 3: Write minimal implementation**

In `src/highcast/backfill.py`, inside `fetch_snapshot`, add this `_safe` block immediately after the `mos` block (line ~104, before `alerts`):

```python
    _safe("ensemble",
          lambda: forecasts.store_ens(conn, forecasts.fetch_openmeteo_ensemble(cfg)),
          counts, errors, progress)
```

Then add this function at module scope in `src/highcast/backfill.py` (after `fetch_snapshot`):

```python
def backfill_ensemble(conn, cfg: AppConfig, progress=print) -> dict:
    """One-time historical grab of all available ensemble history (API cap is
    93 past days). Safe to re-run; INSERT OR REPLACE dedupes by PK."""
    rows = forecasts.fetch_openmeteo_ensemble(cfg, past_days=93, forecast_days=1)
    n = forecasts.store_ens(conn, rows)
    progress(f"ensemble backfill: {n} day-rows")
    return {"ensemble_backfill": n}
```

In `src/highcast/cli.py`, in `cmd_backfill` (around line 13–23), add a line so the one-time grab runs when the user runs `highcast backfill`. After the existing `print(backfill.fetch_snapshot(conn, cfg))` line, add:

```python
    print(backfill.backfill_ensemble(conn, cfg))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_backfill_ensemble.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Suite checkpoint + live smoke (real fetch, one run)**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green.

Then a one-time real backfill against the live endpoint (network), to seed history now:

Run: `.venv/bin/python -c "from highcast import db, backfill; from highcast.config import load_config; cfg=load_config(); backfill.backfill_ensemble(db.connect(cfg.db_path), cfg)"`
Expected: prints `ensemble backfill: N day-rows` with N in the ~90s. Then confirm:
Run: `.venv/bin/python -c "import sqlite3; c=sqlite3.connect('data/highcast.db'); print(c.execute('SELECT COUNT(*), MIN(target_date), MAX(target_date), AVG(n_members) FROM forecast_ens').fetchone())"`
Expected: ~90+ rows spanning ~93 days back to today, avg members ~119.

---

## Self-Review

**Spec coverage (against `2026-07-25-ensemble-uncertainty-features-design.md`, capture subset per the hybrid decision):**
- Component 1 (ensemble fetcher) → Task 3. ✓
- Component 2 (summary storage, `forecast_ens`) → Task 1 (table) + Task 2 (summary stats). ✓
- Component 5 (backfill + one-time seed) → Task 4 (`backfill_ensemble`, wake wiring, live smoke). ✓
- Components 3, 4, 6 (features / model integration / backtest) → **deliberately deferred** per the hybrid decision (can't validate for ~months); NOT in this plan. ✓
- Leakage discipline (run_date/fetched_at = issue time) → Task 3, matches `fetch_openmeteo`. ✓
- Member selection / control exclusion / floor / frac threshold / daily-max-over-climate-day → Task 2, all encoded verbatim from Global Constraints. ✓
- "Capture only, no prediction change" → enforced by scope: no task touches FEATURES/models/predictions/betting. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step has an exact command + expected output. ✓

**Type consistency:** `parse_openmeteo_ensemble(payload, icao, run_date, fetched_at, std_utc_offset, metric_key="high")` defined in Task 2, called with those args in Task 3. `store_ens(conn, rows)` defined Task 1, used Task 4. `fetch_openmeteo_ensemble(cfg, past_days=2, forecast_days=2)` defined Task 3, called in Task 4 (wake: defaults; backfill: `past_days=93`). Row dict keys ≡ `forecast_ens` columns across Tasks 1/2/4. `ENSEMBLE_URL`/`ENSEMBLE_MODELS` constants defined Task 2, asserted Task 3. ✓

**Note for whoever builds the future features plan:** once `forecast_ens` has accumulated enough seam days, that plan adds `ens_std/p10/p90/frac_ge_82/bimodality` to `features.FEATURES`, populates them in `build_features` from `forecast_ens` (run available by the 9am cutoff, NaN otherwise), and runs the with/without walk-forward backtest scored on seam-conditional coverage + CRPS. `ens_mean`/`ens_p50` are stored but not featured (center is covered by `fc_mean`).
