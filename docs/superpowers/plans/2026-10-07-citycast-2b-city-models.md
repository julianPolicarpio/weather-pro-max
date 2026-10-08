# citycast 2b — City Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train shared and per-city LightGBM quantile models for daily high and low at all 24 stations, test them walk-forward against the MOS baseline and the market, pick a variant per station/metric, and paper-trade the gated ones as a new `citycast` sandbox bot.

**Architecture:** Features are built per (station, metric, climate day, as-of instant) strictly from data issued before the as-of instant (spec §3.3). A quantile model predicts the residual from a MOS anchor; quantiles become an integer-°F pmf through `highcast.buckets`. Walk-forward quarterly refits write out-of-sample predictions to `wf_predictions`; evaluation chooses shared vs city per station/metric and applies the gate; a final model writes `live_predictions` that the sandbox bot reads read-only.

**Tech Stack:** Python 3.12, pandas, numpy, LightGBM 4.7, sqlite3, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-city-models-and-laya-design.md` §3.
**Prerequisite:** Plan 2a complete and `citycast coverage` passing.

## Global Constraints

- Commit only when Julian asks. Tests: `.venv/bin/python -m pytest`. No network in unit tests.
- Never write `data/highcast.db` or `data/sandbox.db` from citycast; the sandbox reads `data/history.db` only with `?mode=ro`. Never import `highcast.kalshi_client` under `src/citycast/`.
- **As-of rule:** every feature uses only data issued at or before the as-of instant t; MOS runs need `runtime_utc ≤ t − PUB_LAG_H[model]` (defaults GFS 5, MEX 5, NBS 3, NBE 3 hours until Task 0 measures them; a measured value may only replace a default as `observed max + 0.5 h`, never shorter); climatology uses only years before the target's year; observations use `latentmax.visible_running_extremum(..., asof_utc=t)` (never raw spot maxima).
- Running extremum → bound: high uses `iv[0]`, low uses `iv[1]` (as `highcast.signals`).
- As-of grid: local hours 06, 09, 12, 15, 18, 21 on D−1 and 00, 03, …, 21 on D; drop points at or after the climate-day end.
- Quantiles Q = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98). LightGBM params: `num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8`. Recency half-life 730 days.
- Spread factor k grid {0.6, 0.7, …, 1.6}, chosen on previous folds only; k = 1.0 until 180 days of prior predictions exist.
- Walk-forward: quarterly refits from 2022-Q3; training data from 2022-01-01. Selection period = event dates before 2026-04-01; report period = on/after. Gate (both must hold, spec §3.8 revised 2026-10-09): (1) selection period: chosen-variant minus mosnorm CRPS, cluster bootstrap by event date (2,000 resamples, seed 0), 95 % CI upper bound < 0; (2) report period: the same difference has mean < 0 over ≥ 60 distinct event dates (`REPORT_MIN_DAYS = 60`). Report-period numbers never choose the variant or k.
- Design freeze: every evaluate run logs `code_fingerprint()` (sha256 of `asof, features, dataset, qmodel, walkforward, evaluate` sources) in `eval_log`; a report produced under a fingerprint different from the first run that scored the report period carries a warning that the report period is spent. Real data is not evaluated before `PUB_LAG_H` is set from the Task 0 report.
- mosnorm baseline = Normal(f_MEX, σ = 3.0 °F), same as the sandbox `mos` bot.
- Paper only. The `citycast` sandbox account follows the existing unlock rule unchanged.

## File structure

| File | Responsibility |
|---|---|
| `src/citycast/db.py` (modify) | add `wf_predictions`, `model_choice`, `live_predictions` |
| `src/citycast/asof.py` | as-of grid, lead hours, same-day test |
| `src/citycast/features.py` | `FeatureContext`: MOS lookup, climatology, observation bounds, feature row |
| `src/citycast/dataset.py` | build/load the per-metric training frame |
| `src/citycast/qmodel.py` | quantile model, quantile scaling, pmf |
| `src/citycast/metrics.py` | RPS (the discrete CRPS), binary Brier, cluster bootstrap, normal pmf |
| `src/citycast/walkforward.py` | quarterly folds, calibration k, writes `wf_predictions` |
| `src/citycast/evaluate.py` | selection, gate, market Brier, report |
| `src/citycast/live.py` | final training, live prediction |
| `src/citycast/cli.py` (modify) | `dataset`, `walkforward`, `evaluate`, `train`, `predict` |
| `src/sandbox/bots.py`, `cycle.py`, `cli.py` (modify) | `CitycastBot` and its account |

---

### Task 0: Measure MOS publication lags (controller task, runs 24 h unattended)

Why: the as-of rule needs each model's real delay between `runtime_utc` and when IEM's archive
serves the run. The 2a review found the flat 1 h in the first draft was too short for GFS/MEX
[Likely ~4 h]. Until this task finishes, Task 2 uses the conservative `PUB_LAG_H` defaults
(GFS 5, MEX 5, NBS 3, NBE 3 h).

**Files:** Create `scripts/measure_mos_lag.py`; output `docs/reports/mos-publication-lag-<date>.md`.

- [ ] **Step 1: Write the script**

```python
# scripts/measure_mos_lag.py
"""Poll IEM's MOS archive (the endpoint the backfill uses) every 15 min for 24 h and record when
each new run first appears. Lag = first_seen - runtime. One request per station/model per poll."""
import csv
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATIONS = ("KNYC", "KLAX", "KDEN")
MODELS = ("GFS", "MEX", "NBS", "NBE")
POLL_S = 900
HOURS = 24


def latest_runtimes(station: str, model: str) -> set[str]:
    now = datetime.now(timezone.utc)
    text = httpx.get(URL, params={"station": station, "model": model, "format": "csv",
                                  "sts": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ"),
                                  "ets": (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")},
                     timeout=120).text
    return {row["runtime"][:16] for row in csv.DictReader(io.StringIO(text)) if row.get("runtime")}


def main() -> int:
    seen: dict[tuple[str, str, str], datetime] = {}
    first_poll = True
    end = time.time() + HOURS * 3600
    while time.time() < end:
        now = datetime.now(timezone.utc)
        for st in STATIONS:
            for m in MODELS:
                try:
                    for rt in latest_runtimes(st, m):
                        # runs already present at the first poll have unknown first-seen times
                        seen.setdefault((st, m, rt), None if first_poll else now)
                except Exception as e:  # noqa: BLE001 - one failed poll is not fatal
                    print(f"poll error {st} {m}: {e}", file=sys.stderr)
                time.sleep(1.0)
        first_poll = False
        time.sleep(POLL_S)
    lags: dict[str, list[float]] = {}
    for (st, m, rt), t in seen.items():
        if t is not None:
            run = datetime.fromisoformat(rt.replace(" ", "T")).replace(tzinfo=timezone.utc)
            lags.setdefault(m, []).append((t - run).total_seconds() / 3600)
    out = Path(f"docs/reports/mos-publication-lag-{datetime.now(timezone.utc).date()}.md")
    lines = ["# MOS publication lag (IEM archive first-seen, 15-min polls)", "",
             "| model | runs seen | min h | max h | proposed PUB_LAG_H (max + 0.5, poll-rounded) |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        v = sorted(lags.get(m, []))
        if v:
            lines.append(f"| {m} | {len(v)} | {v[0]:.2f} | {v[-1]:.2f} | {v[-1] + 0.5:.1f} |")
        else:
            lines.append(f"| {m} | 0 | – | – | keep default |")
    out.write_text("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it detached** — `nohup caffeinate -i .venv/bin/python scripts/measure_mos_lag.py > data/mos-lag.log 2>&1 &`. It needs no other task and can run while Tasks 1–3 are built.
- [ ] **Step 3: Apply** — when the report exists, set each `PUB_LAG_H[model]` in `src/citycast/features.py` to the report's proposed value (observed max + 0.5 h; 15-min polling means first-seen can be up to 15 min late, which only makes the lag more conservative). A model with no observed runs keeps its default. Rebuild the datasets (Task 3) if they were built with the defaults.

---

### Task 1: Schema additions and the as-of grid

**Files:**
- Modify: `src/citycast/db.py` (append to `SCHEMA`)
- Create: `src/citycast/asof.py`
- Test: `tests/test_citycast_asof.py`

**Interfaces:**
- Produces: tables below; `GRID: tuple[tuple[int, int], ...]`; `asof_points(st, day: date) -> list[datetime]` (aware UTC, ascending); `lead_hours(st, day, t) -> float`; `same_day(st, day, t) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_asof.py
from contextlib import closing
from datetime import date, datetime, timezone

from citycast import db
from citycast.asof import asof_points, lead_hours, same_day
from citycast.stations import Station

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
LAX = Station("KLAX", "LAX", "CA_ASOS", "LOX", 33.94, -118.39, 32.0, "America/Los_Angeles", -8,
              "KXHIGHLAX", "KXLOWTLAX")


def U(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_grid_winter_new_york():
    pts = asof_points(NYC, date(2024, 1, 5))
    assert len(pts) == 14
    assert pts[0] == U("2024-01-04T11:00:00")        # 06:00 EST on D-1
    assert pts[-1] == U("2024-01-06T02:00:00")       # 21:00 EST on D
    assert same_day(NYC, date(2024, 1, 5), U("2024-01-05T05:00:00"))   # 00:00 EST = day start
    assert lead_hours(NYC, date(2024, 1, 5), U("2024-01-05T05:00:00")) == 24.0


def test_summer_dst_midnight_is_still_pre_day():
    d = date(2024, 7, 5)
    pts = asof_points(LAX, d)
    assert len(pts) == 14
    midnight_pdt = U("2024-07-05T07:00:00")           # 00:00 PDT; climate day starts 08:00Z
    assert midnight_pdt in pts and not same_day(LAX, d, midnight_pdt)


def test_new_tables_exist(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"wf_predictions", "model_choice", "live_predictions"} <= names
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q tests/test_citycast_asof.py`, FAIL (module/tables missing).

- [ ] **Step 3: Implement**

Append to the `SCHEMA` string in `src/citycast/db.py` (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS wf_predictions (station TEXT, metric TEXT, event_date TEXT, asof_utc TEXT,
  variant TEXT, fold TEXT, anchor REAL, run_ext REAL, quantiles_json TEXT NOT NULL,
  pmf_json TEXT NOT NULL, k REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (station, metric, event_date, asof_utc, variant));
CREATE TABLE IF NOT EXISTS model_choice (station TEXT, metric TEXT, variant TEXT NOT NULL,
  crps_sel REAL, mos_crps_sel REAL, gate_ci_hi REAL, gate_pass INTEGER NOT NULL,
  decided_at TEXT NOT NULL, PRIMARY KEY (station, metric));
CREATE TABLE IF NOT EXISTS live_predictions (station TEXT, metric TEXT, event_date TEXT,
  issued_utc TEXT, variant TEXT NOT NULL, anchor REAL, run_ext REAL, quantiles_json TEXT NOT NULL,
  pmf_json TEXT NOT NULL, k REAL NOT NULL, PRIMARY KEY (station, metric, event_date, issued_utc));
```

```python
# src/citycast/asof.py
"""The as-of grid (spec §3.2): when, relative to a climate day, a forecast is made."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.timeutil import climate_day_bounds

GRID: tuple[tuple[int, int], ...] = (tuple((-1, h) for h in (6, 9, 12, 15, 18, 21))
                                      + tuple((0, h) for h in range(0, 24, 3)))


def asof_points(st, day: date) -> list[datetime]:
    end = climate_day_bounds(day, st.std_utc_offset)[1]
    zone = ZoneInfo(st.tz)
    out = []
    for dd, hour in GRID:
        t = datetime.combine(day + timedelta(days=dd), time(hour), tzinfo=zone).astimezone(timezone.utc)
        if t < end:
            out.append(t)
    return sorted(out)


def lead_hours(st, day: date, t: datetime) -> float:
    return (climate_day_bounds(day, st.std_utc_offset)[1] - t).total_seconds() / 3600


def same_day(st, day: date, t: datetime) -> bool:
    return t >= climate_day_bounds(day, st.std_utc_offset)[0]
```

- [ ] **Step 4: Run to verify they pass** — 3 passed.

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q`, all pass.

---

### Task 2: Feature context with leakage tests

**Files:**
- Create: `src/citycast/features.py`
- Test: `tests/test_citycast_features.py`

**Interfaces:**
- Consumes: `asof.lead_hours / same_day`, `stations.app_config`, `highcast.latentmax.visible_running_extremum`.
- Produces: `MOS_MODELS`, `FEATURE_COLS: list[str]`, `FeatureContext(conn, stations: dict, db_path, mos_since: date | None = None)` with `.row(icao, metric, day: date, t: datetime) -> dict | None` (keys `station, metric, event_date, asof_utc` + `FEATURE_COLS`; None when no anchor) and `.truth(icao, metric, day) -> float` (NaN if no final CLI).

`FEATURE_COLS` (exact order): `f_MEX, f_GFS, f_NBS, f_NBE, age_MEX, age_GFS, age_NBS, age_NBE, anchor, spread_mos, clim_mean, clim_sd, prev_obs, run_ext, run_gap, last_tmpf, last_dwpf, doy_sin, doy_cos, lead_h, local_hour, lat, lon, elevation_m`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_features.py
import math
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

from citycast import db
from citycast.features import FEATURE_COLS, FeatureContext
from citycast.stations import Station

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"
D = date(2024, 1, 5)
T_PRE = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)      # 18:00 EST on D-1
T_SAME = datetime(2024, 1, 5, 19, 0, tzinfo=timezone.utc)     # 14:00 EST on D


def _mos(model, runtime, value, n_points=1):
    return {"station": "KNYC", "model": model, "runtime_utc": runtime, "target_date": D.isoformat(),
            "metric": "high", "value_f": value, "n_points": n_points, "fetched_at": F}


def _obs(valid, tmpf, six_max_c=None):
    return {"station": "KNYC", "valid_utc": valid, "source": "metar", "tmpf": tmpf,
            "tmpf_precise": tmpf, "dwpf": 20.0, "six_max_c": six_max_c, "raw": "", "fetched_at": F}


def _seed(c):
    clim = [{"station": "KNYC", "date": f"{y}-01-0{d}", "high_f": 38 + d, "low_f": 25, "final": 1,
             "issued_utc": None, "raw": "{}", "fetched_at": F} for y in (2021, 2022, 2023) for d in range(1, 9)]
    with c:
        db.upsert(c, "cli_daily", clim)
        db.upsert(c, "mos_runs", [_mos("MEX", "2024-01-04T12:00:00+00:00", 44.0),
                                  _mos("NBS", "2024-01-04T13:00:00+00:00", 46.0, n_points=8)])
        db.upsert(c, "obs", [_obs("2024-01-05T14:51:00+00:00", 41.0),
                             _obs("2024-01-05T17:51:00+00:00", 45.0, six_max_c=7.8)])


def _ctx(c, tmp_path):
    return FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")


def test_pre_day_row(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE)
        assert list(r)[:4] == ["station", "metric", "event_date", "asof_utc"]
        assert set(FEATURE_COLS) <= set(r)
        assert r["f_MEX"] == 44.0 and r["f_NBS"] == 46.0 and math.isnan(r["f_GFS"])
        assert r["anchor"] == 45.0 and r["spread_mos"] == 2.0
        assert r["age_MEX"] == 11.0
        assert math.isnan(r["run_ext"]) and r["lead_h"] == 30.0
        assert r["clim_mean"] > 0 and r["local_hour"] == 18


def test_partial_nbs_run_is_a_feature_but_not_the_anchor(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:   # a later NBS run that covers only 2 points of the day
            db.upsert(c, "mos_runs", [_mos("NBS", "2024-01-04T19:00:00+00:00", 40.0, n_points=2)])
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 5, 2, 0, tzinfo=timezone.utc))
        assert r["f_NBS"] == 40.0 and r["npts_NBS"] == 2.0      # newest published run is reported
        assert r["anchor"] == 44.0                                # ...but only full-coverage runs anchor


def test_publication_lag_is_per_model(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        # 16:30Z: NBS (13Z + 3 h) is published, MEX (12Z + 5 h = 17Z) is not
        r = _ctx(c, tmp_path).row("KNYC", "high", D, datetime(2024, 1, 4, 16, 30, tzinfo=timezone.utc))
        assert r["f_NBS"] == 46.0 and math.isnan(r["f_MEX"]) and r["anchor"] == 46.0


def test_same_day_row_uses_visible_running_extremum(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        r = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        assert r["run_ext"] >= 45.0                       # 6-hour group 7.8 C = 46.04 F counts
        assert r["run_gap"] == r["run_ext"] - r["anchor"] and r["last_tmpf"] == 45.0


def test_no_anchor_returns_none(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        assert _ctx(c, tmp_path).row("KNYC", "high", D, T_PRE) is None


def test_leakage_future_data_changes_nothing(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        before = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        with c:
            # a MOS run 30 min before t (inside the 1 h publication lag), an obs after t,
            # and a CLI report from the target's own year
            db.upsert(c, "mos_runs", [_mos("MEX", (T_SAME - timedelta(minutes=30)).isoformat(), 99.0)])
            db.upsert(c, "obs", [_obs("2024-01-05T20:51:00+00:00", 70.0, six_max_c=25.0)])
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2024-01-03", "high_f": 90,
                                        "low_f": 80, "final": 1, "issued_utc": None, "raw": "{}",
                                        "fetched_at": F}])
        after = _ctx(c, tmp_path).row("KNYC", "high", D, T_SAME)
        same = lambda a, b: (math.isnan(a) and math.isnan(b)) or a == b   # noqa: E731
        assert all(same(before[k], after[k]) for k in FEATURE_COLS)


def test_truth_uses_final_reports_only(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "cli_daily", [
                {"station": "KNYC", "date": "2024-01-05", "high_f": 47, "low_f": 30, "final": 1,
                 "issued_utc": None, "raw": "{}", "fetched_at": F},
                {"station": "KNYC", "date": "2024-01-06", "high_f": 50, "low_f": 31, "final": 0,
                 "issued_utc": None, "raw": "{}", "fetched_at": F}])
        ctx = _ctx(c, tmp_path)
        assert ctx.truth("KNYC", "high", date(2024, 1, 5)) == 47.0
        assert math.isnan(ctx.truth("KNYC", "high", date(2024, 1, 6)))
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/features.py
"""Feature rows for (station, metric, climate day, as-of instant) — spec §3.3.

Everything here is strictly as-of t: MOS runs need runtime_utc <= t - 1 h, climatology uses
only years before the target's year, observations go through latentmax with asof_utc=t.
"""
import bisect
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from citycast.asof import lead_hours, same_day
from citycast.stations import app_config
from highcast.latentmax import visible_running_extremum
from highcast.timeutil import climate_day_bounds

MOS_MODELS = ("MEX", "GFS", "NBS", "NBE")
# Publication lag per model (spec §3.3, revised 2026-10-08): conservative defaults until Task 0
# measures first-seen times on IEM; a too-short lag leaks unpublished forecasts into training.
PUB_LAG_H = {"GFS": 5.0, "MEX": 5.0, "NBS": 3.0, "NBE": 3.0}
FULL_DAY_MIN_PTS = 6    # an NBS/NBE run counts toward the anchor only if it covers >= 18 h of the day
CLIM_WINDOW_D = 7
CLIM_MIN_N = 10
NAN = float("nan")

FEATURE_COLS = [*(f"f_{m}" for m in MOS_MODELS), *(f"age_{m}" for m in MOS_MODELS),
                *(f"npts_{m}" for m in MOS_MODELS),
                "anchor", "spread_mos", "clim_mean", "clim_sd", "prev_obs", "run_ext", "run_gap",
                "last_tmpf", "last_dwpf", "doy_sin", "doy_cos", "lead_h", "local_hour",
                "lat", "lon", "elevation_m"]


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class FeatureContext:
    def __init__(self, conn, stations: dict, db_path, mos_since: date | None = None):
        self.conn, self.stations = conn, stations
        self.cfg = {i: app_config(s, db_path) for i, s in stations.items()}
        self._mos: dict[tuple, tuple[list[str], list[float]]] = {}
        sql = ("SELECT station, model, metric, target_date, runtime_utc, value_f, n_points FROM mos_runs "
               + ("WHERE target_date >= ? " if mos_since else "") + "ORDER BY runtime_utc")
        for r in conn.execute(sql, (mos_since.isoformat(),) if mos_since else ()):
            rts, vals = self._mos.setdefault((r[0], r[1], r[2], r[3]), ([], []))
            rts.append(r[4])
            vals.append((float(r[5]), int(r[6])))
        self._cli: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._truth: dict[tuple[str, str, str], float] = {}
        for icao in stations:
            for metric, col in (("high", "high_f"), ("low", "low_f")):
                rows = conn.execute(f"SELECT date, {col} FROM cli_daily WHERE station=? AND final=1 "
                                    f"AND {col} IS NOT NULL", (icao,)).fetchall()
                ds = [date.fromisoformat(d) for d, _ in rows]
                self._cli[(icao, metric)] = (np.array([d.year for d in ds]),
                                             np.array([d.timetuple().tm_yday for d in ds]),
                                             np.array([float(v) for _, v in rows]))
                for d, v in rows:
                    self._truth[(icao, metric, d)] = float(v)
        self._clim_cache: dict[tuple, tuple[float, float]] = {}

    def truth(self, icao: str, metric: str, day: date) -> float:
        return self._truth.get((icao, metric, day.isoformat()), NAN)

    def mos(self, icao, model, metric, day: date, t: datetime) -> tuple[float, float, float]:
        """(value, age_h, n_points) of the latest run published by t; NaNs when none."""
        found = self._mos.get((icao, model, metric, day.isoformat()))
        if not found:
            return NAN, NAN
        rts, vals = found
        i = bisect.bisect_right(rts, (t - timedelta(hours=PUB_LAG_H[model])).isoformat()) - 1
        if i < 0:
            return NAN, NAN
        value, npts = vals[i]
        return value, (t - datetime.fromisoformat(rts[i])).total_seconds() / 3600, float(npts)

    def clim(self, icao, metric, day: date) -> tuple[float, float]:
        key = (icao, metric, day)
        if key not in self._clim_cache:
            years, doys, vals = self._cli[(icao, metric)]
            doy = day.timetuple().tm_yday
            dist = np.abs(doys - doy)
            dist = np.minimum(dist, 366 - dist)
            v = vals[(years < day.year) & (dist <= CLIM_WINDOW_D)]
            self._clim_cache[key] = ((float(v.mean()), float(v.std(ddof=1)))
                                     if len(v) >= CLIM_MIN_N else (NAN, NAN))
        return self._clim_cache[key]

    def bound(self, icao, metric, day: date, t: datetime) -> float:
        extremum = "max" if metric == "high" else "min"
        iv = visible_running_extremum(self.conn, self.cfg[icao], day, extremum, asof_utc=t.isoformat())
        if iv is None:
            return NAN
        return float(iv[0] if metric == "high" else iv[1])

    def last_obs(self, icao, day: date, t: datetime) -> tuple[float, float]:
        start = climate_day_bounds(day, self.stations[icao].std_utc_offset)[0]
        r = self.conn.execute("SELECT tmpf, dwpf FROM obs WHERE station=? AND valid_utc<=? "
                              "AND valid_utc>=? AND tmpf IS NOT NULL ORDER BY valid_utc DESC LIMIT 1",
                              (icao, t.isoformat(), start.isoformat())).fetchone()
        if r is None:
            return NAN, NAN
        return float(r[0]), (NAN if r[1] is None else float(r[1]))

    def row(self, icao: str, metric: str, day: date, t: datetime) -> dict | None:
        st = self.stations[icao]
        r: dict = {"station": icao, "metric": metric, "event_date": day.isoformat(),
                   "asof_utc": t.isoformat()}
        for m in MOS_MODELS:
            r[f"f_{m}"], r[f"age_{m}"], r[f"npts_{m}"] = self.mos(icao, m, metric, day, t)

        def full(m):   # MEX/GFS carry one n_x value; NBS/NBE must cover most of the day
            v = r[f"f_{m}"]
            ok = m in ("MEX", "GFS") or (not _isnan(r[f"npts_{m}"]) and r[f"npts_{m}"] >= FULL_DAY_MIN_PTS)
            return v if ok and not _isnan(v) else NAN
        pair = [full(m) for m in ("NBS", "MEX") if not _isnan(full(m))]
        if pair:
            anchor = float(np.mean(pair))
        elif not _isnan(full("GFS")):
            anchor = full("GFS")
        else:
            anchor = full("NBE")
        if _isnan(anchor):
            return None
        avail = [r[f"f_{m}"] for m in MOS_MODELS if not _isnan(r[f"f_{m}"])]
        r["anchor"], r["spread_mos"] = anchor, float(max(avail) - min(avail))
        r["clim_mean"], r["clim_sd"] = self.clim(icao, metric, day)
        r["prev_obs"] = self.bound(icao, metric, day - timedelta(days=1), t)
        if same_day(st, day, t):
            b = self.bound(icao, metric, day, t)
            r["run_ext"] = b
            r["run_gap"] = NAN if _isnan(b) else b - anchor
            r["last_tmpf"], r["last_dwpf"] = self.last_obs(icao, day, t)
        else:
            r["run_ext"] = r["run_gap"] = r["last_tmpf"] = r["last_dwpf"] = NAN
        doy = day.timetuple().tm_yday
        r["doy_sin"], r["doy_cos"] = math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)
        r["lead_h"] = lead_hours(st, day, t)
        r["local_hour"] = t.astimezone(ZoneInfo(st.tz)).hour
        r["lat"], r["lon"], r["elevation_m"] = st.lat, st.lon, st.elevation_m
        return r
```

- [ ] **Step 4: Run to verify they pass** — 5 passed. If `test_same_day_row_uses_visible_running_extremum` fails because latentmax rejects the 17:51Z six-hour group (its 6-hour window must lie inside the climate day: 11:51–17:51Z does, since the day starts 05:00Z), print `visible_running_extremum` output and fix the **test data**, not latentmax.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 3: Dataset builder

**Files:**
- Create: `src/citycast/dataset.py`
- Test: `tests/test_citycast_dataset.py`

**Interfaces:**
- Consumes: `FeatureContext`, `asof_points`.
- Produces: `DATASET_DIR = Path("data/citycast")`; `build(ctx, metric, since: date, until: date, icaos=None, progress=print) -> pd.DataFrame` (columns: key columns + `FEATURE_COLS` + `y`, rows without anchor skipped, `y` NaN when no final CLI); `save(df, metric)` / `load(metric) -> pd.DataFrame` (pickle at `data/citycast/dataset-{metric}.pkl`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citycast_dataset.py
from contextlib import closing
from datetime import date

from citycast import dataset, db
from citycast.features import FEATURE_COLS, FeatureContext
from test_citycast_features import NYC, _seed


def test_build_one_day_has_grid_rows_and_truth(tmp_path, monkeypatch):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2024-01-05", "high_f": 47,
                                        "low_f": 30, "final": 1, "issued_utc": None, "raw": "{}",
                                        "fetched_at": "x"}])
        ctx = FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")
        df = dataset.build(ctx, "high", date(2024, 1, 5), date(2024, 1, 5), progress=lambda m: None)
        # 14 grid points; NBS (13Z Jan 4, 3 h lag) is usable from 16Z and MEX (12Z, 5 h lag) from 17Z,
        # so 06:00 and 09:00 EST on D-1 (11Z, 14Z) have no anchor
        assert len(df) == 12 and set(FEATURE_COLS) <= set(df.columns)
        assert (df["y"] == 47.0).all() and df["asof_utc"].is_monotonic_increasing
        monkeypatch.setattr(dataset, "DATASET_DIR", tmp_path / "ds")
        dataset.save(df, "high")
        assert dataset.load("high").equals(df)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/dataset.py
"""Per-metric training frame: one row per (station, climate day, as-of point) — spec §3.2–3.4."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from citycast.asof import asof_points
from citycast.features import FEATURE_COLS

DATASET_DIR = Path("data/citycast")
KEY_COLS = ["station", "metric", "event_date", "asof_utc"]


def build(ctx, metric: str, since: date, until: date, icaos=None, progress=print) -> pd.DataFrame:
    rows = []
    for icao in (icaos or sorted(ctx.stations)):
        st = ctx.stations[icao]
        day = since
        while day <= until:
            y = ctx.truth(icao, metric, day)
            for t in asof_points(st, day):
                r = ctx.row(icao, metric, day, t)
                if r is not None:
                    r["y"] = y
                    rows.append(r)
            day += timedelta(days=1)
        progress(f"{metric} {icao}: {len(rows)} rows so far")
    df = pd.DataFrame(rows, columns=KEY_COLS + FEATURE_COLS + ["y"])
    return df.sort_values(["station", "event_date", "asof_utc"]).reset_index(drop=True)


def _path(metric: str) -> Path:
    return DATASET_DIR / f"dataset-{metric}.pkl"


def save(df: pd.DataFrame, metric: str) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(_path(metric))
    return _path(metric)


def load(metric: str) -> pd.DataFrame:
    return pd.read_pickle(_path(metric))
```

- [ ] **Step 4: Run to verify it passes** — 1 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 4: Quantile model and pmf

**Files:**
- Create: `src/citycast/qmodel.py`
- Test: `tests/test_citycast_qmodel.py`

**Interfaces:**
- Consumes: `FEATURE_COLS`, `highcast.buckets.bucket_probs / truncate_at_bound`.
- Produces: `QUANTILES`, `PARAMS`, `recency_weights(event_dates: pd.Series, ref: date) -> np.ndarray`, `QuantileModel(variant: str, station_categories: tuple[str, ...])` with `.fit(df, ref: date) -> self` and `.predict_y_quantiles(df) -> np.ndarray` shape (n, 13), ascending per row, already including the anchor; `scale_quantiles(q: dict[float, float], k: float) -> dict`; `pmf_from_quantiles(q: dict, k: float, bound: float, metric: str) -> dict[int, float]`.

Variants: `"shared"` uses `FEATURE_COLS + ["station"]` (station as categorical with fixed categories); `"city"` uses `FEATURE_COLS` minus `lat, lon, elevation_m`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_qmodel.py
from datetime import date

import numpy as np
import pandas as pd

from citycast.features import FEATURE_COLS
from citycast.qmodel import (QUANTILES, QuantileModel, pmf_from_quantiles, recency_weights,
                             scale_quantiles)


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["anchor"] = rng.uniform(30, 90, n)
    df["station"] = rng.choice(["KNYC", "KLAX"], n)
    df["event_date"] = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    df["y"] = df["anchor"] + rng.normal(0, 2, n)
    return df


def test_recency_weights_halve_every_730_days():
    w = recency_weights(pd.Series(["2024-01-01", "2022-01-01"]), date(2024, 1, 1))
    assert w[0] == 1.0 and abs(w[1] - 0.5) < 0.01


def test_fit_predict_shapes_and_monotone():
    df = _frame()
    for variant in ("shared", "city"):
        m = QuantileModel(variant, ("KLAX", "KNYC")).fit(df, date(2025, 1, 1))
        q = m.predict_y_quantiles(df.head(5))
        assert q.shape == (5, len(QUANTILES))
        assert (np.diff(q, axis=1) > 0).all()
        assert abs(np.median(q[:, 6] - df.head(5)["anchor"].to_numpy())) < 3


def test_scale_quantiles_about_median():
    q = {0.1: 40.0, 0.5: 50.0, 0.9: 60.0}
    assert scale_quantiles(q, 0.5) == {0.1: 45.0, 0.5: 50.0, 0.9: 55.0}


def test_pmf_sums_to_one_and_truncates_high_below_bound():
    q = dict(zip(QUANTILES, np.linspace(40, 60, len(QUANTILES))))
    pmf = pmf_from_quantiles(q, 1.0, float("nan"), "high")
    assert abs(sum(pmf.values()) - 1) < 1e-9
    cut = pmf_from_quantiles(q, 1.0, 52.4, "high")
    assert min(k for k, v in cut.items() if v > 0) >= 52 and abs(sum(cut.values()) - 1) < 1e-9
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/qmodel.py
"""Residual quantile model (spec §3.4) and quantiles -> integer pmf."""
import math
from datetime import date

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from citycast.features import FEATURE_COLS
from highcast.buckets import bucket_probs, truncate_at_bound

QUANTILES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
HALF_LIFE_D = 730.0
MIN_GAP_F = 1e-3
CITY_COLS = [c for c in FEATURE_COLS if c not in ("lat", "lon", "elevation_m")]
SHARED_COLS = FEATURE_COLS + ["station"]


def recency_weights(event_dates: pd.Series, ref: date) -> np.ndarray:
    age = (pd.Timestamp(ref) - pd.to_datetime(event_dates)).dt.days.clip(lower=0).to_numpy()
    return 0.5 ** (age / HALF_LIFE_D)


class QuantileModel:
    def __init__(self, variant: str, station_categories: tuple[str, ...]):
        if variant not in ("shared", "city"):
            raise ValueError(variant)
        self.variant = variant
        self.cats = tuple(station_categories)
        self.cols = SHARED_COLS if variant == "shared" else CITY_COLS
        self.models: dict[float, LGBMRegressor] = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].copy()
        if "station" in X:
            X["station"] = pd.Categorical(X["station"], categories=list(self.cats))
        return X

    def fit(self, df: pd.DataFrame, ref: date) -> "QuantileModel":
        X = self._X(df)
        resid = (df["y"] - df["anchor"]).to_numpy()
        w = recency_weights(df["event_date"], ref)
        self.models = {a: LGBMRegressor(objective="quantile", alpha=a, **PARAMS).fit(X, resid, sample_weight=w)
                       for a in QUANTILES}
        return self

    def predict_y_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        q = np.column_stack([self.models[a].predict(X) for a in QUANTILES])
        q = np.sort(q, axis=1) + df["anchor"].to_numpy()[:, None]
        for j in range(1, q.shape[1]):                       # strictly increasing for the CDF
            q[:, j] = np.maximum(q[:, j], q[:, j - 1] + MIN_GAP_F)
        return q


def scale_quantiles(q: dict[float, float], k: float) -> dict[float, float]:
    m = q[0.5]
    return {a: m + k * (v - m) for a, v in q.items()}


def pmf_from_quantiles(q: dict[float, float], k: float, bound: float, metric: str) -> dict[int, float]:
    pmf = bucket_probs(scale_quantiles(q, k))
    if bound is not None and not math.isnan(bound):
        pmf = truncate_at_bound(pmf, bound, "max" if metric == "high" else "min")
    return pmf
```

- [ ] **Step 4: Run to verify they pass** — 4 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 5: Metrics

**Files:**
- Create: `src/citycast/metrics.py`
- Test: `tests/test_citycast_metrics.py`

**Interfaces:**
- Produces: `rps(pmf: dict[int, float], y: int) -> float` (discrete CRPS over integer °F); `brier(p: float, won: int) -> float`; `cluster_bootstrap(values_by_cluster: dict, n_boot=2000, seed=0) -> tuple[float, float, float]` (mean, 2.5 %, 97.5 % of ratio-of-sums); `normal_pmf(mu: float, sigma: float = 3.0) -> dict[int, float]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_metrics.py
from citycast.metrics import brier, cluster_bootstrap, normal_pmf, rps


def test_rps_zero_for_perfect_and_grows_with_distance():
    assert rps({50: 1.0}, 50) == 0.0
    assert rps({50: 1.0}, 52) == 2.0
    assert rps({49: 0.5, 51: 0.5}, 50) == 0.5


def test_brier():
    assert brier(0.7, 1) == 0.09000000000000002 or abs(brier(0.7, 1) - 0.09) < 1e-12
    assert brier(0.7, 0) == 0.49


def test_cluster_bootstrap_mean_and_ci_order():
    vals = {f"d{i}": [-1.0, -0.5] for i in range(50)}
    m, lo, hi = cluster_bootstrap(vals)
    assert m == -0.75 and lo <= m <= hi < 0


def test_normal_pmf_centered_and_normalised():
    p = normal_pmf(60.0)
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == 60
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/metrics.py
"""Scores (spec §3.8): discrete CRPS (ranked probability score), Brier, cluster bootstrap."""
import math

import numpy as np

SIGMA_F = 3.0   # the sandbox mos baseline's fixed spread


def rps(pmf: dict[int, float], y: int) -> float:
    lo = min(min(pmf), y) - 1
    hi = max(max(pmf), y) + 1
    cum, total = 0.0, 0.0
    for t in range(lo, hi + 1):
        cum += pmf.get(t, 0.0)
        total += (cum - (1.0 if t >= y else 0.0)) ** 2
    return total


def brier(p: float, won: int) -> float:
    return (p - won) ** 2


def cluster_bootstrap(values_by_cluster: dict, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    sums = np.array([sum(v) for v in values_by_cluster.values()], dtype=float)
    counts = np.array([len(v) for v in values_by_cluster.values()], dtype=float)
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pmf(mu: float, sigma: float = SIGMA_F) -> dict[int, float]:
    c = round(mu)
    raw = {t: _phi((t + 0.5 - mu) / sigma) - _phi((t - 0.5 - mu) / sigma) for t in range(c - 20, c + 21)}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}
```

- [ ] **Step 4: Run to verify they pass** — 4 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 6: Walk-forward with calibration

**Files:**
- Create: `src/citycast/walkforward.py`
- Test: `tests/test_citycast_walkforward.py`

**Interfaces:**
- Consumes: `QuantileModel`, `QUANTILES`, `pmf_from_quantiles`, `rps`, `normal_pmf`, `db.upsert`.
- Produces: `K_GRID`, `quarters(first: date, today: date) -> list[tuple[str, date, date]]` (label like `"2022Q3"`, start, end-exclusive; only quarters ending on or before `today`); `choose_k(prev: list[tuple[dict, float, float]], metric) -> float` (prev items = (raw quantiles, bound, y)); `run(conn, df, metric, stations: tuple[str, ...], now_iso, train_start=date(2022, 1, 1), first_fold=date(2022, 7, 1), today=None, progress=print) -> int` (rows written).

Rules inside `run`, per quarter: training rows = `train_start ≤ event_date < start` with `y` not NaN; test rows = `start ≤ event_date < end` with `y` not NaN. Skip the quarter if fewer than 1,000 training rows. Shared model on all stations; city model per station with ≥ 300 training rows for that station. Before predicting a (station, variant) in a quarter, k = `choose_k` over that station/metric/variant's earlier `wf_predictions` (raw quantiles, `run_ext`, truth) whose event dates are before the quarter start; k = 1.0 if those span < 180 days. Also write `mosnorm` rows (`normal_pmf(f_MEX)`, `quantiles_json = "{}"`, `k = 1.0`) for test rows with `f_MEX` present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_walkforward.py
import json
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd

from citycast import db
from citycast.features import FEATURE_COLS
from citycast.walkforward import choose_k, quarters, run


def test_quarters_only_complete_ones():
    q = quarters(date(2022, 7, 1), date(2023, 1, 15))
    assert q == [("2022Q3", date(2022, 7, 1), date(2022, 10, 1)),
                 ("2022Q4", date(2022, 10, 1), date(2023, 1, 1))]


def test_choose_k_widens_overconfident_forecasts():
    rng = np.random.default_rng(1)
    prev = []
    for i in range(400):
        y = 50 + rng.normal(0, 4)
        q = {a: 50 + v for a, v in zip((0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
                                        np.linspace(-2, 2, 13))}
        prev.append((q, float("nan"), round(y)))
    assert choose_k(prev, "high") > 1.0
    assert choose_k([], "high") == 1.0


def _frame():
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    rows = []
    for st in ("KNYC", "KLAX"):
        for d in days:
            y = round(60 + rng.normal(0, 3))
            for hour in ("06", "12", "18"):          # 3 as-of points/day: 1,086 rows before 2022-Q3
                r = {c: rng.normal() for c in FEATURE_COLS}
                r.update(station=st, metric="high", event_date=d.strftime("%Y-%m-%d"),
                         asof_utc=f"{d.strftime('%Y-%m-%d')}T{hour}:00:00+00:00",
                         anchor=60.0, f_MEX=60.0, run_ext=float("nan"), y=y)
                rows.append(r)
    return pd.DataFrame(rows)


def test_run_writes_out_of_sample_rows_for_all_variants(tmp_path):
    df = _frame()
    with closing(db.connect(tmp_path / "h.db")) as c:
        n = run(c, df, "high", ("KLAX", "KNYC"), lambda: "2026-10-07T00:00:00+00:00",
                today=date(2023, 1, 1), progress=lambda m: None)
        got = c.execute("SELECT variant, COUNT(*), MIN(event_date) FROM wf_predictions GROUP BY variant").fetchall()
        assert n > 0
        assert {g[0] for g in got} == {"shared", "city", "mosnorm"}
        assert all(g[2] >= "2022-07-01" for g in got)          # never predicts its training period
        pmf = json.loads(c.execute("SELECT pmf_json FROM wf_predictions LIMIT 1").fetchone()[0])
        assert abs(sum(pmf.values()) - 1) < 1e-6
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/walkforward.py
"""Quarterly walk-forward (spec §3.6–3.7): every stored prediction is out of sample."""
import json
import math
from datetime import date, timedelta

import pandas as pd

from citycast import db
from citycast.metrics import normal_pmf, rps
from citycast.qmodel import QUANTILES, QuantileModel, pmf_from_quantiles

K_GRID = tuple(round(0.6 + 0.1 * i, 1) for i in range(11))
MIN_TRAIN_ROWS = 1000
MIN_CITY_ROWS = 300
MIN_K_SPAN_D = 180


def quarters(first: date, today: date) -> list[tuple[str, date, date]]:
    out = []
    start = date(first.year, 3 * ((first.month - 1) // 3) + 1, 1)
    while True:
        m = start.month + 3
        end = date(start.year + (m > 12), (m - 1) % 12 + 1, 1)
        if end > today:
            return out
        out.append((f"{start.year}Q{(start.month - 1) // 3 + 1}", start, end))
        start = end


def choose_k(prev: list[tuple[dict, float, float]], metric: str) -> float:
    if not prev:
        return 1.0
    best, best_score = 1.0, math.inf
    for k in K_GRID:
        score = sum(rps(pmf_from_quantiles(q, k, b, metric), int(y)) for q, b, y in prev) / len(prev)
        if score < best_score - 1e-12:
            best, best_score = k, score
    return best


def _prev(conn, station, metric, variant, before: date, truth: dict) -> list:
    rows = conn.execute("SELECT event_date, quantiles_json, run_ext FROM wf_predictions WHERE station=? "
                        "AND metric=? AND variant=? AND event_date<?",
                        (station, metric, variant, before.isoformat())).fetchall()
    if not rows or (before - date.fromisoformat(min(r[0] for r in rows))).days < MIN_K_SPAN_D:
        return []
    out = []
    for d, qj, b in rows:
        y = truth.get((station, d))
        if y is not None:
            q = {float(a): v for a, v in json.loads(qj).items()}
            out.append((q, float("nan") if b is None else b, y))
    return out


def _rows(test: pd.DataFrame, qmat, metric, variant, fold, k_by_station, now) -> list[dict]:
    out = []
    for i, r in enumerate(test.itertuples(index=False)):
        q = dict(zip(QUANTILES, (float(v) for v in qmat[i])))
        k = k_by_station[r.station]
        pmf = pmf_from_quantiles(q, k, r.run_ext, metric)
        out.append({"station": r.station, "metric": metric, "event_date": r.event_date,
                    "asof_utc": r.asof_utc, "variant": variant, "fold": fold, "anchor": r.anchor,
                    "run_ext": None if math.isnan(r.run_ext) else r.run_ext,
                    "quantiles_json": json.dumps(q), "pmf_json": json.dumps(pmf), "k": k,
                    "created_at": now})
    return out


def run(conn, df: pd.DataFrame, metric: str, stations: tuple[str, ...], now_iso,
        train_start: date = date(2022, 1, 1), first_fold: date = date(2022, 7, 1),
        today: date | None = None, progress=print) -> int:
    df = df[df["metric"] == metric] if "metric" in df else df
    known = df[df["y"].notna()]
    truth = {(r.station, r.event_date): int(r.y) for r in known[["station", "event_date", "y"]]
             .drop_duplicates(["station", "event_date"]).itertuples(index=False)}
    written = 0
    for label, start, end in quarters(first_fold, today or date.today()):
        train = known[(known["event_date"] >= train_start.isoformat()) & (known["event_date"] < start.isoformat())]
        test = known[(known["event_date"] >= start.isoformat()) & (known["event_date"] < end.isoformat())]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            progress(f"{metric} {label}: skipped ({len(train)} train rows)")
            continue
        now = now_iso()
        rows: list[dict] = []
        shared = QuantileModel("shared", stations).fit(train, start)
        k_shared = {s: choose_k(_prev(conn, s, metric, "shared", start, truth), metric) for s in test["station"].unique()}
        rows += _rows(test, shared.predict_y_quantiles(test), metric, "shared", label, k_shared, now)
        for s in sorted(test["station"].unique()):
            tr, te = train[train["station"] == s], test[test["station"] == s]
            if len(tr) < MIN_CITY_ROWS:
                continue
            city = QuantileModel("city", stations).fit(tr, start)
            k_city = {s: choose_k(_prev(conn, s, metric, "city", start, truth), metric)}
            rows += _rows(te, city.predict_y_quantiles(te), metric, "city", label, k_city, now)
        for r in test[test["f_MEX"].notna()].itertuples(index=False):
            rows.append({"station": r.station, "metric": metric, "event_date": r.event_date,
                         "asof_utc": r.asof_utc, "variant": "mosnorm", "fold": label, "anchor": r.anchor,
                         "run_ext": None, "quantiles_json": "{}",
                         "pmf_json": json.dumps(normal_pmf(r.f_MEX)), "k": 1.0, "created_at": now})
        with conn:
            written += db.upsert(conn, "wf_predictions", rows)
        progress(f"{metric} {label}: {len(rows)} predictions")
    return written
```

- [ ] **Step 4: Run to verify they pass** — 3 passed (the run test takes a few seconds: 2 quarters × 13 quantile models).

- [ ] **Step 5: Full suite** — all pass.

---

### Task 7: Evaluation, selection, gate and report

**Files:**
- Create: `src/citycast/evaluate.py`
- Modify: `src/citycast/db.py` (`model_choice` columns; new `eval_log` table — no live DB has `model_choice` yet, checked 2026-10-09, so the CREATE statement is edited in place)
- Test: `tests/test_citycast_evaluate.py`

**Interfaces:**
- Consumes: `rps`, `brier`, `cluster_bootstrap`, `highcast.buckets.pmf_mass`.
- Produces: `SELECTION_END = "2026-04-01"`, `MARKET_STALE_H = 2`, `REPORT_MIN_DAYS = 60`; `code_fingerprint() -> str`; `canonical_events(conn) -> set[str]` (one event per station/metric/date: most markets, ties → smallest ticker; plan 3 imports it); `scored(conn, metric) -> pd.DataFrame` (one row per wf prediction with truth: columns `station, metric, event_date, asof_utc, variant, rps, pmf`); `select_and_gate(conn, metric, now_iso) -> list[dict]` (writes `model_choice`); `market_brier(conn, metric) -> dict[str, tuple[float, float, float, int]]` (station → (mean model−market Brier, CI lo, CI hi, n contracts), chosen variant, report period and selection period combined but reported by period in the report); `write_report(conn, path: Path) -> Path`.

Selection: for each station, compare `shared` vs `city` mean RPS on rows (same `event_date, asof_utc`) present for both in the selection period; ties → `shared`; a station with no `city` rows → `shared`. Gate: per row diff = chosen RPS − mosnorm RPS on common keys in the selection period, clustered by `event_date`; `sel_pass = ci_hi < 0`. Report confirmation: the same paired diff on report-period rows; `rep_days` = distinct event dates; `gate_pass = sel_pass and rep_days >= REPORT_MIN_DAYS and mean diff < 0`. Each `select_and_gate` call also writes one `eval_log` row `(run_at, metric, fingerprint, report_scored)` where `report_scored = 1` if any report-period paired rows existed.

Schema (replace the `model_choice` statement in `db.SCHEMA` and add `eval_log`):

```sql
CREATE TABLE IF NOT EXISTS model_choice (station TEXT, metric TEXT, variant TEXT NOT NULL,
  crps_sel REAL, mos_crps_sel REAL, gate_ci_hi REAL, sel_pass INTEGER NOT NULL,
  crps_rep REAL, mos_crps_rep REAL, rep_days INTEGER NOT NULL, gate_pass INTEGER NOT NULL,
  decided_at TEXT NOT NULL, PRIMARY KEY (station, metric));
CREATE TABLE IF NOT EXISTS eval_log (run_at TEXT, metric TEXT, fingerprint TEXT NOT NULL,
  report_scored INTEGER NOT NULL, PRIMARY KEY (run_at, metric));
```

Market Brier: for each chosen-variant prediction and each usable market of that event (`kalshi_markets` with matching station, metric, event_date), take the newest candle with `end_utc` in `(asof − 2 h, asof]` having both `yes_bid` and `yes_ask`; model p = `pmf_mass(pmf, lo, hi)`; market p = mid; won = truth in [lo, hi]; diff = brier(model) − brier(market), clustered by `event_ticker`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_evaluate.py
import json
from contextlib import closing

from citycast import db
from citycast.evaluate import market_brier, select_and_gate

F = "2026-10-07T00:00:00+00:00"


def _pred(variant, d, pmf, asof="12:00"):
    return {"station": "KNYC", "metric": "high", "event_date": d, "asof_utc": f"{d}T{asof}:00+00:00",
            "variant": variant, "fold": "x", "anchor": 50.0, "run_ext": None, "quantiles_json": "{}",
            "pmf_json": json.dumps(pmf), "k": 1.0, "created_at": F}


def _seed(c, good="city", rep_days=60, rep_good=True):
    """60 selection-period days where `good` beats everything, then `rep_days` report-period days
    where the chosen variant beats mosnorm (rep_good) or loses to it."""
    preds, cli = [], []
    days = [(f"2025-0{1 + i // 28}-{1 + i % 28:02d}", True) for i in range(60)]
    days += [(f"2026-0{4 + i // 28}-{1 + i % 28:02d}", rep_good) for i in range(rep_days)]
    for d, wins in days:
        cli.append({"station": "KNYC", "date": d, "high_f": 50, "low_f": 30, "final": 1,
                    "issued_utc": None, "raw": "{}", "fetched_at": F})
        sharp, wide = {50: 0.9, 51: 0.1}, {47: 0.5, 53: 0.5}
        preds += [_pred(good, d, sharp if wins else wide),
                  _pred("shared" if good == "city" else "city", d, {49: 0.5, 51: 0.5}),
                  _pred("mosnorm", d, wide if wins else sharp)]
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "wf_predictions", preds)


def test_selects_better_variant_and_passes_gate(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["variant"] == "city" and out[0]["gate_pass"] == 1
        row = c.execute("SELECT variant, gate_pass FROM model_choice WHERE station='KNYC'").fetchone()
        assert tuple(row) == ("city", 1)
        log = c.execute("SELECT metric, report_scored FROM eval_log").fetchall()
        assert [tuple(r) for r in log] == [("high", 1)]


def test_gate_fails_when_model_loses_in_report_period(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_good=False)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["sel_pass"] == 1 and out[0]["gate_pass"] == 0
        assert out[0]["crps_rep"] > out[0]["mos_crps_rep"]


def test_gate_fails_with_too_few_report_days(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=59)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["sel_pass"] == 1 and out[0]["rep_days"] == 59 and out[0]["gate_pass"] == 0


def test_variant_choice_ignores_report_period(tmp_path):
    # city wins the selection period; in the report period shared is far better than city.
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=0)
        rows = []
        for i in range(60):
            d = f"2026-0{4 + i // 28}-{1 + i % 28:02d}"
            rows += [_pred("city", d, {40: 1.0}), _pred("shared", d, {50: 1.0}), _pred("mosnorm", d, {45: 1.0})]
        with c:
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": r["event_date"], "high_f": 50, "low_f": 30,
                                        "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F}
                                       for r in rows[::3]])
            db.upsert(c, "wf_predictions", rows)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["variant"] == "city"                      # chosen on selection only
        assert out[0]["gate_pass"] == 0                         # and then fails confirmation


def test_report_warns_when_code_changed_after_report_was_scored(tmp_path):
    from citycast.evaluate import code_fingerprint, write_report
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        select_and_gate(c, "high", lambda: F)
        assert "WARNING" not in write_report(c, tmp_path / "r.md").read_text()
        with c:
            c.execute("UPDATE eval_log SET fingerprint='0000'")       # first scoring ran on other code
        assert "WARNING" in write_report(c, tmp_path / "r.md").read_text()
        assert len(code_fingerprint()) == 64


def test_canonical_events_picks_one_event_per_city_day(tmp_path):
    from citycast.evaluate import canonical_events
    base = {"series": "KXHIGHCHI", "station": "KMDW", "metric": "high", "event_date": "2024-02-28",
            "lo": 1, "hi": 2, "open_time": None, "close_time": "2024-02-29T05:59:00+00:00",
            "result": "no", "expiration_value": "", "raw": "{}"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [{**base, "ticker": "A1", "event_ticker": "EV-B"},
                                            {**base, "ticker": "A2", "event_ticker": "EV-B"},
                                            {**base, "ticker": "B1", "event_ticker": "EV-A"}])
        assert canonical_events(c) == {"EV-B"}                   # more markets wins


def test_market_brier_uses_fresh_candle_only(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        select_and_gate(c, "high", lambda: F)
        with c:
            db.upsert(c, "kalshi_markets", [{
                "ticker": "M1", "series": "KXHIGHNY", "event_ticker": "E1", "station": "KNYC",
                "metric": "high", "event_date": "2025-01-01", "lo": 50, "hi": 51, "open_time": None,
                "close_time": "2025-01-02T04:59:00+00:00", "result": "yes", "expiration_value": "50",
                "raw": "{}"}])
            db.upsert(c, "kalshi_candles", [
                {"ticker": "M1", "end_utc": "2025-01-01T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.50,
                 "volume": 1.0, "open_interest": 1.0},
                {"ticker": "M1", "end_utc": "2025-01-01T08:00:00+00:00", "yes_bid": 0.10, "yes_ask": 0.20,
                 "volume": 1.0, "open_interest": 1.0}])
        res = market_brier(c, "high")
        mean, lo, hi, n = res["KNYC"]
        # model p = 1.0 (all mass on 50-51), market mid 0.45 from the 11:00 candle; won = 1
        assert n == 1 and abs(mean - (0.0 - 0.3025)) < 1e-9
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/evaluate.py
"""Selection, gate and market comparison (spec §3.8)."""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from citycast.metrics import brier, cluster_bootstrap, rps
from highcast.buckets import pmf_mass

SELECTION_END = "2026-04-01"
MARKET_STALE_H = 2
REPORT_MIN_DAYS = 60
FINGERPRINT_MODULES = ("asof", "features", "dataset", "qmodel", "walkforward", "evaluate")


def code_fingerprint() -> str:
    """sha256 over the modelling code, so a report can tell whether the report period was first
    scored under different code (spec §3.8 design freeze)."""
    h = hashlib.sha256()
    for name in FINGERPRINT_MODULES:
        h.update(name.encode())
        h.update((Path(__file__).parent / f"{name}.py").read_bytes())
    return h.hexdigest()


def _truth(conn, metric) -> dict:
    col = "high_f" if metric == "high" else "low_f"
    return {(s, d): v for s, d, v in conn.execute(
        f"SELECT station, date, {col} FROM cli_daily WHERE final=1 AND {col} IS NOT NULL")}


def scored(conn, metric: str) -> pd.DataFrame:
    truth = _truth(conn, metric)
    rows = []
    for s, d, a, v, pj in conn.execute("SELECT station, event_date, asof_utc, variant, pmf_json FROM wf_predictions "
                                       "WHERE metric=?", (metric,)):
        y = truth.get((s, d))
        if y is None:
            continue
        pmf = {int(k): p for k, p in json.loads(pj).items()}
        rows.append({"station": s, "metric": metric, "event_date": d, "asof_utc": a, "variant": v,
                     "rps": rps(pmf, int(y)), "pmf": pmf, "y": int(y)})
    return pd.DataFrame(rows, columns=["station", "metric", "event_date", "asof_utc", "variant", "rps", "pmf", "y"])


def _paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    keys = ["station", "event_date", "asof_utc"]
    left = df[df["variant"] == a][keys + ["rps"]].rename(columns={"rps": "a"})
    right = df[df["variant"] == b][keys + ["rps"]].rename(columns={"rps": "b"})
    return left.merge(right, on=keys)


def select_and_gate(conn, metric: str, now_iso) -> list[dict]:
    df = scored(conn, metric)
    sel = df[df["event_date"] < SELECTION_END]
    rep = df[df["event_date"] >= SELECTION_END]
    out, report_scored = [], 0
    for station in sorted(sel["station"].unique()):
        s = sel[sel["station"] == station]
        both = _paired(s, "city", "shared")
        variant = "city" if len(both) and both["a"].mean() < both["b"].mean() else "shared"
        gate = _paired(s, variant, "mosnorm")
        if gate.empty:
            continue
        by_day: dict[str, list[float]] = {}
        for r in gate.itertuples(index=False):
            by_day.setdefault(r.event_date, []).append(r.a - r.b)
        _, _, ci_hi = cluster_bootstrap(by_day)
        conf = _paired(rep[rep["station"] == station], variant, "mosnorm")   # report period: confirm only
        rep_days = int(conf["event_date"].nunique())
        report_scored |= rep_days > 0
        sel_pass = ci_hi < 0
        rep_ok = rep_days >= REPORT_MIN_DAYS and float((conf["a"] - conf["b"]).mean()) < 0
        row = {"station": station, "metric": metric, "variant": variant,
               "crps_sel": float(gate["a"].mean()), "mos_crps_sel": float(gate["b"].mean()),
               "gate_ci_hi": ci_hi, "sel_pass": int(sel_pass),
               "crps_rep": float(conf["a"].mean()) if rep_days else None,
               "mos_crps_rep": float(conf["b"].mean()) if rep_days else None,
               "rep_days": rep_days, "gate_pass": int(sel_pass and rep_ok), "decided_at": now_iso()}
        out.append(row)
    with conn:
        conn.executemany("INSERT OR REPLACE INTO model_choice VALUES (:station,:metric,:variant,:crps_sel,"
                         ":mos_crps_sel,:gate_ci_hi,:sel_pass,:crps_rep,:mos_crps_rep,:rep_days,:gate_pass,"
                         ":decided_at)", out)
        conn.execute("INSERT OR REPLACE INTO eval_log VALUES (?,?,?,?)",
                     (now_iso(), metric, code_fingerprint(), int(report_scored)))
    return out


def canonical_events(conn) -> set[str]:
    """One event per (station, metric, event_date): most markets, ties -> smallest event_ticker.
    (The 2a archive has KMDW 2024-02-28 listed under two events; spec §4.1.)"""
    best: dict[tuple, tuple[int, str]] = {}
    for st, metric, d, ev, n in conn.execute(
            "SELECT station, metric, event_date, event_ticker, COUNT(*) FROM kalshi_markets "
            "GROUP BY station, metric, event_date, event_ticker"):
        key = (st, metric, d)
        if key not in best or (-n, ev) < (-best[key][0], best[key][1]):
            best[key] = (n, ev)
    return {ev for _, ev in best.values()}


def market_brier(conn, metric: str, period: str | None = None) -> dict:
    """period None = all dates; 'selection' = before SELECTION_END; 'report' = on/after."""
    choice = {s: v for s, v in conn.execute("SELECT station, variant FROM model_choice WHERE metric=?", (metric,))}
    df = scored(conn, metric)
    df = df[df.apply(lambda r: choice.get(r.station) == r.variant, axis=1)] if len(df) else df
    if period == "selection":
        df = df[df["event_date"] < SELECTION_END]
    elif period == "report":
        df = df[df["event_date"] >= SELECTION_END]
    canon = canonical_events(conn)
    markets = conn.execute("SELECT ticker, event_ticker, station, event_date, lo, hi FROM kalshi_markets "
                           "WHERE metric=? AND NOT (lo IS NULL AND hi IS NULL) AND result IN ('yes','no')",
                           (metric,)).fetchall()
    by_event: dict[tuple[str, str], list] = {}
    for m in markets:
        if m[1] in canon:
            by_event.setdefault((m[2], m[3]), []).append(m)
    diffs: dict[str, dict[str, list[float]]] = {}
    for r in df.itertuples(index=False):
        t = datetime.fromisoformat(r.asof_utc)
        lo_iso = (t - timedelta(hours=MARKET_STALE_H)).isoformat()
        for ticker, event_ticker, _, _, lo, hi in by_event.get((r.station, r.event_date), []):
            c = conn.execute("SELECT yes_bid, yes_ask FROM kalshi_candles WHERE ticker=? AND end_utc>? "
                             "AND end_utc<=? AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL "
                             "ORDER BY end_utc DESC LIMIT 1", (ticker, lo_iso, r.asof_utc)).fetchone()
            if c is None:
                continue
            won = int((lo is None or r.y >= lo) and (hi is None or r.y <= hi))
            p_model = pmf_mass(r.pmf, lo, hi) or 0.0
            mid = (c[0] + c[1]) / 2
            diffs.setdefault(r.station, {}).setdefault(event_ticker, []).append(
                brier(p_model, won) - brier(mid, won))
    out = {}
    for station, clusters in diffs.items():
        mean, lo, hi = cluster_bootstrap(clusters)
        out[station] = (mean, lo, hi, sum(len(v) for v in clusters.values()))
    return out


def _freeze_warning(conn) -> list[str]:
    first = conn.execute("SELECT run_at, fingerprint FROM eval_log WHERE report_scored=1 "
                         "ORDER BY run_at LIMIT 1").fetchone()
    if first is None or first[1] == code_fingerprint():
        return []
    return [f"> **WARNING:** the report period was first scored on {first[0]} under different modelling "
            "code. It is no longer an untouched test; only the sandbox forward test counts.", ""]


def write_report(conn, path: Path) -> Path:
    lines = ["# citycast evaluation", "", *_freeze_warning(conn),
             f"Selection period: event dates before {SELECTION_END}. Report period: on/after. "
             "RPS = discrete CRPS (lower is better). Gate = selection CI hi < 0 AND report-period mean "
             f"below mosnorm over >= {REPORT_MIN_DAYS} days. "
             "Market Brier diff < 0 means the model beat the market mid.", ""]
    for metric in ("high", "low"):
        lines += [f"## {metric}", "",
                  "| station | variant | RPS (sel) | mosnorm RPS (sel) | gate CI hi | RPS (rep) | "
                  "mosnorm RPS (rep) | rep days | gate | market Δ sel [CI] (n) | market Δ report [CI] (n) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        sel_m, rep_m = market_brier(conn, metric, "selection"), market_brier(conn, metric, "report")
        for row in conn.execute("SELECT station, variant, crps_sel, mos_crps_sel, gate_ci_hi, crps_rep, "
                                "mos_crps_rep, rep_days, sel_pass, gate_pass FROM model_choice WHERE metric=? "
                                "ORDER BY station", (metric,)):
            def fmt(x):
                return f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}] ({x[3]})" if x else "no prices"
            def num(x):
                return "—" if x is None else f"{x:.3f}"
            verdict = "PASS" if row[9] else ("fail (report)" if row[8] else "fail")
            lines.append(f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:+.3f} | "
                         f"{num(row[5])} | {num(row[6])} | {row[7]} | {verdict} | "
                         f"{fmt(sel_m.get(row[0]))} | {fmt(rep_m.get(row[0]))} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
```

- [ ] **Step 4: Run to verify they pass** — 7 passed.

- [ ] **Step 5: Full suite** — all pass (update `tests/test_citycast_asof.py` only if it asserts the exact table list; it checks a subset).

---

### Task 8: Final training, live prediction and the CLI commands

**Files:**
- Create: `src/citycast/live.py`
- Modify: `src/citycast/cli.py` (add subcommands)
- Test: `tests/test_citycast_live.py`

**Interfaces:**
- Consumes: `dataset.load`, `QuantileModel`, `choose_k`, `_prev` (walkforward), `FeatureContext`, `pmf_from_quantiles`, `run_chunks` + source chunk builders.
- Produces: `MODEL_DIR = Path("data/citycast/models")`; `train_final(conn, df, metric, stations: tuple[str, ...], now: date) -> Path` (pickle bundle `{"metric", "trained_on", "shared": QuantileModel, "city": {icao: QuantileModel}, "choice": {icao: (variant, k, gate_pass)}}`); `predict(conn, ctx, metric, now: datetime) -> int` (rows written to `live_predictions` for gated stations, event days = station-local today and tomorrow); CLI: `citycast dataset --metric M [--since]`, `citycast walkforward --metric M`, `citycast evaluate`, `citycast train --metric M`, `citycast predict [--no-refresh]`.

`predict` refreshes first (unless `--no-refresh`): runs the `cli`, `mos` and `obs` chunk builders with `since = today − 7 days` (their current chunks always rerun).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citycast_live.py
import json
from contextlib import closing
from datetime import date, datetime, timezone

from citycast import db, live
from citycast.features import FeatureContext
from test_citycast_features import NYC, _seed
from test_citycast_walkforward import _frame


def test_train_then_predict_writes_live_rows_for_gated_station(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "models")
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            c.execute("INSERT INTO model_choice VALUES ('KNYC','high','shared',1.0,1.2,-0.1,1,1.0,1.2,90,1,'x')")
        df = _frame()
        path = live.train_final(c, df, "high", ("KLAX", "KNYC"), date(2023, 1, 1))
        assert path.exists()
        ctx = FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")
        now = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        n = live.predict(c, ctx, "high", now)
        row = c.execute("SELECT event_date, issued_utc, variant, pmf_json FROM live_predictions").fetchone()
        assert n == 1 and row[0] == "2024-01-05" and row[1] == now.isoformat() and row[2] == "shared"
        assert abs(sum(json.loads(row[3]).values()) - 1) < 1e-6
```

(At 18:00 EST on Jan 4 only the Jan 5 event has MOS rows in the seed, so exactly one row is written; Jan 4 has no anchor.)

- [ ] **Step 2: Run to verify it fails** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/live.py
"""Final models and live predictions (spec §3.9)."""
import json
import math
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from citycast import db
from citycast.qmodel import QUANTILES, QuantileModel, pmf_from_quantiles
from citycast.walkforward import MIN_CITY_ROWS, _prev, choose_k

MODEL_DIR = Path("data/citycast/models")


def _bundle_path(metric: str) -> Path:
    return MODEL_DIR / f"{metric}.pkl"


def train_final(conn, df: pd.DataFrame, metric: str, stations: tuple[str, ...], now: date) -> Path:
    known = df[(df["y"].notna()) & (df["metric"] == metric)] if "metric" in df else df[df["y"].notna()]
    truth = {(r.station, r.event_date): int(r.y) for r in known[["station", "event_date", "y"]]
             .drop_duplicates(["station", "event_date"]).itertuples(index=False)}
    choice = {s: (v, g) for s, v, g in conn.execute(
        "SELECT station, variant, gate_pass FROM model_choice WHERE metric=?", (metric,))}
    bundle = {"metric": metric, "trained_on": now.isoformat(),
              "shared": QuantileModel("shared", stations).fit(known, now), "city": {}, "choice": {}}
    for s, (variant, gate) in choice.items():
        if variant == "city":
            tr = known[known["station"] == s]
            if len(tr) < MIN_CITY_ROWS:
                variant = "shared"
            else:
                bundle["city"][s] = QuantileModel("city", stations).fit(tr, now)
        k = choose_k(_prev(conn, s, metric, variant, now, truth), metric)
        bundle["choice"][s] = (variant, k, int(gate))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _bundle_path(metric)
    path.write_bytes(pickle.dumps(bundle))
    return path


def predict(conn, ctx, metric: str, now: datetime) -> int:
    bundle = pickle.loads(_bundle_path(metric).read_bytes())
    rows = []
    for icao, (variant, k, gate) in sorted(bundle["choice"].items()):
        if not gate or icao not in ctx.stations:
            continue
        st = ctx.stations[icao]
        today = now.astimezone(ZoneInfo(st.tz)).date()
        for day in (today, today + timedelta(days=1)):
            r = ctx.row(icao, metric, day, now)
            if r is None:
                continue
            frame = pd.DataFrame([r])
            model = bundle["city"][icao] if variant == "city" else bundle["shared"]
            q = dict(zip(QUANTILES, (float(v) for v in model.predict_y_quantiles(frame)[0])))
            pmf = pmf_from_quantiles(q, k, r["run_ext"], metric)
            rows.append({"station": icao, "metric": metric, "event_date": day.isoformat(),
                         "issued_utc": now.isoformat(), "variant": variant, "anchor": r["anchor"],
                         "run_ext": None if math.isnan(r["run_ext"]) else r["run_ext"],
                         "quantiles_json": json.dumps(q), "pmf_json": json.dumps(pmf), "k": k})
    with conn:
        return db.upsert(conn, "live_predictions", rows)
```

Add to `src/citycast/cli.py` (new imports at the top; new subparsers after `coverage`; new branches before the final `return 2`):

```python
# imports
from datetime import datetime, timedelta, timezone
from pathlib import Path

from citycast import dataset, evaluate, live, walkforward
from citycast.features import FeatureContext
```

```python
    # subparsers (inside main, after the coverage parser)
    ds = sub.add_parser("dataset")
    ds.add_argument("--metric", choices=("high", "low"), required=True)
    ds.add_argument("--since", default="2022-01-01")
    wf = sub.add_parser("walkforward")
    wf.add_argument("--metric", choices=("high", "low"), required=True)
    sub.add_parser("evaluate")
    tr = sub.add_parser("train")
    tr.add_argument("--metric", choices=("high", "low"), required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--no-refresh", action="store_true")
```

```python
        # branches (inside the `with closing(...)` block)
        if args.cmd == "dataset":
            ctx = FeatureContext(conn, stations, args.db)
            df = dataset.build(ctx, args.metric, date.fromisoformat(args.since), date.today())
            print(json.dumps({"rows": len(df), "path": str(dataset.save(df, args.metric))}))
            return 0
        if args.cmd == "walkforward":
            n = walkforward.run(conn, dataset.load(args.metric), args.metric, tuple(sorted(stations)), utcnow_iso)
            print(json.dumps({"written": n}))
            return 0
        if args.cmd == "evaluate":
            for metric in ("high", "low"):
                evaluate.select_and_gate(conn, metric, utcnow_iso)
            path = evaluate.write_report(conn, Path(f"docs/reports/citycast-evaluation-{date.today()}.md"))
            print(json.dumps({"report": str(path)}))
            return 0
        if args.cmd == "train":
            path = live.train_final(conn, dataset.load(args.metric), args.metric, tuple(sorted(stations)), date.today())
            print(json.dumps({"bundle": str(path)}))
            return 0
        if args.cmd == "predict":
            if not args.no_refresh:
                paced, today = Paced(), date.today()
                for source in ("cli", "mos", "obs"):
                    run_chunks(conn, _chunks_for(source, paced, conn, stations, today - timedelta(days=7), today),
                               now_iso=utcnow_iso, progress=lambda m: None)
            ctx = FeatureContext(conn, stations, args.db, mos_since=date.today() - timedelta(days=3))
            now = datetime.now(timezone.utc).replace(microsecond=0)
            written = {m: live.predict(conn, ctx, m, now) for m in ("high", "low")}
            print(json.dumps({"issued_utc": now.isoformat(), **written}))
            return 0
```

- [ ] **Step 4: Run to verify it passes** — `.venv/bin/python -m pytest -q tests/test_citycast_live.py tests/test_citycast_cli.py`, all pass.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 9: `citycast` sandbox bot

**Files:**
- Modify: `src/sandbox/bots.py` (add `CitycastBot`), `src/sandbox/cycle.py` (`ACCOUNTS`), `src/sandbox/cli.py` (`--history-db`, bot list)
- Test: `tests/test_sandbox_citycast_bot.py`; update tests that hard-code the four accounts

**Interfaces:**
- Consumes: `live_predictions`, `model_choice` in `data/history.db` (read-only).
- Produces: `CitycastBot(history_db: str)`, `name = "citycast"`, `pre_day_only = False`, `MAX_AGE_S = 7200`; `forecast_json = {"source": "citycast", "variant": …, "issued_utc": …}`. Account row `("citycast", "bot", "24 cities: citycast model, gated pairs only")` inserted before `manual`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox_citycast_bot.py
import json
import sqlite3
from contextlib import closing

from citycast import db as hdb
from sandbox.bots import CitycastBot
from sandbox_helpers import make_market

ISSUED = "2026-10-08T00:00:00+00:00"


def _history(tmp_path, gate=1, issued=ISSUED):
    p = tmp_path / "history.db"
    with closing(hdb.connect(p)) as c, c:
        c.execute("INSERT INTO model_choice VALUES ('KNYC','high','shared',1,1,-0.1,?,'x')", (gate,))
        c.execute("INSERT INTO live_predictions VALUES ('KNYC','high','2026-10-08',?,'shared',70,NULL,'{}',?,1.0)",
                  (issued, json.dumps({"69": 0.2, "70": 0.5, "71": 0.3})))
    return str(p)


def _markets():
    return [make_market("KXHIGHNY-26OCT08-B69.5", lo=69, hi=70, station="KNYC", metric="high",
                        event_date="2026-10-08"),
            make_market("KXHIGHNY-26OCT08-T70", lo=71, hi=None, station="KNYC", metric="high",
                        event_date="2026-10-08")]


class Snap:
    taken_at = "2026-10-08T01:00:00+00:00"


def test_reads_latest_gated_prediction(tmp_path):
    probs, fj = CitycastBot(_history(tmp_path)).forecast_for_event(_markets(), Snap())
    assert abs(probs["KXHIGHNY-26OCT08-B69.5"] - 0.7) < 1e-9 and fj["source"] == "citycast"


def test_ungated_stale_or_missing_db_gives_nothing(tmp_path):
    assert CitycastBot(_history(tmp_path, gate=0)).forecast_for_event(_markets(), Snap()) == (None, None)
    old = "2026-10-07T20:00:00+00:00"                          # 5 h before the snapshot
    assert CitycastBot(_history(tmp_path / "o", issued=old)).forecast_for_event(_markets(), Snap()) == (None, None)
    assert CitycastBot(str(tmp_path / "none.db")).forecast_for_event(_markets(), Snap()) == (None, None)


def test_never_writes_history_db(tmp_path):
    p = _history(tmp_path)
    before = open(p, "rb").read()
    CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert open(p, "rb").read() == before
```

Before writing these, read `tests/sandbox_helpers.py` and use `make_market`'s real signature; adjust the keyword names above to match it (same values). If `_history(tmp_path / "o", …)` needs the directory, `hdb.connect` creates parents.

- [ ] **Step 2: Run to verify they fail** — FAIL (`CitycastBot` missing).

- [ ] **Step 3: Implement** — add to `src/sandbox/bots.py` after `HighcastBot`:

```python
class CitycastBot(ForecastBot):
    """citycast's 24-city model (spec 2026-10-07 §3.9). Reads data/history.db read-only and
    uses only station/metrics whose model passed the gate; skips predictions older than 2 h."""
    name = "citycast"
    MAX_AGE_S = 7200

    def __init__(self, history_db: str):
        self.history_db = history_db

    def _latest(self, station, metric, event_date, at_or_before):
        path = Path(self.history_db)
        if not path.exists():
            return None
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as c:
                return c.execute(
                    """SELECT lp.pmf_json, lp.variant, lp.issued_utc FROM live_predictions lp
                       JOIN model_choice mc ON mc.station=lp.station AND mc.metric=lp.metric
                       WHERE mc.gate_pass=1 AND lp.station=? AND lp.metric=? AND lp.event_date=?
                         AND lp.issued_utc<=? ORDER BY lp.issued_utc DESC LIMIT 1""",
                    (station, metric, event_date, at_or_before)).fetchone()
        except sqlite3.Error as e:
            log.warning("citycast read failed for %s %s %s: %r", station, metric, event_date, e)
            return None

    def forecast_for_event(self, markets, snapshot):
        first = markets[0]
        if first.station is None or first.metric is None:
            return None, None
        row = self._latest(first.station, first.metric, first.event_date, snapshot.taken_at)
        if row is None:
            return None, None
        age = (datetime.fromisoformat(snapshot.taken_at) - datetime.fromisoformat(row[2])).total_seconds()
        if age > self.MAX_AGE_S:
            return None, None
        try:
            pmf = {int(k): float(v) for k, v in json.loads(row[0]).items()}
        except (ValueError, TypeError, AttributeError) as e:
            log.warning("citycast pmf unreadable for %s: %r", first.event_ticker, e)
            return None, None
        raw = {m.ticker: pmf_mass(pmf, m.lo, m.hi) or 0.0 for m in markets}
        total = sum(raw.values())
        if total <= 0:
            return None, None
        return ({t: v / total for t, v in raw.items()},
                {"source": "citycast", "variant": row[1], "issued_utc": row[2]})

    def pmf_for_event(self, markets, snapshot):
        return self.forecast_for_event(markets, snapshot)[0]
```

In `src/sandbox/cycle.py`, `ACCOUNTS` becomes:

```python
ACCOUNTS: tuple[tuple[str, str, str], ...] = (
    ("favorite", "bot", "control: follow the crowd after fees"),
    ("mos", "bot", "uncalibrated: MOS with a fixed 3.0°F spread"),
    ("highcast", "bot", "KLAX/KMIA only"),
    ("citycast", "bot", "24 cities: citycast model, gated pairs only"),
    ("manual", "manual", "Julian's hand bets"),
)
```

In `src/sandbox/cli.py`: add `p.add_argument("--history-db", default="data/history.db")` next to `--highcast-db`, import `CitycastBot`, and set the bot list to
`bots = [FavoriteBot(), HighcastBot(args.highcast_db), CitycastBot(args.history_db), MosBot(IemMos())]`.

- [ ] **Step 4: Fix tests that hard-code four accounts**

Run: `grep -rn "manual" tests/test_sandbox_*.py | grep -E "ACCOUNTS|4|four|12" ` and `.venv/bin/python -m pytest -q tests/test_sandbox_*.py`. For each failure caused only by the new fifth account (e.g. 12 → 15 report lines, account lists), update the expected value to include `citycast`. Do not change any assertion about gates, money or the unlock rule.

- [ ] **Step 5: Run the bot tests, the sandbox suite with strict warnings, then the full suite**

Run: `.venv/bin/python -m pytest -q tests/test_sandbox_citycast_bot.py && .venv/bin/python -m pytest -q tests/test_sandbox_*.py -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning && .venv/bin/python -m pytest -q`
Expected: all pass; the no-real-money import test still passes (citycast does not import `highcast.kalshi_client`).

---

### Task 10: Run it for real (controller task)

- [ ] **Step 0: Freeze** — confirm `PUB_LAG_H` has been set from the Task 0 report (observed max + 0.5 h) and committed; after Step 3 the modelling code is frozen. Any later change to the fingerprinted modules spends the report period (the report will say so) and the sandbox becomes the only test.
- [ ] **Step 1: Build datasets** — `.venv/bin/citycast dataset --metric high` and `--metric low` (background; record row counts and minutes).
- [ ] **Step 2: Walk-forward** — `.venv/bin/citycast walkforward --metric high` and `--metric low` (background, hours). Record rows per variant.
- [ ] **Step 3: Evaluate** — `.venv/bin/citycast evaluate`. Read `docs/reports/citycast-evaluation-<date>.md`; for each station/metric record variant, RPS vs mosnorm, gate result and market Brier per period. Report-period numbers are read, never used to change choices.
- [ ] **Step 4: Train and predict** — `.venv/bin/citycast train --metric high`, `--metric low`, then `.venv/bin/citycast predict`; check `live_predictions` has rows only for gated pairs.
- [ ] **Step 5: Sandbox** — after Julian approves (the sandbox account list changes), run one `sandbox cycle` and confirm `citycast` candidates appear with `forecast_json.source = "citycast"`. Installing an hourly `citycast predict` launchd job needs Julian's explicit yes.
- [ ] **Step 6: Record** — results to `.superpowers/sdd/sandbox-progress.md` and the brain plan note; tell Julian plainly which cities beat MOS, and whether any beat the market.
