# KLAX Daily High Temperature Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict the daily maximum temperature at KLAX as published in the NWS Daily Climate Report, as a calibrated probability per 1°F bucket, day-ahead and live intraday, served by a local dashboard.

**Architecture:** Fetchers pull free NWS/IEM/Open-Meteo data into a single SQLite file (every row timestamped for leakage-free training). A feature builder feeds a weighted-blend baseline and LightGBM quantile models (day-ahead + intraday), calibrated and converted to 1°F bucket probabilities. FastAPI serves a local dashboard with a Manila-time information clock.

**Tech Stack:** Python 3.14 (`python3 -m venv`, pip), httpx, pandas/numpy, LightGBM, scikit-learn, FastAPI + uvicorn + jinja2, pytest. SQLite via stdlib `sqlite3`.

**Spec:** `docs/superpowers/specs/2026-07-17-lax-high-temp-predictor-design.md` — read it before starting.

## Global Constraints

- **NO GIT.** The user has explicitly forbidden creating a git repository here. Skip every commit convention from any skill; just leave files on disk.
- Station-parameterized: no `KLAX` literal outside `config/klax.yaml` and tests.
- Every stored row carries `fetched_at` (UTC ISO). Training/backtests may only use data available at prediction time (rule per task 9).
- All temperatures stored in °F unless a column is explicitly `_c`. Rounding is **half-up** (`math.floor(x + 0.5)`), never Python's banker's `round()`, whenever mimicking NWS behavior.
- Climate day = midnight-to-midnight **local standard time** (UTC−8 for KLAX year-round, even during PDT).
- Kalshi bucket conventions: "greater than X" is strict; "between A and B" inclusive. Output is a pmf over integer °F.
- The hot tail of any distribution is never clipped to historical ranges.
- Free data only; the sole optional key is a Synoptic token in config (blank = feature degrades gracefully).
- External API response shapes are verified by `-m live` smoke tests before parsers are trusted; if a real payload differs from the fixture in a task, adapt the parser but keep the function signature.
- Network tests are marked `@pytest.mark.live` and excluded by default (`addopts = -m 'not live'`). Run them explicitly with `pytest -m live`.
- Working directory for all commands: the project root (`/Users/julian/Desktop/personal/weather temp predictor`). Activate the venv first: `source .venv/bin/activate`.

---

### Task 1: Project scaffold, config, climate-day time math

**Files:**
- Create: `pyproject.toml`, `config/klax.yaml`, `src/highcast/__init__.py`, `src/highcast/config.py`, `src/highcast/timeutil.py`
- Test: `tests/test_config.py`, `tests/test_timeutil.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `load_config(path="config/klax.yaml") -> AppConfig` where `AppConfig` has `.station: StationConfig` (fields `key, name, icao, iem_id, iem_network, wfo, lat, lon, tz, std_utc_offset, nws_zones: list[str]`), `.display_tz: str`, `.synoptic_token: str`, `.db_path: Path`. `climate_day_bounds(day: date, std_utc_offset: int) -> tuple[datetime, datetime]` (UTC-aware). `utcnow_iso() -> str`. `round_half_up(x: float) -> int`. `c_to_f(c: float) -> float`.

- [ ] **Step 1: Create venv and project config**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Create `pyproject.toml`:

```toml
[project]
name = "highcast"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "httpx>=0.27",
  "pandas>=2.2",
  "numpy>=1.26",
  "lightgbm>=4.3",
  "scikit-learn>=1.4",
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "jinja2>=3.1",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
highcast = "highcast.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/highcast"]

[tool.pytest.ini_options]
markers = ["live: hits real network endpoints"]
addopts = "-m 'not live'"
testpaths = ["tests"]
```

Install: `pip install -e '.[dev]'`
Expected: succeeds. If lightgbm fails on macOS with a libomp error: `brew install libomp`, retry.

- [ ] **Step 2: Write failing tests**

`tests/test_config.py`:

```python
from highcast.config import load_config

def test_load_klax_config():
    cfg = load_config("config/klax.yaml")
    assert cfg.station.icao == "KLAX"
    assert cfg.station.iem_id == "LAX"
    assert cfg.station.std_utc_offset == -8
    assert cfg.station.lat == 33.9425
    assert cfg.display_tz == "Asia/Manila"
    assert str(cfg.db_path).endswith("highcast.db")
    assert "CAZ041" in cfg.station.nws_zones
```

`tests/test_timeutil.py`:

```python
from datetime import date, datetime, timezone
from highcast.timeutil import climate_day_bounds, round_half_up, c_to_f

def test_climate_day_is_local_standard_time_even_in_summer():
    # 2026-07-17 at KLAX (std offset -8): climate day is 08:00 UTC to 08:00 UTC next day
    start, end = climate_day_bounds(date(2026, 7, 17), -8)
    assert start == datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 18, 8, 0, tzinfo=timezone.utc)

def test_round_half_up_is_not_bankers():
    assert round_half_up(0.5) == 1      # Python round() would give 0
    assert round_half_up(1.5) == 2
    assert round_half_up(-0.5) == 0
    assert round_half_up(71.4) == 71

def test_c_to_f():
    assert c_to_f(22.0) == 71.6
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_timeutil.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'highcast'` or import errors.

- [ ] **Step 4: Implement**

`config/klax.yaml`:

```yaml
station:
  key: klax
  name: Los Angeles Intl Airport
  icao: KLAX
  iem_id: LAX
  iem_network: CA_ASOS
  wfo: LOX
  lat: 33.9425
  lon: -118.4081
  tz: America/Los_Angeles
  std_utc_offset: -8
  nws_zones: ["CAZ041"]
display_tz: Asia/Manila
synoptic_token: ""
db_path: data/highcast.db
```

`src/highcast/__init__.py`: empty file.

`src/highcast/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StationConfig:
    key: str
    name: str
    icao: str
    iem_id: str
    iem_network: str
    wfo: str
    lat: float
    lon: float
    tz: str
    std_utc_offset: int
    nws_zones: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    station: StationConfig
    display_tz: str
    synoptic_token: str
    db_path: Path


def load_config(path: str | Path = "config/klax.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        station=StationConfig(**raw["station"]),
        display_tz=raw["display_tz"],
        synoptic_token=raw.get("synoptic_token") or "",
        db_path=Path(raw["db_path"]),
    )
```

`src/highcast/timeutil.py`:

```python
import math
from datetime import date, datetime, timedelta, timezone


def climate_day_bounds(day: date, std_utc_offset: int) -> tuple[datetime, datetime]:
    """NWS climate day: midnight-to-midnight LOCAL STANDARD time (ignores DST)."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(
        hours=std_utc_offset
    )
    return start, start + timedelta(days=1)


def round_half_up(x: float) -> int:
    """NWS rounds .5 up; Python's round() is banker's rounding. Always use this."""
    return math.floor(x + 0.5)


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_timeutil.py -v`
Expected: 5 passed.

---

### Task 2: SQLite layer

**Files:**
- Create: `src/highcast/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `AppConfig.db_path`
- Produces: `connect(db_path) -> sqlite3.Connection` (creates parent dirs, applies schema, `row_factory=sqlite3.Row`); `upsert(conn, table, rows: list[dict]) -> int`; `get_meta(conn, key) -> str | None`; `set_meta(conn, key, value)`. Tables: `cli_reports(station, date PK, high_f, low_f, raw, fetched_at)`, `obs(station, valid_utc, source PK, tmpf, tmpf_precise, dwpf, sknt, drct, skyc, six_max_c, raw, fetched_at)`, `forecasts(station, target_date, model, run_date PK, max_temp_f, fetched_at)`, `alerts(station, event, onset_utc PK, expires_utc, fetched_at)`, `predictions(station, target_date, issued_at_utc, kind PK, point_f, quantiles, buckets)`, `meta(k PK, v)`.

- [ ] **Step 1: Write failing test**

`tests/test_db.py`:

```python
from highcast import db

def test_schema_and_upsert(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rows = [
        {"station": "KLAX", "date": "2026-07-16", "high_f": 84, "low_f": 64,
         "raw": "{}", "fetched_at": "2026-07-17T00:00:00+00:00"},
    ]
    assert db.upsert(conn, "cli_reports", rows) == 1
    # upsert is idempotent and updates in place
    rows[0]["high_f"] = 85
    db.upsert(conn, "cli_reports", rows)
    got = conn.execute("SELECT high_f FROM cli_reports").fetchall()
    assert len(got) == 1 and got[0]["high_f"] == 85

def test_meta_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    assert db.get_meta(conn, "grid_url") is None
    db.set_meta(conn, "grid_url", "https://x")
    assert db.get_meta(conn, "grid_url") == "https://x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `AttributeError`/`ImportError` (module doesn't exist).

- [ ] **Step 3: Implement**

`src/highcast/db.py`:

```python
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cli_reports (
  station TEXT NOT NULL, date TEXT NOT NULL, high_f INTEGER, low_f INTEGER,
  raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, date));
CREATE TABLE IF NOT EXISTS obs (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, source TEXT NOT NULL,
  tmpf REAL, tmpf_precise REAL, dwpf REAL, sknt REAL, drct REAL, skyc TEXT,
  six_max_c REAL, raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, valid_utc, source));
CREATE TABLE IF NOT EXISTS forecasts (
  station TEXT NOT NULL, target_date TEXT NOT NULL, model TEXT NOT NULL,
  run_date TEXT NOT NULL, max_temp_f REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, target_date, model, run_date));
CREATE TABLE IF NOT EXISTS alerts (
  station TEXT NOT NULL, event TEXT NOT NULL, onset_utc TEXT NOT NULL,
  expires_utc TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, event, onset_utc));
CREATE TABLE IF NOT EXISTS predictions (
  station TEXT NOT NULL, target_date TEXT NOT NULL, issued_at_utc TEXT NOT NULL,
  kind TEXT NOT NULL, point_f REAL, quantiles TEXT, buckets TEXT,
  PRIMARY KEY (station, target_date, issued_at_utc, kind));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS idx_obs_valid ON obs (station, valid_utc);
CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts (station, target_date);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})"
    )
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    conn.commit()
    return len(rows)


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    return row["v"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, value))
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v` — Expected: 2 passed.
Then run the whole suite: `pytest -q` — Expected: all pass.

---

### Task 3: HTTP helper + CLI-report fetcher (settlement ground truth)

**Files:**
- Create: `src/highcast/http.py`, `src/highcast/fetchers/__init__.py`, `src/highcast/fetchers/cli_reports.py`
- Test: `tests/test_cli_reports.py`

**Interfaces:**
- Consumes: `db.upsert`, `timeutil.utcnow_iso`, `AppConfig`
- Produces: `http.get_json(url, params=None) -> dict|list`, `http.get_text(url, params=None) -> str` (3 retries, 30s timeout, custom User-Agent). `cli_reports.fetch_year(cfg, year: int) -> list[dict]` (rows matching `cli_reports` table), `cli_reports.parse_rows(payload, icao, fetched_at) -> list[dict]`, `cli_reports.store(conn, rows) -> int`.

- [ ] **Step 1: Write failing parser test**

`tests/test_cli_reports.py`:

```python
from highcast.fetchers import cli_reports

SAMPLE = {
    "data": [
        {"station": "KLAX", "valid": "2025-07-16", "high": 84, "low": 64,
         "precip": 0.0},
        {"station": "KLAX", "valid": "2025-07-17", "high": None, "low": 63},
    ]
}

def test_parse_rows_skips_missing_high_and_maps_fields():
    rows = cli_reports.parse_rows(SAMPLE, "KLAX", "2026-07-17T00:00:00+00:00")
    assert len(rows) == 1
    r = rows[0]
    assert r["station"] == "KLAX"
    assert r["date"] == "2025-07-16"
    assert r["high_f"] == 84
    assert r["low_f"] == 64
    assert r["fetched_at"] == "2026-07-17T00:00:00+00:00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_reports.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/highcast/http.py`:

```python
import time

import httpx

HEADERS = {"User-Agent": "highcast personal weather project (contact: local)"}


def _get(url: str, params: dict | None, tries: int = 3) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = httpx.get(url, params=params, timeout=30, headers=HEADERS,
                          follow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retry any transport/status error
            last = e
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def get_json(url: str, params: dict | None = None):
    return _get(url, params).json()


def get_text(url: str, params: dict | None = None) -> str:
    return _get(url, params).text
```

`src/highcast/fetchers/__init__.py`: empty file.

`src/highcast/fetchers/cli_reports.py`:

```python
import json

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import utcnow_iso

IEM_CLI = "https://mesonet.agron.iastate.edu/api/1/cli.json"


def parse_rows(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    rows = []
    for item in payload.get("data", []):
        if item.get("high") is None:
            continue
        rows.append({
            "station": icao,
            "date": str(item["valid"])[:10],
            "high_f": int(item["high"]),
            "low_f": int(item["low"]) if item.get("low") is not None else None,
            "raw": json.dumps(item),
            "fetched_at": fetched_at,
        })
    return rows


def fetch_year(cfg: AppConfig, year: int) -> list[dict]:
    payload = get_json(IEM_CLI, {"station": cfg.station.icao, "year": year})
    return parse_rows(payload, cfg.station.icao, utcnow_iso())


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "cli_reports", rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_reports.py -v` — Expected: PASS.

- [ ] **Step 5: Add live smoke test and run it**

Append to `tests/test_cli_reports.py`:

```python
import pytest
from highcast.config import load_config

@pytest.mark.live
def test_live_fetch_2025_has_a_full_year():
    cfg = load_config()
    rows = cli_reports.fetch_year(cfg, 2025)
    assert len(rows) >= 300, f"only {len(rows)} rows - check endpoint/params"
    highs = [r["high_f"] for r in rows]
    assert all(40 <= h <= 120 for h in highs)
    print("latest:", rows[-1]["date"], rows[-1]["high_f"])
```

Run: `pytest tests/test_cli_reports.py -m live -v -s`
Expected: PASS, printing a plausible date + high. **If the real payload shape differs from `SAMPLE`** (field names, nesting): update `parse_rows` and the fixture to match reality, keep the signature, re-run both tests.

---

### Task 4: METAR fetcher + parser (precision anchor)

**Files:**
- Create: `src/highcast/fetchers/metar.py`
- Test: `tests/test_metar.py`

**Interfaces:**
- Consumes: `http.get_json`, `http.get_text`, `db.upsert`, `timeutil`
- Produces: `parse_metar(raw: str, valid_utc: str, icao: str, fetched_at: str) -> dict` (row for `obs` table, `source="metar"`; `tmpf_precise` from the T-group at 0.1°C resolution, `tmpf` from the coarse tt/dd group, `six_max_c` from the 6-hr max `1sTTT` remark when present). `fetch_recent(cfg, hours=36) -> list[dict]` (aviationweather.gov). `fetch_archive(cfg, start: date, end: date) -> list[dict]` (IEM). `store(conn, rows) -> int`.

- [ ] **Step 1: Write failing parser tests**

`tests/test_metar.py`:

```python
from highcast.fetchers.metar import parse_metar

RAW = ("KLAX 171753Z 25008KT 10SM FEW015 24/17 A2992 "
       "RMK AO2 SLP132 T02390172 10244 20161 58008")

def test_parse_full_metar():
    row = parse_metar(RAW, "2026-07-17T17:53:00+00:00", "KLAX", "2026-07-17T18:00:00+00:00")
    assert row["source"] == "metar"
    assert row["tmpf"] == 75.2                    # 24 C coarse group
    assert abs(row["tmpf_precise"] - 75.02) < 0.01  # T02390172 -> 23.9 C
    assert abs(row["dwpf"] - 63.0) < 0.1          # 17.2 C from T-group
    assert row["sknt"] == 8.0 and row["drct"] == 250.0
    assert row["skyc"] == "FEW"
    assert row["six_max_c"] == 24.4               # 10244 -> +24.4 C

def test_parse_negative_t_group_and_no_remarks():
    raw = "KLAX 170353Z 00000KT 10SM CLR M01/M03 A3010 RMK AO2 T10061028"
    row = parse_metar(raw, "2026-01-17T03:53:00+00:00", "KLAX", "x")
    assert abs(row["tmpf_precise"] - 30.92) < 0.01  # -0.6 C
    assert row["six_max_c"] is None
    assert row["skyc"] == "CLR"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metar.py -v` — Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/highcast/fetchers/metar.py`:

```python
import re
from datetime import date, timedelta

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json, get_text
from highcast.timeutil import c_to_f, utcnow_iso

AVWX = "https://aviationweather.gov/api/data/metar"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_SIX_MAX = re.compile(r"\b1([01])(\d{3})\b")
_TT_DD = re.compile(r"\b(M?\d{2})/(M?\d{2})\b")
_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b")
_SKY = re.compile(r"\b(FEW|SCT|BKN|OVC|CLR|SKC)\d*\b")


def _signed(sign: str, digits: str) -> float:
    v = int(digits) / 10.0
    return -v if sign == "1" else v


def _int_c(tok: str) -> float:
    return -int(tok[1:]) if tok.startswith("M") else int(tok)


def parse_metar(raw: str, valid_utc: str, icao: str, fetched_at: str) -> dict:
    tmpf = tmpf_precise = dwpf = sknt = drct = six_max_c = None
    skyc = None
    if m := _TT_DD.search(raw):
        tmpf = c_to_f(_int_c(m.group(1)))
        dwpf = c_to_f(_int_c(m.group(2)))
    if m := _T_GROUP.search(raw):
        tmpf_precise = c_to_f(_signed(m.group(1), m.group(2)))
        dwpf = c_to_f(_signed(m.group(3), m.group(4)))
    if m := _WIND.search(raw):
        drct = None if m.group(1) == "VRB" else float(m.group(1))
        sknt = float(m.group(2))
    # 6-hr max lives in remarks only; don't match the wind group by accident
    rmk = raw.split(" RMK ", 1)
    if len(rmk) == 2 and (m := _SIX_MAX.search(rmk[1])):
        six_max_c = _signed(m.group(1), m.group(2))
    sky_tokens = _SKY.findall(raw)
    if sky_tokens:
        skyc = sky_tokens[-1]
    return {
        "station": icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": tmpf, "tmpf_precise": tmpf_precise, "dwpf": dwpf,
        "sknt": sknt, "drct": drct, "skyc": skyc, "six_max_c": six_max_c,
        "raw": raw, "fetched_at": fetched_at,
    }


def fetch_recent(cfg: AppConfig, hours: int = 36) -> list[dict]:
    payload = get_json(AVWX, {"ids": cfg.station.icao, "format": "json",
                              "hours": hours})
    now = utcnow_iso()
    rows = []
    for ob in payload:
        raw = ob.get("rawOb") or ""
        valid = ob.get("reportTime") or ob.get("obsTime")
        if raw and valid:
            rows.append(parse_metar(raw, str(valid), cfg.station.icao, now))
    return rows


def fetch_archive(cfg: AppConfig, start: date, end: date) -> list[dict]:
    """IEM METAR archive, chunked by month to keep responses small."""
    now = utcnow_iso()
    rows = []
    cur = start
    while cur <= end:
        nxt = min(end, (cur.replace(day=1) + timedelta(days=32)).replace(day=1))
        text = get_text(IEM_ASOS, {
            "station": cfg.station.iem_id, "data": "metar",
            "year1": cur.year, "month1": cur.month, "day1": cur.day,
            "year2": nxt.year, "month2": nxt.month, "day2": nxt.day,
            "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
            "report_type": [3, 4],
        })
        for line in text.splitlines()[1:]:
            parts = line.split(",", 2)
            if len(parts) == 3:
                _, valid, raw = parts
                rows.append(parse_metar(raw.strip(), valid.strip() + "+00:00",
                                        cfg.station.icao, now))
        cur = nxt if nxt > cur else end + timedelta(days=1)
    return rows


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "obs", rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metar.py -v` — Expected: 2 passed.

- [ ] **Step 5: Add live smoke tests and run them**

Append to `tests/test_metar.py`:

```python
import pytest
from datetime import date
from highcast.config import load_config
from highcast.fetchers import metar

@pytest.mark.live
def test_live_recent_metars():
    cfg = load_config()
    rows = metar.fetch_recent(cfg, hours=12)
    assert len(rows) >= 8
    assert any(r["tmpf_precise"] is not None for r in rows)

@pytest.mark.live
def test_live_archive_one_week():
    cfg = load_config()
    rows = metar.fetch_archive(cfg, date(2025, 7, 1), date(2025, 7, 7))
    assert len(rows) >= 150   # ~hourly + specials for a week
    assert any(r["six_max_c"] is not None for r in rows)
```

Run: `pytest tests/test_metar.py -m live -v -s`
Expected: PASS. Adapt field names to real payloads if needed (signatures fixed).
Then: `pytest -q` — all pass.

---

### Task 5: Quantization inversion + 5-minute feed fetcher

**Files:**
- Create: `src/highcast/quantize.py`, `src/highcast/fetchers/fivemin.py`
- Test: `tests/test_quantize.py`, `tests/test_fivemin.py`

**Interfaces:**
- Consumes: `timeutil.round_half_up`, `http.get_json`, `db.upsert`
- Produces: `quantize.candidates_from_reported_c(c_whole: int) -> list[int]` (integer °F values consistent with a whole-°C report); `quantize.candidates_from_displayed_f(f_disp: int) -> list[int]`; `quantize.interval(values: list[int]) -> tuple[int, int]`. `fivemin.fetch_recent(cfg, minutes=360) -> list[dict]` (`obs` rows, `source="syn5"`; returns `[]` when `cfg.synoptic_token` is blank). `fivemin.store(conn, rows)`.

- [ ] **Step 1: Write failing quantize tests**

`tests/test_quantize.py`:

```python
from highcast.quantize import candidates_from_reported_c, candidates_from_displayed_f, interval

def test_whole_celsius_maps_back_to_one_or_two_fahrenheit():
    # 71F -> 21.67C -> 22 ; 72F -> 22.22C -> 22 ; 73F -> 22.78C -> 23
    assert candidates_from_reported_c(22) == [71, 72]
    assert candidates_from_reported_c(23) == [73, 74]
    # every whole C from -30..45 must invert to 1 or 2 candidates
    for c in range(-30, 46):
        assert 1 <= len(candidates_from_reported_c(c)) <= 2

def test_displayed_f_roundtrips_through_c():
    # displayed F = round_half_up(C * 9/5 + 32); 22C displays as 72F
    assert candidates_from_displayed_f(72) == [71, 72]

def test_some_displayed_f_values_are_unreachable():
    # 25C displays 77F, 26C displays 79F - the feed can never show 78F.
    # Empty result is correct and documents the display-table gaps.
    assert candidates_from_displayed_f(78) == []

def test_interval():
    assert interval([71, 72]) == (71, 72)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_quantize.py -v` — Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/highcast/quantize.py`:

```python
"""Invert the NWS 5-minute feed's rounding pipeline.

Per trader reports (verified empirically by `highcast verify-quantization`,
Task 14): the public 5-min feed rounds the true F reading to a whole degree,
converts to whole C, then displays F re-derived from that whole C. A displayed
value therefore stands for 1-2 possible true whole-F values.
"""
from highcast.timeutil import round_half_up


def candidates_from_reported_c(c_whole: int) -> list[int]:
    return [f for f in range(-30, 131)
            if round_half_up((f - 32) * 5.0 / 9.0) == c_whole]


def candidates_from_displayed_f(f_disp: int) -> list[int]:
    cs = [c for c in range(-40, 60) if round_half_up(c * 9.0 / 5.0 + 32) == f_disp]
    return sorted({f for c in cs for f in candidates_from_reported_c(c)})


def interval(values: list[int]) -> tuple[int, int]:
    return (min(values), max(values))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_quantize.py -v` — Expected: 3 passed.

- [ ] **Step 5: Write failing fivemin test**

`tests/test_fivemin.py`:

```python
from highcast.fetchers import fivemin
from highcast.config import load_config, AppConfig

SAMPLE = {
    "STATION": [{
        "OBSERVATIONS": {
            "date_time": ["2026-07-17T20:00:00Z", "2026-07-17T20:05:00Z"],
            "air_temp_set_1": [22.0, 23.0],
        }
    }]
}

def test_parse_synoptic_payload():
    rows = fivemin.parse_rows(SAMPLE, "KLAX", "2026-07-17T20:06:00+00:00")
    assert len(rows) == 2
    assert rows[0]["source"] == "syn5"
    assert rows[0]["tmpf"] == 71.6           # 22 C -> displayed F equivalent
    assert rows[0]["valid_utc"].startswith("2026-07-17T20:00")

def test_blank_token_returns_empty(monkeypatch):
    cfg = load_config()
    assert cfg.synoptic_token == ""
    assert fivemin.fetch_recent(cfg) == []
```

- [ ] **Step 6: Run to verify failure, then implement**

Run: `pytest tests/test_fivemin.py -v` — Expected: FAIL.

`src/highcast/fetchers/fivemin.py`:

```python
from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import c_to_f, utcnow_iso

SYNOPTIC = "https://api.synopticdata.com/v2/stations/timeseries"


def parse_rows(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    stations = payload.get("STATION") or []
    if not stations:
        return []
    obs = stations[0].get("OBSERVATIONS", {})
    times = obs.get("date_time") or []
    temps = obs.get("air_temp_set_1") or []
    rows = []
    for t, c in zip(times, temps):
        if c is None:
            continue
        rows.append({
            "station": icao, "valid_utc": t.replace("Z", "+00:00"),
            "source": "syn5", "tmpf": round(c_to_f(float(c)), 2),
            "tmpf_precise": None, "dwpf": None, "sknt": None, "drct": None,
            "skyc": None, "six_max_c": None, "raw": None,
            "fetched_at": fetched_at,
        })
    return rows


def fetch_recent(cfg: AppConfig, minutes: int = 360) -> list[dict]:
    if not cfg.synoptic_token:
        return []
    payload = get_json(SYNOPTIC, {
        "stid": cfg.station.icao, "recent": minutes, "vars": "air_temp",
        "token": cfg.synoptic_token,
    })
    return parse_rows(payload, cfg.station.icao, utcnow_iso())


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "obs", rows)
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_fivemin.py tests/test_quantize.py -v` — Expected: all pass.
Note for the user (put in README later, Task 17): a free Synoptic token from
synopticdata.com goes into `config/klax.yaml -> synoptic_token` to enable the
live 5-minute feed; everything else works without it.
Then: `pytest -q` — all pass.

---

### Task 6: Forecast fetchers (Open-Meteo live + historical, NWS gridpoint, MOS)

**Files:**
- Create: `src/highcast/fetchers/forecasts.py`
- Test: `tests/test_forecasts.py`

**Interfaces:**
- Consumes: `http.get_json`, `db.upsert/get_meta/set_meta`, `timeutil`
- Produces: rows for the `forecasts` table. `MODELS = {"ecmwf": "ecmwf_ifs025", "gfs": "gfs_seamless", "icon": "icon_seamless", "hrrr": "gfs_hrrr"}`. `fetch_openmeteo(cfg) -> list[dict]` (today+2 days, one row per model per target_date). `fetch_openmeteo_lead1(cfg, start: date, end: date) -> list[dict]` (historical lead-1 forecasts from the previous-runs API: hourly `temperature_2m_previous_day1`, maxed over each climate day). `fetch_nws(cfg, conn) -> list[dict]` (model `"nws"`, gridpoint `maxTemperature`). `fetch_mos(cfg, model="MEX") -> list[dict]` (model `"mos_mex"`, IEM). `fetch_mos_archive(cfg, start, end, model="MEX") -> list[dict]`. `store(conn, rows)`.

- [ ] **Step 1: Write failing parser tests**

`tests/test_forecasts.py`:

```python
from datetime import date
from highcast.fetchers import forecasts

OM_MULTI = {
    "daily": {
        "time": ["2026-07-17", "2026-07-18"],
        "temperature_2m_max_ecmwf_ifs025": [84.2, 86.0],
        "temperature_2m_max_gfs_seamless": [85.1, None],
    }
}

def test_parse_openmeteo_multi_model():
    rows = forecasts.parse_openmeteo(OM_MULTI, "KLAX", "2026-07-17", "t0")
    assert {(r["model"], r["target_date"], r["max_temp_f"]) for r in rows} == {
        ("ecmwf", "2026-07-17", 84.2), ("ecmwf", "2026-07-18", 86.0),
        ("gfs", "2026-07-17", 85.1),
    }
    assert all(r["run_date"] == "2026-07-17" for r in rows)

OM_PREV = {
    "hourly": {
        "time": ["2026-07-17T00:00", "2026-07-17T13:00", "2026-07-17T14:00",
                 "2026-07-18T13:00"],
        "temperature_2m_previous_day1_ecmwf_ifs025": [64.0, 83.5, 84.9, 88.0],
    }
}

def test_parse_openmeteo_lead1_takes_daily_max_per_local_day():
    rows = forecasts.parse_openmeteo_lead1(OM_PREV, "KLAX", "t0")
    by_date = {(r["model"], r["target_date"]): r for r in rows}
    r = by_date[("ecmwf", "2026-07-17")]
    assert r["max_temp_f"] == 84.9
    assert r["run_date"] == "2026-07-16"          # lead-1: issued the day before

NWS_GRID = {
    "properties": {"maxTemperature": {"uom": "wmoUnit:degC", "values": [
        {"validTime": "2026-07-17T14:00:00+00:00/PT19H", "value": 29.0},
        {"validTime": "2026-07-18T14:00:00+00:00/PT19H", "value": 30.5},
    ]}}
}

def test_parse_nws_grid_converts_c_to_f():
    rows = forecasts.parse_nws(NWS_GRID, "KLAX", "2026-07-17", "t0",
                               tz="America/Los_Angeles")
    assert rows[0]["model"] == "nws"
    assert rows[0]["target_date"] == "2026-07-17"
    assert abs(rows[0]["max_temp_f"] - 84.2) < 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_forecasts.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/fetchers/forecasts.py`:

```python
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import c_to_f, utcnow_iso

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"
IEM_MOS = "https://mesonet.agron.iastate.edu/api/1/mos.json"

MODELS = {"ecmwf": "ecmwf_ifs025", "gfs": "gfs_seamless",
          "icon": "icon_seamless", "hrrr": "gfs_hrrr"}
_BY_SUFFIX = {v: k for k, v in MODELS.items()}


def parse_openmeteo(payload: dict, icao: str, run_date: str,
                    fetched_at: str) -> list[dict]:
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    rows = []
    for key, values in daily.items():
        if not key.startswith("temperature_2m_max"):
            continue
        suffix = key.removeprefix("temperature_2m_max").lstrip("_")
        model = _BY_SUFFIX.get(suffix, suffix or "openmeteo")
        for d, v in zip(dates, values):
            if v is None:
                continue
            rows.append({"station": icao, "target_date": d, "model": model,
                         "run_date": run_date, "max_temp_f": float(v),
                         "fetched_at": fetched_at})
    return rows


def fetch_openmeteo(cfg: AppConfig) -> list[dict]:
    st = cfg.station
    today_local = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    payload = get_json(OPEN_METEO, {
        "latitude": st.lat, "longitude": st.lon,
        "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
        "timezone": st.tz, "forecast_days": 3,
        "models": ",".join(MODELS.values()),
    })
    return parse_openmeteo(payload, st.icao, today_local, utcnow_iso())


def parse_openmeteo_lead1(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    rows = []
    for key, values in hourly.items():
        if not key.startswith("temperature_2m_previous_day1"):
            continue
        suffix = key.removeprefix("temperature_2m_previous_day1").lstrip("_")
        model = _BY_SUFFIX.get(suffix, suffix or "openmeteo")
        daymax: dict[str, float] = defaultdict(lambda: float("-inf"))
        for t, v in zip(times, values):
            if v is not None:
                d = t[:10]
                daymax[d] = max(daymax[d], float(v))
        for d, v in sorted(daymax.items()):
            run = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
            rows.append({"station": icao, "target_date": d, "model": model,
                         "run_date": run, "max_temp_f": v,
                         "fetched_at": fetched_at})
    return rows


def fetch_openmeteo_lead1(cfg: AppConfig, start: date, end: date) -> list[dict]:
    st = cfg.station
    rows: list[dict] = []
    cur = start
    while cur <= end:  # chunk by ~180 days to keep responses reasonable
        stop = min(end, cur + timedelta(days=180))
        payload = get_json(PREV_RUNS, {
            "latitude": st.lat, "longitude": st.lon,
            "hourly": "temperature_2m_previous_day1",
            "temperature_unit": "fahrenheit", "timezone": st.tz,
            "start_date": cur.isoformat(), "end_date": stop.isoformat(),
            "models": ",".join(MODELS.values()),
        })
        rows.extend(parse_openmeteo_lead1(payload, st.icao, utcnow_iso()))
        cur = stop + timedelta(days=1)
    return rows


def parse_nws(payload: dict, icao: str, run_date: str, fetched_at: str,
              tz: str) -> list[dict]:
    zone = ZoneInfo(tz)
    rows = []
    for item in payload["properties"]["maxTemperature"]["values"]:
        start_iso = item["validTime"].split("/")[0]
        local_day = datetime.fromisoformat(start_iso).astimezone(zone).date()
        if item["value"] is None:
            continue
        rows.append({"station": icao, "target_date": local_day.isoformat(),
                     "model": "nws", "run_date": run_date,
                     "max_temp_f": round(c_to_f(float(item["value"])), 1),
                     "fetched_at": fetched_at})
    return rows


def fetch_nws(cfg: AppConfig, conn) -> list[dict]:
    st = cfg.station
    grid_url = db.get_meta(conn, "nws_grid_url")
    if not grid_url:
        pts = get_json(NWS_POINTS.format(lat=st.lat, lon=st.lon))
        grid_url = pts["properties"]["forecastGridData"]
        db.set_meta(conn, "nws_grid_url", grid_url)
    payload = get_json(grid_url)
    today_local = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    return parse_nws(payload, st.icao, today_local, utcnow_iso(), st.tz)


def parse_mos(payload: dict, icao: str, fetched_at: str,
              model_name: str) -> list[dict]:
    rows = []
    for item in payload.get("data", []):
        nx = item.get("n_x")
        ftime = item.get("ftime") or ""
        runtime = item.get("runtime") or ""
        if nx is None or not ftime:
            continue
        rows.append({"station": icao, "target_date": str(ftime)[:10],
                     "model": model_name, "run_date": str(runtime)[:10],
                     "max_temp_f": float(nx), "fetched_at": fetched_at})
    return rows


def fetch_mos(cfg: AppConfig, model: str = "MEX") -> list[dict]:
    payload = get_json(IEM_MOS, {"station": cfg.station.icao, "model": model})
    return parse_mos(payload, cfg.station.icao, utcnow_iso(),
                     f"mos_{model.lower()}")


def fetch_mos_archive(cfg: AppConfig, start: date, end: date,
                      model: str = "MEX") -> list[dict]:
    rows: list[dict] = []
    cur = start
    while cur <= end:  # IEM MOS API accepts runtime windows; chunk monthly
        stop = min(end, cur + timedelta(days=31))
        payload = get_json(IEM_MOS, {
            "station": cfg.station.icao, "model": model,
            "sts": cur.isoformat() + "T00:00Z", "ets": stop.isoformat() + "T23:59Z",
        })
        rows.extend(parse_mos(payload, cfg.station.icao, utcnow_iso(),
                              f"mos_{model.lower()}"))
        cur = stop + timedelta(days=1)
    return rows


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecasts", rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecasts.py -v` — Expected: 3 passed.

- [ ] **Step 5: Add live smoke tests and run them**

Append to `tests/test_forecasts.py`:

```python
import pytest
from highcast import db as hdb
from highcast.config import load_config

@pytest.mark.live
def test_live_openmeteo_current():
    rows = forecasts.fetch_openmeteo(load_config())
    models = {r["model"] for r in rows}
    assert "ecmwf" in models and "gfs" in models, models
    # If 'hrrr' is missing here, remove it from MODELS and note it in README.

@pytest.mark.live
def test_live_openmeteo_lead1_history():
    rows = forecasts.fetch_openmeteo_lead1(load_config(), date(2024, 7, 1),
                                           date(2024, 7, 31))
    assert len(rows) >= 25

@pytest.mark.live
def test_live_nws_gridpoint(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    rows = forecasts.fetch_nws(load_config(), conn)
    assert any(60 <= r["max_temp_f"] <= 120 for r in rows)

@pytest.mark.live
def test_live_mos():
    rows = forecasts.fetch_mos(load_config())
    assert rows, "MOS endpoint/params need adapting - print raw payload and fix parse_mos"
```

Run: `pytest tests/test_forecasts.py -m live -v -s`
Expected: PASS. The MOS and previous-runs endpoints are the most likely to need
param/field adaptation — if a test fails, print the raw payload, adapt the
parser (signature fixed), update fixtures, re-run. If MOS proves unavailable,
drop `fetch_mos*` calls in Task 8 and the `mos_mex` feature column in Task 9
(the feature frame tolerates missing columns).
Then: `pytest -q` — all pass.

---

### Task 7: Heat-alert fetcher (live + VTEC archive)

**Files:**
- Create: `src/highcast/fetchers/alerts.py`
- Test: `tests/test_alerts.py`

**Interfaces:**
- Consumes: `http.get_json`, `db.upsert`
- Produces: `HEAT_EVENTS = {"Excessive Heat Warning", "Extreme Heat Warning", "Excessive Heat Watch", "Extreme Heat Watch", "Heat Advisory", "Red Flag Warning"}`; `fetch_active(cfg) -> list[dict]` (api.weather.gov by point, heat events only); `fetch_archive_year(cfg, year: int) -> list[dict]` (IEM VTEC, phenomena EH/XH/HT/FW filtered to `cfg.station.nws_zones`); `store(conn, rows)`.

- [ ] **Step 1: Write failing parser test**

`tests/test_alerts.py`:

```python
from highcast.fetchers import alerts

ACTIVE = {"features": [
    {"properties": {"event": "Heat Advisory",
                    "onset": "2026-07-17T18:00:00+00:00",
                    "expires": "2026-07-18T04:00:00+00:00"}},
    {"properties": {"event": "Small Craft Advisory",
                    "onset": "2026-07-17T18:00:00+00:00",
                    "expires": "2026-07-18T04:00:00+00:00"}},
]}

def test_parse_active_keeps_heat_events_only():
    rows = alerts.parse_active(ACTIVE, "KLAX", "t0")
    assert len(rows) == 1
    assert rows[0]["event"] == "Heat Advisory"
    assert rows[0]["onset_utc"] == "2026-07-17T18:00:00+00:00"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_alerts.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/fetchers/alerts.py`:

```python
from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import utcnow_iso

NWS_ALERTS = "https://api.weather.gov/alerts/active"
IEM_VTEC = "https://mesonet.agron.iastate.edu/api/1/vtec/events.json"

HEAT_EVENTS = {"Excessive Heat Warning", "Extreme Heat Warning",
               "Excessive Heat Watch", "Extreme Heat Watch",
               "Heat Advisory", "Red Flag Warning"}
# VTEC phenomena codes for the same events
HEAT_PHENOMENA = {"EH", "XH", "HT", "FW"}
_PHENOM_NAME = {"EH": "Excessive Heat Warning", "XH": "Extreme Heat Warning",
                "HT": "Heat Advisory", "FW": "Red Flag Warning"}


def parse_active(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    rows = []
    for feat in payload.get("features", []):
        p = feat.get("properties", {})
        if p.get("event") not in HEAT_EVENTS:
            continue
        rows.append({"station": icao, "event": p["event"],
                     "onset_utc": p.get("onset") or p.get("effective"),
                     "expires_utc": p.get("expires"), "fetched_at": fetched_at})
    return [r for r in rows if r["onset_utc"]]


def fetch_active(cfg: AppConfig) -> list[dict]:
    st = cfg.station
    payload = get_json(NWS_ALERTS, {"point": f"{st.lat},{st.lon}"})
    return parse_active(payload, st.icao, utcnow_iso())


def parse_archive(payload: dict, cfg: AppConfig, fetched_at: str) -> list[dict]:
    zones = set(cfg.station.nws_zones)
    rows = []
    for ev in payload.get("data", []):
        if ev.get("phenomena") not in HEAT_PHENOMENA:
            continue
        ugc = str(ev.get("ugc") or ev.get("ugcs") or "")
        if zones and not any(z in ugc for z in zones):
            continue
        onset = ev.get("issue") or ev.get("utc_issue")
        if not onset:
            continue
        rows.append({"station": cfg.station.icao,
                     "event": _PHENOM_NAME[ev["phenomena"]],
                     "onset_utc": str(onset),
                     "expires_utc": str(ev.get("expire") or ev.get("utc_expire") or ""),
                     "fetched_at": fetched_at})
    return rows


def fetch_archive_year(cfg: AppConfig, year: int) -> list[dict]:
    payload = get_json(IEM_VTEC, {"wfo": cfg.station.wfo, "year": year})
    return parse_archive(payload, cfg, utcnow_iso())


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "alerts", rows)
```

- [ ] **Step 4: Run tests, then live smoke**

Run: `pytest tests/test_alerts.py -v` — Expected: PASS.

Append live test:

```python
import pytest
from highcast.config import load_config

@pytest.mark.live
def test_live_vtec_2024_has_heat_events():
    rows = alerts.fetch_archive_year(load_config(), 2024)
    assert rows, "adapt parse_archive to the real VTEC payload (print it)"
    assert all(r["event"] in alerts.HEAT_EVENTS for r in rows)
```

Run: `pytest tests/test_alerts.py -m live -v -s` — adapt parser to real payload
shape if needed (LOX issues heat products most summers, so an empty 2024 means
a parsing/filter bug, not a quiet year).
Then: `pytest -q` — all pass.

---

### Task 8: Backfill orchestrator + CLI skeleton

**Files:**
- Create: `src/highcast/backfill.py`, `src/highcast/cli.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: every fetcher module, `db`, `config`
- Produces: `backfill.run(conn, cfg, years=5, progress=print) -> dict` (counts per source; idempotent). `backfill.fill_gaps(conn, cfg, progress=print) -> dict` (METAR archive from last stored ob to now when gap > 2h; yesterday's CLI report if missing). `cli.main(argv=None)` — argparse with subcommands `init-db`, `backfill --years N`, `fetch`, `wake`. Later tasks add `predict`, `backtest`, `verify-quantization`, `serve` to this same parser; each command is a `cmd_<name>(args)` function.

- [ ] **Step 1: Write failing test (gap detection logic, no network)**

`tests/test_backfill.py`:

```python
from datetime import datetime, timedelta, timezone
from highcast import backfill

def test_gap_start_none_when_fresh():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    last = (now - timedelta(minutes=30)).isoformat()
    assert backfill.gap_start(last, now, max_gap_hours=2) is None

def test_gap_start_returns_last_when_stale():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    last = (now - timedelta(hours=30)).isoformat()
    start = backfill.gap_start(last, now, max_gap_hours=2)
    assert start is not None and start.date() == (now - timedelta(hours=30)).date()

def test_gap_start_handles_empty_db():
    now = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)
    start = backfill.gap_start(None, now, max_gap_hours=2)
    assert start == (now - timedelta(days=7))   # bootstrap: last week
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_backfill.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/backfill.py`:

```python
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.config import AppConfig
from highcast.fetchers import alerts, cli_reports, fivemin, forecasts, metar


def gap_start(last_valid_utc: str | None, now: datetime,
              max_gap_hours: int = 2) -> datetime | None:
    if last_valid_utc is None:
        return now - timedelta(days=7)
    last = datetime.fromisoformat(last_valid_utc)
    if now - last <= timedelta(hours=max_gap_hours):
        return None
    return last


def run(conn, cfg: AppConfig, years: int = 5, progress=print) -> dict:
    """Assemble the historical dataset. Idempotent: re-running upserts."""
    counts: dict[str, int] = {}
    this_year = datetime.now(timezone.utc).year
    for year in range(this_year - years, this_year + 1):
        counts[f"cli_{year}"] = cli_reports.store(
            conn, cli_reports.fetch_year(cfg, year))
        counts[f"vtec_{year}"] = alerts.store(
            conn, alerts.fetch_archive_year(cfg, year))
        progress(f"{year}: cli={counts[f'cli_{year}']} vtec={counts[f'vtec_{year}']}")
    start = date(this_year - years, 1, 1)
    end = datetime.now(timezone.utc).date()
    counts["metar"] = metar.store(conn, metar.fetch_archive(cfg, start, end))
    progress(f"metar archive: {counts['metar']}")
    counts["om_lead1"] = forecasts.store(
        conn, forecasts.fetch_openmeteo_lead1(cfg, start, end))
    counts["mos"] = forecasts.store(
        conn, forecasts.fetch_mos_archive(cfg, start, end))
    progress(f"forecasts: om_lead1={counts['om_lead1']} mos={counts['mos']}")
    return counts


def fetch_snapshot(conn, cfg: AppConfig) -> dict:
    """One live fetch cycle: obs + forecasts + alerts + yesterday's CLI."""
    counts = {
        "metar": metar.store(conn, metar.fetch_recent(cfg)),
        "syn5": fivemin.store(conn, fivemin.fetch_recent(cfg)),
        "openmeteo": forecasts.store(conn, forecasts.fetch_openmeteo(cfg)),
        "nws": forecasts.store(conn, forecasts.fetch_nws(cfg, conn)),
        "mos": forecasts.store(conn, forecasts.fetch_mos(cfg)),
        "alerts": alerts.store(conn, alerts.fetch_active(cfg)),
    }
    yesterday = (datetime.now(ZoneInfo(cfg.station.tz)).date()
                 - timedelta(days=1)).isoformat()
    have = conn.execute(
        "SELECT 1 FROM cli_reports WHERE station=? AND date=?",
        (cfg.station.icao, yesterday)).fetchone()
    if not have:
        year = int(yesterday[:4])
        counts["cli"] = cli_reports.store(conn, cli_reports.fetch_year(cfg, year))
    return counts


def fill_gaps(conn, cfg: AppConfig, progress=print) -> dict:
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "SELECT MAX(valid_utc) AS m FROM obs WHERE station=? AND source='metar'",
        (cfg.station.icao,)).fetchone()
    start = gap_start(row["m"] if row else None, now)
    counts = {}
    if start is not None:
        counts["metar_gap"] = metar.store(
            conn, metar.fetch_archive(cfg, start.date(), now.date()))
        progress(f"backfilled metar gap since {start.date()}: {counts['metar_gap']}")
    counts.update(fetch_snapshot(conn, cfg))
    return counts
```

`src/highcast/cli.py`:

```python
import argparse

from highcast import backfill, db
from highcast.config import load_config


def cmd_init_db(args) -> None:
    cfg = load_config(args.config)
    db.connect(cfg.db_path)
    print(f"initialized {cfg.db_path}")


def cmd_backfill(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    counts = backfill.run(conn, cfg, years=args.years)
    print(counts)


def cmd_fetch(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.fetch_snapshot(conn, cfg))


def cmd_wake(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.fill_gaps(conn, cfg))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="highcast")
    p.add_argument("--config", default="config/klax.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    b = sub.add_parser("backfill")
    b.add_argument("--years", type=int, default=5)
    b.set_defaults(func=cmd_backfill)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    sub.add_parser("wake").set_defaults(func=cmd_wake)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
```

- [ ] **Step 4: Run tests, then do the real backfill**

Run: `pytest -q` — Expected: all pass.
Run: `highcast init-db && highcast backfill --years 5`
Expected: progress lines per year; takes minutes (IEM archive is large). Then sanity-check:

```bash
sqlite3 data/highcast.db "SELECT COUNT(*), MIN(date), MAX(date) FROM cli_reports;"
sqlite3 data/highcast.db "SELECT model, COUNT(*) FROM forecasts GROUP BY model;"
sqlite3 data/highcast.db "SELECT COUNT(*) FROM obs WHERE source='metar';"
```

Expected: ~1800+ CLI rows spanning 5+ years; ecmwf/gfs/icon (hrrr coverage
may be shorter) plus mos_mex rows; tens of thousands of METARs. Record actual
counts in the task report.

---

### Task 9: Feature builder (leakage-safe day-ahead features)

**Files:**
- Create: `src/highcast/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `db` tables, `timeutil.climate_day_bounds`, `config`
- Produces: `build_dayahead(conn, cfg, dates: list[str], cutoff_local_hour=9, require_target=True) -> pandas.DataFrame` with one row per date and columns: `target_date, fc_ecmwf, fc_gfs, fc_icon, fc_hrrr, fc_nws, fc_mos_mex, fc_spread, fc_mean, doy_sin, doy_cos, year_index, yesterday_high, morn_tmpf, morn_dwpf, morn_onshore, morn_sky, heat_alert, y` (float NaN where missing; `y` = CLI high, NaN allowed when `require_target=False`). `FEATURES: list[str]` = all columns except `target_date` and `y`. **Availability rule (leakage guard):** a forecast row may be used for `target_date` D only if `run_date < D`, or `run_date == D` and `fetched_at <= cutoff` (cutoff = D at `cutoff_local_hour` in station tz); obs rows only if `valid_utc` within `[climate_day_start(D), cutoff]`.

- [ ] **Step 1: Write failing tests**

`tests/test_features.py`:

```python
import pandas as pd
from datetime import datetime, timezone
from highcast import db as hdb, features
from highcast.config import load_config

CFG = load_config()
ICAO = CFG.station.icao

def _seed(conn):
    hdb.upsert(conn, "cli_reports", [
        {"station": ICAO, "date": "2025-07-15", "high_f": 82, "low_f": 63, "raw": "{}", "fetched_at": "x"},
        {"station": ICAO, "date": "2025-07-16", "high_f": 84, "low_f": 64, "raw": "{}", "fetched_at": "x"},
    ])
    hdb.upsert(conn, "forecasts", [
        # lead-1 row: usable
        {"station": ICAO, "target_date": "2025-07-16", "model": "ecmwf",
         "run_date": "2025-07-15", "max_temp_f": 83.0, "fetched_at": "x"},
        # same-day row fetched AFTER the 09:00 local cutoff (16:00Z = 9am PDT; use 17:00Z): NOT usable
        {"station": ICAO, "target_date": "2025-07-16", "model": "gfs",
         "run_date": "2025-07-16", "max_temp_f": 99.0,
         "fetched_at": "2025-07-16T17:30:00+00:00"},
    ])
    hdb.upsert(conn, "obs", [
        # 08:00 PDT = 15:00Z on 2025-07-16, inside climate day, before cutoff
        {"station": ICAO, "valid_utc": "2025-07-16T15:00:00+00:00", "source": "metar",
         "tmpf": 68.0, "tmpf_precise": 68.2, "dwpf": 60.1, "sknt": 6.0, "drct": 250.0,
         "skyc": "OVC", "six_max_c": None, "raw": "", "fetched_at": "x"},
    ])

def test_dayahead_features_respect_cutoff(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    df = features.build_dayahead(conn, CFG, ["2025-07-16"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["fc_ecmwf"] == 83.0
    assert pd.isna(row["fc_gfs"])            # fetched after cutoff -> excluded
    assert row["y"] == 84
    assert row["yesterday_high"] == 82
    assert abs(row["morn_dwpf"] - 60.1) < 0.01
    assert row["morn_sky"] == 4.0            # OVC ordinal
    assert row["heat_alert"] == 0.0

def test_alert_flag_set_when_overlapping(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    _seed(conn)
    hdb.upsert(conn, "alerts", [
        {"station": ICAO, "event": "Heat Advisory",
         "onset_utc": "2025-07-16T10:00:00+00:00",
         "expires_utc": "2025-07-17T04:00:00+00:00", "fetched_at": "x"},
    ])
    df = features.build_dayahead(conn, CFG, ["2025-07-16"])
    assert df.iloc[0]["heat_alert"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_features.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/features.py`:

```python
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from highcast.config import AppConfig
from highcast.timeutil import climate_day_bounds

FC_MODELS = ["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex"]
SKY_ORD = {"CLR": 0.0, "SKC": 0.0, "FEW": 1.0, "SCT": 2.0, "BKN": 3.0, "OVC": 4.0}
ONSHORE_DEG = 250.0  # sea-breeze direction at LAX-like coastal stations

FEATURES = ([f"fc_{m}" for m in FC_MODELS]
            + ["fc_spread", "fc_mean", "doy_sin", "doy_cos", "year_index",
               "yesterday_high", "morn_tmpf", "morn_dwpf", "morn_onshore",
               "morn_sky", "heat_alert"])


def _cutoff_utc(cfg: AppConfig, d: date, hour_local: int) -> datetime:
    tz = ZoneInfo(cfg.station.tz)
    return datetime(d.year, d.month, d.day, hour_local, tzinfo=tz).astimezone(
        ZoneInfo("UTC"))


def _forecast_features(conn, cfg, d: date, cutoff: datetime) -> dict:
    icao = cfg.station.icao
    out: dict[str, float] = {}
    for m in FC_MODELS:
        row = conn.execute(
            """SELECT max_temp_f FROM forecasts
               WHERE station=? AND target_date=? AND model=?
                 AND (run_date < ? OR (run_date = ? AND fetched_at <= ?))
               ORDER BY run_date DESC, fetched_at DESC LIMIT 1""",
            (icao, d.isoformat(), m, d.isoformat(), d.isoformat(),
             cutoff.isoformat())).fetchone()
        out[f"fc_{m}"] = float(row["max_temp_f"]) if row else np.nan
    vals = [v for v in out.values() if not np.isnan(v)]
    out["fc_mean"] = float(np.mean(vals)) if vals else np.nan
    out["fc_spread"] = float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else np.nan
    return out


def _morning_features(conn, cfg, d: date, cutoff: datetime) -> dict:
    start, _ = climate_day_bounds(d, cfg.station.std_utc_offset)
    row = conn.execute(
        """SELECT tmpf, tmpf_precise, dwpf, sknt, drct, skyc FROM obs
           WHERE station=? AND source='metar' AND valid_utc >= ? AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (cfg.station.icao, start.isoformat(), cutoff.isoformat())).fetchone()
    if row is None:
        return {"morn_tmpf": np.nan, "morn_dwpf": np.nan,
                "morn_onshore": np.nan, "morn_sky": np.nan}
    tmpf = row["tmpf_precise"] if row["tmpf_precise"] is not None else row["tmpf"]
    onshore = np.nan
    if row["sknt"] is not None and row["drct"] is not None:
        onshore = row["sknt"] * math.cos(math.radians(row["drct"] - ONSHORE_DEG))
    return {"morn_tmpf": tmpf if tmpf is not None else np.nan,
            "morn_dwpf": row["dwpf"] if row["dwpf"] is not None else np.nan,
            "morn_onshore": onshore,
            "morn_sky": SKY_ORD.get(row["skyc"], np.nan)}


def _alert_flag(conn, cfg, d: date) -> float:
    start, end = climate_day_bounds(d, cfg.station.std_utc_offset)
    row = conn.execute(
        """SELECT 1 FROM alerts WHERE station=? AND onset_utc <= ?
             AND (expires_utc IS NULL OR expires_utc = '' OR expires_utc >= ?)
           LIMIT 1""",
        (cfg.station.icao, end.isoformat(), start.isoformat())).fetchone()
    return 1.0 if row else 0.0


def build_dayahead(conn, cfg: AppConfig, dates: list[str],
                   cutoff_local_hour: int = 9,
                   require_target: bool = True) -> pd.DataFrame:
    icao = cfg.station.icao
    highs = {r["date"]: r["high_f"] for r in conn.execute(
        "SELECT date, high_f FROM cli_reports WHERE station=?", (icao,))}
    rows = []
    for ds in dates:
        d = date.fromisoformat(ds)
        y = highs.get(ds, np.nan)
        if require_target and (y is None or (isinstance(y, float) and np.isnan(y))):
            continue
        cutoff = _cutoff_utc(cfg, d, cutoff_local_hour)
        doy = d.timetuple().tm_yday
        rec = {"target_date": ds, "y": float(y) if y is not None else np.nan,
               "doy_sin": math.sin(2 * math.pi * doy / 365.25),
               "doy_cos": math.cos(2 * math.pi * doy / 365.25),
               "year_index": float(d.year - 2000),
               "yesterday_high": float(highs.get(
                   (d - timedelta(days=1)).isoformat(), np.nan)),
               "heat_alert": _alert_flag(conn, cfg, d)}
        rec.update(_forecast_features(conn, cfg, d, cutoff))
        rec.update(_morning_features(conn, cfg, d, cutoff))
        rows.append(rec)
    return pd.DataFrame(rows, columns=["target_date", "y", *FEATURES])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v` — Expected: 2 passed.
Then `pytest -q` — all pass.
Sanity-check on real data (backfilled in Task 8):

```bash
python -c "
from highcast import db, features
from highcast.config import load_config
cfg = load_config(); conn = db.connect(cfg.db_path)
dates = [r['date'] for r in conn.execute(
    \"SELECT date FROM cli_reports WHERE station=? ORDER BY date\", (cfg.station.icao,))]
df = features.build_dayahead(conn, cfg, dates)
print(df.shape); print(df[['fc_ecmwf','fc_nws','morn_dwpf','y']].notna().mean())"
```

Expected: >1500 rows; `fc_ecmwf` non-null on most rows with forecast coverage;
`fc_nws` mostly null for the backfilled past (accumulates from live fetches —
this is expected and documented in the spec).

---

### Task 10: Buckets + probabilistic metrics (pure math)

**Files:**
- Create: `src/highcast/buckets.py`, `src/highcast/metrics.py`
- Test: `tests/test_buckets.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: numpy only
- Produces: `buckets.bucket_probs(quantiles: dict[float, float]) -> dict[int, float]` (pmf over integer °F; exponential tails beyond outer quantiles — never clipped; sums to 1.0); `buckets.cdf_from_quantiles(quantiles) -> callable`; `metrics.crps_pmf(pmf: dict[int, float], y: int) -> float`; `metrics.brier(pmf, y, t: int) -> float`; `metrics.pinball(quantiles, y) -> float`; `metrics.reliability(records: list[tuple[dict, int]], level: float) -> float` (empirical coverage of nominal quantile `level`).

- [ ] **Step 1: Write failing tests**

`tests/test_buckets.py`:

```python
from highcast.buckets import bucket_probs

QS = {0.05: 78.0, 0.25: 81.0, 0.50: 83.0, 0.75: 85.0, 0.95: 88.0}

def test_pmf_sums_to_one_and_peaks_near_median():
    pmf = bucket_probs(QS)
    assert abs(sum(pmf.values()) - 1.0) < 1e-6
    mode = max(pmf, key=pmf.get)
    assert 82 <= mode <= 84

def test_hot_tail_is_never_zeroed():
    pmf = bucket_probs(QS)
    assert pmf.get(92, 0.0) > 0.0       # beyond the 95th percentile
    assert pmf.get(95, 0.0) > 0.0

def test_pmf_monotone_cdf():
    pmf = bucket_probs(QS)
    ts = sorted(pmf)
    cum = 0.0
    for t in ts:
        assert pmf[t] >= 0.0
        cum += pmf[t]
    assert abs(cum - 1.0) < 1e-6
```

`tests/test_metrics.py`:

```python
from highcast.metrics import brier, crps_pmf, pinball

def test_crps_zero_for_certain_correct_forecast():
    assert crps_pmf({84: 1.0}, 84) == 0.0

def test_crps_penalizes_distance():
    near = crps_pmf({83: 0.5, 84: 0.5}, 84)
    far = crps_pmf({80: 0.5, 84: 0.5}, 84)
    assert near < far

def test_brier():
    assert brier({84: 0.7, 85: 0.3}, 84, 84) == (0.7 - 1) ** 2
    assert brier({84: 0.7, 85: 0.3}, 85, 84) == 0.7 ** 2

def test_pinball():
    qs = {0.5: 83.0}
    assert pinball(qs, 85) == 0.5 * (85 - 83)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_buckets.py tests/test_metrics.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/buckets.py`:

```python
import numpy as np


def cdf_from_quantiles(quantiles: dict[float, float]):
    """Monotone CDF from (level -> value) pairs with exponential tails."""
    items = sorted(quantiles.items())
    levels = np.array([l for l, _ in items], dtype=float)
    vals = np.maximum.accumulate(np.array([v for _, v in items], dtype=float))
    lo_scale = max(1.0, (vals[1] - vals[0]) if len(vals) > 1 else 1.0)
    hi_scale = max(1.0, (vals[-1] - vals[-2]) if len(vals) > 1 else 1.0)

    def cdf(x: float) -> float:
        if x <= vals[0]:
            return float(levels[0] * np.exp((x - vals[0]) / lo_scale))
        if x >= vals[-1]:
            return float(1.0 - (1.0 - levels[-1]) * np.exp(-(x - vals[-1]) / hi_scale))
        return float(np.interp(x, vals, levels))

    return cdf


def bucket_probs(quantiles: dict[float, float]) -> dict[int, float]:
    cdf = cdf_from_quantiles(quantiles)
    vals = list(quantiles.values())
    lo = int(np.floor(min(vals))) - 15
    hi = int(np.ceil(max(vals))) + 15
    pmf = {t: max(0.0, cdf(t + 0.5) - cdf(t - 0.5)) for t in range(lo, hi + 1)}
    total = sum(pmf.values())
    pmf = {t: p / total for t, p in pmf.items()}
    return {t: p for t, p in pmf.items() if p > 1e-6} | _renorm_tail(pmf)


def _renorm_tail(pmf: dict[int, float]) -> dict[int, float]:
    # keep tiny tail mass explicit so the hot tail is never reported as zero
    return {t: p for t, p in pmf.items() if 1e-9 < p <= 1e-6}
```

`src/highcast/metrics.py`:

```python
def crps_pmf(pmf: dict[int, float], y: int) -> float:
    # iterate the FULL integer range, not just support points - gaps between
    # support points still contribute (cum - indicator)^2 per skipped degree
    lo = min(min(pmf), y)
    hi = max(max(pmf), y)
    cum = 0.0
    score = 0.0
    for t in range(lo, hi + 1):
        cum += pmf.get(t, 0.0)
        score += (cum - (1.0 if t >= y else 0.0)) ** 2
    return score


def brier(pmf: dict[int, float], y: int, t: int) -> float:
    return (pmf.get(t, 0.0) - (1.0 if y == t else 0.0)) ** 2


def pinball(quantiles: dict[float, float], y: float) -> float:
    total = 0.0
    for q, v in quantiles.items():
        total += q * (y - v) if y >= v else (1 - q) * (v - y)
    return total / len(quantiles)


def reliability(records: list[tuple[dict, int]], level: float) -> float:
    """Fraction of outcomes at or below the nominal `level` quantile."""
    hits = [1.0 if y <= qs[level] else 0.0 for qs, y in records]
    return sum(hits) / len(hits) if hits else float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_buckets.py tests/test_metrics.py -v` — Expected: all pass.
Then `pytest -q` — all pass.

---

### Task 11: Baseline blend + seasonal error distribution

**Files:**
- Create: `src/highcast/models/__init__.py`, `src/highcast/models/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `features.FEATURES` frame columns (`fc_*`, `doy_sin/cos`, `y`)
- Produces: `LEVELS = [0.05, 0.10, ..., 0.95]` (19 levels, defined here, reused by every model). `class BaselineBlend` with `.fit(df: DataFrame) -> self` (inverse-MSE weights per `fc_*` column; stores train residuals + day-of-year), `.predict_quantiles(df: DataFrame) -> list[dict[float, float]]` (blend point + residual quantiles from a ±45-day seasonal window around each row's day-of-year), `.point(df) -> np.ndarray`.

- [ ] **Step 1: Write failing test**

`tests/test_baseline.py`:

```python
import numpy as np
import pandas as pd
from highcast.models.baseline import BaselineBlend, LEVELS

def _frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.linspace(0, 4, n)) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "target_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "fc_ecmwf": y + rng.normal(0, 1.0, n),      # good model
        "fc_gfs": y + rng.normal(0, 3.0, n),        # worse model
        "fc_nws": np.nan,                            # entirely missing model
        "doy_sin": np.sin(2 * np.pi * np.arange(n) / 365),
        "doy_cos": np.cos(2 * np.pi * np.arange(n) / 365),
        "y": y,
    })

def test_weights_favor_better_model_and_handle_missing():
    m = BaselineBlend().fit(_frame())
    assert m.weights["fc_ecmwf"] > m.weights["fc_gfs"]
    assert "fc_nws" not in m.weights

def test_quantiles_are_monotone_and_centered():
    df = _frame()
    m = BaselineBlend().fit(df)
    qs = m.predict_quantiles(df.tail(5))
    for q in qs:
        vals = [q[l] for l in LEVELS]
        assert vals == sorted(vals)
        assert q[0.05] < q[0.50] < q[0.95]

def test_point_tracks_truth_reasonably():
    df = _frame()
    m = BaselineBlend().fit(df)
    mae = np.mean(np.abs(m.point(df) - df["y"]))
    assert mae < 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_baseline.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/models/__init__.py`: empty file.

`src/highcast/models/baseline.py`:

```python
import numpy as np
import pandas as pd

LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95
_FC_PREFIX = "fc_"
_EXCLUDE = {"fc_spread", "fc_mean"}
SEASON_WINDOW = 45  # +/- days of day-of-year for the residual pool


class BaselineBlend:
    def fit(self, df: pd.DataFrame) -> "BaselineBlend":
        self.weights: dict[str, float] = {}
        for col in df.columns:
            if not col.startswith(_FC_PREFIX) or col in _EXCLUDE:
                continue
            mask = df[col].notna() & df["y"].notna()
            if mask.sum() < 30:
                continue
            mse = float(np.mean((df.loc[mask, col] - df.loc[mask, "y"]) ** 2))
            self.weights[col] = 1.0 / max(mse, 0.25)
        if not self.weights:
            raise ValueError("no forecast columns with enough data to fit")
        pts = self.point(df)
        ok = df["y"].notna() & ~np.isnan(pts)
        self._resid = (df.loc[ok, "y"] - pts[ok]).to_numpy()
        self._resid_doy = pd.to_datetime(
            df.loc[ok, "target_date"]).dt.dayofyear.to_numpy()
        return self

    def point(self, df: pd.DataFrame) -> np.ndarray:
        num = np.zeros(len(df))
        den = np.zeros(len(df))
        for col, w in self.weights.items():
            v = df[col].to_numpy(dtype=float)
            has = ~np.isnan(v)
            num[has] += w * v[has]
            den[has] += w
        out = np.full(len(df), np.nan)
        out[den > 0] = num[den > 0] / den[den > 0]
        return out

    def _seasonal_resid(self, doy: int) -> np.ndarray:
        dist = np.minimum(np.abs(self._resid_doy - doy),
                          365 - np.abs(self._resid_doy - doy))
        pool = self._resid[dist <= SEASON_WINDOW]
        return pool if len(pool) >= 60 else self._resid

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        pts = self.point(df)
        doys = pd.to_datetime(df["target_date"]).dt.dayofyear.to_numpy()
        out = []
        for p, doy in zip(pts, doys):
            pool = self._seasonal_resid(int(doy))
            qs = np.quantile(pool, LEVELS)
            out.append({l: float(p + q) for l, q in zip(LEVELS, np.sort(qs))})
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baseline.py -v` — Expected: 3 passed.
Then `pytest -q` — all pass.

---

### Task 12: Walk-forward backtest harness

**Files:**
- Create: `src/highcast/backtest.py`
- Modify: `src/highcast/cli.py` (add `backtest` subcommand)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `features.build_dayahead`, `BaselineBlend`, `buckets.bucket_probs`, `metrics`
- Produces: `walk_forward(df, model_factory, min_train=365, refit_every=30) -> list[dict]` where each record is `{"target_date", "y", "point", "quantiles", "crps", "abs_err"}`; `summarize(records, df) -> dict` with keys `n, mae, crps, mae_nws, mae_blend, cov_10, cov_50, cov_90`; model_factory is a zero-arg callable returning an object with `.fit(df)` and `.predict_quantiles(df) -> list[dict]`.

- [ ] **Step 1: Write failing test**

`tests/test_backtest.py`:

```python
import numpy as np
import pandas as pd
from highcast import backtest
from highcast.models.baseline import BaselineBlend

def _frame(n=500, seed=2):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "target_date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "fc_ecmwf": y + rng.normal(0, 1, n),
        "fc_nws": y + rng.normal(0, 1.5, n),
        "doy_sin": np.sin(2 * np.pi * np.arange(n) / 365),
        "doy_cos": np.cos(2 * np.pi * np.arange(n) / 365),
        "y": np.round(y),
    })

def test_walk_forward_only_predicts_after_min_train():
    df = _frame()
    recs = backtest.walk_forward(df, BaselineBlend, min_train=365, refit_every=60)
    assert len(recs) == len(df) - 365
    assert recs[0]["target_date"] == df["target_date"].iloc[365]
    assert all(r["crps"] >= 0 for r in recs)

def test_summarize_reports_calibration_and_baselines():
    df = _frame()
    recs = backtest.walk_forward(df, BaselineBlend, min_train=365, refit_every=60)
    s = backtest.summarize(recs, df)
    assert s["n"] == len(recs)
    assert s["mae"] < 2.5
    assert 0.05 <= s["cov_10"] <= 0.35      # rough calibration on synthetic data
    assert 0.65 <= s["cov_90"] <= 0.99
    assert "mae_nws" in s
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_backtest.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/backtest.py`:

```python
import numpy as np
import pandas as pd

from highcast.buckets import bucket_probs
from highcast.metrics import crps_pmf


def walk_forward(df: pd.DataFrame, model_factory, min_train: int = 365,
                 refit_every: int = 30) -> list[dict]:
    df = df.sort_values("target_date").reset_index(drop=True)
    records: list[dict] = []
    model = None
    for i in range(min_train, len(df)):
        if model is None or (i - min_train) % refit_every == 0:
            model = model_factory()
            model.fit(df.iloc[:i])
        row = df.iloc[[i]]
        y = row["y"].iloc[0]
        if np.isnan(y):
            continue
        qs = model.predict_quantiles(row)[0]
        pmf = bucket_probs(qs)
        records.append({
            "target_date": row["target_date"].iloc[0], "y": int(y),
            "point": qs[0.5], "quantiles": qs,
            "crps": crps_pmf(pmf, int(y)), "abs_err": abs(qs[0.5] - y),
        })
    return records


def _coverage(records: list[dict], level: float) -> float:
    hits = [1.0 if r["y"] <= r["quantiles"][level] else 0.0 for r in records]
    return float(np.mean(hits)) if hits else float("nan")


def summarize(records: list[dict], df: pd.DataFrame) -> dict:
    by_date = df.set_index("target_date")
    mae_nws = np.nanmean([abs(by_date.loc[r["target_date"], "fc_nws"] - r["y"])
                          for r in records]) if "fc_nws" in df else float("nan")
    return {
        "n": len(records),
        "mae": float(np.mean([r["abs_err"] for r in records])),
        "crps": float(np.mean([r["crps"] for r in records])),
        "mae_nws": float(mae_nws),
        "mae_blend": float(np.mean([r["abs_err"] for r in records])),
        "cov_10": _coverage(records, 0.1),
        "cov_50": _coverage(records, 0.5),
        "cov_90": _coverage(records, 0.9),
    }
```

Add to `src/highcast/cli.py` — a new command function and parser entry:

```python
def cmd_backtest(args) -> None:
    import pandas as pd
    from highcast import backtest as bt, features
    from highcast.models.baseline import BaselineBlend

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    df = features.build_dayahead(conn, cfg, dates)
    if args.model == "baseline":
        factory = BaselineBlend
    else:
        from highcast.models.dayahead import DayAheadModel
        factory = DayAheadModel
    recs = bt.walk_forward(df, factory)
    print(pd.Series(bt.summarize(recs, df)).to_string())
```

and inside `build_parser()`:

```python
    t = sub.add_parser("backtest")
    t.add_argument("--model", choices=["baseline", "ml"], default="baseline")
    t.set_defaults(func=cmd_backtest)
```

- [ ] **Step 4: Run tests, then a real baseline backtest**

Run: `pytest tests/test_backtest.py -v` — Expected: 2 passed. `pytest -q` — all pass.
Run: `highcast backtest --model baseline`
Expected: a summary table on real data. Record `mae`, `crps`, `mae_nws` in the
task report — Task 13's ML model must beat these. (`mae_nws` may be NaN until
live NWS fetches accumulate; the ECMWF column then serves as the reference —
note which was used.)

---

### Task 13: Day-ahead LightGBM quantile model + calibration + `predict` command

**Files:**
- Create: `src/highcast/models/dayahead.py`, `src/highcast/models/calibrate.py`
- Modify: `src/highcast/cli.py` (add `predict` subcommand)
- Test: `tests/test_dayahead.py`, `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `features.FEATURES`, `LEVELS` (from `highcast.models.baseline`), `buckets.bucket_probs`
- Produces: `class DayAheadModel` with `.fit(df) -> self` (one LGBMRegressor per level, recency sample weights with 730-day half-life, PIT calibrator fitted on the last 90 days then refit on all data), `.predict_quantiles(df) -> list[dict[float, float]]` (calibrated, non-crossing). `class PITCalibrator` with `.fit(pits: list[float]) -> self`, `.level(nominal: float) -> float`. CLI `highcast predict [--date YYYY-MM-DD]` stores a `predictions` row (`kind="dayahead"`) and prints the bucket table.

- [ ] **Step 1: Write failing calibrator test**

`tests/test_calibrate.py`:

```python
import numpy as np
from highcast.models.calibrate import PITCalibrator

def test_identity_when_calibrated():
    cal = PITCalibrator().fit(list(np.linspace(0.01, 0.99, 200)))
    assert abs(cal.level(0.5) - 0.5) < 0.03

def test_shifts_levels_when_biased_low():
    # outcomes always land in the top of the predicted distribution
    cal = PITCalibrator().fit(list(np.clip(np.linspace(0.5, 1.0, 200), 0, 1)))
    # to truly cover 50%, we must read a higher nominal level
    assert cal.level(0.5) > 0.6
```

- [ ] **Step 2: Run to verify failure, implement calibrator**

Run: `pytest tests/test_calibrate.py -v` — Expected: FAIL.

`src/highcast/models/calibrate.py`:

```python
import numpy as np


class PITCalibrator:
    """Map nominal quantile levels to the levels that actually achieve them.

    PIT value = F_pred(y). If forecasts were perfectly calibrated, PITs are
    uniform. cal.level(q) returns u such that P(PIT <= u) = q, i.e. the raw
    quantile curve should be read at level u to get true coverage q.
    """

    def fit(self, pits: list[float]) -> "PITCalibrator":
        self._sorted = np.sort(np.clip(np.asarray(pits, dtype=float), 0.0, 1.0))
        return self

    def level(self, nominal: float) -> float:
        if len(self._sorted) < 30:
            return nominal
        return float(np.clip(np.quantile(self._sorted, nominal), 0.01, 0.99))
```

Run: `pytest tests/test_calibrate.py -v` — Expected: 2 passed.

- [ ] **Step 3: Write failing model test**

`tests/test_dayahead.py`:

```python
import numpy as np
import pandas as pd
from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.dayahead import DayAheadModel

def _frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = 75 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 2, n)
    df = pd.DataFrame({c: np.nan for c in FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d")
    df["fc_ecmwf"] = y + rng.normal(0, 1, n)
    df["fc_gfs"] = y + rng.normal(0, 2, n)
    df["doy_sin"] = np.sin(2 * np.pi * np.arange(n) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * np.arange(n) / 365)
    df["year_index"] = 24.0
    df["heat_alert"] = 0.0
    df["y"] = np.round(y)
    return df

def test_fit_predict_quantiles_monotone_and_sane():
    df = _frame()
    m = DayAheadModel().fit(df.iloc[:500])
    qs = m.predict_quantiles(df.iloc[500:520])
    assert len(qs) == 20
    for q in qs:
        vals = [q[l] for l in LEVELS]
        assert vals == sorted(vals)
    med_err = np.mean([abs(q[0.5] - y) for q, y in zip(qs, df["y"].iloc[500:520])])
    assert med_err < 3.0

def test_recency_weights_defined():
    df = _frame()
    w = DayAheadModel.recency_weights(df)
    assert w.iloc[-1] > w.iloc[0]           # newer rows weigh more
    assert abs(w.iloc[-1] - 1.0) < 1e-6
```

- [ ] **Step 4: Run to verify failure, implement model**

Run: `pytest tests/test_dayahead.py -v` — Expected: FAIL.

`src/highcast/models/dayahead.py`:

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from highcast.buckets import cdf_from_quantiles
from highcast.features import FEATURES
from highcast.models.baseline import LEVELS
from highcast.models.calibrate import PITCalibrator

HALF_LIFE_DAYS = 730.0   # climate-trend handling: recent years dominate
CAL_HOLDOUT_DAYS = 90


def _lgbm(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300,
                         learning_rate=0.05, num_leaves=15,
                         min_child_samples=20, verbose=-1)


class DayAheadModel:
    @staticmethod
    def recency_weights(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df["target_date"])
        age = (dates.max() - dates).dt.days.astype(float)
        return np.exp(-age * np.log(2) / HALF_LIFE_DAYS)

    def _fit_raw(self, df: pd.DataFrame) -> dict:
        X = df[FEATURES]
        y = df["y"]
        w = self.recency_weights(df)
        return {q: _lgbm(q).fit(X, y, sample_weight=w) for q in LEVELS}

    def _raw_quantiles(self, models: dict, df: pd.DataFrame) -> list[dict]:
        preds = np.column_stack([models[q].predict(df[FEATURES]) for q in LEVELS])
        preds = np.sort(preds, axis=1)               # enforce non-crossing
        return [dict(zip(LEVELS, row)) for row in preds]

    def fit(self, df: pd.DataFrame) -> "DayAheadModel":
        df = df[df["y"].notna()].sort_values("target_date").reset_index(drop=True)
        cut = pd.to_datetime(df["target_date"]).max() - pd.Timedelta(
            days=CAL_HOLDOUT_DAYS)
        train = df[pd.to_datetime(df["target_date"]) <= cut]
        hold = df[pd.to_datetime(df["target_date"]) > cut]
        self.cal = PITCalibrator().fit([])
        if len(train) >= 200 and len(hold) >= 30:
            tmp = self._fit_raw(train)
            pits = [cdf_from_quantiles(qs)(y) for qs, y in
                    zip(self._raw_quantiles(tmp, hold), hold["y"])]
            self.cal = PITCalibrator().fit(pits)
        self.models = self._fit_raw(df)              # refit on everything
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        out = []
        for raw in self._raw_quantiles(self.models, df):
            cdf_levels = sorted(raw)                 # nominal levels
            vals = [raw[l] for l in cdf_levels]
            adj = {}
            for l in LEVELS:
                u = self.cal.level(l)
                adj[l] = float(np.interp(u, cdf_levels, vals))
            out.append(dict(sorted(adj.items())))
        return out
```

Run: `pytest tests/test_dayahead.py -v` — Expected: 2 passed.

- [ ] **Step 5: Add the `predict` command**

Add to `src/highcast/cli.py`:

```python
def cmd_predict(args) -> None:
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from highcast import features
    from highcast.buckets import bucket_probs
    from highcast.models.dayahead import DayAheadModel
    from highcast.timeutil import utcnow_iso

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    target = args.date or datetime.now(ZoneInfo(cfg.station.tz)).date().isoformat()
    hist_dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? AND date < ? ORDER BY date",
        (cfg.station.icao, target))]
    train = features.build_dayahead(conn, cfg, hist_dates)
    model = DayAheadModel().fit(train)
    today = features.build_dayahead(conn, cfg, [target], require_target=False)
    qs = model.predict_quantiles(today)[0]
    pmf = bucket_probs(qs)
    db.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target,
        "issued_at_utc": utcnow_iso(), "kind": "dayahead",
        "point_f": qs[0.5], "quantiles": json.dumps(qs),
        "buckets": json.dumps(pmf),
    }])
    print(f"{cfg.station.icao} {target}  point={qs[0.5]:.1f}F  "
          f"p10={qs[0.1]:.1f} p90={qs[0.9]:.1f}")
    for t in sorted(pmf):
        if pmf[t] >= 0.01:
            print(f"  {t:3d}F  {pmf[t]*100:5.1f}%  {'#' * int(pmf[t] * 200)}")
```

and inside `build_parser()`:

```python
    pr = sub.add_parser("predict")
    pr.add_argument("--date", default=None)
    pr.set_defaults(func=cmd_predict)
```

- [ ] **Step 6: Acceptance gate — the ML must beat the baseline**

Run: `pytest -q` — all pass. Then:

```bash
highcast backtest --model baseline
highcast backtest --model ml
```

Expected: ML `crps` ≤ baseline `crps` AND ML `mae` ≤ baseline `mae` on real
data, and cov_10/cov_50/cov_90 within ±0.07 of 0.10/0.50/0.90 for the ML run.
**If ML loses, do NOT proceed:** check for feature columns that are entirely
NaN in training (kills LightGBM value), verify the availability rule isn't
starving features, and check `recency_weights`. This gate is the spec's
success criterion; escalate to the user if it can't be met.
Then run `highcast predict` and confirm it prints a sane distribution for today.

---

### Task 14: Latent running max, hidden-spike distribution, quantization verification

**Files:**
- Create: `src/highcast/latentmax.py`
- Modify: `src/highcast/cli.py` (add `verify-quantization` subcommand), `src/highcast/fetchers/metar.py` (no change needed — 1-min fetch lives in latentmax)
- Test: `tests/test_latentmax.py`

**Interfaces:**
- Consumes: `quantize`, `climate_day_bounds`, `obs`/`cli_reports` tables, `round_half_up`
- Produces: `visible_running_max(conn, cfg, day: date, asof_utc: str | None = None) -> tuple[float, float] | None` (lo/hi bounds of the true running max in °F from METAR precise temps, 6-hr max groups, and quantized syn5 candidates; None when no obs). `hidden_spike_pmf(conn, cfg, min_days=120) -> dict[int, float]` (distribution of `cli_high − round_half_up(visible_hi)` over history). `fetch_asos1min(cfg, start: date, end: date) -> list[dict]` (IEM 1-min archive, `source="asos1"`, for verification only). CLI `highcast verify-quantization --days 90` prints the spike histogram and boundary checks.

- [ ] **Step 1: Write failing tests**

`tests/test_latentmax.py`:

```python
from datetime import date
from highcast import db as hdb, latentmax
from highcast.config import load_config

CFG = load_config()
ICAO = CFG.station.icao

def _ob(valid, source, **kw):
    base = {"station": ICAO, "valid_utc": valid, "source": source,
            "tmpf": None, "tmpf_precise": None, "dwpf": None, "sknt": None,
            "drct": None, "skyc": None, "six_max_c": None, "raw": "",
            "fetched_at": "x"}
    return base | kw

def test_visible_running_max_combines_sources(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        # climate day 2025-07-16 starts 08:00Z; both obs inside it
        _ob("2025-07-16T18:53:00+00:00", "metar", tmpf_precise=75.02, tmpf=75.2),
        # syn5 stores c_to_f(22 C) = 71.6; whole-C reading -> true F in {71, 72}
        _ob("2025-07-16T20:05:00+00:00", "syn5", tmpf=71.6),
    ])
    lo, hi = latentmax.visible_running_max(conn, CFG, date(2025, 7, 16))
    # metar precise 75.02 dominates both bounds (+/- 0.1 tolerance band)
    assert 74.8 <= lo <= 75.1
    assert 75.0 <= hi <= 75.3

def test_syn5_dominates_when_hotter(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    hdb.upsert(conn, "obs", [
        _ob("2025-07-16T18:53:00+00:00", "metar", tmpf_precise=75.02),
        # c_to_f(26 C) = 78.8; whole-C reading -> true F in {78, 79}
        _ob("2025-07-16T21:05:00+00:00", "syn5", tmpf=78.8),
    ])
    lo, hi = latentmax.visible_running_max(conn, CFG, date(2025, 7, 16))
    assert lo == 78.0 and hi == 79.0

def test_no_obs_returns_none(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    assert latentmax.visible_running_max(conn, CFG, date(2025, 7, 16)) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_latentmax.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/latentmax.py`:

```python
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from highcast.config import AppConfig
from highcast.http import get_text
from highcast.quantize import candidates_from_reported_c
from highcast.timeutil import c_to_f, climate_day_bounds, round_half_up, utcnow_iso

IEM_1MIN = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
METAR_TOL = 0.1  # deg F tolerance on T-group precision


def visible_running_max(conn, cfg: AppConfig, day: date,
                        asof_utc: str | None = None):
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    stop = min(end.isoformat(), asof_utc) if asof_utc else end.isoformat()
    lo = hi = None
    for r in conn.execute(
            """SELECT source, tmpf, tmpf_precise, six_max_c FROM obs
               WHERE station=? AND valid_utc >= ? AND valid_utc < ?""",
            (cfg.station.icao, start.isoformat(), stop)):
        cands: tuple[float, float] | None = None
        if r["source"] == "metar":
            if r["tmpf_precise"] is not None:
                cands = (r["tmpf_precise"] - METAR_TOL, r["tmpf_precise"] + METAR_TOL)
            if r["six_max_c"] is not None:
                v = c_to_f(r["six_max_c"])
                cands = _max_iv(cands, (v - METAR_TOL, v + METAR_TOL))
        elif r["source"] in ("syn5", "asos1") and r["tmpf"] is not None:
            f = r["tmpf"]
            c = (f - 32.0) * 5.0 / 9.0
            if abs(c - round(c)) < 0.02:      # whole-degree C -> quantized feed
                cf = candidates_from_reported_c(int(round(c)))
                cands = (float(min(cf)), float(max(cf)))
            else:                              # decimal reading (e.g. 1-min data)
                cands = (f - 0.5, f + 0.5)
        if cands:
            lo = cands[0] if lo is None else max(lo, cands[0])
            hi = cands[1] if hi is None else max(hi, cands[1])
    return None if lo is None else (lo, hi)


def _max_iv(a, b):
    if a is None:
        return b
    return (max(a[0], b[0]), max(a[1], b[1]))


def hidden_spike_pmf(conn, cfg: AppConfig, min_days: int = 120) -> dict[int, float]:
    """P(cli_high - visible_max = k): how often the official high beats
    everything observable. Learned from history; used as a dashboard
    diagnostic and sanity floor for the intraday model."""
    deltas: Counter[int] = Counter()
    for r in conn.execute(
            "SELECT date, high_f FROM cli_reports WHERE station=? ORDER BY date",
            (cfg.station.icao,)):
        vis = visible_running_max(conn, cfg, date.fromisoformat(r["date"]))
        if vis is None:
            continue
        deltas[int(r["high_f"]) - round_half_up(vis[1])] += 1
    total = sum(deltas.values())
    if total < min_days:
        return {0: 1.0}
    return {k: v / total for k, v in sorted(deltas.items())}


def fetch_asos1min(cfg: AppConfig, start: date, end: date) -> list[dict]:
    now = utcnow_iso()
    text = get_text(IEM_1MIN, {
        "station": cfg.station.iem_id, "vars": "tmpf", "sample": "1min",
        "sts": f"{start.isoformat()}T00:00Z", "ets": f"{end.isoformat()}T23:59Z",
        "what": "download", "tz": "UTC", "delim": "comma",
    })
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3:
            _, valid, tmpf = parts[0], parts[1], parts[2]
            try:
                v = float(tmpf)
            except ValueError:
                continue
            rows.append({"station": cfg.station.icao,
                         "valid_utc": valid.strip().replace(" ", "T") + "+00:00",
                         "source": "asos1", "tmpf": v, "tmpf_precise": None,
                         "dwpf": None, "sknt": None, "drct": None, "skyc": None,
                         "six_max_c": None, "raw": None, "fetched_at": now})
    return rows
```

Add to `src/highcast/cli.py`:

```python
def cmd_verify_quantization(args) -> None:
    from datetime import date, timedelta
    from highcast import latentmax
    from highcast.timeutil import round_half_up

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    pmf = latentmax.hidden_spike_pmf(conn, cfg)
    print("hidden spike pmf (cli_high - visible_max):")
    for k, p in pmf.items():
        print(f"  {k:+d}F  {p*100:5.1f}%")
    neg = sum(p for k, p in pmf.items() if k < -1)
    if neg > 0.02:
        print(f"WARNING: {neg*100:.1f}% of days have CLI high {'<'} visible max "
              "- suspect climate-day boundary bug or feed misparse.")
    # spot-check the 1-min claim on a few recent days
    end = date.today() - timedelta(days=3)
    rows = latentmax.fetch_asos1min(cfg, end - timedelta(days=int(args.days)), end)
    print(f"1-min archive rows fetched: {len(rows)} "
          "(store+compare manually if investigating)")
```

and inside `build_parser()`:

```python
    v = sub.add_parser("verify-quantization")
    v.add_argument("--days", type=int, default=30)
    v.set_defaults(func=cmd_verify_quantization)
```

- [ ] **Step 4: Run tests, then verify against reality**

Run: `pytest tests/test_latentmax.py -v` — Expected: 3 passed. `pytest -q` — all pass.
Run: `highcast verify-quantization --days 30`
Expected: spike pmf concentrated on 0 and +1 (typically ≥85% mass), small mass
at +2; the negative-delta warning must NOT fire. If it fires, debug
`climate_day_bounds` usage and METAR parsing before continuing — the intraday
model's floor depends on this being right. Record the pmf in the task report.

---

### Task 15: Intraday nowcast model + `predict --live`

**Files:**
- Create: `src/highcast/models/intraday.py`
- Modify: `src/highcast/cli.py` (extend `predict` with `--live`)
- Test: `tests/test_intraday.py`

**Interfaces:**
- Consumes: `latentmax.visible_running_max`, `BaselineBlend`, `features.build_dayahead`, `LEVELS`, `buckets.bucket_probs`
- Produces: `build_checkpoint_frame(conn, cfg, dates: list[str], hours=(8, 10, 12, 14, 16, 18, 20), baseline: BaselineBlend | None = None, require_target: bool = True) -> DataFrame` with columns `target_date, hour, run_lo, run_hi, run_width, cur_tmpf, cur_dwpf, cur_onshore, cur_sky, base_p50, base_spread, doy_sin, doy_cos, year_index, heat_alert, y` (hour in station-local time; obs/base features computed as-of that hour; NaN-tolerant). `INTRADAY_FEATURES: list[str]`. `class IntradayModel` with `.fit(frame)`, `.predict_quantiles(frame) -> list[dict]` — after prediction, quantiles are floored at `run_lo` (the high cannot be below what was already observed). CLI: `highcast predict --live` stores `kind="intraday"`.

- [ ] **Step 1: Write failing tests**

`tests/test_intraday.py`:

```python
import numpy as np
import pandas as pd
from highcast.models.intraday import INTRADAY_FEATURES, IntradayModel
from highcast.models.baseline import LEVELS

def _frame(n=800, seed=4):
    rng = np.random.default_rng(seed)
    y = np.round(75 + 8 * np.sin(np.arange(n)) + rng.normal(0, 2, n))
    hour = rng.choice([8, 10, 12, 14, 16, 18], n)
    # visible running max approaches y as the day progresses
    frac = (hour - 6) / 12.0
    run_hi = y - np.maximum(0, (1 - frac) * rng.uniform(2, 10, n))
    df = pd.DataFrame({c: np.nan for c in INTRADAY_FEATURES}, index=range(n))
    df["target_date"] = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    df["hour"] = hour.astype(float)
    df["run_lo"] = run_hi - 1.0
    df["run_hi"] = run_hi
    df["run_width"] = 1.0
    df["base_p50"] = y + rng.normal(0, 2, n)
    df["base_spread"] = 4.0
    df["heat_alert"] = 0.0
    df["y"] = y
    return df

def test_afternoon_is_sharper_than_morning():
    df = _frame()
    m = IntradayModel().fit(df.iloc[:600])
    test = df.iloc[600:]
    def spread(sub):
        qs = m.predict_quantiles(sub)
        return np.mean([q[0.9] - q[0.1] for q in qs])
    assert spread(test[test["hour"] >= 16]) < spread(test[test["hour"] <= 10])

def test_quantiles_floored_at_running_max():
    df = _frame()
    m = IntradayModel().fit(df.iloc[:600])
    sub = df.iloc[600:650]
    qs = m.predict_quantiles(sub)
    for q, lo in zip(qs, sub["run_lo"]):
        assert q[0.05] >= lo - 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_intraday.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`src/highcast/models/intraday.py`:

```python
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from highcast import latentmax
from highcast.config import AppConfig
from highcast.features import SKY_ORD, ONSHORE_DEG, build_dayahead
from highcast.models.baseline import LEVELS, BaselineBlend

INTRADAY_FEATURES = ["hour", "run_lo", "run_hi", "run_width", "cur_tmpf",
                     "cur_dwpf", "cur_onshore", "cur_sky", "base_p50",
                     "base_spread", "doy_sin", "doy_cos", "year_index",
                     "heat_alert"]


def _asof_features(conn, cfg: AppConfig, d: date, asof_utc: str) -> dict:
    row = conn.execute(
        """SELECT tmpf, tmpf_precise, dwpf, sknt, drct, skyc FROM obs
           WHERE station=? AND source='metar' AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (cfg.station.icao, asof_utc)).fetchone()
    out = {"cur_tmpf": np.nan, "cur_dwpf": np.nan, "cur_onshore": np.nan,
           "cur_sky": np.nan}
    if row:
        t = row["tmpf_precise"] if row["tmpf_precise"] is not None else row["tmpf"]
        out["cur_tmpf"] = t if t is not None else np.nan
        out["cur_dwpf"] = row["dwpf"] if row["dwpf"] is not None else np.nan
        if row["sknt"] is not None and row["drct"] is not None:
            out["cur_onshore"] = row["sknt"] * math.cos(
                math.radians(row["drct"] - ONSHORE_DEG))
        out["cur_sky"] = SKY_ORD.get(row["skyc"], np.nan)
    return out


def build_checkpoint_frame(conn, cfg: AppConfig, dates: list[str],
                           hours=(8, 10, 12, 14, 16, 18, 20),
                           baseline: BaselineBlend | None = None,
                           require_target: bool = True) -> pd.DataFrame:
    """Training path: baseline=None fits the blend on `dates` themselves.
    Live path: pass a baseline fitted on HISTORY (a 1-day frame with no
    outcome cannot fit one) and require_target=False."""
    day_df = build_dayahead(conn, cfg, dates, require_target=require_target)
    if day_df.empty:
        return pd.DataFrame(columns=["target_date", "y", *INTRADAY_FEATURES])
    base = baseline or BaselineBlend().fit(day_df)
    base_qs = base.predict_quantiles(day_df)
    tz = ZoneInfo(cfg.station.tz)
    rows = []
    for (_, day_row), qs in zip(day_df.iterrows(), base_qs):
        d = date.fromisoformat(day_row["target_date"])
        for h in hours:
            asof = datetime(d.year, d.month, d.day, h, tzinfo=tz).astimezone(
                ZoneInfo("UTC")).isoformat()
            vis = latentmax.visible_running_max(conn, cfg, d, asof_utc=asof)
            if vis is None:
                continue
            rec = {"target_date": day_row["target_date"], "hour": float(h),
                   "run_lo": vis[0], "run_hi": vis[1],
                   "run_width": vis[1] - vis[0],
                   "base_p50": qs[0.5], "base_spread": qs[0.9] - qs[0.1],
                   "doy_sin": day_row["doy_sin"], "doy_cos": day_row["doy_cos"],
                   "year_index": day_row["year_index"],
                   "heat_alert": day_row["heat_alert"], "y": day_row["y"]}
            rec.update(_asof_features(conn, cfg, d, asof))
            rows.append(rec)
    return pd.DataFrame(rows, columns=["target_date", "y", *INTRADAY_FEATURES])


def _lgbm(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300,
                         learning_rate=0.05, num_leaves=15,
                         min_child_samples=20, verbose=-1)


class IntradayModel:
    def fit(self, frame: pd.DataFrame) -> "IntradayModel":
        frame = frame[frame["y"].notna()]
        X = frame[INTRADAY_FEATURES]
        self.models = {q: _lgbm(q).fit(X, frame["y"]) for q in LEVELS}
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> list[dict[float, float]]:
        preds = np.column_stack(
            [self.models[q].predict(frame[INTRADAY_FEATURES]) for q in LEVELS])
        preds = np.sort(preds, axis=1)
        lows = frame["run_lo"].to_numpy(dtype=float)
        out = []
        for row, lo in zip(preds, lows):
            if not np.isnan(lo):
                row = np.maximum(row, lo)   # the high can't be below observed
            out.append(dict(zip(LEVELS, row.astype(float))))
        return out
```

Modify `cmd_predict` in `src/highcast/cli.py` — replace the whole function with:

```python
def cmd_predict(args) -> None:
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from highcast import features
    from highcast.buckets import bucket_probs
    from highcast.timeutil import utcnow_iso

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    target = args.date or datetime.now(ZoneInfo(cfg.station.tz)).date().isoformat()
    hist_dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? AND date < ? ORDER BY date",
        (cfg.station.icao, target))]
    if args.live:
        from highcast.models.baseline import BaselineBlend
        from highcast.models.intraday import IntradayModel, build_checkpoint_frame
        hist_day = features.build_dayahead(conn, cfg, hist_dates)
        base = BaselineBlend().fit(hist_day)
        train = build_checkpoint_frame(conn, cfg, hist_dates, baseline=base)
        model = IntradayModel().fit(train)
        now_frame = build_checkpoint_frame(
            conn, cfg, [target],
            hours=(datetime.now(ZoneInfo(cfg.station.tz)).hour,),
            baseline=base, require_target=False)
        if now_frame.empty:
            print("no observations yet for the current climate day")
            return
        qs = model.predict_quantiles(now_frame)[0]
        kind = "intraday"
    else:
        from highcast.models.dayahead import DayAheadModel
        train = features.build_dayahead(conn, cfg, hist_dates)
        model = DayAheadModel().fit(train)
        today = features.build_dayahead(conn, cfg, [target], require_target=False)
        qs = model.predict_quantiles(today)[0]
        kind = "dayahead"
    pmf = bucket_probs(qs)
    db.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target,
        "issued_at_utc": utcnow_iso(), "kind": kind,
        "point_f": qs[0.5], "quantiles": json.dumps(qs),
        "buckets": json.dumps({str(k): v for k, v in pmf.items()}),
    }])
    print(f"{cfg.station.icao} {target} [{kind}]  point={qs[0.5]:.1f}F  "
          f"p10={qs[0.1]:.1f} p90={qs[0.9]:.1f}")
    for t in sorted(pmf):
        if pmf[t] >= 0.01:
            print(f"  {t:3d}F  {pmf[t]*100:5.1f}%  {'#' * int(pmf[t] * 200)}")
```

and in `build_parser()`, add to the existing `predict` parser:

```python
    pr.add_argument("--live", action="store_true")
```

Note: the live branch fits the baseline on history and passes it into
`build_checkpoint_frame` — a single-day frame with no outcome cannot fit a
blend, which is why the `baseline`/`require_target` parameters exist.

- [ ] **Step 4: Run tests, then the intraday acceptance check**

Run: `pytest tests/test_intraday.py -v` — Expected: 2 passed. `pytest -q` — all pass.
Acceptance (spec §6.4) on real data:

```bash
python -c "
import numpy as np
from highcast import db, backtest
from highcast.buckets import bucket_probs
from highcast.metrics import crps_pmf
from highcast.config import load_config
from highcast.models.intraday import IntradayModel, build_checkpoint_frame
cfg = load_config(); conn = db.connect(cfg.db_path)
dates = [r['date'] for r in conn.execute(
    \"SELECT date FROM cli_reports WHERE station=? ORDER BY date\", (cfg.station.icao,))]
split = dates[-365]
frame = build_checkpoint_frame(conn, cfg, dates)
train = frame[frame['target_date'] < split]; test = frame[frame['target_date'] >= split]
m = IntradayModel().fit(train)
for h in (8, 12, 14, 16, 18):
    sub = test[test['hour'] == h]
    if len(sub) == 0: continue
    qs = m.predict_quantiles(sub)
    crps = np.mean([crps_pmf(bucket_probs(q), int(y)) for q, y in zip(qs, sub['y'])])
    print(f'hour {h:2d}: n={len(sub):4d} crps={crps:.3f}')"
```

Expected: CRPS strictly decreasing with hour; hour-14 CRPS below the Task 13
day-ahead CRPS. Record the numbers. If not monotone, inspect `run_hi` coverage
(METAR gaps) before touching model params.

---

### Task 16: Dashboard (FastAPI + Manila information clock)

**Files:**
- Create: `src/highcast/clock.py`, `src/highcast/web/__init__.py`, `src/highcast/web/app.py`, `src/highcast/web/templates/index.html`
- Modify: `src/highcast/cli.py` (add `serve` subcommand)
- Test: `tests/test_clock.py`, `tests/test_web.py`

**Interfaces:**
- Consumes: `predictions`/`obs`/`cli_reports` tables, `latentmax.visible_running_max`, `hidden_spike_pmf`, `climate_day_bounds`
- Produces: `clock.events(cfg, day: date) -> list[dict]` (each `{"name", "la", "manila", "done": bool}`, DST-aware). FastAPI app factory `create_app(cfg) -> FastAPI` with routes `GET /` (HTML), `GET /api/today`, `GET /api/clock`, `GET /api/history`. CLI `highcast serve --port 8712`.

- [ ] **Step 1: Write failing clock test**

`tests/test_clock.py`:

```python
from datetime import date
from highcast.clock import events
from highcast.config import load_config

def test_events_are_dst_aware_and_cover_the_cycle():
    evs = events(load_config(), date(2026, 7, 17))
    names = [e["name"] for e in evs]
    assert "Peak heating begins" in names
    assert "Preliminary climate report" in names
    assert "Settlement" in names
    prelim = next(e for e in evs if e["name"] == "Preliminary climate report")
    # 16:35 PDT == 07:35+1d in Manila during July
    assert prelim["la"].endswith("16:35")
    assert prelim["manila"].endswith("07:35")

def test_winter_shift():
    evs = events(load_config(), date(2026, 1, 15))
    prelim = next(e for e in evs if e["name"] == "Preliminary climate report")
    assert prelim["manila"].endswith("08:35")   # PST: +16h
```

- [ ] **Step 2: Run to verify failure, implement clock**

Run: `pytest tests/test_clock.py -v` — Expected: FAIL.

`src/highcast/clock.py`:

```python
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.config import AppConfig
from highcast.timeutil import climate_day_bounds


def _fmt(dt: datetime, tz: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).strftime("%a %H:%M")


def events(cfg: AppConfig, day: date) -> list[dict]:
    la = ZoneInfo(cfg.station.tz)
    et = ZoneInfo("America/New_York")
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    prev = day - timedelta(days=1)
    nxt = day + timedelta(days=1)

    def at(d: date, h: int, m: int = 0, tz=la) -> datetime:
        return datetime(d.year, d.month, d.day, h, m, tzinfo=tz)

    raw = [
        ("00Z model runs land", at(prev, 20)),
        ("Climate day starts (midnight LST)", start),
        ("Morning obs window opens", at(day, 6)),
        ("12Z model runs land / day-ahead sharpest", at(day, 8)),
        ("Peak heating begins", at(day, 12)),
        ("Peak heating ends", at(day, 16)),
        ("Preliminary climate report", at(day, 16, 35)),
        ("Market close", at(day, 23, 59)),
        ("Climate day ends", end),
        ("Settlement", at(nxt, 10, 0, tz=et)),
    ]
    now = datetime.now(timezone.utc)
    return [{"name": n, "la": _fmt(t, cfg.station.tz),
             "manila": _fmt(t, cfg.display_tz), "done": t < now}
            for n, t in raw]
```

Run: `pytest tests/test_clock.py -v` — Expected: 2 passed.

- [ ] **Step 3: Write failing web test**

`tests/test_web.py`:

```python
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi.testclient import TestClient
from highcast import db as hdb
from highcast.config import load_config
from highcast.web.app import create_app

def _client(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    conn = hdb.connect(cfg.db_path)
    today = datetime.now(ZoneInfo(cfg.station.tz)).date().isoformat()
    hdb.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": today,
        "issued_at_utc": "2026-07-17T02:00:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": json.dumps({"0.5": 84.0}),
        "buckets": json.dumps({"83": 0.2, "84": 0.5, "85": 0.3}),
    }])
    return TestClient(create_app(cfg))

def test_today_returns_latest_prediction(tmp_path):
    r = _client(tmp_path).get("/api/today")
    assert r.status_code == 200
    body = r.json()
    assert body["dayahead"]["buckets"]["84"] == 0.5

def test_index_renders(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert "highcast" in r.text.lower()

def test_clock_endpoint(tmp_path):
    r = _client(tmp_path).get("/api/clock")
    assert any(e["name"] == "Settlement" for e in r.json())
```

- [ ] **Step 4: Run to verify failure, implement web app**

Run: `pytest tests/test_web.py -v` — Expected: FAIL.

`src/highcast/web/__init__.py`: empty file.

`src/highcast/web/app.py`:

```python
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from highcast import db, latentmax
from highcast.clock import events
from highcast.config import AppConfig

TEMPLATES = Path(__file__).parent / "templates"


def _latest(conn, icao: str, target: str, kind: str):
    row = conn.execute(
        """SELECT * FROM predictions WHERE station=? AND target_date=? AND kind=?
           ORDER BY issued_at_utc DESC LIMIT 1""", (icao, target, kind)).fetchone()
    if not row:
        return None
    return {"issued_at": row["issued_at_utc"], "point": row["point_f"],
            "quantiles": json.loads(row["quantiles"]),
            "buckets": json.loads(row["buckets"])}


def _freshness(conn, icao: str) -> dict:
    out = {}
    for src in ("metar", "syn5"):
        row = conn.execute(
            "SELECT MAX(valid_utc) AS m FROM obs WHERE station=? AND source=?",
            (icao, src)).fetchone()
        out[src] = row["m"]
    for tbl in ("forecasts", "cli_reports", "alerts"):
        row = conn.execute(
            f"SELECT MAX(fetched_at) AS m FROM {tbl} WHERE station=?",
            (icao,)).fetchone()
        out[tbl] = row["m"]
    return out


def create_app(cfg: AppConfig) -> FastAPI:
    app = FastAPI(title="highcast")
    env = Environment(loader=FileSystemLoader(TEMPLATES))

    def _conn():
        return db.connect(cfg.db_path)

    # computed once at startup: scans all history (too slow per request)
    spike_pmf = latentmax.hidden_spike_pmf(_conn(), cfg)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return env.get_template("index.html").render(
            station=cfg.station.name, icao=cfg.station.icao)

    @app.get("/api/today")
    def today():
        conn = _conn()
        tz = ZoneInfo(cfg.station.tz)
        target = datetime.now(tz).date()
        vis = latentmax.visible_running_max(conn, cfg, target)
        return {
            "target_date": target.isoformat(),
            "dayahead": _latest(conn, cfg.station.icao, target.isoformat(), "dayahead"),
            "intraday": _latest(conn, cfg.station.icao, target.isoformat(), "intraday"),
            "running_max": {"lo": vis[0], "hi": vis[1]} if vis else None,
            "freshness": _freshness(conn, cfg.station.icao),
            "spike_pmf": spike_pmf,
        }

    @app.get("/api/clock")
    def clock():
        tz = ZoneInfo(cfg.station.tz)
        return events(cfg, datetime.now(tz).date())

    @app.get("/api/history")
    def history():
        conn = _conn()
        rows = []
        for r in conn.execute(
                """SELECT c.date, c.high_f, p.point_f FROM cli_reports c
                   LEFT JOIN predictions p ON p.station=c.station
                     AND p.target_date=c.date AND p.kind='dayahead'
                   WHERE c.station=? ORDER BY c.date DESC LIMIT 60""",
                (cfg.station.icao,)):
            rows.append({"date": r["date"], "actual": r["high_f"],
                         "predicted": r["point_f"]})
        errs = [abs(r["predicted"] - r["actual"]) for r in rows
                if r["predicted"] is not None]
        return {"days": rows,
                "mae": sum(errs) / len(errs) if errs else None}

    return app
```

`src/highcast/web/templates/index.html`:

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>highcast — {{ icao }}</title>
<style>
 body{font-family:-apple-system,sans-serif;margin:2rem;max-width:900px}
 h1{font-size:1.3rem} h2{font-size:1rem;margin-top:1.6rem}
 .bar{display:flex;align-items:center;gap:.5rem;font-size:.85rem}
 .bar .fill{background:#e0682f;height:14px;border-radius:3px}
 .bar.live .fill{background:#2f7fe0}
 table{border-collapse:collapse;font-size:.85rem}
 td,th{padding:.2rem .6rem;border-bottom:1px solid #ddd;text-align:left}
 .done{opacity:.45} .stale{color:#c00;font-weight:600}
</style></head><body>
<h1>highcast — {{ station }} ({{ icao }})</h1>
<div id="summary"></div>
<h2>Today's distribution (orange = day-ahead, blue = live)</h2>
<div id="buckets"></div>
<h2>Manila information clock</h2>
<table id="clock"><tr><th>Event</th><th>LA</th><th>Manila</th></tr></table>
<h2>Data freshness</h2><div id="fresh"></div>
<h2>Last 60 days</h2><div id="hist"></div>
<script>
async function j(u){return (await fetch(u)).json()}
function bars(el, pmf, cls){
  const ts = Object.keys(pmf).map(Number).sort((a,b)=>a-b);
  for(const t of ts){
    const p = pmf[t]; if(p < 0.005) continue;
    el.insertAdjacentHTML('beforeend',
      `<div class="bar ${cls}"><span style="width:3em">${t}°F</span>`+
      `<div class="fill" style="width:${p*300}px"></div>`+
      `<span>${(p*100).toFixed(1)}%</span></div>`);
  }
}
async function main(){
  const t = await j('/api/today');
  const s = document.getElementById('summary');
  let txt = `<b>${t.target_date}</b>`;
  if(t.running_max) txt += ` — running max so far: ${t.running_max.lo}–${t.running_max.hi}°F`;
  if(t.dayahead) txt += ` — day-ahead point: ${t.dayahead.point.toFixed(1)}°F`;
  s.innerHTML = txt;
  const b = document.getElementById('buckets');
  if(t.dayahead) bars(b, t.dayahead.buckets, '');
  if(t.intraday){ b.insertAdjacentHTML('beforeend','<hr>'); bars(b, t.intraday.buckets, 'live'); }
  const clock = await j('/api/clock');
  for(const e of clock){
    document.getElementById('clock').insertAdjacentHTML('beforeend',
      `<tr class="${e.done?'done':''}"><td>${e.name}</td><td>${e.la}</td><td>${e.manila}</td></tr>`);
  }
  document.getElementById('fresh').innerHTML = Object.entries(t.freshness)
    .map(([k,v])=>`${k}: ${v ?? '<span class=stale>never</span>'}`).join(' · ');
  const h = await j('/api/history');
  document.getElementById('hist').innerHTML =
    (h.mae!==null?`<b>60-day day-ahead MAE: ${h.mae.toFixed(2)}°F</b><br>`:'') +
    h.days.map(d=>`${d.date}: actual ${d.actual}°F` +
      (d.predicted?`, predicted ${d.predicted.toFixed(1)}°F`:'' )).join('<br>');
}
main();
</script></body></html>
```

Add to `src/highcast/cli.py`:

```python
def cmd_serve(args) -> None:
    import uvicorn
    from highcast.web.app import create_app

    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=args.port)
```

and inside `build_parser()`:

```python
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8712)
    sv.set_defaults(func=cmd_serve)
```

Also add the test dependency: `pip install httpx` is already present (FastAPI's
TestClient needs it).

- [ ] **Step 5: Run tests, then eyeball the real dashboard**

Run: `pytest tests/test_web.py tests/test_clock.py -v` — Expected: all pass.
`pytest -q` — all pass.
Run: `highcast predict && highcast predict --live && highcast serve` then open
`http://127.0.0.1:8712` in a browser. Expected: bucket bars for today, the
Manila clock with past events greyed out, freshness line, 60-day history.
Fix any rendering issue before closing the task.

---

### Task 17: launchd scheduling, auto cycle, README

**Files:**
- Create: `launchd/com.julian.highcast.plist`, `README.md`
- Modify: `src/highcast/cli.py` (extend `wake` into the scheduled auto cycle)
- Test: `tests/test_auto.py`

**Interfaces:**
- Consumes: everything
- Produces: `cli.auto_cycle(conn, cfg) -> list[str]` (actions taken; pure decision logic in `should_predict_dayahead(conn, cfg, now) -> bool` and `in_climate_day(cfg, now) -> bool` so tests need no network). `highcast wake` = fill gaps + snapshot + auto predictions. A launchd plist that runs `highcast wake` every 5 minutes and at load (wake-from-sleep backfill).

- [ ] **Step 1: Write failing decision-logic tests**

`tests/test_auto.py`:

```python
from datetime import datetime, timezone
from highcast import db as hdb
from highcast.cli import in_climate_day, should_predict_dayahead
from highcast.config import load_config

CFG = load_config()

def test_in_climate_day_boundaries():
    # 07:00Z on July 17 is BEFORE the climate day start (08:00Z)
    assert not in_climate_day(CFG, datetime(2026, 7, 17, 7, 0, tzinfo=timezone.utc))
    assert in_climate_day(CFG, datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc))

def test_should_predict_dayahead_once_after_9am_local(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    before9 = datetime(2026, 7, 17, 15, 30, tzinfo=timezone.utc)  # 08:30 PDT
    after9 = datetime(2026, 7, 17, 16, 30, tzinfo=timezone.utc)   # 09:30 PDT
    assert not should_predict_dayahead(conn, CFG, before9)
    assert should_predict_dayahead(conn, CFG, after9)
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-17",
        "issued_at_utc": "2026-07-17T16:20:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": "{}", "buckets": "{}"}])
    assert not should_predict_dayahead(conn, CFG, after9)   # already issued
```

- [ ] **Step 2: Run to verify failure, implement**

Run: `pytest tests/test_auto.py -v` — Expected: FAIL.

Add to `src/highcast/cli.py`:

```python
def in_climate_day(cfg, now) -> bool:
    """True while TODAY'S (local calendar date's) climate day is running.
    Climate days tile time continuously, so the check is specifically against
    the current local date's window - during PDT that window opens at 01:00."""
    from zoneinfo import ZoneInfo
    from highcast.timeutil import climate_day_bounds

    local_date = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    start, end = climate_day_bounds(local_date, cfg.station.std_utc_offset)
    return start <= now < end


def should_predict_dayahead(conn, cfg, now) -> bool:
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo(cfg.station.tz))
    if local.hour < 9:
        return False
    row = conn.execute(
        """SELECT 1 FROM predictions WHERE station=? AND target_date=?
             AND kind='dayahead' LIMIT 1""",
        (cfg.station.icao, local.date().isoformat())).fetchone()
    return row is None


def auto_cycle(conn, cfg) -> list[str]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    actions = []
    if should_predict_dayahead(conn, cfg, now):
        cmd_predict(_ns(config="config/klax.yaml", date=None, live=False))
        actions.append("dayahead")
    if in_climate_day(cfg, now):
        cmd_predict(_ns(config="config/klax.yaml", date=None, live=True))
        actions.append("intraday")
    return actions


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)
```

Then replace `cmd_wake` with:

```python
def cmd_wake(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.fill_gaps(conn, cfg))
    print("auto:", auto_cycle(conn, cfg))
```

`launchd/com.julian.highcast.plist` (replace `__PROJECT__` with the absolute
project path when installing):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.julian.highcast</string>
  <key>ProgramArguments</key><array>
    <string>__PROJECT__/.venv/bin/highcast</string>
    <string>wake</string>
  </array>
  <key>WorkingDirectory</key><string>__PROJECT__</string>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>__PROJECT__/data/launchd.log</string>
  <key>StandardErrorPath</key><string>__PROJECT__/data/launchd.err</string>
</dict></plist>
```

`README.md` — write it with these sections (full prose, not bullets-only):
1. **What this is:** one paragraph — KLAX daily-high probability forecasts
   matching the NWS Daily Climate Report that Kalshi LAHIGH settles on; links
   to the spec and `LAXHIGH.pdf`.
2. **Setup:** venv + `pip install -e '.[dev]'`; optional free Synoptic token
   into `config/klax.yaml` for the live 5-minute feed; `highcast init-db &&
   highcast backfill --years 5`.
3. **Commands:** one line each for `fetch`, `wake`, `predict [--live]`,
   `backtest --model baseline|ml`, `verify-quantization`, `serve`.
4. **Scheduling:** `sed "s|__PROJECT__|$PWD|g" launchd/com.julian.highcast.plist
   > ~/Library/LaunchAgents/com.julian.highcast.plist && launchctl bootstrap
   gui/$(id -u) ~/Library/LaunchAgents/com.julian.highcast.plist`; note that
   RunAtLoad backfills after every Mac wake.
5. **Manila cheat sheet:** the information-clock table from the spec (§8),
   plus the one-line takeaway that ~21:00–24:00 PHT is the sharpest
   high-information window before the day resolves.
6. **Disclaimers:** forecasts are information, not financial advice; verify
   Kalshi eligibility for Philippine residents; NWS data can be revised —
   settlement uses the final climate report.

- [ ] **Step 3: Run everything**

Run: `pytest tests/test_auto.py -v` — Expected: 3 passed.
Run: `pytest -q` — Expected: ALL tests pass.
Run: `highcast wake` — Expected: gap backfill counts + auto actions printed.
Install the launchd job per README §4 and verify:
`launchctl list | grep highcast` — Expected: the job is listed; after 5 minutes
`data/launchd.log` shows a wake cycle.

---

## Plan Self-Review (completed by the author)

**Spec coverage:** ground truth CLI reports (T3), METAR precision + 6-hr groups
(T4), quantized 5-min feed + rounding inversion (T5), multi-model forecasts +
archived-run training data + MOS + NWS (T6), heat alerts live + VTEC history
(T7), backfill-on-wake for Manila sleep hours (T8, T17), leakage-safe features
+ climate-trend features (T9), Kalshi bucket conventions + unclipped hot tail
(T10), baseline yardstick (T11), walk-forward backtest + calibration coverage
metrics (T12), LightGBM day-ahead + recency weighting + PIT calibration + the
must-beat-baseline gate (T13), latent running max + hidden-spike pmf +
quantization/boundary verification (T14), intraday nowcast + floor at observed
max + sharpness acceptance (T15), dashboard with Manila clock + freshness +
history (T16), launchd + auto cycle + README with eligibility/advice
disclaimers (T17). AFD text features, raw news scraping, other stations,
trading integration: out of scope per spec.

**Known simplifications (documented, intentional):** NWS forecast history
accumulates from live fetches rather than backfill (spec §4 note); hidden-spike
pmf is a diagnostic + implicit in intraday training rather than an explicit
convolution; MOS may be dropped if the IEM endpoint resists (T6 fallback).

**Type consistency check:** `LEVELS` defined once in `models/baseline.py`,
imported everywhere; `quantiles` dicts are `{float_level: float_F}`;
`bucket_probs` consumes exactly that; `visible_running_max` returns
`(lo, hi) | None` and both consumers (T15, T16) handle `None`; all fetchers
return `obs`/`forecasts`/`alerts`-shaped dicts consumed by `db.upsert`.

