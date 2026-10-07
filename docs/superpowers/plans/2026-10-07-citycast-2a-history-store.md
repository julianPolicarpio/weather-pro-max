# citycast 2a — History Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data/history.db` and the `citycast backfill` / `citycast coverage` commands that archive NWS CLI truth, MOS runs, METAR observations, and Kalshi settled markets plus hourly candles for all 24 stations.

**Architecture:** New package `src/citycast/` beside `highcast` and `sandbox`. One SQLite file (`data/history.db`) with the schema from spec §2.3. Each source module turns its downloads into "chunks" (one resumable unit of work); a single runner executes chunks, upserts rows in one transaction per chunk, and records finished chunks in `backfill_log`. A paced HTTP client enforces per-host rate limits and backs off on throttling.

**Tech Stack:** Python 3.12, sqlite3, httpx, PyYAML, pytest. Reuses `highcast` parsers by import.

**Spec:** `docs/superpowers/specs/2026-10-07-city-models-and-laya-design.md` (§1, §2).

## Global Constraints

- Git repo: commit only when Julian asks. Tasks end with the full suite, not a commit.
- Tests: `.venv/bin/python -m pytest` (never plain `.venv/bin/pytest`). Unit tests never touch the network: inject a fake `fetch`.
- Never write `data/highcast.db` or `data/sandbox.db`. Never import `highcast.kalshi_client` anywhere under `src/citycast/`.
- `data/history.db`: WAL, `busy_timeout` 30000 ms.
- Pacing: IEM (`mesonet.agron.iastate.edu`) ≤ 1 request/s; Kalshi (`api.elections.kalshi.com`) ≤ 5 requests/s (0.2 s). On HTTP 429 or a body containing "Too many requests": back off 5 s, 15 s, 60 s, then fail the chunk.
- Kalshi is read unauthenticated only. Base URL `https://api.elections.kalshi.com/trade-api/v2`.
- Timestamps stored as UTC ISO strings with `+00:00`.
- CLI truth = final reports only (`final = 1`); the climate day uses standard time (`std_utc_offset`).
- Never derive a daily extreme from spot METAR readings alone; the `obs` table keeps `six_max_c`/`six_min_c`.
- A market's YES interval: strike fields via `highcast.fetchers.kalshi._lo_hi`, else rules text, else unusable (`lo` and `hi` both NULL). Station and metric come from the series alias table, never the rules text.

## File structure

| File | Responsibility |
|---|---|
| `src/citycast/__init__.py` | package marker |
| `src/citycast/__main__.py` | `python -m citycast` → `cli.main` |
| `src/citycast/db.py` | schema, `connect`, `upsert` |
| `src/citycast/stations.py` | registry loader, series index, `app_config` |
| `src/citycast/http.py` | `Paced` client with per-host pacing and backoff |
| `src/citycast/backfill.py` | `Chunk`, `run_chunks` (resume + log) |
| `src/citycast/sources/cli_reports.py` | IEM CLI → `cli_daily` |
| `src/citycast/sources/mos.py` | IEM MOS CSV → `mos_runs` |
| `src/citycast/sources/obs.py` | IEM ASOS → `obs` |
| `src/citycast/sources/kalshi_markets.py` | Kalshi settled markets → `kalshi_markets` |
| `src/citycast/sources/kalshi_candles.py` | Kalshi hourly candles → `kalshi_candles` |
| `src/citycast/coverage.py` | coverage report + acceptance |
| `src/citycast/cli.py` | `citycast init | backfill | coverage` |
| `config/citycast_stations.yaml` | generated + hand-reviewed registry |
| `scripts/build_citycast_stations.py` | one-off registry generator |

---

### Task 1: Package, schema, CLI skeleton

**Files:**
- Create: `src/citycast/__init__.py`, `src/citycast/__main__.py`, `src/citycast/db.py`, `src/citycast/cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`, `[tool.hatch.build.targets.wheel]`)
- Test: `tests/test_citycast_db.py`

**Interfaces:**
- Produces: `citycast.db.DEFAULT_PATH: Path`, `citycast.db.connect(path) -> sqlite3.Connection` (row_factory `sqlite3.Row`), `citycast.db.upsert(conn, table: str, rows: list[dict]) -> int` (does not commit; caller wraps in `with conn:`), `citycast.cli.main(argv: list[str] | None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_db.py
from contextlib import closing

from citycast import db

TABLES = {"stations", "cli_daily", "mos_runs", "obs", "kalshi_markets",
          "kalshi_candles", "backfill_log", "meta"}


def test_connect_creates_schema_wal_and_timeout(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert TABLES <= names
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_obs_columns_match_highcast_obs(tmp_path):
    """latentmax.visible_running_extremum must run unchanged against history.db."""
    from highcast import db as hdb
    with closing(db.connect(tmp_path / "h.db")) as c, \
            closing(hdb.connect(tmp_path / "hc.db")) as hc:
        cols = lambda conn: {r[1] for r in conn.execute("PRAGMA table_info(obs)")}
        assert cols(c) == cols(hc)


def test_upsert_replaces_on_primary_key(tmp_path):
    row = {"station": "KNYC", "date": "2024-01-01", "high_f": 40, "low_f": 30, "final": 1,
           "issued_utc": None, "raw": "{}", "fetched_at": "2026-10-07T00:00:00+00:00"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            assert db.upsert(c, "cli_daily", [row]) == 1
            db.upsert(c, "cli_daily", [{**row, "high_f": 41}])
        assert [tuple(r) for r in c.execute("SELECT high_f FROM cli_daily")] == [(41,)]
        assert db.upsert(c, "cli_daily", []) == 0


def test_cli_init_creates_db(tmp_path, capsys):
    from citycast.cli import main
    p = tmp_path / "sub" / "h.db"
    assert main(["--db", str(p), "init"]) == 0
    assert p.exists() and '"status": "ok"' in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_db.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'citycast'`.

- [ ] **Step 3: Implement**

```python
# src/citycast/__init__.py
"""citycast: multi-city history store, forecasts and decision data (spec 2026-10-07)."""
```

```python
# src/citycast/__main__.py
import sys

from citycast.cli import main

sys.exit(main())
```

```python
# src/citycast/db.py
"""data/history.db — the multi-city archive behind citycast (spec 2026-10-07 §2.3)."""
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path("data/history.db")
BUSY_TIMEOUT_MS = 30_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stations (icao TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cli_daily (station TEXT, date TEXT, high_f INTEGER, low_f INTEGER,
  final INTEGER NOT NULL, issued_utc TEXT, raw TEXT, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, date));
CREATE TABLE IF NOT EXISTS mos_runs (station TEXT, model TEXT, runtime_utc TEXT, target_date TEXT,
  metric TEXT, value_f REAL, n_points INTEGER NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (station, model, runtime_utc, target_date, metric));
CREATE TABLE IF NOT EXISTS obs (
  station TEXT NOT NULL, valid_utc TEXT NOT NULL, source TEXT NOT NULL, tmpf REAL, tmpf_precise REAL,
  dwpf REAL, sknt REAL, drct REAL, skyc TEXT, six_max_c REAL, raw TEXT, fetched_at TEXT NOT NULL,
  six_min_c REAL, slp_hpa REAL, pres_tend_hpa REAL, pres_tend_code INTEGER, vis_sm REAL,
  ceiling_ft INTEGER, sky_lowest_ft INTEGER, n_sky_layers INTEGER, sky_cb INTEGER, wx_present TEXT,
  wx_thunder INTEGER, PRIMARY KEY (station, valid_utc, source));
CREATE INDEX IF NOT EXISTS idx_obs_valid ON obs (station, valid_utc);
CREATE TABLE IF NOT EXISTS kalshi_markets (ticker TEXT PRIMARY KEY, series TEXT NOT NULL,
  event_ticker TEXT NOT NULL, station TEXT NOT NULL, metric TEXT NOT NULL, event_date TEXT NOT NULL,
  lo INTEGER, hi INTEGER, open_time TEXT, close_time TEXT NOT NULL, result TEXT,
  expiration_value TEXT, raw TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_km_event ON kalshi_markets (station, metric, event_date);
CREATE TABLE IF NOT EXISTS kalshi_candles (ticker TEXT, end_utc TEXT, yes_bid REAL, yes_ask REAL,
  volume REAL, open_interest REAL, PRIMARY KEY (ticker, end_utc));
CREATE TABLE IF NOT EXISTS backfill_log (source TEXT, key TEXT, done_at TEXT NOT NULL,
  rows INTEGER NOT NULL, PRIMARY KEY (source, key));
"""


def connect(path=DEFAULT_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE every row (columns from the first row). Never commits."""
    if not rows:
        return 0
    cols = list(rows[0])
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({','.join('?' * len(cols))})")
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)
```

```python
# src/citycast/cli.py
"""citycast init | backfill | coverage (backfill/coverage added in Task 9)."""
import argparse
import json
from contextlib import closing

from citycast import db


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citycast")
    p.add_argument("--db", default=str(db.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    args = p.parse_args(argv)
    if args.cmd == "init":
        with closing(db.connect(args.db)):
            pass
        print(json.dumps({"status": "ok", "db": args.db}))
        return 0
    return 2
```

In `pyproject.toml`, add the script and package:

```toml
[project.scripts]
highcast = "highcast.cli:main"
sandbox = "sandbox.cli:main"
citycast = "citycast.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/highcast", "src/sandbox", "src/citycast"]
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_db.py`
Expected: 4 passed.

- [ ] **Step 5: Install the console script**

Run: `.venv/bin/pip install -e . --no-deps -q && .venv/bin/citycast --db /tmp/citycast-smoke.db init && rm -f /tmp/citycast-smoke.db*`
Expected: `{"status": "ok", "db": "/tmp/citycast-smoke.db"}`. If pip cannot fetch hatchling, use `.venv/bin/python -m citycast` everywhere instead and note it in the report.

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (baseline 1462 passed + 4 new).

---

### Task 2: Station registry

**Files:**
- Create: `src/citycast/stations.py`, `scripts/build_citycast_stations.py`, `config/citycast_stations.yaml` (generated by the script, then reviewed)
- Test: `tests/test_citycast_stations.py`

**Interfaces:**
- Consumes: `sandbox.stations.STATIONS`, `sandbox.kalshi_public.PacedGet / list_temperature_series / open_markets`, `highcast.config.AppConfig / StationConfig`.
- Produces: `Station` dataclass (fields below), `load_stations(path=REGISTRY_PATH) -> dict[str, Station]`, `series_index(stations) -> dict[str, tuple[str, str]]` (series or alias ticker → (icao, metric)), `app_config(st: Station, db_path) -> AppConfig`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_stations.py
from pathlib import Path

import pytest

from citycast.stations import Station, app_config, load_stations, series_index
from sandbox.stations import STATIONS


def test_registry_matches_sandbox_stations():
    st = load_stations()
    assert set(st) == set(STATIONS) and len(st) == 24
    for icao, s in st.items():
        assert (s.tz, s.std_utc_offset) == STATIONS[icao]
        assert s.iem_id == icao[1:]
        assert s.high_series.startswith("KXHIGH") and s.low_series.startswith("KXLOWT")


def test_primary_series_are_48_distinct_and_index_has_no_duplicates():
    st = load_stations()
    primaries = {s.high_series for s in st.values()} | {s.low_series for s in st.values()}
    assert len(primaries) == 48
    idx = series_index(st)
    assert idx[st["KNYC"].high_series] == ("KNYC", "high")
    assert idx[st["KNYC"].low_series] == ("KNYC", "low")


def _st(icao="KNYC", **kw):
    base = dict(icao=icao, iem_id=icao[1:], iem_network="NY_ASOS", wfo="OKX", lat=40.78,
                lon=-73.97, elevation_m=40.0, tz="America/New_York", std_utc_offset=-5,
                high_series="KXHIGHNY", low_series="KXLOWTNYC")
    return Station(**{**base, **kw})


def test_series_index_includes_aliases_and_rejects_double_mapping():
    a = _st(high_aliases=("KXHIGHOLDNY",))
    assert series_index({"KNYC": a})["KXHIGHOLDNY"] == ("KNYC", "high")
    b = _st("KXXX", high_series="KXHIGHNY", low_series="KXLOWTXXX")
    with pytest.raises(ValueError, match="KXHIGHNY"):
        series_index({"KNYC": a, "KXXX": b})


def test_app_config_carries_station_fields(tmp_path):
    cfg = app_config(_st(), tmp_path / "h.db")
    assert cfg.station.icao == "KNYC" and cfg.station.iem_id == "NYC"
    assert cfg.station.std_utc_offset == -5 and cfg.db_path == tmp_path / "h.db"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_stations.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'citycast.stations'`.

- [ ] **Step 3: Implement the loader**

```python
# src/citycast/stations.py
"""The 24 settlement stations (config/citycast_stations.yaml, spec §2.2)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from highcast.config import AppConfig, StationConfig

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "citycast_stations.yaml"


@dataclass(frozen=True)
class Station:
    icao: str
    iem_id: str
    iem_network: str
    wfo: str
    lat: float
    lon: float
    elevation_m: float
    tz: str
    std_utc_offset: int
    high_series: str
    low_series: str
    high_aliases: tuple[str, ...] = ()
    low_aliases: tuple[str, ...] = ()


def load_stations(path: Path = REGISTRY_PATH) -> dict[str, Station]:
    raw = yaml.safe_load(Path(path).read_text())
    out: dict[str, Station] = {}
    for entry in raw["stations"]:
        e = dict(entry)
        e["high_aliases"] = tuple(e.get("high_aliases") or ())
        e["low_aliases"] = tuple(e.get("low_aliases") or ())
        out[e["icao"]] = Station(**e)
    return out


def series_index(stations: dict[str, Station]) -> dict[str, tuple[str, str]]:
    idx: dict[str, tuple[str, str]] = {}
    for st in stations.values():
        for metric, primary, aliases in (("high", st.high_series, st.high_aliases),
                                         ("low", st.low_series, st.low_aliases)):
            for s in (primary, *aliases):
                if s in idx:
                    raise ValueError(f"series {s} mapped twice")
                idx[s] = (st.icao, metric)
    return idx


def app_config(st: Station, db_path) -> AppConfig:
    """A minimal highcast AppConfig so highcast helpers (latentmax, parsers) run per station."""
    return AppConfig(
        station=StationConfig(key=st.icao.lower(), name=st.icao, icao=st.icao, iem_id=st.iem_id,
                              iem_network=st.iem_network, wfo=st.wfo, lat=st.lat, lon=st.lon,
                              tz=st.tz, std_utc_offset=st.std_utc_offset),
        display_tz="Asia/Manila", synoptic_token="", db_path=Path(db_path))
```

- [ ] **Step 4: Write the generator script**

```python
# scripts/build_citycast_stations.py
"""One-off: build config/citycast_stations.yaml from live Kalshi series + IEM metadata.

Run, then REVIEW the output by hand (spec §2.2). Re-run only when Kalshi adds or renames series.
"""
import sys
import time
from pathlib import Path

import httpx
import yaml

from sandbox.kalshi_public import BASE, PacedGet, list_temperature_series, open_markets
from sandbox.stations import STATIONS, metric_from_series, station_from_rules

STATES = {
    "KATL": "GA", "KAUS": "TX", "KBOS": "MA", "KDCA": "VA", "KDEN": "CO", "KDFW": "TX",
    "KEWR": "NJ", "KHOU": "TX", "KLAS": "NV", "KLAX": "CA", "KMDW": "IL", "KMIA": "FL",
    "KMSP": "MN", "KMSY": "LA", "KNYC": "NY", "KOKC": "OK", "KPHL": "PA", "KPHX": "AZ",
    "KSAN": "CA", "KSAT": "TX", "KSDF": "KY", "KSEA": "WA", "KSFO": "CA", "KTTN": "NJ",
}
# Archive-only series that belong to a station but are no longer listed live
# (verified 2026-10-07: KXHIGHHOU holds 70 Houston events). Each is printed with its
# rules text below for a human to confirm before the YAML is trusted.
EXTRA_ALIASES = {"KXHIGHHOU": ("KHOU", "high")}
IEM_NETWORK = "https://mesonet.agron.iastate.edu/geojson/network.py"


def main() -> int:
    get = PacedGet()
    by_station: dict[str, dict[str, str]] = {}
    for s in list_temperature_series(get):
        markets = open_markets(get, s["ticker"]).get("markets") or []
        icao = next((station_from_rules(m.get("rules_primary")) for m in markets
                     if station_from_rules(m.get("rules_primary"))), None)
        if icao not in STATIONS:
            print(f"skip {s['ticker']} (station {icao})", file=sys.stderr)
            continue
        by_station.setdefault(icao, {})[metric_from_series(s["ticker"])] = s["ticker"]

    meta = {}
    for state in sorted(set(STATES.values())):
        j = httpx.get(IEM_NETWORK, params={"network": f"{state}_ASOS"}, timeout=60).json()
        for f in j["features"]:
            meta[(state, f["properties"]["sid"])] = f
        time.sleep(1.0)

    entries = []
    for icao in sorted(STATIONS):
        tz, off = STATIONS[icao]
        state = STATES[icao]
        f = meta[(state, icao[1:])]          # KeyError = wrong state network: fix STATES
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        if p["tzname"] != tz:
            raise SystemExit(f"{icao}: IEM tz {p['tzname']} != sandbox tz {tz}")
        aliases = {"high": [], "low": []}
        for alias, (a_icao, a_metric) in EXTRA_ALIASES.items():
            if a_icao == icao:
                aliases[a_metric].append(alias)
        entries.append({
            "icao": icao, "iem_id": icao[1:], "iem_network": f"{state}_ASOS", "wfo": p["wfo"],
            "lat": round(lat, 4), "lon": round(lon, 4), "elevation_m": float(p["elevation"]),
            "tz": tz, "std_utc_offset": off,
            "high_series": by_station[icao]["high"], "low_series": by_station[icao]["low"],
            "high_aliases": aliases["high"], "low_aliases": aliases["low"],
        })

    for alias in EXTRA_ALIASES:
        j = httpx.get(f"{BASE}/historical/markets", params={"series_ticker": alias, "limit": 1},
                      timeout=60).json()
        rules = (j.get("markets") or [{}])[0].get("rules_primary")
        print(f"CHECK ALIAS {alias}: {rules}", file=sys.stderr)

    out = Path("config/citycast_stations.yaml")
    out.write_text("# Generated by scripts/build_citycast_stations.py, then reviewed by hand.\n"
                   + yaml.safe_dump({"stations": entries}, sort_keys=False))
    print(f"wrote {out} ({len(entries)} stations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Generate and review the registry**

Run: `.venv/bin/python scripts/build_citycast_stations.py`
Expected: `wrote config/citycast_stations.yaml (24 stations)` and one `CHECK ALIAS KXHIGHHOU: …` line. Review: the alias rules text must name a Houston station that settles on the same CLI as `KXHIGHTHOU` (Hobby, CLIHOU). If it names a different station (e.g. Bush Intercontinental), delete the `KXHIGHHOU` alias from the YAML by hand and say so in the report. If a station raises `KeyError`, look up its IEM network on mesonet.agron.iastate.edu and correct `STATES`.

- [ ] **Step 6: Run the tests and the full suite**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_stations.py && .venv/bin/python -m pytest -q`
Expected: 4 passed; full suite passes.

---

### Task 3: Paced HTTP client and chunk runner

**Files:**
- Create: `src/citycast/http.py`, `src/citycast/backfill.py`
- Test: `tests/test_citycast_http.py`, `tests/test_citycast_backfill.py`

**Interfaces:**
- Consumes: `citycast.db.upsert`.
- Produces: `Paced(host_interval=None, fetch=..., sleep=time.sleep, clock=time.monotonic, backoffs=(5, 15, 60))` with `.get_text(url, params=None) -> str` and `.get_json(url, params=None) -> dict`; exceptions `HttpError(status, url)` (attribute `.status`) and `RateLimited(HttpError)`. `Chunk(source: str, key: str, current: bool, run: Callable[[], list[tuple[str, list[dict]]]])`; `run_chunks(conn, chunks, *, refresh=False, now_iso, progress=print) -> dict` with keys `done, skipped, rows, errors` (errors = list of `{"source","key","error"}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_http.py
import pytest

from citycast.http import HttpError, Paced, RateLimited

IEM = "https://mesonet.agron.iastate.edu/x"
KAL = "https://api.elections.kalshi.com/trade-api/v2/x"


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _paced(responses, clk):
    calls = []

    def fetch(url, params):
        calls.append(url)
        return responses.pop(0)
    return Paced(fetch=fetch, sleep=clk.sleep, clock=clk.clock), calls


def test_paces_per_host():
    clk = Clock()
    p, _ = _paced([(200, "a"), (200, "b"), (200, "c")], clk)
    p.get_text(IEM)
    p.get_text(KAL)          # other host: no wait
    p.get_text(IEM)          # same host again: waits the rest of 1.0 s
    assert clk.sleeps == [1.0]


def test_backs_off_on_429_and_too_many_requests_then_succeeds():
    clk = Clock()
    p, calls = _paced([(429, ""), (200, "Too many requests from your IP"), (200, "ok")], clk)
    assert p.get_text(KAL) == "ok"
    assert len(calls) == 3
    assert [s for s in clk.sleeps if s >= 5] == [5, 15]


def test_raises_rate_limited_after_backoffs_exhausted():
    clk = Clock()
    p, _ = _paced([(429, "")] * 4, clk)
    with pytest.raises(RateLimited):
        p.get_text(KAL)


def test_http_error_carries_status():
    clk = Clock()
    p, _ = _paced([(404, "not found")], clk)
    with pytest.raises(HttpError) as e:
        p.get_text(KAL)
    assert e.value.status == 404


def test_get_json_parses():
    clk = Clock()
    p, _ = _paced([(200, '{"a": 1}')], clk)
    assert p.get_json(KAL) == {"a": 1}
```

```python
# tests/test_citycast_backfill.py
from contextlib import closing

from citycast import db
from citycast.backfill import Chunk, run_chunks

NOW = lambda: "2026-10-07T00:00:00+00:00"   # noqa: E731


def _row(day, high):
    return {"station": "KNYC", "date": day, "high_f": high, "low_f": 1, "final": 1,
            "issued_utc": None, "raw": "{}", "fetched_at": NOW()}


def test_runs_logs_and_skips_done_chunks(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return [("cli_daily", [_row("2024-01-01", 40)])]
    with closing(db.connect(tmp_path / "h.db")) as c:
        s1 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], now_iso=NOW, progress=lambda m: None)
        s2 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], now_iso=NOW, progress=lambda m: None)
        s3 = run_chunks(c, [Chunk("cli", "KNYC:2024", False, run)], refresh=True, now_iso=NOW,
                        progress=lambda m: None)
        assert (s1["done"], s2["skipped"], s3["done"], len(calls)) == (1, 1, 1, 2)
        assert c.execute("SELECT rows FROM backfill_log").fetchone()[0] == 1


def test_current_chunk_always_runs_and_is_never_logged(tmp_path):
    run = lambda: [("cli_daily", [_row("2026-10-01", 70)])]   # noqa: E731
    with closing(db.connect(tmp_path / "h.db")) as c:
        for _ in range(2):
            s = run_chunks(c, [Chunk("cli", "KNYC:2026", True, run)], now_iso=NOW, progress=lambda m: None)
            assert s["done"] == 1
        assert c.execute("SELECT COUNT(*) FROM backfill_log").fetchone()[0] == 0


def test_error_is_reported_not_logged_and_does_not_stop_others(tmp_path):
    def bad():
        raise RuntimeError("boom")
    good = lambda: [("cli_daily", [_row("2024-01-02", 41)])]   # noqa: E731
    with closing(db.connect(tmp_path / "h.db")) as c:
        s = run_chunks(c, [Chunk("cli", "A", False, bad), Chunk("cli", "B", False, good)],
                       now_iso=NOW, progress=lambda m: None)
        assert s["done"] == 1 and s["errors"] == [
            {"source": "cli", "key": "A", "error": "RuntimeError: boom"}]
        assert [r[0] for r in c.execute("SELECT key FROM backfill_log")] == ["B"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_http.py tests/test_citycast_backfill.py`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement**

```python
# src/citycast/http.py
"""Paced, throttle-aware GET for IEM and Kalshi (spec §2.4)."""
import json
import time
from urllib.parse import urlparse

import httpx

IEM_HOST = "mesonet.agron.iastate.edu"
KALSHI_HOST = "api.elections.kalshi.com"
DEFAULT_INTERVALS = {IEM_HOST: 1.0, KALSHI_HOST: 0.2}
BACKOFFS_S = (5, 15, 60)


class HttpError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} {url}")
        self.status = status


class RateLimited(HttpError):
    pass


def _httpx_fetch(url: str, params: dict | None) -> tuple[int, str]:
    r = httpx.get(url, params=params, timeout=120, follow_redirects=True,
                  headers={"User-Agent": "citycast/0.1 (research archive)"})
    return r.status_code, r.text


def _throttled(status: int, text: str) -> bool:
    return status == 429 or "too many requests" in text[:300].lower()


class Paced:
    def __init__(self, host_interval: dict[str, float] | None = None, fetch=_httpx_fetch,
                 sleep=time.sleep, clock=time.monotonic, backoffs=BACKOFFS_S):
        self.host_interval = dict(DEFAULT_INTERVALS if host_interval is None else host_interval)
        self.fetch, self.sleep, self.clock, self.backoffs = fetch, sleep, clock, tuple(backoffs)
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            delay = self.host_interval.get(host, 1.0) - (self.clock() - last)
            if delay > 0:
                self.sleep(delay)
        self._last[host] = self.clock()

    def get_text(self, url: str, params: dict | None = None) -> str:
        host = urlparse(url).hostname or ""
        for attempt in range(len(self.backoffs) + 1):
            self._wait(host)
            status, text = self.fetch(url, params)
            if _throttled(status, text):
                if attempt < len(self.backoffs):
                    self.sleep(self.backoffs[attempt])
                    continue
                raise RateLimited(status, url)
            if status >= 400:
                raise HttpError(status, url)
            return text
        raise RateLimited(429, url)   # unreachable; keeps type checkers honest

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.get_text(url, params))
```

```python
# src/citycast/backfill.py
"""Resumable chunk runner (spec §2.4): one transaction and one log row per finished chunk."""
from dataclasses import dataclass
from typing import Callable

from citycast import db


@dataclass(frozen=True)
class Chunk:
    source: str
    key: str
    current: bool       # still filling (e.g. this year): always re-run, never logged as done
    run: Callable[[], list[tuple[str, list[dict]]]]


def is_logged(conn, source: str, key: str) -> bool:
    return conn.execute("SELECT 1 FROM backfill_log WHERE source=? AND key=?",
                        (source, key)).fetchone() is not None


def run_chunks(conn, chunks, *, refresh: bool = False, now_iso: Callable[[], str],
               progress=print) -> dict:
    summary = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
    for ch in chunks:
        if not refresh and not ch.current and is_logged(conn, ch.source, ch.key):
            summary["skipped"] += 1
            continue
        try:
            results = ch.run()
        except Exception as e:  # noqa: BLE001 — one bad chunk must not stop the backfill
            summary["errors"].append({"source": ch.source, "key": ch.key,
                                      "error": f"{type(e).__name__}: {e}"})
            progress(f"ERROR {ch.source} {ch.key}: {type(e).__name__}: {e}")
            continue
        n = 0
        with conn:
            for table, rows in results:
                n += db.upsert(conn, table, rows)
            if not ch.current:
                conn.execute("INSERT OR REPLACE INTO backfill_log VALUES (?,?,?,?)",
                             (ch.source, ch.key, now_iso(), n))
        summary["done"] += 1
        summary["rows"] += n
        progress(f"{ch.source} {ch.key}: {n} rows")
    return summary
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_http.py tests/test_citycast_backfill.py`
Expected: 8 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest -q` — Expected: all pass.

---

### Task 4: NWS CLI source

**Files:**
- Create: `src/citycast/sources/__init__.py` (empty), `src/citycast/sources/cli_reports.py`
- Test: `tests/test_citycast_source_cli.py`

**Interfaces:**
- Consumes: `highcast.fetchers.cli_reports.parse_rows(payload, icao, fetched_at)`, `cli_row_is_final(product, day, std_utc_offset)`, `_product_issued_utc(product)`; `Station`; `Chunk`; `Paced`.
- Produces: `IEM_CLI: str`, `cli_rows(payload: dict, st: Station, fetched_at: str) -> list[dict]` (columns of `cli_daily`), `chunks(paced, stations: dict[str, Station], since: date, today: date, now_iso) -> list[Chunk]` (source `"cli"`, key `"KNYC:2024"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_source_cli.py
import json
from datetime import date

from citycast.sources.cli_reports import chunks, cli_rows
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
PAYLOAD = {"results": [
    # issued the next morning: final
    {"valid": "2024-01-05", "high": 40, "low": 30, "product": "202401060714-KOKX-CDUS41-CLINYC"},
    # issued during the day itself: preliminary
    {"valid": "2024-01-06", "high": 38, "low": 29, "product": "202401062130-KOKX-CDUS41-CLINYC"},
    {"valid": "2024-01-07", "high": "M", "low": 20, "product": "202401080700-KOKX-CDUS41-CLINYC"},
]}


def test_cli_rows_marks_final_and_issue_time():
    rows = {r["date"]: r for r in cli_rows(PAYLOAD, ST, "2026-10-07T00:00:00+00:00")}
    assert set(rows) == {"2024-01-05", "2024-01-06"}       # missing high dropped
    assert rows["2024-01-05"]["final"] == 1 and rows["2024-01-06"]["final"] == 0
    assert rows["2024-01-05"]["issued_utc"] == "2024-01-06T07:14:00+00:00"
    assert rows["2024-01-05"]["high_f"] == 40 and rows["2024-01-05"]["low_f"] == 30
    assert json.loads(rows["2024-01-05"]["raw"])["valid"] == "2024-01-05"


def test_chunks_per_station_year_and_current_year_is_current():
    seen = []

    class FakePaced:
        def get_json(self, url, params):
            seen.append(params)
            return PAYLOAD
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2024, 1, 1), date(2026, 10, 7),
                lambda: "2026-10-07T00:00:00+00:00")
    assert [c.key for c in cs] == ["KNYC:2024", "KNYC:2025", "KNYC:2026"]
    assert [c.current for c in cs] == [False, False, True]
    table, rows = cs[0].run()[0]
    assert table == "cli_daily" and len(rows) == 2
    assert seen == [{"station": "KNYC", "year": 2024}]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_source_cli.py`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/citycast/sources/__init__.py
```

```python
# src/citycast/sources/cli_reports.py
"""IEM CLI archive -> cli_daily. Truth is final reports only (spec §1)."""
import json
from datetime import date

from citycast.backfill import Chunk
from highcast.fetchers.cli_reports import _product_issued_utc, cli_row_is_final, parse_rows

IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"


def cli_rows(payload: dict, st, fetched_at: str) -> list[dict]:
    out = []
    for r in parse_rows(payload, st.icao, fetched_at):
        product = json.loads(r["raw"]).get("product")
        issued = _product_issued_utc(product)
        out.append({"station": st.icao, "date": r["date"], "high_f": r["high_f"],
                    "low_f": r["low_f"],
                    "final": int(cli_row_is_final(product, r["date"], st.std_utc_offset)),
                    "issued_utc": issued.isoformat() if issued else None,
                    "raw": r["raw"], "fetched_at": fetched_at})
    return out


def chunks(paced, stations: dict, since: date, today: date, now_iso) -> list[Chunk]:
    out = []
    # January corrections can still land on last year's December reports.
    first_current = today.year - 1 if today.month == 1 else today.year
    for st in stations.values():
        for year in range(since.year, today.year + 1):
            def run(st=st, year=year):
                payload = paced.get_json(IEM_CLI, {"station": st.icao, "year": year})
                return [("cli_daily", cli_rows(payload, st, now_iso()))]
            out.append(Chunk("cli", f"{st.icao}:{year}", year >= first_current, run))
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_source_cli.py`
Expected: 2 passed. If `final` for 2024-01-05 comes back 0, read `cli_row_is_final` in `src/highcast/fetchers/cli_reports.py` and adjust only the **test's product timestamps** so one is issued after the climate day ends and one before; do not change highcast.

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q`, all pass.

---

### Task 5: MOS runs source

**Files:**
- Create: `src/citycast/sources/mos.py`
- Test: `tests/test_citycast_source_mos.py`

**Interfaces:**
- Produces: `IEM_MOS: str`, `MODELS = ("MEX", "GFS", "NBS", "NBE")`, `parse_mos_runs(text: str, st: Station, model: str, fetched_at: str) -> list[dict]` (columns of `mos_runs`), `chunks(paced, stations, since: date, today: date, now_iso, models=MODELS) -> list[Chunk]` (source `"mos"`, key `"KNYC:MEX:2024-01-01"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_source_mos.py
from datetime import date

from citycast.sources.mos import chunks, parse_mos_runs
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"

MEX = """runtime,ftime,model,n_x,tmp
2024-01-04 12:00:00,2024-01-05 00:00:00,MEX,45,40
2024-01-04 12:00:00,2024-01-05 12:00:00,MEX,30,31
2024-01-04 12:00:00,2024-01-06 00:00:00,MEX,48,44
2024-01-04 12:00:00,2024-01-06 12:00:00,MEX,,33
"""

NBS = """runtime,ftime,model,tmp,dpt
2024-01-05 01:00:00,2024-01-05 06:00:00,NBS,30,20
2024-01-05 01:00:00,2024-01-05 09:00:00,NBS,28,20
2024-01-05 01:00:00,2024-01-05 18:00:00,NBS,41,22
2024-01-05 01:00:00,2024-01-06 03:00:00,NBS,35,22
2024-01-05 01:00:00,2024-01-06 06:00:00,NBS,M,22
"""


def test_mex_uses_n_x_afternoon_for_high_morning_for_low_by_climate_day():
    rows = {(r["target_date"], r["metric"]): r for r in parse_mos_runs(MEX, ST, "MEX", F)}
    # 00Z Jan 5 = 19:00 local Jan 4 (afternoon) -> high of climate day Jan 4
    assert rows[("2024-01-04", "high")]["value_f"] == 45
    # 12Z Jan 5 = 07:00 local (morning) -> low of Jan 5; 00Z Jan 6 -> high of Jan 5
    assert rows[("2024-01-05", "low")]["value_f"] == 30
    assert rows[("2024-01-05", "high")]["value_f"] == 48
    assert ("2024-01-06", "low") not in rows                      # blank n_x skipped
    r = rows[("2024-01-04", "high")]
    assert r["runtime_utc"] == "2024-01-04T12:00:00+00:00" and r["n_points"] == 1
    assert r["model"] == "MEX" and r["station"] == "KNYC"


def test_nbs_aggregates_tmp_within_the_climate_day():
    rows = {(r["target_date"], r["metric"]): r for r in parse_mos_runs(NBS, ST, "NBS", F)}
    # climate day Jan 5 = 05:00Z Jan 5 .. 05:00Z Jan 6 -> 06Z, 09Z, 18Z, 03Z rows
    assert rows[("2024-01-05", "high")]["value_f"] == 41
    assert rows[("2024-01-05", "low")]["value_f"] == 28
    assert rows[("2024-01-05", "high")]["n_points"] == 4


def test_chunks_half_years_per_model_and_last_is_current():
    asked = []

    class FakePaced:
        def get_text(self, url, params):
            asked.append(params)
            return MEX
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2025, 1, 1), date(2025, 9, 1), lambda: F,
                models=("MEX",))
    assert [c.key for c in cs] == ["KNYC:MEX:2025-01-01", "KNYC:MEX:2025-07-01"]
    assert [c.current for c in cs] == [False, True]
    cs[0].run()
    assert asked == [{"station": "KNYC", "model": "MEX", "sts": "2025-01-01T00:00Z",
                      "ets": "2025-07-01T00:00Z", "format": "csv"}]
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest -q tests/test_citycast_source_mos.py`, FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/sources/mos.py
"""IEM MOS archive CSV -> mos_runs, keeping each run's UTC issue time (spec §2.3).

Conventions match highcast.fetchers.forecasts: climate day = (ftime_utc + std offset).date();
MEX/GFS carry the daily max/min in `n_x` on local-afternoon/-morning rows; NBS/NBE have no
n_x, so the max/min is taken over the run's 3-hourly `tmp` points inside the climate day.
"""
import csv
import io
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from citycast.backfill import Chunk

IEM_MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
MODELS = ("MEX", "GFS", "NBS", "NBE")
N_X_MODELS = {"MEX", "GFS"}
_MISSING = (None, "", "M")


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(str(s).strip()[:19].replace(" ", "T")).replace(tzinfo=timezone.utc)


def parse_mos_runs(text: str, st, model: str, fetched_at: str) -> list[dict]:
    zone = ZoneInfo(st.tz)
    agg: dict[tuple[str, str, str], list[float]] = {}
    for item in csv.DictReader(io.StringIO(text)):
        runtime, ftime = item.get("runtime"), item.get("ftime")
        if not runtime or not ftime:
            continue
        ft = _utc(ftime)
        day = (ft + timedelta(hours=st.std_utc_offset)).date().isoformat()
        rt = _utc(runtime).isoformat()
        if model in N_X_MODELS:
            v = item.get("n_x")
            if v in _MISSING:
                continue
            metric = "high" if ft.astimezone(zone).hour >= 12 else "low"
            agg.setdefault((rt, day, metric), []).append(float(v))
        else:
            v = item.get("tmp")
            if v in _MISSING:
                continue
            for metric in ("high", "low"):
                agg.setdefault((rt, day, metric), []).append(float(v))
    return [{"station": st.icao, "model": model, "runtime_utc": rt, "target_date": day,
             "metric": metric, "value_f": max(v) if metric == "high" else min(v),
             "n_points": len(v), "fetched_at": fetched_at}
            for (rt, day, metric), v in sorted(agg.items())]


def _half_years(since: date, today: date):
    start = date(since.year, 1 if since.month <= 6 else 7, 1)
    while start <= today:
        end = date(start.year, 7, 1) if start.month == 1 else date(start.year + 1, 1, 1)
        yield start, end
        start = end


def chunks(paced, stations: dict, since: date, today: date, now_iso, models=MODELS) -> list[Chunk]:
    out = []
    for st in stations.values():
        for model in models:
            for a, b in _half_years(since, today):
                def run(st=st, model=model, a=a, b=b):
                    text = paced.get_text(IEM_MOS, {"station": st.icao, "model": model,
                                                    "sts": f"{a.isoformat()}T00:00Z",
                                                    "ets": f"{b.isoformat()}T00:00Z",
                                                    "format": "csv"})
                    return [("mos_runs", parse_mos_runs(text, st, model, now_iso()))]
                out.append(Chunk("mos", f"{st.icao}:{model}:{a.isoformat()}", b > today, run))
    return out
```

- [ ] **Step 4: Run to verify they pass** — 3 passed.

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q`, all pass.

---

### Task 6: METAR observations source

**Files:**
- Create: `src/citycast/sources/obs.py`
- Test: `tests/test_citycast_source_obs.py`

**Interfaces:**
- Consumes: `highcast.fetchers.metar.parse_metar(raw, valid_utc, icao, fetched_at) -> dict`, `highcast.timeutil.norm_valid`.
- Produces: `IEM_ASOS: str`, `obs_rows(text: str, st: Station, fetched_at: str) -> list[dict]`, `chunks(paced, stations, since, today, now_iso) -> list[Chunk]` (source `"obs"`, key `"KNYC:2024-01"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_source_obs.py
from datetime import date

from citycast.sources.obs import chunks, obs_rows
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"
TEXT = """station,valid,metar
NYC,2024-01-05 17:51,KNYC 051751Z 27008KT 10SM CLR 06/M06 A3012 RMK AO2 SLP199 T00561061 10067 20044
NYC,2024-01-05 18:51,KNYC 051851Z 28009KT 10SM CLR 05/M06 A3013 RMK AO2 SLP203 T00501061
"""


def test_obs_rows_parse_metar_including_six_hour_groups():
    rows = obs_rows(TEXT, ST, F)
    assert len(rows) == 2 and rows[0]["station"] == "KNYC" and rows[0]["source"] == "metar"
    assert rows[0]["valid_utc"] == "2024-01-05T17:51:00+00:00"
    assert rows[0]["six_max_c"] == 6.7 and rows[0]["six_min_c"] == 4.4


def test_chunks_are_months_with_current_month_current():
    asked = []

    class FakePaced:
        def get_text(self, url, params):
            asked.append(params)
            return TEXT
    cs = chunks(FakePaced(), {"KNYC": ST}, date(2026, 8, 1), date(2026, 10, 7), lambda: F)
    assert [c.key for c in cs] == ["KNYC:2026-08", "KNYC:2026-09", "KNYC:2026-10"]
    assert [c.current for c in cs] == [False, False, True]
    table, rows = cs[0].run()[0]
    assert table == "obs" and len(rows) == 2
    assert asked[0]["station"] == "NYC" and asked[0]["month1"] == 8 and asked[0]["month2"] == 9
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/sources/obs.py
"""IEM ASOS METAR archive -> obs (same parser and columns as highcast, incl. 6-hour groups)."""
from datetime import date

from citycast.backfill import Chunk
from highcast.fetchers.metar import parse_metar
from highcast.timeutil import norm_valid

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def obs_rows(text: str, st, fetched_at: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",", 2)
        if len(parts) == 3:
            _, valid, raw = parts
            rows.append(parse_metar(raw.strip(), norm_valid(valid.strip()), st.icao, fetched_at))
    return rows


def _months(since: date, today: date):
    cur = date(since.year, since.month, 1)
    while cur <= today:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt
        cur = nxt


def chunks(paced, stations: dict, since: date, today: date, now_iso) -> list[Chunk]:
    out = []
    for st in stations.values():
        for a, b in _months(since, today):
            def run(st=st, a=a, b=b):
                text = paced.get_text(IEM_ASOS, {
                    "station": st.iem_id, "data": "metar",
                    "year1": a.year, "month1": a.month, "day1": 1,
                    "year2": b.year, "month2": b.month, "day2": 1,
                    "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
                    "report_type": [3, 4]})
                return [("obs", obs_rows(text, st, now_iso()))]
            out.append(Chunk("obs", f"{st.icao}:{a.strftime('%Y-%m')}", b > today, run))
    return out
```

- [ ] **Step 4: Run to verify they pass** — 2 passed. If `valid_utc` differs in format, assert against `highcast.timeutil.norm_valid("2024-01-05 17:51")` instead of the literal and note it; if six-hour values differ by float rounding, use `pytest.approx`.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 7: Kalshi settled markets source

**Files:**
- Create: `src/citycast/sources/kalshi_markets.py`
- Test: `tests/test_citycast_source_kalshi_markets.py`

**Interfaces:**
- Consumes: `highcast.fetchers.kalshi._lo_hi`, `citycast.stations.series_index` output.
- Produces: `KALSHI: str`, `event_date(event_ticker: str) -> str | None`, `interval(m: dict) -> tuple[int | None, int | None] | None`, `market_rows(markets: list[dict], series: str, index: dict) -> list[dict]` (columns of `kalshi_markets`), `fetch_series_markets(paced, series: str) -> list[dict]`, `historical_cutoff(paced) -> str`, `chunks(paced, index: dict) -> list[Chunk]` (source `"kalshi-markets"`, key = series, always current).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_source_kalshi_markets.py
from citycast.sources.kalshi_markets import (chunks, event_date, fetch_series_markets,
                                             interval, market_rows)

IDX = {"KXHIGHNY": ("KNYC", "high")}


def test_event_date_handles_modern_legacy_and_double_dash():
    assert event_date("KXHIGHNY-26OCT04") == "2026-10-04"
    assert event_date("HIGHNY-21AUG06") == "2021-08-06"
    assert event_date("HIGHAUS--23MAY11") == "2023-05-11"
    assert event_date("NODATE") is None


def test_interval_from_strike_fields_matches_live_parser():
    assert interval({"strike_type": "between", "floor_strike": 66, "cap_strike": 67}) == (66, 67)
    assert interval({"strike_type": "greater", "floor_strike": 72}) == (73, None)
    assert interval({"strike_type": "less", "cap_strike": 60}) == (None, 59)


def test_interval_from_legacy_rules_text():
    gt = {"rules_primary": "If the highest temperature ... is strictly greater than 86°F, then ..."}
    lt = {"rules_primary": "... is strictly less than 50°F, then the market resolves to Yes."}
    bt = {"rules_primary": "... is between 66-67°, then the market resolves to Yes."}
    assert interval(gt) == (87, None) and interval(lt) == (None, 49) and interval(bt) == (66, 67)
    assert interval({"rules_primary": "something else"}) is None


def test_market_rows_station_from_series_and_unusable_kept():
    ms = [{"ticker": "HIGHNY-21AUG06-T86", "event_ticker": "HIGHNY-21AUG06",
           "rules_primary": "strictly greater than 86°F", "close_time": "2021-08-07T03:59:00Z",
           "open_time": "2021-08-05T14:00:00Z", "result": "yes", "expiration_value": "88"},
          {"ticker": "HIGHNY-21AUG06-X", "event_ticker": "HIGHNY-21AUG06",
           "rules_primary": "unparseable", "close_time": "2021-08-07T03:59:00Z", "result": "no"},
          {"ticker": "KXHIGHNY-26OCT04-B70.5", "event_ticker": "KXHIGHNY-26OCT04",
           "market_type": "scalar", "close_time": "2026-10-05T04:59:00Z"}]
    rows = {r["ticker"]: r for r in market_rows(ms, "KXHIGHNY", IDX)}
    assert set(rows) == {"HIGHNY-21AUG06-T86", "HIGHNY-21AUG06-X"}   # non-binary dropped
    a = rows["HIGHNY-21AUG06-T86"]
    assert (a["station"], a["metric"], a["event_date"], a["lo"], a["hi"]) == (
        "KNYC", "high", "2021-08-06", 87, None)
    assert a["close_time"] == "2021-08-07T03:59:00+00:00" and a["expiration_value"] == "88"
    b = rows["HIGHNY-21AUG06-X"]
    assert b["lo"] is None and b["hi"] is None


def test_fetch_series_markets_pages_both_endpoints_live_wins():
    pages = {
        ("/historical/markets", None): {"markets": [{"ticker": "A", "v": "hist"}], "cursor": "c1"},
        ("/historical/markets", "c1"): {"markets": [{"ticker": "B", "v": "hist"}], "cursor": ""},
        ("/markets", None): {"markets": [{"ticker": "B", "v": "live"}], "cursor": ""},
    }
    calls = []

    class FakePaced:
        def get_json(self, url, params):
            path = url.split("/trade-api/v2")[1]
            calls.append((path, params))
            return pages[(path, params.get("cursor"))]
    got = {m["ticker"]: m["v"] for m in fetch_series_markets(FakePaced(), "KXHIGHNY")}
    assert got == {"A": "hist", "B": "live"}
    assert calls[-1][1]["status"] == "settled"


def test_chunks_one_per_series_always_current():
    cs = chunks(object(), {"KXHIGHNY": ("KNYC", "high"), "KXLOWTNYC": ("KNYC", "low")})
    assert sorted(c.key for c in cs) == ["KXHIGHNY", "KXLOWTNYC"] and all(c.current for c in cs)
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/sources/kalshi_markets.py
"""Kalshi settled markets (archive + recent) -> kalshi_markets (spec §1 'Legacy market shape')."""
import json
import re
from datetime import datetime

from citycast.backfill import Chunk
from highcast.fetchers.kalshi import _lo_hi

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
_DATE = re.compile(r"(\d{2}[A-Z]{3}\d{2})")
_GT = re.compile(r"strictly greater than (-?\d+)")
_LT = re.compile(r"strictly less than (-?\d+)")
_BETWEEN = re.compile(r"between (-?\d+)\s*(?:-|–|to|and)\s*(-?\d+)")


def event_date(event_ticker: str) -> str | None:
    m = _DATE.search(event_ticker or "")
    if not m:
        return None
    return datetime.strptime(m.group(1), "%y%b%d").date().isoformat()


def interval(m: dict) -> tuple[int | None, int | None] | None:
    if m.get("floor_strike") is not None or m.get("cap_strike") is not None:
        return _lo_hi(m.get("floor_strike"), m.get("cap_strike"))
    rules = m.get("rules_primary") or ""
    if g := _GT.search(rules):
        return int(g.group(1)) + 1, None
    if lt := _LT.search(rules):
        return None, int(lt.group(1)) - 1
    if b := _BETWEEN.search(rules):
        return int(b.group(1)), int(b.group(2))
    return None


def _utc(ts: str | None) -> str | None:
    if not ts:
        return None
    return ts[:-1] + "+00:00" if ts.endswith("Z") else ts


def market_rows(markets: list[dict], series: str, index: dict) -> list[dict]:
    icao, metric = index[series]
    rows = []
    for m in markets:
        if m.get("market_type") not in (None, "binary"):
            continue
        d = event_date(m.get("event_ticker", ""))
        if d is None or not m.get("ticker") or not m.get("close_time"):
            continue
        lo, hi = interval(m) or (None, None)
        rows.append({"ticker": m["ticker"], "series": series, "event_ticker": m["event_ticker"],
                     "station": icao, "metric": metric, "event_date": d, "lo": lo, "hi": hi,
                     "open_time": _utc(m.get("open_time")), "close_time": _utc(m["close_time"]),
                     "result": m.get("result"), "expiration_value": m.get("expiration_value"),
                     "raw": json.dumps(m, sort_keys=True)})
    return rows


def fetch_series_markets(paced, series: str) -> list[dict]:
    """Archive first, then recent settled; a ticker in both keeps the recent copy."""
    out: dict[str, dict] = {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cursor = None
        while True:
            params = {"series_ticker": series, "limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor
            j = paced.get_json(KALSHI + path, params)
            ms = j.get("markets") or []
            for m in ms:
                out[m["ticker"]] = m
            cursor = j.get("cursor")
            if not cursor or not ms:
                break
    return list(out.values())


def historical_cutoff(paced) -> str:
    return _utc(paced.get_json(KALSHI + "/historical/cutoff")["market_settled_ts"])


def chunks(paced, index: dict) -> list[Chunk]:
    out = []
    for series in sorted(index):
        def run(series=series):
            return [("kalshi_markets", market_rows(fetch_series_markets(paced, series), series, index))]
        out.append(Chunk("kalshi-markets", series, True, run))
    return out
```

- [ ] **Step 4: Run to verify they pass** — 6 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 8: Kalshi hourly candles source

**Files:**
- Create: `src/citycast/sources/kalshi_candles.py`
- Test: `tests/test_citycast_source_kalshi_candles.py`

**Interfaces:**
- Consumes: `KALSHI`, `historical_cutoff` (Task 7), `HttpError` (Task 3), `kalshi_markets` rows.
- Produces: `candle_rows(ticker: str, payload: dict) -> list[dict]`, `fetch_candles(paced, ticker, series, open_iso, close_iso, cutoff_iso) -> dict` (merged payload), `chunks(paced, conn, now_iso) -> list[Chunk]` (source `"kalshi-candles"`, key = ticker, never current; only markets with a usable interval and `close_time < now`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_source_kalshi_candles.py
from contextlib import closing

from citycast import db
from citycast.http import HttpError
from citycast.sources.kalshi_candles import candle_rows, chunks, fetch_candles

PAYLOAD = {"candlesticks": [
    {"end_period_ts": 1736056800, "volume": "10.00", "open_interest": "46.00",
     "yes_bid": {"close": "0.3000"}, "yes_ask": {"close": "0.3200"}, "price": {"close": None}},
    {"end_period_ts": 1736060400, "volume": "0.00", "open_interest": "46.00",
     "yes_bid": {"close": None}, "yes_ask": {"close": "0.3300"}},
]}


def test_candle_rows_keep_close_bid_ask():
    rows = candle_rows("T", PAYLOAD)
    assert rows[0] == {"ticker": "T", "end_utc": "2025-01-05T06:00:00+00:00", "yes_bid": 0.30,
                       "yes_ask": 0.32, "volume": 10.0, "open_interest": 46.0}
    assert rows[1]["yes_bid"] is None and rows[1]["yes_ask"] == 0.33


def test_fetch_uses_historical_before_cutoff_and_falls_back_on_404():
    calls = []

    class FakePaced:
        def get_json(self, url, params):
            calls.append(url.split("/trade-api/v2")[1])
            if "/historical/" in url:
                raise HttpError(404, url)
            return PAYLOAD
    got = fetch_candles(FakePaced(), "T", "KXHIGHLAX", "2025-01-04T20:00:00+00:00",
                        "2025-01-06T07:59:00+00:00", "2026-08-08T00:00:00+00:00")
    assert calls == ["/historical/markets/T/candlesticks", "/series/KXHIGHLAX/markets/T/candlesticks"]
    assert len(got["candlesticks"]) == 2


def test_chunks_only_closed_usable_markets(tmp_path):
    base = {"series": "KXHIGHLAX", "event_ticker": "E", "station": "KLAX", "metric": "high",
            "event_date": "2025-01-05", "open_time": "2025-01-04T20:00:00+00:00", "result": "no",
            "expiration_value": "", "raw": "{}"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [
                {**base, "ticker": "OK", "lo": 66, "hi": 67, "close_time": "2025-01-06T07:59:00+00:00"},
                {**base, "ticker": "UNUSABLE", "lo": None, "hi": None, "close_time": "2025-01-06T07:59:00+00:00"},
                {**base, "ticker": "OPEN", "lo": 66, "hi": 67, "close_time": "2099-01-01T00:00:00+00:00"},
            ])

        class FakePaced:
            def get_json(self, url, params=None):
                if url.endswith("/historical/cutoff"):
                    return {"market_settled_ts": "2026-08-08T00:00:00Z"}
                return PAYLOAD
        cs = chunks(FakePaced(), c, lambda: "2026-10-07T00:00:00+00:00")
        assert [ch.key for ch in cs] == ["OK"] and not cs[0].current
        table, rows = cs[0].run()[0]
        assert table == "kalshi_candles" and len(rows) == 2
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/sources/kalshi_candles.py
"""Kalshi hourly candles -> kalshi_candles (close bid/ask per active hour; spec §1)."""
from datetime import datetime, timedelta, timezone

from citycast.backfill import Chunk
from citycast.http import HttpError
from citycast.sources.kalshi_markets import KALSHI, historical_cutoff

MAX_WINDOW_H = 5000


def _num(x) -> float | None:
    return None if x in (None, "") else float(x)


def _close(d) -> float | None:
    return _num(d.get("close")) if isinstance(d, dict) else None


def candle_rows(ticker: str, payload: dict) -> list[dict]:
    return [{"ticker": ticker,
             "end_utc": datetime.fromtimestamp(int(c["end_period_ts"]), timezone.utc).isoformat(),
             "yes_bid": _close(c.get("yes_bid")), "yes_ask": _close(c.get("yes_ask")),
             "volume": _num(c.get("volume")), "open_interest": _num(c.get("open_interest"))}
            for c in payload.get("candlesticks") or []]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def fetch_candles(paced, ticker: str, series: str, open_iso: str | None, close_iso: str,
                  cutoff_iso: str) -> dict:
    end = _ts(close_iso)
    start = _ts(open_iso) if open_iso else end - 7 * 86400
    paths = [f"/historical/markets/{ticker}/candlesticks",
             f"/series/{series}/markets/{ticker}/candlesticks"]
    if close_iso >= cutoff_iso:
        paths.reverse()
    merged: list[dict] = []
    for i, path in enumerate(paths):
        try:
            a = start
            while a < end:
                b = min(end, a + MAX_WINDOW_H * 3600)
                j = paced.get_json(KALSHI + path, {"start_ts": a, "end_ts": b, "period_interval": 60})
                merged.extend(j.get("candlesticks") or [])
                a = b
            return {"candlesticks": merged}
        except HttpError as e:
            if e.status != 404 or i == len(paths) - 1:
                raise
            merged = []
    return {"candlesticks": merged}


def chunks(paced, conn, now_iso) -> list[Chunk]:
    now = now_iso()
    rows = conn.execute(
        "SELECT ticker, series, open_time, close_time FROM kalshi_markets "
        "WHERE NOT (lo IS NULL AND hi IS NULL) AND close_time < ? ORDER BY close_time",
        (now,)).fetchall()
    if not rows:
        return []
    cutoff = historical_cutoff(paced)
    out = []
    for r in rows:
        def run(r=r):
            payload = fetch_candles(paced, r["ticker"], r["series"], r["open_time"],
                                    r["close_time"], cutoff)
            return [("kalshi_candles", candle_rows(r["ticker"], payload))]
        out.append(Chunk("kalshi-candles", r["ticker"], False, run))
    return out
```

- [ ] **Step 4: Run to verify they pass** — 3 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 9: Coverage report and the `backfill` / `coverage` commands

**Files:**
- Create: `src/citycast/coverage.py`
- Modify: `src/citycast/cli.py` (replace whole file)
- Test: `tests/test_citycast_coverage.py`, `tests/test_citycast_cli.py`

**Interfaces:**
- Consumes: everything above; `highcast.timeutil.climate_day_bounds(day, std_utc_offset)`, `highcast.timeutil.utcnow_iso`.
- Produces: `coverage.compute(conn, stations, today: date, since: date = date(2021, 1, 1)) -> dict` with `{"stations": {icao: {...}}, "disagreements": [...], "pass": bool}`; `coverage.render(report) -> str`; CLI `citycast backfill SOURCE [--station K…] [--since YYYY-MM-DD] [--refresh]` and `citycast coverage [--json] [--since] [--station K…]`.

Per-station report keys: `cli_final_pct`, `mos_pct` (dict model → fraction of days since that model's first row with a run issued before the climate day starts), `obs_pct` (days with ≥ 20 METARs / days since `since`), `kalshi_events`, `kalshi_markets`, `kalshi_usable`, `kalshi_with_candles`, `pass` (`cli_final_pct ≥ 0.98` and `mos_pct["MEX"] ≥ 0.98` and `mos_pct["NBS"] ≥ 0.98`). `disagreements` = events whose numeric `expiration_value` ≠ CLI final value for that metric: `{"station","metric","event_date","kalshi","cli"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_coverage.py
from contextlib import closing
from datetime import date, timedelta

from citycast import db
from citycast.coverage import compute, render
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"


def _seed(c, days, missing_cli=0):
    start = date(2026, 1, 1)
    cli, mos = [], []
    for i in range(days):
        d = start + timedelta(days=i)
        if i >= missing_cli:
            cli.append({"station": "KNYC", "date": d.isoformat(), "high_f": 40, "low_f": 30,
                        "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F})
        for model in ("MEX", "NBS"):
            # run issued at 12Z the day before: before the climate day starts (05Z)
            mos.append({"station": "KNYC", "model": model,
                        "runtime_utc": f"{(d - timedelta(days=1)).isoformat()}T12:00:00+00:00",
                        "target_date": d.isoformat(), "metric": "high", "value_f": 41.0,
                        "n_points": 1, "fetched_at": F})
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "mos_runs", mos)
        db.upsert(c, "kalshi_markets", [{
            "ticker": "KXHIGHNY-26JAN05-T40", "series": "KXHIGHNY", "event_ticker": "KXHIGHNY-26JAN05",
            "station": "KNYC", "metric": "high", "event_date": "2026-01-05", "lo": 41, "hi": None,
            "open_time": None, "close_time": "2026-01-06T04:59:00+00:00", "result": "no",
            "expiration_value": "39", "raw": "{}"}])


def test_full_coverage_passes_and_disagreement_listed(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, 100)
        rep = compute(c, {"KNYC": ST}, today=date(2026, 4, 11), since=date(2026, 1, 1))
        s = rep["stations"]["KNYC"]
        assert s["cli_final_pct"] == 1.0 and s["mos_pct"]["MEX"] == 1.0 and s["pass"]
        assert s["kalshi_markets"] == 1 and s["kalshi_usable"] == 1 and s["kalshi_with_candles"] == 0
        assert rep["disagreements"] == [{"station": "KNYC", "metric": "high",
                                         "event_date": "2026-01-05", "kalshi": 39, "cli": 40}]
        assert rep["pass"] and "KNYC" in render(rep)


def test_missing_cli_fails_the_station(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, 100, missing_cli=5)
        rep = compute(c, {"KNYC": ST}, today=date(2026, 4, 11), since=date(2026, 1, 1))
        assert rep["stations"]["KNYC"]["cli_final_pct"] == 0.95 and not rep["pass"]
```

```python
# tests/test_citycast_cli.py
import json

from citycast import cli


def test_backfill_dispatches_source_and_exit_code(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run(conn, chunks, **kw):
        seen["n"] = len(chunks)
        return {"done": len(chunks), "skipped": 0, "rows": 0, "errors": []}
    monkeypatch.setattr(cli, "run_chunks", fake_run)
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: [object(), object()])
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli", "--since", "2024-01-01"]) == 0
    assert seen["n"] == 2 and json.loads(capsys.readouterr().out.splitlines()[-1])["done"] == 2


def test_backfill_exit_1_on_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_chunks", lambda *a, **k: {"done": 0, "skipped": 0, "rows": 0,
                                                            "errors": [{"x": 1}]})
    monkeypatch.setattr(cli, "_chunks_for", lambda source, *a, **k: [])
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "obs"]) == 1


def test_unknown_station_rejected(tmp_path):
    assert cli.main(["--db", str(tmp_path / "h.db"), "backfill", "cli", "--station", "KXXX"]) == 2
```

- [ ] **Step 2: Run to verify they fail** — FAIL (`citycast.coverage` missing; `cli.run_chunks` missing).

- [ ] **Step 3: Implement coverage**

```python
# src/citycast/coverage.py
"""Coverage report and the 2a acceptance check (spec §2.5)."""
from datetime import date, datetime, timedelta

from highcast.timeutil import climate_day_bounds

MIN_PCT = 0.98
REQUIRED_MODELS = ("MEX", "NBS")


def _days(a: date, b: date) -> int:
    return max((b - a).days, 0)


def _mos_pct(conn, st, today: date) -> dict:
    out = {}
    for (model,) in conn.execute("SELECT DISTINCT model FROM mos_runs WHERE station=?", (st.icao,)):
        first = conn.execute("SELECT MIN(target_date) FROM mos_runs WHERE station=? AND model=?",
                             (st.icao, model)).fetchone()[0]
        start = date.fromisoformat(first)
        ok = 0
        rows = conn.execute("SELECT target_date, MIN(runtime_utc) FROM mos_runs WHERE station=? "
                            "AND model=? AND target_date < ? GROUP BY target_date",
                            (st.icao, model, today.isoformat())).fetchall()
        for d, first_run in rows:
            day_start = climate_day_bounds(date.fromisoformat(d), st.std_utc_offset)[0]
            if datetime.fromisoformat(first_run) < day_start:
                ok += 1
        n = _days(start, today)
        out[model] = round(ok / n, 4) if n else 0.0
    return out


def _disagreements(conn, st) -> list[dict]:
    out = []
    rows = conn.execute(
        "SELECT DISTINCT k.metric, k.event_date, k.expiration_value, c.high_f, c.low_f "
        "FROM kalshi_markets k JOIN cli_daily c ON c.station=k.station AND c.date=k.event_date "
        "WHERE k.station=? AND c.final=1 AND k.expiration_value IS NOT NULL "
        "AND k.expiration_value != ''", (st.icao,)).fetchall()
    for metric, d, ev, hi, lo in rows:
        try:
            kv = round(float(ev))
        except ValueError:
            continue
        cv = hi if metric == "high" else lo
        if cv is not None and kv != cv:
            out.append({"station": st.icao, "metric": metric, "event_date": d, "kalshi": kv, "cli": cv})
    return out


def compute(conn, stations: dict, today: date, since: date = date(2021, 1, 1)) -> dict:
    report = {"stations": {}, "disagreements": []}
    n_days = _days(since, today)
    for st in stations.values():
        cli_final = conn.execute("SELECT COUNT(*) FROM cli_daily WHERE station=? AND final=1 "
                                 "AND date>=? AND date<?",
                                 (st.icao, since.isoformat(), today.isoformat())).fetchone()[0]
        obs_days = conn.execute(
            "SELECT COUNT(*) FROM (SELECT substr(datetime(valid_utc, ? || ' hours'), 1, 10) d "
            "FROM obs WHERE station=? GROUP BY d HAVING COUNT(*) >= 20)",
            (str(st.std_utc_offset), st.icao)).fetchone()[0]
        k = conn.execute(
            "SELECT COUNT(DISTINCT event_ticker), COUNT(*), "
            "SUM(NOT (lo IS NULL AND hi IS NULL)), "
            "SUM(EXISTS (SELECT 1 FROM kalshi_candles kc WHERE kc.ticker = km.ticker)) "
            "FROM kalshi_markets km WHERE station=?", (st.icao,)).fetchone()
        mos = _mos_pct(conn, st, today)
        cli_pct = round(cli_final / n_days, 4) if n_days else 0.0
        s = {"cli_final_pct": cli_pct, "mos_pct": mos,
             "obs_pct": round(obs_days / n_days, 4) if n_days else 0.0,
             "kalshi_events": k[0] or 0, "kalshi_markets": k[1] or 0,
             "kalshi_usable": k[2] or 0, "kalshi_with_candles": k[3] or 0}
        s["pass"] = cli_pct >= MIN_PCT and all(mos.get(m, 0.0) >= MIN_PCT for m in REQUIRED_MODELS)
        report["stations"][st.icao] = s
        report["disagreements"].extend(_disagreements(conn, st))
    report["pass"] = all(s["pass"] for s in report["stations"].values())
    return report


def render(report: dict) -> str:
    lines = [f"{'station':8} {'cli':>6} {'MEX':>6} {'NBS':>6} {'GFS':>6} {'NBE':>6} {'obs':>6} "
             f"{'events':>7} {'mkts':>7} {'usable':>7} {'candles':>7}  pass"]
    for icao, s in sorted(report["stations"].items()):
        m = s["mos_pct"]
        lines.append(f"{icao:8} {s['cli_final_pct']:6.1%} {m.get('MEX', 0):6.1%} {m.get('NBS', 0):6.1%} "
                     f"{m.get('GFS', 0):6.1%} {m.get('NBE', 0):6.1%} {s['obs_pct']:6.1%} "
                     f"{s['kalshi_events']:7d} {s['kalshi_markets']:7d} {s['kalshi_usable']:7d} "
                     f"{s['kalshi_with_candles']:7d}  {'yes' if s['pass'] else 'NO'}")
    lines.append(f"disagreements (Kalshi settled value vs CLI final): {len(report['disagreements'])}")
    for d in report["disagreements"]:
        lines.append(f"  {d['station']} {d['metric']} {d['event_date']}: kalshi {d['kalshi']} cli {d['cli']}")
    lines.append(f"ACCEPTANCE: {'PASS' if report['pass'] else 'FAIL'}")
    return "\n".join(lines)
```

- [ ] **Step 4: Replace the CLI**

```python
# src/citycast/cli.py
"""citycast init | backfill SOURCE | coverage (spec §2.4–2.5)."""
import argparse
import json
import sys
from contextlib import closing
from datetime import date

from citycast import coverage, db
from citycast.backfill import run_chunks
from citycast.http import Paced
from citycast.sources import cli_reports, kalshi_candles, kalshi_markets, mos, obs
from citycast.stations import load_stations, series_index
from highcast.timeutil import utcnow_iso

SOURCES = ("cli", "mos", "obs", "kalshi-markets", "kalshi-candles")


def _chunks_for(source, paced, conn, stations, since, today):
    if source == "cli":
        return cli_reports.chunks(paced, stations, since, today, utcnow_iso)
    if source == "mos":
        return mos.chunks(paced, stations, since, today, utcnow_iso)
    if source == "obs":
        return obs.chunks(paced, stations, since, today, utcnow_iso)
    if source == "kalshi-markets":
        idx = {s: v for s, v in series_index(stations).items() if v[0] in stations}
        return kalshi_markets.chunks(paced, idx)
    if source == "kalshi-candles":
        return kalshi_candles.chunks(paced, conn, utcnow_iso)
    raise ValueError(source)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citycast")
    p.add_argument("--db", default=str(db.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    b = sub.add_parser("backfill")
    b.add_argument("source", choices=SOURCES + ("all",))
    b.add_argument("--station")
    b.add_argument("--since", default="2021-01-01")
    b.add_argument("--refresh", action="store_true")
    cov = sub.add_parser("coverage")
    cov.add_argument("--json", action="store_true")
    cov.add_argument("--since", default="2021-01-01")
    cov.add_argument("--station")
    args = p.parse_args(argv)

    stations = load_stations()
    if getattr(args, "station", None):
        if args.station not in stations:
            print(f"unknown station {args.station}", file=sys.stderr)
            return 2
        stations = {args.station: stations[args.station]}

    with closing(db.connect(args.db)) as conn:
        if args.cmd == "init":
            print(json.dumps({"status": "ok", "db": args.db}))
            return 0
        if args.cmd == "backfill":
            paced = Paced()
            today = date.today()
            since = date.fromisoformat(args.since)
            sources = SOURCES if args.source == "all" else (args.source,)
            total = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
            for source in sources:
                chunks = _chunks_for(source, paced, conn, stations, since, today)
                s = run_chunks(conn, chunks, refresh=args.refresh, now_iso=utcnow_iso)
                for k in ("done", "skipped", "rows"):
                    total[k] += s[k]
                total["errors"] += s["errors"]
            print(json.dumps(total))
            return 1 if total["errors"] else 0
        if args.cmd == "coverage":
            rep = coverage.compute(conn, stations, date.today(), date.fromisoformat(args.since))
            print(json.dumps(rep, indent=1) if args.json else coverage.render(rep))
            return 0 if rep["pass"] else 1
    return 2
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_citycast_coverage.py tests/test_citycast_cli.py tests/test_citycast_db.py`
Expected: all pass (the Task 1 `init` test still passes; `init` now loads the registry, which exists since Task 2).

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q`, all pass.

---

### Task 10: Run the real backfill and accept 2a (controller task, not a subagent)

This task writes only `data/history.db` and a report. It needs the network and several hours.

- [ ] **Step 1: One-station smoke test**

Run: `.venv/bin/citycast backfill all --station KNYC --since 2026-09-01`
Expected: JSON summary with `"errors": []`. Then `.venv/bin/citycast coverage --since 2026-09-01 --station KNYC` prints one row.

- [ ] **Step 2: Full weather backfill (≈ 1 h at 1 req/s)**

Run, in the background: `.venv/bin/citycast backfill cli && .venv/bin/citycast backfill mos && .venv/bin/citycast backfill obs`
Expected: each prints a summary; rerun any source whose summary lists errors (finished chunks are skipped).

- [ ] **Step 3: Kalshi backfill (markets ≈ minutes; candles ≈ 5–6 h for ~90,000 markets)**

Run, in the background: `.venv/bin/citycast backfill kalshi-markets && .venv/bin/citycast backfill kalshi-candles`
Expected: summaries with no errors, or reruns until clean.

- [ ] **Step 4: Coverage acceptance**

Run: `.venv/bin/citycast coverage > docs/reports/citycast-coverage-$(date +%F).md; echo exit=$?`
Expected: `ACCEPTANCE: PASS`, exit 0. For every `NO` row: rerun that station's source with `--refresh`; if IEM genuinely lacks the data, record the gap in the report with the IEM URL that shows it. List every disagreement in the report for Julian; truth stays CLI final.

- [ ] **Step 5: Record**

Append the coverage table, row counts per table, DB size, and any gaps to `.superpowers/sdd/sandbox-progress.md` (2a section) and update `~/.agents/projects/weather-temp-predictor/weather-sandbox-plan.md` status. 2b may start only after Step 4 passes.
