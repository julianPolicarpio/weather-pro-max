# Weather Desk Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paper-trade every open Kalshi daily temperature market against real live prices and real order books, so any model (later Laya) must prove itself before real money.

**Architecture:** A new package `src/sandbox/` beside `src/highcast/`, with its own SQLite file `data/sandbox.db`. It reuses highcast's verified fee, bucket, market-parsing and MOS-parsing code by import, never writes to highcast's database, and never imports the real-money client. A launchd job runs a cycle every 15 minutes (snapshot all markets); the first cycle of each UTC hour also settles and lets bots bet. A local FastAPI dashboard on port 8713 shows markets, manual betting, the scoreboard and the audit log.

**Tech Stack:** Python 3.14 (`.venv`), sqlite3, httpx (via `highcast.http.get_json`), numpy, FastAPI + Jinja2 + uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-weather-sandbox-design.md`

## Global Constraints

- **Not a git repo. Never run git commands or offer to commit** (project rule in `CLAUDE.md`). Every task ends with a full-suite run instead of a commit.
- Never import `highcast.kalshi_client` anywhere under `src/sandbox/` (enforced by a test in Task 10).
- Never write to `data/highcast.db`. The `highcast` bot opens it with `?mode=ro`.
- All Kalshi reads are unauthenticated, at most **5 requests per second** (`MIN_INTERVAL_S = 0.2`).
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`.
- Timestamps stored as UTC ISO strings with `+00:00`, seconds precision (`2026-10-04T12:00:00+00:00`).
- Every bot bets **1 contract**, YES side, at most once per bucket ticker; each account starts with **$5.00**.
- `MIN_EDGE` is imported from `highcast.orderbot` (0.03). `SIGMA_F = 3.0`.
- Unlock rule: **≥100 settled bets** AND profit bootstrap 95% CI lower bound > 0 AND (bot − market) Brier bootstrap 95% CI upper bound < 0.
- Dashboard: title "Weather Desk Sandbox", no Kalshi logos or styling, `127.0.0.1:8713` (highcast uses 8712).
- No new third-party dependencies (`python-multipart` is NOT installed — the bet form posts JSON).
- Run tests with `.venv/bin/pytest`. Unit tests must never touch the network (inject a fake `get`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | Add `src/sandbox` to the wheel and a `sandbox` console script |
| `src/sandbox/__init__.py` | Package docstring |
| `src/sandbox/stations.py` | `STATIONS` tz table, `station_from_rules`, `metric_from_series` |
| `src/sandbox/kalshi_public.py` | `PacedGet`, `list_temperature_series`, `open_markets`, `settled_markets`, `fetch_book_payload` |
| `src/sandbox/models.py` | Dataclasses `Market`, `Snapshot`, `Intent`, `Candidate` |
| `src/sandbox/markets.py` | `parse_event_date`, `normalize_ts`, `parse_series_payload`, `usable_bid`, `usable_ask` |
| `src/sandbox/db.py` | `SCHEMA`, `connect`, `get_meta`, `set_meta` |
| `src/sandbox/recorder.py` | `upsert_markets`, `record_snapshot`, `latest_snapshot_id`, `load_snapshot`, `row_to_market`, `record_candidates` |
| `src/sandbox/book.py` | `Book`, `Fill`, `NO_FILL`, `parse_book`, `walk` |
| `src/sandbox/pricing.py` | `mid`, `devig`, `multiclass_brier` |
| `src/sandbox/ledger.py` | `START_CASH`, `open_account`, `cash`, `holds`, `place` |
| `src/sandbox/bots.py` | `MIN_EDGE`, `SIGMA_F`, `Decision`, `evaluate_bucket`, `normal_bucket_probs`, `minutes_between`, `ForecastBot`, `FavoriteBot`, `IemMos`, `MosBot`, `HighcastBot` |
| `src/sandbox/settle.py` | `settle_due` |
| `src/sandbox/scoreboard.py` | `N_REQUIRED`, `bootstrap_mean_ci`, `Scorecard`, `report` |
| `src/sandbox/cycle.py` | `ACCOUNTS`, `utc_now_iso`, `ensure_accounts`, `discover`, `execute_intent`, `run_bot`, `decide_all`, `run_cycle` |
| `src/sandbox/web.py` + `src/sandbox/templates/*.html` | Dashboard |
| `src/sandbox/cli.py` | `sandbox init / cycle / serve / report` |
| `launchd/com.julian.sandbox.cycle.plist`, `launchd/com.julian.sandbox.serve.plist` | Scheduling |
| `tests/sandbox_helpers.py` | Shared synthetic payload/market builders for tests |
| `tests/test_sandbox_*.py` | One test file per module |

---

### Task 1: Package scaffold, stations, paced Kalshi client

**Files:**
- Modify: `pyproject.toml`
- Create: `src/sandbox/__init__.py`, `src/sandbox/stations.py`, `src/sandbox/kalshi_public.py`
- Test: `tests/test_sandbox_stations.py`, `tests/test_sandbox_kalshi_public.py`, `tests/test_sandbox_live.py`

**Interfaces:**
- Produces: `STATIONS: dict[str, tuple[str, int]]`; `station_from_rules(rules_primary: str | None) -> str | None`; `metric_from_series(series: str) -> str | None`; `BASE: str`; `MIN_INTERVAL_S: float`; `class PacedGet(fetch=get_json, min_interval=MIN_INTERVAL_S, clock=time.monotonic, sleep=time.sleep)` callable as `get(path: str, params: dict | None = None) -> dict`; `list_temperature_series(get) -> list[dict]`; `open_markets(get, series: str) -> dict`; `settled_markets(get, series: str) -> dict`; `fetch_book_payload(get, ticker: str) -> dict`.

- [ ] **Step 1: Register the package**

In `pyproject.toml`, change the scripts and wheel sections to:

```toml
[project.scripts]
highcast = "highcast.cli:main"
sandbox = "sandbox.cli:main"
```

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/highcast", "src/sandbox"]
```

Create `src/sandbox/__init__.py`:

```python
"""Weather Desk Sandbox — paper trading on live Kalshi temperature markets.

Never places real orders. Never imports highcast.kalshi_client. Never writes to
data/highcast.db. See docs/superpowers/specs/2026-10-04-weather-sandbox-design.md.
"""
```

Reinstall without touching pinned dependencies:

Run: `.venv/bin/pip install -e . --no-deps`
Expected: `Successfully installed highcast-0.1.0`

- [ ] **Step 2: Write the failing stations test**

Create `tests/test_sandbox_stations.py`:

```python
from zoneinfo import ZoneInfo

from sandbox.stations import STATIONS, metric_from_series, station_from_rules

# The 24 settlement codes measured live on 2026-10-04 from every open series'
# rules_primary. If Kalshi adds a city, this list and STATIONS grow together.
LIVE_CODES = ["AUS", "MDW", "DEN", "LAX", "MIA", "NYC", "PHL", "ATL", "BOS", "DFW",
              "DCA", "EWR", "HOU", "LAS", "MSP", "MSY", "OKC", "PHX", "SAN", "SAT",
              "SDF", "SEA", "SFO", "TTN"]


def test_station_from_rules_reads_the_cli_code():
    rules = ("If the maximum temperature recorded at New York City (CLINYC) for "
             "Oct 4, 2026, is greater than 70° fahrenheit")
    assert station_from_rules(rules) == "KNYC"
    assert station_from_rules("recorded at Chicago (CLIMDW) for") == "KMDW"


def test_station_from_rules_none_when_absent():
    assert station_from_rules("no code here") is None
    assert station_from_rules(None) is None


def test_metric_from_series():
    assert metric_from_series("KXHIGHNY") == "high"
    assert metric_from_series("KXHIGHTATL") == "high"
    assert metric_from_series("KXLOWTNYC") == "low"
    assert metric_from_series("KXRAINNYC") is None


def test_every_live_station_has_a_valid_timezone():
    for code in LIVE_CODES:
        tz, offset = STATIONS[f"K{code}"]
        ZoneInfo(tz)  # raises if the zone name is wrong
        assert offset in (-5, -6, -7, -8)
    assert len(STATIONS) == len(LIVE_CODES)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_stations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.stations'`

- [ ] **Step 4: Implement `src/sandbox/stations.py`**

```python
"""Settlement stations and series metadata for Kalshi daily temperature markets.

The station comes from each market's own rules text, never from a hand list:
"...recorded at New York City (CLINYC)..." -> KNYC. Verified for all 48 live
series on 2026-10-04. highcast.cities.ROSTER is stale (12 cities) and unused.
"""
import re

# ICAO -> (IANA tz, standard-time UTC offset in hours). The offset is what
# highcast.fetchers.forecasts.parse_mos needs to bucket MOS rows by climate day.
STATIONS: dict[str, tuple[str, int]] = {
    "KATL": ("America/New_York", -5),
    "KAUS": ("America/Chicago", -6),
    "KBOS": ("America/New_York", -5),
    "KDCA": ("America/New_York", -5),
    "KDEN": ("America/Denver", -7),
    "KDFW": ("America/Chicago", -6),
    "KEWR": ("America/New_York", -5),
    "KHOU": ("America/Chicago", -6),
    "KLAS": ("America/Los_Angeles", -8),
    "KLAX": ("America/Los_Angeles", -8),
    "KMDW": ("America/Chicago", -6),
    "KMIA": ("America/New_York", -5),
    "KMSP": ("America/Chicago", -6),
    "KMSY": ("America/Chicago", -6),
    "KNYC": ("America/New_York", -5),
    "KOKC": ("America/Chicago", -6),
    "KPHL": ("America/New_York", -5),
    "KPHX": ("America/Phoenix", -7),
    "KSAN": ("America/Los_Angeles", -8),
    "KSAT": ("America/Chicago", -6),
    "KSDF": ("America/Kentucky/Louisville", -5),
    "KSEA": ("America/Los_Angeles", -8),
    "KSFO": ("America/Los_Angeles", -8),
    "KTTN": ("America/New_York", -5),
}

_RULE_CODE = re.compile(r"\(CLI([A-Z]{3})\)")


def station_from_rules(rules_primary: str | None) -> str | None:
    """'(CLINYC)' in the rules text -> 'KNYC'. None when the code is absent."""
    m = _RULE_CODE.search(rules_primary or "")
    return f"K{m.group(1)}" if m else None


def metric_from_series(series: str) -> str | None:
    """KXHIGH* -> 'high', KXLOWT* -> 'low', anything else -> None."""
    if series.startswith("KXHIGH"):
        return "high"
    if series.startswith("KXLOWT"):
        return "low"
    return None
```

- [ ] **Step 5: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_stations.py -v`
Expected: 4 passed

- [ ] **Step 6: Write the failing client test**

Create `tests/test_sandbox_kalshi_public.py`:

```python
from sandbox.kalshi_public import (BASE, PacedGet, fetch_book_payload,
                                   list_temperature_series, open_markets,
                                   settled_markets)


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.t += s


def test_paced_get_waits_between_calls_and_prefixes_base():
    calls = []
    fc = FakeClock()
    get = PacedGet(fetch=lambda url, params: calls.append((url, params)) or {"ok": 1},
                   min_interval=0.2, clock=fc.clock, sleep=fc.sleep)
    assert get("/series", {"a": 1}) == {"ok": 1}
    get("/markets")
    assert calls == [(BASE + "/series", {"a": 1}), (BASE + "/markets", None)]
    assert fc.slept == [0.2]  # second call waited the full interval


def test_list_temperature_series_filters_to_high_and_low():
    def get(path, params=None):
        assert path == "/series"
        assert params == {"category": "Climate and Weather", "limit": 500}
        return {"series": [{"ticker": "KXHIGHNY"}, {"ticker": "KXRAINNYC"},
                           {"ticker": "KXLOWTNYC"}]}
    assert [s["ticker"] for s in list_temperature_series(get)] == ["KXHIGHNY", "KXLOWTNYC"]


def test_market_and_book_paths():
    seen = []

    def get(path, params=None):
        seen.append((path, params))
        return {}
    open_markets(get, "KXHIGHNY")
    settled_markets(get, "KXHIGHNY")
    fetch_book_payload(get, "KXHIGHNY-26OCT04-T70")
    assert seen == [
        ("/markets", {"series_ticker": "KXHIGHNY", "status": "open", "limit": 200}),
        ("/markets", {"series_ticker": "KXHIGHNY", "status": "settled", "limit": 200}),
        ("/markets/KXHIGHNY-26OCT04-T70/orderbook", None),
    ]
```

- [ ] **Step 7: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_kalshi_public.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.kalshi_public'`

- [ ] **Step 8: Implement `src/sandbox/kalshi_public.py`**

```python
"""Unauthenticated, paced reads from Kalshi's public market-data API.

Kalshi's authenticated Basic tier allows 20 reads/s (200 tokens/s, 10 per
request). The unauthenticated limit is undocumented, so we stay at 5/s.
Every function takes `get` so tests inject a fake and never touch the network.
"""
import time

from highcast.http import get_json
from sandbox.stations import metric_from_series

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MIN_INTERVAL_S = 0.2


class PacedGet:
    """get(path, params) -> JSON dict, never faster than one call per interval."""

    def __init__(self, fetch=get_json, min_interval: float = MIN_INTERVAL_S,
                 clock=time.monotonic, sleep=time.sleep):
        self.fetch = fetch
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self._last: float | None = None

    def __call__(self, path: str, params: dict | None = None) -> dict:
        if self._last is not None:
            wait = self._last + self.min_interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()
        return self.fetch(BASE + path, params)


def list_temperature_series(get) -> list[dict]:
    payload = get("/series", {"category": "Climate and Weather", "limit": 500})
    return [s for s in payload.get("series") or []
            if metric_from_series(s.get("ticker", "")) is not None]


def open_markets(get, series: str) -> dict:
    return get("/markets", {"series_ticker": series, "status": "open", "limit": 200})


def settled_markets(get, series: str) -> dict:
    # Verified 2026-10-04: status=settled returns markets whose status is
    # "finalized" with result "yes"/"no".
    return get("/markets", {"series_ticker": series, "status": "settled", "limit": 200})


def fetch_book_payload(get, ticker: str) -> dict:
    return get(f"/markets/{ticker}/orderbook")
```

- [ ] **Step 9: Add a live smoke test (manual only)**

Create `tests/test_sandbox_live.py`:

```python
import pytest

from sandbox.kalshi_public import PacedGet, list_temperature_series, open_markets
from sandbox.stations import STATIONS, station_from_rules


@pytest.mark.live
def test_every_open_temperature_series_maps_to_a_known_station():
    get = PacedGet()
    seen = 0
    for s in list_temperature_series(get):
        markets = open_markets(get, s["ticker"]).get("markets") or []
        if not markets:
            continue
        seen += 1
        st = station_from_rules(markets[0].get("rules_primary"))
        assert st in STATIONS, (s["ticker"], st)
    assert seen > 0
```

- [ ] **Step 10: Run unit tests, then the live smoke test once**

Run: `.venv/bin/pytest tests/test_sandbox_stations.py tests/test_sandbox_kalshi_public.py -v`
Expected: 7 passed

Run: `.venv/bin/pytest tests/test_sandbox_live.py -m live -v`
Expected: 1 passed (needs network; if it fails with a station not in `STATIONS`, add that station's tz and offset to `STATIONS` and `LIVE_CODES`)

- [ ] **Step 11: Full suite (replaces commit — not a git repo)**

Run: `.venv/bin/pytest -q`
Expected: all previously passing tests still pass, plus the new ones.

---

### Task 2: Data models, market parsing, database, recorder

**Files:**
- Create: `src/sandbox/models.py`, `src/sandbox/markets.py`, `src/sandbox/db.py`, `src/sandbox/recorder.py`
- Create: `tests/sandbox_helpers.py`
- Test: `tests/test_sandbox_markets.py`, `tests/test_sandbox_recorder.py`

**Interfaces:**
- Consumes: `metric_from_series`, `station_from_rules` (Task 1); `highcast.fetchers.kalshi.parse_markets(payload, event_ticker) -> list[dict]` with keys `label, lo, hi, ticker, yes_bid, yes_ask, last_price, close_time`.
- Produces:
  - `models.Market(ticker, series, event_ticker, event_date, lo, hi, label, metric, station, yes_bid, yes_ask, yes_bid_size, yes_ask_size, close_time)` (frozen).
  - `models.Snapshot(id: int, taken_at: str, markets: tuple[Market, ...])` with `.events() -> dict[str, list[Market]]`.
  - `models.Intent(account, ticker, side, contracts, price_cap, snapshot_id, p_bot, p_market)` (frozen).
  - `models.Candidate(...)` (mutable; fields listed in Step 3).
  - `markets.parse_event_date(event_ticker: str) -> str`; `markets.normalize_ts(ts: str | None) -> str | None`; `markets.parse_series_payload(series: str, payload: dict) -> list[Market]`; `markets.usable_bid(m: Market) -> float | None`; `markets.usable_ask(m: Market) -> float | None`.
  - `db.connect(path) -> sqlite3.Connection`; `db.get_meta(conn, key) -> str | None`; `db.set_meta(conn, key, value) -> None`.
  - `recorder.record_snapshot(conn, markets: list[Market], taken_at: str) -> int`; `recorder.latest_snapshot_id(conn, at_or_before: str) -> int | None`; `recorder.load_snapshot(conn, snapshot_id: int) -> Snapshot`; `recorder.row_to_market(row) -> Market`; `recorder.record_candidates(conn, candidates: list[Candidate]) -> None`.

- [ ] **Step 1: Create the shared test helpers**

Create `tests/sandbox_helpers.py`:

```python
"""Synthetic payloads shaped exactly like the live Kalshi responses verified on
2026-10-04 (markets: *_dollars price strings, *_size_fp size strings, Z close
times; orderbook: orderbook_fp.{yes,no}_dollars = [[price, qty], ...] bids)."""
from sandbox.models import Market

EVENT = "KXHIGHNY-26OCT04"


def raw_market(ticker, event=EVENT, floor=None, cap=None, bid="0.2000",
               ask="0.2400", bid_size="10.00", ask_size="3.00",
               close="2026-10-05T05:00:00Z", code="NYC"):
    return {
        "ticker": ticker, "event_ticker": event, "market_type": "binary",
        "floor_strike": floor, "cap_strike": cap,
        "yes_bid_dollars": bid, "yes_ask_dollars": ask,
        "yes_bid_size_fp": bid_size, "yes_ask_size_fp": ask_size,
        "close_time": close,
        "rules_primary": (f"If the maximum temperature recorded at Somewhere (CLI{code}) "
                          "for Oct 4, 2026, is greater than 70° fahrenheit"),
    }


def ny_event_payload():
    """Three buckets: ≤68 (cap 69), 69–70 (floor 69 cap 70), ≥71 (floor 70)."""
    return {"markets": [
        raw_market(f"{EVENT}-T69", cap=69, bid="0.1500", ask="0.2000"),
        raw_market(f"{EVENT}-B69.5", floor=69, cap=70, bid="0.2500", ask="0.3000"),
        raw_market(f"{EVENT}-T70", floor=70, bid="0.4000", ask="0.4500"),
    ]}


def make_market(ticker, lo=None, hi=None, bid=0.20, ask=0.24, event=EVENT,
                series="KXHIGHNY", metric="high", station="KNYC",
                close="2026-10-05T05:00:00+00:00", bid_size=10.0, ask_size=3.0):
    label = f"≤{hi}" if lo is None else (f"≥{lo}" if hi is None else f"{lo}–{hi}")
    return Market(ticker=ticker, series=series, event_ticker=event,
                  event_date="2026-10-04", lo=lo, hi=hi, label=label,
                  metric=metric, station=station, yes_bid=bid, yes_ask=ask,
                  yes_bid_size=bid_size, yes_ask_size=ask_size, close_time=close)


def three_buckets(bids=(0.15, 0.25, 0.40), asks=(0.20, 0.30, 0.45), **kw):
    return [
        make_market(f"{EVENT}-T69", None, 68, bids[0], asks[0], **kw),
        make_market(f"{EVENT}-B69.5", 69, 70, bids[1], asks[1], **kw),
        make_market(f"{EVENT}-T70", 71, None, bids[2], asks[2], **kw),
    ]


def book_payload(no=(), yes=()):
    fmt = lambda levels: [[f"{p:.4f}", f"{q:.2f}"] for p, q in levels]
    return {"orderbook_fp": {"no_dollars": fmt(no), "yes_dollars": fmt(yes)}}
```

- [ ] **Step 2: Write the failing parsing test**

Create `tests/test_sandbox_markets.py`:

```python
from sandbox_helpers import EVENT, make_market, ny_event_payload

from sandbox.markets import (normalize_ts, parse_event_date, parse_series_payload,
                             usable_ask, usable_bid)


def test_parse_event_date():
    assert parse_event_date("KXHIGHNY-26OCT04") == "2026-10-04"
    assert parse_event_date("KXLOWTSFO-26JAN31") == "2026-01-31"


def test_normalize_ts():
    assert normalize_ts("2026-10-05T05:00:00Z") == "2026-10-05T05:00:00+00:00"
    assert normalize_ts("2026-10-05T05:00:00+00:00") == "2026-10-05T05:00:00+00:00"
    assert normalize_ts(None) is None


def test_parse_series_payload_builds_markets_with_station_and_sizes():
    ms = parse_series_payload("KXHIGHNY", ny_event_payload())
    assert [m.label for m in ms] == ["≤68", "69–70", "≥71"]
    first = ms[0]
    assert first.ticker == f"{EVENT}-T69"
    assert first.event_date == "2026-10-04"
    assert (first.lo, first.hi) == (None, 68)
    assert first.metric == "high" and first.station == "KNYC"
    assert (first.yes_bid, first.yes_ask) == (0.15, 0.20)
    assert (first.yes_bid_size, first.yes_ask_size) == (10.0, 3.0)
    assert first.close_time == "2026-10-05T05:00:00+00:00"


def test_usable_prices_drop_empty_sides():
    assert usable_ask(make_market("A", ask=1.0)) is None   # no seller
    assert usable_bid(make_market("A", bid=0.0)) is None   # no buyer
    assert usable_ask(make_market("A", ask=0.24)) == 0.24
```

Run: `.venv/bin/pytest tests/test_sandbox_markets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.models'`

- [ ] **Step 3: Implement `src/sandbox/models.py` and `src/sandbox/markets.py`**

`src/sandbox/models.py`:

```python
"""Plain data carried between sandbox modules."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    ticker: str
    series: str
    event_ticker: str
    event_date: str          # Kalshi event date = the station's climate day
    lo: int | None           # inclusive °F; None for the bottom open tail
    hi: int | None           # inclusive °F; None for the top open tail
    label: str
    metric: str | None       # "high" | "low"
    station: str | None      # ICAO from the market's own rules text
    yes_bid: float | None
    yes_ask: float | None
    yes_bid_size: float | None
    yes_ask_size: float | None
    close_time: str | None   # UTC ISO with +00:00


@dataclass(frozen=True)
class Snapshot:
    id: int
    taken_at: str
    markets: tuple[Market, ...]

    def events(self) -> dict[str, list[Market]]:
        out: dict[str, list[Market]] = {}
        for m in self.markets:
            out.setdefault(m.event_ticker, []).append(m)
        return out


@dataclass(frozen=True)
class Intent:
    account: str
    ticker: str
    side: str                # "yes" | "no"
    contracts: int
    price_cap: float
    snapshot_id: int | None
    p_bot: float | None
    p_market: float | None


@dataclass
class Candidate:
    bot: str
    ticker: str
    series: str
    station: str | None
    metric: str | None
    event_date: str
    snapshot_id: int
    decided_at: str
    p_bot: float
    pmf_json: str            # {ticker: p} for the whole event
    p_market: float | None
    yes_bid: float | None
    yes_ask: float | None
    spread: float | None
    ask_size: float | None
    minutes_to_close: float | None
    fee_per_contract: float | None
    edge_after_fee: float | None
    placed: bool
    reason: str              # edge | below_min_edge | no_ask | already_held | no_fill | insufficient_cash
    gates_json: str | None = None    # populated in sub-project 3
    regime_json: str | None = None   # populated in sub-project 3
```

`src/sandbox/markets.py`:

```python
"""Turn a Kalshi /markets payload into Market rows.

Bucket structure (lo/hi/label) comes from highcast's verified parse_markets;
this module only adds what the sandbox needs on top: event date, metric,
station, and top-of-book sizes.
"""
from datetime import datetime

from highcast.fetchers.kalshi import parse_markets
from sandbox.models import Market
from sandbox.stations import metric_from_series, station_from_rules


def parse_event_date(event_ticker: str) -> str:
    """'KXHIGHNY-26OCT04' -> '2026-10-04'."""
    suffix = event_ticker.rsplit("-", 1)[1]
    return datetime.strptime(suffix, "%y%b%d").date().isoformat()


def normalize_ts(ts: str | None) -> str | None:
    """Kalshi's trailing 'Z' -> '+00:00', so stored times compare as strings."""
    if not ts:
        return None
    return ts[:-1] + "+00:00" if ts.endswith("Z") else ts


def _f(x) -> float | None:
    return None if x is None else float(x)


def parse_series_payload(series: str, payload: dict) -> list[Market]:
    raw = {m["ticker"]: m for m in payload.get("markets") or [] if m.get("ticker")}
    metric = metric_from_series(series)
    events = sorted({m.get("event_ticker") for m in raw.values() if m.get("event_ticker")})
    out: list[Market] = []
    for ev in events:
        for r in parse_markets(payload, ev):
            m = raw[r["ticker"]]
            out.append(Market(
                ticker=r["ticker"], series=series, event_ticker=ev,
                event_date=parse_event_date(ev), lo=r["lo"], hi=r["hi"],
                label=r["label"], metric=metric,
                station=station_from_rules(m.get("rules_primary")),
                yes_bid=r["yes_bid"], yes_ask=r["yes_ask"],
                yes_bid_size=_f(m.get("yes_bid_size_fp")),
                yes_ask_size=_f(m.get("yes_ask_size_fp")),
                close_time=normalize_ts(r["close_time"]),
            ))
    return out


def usable_bid(m: Market) -> float | None:
    return m.yes_bid if m.yes_bid is not None and 0.0 < m.yes_bid < 1.0 else None


def usable_ask(m: Market) -> float | None:
    return m.yes_ask if m.yes_ask is not None and 0.0 < m.yes_ask < 1.0 else None
```

- [ ] **Step 4: Run the parsing test**

Run: `.venv/bin/pytest tests/test_sandbox_markets.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the failing recorder test**

Create `tests/test_sandbox_recorder.py`:

```python
from sandbox_helpers import three_buckets

from sandbox.db import connect, get_meta, set_meta
from sandbox.models import Candidate
from sandbox.recorder import (latest_snapshot_id, load_snapshot, record_candidates,
                              record_snapshot)

T1 = "2026-10-04T10:00:00+00:00"
T2 = "2026-10-04T10:15:00+00:00"
BETWEEN = "2026-10-04T10:07:00+00:00"


def test_snapshot_roundtrip():
    conn = connect(":memory:")
    sid = record_snapshot(conn, three_buckets(), T1)
    snap = load_snapshot(conn, sid)
    assert snap.taken_at == T1
    assert [m.label for m in snap.markets] == ["≤68", "69–70", "≥71"]
    assert snap.markets[0].station == "KNYC"
    assert list(snap.events()) == ["KXHIGHNY-26OCT04"]


def test_latest_snapshot_never_looks_ahead():
    conn = connect(":memory:")
    s1 = record_snapshot(conn, three_buckets(), T1)
    s2 = record_snapshot(conn, three_buckets(asks=(0.5, 0.5, 0.5)), T2)
    assert latest_snapshot_id(conn, BETWEEN) == s1
    assert latest_snapshot_id(conn, T2) == s2
    assert latest_snapshot_id(conn, "2026-10-04T09:00:00+00:00") is None


def test_quotes_are_per_snapshot():
    conn = connect(":memory:")
    s1 = record_snapshot(conn, three_buckets(), T1)
    s2 = record_snapshot(conn, three_buckets(asks=(0.5, 0.5, 0.5)), T2)
    assert load_snapshot(conn, s1).markets[0].yes_ask == 0.20
    assert load_snapshot(conn, s2).markets[0].yes_ask == 0.5


def test_meta_roundtrip():
    conn = connect(":memory:")
    assert get_meta(conn, "k") is None
    set_meta(conn, "k", "v")
    assert get_meta(conn, "k") == "v"


def test_record_candidates_stores_rows():
    conn = connect(":memory:")
    sid = record_snapshot(conn, three_buckets(), T1)
    c = Candidate(bot="mos", ticker="KXHIGHNY-26OCT04-T69", series="KXHIGHNY",
                  station="KNYC", metric="high", event_date="2026-10-04",
                  snapshot_id=sid, decided_at=T1, p_bot=0.37, pmf_json="{}",
                  p_market=0.2, yes_bid=0.15, yes_ask=0.20, spread=0.05,
                  ask_size=3.0, minutes_to_close=1140.0, fee_per_contract=0.02,
                  edge_after_fee=0.15, placed=True, reason="edge")
    record_candidates(conn, [c])
    row = conn.execute("SELECT bot, placed, reason, result FROM candidates").fetchone()
    assert tuple(row) == ("mos", 1, "edge", None)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.db'`

- [ ] **Step 7: Implement `src/sandbox/db.py`**

```python
"""sandbox.db schema. Own file; never the highcast database."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS markets (
  ticker TEXT PRIMARY KEY, series TEXT NOT NULL, event_ticker TEXT NOT NULL,
  event_date TEXT NOT NULL, lo INTEGER, hi INTEGER, label TEXT NOT NULL,
  metric TEXT, station TEXT, close_time TEXT, result TEXT, settled_at TEXT);
CREATE INDEX IF NOT EXISTS markets_event ON markets(event_ticker);
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT, taken_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS snapshots_taken ON snapshots(taken_at);
CREATE TABLE IF NOT EXISTS quotes (
  snapshot_id INTEGER NOT NULL, ticker TEXT NOT NULL,
  yes_bid REAL, yes_ask REAL, yes_bid_size REAL, yes_ask_size REAL,
  PRIMARY KEY (snapshot_id, ticker));
CREATE TABLE IF NOT EXISTS accounts (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL, start_cash REAL NOT NULL,
  label TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, ticker TEXT NOT NULL,
  side TEXT NOT NULL, contracts_requested INTEGER NOT NULL, price_cap REAL NOT NULL,
  snapshot_id INTEGER, decided_at TEXT NOT NULL, status TEXT NOT NULL,
  filled_contracts INTEGER NOT NULL, cost REAL NOT NULL, fee REAL NOT NULL,
  levels_json TEXT NOT NULL, book_json TEXT, p_bot REAL, p_market REAL, payout REAL);
CREATE INDEX IF NOT EXISTS orders_account ON orders(account);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bot TEXT NOT NULL, ticker TEXT NOT NULL,
  series TEXT NOT NULL, station TEXT, metric TEXT, event_date TEXT NOT NULL,
  snapshot_id INTEGER NOT NULL, decided_at TEXT NOT NULL, p_bot REAL NOT NULL,
  pmf_json TEXT NOT NULL, p_market REAL, yes_bid REAL, yes_ask REAL, spread REAL,
  ask_size REAL, minutes_to_close REAL, fee_per_contract REAL, edge_after_fee REAL,
  gates_json TEXT, regime_json TEXT, placed INTEGER NOT NULL, reason TEXT NOT NULL,
  result TEXT, asif_profit REAL);
CREATE INDEX IF NOT EXISTS candidates_ticker ON candidates(ticker);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()
```

- [ ] **Step 8: Implement `src/sandbox/recorder.py`**

```python
"""Append-only snapshots and the candidate log."""
import sqlite3
from dataclasses import astuple, fields

from sandbox.models import Candidate, Market, Snapshot


def upsert_markets(conn: sqlite3.Connection, markets: list[Market]) -> None:
    conn.executemany(
        """INSERT INTO markets(ticker, series, event_ticker, event_date, lo, hi, label,
                               metric, station, close_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET close_time=excluded.close_time""",
        [(m.ticker, m.series, m.event_ticker, m.event_date, m.lo, m.hi, m.label,
          m.metric, m.station, m.close_time) for m in markets])


def record_snapshot(conn: sqlite3.Connection, markets: list[Market], taken_at: str) -> int:
    upsert_markets(conn, markets)
    sid = conn.execute("INSERT INTO snapshots(taken_at) VALUES (?)", (taken_at,)).lastrowid
    conn.executemany(
        """INSERT INTO quotes(snapshot_id, ticker, yes_bid, yes_ask, yes_bid_size, yes_ask_size)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(sid, m.ticker, m.yes_bid, m.yes_ask, m.yes_bid_size, m.yes_ask_size)
         for m in markets])
    conn.commit()
    return sid


def latest_snapshot_id(conn: sqlite3.Connection, at_or_before: str) -> int | None:
    """Newest snapshot taken at or before the given time. Never one after it."""
    row = conn.execute("SELECT id FROM snapshots WHERE taken_at <= ? "
                       "ORDER BY taken_at DESC, id DESC LIMIT 1", (at_or_before,)).fetchone()
    return None if row is None else row["id"]


def row_to_market(r) -> Market:
    return Market(ticker=r["ticker"], series=r["series"], event_ticker=r["event_ticker"],
                  event_date=r["event_date"], lo=r["lo"], hi=r["hi"], label=r["label"],
                  metric=r["metric"], station=r["station"], yes_bid=r["yes_bid"],
                  yes_ask=r["yes_ask"], yes_bid_size=r["yes_bid_size"],
                  yes_ask_size=r["yes_ask_size"], close_time=r["close_time"])


def load_snapshot(conn: sqlite3.Connection, snapshot_id: int) -> Snapshot:
    taken = conn.execute("SELECT taken_at FROM snapshots WHERE id=?",
                         (snapshot_id,)).fetchone()["taken_at"]
    rows = conn.execute(
        """SELECT m.*, q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size
           FROM quotes q JOIN markets m ON m.ticker = q.ticker
           WHERE q.snapshot_id = ?
           ORDER BY m.event_ticker, COALESCE(m.hi, 1000)""", (snapshot_id,)).fetchall()
    return Snapshot(id=snapshot_id, taken_at=taken,
                    markets=tuple(row_to_market(r) for r in rows))


_CANDIDATE_COLS = [f.name for f in fields(Candidate)]


def record_candidates(conn: sqlite3.Connection, candidates: list[Candidate]) -> None:
    if not candidates:
        return
    cols = ", ".join(_CANDIDATE_COLS)
    marks = ", ".join("?" for _ in _CANDIDATE_COLS)
    conn.executemany(f"INSERT INTO candidates({cols}) VALUES ({marks})",
                     [astuple(c) for c in candidates])
    conn.commit()
```

- [ ] **Step 9: Run the recorder test**

Run: `.venv/bin/pytest tests/test_sandbox_recorder.py tests/test_sandbox_markets.py -v`
Expected: 9 passed

- [ ] **Step 10: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 3: Order book parsing and the fill walk

**Files:**
- Create: `src/sandbox/book.py`
- Test: `tests/test_sandbox_book.py`

**Interfaces:**
- Consumes: `highcast.fees.taker_fee(contracts, price) -> float` (rounds up to the cent; raises unless 0 < price < 1).
- Produces: `Book(yes: tuple[tuple[float, float], ...], no: tuple[...])` (resting bids as (price, qty)); `Fill(status: str, contracts: int, levels: tuple[tuple[float, float], ...], cost: float, fee: float)` with `.avg_price -> float | None`; `NO_FILL`; `parse_book(payload: dict) -> Book`; `walk(book: Book, side: str, contracts: int, price_cap: float) -> Fill`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_book.py`:

```python
import pytest
from sandbox_helpers import book_payload

from sandbox.book import NO_FILL, parse_book, walk

# NO bids 0.38 x2, 0.36 x5, 0.30 x10  ->  YES asks 0.62 x2, 0.64 x5, 0.70 x10
BOOK = parse_book(book_payload(no=[(0.38, 2), (0.36, 5), (0.30, 10)]))


def test_parse_book_reads_bids():
    assert BOOK.no == ((0.38, 2.0), (0.36, 5.0), (0.30, 10.0))
    assert BOOK.yes == ()


def test_buy_yes_walks_no_bids_cheapest_first():
    f = walk(BOOK, "yes", 3, 0.65)
    assert f.status == "filled" and f.contracts == 3
    assert f.levels == ((0.62, 2.0), (0.64, 1.0))
    assert f.cost == pytest.approx(1.88)
    # per level, rounded up: 0.07*2*0.62*0.38=0.032984 -> 0.04 ; 0.07*0.64*0.36=0.016128 -> 0.02
    assert f.fee == pytest.approx(0.06)
    assert f.avg_price == pytest.approx(1.88 / 3)


def test_price_cap_makes_a_partial_fill():
    f = walk(BOOK, "yes", 3, 0.62)
    assert f.status == "partial" and f.contracts == 2
    assert f.cost == pytest.approx(1.24) and f.fee == pytest.approx(0.04)


def test_nothing_under_the_cap_is_no_fill():
    assert walk(BOOK, "yes", 1, 0.60) == NO_FILL


def test_buy_no_walks_yes_bids():
    book = parse_book(book_payload(yes=[(0.55, 4)]))
    f = walk(book, "no", 1, 0.50)
    assert f.levels == ((0.45, 1.0),)
    assert f.cost == pytest.approx(0.45) and f.fee == pytest.approx(0.02)


def test_fractional_book_sizes_still_fill_whole_contracts():
    book = parse_book(book_payload(no=[(0.40, 1.5), (0.39, 3)]))
    f = walk(book, "yes", 2, 0.61)
    assert f.contracts == 2
    assert f.levels == ((0.60, 1.5), (0.61, 0.5))
    assert f.cost == pytest.approx(1.205)
    assert f.fee == pytest.approx(0.04)  # 0.0252 -> 0.03 ; 0.0083 -> 0.01
    g = walk(book, "yes", 2, 0.60)        # only 1.5 available under the cap
    assert g.status == "partial" and g.contracts == 1 and g.levels == ((0.60, 1.0),)


def test_empty_payload_is_no_fill():
    assert walk(parse_book({}), "yes", 1, 0.99) == NO_FILL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_book.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.book'`

- [ ] **Step 3: Implement `src/sandbox/book.py`**

```python
"""Kalshi order books and the simulated fill. PURE: no network, no database.

Kalshi's book lists resting BIDS only (verified 2026-10-04). Buying YES at p
means hitting a NO bid at 1-p; buying NO means hitting a YES bid at 1-p.
Fees are charged per price level taken, each rounded up to the cent — at
least as large as Kalshi's per-order rounding, so the sandbox never flatters.
"""
import math
from dataclasses import dataclass

from highcast.fees import taker_fee

EPS = 1e-9


@dataclass(frozen=True)
class Book:
    yes: tuple[tuple[float, float], ...]
    no: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Fill:
    status: str              # "filled" | "partial" | "none"
    contracts: int
    levels: tuple[tuple[float, float], ...]
    cost: float
    fee: float

    @property
    def avg_price(self) -> float | None:
        return self.cost / self.contracts if self.contracts else None


NO_FILL = Fill("none", 0, (), 0.0, 0.0)


def _levels(raw) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(p), 4), float(q)) for p, q in (raw or []))


def parse_book(payload: dict) -> Book:
    ob = (payload or {}).get("orderbook_fp") or {}
    return Book(yes=_levels(ob.get("yes_dollars")), no=_levels(ob.get("no_dollars")))


def walk(book: Book, side: str, contracts: int, price_cap: float) -> Fill:
    opposite = book.no if side == "yes" else book.yes
    asks = sorted((round(1.0 - p, 4), q) for p, q in opposite if 0.0 < p < 1.0 and q > 0)
    usable = [(p, q) for p, q in asks if p <= price_cap + EPS]
    available = sum(q for _, q in usable)
    n = min(contracts, int(math.floor(available + EPS)))
    if n <= 0:
        return NO_FILL
    remaining = float(n)
    taken: list[tuple[float, float]] = []
    for p, q in usable:
        if remaining <= EPS:
            break
        take = min(remaining, q)
        taken.append((p, round(take, 6)))
        remaining -= take
    cost = round(sum(p * q for p, q in taken), 4)
    fee = round(sum(taker_fee(q, p) for p, q in taken), 2)
    return Fill("filled" if n == contracts else "partial", n, tuple(taken), cost, fee)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_book.py -v`
Expected: 7 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 4: Pricing (de-vig, Brier) and the ledger

**Files:**
- Create: `src/sandbox/pricing.py`, `src/sandbox/ledger.py`
- Test: `tests/test_sandbox_pricing.py`, `tests/test_sandbox_ledger.py`

**Interfaces:**
- Consumes: `Market`, `Intent` (Task 2); `usable_bid`, `usable_ask` (Task 2); `Fill`, `NO_FILL` (Task 3); `db.connect` (Task 2).
- Produces: `pricing.mid(m: Market) -> float | None`; `pricing.devig(markets: list[Market]) -> dict[str, float]`; `pricing.multiclass_brier(probs: dict[str, float], winner: str) -> float`; `ledger.START_CASH = 5.0`; `ledger.open_account(conn, name, kind, label="") -> None`; `ledger.cash(conn, account) -> float`; `ledger.holds(conn, account, ticker) -> bool`; `ledger.place(conn, intent, fill, decided_at, book_json) -> tuple[int, str]` (order id, final status).

- [ ] **Step 1: Write the failing pricing test**

Create `tests/test_sandbox_pricing.py`:

```python
import pytest
from sandbox_helpers import make_market

from sandbox.pricing import devig, mid, multiclass_brier


def test_mid_uses_whatever_side_exists():
    assert mid(make_market("A", bid=0.20, ask=0.24)) == pytest.approx(0.22)
    assert mid(make_market("A", bid=0.0, ask=0.30)) == pytest.approx(0.30)
    assert mid(make_market("A", bid=0.40, ask=1.0)) == pytest.approx(0.40)
    assert mid(make_market("A", bid=0.0, ask=1.0)) is None


def test_devig_normalises_the_ladder():
    ms = [make_market("A", bid=0.20, ask=0.24), make_market("B", bid=0.50, ask=0.56),
          make_market("C", bid=0.0, ask=0.30)]
    p = devig(ms)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["A"] == pytest.approx(0.22 / 1.05)
    assert p["B"] == pytest.approx(0.53 / 1.05)


def test_multiclass_brier():
    assert multiclass_brier({"A": 0.2, "B": 0.5, "C": 0.3}, "B") == pytest.approx(0.38)
    assert multiclass_brier({"A": 1.0}, "B") == pytest.approx(2.0)  # winner missing counts as 0
```

- [ ] **Step 2: Implement `src/sandbox/pricing.py`**

```python
"""Market probabilities and scoring."""
from sandbox.markets import usable_ask, usable_bid
from sandbox.models import Market


def mid(m: Market) -> float | None:
    bid, ask = usable_bid(m), usable_ask(m)
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    return ask if ask is not None else bid


def devig(markets: list[Market]) -> dict[str, float]:
    """One event's ladder -> probabilities summing to 1 (mids rescaled)."""
    mids = {m.ticker: v for m in markets if (v := mid(m)) is not None}
    total = sum(mids.values())
    return {t: v / total for t, v in mids.items()} if total > 0 else {}


def multiclass_brier(probs: dict[str, float], winner: str) -> float:
    score = sum((p - (1.0 if t == winner else 0.0)) ** 2 for t, p in probs.items())
    return score + (0.0 if winner in probs else 1.0)
```

- [ ] **Step 3: Run the pricing test**

Run: `.venv/bin/pytest tests/test_sandbox_pricing.py -v`
Expected: 3 passed

- [ ] **Step 4: Write the failing ledger test**

Create `tests/test_sandbox_ledger.py`:

```python
import pytest

from sandbox.book import NO_FILL, Fill
from sandbox.db import connect
from sandbox.ledger import START_CASH, cash, holds, open_account, place
from sandbox.models import Intent

T = "2026-10-04T10:00:00+00:00"


def intent(ticker="A", account="mos"):
    return Intent(account=account, ticker=ticker, side="yes", contracts=3,
                  price_cap=0.65, snapshot_id=1, p_bot=0.7, p_market=0.6)


def test_new_account_starts_with_five_dollars():
    conn = connect(":memory:")
    open_account(conn, "mos", "bot", "label")
    open_account(conn, "mos", "bot", "label")  # idempotent
    assert START_CASH == 5.0
    assert cash(conn, "mos") == pytest.approx(5.0)


def test_fill_debits_cost_and_fee():
    conn = connect(":memory:")
    open_account(conn, "mos", "bot")
    fill = Fill("filled", 3, ((0.62, 2.0), (0.64, 1.0)), 1.88, 0.06)
    oid, status = place(conn, intent(), fill, T, '{"no_dollars": []}')
    assert status == "filled" and oid == 1
    assert cash(conn, "mos") == pytest.approx(3.06)
    assert holds(conn, "mos", "A") and not holds(conn, "mos", "B")


def test_insufficient_cash_records_a_rejected_order():
    conn = connect(":memory:")
    open_account(conn, "mos", "bot")
    big = Fill("filled", 10, ((0.62, 10.0),), 6.20, 0.17)
    _, status = place(conn, intent(), big, T, None)
    assert status == "insufficient_cash"
    assert cash(conn, "mos") == pytest.approx(5.0)
    assert not holds(conn, "mos", "A")


def test_no_fill_is_recorded_but_costs_nothing():
    conn = connect(":memory:")
    open_account(conn, "mos", "bot")
    _, status = place(conn, intent(), NO_FILL, T, None)
    assert status == "none"
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    assert cash(conn, "mos") == pytest.approx(5.0)
```

- [ ] **Step 5: Implement `src/sandbox/ledger.py`**

```python
"""Paper accounts. Cash is always derived from orders, never stored."""
import json
import sqlite3

from sandbox.book import NO_FILL, Fill
from sandbox.models import Intent

START_CASH = 5.0


def open_account(conn: sqlite3.Connection, name: str, kind: str, label: str = "") -> None:
    conn.execute("INSERT OR IGNORE INTO accounts(name, kind, start_cash, label) "
                 "VALUES (?, ?, ?, ?)", (name, kind, START_CASH, label))
    conn.commit()


def cash(conn: sqlite3.Connection, account: str) -> float:
    start = conn.execute("SELECT start_cash FROM accounts WHERE name=?", (account,)).fetchone()
    if start is None:
        raise KeyError(account)
    spent, paid = conn.execute(
        "SELECT COALESCE(SUM(cost + fee), 0), COALESCE(SUM(payout), 0) "
        "FROM orders WHERE account=?", (account,)).fetchone()
    return round(start["start_cash"] - spent + paid, 4)


def holds(conn: sqlite3.Connection, account: str, ticker: str) -> bool:
    return conn.execute("SELECT 1 FROM orders WHERE account=? AND ticker=? "
                        "AND filled_contracts > 0 LIMIT 1", (account, ticker)).fetchone() is not None


def place(conn: sqlite3.Connection, intent: Intent, fill: Fill, decided_at: str,
          book_json: str | None) -> tuple[int, str]:
    status, f = fill.status, fill
    if f.contracts > 0 and f.cost + f.fee > cash(conn, intent.account) + 1e-9:
        status, f = "insufficient_cash", NO_FILL
    cur = conn.execute(
        """INSERT INTO orders(account, ticker, side, contracts_requested, price_cap,
                              snapshot_id, decided_at, status, filled_contracts, cost, fee,
                              levels_json, book_json, p_bot, p_market)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (intent.account, intent.ticker, intent.side, intent.contracts, intent.price_cap,
         intent.snapshot_id, decided_at, status, f.contracts, f.cost, f.fee,
         json.dumps(f.levels), book_json, intent.p_bot, intent.p_market))
    conn.commit()
    return cur.lastrowid, status
```

- [ ] **Step 6: Run both tests**

Run: `.venv/bin/pytest tests/test_sandbox_pricing.py tests/test_sandbox_ledger.py -v`
Expected: 7 passed

- [ ] **Step 7: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 5: Bot framework, edge rule, and the `favorite` bot

**Files:**
- Create: `src/sandbox/bots.py`
- Test: `tests/test_sandbox_bots.py`

**Interfaces:**
- Consumes: `Market`, `Snapshot`, `Intent`, `Candidate` (Task 2); `usable_ask`, `usable_bid` (Task 2); `devig` (Task 4); `highcast.fees.fee_per_contract(contracts, price) -> float`; `highcast.orderbot.MIN_EDGE`.
- Produces: `MIN_EDGE`; `SIGMA_F = 3.0`; `Decision(intents: list[Intent], candidates: list[Candidate])` (NamedTuple); `evaluate_bucket(p: float, ask: float | None) -> tuple[bool, float | None, float | None, str]` returning (ok, fee_per_contract, edge_after_fee, reason); `normal_bucket_probs(markets, mu, sigma) -> dict[str, float]`; `minutes_between(start_iso, end_iso) -> float | None`; `class ForecastBot` with `name: str`, `pmf_for_event(markets, snapshot) -> dict[str, float] | None` (override), `decide(snapshot) -> Decision`; `class FavoriteBot` with `name = "favorite"`, `decide(snapshot) -> Decision`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_bots.py`:

```python
import json

import pytest
from sandbox_helpers import three_buckets

from highcast import orderbot
from sandbox.bots import (MIN_EDGE, ForecastBot, FavoriteBot, evaluate_bucket,
                          minutes_between, normal_bucket_probs)
from sandbox.models import Snapshot

T = "2026-10-04T10:00:00+00:00"


def snap(markets):
    return Snapshot(id=7, taken_at=T, markets=tuple(markets))


def test_min_edge_is_the_order_bots():
    assert MIN_EDGE == orderbot.MIN_EDGE == 0.03


def test_normal_bucket_probs_symmetric_ladder():
    p = normal_bucket_probs(three_buckets(), mu=69.5, sigma=3.0)
    t69, mid, t70 = (m.ticker for m in three_buckets())
    assert p[t69] == pytest.approx(0.3694, abs=1e-3)
    assert p[t70] == pytest.approx(0.3694, abs=1e-3)
    assert p[mid] == pytest.approx(0.2611, abs=1e-3)
    assert sum(p.values()) == pytest.approx(1.0)


def test_evaluate_bucket():
    ok, fpc, edge, why = evaluate_bucket(0.50, 0.40)   # fee 0.07*.4*.6=.0168 -> .02
    assert ok and fpc == pytest.approx(0.02) and edge == pytest.approx(0.08) and why == "edge"
    assert evaluate_bucket(0.42, 0.40)[0] is False
    assert evaluate_bucket(0.42, 0.40)[3] == "below_min_edge"
    assert evaluate_bucket(0.50, None) == (False, None, None, "no_ask")


def test_minutes_between():
    assert minutes_between(T, "2026-10-04T11:30:00+00:00") == pytest.approx(90.0)
    assert minutes_between(T, None) is None


class FixedPmf(ForecastBot):
    name = "fixed"

    def __init__(self, mu):
        self.mu = mu

    def pmf_for_event(self, markets, snapshot):
        return None if self.mu is None else normal_bucket_probs(markets, self.mu, 3.0)


def test_forecast_bot_logs_every_bucket_and_bets_only_on_edge():
    d = FixedPmf(69.5).decide(snap(three_buckets()))
    assert len(d.candidates) == 3
    placed = [c for c in d.candidates if c.placed]
    # ≤68: p .369 - ask .20 - fee .02 = .149 -> bet ; others negative
    assert [c.ticker for c in placed] == ["KXHIGHNY-26OCT04-T69"]
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T69"]
    it = d.intents[0]
    assert (it.account, it.side, it.contracts, it.price_cap, it.snapshot_id) == ("fixed", "yes", 1, 0.20, 7)
    c0 = d.candidates[0]
    assert c0.spread == pytest.approx(0.05) and c0.ask_size == 3.0
    assert c0.minutes_to_close == pytest.approx(1140.0)
    assert sum(json.loads(c0.pmf_json).values()) == pytest.approx(1.0)
    assert c0.gates_json is None and c0.regime_json is None


def test_forecast_bot_without_a_forecast_does_nothing():
    d = FixedPmf(None).decide(snap(three_buckets()))
    assert d.intents == [] and d.candidates == []


def test_favorite_buys_the_most_likely_bucket_at_the_ask():
    d = FavoriteBot().decide(snap(three_buckets()))
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T70"]   # mid .425 is largest
    assert d.intents[0].price_cap == 0.45 and d.intents[0].account == "favorite"
    assert d.candidates == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_bots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.bots'`

- [ ] **Step 3: Implement `src/sandbox/bots.py` (framework + favorite)**

```python
"""Sandbox bots. Each bot sees one Snapshot and returns intents + candidates.

Every bot bets 1 contract on the YES side. Stake sizing is out of scope (spec §9).
"""
import json
import math
from datetime import datetime
from typing import NamedTuple

from highcast.fees import fee_per_contract
from highcast.orderbot import MIN_EDGE  # 0.03 — one constant shared with the live desk
from sandbox.markets import usable_ask, usable_bid
from sandbox.models import Candidate, Intent, Market, Snapshot
from sandbox.pricing import devig

SIGMA_F = 3.0   # MOS spread prior in °F — a guess, not a measurement (spec §6)

__all__ = ["MIN_EDGE", "SIGMA_F", "Decision", "evaluate_bucket", "normal_bucket_probs",
           "minutes_between", "ForecastBot", "FavoriteBot"]


class Decision(NamedTuple):
    intents: list[Intent]
    candidates: list[Candidate]


def evaluate_bucket(p: float, ask: float | None) -> tuple[bool, float | None, float | None, str]:
    if ask is None:
        return False, None, None, "no_ask"
    fpc = fee_per_contract(1, ask)
    edge = round(p - ask - fpc, 6)
    if edge >= MIN_EDGE:
        return True, fpc, edge, "edge"
    return False, fpc, edge, "below_min_edge"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_bucket_probs(markets: list[Market], mu: float, sigma: float) -> dict[str, float]:
    """Normal(mu, sigma) over integer °F outcomes, summed into each bucket."""
    raw = {}
    for m in markets:
        upper = 1.0 if m.hi is None else _phi((m.hi + 0.5 - mu) / sigma)
        lower = 0.0 if m.lo is None else _phi((m.lo - 0.5 - mu) / sigma)
        raw[m.ticker] = max(upper - lower, 0.0)
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()} if total > 0 else {}


def minutes_between(start_iso: str, end_iso: str | None) -> float | None:
    if not end_iso:
        return None
    delta = datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
    return delta.total_seconds() / 60.0


class ForecastBot:
    """Base for bots that produce a probability per bucket. Logs every bucket."""
    name = "forecast"

    def pmf_for_event(self, markets: list[Market], snapshot: Snapshot) -> dict[str, float] | None:
        raise NotImplementedError

    def decide(self, snapshot: Snapshot) -> Decision:
        intents: list[Intent] = []
        candidates: list[Candidate] = []
        for markets in snapshot.events().values():
            pmf = self.pmf_for_event(markets, snapshot)
            if not pmf:
                continue
            market_p = devig(markets)
            pmf_json = json.dumps(pmf, sort_keys=True)
            for m in markets:
                p = pmf.get(m.ticker)
                if p is None:
                    continue
                ask, bid = usable_ask(m), usable_bid(m)
                ok, fpc, edge, reason = evaluate_bucket(p, ask)
                candidates.append(Candidate(
                    bot=self.name, ticker=m.ticker, series=m.series, station=m.station,
                    metric=m.metric, event_date=m.event_date, snapshot_id=snapshot.id,
                    decided_at=snapshot.taken_at, p_bot=p, pmf_json=pmf_json,
                    p_market=market_p.get(m.ticker), yes_bid=bid, yes_ask=ask,
                    spread=None if ask is None or bid is None else round(ask - bid, 4),
                    ask_size=m.yes_ask_size,
                    minutes_to_close=minutes_between(snapshot.taken_at, m.close_time),
                    fee_per_contract=fpc, edge_after_fee=edge, placed=ok, reason=reason))
                if ok:
                    intents.append(Intent(account=self.name, ticker=m.ticker, side="yes",
                                          contracts=1, price_cap=ask,
                                          snapshot_id=snapshot.id, p_bot=p,
                                          p_market=market_p.get(m.ticker)))
        return Decision(intents, candidates)


class FavoriteBot:
    """Control: buy YES on each event's most likely bucket at the ask."""
    name = "favorite"

    def decide(self, snapshot: Snapshot) -> Decision:
        intents: list[Intent] = []
        for markets in snapshot.events().values():
            probs = devig(markets)
            buyable = [m for m in markets if usable_ask(m) is not None and m.ticker in probs]
            if not buyable:
                continue
            best = max(buyable, key=lambda m: probs[m.ticker])
            intents.append(Intent(account=self.name, ticker=best.ticker, side="yes",
                                  contracts=1, price_cap=usable_ask(best),
                                  snapshot_id=snapshot.id, p_bot=probs[best.ticker],
                                  p_market=probs[best.ticker]))
        return Decision(intents, [])
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_bots.py -v`
Expected: 8 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 6: The `mos` bot and its IEM MOS source

**Files:**
- Modify: `src/sandbox/bots.py` (append; add names to `__all__`)
- Test: `tests/test_sandbox_mos.py`

**Interfaces:**
- Consumes: `ForecastBot`, `normal_bucket_probs`, `SIGMA_F` (Task 5); `STATIONS` (Task 1); `highcast.fetchers.forecasts.parse_mos(payload, icao, fetched_at, model_name, tz, extremum, std_utc_offset) -> list[dict]` (rows with `target_date`, `run_date`, `max_temp_f`); `highcast.fetchers.forecasts.IEM_MOS`; `highcast.http.get_json`.
- Produces: `class IemMos(fetch=get_json, model="MEX")` with `value(icao: str, metric: str, event_date: str) -> float | None`; `class MosBot(source, sigma=SIGMA_F)` with `name = "mos"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_mos.py`:

```python
import pytest
from sandbox_helpers import three_buckets

from sandbox.bots import IemMos, MosBot
from sandbox.models import Snapshot

T = "2026-10-04T10:00:00+00:00"

# KNYC, standard offset -5. 00Z on Oct 5 = 20:00 EDT Oct 4 (afternoon row -> MAX),
# 12Z on Oct 4 = 08:00 EDT (morning row -> MIN). Both fall on climate day Oct 4.
PAYLOAD = {"data": [
    {"runtime": "2026-10-04 12:00", "ftime": "2026-10-05 00:00", "n_x": 71},
    {"runtime": "2026-10-04 00:00", "ftime": "2026-10-04 12:00", "n_x": 58},
]}


def test_iem_mos_value_picks_the_extremum_for_the_climate_day():
    calls = []
    src = IemMos(fetch=lambda url, params: calls.append(params) or PAYLOAD)
    assert src.value("KNYC", "high", "2026-10-04") == 71.0
    assert src.value("KNYC", "low", "2026-10-04") == 58.0
    assert calls == [{"station": "KNYC", "model": "MEX"}]  # cached per station


def test_iem_mos_unknown_station_or_fetch_error_is_none():
    def boom(url, params):
        raise RuntimeError("down")
    assert IemMos(fetch=boom).value("KNYC", "high", "2026-10-04") is None
    assert IemMos(fetch=lambda u, p: PAYLOAD).value("KXXX", "high", "2026-10-04") is None


class FakeSource:
    def __init__(self, v):
        self.v = v
        self.asked = []

    def value(self, icao, metric, event_date):
        self.asked.append((icao, metric, event_date))
        return self.v


def test_mos_bot_uses_station_metric_and_date_from_the_market():
    src = FakeSource(69.5)
    d = MosBot(src).decide(Snapshot(id=1, taken_at=T, markets=tuple(three_buckets())))
    assert src.asked == [("KNYC", "high", "2026-10-04")]
    assert len(d.candidates) == 3 and all(c.bot == "mos" for c in d.candidates)
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T69"]


def test_mos_bot_skips_events_without_a_forecast():
    d = MosBot(FakeSource(None)).decide(Snapshot(id=1, taken_at=T, markets=tuple(three_buckets())))
    assert d.candidates == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_mos.py -v`
Expected: FAIL with `ImportError: cannot import name 'IemMos'`

- [ ] **Step 3: Append to `src/sandbox/bots.py`**

Add these imports to the top import block of `src/sandbox/bots.py`:

```python
from highcast.fetchers.forecasts import IEM_MOS, parse_mos
from highcast.http import get_json
from sandbox.stations import STATIONS
```

Append to the end of the file:

```python
class IemMos:
    """MOS MEX daily max/min per (station, metric, climate day), fetched once per
    station per instance. Build a fresh instance each cycle."""

    def __init__(self, fetch=get_json, model: str = "MEX"):
        self.fetch = fetch
        self.model = model
        self._payloads: dict[str, dict] = {}

    def value(self, icao: str, metric: str, event_date: str) -> float | None:
        if icao not in STATIONS:
            return None
        if icao not in self._payloads:
            try:
                self._payloads[icao] = self.fetch(IEM_MOS, {"station": icao, "model": self.model})
            except Exception:  # noqa: BLE001 - no forecast this cycle, not a crash
                self._payloads[icao] = {}
        tz, offset = STATIONS[icao]
        rows = parse_mos(self._payloads[icao], icao, "", f"mos_{self.model.lower()}",
                         tz, "max" if metric == "high" else "min", offset)
        matches = sorted((r for r in rows if r["target_date"] == event_date),
                         key=lambda r: r["run_date"])
        return float(matches[-1]["max_temp_f"]) if matches else None


class MosBot(ForecastBot):
    """Normal(MOS, SIGMA_F) over each event's buckets. Uncalibrated by design."""
    name = "mos"

    def __init__(self, source, sigma: float = SIGMA_F):
        self.source = source
        self.sigma = sigma

    def pmf_for_event(self, markets, snapshot):
        first = markets[0]
        if first.station is None or first.metric is None:
            return None
        mu = self.source.value(first.station, first.metric, first.event_date)
        if mu is None:
            return None
        return normal_bucket_probs(markets, mu, self.sigma)
```

Add `"IemMos", "MosBot"` to `__all__`.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_mos.py tests/test_sandbox_bots.py -v`
Expected: 12 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 7: The `highcast` bot (read-only)

**Files:**
- Modify: `src/sandbox/bots.py` (append; add names to `__all__`)
- Test: `tests/test_sandbox_highcast_bot.py`

**Interfaces:**
- Consumes: `ForecastBot` (Task 5); `highcast.buckets.pmf_mass(pmf, lo, hi) -> float | None`, `highcast.buckets.pmf_from_buckets(dict) -> dict[int, float]`; highcast table `predictions(station, target_date, issued_at_utc, kind, point_f, quantiles, buckets, metric)` where `buckets` is JSON `{"67": 0.2, ...}` and `issued_at_utc` looks like `2026-10-03T18:09:54+00:00`.
- Produces: `HIGHCAST_STATIONS = {"KLAX", "KMIA"}`; `class HighcastBot(highcast_db: str)` with `name = "highcast"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_highcast_bot.py`:

```python
import json
import sqlite3

import pytest
from sandbox_helpers import three_buckets

from sandbox.bots import HighcastBot
from sandbox.models import Snapshot

T = "2026-10-04T11:00:00+00:00"


def make_highcast_db(path):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE predictions (station TEXT, target_date TEXT, issued_at_utc TEXT,
                 kind TEXT, point_f REAL, quantiles TEXT, buckets TEXT, metric TEXT)""")
    rows = [
        ("2026-10-04T10:00:00+00:00", {"67": 0.2, "68": 0.2, "69": 0.3, "70": 0.2, "71": 0.1}),
        ("2026-10-04T12:00:00+00:00", {"71": 1.0}),   # AFTER the snapshot: must be ignored
    ]
    for issued, b in rows:
        c.execute("INSERT INTO predictions VALUES ('KLAX','2026-10-04',?, 'intraday', NULL, NULL, ?, 'high')",
                  (issued, json.dumps(b)))
    c.commit()
    c.close()


def lax_buckets():
    return three_buckets(series="KXHIGHLAX", station="KLAX", event="KXHIGHLAX-26OCT04")


def test_reads_latest_prediction_at_or_before_the_snapshot(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    markets = lax_buckets()
    pmf = HighcastBot(str(db)).pmf_for_event(markets, Snapshot(1, T, tuple(markets)))
    t69, mid, t70 = (m.ticker for m in markets)
    assert pmf[t69] == pytest.approx(0.4)
    assert pmf[mid] == pytest.approx(0.5)
    assert pmf[t70] == pytest.approx(0.1)


def test_other_stations_and_missing_db_give_nothing(tmp_path):
    ny = three_buckets()
    assert HighcastBot(str(tmp_path / "x.db")).pmf_for_event(ny, Snapshot(1, T, tuple(ny))) is None
    lax = lax_buckets()
    assert HighcastBot(str(tmp_path / "missing.db")).pmf_for_event(lax, Snapshot(1, T, tuple(lax))) is None


def test_never_writes_to_the_highcast_db(tmp_path):
    db = tmp_path / "highcast.db"
    make_highcast_db(db)
    before = db.read_bytes()
    markets = lax_buckets()
    HighcastBot(str(db)).decide(Snapshot(1, T, tuple(markets)))
    assert db.read_bytes() == before
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_highcast_bot.py -v`
Expected: FAIL with `ImportError: cannot import name 'HighcastBot'`

- [ ] **Step 3: Append to `src/sandbox/bots.py`**

Add these imports to the top import block:

```python
import sqlite3
from contextlib import closing
from pathlib import Path

from highcast.buckets import pmf_from_buckets, pmf_mass
```

Append:

```python
HIGHCAST_STATIONS = {"KLAX", "KMIA"}


class HighcastBot(ForecastBot):
    """The shipped highcast model, read from highcast.db READ-ONLY."""
    name = "highcast"

    def __init__(self, highcast_db: str):
        self.highcast_db = highcast_db

    def _latest_pmf(self, station: str, metric: str, event_date: str,
                    at_or_before: str) -> dict[int, float] | None:
        path = Path(self.highcast_db)
        if not path.exists():
            return None
        uri = f"file:{path.resolve()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as c:
                row = c.execute(
                    """SELECT buckets FROM predictions
                       WHERE station=? AND metric=? AND target_date=? AND issued_at_utc<=?
                         AND buckets IS NOT NULL
                       ORDER BY issued_at_utc DESC LIMIT 1""",
                    (station, metric, event_date, at_or_before)).fetchone()
        except sqlite3.Error:
            return None
        return pmf_from_buckets(json.loads(row[0])) if row else None

    def pmf_for_event(self, markets, snapshot):
        first = markets[0]
        if first.station not in HIGHCAST_STATIONS or first.metric is None:
            return None
        pmf = self._latest_pmf(first.station, first.metric, first.event_date, snapshot.taken_at)
        if not pmf:
            return None
        raw = {m.ticker: pmf_mass(pmf, m.lo, m.hi) or 0.0 for m in markets}
        total = sum(raw.values())
        return {t: v / total for t, v in raw.items()} if total > 0 else None
```

Add `"HIGHCAST_STATIONS", "HighcastBot"` to `__all__`.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_highcast_bot.py -v`
Expected: 3 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 8: Settlement — pay orders and label candidates

**Files:**
- Create: `src/sandbox/settle.py`
- Test: `tests/test_sandbox_settle.py`

**Interfaces:**
- Consumes: `settled_markets(get, series)` (Task 1); tables from Task 2; `record_snapshot`, `record_candidates` (Task 2); `place`, `open_account`, `cash` (Task 4).
- Produces: `settle_due(conn, get, now_iso: str) -> int` (number of markets newly settled).

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_settle.py`:

```python
import pytest
from sandbox_helpers import three_buckets

from sandbox.book import Fill
from sandbox.db import connect
from sandbox.ledger import cash, open_account, place
from sandbox.models import Candidate, Intent
from sandbox.recorder import record_candidates, record_snapshot
from sandbox.settle import settle_due

T = "2026-10-04T10:00:00+00:00"
AFTER = "2026-10-05T12:00:00+00:00"      # markets close 2026-10-05T05:00:00+00:00
A, B, C = (m.ticker for m in three_buckets())


def candidate(ticker, placed, ask, fpc, sid):
    return Candidate(bot="mos", ticker=ticker, series="KXHIGHNY", station="KNYC",
                     metric="high", event_date="2026-10-04", snapshot_id=sid,
                     decided_at=T, p_bot=0.4, pmf_json="{}", p_market=0.3,
                     yes_bid=None, yes_ask=ask, spread=None, ask_size=None,
                     minutes_to_close=None, fee_per_contract=fpc, edge_after_fee=0.1,
                     placed=placed, reason="edge" if placed else "below_min_edge")


def setup():
    conn = connect(":memory:")
    sid = record_snapshot(conn, three_buckets(), T)
    open_account(conn, "mos", "bot")
    place(conn, Intent("mos", A, "yes", 1, 0.20, sid, 0.4, 0.3),
          Fill("filled", 1, ((0.20, 1.0),), 0.20, 0.02), T, None)
    record_candidates(conn, [candidate(A, True, 0.20, 0.02, sid),
                             candidate(B, False, 0.30, 0.02, sid)])
    return conn


def get_settled(path, params=None):
    assert path == "/markets" and params["status"] == "settled"
    return {"markets": [
        {"ticker": A, "status": "finalized", "result": "yes"},
        {"ticker": B, "status": "finalized", "result": "no"},
        {"ticker": C, "status": "finalized", "result": "no"},
    ]}


def test_settle_pays_winners_once():
    conn = setup()
    assert settle_due(conn, get_settled, AFTER) == 3
    assert cash(conn, "mos") == pytest.approx(5.0 - 0.22 + 1.0)
    assert settle_due(conn, get_settled, AFTER) == 0          # idempotent
    assert cash(conn, "mos") == pytest.approx(5.78)


def test_skipped_candidates_are_labelled_too():
    conn = setup()
    settle_due(conn, get_settled, AFTER)
    rows = {r["ticker"]: (r["result"], r["asif_profit"])
            for r in conn.execute("SELECT ticker, result, asif_profit FROM candidates")}
    assert rows[A] == ("yes", pytest.approx(1 - 0.20 - 0.02))
    assert rows[B] == ("no", pytest.approx(-0.30 - 0.02))


def test_settlement_only_touches_outcome_fields():
    conn = setup()
    cols = "bot, ticker, snapshot_id, p_bot, pmf_json, yes_ask, fee_per_contract, placed, reason"
    before = [tuple(r) for r in conn.execute(f"SELECT {cols} FROM candidates ORDER BY id")]
    settle_due(conn, get_settled, AFTER)
    after = [tuple(r) for r in conn.execute(f"SELECT {cols} FROM candidates ORDER BY id")]
    assert before == after


def test_nothing_is_due_before_close():
    conn = setup()
    calls = []
    assert settle_due(conn, lambda p, q=None: calls.append(p) or {}, T) == 0
    assert calls == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_settle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.settle'`

- [ ] **Step 3: Implement `src/sandbox/settle.py`**

```python
"""Settle on Kalshi's own result: $1 per winning contract, no settlement fee."""
import logging
import sqlite3

from sandbox.kalshi_public import settled_markets

log = logging.getLogger(__name__)

_RESULT = "(SELECT result FROM markets WHERE markets.ticker = {t}.ticker)"


def settle_due(conn: sqlite3.Connection, get, now_iso: str) -> int:
    due = [r["series"] for r in conn.execute(
        "SELECT DISTINCT series FROM markets WHERE result IS NULL "
        "AND close_time IS NOT NULL AND close_time <= ?", (now_iso,))]
    newly = 0
    for series in due:
        try:
            payload = settled_markets(get, series)
        except Exception as e:  # noqa: BLE001 - try again next hour
            log.warning("settle fetch failed for %s: %r", series, e)
            continue
        for m in payload.get("markets") or []:
            if m.get("status") in ("finalized", "settled") and m.get("result") in ("yes", "no"):
                newly += conn.execute(
                    "UPDATE markets SET result=?, settled_at=? WHERE ticker=? AND result IS NULL",
                    (m["result"], now_iso, m["ticker"])).rowcount
    o = _RESULT.format(t="orders")
    conn.execute(f"""UPDATE orders
                     SET payout = CASE WHEN {o} = orders.side THEN orders.filled_contracts * 1.0
                                       ELSE 0.0 END
                     WHERE payout IS NULL AND filled_contracts > 0 AND {o} IS NOT NULL""")
    c = _RESULT.format(t="candidates")
    conn.execute(f"""UPDATE candidates
                     SET result = {c},
                         asif_profit = CASE
                           WHEN yes_ask IS NULL OR fee_per_contract IS NULL THEN NULL
                           ELSE ROUND((CASE WHEN {c} = 'yes' THEN 1.0 ELSE 0.0 END)
                                      - yes_ask - fee_per_contract, 4) END
                     WHERE result IS NULL AND {c} IS NOT NULL""")
    conn.commit()
    return newly
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_settle.py -v`
Expected: 4 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 9: Scoreboard and the unlock verdict

**Files:**
- Create: `src/sandbox/scoreboard.py`
- Test: `tests/test_sandbox_scoreboard.py`

**Interfaces:**
- Consumes: `row_to_market` (Task 2); `devig`, `multiclass_brier` (Task 4); `cash` (Task 4); tables from Task 2 with `payout` from Task 8.
- Produces: `N_REQUIRED = 100`; `bootstrap_mean_ci(values, n=2000, seed=20261004) -> tuple[float, float]`; `Scorecard(account, cash, settled_bets, profit, profit_ci, brier_diff, brier_ci, profit_pass, score_pass)` with `.unlocked -> bool`; `report(conn, account) -> Scorecard`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_scoreboard.py`:

```python
import json

import pytest
from sandbox_helpers import make_market

from sandbox.book import Fill
from sandbox.db import connect
from sandbox.ledger import open_account, place
from sandbox.models import Candidate, Intent
from sandbox.recorder import record_candidates, record_snapshot
from sandbox.scoreboard import N_REQUIRED, Scorecard, bootstrap_mean_ci, report

T = "2026-10-04T10:00:00+00:00"


def test_bootstrap_of_a_constant_is_that_constant():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def card(n, p, s):
    return Scorecard("mos", 5.0, n, 0.0, None, None, None, p, s)


def test_unlock_needs_all_three():
    assert N_REQUIRED == 100
    assert card(100, True, True).unlocked
    assert not card(99, True, True).unlocked
    assert not card(100, False, True).unlocked
    assert not card(100, True, False).unlocked


def test_report_profit_and_brier_against_the_market_at_bet_time():
    conn = connect(":memory:")
    ms = [make_market("A", bid=0.20, ask=0.24), make_market("B", bid=0.50, ask=0.56),
          make_market("C", bid=0.0, ask=0.30)]
    sid = record_snapshot(conn, ms, T)
    open_account(conn, "mos", "bot")
    place(conn, Intent("mos", "B", "yes", 1, 0.56, sid, 0.8, 0.5),
          Fill("filled", 1, ((0.56, 1.0),), 0.56, 0.02), T, None)
    pmf = {"A": 0.1, "B": 0.8, "C": 0.1}
    record_candidates(conn, [Candidate(
        bot="mos", ticker="B", series="KXHIGHNY", station="KNYC", metric="high",
        event_date="2026-10-04", snapshot_id=sid, decided_at=T, p_bot=0.8,
        pmf_json=json.dumps(pmf), p_market=0.5, yes_bid=0.5, yes_ask=0.56, spread=0.06,
        ask_size=3.0, minutes_to_close=None, fee_per_contract=0.02, edge_after_fee=0.22,
        placed=True, reason="edge")])
    conn.execute("UPDATE markets SET result = CASE ticker WHEN 'B' THEN 'yes' ELSE 'no' END")
    conn.execute("UPDATE orders SET payout = 1.0")
    conn.commit()
    sc = report(conn, "mos")
    assert sc.settled_bets == 1
    assert sc.profit == pytest.approx(0.42)
    # bot brier .06 ; market (.2095, .5048, .2857) brier .3708 -> diff -.3108
    assert sc.brier_diff == pytest.approx(-0.3108, abs=1e-3)
    assert sc.profit_ci is None and not sc.profit_pass   # n < 2
    assert not sc.unlocked
    assert sc.cash == pytest.approx(5.42)


def test_report_on_an_empty_account():
    conn = connect(":memory:")
    open_account(conn, "favorite", "bot")
    sc = report(conn, "favorite")
    assert (sc.settled_bets, sc.profit, sc.brier_diff, sc.unlocked) == (0, 0.0, None, False)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_scoreboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.scoreboard'`

- [ ] **Step 3: Implement `src/sandbox/scoreboard.py`**

```python
"""Per-account evidence and the unlock verdict (spec §1)."""
import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from sandbox.ledger import cash
from sandbox.pricing import devig, multiclass_brier
from sandbox.recorder import row_to_market

N_REQUIRED = 100
BOOT_N = 2000
SEED = 20261004


def bootstrap_mean_ci(values, n: int = BOOT_N, seed: int = SEED) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 6), round(float(hi), 6)


@dataclass(frozen=True)
class Scorecard:
    account: str
    cash: float
    settled_bets: int
    profit: float
    profit_ci: tuple[float, float] | None
    brier_diff: float | None
    brier_ci: tuple[float, float] | None
    profit_pass: bool
    score_pass: bool

    @property
    def unlocked(self) -> bool:
        return self.settled_bets >= N_REQUIRED and self.profit_pass and self.score_pass


def _winner(conn: sqlite3.Connection, event_ticker: str) -> str | None:
    rows = conn.execute("SELECT ticker, result FROM markets WHERE event_ticker=?",
                        (event_ticker,)).fetchall()
    if not rows or any(r["result"] is None for r in rows):
        return None
    winners = [r["ticker"] for r in rows if r["result"] == "yes"]
    return winners[0] if len(winners) == 1 else None


def _market_probs(conn: sqlite3.Connection, snapshot_id: int, event_ticker: str) -> dict[str, float]:
    rows = conn.execute(
        """SELECT m.*, q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size
           FROM quotes q JOIN markets m ON m.ticker = q.ticker
           WHERE q.snapshot_id = ? AND m.event_ticker = ?""",
        (snapshot_id, event_ticker)).fetchall()
    return devig([row_to_market(r) for r in rows])


def report(conn: sqlite3.Connection, account: str) -> Scorecard:
    orders = conn.execute(
        """SELECT o.*, m.event_ticker FROM orders o JOIN markets m ON m.ticker = o.ticker
           WHERE o.account = ? AND o.filled_contracts > 0 AND o.payout IS NOT NULL
           ORDER BY o.id""", (account,)).fetchall()
    pnl = [o["payout"] - o["cost"] - o["fee"] for o in orders]
    diffs = []
    for o in orders:
        if o["snapshot_id"] is None:
            continue
        cand = conn.execute("SELECT pmf_json FROM candidates WHERE bot=? AND ticker=? "
                            "AND snapshot_id=? LIMIT 1",
                            (account, o["ticker"], o["snapshot_id"])).fetchone()
        winner = _winner(conn, o["event_ticker"])
        market = _market_probs(conn, o["snapshot_id"], o["event_ticker"])
        if cand is None or winner is None or not market:
            continue
        bot = json.loads(cand["pmf_json"])
        diffs.append(multiclass_brier(bot, winner) - multiclass_brier(market, winner))
    pci = bootstrap_mean_ci(pnl) if len(pnl) >= 2 else None
    bci = bootstrap_mean_ci(diffs) if len(diffs) >= 2 else None
    return Scorecard(
        account=account, cash=cash(conn, account), settled_bets=len(pnl),
        profit=round(sum(pnl), 4), profit_ci=pci,
        brier_diff=round(float(np.mean(diffs)), 6) if diffs else None, brier_ci=bci,
        profit_pass=pci is not None and pci[0] > 0,
        score_pass=bci is not None and bci[1] < 0)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_scoreboard.py -v`
Expected: 4 passed

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 10: The cycle — discovery, hourly decisions, fills, and the safety tests

**Files:**
- Create: `src/sandbox/cycle.py`
- Test: `tests/test_sandbox_cycle.py`, `tests/test_sandbox_no_real_money.py`

**Interfaces:**
- Consumes: everything above. `list_temperature_series`, `open_markets`, `fetch_book_payload` (Task 1); `parse_series_payload` (Task 2); `record_snapshot`, `latest_snapshot_id`, `load_snapshot`, `record_candidates` (Task 2); `get_meta`, `set_meta` (Task 2); `parse_book`, `walk` (Task 3); `open_account`, `holds`, `place` (Task 4); `settle_due` (Task 8); any object with `.name` and `.decide(snapshot) -> Decision`.
- Produces: `ACCOUNTS: tuple[tuple[str, str, str], ...]`; `STALE_AFTER_S = 600`; `utc_now_iso() -> str`; `ensure_accounts(conn) -> None`; `discover(get) -> list[Market]`; `execute_intent(conn, get, intent, now_iso) -> str`; `run_bot(conn, get, bot, snapshot, now_iso) -> tuple[int, int]`; `decide_all(conn, get, bots, now_iso) -> dict`; `run_cycle(conn, get, bots, now_iso, *, force_decide=False) -> dict`.

- [ ] **Step 1: Write the failing cycle test**

Create `tests/test_sandbox_cycle.py`:

```python
from sandbox_helpers import book_payload, ny_event_payload

from sandbox.bots import Decision
from sandbox.cycle import decide_all, ensure_accounts, run_cycle
from sandbox.db import connect
from sandbox.ledger import cash
from sandbox.models import Intent
from sandbox.recorder import record_snapshot
from sandbox_helpers import three_buckets

NOW = "2026-10-04T10:00:00+00:00"
SAME_HOUR = "2026-10-04T10:15:00+00:00"
NEXT_HOUR = "2026-10-04T11:00:00+00:00"
T69 = "KXHIGHNY-26OCT04-T69"


def fake_get(path, params=None):
    if path == "/series":
        return {"series": [{"ticker": "KXHIGHNY"}, {"ticker": "KXRAINNYC"}]}
    if path == "/markets" and params["status"] == "open":
        return ny_event_payload() if params["series_ticker"] == "KXHIGHNY" else {"markets": []}
    if path == "/markets" and params["status"] == "settled":
        return {"markets": []}
    if path.endswith("/orderbook"):
        return book_payload(no=[(0.80, 5)])          # YES ask 0.20 x5
    raise AssertionError(path)


class SpyBot:
    name = "favorite"

    def __init__(self):
        self.seen = []

    def decide(self, snapshot):
        self.seen.append(snapshot.taken_at)
        return Decision([Intent("favorite", T69, "yes", 1, 0.20, snapshot.id, 0.3, 0.3)], [])


def test_cycle_records_a_snapshot_and_bets_on_the_hour():
    conn = connect(":memory:")
    ensure_accounts(conn)
    bot = SpyBot()
    out = run_cycle(conn, fake_get, [bot], NOW)
    assert out["status"] == "ok" and out["markets"] == 3 and out["orders"] == 1
    assert bot.seen == [NOW]                                   # never newer than now
    order = conn.execute("SELECT status, cost, fee FROM orders").fetchone()
    assert tuple(order) == ("filled", 0.2, 0.02)
    assert cash(conn, "favorite") == 4.78


def test_decisions_run_once_per_utc_hour():
    conn = connect(":memory:")
    ensure_accounts(conn)
    bot = SpyBot()
    run_cycle(conn, fake_get, [bot], NOW)
    run_cycle(conn, fake_get, [bot], SAME_HOUR)
    assert bot.seen == [NOW]
    run_cycle(conn, fake_get, [bot], NEXT_HOUR)
    assert bot.seen == [NOW, NEXT_HOUR]
    # second hour: already holds T69 -> no second order
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_failed_discovery_records_nothing():
    conn = connect(":memory:")
    ensure_accounts(conn)

    def broken(path, params=None):
        raise RuntimeError("kalshi down")
    out = run_cycle(conn, broken, [SpyBot()], NOW)
    assert out["status"] == "fetch_failed"
    assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0


def test_stale_snapshot_makes_bots_sit_out():
    conn = connect(":memory:")
    ensure_accounts(conn)
    record_snapshot(conn, three_buckets(), "2026-10-04T09:30:00+00:00")
    bot = SpyBot()
    out = decide_all(conn, fake_get, [bot], NOW)
    assert out["status"] == "stale_snapshot" and bot.seen == []


def test_already_held_and_no_fill_reasons_reach_the_candidate_log():
    from sandbox.bots import ForecastBot, normal_bucket_probs

    class Fixed(ForecastBot):
        name = "mos"

        def pmf_for_event(self, markets, snapshot):
            return normal_bucket_probs(markets, 69.5, 3.0)

    conn = connect(":memory:")
    ensure_accounts(conn)
    run_cycle(conn, fake_get, [Fixed()], NOW)
    run_cycle(conn, fake_get, [Fixed()], NEXT_HOUR)
    reasons = [r["reason"] for r in conn.execute(
        "SELECT reason FROM candidates WHERE ticker=? ORDER BY id", (T69,))]
    assert reasons == ["edge", "already_held"]
```

- [ ] **Step 2: Write the failing no-real-money test**

Create `tests/test_sandbox_no_real_money.py`:

```python
import subprocess
import sys

CODE = """
import importlib, pkgutil, sys
import sandbox
for m in pkgutil.iter_modules(sandbox.__path__):
    importlib.import_module("sandbox." + m.name)
print("highcast.kalshi_client" in sys.modules)
"""


def test_sandbox_never_imports_the_real_money_client():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stderr
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_sandbox_cycle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.cycle'`

- [ ] **Step 4: Implement `src/sandbox/cycle.py`**

```python
"""One scheduled run: discover -> snapshot; once per UTC hour also settle and decide."""
import json
import logging
import sqlite3
from datetime import datetime, timezone

from sandbox.book import parse_book, walk
from sandbox.db import get_meta, set_meta
from sandbox.kalshi_public import fetch_book_payload, list_temperature_series, open_markets
from sandbox.ledger import holds, open_account, place
from sandbox.markets import parse_series_payload
from sandbox.models import Intent, Market, Snapshot
from sandbox.recorder import (latest_snapshot_id, load_snapshot, record_candidates,
                              record_snapshot)
from sandbox.settle import settle_due

log = logging.getLogger(__name__)

STALE_AFTER_S = 600
ACCOUNTS: tuple[tuple[str, str, str], ...] = (
    ("favorite", "bot", "control: follow the crowd after fees"),
    ("mos", "bot", "uncalibrated: MOS with a fixed 3.0°F spread"),
    ("highcast", "bot", "KLAX/KMIA only"),
    ("manual", "manual", "Julian's hand bets"),
)
_FILLED = ("filled", "partial")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_accounts(conn: sqlite3.Connection) -> None:
    for name, kind, label in ACCOUNTS:
        open_account(conn, name, kind, label)


def discover(get) -> list[Market]:
    markets: list[Market] = []
    for s in list_temperature_series(get):
        markets.extend(parse_series_payload(s["ticker"], open_markets(get, s["ticker"])))
    return markets


def execute_intent(conn: sqlite3.Connection, get, intent: Intent, now_iso: str) -> str:
    try:
        payload = fetch_book_payload(get, intent.ticker)
    except Exception as e:  # noqa: BLE001 - an unreachable book is a no-fill
        log.warning("book fetch failed for %s: %r", intent.ticker, e)
        payload = {}
    fill = walk(parse_book(payload), intent.side, intent.contracts, intent.price_cap)
    book_json = json.dumps(payload.get("orderbook_fp")) if payload else None
    _, status = place(conn, intent, fill, now_iso, book_json)
    return status


def run_bot(conn: sqlite3.Connection, get, bot, snapshot: Snapshot, now_iso: str) -> tuple[int, int]:
    decision = bot.decide(snapshot)
    by_ticker = {c.ticker: c for c in decision.candidates}
    placed = 0
    for intent in decision.intents:
        cand = by_ticker.get(intent.ticker)
        if holds(conn, intent.account, intent.ticker):
            if cand:
                cand.placed, cand.reason = False, "already_held"
            continue
        status = execute_intent(conn, get, intent, now_iso)
        if status in _FILLED:
            placed += 1
        elif cand:
            cand.placed = False
            cand.reason = "insufficient_cash" if status == "insufficient_cash" else "no_fill"
    record_candidates(conn, decision.candidates)
    return placed, len(decision.candidates)


def _age_s(taken_at: str, now_iso: str) -> float:
    return (datetime.fromisoformat(now_iso) - datetime.fromisoformat(taken_at)).total_seconds()


def decide_all(conn: sqlite3.Connection, get, bots, now_iso: str) -> dict:
    sid = latest_snapshot_id(conn, now_iso)
    if sid is None:
        return {"status": "stale_snapshot", "orders": 0, "candidates": 0}
    snapshot = load_snapshot(conn, sid)
    if _age_s(snapshot.taken_at, now_iso) > STALE_AFTER_S:
        return {"status": "stale_snapshot", "orders": 0, "candidates": 0}
    orders = candidates = 0
    for bot in bots:
        o, c = run_bot(conn, get, bot, snapshot, now_iso)
        orders += o
        candidates += c
    return {"status": "ok", "orders": orders, "candidates": candidates}


def run_cycle(conn: sqlite3.Connection, get, bots, now_iso: str, *, force_decide: bool = False) -> dict:
    try:
        markets = discover(get)
    except Exception as e:  # noqa: BLE001 - skip the whole cycle
        log.warning("discovery failed: %r", e)
        return {"status": "fetch_failed", "error": repr(e)}
    sid = record_snapshot(conn, markets, now_iso)
    summary = {"status": "ok", "snapshot_id": sid, "markets": len(markets),
               "settled": 0, "orders": 0, "candidates": 0, "decided": False}
    hour = now_iso[:13]
    if force_decide or get_meta(conn, "last_decide_hour") != hour:
        summary["settled"] = settle_due(conn, get, now_iso)
        result = decide_all(conn, get, bots, now_iso)
        summary.update(status=result["status"], orders=result["orders"],
                       candidates=result["candidates"], decided=True)
        set_meta(conn, "last_decide_hour", hour)
    return summary
```

- [ ] **Step 5: Run both tests**

Run: `.venv/bin/pytest tests/test_sandbox_cycle.py tests/test_sandbox_no_real_money.py -v`
Expected: 6 passed

- [ ] **Step 6: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 11: Dashboard

**Files:**
- Create: `src/sandbox/web.py`, `src/sandbox/templates/base.html`, `markets.html`, `bet.html`, `scoreboard.html`, `log.html`
- Test: `tests/test_sandbox_web.py`

**Interfaces:**
- Consumes: `connect` (Task 2); `latest_snapshot_id`, `load_snapshot` (Task 2); `fetch_book_payload` (Task 1); `parse_book`, `walk` (Task 3); `devig` (Task 4); `report` (Task 9); `ACCOUNTS`, `ensure_accounts`, `execute_intent`, `utc_now_iso` (Task 10); `PacedGet` (Task 1).
- Produces: `create_app(db_path: str, get=None) -> FastAPI` with routes `GET /`, `GET /bet`, `GET /api/preview`, `POST /api/bet` (JSON body `{ticker, side, contracts, price_cap}`), `GET /scoreboard`, `GET /log`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_web.py`:

```python
from fastapi.testclient import TestClient
from sandbox_helpers import book_payload, three_buckets

from sandbox.cycle import ensure_accounts, utc_now_iso
from sandbox.db import connect
from sandbox.recorder import record_snapshot
from sandbox.web import create_app

T69 = "KXHIGHNY-26OCT04-T69"


def client(tmp_path):
    db = str(tmp_path / "sandbox.db")
    conn = connect(db)
    ensure_accounts(conn)
    record_snapshot(conn, three_buckets(), utc_now_iso())
    conn.close()
    get = lambda path, params=None: book_payload(no=[(0.80, 5)])
    return TestClient(create_app(db, get=get)), db


def test_markets_page_is_unbranded_and_lists_buckets(tmp_path):
    c, _ = client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "Weather Desk Sandbox" in r.text and T69 in r.text
    assert "Kalshi" not in r.text


def test_preview_shows_the_fill_without_writing(tmp_path):
    c, db = client(tmp_path)
    r = c.get("/api/preview", params={"ticker": T69, "side": "yes", "contracts": 2, "price_cap": 0.25})
    assert r.json() == {"status": "filled", "contracts": 2, "cost": 0.4, "fee": 0.03,
                        "levels": [[0.2, 2.0]]}
    assert connect(db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_manual_bet_goes_to_the_manual_account(tmp_path):
    c, db = client(tmp_path)
    r = c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    assert r.json()["status"] == "filled"
    row = connect(db).execute("SELECT account, filled_contracts FROM orders").fetchone()
    assert tuple(row) == ("manual", 1)


def test_bad_bet_is_rejected(tmp_path):
    c, _ = client(tmp_path)
    assert c.post("/api/bet", json={"ticker": T69, "side": "maybe", "contracts": 1,
                                    "price_cap": 0.25}).status_code == 400
    assert c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 0,
                                    "price_cap": 0.25}).status_code == 400


def test_scoreboard_and_log_pages(tmp_path):
    c, _ = client(tmp_path)
    c.post("/api/bet", json={"ticker": T69, "side": "yes", "contracts": 1, "price_cap": 0.25})
    s = c.get("/scoreboard")
    assert s.status_code == 200 and "LOCKED" in s.text and "uncalibrated" in s.text
    lg = c.get("/log")
    assert lg.status_code == 200 and T69 in lg.text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.web'`

- [ ] **Step 3: Implement `src/sandbox/web.py`**

```python
"""Local dashboard. 127.0.0.1 only. Never places real orders."""
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sandbox.book import parse_book, walk
from sandbox.cycle import ACCOUNTS, ensure_accounts, execute_intent, utc_now_iso
from sandbox.db import connect
from sandbox.kalshi_public import PacedGet, fetch_book_payload
from sandbox.models import Intent
from sandbox.pricing import devig
from sandbox.recorder import latest_snapshot_id, load_snapshot
from sandbox.scoreboard import N_REQUIRED, report

TEMPLATES = Path(__file__).parent / "templates"
MANILA = ZoneInfo("Asia/Manila")


class BetIn(BaseModel):
    ticker: str
    side: str
    contracts: int
    price_cap: float


def _manila(ts: str | None) -> str:
    if not ts:
        return ""
    return datetime.fromisoformat(ts).astimezone(MANILA).strftime("%b %d %H:%M MNL")


def _validate(bet: BetIn) -> None:
    if bet.side not in ("yes", "no"):
        raise HTTPException(400, "side must be yes or no")
    if bet.contracts < 1 or bet.contracts > 100:
        raise HTTPException(400, "contracts must be 1..100")
    if not 0.0 < bet.price_cap < 1.0:
        raise HTTPException(400, "price_cap must be between 0 and 1")


def create_app(db_path: str, get=None) -> FastAPI:
    get = get or PacedGet()
    app = FastAPI(title="Weather Desk Sandbox")
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["manila"] = _manila
    labels = {name: label for name, _, label in ACCOUNTS}

    def _snapshot(conn):
        sid = latest_snapshot_id(conn, utc_now_iso())
        return load_snapshot(conn, sid) if sid is not None else None

    @app.get("/", response_class=HTMLResponse)
    def markets(request: Request):
        with closing(connect(db_path)) as conn:
            snap = _snapshot(conn)
            events = []
            if snap:
                for ev, ms in snap.events().items():
                    probs = devig(ms)
                    events.append({"event": ev, "series": ms[0].series, "station": ms[0].station,
                                   "date": ms[0].event_date,
                                   "rows": [(m, probs.get(m.ticker)) for m in ms]})
        return templates.TemplateResponse(request, "markets.html",
                                          {"snap": snap, "events": events})

    @app.get("/bet", response_class=HTMLResponse)
    def bet_page(request: Request):
        with closing(connect(db_path)) as conn:
            snap = _snapshot(conn)
        return templates.TemplateResponse(request, "bet.html",
                                          {"markets": snap.markets if snap else ()})

    @app.get("/api/preview")
    def preview(ticker: str, side: str, contracts: int, price_cap: float):
        bet = BetIn(ticker=ticker, side=side, contracts=contracts, price_cap=price_cap)
        _validate(bet)
        fill = walk(parse_book(fetch_book_payload(get, ticker)), side, contracts, price_cap)
        return {"status": fill.status, "contracts": fill.contracts, "cost": fill.cost,
                "fee": fill.fee, "levels": [list(lv) for lv in fill.levels]}

    @app.post("/api/bet")
    def place_bet(bet: BetIn):
        _validate(bet)
        with closing(connect(db_path)) as conn:
            ensure_accounts(conn)
            snap = _snapshot(conn)
            p_market = None
            if snap:
                ev = next((ms for ms in snap.events().values()
                           if any(m.ticker == bet.ticker for m in ms)), None)
                p_market = devig(ev).get(bet.ticker) if ev else None
            intent = Intent(account="manual", ticker=bet.ticker, side=bet.side,
                            contracts=bet.contracts, price_cap=bet.price_cap,
                            snapshot_id=snap.id if snap else None, p_bot=None,
                            p_market=p_market)
            status = execute_intent(conn, get, intent, utc_now_iso())
        return {"status": status}

    @app.get("/scoreboard", response_class=HTMLResponse)
    def scoreboard(request: Request):
        with closing(connect(db_path)) as conn:
            ensure_accounts(conn)
            cards = [(report(conn, name), labels[name]) for name, _, _ in ACCOUNTS]
        return templates.TemplateResponse(request, "scoreboard.html",
                                          {"cards": cards, "n_required": N_REQUIRED})

    @app.get("/log", response_class=HTMLResponse)
    def order_log(request: Request):
        with closing(connect(db_path)) as conn:
            orders = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 500").fetchall()
        return templates.TemplateResponse(request, "log.html", {"orders": orders})

    return app
```

- [ ] **Step 4: Create the templates**

`src/sandbox/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weather Desk Sandbox</title>
<style>
  :root { --bg: #f6f5f1; --fg: #1d1d1b; --muted: #6b6a65; --line: #d9d7cf; --good: #1f7a4d; --bad: #a3361f; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecebe6; --muted: #9a998f; --line: #34332f; } }
  body { background: var(--bg); color: var(--fg); font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; margin: 0 16px 48px; }
  nav { display: flex; gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
  nav strong { margin-right: auto; }
  a { color: inherit; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 20px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  .muted { color: var(--muted); }
  .good { color: var(--good); } .bad { color: var(--bad); }
  .banner { padding: 8px 12px; border: 1px solid var(--line); border-radius: 6px; display: inline-block; }
</style>
</head>
<body>
<nav><strong>Weather Desk Sandbox</strong>
  <a href="/">Markets</a><a href="/bet">Bet</a><a href="/scoreboard">Scoreboard</a><a href="/log">Log</a></nav>
<p class="muted">Paper trading only. Nothing here places a real order.</p>
{% block body %}{% endblock %}
</body>
</html>
```

`src/sandbox/templates/markets.html`:

```html
{% extends "base.html" %}
{% block body %}
{% if not snap %}<p>No snapshot yet. Run <code>sandbox cycle</code>.</p>{% else %}
<p class="muted">Snapshot {{ snap.taken_at | manila }} · {{ events | length }} events</p>
{% for e in events %}
<h3>{{ e.series }} · {{ e.station }} · {{ e.date }}</h3>
<table>
  <tr><th>Bucket</th><th>Ticker</th><th>Bid</th><th>Ask</th><th>Size at ask</th><th>Market p</th></tr>
  {% for m, p in e.rows %}
  <tr><td>{{ m.label }}</td><td class="muted">{{ m.ticker }}</td>
      <td>{{ "%.2f"|format(m.yes_bid) if m.yes_bid is not none else "—" }}</td>
      <td>{{ "%.2f"|format(m.yes_ask) if m.yes_ask is not none else "—" }}</td>
      <td>{{ "%.0f"|format(m.yes_ask_size) if m.yes_ask_size is not none else "—" }}</td>
      <td>{{ "%.1f%%"|format(p * 100) if p is not none else "—" }}</td></tr>
  {% endfor %}
</table>
{% endfor %}
{% endif %}
{% endblock %}
```

`src/sandbox/templates/bet.html`:

```html
{% extends "base.html" %}
{% block body %}
<h3>Manual bet (account: manual)</h3>
<form id="f" onsubmit="return false">
  <label>Market <select name="ticker">
    {% for m in markets %}<option value="{{ m.ticker }}">{{ m.ticker }} ({{ m.label }})</option>{% endfor %}
  </select></label>
  <label>Side <select name="side"><option>yes</option><option>no</option></select></label>
  <label>Contracts <input name="contracts" type="number" min="1" max="100" value="1"></label>
  <label>Max price <input name="price_cap" type="number" step="0.01" min="0.01" max="0.99" value="0.50"></label>
  <button id="preview">Preview fill</button>
  <button id="confirm" disabled>Confirm paper bet</button>
</form>
<pre id="out" class="muted"></pre>
<script>
const f = document.getElementById('f'), out = document.getElementById('out'), confirmBtn = document.getElementById('confirm');
const body = () => ({ticker: f.ticker.value, side: f.side.value,
                     contracts: Number(f.contracts.value), price_cap: Number(f.price_cap.value)});
document.getElementById('preview').onclick = async () => {
  const q = new URLSearchParams(body());
  const r = await fetch('/api/preview?' + q); const j = await r.json();
  out.textContent = JSON.stringify(j, null, 2);
  confirmBtn.disabled = !(r.ok && j.contracts > 0);
};
confirmBtn.onclick = async () => {
  const r = await fetch('/api/bet', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                     body: JSON.stringify(body())});
  out.textContent = JSON.stringify(await r.json(), null, 2); confirmBtn.disabled = true;
};
</script>
{% endblock %}
```

`src/sandbox/templates/scoreboard.html`:

```html
{% extends "base.html" %}
{% block body %}
{% for sc, label in cards %}
<h3>{{ sc.account }} <span class="muted">· {{ label }}</span></h3>
<p class="banner {{ 'good' if sc.unlocked else 'bad' }}">
  {{ "UNLOCKED" if sc.unlocked else "LOCKED" }}: {{ sc.settled_bets }}/{{ n_required }} settled ·
  profit test {{ "✓" if sc.profit_pass else "✗" }} · score test {{ "✓" if sc.score_pass else "✗" }}</p>
<table>
  <tr><th>Cash</th><th>Profit after fees</th><th>Profit CI (per bet)</th><th>Brier vs market</th><th>Brier CI</th></tr>
  <tr><td>${{ "%.2f"|format(sc.cash) }}</td><td>${{ "%.2f"|format(sc.profit) }}</td>
      <td>{{ "[%.3f, %.3f]"|format(*sc.profit_ci) if sc.profit_ci else "—" }}</td>
      <td>{{ "%+.4f"|format(sc.brier_diff) if sc.brier_diff is not none else "—" }}</td>
      <td>{{ "[%.4f, %.4f]"|format(*sc.brier_ci) if sc.brier_ci else "—" }}</td></tr>
</table>
{% endfor %}
{% endblock %}
```

`src/sandbox/templates/log.html`:

```html
{% extends "base.html" %}
{% block body %}
<table>
  <tr><th>#</th><th>When</th><th>Account</th><th>Ticker</th><th>Side</th><th>Status</th>
      <th>Filled</th><th>Cost</th><th>Fee</th><th>Levels</th><th>Payout</th></tr>
  {% for o in orders %}
  <tr><td>{{ o.id }}</td><td>{{ o.decided_at | manila }}</td><td>{{ o.account }}</td>
      <td>{{ o.ticker }}</td><td>{{ o.side }}</td><td>{{ o.status }}</td>
      <td>{{ o.filled_contracts }}/{{ o.contracts_requested }}</td>
      <td>{{ "%.2f"|format(o.cost) }}</td><td>{{ "%.2f"|format(o.fee) }}</td>
      <td class="muted">{{ o.levels_json }}</td>
      <td>{{ "%.2f"|format(o.payout) if o.payout is not none else "open" }}</td></tr>
  {% endfor %}
</table>
{% endblock %}
```

- [ ] **Step 5: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_web.py -v`
Expected: 5 passed

Note the expected preview fee: 2 contracts at 0.20 = 0.07 × 2 × 0.2 × 0.8 = 0.0224 → $0.03.

- [ ] **Step 6: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

---

### Task 12: CLI, launchd jobs, and live acceptance

**Files:**
- Create: `src/sandbox/cli.py`, `launchd/com.julian.sandbox.cycle.plist`, `launchd/com.julian.sandbox.serve.plist`
- Test: `tests/test_sandbox_cli.py`
- Modify: `README.md` (add a "Sandbox" section)

**Interfaces:**
- Consumes: `connect` (Task 2); `ensure_accounts`, `run_cycle`, `utc_now_iso`, `ACCOUNTS` (Task 10); `FavoriteBot`, `IemMos`, `MosBot`, `HighcastBot` (Tasks 5–7); `report` (Task 9); `create_app` (Task 11); `PacedGet` (Task 1).
- Produces: `main(argv: list[str] | None = None) -> int`; console script `sandbox`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_cli.py`:

```python
from sandbox.cli import main
from sandbox.db import connect


def test_init_creates_all_four_accounts(tmp_path, capsys):
    db = tmp_path / "s.db"
    assert main(["--db", str(db), "init"]) == 0
    names = [r["name"] for r in connect(db).execute("SELECT name FROM accounts ORDER BY name")]
    assert names == ["favorite", "highcast", "manual", "mos"]


def test_report_prints_one_line_per_account(tmp_path, capsys):
    db = tmp_path / "s.db"
    main(["--db", str(db), "init"])
    capsys.readouterr()
    assert main(["--db", str(db), "report"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 and all("LOCKED" in ln for ln in lines)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_sandbox_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sandbox.cli'`

- [ ] **Step 3: Implement `src/sandbox/cli.py`**

```python
"""sandbox init | cycle | serve | report"""
import argparse
import json
import logging
from contextlib import closing

from sandbox.bots import FavoriteBot, HighcastBot, IemMos, MosBot
from sandbox.cycle import ACCOUNTS, ensure_accounts, run_cycle, utc_now_iso
from sandbox.db import connect
from sandbox.kalshi_public import PacedGet
from sandbox.scoreboard import N_REQUIRED, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sandbox")
    p.add_argument("--db", default="data/sandbox.db")
    p.add_argument("--highcast-db", default="data/highcast.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    c = sub.add_parser("cycle")
    c.add_argument("--force-decide", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8713)
    sub.add_parser("report")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "serve":
        import uvicorn
        from sandbox.web import create_app
        uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port)
        return 0

    with closing(connect(args.db)) as conn:
        ensure_accounts(conn)
        if args.cmd == "init":
            print(f"sandbox ready at {args.db}")
        elif args.cmd == "cycle":
            bots = [FavoriteBot(), MosBot(IemMos()), HighcastBot(args.highcast_db)]
            summary = run_cycle(conn, PacedGet(), bots, utc_now_iso(),
                                force_decide=args.force_decide)
            print(json.dumps(summary))
        elif args.cmd == "report":
            for name, _, _ in ACCOUNTS:
                sc = report(conn, name)
                verdict = "UNLOCKED" if sc.unlocked else "LOCKED"
                print(f"{name:9} {verdict} {sc.settled_bets}/{N_REQUIRED} "
                      f"cash ${sc.cash:.2f} profit ${sc.profit:+.2f} "
                      f"profit_test={'pass' if sc.profit_pass else 'fail'} "
                      f"score_test={'pass' if sc.score_pass else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_sandbox_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Create the launchd plists**

`launchd/com.julian.sandbox.cycle.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.julian.sandbox.cycle</string>
  <key>ProgramArguments</key><array>
    <string>__PROJECT__/.venv/bin/sandbox</string>
    <string>cycle</string>
  </array>
  <key>WorkingDirectory</key><string>__PROJECT__</string>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>__PROJECT__/data/sandbox-cycle.log</string>
  <key>StandardErrorPath</key><string>__PROJECT__/data/sandbox-cycle.err</string>
</dict></plist>
```

`launchd/com.julian.sandbox.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.julian.sandbox.serve</string>
  <key>ProgramArguments</key><array>
    <string>__PROJECT__/.venv/bin/sandbox</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key><string>__PROJECT__</string>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>__PROJECT__/data/sandbox-serve.log</string>
  <key>StandardErrorPath</key><string>__PROJECT__/data/sandbox-serve.err</string>
</dict></plist>
```

- [ ] **Step 6: Add the README section**

Append to `README.md`:

````markdown
## Sandbox (paper trading on every Kalshi temperature market)

Spec: `docs/superpowers/specs/2026-10-04-weather-sandbox-design.md`. Paper only —
it never imports the real-money client and never writes to `data/highcast.db`.

```
sandbox init                 # creates data/sandbox.db and four $5 accounts
sandbox cycle --force-decide # one snapshot + settle + bot decisions now
sandbox serve                # dashboard at http://127.0.0.1:8713
sandbox report               # per-account LOCKED/UNLOCKED verdict
```

Schedule (every 15 minutes; bots decide once per UTC hour):

```
sed "s|__PROJECT__|$PWD|g" launchd/com.julian.sandbox.cycle.plist > ~/Library/LaunchAgents/com.julian.sandbox.cycle.plist
sed "s|__PROJECT__|$PWD|g" launchd/com.julian.sandbox.serve.plist > ~/Library/LaunchAgents/com.julian.sandbox.serve.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.julian.sandbox.cycle.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.julian.sandbox.serve.plist
```

Real money stays locked for any model until it has ≥100 settled sandbox bets,
profit after fees with a 95% CI above zero, AND a Brier score better than the
market's own prices with a 95% CI below zero.
````

- [ ] **Step 7: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 8: First live run (needs network)**

Run: `.venv/bin/sandbox init && .venv/bin/sandbox cycle --force-decide`
Expected: one JSON line like `{"status": "ok", "snapshot_id": 1, "markets": 576, "settled": 0, "orders": N, "candidates": M, "decided": true}` with `markets` roughly 500–600 and `candidates` > 0 (from `mos`).

Run: `sqlite3 data/sandbox.db "SELECT bot, COUNT(*), SUM(placed) FROM candidates GROUP BY bot; SELECT account, status, COUNT(*) FROM orders GROUP BY account, status;"`
Expected: rows for `mos` (and `highcast` if highcast has a prediction for today), orders for `favorite` and `mos`.

- [ ] **Step 9: Install the schedule and check the dashboard**

Run the four commands from the README section, then open `http://127.0.0.1:8713`.
Expected: the Markets page lists every open series; Scoreboard shows four LOCKED banners; `launchctl list | grep sandbox` shows both jobs.

- [ ] **Step 10: Acceptance after 3 days (spec §7)**

1. Pick 10 orders from `/log` that have a payout. For each, open its `book_json` and confirm the recorded levels are the cheapest asks under the cap, and compare the market's result with Kalshi's settled market page.
2. Confirm no order was placed on a stale snapshot: `sqlite3 data/sandbox.db "SELECT COUNT(*) FROM orders o JOIN snapshots s ON s.id=o.snapshot_id WHERE (julianday(o.decided_at)-julianday(s.taken_at))*86400 > 600;"` → `0`.
3. Measure growth: `ls -lh data/sandbox.db` and record the size per day in `HANDOFF.md`.
4. Record the first `sandbox report` output in `HANDOFF.md`.

- [ ] **Step 11: Full suite one last time**

Run: `.venv/bin/pytest -q`
Expected: all pass, including every `tests/test_sandbox_*.py`.
