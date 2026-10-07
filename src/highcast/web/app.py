import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from highcast import db, gating, latentmax, paper_account, scoring
from highcast.buckets import (bucket_pair as _bucket_pair, pmf_from_buckets,
                              pmf_mass as _pmf_mass)
from highcast.clock import events, timing_guide
from dataclasses import replace as dc_replace

from highcast.config import AppConfig, load_config, set_bankroll
from highcast.fetchers import kalshi
from highcast.timeutil import climate_day_bounds, f_to_c, round_half_up, utcnow_iso

# In-process cache for live Kalshi market ranges, keyed by target_date (iso).
# The dashboard auto-refreshes every 5 min (see index.html POLL_MS) but we
# don't want that to hammer Kalshi's API, so each date's fetch is reused for
# _KALSHI_CACHE_TTL seconds. Stamped with time.monotonic() so it's immune to
# wall-clock adjustments.
_KALSHI_CACHE: dict[str, tuple[float, list[dict] | None]] = {}
_KALSHI_CACHE_TTL = 120.0

TEMPLATES = Path(__file__).parent / "templates"

# Config paths for stations the dashboard can show BESIDES the one it was
# started with. `highcast serve` (and this server's own `cfg`) still default
# to config/klax.yaml -- KLAX's surface is unchanged by this dict existing.
# Adding a station here is what `?station=KMIA` on /api/bet and /api/timing
# resolves against; see `_cfg_for`. Both configs point at the SAME
# `data/highcast.db` (see config/kmia.yaml's comment), so no second DB
# connection pool is needed -- just a second AppConfig to read station,
# staking and kalshi_series from.
OTHER_STATION_CONFIGS: dict[str, str] = {"KMIA": "config/kmia.yaml"}


def _coverage_summary(conn, cfg: AppConfig, metric: str, target: str,
                      now_iso: str) -> dict:
    """How much of this climate day has an admissible 6-hour group behind it.

    DISPLAY ONLY. The desk had no way to know it was blind on 2026-08-06, when
    KMIA settled a low of 75F at 01:51 local and our best evidence was the
    78.1F METAR two minutes later. `blind_minutes` is elapsed time on hourly
    spot readings alone.

    ~75% is a NORMAL day and not a fault: the windows spanning a climate day's
    start and end straddle the boundary and are inadmissible however well the
    station reports. `missing_groups` is the actionable part — a window that
    WOULD have been usable and never arrived.
    """
    from highcast import latentmax as _lm
    try:
        day = date.fromisoformat(target)
    except (TypeError, ValueError):
        return {"blind_minutes": None, "covered_minutes": None,
                "day_minutes": None, "missing_groups": []}
    cov = _lm.six_hour_coverage(conn, cfg, day, metric, asof_utc=now_iso)
    return {"blind_minutes": sum(g["minutes"] for g in cov["gaps"]),
            "covered_minutes": cov["covered_minutes"],
            "day_minutes": cov["day_minutes"],
            "missing_groups": cov["missing_groups"],
            "gaps": cov["gaps"]}


def _ensemble_summary(conn, cfg: AppConfig, metric: str, target: str) -> dict | None:
    """119-member spread for this market, or None when unfetched.

    Already collected every wake into `forecast_ens` and shown NOWHERE until
    now. On 2026-08-06 KLAX HIGH's members spanned 78.0-92.0F with bimodality
    1.19 -- two competing scenarios -- while KMIA HIGH sat in 84.0-88.4F at
    0.45. That is the difference between a bucket bet and a coin toss, and the
    desk could not see it.

    Returns None rather than omitting the key: a renderer that has to guess
    prints "undefined".
    """
    row = conn.execute(
        """SELECT ens_std, ens_p10, ens_p50, ens_p90, ens_bimodality, n_members
           FROM forecast_ens WHERE station=? AND metric=? AND target_date=?
           ORDER BY run_date DESC, fetched_at DESC LIMIT 1""",
        (cfg.station.icao, metric, target)).fetchone()
    if row is None or row["ens_std"] is None:
        return None
    return {"spread": round(row["ens_std"], 2), "p10": row["ens_p10"],
            "p50": row["ens_p50"], "p90": row["ens_p90"],
            "bimodality": (round(row["ens_bimodality"], 2)
                           if row["ens_bimodality"] is not None else None),
            "n_members": row["n_members"]}


def _latest(conn, icao: str, target: str, kind: str, metric: str = "high"):
    row = conn.execute(
        """SELECT * FROM predictions WHERE station=? AND target_date=? AND kind=?
           AND metric=? ORDER BY issued_at_utc DESC LIMIT 1""",
        (icao, target, kind, metric)).fetchone()
    if not row:
        return None
    return {"issued_at": row["issued_at_utc"], "point": row["point_f"],
            "quantiles": json.loads(row["quantiles"]),
            "buckets": json.loads(row["buckets"])}


def _committed_hour(conn, cfg: AppConfig, metric: str, target: str):
    """Station-local hour a bet was locked at, or None if nothing is committed."""
    from highcast import appbet
    row = appbet.existing(conn, cfg, metric, target)
    return None if row is None else row["placed_local_hour"]


def _committed_bet(conn, cfg: AppConfig, metric: str, target: str) -> dict | None:
    """The app's locked pick for (metric, target), if it has committed one."""
    from highcast import appbet
    row = appbet.existing(conn, cfg, metric, target)
    if row is None:
        return None
    r = dict(row)
    return {"bucket": r["bucket"], "p_model": r["p_model"], "price": r["price"],
            "placed_manila": r["placed_manila"], "reason": r.get("reason"),
            "won": r.get("won"), "actual": r.get("actual")}


def _freshness(conn, icao: str, cfg=None) -> dict:
    """Age of each data source. `syn5` is OPTIONAL — the Synoptic sub-hourly feed
    needs a paid token, and without one the fetcher is a deliberate no-op. Report
    that as "off" rather than letting the UI show a permanent red "never", which
    reads as a broken pipeline instead of an unused option."""
    out = {}
    for src in ("metar", "syn5"):
        row = conn.execute(
            "SELECT MAX(valid_utc) AS m FROM obs WHERE station=? AND source=?",
            (icao, src)).fetchone()
        out[src] = row["m"]
    if cfg is not None and not getattr(cfg, "synoptic_token", ""):
        out["syn5_disabled"] = True
    for tbl in ("forecasts", "cli_reports", "alerts"):
        row = conn.execute(
            f"SELECT MAX(fetched_at) AS m FROM {tbl} WHERE station=?",
            (icao,)).fetchone()
        out[tbl] = row["m"]
    # ⚠️ `alerts` IS EVENT-DRIVEN, SO ITS AGE IS NOT A HEALTH SIGNAL. No active
    # NWS advisory means no rows, which is the normal state — KLAX had none for
    # twelve days in August and the chip read "ALERTS 292h 38m" in warning
    # amber, implying a dead feed when nothing was wrong. Same false-positive
    # the `syn5` branch above already avoids, and `highcast.staleness` exempts
    # this table outright for exactly this reason.
    #
    # What is actually worth showing is whether an advisory is LIVE RIGHT NOW,
    # so report that instead of an age. `active` is the count and `active_events`
    # names them.
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        rows = conn.execute(
            """SELECT event FROM alerts WHERE station=?
                 AND expires_utc > ? AND onset_utc <= ?""",
            (icao, now_iso, now_iso)).fetchall()
        out["alerts_active"] = len(rows)
        out["alerts_active_events"] = sorted({r["event"] for r in rows})
    except Exception:                    # noqa: BLE001 - never break the board
        out["alerts_active"] = None
        out["alerts_active_events"] = []
    return out


def _is_stale(kind: str, issued: datetime, now: datetime, cfg: AppConfig) -> bool:
    """Schedule-aware staleness: stale only when a scheduled regeneration
    should already have replaced this prediction.

    - dayahead regenerates at ~17:00 local (evening guidance for the next day)
      and ~09:00 local (morning refresh). Overnight aging is normal; after
      09:30 local the row must postdate 08:00 local today.
    - intraday regenerates every 5-min wake while the climate day runs; more
      than 30 min old inside the climate day means the auto-cycle is failing.
      Outside the climate day there is nothing to regenerate.
    """
    tz = ZoneInfo(cfg.station.tz)
    local = now.astimezone(tz)
    if kind == "dayahead":
        if (local.hour, local.minute) >= (9, 30):
            eight = datetime(local.year, local.month, local.day, 8, 0, tzinfo=tz)
            return issued < eight.astimezone(timezone.utc)
        return False
    start, end = climate_day_bounds(local.date(), cfg.station.std_utc_offset)
    if start <= now < end:
        return (now - issued).total_seconds() > 30 * 60
    return False


def _add_freshness(pred: dict | None, now: datetime, kind: str,
                   cfg: AppConfig) -> None:
    """Mutate `pred` in place, adding age_minutes/stale computed against `now`."""
    if pred is None:
        return
    issued = datetime.fromisoformat(pred["issued_at"])
    pred["age_minutes"] = int((now - issued).total_seconds() / 60.0)
    pred["stale"] = _is_stale(kind, issued, now, cfg)


def _public_display_f(hi: float) -> int:
    """What a public feed (Kalshi UI, weather sites) would show for `hi`: it
    re-derives °F from a whole-degree °C reading, so precise values near a
    C-degree boundary (e.g. 78.0-78.2°F -> whole C 26 -> back to 79°F) can
    display one degree above the precise round_half_up(hi). Settlement
    always follows the precise sensor max, not this double-rounded value."""
    c_whole = round_half_up(f_to_c(hi))
    return round_half_up(c_whole * 9.0 / 5.0 + 32.0)


def _kalshi_buckets(dayahead: dict | None, intraday: dict | None,
                    running_max: dict | None) -> list[dict]:
    """2-degree (odd/odd+1) buckets spanning the union of the dayahead and
    intraday pmf supports, plus open ≤/≥ tails for any mass outside that
    union. `dead` marks buckets the running max has already ruled out."""
    da_pmf = pmf_from_buckets(dayahead["buckets"]) if dayahead else None
    in_pmf = pmf_from_buckets(intraday["buckets"]) if intraday else None
    keys = set(da_pmf or {}) | set(in_pmf or {})
    if not keys:
        return []

    lo0, _ = _bucket_pair(min(keys))
    pmax_lo, _ = _bucket_pair(max(keys))
    # "dead" = ruled out by the running extremum. For the HIGH the running max
    # rules out buckets below it; for the LOW the ratchet runs the other way, so
    # rather than invert the arithmetic we simply don't mark dead buckets for the
    # low (no false strikeouts).
    is_max = (running_max or {}).get("extremum", "max") == "max"
    dead_floor = math.floor(running_max["lo"]) if (running_max and is_max) else None

    def is_dead(hi: int | None) -> bool:
        return dead_floor is not None and hi is not None and hi < dead_floor

    out = []
    tail_lo = lo0 - 1
    out.append({"label": f"≤{tail_lo}", "lo": None, "hi": tail_lo,
                "p_dayahead": _pmf_mass(da_pmf, None, tail_lo),
                "p_intraday": _pmf_mass(in_pmf, None, tail_lo),
                "dead": is_dead(tail_lo)})
    for lo in range(lo0, pmax_lo + 1, 2):
        hi = lo + 1
        out.append({"label": f"{lo}–{hi}", "lo": lo, "hi": hi,
                    "p_dayahead": _pmf_mass(da_pmf, lo, hi),
                    "p_intraday": _pmf_mass(in_pmf, lo, hi),
                    "dead": is_dead(hi)})
    tail_hi = pmax_lo + 2
    out.append({"label": f"≥{tail_hi}", "lo": tail_hi, "hi": None,
                "p_dayahead": _pmf_mass(da_pmf, tail_hi, None),
                "p_intraday": _pmf_mass(in_pmf, tail_hi, None),
                "dead": is_dead(None)})
    return out


def _cached_kalshi_ranges(cfg: AppConfig, target_date: str,
                          metric: str = "high") -> list[dict] | None:
    """fetch_ranges result for (station, target_date, metric), reused for
    _KALSHI_CACHE_TTL seconds so the dashboard's auto-refresh doesn't hit
    Kalshi's API on every poll.

    STATION IS PART OF THE KEY. Before the KMIA tile this cache only ever saw
    one station, so `metric:date` was unique enough. Serving two stations from
    one process, KLAX and KMIA can both have a "high:2026-07-29" in flight on
    the same day with DIFFERENT Kalshi series and strike ladders -- collapsing
    them onto one cache key would hand one station's ranges (and prices) to
    the other's request.
    """
    from highcast.metric import get_metric
    key = f"{cfg.station.icao}:{metric}:{target_date}"
    now = time.monotonic()
    cached = _KALSHI_CACHE.get(key)
    if cached is not None and (now - cached[0]) < _KALSHI_CACHE_TTL:
        return cached[1]
    ranges = kalshi.fetch_ranges(cfg, target_date, get_metric(cfg, metric))
    _KALSHI_CACHE[key] = (now, ranges)
    return ranges


def _kalshi_market(ranges: list[dict] | None, dayahead: dict | None,
                   intraday: dict | None) -> list[dict] | None:
    """Real Kalshi ranges (from fetch_ranges) with the model's probability
    mass summed per range, and the market's own price beside it. None when
    no live ranges are available (feature disabled, or Kalshi unreachable)
    -- the caller falls back to the synthetic kalshi_buckets in that case."""
    if not ranges:
        return None
    da_pmf = pmf_from_buckets(dayahead["buckets"]) if dayahead else None
    in_pmf = pmf_from_buckets(intraday["buckets"]) if intraday else None
    return [{
        "label": r["label"], "lo": r["lo"], "hi": r["hi"],
        "market_price": kalshi.resolve_price(r),
        # The ASK, kept alongside the resolved price because the disagreement
        # figure normalises asks (overround ~1.09) -- see `_disagreement`.
        "yes_ask": r.get("yes_ask"),
        "p_model_dayahead": _pmf_mass(da_pmf, r["lo"], r["hi"]),
        "p_model_intraday": _pmf_mass(in_pmf, r["lo"], r["hi"]),
    } for r in ranges]


def _ensemble(conn, cfg: AppConfig, metric: str, target: date,
              now: datetime) -> dict | None:
    """The 119-member ensemble's SHAPE for this target — display only.

    ⚠️ LEAK GUARD, and it is not theoretical. `forecast_ens` holds rows whose
    `run_date` is AFTER `target_date` — 234 of 554 on 2026-08-28, leads spanning
    -5 to +2 days. Those are hindcasts. Taking "the latest row for this target"
    would put a run made after the day ended on the desk as though it were a
    forecast, which is error #1's shape. Two filters, both required:
    `run_date <= target_date` AND `fetched_at <= now`.

    ⚠️ IT DOES NOT PREDICT OUTCOMES, MEASURED. Over 96 settled bets with a
    leak-free row, against whether our bet lost:

        ens_std          +0.034  CI [-0.168, +0.233]
        ens_bimodality   -0.038  CI [-0.237, +0.164]
        p90 - p10        +0.032  CI [-0.170, +0.231]

    All three straddle zero. High-spread days lost 29% against 19% on n=48 a
    side, which the correlations say is noise. This is consistent with the
    guidance-spread veto already being dead (measured 2026-08-06: the days it
    would have vetoed hit BETTER). **So it must never become a veto or a
    feature.**

    WHAT IT IS FOR, and this part is arithmetic rather than prediction: when 119
    members span 15 F and a Kalshi rung is 2 F wide, no single-bucket bet can be
    confident whatever the model's probability says. Seeing "8 buckets wide"
    before committing is a fact about the question, not a forecast of the answer.
    """
    row = conn.execute(
        """SELECT n_members, ens_std, ens_p10, ens_p50, ens_p90, ens_bimodality,
                  run_date, fetched_at
           FROM forecast_ens
           WHERE station=? AND metric=? AND target_date=?
             AND run_date <= target_date AND fetched_at <= ?
           ORDER BY fetched_at DESC LIMIT 1""",
        (cfg.station.icao, metric, target.isoformat(), now.isoformat())).fetchone()
    if row is None or row["ens_p10"] is None or row["ens_p90"] is None:
        return None
    span = float(row["ens_p90"]) - float(row["ens_p10"])
    return {"members": row["n_members"], "std": row["ens_std"],
            "p10": row["ens_p10"], "p50": row["ens_p50"], "p90": row["ens_p90"],
            "bimodality": row["ens_bimodality"], "span_f": span,
            # A rung is 2F, so the span in BUCKETS is what the desk actually bets.
            "span_buckets": max(1, round(span / 2.0)),
            "run_date": row["run_date"]}


def _disagreement(conn, cfg: AppConfig, metric: str, market: list[dict] | None,
                  target: date, now: datetime) -> dict | None:
    """Where the market outbids US most, against the model the desk would ACT on.

    The dashboard shows day-ahead and intraday side by side, but only one of
    them is the desk's answer at any hour -- `gating.kind_at_hour` is the single
    rule for which, and every surface must go through it or the page can show a
    disagreement the desk does not actually have.

    ⚠️ DISPLAY ONLY. `marketsensor.disagreement` carries the measurement and,
    more importantly, the three measured reasons this is NOT a gate:
    corr(D, our ask) = -0.798, a D veto is a cheap-contract veto, and the
    trigger question is pre-registered in `scripts/salvage_alert_gate.py`.
    """
    from highcast import marketsensor
    if not market:
        return None
    try:
        gate = gating.gate_state(conn, cfg, target.isoformat(), metric, now)
        local_hour = now.astimezone(ZoneInfo(cfg.station.tz)).hour
        kind = gating.kind_at_hour(cfg.station.icao, metric,
                                   bool(gate.get("within_window")), local_hour)
    except Exception:                              # noqa: BLE001
        kind = "intraday"
    # ⚠️ MIRROR `preferred_prediction`'s FALLBACK, not just its preference.
    # `kind_at_hour` gives the rule; the shipped function then walks an ORDER
    # and returns the first kind that actually HAS a stored prediction. Before
    # the day-ahead run lands, a pre-knee HIGH prefers `dayahead` and every
    # `p_model_dayahead` is None -- so keying on the preference alone made this
    # panel silently blank all morning. That availability fallback is the same
    # mechanism behind the one real cross-kind bet (HANDOFF 3.8b); it has now
    # bitten twice, which is why it is spelled out here rather than assumed.
    order = (kind, "intraday" if kind == "dayahead" else "dayahead")
    for k in order:
        lad = [{"label": r["label"], "p_model": r.get(f"p_model_{k}"),
                "yes_ask": r.get("yes_ask")} for r in market]
        d = marketsensor.disagreement(lad)
        if d:
            d["kind"] = k
            return d
    return None


def _model_pick(cfg: AppConfig, target_date: str, pred: dict | None,
                kind: str, metric: str = "high") -> dict | None:
    """The argmax bucket for `pred` (the resolved latest prediction for
    target_date) -- real Kalshi ranges when fetch_ranges has them (reusing
    the same cached path /api/today uses, so this never triggers a second
    fetch), else the synthetic 2-degree odd-pairs. None if there's no
    prediction to pick from."""
    if pred is None or not pred.get("buckets"):
        return None
    ranges = _cached_kalshi_ranges(cfg, target_date, metric)
    if ranges:
        market = _kalshi_market(ranges, pred, None)
        best = max(market, key=lambda r: r["p_model_dayahead"] if r["p_model_dayahead"] is not None else -1.0)
        return {"label": best["label"], "prob": best["p_model_dayahead"], "source": kind}
    buckets = _kalshi_buckets(pred, None, None)
    best = max(buckets, key=lambda b: b["p_dayahead"] if b["p_dayahead"] is not None else -1.0)
    return {"label": best["label"], "prob": best["p_dayahead"], "source": kind}


def _pmf_pick(pred: dict | None) -> dict | None:
    """Argmax 2-degree (odd/odd+1) bucket for a single prediction's own pmf,
    reusing _bucket_pair/_pmf_mass. Deliberately independent of live Kalshi
    ranges (unlike _model_pick) so capturing field-note context never
    triggers a market fetch."""
    if pred is None or not pred.get("buckets"):
        return None
    pmf = pmf_from_buckets(pred["buckets"])
    keys = sorted(pmf)
    lo0, _ = _bucket_pair(keys[0])
    pmax_lo, _ = _bucket_pair(keys[-1])
    best_label, best_p = None, -1.0
    for lo in range(lo0, pmax_lo + 1, 2):
        hi = lo + 1
        p = _pmf_mass(pmf, lo, hi) or 0.0
        if p > best_p:
            best_label, best_p = f"{lo}–{hi}", p
    return {"point": pred["point"], "pick_label": best_label, "pick_prob": best_p}


def _note_context(conn, cfg: AppConfig, target: date, now: datetime) -> dict:
    """Server-side snapshot of model state for `target`, captured at note-
    write time -- the client is never trusted to supply this."""
    vis = latentmax.visible_running_max(conn, cfg, target)
    running_max = {"lo": round(vis[0], 2), "hi": round(vis[1], 2)} if vis else None
    dayahead = _latest(conn, cfg.station.icao, target.isoformat(), "dayahead")
    intraday = _latest(conn, cfg.station.icao, target.isoformat(), "intraday")
    return {
        "running_max": running_max,
        "dayahead": _pmf_pick(dayahead),
        "intraday": _pmf_pick(intraday),
        "phase": timing_guide(cfg, now)["phase"],
    }


class NoteIn(BaseModel):
    target_date: str
    sky: str


class BankrollIn(BaseModel):
    bankroll: float


class MyBetIn(BaseModel):
    metric: str = "high"
    bucket: str
    contracts: int
    price: float | None = None   # defaults to the current market price


def create_app(cfg: AppConfig, config_path: str = "config/klax.yaml") -> FastAPI:
    app = FastAPI(title="highcast")
    app.state.cfg = cfg  # mutable slot: POST /api/bankroll swaps this live
    env = Environment(loader=FileSystemLoader(TEMPLATES))

    def current() -> AppConfig:
        return app.state.cfg

    # ONE migrating connection, at startup, before anything is served. Every
    # per-request connection below is `migrate=False`.
    #
    # WHY: this used to run `executescript(SCHEMA)` plus eight `ensure_*`
    # migrations on EVERY request, from 16 call sites, on a 60-second poll --
    # a burst of DDL aimed at the same file the 5-minute wake cycle writes.
    # `data/serve.err` carries the result: three `sqlite3.OperationalError:
    # database is locked` tracebacks, all of them raised inside
    # `connect()`'s own schema call while merely trying to READ.
    db.connect(cfg.db_path).close()

    def _reader(c: AppConfig = None):
        """THE ONLY WAY A REQUEST MAY OPEN THE DATABASE. Never migrates.

        ⚠️ Two routes used to call `db.connect(c.db_path)` directly and so ran
        `executescript(SCHEMA)` + 8 migrations on EVERY request — a burst of DDL
        aimed at the file the 5-minute wake cycle writes, from endpoints whose
        own docstrings say "read-only". That is the §2.4 bug, and it survived the
        original fix because the fix only covered `_conn()`.

        Funnelling every request-path connection through one helper is what stops
        it coming back a third time; `test_no_route_opens_a_migrating_connection`
        is the wall. The schema is guaranteed present because `create_app` opens
        one migrating connection at startup, above.
        """
        return db.connect((c or cfg).db_path, migrate=False)

    def _conn():
        return _reader(cfg)

    _other_cfg_cache: dict[str, AppConfig] = {}

    def _cfg_for(station: str | None) -> AppConfig:
        """Resolve the AppConfig a request wants, defaulting to this server's
        own station (KLAX in production) when `station` is omitted -- so every
        existing caller that never passes `?station=` gets byte-identical
        behaviour to before this function existed.

        The server's OWN station always goes through `current()`, never a
        fresh reload -- `current()` is what POST /api/bankroll mutates live,
        and reloading config/klax.yaml from disk here would silently drop that
        edit until the next restart. A DIFFERENT station's yaml is read-only
        from this server's point of view (nothing here can edit KMIA's
        bankroll), so loading it once and caching it is safe.
        """
        c = current()
        if not station or station == c.station.icao:
            return c
        if station not in _other_cfg_cache:
            path = OTHER_STATION_CONFIGS.get(station)
            if path is None:
                raise HTTPException(404, f"unknown station {station!r}")
            _other_cfg_cache[station] = load_config(path)
        return _other_cfg_cache[station]

    # GET /api/demo-account's cache slot -- see that endpoint for why it is
    # unkeyed. Per-app (not module-level) so two servers in one process cannot
    # serve each other's account.
    _demo_cache: tuple[float, dict] | None = None
    _DEMO_TTL = 120.0

    # computed once at startup: scans all history (too slow per request)
    spike_pmf = {"high": latentmax.hidden_spike_pmf(_conn(), cfg, metric="high"),
                 "low": latentmax.hidden_spike_pmf(_conn(), cfg, metric="low")}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return env.get_template("index.html").render(
            station=cfg.station.name, icao=cfg.station.icao)

    @app.get("/api/today")
    def today(metric: str = "high", station: str | None = None):
        """`station` defaults to this server's own (KLAX), so every existing
        caller that omits it is byte-identical to before.

        ⚠️ WHY IT WAS ADDED, 2026-08-15. This endpoint feeds the freshness row,
        including the ALERTS chip. Without a station it always answered for
        KLAX, so a KMIA advisory could never appear there — KMIA had a live Heat
        Advisory on 2026-08-15 that the chip structurally could not show.
        `/api/bet` and `/api/timing` already took `station`; this was the odd one
        out.

        Safe because `_cached_kalshi_ranges` already keys on STATION as well as
        metric and date — collapsing those would hand one station's ladder to
        the other, which is the `metric:date` cross-station bug CLAUDE.md names.
        """
        from highcast.metric import get_metric
        c = _cfg_for(station)
        spec = get_metric(c, metric)
        conn = _reader(c)
        now = datetime.now(timezone.utc)
        tz = ZoneInfo(c.station.tz)
        target = now.astimezone(tz).date()
        icao = c.station.icao
        vis = latentmax.visible_running_extremum(conn, c, target, spec.extremum)
        if vis:
            hi = round(vis[1], 2)
            running_max = {"lo": round(vis[0], 2), "hi": hi,
                           "extremum": spec.extremum,
                           # the °C double-rounding artifact only applies to the high
                           "public_display_f": _public_display_f(hi)
                           if spec.extremum == "max" else None}
        else:
            running_max = None
        dayahead = _latest(conn, icao, target.isoformat(), "dayahead", metric)
        intraday = _latest(conn, icao, target.isoformat(), "intraday", metric)
        _add_freshness(dayahead, now, "dayahead", c)
        _add_freshness(intraday, now, "intraday", c)
        tomorrow_date = target + timedelta(days=1)
        # THE EVENING FORECAST, ASKED FOR BY NAME. Tomorrow's card is the one
        # legitimate consumer of it: at 17:00 local there is nothing else for
        # tomorrow, and a guidance-only look at tomorrow's market is exactly what
        # this card is for. Everywhere else the evening object is deliberately
        # invisible, because it runs 4F cold at KLAX HIGH (n=19, MAE 4.42 vs the
        # morning refresh's 1.45) and was never in any measurement.
        # See db.KIND_EVENING.
        tomorrow_pred = _latest(
            conn, icao, tomorrow_date.isoformat(), db.KIND_EVENING, metric)
        _add_freshness(tomorrow_pred, now, "dayahead", c)
        today_ranges = _cached_kalshi_ranges(c, target.isoformat(), metric)
        tomorrow_ranges = _cached_kalshi_ranges(c, tomorrow_date.isoformat(), metric)
        return {
            "target_date": target.isoformat(),
            "metric": metric,
            "dayahead": dayahead,
            "intraday": intraday,
            "running_max": running_max,
            "freshness": _freshness(conn, icao, c),
            "spike_pmf": spike_pmf.get(metric, {0: 1.0}),
            "kalshi_buckets": _kalshi_buckets(dayahead, intraday, running_max),
            "kalshi_market": _kalshi_market(today_ranges, dayahead, intraday),
            "ensemble": _ensemble(conn, c, metric, target, now),
            "disagreement": _disagreement(
                conn, c, metric, _kalshi_market(today_ranges, dayahead, intraday),
                target, now),
            # no obs constrain a future date: intraday is null and nothing is
            # dead, which _kalshi_buckets(…, None, None) already yields.
            "tomorrow": {
                "target_date": tomorrow_date.isoformat(),
                "prediction": tomorrow_pred,
                "kalshi_buckets": _kalshi_buckets(tomorrow_pred, None, None),
                "kalshi_market": _kalshi_market(tomorrow_ranges, tomorrow_pred, None),
            },
        }

    @app.get("/api/clock")
    def clock(metric: str = "high"):
        tz = ZoneInfo(cfg.station.tz)
        return events(cfg, datetime.now(tz).date(), metric)

    @app.get("/api/timing")
    def timing(metric: str = "high", station: str | None = None):
        """`station` defaults to this server's own station (see `_cfg_for` /
        `/api/bet` above) -- the Miami tile passes `?station=KMIA`."""
        c = _cfg_for(station)
        now = datetime.now(timezone.utc)
        guide = timing_guide(c, now, metric)
        guide["metric"] = metric
        # The best-time panel converts station-local decision hours to the
        # display tz in the browser (so it follows DST rather than a baked-in
        # offset), which needs both zone names.
        guide["station_tz"] = c.station.tz
        guide["display_tz"] = c.display_tz
        conn = _reader(c)
        rec = guide["recommendation"]
        target_date = rec["target_date"]
        gate = gating.gate_state(conn, c, target_date, metric, now)
        # Read the SAME prediction the Decision Desk reads. These two panels
        # used to diverge (timing card showed the nowcast's 81-82 while the desk
        # showed the day-ahead's 79-80); `preferred_prediction` is now the single
        # rule for both. report_out still targets tomorrow, where only a
        # day-ahead exists — the helper falls through to it naturally.
        # `local_hour` matters here too: the desk passes it (via
        # signals.live_decision), so omitting it would reintroduce exactly the
        # divergence this comment block exists to prevent — the desk switching to
        # the nowcast at the HIGH's knee while this card still showed the
        # day-ahead. Only meaningful when the card is on today; tomorrow has no
        # nowcast either way.
        rec_is_today = target_date == now.astimezone(ZoneInfo(c.station.tz)).date().isoformat()
        pred, kind = gating.preferred_prediction(
            conn, c.station.icao, target_date, metric,
            bool(gate.get("within_window")),
            local_hour=(now.astimezone(ZoneInfo(c.station.tz)).hour
                        if rec_is_today else None))
        guide["window_open"] = bool(gate.get("within_window"))
        rec["model_pick"] = _model_pick(c, target_date, pred, kind, metric)
        if rec["model_pick"] is not None:
            rec["model_pick"]["gate"] = {
                "committed": gate["committed"], "confident": gate["confident"],
                "stable": gate["stable"], "within_window": gate["within_window"],
                "n_readings": gate["n_readings"],
                "min_confidence": c.staking.min_confidence}
            rec["model_pick"]["actionable"] = bool(
                gate["committed"] and gate["within_window"])
        return guide

    @app.get("/api/history")
    def history():
        conn = _conn()
        rows = []
        for r in conn.execute(
                """SELECT c.date, c.high_f, p.point_f FROM cli_reports c
                   LEFT JOIN predictions p ON p.station=c.station
                     AND p.target_date=c.date AND p.kind='dayahead'
                     AND p.metric='high'
                     AND p.issued_at_utc = (
                       SELECT MAX(p2.issued_at_utc) FROM predictions p2
                       WHERE p2.station=c.station AND p2.target_date=c.date
                         AND p2.kind='dayahead' AND p2.metric='high')
                   WHERE c.station=? ORDER BY c.date DESC LIMIT 60""",
                (cfg.station.icao,)):
            rows.append({"date": r["date"], "actual": r["high_f"],
                         "predicted": r["point_f"]})
        errs = [abs(r["predicted"] - r["actual"]) for r in rows
                if r["predicted"] is not None]
        return {"days": rows,
                "mae": sum(errs) / len(errs) if errs else None}

    @app.get("/api/day-grade")
    def day_grade_api(station: str | None = None):
        """Input-quality grade for today's two nowcasts.

        ⚠️ The payload carries a `disclaimer` field and any UI MUST render it.
        A letter grade invites "A means it will hit"; it means the opposite kind
        of thing — that the number was computed from complete, current inputs.
        The accuracy estimate is the model's own probability.
        """
        from highcast import daygrade
        c = _cfg_for(station)
        conn = _reader(c)
        return {"grades": [g.as_dict() for g in daygrade.grade_all(conn, c)]}

    @app.get("/api/scorecard")
    def scorecard_api():
        conn = _conn()
        records = scoring.scorecard(conn, cfg)
        return {"records": records, "summary": scoring.summarize(records)}

    @app.get("/api/bet")
    def bet(metric: str = "high", station: str | None = None):
        """The gated decision + sizing for the Decision Desk. Read-only: it does
        NOT record a signal (the wake cycle owns logging), so a page poll never
        pollutes the paper track record.

        `station` is optional and defaults to this server's own station (see
        `_cfg_for`) -- the Miami tile is the only caller that passes it today,
        via `?station=KMIA`."""
        from highcast import signals
        from highcast.metric import get_metric
        c = _cfg_for(station)
        conn = _reader(c)
        now = datetime.now(timezone.utc)
        spec = get_metric(c, metric)
        out = signals.live_decision(conn, c, spec, now)
        d, g = out["decision"], out["gate"]
        return {
            "metric": metric, "target": out["target"], "kind": out["kind"],
            "has_pred": out["pred"] is not None, "gate": g,
            "decision": None if d is None else {
                "place": d.place, "reason": d.reason, "bucket": d.bucket,
                "p_model": d.p_model, "price": d.price, "edge": d.edge,
                "stake": d.sizing.stake if d.sizing else None,
                "stake_frac": d.sizing.stake_frac if d.sizing else None,
                "contracts": d.sizing.contracts if d.sizing else None,
                "capped": d.sizing.capped if d.sizing else None,
                "bankroll_unset": d.sizing.bankroll_unset if d.sizing else None,
                "spread": d.spread},
            "bankroll": c.staking.bankroll, "currency": c.staking.currency,
            "min_confidence": c.staking.min_confidence,
            "min_edge": c.staking.min_edge, "max_spread": c.staking.max_spread,
            # How accurate this metric's nowcast has MEASURED at the hour the
            # displayed prediction was issued, so the desk can never present an
            # early read as though it carried the same weight as a post-knee one.
            # Keyed off the prediction's issued_at (its model's hour), not "now".
            #
            # INTRADAY ONLY. HOUR_ACCURACY is a per-hour curve of the *nowcast*;
            # the day-ahead is a different model with one daily cutoff, and its
            # issued_at hour indexes nothing. Labelling a day-ahead with it put
            # "99% at 18:00" (the nowcast's post-max number) on the evening-issued
            # 73.2F card that was simultaneously vetoed as an untrained regime —
            # a green tick on the single most broken read of the day.
            # When a bet is LOCKED, the honest figure is the accuracy at the hour
            # it was locked — not at the current hour. The HIGH locked at hour 11
            # (52%) but this reported hour 17 (99%), so the gate row read
            # "Hour accuracy 99% at 17:00 ✓" beside an hour-11 pick. That 99% is
            # the transcription number the whole decision-hour work exists to
            # avoid quoting.
            "hour_accuracy": (
                gating.hour_accuracy(c.station.icao, metric,
                                     _committed_hour(conn, c, metric,
                                                     out["target"]))
                or (gating.hour_accuracy(
                        c.station.icao, metric,
                        gating._local_hour(c, out["pred"]["issued_at"]))
                    if out["pred"] and out["kind"] == "intraday" else None)),
            "decision_hour": gating.min_decision_hour(c.station.icao, metric),
            "station_tz": c.station.tz,
            # THE COMMITTED BET IS THE RECOMMENDATION. Once the app has locked a
            # pick for this target date, the card must show THAT for the whole
            # trading window — not the live model's current favourite, which
            # refits every hour and would show a different bucket depending on
            # when you happened to look. A recommendation you cannot come back to
            # is not a recommendation.
            "committed": _committed_bet(conn, c, metric, out["target"])}

    @app.get("/api/cities")
    def cities_api():
        """Predictability of every Kalshi daily temperature market.

        Computed from stored CLI + MOS, so it needs no per-city model. Cheap
        enough to serve per request (two indexed scans per city) and it changes
        only when new days settle."""
        from highcast import cities as _cities
        c = current()
        return {"station": c.station.icao,
                "rows": _cities.survey(_conn(), c.station.icao)}

    @app.get("/api/board")
    def board_api():
        """Station table: the raw physical state each probability is built from.

        Exists because a probability cannot be argued with -- 81% is 81% -- while
        "min stalled 17.9h, pressure falling 1.3 hPa, depression 5.9F" is a
        picture a human can overrule. On 2026-07-29 the KLAX LOW committed 67-68
        at 81% and lost; every number that would have flagged it was already in
        `obs` and shown nowhere.

        DB-only and cheap (one latest-obs read plus the running extremum per
        metric), so it is served per request like `/api/cities` rather than
        cached. Display only: it writes nothing and votes on nothing.
        """
        from highcast import board as _board, latentmax as _lm
        from highcast import preflight as _preflight
        conn = _conn()
        now_iso = datetime.now(timezone.utc).isoformat()

        # EVERY station, not just the one this server was started with. It
        # used to build from `current()` alone, so the panel showed two of the
        # four markets and gave no sign the other two existed -- worse than
        # showing none, because the missing half looks checked.
        cfgs = [current()] + [_cfg_for(i) for i in OTHER_STATION_CONFIGS]
        rows = []
        for c in cfgs:
            # Climatology bands are per station/hour/season, NOT per metric, so
            # the pool is built ONCE PER STATION and shared across its metrics.
            # `obs` is indexed on (station, valid_utc) only and this scan cannot
            # use it -- doing it per metric would double a ~57k-row scan for two
            # identical answers. Measured at ~0.12s per station.
            bands = _board.climatology_bands(conn, c)
            for m in ("low", "high"):
                r = _board.row(conn, c, m)
                r["evidence"] = _board.evidence(r, bands)
                # Display only. `grade` reads a stored column and returns words;
                # it is never consulted by staking, appbet or the commit.
                r["preflight"] = _preflight.grade(conn, c, m, r["target_date"])
                r["coverage"] = _coverage_summary(conn, c, m, r["target_date"],
                                                  now_iso)
                r["ensemble"] = _ensemble_summary(conn, c, m, r["target_date"])
                rows.append(r)
        return {"station": current().station.icao,
                "stations": [c.station.icao for c in cfgs],
                "rows": rows,
                "stall_note_hours": _board.STALL_NOTE_HOURS,
                # Static per-market reliability, served alongside so the panel
                # can answer "which of these do I trust" without a second call.
                "reliability": _board.reliability_table(),
                "climo_window": {"hours": _board.CLIMO_HOUR_WINDOW,
                                 "days": _board.CLIMO_DOY_WINDOW}}

    @app.get("/api/bankroll")
    def get_bankroll():
        s = current().staking
        return {"bankroll": s.bankroll, "currency": s.currency,
                "paper": s.bankroll == 0}

    @app.post("/api/bankroll")
    def post_bankroll(body: BankrollIn):
        try:
            new_val = set_bankroll(config_path, body.bankroll)
        except ValueError as e:
            raise HTTPException(400, str(e))
        c = current()
        app.state.cfg = dc_replace(
            c, staking=dc_replace(c.staking, bankroll=new_val))
        return {"bankroll": new_val, "currency": c.staking.currency,
                "paper": new_val == 0}

    @app.get("/api/pnl")
    def pnl(metric: str = "high"):
        """Paper P&L track record for the Paper Ledger panel."""
        from highcast import signals
        return signals.paper_report(_conn(), cfg, metric)

    @app.get("/api/demo-account")
    def demo_account(refresh: int = 0,
                     bench: int = 0,
                     commit_hour: float | None = Query(None, ge=0, lt=24),
                     confidence: float | None = Query(None, gt=0, le=1),
                     aggressive_frac: float | None = Query(None, gt=0, le=1),
                     min_payoff: float | None = Query(None, ge=0, le=10),
                     min_edge: float | None = Query(None, ge=0, le=1),
                     price_ceiling: float | None = Query(None, gt=0, lt=1),
                     max_stake_frac: float | None = Query(None, gt=0, le=1),
                     balance: float | None = Query(None, ge=0, le=100000)):
        """THE DEMO ACCOUNT panel — `highcast.paper_account`, rendered.

        READ-ONLY: `persist=False`, so a page load never rebuilds
        `paper_account_txns` (the CLI's `highcast paper-account` owns those
        rows). Cached for `_DEMO_TTL` seconds because the seed is a live
        Kalshi balance read and the replay scans the whole archive; the
        dashboard polls every 5 minutes and neither should run per request.
        `?refresh=1` forces a recompute.

        Unlike `_KALSHI_CACHE`, a single unkeyed slot is right here: this
        account is ONE object spanning BOTH stations (`cfgs` below), not a
        per-station read, so there is no station to key on and therefore no
        cross-station collision to guard against.

        ── THE TEST BENCH (`?bench=1`) ──────────────────────────────────────
        Every parameter above is an ARGUMENT TO THE SIMULATION and nothing
        else. None of them writes a config file, mutates
        `gating.MIN_DECISION_HOUR`, or changes what the desk or the order bot
        would do on the next wake — `tests/test_web_testbench.py` asserts the
        shipped constants are byte-identical across a bench request with every
        knob pushed.

        That restraint is deliberate. This dashboard is reachable by anything
        on the machine, these values feed real staking arithmetic, and a
        browser tab left on an experimental setting would quietly become the
        shipped behaviour. So the bench is GET-only and stateless: there is no
        POST, no server-side "current experiment", and two tabs cannot see
        each other's knobs. Bench responses skip the cache for the same
        reason — a what-if must never be served to a plain page load.
        """
        nonlocal _demo_cache
        now = time.monotonic()
        knobs = {"commit_hour": commit_hour, "confidence": confidence,
                 "aggressive_frac": aggressive_frac, "min_payoff": min_payoff,
                 "min_edge": min_edge, "price_ceiling": price_ceiling,
                 "max_stake_frac": max_stake_frac, "balance": balance}
        is_bench = bool(bench) or any(v is not None for v in knobs.values())
        if not refresh and not is_bench and _demo_cache \
                and now - _demo_cache[0] < _DEMO_TTL:
            return _demo_cache[1]
        cfgs = {current().station.icao: current()}
        for icao in OTHER_STATION_CONFIGS:
            cfgs[icao] = _cfg_for(icao)
        if not is_bench:
            out = paper_account.refresh(_conn(), cfgs, persist=False)
            _demo_cache = (now, out)
            return out
        out = paper_account.bench(_conn(), cfgs, **knobs)
        out["bench"] = {k: v for k, v in
                        {**knobs, **out.get("bench", {})}.items()}
        return out

    @app.get("/api/app-bets")
    def app_bets_api(metric: str = "all"):
        """The APP's own committed bets — one per market per day, immutable.

        Read-only here on purpose. The only writer is the launchd wake cycle
        (`cli.auto_cycle`), so a bet exists only if the laptop was awake at the
        decision hour. See highcast/appbet.py for the integrity rules."""
        from highcast import appbet
        return appbet.ledger(_conn(), current(),
                             None if metric == "all" else metric)

    @app.get("/api/my-bets")
    def my_bets(metric: str = "high"):
        """The human's own bet ledger (kind='manual')."""
        from highcast import signals
        return signals.manual_report(_conn(), current(), metric)

    @app.post("/api/my-bet")
    def my_bet(body: MyBetIn):
        """Log a bet the human chose to place, priced off the current market."""
        from highcast import signals
        from highcast.metric import get_metric
        c = current()
        spec = get_metric(c, body.metric)
        conn = _conn()
        now = datetime.now(timezone.utc)
        out = signals.live_decision(conn, c, spec, now)
        market = out["market"]
        if not market:
            raise HTTPException(400, "no live market to price this bet")
        b = next((m for m in market if m["label"] == body.bucket), None)
        if b is None:
            labels = ", ".join(m["label"] for m in market)
            raise HTTPException(400, f"bucket '{body.bucket}' not in the current "
                                     f"market. Available: {labels}")
        price = body.price if body.price is not None else b["market_price"]
        if price is None:
            raise HTTPException(400, "no market price available for that bucket")
        if body.contracts <= 0:
            raise HTTPException(400, "contracts must be a positive integer")
        try:
            sid = signals.record_manual(conn, c, spec, out["target"], body.bucket,
                                        body.contracts, price, market, now)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "id": sid, "target": out["target"],
                "metric": body.metric, "bucket": body.bucket,
                "contracts": body.contracts, "price": round(float(price), 3)}

    @app.post("/api/note")
    def create_note(body: NoteIn):
        sky = body.sky.strip()
        if not sky or len(sky) > 500:
            raise HTTPException(400, "sky must be non-empty and at most 500 characters")
        try:
            target = date.fromisoformat(body.target_date)
        except ValueError:
            raise HTTPException(400, "target_date must be an ISO date (YYYY-MM-DD)")
        conn = _conn()
        now = datetime.now(timezone.utc)
        context = _note_context(conn, cfg, target, now)
        cur = conn.execute(
            """INSERT INTO notes (station, target_date, created_utc, sky, context)
               VALUES (?, ?, ?, ?, ?)""",
            (cfg.station.icao, target.isoformat(), utcnow_iso(), sky,
             json.dumps(context)))
        conn.commit()
        return {"ok": True, "id": cur.lastrowid}

    @app.get("/api/notes")
    def list_notes():
        conn = _conn()
        rows = conn.execute(
            """SELECT n.id, n.target_date, n.created_utc, n.sky, n.context,
                      c.high_f AS actual_high
               FROM notes n
               LEFT JOIN cli_reports c
                 ON c.station = n.station AND c.date = n.target_date
               WHERE n.station = ?
               ORDER BY n.id DESC LIMIT 50""",
            (cfg.station.icao,)).fetchall()
        return [{
            "id": r["id"], "target_date": r["target_date"],
            "created_utc": r["created_utc"], "sky": r["sky"],
            "context": json.loads(r["context"]) if r["context"] else None,
            "actual_high": r["actual_high"],
        } for r in rows]

    return app
