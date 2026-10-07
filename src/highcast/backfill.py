import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast import db
from highcast.config import AppConfig
from highcast.fetchers import alerts, cli_reports, fivemin, forecasts, metar
from highcast.http import get_json
from highcast.metric import get_metric
from highcast.timeutil import utcnow_iso


def gap_start(last_valid_utc: str | None, now: datetime,
              max_gap_hours: int = 2) -> datetime | None:
    if last_valid_utc is None:
        return now - timedelta(days=7)
    last = datetime.fromisoformat(last_valid_utc)
    if now - last <= timedelta(hours=max_gap_hours):
        return None
    return last


def _safe(label, thunk, counts, errors, progress):
    """Run one source's fetch+store in isolation.

    An unattended wake cycle must not lose five working sources because a
    sixth hit a transient network error (a DNS blip, an endpoint 5xx). Each
    source is fenced: on failure we record the error, count 0, and continue,
    so predictions still run on whatever data did arrive.
    """
    try:
        counts[label] = thunk()
    except Exception as e:  # noqa: BLE001 - transient upstream failures are expected
        errors[label] = f"{type(e).__name__}: {e}"
        counts[label] = 0
        progress(f"  [warn] {label} fetch failed, continuing: {errors[label]}")


def run(conn, cfg: AppConfig, years: int = 5, progress=print) -> dict:
    """Assemble the historical dataset. Idempotent: re-running upserts.

    Per-source failures are isolated so a mid-run network error doesn't
    discard everything already fetched; re-running fills whatever gaps remain.
    """
    counts: dict = {}
    errors: dict = {}
    this_year = datetime.now(timezone.utc).year
    for year in range(this_year - years, this_year + 1):
        _safe(f"cli_{year}",
              lambda y=year: cli_reports.store(conn, cli_reports.fetch_year(cfg, y)),
              counts, errors, progress)
        _safe(f"vtec_{year}",
              lambda y=year: alerts.store(conn, alerts.fetch_archive_year(cfg, y)),
              counts, errors, progress)
        progress(f"{year}: cli={counts.get(f'cli_{year}')} vtec={counts.get(f'vtec_{year}')}")
    start = date(this_year - years, 1, 1)
    end = datetime.now(timezone.utc).date()
    _safe("metar",
          lambda: metar.store(conn, metar.fetch_archive(cfg, start, end)),
          counts, errors, progress)
    progress(f"metar archive: {counts.get('metar')}")
    for mk in ("high", "low"):
        spec = get_metric(cfg, mk)
        _safe(f"om_lead1_{mk}",
              lambda s=spec: forecasts.store(
                  conn, forecasts.fetch_openmeteo_lead1(cfg, start, end, s)),
              counts, errors, progress)
    # NOTE: an earlier plan called for bounding this to the most recent 3
    # years because fetch_mos_archive originally issued one HTTP request per
    # day (12Z runtime enumeration, ~1800 requests for a 5-year pull). That
    # implementation was superseded in forecasts.py: fetch_mos_archive now
    # calls IEM's bulk mos.py request endpoint (format=csv) in ~182-day
    # chunks, so a 5-year pull is ~11 requests total (live-verified: a
    # 2-month window returned 923 rows in 1.7s). The per-day-request
    # rationale for truncating no longer applies, so the full `years` span
    # is used here like every other source.
    # BOTH metrics. The MOS N/X field carries the overnight MIN on its
    # morning-valid rows and the afternoon MAX on its afternoon-valid ones; the
    # parser used to discard the morning half outright, so the LOW had ZERO MOS
    # rows while the HIGH had 16,254 back to 2021. MOS is the only
    # station-calibrated guidance either metric gets. See `_mos_row`.
    for mk in ("high", "low"):
        _safe(f"mos_{mk}",
              lambda s=get_metric(cfg, mk): forecasts.store(
                  conn, forecasts.fetch_mos_archive(cfg, start, end, metric=s)),
              counts, errors, progress)
    # NBM station bulletins. The Open-Meteo `nbm` channel only reaches 2024-10;
    # this IEM bulletin archive reaches 2021. Separate model name (`nbm_nbs`).
    for mk in ("high", "low"):
        _safe(f"nbm_{mk}",
              lambda s=get_metric(cfg, mk): forecasts.store(
                  conn, forecasts.fetch_nbm_archive(cfg, start, end, metric=s)),
              counts, errors, progress)
        # ...and the NON-temperature bulletin fields: the marine-layer /
        # burn-off physics (sun, cloud, ceiling, onshore-vs-offshore flow).
        _safe(f"nbsaux_{mk}",
              lambda s=get_metric(cfg, mk): forecasts.store_nbs(
                  conn, forecasts.fetch_nbs_aux_archive(cfg, start, end, metric=s)),
              counts, errors, progress)
    progress(f"forecasts: om_lead1={counts.get('om_lead1')} "
             f"mos={counts.get('mos_high')} nbm_nbs={counts.get('nbm_high')}")
    if errors:
        counts["_errors"] = errors
    return counts


# Minimum gap between intraday guidance captures. Open-Meteo's models refresh
# hourly at best, so capturing more often adds rows without adding information.
INTRADAY_CAPTURE_MIN_GAP_MIN = 55

# NBM station bulletins: how often to refresh, and how far back to re-request.
# The IEM archive posts a few times a day, so hourly is generous. The lookback
# covers the forward bulletin horizon plus slack, so a wake cycle that was
# asleep for a day still closes its own hole without a manual backfill.
NBS_FETCH_MIN_GAP_MIN = 55

# ── CONVECTION (QUEUE item 3) ────────────────────────────────────────────────
# Day-ahead pop/qpf/cape. Dead from 2026-07-29 to 2026-08-07 because the fetch
# lived only in `scripts/fetch_convection.py` and was never wired in — the same
# defect `nbm_nbs` had, and the cure was never generalised.
#
# WHAT IT COST, 2026-08-06: KMIA settled a low of 75F at 01:51 local on a day
# with 0.41" of rain. Our best observation was 78.1F at the 01:53 METAR two
# minutes later, and that report filed NO 2sTTT six-hour min group (~3% of
# synoptic reports omit it), so nothing in the METAR stream ever carried the
# event. The model has no precipitation input at all and predicted 78-79. The
# market priced the cold scenario; we did not.
#
# 55 min, matching NBS: day-ahead convective guidance does not move faster than
# the model runs, and the wake cycle fires every 5 minutes.
CONV_FETCH_MIN_GAP_MIN = 55
CONV_MODELS = ("ncep_nbm_conus", "gfs_seamless", "gfs_hrrr")
CONV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _conv_fetch_due(conn, cfg: AppConfig) -> bool:
    """Convection's due-check. Thin caller of `_feed_fetch_due` — see the
    spec's "GENERALISE, do not copy"."""
    return _feed_fetch_due(conn, "forecast_conv", cfg.station.icao,
                           max_age_min=CONV_FETCH_MIN_GAP_MIN)


def _parse_conv(hourly: dict, icao: str, model: str, fetched_at: str) -> list[dict]:
    """Open-Meteo hourly block -> `forecast_conv` rows.

    ⚠️ `precipitation` SUBSTRING-MATCHES `precipitation_probability`, so the
    QPF column is picked by excluding the probability key rather than by
    trusting key order. The hand-run script needed this guard and it has to
    survive the move.
    """
    times = hourly.get("time") or []
    if not times:
        return []

    def col(name):
        for k in hourly:
            if name in k:
                return hourly[k]
        return [None] * len(times)

    pops, capes = col("precipitation_probability"), col("cape")
    qpfs = [None] * len(times)
    for k in hourly:
        if "precipitation" in k and "probability" not in k:
            qpfs = hourly[k]
    return [{"station": icao, "valid_utc": t, "model": model,
             "pop": p, "qpf": q, "cape": c, "fetched_at": fetched_at}
            for t, p, q, c in zip(times, pops, qpfs, capes)
            if p is not None or q is not None or c is not None]


def fetch_convection(cfg: AppConfig) -> list[dict]:
    """Day-ahead pop/qpf/cape for this station, across `CONV_MODELS`.

    A short window on purpose: the wake cycle keeps the recent days fresh, and
    the 730-day history the hand-run script pulled is a backfill job, not
    something to re-request hourly.
    """
    now = utcnow_iso()
    rows: list[dict] = []
    for model in CONV_MODELS:
        payload = get_json(CONV_URL, {
            "latitude": cfg.station.lat, "longitude": cfg.station.lon,
            "models": model,
            "hourly": ("precipitation_probability_previous_day1,"
                       "precipitation_previous_day1,cape_previous_day1"),
            "past_days": 3, "forecast_days": 1, "timezone": "UTC"})
        rows.extend(_parse_conv(payload.get("hourly", {}) or {},
                                cfg.station.icao, model, now))
    return rows
NBS_REFRESH_LOOKBACK_DAYS = 3


def _feed_fetch_due(conn, table: str, station: str, *,
                    model: str | None = None,
                    max_age_min: float) -> bool:
    """ONE due-check for every rate-limited feed. Per station, fails OPEN.

    THE SPEC ASKED FOR THIS BY NAME (2026-08-02 convection-feed-revival, §4.1):
    "GENERALISE, do not copy. Two near-identical due-checks is how the third
    one gets written wrong." This project has been bitten four times by
    one-truth-two-copies — `orderbot.CONFIDENCE_FLOOR` shadowing `gating`'s,
    `fc_anchor` re-entering its own blend, a settlement predicate that guarded
    fetching but not grading, and 13 scripts reading a config field that had
    gone per-metric.

    PER STATION, always: a shared clock is why `forecast_conv` held zero KLAX
    rows for a year — one station's fetch would satisfy the other's check.

    FAILS OPEN on a missing or unparseable stamp. One extra HTTP call costs
    nothing; a feed going quiet cost a whole bucket on 2026-08-06.

    `table` is interpolated into the SQL, so it must never come from user
    input — every caller passes a literal.
    """
    sql = f"SELECT max(fetched_at) AS f FROM {table} WHERE station=?"
    args: tuple = (station,)
    if model is not None:
        sql += " AND model=?"
        args = (station, model)
    row = conn.execute(sql, args).fetchone()
    if row is None or row["f"] is None:
        return True
    try:
        last = datetime.fromisoformat(row["f"])
    except (TypeError, ValueError):
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    return age_min >= max_age_min


def _nbs_fetch_due(conn, cfg: AppConfig) -> bool:
    """NBS's due-check. Thin caller of `_feed_fetch_due`.

    Keyed on `fetched_at`, not `target_date`: the question is when we last
    ASKED IEM, and a run that legitimately returns no new targets must still
    reset the clock or every wake cycle would re-request.
    """
    return _feed_fetch_due(conn, "forecasts", cfg.station.icao,
                           model="nbm_nbs",
                           max_age_min=NBS_FETCH_MIN_GAP_MIN)


def _intraday_capture_due(conn, cfg: AppConfig) -> bool:
    """Has it been long enough since the last capture for this station?

    Keyed off the LAST STORED ROW rather than a timer, so a restart, a laptop
    sleep, or a crash cannot lose the cadence or double-fetch on resume. Failing
    OPEN (returning True when the table is empty or unreadable) is deliberate:
    the cost of an extra fetch is one HTTP call, while the cost of skipping is a
    permanent hole in a dataset that can never be backfilled.
    """
    try:
        row = conn.execute(
            "SELECT MAX(fetched_at) AS m FROM forecast_intraday WHERE station=?",
            (cfg.station.icao,)).fetchone()
    except Exception:  # noqa: BLE001 - a missing table must not stop the capture
        return True
    if row is None or row["m"] is None:
        return True
    try:
        last = datetime.fromisoformat(row["m"])
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    gap = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    return gap >= INTRADAY_CAPTURE_MIN_GAP_MIN


# `cli_row_is_final` used to live here. It now lives in
# `highcast.fetchers.cli_reports` (2026-08-02), because settlement
# (`pick.settle`, `appbet.settle`) needs the SAME predicate this module's
# re-fetch decision uses below -- one definition, not a second copy grown at
# the settle call sites. Imported via the `cli_reports` module already
# imported above; see that module for the full docstring and incident history.


def fetch_snapshot(conn, cfg: AppConfig, progress=print) -> dict:
    """One live fetch cycle: obs + forecasts + alerts + yesterday's CLI.

    Each source is fenced (see `_safe`) so one transient failure can't abort
    the cycle. A `_errors` key summarises any sources that failed this cycle.
    """
    counts: dict = {}
    errors: dict = {}
    _safe("metar", lambda: metar.store(conn, metar.fetch_recent(cfg)),
          counts, errors, progress)
    _safe("syn5", lambda: fivemin.store(conn, fivemin.fetch_recent(cfg)),
          counts, errors, progress)
    for mk in ("high", "low"):
        spec = get_metric(cfg, mk)
        _safe(f"openmeteo_{mk}",
              lambda s=spec: forecasts.store(conn, forecasts.fetch_openmeteo(cfg, s)),
              counts, errors, progress)
        _safe(f"nws_{mk}",
              lambda s=spec: forecasts.store(conn, forecasts.fetch_nws(cfg, conn, s)),
              counts, errors, progress)
    # INTRADAY GUIDANCE CAPTURE. Forward-accumulation for a hypothesis that cannot
    # be tested on history: at the HIGH's h11 commit the guidance in use is 11.1
    # HOURS old, because same-day model runs only reach `forecasts` at 06:55Z the
    # NEXT morning. Freshness is a property of WHEN a row was fetched, so it can
    # never be backfilled -- every day not captured is permanently missing.
    #
    # Rate-limited to hourly: the wake cycle runs every 300s and Open-Meteo only
    # refreshes hourly, so 288 fetches a day would be noise against their API.
    #
    # Writes to `forecast_intraday`, NEVER to `forecasts`. The LOW's feature cutoff
    # is 20:00 local the PREVIOUS day, so a midday row landing in `forecasts` would
    # fall before the next target's LOW cutoff and silently move `base_p50` for the
    # FROZEN markets. Nothing reads this table yet.
    if _intraday_capture_due(conn, cfg):
        for mk in ("high", "low"):
            _safe(f"intraday_{mk}",
                  lambda s=get_metric(cfg, mk): forecasts.store_intraday(
                      conn, forecasts.fetch_intraday(cfg, s)),
                  counts, errors, progress)
        # HOURLY TEMPERATURE CURVE, on the same clock. The daily min/max tables
        # cannot express "tonight will be colder than this morning was", which
        # is the 11%-of-days evening-minimum case the LOW keeps losing on.
        # Metric-agnostic: one series serves both. Cannot be backfilled.
        _safe("hourly",
              lambda: forecasts.store_hourly(conn, forecasts.fetch_hourly(cfg)),
              counts, errors, progress)
    for mk in ("high", "low"):
        _safe(f"mos_{mk}",
              lambda s=get_metric(cfg, mk): forecasts.store(
                  conn, forecasts.fetch_mos(cfg, metric=s)),
              counts, errors, progress)
    # ── NBM STATION BULLETINS ───────────────────────────────────────────────
    #
    # ⚠️ THIS WAS MISSING, AND IT SILENTLY BROKE THE DAY-AHEAD LOW.
    # `nbm_nbs` and its aux fields feed SEVEN of the LOW's day-ahead features
    # (fc_nbm_nbs, fcc_nbm_nbs, nbs_sol_mean, nbs_sky_mean, nbs_cig_min,
    # nbs_offshore, nbs_tsd_mean), but they were populated ONLY by the manual
    # `highcast backfill` command -- never by the wake cycle. So the data simply
    # ran out after the last manual run (KLAX 2026-07-26, KMIA 2026-07-28), and
    # from the first uncovered target the model ran on NaN in a third of its
    # inputs. It did not fail loudly; it drifted 5-7F cold against unanimous
    # guidance and put 35% on "<=60F" in July. Found 2026-07-31.
    #
    # `gating.nbs_guidance_ready` is the safety net that refuses to trade on the
    # NaN regime. THIS is the actual cure. Both stay: a fetch can fail again.
    #
    # Rate-limited to hourly on the same clock as the intraday capture -- the
    # IEM bulletin archive posts a few times a day, so the 5-minute wake cycle
    # would otherwise hammer it ~288 times daily for the same rows.
    if _nbs_fetch_due(conn, cfg):
        today = datetime.now(timezone.utc).date()
        span_start = today - timedelta(days=NBS_REFRESH_LOOKBACK_DAYS)
        for mk in ("high", "low"):
            _safe(f"nbm_{mk}",
                  lambda s=get_metric(cfg, mk): forecasts.store(
                      conn, forecasts.fetch_nbm_archive(
                          cfg, span_start, today, metric=s)),
                  counts, errors, progress)
            _safe(f"nbsaux_{mk}",
                  lambda s=get_metric(cfg, mk): forecasts.store_nbs(
                      conn, forecasts.fetch_nbs_aux_archive(
                          cfg, span_start, today, metric=s)),
                  counts, errors, progress)
    # Convection — QUEUE item 3. Rate-limited like nbm_nbs; see
    # `_conv_fetch_due` for what its nine-day outage cost on 2026-08-06.
    # COLLECTION ONLY: nothing in `features`, `models/` or `staking` reads
    # `forecast_conv`, and the gate on ever using it is that item's Task 0
    # (does the FORECAST identify the same days the OBSERVATION does?).
    if _conv_fetch_due(conn, cfg):
        _safe("conv", lambda: db.upsert(conn, "forecast_conv",
                                        fetch_convection(cfg)),
              counts, errors, progress)
    _safe("ensemble",
          lambda: forecasts.store_ens(conn, forecasts.fetch_openmeteo_ensemble(cfg)),
          counts, errors, progress)
    _safe("aloft",
          lambda: forecasts.store_aloft(conn, forecasts.fetch_openmeteo_aloft(cfg)),
          counts, errors, progress)
    _safe("alerts", lambda: alerts.store(conn, alerts.fetch_active(cfg)),
          counts, errors, progress)

    def _yesterday_cli():
        yesterday = (datetime.now(ZoneInfo(cfg.station.tz)).date()
                     - timedelta(days=1)).isoformat()
        have = conn.execute(
            "SELECT raw FROM cli_reports WHERE station=? AND date=?",
            (cfg.station.icao, yesterday)).fetchone()
        # PRESENCE IS NOT ENOUGH -- the stored row may be a PRELIMINARY report.
        # This used to be `if have: return 0`, which froze whatever landed first.
        # On 2026-07-30 NWS issued LAX's report at 18:29 local, hours before the
        # climate day closed; it said "low 70 at 5:52 AM" and the day went on to
        # bottom at 67 at 11:59 PM. The row was never revisited, the ">=69" bet
        # was graded a WIN when it lost, and an audit that checked `app_bets`
        # against `cli_reports` confirmed nothing because both read the same
        # corrupted row. See `tests/test_cli_preliminary.py`.
        if have:
            try:
                product = json.loads(have["raw"]).get("product")
            except Exception:
                product = None
            if cli_reports.cli_row_is_final(
                    product, yesterday, cfg.station.std_utc_offset):
                return 0
        return cli_reports.store(conn, cli_reports.fetch_year(cfg, int(yesterday[:4])))

    _safe("cli", _yesterday_cli, counts, errors, progress)
    if errors:
        counts["_errors"] = errors
    return counts


def backfill_ensemble(conn, cfg: AppConfig, progress=print) -> dict:
    """One-time historical grab of all available ensemble history (API cap is
    93 past days). Safe to re-run; INSERT OR REPLACE dedupes by PK."""
    rows = forecasts.fetch_openmeteo_ensemble(cfg, past_days=93, forecast_days=1)
    n = forecasts.store_ens(conn, rows)
    progress(f"ensemble backfill: {n} day-rows")
    return {"ensemble_backfill": n}


def backfill_aloft(conn, cfg: AppConfig, progress=print) -> dict:
    """One-time seed of available 850 hPa history (~2 weeks; live-only for KLAX).
    Safe to re-run; INSERT OR REPLACE dedupes by PK. Accumulates forward."""
    rows = forecasts.fetch_openmeteo_aloft(cfg, past_days=90, forecast_days=1)
    n = forecasts.store_aloft(conn, rows)
    progress(f"aloft backfill: {n} day-rows")
    return {"aloft_backfill": n}


# Every column `parse_metar` derives from the raw text. A field missing from this
# tuple is silently never backfilled: `reparse-obs` reports "updated 0" and the
# column stays NULL forever, because the comparison that decides whether a row
# changed only looks at the columns listed here. That happened on 2026-07-30 when
# the environment fields were added to the parser but not to this list.
_DERIVED_COLS = ("tmpf", "tmpf_precise", "dwpf", "sknt", "drct", "skyc",
                 "six_max_c", "six_min_c", "slp_hpa", "pres_tend_hpa",
                 "pres_tend_code",
                 "vis_sm", "ceiling_ft", "sky_lowest_ft", "n_sky_layers",
                 "sky_cb", "wx_present", "wx_thunder")


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def reparse_obs(conn, cfg: AppConfig, progress=print) -> dict:
    """Re-run parse_metar over every stored raw METAR and update the derived
    columns in place. Backfills six_min_c for history without any network.

    The archive holds tens of thousands of raw reports parsed by older
    versions of parse_metar (notably before the 6-hour MINIMUM group 2sTTT
    was read at all). Rather than re-fetch them, re-derive from the raw text
    we already have. Idempotent: rows whose derived columns already match the
    current parser are left untouched, so a second run reports 0 changed.
    """
    rows = conn.execute(
        "SELECT valid_utc, raw, " + ", ".join(_DERIVED_COLS) + " FROM obs "
        "WHERE station=? AND source='metar' AND raw IS NOT NULL AND raw != ''",
        (cfg.station.icao,)).fetchall()
    updates = []
    for r in rows:
        parsed = metar.parse_metar(r["raw"], r["valid_utc"], cfg.station.icao, "")
        if all(_same(parsed[c], r[c]) for c in _DERIVED_COLS):
            continue
        updates.append([parsed[c] for c in _DERIVED_COLS]
                       + [cfg.station.icao, r["valid_utc"]])
    if updates:
        conn.executemany(
            "UPDATE obs SET " + ", ".join(f"{c}=?" for c in _DERIVED_COLS)
            + " WHERE station=? AND valid_utc=? AND source='metar'", updates)
        conn.commit()
    progress(f"reparse-obs: scanned {len(rows)} raw METAR(s), "
             f"updated {len(updates)}")
    return {"scanned": len(rows), "changed": len(updates)}


def fill_gaps(conn, cfg: AppConfig, progress=print) -> dict:
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "SELECT MAX(valid_utc) AS m FROM obs WHERE station=? AND source='metar'",
        (cfg.station.icao,)).fetchone()
    start = gap_start(row["m"] if row else None, now)
    counts: dict = {}
    errors: dict = {}
    if start is not None:
        _safe("metar_gap",
              lambda: metar.store(conn, metar.fetch_archive(cfg, start.date(), now.date())),
              counts, errors, progress)
        progress(f"backfilled metar gap since {start.date()}: {counts.get('metar_gap')}")
    snap = fetch_snapshot(conn, cfg, progress=progress)
    snap_errors = snap.pop("_errors", {})
    counts.update(snap)
    errors.update(snap_errors)
    if errors:
        counts["_errors"] = errors
    return counts
