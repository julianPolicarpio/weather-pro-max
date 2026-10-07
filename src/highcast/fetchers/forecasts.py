import csv
import io
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json, get_text
from highcast.metric import MetricSpec, get_metric
from highcast.timeutil import c_to_f, climate_day_bounds, utcnow_iso

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"
IEM_MOS = "https://mesonet.agron.iastate.edu/api/1/mos.json"
# Bulk MOS archive service: unlike /api/1/mos.json (single runtime only, see
# fetch_mos below), this accepts a real sts/ets date range and returns every
# archived MOS cycle in one CSV response. Verified live 2026-07-17 for
# the station/MEX across 2021-2024 windows (see task-8-report.md for evidence).
IEM_MOS_ARCHIVE = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"

MODELS = {"ecmwf": "ecmwf_ifs025", "gfs": "gfs_seamless",
          "icon": "icon_seamless", "hrrr": "gfs_hrrr",
          # Added 2026-07-24: NBM (a calibrated multi-model blend that also
          # carries the daily MIN — the low's biggest gap), NAM Nest, and
          # ECMWF-AIFS (the Euro's GraphCast-style AI model). All three serve
          # both temperature_2m_max/min and the lead-1 archive.
          "nbm": "ncep_nbm_conus", "nam": "ncep_nam_conus",
          "aifs": "ecmwf_aifs025_single"}
_BY_SUFFIX = {v: k for k, v in MODELS.items()}

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODELS = "gfs025,ecmwf_ifs025,icon_seamless"
_ENS_MEMBER_RE = re.compile(r"^temperature_2m_member\d+_")
_ENS_FLOOR = 10          # min usable members for a summary; else NaN
_ENS_FRAC_THRESH = 82.0  # seam upper edge — fraction of members at/above this
_ENS_STAT_KEYS = ["ens_mean", "ens_std", "ens_p10", "ens_p50", "ens_p90",
                  "ens_frac_ge_82", "ens_bimodality"]

# Default MetricSpec for the pure parse_* functions below, which have no
# `cfg` in scope to call metric.get_metric(cfg, "high") themselves. This
# mirrors that "high" spec exactly except kalshi_series (never read by these
# functions, and only resolvable from cfg) so existing high call sites that
# don't pass `metric` explicitly keep working unchanged.
_HIGH_METRIC = MetricSpec(
    key="high", target_col="high_f",
    openmeteo_var="temperature_2m_max", nws_field="maxTemperature",
    mos_field="n_x", kalshi_series="", extremum="max", tail_dir="floor",
    cutoff_local_hour=9, cutoff_day_offset=0,
    fc_models=["ecmwf", "gfs", "icon", "hrrr", "nws", "mos_mex"])


def parse_openmeteo(payload: dict, icao: str, run_date: str, fetched_at: str,
                    metric: MetricSpec = _HIGH_METRIC) -> list[dict]:
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    rows = []
    for key, values in daily.items():
        if not key.startswith(metric.openmeteo_var):
            continue
        suffix = key.removeprefix(metric.openmeteo_var).lstrip("_")
        model = _BY_SUFFIX.get(suffix, suffix or "openmeteo")
        for d, v in zip(dates, values):
            if v is None:
                continue
            rows.append({"station": icao, "target_date": d, "model": model,
                         "run_date": run_date, "max_temp_f": float(v),
                         "fetched_at": fetched_at, "metric": metric.key})
    return rows


def _ens_summary(member_extremes: list[float], metric_key: str = "high") -> dict:
    """Summary stats of one day's ~119 member daily extremes. Below the member
    floor, every stat is None (LightGBM will later treat these as missing).

    ⚠️ `ens_frac_ge_82` IS HIGH-ONLY. 82F is the KLAX marine-layer seam's upper
    edge — "what fraction of members clear the seam". It has no meaning for a
    LOW, and choosing some low-side threshold here would be inventing a
    constant nobody has measured. It comes back None for the low.
    """
    arr = np.array([m for m in member_extremes if m is not None], dtype=float)
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
        "ens_frac_ge_82": (float((arr >= _ENS_FRAC_THRESH).mean())
                           if metric_key == "high" else None),
        # normalized largest gap between adjacent sorted members: big when the
        # members split into two clumps (the marine-layer seam), ~0 when tight.
        "ens_bimodality": float(gaps.max() / (std + 1e-6)) if gaps.size else 0.0,
    }


def parse_openmeteo_ensemble(payload: dict, icao: str, run_date: str,
                             fetched_at: str, std_utc_offset: int,
                             metric_key: str = "high") -> list[dict]:
    """One summary row per target_date in the payload. Each member's daily
    EXTREME is taken over the NWS climate day (`climate_day_bounds`) — max for
    the high, MIN for the low.

    ⚠️ THE AGGREGATION USED TO BE HARDCODED `max`. `metric_key` was stamped on
    every row while the arithmetic ignored it, so asking for the low would have
    produced rows LABELLED "low" carrying each member's daily MAXIMUM, silently,
    into a table the desk reads. Nothing had ever asked for the low, so it never
    fired. Fixed 2026-08-07 when the station table started showing the lows.
    """
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
        agg = min if metric_key == "low" else max
        extremes = []
        for col in member_cols:
            vals = [hourly[col][i] for i in idx if hourly[col][i] is not None]
            if vals:
                extremes.append(agg(vals))
        rows.append({"station": icao, "target_date": d.isoformat(),
                     "run_date": run_date, "metric": metric_key,
                     "fetched_at": fetched_at,
                     **_ens_summary(extremes, metric_key)})
    return rows


def fetch_openmeteo_ensemble(cfg: AppConfig, past_days: int = 2,
                             forecast_days: int = 2) -> list[dict]:
    """Ensemble members summarised per day, for BOTH metrics. `past_days` <= 93
    (API cap). Capture-only: callers store into forecast_ens via store_ens.

    ONE HTTP CALL, TWO METRICS. The member temperatures are the same hourly
    payload; the high and the low are just different aggregations of it, so
    fetching twice would double the request for nothing.

    The LOW was absent entirely until 2026-08-07 — `forecast_ens` held 164
    KLAX-high and 145 KMIA-high rows and zero low rows, so both LOW markets
    showed no ensemble on the station table. The lows are the markets where
    KMIA keeps getting hurt.
    """
    st = cfg.station
    payload = get_json(ENSEMBLE_URL, {
        "latitude": st.lat, "longitude": st.lon,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
        "timezone": "GMT", "past_days": past_days, "forecast_days": forecast_days,
        "models": ENSEMBLE_MODELS,
    })
    run_date = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    rows = []
    for metric_key in ("high", "low"):
        rows.extend(parse_openmeteo_ensemble(
            payload, st.icao, run_date, utcnow_iso(), st.std_utc_offset,
            metric_key))
    return rows


def fetch_openmeteo(cfg: AppConfig, metric: MetricSpec | None = None) -> list[dict]:
    metric = metric or get_metric(cfg, "high")
    st = cfg.station
    today_local = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    payload = get_json(OPEN_METEO, {
        "latitude": st.lat, "longitude": st.lon,
        "daily": metric.openmeteo_var, "temperature_unit": "fahrenheit",
        "timezone": st.tz, "forecast_days": 3,
        "models": ",".join(MODELS.values()),
    })
    return parse_openmeteo(payload, st.icao, today_local, utcnow_iso(), metric)


def parse_openmeteo_lead1(payload: dict, icao: str, fetched_at: str,
                          metric: MetricSpec = _HIGH_METRIC) -> list[dict]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    agg = min if metric.extremum == "min" else max
    seed = float("inf") if metric.extremum == "min" else float("-inf")
    rows = []
    for key, values in hourly.items():
        if not key.startswith("temperature_2m_previous_day1"):
            continue
        suffix = key.removeprefix("temperature_2m_previous_day1").lstrip("_")
        model = _BY_SUFFIX.get(suffix, suffix or "openmeteo")
        dayagg: dict[str, float] = defaultdict(lambda: seed)
        for t, v in zip(times, values):
            if v is not None:
                d = t[:10]
                dayagg[d] = agg(dayagg[d], float(v))
        for d, v in sorted(dayagg.items()):
            run = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
            # fetched_at must reflect when the lead-1 forecast was ISSUED (its
            # run day), not when we backfilled it. Storing the backfill wall-time
            # (the passed `fetched_at`) made every historical forecast look like
            # it arrived in the future, so the LOW's prior-evening cutoff rejected
            # them all (the HIGH's run_date<target cutoff masked the bug). Anchor
            # it to run-day noon UTC — safely before any prediction cutoff.
            rows.append({"station": icao, "target_date": d, "model": model,
                         "run_date": run, "max_temp_f": v,
                         "fetched_at": f"{run}T12:00:00+00:00",
                         "metric": metric.key})
    return rows


def fetch_openmeteo_lead1(cfg: AppConfig, start: date, end: date,
                          metric: MetricSpec | None = None) -> list[dict]:
    metric = metric or get_metric(cfg, "high")
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
        rows.extend(parse_openmeteo_lead1(payload, st.icao, utcnow_iso(), metric))
        cur = stop + timedelta(days=1)
    return rows


def parse_nws(payload: dict, icao: str, run_date: str, fetched_at: str,
              tz: str, metric: MetricSpec = _HIGH_METRIC) -> list[dict]:
    zone = ZoneInfo(tz)
    rows = []
    for item in payload["properties"][metric.nws_field]["values"]:
        start_iso = item["validTime"].split("/")[0]
        local_day = datetime.fromisoformat(start_iso).astimezone(zone).date()
        if item["value"] is None:
            continue
        rows.append({"station": icao, "target_date": local_day.isoformat(),
                     "model": "nws", "run_date": run_date,
                     "max_temp_f": c_to_f(float(item["value"])),
                     "fetched_at": fetched_at, "metric": metric.key})
    return rows


def fetch_nws(cfg: AppConfig, conn, metric: MetricSpec | None = None) -> list[dict]:
    metric = metric or get_metric(cfg, "high")
    st = cfg.station
    grid_url = db.get_meta(conn, "nws_grid_url")
    if not grid_url:
        pts = get_json(NWS_POINTS.format(lat=st.lat, lon=st.lon))
        grid_url = pts["properties"]["forecastGridData"]
        db.set_meta(conn, "nws_grid_url", grid_url)
    payload = get_json(grid_url)
    today_local = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    return parse_nws(payload, st.icao, today_local, utcnow_iso(), st.tz, metric)


def _mos_row(runtime: str, ftime: str, nx, icao: str, model_name: str,
            fetched_at: str, zone: ZoneInfo, extremum: str,
            std_utc_offset: int) -> dict | None:
    # REALITY CHECK (differs from the brief): IEM's MOS "n_x" column is the
    # NWS MOS N/X field, which alternates between the overnight MIN and the
    # following afternoon's MAX at each ~12h forecast step - it is NOT
    # already a daily max keyed by a UTC calendar date. Confirmed live for
    # the station/MEX: ftime valid ~00Z (=~5pm local) carries the high (e.g. 82F),
    # ftime valid ~12Z (=~5am local) carries the low (e.g. 69F), and both can
    # fall on the same naive `ftime[:10]` UTC-date bucket. Naively upserting
    # every n_x value under `target_date = ftime[:10]` intermittently
    # clobbers the true daily max with the overnight min (same primary key).
    # Fix: convert ftime to the station's local time and keep only the rows
    # carrying the extremum we want, bucketed by CLIMATE day.
    #
    # 2026-07-26 -- `extremum="min"` ADDED, and it matters a lot. This function
    # used to hard-drop every morning-valid row with the comment "overnight low,
    # not the daily max we want", and `fetch_mos` claimed "IEM MOS has no
    # min-temp field for the low metric". THAT CLAIM WAS WRONG: the min is in the
    # same `n_x` column, on the morning-valid rows, and it was being discarded.
    # The LOW consequently had ZERO MOS rows while the HIGH had 16,254 back to
    # 2021 -- the LOW was missing its only station-calibrated guidance channel
    # and running on raw NWP grid points alone. Verified live against the archive
    # for 2026-07-19..25: local-05:00 rows read 69/66/67/68/67/65/65 F.
    #
    # Bucketing is by CLIMATE day (08:00Z->08:00Z, DST-invariant) rather than
    # naive local date, because a 00:00-00:59 local valid time belongs to the
    # PREVIOUS climate day. MOS only emits 12Z/00Z (05:00/17:00 local) so the two
    # agree in practice today, and `test_mos_min` pins that the HIGH's mapping is
    # byte-identical to the old local-date one -- but being right by construction
    # is what keeps a future cycle change from silently misfiling a day.
    #
    # Shared by parse_mos (single-cycle JSON, from /api/1/mos.json) and
    # parse_mos_csv (bulk-range CSV, from the archive request service) so
    # the target_date/run_date mapping logic lives in exactly one place.
    if nx in (None, "", "M") or not ftime or not runtime:
        return None
    ftime_utc = datetime.fromisoformat(str(ftime).replace(" ", "T")).replace(
        tzinfo=timezone.utc)
    ftime_local = ftime_utc.astimezone(zone)
    afternoon = ftime_local.hour >= 12
    if afternoon != (extremum == "max"):
        return None      # this row carries the OTHER extremum
    runtime_local = datetime.fromisoformat(str(runtime).replace(" ", "T")).replace(
        tzinfo=timezone.utc).astimezone(zone)
    climate_day = (ftime_utc + timedelta(hours=std_utc_offset)).date()
    return {"station": icao, "target_date": climate_day.isoformat(),
            "model": model_name, "run_date": runtime_local.date().isoformat(),
            "max_temp_f": float(nx), "fetched_at": fetched_at}


def parse_mos(payload: dict, icao: str, fetched_at: str,
              model_name: str, tz: str, extremum: str,
              std_utc_offset: int) -> list[dict]:
    """`extremum` and `std_utc_offset` are REQUIRED, deliberately. Defaulting
    the offset to 0 made `_mos_row` bucket by UTC date instead of climate day,
    which silently shifted the HIGH's target_date by one for every 00Z-valid row
    -- a wrong answer rather than the old behaviour. There is no safe default for
    "which climate day is this", so callers must say."""
    zone = ZoneInfo(tz)
    rows = []
    for item in payload.get("data", []):
        row = _mos_row(item.get("runtime") or "", item.get("ftime") or "",
                       item.get("n_x"), icao, model_name, fetched_at, zone,
                       extremum, std_utc_offset)
        if row is not None:
            rows.append(row)
    return rows


def parse_mos_csv(text: str, icao: str, fetched_at: str,
                  model_name: str, tz: str, extremum: str,
                  std_utc_offset: int) -> list[dict]:
    """Parse the bulk MOS archive CSV (request/mos.py?format=csv).

    Columns of interest: runtime, ftime, n_x (station/model columns are also
    present but every row already matches the requested station+model, so
    they aren't filtered on here). Column set varies slightly by era (older
    archives carry extra fields like typ/psn) - DictReader handles that fine
    since we only pull the three columns we need.
    """
    zone = ZoneInfo(tz)
    rows = []
    for item in csv.DictReader(io.StringIO(text)):
        row = _mos_row(item.get("runtime") or "", item.get("ftime") or "",
                       item.get("n_x"), icao, model_name, fetched_at, zone,
                       extremum, std_utc_offset)
        if row is not None:
            rows.append(row)
    return rows


def fetch_mos(cfg: AppConfig, model: str = "MEX",
             metric: MetricSpec | None = None) -> list[dict]:
    metric = metric or get_metric(cfg, "high")
    if metric.mos_field is None:
        return []
    payload = get_json(IEM_MOS, {"station": cfg.station.icao, "model": model})
    rows = parse_mos(payload, cfg.station.icao, utcnow_iso(),
                     f"mos_{model.lower()}", cfg.station.tz,
                     metric.extremum, cfg.station.std_utc_offset)
    for r in rows:
        r["metric"] = metric.key
    return rows


def parse_nbs_csv(text: str, icao: str, fetched_at: str, model_name: str,
                  tz: str, extremum: str, std_utc_offset: int) -> list[dict]:
    """Daily extremum per (run, climate day) from NBM/NBS 3-hourly `tmp`.

    WHY A SEPARATE PARSER FROM MOS. The NBS bulletin has no `n_x` column at all
    (verified live: 759 rows, `n_x` empty on every one). It carries 3-hourly
    `tmp` plus a sparse `txn`. Aggregating `tmp` over the climate day is both
    more available than `txn` and uses the same 08:00Z->08:00Z definition the
    rest of the codebase settles on, instead of trusting the bulletin's own
    day boundary.

    KNOWN BIAS, stated so it is never mistaken for a bug: 3-hourly sampling
    cannot see a peak that falls between steps, so the derived max runs slightly
    COOL and the derived min slightly WARM. It is a systematic offset, which is
    exactly what a tree absorbs into its split thresholds and what the LOW's
    trailing `fcc_*` correction removes outright.

    `run_date` is the runtime's STATION-LOCAL date, matching `_mos_row`, so the
    existing day-ahead leakage filter in `features._forecast_features`
    (`run_date < cutoff_date`) applies unchanged.
    """
    zone = ZoneInfo(tz)
    agg: dict[tuple[str, str], float] = {}
    pick = max if extremum == "max" else min
    for item in csv.DictReader(io.StringIO(text)):
        tmp, ftime, runtime = item.get("tmp"), item.get("ftime"), item.get("runtime")
        if tmp in (None, "", "M") or not ftime or not runtime:
            continue
        ftime_utc = datetime.fromisoformat(str(ftime).replace(" ", "T")).replace(
            tzinfo=timezone.utc)
        climate_day = (ftime_utc + timedelta(hours=std_utc_offset)).date()
        run_local = datetime.fromisoformat(
            str(runtime).replace(" ", "T")).replace(
                tzinfo=timezone.utc).astimezone(zone).date()
        key = (climate_day.isoformat(), run_local.isoformat())
        v = float(tmp)
        agg[key] = v if key not in agg else pick(agg[key], v)
    return [{"station": icao, "target_date": td, "model": model_name,
             "run_date": rd, "max_temp_f": v, "fetched_at": fetched_at}
            for (td, rd), v in sorted(agg.items())]


# Station-local hours over which each metric's extremum forms. The NBS aux
# fields are averaged over this window only -- cloud cover at 3am tells you
# nothing about whether a 2pm max burns off, and averaging over the whole day
# would dilute the signal that matters into one that doesn't.
_AUX_WINDOW = {"max": (9, 17), "min": (0, 8)}


# NBM/NBS uses -88 for "unlimited" ceiling / cloud base (i.e. NO deck at all).
# Taken literally it is catastrophic for these features: `min(cig)` over a window
# containing one clear period returns -88, so the CLEAREST possible sky scores as
# the LOWEST possible marine-layer deck -- the signal exactly inverted, on
# precisely the hot burn-off days the feature exists to identify. Map it to a
# large sentinel so "no ceiling" sorts as high, which is what it physically means.
_UNLIMITED_CIG = -88.0
_CIG_CLEAR = 999.0


def _num(v):
    if v in (None, "", "M"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _num_cig(v):
    """`_num` for ceiling-like fields, with the -88 unlimited sentinel remapped."""
    x = _num(v)
    if x is None:
        return None
    return _CIG_CLEAR if x == _UNLIMITED_CIG else x


def parse_nbs_aux_csv(text: str, icao: str, fetched_at: str, tz: str,
                      extremum: str, std_utc_offset: int) -> list[dict]:
    """Marine-layer / burn-off predictors from the NBS bulletin, per (run, day).

    WHY THESE FIELDS. At a coastal station the daily max is governed by whether
    and when the marine layer burns off, which is a threshold phenomenon that
    temperature guidance alone represents badly -- it is the mechanism behind the
    HIGH's weakness and behind the -10 F misses on days like 2026-07-17.
    `sol` (forecast solar radiation) is the most direct available proxy for it,
    `sky`/`cig`/`lcb` describe the deck itself, and `wdr` distinguishes onshore
    sea-breeze from offshore/downslope flow.

    THIS SUBSUMES §5b. The handoff's long-standing "Santa Ana / offshore pressure
    gradient" idea needed a second station's SLP (LAX-KDAG) fetched and
    maintained. `nbs_offshore` -- the share of the window with a land-sector wind
    direction -- captures the same regime from data already being downloaded.

    All fields verified 100% populated for KLAX back to 2021.
    """
    zone = ZoneInfo(tz)
    lo_h, hi_h = _AUX_WINDOW[extremum]
    acc: dict[tuple[str, str], dict[str, list]] = {}
    for item in csv.DictReader(io.StringIO(text)):
        ftime, runtime = item.get("ftime"), item.get("runtime")
        if not ftime or not runtime:
            continue
        ftime_utc = datetime.fromisoformat(str(ftime).replace(" ", "T")).replace(
            tzinfo=timezone.utc)
        local_hour = ftime_utc.astimezone(zone).hour
        if not (lo_h <= local_hour < hi_h):
            continue
        climate_day = (ftime_utc + timedelta(hours=std_utc_offset)).date()
        run_local = datetime.fromisoformat(
            str(runtime).replace(" ", "T")).replace(
                tzinfo=timezone.utc).astimezone(zone).date()
        b = acc.setdefault((climate_day.isoformat(), run_local.isoformat()),
                           {k: [] for k in
                            ("sol", "sky", "cig", "lcb", "wdr", "wsp", "tsd", "dpt")})
        for k in b:
            v = (_num_cig(item.get(k)) if k in ("cig", "lcb")
                 else _num(item.get(k)))
            if v is not None:
                b[k].append(v)

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    rows = []
    for (td, rd), b in sorted(acc.items()):
        # Offshore at a west-facing coast = wind FROM the land sector. LAX's
        # onshore sea breeze is ~250 deg (features.ONSHORE_DEG); the opposite
        # half-circle (0-135 deg, i.e. N through SE) is the downslope/offshore
        # regime that produces the hot days.
        offshore = ([1.0 if (d <= 135 or d >= 340) else 0.0 for d in b["wdr"]]
                    if b["wdr"] else [])
        rows.append({
            "station": icao, "target_date": td, "run_date": rd,
            "nbs_sol_max": max(b["sol"]) if b["sol"] else None,
            "nbs_sol_mean": _mean(b["sol"]), "nbs_sky_mean": _mean(b["sky"]),
            "nbs_cig_min": min(b["cig"]) if b["cig"] else None,
            "nbs_lcb_min": min(b["lcb"]) if b["lcb"] else None,
            "nbs_offshore": _mean(offshore), "nbs_wsp_mean": _mean(b["wsp"]),
            "nbs_tsd_mean": _mean(b["tsd"]), "nbs_dpt_mean": _mean(b["dpt"]),
            "nbs_n": max((len(v) for v in b.values()), default=0),
            "fetched_at": fetched_at})
    return rows


def fetch_nbs_aux_archive(cfg: AppConfig, start: date, end: date,
                          model: str = "NBS",
                          metric: MetricSpec | None = None) -> list[dict]:
    """Bulk NBS aux fields over a date range. Same endpoint/chunking as
    `fetch_nbm_archive`; separate call so the temperature channel and the
    marine-layer channel can be backfilled and reasoned about independently."""
    metric = metric or get_metric(cfg, "high")
    rows: list[dict] = []
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=182))
        text = get_text(IEM_MOS_ARCHIVE, {
            "station": cfg.station.icao, "model": model,
            "sts": f"{cur.isoformat()}T00:00Z",
            "ets": f"{(stop + timedelta(days=1)).isoformat()}T00:00Z",
            "format": "csv",
        })
        rows.extend(parse_nbs_aux_csv(text, cfg.station.icao, utcnow_iso(),
                                      cfg.station.tz, metric.extremum,
                                      cfg.station.std_utc_offset))
        cur = stop + timedelta(days=1)
    for r in rows:
        r["metric"] = metric.key
    return rows


def store_nbs(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_nbs", rows)


def fetch_nbm_archive(cfg: AppConfig, start: date, end: date,
                      model: str = "NBS",
                      metric: MetricSpec | None = None) -> list[dict]:
    """NBM (National Blend of Models) station bulletins from IEM's bulk archive.

    NBM is the NWS's own operational, station-calibrated blend and is usually the
    strongest single guidance for station temperatures. The Open-Meteo
    `ncep_nbm_conus` channel already in `MODELS` only reaches back to **2024-10**
    (verified: the previous-day archive returns all-null for 2022, 2023 and
    mid-2024), which is why `nbm` had just 664 rows. This IEM bulletin archive is
    populated back to **at least 2021** -- verified live, 644 rows / 28 runs for
    the same June window in each of 2021-2025.

    Stored under a DISTINCT model name (`nbm_nbs`) rather than merged into `nbm`:
    the two come from different sources with different definitions (grid
    interpolation vs station bulletin, daily field vs 3-hourly aggregate) and so
    carry different biases. Merging them would splice two series at 2024-10 and
    silently change what the feature means partway through training.
    """
    metric = metric or get_metric(cfg, "high")
    rows: list[dict] = []
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=182))
        text = get_text(IEM_MOS_ARCHIVE, {
            "station": cfg.station.icao, "model": model,
            "sts": f"{cur.isoformat()}T00:00Z",
            "ets": f"{(stop + timedelta(days=1)).isoformat()}T00:00Z",
            "format": "csv",
        })
        rows.extend(parse_nbs_csv(text, cfg.station.icao, utcnow_iso(),
                                  f"nbm_{model.lower()}", cfg.station.tz,
                                  metric.extremum, cfg.station.std_utc_offset))
        cur = stop + timedelta(days=1)
    for r in rows:
        r["metric"] = metric.key
    return rows


def fetch_mos_archive(cfg: AppConfig, start: date, end: date,
                      model: str = "MEX",
                      metric: MetricSpec | None = None) -> list[dict]:
    # /api/1/mos.json (used by fetch_mos above) takes no sts/ets range at all
    # - it always returns just the latest cycle, which is why an earlier
    # version of this function had to enumerate one runtime per day and
    # stalled across a multi-year backfill (~2000 requests, each retried 3x
    # on 404 with exponential backoff). IEM's bulk MOS *request* service
    # (request/mos.py, format=csv) is different: it accepts a real sts/ets
    # date range and returns every archived cycle in that window in one CSV
    # response. Verified live 2026-07-17 for the station/MEX: a 6-month window
    # (2021-01-01..2021-07-01) returned 5430 rows across 362 distinct
    # runtimes in ~19s, HTTP 200, n_x populated on every row. Chunk by ~6
    # months per request to keep individual responses small.
    metric = metric or get_metric(cfg, "high")
    rows: list[dict] = []
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=182))
        text = get_text(IEM_MOS_ARCHIVE, {
            "station": cfg.station.icao, "model": model,
            "sts": f"{cur.isoformat()}T00:00Z",
            "ets": f"{(stop + timedelta(days=1)).isoformat()}T00:00Z",
            "format": "csv",
        })
        rows.extend(parse_mos_csv(text, cfg.station.icao, utcnow_iso(),
                                  f"mos_{model.lower()}", cfg.station.tz,
                                  metric.extremum, cfg.station.std_utc_offset))
        cur = stop + timedelta(days=1)
    for r in rows:
        r["metric"] = metric.key
    return rows


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecasts", rows)


def fetch_intraday(cfg: AppConfig, metric: MetricSpec | None = None) -> list[dict]:
    """Guidance AS IT STANDS RIGHT NOW, stamped with the true fetch instant.

    Identical query to `fetch_openmeteo`; the difference is entirely in where it
    is stored and that it is called repeatedly through the day. Every call is a
    new snapshot keyed by `fetched_at`, so the table accumulates a record of what
    was knowable at each hour rather than one row per model per day.

    Capture-only. Callers store via `store_intraday`; nothing reads it yet.
    """
    metric = metric or get_metric(cfg, "high")
    st = cfg.station
    today_local = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    payload = get_json(OPEN_METEO, {
        "latitude": st.lat, "longitude": st.lon,
        "daily": metric.openmeteo_var, "temperature_unit": "fahrenheit",
        "timezone": st.tz, "forecast_days": 2,
        "models": ",".join(MODELS.values()),
    })
    # `utcnow_iso()` is evaluated AFTER the request returns, so the stamp is never
    # earlier than the moment the data actually existed here. Stamping before the
    # call would claim knowledge fractionally sooner than we had it -- a small lie
    # in exactly the direction that would flatter a future freshness arm.
    return parse_openmeteo(payload, st.icao, today_local, utcnow_iso(), metric)


def store_intraday(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_intraday", rows)


# The hourly capture's model list is DELIBERATELY SEPARATE from `MODELS`.
#
# `MODELS` maps "gfs" -> `gfs_seamless`, and over CONUS Open-Meteo's *seamless*
# product IS HRRR for the first 48 hours. Verified against the live API
# 2026-07-31: `gfs_seamless` and `gfs_hrrr` returned BYTE-IDENTICAL 48-hour
# series, and in the stored daily table `fc_gfs` == `fc_hrrr` on 3,733 of 3,735
# lead-0/lead-1 pairs since 2024-01-18. They are one channel, not two.
#
# So the hourly capture asks for `gfs_global` -- the real global GFS, which the
# same probe showed is genuinely different (min 83.6F vs the seamless series'
# corrupted 73.1F) -- and keeps `gfs_hrrr` under its own name. Seven DISTINCT
# products, no duplicate.
#
# ⚠️ `gfs_hrrr` STILL BREAKS past HRRR's real horizon (~18-20h): 96.7F at 18:00Z
# then 78.0F at 21:00Z for KMIA, while every other model held 88-96F. That is
# Open-Meteo's data, not a parse bug -- the probe above reproduced it at the API.
# It is NOT filtered here, because the horizon is DERIVABLE from what is already
# stored (`valid_utc` - `fetched_at`), so a later consumer can drop the bad tail
# without this capture silently discarding rows it can never re-fetch.
# Any feature built on this table MUST filter on that horizon.
#
# `MODELS` itself is left ALONE: it feeds the daily `forecasts` table, which is
# a live model input, and changing it is a model change needing a gate.
HOURLY_MODELS = {"ecmwf": "ecmwf_ifs025", "gfs": "gfs_global",
                 "icon": "icon_seamless", "hrrr": "gfs_hrrr",
                 "nbm": "ncep_nbm_conus", "nam": "ncep_nam_conus",
                 "aifs": "ecmwf_aifs025_single"}


def fetch_hourly(cfg: AppConfig) -> list[dict]:
    """The per-hour temperature curve each model is predicting, right now.

    WHY THIS EXISTS. Every other guidance table stores a DAILY min/max, which
    cannot express "tonight will be colder than this morning was" -- and that is
    exactly the case the LOW keeps losing on. On 11.0% of KLAX days the
    climate-day minimum lands at h18+, the same evening, seventeen hours after
    the h4 commit, and nothing in the model's inputs hints at it.

    METRIC-AGNOSTIC BY DESIGN. It is one temperature series per model, from
    which both a morning minimum and an evening minimum can be derived later.
    Splitting it per metric would store the same numbers twice.

    TIMEZONE IS PINNED TO UTC, not the station's local zone, because the rows
    are keyed by `valid_utc` and joined against `obs.valid_utc`. Asking
    Open-Meteo for local time and converting here would add a DST bug for no
    benefit.

    Capture-only; `store_hourly` writes it and nothing reads it.
    """
    st = cfg.station
    payload = get_json(OPEN_METEO, {
        "latitude": st.lat, "longitude": st.lon,
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
        "timezone": "UTC", "forecast_days": 2,
        "models": ",".join(HOURLY_MODELS.values()),
    })
    # Stamped AFTER the response returns, so the record never claims knowledge
    # fractionally earlier than we had it -- the same rule `fetch_intraday`
    # keeps, and for the same reason: a freshness arm measured against an
    # optimistic stamp would flatter itself.
    fetched = utcnow_iso()
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    rows = []
    for key, series in hourly.items():
        if not key.startswith("temperature_2m_") or not isinstance(series, list):
            continue
        model = key[len("temperature_2m_"):]
        for t, v in zip(times, series):
            if v is None:
                continue
            # Open-Meteo returns 'YYYY-MM-DDTHH:MM' with no offset under
            # timezone=UTC; make the stored key explicit so a later
            # `datetime.fromisoformat` is tz-aware and comparable to `obs`.
            rows.append({"station": st.icao,
                         "valid_utc": t if len(t) > 16 else f"{t}:00+00:00",
                         "model": model, "fetched_at": fetched,
                         "temp_f": float(v)})
    return rows


def store_hourly(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_hourly", rows)


def store_ens(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_ens", rows)


ALOFT_MODELS = "gfs_seamless"


def parse_openmeteo_aloft(payload: dict, icao: str, run_date: str,
                          fetched_at: str, std_utc_offset: int,
                          metric_key: str = "high") -> list[dict]:
    """One row per target date: the 850 hPa temp, 2 m temp, and their difference
    (inversion strength) at the 9am-LST cutoff hour. Positive inversion = warm
    cap aloft = marine layer holds (cool high); negative = unstable = clears."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    a850 = hourly.get("temperature_850hPa", [])
    a2 = hourly.get("temperature_2m", [])
    if not times:
        return []
    target_hour = (9 - std_utc_offset) % 24        # 9am LST in UTC; -8 -> 17
    seen = {}
    for t, x, y in zip(times, a850, a2):
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        if dt.hour == target_hour and x is not None and y is not None:
            seen[dt.date().isoformat()] = (float(x), float(y))
    return [{"station": icao, "target_date": d, "run_date": run_date,
             "metric": metric_key, "t850": x, "t2m": y, "inversion": x - y,
             "fetched_at": fetched_at} for d, (x, y) in seen.items()]


def fetch_openmeteo_aloft(cfg: AppConfig, past_days: int = 2,
                          forecast_days: int = 2) -> list[dict]:
    """Fetch 850 hPa + 2 m temps and summarize the morning inversion per day.
    850 hPa is live-only for KLAX (~2 weeks back), so this is a forward-
    accumulation capture; callers store via store_aloft."""
    st = cfg.station
    payload = get_json(OPEN_METEO, {
        "latitude": st.lat, "longitude": st.lon,
        "hourly": "temperature_850hPa,temperature_2m",
        "temperature_unit": "fahrenheit", "timezone": "GMT",
        "past_days": past_days, "forecast_days": forecast_days,
        "models": ALOFT_MODELS,
    })
    run_date = datetime.now(ZoneInfo(st.tz)).date().isoformat()
    return parse_openmeteo_aloft(payload, st.icao, run_date, utcnow_iso(),
                                 st.std_utc_offset, "high")


def store_aloft(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "forecast_aloft", rows)
