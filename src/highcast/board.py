"""The Station Table — raw physical state, decoded, with the source kept.

WHAT THIS IS FOR. The model hands Julian a probability, which he cannot argue
with: 81% is 81%. This hands him the facts the probability is made of, so a read
can be overruled. On 2026-07-29 the KLAX LOW committed 67-68 at 81% and lost;
"min unchanged 13.1h, pressure falling 1.3 hPa, wind borderline onshore,
dewpoint depression 11F" is a picture he could have acted against.

WHY DECODED RATHER THAN RAW. Raw METAR is the right SOURCE and the wrong
INTERFACE. In

    METAR KLAX 292353Z 24012KT 10SM FEW010 24/18 A2982 RMK AO2 SLP097
          T02440183 10267 20239 58013 $

the two numbers that settle the bet are `10267` and `20239`, sitting undelimited
beside the pressure group `58013`, all three five digits. Using them by hand means
spotting them among lookalikes, knowing `1xxxx` is max and `2xxxx` is min, knowing
digit 2 is the sign and 3-5 are tenths of Celsius, converting to Fahrenheit, AND
knowing whether that report's six-hour window is admissible for this station's
climate day. The last step is station-dependent -- KLAX rejects the 12Z report,
KMIA rejects 06Z -- and it is where this project's hardest bug lived. So the board
decodes, and `--raw` keeps the source line underneath for when a number is doubted.

FREEZE STATUS: display only. Reads `obs`, `latentmax`, the `gating` constants,
and (for the GATES line) `gating.decision_gates` itself -- the same aggregator
the CLI and dashboard must agree with, never a re-implementation of its gates.
It computes no prediction, writes nothing, and must never be given a vote in
`staking.evaluate` or `appbet`.
"""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

from highcast import gating, latentmax
from highcast.config import AppConfig
from highcast.features import ONSHORE_DEG
from highcast.fetchers.metar import PRES_TEND_SHAPE
from highcast.gating import (EXTREMUM_STILL_FUTURE, HOUR_ACCURACY,
                             MIN_DECISION_HOUR, hour_calib)
from highcast.metric import get_metric

# Cloud-cover codes, worst-case (most cloud) last. `obs.skyc` stores the LAST
# token in the report, which is the highest layer -- so a "FEW" here does not
# rule out a lower deck, and the board says the code rather than interpreting it.
SKY_WORDS = {"CLR": "clear", "SKC": "clear", "FEW": "few", "SCT": "scattered",
             "BKN": "broken", "OVC": "overcast", "VV": "obscured"}

# How long a running extremum must sit unchanged before it is worth remarking on.
# Their board's "held 460m" on the day we lost was 7.7h; ours was 13.1h. Below
# ~3h a flat reading is just the ordinary overnight plateau.
STALL_NOTE_HOURS = 3.0

# ── CLIMATOLOGY BANDS ───────────────────────────────────────────────────────
#
# "dewpoint depression 5.9F" is unreadable without knowing what this station
# does on an ordinary night in July. The number alone forces Julian to supply
# the physics; the PERCENTILE supplies it for him -- "5.9F, drier than 78% of
# comparable hours here" is a statement he can act on cold.
#
# CONDITIONED ON HOUR AND SEASON, because neither alone is enough: KLAX's
# dewpoint depression at 4am in January and 2pm in July belong to different
# populations, and pooling them makes every reading look "typical". The window
# is +/-1 hour and +/-30 days across all years on record.
#
# WHAT IT DELIBERATELY DOES NOT SAY: which direction is good for the bet. That
# claim is not measured, so the band reports UNUSUAL vs TYPICAL only. Reading a
# valence into it would be inventing physics on the dashboard, which is the
# error class this project has had to unwind twice.
CLIMO_HOUR_WINDOW = 1
CLIMO_DOY_WINDOW = 30
CLIMO_MIN_SAMPLES = 120
# A leap year, so 29 February has a day-of-year and the seasonal window is
# computed the same way on both sides of the comparison.
_DOY_REF = 2000

# Percentiles drawn: the track spans p10-p90 and the shaded arc is p25-p75, so
# the strip reads like an instrument's green arc rather than a generic bar.
CLIMO_PCTS = (10, 25, 50, 75, 90)


def admissible_group_reports(cfg: AppConfig, day: dt.date) -> list[dict]:
    """The 6-hourly synoptic reports whose window lies wholly inside this climate
    day, with the instant each one FILES and its station-local clock time.

    A report at synoptic hour R covers `(R-6h, R]` and is transmitted in the
    `(R-1):53Z` observation. A report whose window straddles the climate-day
    boundary must be REJECTED -- folding it in would import temperatures from the
    previous day, which is the boundary error §19 of the competitor doc suspects in
    their own series, and which `latentmax` already refuses.

    Derived from `std_utc_offset`, never hardcoded, because WHICH report is
    rejected differs by station: KLAX (day 08Z->08Z) drops the 12Z report, KMIA
    (05Z->05Z) drops the 06Z. Verified against the competitor doc's independently
    observed posting times on both stations -- KLAX's afternoon max files 16:53
    local against their "Pacific ~4:50-5:00 PM", KMIA's 19:53 against their
    "Eastern ~7:50-7:55 PM".
    """
    start, stop = latentmax.climate_day_bounds(day, cfg.station.std_utc_offset)
    tz = ZoneInfo(cfg.station.tz)
    out = []
    # Walk every synoptic report instant that could serve this day, including the
    # one just past the end (its window may still close inside the day).
    cursor = start - dt.timedelta(hours=6)
    while cursor <= stop + dt.timedelta(hours=6):
        if cursor.hour % 6 == 0:
            win_start, win_end = cursor - dt.timedelta(hours=6), cursor
            files_at = cursor - dt.timedelta(minutes=7)   # (R-1):53Z
            if win_start >= start and win_end <= stop:
                out.append({
                    "report_hour": cursor.hour,
                    "window": f"{win_start.hour:02d}-{win_end.hour:02d}Z",
                    "files_utc": files_at,
                    "files_local": files_at.astimezone(tz).strftime("%H:%M"),
                })
        cursor += dt.timedelta(hours=1)
    return out


def next_group(cfg: AppConfig, day: dt.date, now: dt.datetime) -> dict | None:
    """The next admissible 6-hour group still to file, or None if all are in.

    This is the actionable number: the running extremum cannot move on a group
    that has not been transmitted yet, so "check back at 22:53" is a better
    instruction than any probability.
    """
    for g in admissible_group_reports(cfg, day):
        if g["files_utc"] > now:
            return {**g, "hours_away": (g["files_utc"] - now).total_seconds() / 3600}
    return None


def latest_group(conn, cfg: AppConfig, day: dt.date, extremum: str,
                 now: dt.datetime) -> dict | None:
    """The most recent admissible 6-hour group ALREADY on record for this day.

    Shown because these are the numbers that settle the bet -- the hourly
    snapshots only sample the extreme and miss it (41% of the time for the HIGH),
    while `1sTTT`/`2sTTT` are derived from continuous sampling. Displaying the
    group makes visible what `latentmax` is already using.
    """
    col = "six_min_c" if extremum == "min" else "six_max_c"
    admissible = {g["report_hour"]: g for g in admissible_group_reports(cfg, day)}
    if not admissible:
        return None
    rows = conn.execute(
        f"""SELECT valid_utc, {col} AS c FROM obs
            WHERE station=? AND source='metar' AND {col} IS NOT NULL
              AND valid_utc <= ? ORDER BY valid_utc DESC LIMIT 12""",
        (cfg.station.icao, now.isoformat())).fetchall()
    for r in rows:
        v = dt.datetime.fromisoformat(r["valid_utc"])
        # the (R-1):53Z observation belongs to synoptic hour R
        report_hour = (v + dt.timedelta(minutes=7)).hour
        g = admissible.get(report_hour)
        if g is None or abs((v - g["files_utc"]).total_seconds()) > 3600:
            continue
        return {"report_hour": report_hour, "window": g["window"],
                "files_local": g["files_local"], "celsius": r["c"],
                "fahrenheit": r["c"] * 9.0 / 5 + 32}
    return None


def _stall_hours(conn, cfg: AppConfig, day: dt.date, extremum: str,
                 now: dt.datetime) -> float | None:
    """Hours since the running extremum last moved.

    Uses SPOT readings only, deliberately. The six-hour groups arrive in lumps at
    synoptic times, so including them would make the stall clock jump backwards
    whenever a group lands -- which reads as "the extremum just moved" when in
    fact a continuously-sampled window was merely published. The stall is a
    statement about the observation trace, not about settlement.
    """
    start, stop = latentmax.climate_day_bounds(day, cfg.station.std_utc_offset)
    rows = conn.execute(
        """SELECT valid_utc, COALESCE(tmpf_precise, tmpf) AS v FROM obs
           WHERE station=? AND valid_utc>=? AND valid_utc<? AND source='metar'
             AND COALESCE(tmpf_precise, tmpf) IS NOT NULL
           ORDER BY valid_utc""",
        (cfg.station.icao, start.isoformat(), min(stop, now).isoformat())
    ).fetchall()
    vals = [r["v"] for r in rows]
    if not vals:
        return None
    best = min(vals) if extremum == "min" else max(vals)
    for r in rows:
        if abs(r["v"] - best) < 0.05:
            last_set = dt.datetime.fromisoformat(r["valid_utc"])
    return (now - last_set).total_seconds() / 3600.0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile. Hand-rolled to keep `board` free of numpy:
    this module is imported by the web app on every request and by the CLI on a
    phone over ssh, and the whole point of it is to be cheap."""
    if not sorted_vals:
        raise ValueError("empty")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def climatology_bands(conn, cfg: AppConfig, now: dt.datetime | None = None,
                      columns: tuple[str, ...] = ("dwpf", "depression",
                                                  "sknt", "slp_hpa")) -> dict:
    """Where each reading sits in THIS station's own history at this hour and
    season. Returns {column: {p10..p90, n}}, or omits a column with too few
    comparable hours rather than drawing a band nobody should trust.

    ONE TABLE SCAN for all columns. `obs` is indexed on (station, valid_utc)
    only, so filtering by hour-of-day cannot use an index -- doing it per column
    would mean four scans of ~57k rows per market per request. Fetch once,
    bucket in Python.

    HOUR MATCHING IS DONE IN UTC, deliberately. SQLite has no timezone support,
    so the alternative is decoding every stored stamp in Python. The target UTC
    hour is derived from the CURRENT local hour, and every comparison day lies
    within +/-30 days, so all of them share today's DST state -- except across a
    transition, where part of the window is off by one hour. That is a one-hour
    smear on a band already +/-1 hour wide, twice a year.

    `depression` is not a stored column; it is computed here so the band matches
    the figure the panel prints beside it.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    # Day-of-year is measured on a fixed LEAP reference year for both today and
    # every historical row, so 29 February is representable and the two sides
    # cannot drift by a day across leap boundaries.
    ltoday = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    doy = dt.date(_DOY_REF, ltoday.month, ltoday.day).timetuple().tm_yday
    rows = conn.execute(
        """SELECT valid_utc, tmpf_precise, tmpf, dwpf, sknt, slp_hpa
           FROM obs WHERE station=? AND source='metar'""",
        (cfg.station.icao,)).fetchall()
    pools: dict[str, list[float]] = {c: [] for c in columns}
    for r in rows:
        v = r["valid_utc"]
        # Stamps are ISO 'YYYY-MM-DDTHH:MM:SS+00:00'; slicing beats parsing
        # 57k datetimes on a request path.
        try:
            hh = int(v[11:13])
            mo, dd = int(v[5:7]), int(v[8:10])
        except (TypeError, ValueError):
            continue
        if min((hh - now.hour) % 24, (now.hour - hh) % 24) > CLIMO_HOUR_WINDOW:
            continue
        try:
            rdoy = dt.date(_DOY_REF, mo, dd).timetuple().tm_yday
        except ValueError:
            continue
        if min((rdoy - doy) % 366, (doy - rdoy) % 366) > CLIMO_DOY_WINDOW:
            continue
        t = r["tmpf_precise"] if r["tmpf_precise"] is not None else r["tmpf"]
        for c in columns:
            if c == "depression":
                val = None if (t is None or r["dwpf"] is None) else t - r["dwpf"]
            else:
                val = r[c] if c in r.keys() else None
            if val is not None:
                pools[c].append(float(val))
    out = {}
    for c, vals in pools.items():
        if len(vals) < CLIMO_MIN_SAMPLES:
            continue
        vals.sort()
        band = {f"p{p}": _percentile(vals, p) for p in CLIMO_PCTS}
        out[c] = {**band, "n": len(vals)}
    return out


def band_position(band: dict | None, value: float | None) -> dict | None:
    """Today's reading placed on its band: a 0-1 track offset plus a plain-word
    verdict. `None` whenever either side is missing, so the caller draws nothing
    rather than a strip anchored on a guess.

    `pct` is an APPROXIMATION interpolated from the five stored percentiles, not
    a true rank -- storing the whole sorted pool to get an exact one would cost
    far more than the extra precision is worth on a 200px strip.
    """
    if band is None or value is None:
        return None
    lo, hi = band["p10"], band["p90"]
    span = hi - lo
    # A degenerate band (every comparable hour identical) has no inside or
    # outside; report the centre and no verdict rather than dividing by zero.
    if span <= 0:
        return {"offset": 0.5, "pct": 50.0, "verdict": "typical",
                "clamped": False}
    offset = (value - lo) / span
    clamped = offset < 0 or offset > 1
    knots = [(band["p10"], 10.0), (band["p25"], 25.0), (band["p50"], 50.0),
             (band["p75"], 75.0), (band["p90"], 90.0)]
    pct = knots[0][1] if value <= knots[0][0] else knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= value <= x1:
            pct = y0 if x1 == x0 else y0 + (y1 - y0) * (value - x0) / (x1 - x0)
            break
    if value < band["p10"] or value > band["p90"]:
        verdict = "unusual"
    elif value < band["p25"] or value > band["p75"]:
        verdict = "uncommon"
    else:
        verdict = "typical"
    return {"offset": min(1.0, max(0.0, offset)), "pct": pct,
            "verdict": verdict, "clamped": clamped,
            "arc_lo": (band["p25"] - lo) / span,
            "arc_hi": (band["p75"] - lo) / span}


def reliability_table(highlight: tuple[str, str] | None = None) -> list[dict]:
    """Every tradeable market ranked by MEASURED accuracy at its commit hour.

    Answers the question "which of these do I actually trust", which is the one
    Julian has asked under several framings ("which is more stable to trade").
    It is a STATIC per-market fact, not a daily signal -- the same answer every
    day -- so it belongs beside the physical state, not in a pick.

    THREE NUMBERS, AND THEY MUST TRAVEL TOGETHER:

      hit    measured modal-bucket accuracy at the commit hour, parity-averaged
      calib  stated probability MINUS realised hit. Positive = the model
             OVERSTATES and its number needs shading before you believe it.
      ahead  share of the extremum still in the future at that hour.

    Accuracy alone flatters a market: it climbs all day purely because the answer
    stops being unknown, which is the trap the 98.8% hours exposed. A market that
    is 62% right with 85% still ahead is doing something much harder than one
    that is 62% right with 10% ahead. Showing hit without `ahead` is how a
    transcription hour gets mistaken for a forecasting one.

    WHAT IT SAYS TODAY (rewritten 2026-08-01 — THE RANKING INVERTED when the
    commit hours moved later, and the old text now says the opposite of the
    truth):

        market      hour   hit     calib      still ahead
        KMIA HIGH   h14   79.7%   -0.0281       49.7%
        KLAX HIGH   h13   69.5%   +0.0145       51.7%
        KLAX LOW    h6    68.2%   -0.0097       49.9%
        KMIA LOW    h7    66.3%   -0.0175       41.7%

    **KMIA HIGH is now the most trustworthy market on the board** -- clearly the
    highest accuracy, and it UNDER-states its own bucket (a negative calib is the
    safe direction to be wrong in). It is also the only market whose >=80% band is
    measured well enough to act on (n=244).

    KLAX LOW was the most trustworthy market at the OLD hours and this docstring
    said so; at h6 it is now third. KLAX HIGH was the worst on both counts and is
    now second on accuracy -- but it is the ONLY market left that still
    OVERSTATES (+0.0145), and its >=80% band overstates by a further +0.109, so
    it remains the market to trust least at high confidence.

    None of this is a model improvement: it is the same models asked at an hour
    where roughly half the extremum has already happened.
    """
    out = []
    for station, metrics in MIN_DECISION_HOUR.items():
        for metric, hour in metrics.items():
            acc = HOUR_ACCURACY.get(station, {}).get(metric, {}).get(hour)
            out.append({
                "station": station, "metric": metric, "commit_hour": hour,
                "hit": acc[1] if acc else None,
                "mae": acc[0] if acc else None,
                "calib": hour_calib(station, metric, hour),
                "still_future":
                    EXTREMUM_STILL_FUTURE.get(station, {})
                    .get(metric, {}).get(hour),
                "is_current": highlight == (station, metric),
            })
    # Most accurate first. Unmeasured markets sort last rather than being hidden:
    # a missing row is a data gap worth seeing, same contract as `cities.survey`.
    out.sort(key=lambda r: (r["hit"] is None, -(r["hit"] or 0)))
    return out


def row(conn, cfg: AppConfig, metric: str,
        now: dt.datetime | None = None) -> dict:
    """One market's physical state. Pure read; no network, no model."""
    now = now or dt.datetime.now(dt.timezone.utc)
    tz = ZoneInfo(cfg.station.tz)
    icao = cfg.station.icao
    spec = get_metric(cfg, metric)
    local = now.astimezone(tz)
    day = local.date()
    # The climate day may still be open on the PREVIOUS calendar date when the
    # station's std offset pushes the boundary past local midnight -- KLAX's day
    # runs 01:00->01:00 under PDT. Fall back one day if today's has not begun.
    start, _ = latentmax.climate_day_bounds(day, cfg.station.std_utc_offset)
    if now < start:
        day = day - dt.timedelta(days=1)

    o = conn.execute(
        """SELECT valid_utc, tmpf_precise, tmpf, dwpf, drct, sknt, skyc,
                  six_max_c, six_min_c, slp_hpa, pres_tend_hpa, pres_tend_code,
                  raw
           FROM obs WHERE station=? AND source='metar'
           ORDER BY valid_utc DESC LIMIT 1""", (icao,)).fetchone()
    iv = latentmax.visible_running_extremum(conn, cfg, day, spec.extremum)
    commit_h = MIN_DECISION_HOUR.get(icao, {}).get(metric)
    acc = HOUR_ACCURACY.get(icao, {}).get(metric, {}).get(commit_h)
    fut = EXTREMUM_STILL_FUTURE.get(icao, {}).get(metric, {}).get(commit_h)

    # ONE verdict, from `gating.decision_gates`, so the terminal and the
    # dashboard can never disagree about whether today is actionable. `hour`
    # is always the market's COMMIT hour (`MIN_DECISION_HOUR`), never the
    # clock hour `row` happens to be called at -- the confidence bands and
    # floor are keyed to that one measured hour, so evaluating at any other
    # hour would just read as an unmeasured market. `p_model` is the current
    # top pick's own stated probability (`gating.committed_state`), a
    # DB-only read matching this function's "no network, no model" contract.
    # None when the station/metric has no measured commit hour at all --
    # `decision_gates` requires an int hour and there is no honest one to
    # substitute.
    gates = None
    if commit_h is not None:
        p_model = gating.committed_state(conn, cfg, day.isoformat(), metric)["prob"]
        gates = gating.decision_gates(conn, cfg, metric=metric,
                                      target=day.isoformat(), hour=commit_h,
                                      p_model=p_model)

    t = None if o is None else (o["tmpf_precise"] if o["tmpf_precise"] is not None
                                else o["tmpf"])
    dwpf = None if o is None else o["dwpf"]
    # Pressure needs a fallback on BOTH fields. The 3-hour tendency group files
    # only at synoptic 3-hourly times (~28% of reports) and SLP is absent from
    # ~15% of them, so keying either to the LATEST report drops the whole pressure
    # line for no good reason -- pressure changes slowly, and a reading an hour old
    # is worth far more than a blank. Each is stamped with its own age.
    slp = conn.execute(
        """SELECT valid_utc, slp_hpa FROM obs
           WHERE station=? AND source='metar' AND slp_hpa IS NOT NULL
             AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (icao, now.isoformat())).fetchone()
    slp_age = (None if slp is None else
               (now - dt.datetime.fromisoformat(slp["valid_utc"])
                ).total_seconds() / 3600.0)
    tend = conn.execute(
        """SELECT valid_utc, pres_tend_hpa, pres_tend_code FROM obs
           WHERE station=? AND source='metar' AND pres_tend_hpa IS NOT NULL
             AND valid_utc <= ?
           ORDER BY valid_utc DESC LIMIT 1""",
        (icao, now.isoformat())).fetchone()
    tend_c = None if tend is None else tend["pres_tend_code"]
    tend_age = (None if tend is None else
                (now - dt.datetime.fromisoformat(tend["valid_utc"])
                 ).total_seconds() / 3600.0)
    return {
        "station": icao, "metric": metric, "target_date": day.isoformat(),
        "local_hour": local.hour, "local_time": local.strftime("%H:%M"),
        "obs_utc": None if o is None else o["valid_utc"],
        "tmpf": t, "dwpf": dwpf,
        "depression": None if (t is None or dwpf is None) else t - dwpf,
        "drct": None if o is None else o["drct"],
        "sknt": None if o is None else o["sknt"],
        "sky": None if o is None else SKY_WORDS.get(
            (o["skyc"] or "")[:3], o["skyc"]),
        # The MODEL's onshore feature is a cosine projection of wind onto LAX's
        # sea-breeze bearing (features.ONSHORE_DEG = 250), not a boolean. Report
        # the same quantity so the board and the model agree, and report it ONLY
        # for the station it was derived for -- 250 degrees is meaningless at
        # KMIA, and transplanting station constants is the error class this
        # project has had to unwind twice.
        "onshore_component": (
            None if (o is None or o["drct"] is None or o["sknt"] is None
                     or icao != "KLAX")
            else o["sknt"] * math.cos(math.radians(o["drct"] - ONSHORE_DEG))),
        "onshore_deg": ONSHORE_DEG if icao == "KLAX" else None,
        "slp_hpa": None if slp is None else slp["slp_hpa"],
        "slp_age_h": slp_age,
        "pres_tend_hpa": None if tend is None else tend["pres_tend_hpa"],
        "pres_tend_shape": None if tend_c is None else PRES_TEND_SHAPE.get(tend_c),
        "pres_tend_age_h": tend_age,
        "running_lo": None if iv is None else iv[0],
        "running_hi": None if iv is None else iv[1],
        "settles_to": None if iv is None else round(iv[1] if spec.extremum == "min"
                                                    else iv[0]),
        "stall_hours": _stall_hours(conn, cfg, day, spec.extremum, now),
        "latest_group": latest_group(conn, cfg, day, spec.extremum, now),
        "next_group": next_group(cfg, day, now),
        "commit_hour": commit_h,
        "commit_hit": None if acc is None else acc[1],
        "commit_still_future": fut,
        "gates": gates,
        "raw": None if o is None else o["raw"],
    }


# The evidence rows, in reading order, with the `obs`-derived value each one
# shows and the climatology pool it is placed against. Ordered by how directly
# each bears on a temperature extreme: moisture first (a saturated column cannot
# cool or warm freely), then the sea breeze, then the synoptic pressure field.
# `band_col` is None where no honest pool exists -- sky is a category, not a
# quantity, and interpolating a percentile through cloud codes would be a number
# with no meaning behind it.
EVIDENCE_ROWS = (
    {"key": "depression", "label": "dewpoint depression", "band_col": "depression",
     "unit": "°F", "digits": 1,
     "hint": "how far the air sits above saturation"},
    {"key": "dwpf", "label": "dewpoint", "band_col": "dwpf",
     "unit": "°F", "digits": 0,
     "hint": "the moisture the column is carrying"},
    {"key": "sknt", "label": "wind speed", "band_col": "sknt",
     "unit": "kt", "digits": 0,
     "hint": "mixing strength"},
    {"key": "slp_hpa", "label": "sea-level pressure", "band_col": "slp_hpa",
     "unit": " hPa", "digits": 1,
     "hint": "where this sits in the synoptic pattern"},
)


def evidence(r: dict, bands: dict | None) -> list[dict]:
    """The physical readings, each placed on its own climatology.

    Assembled here rather than in the browser so the placement is unit-testable
    and so the terminal and the dashboard cannot drift into showing different
    verdicts for the same observation -- `render` and `renderBoard` both consume
    this list.

    Rows whose value is missing are DROPPED, not shown blank. A row reading "—"
    invites the eye to treat absence as a reading; the pressure fields in
    particular are absent from most reports by design (the 3-hour tendency group
    files only at synoptic times) and would leave permanent gaps.
    """
    out = []
    for spec in EVIDENCE_ROWS:
        v = r.get(spec["key"])
        if v is None:
            continue
        pos = band_position((bands or {}).get(spec["band_col"]), v)
        out.append({
            "key": spec["key"], "label": spec["label"], "hint": spec["hint"],
            "value": v,
            "text": f"{v:.{spec['digits']}f}{spec['unit']}",
            "band": pos,
        })
    return out


def render(rows: list[dict], show_raw: bool = False) -> str:
    """Fixed-width text, one block per market. Terminal-first on purpose: this is
    read at 02:00 Manila on a phone over ssh as often as on the dashboard."""
    out = []
    for r in rows:
        head = (f"== {r['station']}  {r['metric'].upper()}  target "
                f"{r['target_date']}  (local {r['local_time']}) ==")
        out.append(head)
        # THE fail-closed verdict from `gating.decision_gates` -- ahead of
        # everything else, because it answers whether the desk may act on
        # this market AT ALL. Named refusals, not a bare no: a desk shown one
        # reason fixes it and is surprised by the next. Absent (rather than
        # printed as "unmeasured") for a station/metric with no measured
        # commit hour at all -- see `row`.
        gv = r.get("gates")
        if gv is not None:
            out.append("   GATES: clear" if gv.ok else
                       f"   GATES: REFUSE ({', '.join(gv.refusals)})")
        # The day's verdict goes ABOVE the readings: "is this worth paying for"
        # comes before "what is it". Only the measured markets get one; the rest
        # print the reason there is none rather than an empty line.
        pf = r.get("preflight")
        if pf:
            LABEL = {"CLEAR": "CLEAR TO TRADE", "FAIR": "ORDINARY DAY",
                     "AVOID": "SIT THIS ONE OUT"}
            name = LABEL.get(pf.get("band")) or (
                "NO READ YET" if pf.get("status") == "pending" else "NO DAILY READ")
            gap = ("" if pf.get("gap") is None else
                   f"   (ratchet - guidance {pf['gap']:+.2f}F)")
            out.append(f"   >> {name}{gap}")
            if pf.get("note"):
                out.append(f"      {pf['note']}")
        if r["tmpf"] is None:
            out.append("   no observations")
            continue
        out.append(
            f"   temp {r['tmpf']:.1f}F   dewpt "
            f"{'--' if r['dwpf'] is None else format(r['dwpf'], '.0f') + 'F'}"
            f"   depression "
            f"{'--' if r['depression'] is None else format(r['depression'], '.1f') + 'F'}"
            f"   sky {r['sky'] or '--'}")
        # The climatology placement, when the caller supplied bands. Same list
        # the dashboard draws as strips -- rendered here as the percentile and
        # the word, because a 40-column ssh session cannot draw an arc.
        for e in r.get("evidence") or []:
            if not e.get("band"):
                continue
            out.append(f"     {e['label']:<22}{e['text']:>10}   "
                       f"p{e['band']['pct']:.0f} of this hour/season"
                       f"   {e['band']['verdict']}")
        # Direction is absent on calm/variable reports while SPEED is still
        # present. Printing a bare "--" contradicted the evidence row above,
        # which lists the speed -- so say which half is missing.
        if r["drct"] is None:
            wind = ("--" if r["sknt"] is None
                    else f"{r['sknt']:.0f}kt, direction not reported")
        else:
            wind = f"{r['drct']:.0f}deg @ {r['sknt']:.0f}kt"
        # Raw degrees ALWAYS, projection only where it is calibrated. A boolean
        # verdict would hide that 240deg sits 10deg off LAX's 250deg bearing --
        # the sort of borderline a human catches and a flag does not.
        onsh = ("" if r["onshore_component"] is None else
                f"   onshore proj {r['onshore_component']:+.1f}kt "
                f"(vs {r['onshore_deg']:.0f}deg)")
        out.append(f"   wind {wind}{onsh}")
        if r["slp_hpa"] is not None:
            if r["pres_tend_hpa"] is None:
                tend = "none on record"
            else:
                age = ("" if not r["pres_tend_age_h"]
                       else f", {r['pres_tend_age_h']:.0f}h old")
                tend = (f"{r['pres_tend_hpa']:+.1f} hPa/3h "
                        f"({r['pres_tend_shape']}{age})")
            sage = ("" if not r.get("slp_age_h") or r["slp_age_h"] < 1.0
                    else f" ({r['slp_age_h']:.0f}h old)")
            out.append(f"   SLP {r['slp_hpa']:.1f} hPa{sage}   tendency {tend}")
        if r["running_lo"] is not None:
            stall = ("" if r["stall_hours"] is None else
                     f"   unchanged {r['stall_hours']:.1f}h"
                     + ("  <-- STALLED" if r["stall_hours"] >= STALL_NOTE_HOURS
                        else ""))
            out.append(f"   running {r['metric']} "
                       f"[{r['running_lo']:.2f}, {r['running_hi']:.2f}]"
                       f" -> settles {r['settles_to']}{stall}")
        # The 6-hour groups: what settles the bet, and when the next one lands.
        g = r.get("latest_group")
        if g:
            out.append(f"   6h {r['metric']} group ({g['window']}, filed "
                       f"{g['files_local']}): {g['celsius']:.1f}C = "
                       f"{g['fahrenheit']:.2f}F")
        n = r.get("next_group")
        if n:
            out.append(f"   next group {n['window']} files {n['files_local']} "
                       f"local (in {n['hours_away']:.1f}h) <-- can still move it")
        elif r.get("latest_group"):
            out.append("   all admissible groups are in — the extremum is settled")
        if r["commit_hour"] is not None:
            hit = ("--" if r["commit_hit"] is None
                   else f"{r['commit_hit']:.1%}")
            fut = ("--" if r["commit_still_future"] is None
                   else f"{r['commit_still_future']:.0%}")
            out.append(f"   commit h{r['commit_hour']}: bucket right {hit}, "
                       f"still a forecast {fut}")
        if show_raw:
            out.append(f"   raw: {r['raw']}")
        out.append("")
    return "\n".join(out)


def render_reliability(rows: list[dict]) -> str:
    """The ranked table. `hit` and `still ahead` are printed side by side on
    purpose -- see `reliability_table` for why neither means much alone."""
    out = ["== WHICH MARKET TO TRUST (measured, static) ==",
           f"   {'market':12s}{'commit':>7}{'bucket right':>14}"
           f"{'overconfident by':>18}{'still ahead':>13}"]
    for r in rows:
        hit = "--" if r["hit"] is None else f"{r['hit']:.1%}"
        cal = "--" if r["calib"] is None else f"{r['calib']:+.1%}"
        fut = "--" if r["still_future"] is None else f"{r['still_future']:.0%}"
        mark = "  <- you" if r["is_current"] else ""
        out.append(f"   {r['station'] + ' ' + r['metric']:12s}"
                   f"{'h' + str(r['commit_hour']):>7}{hit:>14}{cal:>18}"
                   f"{fut:>13}{mark}")
    out.append("   accuracy alone flatters a market: it rises all day only")
    out.append("   because the answer stops being unknown. Read it with "
               "'still ahead'.")
    return "\n".join(out)
