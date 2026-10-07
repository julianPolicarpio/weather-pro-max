from collections import Counter
from datetime import date, datetime, timedelta, timezone

from highcast.config import AppConfig
from highcast.http import get_text
from highcast.metric import get_metric
from highcast.quantize import candidates_from_reported_c
from highcast.timeutil import c_to_f, climate_day_bounds, f_to_c, round_half_up, utcnow_iso

IEM_1MIN = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
METAR_TOL = 0.1  # deg F tolerance on T-group precision


def _as_utc(ts: str) -> datetime:
    """Parse an ISO timestamp; a missing offset means UTC (every timestamp in
    `obs` is UTC by construction, see timeutil.norm_valid)."""
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _six_hour_window(valid_utc: str) -> tuple[datetime, datetime]:
    """The (start, end] UTC window a report's 6-hourly remark group covers.

    An ASOS synoptic report is filed a few minutes before the hour it belongs
    to (KLAX files at :53), so a report at HH:53Z carries synoptic hour HH+1
    and its 1sTTT/2sTTT groups summarise the 6 hours ending there. A report
    landing exactly on the hour is already at its synoptic hour. Pure: no DB,
    no config -- the guard is checked against the climate-day bounds by the
    caller.
    """
    dt = _as_utc(valid_utc)
    end = dt.replace(minute=0, second=0, microsecond=0)
    if dt != end:
        end += timedelta(hours=1)
    return end - timedelta(hours=6), end


def visible_running_extremum(conn, cfg: AppConfig, day: date, extremum: str,
                              asof_utc: str | None = None):
    """Running MAX (extremum="max") or MIN (extremum="min") of the interval
    of true temperatures consistent with everything observed so far in the
    climate day. The settlement high can't be below what's already been
    observed; symmetrically the settlement low can't be above it.

    METAR rows contribute two kinds of evidence: the hourly spot reading, and
    (on synoptic reports) the 6-hourly extremum group, which is derived from
    continuous sampling and so can legitimately sit outside every spot
    reading. A 6-hourly group is only admissible when its whole 6-hour window
    lies inside [climate day start, stop) -- at KLAX the 12Z report's 06-12Z
    window straddles the 08:00Z climate-day start, and folding it in would
    import a temperature belonging to the PREVIOUS climate day.
    """
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    stop = min(end.isoformat(), asof_utc) if asof_utc else end.isoformat()
    stop_dt = _as_utc(stop)
    iv_combine = _max_iv if extremum == "max" else _min_iv
    six_col = "six_max_c" if extremum == "max" else "six_min_c"
    running: tuple[float, float] | None = None
    for r in conn.execute(
            """SELECT source, valid_utc, tmpf, tmpf_precise, six_max_c, six_min_c
               FROM obs
               WHERE station=? AND valid_utc >= ? AND valid_utc < ?""",
            (cfg.station.icao, start.isoformat(), stop)):
        cands: tuple[float, float] | None = None
        if r["source"] == "metar":
            if r["tmpf_precise"] is not None:
                cands = (r["tmpf_precise"] - METAR_TOL, r["tmpf_precise"] + METAR_TOL)
            if r[six_col] is not None:
                w_start, w_end = _six_hour_window(r["valid_utc"])
                if w_start >= start and w_end <= stop_dt:
                    v = c_to_f(r[six_col])
                    cands = iv_combine(cands, (v - METAR_TOL, v + METAR_TOL))
        elif r["source"] in ("syn5", "asos1") and r["tmpf"] is not None:
            f = r["tmpf"]
            c = f_to_c(f)
            if abs(c - round(c)) < 0.02:      # whole-degree C -> quantized feed
                cf = candidates_from_reported_c(int(round(c)))
                cands = (float(min(cf)), float(max(cf)))
            else:                              # decimal reading (e.g. 1-min data)
                cands = (f - 0.5, f + 0.5)
        if cands:
            running = iv_combine(running, cands)
    return running


def six_hour_coverage(conn, cfg: AppConfig, day: date, extremum: str,
                      asof_utc: str | None = None) -> dict:
    """Which stretches of a climate day NO admissible 6-hour group covers.

    REPORT ONLY. It changes nothing about what `visible_running_extremum`
    accepts — the admissibility rule there is correct and this must not relax
    it. This answers a different question: *where are we relying on hourly
    spot readings alone?*

    WHY IT EXISTS, 2026-08-06. KMIA settled a low of 75F at 01:51 local. The
    01:53 METAR — two minutes later — read 78.1F, the model predicted 78-79 on
    exactly that evidence, and the market priced the cold scenario while we did
    not. Two things defeated the 2sTTT backstop that day: the 05:53Z report
    filed no group at all (~3% of synoptic reports omit one), and the group
    that would have covered the event straddles the climate-day start and is
    therefore inadmissible anyway.

    TWO GAPS EXIST ON EVERY DAY, even a perfect one, and that is not a defect —
    it falls out of the boundary rule. The groups spanning the day's start and
    end necessarily straddle it. For a station filing at :53 that leaves ~53
    minutes uncovered at the open and up to ~5 hours at the close. Those are
    exactly where the KMIA low tends to form.

    `asof_utc` bounds the answer to elapsed time — REQUIRED for live use. The
    wake cycle asks every 5 minutes, so it is usually mid-day: at 09:00Z the
    12:00Z and 18:00Z windows have not happened yet, and reporting them as
    unreported would cry wolf on nearly every wake. They come back under
    `pending_groups` instead.

    Returns `{gaps: [{start, end, minutes}], missing_groups, pending_groups,
    covered_minutes, day_minutes}`.
    """
    start, stop = climate_day_bounds(day, cfg.station.std_utc_offset)
    horizon = min(stop, _as_utc(asof_utc)) if asof_utc else stop
    if horizon < start:
        horizon = start
    six_col = "six_max_c" if extremum == "max" else "six_min_c"
    windows: list[tuple[datetime, datetime]] = []
    missing: list[str] = []
    by_window: dict[datetime, tuple[datetime, bool]] = {}
    for r in conn.execute(
            f"""SELECT valid_utc, {six_col} AS six FROM obs
                WHERE station=? AND source='metar'
                  AND valid_utc >= ? AND valid_utc < ?
                ORDER BY valid_utc""",
            (cfg.station.icao, (start - timedelta(hours=7)).isoformat(),
             (stop + timedelta(hours=7)).isoformat())):
        w_start, w_end = _six_hour_window(r["valid_utc"])
        if w_end.hour % 6 != 0:
            continue
        # ⚠️ ASK PER WINDOW, NOT PER REPORT. Several reports can land inside
        # one 6-hourly window — the routine :53 filing plus any number of
        # SPECIs, which fire whenever conditions jump and carry no group. Two
        # earlier versions of this loop asked "did THIS report have a group?"
        # and named four SPECIs as missing on 2026-08-06 (then one at 17:51,
        # sitting beside the routine 17:53 that did carry it). The question is
        # whether ANY report closed that window, so the group is recorded
        # against the window and judged after the scan.
        if w_start >= start and w_end <= stop:
            by_window.setdefault(w_end, (w_start, False))
            if r["six"] is not None:
                by_window[w_end] = (w_start, True)

    pending: list[str] = []
    for w_end, (w_start, had_group) in sorted(by_window.items()):
        if had_group:
            windows.append((w_start, w_end))
        elif w_end > horizon:
            pending.append(w_end.isoformat())
        else:
            missing.append(w_end.isoformat())
    # A window nobody reported at all leaves no row to scan, so walk the
    # expected synoptic hours too rather than only the ones that produced obs.
    #
    # ⚠️ ONLY WINDOWS THAT WOULD HAVE BEEN ADMISSIBLE COUNT. The window ending
    # at the day's first synoptic hour always straddles the climate-day start
    # and is refused regardless, so its absence costs nothing and naming it
    # would be a permanent false alarm on every single day.
    w = start
    while w <= stop:
        if (w.hour % 6 == 0 and w not in by_window
                and w - timedelta(hours=6) >= start and w <= stop):
            (pending if w > horizon else missing).append(w.isoformat())
        w += timedelta(hours=1)
    missing.sort(); pending.sort()

    windows.sort()
    merged: list[list[datetime]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    gaps, cursor = [], start
    for a, b in merged:
        if a > cursor:
            gaps.append((cursor, min(a, horizon)))
        cursor = max(cursor, b)
    if cursor < horizon:
        gaps.append((cursor, horizon))
    gaps = [(a, b) for a, b in gaps if b > a]

    covered = sum((b - a).total_seconds() for a, b in merged) / 60.0
    return {
        "station": cfg.station.icao, "day": day.isoformat(), "extremum": extremum,
        "gaps": [{"start": a.isoformat(), "end": b.isoformat(),
                  "minutes": round((b - a).total_seconds() / 60.0)}
                 for a, b in gaps],
        "missing_groups": missing,
        "pending_groups": pending,
        "covered_minutes": round(covered),
        "day_minutes": round((horizon - start).total_seconds() / 60.0),
    }


def visible_running_max(conn, cfg: AppConfig, day: date,
                        asof_utc: str | None = None):
    return visible_running_extremum(conn, cfg, day, "max", asof_utc)


def _max_iv(a, b):
    if a is None:
        return b
    return (max(a[0], b[0]), max(a[1], b[1]))


def _min_iv(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]))


def hidden_spike_pmf(conn, cfg: AppConfig, min_days: int = 120,
                      metric: str = "high") -> dict[int, float]:
    """P(cli_<metric> - visible_<extremum> = k): how often the official
    high/low beats everything observable. Learned from history; used as a
    dashboard diagnostic and sanity floor for the intraday model."""
    spec = get_metric(cfg, metric)
    deltas: Counter[int] = Counter()
    for r in conn.execute(
            "SELECT date, high_f, low_f FROM cli_reports WHERE station=? ORDER BY date",
            (cfg.station.icao,)):
        vis = visible_running_extremum(conn, cfg, date.fromisoformat(r["date"]), spec.extremum)
        if vis is None:
            continue
        bound = vis[1] if spec.extremum == "max" else vis[0]
        deltas[int(r[spec.target_col]) - round_half_up(bound)] += 1
    total = sum(deltas.values())
    if total < min_days:
        return {0: 1.0}
    return {k: v / total for k, v in sorted(deltas.items())}


def fetch_asos1min(cfg: AppConfig, start: date, end: date) -> list[dict]:
    """Fetch IEM's 1-minute ASOS archive.

    NOTE on real payload shape (differs from the naive 3-column assumption):
    the CSV header is `station,station_name,valid(UTC),tmpf` (4 columns —
    station_name sits between station and valid), the valid(UTC) field has
    no seconds (`YYYY-MM-DD HH:MM`), and tmpf is a bare whole-degree F
    integer or the literal `M` for missing. Verified live against LAX,
    2025-07-01..02 (Task 14).
    """
    now = utcnow_iso()
    text = get_text(IEM_1MIN, {
        "station": cfg.station.iem_id, "vars": "tmpf", "sample": "1min",
        "sts": f"{start.isoformat()}T00:00Z", "ets": f"{end.isoformat()}T23:59Z",
        "what": "download", "tz": "UTC", "delim": "comma",
    })
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4:
            _, _name, valid, tmpf = parts[0], parts[1], parts[2], parts[3]
            try:
                v = float(tmpf)
            except ValueError:
                continue
            valid_iso = valid.strip().replace(" ", "T")
            if valid_iso.count(":") == 1:   # no seconds field -> pad to canonical form
                valid_iso += ":00"
            rows.append({"station": cfg.station.icao,
                         "valid_utc": valid_iso + "+00:00",
                         "source": "asos1", "tmpf": v, "tmpf_precise": None,
                         "dwpf": None, "sknt": None, "drct": None, "skyc": None,
                         "six_max_c": None, "six_min_c": None,
                         "raw": None, "fetched_at": now})
    return rows
