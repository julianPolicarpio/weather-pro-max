"""Regime-conditional pmf reshape for the KMIA LOW at h7.

Design: an internal design note

WHAT THIS FIXES. The intraday model already consumes the observed running
minimum as `run_lo`/`run_hi`, but nothing tells it WHICH of its two information
sources to trust today. `models/intraday.py` measured the same blind spot from
the other side and left the fix unshipped: guidance error is flat across
regimes while the bound's DOUBLES, and the model weights them identically.

On 2026-08-01 that ran in reverse -- the minimum had already landed (82.04F,
flat three hours, sun up), which is the regime where the bound is the reliable
side, and the model still emitted 80.0 tracking guidance 2-5F below an observed
fact. It put 37.4% on the bucket adjacent to the observed minimum, where the
regime-conditional rate is 22%.

WHY A RESHAPE AND NOT A SHIFT. The drop magnitude D = bound - settled is
zero-inflated: 65% of days never go below the bound at all, and the MEDIAN D is
0.16F in every single regime cell. All the signal is in the tail probability and
none of it in the centre, so a point forecast cannot carry it and a smooth
unimodal pmf cannot represent it.

PURE MODULE. No network, no DB writes, no model. It reads obs and forecasts and
returns a dict. That is what makes the money-free parts testable without either.
"""
from __future__ import annotations

import datetime as dt
import statistics
from zoneinfo import ZoneInfo

from highcast import latentmax

# The one (station, metric, hour) this layer may touch. Widening it is a
# deliberate edit guarded by a test -- the HIGH markets have the opposite shape
# (the extremum lands late, so the bound is uninformative most of the day) and
# would need their own gate.
SCOPE = ("KMIA", "low", 7)
SEASON_MONTHS = (6, 7, 8, 9)

# Pre-registered in spec 3.3. May NOT be lowered after seeing results.
MIN_CELL_N = 40

# Bound level bands. REQUIRED as a key dimension: D is not translation-
# invariant. Measured at h7 over 789 Jun-Sep days, P(D >= 1.5F) runs
# 22.2 / 34.9 / 40.4 / 36.6 % across these bands, so pooling would import
# cool-morning behaviour into a warm-morning forecast.
LEVEL_BANDS = ((None, 79.0), (79.0, 81.0), (81.0, 83.0), (83.0, None))

# Guidance-disagreement bands. bound - guidance_median; positive means guidance
# is COLDER than what has already been observed. P(D >= 1.5F) runs
# 19.6 / 36.0 / 40.1 / 44.1 % across these -- disagreement doubles the drop rate.
GAP_BANDS = ((None, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, None))


def _band(value: float, bands) -> int | None:
    """Index of the band containing `value`. Inclusive low, exclusive high."""
    for i, (lo, hi) in enumerate(bands):
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return i
    return None


def regime_key(bound: float | None, gap: float | None) -> tuple[int, int] | None:
    """(level_index, gap_index), or None meaning "fail closed".

    None on any missing or NaN input. For an unattended caller "I could not
    tell" and "no" must be the same answer -- the rule `confidence_floor_ok`
    already follows.
    """
    if bound is None or gap is None:
        return None
    if bound != bound or gap != gap:          # NaN
        return None
    lv, gp = _band(bound, LEVEL_BANDS), _band(gap, GAP_BANDS)
    if lv is None or gp is None:
        return None
    return (lv, gp)


def bound_for(conn, cfg, day: dt.date, hour: int) -> float | None:
    """Upper end of the running-MIN interval as of `hour` LOCAL. Never later.

    Upper end, not lower: it is the tightest value consistent with everything
    observed, and the settled low cannot exceed it.

    THE HOUR IS LOAD-BEARING. The bound is still falling between h7 and h8 on
    40.2% of days, so evaluating one hour late inflated the measured skill by
    ~7 points on the worked example. Callers pass the DECISION hour.
    """
    asof = dt.datetime(day.year, day.month, day.day, hour,
                       tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)
    iv = latentmax.visible_running_extremum(conn, cfg, day, "min",
                                            asof_utc=asof.isoformat())
    return None if iv is None else iv[1]


def guidance_median(conn, station: str, day: dt.date) -> float | None:
    """Unweighted median guidance low from runs STRICTLY BEFORE the target date.

    `max_temp_f` holds the LOW when metric='low' -- the column name is a
    historical wart, not a bug.

    NO MODEL ALLOW-LIST. Naming a subset would need its own justification and
    the set has changed twice this month. Note `gfs` and `hrrr` are a known
    duplicated channel, so one model effectively votes twice here; that is
    deliberate and recorded in spec 6, because fixing it inside this change
    would confound the gate.

    `run_date < target_date` IS STRICT, and `fetched_at` IS NOT USED. `run_date`
    is a DATE, so a same-day row may be an 18Z run that did not exist at the 11Z
    decision. And `fetched_at` cannot stand in: `db.upsert` is INSERT OR REPLACE
    and the `forecasts` primary key excludes it, so it is overwritten on every
    re-fetch -- the artifact found 2026-07-31.
    """
    iso = day.isoformat()
    rows = conn.execute(
        """SELECT max_temp_f FROM forecasts
           WHERE station=? AND metric='low' AND target_date=?
             AND run_date < ? AND max_temp_f IS NOT NULL""",
        (station, iso, iso)).fetchall()
    vals = [float(r[0]) for r in rows]
    return statistics.median(vals) if vals else None
