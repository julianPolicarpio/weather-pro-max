"""CONVECTION FROM OUR OWN OBSERVATIONS — because the NWS feed cannot see it.

⚠️ THE DEFECT THIS REPLACES. `notify.notify_unstable`'s storm half read the
`alerts` table for `STORM_EVENTS`. That table has **never contained a single
storm product for either station** — its own docstring admits the archive fetch
requests `HEAT_PHENOMENA` only and its VTEC map has no thunderstorm codes. So:

    alert_stamps WHERE kind='unstable' AND key LIKE '%storm%'  ->  0 rows

Zero storm warnings in a month of operation, through a Miami August that
thunderstorms almost daily, while the HEAT half fired routinely. **A guard that
cannot fire** — the same failure as the CLI staleness check (error #8) and the
`daygrade` advisory check (error #11), both of which looked like coverage until
someone asked whether they had ever gone off.

Severe Thunderstorm Warnings are also the wrong instrument even when present:
they need 58 mph winds or 1" hail. An ordinary Miami downdraft clears no such
bar and still moves the temperature 13 °F, which is six Kalshi rungs.

WHAT WE ACTUALLY HAVE, from the METARs already in `obs`:

    KMIA  wx_thunder  3,895 rows over  696 days   (2020-01-30 .. )
    KMIA  sky_cb      7,076 rows over 1,071 days
    KLAX  wx_thunder     22 days                  (LA barely convects)

WHY IT IS WORTH AN ALERT, measured 2026-08-28 over 2,430 KMIA climate days.
Does the daily LOW break >=2 °F below what was already visible?

    after thunder ONSET          199/695 = 28.6%
    quiet-day control            92/1733 =  5.3%      -> 5.4x

and the QUEUE's original framing reproduces too — the low falls >=2 °F after the
h7 commit on 19.8% of thunder days against 5.5% of quiet ones (3.6x).

⚠️ THIS IS A MONITOR, NOT A PREDICTOR, AND THE DISTINCTION IS MEASURED. Rain and
convection have been tested as MODEL FEATURES three times and failed every time:
`convection_eval.py` Task 1 (KMIA HIGH h13, dHIT +0.0%), `convection_kmia_low_gate.py`
Task 1b (KMIA LOW h7, +1.9 pts against a random arm that also scored +1.9), and
Julian's Phase 0 bundle B3 (observed METAR rain, h9-h13, every cell REJECT).
**Nothing here may be routed into a model, a stake or a gate.** It tells a human
that a committed low is under attack, which is the one job it has evidence for.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from highcast.config import AppConfig

#: How recently thunder must have been reported to count as "happening now".
#: One hour, because KMIA files at :53 and a storm that ended 40 minutes ago has
#: usually already done its damage to the temperature.
WINDOW_MIN = 60

#: How far back to look for the PRE-STORM temperature, which is a different
#: question from "is it storming".
#:
#: ⚠️ THESE MUST NOT BE THE SAME NUMBER. The first version measured the swing
#: inside the 60-minute detection window and reported the real 2026-08-24 KMIA
#: downdraft as a **1.98F** swing — the window opened at 16:10Z and the pre-storm
#: peak was 89.96F at 15:53Z, seventeen minutes earlier. The true drop was 13F.
#: An alert whose whole job is "it has already fallen off a cliff" cannot quote a
#: number that starts measuring at the bottom of the cliff.
SWING_LOOKBACK_MIN = 180


def active(conn, cfg: AppConfig, now: datetime,
           window_min: int = WINDOW_MIN) -> dict | None:
    """Convection reported at this station within `window_min`, else None.

    `wx_thunder` is the primary (METAR `TS` in the present-weather group);
    `sky_cb` (a cumulonimbus layer) is reported alongside and is kept as
    corroboration rather than as a trigger of its own — a CB layer with no
    thunder is a cloud, not an event.
    """
    since = (now - timedelta(minutes=window_min)).isoformat()
    rows = conn.execute(
        """SELECT valid_utc, wx_present, wx_thunder, sky_cb, tmpf_precise
           FROM obs WHERE station=? AND valid_utc > ? AND valid_utc <= ?
           ORDER BY valid_utc""",
        (cfg.station.icao, since, now.isoformat())).fetchall()
    thunder = [r for r in rows if r["wx_thunder"]]
    if not thunder:
        return None
    last = thunder[-1]
    # The swing is read over SWING_LOOKBACK_MIN, not the detection window — see
    # the constant. Peak BEFORE the storm minus the latest reading, so the number
    # is the fall that has actually happened rather than the spread of whatever
    # the short window happened to contain.
    swing_since = (now - timedelta(minutes=SWING_LOOKBACK_MIN)).isoformat()
    trend = conn.execute(
        """SELECT tmpf_precise FROM obs
           WHERE station=? AND valid_utc > ? AND valid_utc <= ?
             AND tmpf_precise IS NOT NULL ORDER BY valid_utc""",
        (cfg.station.icao, swing_since, now.isoformat())).fetchall()
    temps = [r["tmpf_precise"] for r in trend]
    drop = (max(temps) - temps[-1]) if len(temps) >= 2 else None
    return {
        "since_utc": thunder[0]["valid_utc"],
        "latest_utc": last["valid_utc"],
        "wx": last["wx_present"] or "TS",
        "cb": bool(last["sky_cb"]),
        "reports": len(thunder),
        "temp_swing_f": drop,
    }
