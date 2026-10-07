"""Pick the market snapshot the desk ACTUALLY traded against.

⚠️ WHY THIS MODULE EXISTS — a bug that invalidated a full day of analysis,
2026-08-12. Four analysis scripts selected a day's ladder with

    for r in conn.execute("... ORDER BY created_utc"):
        priced.setdefault(r["target_date"], lad)      # <- keeps the FIRST

which keeps the EARLIEST snapshot of the day. The model's prediction, meanwhile,
is made at the commit hour. For KMIA HIGH that paired a 17:00Z prediction against
a 01:02Z quote — SIXTEEN HOURS apart, and 01:02Z is before the Miami climate day
(05:00Z) has even begun.

The model at commit has the whole morning's observations and a running extremum
ratchet. The market at 01:02Z has none of it. Comparing them is not a measurement
of skill, it is a lookahead advantage, and it produced the absurd conclusion that
the model beats the market by 36 points on its own picks — flatly contradicted by
the live ledger (PLACE 43%, SKIP 91%).

The rule: a decision made at time T may only see a price quoted at or after T,
and the nearest one. Anything else is trading on hindsight.

`same_day_only` defaults True because a quote from the following calendar day is
never the one the desk faced.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo


def commit_instant(target_date: str, hour: int, station_tz: str) -> datetime:
    """UTC instant of `hour` local on `target_date` — when the desk commits."""
    d = datetime.fromisoformat(target_date)
    return datetime(d.year, d.month, d.day, hour,
                    tzinfo=ZoneInfo(station_tz)).astimezone(ZoneInfo("UTC"))


def ladders_at_commit(conn, icao: str, metric: str, hour: int, station_tz: str,
                      *, same_day_only: bool = True) -> dict[str, list[dict]]:
    """{target_date: ladder} using the FIRST snapshot at or after the commit.

    A day with no snapshot at or after its commit hour is OMITTED rather than
    back-filled with an earlier one — the honest answer is "the desk had no price
    then", and silently substituting a stale quote is the bug this module exists
    to prevent.
    """
    out: dict[str, list[dict]] = {}
    rows = conn.execute(
        """SELECT target_date, created_utc, market_json FROM signals
           WHERE station=? AND metric=? AND market_json IS NOT NULL
             AND market_json != '' ORDER BY target_date, created_utc""",
        (icao, metric)).fetchall()
    for r in rows:
        td = r["target_date"]
        if td in out:
            continue                       # already have the first at/after
        commit = commit_instant(td, hour, station_tz)
        try:
            created = datetime.fromisoformat(r["created_utc"])
        except ValueError:
            continue
        if created < commit:
            continue
        if same_day_only and created.date() != commit.date():
            continue
        try:
            lad = json.loads(r["market_json"])
        except Exception:
            continue
        if lad and any(x.get("yes_ask") is not None for x in lad):
            out[td] = lad
    return out
