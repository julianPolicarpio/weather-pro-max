"""STALENESS GUARD — has any feed stopped delivering? QUEUE item 4.

WHY. A feed has died silently TWICE and nobody noticed for days:

  * `nbm_nbs` — the fetch existed, the rows stopped, the model kept predicting
    from whatever it had.
  * `forecast_conv` — dead 2026-07-29 → 08-07 because the fetch lived only in a
    script that nothing called. Nine days.

Both were found by accident. Fixing feed #2 without this means feed #3 dies the
same way. This is worth more than any single feed.

⚠️ THE SIGNAL IS "NEW DATA ARRIVED", NOT "THE FETCHER RAN". `fetched_at` only
advances when rows are written, which is exactly right: a fetcher that runs every
five minutes and stores nothing for three days IS a dead feed, and one that
happily returns the same stale payload forever is the failure mode this exists to
catch.

⚠️ THE THRESHOLD IS THE SOURCE'S PUBLICATION CADENCE, NOT THE FETCH CADENCE.
`forecast_conv` is fetched every 55 minutes but Open-Meteo republishes it roughly
every 6 hours, so healthy gaps of 9h are normal — the observed p90 is 39h. A
threshold set from the fetch interval would cry wolf every day. Every number
below is set generously against the MEASURED p90 gap (2026-07-13 → 08-13),
because a noisy guard gets ignored, and an ignored guard is worse than none.

⚠️ THREE FEEDS ARE EXEMPT AND MUST STAY EXEMPT:
  * `alerts` — event-driven. No active NWS alert means no rows, which is the
    normal state, not a failure. Its observed p90 gap is 25h for that reason.
  * `obs_1min` — an ARCHIVE with no live path (IEM lags 24-36h). It will never
    refresh and would alert forever.
  * `twc_obs` — zero rows, ever. Nothing writes it.
An exemption is a claim that ABSENCE IS NORMAL for that feed. Do not add one to
silence a feed that is genuinely broken.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from highcast.config import AppConfig


@dataclass(frozen=True)
class Feed:
    table: str
    label: str
    max_age_min: int
    source: str | None = None      # `obs` holds several feeds in one table
    per_station: bool = True
    why: str = ""
    # "fetched_at" = age of the newest write. "yesterday_row" = does a row exist
    # for yesterday, station-local. See the CLI note below for why both exist.
    check: str = "fetched_at"


# max_age_min is set well above the measured p90 gap over 2026-07-13..08-13.
# The p90 is quoted in `why` so a future reader can see the headroom rather than
# guess at it.
FEEDS: tuple[Feed, ...] = (
    Feed("obs", "metar", 180, source="metar",
         why="METARs file hourly; p90 gap 64min. 3h = three missed filings."),
    Feed("forecast_hourly", "openmeteo hourly", 360,
         why="p90 gap 57min; 6h is 6x headroom."),
    Feed("forecast_intraday", "intraday capture", 240,
         why="fetched every 55min, p90 gap 54min."),
    Feed("forecasts", "guidance (day-ahead)", 1440,
         why="p90 gap 286min; a full day without any guidance row is broken."),
    Feed("forecast_nbs", "NBM station bulletin", 2880,
         why="IEM posts a few times a day; p90 gap 23h. 48h avoids weekend gaps."),
    Feed("forecast_conv", "convection", 4320,
         why="source republishes ~6h but p90 gap is 39h. THIS is the feed whose "
             "9-day death started this item."),
    Feed("forecast_ens", "ensemble", 2880,
         why="p90 gap 17h."),
    Feed("forecast_aloft", "aloft", 2880,
         why="p90 gap 17h."),
    # ⚠️ CLI IS CHECKED SEMANTICALLY, NOT BY TIMESTAMP, and the difference is not
    # cosmetic — a `fetched_at` check here would have fired for FOURTEEN
    # CONSECUTIVE DAYS during normal operation.
    #
    # `backfill._yesterday_cli` short-circuits with `return 0` once yesterday's
    # row is present AND final, so a healthy run writes NOTHING and `fetched_at`
    # stops advancing. When it does write, it calls `fetch_year()` and `upsert`
    # (INSERT OR REPLACE), rewriting every row of the year with one fresh
    # timestamp — which also destroys the per-day write history.
    #
    # Observed: only 7 distinct `fetched_at` values since 2026-06-01, with a
    # 379h gap from 07-28 to 08-13 that was NOT an outage. A frozen timestamp on
    # this table means the feed is WORKING.
    #
    # The question that actually matters is whether yesterday's settlement is on
    # hand, so that is what gets asked.
    Feed("cli_reports", "CLI settlement", 2880, check="yesterday_row",
         why="timestamp is meaningless here (see comment); asks for yesterday's "
             "row instead. 48h of slack for late NWS publication."),
)

# Documented refusals. Kept in code so nobody re-adds them as 'missing feeds'.
EXEMPT: dict[str, str] = {
    "alerts": "event-driven — no active NWS alert means no rows, which is normal",
    "obs_1min": "archive only; IEM lags 24-36h and there is no live path",
    "twc_obs": "never populated by anything",
    "syn5": "Synoptic 5-minute feed is unconfigured (no token) — QUEUE item 13",
}


def _parse(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts if ("+" in ts or ts.endswith("Z"))
                                      else ts + "+00:00")
    except ValueError:
        return None


def check(conn, cfg: AppConfig, now: datetime | None = None) -> list[dict]:
    """One row per feed: age, threshold, and whether it is stale.

    Returns EVERY feed, not just the stale ones, so the wake line can show the
    healthy ones too — a guard that only speaks up on failure gives no
    confidence that it is still watching.
    """
    now = now or datetime.now(timezone.utc)
    icao = cfg.station.icao
    out = []
    for f in FEEDS:
        base = {"label": f.label, "table": f.table,
                "max_age_min": f.max_age_min, "reason": ""}
        try:
            if f.check == "yesterday_row":
                out.append({**base, **_check_yesterday_row(conn, cfg, f, now)})
            else:
                out.append({**base, **_check_fetched_at(conn, icao, f, now)})
        except Exception as e:                 # noqa: BLE001 - missing table etc
            # FAIL LOUD, not silent. A guard that swallows its own error is the
            # exact failure it was built to prevent.
            out.append({**base, "age_min": None, "stale": True,
                        "reason": f"check failed: {type(e).__name__}"})
    return out


def _check_fetched_at(conn, icao: str, f: Feed, now: datetime) -> dict:
    sql = f"SELECT MAX(fetched_at) m FROM {f.table} WHERE 1=1"
    args: list = []
    if f.per_station:
        sql += " AND station=?"
        args.append(icao)
    if f.source:
        sql += " AND source=?"
        args.append(f.source)
    row = conn.execute(sql, args).fetchone()
    last = _parse(row["m"] if row else None)
    if last is None:
        return {"age_min": None, "stale": True, "reason": "no rows ever"}
    age = (now - last).total_seconds() / 60.0
    return {"age_min": age, "stale": age > f.max_age_min,
            "last_utc": last.isoformat()}


def _check_yesterday_row(conn, cfg: AppConfig, f: Feed, now: datetime) -> dict:
    """How far behind TODAY is the newest settlement row, station-local?

    ⚠️ STATION-LOCAL, not UTC. At 05:10Z KMIA is already on the new day while
    KLAX is still on the previous one — the same bug that once produced a bogus
    "0 of 4 missed" commit report.

    ⚠️ AGE IS MEASURED IN WHOLE DAYS, and the first version of this got it wrong
    in a way that could never fire. It measured "minutes since local midnight",
    which maxes out at 1,440 — so a 2,880-minute threshold was unreachable and a
    permanently dead CLI feed would have stayed silent forever. A guard that
    cannot fire is worse than no guard, because it looks like coverage.

    Normal state is newest == yesterday (age 1 day) because today's settlement is
    not published until tomorrow. `max_age_min` of 2,880 therefore means "two
    consecutive days missing", matching QUEUE item 4's note that one missing day
    is not yet an outage.
    """
    from zoneinfo import ZoneInfo

    today = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    row = conn.execute(
        f"SELECT MAX(date) m FROM {f.table} WHERE station=?",
        (cfg.station.icao,)).fetchone()
    newest = row["m"] if row else None
    if not newest:
        return {"age_min": None, "stale": True, "reason": "no rows ever"}
    lag_days = (today - date.fromisoformat(newest)).days
    age = lag_days * 1440.0
    return {"age_min": age, "stale": age > f.max_age_min, "last_utc": newest,
            "reason": "" if age <= f.max_age_min
                      else f"newest settlement is {newest} ({lag_days}d behind)"}


def stale(conn, cfg: AppConfig, now: datetime | None = None) -> list[dict]:
    return [r for r in check(conn, cfg, now) if r["stale"]]


def wake_line(conn, cfg: AppConfig, now: datetime | None = None) -> str:
    """One line for the wake-cycle log. Names the offenders, counts the rest."""
    rows = check(conn, cfg, now)
    bad = [r for r in rows if r["stale"]]
    if not bad:
        worst = max((r for r in rows if r["age_min"] is not None),
                    key=lambda r: r["age_min"] / r["max_age_min"], default=None)
        tail = ""
        if worst:
            tail = (f" (oldest: {worst['label']} "
                    f"{worst['age_min'] / 60:.1f}h of "
                    f"{worst['max_age_min'] / 60:.0f}h)")
        return f"feeds: all {len(rows)} fresh{tail}"
    parts = []
    for r in bad:
        if r["age_min"] is None:
            parts.append(f"{r['label']} [{r['reason']}]")
        else:
            parts.append(f"{r['label']} {r['age_min'] / 60:.1f}h"
                         f" (max {r['max_age_min'] / 60:.0f}h)")
    return f"⚠️ feeds STALE {len(bad)}/{len(rows)}: " + ", ".join(parts)
