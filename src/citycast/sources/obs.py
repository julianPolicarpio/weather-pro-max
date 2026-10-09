"""IEM ASOS METAR archive -> obs (same parser and columns as highcast, incl. 6-hour groups)."""
from datetime import date, datetime, timedelta, timezone
from functools import partial

from citycast.backfill import Chunk
from highcast.fetchers.metar import parse_metar
from highcast.timeutil import norm_valid

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def obs_rows(text: str, st, fetched_at: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",", 2)
        if len(parts) == 3:
            _, valid, raw = parts
            rows.append(parse_metar(raw.strip(), norm_valid(valid.strip()), st.icao, fetched_at))
    return rows


def _months(since: date, today: date):
    cur = date(since.year, since.month, 1)
    while cur <= today:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt
        cur = nxt


def _fetch(paced, st, a: date, b: date, now_iso) -> list[tuple[str, list[dict]]]:
    """METARs from a 00Z up to b 00Z (UTC)."""
    text = paced.get_text(IEM_ASOS, {
        "station": st.iem_id, "data": "metar",
        "year1": a.year, "month1": a.month, "day1": a.day,
        "year2": b.year, "month2": b.month, "day2": b.day,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
        "report_type": [3, 4]})
    return [("obs", obs_rows(text, st, now_iso()))]


def chunks(paced, stations: dict, since: date, today: date, now_iso,
           complete_before: date | None = None) -> list[Chunk]:
    complete_before = complete_before or today
    out = []
    for st in stations.values():
        for a, b in _months(since, today):
            out.append(Chunk("obs", f"{st.icao}:{a.strftime('%Y-%m')}", b > complete_before,
                             partial(_fetch, paced, st, a, b, now_iso)))
    return out


def recent_chunks(paced, stations: dict, now: datetime, now_iso, days: int = 3) -> list[Chunk]:
    """The last `days` UTC days through tomorrow, for `citycast predict`: current, so never logged. A
    month chunk fetched by predict could be logged before IEM is complete and then never refetched."""
    now = now.astimezone(timezone.utc)
    a, b = (now - timedelta(days=days)).date(), (now + timedelta(days=1)).date()
    return [Chunk("obs", f"{st.icao}:recent", True, partial(_fetch, paced, st, a, b, now_iso))
            for st in stations.values()]
