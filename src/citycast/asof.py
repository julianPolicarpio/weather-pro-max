"""The as-of grid (spec §3.2): when, relative to a climate day, a forecast is made."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.timeutil import climate_day_bounds

GRID: tuple[tuple[int, int], ...] = (tuple((-1, h) for h in (6, 9, 12, 15, 18, 21))
                                      + tuple((0, h) for h in range(0, 24, 3)))


def asof_points(st, day: date) -> list[datetime]:
    end = climate_day_bounds(day, st.std_utc_offset)[1]
    zone = ZoneInfo(st.tz)
    out = []
    for dd, hour in GRID:
        t = datetime.combine(day + timedelta(days=dd), time(hour), tzinfo=zone).astimezone(timezone.utc)
        if t < end:
            out.append(t)
    return sorted(out)


def lead_hours(st, day: date, t: datetime) -> float:
    return (climate_day_bounds(day, st.std_utc_offset)[1] - t).total_seconds() / 3600


def same_day(st, day: date, t: datetime) -> bool:
    return t >= climate_day_bounds(day, st.std_utc_offset)[0]
