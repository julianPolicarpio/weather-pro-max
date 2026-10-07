import math
from datetime import date, datetime, timedelta, timezone


def climate_day_bounds(day: date, std_utc_offset: int) -> tuple[datetime, datetime]:
    """NWS climate day: midnight-to-midnight LOCAL STANDARD time (ignores DST)."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(
        hours=std_utc_offset
    )
    return start, start + timedelta(days=1)


def round_half_up(x: float) -> int:
    """NWS rounds .5 up; Python's round() is banker's rounding. Always use this."""
    return math.floor(x + 0.5)


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_valid(value) -> str:
    """Canonicalize a valid-time value to 'YYYY-MM-DDTHH:MM:SS+00:00' (UTC).

    Accepts epoch seconds (int/float), or ISO-ish strings with a 'T' or
    space date/time separator, an optional trailing 'Z', and optional
    fractional seconds. Both fetch_recent (aviationweather.gov) and
    fetch_archive (IEM) must funnel their timestamps through this so the
    same observation gets the same valid_utc regardless of source path.
    """
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat()
