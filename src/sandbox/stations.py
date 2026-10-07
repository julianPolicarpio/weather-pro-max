"""Settlement stations and series metadata for Kalshi daily temperature markets.

The station comes from each market's own rules text, never from a hand list:
"...recorded at New York City (CLINYC)..." -> KNYC. Verified for all 48 live
series on 2026-10-04. highcast.cities.ROSTER is stale (12 cities) and unused.
"""
import re
from datetime import date, datetime

from highcast.timeutil import climate_day_bounds

# ICAO -> (IANA tz, standard-time UTC offset in hours). The offset is what
# highcast.fetchers.forecasts.parse_mos needs to bucket MOS rows by climate day.
STATIONS: dict[str, tuple[str, int]] = {
    "KATL": ("America/New_York", -5),
    "KAUS": ("America/Chicago", -6),
    "KBOS": ("America/New_York", -5),
    "KDCA": ("America/New_York", -5),
    "KDEN": ("America/Denver", -7),
    "KDFW": ("America/Chicago", -6),
    "KEWR": ("America/New_York", -5),
    "KHOU": ("America/Chicago", -6),
    "KLAS": ("America/Los_Angeles", -8),
    "KLAX": ("America/Los_Angeles", -8),
    "KMDW": ("America/Chicago", -6),
    "KMIA": ("America/New_York", -5),
    "KMSP": ("America/Chicago", -6),
    "KMSY": ("America/Chicago", -6),
    "KNYC": ("America/New_York", -5),
    "KOKC": ("America/Chicago", -6),
    "KPHL": ("America/New_York", -5),
    "KPHX": ("America/Phoenix", -7),
    "KSAN": ("America/Los_Angeles", -8),
    "KSAT": ("America/Chicago", -6),
    "KSDF": ("America/Kentucky/Louisville", -5),
    "KSEA": ("America/Los_Angeles", -8),
    "KSFO": ("America/Los_Angeles", -8),
    "KTTN": ("America/New_York", -5),
}

_RULE_CODE = re.compile(r"\(CLI([A-Z]{3})\)")


def station_from_rules(rules_primary: str | None) -> str | None:
    """'(CLINYC)' in the rules text -> 'KNYC'. None when the code is absent."""
    m = _RULE_CODE.search(rules_primary or "")
    return f"K{m.group(1)}" if m else None


def metric_from_series(series: str) -> str | None:
    """KXHIGH* -> 'high', KXLOWT* -> 'low', anything else -> None."""
    if series.startswith("KXHIGH"):
        return "high"
    if series.startswith("KXLOWT"):
        return "low"
    return None


def climate_day_start_utc(icao: str | None, event_date: str | date) -> datetime | None:
    """When the station's climate day for `event_date` begins, in UTC: local STANDARD
    midnight (NWS ignores daylight saving). None for a station not in STATIONS."""
    if icao not in STATIONS:
        return None
    day = event_date if isinstance(event_date, date) else date.fromisoformat(event_date)
    return climate_day_bounds(day, STATIONS[icao][1])[0]


def event_started(market, at_iso: str) -> bool:
    """True once the market's climate day has begun at `at_iso`. Fails closed: a market
    whose start cannot be computed (unknown or missing station) counts as started."""
    start = climate_day_start_utc(market.station, market.event_date)
    return start is None or datetime.fromisoformat(at_iso) >= start
