import json
from datetime import datetime, timedelta, timezone

from highcast import db
from highcast.config import AppConfig
from highcast.http import get_json
from highcast.timeutil import utcnow_iso

IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"


def _to_int_or_none(value):
    """IEM CLI reports use the string "M" (and sometimes other non-numeric
    junk) as a missing-value sentinel. Treat that - or anything else that
    can't parse as an int - as missing rather than raising."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _product_issued_utc(product: str | None) -> datetime | None:
    """UTC instant an IEM CLI product was issued, parsed from the leading 12
    digits of its id (YYYYMMDDHHMM), e.g. "202608012033-KMFL-CDUS42-CLIMIA"
    -> 2026-08-01 20:33 UTC. None if `product` is missing or unparseable."""
    if not product or len(product) < 12 or not product[:12].isdigit():
        return None
    try:
        return datetime.strptime(product[:12], "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def cli_row_is_final(product: str | None, day: str, std_utc_offset: int) -> bool:
    """Was this CLI product issued AFTER the climate day it describes closed?

    MOVED HERE FROM `backfill.py` (2026-08-02) so re-fetch decisions
    (`backfill._yesterday_cli`) and settlement (`pick.settle`, `appbet.settle`)
    share ONE definition instead of each growing their own. This project has
    already paid for a duplicated rule twice -- `orderbot.CONFIDENCE_FLOOR`
    shadowed `gating.CONFIDENCE_FLOOR` and silenced a re-measurement, and
    `fc_anchor` re-entered a blend it was derived from -- so a second copy of
    this predicate is not a style nit, it is how this project's bugs recur.

    A report issued while the day is still running is PRELIMINARY: NWS may
    revise it, and any extremum set after the issue time is missing from it.
    Caught live twice. KLAX 2026-07-30 (see `tests/test_cli_preliminary.py`)
    corrupted `backfill`'s re-fetch cache. KMIA 2026-08-01 went further: the
    only product IEM ever issued for that date (202608012033-KMFL-CDUS42-
    CLIMIA, "low 82 at 6:24 AM") was filed 8.5h before the close, missed a
    storm that pushed the true low to 74F, and settlement graded a >=81 bet
    WON when it had lost -- see `scripts/migrate_prelim_settlement.py`.

    `product` is an IEM id like `202607311139-KLOX-CDUS46-CLILAX`, whose
    leading 12 digits are a UTC stamp. The climate day for `day` closes at
    `day+1 00:00` LOCAL STANDARD time, i.e. `-std_utc_offset` hours past
    midnight UTC -- derived per station, never hardcoded, because KLAX closes
    at 08:00Z and KMIA at 05:00Z.

    Fails toward freshness: a missing or unparseable stamp returns False, so
    a re-fetch decision retries and a settlement stays ungraded rather than
    trusting a row of unknown freshness.
    """
    issued = _product_issued_utc(product)
    if issued is None:
        return False
    try:
        closes = (datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                  + timedelta(days=1, hours=-std_utc_offset))
    except (TypeError, ValueError):
        return False
    return issued >= closes


def parse_rows(payload: dict, icao: str, fetched_at: str) -> list[dict]:
    """One row per DATE. When IEM's payload holds more than one CLI product
    for the same date -- a preliminary one and a later correction -- keep
    only the one issued LATEST, never whichever lands last in the payload
    (which was arbitrary). A row with a missing/unparseable product id never
    displaces one whose issuance time IS known; see `cli_row_is_final`'s
    docstring for why a preliminary product must not be trusted.

    ⚠️ This alone would NOT have prevented the KMIA 2026-08-01 incident: IEM
    held only the one preliminary product for that date, so there was no
    later one to prefer. It closes a real, separate gap (arbitrary payload-
    order wins) -- `cli_row_is_final` at the settlement path is the actual
    fix for the incident itself.
    """
    by_date: dict[str, dict] = {}
    issued_by_date: dict[str, datetime | None] = {}
    for item in payload.get("results", []):
        high = _to_int_or_none(item.get("high"))
        if high is None:
            continue
        d = str(item["valid"])[:10]
        issued = _product_issued_utc(item.get("product"))
        row = {
            "station": icao,
            "date": d,
            "high_f": high,
            "low_f": _to_int_or_none(item.get("low")),
            "raw": json.dumps(item),
            "fetched_at": fetched_at,
        }
        prior_issued = issued_by_date.get(d)
        is_new_date = d not in by_date
        replaces = is_new_date or (
            issued is not None and (prior_issued is None or issued > prior_issued))
        if replaces:
            by_date[d] = row
            issued_by_date[d] = issued
    return list(by_date.values())


def fetch_year(cfg: AppConfig, year: int) -> list[dict]:
    payload = get_json(IEM_CLI, {"station": cfg.station.icao, "year": year})
    return parse_rows(payload, cfg.station.icao, utcnow_iso())


def store(conn, rows: list[dict]) -> int:
    return db.upsert(conn, "cli_reports", rows)
