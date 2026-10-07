"""A PRELIMINARY CLI report must be re-fetched until the final one lands.

CAUGHT 2026-08-01, and it had already corrupted the ledger.

NWS usually issues LAX's daily climate report around 01:30 local the FOLLOWING
day — after the climate day (01:00→01:00 local) has closed, so it is final. But
on 2026-07-30 it issued one at **18:29 local the SAME day**, hours before the day
ended. `backfill._yesterday_cli` stored it and then never looked again:

    have = SELECT 1 FROM cli_reports WHERE station=? AND date=?
    if have: return 0          # <-- frozen forever

That preliminary report said `low 70 at 5:52 AM`. The climate day then cooled to
**67 at 11:59 PM** — an evening minimum, which is the regime that produces 35% of
KLAX's daily lows. The final product (202607311139) carried 67.

WHAT THE STALE ROW COST:
  * `app_bets` KLAX low 2026-07-30, bucket ">=69", was graded WON. At the true
    low of 67 it LOST. The placed record is 2W-4L, not 3W-3L.
  * An audit that "verified" `app_bets.actual` against `cli_reports` confirmed
    nothing, because both sides read the same corrupted row.
  * The live LOW scorecard was overstated.

Only 1 of 1,144 stored station-days was affected, so this is a race, not
systemic rot. But it lands hardest on exactly the evening-minimum days a LOW
trader cares about, and it is silent.

THE RULE: a stored CLI row is trustworthy only if its product was issued AFTER
the climate day it describes had closed. Anything else is provisional and must
be re-fetched. Fails toward freshness — an unparseable or missing product stamp
counts as provisional.
"""
import datetime as dt

from highcast.fetchers.cli_reports import cli_row_is_final


def _product(stamp: str) -> str:
    """An IEM CLI product id: YYYYMMDDHHMM-KLOX-CDUS46-CLILAX (stamp is UTC)."""
    return f"{stamp}-KLOX-CDUS46-CLILAX"


def test_a_report_issued_after_the_climate_day_closed_is_final():
    """The normal case: LAX's CLI lands ~01:30 local the next day. The 2026-07-30
    climate day closes 01:00 local on 07-31 = 08:00Z, so a 11:39Z product is
    comfortably final."""
    assert cli_row_is_final(_product("202607311139"), "2026-07-30", -8) is True


def test_the_report_that_broke_the_ledger_is_rejected():
    """THE REGRESSION. 202607310129 = 01:29Z on 07-31 = 18:29 local on 07-30,
    issued before the climate day closed. It reported low 70; the day finished at
    67."""
    assert cli_row_is_final(_product("202607310129"), "2026-07-30", -8) is False


def test_a_report_issued_during_the_day_it_describes_is_rejected():
    assert cli_row_is_final(_product("202607301800"), "2026-07-30", -8) is False


def test_the_boundary_is_the_climate_day_close_not_midnight_utc():
    """KLAX's climate day is 08:00Z->08:00Z. A product at 07:59Z on 07-31 is one
    minute EARLY and must be refused; 08:01Z is final. Using midnight UTC instead
    would accept the early one."""
    assert cli_row_is_final(_product("202607310759"), "2026-07-30", -8) is False
    assert cli_row_is_final(_product("202607310801"), "2026-07-30", -8) is True


def test_kmia_uses_its_own_offset_not_klax_s():
    """Both stations share one database; a hardcoded offset here would silently
    mis-judge one of them. KMIA's climate day is 05:00Z->05:00Z, so a 06:00Z
    product on the following day is final for KMIA but would be EARLY for KLAX."""
    assert cli_row_is_final(_product("202607310600"), "2026-07-30", -5) is True
    assert cli_row_is_final(_product("202607310600"), "2026-07-30", -8) is False


def test_missing_or_unparseable_product_is_treated_as_provisional():
    """Fail toward freshness: if we cannot prove the row is final, re-fetch. The
    cost of an extra request is one HTTP call; the cost of trusting a stale row
    is a mis-graded bet."""
    for bad in (None, "", "not-a-product", "20260731-KLOX", "abcdefghijkl-KLOX"):
        assert cli_row_is_final(bad, "2026-07-30", -8) is False


# ── the SECOND incident: KMIA 2026-08-01, the settlement path ──────────────
#
# This one was worse than 2026-07-30: it did not just corrupt `backfill`'s
# re-fetch cache, it mis-graded a live bet from a LOSS into a WIN. IEM's only
# CLI product for this date is `202608012033-KMFL-CDUS42-CLIMIA`, issued
# 20:33Z with "low 82 at 6:24 AM". KMIA's climate day is 05:00Z->05:00Z, so
# that report was filed 8.5 HOURS before the day ended -- and a storm at
# 20:43Z (ten minutes later) dropped the station from 87.98F to 75.02F, with
# the true minimum landing at 21:22Z. True low 74F, confirmed by spot obs
# (73.94F), the 23:53Z six-hour group (23.3C), and the market's own final
# pricing (73-74 at 0.99, >=81 at 0.005). See
# scripts/migrate_prelim_settlement.py for the full account and the one-shot
# correction of the record it had already produced.

def test_kmia_2026_08_01_incident_product_is_rejected():
    """REGRESSION. This exact product, for this exact date, must never be
    treated as final -- that is what let a >=81 bet settle WON when it had
    LOST. If this test ever passes with `is True`, the settlement guard in
    `pick.settle` / `appbet.settle` has stopped protecting against the
    incident it was built for."""
    assert cli_row_is_final(
        "202608012033-KMFL-CDUS42-CLIMIA", "2026-08-01", -5) is False


def test_cli_row_is_final_has_exactly_one_definition():
    """The predicate must never grow a second copy. This project has already
    paid for that mistake twice -- `orderbot.CONFIDENCE_FLOOR` shadowed
    `gating.CONFIDENCE_FLOOR` and silenced a re-measurement, and `fc_anchor`
    re-entered a blend it was derived from -- so this is pinned directly
    rather than trusted to code review. If this fails, someone pasted a
    second `cli_row_is_final` instead of importing the one in
    `highcast.fetchers.cli_reports`."""
    import pathlib
    import re

    src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    defs = sorted(
        p for p in src_root.rglob("*.py")
        if re.search(r"(?m)^def cli_row_is_final\(", p.read_text()))
    assert defs == [src_root / "highcast" / "fetchers" / "cli_reports.py"]
