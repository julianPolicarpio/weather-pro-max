"""One-shot correction: KMIA LOW 2026-08-01 was settled on a PRELIMINARY CLI.

THE DEFECT. `fetchers/cli_reports.py` stores whatever IEM returns for a date
without ever inspecting the product's ISSUANCE TIME. For KMIA 2026-08-01 the
only product IEM holds is:

    202608012033-KMFL-CDUS42-CLIMIA     issued 2026-08-01 20:33Z
    high 94 (1238 PM)   low 82 (624 AM)

KMIA's climate day is 05:00Z -> 05:00Z, so that report was issued **8.5 hours
before the day ended** and physically could not see what happened afterward. At
20:43Z a thunderstorm took the station from 87.98F to 75.02F in 50 minutes, and
the true minimum landed at 21:22Z.

THE TRUE LOW IS 74F, confirmed three independent ways:
    spot readings          73.94F at 2026-08-01T21:22Z
    six-hour remark group  23.3C = 73.94F in the 23:53Z METAR (2sTTT)
    the market itself      final snapshot 03:57Z had 73-74 at 0.99 / >=81 at 0.005
`latentmax.visible_running_extremum` returns 73.84F -> settles 74F.

So the `>=81` bet LOST and was recorded WON. This is the direction that hides
problems: a false win also makes `gating.storm_warning_active` (shipped for this
very event) look unnecessary.

⚠️ RE-FETCHING DOES NOT FIX IT. IEM holds exactly one product for this date and
it is the preliminary one; no corrected CLI exists to pull. That is why this is a
correction script and not a re-fetch.

WHAT IT CHANGES
  app_bets   KMIA/low/2026-08-01   actual 82 -> 74,  won 1 -> 0
  cli_reports KMIA/2026-08-01      low_f 82 -> NULL

  The CLI low is NULLED, not overwritten with 74. 74 is OUR number, derived from
  observations; `cli_reports` is supposed to hold what the NWS published. Writing
  a derived value into a source-of-truth table would launder a reconstruction
  into an observation, and the next walk-forward could not tell them apart. NULL
  is the honest representation of "the published value is not trustworthy and no
  corrected one exists" -- and every consumer already skips NULL rows (they
  filter `low_f IS NOT NULL`), so the day simply drops out of KMIA LOW training
  rather than poisoning it. high_f stays 94, which is correct and unaffected.

  An audit row goes into `notes` so the reason survives in the database and not
  only in this file.

SAFEGUARDS, all required:
  1. back up the DB first and print the backup path
  2. assert the CURRENT state is exactly the defective one, else STOP. Re-running
     after a successful run finds the corrected state and exits cleanly, so this
     is idempotent rather than merely safe-once.
  3. one transaction
  4. re-read and print the corrected rows

Usage:  .venv/bin/python scripts/migrate_prelim_settlement.py [--apply]

Without --apply it reports what it would do and changes nothing.
"""
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

DB = "data/highcast.db"
BACKUP = "data/highcast.db.pre-prelim-settlement"
STATION, TARGET = "KMIA", "2026-08-01"
BAD_ACTUAL, BAD_WON, BAD_LOW = 82, 1, 82
TRUE_LOW = 74
PRODUCT = "202608012033-KMFL-CDUS42-CLIMIA"


def state(conn):
    b = conn.execute(
        "SELECT actual, won FROM app_bets WHERE station=? AND metric='low' "
        "AND target_date=?", (STATION, TARGET)).fetchone()
    c = conn.execute(
        "SELECT low_f FROM cli_reports WHERE station=? AND date=?",
        (STATION, TARGET)).fetchone()
    return b, c


def main() -> int:
    apply = "--apply" in sys.argv
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    bet, cli = state(conn)
    if bet is None or cli is None:
        print("STOP: expected rows not found; nothing done.")
        return 1

    already = (bet["actual"] == TRUE_LOW and bet["won"] == 0
               and cli["low_f"] is None)
    if already:
        print("Already corrected (actual=74, won=0, cli low_f=NULL). "
              "Nothing to do.")
        return 0

    if not (bet["actual"] == BAD_ACTUAL and bet["won"] == BAD_WON
            and cli["low_f"] == BAD_LOW):
        print("STOP: the database is in neither the defective nor the corrected "
              f"state. Found bet(actual={bet['actual']}, won={bet['won']}), "
              f"cli(low_f={cli['low_f']}). Refusing to guess.")
        return 1

    print(f"WILL CORRECT {STATION} LOW {TARGET}:")
    print(f"  app_bets    actual {BAD_ACTUAL} -> {TRUE_LOW},  won {BAD_WON} -> 0")
    print(f"  cli_reports low_f  {BAD_LOW} -> NULL   (preliminary product "
          f"{PRODUCT})")
    if not apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    shutil.copyfile(DB, BACKUP)
    print(f"\nbackup written: {BACKUP}")
    note = (f"SETTLEMENT CORRECTION. The >=81 bet was graded WON against CLI "
            f"product {PRODUCT}, issued 2026-08-01T20:33Z -- 8.5h before KMIA's "
            f"climate day ended (05:00Z) and 49min before the true minimum at "
            f"21:22Z. True low 74F, confirmed by spot obs (73.94F), the 23:53Z "
            f"six-hour group (23.3C), and the market's final pricing (73-74 at "
            f"0.99, >=81 at 0.005). The bet LOST. cli_reports.low_f nulled "
            f"because no corrected NWS product exists to replace it.")
    with conn:
        conn.execute(
            "UPDATE app_bets SET actual=?, won=0 WHERE station=? AND "
            "metric='low' AND target_date=?", (TRUE_LOW, STATION, TARGET))
        conn.execute(
            "UPDATE cli_reports SET low_f=NULL WHERE station=? AND date=?",
            (STATION, TARGET))
        conn.execute(
            "INSERT INTO notes (station, target_date, created_utc, sky, context) "
            "VALUES (?,?,?,?,?)",
            (STATION, TARGET, datetime.now(timezone.utc).isoformat(),
             "settlement-correction", note))

    bet, cli = state(conn)
    print(f"\ncorrected: app_bets actual={bet['actual']} won={bet['won']}, "
          f"cli_reports low_f={cli['low_f']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
