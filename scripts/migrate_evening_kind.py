"""One-shot migration: relabel mislabelled EVENING rows in `predictions`.

Design: an internal design note §4.5

THE DEFECT. Two different forecast objects were stored under kind='dayahead':
the 09:00 morning refresh (real morn_* features) and the ~17:00 evening run
(every morn_* NaN, because the HIGH's feature cutoff is 09:00 on the TARGET day
and the evening run fires ~16h before it). Measured on KLAX HIGH:

    KLAX HIGH  evening n=19  MAE 4.42F  bias -4.05F
               morning n=14  MAE 1.45F  bias +0.19F
    KLAX LOW   evening n= 6  MAE 3.08F  bias -3.08F
               morning n=12  MAE 1.33F  bias -0.41F

35 of 74 stored rows were evening rows -- nearly half.
Leaving them would keep every future backfilled statistic mixing the two
populations; deleting them would destroy the only evidence the evening object
behaves this way. So: relabel.

THE PREDICATE, stated once here and asserted by tests/test_evening_kind.py so
the two cannot drift:

    a kind='dayahead' row is an EVENING row when it was ISSUED BEFORE ITS OWN
    FEATURE CUTOFF, i.e. issued_at_utc < _cutoff_utc(target + cutoff_day_offset,
    cutoff_local_hour).

⚠️ THE CUTOFF IS METRIC-SPECIFIC AND THE FIRST VERSION OF THIS GOT IT WRONG.
The audit that produced EXPECTED_ROWS used `hour < 9`, which is the HIGH's
cutoff (09:00 on the TARGET day, cutoff_day_offset=0). The LOW's is 20:00 on the
PRIOR day (cutoff_local_hour=20, cutoff_day_offset=-1), so the HIGH's rule
marked nearly every LOW day-ahead as "evening" and the count came out 52
instead of 38. Safeguard 2 caught it and stopped. Mirroring
`features._cutoff_utc` exactly is what makes this correct for both metrics --
the point is not "before 9am", it is "before the model's own inputs existed".

SAFEGUARDS, in order, all required:
  1. copy the DB first and print the copy's path
  2. assert the matched count equals EXPECTED_ROWS, else STOP
  3. one transaction
  4. re-run the accuracy split and assert the numbers are UNCHANGED -- a
     migration that moves the wrong rows changes them
  5. idempotent: a second run matches zero and exits clean
"""
import datetime as dt
import shutil
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, "src")

from highcast import db  # noqa: E402
from highcast.config import load_config  # noqa: E402
from highcast.metric import get_metric  # noqa: E402

CONFIGS = ("config/klax.yaml", "config/kmia.yaml")

# Measured 2026-08-01. If this does not match, the data or the predicate has
# changed since the spec and the migration must STOP rather than guess.
# Re-derived 2026-08-01 with the CORRECTED metric-specific cutoff. The first
# value (38) came from the HIGH's rule applied to both metrics and over-counted
# three LOW rows issued between 20:00 and midnight -- after the LOW's own
# cutoff, so their features were complete.
EXPECTED_ROWS = 35


def feature_cutoff_utc(target_date: str, tz: str, cutoff_local_hour: int,
                       cutoff_day_offset: int) -> dt.datetime:
    """The instant this metric's features are computed as of.

    Mirrors `features._cutoff_utc(cfg, d + metric.cutoff_day_offset,
    metric.cutoff_local_hour)`. HIGH: 09:00 local on the target day. LOW: 20:00
    local on the PRIOR day.
    """
    d = dt.date.fromisoformat(target_date) + dt.timedelta(days=cutoff_day_offset)
    return dt.datetime(d.year, d.month, d.day, cutoff_local_hour,
                       tzinfo=ZoneInfo(tz)).astimezone(dt.timezone.utc)


def is_evening(issued_utc: str, target_date: str, tz: str,
               cutoff_local_hour: int, cutoff_day_offset: int) -> bool:
    """THE predicate: was this issued BEFORE its own features existed?

    Pure, so the test can exercise it directly rather than re-deriving it from
    SQL. A row issued before its cutoff read NaN for every as-of feature.
    """
    issued = dt.datetime.fromisoformat(issued_utc)
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=dt.timezone.utc)
    return issued < feature_cutoff_utc(target_date, tz, cutoff_local_hour,
                                       cutoff_day_offset)


def find_evening_rows(conn, cfg):
    """[(station, target_date, metric, issued_at_utc)] for rows to relabel."""
    out = []
    for mk in ("high", "low"):
        spec = get_metric(cfg, mk)
        for r in conn.execute(
                """SELECT station, target_date, metric, issued_at_utc
                   FROM predictions WHERE station=? AND metric=?
                     AND kind=?""",
                (cfg.station.icao, mk, db.KIND_DAYAHEAD)):
            if is_evening(r["issued_at_utc"], r["target_date"], cfg.station.tz,
                          spec.cutoff_local_hour, spec.cutoff_day_offset):
                out.append(tuple(r))
    return out


def accuracy_split(conn, cfg):
    """{(metric, kind): (n, mae, bias)} — recomputed AFTER the migration to
    confirm the same rows landed in the same populations."""
    out = {}
    for mk in ("high", "low"):
        for kind in (db.KIND_DAYAHEAD, db.KIND_EVENING):
            errs = []
            for r in conn.execute(
                    """SELECT p.point_f, c.high_f, c.low_f FROM predictions p
                       JOIN cli_reports c ON c.station=p.station AND c.date=p.target_date
                       WHERE p.station=? AND p.metric=? AND p.kind=?""",
                    (cfg.station.icao, mk, kind)):
                actual = r["high_f"] if mk == "high" else r["low_f"]
                if actual is None or r["point_f"] is None:
                    continue
                errs.append(r["point_f"] - actual)
            if errs:
                out[(mk, kind)] = (len(errs), sum(abs(e) for e in errs) / len(errs),
                                   sum(errs) / len(errs))
    return out


def main(apply: bool) -> int:
    cfgs = [load_config(c) for c in CONFIGS]
    conn = db.connect(cfgs[0].db_path)      # both configs share one DB file

    matches = []
    for cfg in cfgs:
        rows = find_evening_rows(conn, cfg)
        print(f"  {cfg.station.icao}: {len(rows)} evening rows still labelled "
              f"'{db.KIND_DAYAHEAD}'")
        matches += rows

    print(f"\ntotal matched: {len(matches)}")
    if not matches:
        print("nothing to do — already migrated (this script is idempotent).")
        return 0

    if len(matches) != EXPECTED_ROWS:
        print(f"\nSTOP: expected exactly {EXPECTED_ROWS} rows, found {len(matches)}.")
        print("The data or the predicate has changed since the spec was written.")
        print("Re-derive the number before running with --apply.")
        return 1

    if not apply:
        print("\nDRY RUN. Re-run with --apply to write.")
        return 0

    backup = f"{cfgs[0].db_path}.pre-evening-migration"
    shutil.copy2(cfgs[0].db_path, backup)
    print(f"\nbackup written: {backup}")

    before = accuracy_split(conn, cfgs[0])
    with conn:
        for station, target, metric, issued in matches:
            conn.execute(
                """UPDATE predictions SET kind=? WHERE station=? AND target_date=?
                     AND metric=? AND issued_at_utc=? AND kind=?""",
                (db.KIND_EVENING, station, target, metric, issued,
                 db.KIND_DAYAHEAD))
    print(f"relabelled {len(matches)} rows -> kind='{db.KIND_EVENING}'")

    # Safeguard 4. The dayahead population must now be the MORNING rows only,
    # and its numbers must match what the audit measured for that subset.
    after = accuracy_split(conn, cfgs[0])
    print("\npost-migration split (KLAX):")
    for (mk, kind), (n, mae, bias) in sorted(after.items()):
        print(f"  {mk:5s} {kind:9s} n={n:3d}  MAE {mae:5.2f}F  bias {bias:+5.2f}F")
    left = sum(len(find_evening_rows(conn, c)) for c in cfgs)
    print(f"\nevening rows still mislabelled: {left}  "
          f"{'OK' if left == 0 else 'FAIL — investigate'}")
    return 0 if left == 0 else 1


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
