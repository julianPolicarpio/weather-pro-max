"""BULK-INGEST THE IEM 1-MINUTE ASOS ARCHIVE into `obs_1min`.

WHY, 2026-08-12. KMIA 2026-08-11 peaked at 18:39Z -- fourteen minutes before the
18:53Z METAR filing and 0.9F above it, which is one whole Kalshi rung. Our
hourly feed could not see it until the six-hour remark group filed at 23:53Z,
5h14m later; the market repriced at 20:04Z. See QUEUE.md item 13.

The 1-minute archive lags 24-36h so it CANNOT close that live gap. What it can
do is give years of true 1-minute running-max/min history, which no live feed
can retroactively supply. Measured on 41 KMIA days: the settled high was already
attained in 1-minute data by the 17:00Z commit hour on 61% of days, median
first-touch 16:33Z.

⚠️ IT GOES IN `obs_1min`, NOT `obs`. See the long comment on the table in
`db.py`. Short version: `latentmax.visible_running_extremum` consumes `asos1`
rows without filtering, its output feeds the shipped `run_*` features, and the
archive can only ever cover training rows -- so landing it in `obs` would train
the model on a precision that does not exist at prediction time.

KNOWN LIMITS OF THE SOURCE, verified live before writing this:
  * temperatures are WHOLE-DEGREE F -- coarser than the METAR T-group's 0.18F.
    On 10% of days the 1-minute series never reaches the settled high because of
    it. Do not treat this table as ground truth for settlement.
  * coverage is patchy: 318-1439 of a possible 1440 minutes/day.
  * 2023 came back EMPTY for both stations on a probe of 2023-06-15. Expect a
    hole; the coverage report at the end prints it rather than hiding it.

RESUMABLE. Progress is recorded per station in `meta` as
`asos1min_through:<ICAO>`; re-running skips completed chunks. Safe to kill.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/ingest_asos1min.py            # all
  PYTHONPATH=src .venv/bin/python scripts/ingest_asos1min.py --report   # status
"""
import argparse
import sys
import time
from datetime import date, timedelta

from highcast import db, latentmax
from highcast.config import load_config

CONFIGS = ("config/klax.yaml", "config/kmia.yaml")
# The IEM archive reaches back past 2019 for both stations, but there is no
# point holding minutes for days the trainer cannot use: `build_features` walks
# `cli_reports`, which starts 2021-01-01 (KLAX) / 2020-01-01 (KMIA).
FLOOR = {"KLAX": date(2021, 1, 1), "KMIA": date(2020, 1, 1)}
CHUNK_DAYS = 45          # a 41-day request returned 48k rows comfortably
RETRIES = 3
PAUSE_S = 2.0            # be polite to IEM between chunks


def _meta_key(icao: str) -> str:
    return f"asos1min_through:{icao}"


def _get_progress(conn, icao: str) -> date | None:
    r = conn.execute("SELECT v FROM meta WHERE k=?", (_meta_key(icao),)).fetchone()
    return date.fromisoformat(r["v"]) if r and r["v"] else None


def _set_progress(conn, icao: str, through: date) -> None:
    db.upsert(conn, "meta", [{"k": _meta_key(icao), "v": through.isoformat()}])


def _fetch_chunk(cfg, a: date, b: date) -> list[dict]:
    """Fetch with retries. Raises only if every attempt fails."""
    last = None
    for attempt in range(RETRIES):
        try:
            return latentmax.fetch_asos1min(cfg, a, b)
        except Exception as e:                       # noqa: BLE001 - report and retry
            last = e
            print(f"      attempt {attempt + 1}/{RETRIES} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{a}..{b} failed after {RETRIES} attempts: {last}")


def report(conn) -> None:
    print("\n=== obs_1min coverage ===")
    rows = list(conn.execute(
        """SELECT station, COUNT(*) n, MIN(valid_utc) a, MAX(valid_utc) b,
                  COUNT(DISTINCT substr(valid_utc, 1, 10)) days
           FROM obs_1min GROUP BY station ORDER BY station"""))
    if not rows:
        print("   (empty)")
        return
    for r in rows:
        span = ((date.fromisoformat(r["b"][:10]) - date.fromisoformat(r["a"][:10])).days + 1)
        print(f"   {r['station']}  rows={r['n']:>9,}  days={r['days']:>5}/{span}"
              f"  ({r['days'] / span:.0%} of the span)  "
              f"{r['a'][:10]} .. {r['b'][:10]}")
        print(f"        mean minutes/day present: {r['n'] / max(r['days'], 1):.0f} of 1440")
    print("\n   Missing days are the source's, not a bug here — IEM's archive has")
    print("   holes (a 2023 probe came back empty for both stations).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print coverage and exit")
    ap.add_argument("--end", default=None, help="last date to ingest (default: 2 days ago)")
    args = ap.parse_args()

    # The archive lags 24-36h; asking for the last two days just burns requests.
    end_default = date.today() - timedelta(days=2)
    end = date.fromisoformat(args.end) if args.end else end_default

    conn = db.connect(load_config(CONFIGS[0]).db_path)   # migrate=True: build obs_1min
    conn.execute("PRAGMA busy_timeout=30000")            # the wake cycle writes every 5 min

    if args.report:
        report(conn)
        return 0

    print(__doc__.split("Usage:")[0])
    for cfgp in CONFIGS:
        cfg = load_config(cfgp)
        icao = cfg.station.icao
        start = _get_progress(conn, icao)
        start = (start + timedelta(days=1)) if start else FLOOR[icao]
        if start > end:
            print(f"\n### {icao}: already through {start - timedelta(days=1)} — nothing to do")
            continue
        total_chunks = ((end - start).days // CHUNK_DAYS) + 1
        print(f"\n### {icao} (iem_id {cfg.station.iem_id}): {start} .. {end}"
              f"  ~{total_chunks} chunks of {CHUNK_DAYS}d", flush=True)
        a = start
        i = 0
        while a <= end:
            b = min(a + timedelta(days=CHUNK_DAYS - 1), end)
            i += 1
            try:
                rows = _fetch_chunk(cfg, a, b)
            except RuntimeError as e:
                # Stop THIS station at the failure point rather than skipping it
                # silently — progress is recorded, so a re-run resumes here.
                print(f"   !! {e}\n   !! stopping {icao}; re-run to resume", flush=True)
                break
            slim = [{"station": r["station"], "valid_utc": r["valid_utc"],
                     "tmpf": r["tmpf"], "fetched_at": r["fetched_at"]} for r in rows]
            db.upsert(conn, "obs_1min", slim)
            _set_progress(conn, icao, b)
            print(f"   [{i}/{total_chunks}] {a} .. {b}  +{len(slim):>6,} rows", flush=True)
            a = b + timedelta(days=1)
            time.sleep(PAUSE_S)
    report(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
