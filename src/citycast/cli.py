"""citycast init | backfill SOURCE | coverage (spec §2.4–2.5)."""
import argparse
import json
import sys
from contextlib import closing
from datetime import date

from citycast import coverage, db
from citycast.backfill import run_chunks
from citycast.http import Paced
from citycast.sources import cli_reports, kalshi_candles, kalshi_markets, mos, obs
from citycast.stations import load_stations, series_index
from highcast.timeutil import utcnow_iso

SOURCES = ("cli", "mos", "obs", "kalshi-markets", "kalshi-candles")


def _chunks_for(source, paced, conn, stations, since, today):
    if source == "cli":
        return cli_reports.chunks(paced, stations, since, today, utcnow_iso)
    if source == "mos":
        return mos.chunks(paced, stations, since, today, utcnow_iso)
    if source == "obs":
        return obs.chunks(paced, stations, since, today, utcnow_iso)
    if source == "kalshi-markets":
        idx = {s: v for s, v in series_index(stations).items() if v[0] in stations}
        return kalshi_markets.chunks(paced, idx)
    if source == "kalshi-candles":
        return kalshi_candles.chunks(paced, conn, utcnow_iso)
    raise ValueError(source)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citycast")
    p.add_argument("--db", default=str(db.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    b = sub.add_parser("backfill")
    b.add_argument("source", choices=SOURCES + ("all",))
    b.add_argument("--station")
    b.add_argument("--since", default="2021-01-01")
    b.add_argument("--refresh", action="store_true")
    cov = sub.add_parser("coverage")
    cov.add_argument("--json", action="store_true")
    cov.add_argument("--since", default="2021-01-01")
    cov.add_argument("--station")
    args = p.parse_args(argv)

    stations = load_stations()
    if getattr(args, "station", None):
        if args.station not in stations:
            print(f"unknown station {args.station}", file=sys.stderr)
            return 2
        stations = {args.station: stations[args.station]}

    with closing(db.connect(args.db)) as conn:
        if args.cmd == "init":
            print(json.dumps({"status": "ok", "db": args.db}))
            return 0
        if args.cmd == "backfill":
            paced = Paced()
            today = date.today()
            since = date.fromisoformat(args.since)
            sources = SOURCES if args.source == "all" else (args.source,)
            total = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
            for source in sources:
                chunks = _chunks_for(source, paced, conn, stations, since, today)
                s = run_chunks(conn, chunks, refresh=args.refresh, now_iso=utcnow_iso)
                for k in ("done", "skipped", "rows"):
                    total[k] += s[k]
                total["errors"] += s["errors"]
            print(json.dumps(total))
            return 1 if total["errors"] else 0
        if args.cmd == "coverage":
            rep = coverage.compute(conn, stations, date.today(), date.fromisoformat(args.since))
            print(json.dumps(rep, indent=1) if args.json else coverage.render(rep))
            return 0 if rep["pass"] else 1
    return 2
