"""citycast init | backfill SOURCE | coverage (spec §2.4–2.5)."""
import argparse
import json
import sys
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from citycast import coverage, dataset, db, evaluate, live, walkforward
from citycast.backfill import run_chunks
from citycast.features import FeatureContext
from citycast.fingerprint import code_fingerprint
from citycast.http import Paced
from citycast.sources import cli_reports, kalshi_candles, kalshi_markets, mos, obs
from citycast.stations import load_stations, series_index
from highcast.timeutil import utcnow_iso

# Kalshi requests stay paced to <= 5/s by Paced's lock; 3 in flight reach that pace
# (measured ~0.35-0.55 s per request, 1.8 markets/s sequential on 2026-10-07).
CANDLE_WORKERS = 3
# IEM ingests a day or so late: a month/half-year is only final 2 UTC days after it ends.
COMPLETE_MARGIN_D = 2
SOURCES = ("cli", "mos", "obs", "kalshi-markets", "kalshi-candles")

REPORT_DIR = Path("docs/reports")


def _chunks_for(source, paced, conn, stations, since, today, candle_since=None,
                complete_before=None):
    if source == "cli":
        return cli_reports.chunks(paced, stations, since, today, utcnow_iso,
                                  complete_before=complete_before)
    if source == "mos":
        return mos.chunks(paced, stations, since, today, utcnow_iso,
                                  complete_before=complete_before)
    if source == "obs":
        return obs.chunks(paced, stations, since, today, utcnow_iso,
                                  complete_before=complete_before)
    if source == "kalshi-markets":
        idx = {s: v for s, v in series_index(stations).items() if v[0] in stations}
        return kalshi_markets.chunks(paced, idx)
    if source == "kalshi-candles":
        return kalshi_candles.chunks(paced, conn, utcnow_iso, since=candle_since)
    raise ValueError(source)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citycast")
    p.add_argument("--db", default=str(db.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    b = sub.add_parser("backfill")
    b.add_argument("source", choices=SOURCES + ("all",))
    b.add_argument("--station")
    b.add_argument("--since", default=None)
    b.add_argument("--refresh", action="store_true")
    cov = sub.add_parser("coverage")
    cov.add_argument("--json", action="store_true")
    cov.add_argument("--since", default="2021-01-01")
    cov.add_argument("--station")
    ds = sub.add_parser("dataset")
    ds.add_argument("--metric", choices=("high", "low"), required=True)
    ds.add_argument("--since", default="2022-01-01")
    wf = sub.add_parser("walkforward")
    wf.add_argument("--metric", choices=("high", "low"), required=True)
    sub.add_parser("evaluate")
    tr = sub.add_parser("train")
    tr.add_argument("--metric", choices=("high", "low"), required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--no-refresh", action="store_true")
    args = p.parse_args(argv)
    # One hash per command, taken before anything fingerprinted is read or built: it is stamped on the
    # dataset, recorded by walkforward and logged by evaluate, so an edit mid-run cannot mismatch them.
    fp = code_fingerprint() if args.cmd in ("dataset", "walkforward", "evaluate") else None

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
            today = datetime.now(timezone.utc).date()
            complete_before = today - timedelta(days=COMPLETE_MARGIN_D)
            since = date.fromisoformat(args.since or "2021-01-01")
            sources = SOURCES if args.source == "all" else (args.source,)
            total = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
            for source in sources:
                chunks = _chunks_for(source, paced, conn, stations, since, today,
                                     candle_since=args.since,
                                     complete_before=complete_before)
                s = run_chunks(conn, chunks, refresh=args.refresh, now_iso=utcnow_iso,
                               workers=CANDLE_WORKERS if source == "kalshi-candles" else 1)
                for k in ("done", "skipped", "rows"):
                    total[k] += s[k]
                total["errors"] += s["errors"]
            print(json.dumps(total))
            return 1 if total["errors"] else 0
        if args.cmd == "coverage":
            rep = coverage.compute(conn, stations, datetime.now(timezone.utc).date(), date.fromisoformat(args.since))
            print(json.dumps(rep, indent=1) if args.json else coverage.render(rep))
            return 0 if rep["pass"] else 1
        if args.cmd == "dataset":
            ctx = FeatureContext(conn, stations, args.db)
            df = dataset.build(ctx, args.metric, date.fromisoformat(args.since), date.today())
            print(json.dumps({"rows": len(df), "path": str(dataset.save(df, args.metric, fp))}))
            return 0
        if args.cmd == "walkforward":
            n = walkforward.run(conn, dataset.load(args.metric), args.metric, tuple(sorted(stations)), utcnow_iso,
                                fingerprint=fp)
            print(json.dumps({"written": n}))
            return 0
        if args.cmd == "evaluate":
            path = evaluate.evaluate_all(conn, REPORT_DIR / f"citycast-evaluation-{date.today()}.md", utcnow_iso,
                                         fingerprint=fp)
            print(json.dumps({"report": str(path)}))
            return 0
        if args.cmd == "train":
            path = live.train_final(conn, dataset.load(args.metric), args.metric, tuple(sorted(stations)), date.today())
            print(json.dumps({"bundle": str(path)}))
            return 0
        if args.cmd == "predict":
            if not args.no_refresh:
                paced, now_utc = Paced(), datetime.now(timezone.utc)
                today = now_utc.date()                    # UTC, as backfill: local dates shift chunk edges
                complete_before = today - timedelta(days=COMPLETE_MARGIN_D)
                refresh = [_chunks_for("cli", paced, conn, stations, today - timedelta(days=7), today,
                                       complete_before=complete_before),
                           mos.recent_chunks(paced, stations, now_utc, utcnow_iso),   # never logged
                           obs.recent_chunks(paced, stations, now_utc, utcnow_iso)]
                refresh_errors = []
                for chunks in refresh:
                    s = run_chunks(conn, chunks, now_iso=utcnow_iso, progress=lambda m: None)
                    refresh_errors += s["errors"]
                if refresh_errors:   # stale inputs must never feed the paper-trading bot
                    print(json.dumps({"refresh_errors": refresh_errors}))
                    return 1
            ctx = FeatureContext(conn, stations, args.db, mos_since=date.today() - timedelta(days=3))
            now = datetime.now(timezone.utc).replace(microsecond=0)
            written = {m: live.predict(conn, ctx, m, now) for m in ("high", "low")}
            print(json.dumps({"issued_utc": now.isoformat(), **written}))
            return 0
    return 2
