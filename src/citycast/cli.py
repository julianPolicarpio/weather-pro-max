"""citycast init | backfill SOURCE | coverage (spec §2.4–2.5)."""
import argparse
import json
import sys
import time
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from citycast import (coverage, dataset, db, decisions, evaluate, final_report, gate_eval, laya_io, live, meta,
                      walkforward)
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
# One retry of the failed refresh chunks, after this pause (IEM timeouts are usually transient).
RETRY_PAUSE_S = 30
SOURCES = ("cli", "mos", "obs", "kalshi-markets", "kalshi-candles")

REPORT_DIR = Path("docs/reports")
B2_PATH = Path("data/citycast/models/gate-b2.pkl")
NO_LAYA = ("run this from ~/projects/laya-weather-gate/.venv/bin/citycast "
           "(laya is not installed in this environment)")


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
    dec = sub.add_parser("decisions")
    dec.add_argument("action", choices=("build", "export"))
    dec.add_argument("--out", default="data/citycast/laya")
    gate = sub.add_parser("gate")
    gate.add_argument("action", choices=("score", "final-report"))
    gate.add_argument("--contender", choices=("B2", "L0", "L1"))
    gate.add_argument("--model-ref")
    gate.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "gate" and args.action == "score":
        if not args.contender:
            p.error("gate score requires --contender")
        if args.contender == "L1":
            if not args.model_ref:
                p.error("gate score --contender L1 requires --model-ref REPO@REV")
            try:
                final_report.pinned_ref(args.model_ref)
            except ValueError as e:
                p.error(str(e))
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
                refresh_errors, failed = [], []
                for chunks in refresh:
                    s = run_chunks(conn, chunks, now_iso=utcnow_iso, progress=lambda m: None)
                    refresh_errors += s["errors"]
                    by_key = {(c.source, c.key): c for c in chunks}
                    failed += [by_key[(e["source"], e["key"])] for e in s["errors"]
                               if (e["source"], e["key"]) in by_key]
                if failed:   # retry only the chunks that errored; errors persisting after it count
                    time.sleep(RETRY_PAUSE_S)
                    retry = run_chunks(conn, failed, now_iso=utcnow_iso, progress=lambda m: None)
                    retried = {(c.source, c.key) for c in failed}
                    refresh_errors = [e for e in refresh_errors if (e["source"], e["key"]) not in retried] \
                        + retry["errors"]
                # a station whose refresh still failed gets no forecast; stale inputs must never feed the bot
                skip = frozenset(e["key"].split(":", 1)[0] for e in refresh_errors)
            else:
                refresh_errors, skip = [], frozenset()
            now = datetime.now(timezone.utc).replace(microsecond=0)
            out = {"issued_utc": now.isoformat(), "high": 0, "low": 0,
                   "skipped_stations": sorted(skip), "refresh_errors": refresh_errors}
            if set(stations) <= skip:   # every station blocked: write nothing
                print(json.dumps(out))
                return 1
            ctx = FeatureContext(conn, stations, args.db, mos_since=date.today() - timedelta(days=3))
            out.update({m: live.predict(conn, ctx, m, now, skip) for m in ("high", "low")})
            print(json.dumps(out))
            return 3 if skip else 0
        if args.cmd == "decisions" and args.action == "build":
            print(json.dumps({"rows": decisions.build(conn, stations)}))
            return 0
        if args.cmd == "decisions" and args.action == "export":
            s = gate_eval.split(gate_eval.load(conn))   # train and validation only: the test split is never exported
            out = Path(args.out)
            n_tr = laya_io.export(laya_io.sample_train(s["train"]), out / "train.jsonl")
            n_va = laya_io.export(s["validation"], out / "validation.jsonl")
            print(json.dumps({"train": n_tr, "validation": n_va, "dir": str(out)}))
            return 0
        if args.cmd == "gate" and args.action == "score":
            s = gate_eval.split(gate_eval.load(conn))
            if args.contender == "B2":
                m = meta.MetaLabeler(tuple(sorted(stations))).fit(s["train"])
                meta.save(m, B2_PATH)
                for k in ("train", "validation"):   # test is scored only by final-report
                    with conn:
                        conn.executemany("INSERT OR REPLACE INTO gate_scores VALUES (?,?,?,?)",
                                         [("B2", t, a, float(x)) for t, a, x
                                          in zip(s[k]["ticker"], s[k]["asof_utc"], m.predict(s[k]))])
            else:
                try:
                    fn = laya_io.router_predict_fn(None if args.contender == "L0" else args.model_ref)
                except ModuleNotFoundError as e:
                    if e.name != "laya":    # torch, peft...: a different problem, shown as it is
                        raise
                    print(NO_LAYA, file=sys.stderr)
                    return 2
                key = "L0" if args.contender == "L0" else final_report.l1_key(args.model_ref)
                laya_io.score(s["validation"], fn, key, conn)
            print(json.dumps({"scored": args.contender}))
            return 0
        if args.cmd == "gate" and args.action == "final-report":
            # Everything that can fail cheaply fails before the first (hours-long) test scoring run.
            final_report.ensure_not_run(conn, args.force)
            row = conn.execute("SELECT value FROM meta WHERE key='laya_l1_ref'").fetchone()
            try:
                ref = json.loads(row[0]) if row else None
            except ValueError:
                ref = None
            if not ref or not isinstance(ref, str):
                raise RuntimeError("laya_l1_ref not set: record the fine-tuned repo@revision first, as a JSON "
                                   "string (plan 3 Task 8 Step 5)")
            try:
                key = final_report.l1_key(ref)
            except ValueError as e:
                raise RuntimeError(f"laya_l1_ref invalid: {e}") from e
            b2 = meta.load_model(B2_PATH)
            try:
                fns = {"L0": laya_io.router_predict_fn(None), "L1": laya_io.router_predict_fn(ref)}
            except ModuleNotFoundError as e:
                if e.name != "laya":    # torch, peft...: a different problem, shown as it is
                    raise
                print(NO_LAYA, file=sys.stderr)
                return 2
            s = gate_eval.split(gate_eval.load(conn))
            scores = {"B2": (None, b2.predict(s["validation"]), b2.predict(s["test"]))}
            for name, fn in fns.items():
                cached_as = key if name == "L1" else name     # L1 scores are cached per exact model revision
                scores[name] = (None, laya_io.score(s["validation"], fn, cached_as, conn),
                                laya_io.score(s["test"], fn, cached_as, conn))
            out = final_report.run_final(conn, scores, REPORT_DIR / f"gate-final-{date.today()}.md",
                                         force=args.force, now_iso=utcnow_iso, l1_ref=ref)
            print(json.dumps({"winner": out["winner"], "margin": out["margin"], "model_ref": out["model_ref"]}))
            return 0
    return 2
