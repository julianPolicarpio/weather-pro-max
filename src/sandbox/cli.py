"""sandbox init | cycle | serve | report"""
import argparse
import fcntl
import json
import logging
import sqlite3
import sys
from contextlib import closing

from sandbox.bots import FavoriteBot, HighcastBot, IemMos, MosBot
from sandbox.cycle import ACCOUNTS, ensure_accounts, run_cycle, utc_now_iso
from sandbox.db import NOT_INITIALISED, NotInitialised, connect, connect_readonly, require_initialised
from sandbox.kalshi_public import PacedGet
from sandbox.scoreboard import (N_EVENTS_REQUIRED, N_REQUIRED, confirmation_status, report,
                                verdict_text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sandbox")
    p.add_argument("--db", default="data/sandbox.db")
    p.add_argument("--highcast-db", default="data/highcast.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    c = sub.add_parser("cycle")
    c.add_argument("--force-decide", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8713)
    sub.add_parser("report")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per request otherwise

    if args.cmd == "serve":
        import uvicorn
        from sandbox.web import create_app
        uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port)
        return 0

    if args.cmd == "cycle":
        # One cycle at a time: launchd starts one every 15 minutes, and a slow one (Kalshi
        # pacing, MOS, books) must not overlap the next on the same database.
        with open(f"{args.db}.cycle.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"status": "locked_out"}))
                return 0
            return _cycle(args)

    if args.cmd == "report":
        try:
            conn = _open_for_report(args.db)
        except NotInitialised as e:
            print(e, file=sys.stderr)
            return 1
        with closing(conn):
            _print_report(conn)
        return 0

    with closing(connect(args.db)) as conn:   # init
        ensure_accounts(conn)
        print(f"sandbox ready at {args.db}")
    return 0


def _open_for_report(path: str) -> sqlite3.Connection:
    """Read-only open that turns a missing file or an empty database into NotInitialised."""
    try:
        conn = connect_readonly(path)
    except sqlite3.OperationalError as e:   # mode=ro never creates the file
        raise NotInitialised(NOT_INITIALISED) from e
    try:
        require_initialised(conn, tuple(name for name, _, _ in ACCOUNTS))
    except NotInitialised:
        conn.close()
        raise
    return conn


def _ci(ci, places: int) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.{places}f}, {ci[1]:.{places}f}]"


def _print_report(conn) -> None:
    for name, _, _ in ACCOUNTS:
        sc = report(conn, name)
        verdict = verdict_text(confirmation_status(conn, name))
        print(f"{name:9} {verdict} settled {sc.settled_bets}/{N_REQUIRED} "
              f"scored {sc.brier_n}/{N_REQUIRED} "
              f"events {sc.n_events}/{N_EVENTS_REQUIRED} "
              f"scored-events {sc.brier_events}/{N_EVENTS_REQUIRED} "
              f"cash ${sc.cash:.2f} profit ${sc.profit:+.2f} "
              f"profit_test={'pass' if sc.profit_pass else 'fail'} "
              f"score_test={'pass' if sc.score_pass else 'fail'}")
        for segment in ("pre_day", "same_day"):
            split = report(conn, name, segment=segment)
            diff = "n/a" if split.brier_diff is None else f"{split.brier_diff:+.4f}"
            print(f"  {segment}: settled {split.settled_bets} profit ${split.profit:+.2f} "
                  f"profit CI {_ci(split.profit_ci, 2)} Brier diff {diff} "
                  f"Brier CI {_ci(split.brier_ci, 4)}")


def _cycle(args) -> int:
    with closing(connect(args.db)) as conn:
        ensure_accounts(conn)
        # favorite first (cheap), mos last: its MOS fetches are the slow part of an hour
        bots = [FavoriteBot(), HighcastBot(args.highcast_db), MosBot(IemMos())]
        summary = run_cycle(conn, PacedGet(), bots, utc_now_iso,
                            force_decide=args.force_decide)
    print(json.dumps(summary))
    # Non-zero marks a bad run: launchd records the last exit status, visible with
    # `launchctl print gui/$(id -u)/com.julian.sandbox.cycle` (not in the log files).
    if (summary["status"] != "ok" or summary.get("bot_errors") or summary.get("stale_bots")
            or summary.get("confirmation_errors")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
