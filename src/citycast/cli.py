"""citycast init | backfill | coverage (backfill/coverage added in Task 9)."""
import argparse
import json
from contextlib import closing

from citycast import db


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citycast")
    p.add_argument("--db", default=str(db.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    args = p.parse_args(argv)
    if args.cmd == "init":
        with closing(db.connect(args.db)):
            pass
        print(json.dumps({"status": "ok", "db": args.db}))
        return 0
    return 2
