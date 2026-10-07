# src/citycast/backfill.py
"""Resumable chunk runner (spec §2.4): one transaction and one log row per finished chunk."""
from dataclasses import dataclass
from typing import Callable

from citycast import db


@dataclass(frozen=True)
class Chunk:
    source: str
    key: str
    current: bool       # still filling (e.g. this year): always re-run, never logged as done
    run: Callable[[], list[tuple[str, list[dict]]]]


def is_logged(conn, source: str, key: str) -> bool:
    return conn.execute("SELECT 1 FROM backfill_log WHERE source=? AND key=?",
                        (source, key)).fetchone() is not None


def run_chunks(conn, chunks, *, refresh: bool = False, now_iso: Callable[[], str],
               progress=print) -> dict:
    summary = {"done": 0, "skipped": 0, "rows": 0, "errors": []}
    for ch in chunks:
        if not refresh and not ch.current and is_logged(conn, ch.source, ch.key):
            summary["skipped"] += 1
            continue
        try:
            results = ch.run()
        except Exception as e:  # noqa: BLE001 — one bad chunk must not stop the backfill
            summary["errors"].append({"source": ch.source, "key": ch.key,
                                      "error": f"{type(e).__name__}: {e}"})
            progress(f"ERROR {ch.source} {ch.key}: {type(e).__name__}: {e}")
            continue
        n = 0
        with conn:
            for table, rows in results:
                n += db.upsert(conn, table, rows)
            if not ch.current:
                conn.execute("INSERT OR REPLACE INTO backfill_log VALUES (?,?,?,?)",
                             (ch.source, ch.key, now_iso(), n))
        summary["done"] += 1
        summary["rows"] += n
        progress(f"{ch.source} {ch.key}: {n} rows")
    return summary
