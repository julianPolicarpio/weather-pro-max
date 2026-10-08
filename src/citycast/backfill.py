# src/citycast/backfill.py
"""Resumable chunk runner (spec §2.4): one transaction and one log row per finished chunk."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def _store(conn, ch, results, summary, now_iso, progress) -> None:
    """Upsert one chunk's results (calling thread only); record success or error."""
    n = 0
    try:
        with conn:   # one transaction: a write error rolls back this chunk only
            for table, rows in results:
                n += db.upsert(conn, table, rows)
            # An empty answer may be a transient outage, so it is not logged as done:
            # the next run asks again (a truly empty chunk costs one request per run).
            if not ch.current and n > 0:
                conn.execute("INSERT OR REPLACE INTO backfill_log VALUES (?,?,?,?)",
                             (ch.source, ch.key, now_iso(), n))
    except Exception as e:  # noqa: BLE001
        _error(ch, e, summary, progress)
        return
    summary["done"] += 1
    summary["rows"] += n
    progress(f"{ch.source} {ch.key}: {n} rows")


def _error(ch, e, summary, progress) -> None:
    summary["errors"].append({"source": ch.source, "key": ch.key,
                              "error": f"{type(e).__name__}: {e}"})
    progress(f"ERROR {ch.source} {ch.key}: {type(e).__name__}: {e}")


def run_chunks(conn, chunks, *, refresh: bool = False, now_iso: Callable[[], str],
               progress=print, workers: int = 1) -> dict:
    summary = {"done": 0, "skipped": 0, "rows": 0, "errors": []}

    def todo():
        for ch in chunks:
            if not refresh and not ch.current and is_logged(conn, ch.source, ch.key):
                summary["skipped"] += 1
                continue
            yield ch

    if workers <= 1:
        for ch in todo():
            try:
                results = ch.run()
            except Exception as e:  # noqa: BLE001 — one bad chunk must not stop the backfill
                _error(ch, e, summary, progress)
                continue
            _store(conn, ch, results, summary, now_iso, progress)
        return summary

    it = todo()
    pending: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        exhausted = False
        while True:
            while not exhausted and len(pending) < 2 * workers:
                ch = next(it, None)
                if ch is None:
                    exhausted = True
                    break
                pending[ex.submit(ch.run)] = ch
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                ch = pending.pop(f)
                try:
                    results = f.result()
                except Exception as e:  # noqa: BLE001
                    _error(ch, e, summary, progress)
                    continue
                _store(conn, ch, results, summary, now_iso, progress)
    return summary
