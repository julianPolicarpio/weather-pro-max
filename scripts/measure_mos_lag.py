# scripts/measure_mos_lag.py
"""Poll IEM's MOS archive (the endpoint the backfill uses) every 15 min for 24 h and record when
each new run first appears. Lag = first_seen - runtime. One request per station/model per poll."""
import csv
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATIONS = ("KNYC", "KLAX", "KDEN")
MODELS = ("GFS", "MEX", "NBS", "NBE")
POLL_S = 900
HOURS = 24


def latest_runtimes(station: str, model: str) -> set[str]:
    now = datetime.now(timezone.utc)
    text = httpx.get(URL, params={"station": station, "model": model, "format": "csv",
                                  "sts": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ"),
                                  "ets": (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")},
                     timeout=120).text
    if not text.startswith("runtime"):   # throttled ("Too many requests") or an error page
        raise RuntimeError(f"not a MOS CSV: {text[:80]!r}")
    return {row["runtime"][:16] for row in csv.DictReader(io.StringIO(text)) if row.get("runtime")}


def main() -> int:
    seen: dict[tuple[str, str, str], datetime] = {}
    first_poll = True
    end = time.time() + HOURS * 3600
    while time.time() < end:
        now = datetime.now(timezone.utc)
        for st in STATIONS:
            for m in MODELS:
                try:
                    for rt in latest_runtimes(st, m):
                        # runs already present at the first poll have unknown first-seen times
                        seen.setdefault((st, m, rt), None if first_poll else now)
                except Exception as e:  # noqa: BLE001 - one failed poll is not fatal
                    print(f"poll error {st} {m}: {e}", file=sys.stderr)
                time.sleep(5.0)   # IEM throttles bulk MOS requests even at 1/s
        first_poll = False
        time.sleep(POLL_S)
    lags: dict[str, list[float]] = {}
    for (st, m, rt), t in seen.items():
        if t is not None:
            run = datetime.fromisoformat(rt.replace(" ", "T")).replace(tzinfo=timezone.utc)
            lags.setdefault(m, []).append((t - run).total_seconds() / 3600)
    out = Path(f"docs/reports/mos-publication-lag-{datetime.now(timezone.utc).date()}.md")
    lines = ["# MOS publication lag (IEM archive first-seen, 15-min polls)", "",
             "| model | runs seen | min h | max h | proposed PUB_LAG_H (max + 0.5, poll-rounded) |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        v = sorted(lags.get(m, []))
        if v:
            lines.append(f"| {m} | {len(v)} | {v[0]:.2f} | {v[-1]:.2f} | {v[-1] + 0.5:.1f} |")
        else:
            lines.append(f"| {m} | 0 | – | – | keep default |")
    out.write_text("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
