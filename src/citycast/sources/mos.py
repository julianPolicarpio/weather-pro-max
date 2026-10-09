"""IEM MOS archive CSV -> mos_runs, keeping each run's UTC issue time (spec §2.3).

Conventions match highcast.fetchers.forecasts: climate day = (ftime_utc + std offset).date();
MEX/GFS carry the daily max/min in `n_x` on local-afternoon/-morning rows; NBS/NBE have no
n_x, so the max/min is taken over the run's 3-hourly `tmp` points inside the climate day.
"""
import csv
import io
from datetime import date, datetime, timedelta, timezone
from functools import partial
from zoneinfo import ZoneInfo

from citycast.backfill import Chunk

IEM_MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
MODELS = ("MEX", "GFS", "NBS", "NBE")
N_X_MODELS = {"MEX", "GFS"}
_MISSING = (None, "", "M")


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(str(s).strip()[:19].replace(" ", "T")).replace(tzinfo=timezone.utc)


def parse_mos_runs(text: str, st, model: str, fetched_at: str) -> list[dict]:
    zone = ZoneInfo(st.tz)
    agg: dict[tuple[str, str, str], list[float]] = {}
    for item in csv.DictReader(io.StringIO(text)):
        runtime, ftime = item.get("runtime"), item.get("ftime")
        if not runtime or not ftime:
            continue
        ft = _utc(ftime)
        day = (ft + timedelta(hours=st.std_utc_offset)).date().isoformat()
        rt = _utc(runtime).isoformat()
        if model in N_X_MODELS:
            v = item.get("n_x")
            if v in _MISSING:
                continue
            metric = "high" if ft.astimezone(zone).hour >= 12 else "low"
            agg.setdefault((rt, day, metric), []).append(float(v))
        else:
            v = item.get("tmp")
            if v in _MISSING:
                continue
            for metric in ("high", "low"):
                agg.setdefault((rt, day, metric), []).append(float(v))
    return [{"station": st.icao, "model": model, "runtime_utc": rt, "target_date": day,
             "metric": metric, "value_f": max(v) if metric == "high" else min(v),
             "n_points": len(v), "fetched_at": fetched_at}
            for (rt, day, metric), v in sorted(agg.items())]


def _half_years(since: date, today: date):
    start = date(since.year, 1 if since.month <= 6 else 7, 1)
    while start <= today:
        end = date(start.year, 7, 1) if start.month == 1 else date(start.year + 1, 1, 1)
        yield start, end
        start = end


def _fetch(paced, st, model: str, sts: str, ets: str, now_iso) -> list[tuple[str, list[dict]]]:
    text = paced.get_text(IEM_MOS, {"station": st.icao, "model": model, "sts": sts, "ets": ets, "format": "csv"})
    return [("mos_runs", parse_mos_runs(text, st, model, now_iso()))]


def chunks(paced, stations: dict, since: date, today: date, now_iso, models=MODELS,
           complete_before: date | None = None) -> list[Chunk]:
    complete_before = complete_before or today
    out = []
    for st in stations.values():
        for model in models:
            for a, b in _half_years(since, today):
                run = partial(_fetch, paced, st, model, f"{a.isoformat()}T00:00Z", f"{b.isoformat()}T00:00Z",
                              now_iso)
                out.append(Chunk("mos", f"{st.icao}:{model}:{a.isoformat()}", b > complete_before, run))
    return out


def recent_chunks(paced, stations: dict, now: datetime, now_iso, days: int = 3, models=MODELS) -> list[Chunk]:
    """Runs issued in the last `days` days, for `citycast predict`: current, so never logged (a logged
    chunk would be skipped by the backfill), and small (a half-year chunk refetches months per call)."""
    now = now.astimezone(timezone.utc)
    sts = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")
    return [Chunk("mos", f"{st.icao}:{model}:recent", True, partial(_fetch, paced, st, model, sts, ets, now_iso))
            for st in stations.values() for model in models]
