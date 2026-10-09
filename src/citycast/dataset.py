"""Per-metric training frame: one row per (station, climate day, as-of point) — spec §3.2–3.4."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from citycast.asof import asof_points
from citycast.features import FEATURE_COLS
from citycast.fingerprint import code_fingerprint

DATASET_DIR = Path("data/citycast")
KEY_COLS = ["station", "metric", "event_date", "asof_utc"]


def build(ctx, metric: str, since: date, until: date, icaos=None, progress=print) -> pd.DataFrame:
    rows = []
    for icao in (icaos or sorted(ctx.stations)):
        st = ctx.stations[icao]
        day = since
        while day <= until:
            y = ctx.truth(icao, metric, day)
            for t in asof_points(st, day):
                r = ctx.row(icao, metric, day, t)
                if r is not None:
                    r["y"] = y
                    rows.append(r)
            day += timedelta(days=1)
        progress(f"{metric} {icao}: {len(rows)} rows so far")
    df = pd.DataFrame(rows, columns=KEY_COLS + FEATURE_COLS + ["y"])
    return df.sort_values(["station", "event_date", "asof_utc"]).reset_index(drop=True)


def _path(metric: str) -> Path:
    return DATASET_DIR / f"dataset-{metric}.pkl"


def save(df: pd.DataFrame, metric: str) -> Path:
    df.attrs["fingerprint"] = code_fingerprint()   # walkforward records it; evaluate refuses a stale one
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(_path(metric))
    return _path(metric)


def load(metric: str) -> pd.DataFrame:
    return pd.read_pickle(_path(metric))
