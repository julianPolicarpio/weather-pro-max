"""QC for the IEM 1-minute ASOS archive in `obs_1min`.

WHY THIS EXISTS. The raw 1-minute extreme is NOT the value that settles. Measured
2026-08-12 over the full ingest (QUEUE item 13):

    KLAX high  exact match vs CLI 29.5%   modal gap +1F (56.4%)   mean +1.29F
    KMIA high  exact match vs CLI 31.7%   modal gap +1F (59.8%)   mean +0.90F
    KLAX low   exact match         68.6%  -1F on 24.4%            mean -0.80F
    KMIA low   exact match         69.0%  -1F on 21.9%            mean -0.58F

Two separate defects, and they need different treatment:

1. RESOLUTION. The official daily extreme comes from the QC'd 5-MINUTE
   observations. This archive is 1,440 samples a day, and the maximum of more
   samples is larger. That is why highs run hot and lows run cold -- the raw
   series is a WIDER range than the one that settles. A 5-minute rolling mean
   reproduces the official basis.

2. SPIKES. The 1-minute feed is not quality-controlled. ~7.5% of KLAX high days
   sit >=3F above settlement, with individual days at +36, +46, +47, +52, +56F,
   and lows at -89 and -136F. A rolling mean does not fix these -- it smears one
   bad sample across five windows -- so spikes must be removed BEFORE averaging.

⚠️ ORDER MATTERS: despike, then average -- but only for MODERATE faults, which
is the subtle part. A +50F spike is caught either way, because even after being
smeared across five windows it is still far from the local median. A +12F spike
averaged first becomes +2.4F, lands inside DESPIKE_TOL_F, and SURVIVES. Those
mid-sized faults are the ones that quietly move a daily extreme by a rung.

⚠️ GAPS ARE NOT ZEROS. Coverage is 1,244-1,312 of 1,440 minutes on a covered
day, so the series is reindexed onto a dense minute grid with NaN holes and
every window requires a minimum number of real samples. Averaging across a
two-hour hole would invent a reading.

Nothing here is wired into features. `db.py` explains why `obs_1min` is
deliberately not `obs`; that argument is unchanged by QC.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from datetime import date, datetime, timezone

import numpy as np

from highcast.config import AppConfig
from highcast.timeutil import climate_day_bounds

# A despike window of 11 minutes is wide enough that a 2-3 minute glitch cannot
# dominate its own median, and narrow enough that a real frontal passage (the
# fastest genuine signal at these stations) is not flattened.
DESPIKE_WINDOW = 11
# 5F from the local median in one minute is not meteorology at KLAX or KMIA.
# Deliberately loose: the point is to remove +46F sensor faults, not to shave
# legitimate tails. Tightening this is a modelling decision, not a QC one.
DESPIKE_TOL_F = 5.0
# The official basis. 5 one-minute samples -> the 5-minute observation.
MEAN_WINDOW = 5
MIN_SAMPLES = 3          # of MEAN_WINDOW, after despiking


def _to_minute_grid(valid_utc: list[str], tmpf: list[float]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Reindex an irregular series onto a dense per-minute grid, NaN for holes.

    Returns (epoch_minutes, values). Duplicate minutes keep the LAST value,
    matching `INSERT OR REPLACE` in the ingest.
    """
    if not valid_utc:
        return np.empty(0, dtype=np.int64), np.empty(0)
    mins = np.array([
        int(datetime.fromisoformat(v).replace(tzinfo=timezone.utc).timestamp()) // 60
        for v in valid_utc], dtype=np.int64)
    vals = np.asarray(tmpf, dtype=float)
    order = np.argsort(mins, kind="stable")
    mins, vals = mins[order], vals[order]
    lo, hi = int(mins[0]), int(mins[-1])
    grid = np.arange(lo, hi + 1, dtype=np.int64)
    out = np.full(grid.shape, np.nan)
    out[mins - lo] = vals            # later duplicates overwrite earlier ones
    return grid, out


@contextmanager
def _quiet():
    """All-NaN windows are EXPECTED, not exceptional — coverage is ~1,250 of
    1,440 minutes, so every gap produces one. numpy warns per call; silencing it
    here keeps a real warning elsewhere visible instead of buried in noise.

    Must be a real context manager: an earlier version returned an already
    entered `catch_warnings()`, which raises "Cannot enter twice" on `with`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _rolling_nanmedian(x: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    padded = np.concatenate([np.full(pad, np.nan), x, np.full(pad, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, w)
    with _quiet():
        return np.nanmedian(view, axis=1)


def _rolling_nanmean(x: np.ndarray, w: int, min_samples: int) -> np.ndarray:
    pad = w // 2
    padded = np.concatenate([np.full(pad, np.nan), x, np.full(pad, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, w)
    counts = np.count_nonzero(~np.isnan(view), axis=1)
    with _quiet():
        means = np.nanmean(view, axis=1)
    means[counts < min_samples] = np.nan
    return means


def despike(vals: np.ndarray, window: int = DESPIKE_WINDOW,
            tol: float = DESPIKE_TOL_F) -> np.ndarray:
    """Drop samples that sit more than `tol` from their local median."""
    med = _rolling_nanmedian(vals, window)
    bad = np.abs(vals - med) > tol
    out = vals.copy()
    out[bad] = np.nan
    return out


def qc_series(valid_utc: list[str], tmpf: list[float], *,
              do_despike: bool = True, do_mean: bool = True
              ) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: minute grid -> despike -> 5-minute mean.

    The two flags exist so a measurement can attribute the improvement to each
    step rather than reporting only the combination. Shipped use wants both.
    """
    grid, vals = _to_minute_grid(valid_utc, tmpf)
    if grid.size == 0:
        return grid, vals
    if do_despike:
        vals = despike(vals)
    if do_mean:
        vals = _rolling_nanmean(vals, MEAN_WINDOW, MIN_SAMPLES)
    return grid, vals


def load_station(conn, icao: str) -> tuple[list[str], list[float]]:
    rows = conn.execute(
        "SELECT valid_utc, tmpf FROM obs_1min WHERE station=? ORDER BY valid_utc",
        (icao,)).fetchall()
    return [r["valid_utc"] for r in rows], [r["tmpf"] for r in rows]


def daily_extremes(cfg: AppConfig, grid: np.ndarray, vals: np.ndarray,
                   days: list[date], extremum: str,
                   min_minutes: int = 1000) -> dict[str, float]:
    """Extreme of `vals` inside each CLIMATE day (local standard midnight).

    `min_minutes` drops thinly-covered days rather than reporting an extreme
    from a fragment — a day with 200 surviving minutes has probably missed the
    peak entirely, and scoring it would understate the method.
    """
    out: dict[str, float] = {}
    agg = np.nanmax if extremum == "max" else np.nanmin
    for d in days:
        start, stop = climate_day_bounds(d, cfg.station.std_utc_offset)
        a = int(start.timestamp()) // 60
        b = int(stop.timestamp()) // 60
        i0, i1 = np.searchsorted(grid, a), np.searchsorted(grid, b)
        if i1 <= i0:
            continue
        seg = vals[i0:i1]
        n = int(np.count_nonzero(~np.isnan(seg)))
        if n < min_minutes:
            continue
        with _quiet():
            out[d.isoformat()] = float(agg(seg))
    return out
