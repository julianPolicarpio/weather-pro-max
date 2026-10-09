"""Quarterly walk-forward (spec §3.6–3.7): every stored prediction is out of sample."""
import json
import math
from datetime import date, timedelta

import pandas as pd

from citycast import db
from citycast.fingerprint import code_fingerprint
from citycast.metrics import normal_pmf, rps
from citycast.qmodel import QUANTILES, QuantileModel, pmf_from_quantiles
from highcast.buckets import truncate_at_bound

K_GRID = tuple(round(0.6 + 0.1 * i, 1) for i in range(11))
MIN_TRAIN_ROWS = 1000
MIN_CITY_ROWS = 300
MIN_K_SPAN_D = 180
EMBARGO_D = 1       # spec §3.7: y(start-1) is unknown at the as-of points of day `start`


def quarters(first: date, today: date) -> list[tuple[str, date, date]]:
    out = []
    start = date(first.year, 3 * ((first.month - 1) // 3) + 1, 1)
    while True:
        m = start.month + 3
        end = date(start.year + (m > 12), (m - 1) % 12 + 1, 1)
        if end > today:
            return out
        out.append((f"{start.year}Q{(start.month - 1) // 3 + 1}", start, end))
        start = end


def choose_k(prev: list[tuple[dict, float, float]], metric: str) -> float:
    if not prev:
        return 1.0
    best, best_score = 1.0, math.inf
    for k in K_GRID:
        score = sum(rps(pmf_from_quantiles(q, k, b, metric), int(y)) for q, b, y in prev) / len(prev)
        if score < best_score - 1e-12:
            best, best_score = k, score
    return best


def _prev(conn, station, metric, variant, before: date, truth: dict) -> list:
    rows = conn.execute("SELECT event_date, quantiles_json, run_ext FROM wf_predictions WHERE station=? "
                        "AND metric=? AND variant=? AND event_date<?",
                        (station, metric, variant, before.isoformat())).fetchall()
    if not rows or (before - date.fromisoformat(min(r[0] for r in rows))).days < MIN_K_SPAN_D:
        return []
    out = []
    for d, qj, b in rows:
        y = truth.get((station, d))
        if y is not None:
            q = {float(a): v for a, v in json.loads(qj).items()}
            out.append((q, float("nan") if b is None else b, y))
    return out


def _rows(test: pd.DataFrame, qmat, metric, variant, fold, k_by_station, now) -> list[dict]:
    out = []
    for i, r in enumerate(test.itertuples(index=False)):
        q = dict(zip(QUANTILES, (float(v) for v in qmat[i])))
        k = k_by_station[r.station]
        pmf = pmf_from_quantiles(q, k, r.run_ext, metric)
        out.append({"station": r.station, "metric": metric, "event_date": r.event_date,
                    "asof_utc": r.asof_utc, "variant": variant, "fold": fold, "anchor": r.anchor,
                    "run_ext": None if math.isnan(r.run_ext) else r.run_ext,
                    "quantiles_json": json.dumps(q), "pmf_json": json.dumps(pmf), "k": k,
                    "created_at": now})
    return out


def run(conn, df: pd.DataFrame, metric: str, stations: tuple[str, ...], now_iso,
        train_start: date = date(2022, 1, 1), first_fold: date = date(2022, 7, 1),
        today: date | None = None, progress=print, fingerprint: str | None = None) -> int:
    """`fingerprint` is the code hash the command took at its start; None hashes the code here, once, before
    anything else runs. Nothing below hashes again, so an edit during a long run cannot change what is recorded."""
    code_fp = fingerprint if fingerprint is not None else code_fingerprint()
    dataset_fp = df.attrs.get("fingerprint")
    df = df[df["metric"] == metric] if "metric" in df else df
    known = df[df["y"].notna()]
    if (known.groupby(["station", "event_date"])["y"].nunique() > 1).any():
        raise ValueError("inconsistent y within a (station, event_date)")
    truth = {(r.station, r.event_date): int(r.y) for r in known[["station", "event_date", "y"]]
             .drop_duplicates(["station", "event_date"]).itertuples(index=False)}
    with conn:                       # a run is reproducible from the current code, never from stale rows
        conn.execute("DELETE FROM wf_predictions WHERE metric=?", (metric,))
        # finished_at stays NULL until the last fold: evaluate refuses a crashed or running run
        conn.execute("INSERT OR REPLACE INTO wf_runs VALUES (?,?,?,?,NULL,NULL)",
                     (metric, code_fp, dataset_fp, now_iso()))
    written = 0
    for label, start, end in quarters(first_fold, today or date.today()):
        cutoff = start - timedelta(days=EMBARGO_D)
        train = known[(known["event_date"] >= train_start.isoformat()) & (known["event_date"] < cutoff.isoformat())]
        test = known[(known["event_date"] >= start.isoformat()) & (known["event_date"] < end.isoformat())]
        if test.empty:
            progress(f"{metric} {label}: skipped (no test rows)")
            continue
        if len(train) < MIN_TRAIN_ROWS:
            progress(f"{metric} {label}: skipped ({len(train)} train rows)")
            continue
        now = now_iso()
        rows: list[dict] = []
        shared = QuantileModel("shared", stations).fit(train, start)
        k_shared = {s: choose_k(_prev(conn, s, metric, "shared", cutoff, truth), metric) for s in test["station"].unique()}
        rows += _rows(test, shared.predict_y_quantiles(test), metric, "shared", label, k_shared, now)
        for s in sorted(test["station"].unique()):
            tr, te = train[train["station"] == s], test[test["station"] == s]
            if len(tr) < MIN_CITY_ROWS:
                continue
            city = QuantileModel("city", stations).fit(tr, start)
            k_city = {s: choose_k(_prev(conn, s, metric, "city", cutoff, truth), metric)}
            rows += _rows(te, city.predict_y_quantiles(te), metric, "city", label, k_city, now)
        for r in test[test["f_MEX"].notna()].itertuples(index=False):
            pmf = normal_pmf(r.f_MEX)
            if not math.isnan(r.run_ext):   # same-day: the baseline gets the bound the model gets (I2)
                pmf = truncate_at_bound(pmf, r.run_ext, "max" if metric == "high" else "min")
            rows.append({"station": r.station, "metric": metric, "event_date": r.event_date,
                         "asof_utc": r.asof_utc, "variant": "mosnorm", "fold": label, "anchor": r.anchor,
                         "run_ext": None if math.isnan(r.run_ext) else r.run_ext, "quantiles_json": "{}",
                         "pmf_json": json.dumps(pmf), "k": 1.0, "created_at": now})
        with conn:
            written += db.upsert(conn, "wf_predictions", rows)
        progress(f"{metric} {label}: {len(rows)} predictions")
    with conn:
        conn.execute("UPDATE wf_runs SET finished_at=?, rows=? WHERE metric=?", (now_iso(), written, metric))
    return written
