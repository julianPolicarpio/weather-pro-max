"""Selection, gate and market comparison (spec §3.8)."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from citycast.fingerprint import code_fingerprint
from citycast.metrics import brier, cluster_bootstrap, rps
from highcast.buckets import pmf_mass

SELECTION_END = "2026-04-01"
MARKET_STALE_H = 2
REPORT_MIN_DAYS = 60


def check_inputs(conn, metric: str) -> None:
    """Refuse to score wf_predictions that are partial (a crashed or running walk-forward) or were
    produced by a dataset or walk-forward under code other than the current code (spec §3.8)."""
    run = conn.execute("SELECT fingerprint, dataset_fingerprint, started_at, finished_at, rows FROM wf_runs "
                       "WHERE metric=?", (metric,)).fetchone()
    if run is None:
        raise RuntimeError(f"{metric}: no walk-forward run is recorded; run `citycast walkforward --metric {metric}`")
    fp, ds_fp, started, finished, rows = run
    if finished is None:
        raise RuntimeError(f"{metric}: the walk-forward run started {started} never finished; rerun it")
    n = conn.execute("SELECT COUNT(*) FROM wf_predictions WHERE metric=?", (metric,)).fetchone()[0]
    if rows != n:
        raise RuntimeError(f"{metric}: wf_predictions holds {n} rows but the walk-forward run wrote {rows}; "
                           "rerun it")
    cur = code_fingerprint()
    if fp != cur:
        raise RuntimeError(f"{metric}: walk-forward ran under different code ({fp[:12]}, current {cur[:12]}); "
                           "rerun it")
    if ds_fp != cur:
        raise RuntimeError(f"{metric}: the dataset was built under different code ({(ds_fp or 'none')[:12]}, "
                           f"current {cur[:12]}); rebuild it with `citycast dataset` and rerun walk-forward")


def _truth(conn, metric) -> dict:
    col = "high_f" if metric == "high" else "low_f"
    return {(s, d): v for s, d, v in conn.execute(
        f"SELECT station, date, {col} FROM cli_daily WHERE final=1 AND {col} IS NOT NULL")}


def scored(conn, metric: str) -> pd.DataFrame:
    truth = _truth(conn, metric)
    rows = []
    for s, d, a, v, pj in conn.execute("SELECT station, event_date, asof_utc, variant, pmf_json FROM wf_predictions "
                                       "WHERE metric=?", (metric,)):
        y = truth.get((s, d))
        if y is None:
            continue
        pmf = {int(k): p for k, p in json.loads(pj).items()}
        rows.append({"station": s, "metric": metric, "event_date": d, "asof_utc": a, "variant": v,
                     "rps": rps(pmf, int(y)), "pmf": pmf, "y": int(y)})
    return pd.DataFrame(rows, columns=["station", "metric", "event_date", "asof_utc", "variant", "rps", "pmf", "y"])


def _paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    keys = ["station", "event_date", "asof_utc"]
    left = df[df["variant"] == a][keys + ["rps"]].rename(columns={"rps": "a"})
    right = df[df["variant"] == b][keys + ["rps"]].rename(columns={"rps": "b"})
    return left.merge(right, on=keys)


def select_and_gate(conn, metric: str, now_iso) -> list[dict]:
    df = scored(conn, metric)
    sel = df[df["event_date"] < SELECTION_END]
    rep = df[df["event_date"] >= SELECTION_END]
    out, report_scored = [], 0
    for station in sorted(sel["station"].unique()):
        s = sel[sel["station"] == station]
        both = _paired(s, "city", "shared")
        variant = "city" if len(both) and both["a"].mean() < both["b"].mean() else "shared"
        gate = _paired(s, variant, "mosnorm")
        if gate.empty:
            continue
        by_day: dict[str, list[float]] = {}
        for r in gate.itertuples(index=False):
            by_day.setdefault(r.event_date, []).append(r.a - r.b)
        _, _, ci_hi = cluster_bootstrap(by_day)
        conf = _paired(rep[rep["station"] == station], variant, "mosnorm")   # report period: confirm only
        rep_days = int(conf["event_date"].nunique())
        report_scored |= rep_days > 0
        sel_pass = ci_hi < 0
        rep_ok = rep_days >= REPORT_MIN_DAYS and float((conf["a"] - conf["b"]).mean()) < 0
        row = {"station": station, "metric": metric, "variant": variant,
               "crps_sel": float(gate["a"].mean()), "mos_crps_sel": float(gate["b"].mean()),
               "gate_ci_hi": ci_hi, "sel_pass": int(sel_pass),
               "crps_rep": float(conf["a"].mean()) if rep_days else None,
               "mos_crps_rep": float(conf["b"].mean()) if rep_days else None,
               "rep_days": rep_days, "gate_pass": int(sel_pass and rep_ok), "decided_at": now_iso()}
        out.append(row)
    with conn:
        conn.execute("DELETE FROM model_choice WHERE metric=?", (metric,))
        conn.executemany("INSERT OR REPLACE INTO model_choice VALUES (:station,:metric,:variant,:crps_sel,"
                         ":mos_crps_sel,:gate_ci_hi,:sel_pass,:crps_rep,:mos_crps_rep,:rep_days,:gate_pass,"
                         ":decided_at)", out)
        conn.execute("INSERT OR REPLACE INTO eval_log VALUES (?,?,?,?)",
                     (now_iso(), metric, code_fingerprint(), int(report_scored)))
    return out


def canonical_events(conn) -> set[str]:
    """One event per (station, metric, event_date): most markets, ties -> smallest event_ticker.
    (The 2a archive has KMDW 2024-02-28 listed under two events; spec §4.1.)"""
    best: dict[tuple, tuple[int, str]] = {}
    for st, metric, d, ev, n in conn.execute(
            "SELECT station, metric, event_date, event_ticker, COUNT(*) FROM kalshi_markets "
            "GROUP BY station, metric, event_date, event_ticker"):
        key = (st, metric, d)
        if key not in best or (-n, ev) < (-best[key][0], best[key][1]):
            best[key] = (n, ev)
    return {ev for _, ev in best.values()}


def market_brier(conn, metric: str, period: str | None = None) -> dict:
    """period None = all dates; 'selection' = before SELECTION_END; 'report' = on/after."""
    choice = {s: v for s, v in conn.execute("SELECT station, variant FROM model_choice WHERE metric=?", (metric,))}
    df = scored(conn, metric)
    df = df[df.apply(lambda r: choice.get(r.station) == r.variant, axis=1)] if len(df) else df
    if period == "selection":
        df = df[df["event_date"] < SELECTION_END]
    elif period == "report":
        df = df[df["event_date"] >= SELECTION_END]
    canon = canonical_events(conn)
    markets = conn.execute("SELECT ticker, event_ticker, station, event_date, lo, hi, result FROM kalshi_markets "
                           "WHERE metric=? AND NOT (lo IS NULL AND hi IS NULL) AND result IN ('yes','no')",
                           (metric,)).fetchall()
    by_event: dict[tuple[str, str], list] = {}
    for m in markets:
        if m[1] in canon:
            by_event.setdefault((m[2], m[3]), []).append(m)
    diffs: dict[str, dict[str, list[float]]] = {}
    for r in df.itertuples(index=False):
        t = datetime.fromisoformat(r.asof_utc)
        lo_iso = (t - timedelta(hours=MARKET_STALE_H)).isoformat()
        for ticker, event_ticker, _, _, lo, hi, result in by_event.get((r.station, r.event_date), []):
            c = conn.execute("SELECT yes_bid, yes_ask FROM kalshi_candles WHERE ticker=? AND end_utc>? "
                             "AND end_utc<=? AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL "
                             "ORDER BY end_utc DESC LIMIT 1", (ticker, lo_iso, r.asof_utc)).fetchone()
            if c is None:
                continue
            won = 1 if result == "yes" else 0          # Kalshi's own settlement (spec §4.1)
            p_model = pmf_mass(r.pmf, lo, hi) or 0.0
            mid = (c[0] + c[1]) / 2
            diffs.setdefault(r.station, {}).setdefault(event_ticker, []).append(
                brier(p_model, won) - brier(mid, won))
    out = {}
    for station, clusters in diffs.items():
        mean, lo, hi = cluster_bootstrap(clusters)
        out[station] = (mean, lo, hi, sum(len(v) for v in clusters.values()))
    return out


def _freeze_warning(conn) -> list[str]:
    first = conn.execute("SELECT run_at, fingerprint FROM eval_log WHERE report_scored=1 "
                         "ORDER BY run_at LIMIT 1").fetchone()
    if first is None:
        return []
    seen = {r[0] for r in conn.execute("SELECT DISTINCT fingerprint FROM eval_log WHERE run_at>=?", (first[0],))}
    if len(seen) <= 1 and first[1] == code_fingerprint():
        return []
    return [f"> **WARNING:** the report period was first scored on {first[0]} under different modelling "
            "code. It is no longer an untouched test; only the sandbox forward test counts.", ""]


def write_report(conn, path: Path) -> Path:
    lines = ["# citycast evaluation", "", *_freeze_warning(conn),
             f"Selection period: event dates before {SELECTION_END}. Report period: on/after. "
             "RPS = discrete CRPS (lower is better). Gate = selection CI hi < 0 AND report-period mean "
             f"below mosnorm over >= {REPORT_MIN_DAYS} days. "
             "Market Brier diff < 0 means the model beat the market mid. "
             "Selection-period columns are in-sample for the variant choice; the report-period columns "
             "are the out-of-sample check.", ""]
    for metric in ("high", "low"):
        lines += [f"## {metric}", "",
                  "| station | variant | RPS (sel) | mosnorm RPS (sel) | gate CI hi | RPS (rep) | "
                  "mosnorm RPS (rep) | rep days | gate | market Δ sel [CI] (n) | market Δ report [CI] (n) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        sel_m, rep_m = market_brier(conn, metric, "selection"), market_brier(conn, metric, "report")
        for row in conn.execute("SELECT station, variant, crps_sel, mos_crps_sel, gate_ci_hi, crps_rep, "
                                "mos_crps_rep, rep_days, sel_pass, gate_pass FROM model_choice WHERE metric=? "
                                "ORDER BY station", (metric,)):
            def fmt(x):
                return f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}] ({x[3]})" if x else "no prices"
            def num(x):
                return "—" if x is None else f"{x:.3f}"
            verdict = "PASS" if row[9] else ("fail (report)" if row[8] else "fail")
            lines.append(f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:+.3f} | "
                         f"{num(row[5])} | {num(row[6])} | {row[7]} | {verdict} | "
                         f"{fmt(sel_m.get(row[0]))} | {fmt(rep_m.get(row[0]))} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
