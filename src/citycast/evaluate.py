"""Selection, gate and market comparison (spec §3.8)."""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from citycast.fingerprint import code_fingerprint
from citycast.metrics import brier, cluster_bootstrap, rps
from highcast.buckets import pmf_mass

SELECTION_END = "2026-04-01"
MARKET_STALE_H = 2
REPORT_MIN_DAYS = 60
METRICS = ("high", "low")
_STR_COLS = ("station", "event_date", "asof_utc", "variant")


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


def scored(conn, metric: str, variants: tuple[str, ...] | None = None, keep_pmf: bool = False,
           station: str | None = None) -> pd.DataFrame:
    """Every wf_predictions row with a final CLI truth, scored by RPS. Rows stream from SQL (variant and
    station filters run there); a parsed pmf (~3 KB a row) is kept only when keep_pmf."""
    truth = _truth(conn, metric)
    sql, params = ("SELECT station, event_date, asof_utc, variant, run_ext, pmf_json FROM wf_predictions "
                   "WHERE metric=?"), [metric]
    if station is not None:
        sql += " AND station=?"
        params.append(station)
    if variants is not None:
        sql += f" AND variant IN ({','.join('?' * len(variants))})"
        params += list(variants)
    cols: dict[str, list] = {c: [] for c in (*_STR_COLS, "rps", "y", "run_ext")}
    pmfs, one = [], {}
    for s, d, a, v, b, pj in conn.execute(sql, params):
        y = truth.get((s, d))
        if y is None:
            continue
        pmf = {int(k): p for k, p in json.loads(pj).items()}
        for col, x in zip(_STR_COLS, (s, d, a, v)):
            cols[col].append(one.setdefault(x, x))          # one object per distinct string, not per row
        cols["rps"].append(rps(pmf, int(y)))
        cols["y"].append(int(y))
        cols["run_ext"].append(math.nan if b is None else float(b))
        if keep_pmf:
            pmfs.append(pmf)
    df = pd.DataFrame(cols).astype({**{c: str for c in _STR_COLS}, "rps": float, "y": "int64",
                                    "run_ext": float})       # typed even when empty
    df.insert(1, "metric", metric)
    if keep_pmf:
        df.insert(df.columns.get_loc("y"), "pmf", pmfs)
    return df


def _paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    keys = ["station", "event_date", "asof_utc"]
    left = df[df["variant"] == a][keys + ["rps", "run_ext"]].rename(columns={"rps": "a"})
    right = df[df["variant"] == b][keys + ["rps"]].rename(columns={"rps": "b"})
    return left.merge(right, on=keys)


def _choose(df: pd.DataFrame, metric: str) -> list[dict]:
    sel = df[df["event_date"] < SELECTION_END]
    rep = df[df["event_date"] >= SELECTION_END]
    out = []
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
        sel_pass = ci_hi < 0
        rep_ok = rep_days >= REPORT_MIN_DAYS and float((conf["a"] - conf["b"]).mean()) < 0
        out.append({"station": station, "metric": metric, "variant": variant,
                    "crps_sel": float(gate["a"].mean()), "mos_crps_sel": float(gate["b"].mean()),
                    "gate_ci_hi": ci_hi, "sel_pass": int(sel_pass),
                    "crps_rep": float(conf["a"].mean()) if rep_days else None,
                    "mos_crps_rep": float(conf["b"].mean()) if rep_days else None,
                    "rep_days": rep_days, "gate_pass": int(sel_pass and rep_ok)})
    return out


def choose_and_gate(conn, metric: str) -> list[dict]:
    """The model_choice rows of one metric, computed but not written (evaluate_all stamps decided_at and
    writes them). The variant is chosen on the selection period only; the report period only confirms."""
    return _choose(scored(conn, metric), metric)


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


def market_brier(conn, metric: str, choice: dict[str, str]) -> dict[str, dict[str, tuple]]:
    """Model minus market-mid Brier of each station's chosen variant (choice: station -> variant), both
    periods from one pass: {"selection": {station: (mean, lo, hi, n)}, "report": {...}}. pmfs are loaded
    one station and one variant at a time."""
    canon = canonical_events(conn)
    markets = conn.execute("SELECT ticker, event_ticker, station, event_date, lo, hi, result FROM kalshi_markets "
                           "WHERE metric=? AND NOT (lo IS NULL AND hi IS NULL) AND result IN ('yes','no')",
                           (metric,)).fetchall()
    by_event: dict[tuple[str, str], list] = {}
    for m in markets:
        if m[1] in canon:
            by_event.setdefault((m[2], m[3]), []).append(m)
    out: dict[str, dict[str, tuple]] = {"selection": {}, "report": {}}
    for station, variant in sorted(choice.items()):
        diffs: dict[str, dict[str, list[float]]] = {"selection": {}, "report": {}}
        for r in scored(conn, metric, (variant,), keep_pmf=True, station=station).itertuples(index=False):
            events = by_event.get((r.station, r.event_date))
            if not events:
                continue
            period = "selection" if r.event_date < SELECTION_END else "report"
            t = datetime.fromisoformat(r.asof_utc)
            lo_iso = (t - timedelta(hours=MARKET_STALE_H)).isoformat()
            for ticker, event_ticker, _, _, lo, hi, result in events:
                c = conn.execute("SELECT yes_bid, yes_ask FROM kalshi_candles WHERE ticker=? AND end_utc>? "
                                 "AND end_utc<=? AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL "
                                 "ORDER BY end_utc DESC LIMIT 1", (ticker, lo_iso, r.asof_utc)).fetchone()
                if c is None:
                    continue
                won = 1 if result == "yes" else 0          # Kalshi's own settlement (spec §4.1)
                p_model = pmf_mass(r.pmf, lo, hi) or 0.0
                mid = (c[0] + c[1]) / 2
                diffs[period].setdefault(event_ticker, []).append(brier(p_model, won) - brier(mid, won))
        for period, clusters in diffs.items():
            if clusters:
                mean, lo, hi = cluster_bootstrap(clusters)
                out[period][station] = (mean, lo, hi, sum(len(v) for v in clusters.values()))
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


def write_report(results: dict[str, dict], warning: list[str]) -> str:
    """Render already-computed results (pure: reads no DB, writes no file).
    results[metric] = {"choices": [model_choice rows], "market": market_brier(...)}."""
    lines = ["# citycast evaluation", "", *warning,
             f"Selection period: event dates before {SELECTION_END}. Report period: on/after. "
             "RPS = discrete CRPS (lower is better). Gate = selection CI hi < 0 AND report-period mean "
             f"below mosnorm over >= {REPORT_MIN_DAYS} days. "
             "Market Brier diff < 0 means the model beat the market mid. "
             "Selection-period columns are in-sample for the variant choice; the report-period columns "
             "are the out-of-sample check.", ""]

    def fmt(x):
        return f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}] ({x[3]})" if x else "no prices"

    def num(x):
        return "—" if x is None else f"{x:.3f}"
    for metric in METRICS:
        res = results[metric]
        sel_m, rep_m = res["market"]["selection"], res["market"]["report"]
        lines += [f"## {metric}", "",
                  "| station | variant | RPS (sel) | mosnorm RPS (sel) | gate CI hi | RPS (rep) | "
                  "mosnorm RPS (rep) | rep days | gate | market Δ sel [CI] (n) | market Δ report [CI] (n) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in res["choices"]:
            verdict = "PASS" if row["gate_pass"] else ("fail (report)" if row["sel_pass"] else "fail")
            lines.append(f"| {row['station']} | {row['variant']} | {row['crps_sel']:.3f} | "
                         f"{row['mos_crps_sel']:.3f} | {row['gate_ci_hi']:+.3f} | {num(row['crps_rep'])} | "
                         f"{num(row['mos_crps_rep'])} | {row['rep_days']} | {verdict} | "
                         f"{fmt(sel_m.get(row['station']))} | {fmt(rep_m.get(row['station']))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def evaluate_all(conn, report_path: Path, now_iso) -> Path:
    """`citycast evaluate`: check the inputs of both metrics, compute everything, then write model_choice
    and eval_log in ONE transaction and the report file last. A failure before that commits nothing, so a
    crash never records the report period as scored (spec §3.8)."""
    for metric in METRICS:
        check_inputs(conn, metric)
    results: dict[str, dict] = {}
    for metric in METRICS:
        choices = _choose(scored(conn, metric), metric)
        choice = {r["station"]: r["variant"] for r in choices}
        results[metric] = {"choices": choices, "market": market_brier(conn, metric, choice)}
    text = write_report(results, _freeze_warning(conn))
    run_at, fp = now_iso(), code_fingerprint()
    with conn:
        for metric in METRICS:
            choices = results[metric]["choices"]
            conn.execute("DELETE FROM model_choice WHERE metric=?", (metric,))
            conn.executemany("INSERT OR REPLACE INTO model_choice VALUES (:station,:metric,:variant,:crps_sel,"
                             ":mos_crps_sel,:gate_ci_hi,:sel_pass,:crps_rep,:mos_crps_rep,:rep_days,:gate_pass,"
                             ":decided_at)", [{**r, "decided_at": run_at} for r in choices])
            conn.execute("INSERT OR REPLACE INTO eval_log VALUES (?,?,?,?)",
                         (run_at, metric, fp, int(any(r["rep_days"] > 0 for r in choices))))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    return report_path
