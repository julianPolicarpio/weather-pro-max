"""Laya I/O (spec §4.4–4.5): state text, training records, export, cached scoring."""
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from citycast.gate_eval import VAL_END

QUESTION = {"win": {"type": "noul", "instructions": "Will this YES contract settle YES?"}}
SAME_DAY_LEAD_H = 24.0


def _r(x, nd):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _bucket(lo, hi) -> str:
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"<={int(hi)}"
    if hi is None or (isinstance(hi, float) and math.isnan(hi)):
        return f">={int(lo)}"
    return f"{int(lo)}-{int(hi)}"


def state(row) -> str:
    g = row.get if isinstance(row, dict) else row.__getitem__
    return json.dumps({
        "city": g("station"), "metric": g("metric"), "bucket": _bucket(g("lo"), g("hi")),
        "hours_to_close": _r(g("hours_to_close"), 1),
        "lead": "same_day" if g("lead_h") <= SAME_DAY_LEAD_H else "pre_day",
        "model_p": _r(g("p_model"), 3), "model_median": _r(g("model_median"), 1),
        "model_iqr": _r(g("model_iqr"), 1), "ask": _r(g("ask"), 2), "bid": _r(g("bid"), 2),
        "market_p": _r(g("market_p"), 3), "edge_after_fee": _r(g("edge"), 3),
        "gap_to_lo": _r(g("run_gap_to_lo"), 1), "gap_to_hi": _r(g("run_gap_to_hi"), 1),
        "month": int(g("month"))}, separators=(",", ":"))


def to_record(row) -> dict:
    """One training record. Schema pinned in laya-weather-gate/docs/format.md (Task 1)."""
    g = row.get if isinstance(row, dict) else row.__getitem__
    return {"state": state(row), "questions": QUESTION,
            "expected": {"win": bool(g("won"))}}   # laya.train.target_from_expected: noul boolean


def sample_train(df: pd.DataFrame, frac: float = 0.6, cap: int = 100_000, seed: int = 0) -> pd.DataFrame:
    keys = (df["station"] + "|" + df["metric"] + "|" + df["event_date"])
    days = keys.drop_duplicates().sample(frac=frac, random_state=seed)
    out = df[keys.isin(set(days))]
    return out.sample(n=cap, random_state=seed) if len(out) > cap else out


def export(df: pd.DataFrame, path: Path) -> int:
    """Write train/validation records as JSONL. The test split (event_date >= VAL_END) is never exported."""
    if (df["event_date"].astype(str) >= VAL_END).any():
        raise ValueError(f"refusing to export rows with event_date >= {VAL_END} (test split)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(to_record(row)) + "\n")
    return len(df)


def export_states(df: pd.DataFrame, path: Path) -> int:
    """Write the model inputs only, for scoring elsewhere (Kaggle): one JSONL line per distinct
    (ticker, asof_utc), sorted by it, as {"ticker", "asof_utc", "state"}. No outcome field is ever written."""
    rows = df.drop_duplicates(["ticker", "asof_utc"]).sort_values(["ticker", "asof_utc"]).to_dict("records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({"ticker": row["ticker"], "asof_utc": row["asof_utc"], "state": state(row)}) + "\n")
    return len(rows)


def import_scores(conn, path: Path, contender: str) -> int:
    """Store P(win) scores computed elsewhere, from JSONL lines {"ticker", "asof_utc", "p_win"}.
    Every line is validated before anything is written (p_win finite and in [0, 1], key present in
    decision_rows, no key twice); the first bad line raises ValueError and nothing is stored."""
    known = {(t, a) for t, a in conn.execute("SELECT ticker, asof_utc FROM decision_rows")}
    seen, out = set(), []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            ticker, asof, p = rec["ticker"], rec["asof_utc"], rec["p_win"]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"line {n}: not a {{ticker, asof_utc, p_win}} JSON object ({e!r})") from e
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"line {n}: p_win must be a number in [0, 1], got {p!r}")
        key = (ticker, asof)
        if not (isinstance(ticker, str) and isinstance(asof, str)) or key not in known:
            raise ValueError(f"line {n}: {ticker} @ {asof} is not in decision_rows")
        if key in seen:
            raise ValueError(f"line {n}: duplicate key {ticker} @ {asof}")
        seen.add(key)
        out.append((contender, ticker, asof, float(p)))
    with conn:
        conn.executemany("INSERT OR REPLACE INTO gate_scores VALUES (?,?,?,?)", out)
    return len(out)


def missing(df: pd.DataFrame, contender: str, conn) -> int:
    """Rows of `df` with no stored score under `contender` (what `score` would have to compute)."""
    cached = {(t, a) for t, a in conn.execute("SELECT ticker, asof_utc FROM gate_scores WHERE contender=?",
                                              (contender,))}
    return sum(k not in cached for k in zip(df["ticker"], df["asof_utc"]))


def score(df: pd.DataFrame, predict_fn: Callable[[list[str]], list[float]], contender: str, conn,
          batch: int = 256) -> np.ndarray:
    """P(win) per row of `df`, in order. Stored scores are reused; each chunk of new scores is
    committed as soon as it returns, so a crash keeps every finished chunk (resumable)."""
    cached = {(t, a): p for t, a, p in conn.execute(
        "SELECT ticker, asof_utc, p_win FROM gate_scores WHERE contender=?", (contender,))}
    rows = df.to_dict("records")
    todo, seen = [], set()
    for row in rows:
        key = (row["ticker"], row["asof_utc"])
        if key not in cached and key not in seen:
            seen.add(key)
            todo.append((key, state(row)))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        preds = [float(p) for p in predict_fn([s for _, s in chunk])]
        if len(preds) != len(chunk):
            raise ValueError(f"predict_fn returned {len(preds)} scores for {len(chunk)} states")
        with conn:
            conn.executemany("INSERT OR REPLACE INTO gate_scores VALUES (?,?,?,?)",
                             [(contender, k[0], k[1], p) for (k, _), p in zip(chunk, preds)])
        cached.update({k: p for (k, _), p in zip(chunk, preds)})
    return np.array([cached[(r["ticker"], r["asof_utc"])] for r in rows], dtype=float)


def router_predict_fn(model_ref: str | None, device: str | None = None) -> Callable[[list[str]], list[float]]:
    """Batched Laya inference. `model_ref` None = zero-shot base; else "repo@revision".
    The loading call is the one pinned in laya-weather-gate/docs/format.md (Task 1)."""
    import laya   # only in the separate Laya venv (laya-weather-gate/.venv); never in this repo's .venv
    if model_ref is None:
        agent = laya.load("convaiinnovations/laya", device=device)
    else:
        repo, _, rev = model_ref.partition("@")
        agent = laya.load(repo, device=device, revision=rev or None)

    def fn(states: list[str]) -> list[float]:
        return [float(r["answers"]["win"]["noul"])
                for r in agent.predict_batch(states, QUESTION, batch_size=32)]
    return fn
