# citycast 3 — Laya Decision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the historical bet/skip dataset, train a LightGBM meta-labeler, fine-tune Laya on Kaggle, compare every contender exactly once on held-out months, and run the winner as the sandbox `gate` bot.

**Architecture:** `decision_rows` joins out-of-sample city-model forecasts (`wf_predictions`, chosen variant) with the Kalshi candle price at each as-of point and the CLI outcome. One shared function, `gate_features`, computes a row's features both for history and for the live bot (no train/serve skew). Contenders output P(win); a common rule bets when P − ask − fee ≥ margin, at most once per ticker. The test split is read only by a once-only final report.

**Tech Stack:** Python 3.12, pandas, LightGBM, sqlite3, pytest; `laya` (optional extra, inference only) and Kaggle 2×T4 for the fine-tune.

**Spec:** `docs/superpowers/specs/2026-10-07-city-models-and-laya-design.md` §4.
**Prerequisite:** Plan 2b complete (`wf_predictions` and `model_choice` filled).

## Global Constraints

- Commit only when Julian asks. Tests: `.venv/bin/python -m pytest`; unit tests never load Laya or touch the network (inject a `predict_fn`).
- Never write `data/highcast.db` / `data/sandbox.db` from citycast; sandbox reads `data/history.db` with `?mode=ro`. Never import `highcast.kalshi_client`.
- Splits by event_date: train < 2026-04-01; validation 2026-04-01 … 2026-06-30; test 2026-07-01 … 2026-09-30. The test split is read only by `citycast gate final-report`, which refuses a second run without `--force`.
- Win/loss in decision rows = Kalshi's own `result` ("yes"/"no"), not the CLI (14 archive events settled differently; 6 changed a payout); markets without a yes/no result (8 cancelled events) are excluded; one canonical event per city-day (`citycast.evaluate.canonical_events`, from plan 2b).
- Decision rule for every contender: bet iff P(win) − ask − fee ≥ margin; at most one bet per ticker (the earliest as-of point where the rule fires); 1 contract; fee = `highcast.fees.fee_per_contract(1, ask)`.
- Margins tuned on validation over {0.00, 0.01, …, 0.10} maximising profit per decision with ≥ 50 bets; B1 uses `MIN_EDGE` (0.03) untuned.
- Profit per decision = Σ profit of bets ÷ number of distinct tickers in the split; clusters = city-day (station, metric, event_date); cluster bootstrap 2,000 resamples, seed 0.
- Success (spec §4.7): L1 beats B1, B2 and L0 on test profit per decision with 95 % CI lower bound > 0 for each difference, and L1's P(win) Brier ≤ B2's. Otherwise B2 if it beats B1 the same way, else B1.
- Laya: P(win) only, never stake size. Real money stays locked by the sandbox unlock rule.
- `market_p` in decision rows = candle mid ((bid+ask)/2, or ask when bid is missing); no de-vig (spec §4.1 is satisfied by the mid; the de-vigged variant is dropped as unneeded).

## File structure

| File | Responsibility |
|---|---|
| `docs/laya-format.md` + `tests/fixtures/citycast/laya_record.json` | pinned notebook data format and checkpoint-loading call (Task 1) |
| `src/citycast/db.py` (modify) | `decision_rows`, `gate_scores` tables |
| `src/citycast/decisions.py` | `gate_features`, `build` |
| `src/citycast/gate_eval.py` | splits, bet selection, profit/decision, margin tuning, comparisons |
| `src/citycast/meta.py` | B2 LightGBM meta-labeler |
| `src/citycast/laya_io.py` | Laya state/question, training sample, JSONL export, scoring |
| `src/citycast/final_report.py` | once-only test comparison and winner record |
| `src/citycast/cli.py` (modify) | `citycast decisions build|export`, `citycast gate score|final-report` |
| `src/sandbox/bots.py`, `cycle.py`, `cli.py` (modify) | `min_edge` per bot; `GateBot` and its account |

---

### Task 1: Pin Laya's training format and checkpoint loading (controller + Julian; no subagent)

This removes the one unknown in the spec (§4.5) before any export code is written.

- [ ] **Step 1:** Download and read `https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` (raw JSON via `https://raw.githubusercontent.com/NandhaKishorM/laya/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`). Record in `docs/laya-format.md`, quoting the notebook cells: (a) the exact per-record schema of the training dataset (field names for state, questions, gold answers; how a `noul` gold answer is written); (b) the file format it reads (JSONL / HF dataset); (c) how it pushes the result and how `Router` (or another class) loads a fine-tuned checkpoint by repo id and revision; (d) the installed package name and version to pin.
- [ ] **Step 2:** Write one record in that exact schema to `tests/fixtures/citycast/laya_record.json` using the example state from spec §4.4 and gold answer "yes".
- [ ] **Step 3:** Install for inference: add to `pyproject.toml` `[project.optional-dependencies]` the line `laya = ["laya==<version from Step 1>"]`, then `.venv/bin/pip install -e ".[laya]"`. Run a zero-shot smoke test on 3 hand-written states (one obvious winner p_model 0.95 / ask 0.40, one obvious loser, one neutral) and record the three P(win) values and latency in `docs/laya-format.md`.
- [ ] **Step 4:** If the notebook's format cannot represent a `noul` gold label, stop and tell Julian before Task 5 (the fallback is a `choice` question with criteria `{"yes": …, "no": …}`, decided with him).

---

### Task 2: Decision rows and the shared feature function

**Files:**
- Modify: `src/citycast/db.py` (append tables)
- Create: `src/citycast/decisions.py`
- Test: `tests/test_citycast_decisions.py`

**Interfaces:**
- Consumes: `wf_predictions`, `model_choice`, `kalshi_markets` (incl. `result`), `kalshi_candles`; `citycast.evaluate.canonical_events` (plan 2b Task 7); `highcast.buckets.pmf_mass`; `highcast.fees.fee_per_contract`; `highcast.timeutil.climate_day_bounds`; `citycast.stations.load_stations`.
- Produces: `GATE_FEATURES: list[str]`; `gate_features(*, station, metric, event_date, lo, hi, pmf, quantiles, run_ext, bid, ask, asof_iso, close_iso, climate_end_iso) -> dict`; `build(conn, stations: dict, progress=print) -> int` (rows written, replaces table contents).

`GATE_FEATURES` (exact): `p_model, model_median, model_iqr, ask, bid, mid, spread, market_p, fee, edge, hours_to_close, lead_h, run_gap_to_lo, run_gap_to_hi, month`. `model_iqr` = q(0.80) − q(0.20) (Q has no 0.25/0.75).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_decisions.py
import json
import math
from contextlib import closing

from citycast import db
from citycast.decisions import GATE_FEATURES, build, gate_features
from citycast.stations import Station

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
Q = {0.02: 44.0, 0.05: 45.0, 0.1: 46.0, 0.2: 47.0, 0.3: 48.0, 0.4: 49.0, 0.5: 50.0, 0.6: 51.0,
     0.7: 52.0, 0.8: 53.0, 0.9: 54.0, 0.95: 55.0, 0.98: 56.0}
PMF = {49: 0.25, 50: 0.5, 51: 0.25}


def test_gate_features_values():
    f = gate_features(station="KNYC", metric="high", event_date="2025-01-05", lo=50, hi=51, pmf=PMF,
                      quantiles=Q, run_ext=48.0, bid=0.40, ask=0.44, asof_iso="2025-01-05T12:00:00+00:00",
                      close_iso="2025-01-06T04:59:00+00:00", climate_end_iso="2025-01-06T05:00:00+00:00")
    assert set(GATE_FEATURES) <= set(f)
    assert f["p_model"] == 0.75 and f["model_median"] == 50.0 and f["model_iqr"] == 6.0
    assert abs(f["mid"] - 0.42) < 1e-9 and abs(f["spread"] - 0.04) < 1e-9 and f["market_p"] == f["mid"]
    assert f["fee"] == 0.02 and abs(f["edge"] - (0.75 - 0.44 - 0.02)) < 1e-9
    assert f["run_gap_to_lo"] == -2.0 and f["run_gap_to_hi"] == -3.0 and f["month"] == 1
    assert abs(f["hours_to_close"] - 16.9833) < 1e-3 and f["lead_h"] == 17.0


def test_gate_features_missing_bid_and_pre_day():
    f = gate_features(station="KNYC", metric="high", event_date="2025-01-05", lo=None, hi=49, pmf=PMF,
                      quantiles=Q, run_ext=None, bid=None, ask=0.30, asof_iso="2025-01-04T12:00:00+00:00",
                      close_iso="2025-01-06T04:59:00+00:00", climate_end_iso="2025-01-06T05:00:00+00:00")
    assert f["mid"] == 0.30 and math.isnan(f["spread"]) and math.isnan(f["run_gap_to_lo"])


def _seed(c):
    F = "2026-10-07T00:00:00+00:00"
    with c:
        c.execute("INSERT INTO model_choice VALUES ('KNYC','high','shared',1,1,-0.1,1,'x')")
        db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2025-01-05", "high_f": 50, "low_f": 30,
                                    "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F}])
        base = {"station": "KNYC", "metric": "high", "event_date": "2025-01-05", "fold": "x",
                "anchor": 50.0, "run_ext": None, "quantiles_json": json.dumps(Q), "k": 1.0, "created_at": F}
        db.upsert(c, "wf_predictions", [
            {**base, "asof_utc": "2025-01-05T12:00:00+00:00", "variant": "shared", "pmf_json": json.dumps(PMF)},
            {**base, "asof_utc": "2025-01-05T12:00:00+00:00", "variant": "city", "pmf_json": json.dumps({40: 1.0})}])
        mk = {"series": "KXHIGHNY", "event_ticker": "E", "station": "KNYC", "metric": "high",
              "event_date": "2025-01-05", "open_time": "2025-01-04T15:00:00+00:00",
              "close_time": "2025-01-06T04:59:00+00:00", "result": "no", "expiration_value": "50", "raw": "{}"}
        db.upsert(c, "kalshi_markets", [{**mk, "ticker": "A", "lo": 50, "hi": 51, "result": "yes"},
                                        {**mk, "ticker": "B", "lo": 52, "hi": 53},
                                        {**mk, "ticker": "C", "lo": 48, "hi": 49},
                                        # cancelled market: no yes/no result, never a decision row
                                        {**mk, "ticker": "D", "lo": 46, "hi": 47, "result": ""}])
        db.upsert(c, "kalshi_candles", [
            {"ticker": "A", "end_utc": "2025-01-05T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.44, "volume": 1, "open_interest": 1},
            {"ticker": "B", "end_utc": "2025-01-05T09:00:00+00:00", "yes_bid": 0.10, "yes_ask": 0.12, "volume": 1, "open_interest": 1},
            {"ticker": "C", "end_utc": "2025-01-05T11:30:00+00:00", "yes_bid": None, "yes_ask": 0.995, "volume": 1, "open_interest": 1}])


def test_build_uses_chosen_variant_fresh_candles_and_outcome(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        assert build(c, {"KNYC": NYC}, progress=lambda m: None) == 1     # B stale (>2 h), C ask > 0.99
        r = c.execute("SELECT ticker, p_model, won, profit_if_bet FROM decision_rows").fetchone()
        assert r[0] == "A" and r[1] == 0.75 and r[2] == 1 and abs(r[3] - (1 - 0.44 - 0.02)) < 1e-9
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `src/citycast/db.py`:

```sql
CREATE TABLE IF NOT EXISTS decision_rows (ticker TEXT, asof_utc TEXT, event_ticker TEXT NOT NULL,
  station TEXT NOT NULL, metric TEXT NOT NULL, event_date TEXT NOT NULL, lo INTEGER, hi INTEGER,
  p_model REAL, model_median REAL, model_iqr REAL, ask REAL, bid REAL, mid REAL, spread REAL,
  market_p REAL, fee REAL, edge REAL, hours_to_close REAL, lead_h REAL, run_gap_to_lo REAL,
  run_gap_to_hi REAL, month INTEGER, won INTEGER NOT NULL, profit_if_bet REAL NOT NULL,
  PRIMARY KEY (ticker, asof_utc));
CREATE TABLE IF NOT EXISTS gate_scores (contender TEXT, ticker TEXT, asof_utc TEXT, p_win REAL NOT NULL,
  PRIMARY KEY (contender, ticker, asof_utc));
```

```python
# src/citycast/decisions.py
"""Historical bet/skip rows (spec §4.1) and the one feature function shared with the live gate bot."""
import json
from datetime import date, datetime, timedelta

from citycast.evaluate import canonical_events
from highcast.buckets import pmf_mass
from highcast.fees import fee_per_contract
from highcast.timeutil import climate_day_bounds

GATE_FEATURES = ["p_model", "model_median", "model_iqr", "ask", "bid", "mid", "spread", "market_p",
                 "fee", "edge", "hours_to_close", "lead_h", "run_gap_to_lo", "run_gap_to_hi", "month"]
STALE_H = 2
NAN = float("nan")


def _h(a_iso: str, b_iso: str) -> float:
    return (datetime.fromisoformat(b_iso) - datetime.fromisoformat(a_iso)).total_seconds() / 3600


def gate_features(*, station, metric, event_date, lo, hi, pmf, quantiles, run_ext, bid, ask,
                  asof_iso, close_iso, climate_end_iso) -> dict:
    q = {float(a): float(v) for a, v in quantiles.items()}
    p = pmf_mass(pmf, lo, hi) or 0.0
    fee = fee_per_contract(1, ask)
    mid = (bid + ask) / 2 if bid is not None else ask
    have_run = run_ext is not None and run_ext == run_ext
    return {"station": station, "metric": metric, "event_date": event_date, "lo": lo, "hi": hi,
            "p_model": p, "model_median": q[0.5], "model_iqr": q[0.8] - q[0.2],
            "ask": ask, "bid": NAN if bid is None else bid, "mid": mid,
            "spread": NAN if bid is None else ask - bid, "market_p": mid, "fee": fee,
            "edge": p - ask - fee, "hours_to_close": _h(asof_iso, close_iso),
            "lead_h": _h(asof_iso, climate_end_iso),
            "run_gap_to_lo": run_ext - lo if have_run and lo is not None else NAN,
            "run_gap_to_hi": run_ext - hi if have_run and hi is not None else NAN,
            "month": int(event_date[5:7])}


def build(conn, stations: dict, progress=print) -> int:
    choice = {(s, m): v for s, m, v in conn.execute("SELECT station, metric, variant FROM model_choice")}
    canon = canonical_events(conn)
    markets: dict[tuple, list] = {}
    # won/profit come from what Kalshi paid (spec §4.1), so only settled yes/no markets qualify
    for m in conn.execute("SELECT ticker, event_ticker, station, metric, event_date, lo, hi, open_time, "
                          "close_time, result FROM kalshi_markets WHERE NOT (lo IS NULL AND hi IS NULL) "
                          "AND result IN ('yes','no')"):
        if m[1] in canon:
            markets.setdefault((m[2], m[3], m[4]), []).append(m)
    rows = []
    preds = conn.execute("SELECT station, metric, event_date, asof_utc, variant, quantiles_json, pmf_json, "
                         "run_ext FROM wf_predictions WHERE variant IN ('shared','city')").fetchall()
    for s, metric, d, asof, variant, qj, pj, run_ext in preds:
        if choice.get((s, metric)) != variant or s not in stations:
            continue
        pmf = {int(k): v for k, v in json.loads(pj).items()}
        q = json.loads(qj)
        end_iso = climate_day_bounds(date.fromisoformat(d), stations[s].std_utc_offset)[1].isoformat()
        stale_iso = (datetime.fromisoformat(asof) - timedelta(hours=STALE_H)).isoformat()
        for ticker, event_ticker, _, _, _, lo, hi, open_t, close_t, result in markets.get((s, metric, d), []):
            if (open_t and open_t > asof) or close_t <= asof:
                continue
            c = conn.execute("SELECT yes_bid, yes_ask FROM kalshi_candles WHERE ticker=? AND end_utc>? "
                             "AND end_utc<=? AND yes_ask IS NOT NULL ORDER BY end_utc DESC LIMIT 1",
                             (ticker, stale_iso, asof)).fetchone()
            if c is None or not (0.01 <= c[1] <= 0.99):
                continue
            f = gate_features(station=s, metric=metric, event_date=d, lo=lo, hi=hi, pmf=pmf, quantiles=q,
                              run_ext=run_ext, bid=c[0], ask=c[1], asof_iso=asof, close_iso=close_t,
                              climate_end_iso=end_iso)
            won = int(result == "yes")
            rows.append({**f, "ticker": ticker, "asof_utc": asof, "event_ticker": event_ticker,
                         "won": won, "profit_if_bet": won - f["ask"] - f["fee"]})
    cols = ["ticker", "asof_utc", "event_ticker", "station", "metric", "event_date", "lo", "hi",
            *GATE_FEATURES, "won", "profit_if_bet"]
    with conn:
        conn.execute("DELETE FROM decision_rows")
        n = 0
        if rows:
            conn.executemany(f"INSERT INTO decision_rows ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                             [tuple(r[c] for c in cols) for r in rows])
            n = len(rows)
    progress(f"decision_rows: {n}")
    return n
```

- [ ] **Step 4: Run to verify they pass** — 3 passed. (The `hours_to_close` expectation: 12:00Z → 04:59Z next day = 16 h 59 min = 16.9833 h.)

- [ ] **Step 5: Full suite** — all pass.

---

### Task 3: Evaluation core

**Files:**
- Create: `src/citycast/gate_eval.py`
- Test: `tests/test_citycast_gate_eval.py`

**Interfaces:**
- Consumes: `citycast.metrics.cluster_bootstrap`.
- Produces: `TRAIN_END, VAL_END, TEST_END`, `MARGINS`, `MIN_BETS = 50`; `load(conn) -> pd.DataFrame` (all decision rows, sorted by ticker, asof); `split(df) -> dict[str, pd.DataFrame]` (`train`, `validation`, `test`); `pick_bets(df, p: np.ndarray, margin) -> pd.Series[bool]` (row is the ticker's first qualifying row); `ticker_profit(df, bet_mask) -> pd.DataFrame` (one row per ticker: `ticker, cluster, profit`, profit 0 when not bet); `profit_per_decision(df, p, margin) -> tuple[float, int]` (value, number of bets); `tune_margin(df, p) -> float`; `compare(df, p_a, m_a, p_b, m_b) -> tuple[float, float, float]` (mean per-decision profit difference A − B and its 95 % CI); `brier_score(df, p) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_gate_eval.py
import numpy as np
import pandas as pd

from citycast.gate_eval import (brier_score, compare, pick_bets, profit_per_decision, split,
                                ticker_profit, tune_margin)


def _df():
    rows = []
    for i in range(200):
        won = i % 2
        for h in ("10", "12"):
            rows.append({"ticker": f"T{i}", "asof_utc": f"2026-05-01T{h}:00:00+00:00", "station": "KNYC",
                         "metric": "high", "event_date": f"2026-05-{1 + i % 28:02d}", "ask": 0.40,
                         "fee": 0.02, "won": won, "profit_if_bet": won - 0.42})
    return pd.DataFrame(rows)


def test_split_by_event_date():
    df = pd.DataFrame({"event_date": ["2026-03-31", "2026-04-01", "2026-06-30", "2026-07-01", "2026-09-30", "2026-10-01"]})
    s = split(df)
    assert (len(s["train"]), len(s["validation"]), len(s["test"])) == (1, 2, 2)


def test_bet_once_per_ticker_at_first_qualifying_row():
    df = _df()
    p = np.where(df["asof_utc"].str.contains("T12"), 0.9, 0.1)     # qualifies only at 12:00
    mask = pick_bets(df, p, 0.0)
    assert mask.sum() == 200 and df[mask]["asof_utc"].str.contains("T12").all()


def test_perfect_oracle_profit_and_comparison():
    df = _df()
    oracle = df["won"].to_numpy().astype(float)
    ppd, n = profit_per_decision(df, oracle, 0.0)
    assert n == 100 and abs(ppd - (100 * 0.58) / 200) < 1e-9
    never = np.zeros(len(df))
    mean, lo, hi = compare(df, oracle, 0.0, never, 0.0)
    assert abs(mean - 0.29) < 1e-9 and lo > 0
    assert len(ticker_profit(df, pick_bets(df, oracle, 0.0))) == 200


def test_tune_margin_requires_min_bets_and_brier():
    df = _df()
    oracle = df["won"].to_numpy().astype(float)
    assert tune_margin(df, oracle) == 0.0
    assert brier_score(df, oracle) == 0.0
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/gate_eval.py
"""Bet selection, profit per decision and comparisons (spec §4.2–4.3, §4.7)."""
import numpy as np
import pandas as pd

from citycast.metrics import cluster_bootstrap

TRAIN_END, VAL_END, TEST_END = "2026-04-01", "2026-07-01", "2026-10-01"
MARGINS = tuple(round(0.01 * i, 2) for i in range(11))
MIN_BETS = 50


def load(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM decision_rows ORDER BY ticker, asof_utc", conn)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df["event_date"]
    return {"train": df[d < TRAIN_END], "validation": df[(d >= TRAIN_END) & (d < VAL_END)],
            "test": df[(d >= VAL_END) & (d < TEST_END)]}


def pick_bets(df: pd.DataFrame, p, margin: float) -> pd.Series:
    ok = pd.Series(np.asarray(p, dtype=float) - df["ask"].to_numpy() - df["fee"].to_numpy() >= margin - 1e-12,
                   index=df.index)
    order = df.assign(_ok=ok).sort_values(["ticker", "asof_utc"])
    first = order[order["_ok"]].groupby("ticker").head(1).index
    return pd.Series(df.index.isin(first), index=df.index)


def ticker_profit(df: pd.DataFrame, bet_mask: pd.Series) -> pd.DataFrame:
    cluster = df["station"] + "|" + df["metric"] + "|" + df["event_date"]
    bet = df.assign(cluster=cluster, profit=np.where(bet_mask, df["profit_if_bet"], 0.0))
    return bet.groupby("ticker", as_index=False).agg(cluster=("cluster", "first"), profit=("profit", "sum"))


def profit_per_decision(df: pd.DataFrame, p, margin: float) -> tuple[float, int]:
    mask = pick_bets(df, p, margin)
    tp = ticker_profit(df, mask)
    return float(tp["profit"].sum() / len(tp)) if len(tp) else 0.0, int(mask.sum())


def tune_margin(df: pd.DataFrame, p) -> float:
    best, best_v = MARGINS[-1], -np.inf
    for m in MARGINS:
        v, n = profit_per_decision(df, p, m)
        if n >= MIN_BETS and v > best_v + 1e-12:
            best, best_v = m, v
    return best


def compare(df: pd.DataFrame, p_a, m_a: float, p_b, m_b: float) -> tuple[float, float, float]:
    a = ticker_profit(df, pick_bets(df, p_a, m_a)).set_index("ticker")
    b = ticker_profit(df, pick_bets(df, p_b, m_b)).set_index("ticker")
    diff = a["profit"] - b["profit"]
    clusters: dict[str, list[float]] = {}
    for ticker, d in diff.items():
        clusters.setdefault(a.at[ticker, "cluster"], []).append(float(d))
    return cluster_bootstrap(clusters)


def brier_score(df: pd.DataFrame, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - df["won"].to_numpy()) ** 2))
```

- [ ] **Step 4: Run to verify they pass** — 4 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 4: B2 meta-labeler

**Files:**
- Create: `src/citycast/meta.py`
- Test: `tests/test_citycast_meta.py`

**Interfaces:**
- Produces: `META_FEATURES = GATE_FEATURES + ["station", "metric"]`; `MetaLabeler(station_categories)` with `.fit(df) -> self`, `.predict(df) -> np.ndarray` (P(win)); `save(model, path)`, `load_model(path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citycast_meta.py
import numpy as np
import pandas as pd

from citycast.decisions import GATE_FEATURES
from citycast.meta import MetaLabeler, load_model, save


def test_learns_signal_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({c: rng.normal(size=n) for c in GATE_FEATURES})
    df["p_model"] = rng.uniform(0, 1, n)
    df["won"] = (rng.uniform(0, 1, n) < df["p_model"]).astype(int)
    df["station"], df["metric"] = rng.choice(["KNYC", "KLAX"], n), "high"
    m = MetaLabeler(("KLAX", "KNYC")).fit(df)
    p = m.predict(df)
    assert p.shape == (n,) and 0 <= p.min() and p.max() <= 1
    assert np.corrcoef(p, df["p_model"])[0, 1] > 0.8
    save(m, tmp_path / "b2.pkl")
    assert np.allclose(load_model(tmp_path / "b2.pkl").predict(df), p)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/meta.py
"""B2: LightGBM P(win) meta-labeler on the gate features (spec §4.3)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from citycast.decisions import GATE_FEATURES
from citycast.qmodel import PARAMS

META_FEATURES = GATE_FEATURES + ["station", "metric"]


class MetaLabeler:
    def __init__(self, station_categories: tuple[str, ...]):
        self.cats = tuple(station_categories)
        self.model: LGBMClassifier | None = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[META_FEATURES].copy()
        X["station"] = pd.Categorical(X["station"], categories=list(self.cats))
        X["metric"] = pd.Categorical(X["metric"], categories=["high", "low"])
        return X

    def fit(self, df: pd.DataFrame) -> "MetaLabeler":
        params = {**PARAMS, "n_estimators": 300}
        self.model = LGBMClassifier(objective="binary", **params).fit(self._X(df), df["won"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df))[:, 1]


def save(model: MetaLabeler, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    return path


def load_model(path: Path) -> MetaLabeler:
    return pickle.loads(Path(path).read_bytes())
```

- [ ] **Step 4: Run to verify it passes** — 1 passed.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 5: Laya state, training sample, export and scoring

**Files:**
- Create: `src/citycast/laya_io.py`
- Test: `tests/test_citycast_laya_io.py` (uses `tests/fixtures/citycast/laya_record.json` from Task 1)

**Interfaces:**
- Consumes: Task 1's pinned record schema (`docs/laya-format.md`).
- Produces: `QUESTION: dict`; `state(row) -> str` (compact JSON, keys in spec §4.4 order); `to_record(row) -> dict` (one training record in the pinned schema); `sample_train(df, frac=0.6, cap=100_000, seed=0) -> pd.DataFrame` (all rows of a random 60 % of city-days, capped); `export(df, path) -> int`; `score(df, predict_fn, contender: str, conn) -> np.ndarray` (calls `predict_fn(state_str) -> float`, stores `gate_scores`, reuses stored scores); `router_predict_fn(model_ref: str | None) -> Callable[[str], float]` (lazy `import laya`; `None` = zero-shot base checkpoint; loading call exactly as pinned in Task 1).

The record schema below is the default; **Task 1's `docs/laya-format.md` governs**. If it differs, change only `to_record` and make the fixture test pass.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citycast_laya_io.py
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from citycast import db
from citycast.laya_io import QUESTION, export, sample_train, score, state, to_record

ROW = {"station": "KLAX", "metric": "high", "event_date": "2026-01-05", "lo": 66, "hi": 67,
       "hours_to_close": 14.52, "lead_h": 20.0, "p_model": 0.41, "model_median": 66.8, "model_iqr": 3.1,
       "ask": 0.33, "bid": 0.30, "market_p": 0.315, "edge": 0.06, "run_gap_to_lo": float("nan"),
       "month": 1, "won": 1, "ticker": "T", "asof_utc": "2026-01-05T00:00:00+00:00"}


def test_state_is_compact_and_ordered():
    s = json.loads(state(ROW))
    assert list(s) == ["city", "metric", "bucket", "hours_to_close", "lead", "model_p", "model_median",
                       "model_iqr", "ask", "bid", "market_p", "edge_after_fee", "running_gap", "month"]
    assert s["bucket"] == "66-67" and s["lead"] == "same_day" and s["running_gap"] is None
    assert len(state(ROW)) < 400


def test_record_matches_pinned_fixture_shape():
    fixture = json.loads(Path("tests/fixtures/citycast/laya_record.json").read_text())
    rec = to_record(ROW)
    assert set(rec) == set(fixture)
    assert json.dumps(QUESTION) in json.dumps(rec) or rec.get("questions") == QUESTION


def test_sample_keeps_whole_city_days_and_caps():
    df = pd.DataFrame([{**ROW, "event_date": f"2026-01-{1 + i % 20:02d}", "ticker": f"T{i}"} for i in range(400)])
    s = sample_train(df, cap=100)
    assert len(s) <= 100
    per_day = df.groupby("event_date").size()
    kept = s.groupby("event_date").size()
    assert (kept == per_day[kept.index]).all() or len(s) == 100


def test_export_and_cached_scoring(tmp_path):
    df = pd.DataFrame([ROW, {**ROW, "ticker": "U"}])
    assert export(df, tmp_path / "train.jsonl") == 2
    calls = []

    def fn(s):
        calls.append(s)
        return 0.7
    with closing(db.connect(tmp_path / "h.db")) as c:
        p1 = score(df, fn, "L0", c)
        p2 = score(df, fn, "L0", c)
    assert np.allclose(p1, 0.7) and np.allclose(p2, 0.7) and len(calls) == 2
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/laya_io.py
"""Laya I/O (spec §4.4–4.5): state text, training records, export, cached scoring."""
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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
        "running_gap": _r(g("run_gap_to_lo"), 1), "month": int(g("month"))}, separators=(",", ":"))


def to_record(row) -> dict:
    """One training record. Schema pinned in docs/laya-format.md (Task 1)."""
    g = row.get if isinstance(row, dict) else row.__getitem__
    return {"state": state(row), "questions": QUESTION,
            "answers": {"win": {"noul": float(g("won"))}}}


def sample_train(df: pd.DataFrame, frac: float = 0.6, cap: int = 100_000, seed: int = 0) -> pd.DataFrame:
    keys = (df["station"] + "|" + df["metric"] + "|" + df["event_date"])
    days = keys.drop_duplicates().sample(frac=frac, random_state=seed)
    out = df[keys.isin(set(days))]
    return out.sample(n=cap, random_state=seed) if len(out) > cap else out


def export(df: pd.DataFrame, path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(to_record(row)) + "\n")
    return len(df)


def score(df: pd.DataFrame, predict_fn: Callable[[str], float], contender: str, conn) -> np.ndarray:
    cached = {(t, a): p for t, a, p in conn.execute(
        "SELECT ticker, asof_utc, p_win FROM gate_scores WHERE contender=?", (contender,))}
    out, new = [], []
    for row in df.to_dict("records"):
        key = (row["ticker"], row["asof_utc"])
        if key not in cached:
            cached[key] = float(predict_fn(state(row)))
            new.append((contender, *key, cached[key]))
        out.append(cached[key])
    if new:
        with conn:
            conn.executemany("INSERT OR REPLACE INTO gate_scores VALUES (?,?,?,?)", new)
    return np.array(out, dtype=float)


def router_predict_fn(model_ref: str | None) -> Callable[[str], float]:
    """Laya inference. `model_ref` None = zero-shot base; else the fine-tuned repo@revision.
    The loading call is the one pinned in docs/laya-format.md (Task 1)."""
    from laya import Router   # optional extra: pip install -e ".[laya]"
    router = Router() if model_ref is None else Router(model=model_ref)

    def fn(state_str: str) -> float:
        return float(router.predict(state_str, QUESTION)["answers"]["win"]["noul"])
    return fn
```

- [ ] **Step 4: Run to verify they pass** — 4 passed. If Task 1 pinned a different loading call, update `router_predict_fn` to it (untested by unit tests; exercised in Task 8).

- [ ] **Step 5: Full suite** — all pass.

---

### Task 6: Once-only final report and the CLI

**Files:**
- Create: `src/citycast/final_report.py`
- Modify: `src/citycast/cli.py`
- Test: `tests/test_citycast_final_report.py`

**Interfaces:**
- Consumes: Tasks 2–5; `highcast.orderbot.MIN_EDGE`.
- Produces: `run_final(conn, scores: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], report_path, force=False, now_iso) -> dict` where `scores[name] = (p_train_unused, p_validation, p_test)` for `B2`, `L0`, `L1` (B0/B1 computed inside); writes the report, stores `meta['gate_choice']` = JSON `{"winner", "margin", "decided_at"}` and `meta['final_report_run_at']`; raises `RuntimeError` on a second run without `force`. CLI: `citycast decisions build`, `citycast decisions export --out DIR` (writes `train.jsonl`, `validation.jsonl` from the sample/splits), `citycast gate score --contender B2|L0|L1 [--model-ref REPO@REV]`, `citycast gate final-report [--force]`.

Winner rule (spec §4.7): `L1` if each of `compare(test, L1 vs X)` for X ∈ {B1, B2, L0} has CI lower > 0 and `brier(L1) ≤ brier(B2)`; else `B2` if `compare(test, B2 vs B1)` CI lower > 0; else `B1`. Margins: B1 = MIN_EDGE; others = `tune_margin(validation, p_validation)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citycast_final_report.py
import json
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from citycast import db
from citycast.final_report import run_final


def _seed(c):
    rows = []
    for split_date in ("2026-05-10", "2026-08-10"):
        for i in range(300):
            won = int(i % 3 == 0)
            rows.append({"ticker": f"{split_date}-{i}", "asof_utc": f"{split_date}T12:00:00+00:00",
                         "event_ticker": "E", "station": "KNYC", "metric": "high",
                         "event_date": f"{split_date[:8]}{1 + i % 28:02d}", "lo": 1, "hi": 2,
                         "p_model": 0.5, "model_median": 1, "model_iqr": 1, "ask": 0.30, "bid": 0.28,
                         "mid": 0.29, "spread": 0.02, "market_p": 0.29, "fee": 0.02, "edge": 0.18,
                         "hours_to_close": 10, "lead_h": 10, "run_gap_to_lo": None, "run_gap_to_hi": None,
                         "month": 5, "won": won, "profit_if_bet": won - 0.32})
    with c:
        db.upsert(c, "decision_rows", rows)


def test_oracle_l1_wins_and_second_run_refused(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        df = pd.read_sql_query("SELECT * FROM decision_rows ORDER BY ticker, asof_utc", c)
        val, test = df[df.event_date < "2026-07-01"], df[df.event_date >= "2026-07-01"]
        oracle = lambda d: d["won"].to_numpy().astype(float)          # noqa: E731
        flat = lambda d: np.full(len(d), 0.5)                          # noqa: E731
        scores = {"B2": (None, flat(val), flat(test)), "L0": (None, flat(val), flat(test)),
                  "L1": (None, oracle(val), oracle(test))}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=lambda: "2026-10-08T00:00:00+00:00")
        assert out["winner"] == "L1" and (tmp_path / "r.md").exists()
        assert json.loads(c.execute("SELECT value FROM meta WHERE key='gate_choice'").fetchone()[0])["winner"] == "L1"
        with pytest.raises(RuntimeError):
            run_final(c, scores, tmp_path / "r.md", now_iso=lambda: "x")
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/citycast/final_report.py
"""The one-time test comparison (spec §4.6–4.7)."""
import json
from pathlib import Path

import numpy as np

from citycast import gate_eval as ge
from highcast.orderbot import MIN_EDGE


def run_final(conn, scores: dict, report_path, force: bool = False, now_iso=None) -> dict:
    if conn.execute("SELECT value FROM meta WHERE key='final_report_run_at'").fetchone() and not force:
        raise RuntimeError("final report already run; the test split is spent (use --force deliberately)")
    s = ge.split(ge.load(conn))
    val, test = s["validation"], s["test"]
    p_test = {"B0": np.zeros(len(test)), "B1": test["p_model"].to_numpy()}
    margin = {"B0": 1.0, "B1": MIN_EDGE}
    for name in ("B2", "L0", "L1"):
        _, p_v, p_t = scores[name]
        margin[name] = ge.tune_margin(val, p_v)
        p_test[name] = np.asarray(p_t, dtype=float)
    ppd = {n: ge.profit_per_decision(test, p_test[n], margin[n]) for n in p_test}
    cmp = {f"{a}-{b}": ge.compare(test, p_test[a], margin[a], p_test[b], margin[b])
           for a, b in (("L1", "B1"), ("L1", "B2"), ("L1", "L0"), ("B2", "B1"))}
    brier = {n: ge.brier_score(test, p_test[n]) for n in ("B1", "B2", "L0", "L1")}
    if all(cmp[f"L1-{x}"][1] > 0 for x in ("B1", "B2", "L0")) and brier["L1"] <= brier["B2"]:
        winner = "L1"
    elif cmp["B2-B1"][1] > 0:
        winner = "B2"
    else:
        winner = "B1"
    now = now_iso()
    lines = ["# Decision gate — final test report", "", f"Run at {now}. Test split: event dates "
             f"{ge.VAL_END} to {ge.TEST_END} (exclusive), {len(test)} rows, {test['ticker'].nunique()} tickers.", "",
             "| contender | margin | profit/decision | bets | Brier |", "|---|---|---|---|---|"]
    for n in ("B0", "B1", "B2", "L0", "L1"):
        lines.append(f"| {n} | {margin[n]:.2f} | {ppd[n][0]:+.4f} | {ppd[n][1]} | "
                     f"{brier.get(n, float('nan')):.4f} |")
    lines += ["", "| comparison | mean diff | 95% CI |", "|---|---|---|"]
    for k, (m, lo, hi) in cmp.items():
        lines.append(f"| {k} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", f"**Winner: {winner}** (rule: spec §4.7)."]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n")
    choice = {"winner": winner, "margin": margin[winner], "decided_at": now}
    with conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('gate_choice', ?)", (json.dumps(choice),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('final_report_run_at', ?)", (now,))
    return {**choice, "ppd": ppd, "compare": cmp, "brier": brier}
```

Add to `src/citycast/cli.py`:

```python
# imports
from citycast import decisions, final_report, gate_eval, laya_io, meta

B2_PATH = Path("data/citycast/models/gate-b2.pkl")
```

```python
    # subparsers
    dec = sub.add_parser("decisions")
    dec.add_argument("action", choices=("build", "export"))
    dec.add_argument("--out", default="data/citycast/laya")
    gate = sub.add_parser("gate")
    gate.add_argument("action", choices=("score", "final-report"))
    gate.add_argument("--contender", choices=("B2", "L0", "L1"))
    gate.add_argument("--model-ref")
    gate.add_argument("--force", action="store_true")
```

```python
        # branches
        if args.cmd == "decisions" and args.action == "build":
            print(json.dumps({"rows": decisions.build(conn, stations)}))
            return 0
        if args.cmd == "decisions" and args.action == "export":
            s = gate_eval.split(gate_eval.load(conn))
            out = Path(args.out)
            n_tr = laya_io.export(laya_io.sample_train(s["train"]), out / "train.jsonl")
            n_va = laya_io.export(s["validation"], out / "validation.jsonl")
            print(json.dumps({"train": n_tr, "validation": n_va, "dir": str(out)}))
            return 0
        if args.cmd == "gate" and args.action == "score":
            s = gate_eval.split(gate_eval.load(conn))
            if args.contender == "B2":
                m = meta.MetaLabeler(tuple(sorted(stations))).fit(s["train"])
                meta.save(m, B2_PATH)
                fn_rows = {k: m.predict(v) for k, v in s.items() if k != "test"}
                for k, p in fn_rows.items():   # store validation (and train) scores; test is scored in final-report
                    with conn:
                        conn.executemany("INSERT OR REPLACE INTO gate_scores VALUES (?,?,?,?)",
                                         [("B2", t, a, float(x)) for t, a, x in zip(s[k]["ticker"], s[k]["asof_utc"], p)])
            else:
                fn = laya_io.router_predict_fn(None if args.contender == "L0" else args.model_ref)
                laya_io.score(s["validation"], fn, args.contender, conn)
            print(json.dumps({"scored": args.contender}))
            return 0
        if args.cmd == "gate" and args.action == "final-report":
            s = gate_eval.split(gate_eval.load(conn))
            b2 = meta.load_model(B2_PATH)
            ref = json.loads((conn.execute("SELECT value FROM meta WHERE key='laya_l1_ref'").fetchone() or ['null'])[0])
            scores = {"B2": (None, b2.predict(s["validation"]), b2.predict(s["test"])),
                      "L0": (None, laya_io.score(s["validation"], laya_io.router_predict_fn(None), "L0", conn),
                             laya_io.score(s["test"], laya_io.router_predict_fn(None), "L0", conn)),
                      "L1": (None, laya_io.score(s["validation"], laya_io.router_predict_fn(ref), "L1", conn),
                             laya_io.score(s["test"], laya_io.router_predict_fn(ref), "L1", conn))}
            out = final_report.run_final(conn, scores, Path(f"docs/reports/gate-final-{date.today()}.md"),
                                         force=args.force, now_iso=utcnow_iso)
            print(json.dumps({"winner": out["winner"], "margin": out["margin"]}))
            return 0
```

`meta['laya_l1_ref']` is set by hand in Task 8 (JSON string of the pushed `repo@revision`).

- [ ] **Step 4: Run to verify it passes** — `.venv/bin/python -m pytest -q tests/test_citycast_final_report.py tests/test_citycast_cli.py`, all pass.

- [ ] **Step 5: Full suite** — all pass.

---

### Task 7: Per-bot margin and the sandbox `gate` bot

**Files:**
- Modify: `src/sandbox/bots.py` (`evaluate_bucket` margin argument; `ForecastBot.min_edge`; `CitycastBot._latest` returns quantiles and run_ext; new `GateBot`), `src/sandbox/cycle.py` (`ACCOUNTS`), `src/sandbox/cli.py` (bot list)
- Test: `tests/test_sandbox_gate_bot.py`; existing sandbox tests unchanged except account-count updates

**Interfaces:**
- Consumes: `citycast.decisions.gate_features`, `citycast.meta.load_model`, `citycast.laya_io.router_predict_fn / state`, `meta['gate_choice']` in history.db.
- Produces: `evaluate_bucket(p, ask, min_edge=MIN_EDGE)`; `ForecastBot.min_edge = MIN_EDGE` used by `decide`; `GateBot(history_db, b2_path="data/citycast/models/gate-b2.pkl")` with `name = "gate"`, `min_edge` = chosen margin; forecast_json `{"source": "gate", "winner": …, "citycast_issued_utc": …}`; account `("gate", "bot", "decision gate: winner of the held-out test (paper)")` before `manual`.

Behaviour: `GateBot` reads `gate_choice` (read-only); if absent → no forecasts. It takes the same latest gated `live_predictions` row as `CitycastBot`, computes `gate_features` per market (bid/ask from the snapshot market, `asof_iso = snapshot.taken_at`, climate end from `sandbox.stations`), then P(win): B1 → `p_model`; B2 → `MetaLabeler.predict`; L1 → `router_predict_fn(meta['laya_l1_ref'])` on `state(row)`. Returns `{ticker: P(win)}` (not normalised — each is an independent probability), and the base `decide` bets when P − ask − fee ≥ `self.min_edge`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox_gate_bot.py
import json
from contextlib import closing

from citycast import db as hdb
from sandbox.bots import ForecastBot, GateBot, evaluate_bucket
from sandbox_helpers import make_market
from test_sandbox_citycast_bot import Snap, _history

Q = {str(a): v for a, v in zip((0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
                                (64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76))}


def test_evaluate_bucket_margin_argument_keeps_default():
    assert evaluate_bucket(0.60, 0.50)[0] is True                  # default MIN_EDGE 0.03
    assert evaluate_bucket(0.60, 0.50, min_edge=0.10)[0] is False
    assert ForecastBot.min_edge == 0.03


def _with_choice(tmp_path, winner="B1", margin=0.05):
    p = _history(tmp_path)
    with closing(hdb.connect(p)) as c, c:
        c.execute("UPDATE live_predictions SET quantiles_json=?", (json.dumps(Q),))
        c.execute("INSERT INTO meta VALUES ('gate_choice', ?)", (json.dumps({"winner": winner, "margin": margin}),))
    return p


def test_b1_gate_returns_model_probability_and_margin(tmp_path):
    bot = GateBot(_with_choice(tmp_path))
    markets = [make_market("KXHIGHNY-26OCT08-B69.5", lo=69, hi=70, station="KNYC", metric="high",
                           event_date="2026-10-08", yes_bid=0.40, yes_ask=0.45,
                           close_time="2026-10-09T03:59:00+00:00")]
    probs, fj = bot.forecast_for_event(markets, Snap())
    assert abs(probs["KXHIGHNY-26OCT08-B69.5"] - 0.7) < 1e-9
    assert bot.min_edge == 0.05 and fj["source"] == "gate" and fj["winner"] == "B1"


def test_no_choice_means_no_bets(tmp_path):
    bot = GateBot(_history(tmp_path))
    assert bot.forecast_for_event([make_market("X", lo=69, hi=70, station="KNYC", metric="high",
                                               event_date="2026-10-08")], Snap()) == (None, None)
```

Adapt `make_market` keyword names to the real helper (read `tests/sandbox_helpers.py` first), keeping the values.

- [ ] **Step 2: Run to verify they fail** — FAIL (`GateBot` missing; `evaluate_bucket` has no `min_edge`).

- [ ] **Step 3: Implement** — in `src/sandbox/bots.py`:

```python
def evaluate_bucket(p: float, ask: float | None, min_edge: float = MIN_EDGE) -> tuple[bool, float | None, float | None, str]:
    if ask is None:
        return False, None, None, "no_ask"
    fpc = fee_per_contract(1, ask)
    edge = round(p - ask - fpc, 6)
    if edge >= min_edge:
        return True, fpc, edge, "edge"
    return False, fpc, edge, "below_min_edge"
```

In `ForecastBot`: add class attribute `min_edge = MIN_EDGE` and change the call in `decide` to `evaluate_bucket(p, ask, self.min_edge)`.

In `CitycastBot._latest`, select `lp.pmf_json, lp.variant, lp.issued_utc, lp.quantiles_json, lp.run_ext` (CitycastBot keeps using indexes 0–2).

Add after `CitycastBot`:

```python
class GateBot(CitycastBot):
    """Bet/skip gate (spec 2026-10-07 §4.7): the held-out test's winner turns citycast's
    forecast into P(win) per bucket; bets when P - ask - fee >= the tuned margin."""
    name = "gate"

    def __init__(self, history_db: str, b2_path: str = "data/citycast/models/gate-b2.pkl"):
        super().__init__(history_db)
        self.b2_path = b2_path
        self._choice = None
        self._predict = None

    def _meta(self, key: str):
        path = Path(self.history_db)
        if not path.exists():
            return None
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as c:
                row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        except sqlite3.Error as e:
            log.warning("gate meta read failed: %r", e)
            return None
        return json.loads(row[0]) if row else None

    def _load(self) -> bool:
        if self._choice is not None:
            return True
        choice = self._meta("gate_choice")
        if not choice:
            return False
        winner = choice["winner"]
        if winner == "B2":
            from citycast.meta import load_model
            model = load_model(Path(self.b2_path))
            self._predict = lambda rows: list(model.predict(pd.DataFrame(rows)))
        elif winner == "L1":
            from citycast.laya_io import router_predict_fn, state
            fn = router_predict_fn(self._meta("laya_l1_ref"))
            self._predict = lambda rows: [fn(state(r)) for r in rows]
        else:
            self._predict = lambda rows: [r["p_model"] for r in rows]
        self._choice, self.min_edge = choice, float(choice["margin"])
        return True

    def forecast_for_event(self, markets, snapshot):
        from citycast.decisions import gate_features
        first = markets[0]
        if first.station is None or first.metric is None or not self._load():
            return None, None
        row = self._latest(first.station, first.metric, first.event_date, snapshot.taken_at)
        if row is None:
            return None, None
        age = (datetime.fromisoformat(snapshot.taken_at) - datetime.fromisoformat(row[2])).total_seconds()
        if age > self.MAX_AGE_S:
            return None, None
        pmf = {int(k): float(v) for k, v in json.loads(row[0]).items()}
        quantiles = json.loads(row[3])
        end_iso = climate_day_bounds(date.fromisoformat(first.event_date), STATIONS[first.station][1])[1].isoformat()
        rows, tickers = [], []
        for m in markets:
            ask = usable_ask(m)
            if ask is None or m.close_time is None:
                continue
            rows.append(gate_features(station=m.station, metric=m.metric, event_date=m.event_date, lo=m.lo,
                                      hi=m.hi, pmf=pmf, quantiles=quantiles, run_ext=row[4], bid=usable_bid(m),
                                      ask=ask, asof_iso=snapshot.taken_at, close_iso=m.close_time,
                                      climate_end_iso=end_iso))
            tickers.append(m.ticker)
        if not rows:
            return None, None
        probs = dict(zip(tickers, (float(p) for p in self._predict(rows))))
        return probs, {"source": "gate", "winner": self._choice["winner"], "citycast_issued_utc": row[2]}
```

Add imports at the top of `bots.py`: `from datetime import date` (extend the existing datetime import), `import pandas as pd`, and `from highcast.timeutil import climate_day_bounds`.

In `src/sandbox/cycle.py` add `("gate", "bot", "decision gate: winner of the held-out test (paper)")` before `manual`; in `src/sandbox/cli.py` add `GateBot(args.history_db)` after `CitycastBot(args.history_db)`.

- [ ] **Step 4: Run the gate tests, the strict sandbox suite and the full suite**

Run: `.venv/bin/python -m pytest -q tests/test_sandbox_gate_bot.py && .venv/bin/python -m pytest -q tests/test_sandbox_*.py -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning && .venv/bin/python -m pytest -q`
Expected: all pass after updating only account-count expectations for the sixth account; the no-real-money import test still passes.

---

### Task 8: Run it (controller task; Kaggle and Hugging Face steps need Julian)

- [ ] **Step 1:** `.venv/bin/citycast decisions build` — record row counts per split and per station.
- [ ] **Step 2:** `.venv/bin/citycast gate score --contender B2` and `--contender L0` (L0 needs the `laya` extra installed).
- [ ] **Step 3:** `.venv/bin/citycast decisions export --out data/citycast/laya` → `train.jsonl`, `validation.jsonl`.
- [ ] **Step 4 (Julian):** Upload the two files as a private Kaggle dataset; run the official notebook on 2×T4 pointed at them (including its calibration-temperature step); push the result to the private Hugging Face repo `julianPolicarpio/laya-weather-gate`. Record the notebook version, run time and the pushed revision in `docs/laya-format.md`. Claude prepares the notebook edits but does not use Julian's Kaggle or HF credentials.
- [ ] **Step 5:** `sqlite3`-free record of the reference: `.venv/bin/python -c "import json,sqlite3; c=sqlite3.connect('data/history.db'); c.execute(\"INSERT OR REPLACE INTO meta VALUES ('laya_l1_ref', ?)\", (json.dumps('julianPolicarpio/laya-weather-gate@<revision>'),)); c.commit()"` then `.venv/bin/citycast gate score --contender L1 --model-ref julianPolicarpio/laya-weather-gate@<revision>`.
- [ ] **Step 6:** `.venv/bin/citycast gate final-report` — once. Read `docs/reports/gate-final-<date>.md` and tell Julian the winner and the numbers plainly, including if Laya lost.
- [ ] **Step 7:** After Julian approves the extra sandbox accounts, run one `sandbox cycle` and confirm `gate` candidates carry `forecast_json.source = "gate"`. Record everything in `.superpowers/sdd/sandbox-progress.md` and the brain plan note.
