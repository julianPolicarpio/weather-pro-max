import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citycast import db
from citycast.laya_io import QUESTION, export, sample_train, score, state, to_record

ROW = {"station": "KLAX", "metric": "high", "event_date": "2026-01-05", "lo": 66, "hi": 67,
       "hours_to_close": 14.52, "lead_h": 20.0, "p_model": 0.41, "model_median": 66.8, "model_iqr": 3.1,
       "ask": 0.33, "bid": 0.30, "market_p": 0.315, "edge": 0.06, "run_gap_to_lo": float("nan"),
       "run_gap_to_hi": float("nan"), "month": 1, "won": 1, "ticker": "T", "asof_utc": "2026-01-05T00:00:00+00:00"}


def test_state_is_compact_and_ordered():
    s = json.loads(state(ROW))
    assert list(s) == ["city", "metric", "bucket", "hours_to_close", "lead", "model_p", "model_median",
                       "model_iqr", "ask", "bid", "market_p", "edge_after_fee", "gap_to_lo", "gap_to_hi", "month"]
    assert s["bucket"] == "66-67" and s["lead"] == "same_day"
    assert s["gap_to_lo"] is None and s["gap_to_hi"] is None
    assert len(state(ROW)) < 400


def test_state_carries_the_running_high_gap():
    row = {**ROW, "run_gap_to_hi": 1.24, "lead_h": 3.0}
    s = json.loads(state(row))
    assert s["lead"] == "same_day" and s["gap_to_hi"] == 1.2 and s["gap_to_lo"] is None
    assert json.loads(state({**row, "run_gap_to_lo": -0.46}))["gap_to_lo"] == -0.5
    assert len(state({**row, "run_gap_to_lo": -0.46})) < 400


def test_record_matches_pinned_fixture_shape():
    fixture = json.loads(Path("tests/fixtures/citycast/laya_record.json").read_text())
    rec = to_record(ROW)
    assert set(rec) == set(fixture)
    assert json.dumps(QUESTION) in json.dumps(rec) or rec.get("questions") == QUESTION
    assert rec["questions"] == fixture["questions"]
    assert rec["expected"] == fixture["expected"] == {"win": True}
    assert to_record({**ROW, "won": 0})["expected"] == {"win": False}


def _city_day_frame(rows_per_day=3):
    recs = [{**ROW, "station": st, "metric": m, "event_date": f"2026-01-{d:02d}",
             "ticker": f"{st}-{m}-{d}-{k}"}
            for st in ("KLAX", "KMIA") for m in ("high", "low") for d in range(1, 21)
            for k in range(rows_per_day)]
    return pd.DataFrame(recs)   # 80 city-days x 3 rows = 240 rows


def _days(frame):
    return frame.groupby(["station", "metric", "event_date"]).size()


def test_sample_keeps_whole_city_days():
    df = _city_day_frame()
    s = sample_train(df, frac=0.6, cap=1000, seed=3)
    kept = _days(s)
    assert len(kept) == 48                       # 60% of the 80 city-days, no cap involved
    assert (kept == 3).all()                     # every kept city-day is complete
    assert len(s) == 144


def test_sample_is_deterministic_per_seed_and_varies_across_seeds():
    df = _city_day_frame()
    a = sample_train(df, frac=0.6, cap=1000, seed=3)
    b = sample_train(df, frac=0.6, cap=1000, seed=3)
    c = sample_train(df, frac=0.6, cap=1000, seed=4)
    assert a.equals(b)
    assert set(a["ticker"]) != set(c["ticker"])


def test_sample_caps_to_exactly_cap_rows_deterministically():
    df = _city_day_frame()
    s = sample_train(df, frac=0.6, cap=50, seed=3)
    assert len(s) == 50 and s.equals(sample_train(df, frac=0.6, cap=50, seed=3))
    assert set(s["ticker"]) <= set(sample_train(df, frac=0.6, cap=1000, seed=3)["ticker"])
    assert len(sample_train(df, frac=0.6, cap=144, seed=3)) == 144   # at the cap: untouched


def test_export_writes_jsonl(tmp_path):
    df = pd.DataFrame([ROW, {**ROW, "ticker": "U"}])
    assert export(df, tmp_path / "sub" / "train.jsonl") == 2
    lines = (tmp_path / "sub" / "train.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == to_record(ROW)


def test_export_refuses_the_test_split(tmp_path):
    df = pd.DataFrame([ROW, {**ROW, "ticker": "U", "event_date": "2026-07-01"}])
    with pytest.raises(ValueError):
        export(df, tmp_path / "train.jsonl")
    assert not (tmp_path / "train.jsonl").exists()
    last_val = pd.DataFrame([{**ROW, "event_date": "2026-06-30"}])
    assert export(last_val, tmp_path / "val.jsonl") == 1


def test_batched_scoring_is_cached(tmp_path):
    df = pd.DataFrame([ROW, {**ROW, "ticker": "U"}])
    sizes = []

    def fn(states):
        sizes.append(len(states))
        return [0.7] * len(states)
    with closing(db.connect(tmp_path / "h.db")) as c:
        p1 = score(df, fn, "L0", c, batch=1)
        assert sizes == [1, 1]
        p2 = score(df, fn, "L0", c, batch=1)
    assert np.allclose(p1, 0.7) and np.allclose(p2, 0.7) and sizes == [1, 1]


def test_batch_chunks_and_df_order(tmp_path):
    df = pd.DataFrame([{**ROW, "ticker": f"T{i}"} for i in range(5)])
    sizes = []

    def fn(states):
        sizes.append(len(states))
        return [json.loads(s)["month"] / 100 + i / 1000 for i, s in enumerate(states)]
    with closing(db.connect(tmp_path / "h.db")) as c:
        p = score(df, fn, "L1", c, batch=2)
    assert sizes == [2, 2, 1] and len(p) == 5


def _insert_scores(conn, contender, items):
    with conn:
        conn.executemany("INSERT INTO gate_scores VALUES (?,?,?,?)",
                         [(contender, t, ROW["asof_utc"], p) for t, p in items])


def test_score_returns_each_rows_own_value_in_df_order(tmp_path):
    order = ["T4", "T1", "T5", "T0", "T3", "T2"]
    df = pd.DataFrame([{**ROW, "ticker": t, "p_model": 0.1 + 0.1 * int(t[1])} for t in order])
    cached = {"T1": 0.91, "T0": 0.92, "T2": 0.93}
    calls = []

    def fn(states):
        calls.append(len(states))
        return [json.loads(s)["model_p"] / 10 for s in states]   # T4 -> 0.05, T5 -> 0.06, T3 -> 0.04
    with closing(db.connect(tmp_path / "h.db")) as c:
        _insert_scores(c, "L2", cached.items())
        p = score(df, fn, "L2", c, batch=2)
    expected = {"T4": 0.05, "T5": 0.06, "T3": 0.04, **cached}
    assert np.allclose(p, [expected[t] for t in order])
    assert calls == [2, 1]                       # only the three uncached rows were scored


def test_duplicate_keys_are_scored_once_and_share_the_score(tmp_path):
    df = pd.DataFrame([{**ROW, "ticker": "A"}, {**ROW, "ticker": "B"}, {**ROW, "ticker": "A"}])
    seen = []

    def fn(states):
        seen.append(len(states))
        return [0.1 * (len(seen) + i) for i in range(len(states))]
    with closing(db.connect(tmp_path / "h.db")) as c:
        p = score(df, fn, "L3", c, batch=1)
        assert c.execute("SELECT COUNT(*) FROM gate_scores").fetchone()[0] == 2
    assert seen == [1, 1]                        # two unique keys, one call each
    assert p[0] == p[2] and p[0] != p[1]
    seen.clear()
    with closing(db.connect(tmp_path / "h2.db")) as c:
        score(df, fn, "L3", c, batch=256)
    assert seen == [2]                           # one chunk of the two unique states, not three


@pytest.mark.parametrize("n_returned", [1, 3])
def test_wrong_length_output_raises_and_stores_nothing_for_that_chunk(tmp_path, n_returned):
    df = pd.DataFrame([{**ROW, "ticker": f"T{i}"} for i in range(4)])
    calls = {"n": 0}

    def fn(states):
        calls["n"] += 1
        return [0.5] * (len(states) if calls["n"] == 1 else n_returned)
    with closing(db.connect(tmp_path / "h.db")) as c:
        with pytest.raises(ValueError, match="scores for 2 states"):
            score(df, fn, "L4", c, batch=2)
        stored = sorted(r["ticker"] for r in c.execute("SELECT ticker FROM gate_scores"))
    assert stored == ["T0", "T1"]                # first chunk kept, bad chunk absent


def test_scoring_resumes_after_a_crash(tmp_path):
    df = pd.DataFrame([ROW, {**ROW, "ticker": "U"}])
    n = {"calls": 0}

    def bad(states):
        n["calls"] += 1
        if n["calls"] == 2:
            raise RuntimeError("boom")
        return [0.6] * len(states)
    with closing(db.connect(tmp_path / "h.db")) as c:
        with pytest.raises(RuntimeError):
            score(df, bad, "L1", c, batch=1)
        stored = c.execute("SELECT ticker, p_win FROM gate_scores WHERE contender='L1'").fetchall()
        assert [(r["ticker"], r["p_win"]) for r in stored] == [("T", 0.6)]
        with closing(db.connect(tmp_path / "h.db")) as other:   # a separate connection sees only committed rows
            seen = other.execute("SELECT ticker, p_win FROM gate_scores WHERE contender='L1'").fetchall()
        assert [(r["ticker"], r["p_win"]) for r in seen] == [("T", 0.6)]
        sizes = []

        def good(states):
            sizes.append(len(states))
            return [0.8] * len(states)
        p = score(df, good, "L1", c, batch=1)
    assert sizes == [1] and np.allclose(p, [0.6, 0.8])
