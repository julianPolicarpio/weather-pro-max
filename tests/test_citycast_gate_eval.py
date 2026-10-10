import numpy as np
import pandas as pd
import pytest

from citycast.gate_eval import (MIN_BETS, brier_score, compare, pick_bets, profit_per_decision, split,
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


def test_tune_margin_picks_best_margin_and_brier():
    df = _df()
    oracle = df["won"].to_numpy().astype(float)
    assert tune_margin(df, oracle) == 0.0
    assert brier_score(df, oracle) == 0.0


def test_tune_margin_none_when_contender_cannot_reach_min_bets():
    df = _df()
    ticker_no = df["ticker"].str[1:].astype(int)
    p = np.where(ticker_no < 40, 0.9, 0.1)          # only 40 tickers qualify at margin 0.00
    assert profit_per_decision(df, p, 0.0)[1] == 40 < MIN_BETS
    assert tune_margin(df, p) is None


def test_tune_margin_enforces_min_bets_floor():
    # edge = p - ask - fee. 49 strong winners (edge .06), 11 losers at edge .02, 10 losers at edge .015.
    rows = []
    for i, (edge_p, won) in enumerate([(0.48, 1)] * 49 + [(0.44, 0)] * 11 + [(0.435, 0)] * 10):
        rows.append({"ticker": f"T{i}", "asof_utc": "2026-05-01T10:00:00+00:00", "station": "KNYC",
                     "metric": "high", "event_date": "2026-05-01", "ask": 0.40, "fee": 0.02,
                     "won": won, "profit_if_bet": won - 0.42, "p": edge_p})
    df = pd.DataFrame(rows)
    p = df["p"].to_numpy()
    v5, n5 = profit_per_decision(df, p, 0.05)
    v2, n2 = profit_per_decision(df, p, 0.02)
    assert n5 == 49 and n2 == 60 and v5 > v2          # margin 0.05 is best on profit but below the floor
    assert tune_margin(df, p) == 0.02


def test_pick_bets_rejects_duplicate_index():
    df = pd.concat([_df().iloc[:4], _df().iloc[:4]])
    with pytest.raises(ValueError, match="df index must be unique"):
        pick_bets(df, np.full(len(df), 0.9), 0.0)


def test_pick_bets_takes_earliest_qualifying_row_not_latest():
    rows = [{"ticker": "T0", "asof_utc": f"2026-05-01T{h}:00:00+00:00", "station": "KNYC", "metric": "high",
             "event_date": "2026-05-01", "ask": 0.40, "fee": 0.02, "won": 1, "profit_if_bet": 0.58}
            for h in ("10", "12", "14")]
    df = pd.DataFrame(rows)
    mask = pick_bets(df, np.array([0.1, 0.9, 0.9]), 0.0)    # rule fires at rows 2 and 3
    assert mask.tolist() == [False, True, False]
