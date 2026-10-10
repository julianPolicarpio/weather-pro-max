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
    if not df.index.is_unique:
        raise ValueError("df index must be unique")
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


def tune_margin(df: pd.DataFrame, p) -> float | None:
    """Margin with the best profit per decision among those reaching MIN_BETS bets.

    None when no margin reaches MIN_BETS (the bet count only falls as the margin rises,
    so margin 0.00 already has fewer): an untunable contender cannot win (spec §4.7
    revised), so there is never a silent fallback margin.
    """
    best, best_v = None, -np.inf
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
