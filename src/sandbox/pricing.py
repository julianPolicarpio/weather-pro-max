"""Market probabilities and scoring."""
from sandbox.markets import usable_ask, usable_bid
from sandbox.models import Market


def mid(m: Market) -> float | None:
    bid, ask = usable_bid(m), usable_ask(m)
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    return ask if ask is not None else bid


def devig(markets: list[Market]) -> dict[str, float]:
    """One event's ladder -> probabilities summing to 1 (mids rescaled)."""
    mids = {m.ticker: v for m in markets if (v := mid(m)) is not None}
    total = sum(mids.values())
    return {t: v / total for t, v in mids.items()} if total > 0 else {}


def multiclass_brier(probs: dict[str, float], winner: str) -> float:
    score = sum((p - (1.0 if t == winner else 0.0)) ** 2 for t, p in probs.items())
    return score + (0.0 if winner in probs else 1.0)
