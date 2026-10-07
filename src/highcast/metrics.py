def crps_pmf(pmf: dict[int, float], y: int) -> float:
    # iterate the FULL integer range, not just support points - gaps between
    # support points still contribute (cum - indicator)^2 per skipped degree.
    # Clamp y into the pmf's own support window first: an out-of-range y
    # (bad data, a bug upstream, whatever) must not turn `range(lo, hi + 1)`
    # into a multi-million-iteration loop. Once clamped, y beyond either
    # edge behaves like the corresponding edge value - a constant tail
    # penalty rather than one that grows (and iterates) without bound.
    support_lo, support_hi = min(pmf), max(pmf)
    y_clamped = max(support_lo, min(support_hi, y))
    cum = 0.0
    score = 0.0
    for t in range(support_lo, support_hi + 1):
        cum += pmf.get(t, 0.0)
        score += (cum - (1.0 if t >= y_clamped else 0.0)) ** 2
    return score


def brier(pmf: dict[int, float], y: int, t: int) -> float:
    return (pmf.get(t, 0.0) - (1.0 if y == t else 0.0)) ** 2


def pinball(quantiles: dict[float, float], y: float) -> float:
    total = 0.0
    for q, v in quantiles.items():
        total += q * (y - v) if y >= v else (1 - q) * (v - y)
    return total / len(quantiles)


def reliability(records: list[tuple[dict, int]], level: float) -> float:
    """Fraction of outcomes at or below the nominal `level` quantile."""
    hits = []
    for qs, y in records:
        v = qs.get(level)
        if v is None:
            raise KeyError(
                f"reliability: quantile level {level!r} not found in record "
                f"(available levels: {sorted(qs)})")
        hits.append(1.0 if y <= v else 0.0)
    return sum(hits) / len(hits) if hits else float("nan")
