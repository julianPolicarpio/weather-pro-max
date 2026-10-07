import numpy as np


def bucket_pair(t: int) -> tuple[int, int]:
    """The (odd, odd+1) 2-degree kalshi bucket containing integer temp t."""
    lo = t if t % 2 == 1 else t - 1
    return lo, lo + 1


def ladder_lo(t: int, parity: int) -> int:
    """Low edge of the 2F bucket containing `t` on a ladder of given parity.

    `parity` is the parity of the ladder's LOW EDGES: 0 = even-based
    (…68-69, 70-71…), 1 = odd-based (…69-70, 71-72…).
    """
    return t - ((t - parity) % 2)


def modal_bucket(pmf: dict[int, float], parity: int) -> tuple[int, float]:
    """(low edge, probability) of the heaviest 2F bucket on a `parity` ladder."""
    agg: dict[int, float] = {}
    for k, v in pmf.items():
        lo = ladder_lo(int(k), parity)
        agg[lo] = agg.get(lo, 0.0) + v
    lo = max(agg, key=agg.get)
    return lo, agg[lo]


def expected_bucket_hit(pmf: dict[int, float], y: int) -> float:
    """P(the model's modal 2F bucket contains `y`) averaged over BOTH ladders.

    ⚠️ THE LADDER PARITY IS NOT FIXED. Kalshi centres each day's ladder on the
    consensus forecast, so which parity you get is effectively a coin flip.
    Measured over 68 settled days per series:

        KXLOWTLAX  36 odd / 32 even        KXHIGHMIA  35 odd / 33 even
        KXHIGHLAX  34 odd / 34 even        KXLOWTMIA  29 odd / 39 even

    Every bucket-hit figure in this project used to assume EVEN (hour_curve's
    `kalshi_bucket`, cities' `_kalshi_lo`), which is right about half the time
    and silently wrong the rest. That fed `gating.HOUR_ACCURACY` and, through
    it, the choice of `MIN_DECISION_HOUR` -- so the bug reached the commit
    hours, not just the display.

    Averaging the two parities is the honest estimator: the true ladder is
    unknowable for historical days (Kalshi only serves ~68 days of settled
    markets), and a bettor facing an unknown ladder experiences exactly this
    expectation. Returns 0.0, 0.5 or 1.0 per day; the MEAN over many days is
    the quantity to read.

    NOTE this is for SCORING ONLY. The live path never guesses: it reads each
    market's real floor/cap from the Kalshi API and sums `pmf_mass` over it.
    """
    hits = 0.0
    for parity in (0, 1):
        lo, _ = modal_bucket(pmf, parity)
        if lo <= int(y) <= lo + 1:
            hits += 1.0
    return hits / 2.0


def pmf_from_buckets(buckets: dict | None) -> dict[int, float] | None:
    """Convert a str-keyed per-degree bucket dict (as stored/serialized via
    JSON) to an int-keyed pmf. None-safe: passing None returns None so
    callers can chain straight off an optional stored prediction."""
    if buckets is None:
        return None
    return {int(k): v for k, v in buckets.items()}


def argmax_pair(pmf: dict | None) -> dict | None:
    """The model's most-confident odd-based 2F bucket. Returns
    {label, lo, hi, prob} or None for an empty/None pmf. This is the single
    shared definition of "the pick" — the display gate, the staking ruleset,
    and the scorecard all read it, so the number on screen can never disagree
    with the number being judged/sized."""
    if not pmf:
        return None
    lo0 = min(pmf)
    lo0 = lo0 if lo0 % 2 == 1 else lo0 - 1
    hi_key = max(pmf)
    pmax_lo = hi_key if hi_key % 2 == 1 else hi_key - 1
    best = None
    for lo in range(lo0, pmax_lo + 1, 2):
        hi = lo + 1
        p = sum(v for k, v in pmf.items() if lo <= k <= hi)
        if best is None or p > best["prob"]:
            best = {"label": f"{lo}-{hi}", "lo": lo, "hi": hi, "prob": p}
    return best


def pmf_mass(pmf: dict | None, lo: int | None, hi: int | None) -> float | None:
    """Sum of a per-degree pmf over the inclusive integer range [lo, hi];
    either bound may be None for an open tail. None in, None out (no
    prediction available yet). Shared by the synthetic _kalshi_buckets and
    the real-market _kalshi_market so the mass-summing logic lives once."""
    if pmf is None:
        return None
    if lo is None:
        vals = (v for k, v in pmf.items() if k <= hi)
    elif hi is None:
        vals = (v for k, v in pmf.items() if k >= lo)
    else:
        vals = (v for k, v in pmf.items() if lo <= k <= hi)
    return round(sum(vals), 4)


def cdf_from_quantiles(quantiles: dict[float, float]):
    """Monotone CDF from (level -> value) pairs with exponential tails."""
    items = sorted(quantiles.items())
    levels = np.array([l for l, _ in items], dtype=float)
    vals = np.maximum.accumulate(np.array([v for _, v in items], dtype=float))
    lo_scale = max(1.0, (vals[1] - vals[0]) if len(vals) > 1 else 1.0)
    hi_scale = max(1.0, (vals[-1] - vals[-2]) if len(vals) > 1 else 1.0)

    def cdf(x: float) -> float:
        if x <= vals[0]:
            return float(levels[0] * np.exp((x - vals[0]) / lo_scale))
        if x >= vals[-1]:
            return float(1.0 - (1.0 - levels[-1]) * np.exp(-(x - vals[-1]) / hi_scale))
        return float(np.interp(x, vals, levels))

    return cdf


def bucket_probs(quantiles: dict[float, float]) -> dict[int, float]:
    cdf = cdf_from_quantiles(quantiles)
    vals = list(quantiles.values())
    lo = int(np.floor(min(vals))) - 15
    hi = int(np.ceil(max(vals))) + 15
    pmf = {t: max(0.0, cdf(t + 0.5) - cdf(t - 0.5)) for t in range(lo, hi + 1)}
    total = sum(pmf.values())
    # no probability-threshold filter: every bucket in the support window keeps
    # its exact normalized mass so the hot tail is never hard-zeroed
    return {t: p / total for t, p in pmf.items()}


def truncate_at_bound(pmf: dict | None, bound: float | None,
                      extremum: str) -> dict | None:
    """Drop the mass a settled ratchet has already made IMPOSSIBLE, renormalised.

    A daily MINIMUM can only fall. Once the running min stands at 68.10, no
    outcome at 69 or above can occur, yet the nowcast still assigned 1.1% to
    "≥70" on 2026-08-01 -- probability spent on a dead bucket, and every live
    bucket diluted by exactly that much. The same holds mirrored for a MAXIMUM.

    ⚠️ THE BOUND IS ROUNDED BEFORE COMPARING, because settlement rounds. With a
    running min of 68.10 the day can still settle at 68 (anything in
    [67.50, 68.49] rounds there), so 68 must stay ALIVE; only 69 and above are
    dead. Comparing against the raw 68.10 would kill 68 and throw away the most
    likely outcome on the board.

    Returns a NEW dict; the input is untouched. `None` in, `None` out. If a
    degenerate bound would kill everything, the original pmf is returned rather
    than an empty or all-zero one -- a caller that has to choose a bucket must
    still get a distribution, and a silent empty dict would surface as a
    mysterious "no prediction".
    """
    if pmf is None or bound is None:
        return pmf
    limit = round(bound)
    if extremum == "min":
        kept = {k: v for k, v in pmf.items() if k <= limit}
    else:
        kept = {k: v for k, v in pmf.items() if k >= limit}
    total = sum(kept.values())
    if not kept or total <= 0:
        return dict(pmf)
    return {k: v / total for k, v in kept.items()}


def dead_mass(pmf: dict | None, bound: float | None, extremum: str) -> float:
    """Share of the pmf sitting on outcomes the ratchet has already ruled out.

    Reported rather than silently discarded: a large dead mass means the model
    and the observations disagree about what is still possible, which is worth
    seeing on the board even after the mass itself is removed.
    """
    if pmf is None or bound is None:
        return 0.0
    total = sum(pmf.values())
    if total <= 0:
        return 0.0
    limit = round(bound)
    if extremum == "min":
        bad = sum(v for k, v in pmf.items() if k > limit)
    else:
        bad = sum(v for k, v in pmf.items() if k < limit)
    return bad / total
