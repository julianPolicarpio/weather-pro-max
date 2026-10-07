"""Live Kalshi market-structure fetch for the daily high-temp series.

Verified live 2026-07-19 against
    GET https://api.elections.kalshi.com/trade-api/v2/markets
        ?series_ticker=KXHIGHLAX&status=open&limit=50
(unauthenticated). Each market in the response is one of three shapes,
distinguished by which of floor_strike/cap_strike is present:

  * top open tail    -- floor_strike set, cap_strike null.
    e.g. floor_strike=80, yes_sub_title="81° or above". rules_primary reads
    "greater than 80°" (strict), so the open range starts at floor_strike+1.
  * bottom open tail -- cap_strike set, floor_strike null.
    e.g. cap_strike=73, yes_sub_title="72° or below" -> hi = cap_strike-1.
  * between bucket   -- both set, e.g. floor_strike=79, cap_strike=80,
    yes_sub_title="79° to 80°" -> lo/hi map directly onto the strikes,
    no offset needed.

The subtitle strings above were cross-checked against every shape in the live
payload and matched this floor_strike/cap_strike arithmetic exactly, so
parse_markets derives lo/hi from the strikes (robust to formatting) rather
than parsing the subtitle text (fragile).
"""
from datetime import date

from highcast.config import AppConfig
from highcast.http import get_json
from highcast.metric import MetricSpec

MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
           "AUG", "SEP", "OCT", "NOV", "DEC"]


def event_date_fmt(target_date: str) -> str:
    """iso date -> Kalshi event-ticker date suffix, e.g. '2026-07-19' -> '26JUL19'."""
    d = date.fromisoformat(target_date)
    return f"{d:%y}{_MONTHS[d.month - 1]}{d:%d}"


def event_ticker(series: str, target_date: str) -> str:
    """target_date iso -> full event ticker, e.g. ('KXHIGHLAX', '2026-07-19')
    -> 'KXHIGHLAX-26JUL19'."""
    return f"{series}-{event_date_fmt(target_date)}"


def _lo_hi(floor_strike, cap_strike) -> tuple[int | None, int | None]:
    if floor_strike is not None and cap_strike is not None:
        return int(floor_strike), int(cap_strike)
    if floor_strike is not None:
        return int(floor_strike) + 1, None
    if cap_strike is not None:
        return None, int(cap_strike) - 1
    return None, None


def _label(lo: int | None, hi: int | None) -> str | None:
    if lo is None and hi is None:
        return None
    if lo is None:
        return f"≤{hi}"
    if hi is None:
        return f"≥{lo}"
    return f"{lo}–{hi}"


def _to_float(x) -> float | None:
    return None if x is None else float(x)


def parse_markets(payload: dict, event_ticker_value: str) -> list[dict]:
    """Pure, fixture-testable: payload from MARKETS_URL -> sorted list of
    {"label", "lo", "hi", "ticker", "yes_bid", "yes_ask", "last_price",
    "close_time"} for the given event_ticker. lo/hi are integer °F bounds, None for an open tail.
    Prices are floats in [0, 1], or None if absent.

    `close_time` is retained because it answers "how long do I have to act", and
    dropping it made a load-bearing question unanswerable from our own data. It
    was worth keeping: measured 2026-07-30 for target 2026-07-30, every market
    stays open ~11 hours PAST the last of the four commit hours --

        KMIA low  commits 17:00 MNL, closes 13:00 MNL next day
        KLAX low  commits 19:00 MNL, closes 16:00 MNL next day
        KMIA high commits 23:00 MNL, closes 12:59 MNL next day
        KLAX high commits 02:00 MNL, closes 15:59 MNL next day

    -- which refutes the assumption that the four markets must be taken in
    sequence, and confirms HANDOFF's previously-unverified "closes 15:59 MNL the
    next day" for KLAX exactly. A best-of-four choice is genuinely available, and
    available at a waking hour rather than at 02:00.
    """
    out = []
    for m in payload.get("markets") or []:
        if m.get("event_ticker") != event_ticker_value:
            continue
        if m.get("market_type") != "binary":
            continue
        lo, hi = _lo_hi(m.get("floor_strike"), m.get("cap_strike"))
        label = _label(lo, hi)
        if label is None:
            continue
        out.append({
            "label": label, "lo": lo, "hi": hi,
            # THE MARKET'S OWN TICKER, retained because it is the only string
            # an order can be placed against. It was being dropped, which left
            # the order bot CONSTRUCTING one from the series + date + bucket
            # label -- a guess containing an en dash, which no exchange ticker
            # has ever contained. Every order would have been rejected. The
            # real value is right here in the payload we already fetch.
            "ticker": m.get("ticker"),
            "yes_bid": _to_float(m.get("yes_bid_dollars")),
            "yes_ask": _to_float(m.get("yes_ask_dollars")),
            "last_price": _to_float(m.get("last_price_dollars")),
            # Kept verbatim as the API's ISO string; callers convert. Absent on
            # older fixtures, so consumers must treat None as "unknown", never as
            # "closed" -- refusing to act on a missing close time would be worse
            # than acting without it.
            "close_time": m.get("close_time"),
        })
    out.sort(key=lambda r: r["hi"] if r["hi"] is not None else float("inf"))
    return out


def resolve_price(r: dict) -> float | None:
    """market_price for a parsed range: a LIVE two-sided book wins, else the last
    trade, else whichever single side exists, else None.

    PRECEDENCE INVERTED 2026-07-31. This used to return `last_price` whenever it
    was present, so a trade from hours ago overrode a live bid/ask that said
    otherwise. Measured across 26,460 stored quotes, 16.1% sat OUTSIDE their own
    book -- median deviation 1c, 78.9% within 2c, but a 52c worst case.

    The small median is why this was NOT the cause of the losing bets (they were
    priced within 0-2c of their book at commit) -- and the 52c tail is why an
    unattended caller must not inherit the old behaviour anyway.
    """
    bid, ask = r.get("yes_bid"), r.get("yes_ask")
    if bid is not None and ask is not None and ask >= bid:
        return round((bid + ask) / 2, 4)
    lp = r.get("last_price")
    if lp is not None:
        return lp
    return bid if bid is not None else ask


def fetch_ranges(
    cfg: AppConfig, target_date: str, metric: MetricSpec | None = None
) -> list[dict] | None:
    """Markets for target_date's event -> parse_markets(...), or None on any
    network/shape error, or when kalshi_series is blank (feature disabled).
    series comes from metric.kalshi_series when a MetricSpec is passed
    (e.g. the low series), else falls back to cfg.station.kalshi_series
    (the existing high-series behavior, unchanged for callers that pass no
    metric). Never raises -- the dashboard must not break when Kalshi is
    unreachable."""
    series = metric.kalshi_series if metric is not None else cfg.station.kalshi_series
    if not series:
        return None
    try:
        ticker = event_ticker(series, target_date)
        payload = get_json(MARKETS_URL, {
            "series_ticker": series, "status": "open", "limit": 50,
        })
        ranges = parse_markets(payload, ticker)
        return ranges or None
    except Exception:  # noqa: BLE001 - any failure means "unavailable", not a crash
        return None
