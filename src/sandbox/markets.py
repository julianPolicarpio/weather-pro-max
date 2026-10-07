"""Turn a Kalshi /markets payload into Market rows.

Bucket structure (lo/hi/label) comes from highcast's verified parse_markets;
this module only adds what the sandbox needs on top: event date, metric,
station, and top-of-book sizes.
"""
from datetime import datetime

from highcast.fetchers.kalshi import parse_markets
from sandbox.models import Market
from sandbox.stations import metric_from_series, station_from_rules


def parse_event_date(event_ticker: str) -> str:
    """'KXHIGHNY-26OCT04' -> '2026-10-04'."""
    suffix = event_ticker.rsplit("-", 1)[1]
    return datetime.strptime(suffix, "%y%b%d").date().isoformat()


def normalize_ts(ts: str | None) -> str | None:
    """Kalshi's trailing 'Z' -> '+00:00', so stored times compare as strings."""
    if not ts:
        return None
    return ts[:-1] + "+00:00" if ts.endswith("Z") else ts


def _f(x) -> float | None:
    return None if x is None else float(x)


def parse_series_payload(series: str, payload: dict) -> list[Market]:
    raw = {m["ticker"]: m for m in payload.get("markets") or [] if m.get("ticker")}
    metric = metric_from_series(series)
    events = sorted({m.get("event_ticker") for m in raw.values() if m.get("event_ticker")})
    out: list[Market] = []
    seen: set[str] = set()
    for ev in events:
        for r in parse_markets(payload, ev):
            if r["ticker"] in seen:   # a market repeated in the payload
                continue
            seen.add(r["ticker"])
            m = raw[r["ticker"]]
            out.append(Market(
                ticker=r["ticker"], series=series, event_ticker=ev,
                event_date=parse_event_date(ev), lo=r["lo"], hi=r["hi"],
                label=r["label"], metric=metric,
                station=station_from_rules(m.get("rules_primary")),
                yes_bid=r["yes_bid"], yes_ask=r["yes_ask"],
                yes_bid_size=_f(m.get("yes_bid_size_fp")),
                yes_ask_size=_f(m.get("yes_ask_size_fp")),
                close_time=normalize_ts(r["close_time"]),
            ))
    return out


def usable_bid(m: Market) -> float | None:
    return m.yes_bid if m.yes_bid is not None and 0.0 < m.yes_bid < 1.0 else None


def usable_ask(m: Market) -> float | None:
    return m.yes_ask if m.yes_ask is not None and 0.0 < m.yes_ask < 1.0 else None
