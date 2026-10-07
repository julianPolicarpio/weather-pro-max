"""Plain data carried between sandbox modules."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    ticker: str
    series: str
    event_ticker: str
    event_date: str          # Kalshi event date = the station's climate day
    lo: int | None           # inclusive °F; None for the bottom open tail
    hi: int | None           # inclusive °F; None for the top open tail
    label: str
    metric: str | None       # "high" | "low"
    station: str | None      # ICAO from the market's own rules text
    yes_bid: float | None
    yes_ask: float | None
    yes_bid_size: float | None
    yes_ask_size: float | None
    close_time: str | None   # UTC ISO with +00:00


@dataclass(frozen=True)
class Snapshot:
    id: int
    taken_at: str
    markets: tuple[Market, ...]

    def events(self) -> dict[str, list[Market]]:
        out: dict[str, list[Market]] = {}
        for m in self.markets:
            out.setdefault(m.event_ticker, []).append(m)
        return out


@dataclass(frozen=True)
class Intent:
    account: str
    ticker: str
    side: str                # "yes" | "no"
    contracts: int
    price_cap: float
    snapshot_id: int | None
    p_bot: float | None
    p_market: float | None
    priority: float = 0.0    # funding order within a bot's hour: higher first (edge or p)


@dataclass
class Candidate:
    bot: str
    ticker: str
    series: str
    station: str | None
    metric: str | None
    event_date: str
    snapshot_id: int
    decided_at: str
    p_bot: float
    pmf_json: str            # {ticker: p} for the whole event
    p_market: float | None
    yes_bid: float | None
    yes_ask: float | None
    spread: float | None
    ask_size: float | None
    minutes_to_close: float | None
    fee_per_contract: float | None
    edge_after_fee: float | None
    placed: bool
    reason: str              # edge | below_min_edge | no_ask | already_held | no_fill |
                             # insufficient_cash | stale_snapshot | bot_error
    gates_json: str | None = None    # populated in sub-project 3
    regime_json: str | None = None   # populated in sub-project 3
    forecast_json: str | None = None  # the forecast behind the pmf, e.g. MOS mu/sigma
