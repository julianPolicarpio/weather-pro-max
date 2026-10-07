"""Kalshi order books and the simulated fill. PURE: no network, no database.

Kalshi's book lists resting BIDS only (verified 2026-10-04). Buying YES at p
means hitting a NO bid at 1-p; buying NO means hitting a YES bid at 1-p.
Fees are charged per price level taken, each rounded up to the cent — at
least as large as Kalshi's per-order rounding, so the sandbox never flatters.
`taker_fee` is called with fractional quantities on fractional book levels, which
`highcast.fees` supports because it does not coerce `contracts` to int.
"""
import math
from dataclasses import dataclass

from highcast.fees import taker_fee

EPS = 1e-9


@dataclass(frozen=True)
class Book:
    yes: tuple[tuple[float, float], ...]
    no: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Fill:
    status: str              # "filled" | "partial" | "none"
    contracts: int
    levels: tuple[tuple[float, float], ...]
    cost: float
    fee: float

    @property
    def avg_price(self) -> float | None:
        return self.cost / self.contracts if self.contracts else None


NO_FILL = Fill("none", 0, (), 0.0, 0.0)


def _levels(raw) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(p), 4), float(q)) for p, q in (raw or []))


def parse_book(payload: dict) -> Book:
    ob = (payload or {}).get("orderbook_fp") or {}
    return Book(yes=_levels(ob.get("yes_dollars")), no=_levels(ob.get("no_dollars")))


def walk(book: Book, side: str, contracts: int, price_cap: float) -> Fill:
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    opposite = book.no if side == "yes" else book.yes
    asks = sorted((round(1.0 - p, 4), q) for p, q in opposite if 0.0 < p < 1.0 and q > 0)
    usable = [(p, q) for p, q in asks if p <= price_cap + EPS]
    available = sum(q for _, q in usable)
    n = min(contracts, int(math.floor(available + EPS)))
    if n <= 0:
        return NO_FILL
    remaining = float(n)
    taken: list[tuple[float, float]] = []
    for p, q in usable:
        if remaining <= EPS:
            break
        take = min(remaining, q)
        taken.append((p, round(take, 6)))
        remaining -= take
    cost = round(sum(p * q for p, q in taken), 4)
    fee = round(sum(taker_fee(q, p) for p, q in taken), 2)
    return Fill("filled" if n == contracts else "partial", n, tuple(taken), cost, fee)
