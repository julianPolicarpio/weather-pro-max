"""Kalshi trading fees — the term `staking.evaluate` has never modelled.

WHY THIS EXISTS. The edge gate compares `p_model - price >= min_edge` on a GROSS
basis and `appbet.payout_if_win = stake / price` carries no fee term at all. The
fee peaks at P=0.50, which is precisely where a genuine edge trade lives, so the
omission is worst exactly where it matters.

THE FORMULA, verified 2026-07-30 against Kalshi's published fee schedule and
cross-checked against nowcast.trade's own calculator page:

    taker fee = round_up(0.07 * C * P * (1-P))        C contracts, P price in $
    maker fee = round_up(0.0175 * C * P * (1-P))      exactly 25% of taker

Their worked example: 10 contracts at 65c -> 0.07*10*0.65*0.35 = 0.15925, which
rounds UP to $0.16. That is the number their page shows, which is what pins the
rounding direction.

THREE DETAILS THAT ARE EASY TO GET WRONG:

1. **Rounding is UP, and it is PER ORDER, not per contract.** `C` sits inside the
   round-up. At Julian's scale this dominates: a 1-contract order at 58c owes
   0.0171 and pays $0.02 — an effective 3.4% of stake, double the unrounded 1.7%.
   A 2-contract order owes 0.0342 and pays $0.04, i.e. $0.02 each. So the
   per-contract cost FALLS as size rises, and any per-contract figure must divide
   the rounded order fee by the contract count rather than re-rounding per unit.

2. **There is NO settlement or exercise fee.** Holding a winner to resolution
   costs nothing, so breakeven is entry price plus the entry fee, full stop. An
   earlier draft of the Pick-of-the-Day design assumed a settlement term; there
   isn't one.

3. **The max is 1.75c per contract, at P=0.50.** Useful invariant for tests:
   0.07 * 1 * 0.5 * 0.5 = 0.0175 exactly.

⚠️ UNVERIFIED CAVEAT TO DETAIL 1 (raised 2026-08-01, NOT yet confirmed against
Kalshi's own docs): some order types are reported to round the fee PER FILL
rather than per order. If true, an order that fills in several pieces pays
several round-ups and this module UNDERSTATES it. Two consequences:

  * Many small fills cost more than one large fill.
  * It cuts against preferring MAKER orders purely for the 4x lower rate --
    resting limit orders are exactly the ones most likely to fill in pieces.

Harmless at the sizes this project trades (a 1-contract order cannot split), so
nothing here is changed on an unverified report. Confirm against the fee
schedule before sizing up or before building a maker-preferring order router.

WHAT THIS MODULE DOES NOT DO. It does not change `staking.evaluate`'s gate. The
gate flip (comparing `min_edge` against NET rather than gross) alters when the
desk says "place", which is decision-affecting and therefore frozen during the
live test week. This module exists so surfaces can DISPLAY net alongside gross,
and so the Pick-of-the-Day scorer can compute an honest EV. Flipping the gate is
a separate, deliberate change.
"""
from __future__ import annotations

import math

TAKER_RATE = 0.07
MAKER_RATE = 0.0175      # exactly 25% of TAKER_RATE, per Kalshi's schedule


def _round_up_cents(dollars: float) -> float:
    """Round UP to the next whole cent.

    `round(x, 9)` first is not decoration: binary floating point makes some exact
    cent values land a hair above their true value (a fee of exactly $0.01 can
    compute as 0.010000000000000002), and a naive `ceil(x * 100)` would then bill
    two cents instead of one. Nine places is far beyond any real fee's precision
    and well inside double precision, so it removes the dust without moving any
    genuine value across a cent boundary.
    """
    return math.ceil(round(dollars * 100.0, 9)) / 100.0


def _raw(rate: float, contracts: int, price: float) -> float:
    if contracts <= 0:
        return 0.0
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0, 1) exclusive, got {price!r}")
    return rate * contracts * price * (1.0 - price)


def taker_fee(contracts: int, price: float) -> float:
    """Total taker fee in dollars for an order of `contracts` at `price`.

    Rounded up once for the whole order, matching Kalshi's `round up(...)`.
    """
    return _round_up_cents(_raw(TAKER_RATE, contracts, price))


def maker_fee(contracts: int, price: float) -> float:
    """Total maker fee in dollars — 25% of taker, same rounding.

    Relevant because a resting limit order pays this instead of the taker fee, so
    working the book rather than crossing it is a real reduction in drag. The
    round-up compresses the advantage at tiny sizes (at 1 contract and 58c it is
    $0.01 against $0.02, a halving rather than a quartering).
    """
    return _round_up_cents(_raw(MAKER_RATE, contracts, price))


def fee_per_contract(contracts: int, price: float, maker: bool = False) -> float:
    """The order's fee spread over its contracts — the figure EV needs.

    Divides the ROUNDED order fee by the count. Do not re-round afterwards: the
    per-contract cost is a genuine fraction of a cent whenever the order holds
    more than one contract, and rounding it again would re-inflate the drag this
    function exists to state correctly.
    """
    if contracts <= 0:
        return 0.0
    fee = maker_fee(contracts, price) if maker else taker_fee(contracts, price)
    return fee / contracts


def breakeven(price: float, contracts: int = 1, maker: bool = False) -> float:
    """Win probability needed to break even, net of fees.

    There is no settlement fee, so this is just price + per-contract entry fee.
    At 1 contract: 80c needs 82.0%, 87c needs 88.0%, 95c needs 96.0%. Those are
    each 0.3-1.0 point HIGHER than the unrounded arithmetic suggests, which is
    why the rounding is modelled rather than ignored.
    """
    return price + fee_per_contract(contracts, price, maker=maker)


def net_ev(p_win: float, price: float, contracts: int = 1,
           maker: bool = False) -> float:
    """Expected value per contract, net of fees, for a $1 binary payout.

    `p_win` must already be the probability you actually BELIEVE — for this
    project that means the model's stated probability shaded by its measured
    calibration offset (`gating.HOUR_CALIB`), not the raw `p_model`. Feeding a
    known-overconfident probability in here produces a confidently wrong EV,
    which is the failure mode the calibration measurement exists to prevent.
    """
    return p_win - price - fee_per_contract(contracts, price, maker=maker)
