"""Should the desk place an order today, and at what price? PURE DECISION LOGIC.

Design: an internal design note.

THIS MODULE KNOWS NOTHING ABOUT HTTP. It takes a market snapshot and a gate
verdict and returns an intent. `kalshi_client` does the network. Keeping them
apart is what makes the money rails testable without a network and the transport
testable without a model -- neither is testable if they are one file.

THE PRICE CEILING IS THE POINT OF THIS MODULE.
Julian does not hold to settlement: he rests a sell limit near 99c and exits when
the market reprices to near-certainty. That changes TIMING, not expected value.
What decides profit is the entry price against the true hit rate:

    buy at 0.70 -> needs 73.5%       buy at 0.90 -> needs 92.9%
    buy at 0.80 -> needs 83.7%       buy at 0.95 -> needs 98.0%

KMIA HIGH measures 84.2% in August (n=155), so **0.81 is the highest price that
can profit** on an AVERAGE day, and the ceiling was 0.80.

2026-08-02: the ceiling is 0.84 and an EDGE RULE now sits beside it. The table
above is the unconditional hit rate; on days the model states 90%+ it actually
lands 96.6% (n=558 day-ladders), and those days CAN profit at 0.84. What makes
that safe is not the ceiling but `MIN_EDGE`: the bot must clear
ask + fee + 3 points on the CALIBRATED number for the specific day, so a
mediocre day at 0.84 is refused while a 96%-day at 0.84 is taken. Until this
existed the bot could buy at the ceiling with a calibrated probability BELOW it
-- negative EV that passed every rail.

The reason this rail exists rather than being left to judgement: in a simulated
week KLAX HIGH went SIX FOR SIX and still carried EV -20.5c/contract, because its
average entry was 0.91 against 72.6% accuracy. **A high win rate above 85c is a
losing system that feels like a winning one.** Nothing else here matters as much.

EVERYTHING FAILS CLOSED. A missing price, a crossed book, an unparseable field,
an exception -- all return place=False. For an unattended caller "I could not
tell" and "no" are the same answer.
"""
from __future__ import annotations

from dataclasses import dataclass

# The one market with an admissible confidence floor. The other three failed the
# "measurably right 4 days in 5" rule; KLAX HIGH is excluded by two independent
# lines of evidence (its own calibration AND its prices).
TRADED_STATION = "KMIA"
TRADED_METRIC = "high"
TRADED_HOUR = 14                 # station-local; 02:00 Manila in summer

# The most this bot will pay per contract.
#
# 2026-08-02: 0.80 -> 0.84, Julian's stated tolerance ("70 to 84 cents, just not
# 90 — there's nothing there"). Safe to raise ONLY because MIN_EDGE below now
# exists: before it, nothing stopped the bot buying at the ceiling with a
# calibrated probability BELOW it, which loses money by construction. The
# ceiling caps the downside per contract; the edge rule is what makes the trade
# worth taking. Raising one without the other is how you lose slowly.
PRICE_CEILING = 0.84


def taker_fee(price: float, count: int = 1) -> float:
    """Kalshi's taker fee: roundup(0.07 * C * P * (1-P)), to the cent.

    Charged ON TOP of the price, so a 0.80 fill costs 0.82. Ignoring it is how
    a 3-cent 'edge' becomes a 1-cent loss: the fee peaks at mid-ladder prices,
    exactly where this bot operates.
    """
    import math
    return math.ceil(0.07 * count * price * (1.0 - price) * 100) / 100


# THE EDGE RULE — added 2026-08-02, because the bot had NONE.
#
# It checked a confidence floor and a price ceiling and nothing that compared
# the two. A calibrated 0.72 against an 0.80 ask passed both gates and is -10%
# after fees. Every rail was about NOT trading; none was about the trade being
# WORTH IT.
#
# The requirement is fee-aware: calibrated >= ask + fee + MIN_EDGE. 3 points
# matches the margin `signals` already demands of the paper ledger, so the bot
# and the ledger now agree on what "an edge" means.
MIN_EDGE = 0.03

# The floor lives in `gating`, keyed by (station, metric, hour), and is READ
# rather than copied.
#
# ⚠️ IT WAS COPIED HERE AS A BARE 0.70 UNTIL 2026-08-02, so re-measuring it in
# `gating` (0.70 -> 0.60) silently did nothing to the bot: the stale duplicate
# still bound. A constant with two homes has one that is wrong.
def confidence_floor() -> float:
    from highcast import gating
    return gating.CONFIDENCE_FLOOR.get(
        (TRADED_STATION, TRADED_METRIC, TRADED_HOUR), 1.01)


@dataclass(frozen=True)
class Intent:
    """What the bot wants to do. `place=False` always carries a reason.

    `edge` is `calibrated - ask` whenever both were known at the point of
    refusal or placement -- i.e. from the market-selection step onward. It is
    None for refusals that never get that far (scope, hour, frequency, the
    gate verdict, calibration/floor), because there is no ask to compare
    against yet. See `gating.MAX_EDGE` for why this is logged at all.
    """
    place: bool
    reason: str
    bucket: str | None = None
    limit_price: float | None = None
    contracts: int = 0
    edge: float | None = None


def _no(reason: str, *, bucket: str | None = None,
        edge: float | None = None) -> Intent:
    return Intent(False, reason, bucket=bucket, edge=edge)


def decide(*, station: str, metric: str, hour: int, target: str,
           ranges: list[dict] | None, gates_ok: bool,
           refusals: tuple[str, ...], calibrated: float | None,
           orders_today: int, budget: float) -> Intent:
    """The whole decision. Ordered cheapest-refusal-first so a log line names the
    most fundamental reason rather than an incidental one."""
    # --- scope ---------------------------------------------------------------
    if station != TRADED_STATION or metric != TRADED_METRIC:
        return _no(f"market {station} {metric} is not traded — this bot is "
                   f"{TRADED_STATION} {TRADED_METRIC} only (the others have no "
                   f"admissible confidence floor)")
    if hour != TRADED_HOUR:
        return _no(f"hour {hour} is not the commit hour {TRADED_HOUR}, where "
                   f"every number behind this bot was measured")

    # --- frequency: the wake loop runs every 5 minutes -----------------------
    if orders_today >= 1:
        return _no(f"already ordered {orders_today} time(s) for {target} — "
                   f"the cap is one per day")

    # --- the gates -----------------------------------------------------------
    if not gates_ok:
        return _no(f"decision_gates refused: {', '.join(refusals) or 'unknown'}")

    # --- calibrated confidence ----------------------------------------------
    # None means the band is UNMEASURED. It must never be read as "use the raw
    # number" -- that is the region all three losing bets came from.
    if calibrated is None:
        return _no("calibration unavailable (unmeasured band) — refuse")
    try:
        p = float(calibrated)
    except (TypeError, ValueError):
        return _no("calibration unparseable — refuse")
    if p != p:                                    # NaN
        return _no("calibration is NaN — refuse")
    floor = confidence_floor()
    if p < floor:
        return _no(f"calibrated {p:.1%} below the measured floor {floor:.0%}")

    # --- the market ----------------------------------------------------------
    if not ranges:
        return _no("no market snapshot — refuse")
    priced = []
    for m in ranges:
        bid, ask = m.get("yes_bid"), m.get("yes_ask")
        if bid is None or ask is None or ask < bid:   # crossed book is nonsense
            continue
        if m.get("p_model") is None:
            continue
        priced.append((m, float(ask)))
    if not priced:
        return _no("no bucket has a usable two-sided book — refuse")

    pick, ask = max(priced, key=lambda t: t[0]["p_model"])
    # Raw disagreement, no fee -- this is what `gating.MAX_EDGE` measures and
    # what gets logged on every decision from here on, place or refuse.
    edge = p - ask

    # --- THE CEILING ---------------------------------------------------------
    if ask > PRICE_CEILING + 1e-9:
        return _no(f"ask {ask:.2f} above the {PRICE_CEILING:.2f} ceiling — at "
                   f"{ask:.2f} this needs a hit rate the market has never "
                   f"measured; a high win rate up here is still negative EV",
                   bucket=pick["label"], edge=edge)

    # --- THE MAXIMUM EDGE: is this disagreement even plausible? --------------
    # `gating.MAX_EDGE` is read live, never copied here -- see that constant's
    # own comment for the full derivation and why it is still PROVISIONAL.
    # A calibrated number this far from a liquid market is usually OUR error,
    # not free money (HANDOFF-2026-08-03.md §0, §4.2), so this refuses rather
    # than sizing up into it. The reason carries the investigation checklist
    # so whoever resolves the alert knows what to check first.
    from highcast import gating
    if edge > gating.MAX_EDGE + 1e-9:
        return _no(
            f"edge_implausible: calibrated {p:.1%} vs ask {ask:.2f} is a "
            f"{edge:+.1%} disagreement with a liquid market — above "
            f"MAX_EDGE {gating.MAX_EDGE:.0%}. A gap this large is usually OUR "
            f"error, not free money. Before trusting this number, check: "
            f"is our latest model run current, or did a fetch fail; did the "
            f"source data update or get revised since we ran; have we "
            f"ingested the obs the market has; is the market pricing a "
            f"forecast product we do not consume; is this bucket's price "
            f"stale rather than wrong (check size and recent volume, not "
            f"just last price).",
            bucket=pick["label"], edge=edge)

    # --- THE EDGE: is this trade actually worth taking? ----------------------
    # Fee-aware, because Kalshi charges roundup(0.07*C*P*(1-P)) ON TOP of the
    # price and it peaks at mid-ladder — exactly where this bot buys.
    fee = taker_fee(ask)
    need = ask + fee + MIN_EDGE
    if p < need:
        return _no(f"no edge: calibrated {p:.1%} vs ask {ask:.2f} + fee "
                   f"{fee:.2f} + {MIN_EDGE:.0%} margin = {need:.1%} required — "
                   f"buying here is negative EV even when the model is right",
                   bucket=pick["label"], edge=edge)

    # --- size ----------------------------------------------------------------
    # A limit ABOVE the ask is a market order in disguise, so cap at the ask.
    limit = min(ask, PRICE_CEILING)
    contracts = int(budget // limit) if limit > 0 else 0
    if contracts < 1:
        return _no(f"budget {budget:.2f} buys no contract at {limit:.2f} — a "
                   f"zero-contract order is not a position",
                   bucket=pick["label"], edge=edge)

    return Intent(True,
                  f"PLACE {pick['label']} — calibrated {p:.1%}, limit "
                  f"{limit:.2f}, {contracts} contract(s)",
                  bucket=pick["label"], limit_price=limit, contracts=contracts,
                  edge=edge)
