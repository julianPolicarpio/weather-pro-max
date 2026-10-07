"""Decision-support staking policy for the Kalshi high/low markets.

This module is DECISION SUPPORT ONLY. It turns the model's probabilities and the
live market prices into a disciplined "place / don't place, and if place, this
much" recommendation. It never places trades and it is not personalized
financial advice — the human sets the bankroll, reads the reasons, and decides.

The whole point is to remove the emotional human factor (chasing, revenge-betting,
oversizing a hunch) by making the decision a deterministic function of:

  1. Is the model CONFIDENT enough?     (top 2F bucket prob >= min_confidence)
  2. Is the pick still COMMITTED and     (stability + valid trading window —
     inside the tradeable window?         supplied by the caller as gate flags)
  3. Is there real EDGE vs the market?   (model prob - market price >= min_edge)
  4. If all yes, size with FRACTIONAL     (quarter-Kelly by default, hard-capped
     KELLY + a hard cap.                   at max_stake_frac of bankroll)

If any gate fails the answer is NO_BET with a plain-language reason. Sizing is
metric-agnostic: it operates on a pmf / market ranges, so the identical policy
serves both the daily-high and daily-low markets.
"""
import dataclasses
from dataclasses import dataclass

from highcast.config import StakingConfig


def full_kelly(p: float, price: float) -> float:
    """Full-Kelly stake fraction for a binary contract: buy YES at `price`
    (dollars in (0,1)), settles to $1 if the event happens (prob `p`) else $0.

    f* = (p - price) / (1 - price)   [ = edge / (1 - price) ]

    Positive only when p > price (you have an edge). Returns 0.0 for a
    degenerate/undefined price (<=0 or >=1) rather than dividing by zero. The
    result can exceed 1.0 for a huge edge on a cheap contract; callers apply the
    fractional multiplier and the hard cap, so this stays the pure textbook
    quantity."""
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return (p - price) / (1.0 - price)


@dataclass(frozen=True)
class Sizing:
    stake: float          # dollars to place (0 if bankroll unset)
    stake_frac: float     # fraction of bankroll
    kelly_full: float     # the raw full-Kelly fraction (pre-shrink/cap)
    contracts: int        # approx whole contracts = floor(stake / price)
    capped: bool          # True if the hard cap bound the size
    bankroll_unset: bool  # True if bankroll <= 0 (stake forced to 0)


def size(p: float, price: float, cfg: StakingConfig) -> Sizing:
    """Fractional-Kelly stake (dollars) for one bucket, hard-capped. Assumes the
    edge gate has already passed; if there is no positive Kelly it returns a
    zero Sizing. Pure calculator — this is the "how much to place given the
    confidence and the price" piece."""
    kelly = full_kelly(p, price)
    if kelly <= 0.0:
        return Sizing(0.0, 0.0, kelly, 0, False, cfg.bankroll <= 0.0)
    raw = cfg.kelly_fraction * kelly
    frac = min(raw, cfg.max_stake_frac)
    capped = raw > cfg.max_stake_frac
    bankroll_unset = cfg.bankroll <= 0.0
    stake = 0.0 if bankroll_unset else frac * cfg.bankroll
    contracts = int(stake // price) if (price > 0.0 and stake > 0.0) else 0
    return Sizing(round(stake, 2), round(frac, 6), round(kelly, 6),
                  contracts, capped, bankroll_unset)


@dataclass(frozen=True)
class Decision:
    place: bool
    reason: str
    bucket: str | None       # the range label being recommended, e.g. "77-78"
    # ⚠️ p_model IS THE MODEL'S RAW NUMBER AND MUST STAY THAT WAY.
    # `signals.record_signal` writes it to a column that already holds several
    # thousand historical rows, and `scripts/band_calib_eval.py` measures
    # calibration by comparing stated probabilities against outcomes. If this
    # field became the CALIBRATED value, new rows would be corrected and old
    # ones raw in the same column, and the next calibration run would silently
    # correct an already-corrected number. The calibrated figure lives in
    # `p_hat`, and `edge` is computed from THAT.
    p_model: float | None    # model probability for that bucket, UNCORRECTED
    price: float | None      # market YES price for that bucket
    edge: float | None       # p_hat - price  (calibrated, not raw)
    sizing: Sizing | None    # None when place is False
    spread: float | None = None  # bid/ask spread of the bucket (None if unknown)
    # The band-calibrated probability the decision was actually made on. Equal
    # to p_model when no calibration was supplied.
    p_hat: float | None = None
    # TRUE only for a REGIME veto — the model was running outside what it was
    # trained and validated on, so the number itself is untrustworthy. This is
    # categorically different from "no edge" or "too illiquid", which are sound
    # numbers that simply are not worth trading. Consumers that want the model's
    # honest opinion (the app's own bet ledger) may ignore place=False, but must
    # never ignore this.
    veto: bool = False


def shown_edge(p_raw, p_hat) -> str:
    """"model 73%" or "model 73%->68% calibrated" -- the same phrasing the
    no-edge message uses, so the two refusals read consistently."""
    return (f"model {p_raw:.0%}" if p_hat == p_raw
            else f"model {p_raw:.0%}->{p_hat:.0%} calibrated")


def _no(reason: str) -> Decision:
    return Decision(False, reason, None, None, None, None, None)


def blocked(reason: str, of: Decision | None = None) -> Decision:
    """A veto applied AFTER evaluate(), for conditions evaluate() cannot see.

    evaluate() only receives probabilities and prices; it has no way to know
    whether the model that produced those probabilities was operating inside the
    regime it was trained on. Callers that do know (see
    `signals.live_decision`) use this to force a no-trade while keeping the
    stated reason specific enough to act on.

    Pass `of` to CARRY THE OBSERVED NUMBERS THROUGH the veto (bucket, p_model,
    price, edge, spread) while still forcing place=False. Without it the vetoed
    row logs NULL for all of them, and `signals` is the only record of what the
    model thought minute-by-minute -- that trail is what made the 17-hour inert
    forecast in HANDOFF-2026-07-26.md item 0b reconstructable at all. A veto
    should suppress the ACTION, not erase the evidence."""
    if of is None:
        return dataclasses.replace(_no(reason), veto=True)
    return dataclasses.replace(of, place=False, reason=reason, sizing=None,
                               veto=True)


def _bucket_bounds(label: str | None) -> tuple[float | None, float | None]:
    """(lo, hi) of a Kalshi bucket label, or (None, None) when unbounded or
    unparseable. `≥81` and `≤75` return None on their open side deliberately —
    an unbounded bucket can never be proven impossible by a ratchet."""
    import re
    t = (label or "").strip().replace("\u2265", ">=").replace("\u2264", "<=")
    m = re.match(r"^(\d+)\D+(\d+)$", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _ratchet_kills(label, extremum: str, ratchet) -> str | None:
    """Is this bucket already impossible? Returns the reason, or None.

    ⚠️ DIRECTION. A settlement LOW is the MINIMUM over the climate day, so once
    a value is observed the minimum can only fall FURTHER: the dead buckets are
    the ones ABOVE the running min. A HIGH mirrors it — dead buckets sit BELOW
    the running max. The first version of this function had it backwards and
    would have refused exactly the live buckets.

    Uses the INTERVAL, not a point. `visible_running_extremum` returns (lo, hi)
    because a T-group is precise to 0.1F and a 6-hour group can sit outside
    every spot reading; refuse only when no value in the bucket survives.
    """
    if not ratchet:
        return None
    lo, hi = _bucket_bounds(label)
    if lo is None or hi is None:
        return None
    r_lo, r_hi = float(ratchet[0]), float(ratchet[1])
    if extremum == "min":
        # The low can only go DOWN. Dead if the whole bucket sits above the
        # highest value the running minimum could still be.
        if lo > r_hi:
            return (f"already impossible: the low is at most {r_hi:.1f}F "
                    f"(observed), so {label} can never settle — a minimum only "
                    f"falls further")
    else:
        # The high can only go UP.
        if hi < r_lo:
            return (f"already impossible: the high is at least {r_lo:.1f}F "
                    f"(observed), so {label} can never settle — a maximum only "
                    f"rises further")
    return None


def evaluate(ranges: list[dict], cfg: StakingConfig, *,
             within_window: bool = True,
             calib=None,
             floor: float | None = None,
             extremum: str = "max",
             ratchet: tuple[float, float] | None = None) -> Decision:
    """The ruleset (EDGE-based). `ranges` is a list of dicts (one per tradeable
    market bucket) each with: label, p_model (model probability mass, 0..1), and
    market_price (market YES price 0..1, or None). Shape matches
    `web.app._kalshi_market`.

    The bet criterion is EDGE, not confidence. For a *calibrated* model the +EV
    bet is where the model's probability exceeds the market's by enough — at ANY
    confidence level — and fractional-Kelly sizing automatically shrinks the
    stake when the edge (hence the certainty) is small. High-confidence gating
    was removed on purpose: confidence only arrives once the outcome has formed,
    by which point the market has already priced it, so requiring it structurally
    guarantees no edge. Confidence/stability are still reported for context, they
    just don't block.

    Gate order (fail-closed):
      within_window     -> still inside the tradeable window
      prices available  -> can't assess edge without a market price
      edge              -> the model's top bucket's (p_model - price) >= min_edge
      liquidity         -> bid/ask spread <= max_spread (slippage would eat edge)
    We bet the model's most-backed bucket (highest p_model) when IT is
    underpriced — betting the app's genuine read, not fishing a mispriced tail."""
    if not within_window:
        return _no("past the tradeable window — too late to place")
    priced = [r for r in ranges if r.get("p_model") is not None
              and r.get("market_price") is not None]
    if not priced:
        return _no("no live market prices — cannot assess edge, do not place")
    # the model's most-backed bucket (its genuine read)
    top = max(priced, key=lambda r: r["p_model"])
    # ── RATCHET GATE ────────────────────────────────────────────────────────
    # Before any probability reasoning: has the day already ruled this bucket
    # out? Arithmetic on `latentmax.visible_running_extremum`, not an opinion.
    # Runs before the edge check so a dead bucket says it is dead rather than
    # "no edge" -- impossibility is the more useful message.
    dead = _ratchet_kills(top.get("label"), extremum, ratchet)
    if dead:
        return Decision(False, dead, top["label"], top["p_model"],
                        top["market_price"], None, None)
    # ── CALIBRATE, AFTER the pick is chosen ─────────────────────────────────
    # Deliberately applied to the SELECTED bucket only, never to every bucket
    # before the argmax: band offsets differ per band, so correcting all of them
    # could REORDER them and hand back a different bucket than the model's
    # actual read. The pick is the model's; only the number attached to it is
    # corrected.
    p_raw = top["p_model"]
    p_hat = p_raw
    if calib is not None:
        p_hat = calib(p_raw)
        if p_hat is None:
            # Unmeasured band. Refusing is the whole point -- the three bets
            # this project lost on 2026-07-29/30 were all stated 81%, which is
            # above every band any market has enough records to measure.
            return Decision(False,
                            f"unmeasured confidence: {top['label']} model "
                            f"{p_raw:.0%} sits above the highest band this "
                            f"market has been calibrated on — refusing to bet "
                            f"a number that has never been checked",
                            top["label"], p_raw, top["market_price"],
                            round(p_raw - top["market_price"], 4), None,
                            p_hat=None)
    edge = p_hat - top["market_price"]
    if edge < cfg.min_edge:
        shown = (f"model {p_raw:.0%}" if p_hat == p_raw else
                 f"model {p_raw:.0%}->{p_hat:.0%} calibrated")
        return Decision(False,
                        f"no edge: {top['label']} {shown} vs "
                        f"market {top['market_price']:.0%} = {edge:+.0%} "
                        f"< {cfg.min_edge:.0%} required — do not place",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), None, p_hat=p_hat)
    # ── OPTIONAL CONFIDENCE GATE (cfg.confidence_gate, OFF by default) ──────
    # Runs AFTER the edge check so a no-edge bet still says "no edge", which is
    # the more actionable message.
    #
    # `floor` is the market's own measured `gating.CONFIDENCE_FLOOR` entry,
    # supplied by the caller. FAIL-OPEN on None, which is the OPPOSITE of
    # `gating.confidence_floor_ok`: that one is for the unattended bot, where
    # "I could not tell" must mean "no". This is decision support for a human
    # who can read the number and disagree, and three of four markets have
    # never qualified for a floor — failing closed would silently stop the desk
    # recommending anything on them.
    #
    # ⚠️ WHY THIS DEFAULTS OFF. On the 14 settled PLACE bets, gating at 0.80
    # deletes the group that made money (hit 33%, ask 0.31, +0.009/contract)
    # and keeps the one that lost it (hit 60%, ask 0.60, -0.016/contract).
    # Confidence and price arrive together — see this function's own docstring
    # on why high-confidence gating was removed in the first place.
    if cfg.confidence_gate and floor is not None and p_hat < floor:
        return Decision(False,
                        f"below the measured confidence floor: {top['label']} "
                        f"{p_hat:.0%} < {floor:.0%} required for this market — "
                        f"do not place",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), None, p_hat=p_hat)
    # ── MAX EDGE ────────────────────────────────────────────────────────────
    # AFTER the min-edge check, so a no-edge bet still says "no edge" -- the
    # commoner and more actionable message, and a bet cannot be both too small
    # and too large.
    #
    # Mirrors `gating.MAX_EDGE`, which has guarded the order bot since
    # 2026-08-02. Without it the desk recommends bets the bot refuses.
    max_edge = getattr(cfg, "max_edge", 0.0) or 0.0
    if max_edge > 0 and edge > max_edge:
        return Decision(False,
                        f"edge too large to be real: {top['label']} "
                        f"{shown_edge(p_raw, p_hat)} vs market "
                        f"{top['market_price']:.0%} = {edge:+.0%}, above the "
                        f"{max_edge:.0%} ceiling — an edge this size is "
                        f"usually our own error, not free money "
                        f"(corr(edge, won) = -0.571 over 48 settled bets)",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), None, p_hat=p_hat)
    # Liquidity gate: even with edge, a wide bid/ask means you'd cross a big
    # spread to get filled and the slippage eats the edge. Only checked when the
    # book gives both sides; last-price-only markets skip it (can't assess).
    bid, ask = top.get("yes_bid"), top.get("yes_ask")
    spread = round(ask - bid, 4) if (bid is not None and ask is not None) else None
    if spread is not None and spread > cfg.max_spread:
        return Decision(False,
                        f"market too thin: {top['label']} bid/ask spread "
                        f"{spread*100:.0f}¢ > {cfg.max_spread*100:.0f}¢ max "
                        f"— slippage would eat the edge, do not place",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), None, spread, p_hat=p_hat)
    # CROSSING COST vs EDGE. `max_spread` above is ABSOLUTE, and 10¢ means
    # something completely different at 90¢ than at 7¢: of 11,656 tradeable
    # quotes that PASS the absolute rule, 12.4% still had a spread above 35% of
    # mid (measured 2026-08-01).
    #
    # Half the spread is what you actually pay to cross, so it is compared
    # DIRECTLY against the edge rather than against an invented threshold. If
    # getting filled costs more than the whole reason for the trade, there is no
    # trade. Self-normalising: no constant to defend, and it scales with price.
    #
    # ⚠️ HONEST SCOPE — this is a small rail, not a headline fix.
    #   * Marginal effect: 31 extra refusals (0.4%) among quotes that clear
    #     `min_edge`, beyond what the absolute cap already catches. Most thin
    #     books carry small edges that `min_edge` refuses first.
    #   * It does NOT catch the case that prompted the review. KMIA ≥81 quoted
    #     0.03/0.11 (8¢ spread, 114% of mid) but claimed a +49% edge, which
    #     trivially covers a 4¢ crossing cost — so this gate passes it, CORRECTLY
    #     by its own logic. What was wrong there was the EDGE (empirically 13%,
    #     matching the market), and that is `pick.MAX_PLAUSIBLE_EDGE`'s job.
    #     Do not mistake this rail for protection against a spurious edge.
    if spread is not None and (spread / 2.0) > edge:
        return Decision(False,
                        f"crossing costs more than the edge: {top['label']} "
                        f"half-spread {spread/2*100:.1f}¢ > edge {edge*100:.1f}¢ "
                        f"(book {bid:.2f}/{ask:.2f}) — do not place",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), None, spread, p_hat=p_hat)
    sz = size(p_hat, top["market_price"], cfg)
    if sz.stake <= 0.0 and sz.bankroll_unset:
        return Decision(True,
                        f"PLACE {top['label']} — edge {edge:+.0%}, but bankroll is "
                        f"unset (config staking.bankroll=0): set it to get an amount",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), sz, spread, p_hat=p_hat)
    # A "PLACE" that buys ZERO whole contracts is not a position, and recording
    # it as one poisons the scorecard: 3 of the app's first 6 "placed" bets
    # bought 0 contracts (prices 47-58c against a $0.40 cap) and were still
    # graded W/L, so the record described trades that never existed. Refuse
    # explicitly, and say WHY -- this is a bankroll limit, not a model opinion.
    if sz.contracts < 1:
        need = top["market_price"] / cfg.max_stake_frac if cfg.max_stake_frac else 0
        return Decision(False,
                        f"unbuyable: {top['label']} at "
                        f"{top['market_price']*100:.0f}¢ needs "
                        f"{cfg.currency} {top['market_price']:.2f} but the "
                        f"{cfg.max_stake_frac:.0%} cap allows only "
                        f"{cfg.currency} {sz.stake:.2f} — bankroll of about "
                        f"{cfg.currency} {need:.0f} would buy one contract",
                        top["label"], p_raw, top["market_price"],
                        round(edge, 4), sz, spread, p_hat=p_hat)
    cap_note = " (hard-capped)" if sz.capped else ""
    shown = (f"{p_raw:.0%}" if p_hat == p_raw
             else f"{p_raw:.0%}->{p_hat:.0%} calibrated")
    return Decision(True,
                    f"PLACE {top['label']}: {shown} vs market "
                    f"{top['market_price']:.0%} (edge {edge:+.0%}) -> "
                    f"{cfg.currency} {sz.stake:.2f} = {sz.stake_frac:.1%} of "
                    f"bankroll{cap_note}, {sz.contracts} contracts",
                    top["label"], p_raw, top["market_price"],
                    round(edge, 4), sz, spread, p_hat=p_hat)
