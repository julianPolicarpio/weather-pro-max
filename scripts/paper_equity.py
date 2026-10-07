"""What would the app's account have done — replayed at any starting bankroll.

WHY THIS IS A SCRIPT AND NOT A LIVE `bankroll = winnings` SETTING.

Making the live bankroll track the app's own P&L is circular in a way that
breaks the one thing the ledger is for. Three concrete problems:

  1. THE APP IS CURRENTLY DOWN. Placed bets are 2W-4L, -0.66 units. A
     winnings-driven bankroll would SHRINK, and since $8 x 5% already fails to
     buy anything above 40c, shrinking locks out more of the board -- leaving
     only the cheapest contracts, which are exactly the wide-disagreement ones
     the model is worst at. That is the trap we just escaped, tightened.

  2. IT CONTAMINATES THE MEASUREMENT. If which bets get RECORDED depends on
     past results, the ledger stops being a clean read of the model and becomes
     a read of the model-plus-its-own-history. At n=12 that is fatal: we are
     still trying to establish whether there is an edge at all.

  3. IT IS IRREVERSIBLE. A bet not recorded because the bankroll was too small
     that week can never be recovered. A simulated one can be re-run at any
     size, forever.

So: keep `bankroll: 0` live (the app records its genuine opinion on every bet),
and compute the equity curve HERE, over the recorded ledger. This is strictly
more informative -- it answers "what would $8 have done, or $50, or $200" from
the same data, and it can be re-run after every settlement.

FEES ARE MODELLED. `appbet.ledger` computes flat-stake P&L with no fee, which
overstates every result; at 1 contract Kalshi's fee rounds UP to 2c at every
price, which is a large slice of a 30c position.

⚠️ Bets whose sizing buys ZERO whole contracts are SKIPPED, not silently counted
as free wins -- that is the same bug that put three phantom positions in the
live ledger on 2026-07-29/30.
"""
import sys

from highcast import fees
from highcast.config import load_config
from highcast.db import connect

STARTS = [8.0, 25.0, 50.0, 200.0]
KELLY_FRACTION = 0.25
MAX_STAKE_FRAC = 0.05

# Bets already decided by the ratchet but not yet graded in the table. Keeping
# them out would flatter the curve by dropping known losses.
FORCED = {("KLAX", "2026-07-30", "low"): 0, ("KLAX", "2026-07-30", "high"): 0}


def replay(rows, start, frac=MAX_STAKE_FRAC, kelly=KELLY_FRACTION):
    bank = start
    n_placed = n_skipped = wins = 0
    curve = [(None, bank)]
    for r in rows:
        won = r["won"]
        if won is None:
            won = FORCED.get((r["station"], r["target_date"], r["metric"]))
        if won is None:
            continue
        p, price = r["p_model"], r["price"]
        if not price or not (0 < price < 1) or p is None:
            continue
        # fractional Kelly, hard-capped — the shipped sizing rule
        edge = p - price
        k = edge / (1.0 - price) if edge > 0 else 0.0
        stake = min(kelly * k, frac) * bank
        contracts = int(stake // price)
        if contracts < 1:
            n_skipped += 1
            curve.append((r["target_date"], bank))
            continue
        fee = fees.taker_fee(contracts, price)
        bank += (contracts * (1 - price) - fee) if won else (-contracts * price - fee)
        n_placed += 1
        wins += won
        curve.append((r["target_date"], bank))
    return bank, n_placed, n_skipped, wins, curve


def main():
    cfg = load_config("config/klax.yaml")
    conn = connect(cfg.db_path)
    rows = [dict(r) for r in conn.execute(
        """SELECT station, target_date, metric, bucket, p_model, price, won
           FROM app_bets ORDER BY target_date, station, metric""")]
    print("PAPER EQUITY — the app's own ledger, replayed with fees\n")
    print("  sizing: quarter-Kelly capped at %.0f%% of bank; 0-contract bets SKIPPED\n"
          % (MAX_STAKE_FRAC * 100))
    print("  %10s %9s %9s %9s %9s %9s" %
          ("start", "end", "P&L", "placed", "skipped", "won"))
    for s in STARTS:
        end, np_, ns, w, _ = replay(rows, s)
        print("  %10.2f %9.2f %+9.2f %9d %9d %9d" % (s, end, end - s, np_, ns, w))
    print("\n  the skipped column IS the finding: at a small bank the cap silently")
    print("  removes most of the board, and what survives is the cheap tail.")
    end, np_, ns, w, curve = replay(rows, 25.0)
    print("\n  equity curve at $25 start:")
    for d, b in curve:
        print("     %-12s %7.2f" % (d or "start", b))


if __name__ == "__main__":
    sys.exit(main())
