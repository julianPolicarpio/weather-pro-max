"""THE PAPER DEMO ACCOUNT — one coherent view of the paper-trading record.

WHY THIS EXISTS. Today the record is spread across four things that do not
reconcile: `app_bets` (the app's own committed bets), `signals` (every
5-minute decision snapshot), `bot_orders` (the order bot's own decisions), and
`scoring.py` / the dashboard. This module does not touch any of them. It ADDS
a read-only replay: one account, seeded from the real Kalshi balance, with a
transaction ledger, open positions, and an equity curve, built entirely by
REPLAYING `app_bets` — the only evidence this project has.

`app_bets` MUST STAY BYTE-IDENTICAL. Every function here reads it; nothing
here writes to it, migrates it in place, or deletes a row. See
`highcast.migrate.ensure_paper_account_tables` — the only schema change this
module needs is a brand-new table, `paper_account_txns`, that this module
owns exclusively.

TWO LENSES, DELIBERATELY KEPT SEPARATE (do not merge their numbers):

1. `unit_replay` — THE RECONCILIATION LEDGER. Fixed at ONE CONTRACT per bet,
   regardless of price or bankroll. This is what proves the replay is honest:
   it must reproduce the archive's hand-computed record exactly (14W-6L over
   20 graded bets; PLACE 3/8 won, net +$0.015/contract; SKIP 11/12 won, net
   +$1.990/contract — see `tests/test_paper_account.py`). Buying "1 contract
   no matter the price" is unrealistic on its own (a $0.005 bucket would be
   200 contracts of real money) but that unrealism is exactly why it is the
   RIGHT tool for measuring the model's raw hit rate: it holds the unit fixed
   so no single lucky/unlucky size dominates the curve. It seeds from the same
   opening balance as the account below, but its balance curve is a hit-rate
   instrument, not a claim about what would really have been bought.

2. `daily_plan_sim` — JULIAN'S ACTUAL OPERATING PLAN: ONE MARKET A DAY, sized
   as a fraction of the LIVE (compounding) balance, until the balance grows
   enough to widen out. The market traded is not hardcoded — it is whichever
   (station, metric) currently measures most accurate at its own decision hour
   in `gating.HOUR_ACCURACY` (see `select_daily_market`), so a future re-
   measurement moves the pick automatically instead of leaving a stale name in
   code. Selecting on ACCURACY rather than edge is deliberate: this project's
   own edge signal correlates -0.619 with winning, so ranking candidates by
   edge would select on the very quantity that is broken.

   Accuracy alone is not sufficient — the most accurate market is also the one
   the market agrees with, and it often prices 94-97c where a win pays 3c. The
   existing `orderbot.PRICE_CEILING` (0.84) and each station's own
   `staking.min_edge` still gate every trade; this module does not relax them.

   THE STAKE COMPOUNDS: `staking.size` is called with `bankroll` set to the
   RUNNING balance at that point in the replay, not the fixed opening figure,
   so the dollar stake grows as paper winnings accrue — `max_stake_frac` and
   `kelly_fraction` stay exactly as configured; nothing here adds a new knob.

   THE ABOVE-CAP OVERRIDE. At $5.50 a single contract at 50c is already ~9% of
   the balance — above the 5% cap — and the honest alternative to accepting
   that is never trading at all while the balance is this small. So when the
   capped size rounds down to zero contracts but ONE contract is still
   affordable in absolute dollars, this buys exactly one and labels the row
   "ABOVE the cap, accepted at stage N" with the real percentage. It never
   rounds up silently and it never raises `max_stake_frac` to make a trade
   fit — the cap keeps its configured value; only the per-trade acceptance is
   overridden, and that override is what the ledger row records. A bet that
   is not even affordable at 100% of the balance still gets a plain `NO_TRADE`.

   THE STAGE LADDER (`stage_ladder`) falls out of the cap arithmetic rather
   than being invented: stage N's upper edge is the balance at which a
   contract at that stage's reference price first fits `max_stake_frac`
   exactly (`price / max_stake_frac`). It exists so "wait for the balance to
   grow" has a visible, falsifiable target instead of being a vibe.

3. `commit_hour_variants` — MEASUREMENT ONLY, not a third account. Julian's
   plan trades ONE bet a day at the shipped `gating.MIN_DECISION_HOUR`; this
   lens asks whether a DIFFERENT hour for the SAME market would have done
   better, without moving the constant or placing a second daily bet.

   WHY IT EXISTS. `MIN_DECISION_HOUR["KMIA"]["high"]` (14) was chosen because
   accuracy jumps there — but accuracy was never compared against what the
   market actually CHARGES at that hour versus an earlier one. A hand-run
   comparison suggested hour 13 might net more (higher edge, much cheaper
   entry) despite lower accuracy, but that comparison mixed a parity-averaged
   walk-forward hit rate against a real five-day mean ask — NOT the same
   object, and parity-averaging has already inverted one conclusion in this
   project (see `cities.py`). This lens settles it on MATCHED data instead:
   for each settled day, the price and model probability the desk actually
   saw at each candidate hour, read from `signals` (already captured every 5
   minutes by the wake cycle — NO new live fetch, NO change to what gets
   captured). Grades each hour's picks the same way `appbet.settle` does,
   reusing `appbet._final_settled_actual` rather than re-deriving the
   finality check (this project has paid for a duplicated grading rule
   before — see `appbet._final_settled_actual`'s own docstring).

   The default hour (`gating.MIN_DECISION_HOUR`) is ALWAYS included and
   flagged `is_default=True` — it is the only variant meant to be read as the
   live recommendation. Any other hour is opt-in (`extra_hours=`, a CLI flag
   or a config value — never a code edit) and exists purely to be compared.
   Nothing here changes `gating.MIN_DECISION_HOUR`; moving it on the strength
   of this measurement is a separate, deliberate decision for later.

Nothing here ever calls `place_order` or touches the arming literal. Reading
the Kalshi balance needs credentials only (`kalshi_client`'s read/trade
split); this module always constructs its client with `armed=False`.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3
from typing import Any

from highcast import db, fees, kalshi_client, staking
from highcast.config import AppConfig

# The starting figure used ONLY when no Kalshi credentials are configured
# (a bare test machine, or a dev box with no ~/.highcast/credentials). This
# must never silently become the number reported for a real run — every path
# that returns it also returns a `note` saying so, and callers must show that
# note rather than swallow it. Deliberately 0.0, matching `StakingConfig
# .bankroll`'s own "never assume the user's money" default.
FALLBACK_STARTING_BALANCE_USD = 0.0


# ─────────────────────────── opening balance ────────────────────────────────

def seed_opening_balance(env: dict[str, str] | None = None, *,
                         credentials_path=None,
                         fallback_usd: float = FALLBACK_STARTING_BALANCE_USD,
                         transport: Any = None) -> dict:
    """(balance_usd, source, note) — the account's real opening balance.

    Reads ONLY. Builds `KalshiClient(..., armed=False, ...)`: balance is a
    read (`kalshi_client`'s read/trade split needs credentials, not the
    arming literal), and this function must never be the thing that flips
    that switch. Falls back to `fallback_usd` — never raises — when
    credentials are absent or the read fails, so a machine with no trading
    setup still gets a usable (clearly-labelled) account instead of an error.
    """
    env = dict(os.environ) if env is None else env
    kid, pem = kalshi_client.load_credentials(env, path=credentials_path)
    if not (kid and pem):
        return {
            "balance_usd": fallback_usd, "source": "fallback",
            "note": (f"no Kalshi credentials configured — using fallback "
                     f"starting balance ${fallback_usd:.2f}"),
        }
    client = kalshi_client.KalshiClient(kid, pem, armed=False, transport=transport)
    out = client.balance()
    bal = out.get("balance_usd")
    if bal is None:
        return {
            "balance_usd": fallback_usd, "source": "fallback",
            "note": (f"Kalshi balance read failed ({out.get('reason')}) — "
                     f"using fallback starting balance ${fallback_usd:.2f}"),
        }
    return {
        "balance_usd": bal, "source": "kalshi_balance",
        "note": f"seeded from the live Kalshi balance: ${bal:.2f}",
    }


# ─────────────────────────── shared helpers ──────────────────────────────────

def _is_place(reason: str | None) -> bool:
    """`app_bets.reason` starts with 'PLACE' when the staking rule said place;
    everything else (including NULL, on the earliest rows written before the
    `reason` column existed) counts as SKIP. Matches the reconciliation gate's
    own definition exactly — do not diverge from this in either replay."""
    return bool(reason) and reason.startswith("PLACE")


def _all_bets(conn: sqlite3.Connection) -> list[dict]:
    """Every row in `app_bets`, oldest first. READ ONLY — nothing here writes
    to this table, ever."""
    return [dict(r) for r in conn.execute(
        "SELECT * FROM app_bets ORDER BY placed_utc, station, metric")]


def _txn_row(mode: str, r: dict, txn_type: str, ts: str, price: float | None,
            contracts: int, fee: float, cash_delta: float, balance_after: float,
            rule: str, won: int | None, reason: str | None) -> dict:
    return {
        "mode": mode, "station": r["station"], "metric": r["metric"],
        "target_date": r["target_date"], "txn_type": txn_type, "txn_utc": ts,
        "bucket": r["bucket"], "price": price, "contracts": int(contracts),
        "fee": round(fee, 4), "cash_delta": round(cash_delta, 4),
        "balance_after": round(balance_after, 4), "rule": rule,
        "won": won, "reason": reason,
    }


def _drawdown(curve: list[dict]) -> tuple[float, float | None]:
    """(max_drawdown_usd, max_drawdown_pct_of_peak) over a balance curve —
    the worst peak-to-trough decline seen at any point, not just start-to-
    end. `pct` is None when the peak is $0 (nothing to express a fraction
    of)."""
    peak = curve[0]["balance"] if curve else 0.0
    worst_usd = 0.0
    worst_pct = None
    for pt in curve:
        bal = pt["balance"]
        peak = max(peak, bal)
        dd = peak - bal
        if dd > worst_usd:
            worst_usd = dd
            worst_pct = (dd / peak) if peak > 0 else None
    return round(worst_usd, 4), worst_pct


# ─────────────────────── 1. the reconciliation ledger ────────────────────────

def unit_replay(conn: sqlite3.Connection, opening_balance: float) -> dict:
    """Replay every `app_bets` row chronologically at ONE CONTRACT per bet.

    Two ledger rows per graded bet (OPEN at `placed_utc`, SETTLE at
    `settled_utc`), interleaved in true time order across every bet — not
    grouped bet-by-bet — so the equity curve reflects the actual sequence of
    cash movements. An ungraded bet (`won IS NULL`) contributes only its OPEN
    row and shows up in `open_positions`, never counted as a win or a loss.

    THE FEE. Charged once, on entry, via `highcast.fees.taker_fee(1, price)`
    — the real Kalshi taker fee for a 1-contract order. There is no
    settlement fee (see `fees` module docstring), so a win's SETTLE row is a
    clean +$1.00 and a loss's is $0.00; the entry fee is the only drag.

    Returns the RECONCILIATION breakdown (`place` / `skip`, each with `n`,
    `wins`, `net_per_contract`) that `tests/test_paper_account.py` pins
    against the archive's hand-computed numbers.
    """
    rows = _all_bets(conn)
    events: list[tuple[str, str, dict]] = []
    for r in rows:
        events.append((r["placed_utc"], "open", r))
        if r["won"] is not None and r["settled_utc"]:
            events.append((r["settled_utc"], "settle", r))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "open" else 1))

    balance = opening_balance
    curve = [{"utc": None, "balance": round(balance, 4)}]
    txns: list[dict] = []
    place_n = place_wins = 0
    place_net = 0.0
    skip_n = skip_wins = 0
    skip_net = 0.0

    for ts, kind, r in events:
        price = r["price"]
        rule = "PLACE" if _is_place(r["reason"]) else "SKIP"
        if kind == "open":
            fee = fees.taker_fee(1, price)
            delta = -(price + fee)
            balance += delta
            txns.append(_txn_row("unit_replay", r, "OPEN", ts, price, 1, fee,
                                 delta, balance, rule, None,
                                 "opened 1 contract"))
        else:
            won = int(bool(r["won"]))
            delta = 1.0 if won else 0.0
            balance += delta
            fee = fees.taker_fee(1, price)
            net = (1.0 - price - fee) if won else (-price - fee)
            txns.append(_txn_row("unit_replay", r, "SETTLE", ts, price, 1, 0.0,
                                 delta, balance, rule, won,
                                 "won — settled at $1.00" if won else
                                 "lost — settled at $0.00"))
            if rule == "PLACE":
                place_n += 1
                place_wins += won
                place_net += net
            else:
                skip_n += 1
                skip_wins += won
                skip_net += net
        curve.append({"utc": ts, "balance": round(balance, 4)})

    open_positions = [
        {"station": r["station"], "metric": r["metric"],
         "target_date": r["target_date"], "bucket": r["bucket"],
         "price": r["price"], "placed_utc": r["placed_utc"],
         "rule": "PLACE" if _is_place(r["reason"]) else "SKIP"}
        for r in rows if r["won"] is None
    ]

    n_graded = place_n + skip_n
    wins = place_wins + skip_wins
    dd_usd, dd_pct = _drawdown(curve)
    return {
        "opening_balance": round(opening_balance, 4),
        "closing_balance": round(balance, 4),
        "txns": txns,
        "curve": curve,
        "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_pct,
        "n_graded": n_graded, "wins": wins, "losses": n_graded - wins,
        "win_rate": (wins / n_graded) if n_graded else None,
        "open_positions": open_positions,
        "reconciliation": {
            "place": {"n": place_n, "wins": place_wins,
                      "net_per_contract": round(place_net, 4)},
            "skip": {"n": skip_n, "wins": skip_wins,
                     "net_per_contract": round(skip_net, 4)},
        },
    }


# ─────────────────── 2. what the real account could buy ──────────────────────

def select_daily_market() -> dict | None:
    """The ONE (station, metric) Julian's accumulation plan trades — the most
    measured-accurate market AT ITS OWN DECISION HOUR, per `gating
    .HOUR_ACCURACY`. Computed fresh every call, never hardcoded, so a future
    re-measurement (a new `hour_curve.py` run) moves the pick with it.

    Ranked by modal-bucket HIT RATE (descending), MAE ascending as the
    tie-break — accuracy, not edge: this project's own edge signal correlates
    -0.619 with winning (see CLAUDE.md), so sorting candidates by edge would
    hand the daily slot to the broken quantity. Only (station, metric) pairs
    with BOTH a `gating.MIN_DECISION_HOUR` entry and a matching
    `HOUR_ACCURACY` row at that hour are eligible — the same "no fallback
    across stations" contract `gating` itself enforces. None when nothing
    qualifies (an empty accuracy table, e.g. in a from-scratch test db)."""
    from highcast import gating
    candidates = []
    for station, metrics in gating.MIN_DECISION_HOUR.items():
        for metric, hour in metrics.items():
            row = gating.HOUR_ACCURACY.get(station, {}).get(metric, {}).get(hour)
            if row is None:
                continue
            # ELIGIBILITY BEFORE RANKING. A market with no measured
            # `CONFIDENCE_FLOOR` at its commit hour is one the bot may never
            # fire on — the floor's own comment says so ("a market with no
            # such threshold is ABSENT here, and an absent market never
            # fires") — so it is not a market this plan should simulate
            # trading either.
            #
            # ⚠️ THIS IS LOAD-BEARING, AND IT WAS ADDED THE DAY IT BIT.
            # Moving KMIA HIGH 14 -> 13 on 2026-08-05 dropped its raw hit rate
            # to 67.6%, below KLAX HIGH's 69.5%, and a pure hit-rate ranking
            # silently handed the daily slot to KLAX HIGH — the one market
            # `gating` excludes on two independent grounds (no threshold
            # reaches a 0.80 lower bound; its favoured bucket trades 0.75-0.96
            # against a model that can justify ~69c). Ranking on accuracy is
            # right; ranking WITHOUT this filter is how the accuracy criterion
            # picked the market the project had already refused.
            if (station, metric, hour) not in gating.CONFIDENCE_FLOOR:
                continue
            mae, hit = row
            candidates.append({"station": station, "metric": metric,
                               "hour": hour, "mae": mae, "hit": hit})
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c["hit"], c["mae"]))
    return candidates[0]


# Reference prices used only to LABEL the stage ladder's boundaries — the
# boundaries themselves are computed (`price / max_stake_frac`), not these
# numbers. 1.00 stands in for "$1 notional", the round figure Julian's plan
# uses once the cap covers a full dollar.
_STAGE_REFERENCE_PRICES = (0.40, 0.60, None, 1.00)  # None -> orderbot.PRICE_CEILING


def stage_ladder(max_stake_frac: float, price_ceiling: float) -> list[dict]:
    """The accumulation ladder: each stage's balance range and what it buys.

    Stage N's upper edge is the balance at which one contract at that stage's
    reference price first fits `max_stake_frac` exactly
    (`reference_price / max_stake_frac`) — e.g. at the shipped 5% cap, 40c
    first fits at $8, the 0.84 price ceiling first fits at $16.80, and a full
    $1 notional first fits at $20. These fall out of the cap arithmetic; nothing here changes what actually gets bought, only how
    the balance is narrated.
    """
    notes = [
        "1 contract fits the {frac:.0%} cap up to 40c",
        "1 contract fits the {frac:.0%} cap up to 60c",
        "1 contract fits the {frac:.0%} cap even at the {ceiling:.2f} price ceiling",
        "$1 notional (~2 contracts) fits the {frac:.0%} cap; four markets at "
        "once would deploy about {four:.0%} of the balance",
    ]
    stages = []
    lo = 0.0
    for i, (ref, note) in enumerate(zip(_STAGE_REFERENCE_PRICES, notes), start=1):
        price = price_ceiling if ref is None else ref
        hi = price / max_stake_frac if max_stake_frac > 0 else None
        stages.append({
            "stage": i, "lo": round(lo, 2), "hi": (round(hi, 2) if hi else None),
            "note": note.format(frac=max_stake_frac, ceiling=price_ceiling,
                                four=max_stake_frac * 4),
        })
        lo = hi if hi is not None else lo
    stages.append({
        "stage": len(notes) + 1, "lo": round(lo, 2), "hi": None,
        "note": "full ladder open — the cap no longer binds at the price ceiling",
    })
    return stages


def stage_for_balance(balance: float, ladder: list[dict]) -> dict:
    """Which stage `balance` currently sits in, and (via `hi`) the balance
    that unlocks the next one."""
    for s in ladder:
        if s["hi"] is None or balance < s["hi"]:
            return s
    return ladder[-1]


def _daily_plan_decision(r: dict, cfg: AppConfig, balance: float,
                         price_ceiling: float, ladder: list[dict]) -> dict:
    """One bet's accept/refuse verdict under the accumulation plan. Reuses
    `staking.full_kelly` / `staking.size` for the actual math (never
    reimplements Kelly), but — unlike `staking.evaluate` — does not refuse
    outright when the capped size rounds to zero contracts: it checks whether
    ONE contract is affordable in absolute dollars and, if so, takes it with
    an explicit "ABOVE the cap" reason instead of a silent round-up. Returns
    a dict with at least {trade: bool, contracts: int, reason: str}."""
    price, p_model = r["price"], r["p_model"]
    scfg = cfg.staking
    if price is None or p_model is None or not (0.0 < price < 1.0):
        return {"trade": False, "contracts": 0, "reason": "no usable price"}
    if price > price_ceiling + 1e-9:
        return {"trade": False, "contracts": 0,
                "reason": (f"price {price:.2f} above the {price_ceiling:.2f} "
                           f"ceiling — the market agrees, no edge worth taking")}
    edge = p_model - price
    if edge < scfg.min_edge:
        return {"trade": False, "contracts": 0,
                "reason": (f"no edge: model {p_model:.0%} vs market {price:.0%} "
                           f"= {edge:+.0%} < {scfg.min_edge:.0%} required")}
    sized = staking.size(p_model, price, dataclasses.replace(scfg, bankroll=balance))
    if sized.contracts >= 1:
        return {"trade": True, "contracts": sized.contracts, "override": False,
                "reason": (f"PLACE {sized.contracts} contract(s) @{price:.2f} = "
                           f"{sized.stake_frac:.1%} of bankroll"
                           + (" (capped)" if sized.capped else ""))}
    if price > balance:
        return {"trade": False, "contracts": 0,
                "reason": (f"insufficient bankroll: 1 contract @{price:.2f} "
                           f"costs more than the ${balance:.2f} balance")}
    fee = fees.taker_fee(1, price)
    cost = price + fee
    frac = (cost / balance) if balance > 0 else None
    stage = stage_for_balance(balance, ladder)
    return {"trade": True, "contracts": 1, "override": True,
            "reason": (f"1 contract @{price:.2f} = ${cost:.2f} = "
                       f"{(frac or 0) * 100:.1f}% of bankroll — ABOVE the "
                       f"{scfg.max_stake_frac:.0%} cap, accepted at stage "
                       f"{stage['stage']}")}


def daily_plan_sim(conn: sqlite3.Connection, cfgs_by_station: dict[str, AppConfig],
                   opening_balance: float) -> dict:
    """Replay the ONE selected market's `app_bets` history under Julian's
    accumulation plan: compounding stake, price ceiling, min-edge, and the
    above-cap override — see the module docstring for the full rationale.

    Only bets for `select_daily_market()`'s (station, metric) are replayed;
    every other historical bet belongs to a market this plan does not trade
    yet and is out of scope for this lens (see `unit_replay` for the full
    archive's own record)."""
    from highcast import orderbot
    market = select_daily_market()
    if market is None:
        return {"market": None,
                "note": "no measured market available in gating.HOUR_ACCURACY "
                        "— cannot select a daily market"}
    station, metric = market["station"], market["metric"]
    cfg = cfgs_by_station.get(station)
    if cfg is None:
        return {"market": market,
                "note": f"no config loaded for {station} — cannot simulate"}
    price_ceiling = orderbot.PRICE_CEILING
    ladder = stage_ladder(cfg.staking.max_stake_frac, price_ceiling)

    rows = [r for r in _all_bets(conn)
            if r["station"] == station and r["metric"] == metric]

    balance = opening_balance
    curve = [{"utc": None, "balance": round(balance, 4)}]
    txns: list[dict] = []
    detail: list[dict] = []
    n_trades = n_overrides = n_no_trade = 0

    for r in rows:
        rule = "PLACE" if _is_place(r["reason"]) else "SKIP"
        stage = stage_for_balance(balance, ladder)
        decision = _daily_plan_decision(r, cfg, balance, price_ceiling, ladder)
        price = r["price"]
        if decision["trade"]:
            n_trades += 1
            n_overrides += int(decision.get("override", False))
            contracts = decision["contracts"]
            fee = fees.taker_fee(contracts, price)
            cost = contracts * price + fee
            losses_to_zero = round(balance / cost, 1) if cost > 0 else None
            balance_before = balance
            balance -= cost
            txns.append(_txn_row("daily_plan", r, "OPEN", r["placed_utc"], price,
                                 contracts, fee, -cost, balance, rule, None,
                                 decision["reason"]))
            curve.append({"utc": r["placed_utc"], "balance": round(balance, 4)})
            won = None
            if r["won"] is not None and r["settled_utc"]:
                won = int(bool(r["won"]))
                delta = float(contracts) if won else 0.0
                balance += delta
                txns.append(_txn_row("daily_plan", r, "SETTLE", r["settled_utc"],
                                     price, contracts, 0.0, delta, balance, rule,
                                     won, "won" if won else "lost"))
                curve.append({"utc": r["settled_utc"], "balance": round(balance, 4)})
            detail.append({
                "station": r["station"], "metric": r["metric"],
                "target_date": r["target_date"], "bucket": r["bucket"],
                "price": price, "rule": rule, "stage": stage["stage"],
                "balance_before": round(balance_before, 4),
                "losses_to_zero": losses_to_zero, "won": won, **decision,
            })
        else:
            n_no_trade += 1
            txns.append(_txn_row("daily_plan", r, "NO_TRADE", r["placed_utc"],
                                 price, 0, 0.0, 0.0, balance, rule, None,
                                 decision["reason"]))
            curve.append({"utc": r["placed_utc"], "balance": round(balance, 4)})
            detail.append({
                "station": r["station"], "metric": r["metric"],
                "target_date": r["target_date"], "bucket": r["bucket"],
                "price": price, "rule": rule, "stage": stage["stage"],
                "balance_before": round(balance, 4), "losses_to_zero": None,
                "won": None, **decision,
            })

    open_positions = [d for d in detail if d["trade"] and d["won"] is None]
    graded_traded = [t for t in txns if t["txn_type"] == "SETTLE"]
    wins = sum(t["won"] for t in graded_traded)
    dd_usd, dd_pct = _drawdown(curve)
    return {
        "market": market, "ladder": ladder,
        "current_stage": stage_for_balance(balance, ladder),
        "price_ceiling": price_ceiling,
        "opening_balance": round(opening_balance, 4),
        "closing_balance": round(balance, 4),
        "txns": txns, "curve": curve, "detail": detail,
        "max_drawdown_usd": dd_usd, "max_drawdown_pct": dd_pct,
        "open_positions": open_positions,
        "n_bets_seen": len(rows), "n_trades": n_trades,
        "n_overrides": n_overrides, "n_no_trade": n_no_trade,
        "n_graded_traded": len(graded_traded), "wins_traded": wins,
        "losses_traded": len(graded_traded) - wins,
    }


# ────────────── 2b. aggressive sizing arms (measurement only) ────────────────
#
# WHAT THIS SETTLES. Julian asked for the stake cap to come off when the model
# is very confident — "remove the cap if its super confident bet 4 dollar or 5".
# This runs that rule against the alternative it kept losing to in argument,
# on the same days, and lets the ledger decide.
#
#   A confidence_only     — the request, taken literally. Confidence clears the
#                           floor ⇒ commit AGGRESSIVE_FRAC of the balance.
#   B confidence_x_return — the same instinct with the PAYOFF in the
#                           arithmetic: refuse a bucket whose win pays less
#                           than MIN_PAYOFF_RATIO per dollar risked, and scale
#                           the stake by how much it actually pays.
#   C control             — the shipped rule (`_daily_plan_decision`), verbatim.
#
# WHY B IS NOT A STRAW MAN. On the 28 settled `app_bets` (2026-08-05) the
# market price correlates +0.596 with winning, the model's own edge signal
# -0.506. The days the model is most confident are the days the market already
# agrees, and those trade at 84-99c: arm A stakes ~85% of the account to win a
# few cents. That is the specific failure B is built to avoid.
#
# NONE OF THIS IS A CONFIG KNOB. `max_stake_frac` and `kelly_fraction` keep
# their shipped values everywhere; the fractions below exist only inside this
# simulation and are never read by `staking.size` on the live path.

AGGRESSIVE_FRAC = 0.85       # arm A: share of the running balance on a confident day
MIN_PAYOFF_RATIO = 0.35      # arm B: (1-price)/price floor — 74c is the dearest it buys
ARM_CONFIDENCE = 0.80        # both aggressive arms: "super confident" as a number
RUIN_BALANCE = 0.01          # below this there is no next bet


def _arm_decision(arm: str, r: dict, balance: float, cfg: AppConfig, *,
                  confidence: float | None = None,
                  aggressive_frac: float | None = None,
                  min_payoff: float | None = None) -> dict:
    """One arm's verdict on one bet. Pure — no DB, no config mutation.

    `r` needs only `price` and `p_model`, so the aggressive arms can be probed
    directly in tests without building a whole archive row.

    The three keyword overrides are the test bench's knobs. They are PARAMETERS
    rather than reassigned module globals on purpose: uvicorn can serve two
    requests concurrently, and a global swapped inside a try/finally would let
    one browser tab's experiment leak into another's results for as long as the
    two overlap.
    """
    conf_gate = ARM_CONFIDENCE if confidence is None else float(confidence)
    frac_full = AGGRESSIVE_FRAC if aggressive_frac is None else float(aggressive_frac)
    payoff_floor = MIN_PAYOFF_RATIO if min_payoff is None else float(min_payoff)
    price, p_model = r.get("price"), r.get("p_model")
    if price is None or p_model is None or not (0.0 < price < 1.0):
        return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                "reason": "no usable price"}
    if balance < RUIN_BALANCE:
        return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                "reason": "ruined — no balance left to bet"}
    if p_model < conf_gate:
        return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                "reason": (f"not confident enough: {p_model:.0%} < "
                           f"{conf_gate:.0%}")}
    payoff = (1.0 - price) / price
    if arm == "confidence_x_return" and payoff < payoff_floor:
        return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                "reason": (f"payoff too thin: {price:.2f} pays "
                           f"{payoff:.0%} per dollar risked, under the "
                           f"{payoff_floor:.0%} floor — confident but not "
                           f"worth the balance")}
    frac = frac_full
    if arm == "confidence_x_return":
        # Scale INTO the payoff rather than betting a flat fraction: a 45c
        # bucket (122% payoff) gets the full aggressive fraction, a 70c one
        # (43%) gets about a third of it. Capped at AGGRESSIVE_FRAC so B is
        # never MORE aggressive than A — the comparison has to be one-sided.
        frac = min(frac_full, frac_full * payoff)
    budget = balance * frac
    contracts = int(budget // price)
    if contracts < 1:
        if price + fees.taker_fee(1, price) > balance:
            return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                    "reason": (f"insufficient balance: 1 contract @{price:.2f} "
                               f"costs more than ${balance:.2f}")}
        contracts = 1
    cost = contracts * price + fees.taker_fee(contracts, price)
    if cost > balance:
        contracts = max(0, int((balance - fees.taker_fee(1, price)) // price))
        if contracts < 1:
            return {"trade": False, "contracts": 0, "stake_frac": 0.0,
                    "reason": f"insufficient balance at {price:.2f}"}
        cost = contracts * price + fees.taker_fee(contracts, price)
    return {"trade": True, "contracts": contracts,
            "stake_frac": (cost / balance) if balance > 0 else 0.0,
            "reason": (f"{contracts} contract(s) @{price:.2f} = ${cost:.2f} = "
                       f"{cost / balance:.0%} of balance "
                       f"(confidence {p_model:.0%}, pays {payoff:.0%})")}


def _arm_rows_for_test(conn: sqlite3.Connection,
                       cfgs_by_station: dict[str, AppConfig]) -> list[dict]:
    """The bets `sizing_arms` scores, exposed for tests that need to know the
    fixture is non-empty without duplicating the selection rule."""
    market = select_daily_market()
    if market is None:
        return []
    return [r for r in _all_bets(conn)
            if r["station"] == market["station"] and r["metric"] == market["metric"]]


def sizing_arms(conn: sqlite3.Connection,
                cfgs_by_station: dict[str, AppConfig],
                opening_balance: float, *,
                confidence: float | None = None,
                aggressive_frac: float | None = None,
                min_payoff: float | None = None) -> dict:
    """Replay the selected daily market under all three sizing arms.

    Every arm sees EVERY bet — an arm that stands down records a no-trade row
    rather than dropping the day, so no arm can win by trading less. Ruin is
    absorbing: once an arm's balance falls below `RUIN_BALANCE` it records the
    date and never trades again, because a 0.5c balance is not a position size,
    it is a dead account.

    MEASUREMENT ONLY — writes nothing, persists nothing, and is never consulted
    by the order bot or by `staking`.
    """
    from highcast import orderbot
    market = select_daily_market()
    if market is None:
        return {"market": None, "arms": [],
                "note": "no measured market available — cannot select a market"}
    station, metric = market["station"], market["metric"]
    cfg = cfgs_by_station.get(station)
    if cfg is None:
        return {"market": market, "arms": [],
                "note": f"no config loaded for {station} — cannot simulate"}
    rows = _arm_rows_for_test(conn, cfgs_by_station)
    ladder = stage_ladder(cfg.staking.max_stake_frac, orderbot.PRICE_CEILING)

    arms = []
    for arm in ("confidence_only", "confidence_x_return", "control"):
        balance = opening_balance
        out_rows, curve = [], [{"utc": None, "balance": round(balance, 4)}]
        contracts_total = 0
        ruin_date = None
        max_frac = 0.0
        for r in rows:
            if arm == "control":
                d = _daily_plan_decision(r, cfg, balance, orderbot.PRICE_CEILING,
                                         ladder)
                d.setdefault("stake_frac", 0.0)
                if d["trade"] and balance > 0:
                    c = d["contracts"] * r["price"] + fees.taker_fee(
                        d["contracts"], r["price"])
                    d["stake_frac"] = c / balance
            else:
                d = _arm_decision(arm, r, balance, cfg,
                                  confidence=confidence,
                                  aggressive_frac=aggressive_frac,
                                  min_payoff=min_payoff)
            row = {"target_date": r["target_date"], "bucket": r["bucket"],
                   "price": r["price"], "p_model": r["p_model"],
                   "won": None, "balance_before": round(balance, 4), **d}
            if d["trade"] and ruin_date is None:
                n = d["contracts"]
                cost = n * r["price"] + fees.taker_fee(n, r["price"])
                balance -= cost
                contracts_total += n
                max_frac = max(max_frac, row["stake_frac"])
                if r["won"] is not None and r["settled_utc"]:
                    row["won"] = int(bool(r["won"]))
                    balance += float(n) if row["won"] else 0.0
                curve.append({"utc": r["placed_utc"], "balance": round(balance, 4)})
                if balance < RUIN_BALANCE and ruin_date is None:
                    ruin_date = r["target_date"]
                    row["reason"] += " — RUINED: balance exhausted"
            row["balance_after"] = round(balance, 4)
            out_rows.append(row)
        dd_usd, dd_pct = _drawdown(curve)
        traded = [x for x in out_rows if x["trade"]]
        graded = [x for x in traded if x["won"] is not None]
        net = round(balance - opening_balance, 4)
        # An OPEN position has already left the cash balance but has not paid
        # out yet, so comparing arms on cash alone punishes whichever arm
        # happens to hold a live bet on the day this is run. `equity_at_cost`
        # is the fair cross-arm figure; `closing_balance` stays the cash truth.
        openers = [x for x in traded if x["won"] is None]
        open_cost = sum(x["balance_before"] - x["balance_after"] for x in openers)
        arms.append({
            "arm": arm,
            "opening_balance": round(opening_balance, 4),
            "closing_balance": round(balance, 4),
            "equity_at_cost": round(balance + open_cost, 4),
            "open_contracts": sum(x["contracts"] for x in openers),
            "open_cost": round(open_cost, 4),
            "net": net,
            "net_settled": round(net + open_cost, 4),
            "net_per_contract": (round(net / contracts_total, 4)
                                 if contracts_total else None),
            "contracts": contracts_total,
            "n_trades": len(traded), "n_no_trade": len(out_rows) - len(traded),
            "n_graded": len(graded),
            "wins": sum(x["won"] for x in graded),
            "max_stake_frac_seen": round(max_frac, 4),
            "max_drawdown_usd": dd_usd, "max_drawdown_pct": dd_pct,
            "ruin_date": ruin_date,
            "curve": curve, "rows": out_rows,
        })
    return {"market": market, "arms": arms,
            "confidence_threshold": (ARM_CONFIDENCE if confidence is None
                                     else float(confidence)),
            "aggressive_frac": (AGGRESSIVE_FRAC if aggressive_frac is None
                                else float(aggressive_frac)),
            "min_payoff_ratio": (MIN_PAYOFF_RATIO if min_payoff is None
                                 else float(min_payoff))}


# ─────────────── 3. commit-hour variants (measurement only) ──────────────────

def _local_hour(cfg: AppConfig, iso_utc: str) -> int | None:
    """Station-local wall-clock hour of an ISO-8601 UTC timestamp string, or
    None if it doesn't parse. A small local copy rather than importing
    `gating._local_hour` — this one has no safety nuance worth sharing, unlike
    the finality check below."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    try:
        ts = datetime.fromisoformat(iso_utc)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(ZoneInfo(cfg.station.tz)).hour


def _signal_at_each_hour(conn: sqlite3.Connection, cfg: AppConfig, station: str,
                         metric: str) -> dict[tuple[str, int], dict]:
    """{(target_date, station_local_hour): first `signals` row seen at that
    hour}. `signals` is written every 5 minutes by the wake cycle regardless
    of whether a real bet was placed (see `signals.record_live`), so this
    needs no new capture — it is a pure re-read of evidence that already
    exists. "First" mirrors `appbet.place`'s own convention (the earliest
    wake at or after the hour ticks over), so a variant's recorded price is
    what a bet placed right at that hour would actually have seen."""
    rows = conn.execute(
        """SELECT target_date, created_utc, bucket, p_model, price
             FROM signals WHERE station=? AND metric=?
               AND bucket IS NOT NULL AND price IS NOT NULL
             ORDER BY created_utc ASC""", (station, metric)).fetchall()
    first: dict[tuple[str, int], dict] = {}
    for r in rows:
        h = _local_hour(cfg, r["created_utc"])
        if h is None:
            continue
        key = (r["target_date"], h)
        if key not in first:
            first[key] = dict(r)
    return first


def commit_hour_variants(conn: sqlite3.Connection,
                         cfgs_by_station: dict[str, AppConfig],
                         market: dict | None,
                         extra_hours: tuple[int, ...] = ()) -> dict:
    """Side-by-side comparison of the shipped commit hour against opt-in
    alternate hours for `market` (from `select_daily_market`) — see the
    module docstring's point 3. Every variant is graded at 1 contract with the
    real taker fee, same as `unit_replay`, since this is a measurement
    instrument, not a claim about position size.

    Returns `{"variants": [...]}`; each variant has `hour`, `is_default`, `n`
    (signals rows found at that hour), `n_graded`, `wins`, `win_rate`,
    `mean_entry_price`, `net_per_contract_total` (summed net over graded
    bets), and `mean_net_per_bet` (the fair cross-variant comparison — a
    variant that trades on fewer settled days should not win on a bigger
    total alone).
    """
    from highcast import appbet
    if market is None:
        return {"variants": [], "note": "no daily market selected"}
    station, metric, default_hour = market["station"], market["metric"], market["hour"]
    cfg = cfgs_by_station.get(station)
    if cfg is None:
        return {"variants": [], "note": f"no config loaded for {station}"}
    hours = sorted({int(default_hour), *(int(h) for h in extra_hours)})
    by_hour = _signal_at_each_hour(conn, cfg, station, metric)

    variants = []
    for h in hours:
        picks = sorted((td, row) for (td, hh), row in by_hour.items() if hh == h)
        n_graded = wins = 0
        net_total = 0.0
        prices = []
        bets = []
        for td, row in picks:
            price = row["price"]
            if price is None or not (0.0 < price < 1.0):
                continue
            prices.append(price)
            actual = appbet._final_settled_actual(conn, cfg, metric, td)
            if actual is None:
                bets.append({"target_date": td, "price": price,
                            "bucket": row["bucket"], "won": None})
                continue
            lo, hi = appbet._bucket_bounds(row["bucket"])
            won = (int(lo <= actual <= hi) if lo is not None else None)
            if won is None:
                continue
            fee = fees.taker_fee(1, price)
            net = (1.0 - price - fee) if won else (-price - fee)
            n_graded += 1
            wins += won
            net_total += net
            bets.append({"target_date": td, "price": price,
                        "bucket": row["bucket"], "won": won,
                        "net": round(net, 4)})
        variants.append({
            "hour": h, "is_default": (h == default_hour),
            "n": len(picks), "n_graded": n_graded, "wins": wins,
            "win_rate": (wins / n_graded) if n_graded else None,
            "mean_entry_price": (sum(prices) / len(prices)) if prices else None,
            "net_per_contract_total": round(net_total, 4),
            "mean_net_per_bet": (round(net_total / n_graded, 4)
                                if n_graded else None),
            "bets": bets,
        })
    return {"station": station, "metric": metric, "default_hour": default_hour,
            "variants": variants}


# ─────────────────────────── persistence ─────────────────────────────────────

def _persist(conn: sqlite3.Connection, mode: str, txns: list[dict]) -> None:
    """Idempotent rebuild of this mode's rows in `paper_account_txns`. Fully
    derived from `app_bets` every time — deleting and re-inserting is safe
    because nothing outside this module writes to this table, and because
    `app_bets` itself is never touched."""
    conn.execute("DELETE FROM paper_account_txns WHERE mode = ?", (mode,))
    if txns:
        cols = ["mode", "station", "metric", "target_date", "txn_type",
                "txn_utc", "bucket", "price", "contracts", "fee",
                "cash_delta", "balance_after", "rule", "won", "reason"]
        conn.executemany(
            f"INSERT INTO paper_account_txns ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [[t[c] for c in cols] for t in txns])
    conn.commit()


# ─────────────────────────── orchestrator ─────────────────────────────────────

def refresh(conn: sqlite3.Connection, cfgs_by_station: dict[str, AppConfig], *,
           env: dict[str, str] | None = None,
           credentials_path=None,
           fallback_usd: float = FALLBACK_STARTING_BALANCE_USD,
           transport: Any = None,
           variant_hours: tuple[int, ...] = (),
           persist: bool = True) -> dict:
    """Seed the opening balance, run the reconciliation replay and the daily
    plan sim, persist their ledger rows to `paper_account_txns`, and return
    everything a CLI or a future web view needs. The only writes this
    performs are to `paper_account_txns` (via `_persist`) and to `meta`
    (recording the seed) — `app_bets` and `signals` are read-only throughout.

    `variant_hours` are OPT-IN alternate commit hours (e.g. `(13,)`) compared
    against the selected market's shipped decision hour via
    `commit_hour_variants` — measurement only, never persisted as ledger
    rows and never a second daily position (see the module docstring).

    `persist=False` computes and returns exactly the same payload while
    writing NOTHING — no ledger rows, no `meta` seed. It exists for the web
    view (`GET /api/demo-account`), which is hit by every page load and by a
    poll timer: a reader that also rebuilds the table would race the CLI's own
    rebuild of the same rows for no gain, since the view renders the returned
    dict and never reads `paper_account_txns` back.
    """
    seed = seed_opening_balance(env, credentials_path=credentials_path,
                                fallback_usd=fallback_usd, transport=transport)
    bal = seed["balance_usd"]
    unit = unit_replay(conn, bal)
    sim = daily_plan_sim(conn, cfgs_by_station, bal)
    variants = commit_hour_variants(conn, cfgs_by_station, sim.get("market"),
                                    extra_hours=variant_hours)
    arms = sizing_arms(conn, cfgs_by_station, bal)
    if persist:
        _persist(conn, "unit_replay", unit["txns"])
        _persist(conn, "daily_plan", sim.get("txns", []))
        db.set_meta(conn, "paper_account_opening_balance_usd", repr(bal))
        db.set_meta(conn, "paper_account_opening_balance_source", seed["source"])
    return {"seed": seed, "unit_replay": unit, "daily_plan": sim,
            "sizing_arms": arms, "commit_hour_variants": variants}


# ─────────────────────────── the test bench ──────────────────────────────────

def bench(conn: sqlite3.Connection, cfgs_by_station: dict[str, AppConfig], *,
          commit_hour: float | None = None,
          confidence: float | None = None,
          aggressive_frac: float | None = None,
          min_payoff: float | None = None,
          min_edge: float | None = None,
          price_ceiling: float | None = None,
          max_stake_frac: float | None = None,
          balance: float | None = None,
          env: dict[str, str] | None = None,
          credentials_path=None,
          transport: Any = None) -> dict:
    """`refresh`, re-run with the knobs changed. WRITES NOTHING, EVER.

    Every override travels as a PARAMETER — `dataclasses.replace` for the
    staking config, keyword arguments for the sizing arms. Nothing is assigned
    to a module global, so two concurrent bench requests cannot see each
    other's knobs and nothing can be left behind for the live desk.
    `tests/test_web_testbench.py` asserts exactly that.

    THE ONE THING THIS CANNOT DO is change what the model itself predicted at
    a different hour. `commit_hour` re-reads the price and probability the desk
    ACTUALLY SAW at that hour from `signals` (via `commit_hour_variants`) —
    real captured evidence, not a re-run of the model. So the bench answers
    "what would this rule have done on the days we have?", never "what would a
    model trained at that hour have said?". The second question needs a
    walk-forward, not a web request.
    """
    if balance is not None:
        seed = {"balance_usd": float(balance), "source": "bench",
                "note": f"bench override — opening balance ${balance:.2f}"}
    else:
        seed = seed_opening_balance(env, credentials_path=credentials_path,
                                    transport=transport)
    bal = seed["balance_usd"]

    cfgs = dict(cfgs_by_station)
    if min_edge is not None or max_stake_frac is not None:
        for icao, c in cfgs.items():
            s = c.staking
            cfgs[icao] = dataclasses.replace(c, staking=dataclasses.replace(
                s,
                min_edge=s.min_edge if min_edge is None else float(min_edge),
                max_stake_frac=(s.max_stake_frac if max_stake_frac is None
                                else float(max_stake_frac))))

    unit = unit_replay(conn, bal)
    sim = daily_plan_sim(conn, cfgs, bal)
    arms = sizing_arms(conn, cfgs, bal, confidence=confidence,
                       aggressive_frac=aggressive_frac, min_payoff=min_payoff)

    market = sim.get("market") or select_daily_market()
    shipped_hour = (market or {}).get("hour")
    extra: tuple[float, ...] = ()
    if commit_hour is not None and market is not None:
        extra = (float(commit_hour),)
    variants = commit_hour_variants(conn, cfgs, market, extra_hours=extra)
    return {"seed": seed, "unit_replay": unit, "daily_plan": sim,
            "sizing_arms": arms, "commit_hour_variants": variants,
            "bench": {"commit_hour": (float(commit_hour)
                                      if commit_hour is not None
                                      else shipped_hour),
                      "shipped_commit_hour": shipped_hour,
                      "is_bench": True}}


# ─────────────────────────── CLI rendering ─────────────────────────────────────

def render(out: dict) -> str:
    """Human-readable summary for `highcast paper-account`. Data layer + CLI
    only — the web view is a separate task."""
    seed, u, s = out["seed"], out["unit_replay"], out["daily_plan"]
    lines = ["PAPER DEMO ACCOUNT", ""]
    lines.append(f"  opening balance   ${seed['balance_usd']:.2f}  "
                 f"({seed['source']})")
    lines.append(f"  {seed['note']}")
    lines.append("")
    lines.append("── RECONCILIATION LEDGER (1 contract/bet — the model's raw "
                 "hit rate) ──")
    graded_line = f"  graded bets       {u['n_graded']}   W-L {u['wins']}-{u['losses']}"
    if u["win_rate"] is not None:
        graded_line += f"   win rate {u['win_rate']*100:.0f}%"
    lines.append(graded_line)
    p, sk = u["reconciliation"]["place"], u["reconciliation"]["skip"]
    lines.append(f"  rule said PLACE   {p['wins']}/{p['n']} won   "
                 f"net {p['net_per_contract']:+.3f}/contract")
    lines.append(f"  rule said SKIP    {sk['wins']}/{sk['n']} won   "
                 f"net {sk['net_per_contract']:+.3f}/contract")
    lines.append(f"  balance           ${u['opening_balance']:.2f} -> "
                 f"${u['closing_balance']:.2f}")
    lines.append(f"  max drawdown      ${u['max_drawdown_usd']:.2f}"
                 + (f" ({u['max_drawdown_pct']*100:.0f}% of peak)"
                    if u['max_drawdown_pct'] is not None else ""))
    lines.append(f"  open positions    {len(u['open_positions'])}")
    for op in u["open_positions"]:
        lines.append(f"     {op['station']} {op['metric']:4} {op['target_date']} "
                     f"{op['bucket']} @ {op['price']:.3f} ({op['rule']})")
    lines.append("")
    lines.append("── ACCUMULATION PLAN (one market/day, compounding stake, "
                 "shipped caps) ──")
    if s.get("market") is None:
        lines.append(f"  {s.get('note', 'no daily market selected')}")
    else:
        m = s["market"]
        lines.append(f"  selected market   {m['station']} {m['metric'].upper()} "
                     f"(decision hour {m['hour']}) — measured MAE {m['mae']:.3f}, "
                     f"hit {m['hit']*100:.1f}% — the most accurate market on file")
        lines.append(f"  price ceiling     {s['price_ceiling']:.2f}   "
                     f"(bets priced above it are refused — the market already "
                     f"agrees)")
        lines.append(f"  bets seen         {s['n_bets_seen']}   "
                     f"traded {s['n_trades']} (of which {s['n_overrides']} "
                     f"above-cap override)   no-trade {s['n_no_trade']}")
        if s["n_graded_traded"]:
            lines.append(f"  of the traded, graded {s['n_graded_traded']}: "
                         f"{s['wins_traded']}W-{s['losses_traded']}L")
        lines.append(f"  balance           ${s['opening_balance']:.2f} -> "
                     f"${s['closing_balance']:.2f}")
        lines.append(f"  max drawdown      ${s['max_drawdown_usd']:.2f}"
                     + (f" ({s['max_drawdown_pct']*100:.0f}% of peak)"
                        if s['max_drawdown_pct'] is not None else ""))
        cur = s["current_stage"]
        hi = f"${cur['hi']:.2f}" if cur["hi"] is not None else None
        lines.append(f"  current stage     {cur['stage']}  (${cur['lo']:.2f}"
                     + (f"-{hi}" if hi else "+") + f")   {cur['note']}")
        if hi:
            lines.append(f"                    next stage unlocks at {hi}")
        lines.append("")
        lines.append("  stage ladder:")
        for st in s["ladder"]:
            rng = f"${st['lo']:.2f}-${st['hi']:.2f}" if st["hi"] is not None \
                else f"${st['lo']:.2f}+"
            mark = " <-- current" if st["stage"] == cur["stage"] else ""
            lines.append(f"     stage {st['stage']}  {rng:<16} {st['note']}{mark}")
        lines.append("")
        lines.append("  per-bet detail (selected market only):")
        for d in s["detail"]:
            tag = ("TRADE" + (" (override)" if d.get("override") else "")
                  if d["trade"] else "no trade")
            extra = (f"  [{d['losses_to_zero']} consecutive losses at this "
                     f"stake would zero the account]"
                     if d.get("losses_to_zero") is not None else "")
            lines.append(f"     {d['target_date']} {d['bucket']} @ "
                         f"{d['price']:.3f} stage {d['stage']}  [{tag}]  "
                         f"{d['reason']}{extra}")
        lines.append("")
        lines.append("NOTE: the reconciliation ledger and the accumulation "
                     "plan are TWO DIFFERENT LENSES on the same archive — the "
                     "first is a fixed-unit accuracy instrument covering every "
                     "historical bet, the second replays ONLY the selected "
                     "daily market with a real, compounding stake. Do not "
                     "average them together.")

    arms = out.get("sizing_arms") or {}
    if arms.get("arms"):
        lines.append("")
        lines.append("── SIZING ARMS (measurement only — the shipped cap is "
                     "unchanged) ──")
        lines.append(f"  {'arm':<22}{'equity':>8}{'net':>8}{'/ctr':>8}"
                     f"{'trades':>7}{'W/G':>7}{'max stake':>10}{'ruin':>12}")
        for a in arms["arms"]:
            npc = a["net_per_contract"]
            lines.append(
                f"  {a['arm']:<22}{a['equity_at_cost']:>8.2f}"
                f"{a['net_settled']:>+8.2f}"
                f"{(npc if npc is not None else 0):>+8.3f}{a['n_trades']:>7}"
                f"{f'{a['wins']}/{a['n_graded']}':>7}"
                f"{a['max_stake_frac_seen']:>9.0%}"
                f"{str(a['ruin_date'] or '—'):>12}")
        lines.append("  equity counts an OPEN bet at cost — cash alone would "
                     "show an arm holding a live position as though it had "
                     "already lost the stake.")

    v = out.get("commit_hour_variants") or {}
    lines.append("")
    lines.append("── COMMIT-HOUR VARIANTS (measurement only — NOT extra daily "
                 "bets) ──")
    if not v.get("variants"):
        lines.append(f"  {v.get('note', 'no variants configured')}")
    else:
        lines.append(f"  {v['station']} {v['metric'].upper()} — comparing "
                     f"{len(v['variants'])} candidate commit hour(s), each "
                     f"read from already-captured `signals` snapshots (no new "
                     f"fetch), graded 1 contract/bet like the reconciliation "
                     f"ledger")
        lines.append(f"  {'hour':>6} {'n':>4} {'graded':>7} {'win%':>7} "
                     f"{'mean price':>11} {'mean net/bet':>13} {'total net':>10}")
        for var in v["variants"]:
            wr = f"{var['win_rate']*100:.0f}%" if var["win_rate"] is not None else "  n/a"
            mp = f"{var['mean_entry_price']:.3f}" if var["mean_entry_price"] is not None else "n/a"
            mn = (f"{var['mean_net_per_bet']:+.3f}"
                 if var["mean_net_per_bet"] is not None else "n/a")
            tag = " (default/live)" if var["is_default"] else " (measurement)"
            lines.append(f"  {var['hour']:>6} {var['n']:>4} {var['n_graded']:>7} "
                         f"{wr:>7} {mp:>11} {mn:>13} "
                         f"{var['net_per_contract_total']:>+10.3f}{tag}")
        lines.append("  headline = mean net/bet (per-contract), NOT total — a "
                     "variant with fewer graded days should not win on total "
                     "alone. Only the default/live hour is the actual "
                     "recommendation; gating.MIN_DECISION_HOUR is unchanged "
                     "regardless of what this table shows.")
    return "\n".join(lines)
