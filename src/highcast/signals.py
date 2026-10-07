"""Paper-trading signal log: the evidence base for "does this actually make
money?". This project never stored Kalshi prices, so profitability was
unmeasurable. The logger fixes that: each evaluation snapshots the model
probabilities AND the live market prices AND the gate/staking decision. Once a
day settles (its CLI value lands) we score paper P&L.

Everything is metric-agnostic and BANKROLL-INDEPENDENT at its core: the headline
number is per-contract P&L (unit stake), so it measures the STRATEGY's edge
regardless of how the human sizes. If a bankroll is set we also report the
scaled dollars. This is decision support and a track record — not a promise, not
advice, and it never places a trade.
"""
import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from highcast import gating, staking
from highcast import latentmax
from highcast.buckets import (dead_mass, pmf_from_buckets, pmf_mass,
                              truncate_at_bound)
from highcast.config import AppConfig
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds

logger = logging.getLogger(__name__)

# Below this price the market considers a bucket all but dead. If a
# just-settled value lands there anyway, the disagreement is almost always
# our CLI source being wrong, not a live order book -- see
# `check_settlement_against_market`. Chosen from the incident itself: the
# market held KMIA's true bucket (73-74) at 0.99 and the WRONGLY-settled one
# (>=81) at 0.005 -- a two-order-of-magnitude gap, not a close call.
MARKET_DISAGREEMENT_FLOOR = 0.02


def _regime_veto(conn, cfg: AppConfig, spec, target: str, kind: str,
                 pred: dict) -> str | None:
    """Reason to refuse the trade because the model was out of its regime, or
    None if the prediction is inside what has actually been validated.

    Keyed off the prediction's OWN `issued_at` -- what the model saw when it ran
    -- never off "now". Checking "now" is what let the day-ahead bug back
    through on its first fix attempt.
    """
    issued = pred.get("issued_at")
    if kind == "dayahead":
        if not gating.morning_obs_ready(conn, cfg, spec, target, as_of=issued):
            return ("this day-ahead forecast was issued before any morning "
                    "observation existed, so every morn_* feature was NaN — a "
                    "regime the model was never trained on. Not tradeable; wait "
                    "for the morning refresh.")
        # Same class of failure, different feed. The LOW carries seven
        # NBS-derived columns and the `nbm_nbs` fetch went silent on
        # 2026-07-26, after which the day-ahead LOW ran 5-7F cold against
        # unanimous guidance. See `gating.nbs_guidance_ready`.
        #
        # SCOPED TO METRICS THAT ACTUALLY READ NBS. The HIGH consumes none of
        # those columns, so vetoing it on a missing feed would block a sound
        # prediction over data it never looks at.
        if (gating.metric_uses_nbs(spec.key)
                and not gating.nbs_guidance_ready(conn, cfg, target)):
            return ("no NBS guidance stored for this target, so every nbs_* "
                    "feature was NaN — the same never-trained regime as a "
                    "missing morning observation. Not tradeable until the "
                    "nbm_nbs fetch is working again.")
    if kind == "intraday" and not gating.decision_hour_ready(cfg, spec, issued):
        knee = gating.min_decision_hour(cfg.station.icao, spec.key)
        # decision_hour_ready fails closed to False both when it's simply too
        # early AND when the station has no measured knee at all -- an
        # unmeasured station can't be compared against "the decision hour",
        # so it gets its own message instead of formatting a hit-rate
        # comparison against a knee that doesn't exist.
        if knee is None:
            return ("this station's intraday decision hour has not been "
                    "measured yet — refusing to trade until it is.")
        acc = gating.hour_accuracy(cfg.station.icao, spec.key,
                                   gating._local_hour(cfg, issued))
        now_hit = f"{acc['hit']:.0%}" if acc else "unmeasured"
        knee_acc = gating.hour_accuracy(cfg.station.icao, spec.key, knee)
        knee_hit = f"{knee_acc['hit']:.0%}" if knee_acc else "unmeasured"
        return (f"too early to act: the nowcast refits every hour, and at this "
                f"hour its modal bucket has historically been right only "
                f"{now_hit} of the time versus {knee_hit} from "
                f"{knee}:00 {cfg.station.tz.split('/')[-1].replace('_', ' ')} "
                f"onward. The pick is still moving — wait for {knee}:00.")
    return None


def live_decision(conn, cfg: AppConfig, spec, now: datetime) -> dict:
    """Orchestrate one evaluation for (metric, today): latest prediction ->
    gate -> live Kalshi market -> staking decision. Returns everything the CLI
    prints and the logger records. `pred` is None when nothing is stored yet."""
    from highcast.fetchers import kalshi
    target = now.astimezone(ZoneInfo(cfg.station.tz)).date().isoformat()
    gate = gating.gate_state(conn, cfg, target, spec.key, now)
    # WHICH MODEL THE DESK READS (measured 2026-07-25, N=1364 matched days,
    # paired bootstrap). Inside the trading window the DAY-AHEAD model is
    # significantly sharper than the intraday nowcast at the decision hour:
    #   intraday  MAE 1.620  CRPS 1.178
    #   dayahead  MAE 1.509  CRPS 1.080   (dMAE -0.111 [-0.166,-0.054])
    # The nowcast only overtakes later in the day, once the extremum is actually
    # forming (CRPS 0.43 by 2pm) — i.e. AFTER the window has closed and the
    # market has priced the same observations. So: day-ahead while the decision
    # is live, nowcast afterwards when the panel is informational.
    # ...and once the extremum is actually observed the nowcast overtakes it, so
    # the preference is hour-bounded, not window-bounded (see
    # `gating.preferred_prediction`'s 2026-07-26 amendment).
    local_hour = now.astimezone(ZoneInfo(cfg.station.tz)).hour
    pred, kind = gating.preferred_prediction(
        conn, cfg.station.icao, target, spec.key, bool(gate.get("within_window")),
        local_hour=local_hour)
    # Once today's window has closed, today's contract is no longer actionable —
    # the desk used to sit on that dead market for the rest of the day. Roll to
    # tomorrow when its window is open AND a forecast for it exists, so the desk
    # always shows the next decision the human can actually make. (The timing
    # card already pivots this way; this keeps the two consistent.)
    if not gate.get("within_window"):
        nxt = (date.fromisoformat(target) + timedelta(days=1)).isoformat()
        ngate = gating.gate_state(conn, cfg, nxt, spec.key, now)
        if ngate.get("within_window"):
            npred, nkind = gating.preferred_prediction(
                conn, cfg.station.icao, nxt, spec.key, True)
            if npred is not None:
                target, gate, pred, kind = nxt, ngate, npred, nkind
    if pred is None:
        return {"target": target, "kind": None, "pred": None, "gate": None,
                "market": None, "decision": None}
    pmf = pmf_from_buckets(pred["buckets"])
    # ── DROP OUTCOMES THE RATCHET HAS ALREADY RULED OUT ──────────────────────
    # A daily MIN can only fall and a MAX can only rise, so once the running
    # extremum is set, everything past it is dead. The nowcast does not know
    # that -- on 2026-08-01 it still put 1.1% on ">=70" for KLAX LOW with the
    # running min already at 68.10. That is probability spent on an outcome
    # that cannot happen, and every LIVE bucket is diluted by exactly as much.
    #
    # Only applied to the CURRENT climate day: a rolled-forward target has no
    # observations yet, so there is no bound and nothing to truncate.
    # `truncate_at_bound` rounds the bound first, because settlement rounds --
    # a running min of 68.10 can still settle at 68.
    bound = None
    if target == now.astimezone(ZoneInfo(cfg.station.tz)).date().isoformat():
        iv = latentmax.visible_running_extremum(
            conn, cfg, date.fromisoformat(target), spec.extremum,
            asof_utc=now.isoformat())
        if iv is not None:
            bound = iv[1] if spec.extremum == "min" else iv[0]
    dead = dead_mass(pmf, bound, spec.extremum)
    pmf = truncate_at_bound(pmf, bound, spec.extremum)
    ranges = kalshi.fetch_ranges(cfg, target, metric=spec)
    market = None if not ranges else [
        {"label": r["label"], "lo": r["lo"], "hi": r["hi"],
         "p_model": pmf_mass(pmf, r["lo"], r["hi"]),
         "market_price": kalshi.resolve_price(r),
         "yes_bid": r.get("yes_bid"), "yes_ask": r.get("yes_ask")} for r in ranges]
    # Band calibration, bound to THIS market's commit hour. The scalar
    # `hour_calib` is an average over a curve that is not flat, and the staking
    # gate only ever fires from the top of that curve -- the one band that
    # overstates. `band_calib` returns None above the highest measured band,
    # which `evaluate` turns into a refusal rather than a raw-number bet.
    knee = gating.min_decision_hour(cfg.station.icao, spec.key)
    _calib = (None if knee is None else
              (lambda p: gating.band_calib(cfg.station.icao, spec.key, knee, p)))
    decision = staking.evaluate(market or [], cfg.staking,
                                within_window=gate["within_window"],
                                calib=_calib)
    # FAIL CLOSED whenever the model that produced these probabilities was
    # running OUTSIDE the regime it was trained and validated in. `evaluate`
    # sees only probabilities and prices, so it cannot tell a real edge from a
    # broken model -- and a 40-70 point edge against a liquid market is
    # essentially always the latter. Both cases below were caught live on real
    # money (2026-07-26); see the gating functions for the full post-mortems.
    # Checked at EVERY hour, not only when the numbers happen to clear the edge
    # gate. Gating this on `decision.place` meant a pre-knee read was presented
    # as an ordinary "no edge: 68-69 model 65% vs market 74%" line, which reads as
    # a normal informative answer when it is actually a number from an hour whose
    # bucket is right 67% of the time. `of=decision` keeps the observed
    # probability/price on the logged row -- suppress the action, keep the trail.
    veto = _regime_veto(conn, cfg, spec, target, kind, pred)
    if veto:
        decision = staking.blocked(veto, of=decision)
    return {"target": target, "kind": kind, "pred": pred, "gate": gate,
            "market": market, "decision": decision, "dead_mass": dead}


def record_signal(conn, cfg: AppConfig, spec, target: str, kind: str,
                  gate: dict, market, decision, now: datetime) -> int:
    """Insert one signal row (settlement fields left NULL). Returns row id."""
    d = decision
    sz = d.sizing
    cur = conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
             committed, within_window, place, bucket, p_model, price, edge,
             stake, contracts, market_json, reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, target, spec.key, kind, now.isoformat(),
         int(bool(gate["committed"])), int(bool(gate["within_window"])),
         int(bool(d.place)), d.bucket, d.p_model, d.price, d.edge,
         (sz.stake if sz else None), (sz.contracts if sz else None),
         json.dumps(market or []), d.reason))
    conn.commit()
    return cur.lastrowid


def record_live(conn, cfg: AppConfig, spec, now: datetime) -> dict:
    """live_decision + persist it (when there is a prediction). Adds
    `signal_id` to the returned dict (None if nothing was recorded)."""
    out = live_decision(conn, cfg, spec, now)
    out["signal_id"] = None
    if out["pred"] is not None:
        out["signal_id"] = record_signal(
            conn, cfg, spec, out["target"], out["kind"], out["gate"],
            out["market"], out["decision"], now)
    return out


def _bucket_bounds(bucket: str | None, market_json: str):
    """(lo, hi) for a recommended bucket, from the stored market snapshot; falls
    back to parsing a 'lo-hi' / 'lo–hi' label. None,None if undeterminable."""
    if not bucket:
        return None, None
    try:
        for r in json.loads(market_json or "[]"):
            if r.get("label") == bucket:
                return r.get("lo"), r.get("hi")
    except (ValueError, TypeError):
        pass
    for dash in ("–", "-"):
        if dash in bucket:
            a, _, b = bucket.partition(dash)
            try:
                return int(a), int(b)
            except ValueError:
                return None, None
    return None, None


def edge_report(conn, cfg: AppConfig, metric: str, n_boot: int = 2000,
                seed: int = 12345) -> dict:
    """Do the model's probabilities beat the MARKET's, out-of-sample? Over every
    settled shadow snapshot (which stored both the model's per-bucket prob AND
    the market price = implied prob), score each with the multi-category Brier
    (lower=better). Market prices are de-vigged (normalized to sum 1) for a fair
    probability-vs-probability comparison. Edge exists only if the model's Brier
    is lower AND a bootstrap 95% CI on the per-day difference clears zero."""
    import numpy as np
    rows = conn.execute(
        """SELECT actual, market_json FROM signals
           WHERE station=? AND metric=? AND kind='shadow'
             AND settled_utc IS NOT NULL AND actual IS NOT NULL""",
        (cfg.station.icao, metric)).fetchall()
    diffs, mbs, kbs = [], [], []
    for r in rows:
        try:
            market = json.loads(r["market_json"])
        except (TypeError, ValueError):
            continue
        if not market:
            continue
        a = r["actual"]
        mp = [b.get("p_model") for b in market]
        pr = [b.get("market_price") for b in market]
        if any(x is None for x in mp) or any(x is None for x in pr):
            continue
        y = [1.0 if (b.get("lo") is not None and b.get("hi") is not None
                     and b["lo"] <= a <= b["hi"]) else 0.0 for b in market]
        if sum(y) != 1:          # need exactly one winning bucket in the snapshot
            continue
        s = sum(pr)
        if s <= 0:
            continue
        mkt = [p / s for p in pr]                    # de-vig
        mb = sum((mp[i] - y[i]) ** 2 for i in range(len(y)))
        kb = sum((mkt[i] - y[i]) ** 2 for i in range(len(y)))
        mbs.append(mb); kbs.append(kb); diffs.append(mb - kb)
    n = len(diffs)
    if n == 0:
        return {"metric": metric, "n": 0, "model_brier": None,
                "market_brier": None, "diff": None, "ci_lo": None,
                "ci_hi": None, "verdict": "no scored days yet"}
    d = np.asarray(diffs)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if ci_hi < 0:
        verdict = "MODEL BEATS MARKET (edge)"
    elif ci_lo > 0:
        verdict = "market beats model (negative edge)"
    else:
        verdict = "no significant edge (CI straddles 0)"
    return {"metric": metric, "n": n,
            "model_brier": round(float(np.mean(mbs)), 4),
            "market_brier": round(float(np.mean(kbs)), 4),
            "diff": round(float(d.mean()), 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "verdict": verdict}


def record_shadow(conn, cfg: AppConfig, spec, target: str, market, now: datetime):
    """Forward 'bet the pick' simulation: log a paper bet on the model's TOP
    bucket in `market` (1 contract at the live price), IGNORING the edge/window
    gates — so we accumulate a win/P&L track record of the raw forecast. Tracked
    as kind='shadow', separate from the real edge-based ledger; dedups per
    (target, metric); settle_signals scores it. Returns the row id or None."""
    if not market:
        return None
    top = max(market, key=lambda m: (m.get("p_model") if m.get("p_model")
                                      is not None else -1.0))
    price = top.get("market_price")
    if price is None:
        return None
    if conn.execute(
            "SELECT 1 FROM signals WHERE station=? AND target_date=? "
            "AND metric=? AND kind='shadow'",
            (cfg.station.icao, target, spec.key)).fetchone():
        return None
    p = top.get("p_model")
    cur = conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
             committed, within_window, place, bucket, p_model, price, edge,
             stake, contracts, market_json, reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, target, spec.key, "shadow", now.isoformat(),
         0, 0, 1, top["label"], p, price,
         (p - price) if p is not None else None,
         round(price, 4), 1, json.dumps(market),
         "shadow: bet the model's top bucket (ungated)"))
    conn.commit()
    return cur.lastrowid


def record_shadow_live(conn, cfg: AppConfig, spec, now: datetime):
    """Convenience: run one live_decision and log a shadow bet from its market."""
    out = live_decision(conn, cfg, spec, now)
    if out["pred"] is None:
        return None
    return record_shadow(conn, cfg, spec, out["target"], out["market"], now)


def check_settlement_against_market(conn, cfg: AppConfig, metric: str,
                                    target: str, actual: int) -> dict | None:
    """Cross-check a just-settled value against the market's own last read.

    THE IDEA. `market_json` snapshots every bucket's yes_bid/yes_ask each time
    a live evaluation runs. The LAST snapshot taken before the climate day
    closed is, in effect, the market's own verdict: Kalshi traders had every
    observation available by then and priced accordingly. If our settled
    value lands in a bucket the market had all but written off, that
    disagreement is almost always OUR settlement source being wrong, not a
    live order book being wrong -- which is exactly what happened on KMIA
    2026-08-01: settled 82 while the market held 73-74 at 0.99 and >=81 at
    0.005 (true low 74F). See `scripts/migrate_prelim_settlement.py`.

    Cheap and advisory only: this never blocks or alters a settlement, it
    only surfaces a loud reason to distrust one. Returns a dict describing
    the disagreement AND logs it at WARNING (on purpose -- not a silent debug
    line) when the settled value's bucket priced under
    `MARKET_DISAGREEMENT_FLOOR`. Returns None otherwise, including when there
    is simply no signals snapshot for this market/day -- the common case,
    and not itself suspicious.
    """
    try:
        end = climate_day_bounds(date.fromisoformat(target),
                                 cfg.station.std_utc_offset)[1]
    except (TypeError, ValueError):
        return None
    rows = conn.execute(
        """SELECT created_utc, market_json FROM signals
           WHERE station=? AND target_date=? AND metric=?
             AND market_json IS NOT NULL
           ORDER BY created_utc DESC""",
        (cfg.station.icao, target, metric)).fetchall()
    snapshot = None
    for r in rows:
        try:
            created = datetime.fromisoformat(r["created_utc"])
        except (TypeError, ValueError):
            continue
        if created < end:
            snapshot = r
            break
    if snapshot is None:
        return None
    try:
        market = json.loads(snapshot["market_json"])
    except (TypeError, ValueError):
        return None
    bucket = next((b for b in (market or [])
                   if b.get("lo") is not None and b.get("hi") is not None
                   and b["lo"] <= actual <= b["hi"]), None)
    if bucket is None:
        return None
    price = bucket.get("yes_bid")
    if price is None:
        price = bucket.get("market_price")
    if price is None or price >= MARKET_DISAGREEMENT_FLOOR:
        return None
    info = {"station": cfg.station.icao, "metric": metric, "target_date": target,
            "actual": actual, "bucket": bucket.get("label"), "price": price}
    logger.warning(
        "SETTLEMENT DISAGREES WITH MARKET: %s %s %s settled %s (bucket %s), "
        "but the last pre-close market snapshot priced that bucket at %.3f "
        "-- the settlement source is very likely wrong (see "
        "scripts/migrate_prelim_settlement.py for the KMIA 2026-08-01 case "
        "this guards against).",
        cfg.station.icao, metric, target, actual, bucket.get("label"), price)
    return info


def settle_signals(conn, cfg: AppConfig, now: datetime) -> int:
    """Fill actual/won/pnl for every unsettled signal whose target date now has
    a settled CLI value for its metric. Non-placed signals settle to pnl 0
    (they still mark settled so they aren't reprocessed). Returns count settled."""
    rows = conn.execute(
        "SELECT * FROM signals WHERE station=? AND settled_utc IS NULL",
        (cfg.station.icao,)).fetchall()
    n = 0
    for r in rows:
        col = get_metric(cfg, r["metric"]).target_col
        arow = conn.execute(
            f"SELECT {col} AS v FROM cli_reports WHERE station=? AND date=?",
            (cfg.station.icao, r["target_date"])).fetchone()
        if arow is None or arow["v"] is None:
            continue  # not settled yet — leave for a later run
        actual = int(arow["v"])
        if r["place"] and r["bucket"] and r["price"] is not None:
            lo, hi = _bucket_bounds(r["bucket"], r["market_json"])
            won = 1 if (lo is not None and hi is not None and lo <= actual <= hi) else 0
            price = r["price"]
            pnl_unit = (1.0 - price) if won else -price
            contracts = r["contracts"] or 0
            pnl = contracts * (1.0 - price) if won else -contracts * price
        else:
            won, pnl_unit, pnl = None, 0.0, 0.0
        conn.execute(
            """UPDATE signals SET actual=?, won=?, pnl_unit=?, pnl=?, settled_utc=?
               WHERE id=?""",
            (actual, won, round(pnl_unit, 4), round(pnl, 2), now.isoformat(), r["id"]))
        n += 1
    conn.commit()
    return n


def shadow_report(conn, cfg: AppConfig, metric: str) -> dict:
    """The shadow 'bet the pick' track record (kind='shadow'): does the raw
    forecast win money if you always bet its top bucket at the market price?"""
    return paper_report(conn, cfg, metric, kind="shadow")


def manual_report(conn, cfg: AppConfig, metric: str) -> dict:
    """The 'you' ledger (kind='manual'): the track record of the bets the HUMAN
    actually chose to place — your own win rate + P&L, separate from the
    automated shadow and edge ledgers."""
    return paper_report(conn, cfg, metric, kind="manual")


def record_manual(conn, cfg: AppConfig, spec, target: str, bucket: str,
                  contracts: int, price: float, market, now: datetime):
    """Log a bet the HUMAN chose to place (kind='manual', place=1): your bucket,
    size, and price — snapshotted with the market for settlement + edge scoring.
    Settles via settle_signals; reported by manual_report. The bucket must be one
    of the current market's buckets (so settlement can resolve its bounds)."""
    b = next((m for m in (market or []) if m.get("label") == bucket), None)
    if b is None:
        raise ValueError(f"bucket {bucket!r} is not in the current market")
    contracts = int(contracts)
    price = float(price)
    p_model = b.get("p_model")
    cur = conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
             committed, within_window, place, bucket, p_model, price, edge,
             stake, contracts, market_json, reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, target, spec.key, "manual", now.isoformat(),
         0, 0, 1, bucket, p_model, price,
         (p_model - price) if p_model is not None else None,
         round(price * contracts, 2), contracts, json.dumps(market),
         "manual bet (human-placed)"))
    conn.commit()
    return cur.lastrowid


def paper_report(conn, cfg: AppConfig, metric: str, kind: str = "real") -> dict:
    """Aggregate paper P&L over the FIRST placed signal per target date (the
    position you'd actually have taken), for `metric`. Headline is per-contract
    (bankroll-independent) so it measures edge; dollar totals shown too when a
    bankroll was set at record time. kind='real' is the edge-based ledger
    (excludes shadow); kind='shadow' is the ungated 'bet the pick' simulation."""
    if kind == "shadow":
        kind_sql = "AND kind = 'shadow'"
    elif kind == "manual":
        kind_sql = "AND kind = 'manual'"
    else:  # 'real' — the automated edge-based ledger only
        kind_sql = "AND kind NOT IN ('shadow', 'manual')"
    rows = conn.execute(
        f"""SELECT s.* FROM signals s
           JOIN (SELECT target_date, MIN(created_utc) mc FROM signals
                 WHERE station=? AND metric=? AND place=1 {kind_sql}
                 GROUP BY target_date) f
             ON s.target_date=f.target_date AND s.created_utc=f.mc
           WHERE s.station=? AND s.metric=? AND s.place=1
             {kind_sql.replace('kind', 's.kind')}""",
        (cfg.station.icao, metric, cfg.station.icao, metric)).fetchall()
    settled = [r for r in rows if r["settled_utc"] is not None and r["won"] is not None]
    pending = len(rows) - len(settled)
    n = len(settled)
    if n == 0:
        return {"metric": metric, "n_paper_trades": 0, "pending": pending,
                "wins": 0, "win_rate": None, "pnl_per_contract": 0.0,
                "roi_per_contract": None, "avg_edge": None,
                "total_staked": 0.0, "total_pnl": 0.0}
    wins = sum(r["won"] for r in settled)
    sum_unit = sum(r["pnl_unit"] for r in settled)
    sum_price = sum(r["price"] for r in settled)
    sum_stake = sum((r["contracts"] or 0) * r["price"] for r in settled)
    sum_pnl = sum(r["pnl"] or 0.0 for r in settled)
    return {
        "metric": metric, "n_paper_trades": n, "pending": pending, "wins": wins,
        "win_rate": round(wins / n, 4),
        "pnl_per_contract": round(sum_unit, 4),
        "roi_per_contract": round(sum_unit / sum_price, 4) if sum_price else None,
        "avg_edge": round(sum(r["edge"] for r in settled) / n, 4),
        "total_staked": round(sum_stake, 2), "total_pnl": round(sum_pnl, 2)}
