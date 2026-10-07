"""Where the order bot meets the wake cycle. Orchestration only.

Design: an internal design note.

THE THREE-WAY SPLIT, and why it is worth a third file:

    orderbot.py       decides WHAT to order      (no network)
    kalshi_client.py  can SEND one               (no weather)
    botrunner.py      joins them                 (no capability of its own)

`botrunner` calls `place_order`, but holds no ability to authenticate or reach a
trading endpoint -- `tests/test_freeze_contracts` bans those everywhere except
`kalshi_client`. A call site is not a capability.

IDEMPOTENCY IS STRUCTURAL. The wake cycle fires every five minutes, so the commit
hour brings roughly a dozen invocations. `bot_orders` has PRIMARY KEY
(station, metric, target_date) and this writes with INSERT OR IGNORE, so the
FIRST decision of the hour is the record and later ticks cannot overwrite it.
Same principle as `app_bets`: a decision rewritten as the model changes its mind
is hindsight, not a decision.

REFUSALS ARE RECORDED TOO. Two weeks of "why not" is the evidence the arming
decision rests on. Logging only orders would leave an empty table, and "never
qualified" would be indistinguishable from "never ran".
"""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

from highcast import gating, kalshi_client, orderbot
from highcast.buckets import bucket_probs  # noqa: F401  (kept for symmetry)


def _today(cfg, now: dt.datetime) -> tuple[str, int]:
    local = now.astimezone(ZoneInfo(cfg.station.tz))
    return local.date().isoformat(), local.hour


def run_once(conn, cfg, now: dt.datetime, *, ranges=None, env=None,
             gates_ok: bool | None = None, calibrated: float | None = None,
             transport=None) -> dict:
    """One evaluation. Safe to call every wake; writes at most one row per day.

    NEVER RAISES. This runs inside a cycle that also fetches observations and
    issues predictions -- a trading bug must not cost the weather pipeline. Any
    exception becomes a recorded refusal.

    `ranges`, `gates_ok` and `calibrated` are injectable so the wiring can be
    tested without a network or a fitted model; production passes none of them.
    """
    env = dict(env if env is not None else {})
    station, metric = orderbot.TRADED_STATION, orderbot.TRADED_METRIC
    try:
        if cfg.station.icao != station:
            return {"place": False, "reason": f"not {station}"}
        target, local_hour = _today(cfg, now)
        if local_hour != orderbot.TRADED_HOUR:
            return {"place": False, "reason": f"hour {local_hour} is not the commit hour"}

        # Already decided today? The PK would reject the write anyway; checking
        # first also avoids re-hitting the market on every tick of the hour.
        if conn.execute(
                """SELECT 1 FROM bot_orders WHERE station=? AND metric=?
                   AND target_date=?""", (station, metric, target)).fetchone():
            return {"place": False, "reason": "already decided today"}

        if ranges is None:
            # `fetch_ranges` returns the RAW Kalshi ladder with p_model=None --
            # it knows nothing about the model. The per-bucket probability comes
            # from the stored prediction's pmf, exactly as `signals.live_decision`
            # builds it. Skipping this step is not a loud failure: every bucket
            # has p_model None, `pick` stays None, and the bot logs
            # "unmeasured_band" forever -- a refusal that reads like a gate
            # decision but actually means the market was never scored. Caught by
            # running it live before the dry-run fortnight started.
            from highcast.buckets import pmf_from_buckets, pmf_mass
            from highcast.fetchers import kalshi as kfetch
            from highcast.metric import get_metric
            spec = get_metric(cfg, metric)
            pred, _kind = gating.preferred_prediction(
                conn, station, target, metric, True,
                local_hour=orderbot.TRADED_HOUR)
            raw = kfetch.fetch_ranges(cfg, target, metric=spec)
            if pred is None or not raw:
                ranges = []
            else:
                pmf = pmf_from_buckets(pred["buckets"])
                ranges = [{"label": r["label"], "lo": r["lo"], "hi": r["hi"],
                           "ticker": r.get("ticker"),
                           "p_model": pmf_mass(pmf, r["lo"], r["hi"]),
                           "market_price": kfetch.resolve_price(r),
                           "yes_bid": r.get("yes_bid"),
                           "yes_ask": r.get("yes_ask")} for r in raw]

        pick = None
        for m in (ranges or []):
            if m.get("p_model") is None:
                continue
            if pick is None or m["p_model"] > pick["p_model"]:
                pick = m

        if calibrated is None and pick is not None:
            calibrated = gating.band_calib(station, metric,
                                           orderbot.TRADED_HOUR, pick["p_model"])
        if gates_ok is None:
            v = gating.decision_gates(
                conn, cfg, metric=metric, target=target,
                hour=orderbot.TRADED_HOUR,
                p_model=(pick or {}).get("p_model"))
            gates_ok, refusals = v.ok, v.refusals
        else:
            refusals = ()

        intent = orderbot.decide(
            station=station, metric=metric, hour=local_hour, target=target,
            ranges=ranges, gates_ok=bool(gates_ok), refusals=tuple(refusals),
            calibrated=calibrated, orders_today=0, budget=1.00)

        # THE EDGE, COMPUTED HERE RATHER THAN TAKEN FROM `intent.edge` ALONE.
        # `orderbot.decide` only fills in `edge` from the market-selection step
        # onward -- a refusal earlier than that (scope, hour, frequency, the
        # gate verdict, calibration/floor) never reaches it, so `intent.edge`
        # is None there even though `calibrated` and the ladder's ask are
        # already both in hand. Same shape as the bucket-recording fix below:
        # compute from what THIS layer already holds instead of trusting a
        # decision object that may have returned early. `gating.MAX_EDGE`'s
        # own comment explains why this needs to accrue on every decision, not
        # just the ones that trip the gate.
        market_ask = (pick or {}).get("yes_ask")
        edge = intent.edge
        if edge is None and calibrated is not None and market_ask is not None:
            try:
                edge = float(calibrated) - float(market_ask)
            except (TypeError, ValueError):
                edge = None

        # BOTH are required for the unattended path: the literal, and an
        # unexpired end date. The literal has to live in the launchd plist for
        # the bot to fire at 02:00 Manila while Julian sleeps -- and a literal in
        # a plist is armed every day until removed. The window makes forgetting
        # fail SAFE. See kalshi_client.ARM_UNTIL_ENV.
        armed = (kalshi_client.is_armed(env)
                 and kalshi_client.arming_window_ok(env, target))
        kid, pem = kalshi_client.load_credentials(env)
        client = kalshi_client.KalshiClient(key_id=kid, private_key_pem=pem,
                                            armed=armed, transport=transport)
        res = {"dry_run": not armed, "placed": False, "reason": intent.reason,
               "env": "demo" if client.is_demo else "production"}
        if intent.place:
            tkr = _ticker(pick)
            if tkr is None:
                # No real ticker means no order that could be accepted. Refusing
                # beats sending a constructed one: a rejected order at the
                # commit hour is a missed day, and it would be logged as an
                # exchange error rather than as our own missing data.
                res["reason"] = "no market ticker in the ladder — not sending"
                intent = orderbot.Intent(place=False, reason=res["reason"],
                                         bucket=intent.bucket,
                                         limit_price=intent.limit_price,
                                         contracts=intent.contracts,
                                         edge=intent.edge)
            else:
                res = client.place_order(
                    ticker=tkr, side="yes", count=intent.contracts,
                    limit_price=intent.limit_price)

        # RECORD THE BUCKET EVEN WHEN REFUSING. `orderbot.decide` returns on the
        # gate check BEFORE it selects a bucket, so `intent.bucket` is None on
        # every refusal -- and refusals are most of the fortnight. Two weeks of
        # "confidence_floor" rows with no bucket would say the gate fired but
        # not what it fired ON, which is precisely the question the arming
        # review has to answer. `place` already distinguishes "ordered" from
        # "considered", so this column can safely mean the day's pick.
        _record(conn, station, metric, target, now, local_hour, intent,
                calibrated, market_ask, res,
                bucket=intent.bucket or (pick or {}).get("label"), edge=edge)
        return {"place": intent.place, "reason": intent.reason,
                "dry_run": res.get("dry_run", True),
                "placed": bool(res.get("placed"))}
    except Exception as e:  # noqa: BLE001 - the weather pipeline must survive
        return {"place": False, "reason": f"orderbot error: {type(e).__name__}: {e}"}


def _ticker(pick: dict | None) -> str | None:
    """The chosen bucket's REAL Kalshi ticker, or None.

    ⚠️ THIS USED TO CONSTRUCT ONE: f"{series}-{target}-{bucket}", yielding
    "KXHIGHMIA-2026-08-01-92–93". No exchange ticker has ever contained an
    EN DASH, and Kalshi's date suffix is "26AUG01", not "2026-08-01". Every
    single order would have been rejected -- and rejected at the commit hour,
    logged as an exchange error, on the one day of the fortnight the bot finally
    wanted to trade. The old docstring said the suffix "must be confirmed
    against a live event listing rather than guessed", and then guessed.

    The listing was already being fetched. `parse_markets` simply dropped the
    `ticker` field; now it keeps it, so the value is carried straight through
    from the ladder rather than reconstructed. Returning None when it is absent
    is deliberate: no ticker means no order that could be accepted, and a
    refusal we can name beats a rejection we would have to diagnose.
    """
    return (pick or {}).get("ticker")


def _record(conn, station, metric, target, now, local_hour, intent,
            calibrated, ask, res, bucket=None, edge=None) -> None:
    """INSERT OR IGNORE: the first decision of the day wins and is immutable.

    `edge` (calibrated - ask) is logged whenever it is known, place or
    refuse -- see `gating.MAX_EDGE` for why the distribution needs to
    accrue on every decision, not just the ones that trip the gate."""
    conn.execute(
        """INSERT OR IGNORE INTO bot_orders
             (station, metric, target_date, decided_utc, local_hour, place,
              reason, bucket, limit_price, contracts, calibrated, ask, edge,
              dry_run, env, placed, response)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (station, metric, target, now.isoformat(), local_hour,
         1 if intent.place else 0, intent.reason,
         bucket if bucket is not None else intent.bucket,
         intent.limit_price, intent.contracts,
         calibrated, ask, edge,
         1 if res.get("dry_run", True) else 0, res.get("env"),
         1 if res.get("placed") else 0, json.dumps(
             {k: v for k, v in res.items() if k != "reason"})))
    conn.commit()
