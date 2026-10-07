"""THE APP'S OWN BET — one committed pick per market per day, and no cheating.

This is a head-to-head scorecard: the app commits to a bet, Julian records his own
in a spreadsheet, and after a week the win counts are compared. That only means
anything if the app cannot quietly advantage itself, so the integrity rules are
enforced HERE, in code, not by convention.

THE FOUR RULES

1. **ONE bet per (station, metric, target_date), and it is IMMUTABLE.** Written
   with INSERT OR IGNORE: the first commit wins and every later wake silently
   leaves it alone. Without this the app would drift its "bet" toward whatever the
   model currently believes, which is not a bet at all -- it is a running
   commentary that always looks right in hindsight.

2. **Never place once the answer is knowable.** Refused if a settled CLI value
   already exists for that target date. A bet placed after the outcome is public
   is not a prediction.

3. **Only at or after the metric's measured decision hour** (`MIN_DECISION_HOUR`),
   and only while the extremum is still mostly ahead. This is the same gate the
   desk uses, so the app bets exactly what it would have told a human to bet.

4. **Only from a live wake cycle.** The launchd job runs every 5 minutes *while
   the laptop is awake*, and `placed_utc` is stamped with the real instant. There
   is no backfill path into this table. **A closed laptop therefore produces NO
   bet for that day -- never a late one.** A missing day is an honest missing day;
   silently filling it in later would be exactly the cheating this guards against.

Columns mirror Julian's spreadsheet so the two ledgers can be compared directly:
    what prediction | date | time manila | bucket | bet amount | bet winnings
    | win or lose

`bet_winnings` is the Kalshi payout if the bet wins: each contract settles at
$1.00, so staking `amount` at price `p` returns `amount / p`. It is recorded at
placement time from the price actually available then -- not recomputed later.
"""
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast import db, gating
from highcast.config import AppConfig
from highcast.metric import get_metric

# Flat stake per bet, in the config's currency. Deliberately FLAT rather than
# Kelly-sized: the comparison being run is "how often is each side right", and
# variable sizing would let a lucky big bet mask a poor hit rate.
# How much of the decision hour's "still ahead" share must survive for a late
# commit to still count as a prediction. 0.75 tolerates the ordinary case of a
# wake landing an hour or two past the knee while the extremum is genuinely still
# forming, and refuses the case that actually happened -- a job started at h12
# when the knee was h5 and 71% of the LOW was already decided.
LATE_FRACTION = 0.75

DEFAULT_STAKE = 1.0


def model_fingerprint(cfg) -> str:
    """A short, stable description of the model that produced a bet.

    WITHOUT THIS A LEDGER IS UNAUDITABLE. Every knob below has been changed at
    least once in this project, and a row that does not say which settings made
    it cannot be compared against a row made under different ones -- you end up
    averaging two different models and calling it a hit rate. Recorded at
    PLACEMENT time, never recomputed, so it describes what actually ran.

    Deliberately terse and human-readable rather than a hash: the point is that a
    human reading the ledger a month later can see the difference at a glance.
    """
    m = cfg.model
    # `k` is PER-METRIC as of 2026-08-02 (KLAX LOW ships k=1, KLAX HIGH stays
    # k=0), so the fingerprint records BOTH -- `k<low>/<high>`. Recording only
    # the station-wide default would make two bets from genuinely different
    # models carry the same fingerprint, which is exactly the unauditable ledger
    # this function exists to prevent.
    return (f"k{m.intraday_bandwidth_for('low')}/{m.intraday_bandwidth_for('high')}"
            f"|lg{int(m.late_commit_guard)}"
            f"|ha{int(m.hybrid_anchor)}"
            f"|rl{int(m.intraday_residual_low)}"
            f"|rh{int(m.intraday_residual_high)}"
            f"|cal{cfg.calibration.mode}")


def _settled_actual(conn, cfg: AppConfig, metric: str, target: str):
    col = get_metric(cfg, metric).target_col
    row = conn.execute(
        f"SELECT {col} AS v FROM cli_reports WHERE station=? AND date=?",
        (cfg.station.icao, target)).fetchone()
    return None if row is None or row["v"] is None else int(row["v"])


def _final_settled_actual(conn, cfg: AppConfig, metric: str, target: str):
    """Like `_settled_actual`, but refuses to grade against a CLI row whose
    product has not been proven FINAL for `target`.

    ⚠️ THE INCIDENT THIS GUARDS. KMIA 2026-08-01: the only CLI product IEM
    ever issued for that date (202608012033-KMFL-CDUS42-CLIMIA) was filed
    8.5h before the climate day closed and reported low 82; the true low was
    74F. `settle()` used to grade a >=81 bet WON against that row. It had
    lost. See `scripts/migrate_prelim_settlement.py`.

    `cli_reports.cli_row_is_final` is the ONE definition of "final" -- shared
    with `pick.settle` and `backfill._yesterday_cli`'s re-fetch decision, not
    a copy grown here. Returns None when a value exists but is still
    preliminary (not `_settled_actual`'s "no value yet"), so `settle()`'s
    existing "skip and retry next wake" path covers this case too, without
    any change to `settle()`'s control flow.

    Deliberately a SEPARATE function from `_settled_actual`: that one backs
    RULE 2 in `place()` ("don't bet once the answer is public"), which stays
    unchanged here -- a preliminary report does not make the outcome public,
    so it must not block a brand-new bet either. Only GRADING an existing bet
    needs the finality proof; this function is not reused for that gate.
    """
    from highcast.fetchers import cli_reports
    col = get_metric(cfg, metric).target_col
    row = conn.execute(
        f"SELECT {col} AS v, raw FROM cli_reports WHERE station=? AND date=?",
        (cfg.station.icao, target)).fetchone()
    if row is None or row["v"] is None:
        return None
    product = None
    if row["raw"]:
        try:
            product = json.loads(row["raw"]).get("product")
        except (TypeError, ValueError):
            product = None
    if not cli_reports.cli_row_is_final(product, target, cfg.station.std_utc_offset):
        return None
    return int(row["v"])


def existing(conn, cfg: AppConfig, metric: str, target: str):
    return conn.execute(
        """SELECT * FROM app_bets WHERE station=? AND metric=? AND target_date=?""",
        (cfg.station.icao, metric, target)).fetchone()


def _features_at_commit(conn, cfg: AppConfig, spec, target: str, now) -> str | None:
    """JSON of the nowcast's input row at commit time. None on any failure --
    a diagnostic must never be able to block a bet."""
    import json
    from zoneinfo import ZoneInfo as _Z
    try:
        from highcast.models.intraday import build_checkpoint_frame
        hour = now.astimezone(_Z(cfg.station.tz)).hour
        # PER-METRIC -- see `ModelConfig.intraday_bandwidth_for`. This frame has
        # to be built with the SAME hours the live model trained on, or the
        # captured features describe a different model than the one that bet.
        _m = getattr(cfg, "model", None)
        k = (_m.intraday_bandwidth_for(spec.key) if _m is not None else 0)
        hrs = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
        fr = build_checkpoint_frame(conn, cfg, [target], hours=hrs,
                                    require_target=False, metric=spec)
        fr = fr[fr["hour"] == float(hour)] if "hour" in fr.columns else fr
        if fr.empty:
            return None
        row = {c: (None if fr[c].isna().iloc[0] else float(fr[c].iloc[0]))
               for c in fr.columns if c not in ("target_date",)}
        return json.dumps(row, sort_keys=True)
    except Exception:  # noqa: BLE001 - see docstring
        return None


def place(conn, cfg: AppConfig, spec, now: datetime,
          stake: float = DEFAULT_STAKE) -> dict:
    """Commit the app's bet for today's market, if all four rules allow it.

    Returns {"placed": bool, "reason": str, ...}. Never raises on a refusal --
    the wake cycle calls this every 5 minutes and a refusal is the normal case.
    """
    from highcast import signals

    target = now.astimezone(ZoneInfo(cfg.station.tz)).date().isoformat()
    prior = existing(conn, cfg, spec.key, target)
    if prior is not None:
        # RULE 1. Already committed — locked, whatever the model now thinks.
        return {"placed": False, "reason": "already committed today",
                "bucket": prior["bucket"]}

    if _settled_actual(conn, cfg, spec.key, target) is not None:
        # RULE 2. The outcome is public; betting now would be hindsight.
        return {"placed": False, "reason": "outcome already settled — refusing"}

    local_hour = now.astimezone(ZoneInfo(cfg.station.tz)).hour
    knee = gating.min_decision_hour(cfg.station.icao, spec.key)
    if knee is None:
        # RULE 3 (unmeasured station). This commits REAL MONEY, so an
        # unmeasured decision hour must refuse outright, never default to 0
        # -- that would let the app bet at any hour with no measured basis
        # for calling the pick settled.
        return {"placed": False,
                "reason": "no measured decision hour for this station — refusing to bet blind"}
    if local_hour < knee:
        # RULE 3. Too early: the model's pick is still moving between hours.
        return {"placed": False,
                "reason": f"before the decision hour ({knee}:00 local)"}
    # RULE 3b. TOO LATE IS ALSO CHEATING, and until 2026-07-29 nothing checked it.
    # The gate was "at or after the knee" with no upper bound, so a job started
    # late in the day caught up and committed immediately -- with hours of extra
    # observations the decision hour exists to exclude. Caught when KMIA's launchd
    # job was first loaded at 12:07 ET: both bets committed at local hour 12
    # instead of h5/h11, and the LOW's had 71% of its outcome already decided
    # (still-future 29% at h12 versus 76% at its h5 knee). That is transcription
    # wearing a prediction's timestamp.
    #
    # The test is on STILL-FUTURE, not on a clock offset, because that measures
    # the thing that actually matters -- how much of the answer is still unknown.
    # An hour two past the knee is fine if the extremum has not formed; an hour
    # one past it is not, if it has.
    #
    # KLAX has the same hole and has only ever avoided it because its job has run
    # continuously. Any restart, migration or launchctl reload mid-day would have
    # produced the same contaminated row.
    if bool(getattr(getattr(cfg, "model", None), "late_commit_guard", False)):
      knee_fut = gating.extremum_still_future(cfg.station.icao, spec.key, knee)
      now_fut = gating.extremum_still_future(cfg.station.icao, spec.key, local_hour)
      if (knee_fut is not None and now_fut is not None
              and now_fut < knee_fut * LATE_FRACTION):
          return {"placed": False,
                  "reason": (f"too late: at hour {local_hour} only {now_fut:.0%} of "
                             f"the {spec.key} is still ahead, versus {knee_fut:.0%} at "
                             f"the {knee}:00 decision hour — that is hindsight, "
                             f"refusing")}

    out = signals.live_decision(conn, cfg, spec, now)
    pred, market = out.get("pred"), out.get("market")
    if pred is None:
        return {"placed": False, "reason": "no prediction available"}
    if not market:
        return {"placed": False, "reason": "no live market prices"}

    # RULE 5. Never bet a pick the desk has VETOED. "No edge" is fine to bet —
    # this ledger measures accuracy, not profit, and gating on edge would leave
    # most days blank. But a regime veto means the app has already judged its own
    # number untrustworthy (a day-ahead built with every morning feature NaN, say),
    # and betting it records a loss the app knew was coming. That is not a
    # forecast being wrong; it is knowingly logging garbage into the scorecard.
    decision = out.get("decision")
    if decision is not None and getattr(decision, "veto", False):
        return {"placed": False, "reason": f"vetoed: {decision.reason}"}

    top = max(market, key=lambda r: (r.get("p_model") or 0.0))
    price = top.get("market_price")
    if price is None or not (0.0 < price < 1.0):
        return {"placed": False, "reason": "no usable price on the model's pick"}

    placed_manila = now.astimezone(ZoneInfo(cfg.display_tz)).strftime("%H:%M")
    db.upsert_ignore(conn, "app_bets", [{
        "station": cfg.station.icao, "metric": spec.key, "target_date": target,
        "placed_utc": now.isoformat(), "placed_manila": placed_manila,
        "placed_local_hour": local_hour,
        "bucket": top["label"], "p_model": top.get("p_model"), "price": price,
        "stake": float(stake), "payout_if_win": float(stake) / price,
        "actual": None, "won": None, "settled_utc": None,
        # Stored so the Discord alert can quote the desk verbatim instead of
        # re-deriving it later against a market that has since moved.
        "reason": (out.get("decision").reason
                   if out.get("decision") is not None else None),
        "notified_utc": None,
    }])
    return {"placed": True, "reason": "committed", "bucket": top["label"],
            "price": price, "p_model": top.get("p_model"),
            "placed_manila": placed_manila}


def settle(conn, cfg: AppConfig, now: datetime) -> int:
    """Fill actual / won for every committed bet whose CLI value has landed
    AND been proven FINAL (see `_final_settled_actual`). A bet whose only
    stored CLI value is preliminary stays unsettled and is retried on the
    next call -- never graded wrong. Only ever writes the settlement columns
    — the bet itself is untouched, so a settled row still shows the price and
    bucket committed to at placement.
    """
    from highcast import signals
    rows = conn.execute(
        "SELECT * FROM app_bets WHERE station=? AND settled_utc IS NULL",
        (cfg.station.icao,)).fetchall()
    n = 0
    for r in rows:
        actual = _final_settled_actual(conn, cfg, r["metric"], r["target_date"])
        if actual is None:
            continue
        signals.check_settlement_against_market(
            conn, cfg, r["metric"], r["target_date"], actual)
        lo, hi = _bucket_bounds(r["bucket"])
        won = int(lo is not None and lo <= actual <= hi) if lo is not None else None
        conn.execute(
            """UPDATE app_bets SET actual=?, won=?, settled_utc=?
               WHERE station=? AND metric=? AND target_date=?""",
            (actual, won, now.isoformat(), cfg.station.icao, r["metric"],
             r["target_date"]))
        n += 1
    conn.commit()
    return n


def _bucket_bounds(label: str | None):
    """Parse a Kalshi range label into (lo, hi). Handles the open tails."""
    if not label:
        return None, None
    s = label.replace("–", "-").replace("−", "-").strip()
    try:
        if s.startswith("≤"):
            return -999, int(float(s[1:]))
        if s.startswith("≥"):
            return int(float(s[1:])), 999
        a, b = s.split("-")
        return int(float(a)), int(float(b))
    except (ValueError, IndexError):
        return None, None


def missing_commits(conn, cfgs_by_station: dict[str, AppConfig],
                    now: datetime | None = None,
                    lookback_days: int = 1) -> list[dict]:
    """Commit hours that have passed with NO `app_bets` row written.

    WHY THIS EXISTS. A power outage on 2026-08-08/09 took the laptop down for
    11.6h then 9.6h and cost four commits — KLAX high on 08-08, and KLAX high,
    KMIA high and KMIA low on 08-09. **Nothing said so.** No alert, no log
    line, no dashboard flag; Julian found out by asking two days later.

    This is NOT the feed-staleness problem (QUEUE item 4). Every feed can be
    perfectly healthy while the process simply was not running at 17:00Z, and
    the only trace is a row that does not exist. Absence has no timestamp, so
    nothing downstream can notice it — the check has to be explicit.

    ⚠️ THE TARGET DATE IS STATION-LOCAL. At 05:10Z on 2026-08-10, KMIA is on
    08-10 (01:10 EDT) while KLAX is still on 08-09 (22:10 PDT). Comparing a
    station's clock HOUR against its commit hour without asking which DAY that
    hour belongs to invents misses for a day that has not started — I made
    exactly that error while diagnosing the outage.

    A REFUSAL COUNTS AS COMMITTED. `app_bets` records the DECISION, not the
    trade: "no edge" and "past the tradeable window" are both rows, and both
    mean the cycle ran. Only an absent row is a miss.

    Each entry carries `still_actionable` — an overdue commit inside the
    tradeable window can still be placed by hand; one past the window is
    history. Collapsing the two would bury the case worth acting on.
    """
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    for icao, cfg in sorted(cfgs_by_station.items()):
        local = now.astimezone(ZoneInfo(cfg.station.tz))
        for back in range(lookback_days, -1, -1):
            target = local.date() - timedelta(days=back)
            for metric in ("low", "high"):
                hour = gating.MIN_DECISION_HOUR.get(icao, {}).get(metric)
                if hour is None:
                    continue
                # Only judge an hour that has actually arrived. For an earlier
                # target date the whole day is behind us; for today, compare
                # against the local clock.
                if back == 0 and local.hour < hour:
                    continue
                if existing(conn, cfg, metric, target.isoformat()):
                    continue
                cutoff = gating.personal_cutoff_instant(cfg, target, metric)
                out.append({
                    "station": icao, "metric": metric,
                    "target_date": target.isoformat(),
                    "commit_hour": hour,
                    "still_actionable": bool(cutoff and now < cutoff),
                    "window_closes": cutoff.isoformat() if cutoff else None,
                })
    return out


def coverage(conn, cfg: AppConfig, days: int = 7) -> dict:
    """Which recent days the app actually managed to bet on — and which it missed.

    A missed day is SILENT otherwise, and silence is dangerous here: during a
    week-long head-to-head, "the app went 4-1" means something very different if
    it only bet on 5 of 7 days. The integrity rules guarantee the app never bets
    late; they cannot guarantee it was awake, so the gaps have to be surfaced.

    A day counts as EXPECTED once its climate day has settled (a CLI value
    exists) — before that there is nothing to have missed. The usual cause of a
    gap is the laptop asleep at the decision hour; on this machine the battery
    power profile sleeps after 1 minute while AC never does.
    """
    have = {(r["metric"], r["target_date"]) for r in conn.execute(
        "SELECT metric, target_date FROM app_bets WHERE station=?",
        (cfg.station.icao,))}
    recent = [r["date"] for r in conn.execute(
        """SELECT date FROM cli_reports WHERE station=?
             AND (high_f IS NOT NULL OR low_f IS NOT NULL)
           ORDER BY date DESC LIMIT ?""", (cfg.station.icao, days))]
    out = {}
    for metric in ("high", "low"):
        missing = [d for d in recent if (metric, d) not in have]
        out[metric] = {"expected": len(recent),
                       "committed": len(recent) - len(missing),
                       "missing": sorted(missing, reverse=True)}
    return {"days": days, "per_metric": out}


def ledger(conn, cfg: AppConfig, metric: str | None = None) -> dict:
    """The app's committed bets, newest first, in Julian's column order."""
    q = "SELECT * FROM app_bets WHERE station=?"
    args: list = [cfg.station.icao]
    if metric:
        q += " AND metric=?"
        args.append(metric)
    rows = [dict(r) for r in conn.execute(q + " ORDER BY target_date DESC, metric",
                                          args)]
    settled = [r for r in rows if r["won"] is not None]
    wins = sum(r["won"] for r in settled)
    staked = sum(r["stake"] for r in settled)
    returned = sum(r["payout_if_win"] for r in settled if r["won"])
    return {
        "rows": rows,
        "n": len(rows), "n_settled": len(settled), "wins": wins,
        "win_rate": (wins / len(settled)) if settled else None,
        "staked": staked, "returned": returned, "pnl": returned - staked,
        "currency": cfg.staking.currency,
        "coverage": coverage(conn, cfg),
    }
