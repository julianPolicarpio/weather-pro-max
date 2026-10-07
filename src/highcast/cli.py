import argparse

from highcast import backfill, db
from highcast.config import load_config


def cmd_init_db(args) -> None:
    cfg = load_config(args.config)
    db.connect(cfg.db_path)
    print(f"initialized {cfg.db_path}")


def cmd_backfill(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    counts = backfill.run(conn, cfg, years=args.years)
    print(counts)
    print(backfill.backfill_ensemble(conn, cfg))
    print(backfill.backfill_aloft(conn, cfg))


def cmd_reparse_obs(args) -> None:
    """Re-derive obs columns from the stored raw METARs (no network). Needed
    after the parser learns a new remark group -- e.g. the 6-hour MINIMUM
    (2sTTT), which history was never parsed for."""
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.reparse_obs(conn, cfg))


def cmd_board(args) -> None:
    """Print the Station Table: raw physical state, decoded, source optional.

    Manual-analysis aid, not a recommender. It shows the facts a probability is
    made of -- running extremum with its stall clock, dewpoint depression, wind
    with the raw bearing, pressure and its 3-hour tendency -- so a model read can
    be argued with. Display only; it writes nothing and votes on nothing.
    """
    from highcast import board, preflight
    cfgs = ([load_config(c) for c in args.configs] if args.configs
            else [load_config(args.config)])
    rows = []
    for cfg in cfgs:
        conn = db.connect(cfg.db_path)
        # One scan per station serves every metric: the climatology pool is
        # keyed by hour and season, not by which extremum is being read.
        bands = board.climatology_bands(conn, cfg)
        metrics = [args.metric] if args.metric else ["low", "high"]
        for m in metrics:
            r = board.row(conn, cfg, m)
            r["evidence"] = board.evidence(r, bands)
            r["preflight"] = preflight.grade(conn, cfg, m, r["target_date"])
            rows.append(r)
    print(board.render(rows, show_raw=args.raw))
    cur = (cfgs[0].station.icao, args.metric) if args.metric else None
    print(board.render_reliability(board.reliability_table(highlight=cur)))


def cmd_fetch(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.fetch_snapshot(conn, cfg))


def in_climate_day(cfg, now) -> bool:
    """True while TODAY'S (local calendar date's) climate day is running.
    Climate days tile time continuously, so the check is specifically against
    the current local date's window - during PDT that window opens at 01:00."""
    from zoneinfo import ZoneInfo
    from highcast.timeutil import climate_day_bounds

    local_date = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    start, end = climate_day_bounds(local_date, cfg.station.std_utc_offset)
    return start <= now < end


def should_predict_dayahead(conn, cfg, now, metric: str = "high") -> bool:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo(cfg.station.tz))
    if local.hour < 9:
        return False
    # Only a forecast issued at/after today 08:00 local counts as done: an
    # evening-before forecast for today must not block the sharper 9am refresh
    # (which sees morning obs + the 12Z guidance).
    cutoff = datetime(local.year, local.month, local.day, 8, 0,
                      tzinfo=ZoneInfo(cfg.station.tz)
                      ).astimezone(timezone.utc).isoformat(timespec="seconds")
    # metric-scoped: a stored HIGH forecast must not satisfy the LOW's check
    row = conn.execute(
        """SELECT 1 FROM predictions WHERE station=? AND target_date=?
             AND kind='dayahead' AND metric=? AND issued_at_utc >= ? LIMIT 1""",
        (cfg.station.icao, local.date().isoformat(), metric, cutoff)).fetchone()
    return row is None


# The evening run is a DIFFERENT OBJECT from the 09:00 day-ahead and gets its
# own kind so nothing can conflate them.
#
# WHY, measured 2026-08-01 on stored predictions vs settled CLI:
#
#     KLAX HIGH   evening (morn_* all NaN)   n=19   MAE 4.42F   bias -4.05F
#                 morning refresh            n=14   MAE 1.45F   bias +0.19F
#
# Three times the error and a systematic 4F cold bias. The cause is a train/
# serve skew, not a bad model: the HIGH's feature cutoff is 09:00 LOCAL ON THE
# TARGET DAY, and the evening run fires ~17:00 the PREVIOUS day, so every
# `morn_*` observation it reads is still ~16 hours in the future and comes back
# NaN. `features.build_features` takes no as-of parameter, so a BACKTEST always
# reconstructs real morning values -- which means the walk-forward behind
# `gating.preferred_prediction` ("day-ahead MAE 1.509 beats intraday 1.620")
# measured the MORNING object exclusively and never saw this one at all.
#
# 38 of 74 stored day-ahead rows were evening rows, so the contamination was the
# majority case. Sharing a kind meant `preferred_prediction` served a 4F-cold
# forecast as the dashboard headline every morning until 09:00, and computed a
# 69% "edge" from it. Separating the kinds makes that unrepresentable rather
# than merely guarded -- `signals._regime_veto` already DETECTED it correctly and
# was simply never consulted by the code that picks the prediction.
#
# Design: docs/superpowers/specs/2026-08-01-evening-dayahead-kind-separation-design.md
#
# Defined in `db` (a stored column value, and `web/app.py` needs it without
# importing this module). Re-exported here because every call site is here.
EVENING_KIND = db.KIND_EVENING


def should_predict_tomorrow(conn, cfg, now, metric: str = "high") -> bool:
    """After 5pm local, issue the evening guidance-only forecast for tomorrow's
    market — unless one already exists for that target date.

    Checks for `EVENING_KIND`, not 'dayahead': after the split those are
    different rows, and looking for the wrong one would re-issue the evening
    forecast on every wake for the rest of the night.
    """
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo(cfg.station.tz))
    if local.hour < 17:
        return False
    tomorrow = (local.date() + timedelta(days=1)).isoformat()
    row = conn.execute(
        """SELECT 1 FROM predictions WHERE station=? AND target_date=?
             AND kind=? AND metric=? LIMIT 1""",
        (cfg.station.icao, tomorrow, EVENING_KIND, metric)).fetchone()
    return row is None


def auto_cycle(conn, cfg, config_path) -> list[str]:
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    now = datetime.now(timezone.utc)
    actions = []
    tomorrow = (now.astimezone(ZoneInfo(cfg.station.tz)).date()
                + timedelta(days=1)).isoformat()
    for mk in ("high", "low"):
        suffix = "" if mk == "high" else "_low"
        if should_predict_dayahead(conn, cfg, now, mk):
            cmd_predict(_ns(config=config_path, date=None, live=False, metric=mk))
            actions.append(f"dayahead{suffix}")
        if should_predict_tomorrow(conn, cfg, now, mk):
            cmd_predict(_ns(config=config_path, date=tomorrow, live=False,
                            metric=mk, kind=EVENING_KIND))
            actions.append(f"evening{suffix}")
    # THE ORDER BOT. Fenced in its own try/except on top of run_once's internal
    # guard: this cycle also fetches observations and issues predictions, and a
    # trading bug must never cost the weather pipeline. It is a no-op except at
    # KMIA HIGH's commit hour, and dry-run unless explicitly armed.
    #
    # `orderbot_enabled` is the OUTERMOST of three independent off-switches and
    # is checked BEFORE the import, so a disabled bot does not even load. The
    # other two (the `HIGHCAST_TRADING` literal and an unexpired
    # `HIGHCAST_TRADING_UNTIL`) are unchanged and still required to arm. Turned
    # off on both stations 2026-08-02 to focus on the paper demo account.
    if not getattr(cfg, "orderbot_enabled", True):
        # Silent by design: this is a no-op every wake for as long as it is off,
        # and an `orderbot:disabled` line on all ~288 daily cycles would be pure
        # noise. `highcast paper-account` reports the flag's state instead, so
        # "why is the bot not trading?" has one honest place to look.
        pass
    else:
        try:
            import os as _os
            from highcast import botrunner
            _b = botrunner.run_once(conn, cfg, now, env=dict(_os.environ))
            if _b.get("place") or "already" not in (_b.get("reason") or ""):
                if _b.get("reason") and "not the commit hour" not in _b["reason"] \
                        and "not KMIA" not in _b["reason"]:
                    actions.append(
                        f"orderbot:{'PLACE' if _b.get('place') else 'no'}")
        except Exception as _e:  # noqa: BLE001
            actions.append(f"orderbot_error:{type(_e).__name__}")

    if in_climate_day(cfg, now):
        cmd_predict(_ns(config=config_path, date=None, live=True))
        actions.append("intraday")
    # Accumulate the paper-trade track record: settle whatever the day's CLI
    # value now allows, then snapshot today's high signal (model probs + live
    # market prices + decision). Fenced — a Kalshi/network hiccup must never
    # break the wake cycle that keeps predictions fresh.
    try:
        from highcast import signals
        from highcast.metric import get_metric
        signals.settle_signals(conn, cfg, now)
        out = signals.record_live(conn, cfg, get_metric(cfg, "high"), now)
        if out.get("signal_id"):
            actions.append("signal_high")
        if signals.record_shadow_live(conn, cfg, get_metric(cfg, "high"), now):
            actions.append("shadow_high")
    except Exception:  # noqa: BLE001 - logging must not abort the wake
        pass
    # Mirror for the LOW: an intraday nowcast through the climate day + its
    # signal, so the low predictor stays fresh and its paper ledger accrues too.
    try:
        from highcast import signals
        from highcast.metric import get_metric
        if in_climate_day(cfg, now):
            cmd_predict(_ns(config=config_path, date=None, live=True, metric="low"))
            actions.append("intraday_low")
        out_lo = signals.record_live(conn, cfg, get_metric(cfg, "low"), now)
        if out_lo.get("signal_id"):
            actions.append("signal_low")
        if signals.record_shadow_live(conn, cfg, get_metric(cfg, "low"), now):
            actions.append("shadow_low")
    except Exception:  # noqa: BLE001 - logging must not abort the wake
        pass
    # THE APP'S OWN COMMITTED BET. This is the ONLY path that writes `app_bets`,
    # and it runs only from the launchd wake — i.e. only while the laptop is
    # awake. That is deliberate: a closed laptop yields NO bet for that day
    # rather than a late one placed with hindsight. See highcast/appbet.py.
    try:
        from highcast import appbet
        from highcast.metric import get_metric
        appbet.settle(conn, cfg, now)
        for mk in ("high", "low"):
            if appbet.place(conn, cfg, get_metric(cfg, mk), now)["placed"]:
                actions.append(f"app_bet_{mk}")
        # Discord alert for anything committed but not yet announced. Driven off
        # `notified_utc IS NULL`, not "placed this tick", so a send that failed
        # while the webhook was down is retried next wake instead of lost — and a
        # successful one is never repeated by the 5-minute cycle.
        from highcast import notify
        n = notify.notify_pending(conn, cfg, now)
        n += notify.notify_settled(conn, cfg, now)
        # WEATHERGUARD: warn when an advisory means outside factors are in play.
        # Not metric-scoped -- a storm over the station affects both the high and
        # the low, so this fires once per day per advisory kind, not per market.
        n += notify.notify_unstable(conn, cfg, now)
        # MISSED COMMIT: the commit hour passed and no bet was recorded, i.e.
        # the wake cycle was not running. Fires ONLY while the window is still
        # open — a miss he cannot act on goes to the wake log, not his phone.
        # The 2026-08-08/09 outage cost four commits and said nothing.
        n += notify.notify_missed_commit(conn, cfg, now)
        for mk in ("high", "low"):
            n += notify.notify_divergence(conn, cfg, get_metric(cfg, mk), now)
            n += notify.notify_price_drop(conn, cfg, get_metric(cfg, mk), now)
            n += notify.notify_low_preview(conn, cfg, get_metric(cfg, mk), now)
        if n:
            actions.append(f"discord_{n}")
    except Exception:  # noqa: BLE001 - a betting hiccup must not abort the wake
        pass
    # GRADE PUBLISHED PICKS. `pick.settle` was written and tested and then wired
    # to nothing, so every `kind='verdict'` row would keep `won`/`actual` NULL
    # forever -- and `pick.KILL_AFTER_PICKS`, the feature's one pre-registered
    # falsification, cannot fire on a column nothing populates. Grading is the
    # half of that promise that runs unattended, so it belongs on the wake.
    #
    # ITS OWN FENCE, not a share of the appbet block above: that block also sends
    # the Discord alerts, and a grading bug must not swallow them. The error is
    # APPENDED rather than swallowed (the orderbot's pattern) -- this gap went
    # unnoticed precisely because nothing said anything.
    #
    # ⚠️ A NO-OP UNTIL THE WRITER IS WIRED. `pick.record` is still called from
    # nowhere, so `picks` is empty and this settles 0 rows; it is correct in
    # advance rather than another thing to remember later. Wiring the writer is
    # a separate, still-open decision (2026-08-02 unfreeze record).
    try:
        from highcast import pick
        n_picks = pick.settle(conn, cfg, now)
        if n_picks:
            actions.append(f"picks_settled_{n_picks}")
    except Exception as _e:  # noqa: BLE001 - grading must never abort the wake
        actions.append(f"pick_settle_error:{type(_e).__name__}")
    return actions


def _ns(**kw):
    return argparse.Namespace(**kw)


def _coverage_line(conn, cfg) -> str | None:
    """One line when a six-hour group we were ENTITLED to is missing.

    SILENT ON A NORMAL DAY, on purpose. Every climate day is structurally
    ~75% covered — the windows spanning its start and end straddle the
    boundary and are inadmissible however well the station reports — so
    printing coverage unconditionally would emit 288 identical lines a day and
    train the reader to skip them. This fires only when a window that WOULD
    have been usable went unreported, which is the state that cost a bucket on
    2026-08-06 (KMIA settled 75F at 01:51 local; our best evidence was the
    78.1F METAR two minutes later, and the 12:00Z group never arrived).

    Windows that have not closed yet come back as `pending`, never `missing` —
    otherwise this would cry wolf on nearly every wake for most of the day.
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    from highcast import latentmax
    now = datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.station.tz)
    today = now.astimezone(tz).date()
    out = []
    for metric in ("high", "low"):
        for day in (today - timedelta(days=1), today):
            cov = latentmax.six_hour_coverage(conn, cfg, day, metric,
                                              asof_utc=now.isoformat())
            if not cov["missing_groups"]:
                continue
            blind = sum(g["minutes"] for g in cov["gaps"])
            hours = ", ".join(m[11:16] + "Z" for m in cov["missing_groups"])
            out.append(f"  [blind] {cfg.station.icao} {metric.upper()} {day}: "
                       f"no 6-hour group for {hours} — "
                       f"{blind} min on spot readings alone "
                       f"({cov['covered_minutes']}/{cov['day_minutes']} covered)")
    return "\n".join(out) if out else None


def _missed_commit_lines(conn, cfg) -> str | None:
    """One line per commit hour that passed with no `app_bets` row.

    THE OUTAGE THIS EXISTS FOR, 2026-08-08/09: the laptop was down 11.6h then
    9.6h and four commits were lost — KLAX high on 08-08, and KLAX high, KMIA
    high and KMIA low on 08-09. **Nothing said so.** Every feed was healthy;
    the process simply was not running at 17:00Z, and absence leaves no
    timestamp for anything downstream to notice.

    Reports only THIS station — the wake cycle runs per-config, so KMIA's job
    reports KMIA. Silent when nothing is missing.
    """
    from datetime import datetime, timezone

    from highcast import appbet
    rows = appbet.missing_commits(conn, {cfg.station.icao: cfg},
                                  datetime.now(timezone.utc), lookback_days=1)
    out = []
    for m in rows:
        urgency = ("STILL PLACEABLE — the window is open until "
                   f"{m['window_closes'][11:16]}Z" if m["still_actionable"]
                   else "window closed — nothing to recover")
        out.append(f"  [MISSED COMMIT] {m['station']} {m['metric'].upper()} "
                   f"{m['target_date']} h{m['commit_hour']}: no bet recorded — "
                   f"{urgency}")
    return "\n".join(out) if out else None


def cmd_wake(args) -> None:
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    print(backfill.fill_gaps(conn, cfg))
    # Report-only. Says WHEN we were blind between hourly filings; changes
    # nothing about what gets predicted, committed or traded.
    gaps = _coverage_line(conn, cfg)
    if gaps:
        print(gaps)
    actions = auto_cycle(conn, cfg, args.config)
    print("auto:", actions)
    # AFTER auto_cycle, so a commit made on THIS wake is already recorded and
    # does not get reported as missing on the very cycle that made it.
    missed = _missed_commit_lines(conn, cfg)
    if missed:
        print(missed)
    # STALENESS GUARD (QUEUE item 4). Printed EVERY wake, including when
    # everything is healthy — a guard that only speaks on failure gives no
    # evidence it is still watching, and the two feeds that died silently
    # (`nbm_nbs`, `forecast_conv`) both died while the log looked normal.
    #
    # It covers a DIFFERENT failure from its neighbour here: this one is
    # "the app is running and a feed stopped", where the missed-commit line is
    # "the app is running and skipped a commit".
    #
    # ⚠️ NOTHING NOW COVERS "the app is not running at all" — the dead man's
    # switch was removed 2026-08-28 (see below). An outage that stops the wake
    # cycle stops every alert with it, silently. That is a deliberate,
    # instructed gap, not an oversight.
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from highcast import notify as _notify
    from highcast import staleness
    try:
        print(" ", staleness.wake_line(conn, cfg))
        n_stale = _notify.notify_stale_feeds(conn, cfg, _dt.now(_tz.utc))
        if n_stale:
            print(f"  stale-feed alerts sent: {n_stale}")
    except Exception as e:  # noqa: BLE001 - a monitor must never break the wake
        print(f"  staleness check failed: {type(e).__name__}: {e}")
    # MARKET SENSOR (QUEUE item 13). Reads the ladder we already store every 5
    # minutes and says two things, both MONITORING ONLY:
    #   * divergence — the market implies an extreme our observations cannot yet
    #     reach, i.e. we are blind RIGHT NOW rather than in hindsight;
    #   * grade conflict — the market's confident call contradicts a grade we
    #     recorded, which is the false-WIN failure MEMORY.md says has bitten
    #     twice off a preliminary CLI.
    #
    # ⚠️ NEVER a trading input. It is the counterparty's own opinion, and the
    # market already beats our model at 78 of 81 hours — routing it into the
    # model would launder their view into ours. Pinned by
    # `test_the_sensor_is_never_wired_into_the_model`.
    #
    # Silent on a normal day: divergence fires on ~25% of day-markets, too often
    # for Discord but fine for the log.
    try:
        from highcast import marketsensor
        line = marketsensor.wake_line(conn, cfg, _dt.now(_tz.utc))
        if line:
            print(line)
    except Exception as e:  # noqa: BLE001 - a monitor must never break the wake
        print(f"  market sensor failed: {type(e).__name__}: {e}")
    # DAY GRADE — input quality for today's two nowcasts. ⚠️ INPUT quality only:
    # it does NOT predict accuracy, and `scripts/day_grade_gate.py` measured
    # every candidate factor flipping sign between markets. `stated` is the
    # accuracy grade; this is the plumbing grade.
    try:
        from highcast import daygrade
        print(daygrade.wake_line(conn, cfg, _dt.now(_tz.utc)))
    except Exception as e:  # noqa: BLE001
        print(f"  day grade failed: {type(e).__name__}: {e}")
    # ⚠️ THE DEAD MAN'S SWITCH WAS REMOVED 2026-08-28, at Julian's instruction.
    # `heartbeat_url` was empty in both station yamls for its entire life, so it
    # never pinged once — an unarmed guard, which this project has repeatedly
    # measured to be WORSE than no guard because it reads as coverage (errors
    # #8, #11, and the storm half of `notify_unstable`). If off-machine
    # liveness monitoring is ever wanted again, add it back ARMED or not at all.


def cmd_creds_check(args) -> None:
    """Print whether the trading setup is correct. The report is built in
    `kalshi_client` (which owns credentials); this only prints it."""
    import os
    from highcast import kalshi_client

    lines, ok = kalshi_client.setup_report(dict(os.environ))
    print("\n".join(lines))
    raise SystemExit(0 if ok else 1)


def cmd_order_test(args) -> None:
    """Place a 1c order, read it back, cancel it.

    Every other guard in this project is about NOT sending an order; this is the
    only thing that answers "and when it does send one, does it work?". Without
    it, a bad signature or a renamed field would first show up on the day the
    bot actually wanted to trade, looking exactly like a gate refusal.

    Like `creds-check`, the work lives in `kalshi_client` — this only prints.
    """
    import os
    from highcast import kalshi_client

    ticker, price_cents, count = args.ticker, args.price_cents, 1
    if args.at_market:
        # Resolve the MODEL'S OWN PICK, so a real order buys what the bot would
        # buy rather than whichever bucket the exchange happened to list first.
        # This lives in the CLI, not in `kalshi_client`: that module knows
        # nothing about weather, and it stays that way.
        pick, note = _bot_pick(args.config)
        print(note)
        if pick is None:
            raise SystemExit(1)
        ticker = pick["ticker"]
        price_cents = int(round((pick["yes_ask"] or 0) * 100))
        count = max(1, int((args.dollars * 100) // max(price_cents, 1)))

    lines, ok = kalshi_client.smoke_test(
        dict(os.environ), demo=args.demo, series=args.series,
        ticker=ticker, price_cents=price_cents, count=count,
        allow_fill=args.at_market, max_dollars=args.dollars,
        cancel=True if args.cancel else None)
    print("\n".join(lines))
    raise SystemExit(0 if ok else 1)


def _bot_pick(config_path):
    """(pick, note) — the bucket the order bot would buy right now, and a
    human-readable account of whether its own gates would allow it.

    A REAL ORDER IS PRINTED WITH THE GATE VERDICT ATTACHED, even when the
    verdict is "refuse". Placing manually against a refusal is a legitimate
    choice; doing it without being told is not.
    """
    import datetime as dt
    from zoneinfo import ZoneInfo

    from highcast import gating, orderbot
    from highcast.buckets import pmf_from_buckets, pmf_mass
    from highcast.fetchers import kalshi as kfetch
    from highcast.metric import get_metric

    cfg = load_config(config_path)
    conn = db.connect(cfg.db_path)
    # STATION-LOCAL date, not `date.today()`.
    #
    # This ran on the machine's local date (Manila) and the market is Miami's.
    # The two disagree for the whole Manila evening and night -- INCLUDING
    # 02:00 Manila, which is KMIA HIGH's commit hour and the exact moment this
    # command is meant to be used. It would have priced tomorrow's market, found
    # no open ladder, and reported "nothing to buy" on the one day it mattered.
    # `botrunner._today` already does this correctly; this helper did not.
    target = dt.datetime.now(dt.timezone.utc).astimezone(
        ZoneInfo(cfg.station.tz)).date().isoformat()
    spec = get_metric(cfg, orderbot.TRADED_METRIC)
    pred, kind = gating.preferred_prediction(
        conn, cfg.station.icao, target, orderbot.TRADED_METRIC, True,
        local_hour=orderbot.TRADED_HOUR)
    raw = kfetch.fetch_ranges(cfg, target, metric=spec)
    if pred is None or not raw:
        return None, "no prediction or no open ladder for today — nothing to buy"
    pmf = pmf_from_buckets(pred["buckets"])
    ranges = [{"label": r["label"], "ticker": r.get("ticker"),
               "yes_ask": r.get("yes_ask"),
               "p_model": pmf_mass(pmf, r["lo"], r["hi"])} for r in raw]
    pick = max(ranges, key=lambda m: m["p_model"])
    if not pick.get("ticker") or not pick.get("yes_ask"):
        return None, f"{pick['label']} has no ticker or no ask — refusing to guess"
    cal = gating.band_calib(cfg.station.icao, orderbot.TRADED_METRIC,
                            orderbot.TRADED_HOUR, pick["p_model"])
    v = gating.decision_gates(conn, cfg, metric=orderbot.TRADED_METRIC,
                              target=target, hour=orderbot.TRADED_HOUR,
                              p_model=pick["p_model"])
    note = "\n".join([
        f"THE BOT'S PICK for {target} ({kind} prediction)",
        f"  bucket    {pick['label']}   ticker {pick['ticker']}",
        f"  model     {pick['p_model']:.1%}   calibrated "
        + (f"{cal:.0%}" if cal is not None else "UNMEASURED"),
        f"  ask       {pick['yes_ask']:.2f}",
        f"  gates     {'PASS' if v.ok else 'REFUSE — ' + ', '.join(v.refusals)}",
        "",
    ])
    if not v.ok:
        note += ("⚠️  The bot would NOT buy this today. Placing anyway is your\n"
                 "    call — it is a manual trade, not the model's decision, and\n"
                 "    it will not appear in the dry-run evidence as one.\n")
    return pick, note


def cmd_bot_log(args) -> None:
    """The order bot's daily decisions — the dry-run evidence.

    This is what the arming decision reads after two weeks: did it fire, at what
    price, and when it refused, why.
    """
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    rows = conn.execute(
        """SELECT * FROM bot_orders ORDER BY target_date DESC LIMIT ?""",
        (args.limit,)).fetchall()
    if not rows:
        print("no decisions recorded yet — the bot runs at KMIA HIGH h14 "
              "(02:00 Manila)")
        return
    print(f"{'date':12s}{'place':>6}{'bucket':>10}{'limit':>7}{'ask':>6}"
          f"{'calib':>7}  reason")
    fired = 0
    for r in rows:
        fired += r["place"]
        print(f"{r['target_date']:12s}{'YES' if r['place'] else 'no':>6}"
              f"{r['bucket'] or '-':>10}"
              f"{(f'{r["limit_price"]:.2f}' if r['limit_price'] else '-'):>7}"
              f"{(f'{r["ask"]:.2f}' if r['ask'] else '-'):>6}"
              f"{(f'{r["calibrated"]:.0%}' if r['calibrated'] else '-'):>7}"
              f"  {(r['reason'] or '')[:64]}")
    print(f"\n{fired}/{len(rows)} days would have ordered. "
          f"All dry-run: {all(r['dry_run'] for r in rows)}")


def cmd_backtest(args) -> None:
    import pandas as pd
    from highcast import backtest as bt, features
    from highcast.metric import get_metric
    from highcast.models.baseline import BaselineBlend

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, args.metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    df = features.build_features(conn, cfg, spec, dates)
    if args.model == "baseline":
        # The blend production feeds the nowcast is metric-scoped too: the LOW
        # uses bias-corrected guidance, the HIGH does not.
        pc = features.uses_bias_correction(spec.key)
        factory = lambda: BaselineBlend(prefer_corrected=pc)  # noqa: E731
    else:
        # Backtest the model PRODUCTION actually runs, not a lookalike: the CLI
        # used to hardcode DayAheadModel and so never measured the shipped
        # hybrid (config `model.hybrid_anchor`).
        #
        # ...and it ALSO has to use production's FEATURE COLUMNS. Until
        # 2026-07-26 this passed no `feature_cols`, so it silently backtested the
        # full 42-column `FEATURES` while production runs `model_features(metric)`
        # — 38 for the LOW, 20 for the HIGH. That is a different model, and its
        # numbers are not the shipped model's. Every measured decision this
        # session (bias, NBM, marine-layer) was about WHICH COLUMNS each metric
        # gets, so backtesting the undifferentiated set measures none of them.
        from highcast.models.dayahead import DayAheadModel, HybridDayAheadModel
        cols = features.model_features(spec.key)
        use_hybrid = bool(getattr(getattr(cfg, "model", None),
                                  "hybrid_anchor", False))
        klass = HybridDayAheadModel if use_hybrid else DayAheadModel
        factory = lambda: klass(cal_mode=cfg.calibration.mode,  # noqa: E731
                                feature_cols=cols)
        print(f"# model: {'hybrid' if use_hybrid else 'plain'} day-ahead "
              f"(metric={args.metric}, {len(cols)} feature cols, "
              f"cal={cfg.calibration.mode})")
    recs = bt.walk_forward(df, factory, refit_every=args.refit_every)
    print(pd.Series(bt.summarize(recs, df)).to_string())


def cmd_calibration_backtest(args) -> None:
    import json

    from highcast import backtest as bt, features
    from highcast.metric import get_metric

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, "high")
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    df = features.build_features(conn, cfg, spec, dates)
    result = bt.compare_calibration_modes(df, refit_every=args.refit_every,
                                          metric=spec.key)
    print(json.dumps(result, indent=2, default=float))


_ARM_TOKENS = {
    "intraday": {"kind": "intraday", "residual": False},
    "intraday-residual": {"kind": "intraday", "residual": True},
    "dayahead": {"kind": "dayahead", "residual": False},
}


def cmd_verify_hour(args) -> None:
    """Re-runnable decision-hour verification: walk-forward every requested
    arm at a single station-local hour, restrict to the days matched across
    all of them, and paired-bootstrap each arm against the first (baseline)
    arm. Replaces the throwaway scratch scripts that used to produce the
    headline decision-hour numbers -- this is the re-checkable version."""
    import json

    from highcast import backtest as bt
    from highcast.metric import get_metric

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, args.metric)
    dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? ORDER BY date",
        (cfg.station.icao,))]
    tokens = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = []
    for tok in tokens:
        if tok not in _ARM_TOKENS:
            raise ValueError(
                f"unknown --arms token: {tok!r} (choose from "
                f"{sorted(_ARM_TOKENS)})")
        spec_arm = _ARM_TOKENS[tok]
        arms.append({"label": tok, **spec_arm})

    result = bt.compare_at_hour(conn, cfg, spec, dates, args.hour, arms,
                                min_train=args.min_train,
                                refit_every=args.refit_every)

    if args.json:
        print(json.dumps(result, indent=2, default=float))
        return

    print(f"=== verify-hour: {spec.key.upper()} @ {cfg.station.icao} — "
          f"hour={args.hour:02d}:00 station-local wall clock "
          f"(tz={cfg.station.tz}) ===")
    print(f"n_matched={result['n_matched']} day(s) across arms: "
          f"{', '.join(a['label'] for a in arms)}\n")
    print(f"{'label':<20}{'n':>6}{'mae':>9}{'crps':>9}{'cov_50':>9}{'cov_90':>9}")
    for arm in arms:
        s = result["arms"][arm["label"]]
        print(f"{arm['label']:<20}{s['n']:>6}{s['mae']:>9.3f}{s['crps']:>9.3f}"
              f"{s['cov_50']:>9.2f}{s['cov_90']:>9.2f}")

    baseline_label = arms[0]["label"]
    print(f"\nvs baseline ({baseline_label}); d* = arm - baseline "
          f"(negative = arm is BETTER):")
    for arm in arms:
        label = arm["label"]
        if label == baseline_label:
            continue
        d = result["vs_baseline"][label]
        lo, hi = d["dmae_ci"]
        significant = lo > 0 or hi < 0
        verdict = "SIGNIFICANT" if significant else "not significant"
        print(f"  {label:<18} dMAE={d['dmae']:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  "
              f"dCRPS={d['dcrps']:+.3f}  95% CI={d['dcrps_ci']}  -> {verdict}")


def cmd_predict(args) -> None:
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from highcast import features
    from highcast.buckets import bucket_probs
    from highcast.metric import get_metric
    from highcast.timeutil import utcnow_iso

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    # TRAP: cmd_predict is also called by auto_cycle via _ns(config=, date=,
    # live=), which sets NO `metric` attribute. Read it defensively so the
    # scheduled high pipeline never crashes with AttributeError.
    spec = get_metric(cfg, getattr(args, "metric", "high"))
    target = args.date or datetime.now(ZoneInfo(cfg.station.tz)).date().isoformat()
    hist_dates = [r["date"] for r in conn.execute(
        "SELECT date FROM cli_reports WHERE station=? AND date < ? ORDER BY date",
        (cfg.station.icao, target))]
    if args.live:
        from highcast.models.baseline import BaselineBlend
        from highcast.models.intraday import IntradayModel, build_checkpoint_frame
        from highcast.models.intraday import CHECKPOINT_HOURS, MIN_MATCHED_ROWS
        from highcast.models.intraday import INTRADAY_FEATURES, features_for
        hist_day = features.build_features(conn, cfg, spec, hist_dates)
        # Must match `build_checkpoint_frame`'s own default: the live path fits the
        # baseline itself (a 1-day frame cannot), so passing a differently-
        # configured blend here than training used would serve the nowcast a
        # `base_p50` on a different scale from the one it was fitted against.
        base = BaselineBlend(
            prefer_corrected=features.uses_bias_correction(spec.key)).fit(hist_day)
        # TRAIN AT THE HOUR WE ARE ABOUT TO PREDICT AT. Production used to train
        # on the fixed CHECKPOINT_HOURS (8..20) and then predict at now.hour,
        # so any hour outside that range had its `hour` feature extrapolated --
        # LightGBM clamps to the nearest split, applying the wrong regime's
        # residual. Measured on a 2025-01-01 holdout (N=569, paired bootstrap),
        # matched-hour training versus the fixed default:
        #     LOW  @3  MAE 1.719 -> 1.016  dMAE -0.703 [-0.811,-0.593]  SIG
        #     LOW  @6  MAE 0.847 -> 0.744  dMAE -0.103 [-0.161,-0.042]  SIG
        #     LOW  @20 MAE 0.340 -> 0.332  dMAE -0.008 ns, dCRPS -0.033 SIG
        #     HIGH @3  MAE 1.609 -> 1.472  dMAE -0.136 [-0.215,-0.061]  SIG
        #     HIGH @9  MAE 1.387 -> 1.397  dMAE +0.010 ns, dCRPS -0.015 ns
        # i.e. a large win where it was extrapolating, a significant win at the
        # LOW's own decision hour (6), and a wash at the HIGH's (9). Reproduce
        # with `scripts/hour_sweep.py`.
        # 2026-07-29 amendment: the matched-hour argument above is about REGIME,
        # not about sample size, and training on one hour also costs ~23/24 of
        # the rows. `intraday_hour_bandwidth` widens the window to {h-k..h+k}
        # while keeping it local, which measured strictly better for the HIGH on
        # all three axes (MAE, bucket hit, and churn). See the constant's own
        # comment in config.py for the sweep. k=0 reproduces the old behaviour
        # exactly, and that is still the default -- KLAX must not move mid-test.
        hour = datetime.now(ZoneInfo(cfg.station.tz)).hour
        # PER-METRIC. KLAX measured SHIP for the LOW and NEUTRAL for the HIGH
        # (scripts/klax_bandwidth_eval.py), so reading the station-wide int here
        # would apply the LOW's answer to the HIGH.
        _m = getattr(cfg, 'model', None)
        k = (_m.intraday_bandwidth_for(spec.key) if _m is not None else 0)
        # Wrap the clock: hour 0 with k=1 needs {23, 0, 1}, not {-1, 0, 1}.
        train_hours = tuple(sorted({(hour + d) % 24 for d in range(-k, k + 1)}))
        train = build_checkpoint_frame(conn, cfg, hist_dates, baseline=base,
                                       metric=spec, hours=train_hours)
        # One row per day at a single hour, so guard the small-sample case: an
        # hour with sparse observational history falls back to the wide frame
        # rather than fitting on almost nothing.
        usable = int(train["y"].notna().sum()) if not train.empty else 0
        if usable < MIN_MATCHED_ROWS:
            print(f"[warn] only {usable} training rows at hours {train_hours}; "
                  f"falling back to hours {CHECKPOINT_HOURS}")
            train = build_checkpoint_frame(conn, cfg, hist_dates, baseline=base,
                                           metric=spec)
        _resid = bool(getattr(getattr(cfg, 'model', None),
                              f'intraday_residual_{spec.key}', False))
        # Per-market feature set. Only the KLAX HIGH carries the warming-rate
        # columns (measured +2.5 pts bucket hit, 97.5% CI [+0.3,+4.9], shipped
        # 2026-07-31); every other market gets exactly INTRADAY_FEATURES, so the
        # three still-frozen markets are unaffected. See `intraday.features_for`.
        _extra = [c for c in features_for(cfg.station.icao, spec.key)
                  if c not in INTRADAY_FEATURES]
        model = IntradayModel(extremum=spec.extremum, residual=_resid,
                              extra_features=_extra).fit(train)
        now_frame = build_checkpoint_frame(
            conn, cfg, [target], hours=(hour,),
            baseline=base, require_target=False, metric=spec)
        if now_frame.empty:
            print("no observations yet for the current climate day")
            return
        qs = model.predict_quantiles(now_frame)[0]
        kind = "intraday"
    else:
        from highcast.models.dayahead import DayAheadModel
        train = features.build_features(conn, cfg, spec, hist_dates)
        # Metric-scoped columns: the trailing-bias features are LOW-only, measured
        # (LOW dMAE -0.091 SIG, HIGH +0.005 ns). See `features.model_features`.
        cols = features.model_features(spec.key)
        if getattr(cfg, "model", None) and cfg.model.hybrid_anchor:
            from highcast.models.dayahead import HybridDayAheadModel
            model = HybridDayAheadModel(cal_mode=cfg.calibration.mode,
                                        feature_cols=cols).fit(train)
        else:
            model = DayAheadModel(cal_mode=cfg.calibration.mode,
                                  feature_cols=cols).fit(train)
        today = features.build_features(conn, cfg, spec, [target],
                                        require_target=False)
        qs = model.predict_quantiles(today)[0]
        # `kind` may be overridden to 'evening'. See EVENING_KIND in this module
        # and docs/superpowers/specs/2026-08-01-evening-dayahead-kind-separation-design.md
        kind = getattr(args, "kind", None) or "dayahead"
    pmf = bucket_probs(qs)
    # The bound-vs-guidance gap, captured because it is ALREADY in the live
    # frame and would otherwise be discarded. `preflight` grades the day on it.
    #
    # ⚠️ WRITTEN AFTER `qs`, AND READ BY NOTHING IN THIS FUNCTION. The model has
    # already predicted by this line; this is a record of a frame column, not an
    # input. Feeding it back into the model was measured and REJECTED
    # (`scripts/klax_low_gap_eval.py`), so its only legitimate use is display.
    bound_gap = None
    if kind == "intraday" and "cur_bound_gap" in now_frame.columns:
        v = now_frame["cur_bound_gap"].iloc[0]
        bound_gap = None if v is None or v != v else float(v)
    db.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target,
        "issued_at_utc": utcnow_iso(), "kind": kind,
        "point_f": qs[0.5], "quantiles": json.dumps(qs),
        "buckets": json.dumps({str(k): v for k, v in pmf.items()}),
        "metric": spec.key, "bound_gap": bound_gap,
    }])
    print(f"{cfg.station.icao} {target} [{kind}]  point={qs[0.5]:.1f}F  "
          f"p10={qs[0.1]:.1f} p90={qs[0.9]:.1f}")
    for t in sorted(pmf):
        if pmf[t] >= 0.01:
            print(f"  {t:3d}F  {pmf[t]*100:5.1f}%  {'#' * int(pmf[t] * 200)}")



def cmd_twc_log(args) -> None:
    """Record one Weather Company hourly reading (the hourly market's truth)."""
    from highcast import twc
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    n = twc.record(conn, cfg, args.valid, args.tempf, source=args.source)
    print(f"recorded {n} TWC reading(s): {args.valid} -> {args.tempf}F")


def cmd_twc_compare(args) -> None:
    """TWC vs our METAR — and how often they would settle a 1F strike apart."""
    from highcast import twc
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    pairs = twc.paired(conn, cfg)
    if not pairs:
        print("no paired hours yet. Log TWC readings with `highcast twc-log "
              "--valid <iso-hour> --tempf <F>` (weather.com/kalshi, Hourly Temps).")
        return
    for col, label in (("metar_precise", "METAR precise (T-group)"),
                       ("metar", "METAR reported (C->F)")):
        r = twc.strike_disagreement(pairs, col)
        print(f"\n=== TWC vs {label}  (n={r['n']}) ===")
        print(f"  bias {r['bias']:+.3f}F   MAE {r['mae']:.3f}F   "
              f"max |diff| {r['max_abs']:.2f}F   exact match {r['exact']:.1%}")
        print(f"  HOURS WHERE A 1F STRIKE WOULD FLIP: "
              f"{r['hours_with_a_flipped_strike']:.1%}  "
              f"({r['total_flipped_strikes']} strikes total)")
    print("\nThe flip rate is the number that matters: on those hours a model "
          "trained on METAR is wrong about a market settled on TWC, however "
          "good the model is.")


def cmd_notify_test(args) -> None:
    """Send a sample of each Discord alert, to prove the webhook works.

    Uses obviously-fake numbers and says TEST in every title, so a message that
    lands here can never be mistaken for a real call.
    """
    from highcast import notify
    cfg = load_config(args.config)
    if not cfg.discord_webhook_url:
        print("No webhook configured. Add this to config/klax.yaml:\n"
              '  discord_webhook_url: "https://discord.com/api/webhooks/..."\n'
              "Get one in Discord: Server Settings -> Integrations -> Webhooks.")
        return
    sample = {"metric": "low", "target_date": "TEST", "bucket": "69–70",
              "p_model": 0.88, "price": 0.66, "stake": 1.0,
              "payout_if_win": 1.52, "placed_manila": "19:04",
              "actual": 69, "won": 1}
    msgs = [
        ("committed", notify.build_message(cfg, sample, {
            "hour_hit": 0.624, "still_future": 0.854,
            "reason": "TEST MESSAGE — no real bet was placed."})),
        ("moved", notify.build_divergence(cfg, sample, "68–69", 0.58)),
        ("settled", notify.build_settlement(cfg, sample)),
    ]
    for name, m in msgs:
        m["embeds"][0]["title"] = "[TEST] " + m["embeds"][0]["title"]
        ok = notify.send(cfg, m)
        print(f"  {name:10} {'sent' if ok else 'FAILED'}")
    print("If nothing arrived, re-copy the webhook URL — a wrong one returns 404 "
          "and `send` reports FAILED rather than raising.")


def cmd_verify_quantization(args) -> None:
    from datetime import date, timedelta
    from highcast import latentmax
    from highcast.metric import get_metric
    from highcast.timeutil import round_half_up

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, args.metric)
    pmf = latentmax.hidden_spike_pmf(conn, cfg, metric=spec.key)
    if spec.extremum == "max":
        print("hidden spike pmf (cli_high - visible_max):")
    else:
        print("hidden spike pmf (cli_low - visible_min):")
    for k, p in pmf.items():
        print(f"  {k:+d}F  {p*100:5.1f}%")
    if spec.extremum == "max":
        # HIGH smell: too much mass BELOW the visible max (CLI high < what we saw).
        neg = sum(p for k, p in pmf.items() if k < -1)
        if neg > 0.02:
            print(f"WARNING: {neg*100:.1f}% of days have CLI high {'<'} visible max "
                  "- suspect climate-day boundary bug or feed misparse.")
    else:
        # LOW smell: the mirror — too much mass ABOVE the visible min
        # (CLI low > what we saw).
        pos = sum(p for k, p in pmf.items() if k > 1)
        if pos > 0.02:
            print(f"WARNING: {pos*100:.1f}% of days have CLI low {'>'} visible min "
                  "- suspect climate-day boundary bug or feed misparse.")
    # spot-check the 1-min claim on a few recent days
    end = date.today() - timedelta(days=3)
    rows = latentmax.fetch_asos1min(cfg, end - timedelta(days=int(args.days)), end)
    print(f"1-min archive rows fetched: {len(rows)} "
          "(store+compare manually if investigating)")


def cmd_serve(args) -> None:
    import uvicorn
    from highcast.web.app import create_app

    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg, config_path=args.config),
                host="127.0.0.1", port=args.port)


def cmd_log(args) -> None:
    import json
    from highcast import scoring

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    kind = None if args.kind == "all" else args.kind
    records = scoring.scorecard(conn, cfg, kind=kind, metric=args.metric)
    summary = scoring.summarize(records)

    if args.json:
        print(json.dumps({"records": records, "summary": summary}, indent=2))
        return

    header = (f"{'date':<12}{'kind':<10}{'bucket':>8}{'b?':>4}{'tail':>7}{'t?':>4}"
              f"{'actual':>8}{'±err':>7}")
    print(header)
    for r in records:
        bmark = "✓" if r["bucket_hit"] else "✗"
        tmark = "✓" if r["tail_hit"] else "✗"
        print(f"{r['target_date']:<12}{r['kind']:<10}{r['pick_label']:>8}{bmark:>4}"
              f"{r['tail_label']:>7}{tmark:>4}{r['actual']:>8}{r['abs_error']:>7.2f}")
    print()
    print("SUMMARY")
    for k, s in summary.items():
        if s["n"] == 0:
            print(f"  {k:<10} n=0")
            continue
        print(f"  {k:<10} n={s['n']} bucket-hit={s['bucket_hit_rate']*100:.0f}% "
              f"tail-hit={s['tail_hit_rate']*100:.0f}% "
              f"point-hit={s['point_hit_rate']*100:.0f}% "
              f"within1={s['within1_rate']*100:.0f}% MAE={s['mae']:.2f}°F")


def cmd_bet(args) -> None:
    """Decision support: gate the latest prediction and, if it clears, size a
    stake vs the live Kalshi market. Also LOGS the evaluation (model probs +
    market prices + decision) as a signal so paper P&L can be tracked. Claude
    dictates the rule + amount; the human places the trade."""
    from datetime import datetime, timezone
    from highcast import signals
    from highcast.metric import get_metric

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, args.metric)
    now = datetime.now(timezone.utc)
    out = signals.record_live(conn, cfg, spec, now)
    if out["pred"] is None:
        print(f"no {args.metric} prediction stored for {out['target']} yet — run "
              f"`highcast predict --metric {args.metric}` (and --live) first")
        return
    gate, market, d = out["gate"], out["market"], out["decision"]
    print(f"=== {spec.key.upper()} bet check — {cfg.station.name} {out['target']} ===")
    if gate["label"]:
        print(f"latest pick: {gate['label']} @ {gate['prob']*100:.0f}%  "
              f"(confident={gate['confident']}, stable={gate['stable']} over "
              f"{gate['n_readings']} reads)")
    print(f"trading window: {'OPEN' if gate['within_window'] else 'CLOSED'} "
          f"(cutoff {gate['cutoff_manila']} Manila)")
    if market is None:
        print("Kalshi market: unavailable — cannot assess edge")
    print()
    print("DECISION:", "PLACE" if d.place else "DO NOT PLACE")
    print("   " + d.reason)
    if d.place and d.sizing and not d.sizing.bankroll_unset:
        sz = d.sizing
        print(f"   stake: {cfg.staking.currency} {sz.stake:.2f} "
              f"({sz.stake_frac*100:.1f}% of bankroll, ~{sz.contracts} contracts,"
              f" {'capped' if sz.capped else 'uncapped'})")
    if out["signal_id"]:
        print(f"   (logged as signal #{out['signal_id']})")
    print()
    print("NOTE: decision support only — not financial advice, not a profit "
          "promise. You place the trade; verify Kalshi eligibility yourself.")


def cmd_settle(args) -> None:
    """Score paper P&L for every logged signal whose day has now settled, and
    grade published Pick of the Day verdicts against the same CLI values.

    Both ledgers settle here, deliberately: `highcast settle` is the command a
    human reaches for to catch up a laptop that was closed, and one that caught
    up only half the record would leave `picks.won` NULL exactly when someone
    went looking for the graded record.

    UNFENCED, unlike the wake cycle's copy. A wake must survive anything; an
    explicit command that fails should say so rather than print a reassuring
    count. `signals` settles first and commits its own work, so a pick-side
    failure cannot cost the signal grading.
    """
    from datetime import datetime, timezone
    from highcast import pick, signals
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    now = datetime.now(timezone.utc)
    n = signals.settle_signals(conn, cfg, now)
    print(f"settled {n} signal(s)")
    print(f"graded {pick.settle(conn, cfg, now)} pick(s)")


def cmd_pnl(args) -> None:
    """Paper track record: win rate + per-contract P&L (edge) per metric."""
    from highcast import signals
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    metrics = ["high", "low"] if args.metric == "all" else [args.metric]
    if getattr(args, "manual", False):
        report, tag = signals.manual_report, "YOUR bets"
    elif getattr(args, "shadow", False):
        report, tag = signals.shadow_report, "SHADOW bet-the-pick"
    else:
        report, tag = signals.paper_report, "paper"
    for m in metrics:
        r = report(conn, cfg, m)
        print(f"=== {m.upper()} {tag} P&L ===")
        if r["n_paper_trades"] == 0:
            print(f"  no settled paper trades yet ({r['pending']} pending)")
            continue
        print(f"  paper trades: {r['n_paper_trades']} (pending {r['pending']})")
        print(f"  win rate:     {r['win_rate']*100:.0f}%  ({r['wins']}/{r['n_paper_trades']})")
        print(f"  avg edge:     {r['avg_edge']*100:+.1f}%")
        roi = r["roi_per_contract"]
        print(f"  P&L/contract: {r['pnl_per_contract']:+.3f}"
              + (f"  (ROI {roi*100:+.1f}% per $ staked)" if roi is not None else ""))
        if r["total_staked"] > 0:
            print(f"  $ staked/P&L: {cfg.staking.currency} {r['total_staked']:.2f} "
                  f"/ {r['total_pnl']:+.2f}")
    print()
    print("NOTE: a paper track record for edge measurement only — past paper "
          "P&L is not a profit promise.")


def cmd_simulate(args) -> None:
    """Log a shadow 'bet the pick' paper bet on today's live market for both
    metrics, settle what's due, and print the shadow track record. This bets the
    model's top bucket ungated (ignores edge/window), so it accrues a running
    'does the forecast win' record separate from the real edge-based ledger."""
    from datetime import datetime, timezone

    from highcast import signals
    from highcast.metric import get_metric
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    now = datetime.now(timezone.utc)
    for m in ("high", "low"):
        sid = signals.record_shadow_live(conn, cfg, get_metric(cfg, m), now)
        print(f"{m}: " + (f"logged shadow bet #{sid}" if sid else
                          "no bet (no market/prediction, or already logged today)"))
    print(f"settled {signals.settle_signals(conn, cfg, now)} signal(s)")
    for m in ("high", "low"):
        r = signals.shadow_report(conn, cfg, m)
        print(f"=== {m.upper()} SHADOW bet-the-pick P&L ===")
        if r["n_paper_trades"] == 0:
            print(f"  no settled shadow bets yet ({r['pending']} pending)")
        else:
            print(f"  {r['n_paper_trades']} bets · win {r['win_rate']*100:.0f}% "
                  f"({r['wins']}/{r['n_paper_trades']}) · "
                  f"P&L/contract {r['pnl_per_contract']:+.3f}")


def cmd_my_bet(args) -> None:
    """Log a bet YOU chose to place, priced off the current live market."""
    from datetime import datetime, timezone

    from highcast import signals
    from highcast.metric import get_metric
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    spec = get_metric(cfg, args.metric)
    now = datetime.now(timezone.utc)
    out = signals.live_decision(conn, cfg, spec, now)
    market = out["market"]
    if not market:
        print("no live market to price this bet")
        return
    b = next((m for m in market if m["label"] == args.bucket), None)
    if b is None:
        print(f"bucket '{args.bucket}' not in the current market. available:")
        for m in market:
            print(f"  {m['label']:>8}  @ {m.get('market_price')}")
        return
    price = args.price if args.price is not None else b["market_price"]
    sid = signals.record_manual(conn, cfg, spec, out["target"], args.bucket,
                                args.contracts, price, market, now)
    print(f"logged your bet #{sid}: {args.contracts} × {args.bucket} @ {price} "
          f"on {out['target']} ({args.metric}). settles when the CLI value lands.")


EXPORT_HEADERS = [
    "prediction(highest temp/lowest temp)",
    "date(date placed)",
    "time(time placed)",
    "bucket(what bucket did i bet on)",
    "bet amount(how much did i bet)",
    "bet winning(the amount i will win if the bet is correct)",
    "did the prediction hit Y/N",
    # appended so the sheet can be audited -- without these a row cannot be
    # checked against the official climate report
    "LA climate day",
    "settled actual (F)",
    "price paid per contract",
    "contracts",
]


def cmd_export_bets(args) -> None:
    """Export your logged bets as CSV with the exact headers Julian tracks.

    Date and time are rendered in `display_tz` (Manila) because that is when the
    bet was actually placed from his chair; the LA climate day is a separate
    column, since a bet placed Sunday evening in Manila belongs to Sunday's LA
    climate day only for the LOW and Saturday's for nothing at all -- keeping
    both avoids the timezone confusion that has bitten this project repeatedly.

    "bet winning" is the PROFIT if the bucket resolves YES (contracts x
    (1 - price)), not the gross payout, since that is the number that matters
    next to "bet amount".
    """
    import csv
    import sys
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    tz = ZoneInfo(cfg.display_tz)
    where = ["station = ?", "kind = ?", "place = 1"]
    params: list = [cfg.station.icao, args.kind]
    if args.metric != "both":
        where.append("metric = ?")
        params.append(args.metric)
    rows = conn.execute(
        f"SELECT * FROM signals WHERE {' AND '.join(where)} ORDER BY created_utc",
        params).fetchall()

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        w = csv.writer(out)
        w.writerow(EXPORT_HEADERS)
        for r in rows:
            placed = datetime.fromisoformat(r["created_utc"]).astimezone(tz)
            price = r["price"]
            n = r["contracts"]
            cost = round(n * price, 2) if (price is not None and n) else None
            win = round(n * (1.0 - price), 2) if (price is not None and n) else None
            if r["won"] is None:
                hit = ""            # not settled yet -- leave blank, not "N"
            else:
                hit = "Y" if r["won"] else "N"
            w.writerow([
                "highest temp" if r["metric"] == "high" else "lowest temp",
                placed.strftime("%Y-%m-%d"),
                placed.strftime("%H:%M"),
                r["bucket"], cost, win, hit,
                r["target_date"],
                r["actual"] if r["actual"] is not None else "",
                price, n,
            ])
    finally:
        if args.out:
            out.close()
            print(f"wrote {len(rows)} bet(s) to {args.out}")


def cmd_edge_report(args) -> None:
    """Do the model's probabilities beat the market's, out-of-sample?"""
    from highcast import signals
    cfg = load_config(args.config)
    conn = db.connect(cfg.db_path)
    metrics = ["high", "low"] if args.metric == "all" else [args.metric]
    for m in metrics:
        r = signals.edge_report(conn, cfg, m)
        print(f"=== {m.upper()} edge report — model probs vs market (Brier, lower is better) ===")
        if r["n"] < 5:
            print(f"  only {r['n']} scored day(s) — need dozens before this means anything")
            continue
        print(f"  N={r['n']}  model {r['model_brier']:.3f}  vs  market {r['market_brier']:.3f}")
        print(f"  diff {r['diff']:+.3f}  95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        print(f"  -> {r['verdict']}")
    print("\nNOTE: needs weeks of settled days to conclude; a positive result is "
          "edge, not a promise. Bootstrap CI must clear 0 to trust it.")


def cmd_paper_account(args) -> None:
    """PAPER DEMO ACCOUNT — one coherent, read-only view of the paper-trading
    record. Seeds the opening balance from the real Kalshi balance (reads
    need credentials only, never the trade-arming flag), replays `app_bets`
    as a fixed-1-contract reconciliation ledger, and separately simulates
    Julian's one-market-a-day accumulation plan with a real compounding
    stake. See `highcast/paper_account.py`. Data layer + CLI only — never
    places an order, never touches arming, never writes to `app_bets`.
    """
    import os

    from highcast import paper_account

    cfgs_list = [load_config(c) for c in args.configs]
    cfgs = {c.station.icao: c for c in cfgs_list}
    conn = db.connect(cfgs_list[0].db_path)
    kwargs = {}
    if args.starting_balance is not None:
        kwargs["fallback_usd"] = args.starting_balance
    variant_hours = tuple(args.variant_hours or ())
    out = paper_account.refresh(conn, cfgs, env=dict(os.environ),
                                variant_hours=variant_hours, **kwargs)
    print(paper_account.render(out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="highcast")
    p.add_argument("--config", default="config/klax.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    bl = sub.add_parser("bot-log", help="the order bot's daily decisions")
    bl.add_argument("--limit", type=int, default=30)
    bl.set_defaults(func=cmd_bot_log)
    sub.add_parser("creds-check",
                   help="is the Kalshi trading setup correct? (prints no key material)"
                   ).set_defaults(func=cmd_creds_check)
    ot = sub.add_parser(
        "order-test",
        help="place a 1c order, read it back, cancel it — proves the path works")
    ot.add_argument("--demo", action="store_true",
                    help="use the demo (fake money) account — start here")
    ot.add_argument("--series", default="KXHIGHMIA")
    ot.add_argument("--ticker", default=None,
                    help="skip market discovery and use this exact ticker")
    ot.add_argument("--price-cents", type=int, default=1,
                    help="limit price in cents (hard-capped; default 1)")
    ot.add_argument("--at-market", action="store_true",
                    help="REAL ORDER: buy the bot's pick at the current ask, "
                         "sized to --dollars. It is meant to fill.")
    ot.add_argument("--dollars", type=float, default=1.00,
                    help="total spend cap for --at-market (hard-capped at 1.00)")
    ot.add_argument("--cancel", action="store_true",
                    help="cancel afterwards even in --at-market mode")
    ot.set_defaults(func=cmd_order_test)
    b = sub.add_parser("backfill")
    b.add_argument("--years", type=int, default=5)
    b.set_defaults(func=cmd_backfill)
    sub.add_parser("reparse-obs").set_defaults(func=cmd_reparse_obs)
    bd = sub.add_parser("board",
                        help="station table: decoded physical state per market")
    bd.add_argument("--metric", choices=("low", "high"),
                    help="one metric only (default: both)")
    bd.add_argument("--configs", nargs="+",
                    help="several stations, e.g. --configs config/klax.yaml "
                         "config/kmia.yaml")
    bd.add_argument("--raw", action="store_true",
                    help="append the source METAR under each block")
    bd.set_defaults(func=cmd_board)
    sub.add_parser("notify-test",
                   help="send a sample Discord alert of each type"
                   ).set_defaults(func=cmd_notify_test)
    tl = sub.add_parser("twc-log", help="record a Weather Company hourly reading")
    tl.add_argument("--valid", required=True, help="ISO hour, e.g. 2026-07-27T02:00Z")
    tl.add_argument("--tempf", type=float, required=True)
    tl.add_argument("--source", default="manual")
    tl.set_defaults(func=cmd_twc_log)
    sub.add_parser("twc-compare",
                   help="TWC vs METAR: bias, MAE and 1F-strike flip rate"
                   ).set_defaults(func=cmd_twc_compare)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    sub.add_parser("wake").set_defaults(func=cmd_wake)
    t = sub.add_parser("backtest")
    t.add_argument("--model", choices=["baseline", "ml"], default="baseline")
    t.add_argument("--metric", choices=["high", "low"], default="high")
    t.add_argument("--refit-every", type=int, default=30)
    t.set_defaults(func=cmd_backtest)
    cb = sub.add_parser("calibration-backtest")
    cb.add_argument("--refit-every", type=int, default=30)
    cb.set_defaults(func=cmd_calibration_backtest)
    pr = sub.add_parser("predict")
    pr.add_argument("--date", default=None)
    pr.add_argument("--live", action="store_true")
    pr.add_argument("--metric", choices=["high", "low"], default="high")
    pr.set_defaults(func=cmd_predict)
    v = sub.add_parser("verify-quantization")
    v.add_argument("--days", type=int, default=30)
    v.add_argument("--metric", choices=["high", "low"], default="high")
    v.set_defaults(func=cmd_verify_quantization)
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8712)
    sv.set_defaults(func=cmd_serve)
    lg = sub.add_parser("log")
    lg.add_argument("--kind", choices=["dayahead", "intraday", "all"], default="all")
    lg.add_argument("--metric", choices=["high", "low"], default="high")
    lg.add_argument("--json", action="store_true")
    lg.set_defaults(func=cmd_log)
    bet = sub.add_parser("bet")
    bet.add_argument("--metric", choices=["high", "low"], default="high")
    bet.set_defaults(func=cmd_bet)
    sub.add_parser("settle").set_defaults(func=cmd_settle)
    pnl = sub.add_parser("pnl")
    pnl.add_argument("--metric", choices=["high", "low", "all"], default="all")
    pnl.add_argument("--shadow", action="store_true",
                     help="show the ungated 'bet the pick' shadow record")
    pnl.add_argument("--manual", action="store_true",
                     help="show YOUR own bet ledger (bets you logged)")
    pnl.set_defaults(func=cmd_pnl)
    sub.add_parser("simulate").set_defaults(func=cmd_simulate)
    mb = sub.add_parser("my-bet")
    mb.add_argument("--metric", choices=["high", "low"], default="high")
    mb.add_argument("--bucket", required=True)
    mb.add_argument("--contracts", type=int, required=True)
    mb.add_argument("--price", type=float, default=None)
    mb.set_defaults(func=cmd_my_bet)

    eb = sub.add_parser("export-bets")
    eb.add_argument("--metric", choices=["high", "low", "both"], default="both")
    eb.add_argument("--kind", choices=["manual", "real", "shadow"],
                    default="manual")
    eb.add_argument("--out", default=None, help="CSV path (default: stdout)")
    eb.set_defaults(func=cmd_export_bets)
    er = sub.add_parser("edge-report")
    er.add_argument("--metric", choices=["high", "low", "all"], default="all")
    er.set_defaults(func=cmd_edge_report)
    vh = sub.add_parser("verify-hour")
    vh.add_argument("--metric", choices=["high", "low"], default="high")
    vh.add_argument("--hour", type=int, required=True)
    vh.add_argument("--arms", default="intraday,dayahead")
    vh.add_argument("--refit-every", type=int, default=30)
    vh.add_argument("--min-train", type=int, default=365)
    vh.add_argument("--json", action="store_true")
    vh.set_defaults(func=cmd_verify_hour)

    pa = sub.add_parser(
        "paper-account",
        help="paper demo account — seeded from the real Kalshi balance, "
             "replays app_bets as a ledger (read-only reporting)")
    pa.add_argument("--configs", nargs="+",
                    default=["config/klax.yaml", "config/kmia.yaml"],
                    help="station configs sharing the account's db "
                         "(default: both KLAX and KMIA)")
    pa.add_argument("--starting-balance", type=float, default=None,
                    help="fallback opening balance in USD, used only when no "
                         "Kalshi credentials are configured (default: "
                         "highcast.paper_account.FALLBACK_STARTING_BALANCE_USD)")
    pa.add_argument("--variant-hours", type=int, nargs="*", default=None,
                    help="opt-in alternate commit hour(s) to compare against "
                         "the selected market's shipped decision hour, e.g. "
                         "--variant-hours 13 (measurement only; never changes "
                         "gating.MIN_DECISION_HOUR or places a second bet)")
    pa.set_defaults(func=cmd_paper_account)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


# Without this, `python -m highcast.cli <cmd>` imports the module, defines main(),
# and exits 0 having done NOTHING — silently. The `highcast` console script
# (pyproject [project.scripts]) always worked, so the gap only showed up for
# anyone following a `python -m` invocation, which then looked like a command
# that ran fine and produced no output.
if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
