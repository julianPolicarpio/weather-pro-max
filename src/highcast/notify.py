"""Discord alerts for the app's committed bets.

A WEBHOOK, NOT A BOT. The alerts are one-way, so a webhook is the whole job: one
HTTPS POST, no token to rotate, no gateway connection, no always-on process to
supervise. A real bot only earns its complexity if Discord needs to send
something back (slash commands), which nothing here needs.

FIVE EVENT TYPES. Four stamp a column on the `app_bets` row they describe;
the fifth (the LOW preview) fires BEFORE any bet row exists, so it stamps the
separate `alert_stamps` table instead (see `notify_low_preview`):

  notified_utc         the bet was committed        -> "here is the call"
  diverged_utc         the model later disagreed    -> "heads up, it moved"
  price_alert_utc      the contract got cheaper     -> "now worth placing"
  settle_notified_utc  the CLI value landed         -> "won / lost"
  alert_stamps(kind='low_preview')  early LOW read, pre-commit -> "here's how it's leaning"

`diverged_utc` and `price_alert_utc` are the two halves of "the situation
changed after we committed": one is the model moving away from the price, the
other is the price moving toward the model. Neither rewrites the bet.

MULTI-STATION. KLAX and KMIA post to the SAME webhook (one Discord channel),
so every title below carries `cfg.station.icao` -- without it, two "LOW ·
2026-07-29 · model moved" messages an hour apart, one for each station, would
be indistinguishable. `build_message` (the commit alert) already had this;
the others did not until this station's alerts existed to expose it.

THE DIVERGENCE ALERT DOES NOT CHANGE THE BET. The committed pick is immutable —
that is the anti-cheat guarantee, and a bet that rewrites itself is not a
prediction. But "the bet is locked" and "you should not be told" are different
claims, and only the first one is true. It fires ONCE, the first time the live
model's top bucket differs from what was committed.

EXACTLY-ONCE IS THE HARD PART. The wake cycle runs every 5 minutes, so the naive
"send when the bet exists" fires ~200 times a day. Two mechanisms prevent that:

  1. `app_bets.notified_utc` — set only after Discord returns success. A bet with
     it set is never re-sent.
  2. Because it is set AFTER the POST, a failed send leaves it NULL and the next
     wake retries. A dropped alert self-heals within 5 minutes instead of being
     lost, and a successful one is never duplicated.

Unconfigured is a silent no-op, matching `synoptic_token`: an optional
integration that is switched off must never look like a failure.
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from highcast import appbet, gating, marketsensor

from highcast.config import AppConfig

TIMEOUT = 10.0

PREVIEW_HOURS: dict[str, dict[str, tuple[int, ...]]] = {
    # STATION -> METRIC -> the local hours to preview at, all BEFORE that
    # metric's commit hour. Station-local, never Manila: the LA/Manila offset is
    # 15h under PDT and 16h under PST, so a Manila literal would drift in
    # November.
    #
    # ⚠️ THIS TABLE IS HOUR-KEYED AND IT DRIFTED ONCE. When the decision hours
    # moved on 2026-07-31 (KLAX low h4->h6, KLAX high h11->h13, KMIA low h5->h7,
    # KMIA high h11->h14) every hour-keyed table had to move with them.
    # HOUR_CALIB and BAND_CALIB were caught because they FAIL LOUDLY when
    # orphaned -- a missing key formats as None and the eval crashes. This one
    # kept working: the alerts still arrived, just at hours nobody had chosen.
    # KMIA low previewed at h3/h4 against an h7 commit, i.e. 4 and 3 hours early
    # instead of 2 and 1, for two days before anyone looked. Found 2026-08-01.
    #
    # TWO KINDS OF ENTRY LIVE HERE, and the difference decides what to do when
    # a commit hour moves again:
    #
    #   RELATIVE -- "the two hours before commit", with no independent reason to
    #     prefer those clock hours. These MUST be re-derived whenever the commit
    #     hour moves. `test_kmia_low_previews_the_two_hours_before_commit` pins
    #     it so the drift cannot repeat silently.
    #
    #   ABSOLUTE -- a specific clock hour chosen because something measurable
    #     happens at it. These do NOT move with the commit hour; the measurement
    #     is about the atmosphere, not about when we choose to act.
    "KLAX": {
        # ABSOLUTE. h2 is where overnight MAE collapses 1.67 -> 1.06, the
        # largest single gain of the night; h3 holds at 0.98. That is a fact
        # about when information arrives, so it did not move when LOW's commit
        # hour went h4 -> h6. These now sit 4 and 3 hours ahead of commit, which
        # is EARLIER WARNING, not a stale value.
        "low": (2, 3),
        # ⚠️ DISABLED 2026-08-01. Was (6, 8) = 21:00/23:00 Manila.
        #
        # Both hours precede the 09:00 day-ahead refresh, so every one of these
        # previews was built on the EVENING forecast -- which measured MAE 4.42F
        # with a -4.05F cold bias against the morning refresh's 1.45F (n=19 vs
        # 14). Julian was being pinged twice a night with a number that ran four
        # degrees cold, and the dashboard computed a 69% "edge" from it.
        #
        # `db.KIND_EVENING` now keeps that object out of `preferred_prediction`
        # entirely, so these hours would preview the INTRADAY instead -- at
        # hours the intraday is weakest, before the max has formed. Neither
        # object is worth an alert at h6/h8.
        #
        # They were also chosen when the HIGH committed at h11; it commits at
        # h13 now and they were never re-measured -- the same staleness found in
        # KMIA low's preview hours on the same day.
        #
        # Re-deriving the right hours is measurement work, not a guess. Removing
        # an alert that ran 4F cold needs no further justification; choosing its
        # replacement does.
        "high": (),
    },
    "KMIA": {
        # RELATIVE. LOW commits h7, so its two prior hours are 5 and 6.
        # Was (3, 4) against the old h5 commit and was NOT updated when the
        # commit hour moved -- see the warning above.
        "low": (5, 6),
        # No HIGH previews.
        #
        # ⚠️ THE REASON RECORDED HERE IS STALE AND THE QUESTION IS NOW OPEN.
        # It used to read: "flat 52-54% from h2 to h10 with 99% still ahead, so
        # an early read carries essentially the same content as the h11 commit".
        # HIGH now commits h14, so that comparison was against a different hour
        # and says nothing about whether h12/h13 are worth previewing.
        #
        # Deliberately NOT guessed at, because this is the market the order bot
        # trades: a preview here would be the only warning before the bot
        # commits, which makes it the one worth measuring rather than assuming.
    },
}


def log_alert(conn, cfg: AppConfig, *, metric: str, target: str, kind: str,
              now: datetime, bucket=None, p_model=None, kalshi_pct=None,
              edge=None, market=None, local_hour=None) -> None:
    """Record an alert that was actually DELIVERED, with the Kalshi ladder as it
    stood at that instant.

    Called only after `send` returns success, so the table is a record of what
    Julian was really told -- not what we intended to tell him.

    `kalshi_pct` is the price on the ALERTED bucket, expressed as a percentage
    because that is how Kalshi shows it and how he reads it. Storing it beside
    `p_model` is what makes the week analysable: a run of "68-69, model 74%"
    alerts says nothing on its own, because 74% is a triumph against a market at
    60 and a losing trade against one at 80.

    NEVER raises. A logging failure must not lose an alert that already went out,
    and must not break the wake cycle that keeps forecasts fresh.
    """
    import json as _json
    try:
        conn.execute(
            """INSERT INTO alert_log (station, metric, target_date, kind, sent_utc,
                 local_hour, bucket, p_model, kalshi_pct, edge, market_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (cfg.station.icao, metric, target, kind, now.isoformat(), local_hour,
             bucket, p_model,
             None if kalshi_pct is None else round(float(kalshi_pct) * 100, 1),
             edge, None if market is None else _json.dumps(market)))
        conn.commit()
    except Exception:  # noqa: BLE001 - see docstring
        pass


def _colour(won, placed_ok: bool) -> int:
    if won is None:
        return 0x4C8DF5 if placed_ok else 0x8A8F98
    return 0x2FA84F if won else 0xC62D2D


# A very large edge is a symptom of OUR staleness, not free money -- the market
# has priced the bucket near zero because it is dead, and we have not caught up.
# See HANDOFF 4.2 and `gating.MAX_EDGE`. Above this, the alert says so instead of
# printing a bare "+50%".
_SUSPECT_EDGE = 0.30


def _alert_is_place(reason: str | None) -> bool:
    """Did the desk actually want this bet?

    ONE DEFINITION, matching `paper_account._is_place`: the reason starts with
    PLACE, and NULL counts as SKIP. Diverging here would let the alert disagree
    with the ledger about what happened on the same row.
    """
    return bool(reason) and reason.startswith("PLACE")


def build_message(cfg: AppConfig, bet: dict, extra: dict | None = None) -> dict:
    """The Discord payload for one `app_bets` row -- COMMITTED or REFUSED.

    ⚠️ A REFUSAL MUST NOT LOOK LIKE A BET. This function used to render both
    identically: same "Committed at HH:MM Manila" title, same Edge field, same
    "Stake -> payout if it wins". Measured on the live `alert_log` 2026-08-10,
    **26 of 44 commit alerts (59%) were refusals** dressed that way -- including
    a KLAX low showing "+43% edge, USD 1.00 -> USD 50.00" against a market
    pricing the bucket at 2c, under a reason that read "too late to place".

    Worse, the refusals were RIGHT: most carry won=1. The desk declined, the
    market was correct, and the alert advertised the decision as a windfall.

    So the payload branches. A refusal gets its own title, a muted colour, the
    reason FIRST, and no stake/payout line -- there is no stake. A genuine
    PLACE is unchanged.

    `bucket right` and `still a forecast` always travel together: accuracy alone
    climbs through the day purely because the answer stops being unknown, so
    quoting it on its own would overstate the call.
    """
    extra = extra or {}
    metric = str(bet["metric"]).upper()
    pct = bet.get("p_model")
    pct_s = f"{pct:.0%}" if isinstance(pct, (int, float)) else "—"
    price = bet.get("price")
    manila = bet.get("placed_manila", "")
    reason = extra.get("reason") or bet.get("reason") or "committed at the decision hour"
    placed = _alert_is_place(reason)

    fields = []
    if not placed:
        # On a refusal the reason IS the message. It used to be the last field,
        # under the numbers that made it look like an opportunity.
        fields.append({"name": "Why no bet", "value": reason[:1000],
                       "inline": False})
    fields += [
        {"name": "Bucket", "value": f"**{bet['bucket']}**", "inline": True},
        {"name": "Model", "value": f"**{pct_s}**", "inline": True},
        # Shown as a PERCENTAGE, not a 0-1 price: that is how Kalshi displays it
        # and how it reads next to the model's own percentage. "0.80" and "74%"
        # side by side invites comparing two different scales.
        {"name": "Kalshi says",
         "value": f"**{price*100:.0f}%**" if isinstance(price, (int, float)) else "—",
         "inline": True},
    ]
    if isinstance(price, (int, float)) and isinstance(pct, (int, float)):
        edge = pct - price
        # A huge edge against a near-dead bucket is this project's
        # best-documented failure mode, not a windfall. Label it.
        note = ("  ⚠️ likely stale, not value"
                if (not placed and edge >= _SUSPECT_EDGE) else "")
        fields.append({"name": "Edge (model − Kalshi)",
                       "value": f"**{edge:+.0%}**{note}", "inline": True})
    hit, ahead = extra.get("hour_hit"), extra.get("still_future")
    if hit is not None:
        fields.append({"name": "Bucket right at this hour",
                       "value": f"{hit:.0%}", "inline": True})
    if ahead is not None:
        fields.append({"name": "Still a forecast",
                       "value": f"{ahead:.0%}", "inline": True})
    # WHERE THE MARKET OUTBIDS US MOST. Shown on EVERY commit alert, placed or
    # refused, with no threshold — see `marketsensor.disagreement` for the
    # measurement and, more importantly, for the three measured reasons this is
    # NOT a veto. It is here so the number is in front of the human at the
    # moment of the decision; on KMIA 2026-08-18 nothing on the desk showed that
    # Kalshi was paying 36% for a rung the model had at 2.6%, and that rung
    # settled.
    dis = extra.get("disagreement")
    # Same rule as the dashboard: if the printed figure rounds to +0% there is
    # no disagreement to report, only a sentence that implies one.
    if (dis and round(dis.get("gap", 0) * 100) >= 1
            and dis["label"] != bet.get("bucket")):
        fields.append({
            "name": "Market outbids us most on",
            "value": (f"**{dis['label']}** — Kalshi {dis['market']:.0%} vs our "
                      f"{dis['model']:.0%}  (gap **{dis['gap']:+.0%}**)"),
            "inline": False})
    if placed:
        # ONLY on a real commit. There is no stake on a refusal, and printing
        # "USD 1.00 -> USD 50.00" beside "do not place" is the single most
        # misleading element this embed ever contained.
        fields.append({
            "name": "Stake → payout if it wins",
            "value": f"{cfg.staking.currency} {bet['stake']:.2f} → "
                     f"{cfg.staking.currency} {bet['payout_if_win']:.2f}",
            "inline": True})
        fields.append({"name": "Reason", "value": reason[:1000], "inline": False})

    title = f"{metric} · {bet['target_date']} · {cfg.station.icao}"
    if not placed:
        title = f"NO BET · {title}"
    return {
        "username": "highcast",
        "embeds": [{
            "title": title,
            "description": (f"Committed at **{manila} Manila**" if placed
                            else f"Desk declined at **{manila} Manila** — "
                                 f"nothing was staked."),
            "color": _colour(bet.get("won"), True) if placed else 0x5B6470,
            "fields": fields,
            "footer": {"text": "Paper record — place the trade yourself." if placed
                       else "Paper record — no position was taken."},
        }],
    }


def send(cfg: AppConfig, payload: dict) -> bool:
    """POST to the webhook. Returns True only on a 2xx.

    Never raises: a notification failure must not break the wake cycle that
    keeps predictions fresh. Returning False leaves `notified_utc` NULL so the
    next wake retries.
    """
    if not cfg.discord_webhook_url:
        return False
    try:
        r = httpx.post(cfg.discord_webhook_url, json=payload, timeout=TIMEOUT)
        return 200 <= r.status_code < 300
    except Exception:  # noqa: BLE001 - network is expected to be flaky
        return False


# How far the COMMITTED bucket's own probability may fall before the alert fires,
# independently of whether some other bucket has overtaken it.
#
# WHY THIS EXISTS -- measured on the logged 2026-07-29 KLAX LOW predictions:
#
#     LA h4  19:03 MNL  P(67-68) 81.2%   <- committed here
#     LA h5  20:00 MNL  P(67-68) 54.7%   <- 26.5 points gone, NOTHING FIRED
#     LA h6  21:03 MNL  P(67-68) 55.9%   <- still nothing
#     LA h7  22:00 MNL  P(67-68) 31.5%   <- >=69 finally ranks first, alert fires
#
# The old rule watched the wrong variable: it asked "has a DIFFERENT bucket taken
# the lead", and 67-68 nominally held the lead at 54.7% because the departing mass
# split across several higher buckets. Two hours of warning were lost while the
# LAX market was still open. Reproduced independently: with these thresholds the
# alert fires at 20:00 MNL instead of 22:00.
#
# ⚠️ THIS IS A BAND-AID AND IS LABELLED AS ONE IN THE MESSAGE. It tells Julian
# sooner that the model has lost confidence. It does NOT stop the model committing
# at 81% to a pick the observation trace did not support -- that bad bet was locked
# at 19:03 and no alert can unwind it. The real fix is a model-side one (advection
# / guidance reliability) and is out of scope while the model is frozen.
DIVERGENCE_DROP = 0.20    # points lost from the probability recorded at commit
DIVERGENCE_FLOOR = 0.60   # or an absolute floor, whichever trips first


def divergence_reason(committed_p: float | None, held_p: float | None,
                      top_label: str, held_label: str) -> str | None:
    """Which divergence rule (if any) has tripped. None means stay silent.

    Two independent triggers, because they catch different failures:
      "overtaken" -- another bucket now ranks first (the original rule)
      "collapsed" -- the committed bucket is still nominally ahead but its own
                     probability has fallen off a cliff (the 2026-07-29 case)
    Returned as text so the message can be worded honestly for each.
    """
    if top_label != held_label:
        return "overtaken"
    if held_p is None:
        return None
    if committed_p is not None and (committed_p - held_p) >= DIVERGENCE_DROP:
        return "collapsed"
    # FLOOR TRIGGER: fell THROUGH the floor, not merely started under it.
    #
    # Caught live on the KMIA LOW 2026-07-30, which committed at 53% -- already
    # below the 60% floor. The old rule (`held_p < FLOOR`) fired "confidence
    # collapsed" the instant the bet existed, on a probability that had not moved
    # at all: the alert read "53% -> 53% (0%)", claiming a collapse while showing
    # a zero-point change. A pick that was ALWAYS weak has not collapsed; the
    # commit alert already said its edge was thin. The floor only means something
    # if confidence was above it to begin with, so require that.
    if (committed_p is not None and committed_p >= DIVERGENCE_FLOOR
            and held_p < DIVERGENCE_FLOOR):
        return "collapsed"
    return None


#: Signal kinds that are not a decision the desk acted on. "shadow" is written
#: alongside the real row and would otherwise win the "nearest at/before the
#: commit" race on a tie.
_NOT_A_DECISION = ("shadow", "manual")


def _kind_at_commit(conn, cfg: AppConfig, bet: dict) -> str | None:
    """Which model actually produced the committed number, from the record.

    Ground truth first: the `signals` row the desk wrote at or just before the
    commit carries the kind it served. Only if none exists (rows predating this
    logging) fall back to the hour rule, which is a PREFERENCE and can differ
    from what was available -- see `divergence_baseline`.
    """
    placed = bet.get("placed_utc")
    if placed:
        row = conn.execute(
            f"""SELECT kind FROM signals
                  WHERE station=? AND metric=? AND target_date=?
                    AND created_utc <= ? AND kind NOT IN
                        ({','.join('?' * len(_NOT_A_DECISION))})
                ORDER BY created_utc DESC LIMIT 1""",
            (cfg.station.icao, bet["metric"], bet["target_date"], placed,
             *_NOT_A_DECISION)).fetchone()
        if row and row["kind"]:
            return row["kind"]
    hour = bet.get("placed_local_hour")
    if hour is None:
        return None
    # A bet is only ever placed inside its window, so `within_window=True` is a
    # fact about the commit, not an assumption about now.
    return gating.kind_at_hour(cfg.station.icao, bet["metric"], True, int(hour))


def divergence_baseline(conn, cfg: AppConfig, bet: dict, kind_now: str | None):
    """What `held_p` should be compared AGAINST, given which model is talking.

    ⚠️ THE COMMITTED PROBABILITY IS NOT ALWAYS A VALID BASELINE. Subtracting a
    day-ahead probability from an intraday one measures the MODEL SWAP, not a
    change of mind, and `divergence_reason`'s two probability triggers cannot
    tell the difference.

    ⚠️⚠️ THE SWAP IS CAUSED BY AVAILABILITY, NOT BY THE HOUR. My first trace of
    this said "a HIGH committed before `MIN_DECISION_HOUR` is priced day-ahead"
    and was WRONG — see error #16 in the handoff. `preferred_prediction` walks
    an ORDER and returns the first kind that HAS a stored prediction, so a bet
    committed before the nowcast has run for that target gets a day-ahead
    number whatever its metric or hour. The only settled bet that has ever
    spanned a switch is a LOW (KLAX 2026-08-09, committed h19 at 45% off the
    day-ahead because no intraday prediction existed until 05:13Z), and the
    hour-based story says LOWs are immune.

    So the committing kind is read from GROUND TRUTH — the `signals` row the
    desk actually wrote at the commit — and `gating.kind_at_hour` is only a
    fallback for rows predating that logging. Reconstructing it from the hour
    is additionally unsafe across time: `MIN_DECISION_HOUR` moved from 11 to 13
    on 2026-07-31, so the same hour means different things before and after.

    THE CHOICE, made deliberately: keep reading the CURRENTLY PREFERRED model —
    after the knee it is the measured-sharper one and silencing it would throw
    that away — but RE-BASELINE to that model's own first reading on the
    committed bucket. The question stays "has the best available model lost
    faith?" and stops being "do two different forecasters disagree?".

    The baseline is recovered from the stored `signals` ladder rather than kept
    in a new column: the desk already writes one row per wake with the full
    per-rung pmf, so the number is on disk and no migration is needed.

    ⚠️ SCOPE, honestly: this has never yet changed an alert. On the one real
    switch the alert fired "overtaken", which compares top against held INSIDE
    ONE LADDER and was never cross-model. This is a latent-bug guard, not a
    fix for observed damage — do not cite it as one.

    Returns `(baseline_p, switched)`. `switched` is False when the committing
    model is still the one talking, in which case the baseline is exactly
    `bet['p_model']` and behaviour is unchanged.
    """
    committed = bet.get("p_model")
    if kind_now is None:
        return committed, False
    kind_commit = _kind_at_commit(conn, cfg, bet)
    if kind_commit is None or kind_commit == kind_now:
        return committed, False
    row = conn.execute(
        """SELECT market_json FROM signals
             WHERE station=? AND metric=? AND target_date=? AND kind=?
               AND market_json IS NOT NULL AND market_json != ''
               AND created_utc >= ?
           ORDER BY created_utc LIMIT 1""",
        (cfg.station.icao, bet["metric"], bet["target_date"], kind_now,
         bet.get("placed_utc") or "")).fetchone()
    if row:
        try:
            rung = next((m for m in json.loads(row["market_json"])
                         if m.get("label") == bet["bucket"]), None)
        except (ValueError, TypeError):
            rung = None
        if rung and isinstance(rung.get("p_model"), (int, float)):
            return float(rung["p_model"]), True
    # No stored reading of the new model yet. Returning None suppresses BOTH
    # probability triggers for this cycle (see `divergence_reason`) rather than
    # falling back to the cross-model subtraction this function exists to stop.
    # "overtaken" is unaffected -- it compares top against held inside ONE
    # ladder, so it was never cross-model and still fires.
    return None, True


#: Our bucket is worth this or less -> nothing left to exit. Descriptive, not a
#: trigger: measured at 1c median on the days where the market had run away.
_GONE = 0.10
#: The market leader is this or more -> the market has genuinely made up its mind.
_DECIDED = 0.60


def market_position(bet_bucket: str, market: list[dict] | None) -> dict | None:
    """Where the MARKET stands on our bucket, and can we still get out?

    Returns None when there is no priced ladder to read. Never raises: this
    decorates an alert, and a decoration must not be able to suppress it.
    """
    priced = [m for m in (market or [])
              if isinstance(m.get("market_price"), (int, float))]
    if not priced:
        return None
    leader = max(priced, key=lambda m: m["market_price"])
    ours = next((m for m in priced if m["label"] == bet_bucket), None)
    ours_price = ours["market_price"] if ours else None
    leads = leader["label"] == bet_bucket
    if leads:
        verdict = ("The market still has **your** bucket ahead — on this one "
                   "the model is the outlier, not the news.")
    elif ours_price is not None and ours_price <= _GONE \
            and leader["market_price"] >= 0.90:
        verdict = (f"⚠️ Your bucket is down to **{ours_price*100:.0f}%** — the "
                   f"position is effectively gone. This is a post-mortem, not "
                   f"a warning.")
    elif leader["market_price"] >= _DECIDED and (ours_price or 0) > _GONE:
        verdict = (f"⚠️ The market has moved against you and your bucket is "
                   f"still trading at **{ours_price*100:.0f}%** — this is the "
                   f"window in which exiting is still worth something.")
    else:
        verdict = "The market has not made up its mind either."
    return {"leader": leader["label"], "leader_price": leader["market_price"],
            "ours_price": ours_price, "leads": leads, "verdict": verdict}


def build_divergence(cfg: AppConfig, bet: dict, now_bucket: str,
                     now_p: float | None, held_price: float | None = None,
                     new_price: float | None = None,
                     reason: str = "overtaken",
                     held_p: float | None = None,
                     market: list[dict] | None = None,
                     baseline_p: float | None = None,
                     baseline_kind: str | None = None) -> dict:
    """"The model has moved off its committed bet." Informational only.

    Worded carefully: the bet has NOT changed and cannot. This exists because the
    alternative — silently letting the committed pick drift toward whatever the
    model currently believes — is precisely the hindsight this ledger is built to
    prevent. So the pick stays put and the disagreement is surfaced instead.

    `reason` selects the wording. "collapsed" must NOT say "the model now favours
    <bucket>" when that bucket IS the committed one — the message would read as
    agreement while reporting a 26-point loss of confidence.

    ⚠️ `market` (the full ladder) drives the MARKET POSITION line, which is the
    half of this alert that actually carries information — see
    `scripts/divergence_alert_eval.py`, run 2026-08-16 over 68 settled
    day-markets:

      * The trigger itself is sound and was left alone: it fired on 32 of 68
        days, and P(bet lost) was 50% when it fired against 8% when it stayed
        quiet (Fisher p = 2.8e-4, the single pre-specified comparison).
      * Requiring the disagreement to PERSIST does essentially nothing. N = 1,
        2, 3, 4, 6 consecutive snapshots give 30, 30, 30, 30, 29 fires with 16
        losers throughout. Divergence, once it starts, does not reverse.
      * Suppressing the alert once the market has concentrated does not help
        and starts hurting below 0.90: separation goes +45% (no filter) to
        +47% at < 0.95, then +38% at < 0.90 and +25% at < 0.85 (p 0.06).
      * By the time the market is >= 0.95 on someone else's bucket, ours is at
        1c in 14 of 14 cases. Perfect separation, no salvage: a death
        certificate, not a warning.

    ⚠️ Numbers corrected 2026-08-16 after error #16 — the first run filtered
    replay snapshots by a RECONSTRUCTED model kind and dropped a third of them.
    The live column (`diverged_utc`) was never affected and is unchanged.

    So the trigger keeps firing exactly as before and the MESSAGE says where the
    market stands and whether the position can still be exited. No threshold was
    fitted here — every number below is descriptive. The one rule that separates
    WITH salvage value (market >= 0.60 against us while ours still trades >= 20c:
    15 fires, 11 losers, 73% vs 15%, p = 3.8e-5, surviving x40 Bonferroni at
    0.0015) is reported in the message and is deliberately NOT a trigger yet —
    it is still a cell chosen from a sweep, and this project has killed four
    ideas that looked this good on ~60 days.
    """
    pct = f"{now_p:.0%}" if isinstance(now_p, (int, float)) else "—"
    # Kalshi's own percentage on BOTH buckets. "The model moved" is not actionable
    # without it: if the market still has the committed bucket ahead, the model is
    # the outlier rather than the news -- which is exactly what happened on
    # 2026-07-28, when the nowcast swung to >=70 at 77% while Kalshi held 68-69 at
    # 83%, and Kalshi was right.
    held_pct = f"{held_price*100:.0f}%" if isinstance(held_price, (int, float)) else "—"
    new_pct = f"{new_price*100:.0f}%" if isinstance(new_price, (int, float)) else "—"
    collapsed = reason == "collapsed"
    committed_p = bet.get("p_model")
    # ⚠️ MEASURE THE DROP FROM THE BASELINE THE TRIGGER USED, not from the
    # committed number. When the desk swapped models at the knee those differ,
    # and quoting the committed one would print a drop that never happened —
    # which is exactly the false "81% → 53%" shape this whole fix exists to
    # stop. `baseline_p` is None whenever no swap occurred, so the ordinary
    # case still shows the number Julian saw at commit.
    base = baseline_p if isinstance(baseline_p, (int, float)) else committed_p
    if collapsed:
        was = f"{base:.0%}" if isinstance(base, (int, float)) else "—"
        nowp = f"{held_p:.0%}" if isinstance(held_p, (int, float)) else "—"
        drop = (f"{(base - held_p):.0%}"
                if isinstance(base, (int, float))
                and isinstance(held_p, (int, float)) else "—")
        title_suffix = "confidence collapsed"
        description = (
            "Your committed bet is **unchanged** — this is a heads-up that the "
            "model has lost confidence in it. The bucket is still nominally "
            "ahead, so nothing has overtaken it; the probability itself has "
            "fallen away.")
        second = {"name": "Confidence on your bucket",
                  "value": f"**{was} → {nowp}**  ({drop})", "inline": True}
    else:
        title_suffix = "model moved"
        description = ("Your committed bet is **unchanged** — this is only a "
                       "heads-up that the live model now favours a different "
                       "bucket.")
        second = {"name": "Model now favours", "value": f"**{now_bucket}** ({pct})",
                  "inline": True}
    # The market position, appended AFTER the model fields on purpose: the
    # measurement says this is the informative half, and it should be the last
    # thing read. `now_bucket` is the MODEL's top bucket and can differ from the
    # market's leader -- that difference is the whole point of showing both.
    pos = market_position(bet["bucket"], market)
    extra = []
    # Disclose the swap. The alert is now measuring a DIFFERENT model's change
    # of mind than the one that made the pick, and hiding that would make the
    # quoted percentages irreconcilable with the commit message Julian still
    # has in the channel.
    if isinstance(baseline_p, (int, float)) and baseline_kind:
        label = {"intraday": "nowcast", "dayahead": "day-ahead"}.get(
            baseline_kind, baseline_kind)
        committed_pct = (f"{committed_p:.0%}"
                         if isinstance(committed_p, (int, float)) else "—")
        extra.append({"name": "Which model is speaking", "value":
                      (f"Committed at **{committed_pct}** by the day-ahead "
                       f"model. The **{label}** took over at the decision hour "
                       f"and is the sharper model from then on, so the change "
                       f"above is measured from ITS first reading "
                       f"(**{baseline_p:.0%}**) — not from the committed "
                       f"number, which a different model produced."),
                      "inline": False})
    if pos:
        ours = (f"{pos['ours_price']*100:.0f}%"
                if pos["ours_price"] is not None else "—")
        extra.append({"name": "Market position", "value":
                      (f"leader **{pos['leader']}** "
                       f"{pos['leader_price']*100:.0f}%  ·  "
                       f"yours {bet['bucket']} {ours}\n{pos['verdict']}"),
                      "inline": False})
    return {"username": "highcast", "embeds": [{
        "title": f"{str(bet['metric']).upper()} · {cfg.station.icao} · "
                 f"{bet['target_date']} · {title_suffix}",
        "description": description,
        "color": 0xE0A030,
        "fields": [
            {"name": "Committed bet", "value": f"**{bet['bucket']}**", "inline": True},
            second,
            # The Kalshi price on both buckets, because "the model moved" is not
            # actionable without knowing whether the MARKET moved with it. If
            # Kalshi still has the committed bucket ahead, the model is the
            # outlier, not the news.
            {"name": "Kalshi says",
             "value": (f"{bet['bucket']} {held_pct}  ·  {now_bucket} {new_pct}"),
             "inline": False},
        ] + extra,
        "footer": {"text": ("The bet is locked by design. This is an early "
                            "warning, not a fix — the pick was committed before "
                            "this was knowable.") if collapsed else
                   "The bet is locked by design. No action required."},
    }]}


def build_price_drop(cfg: AppConfig, bet: dict, price: float, p_model: float,
                     edge: float) -> dict:
    """"The contract got cheap enough to be worth buying." Informational only.

    The complement of the divergence alert. That one fires when the MODEL moves
    off the committed bucket; this one fires when the same bucket is still the
    call but the MARKET has come to you. Both leave the committed bet untouched.

    Why this exists: 68-69 committed at a price of 0.80 on 2026-07-28. The pick
    was sound -- model 74% -- but at 80c you stake $1 to win $1.25, and the edge
    gate (-6%) correctly said do not place. Nothing about that verdict is
    permanent: it is a statement about the PRICE, and the price moves all day.
    Without this alert the only way to catch the price coming down is to sit
    watching the dashboard, which is exactly what the alerts exist to avoid.
    """
    return {"username": "highcast", "embeds": [{
        "title": f"{str(bet['metric']).upper()} · {cfg.station.icao} · "
                 f"{bet['target_date']} · price dropped — now worth placing",
        "description": f"**{bet['bucket']}** is cheaper than when the app "
                       f"committed, and the edge gate now passes.",
        "color": 0x2FA84F,
        "fields": [
            {"name": "Bucket", "value": f"**{bet['bucket']}**", "inline": True},
            {"name": "Price", "value": f"{bet['price']:.2f} → **{price:.2f}**",
             "inline": True},
            {"name": "Model", "value": f"**{p_model:.0%}**", "inline": True},
            {"name": "Edge now", "value": f"**{edge:+.0%}** "
                                          f"(needs {cfg.staking.min_edge:+.0%})",
             "inline": True},
            {"name": "Pays per 1.00 staked", "value": f"**{1.0 / price:.2f}** "
                                                      f"(was {1.0 / bet['price']:.2f})",
             "inline": True},
        ],
        "footer": {"text": "The committed bet is unchanged. Place it yourself "
                           "if you want it."},
    }]}


def build_settlement(cfg: AppConfig, bet: dict) -> dict:
    """Win/lose, once the NWS climate report lands.

    No P&L, by request. The stake is a flat notional $1 that exists only so the
    ledger has a consistent unit -- reporting a dollar figure on it would invite
    reading paper pennies as performance. The question this alert answers is
    "was the call right", and the win/lose plus the actual temperature answer it
    completely. The dashboard still totals P&L if it is ever wanted.
    """
    won = bet.get("won")
    return {"username": "highcast", "embeds": [{
        "title": f"{str(bet['metric']).upper()} · {cfg.station.icao} · "
                 f"{bet['target_date']} · {'WON' if won else 'LOST'}",
        "description": f"Settled at **{bet['actual']}°F**",
        "color": _colour(won, True),
        "fields": [
            {"name": "Bet", "value": f"{bet['bucket']}", "inline": True},
            {"name": "Actual", "value": f"**{bet['actual']}°F**", "inline": True},
        ],
        "footer": {"text": "Settled on the NWS Climatological Report."},
    }]}


def _stamp(conn, cfg, bet, column, now):
    conn.execute(
        f"""UPDATE app_bets SET {column}=? WHERE station=? AND metric=?
            AND target_date=?""",
        (now.isoformat(), cfg.station.icao, bet["metric"], bet["target_date"]))
    conn.commit()


def notify_settled(conn, cfg: AppConfig, now: datetime) -> int:
    """Announce every settled bet that has not been reported yet."""
    if not cfg.discord_webhook_url:
        return 0
    rows = conn.execute(
        """SELECT * FROM app_bets WHERE station=? AND won IS NOT NULL
             AND settle_notified_utc IS NULL""", (cfg.station.icao,)).fetchall()
    sent = 0
    for r in rows:
        bet = dict(r)
        if send(cfg, build_settlement(cfg, bet)):
            _stamp(conn, cfg, bet, "settle_notified_utc", now)
            log_alert(conn, cfg, metric=bet["metric"], target=bet["target_date"],
                      kind="settled", now=now, bucket=bet["bucket"],
                      p_model=bet.get("p_model"), kalshi_pct=bet.get("price"))
            sent += 1
    return sent


def notify_divergence(conn, cfg: AppConfig, spec, now: datetime) -> int:
    """Fire once if the live model has moved off today's committed bet."""
    if not cfg.discord_webhook_url:
        return 0
    from highcast import signals
    from zoneinfo import ZoneInfo as _Z

    target = now.astimezone(_Z(cfg.station.tz)).date().isoformat()
    row = conn.execute(
        """SELECT * FROM app_bets WHERE station=? AND metric=? AND target_date=?
             AND diverged_utc IS NULL AND won IS NULL""",
        (cfg.station.icao, spec.key, target)).fetchone()
    if row is None:
        return 0
    bet = dict(row)
    out = signals.live_decision(conn, cfg, spec, now)
    # THE TWO MUST BE ABOUT THE SAME DAY. Once a metric's window closes the desk
    # rolls to TOMORROW, so `live_decision` starts returning a different target
    # with its own strike ladder. Comparing that against today's locked bet fired
    # a bogus "model moved to 66–67" on 2026-07-27: the bet was Jul 27 (69–70),
    # the quote was Jul 28. Different days, different markets, no disagreement.
    if out.get("target") != bet["target_date"]:
        return 0
    market = out.get("market")
    if not market:
        return 0
    top = max(market, key=lambda r: (r.get("p_model") or 0.0))
    held = next((m for m in market if m["label"] == bet["bucket"]), None)
    # Two triggers, not one: a different bucket taking the lead OR the committed
    # bucket's own confidence collapsing while it nominally still leads. The
    # second is the 2026-07-29 case that cost two hours of warning -- see
    # DIVERGENCE_DROP.
    # ⚠️ NOT `bet['p_model']` -- that can be a DAY-AHEAD number while `held` is
    # the intraday nowcast. `divergence_baseline` re-baselines across the model
    # switch; see its docstring for the measurement and the false alert it fixes.
    baseline, switched = divergence_baseline(conn, cfg, bet, out.get("kind"))
    reason = divergence_reason(baseline, (held or {}).get("p_model"),
                               top["label"], bet["bucket"])
    if reason is None:
        return 0
    if send(cfg, build_divergence(cfg, bet, top["label"], top.get("p_model"),
                                  held_price=(held or {}).get("market_price"),
                                  new_price=top.get("market_price"),
                                  reason=reason,
                                  held_p=(held or {}).get("p_model"),
                                  market=market,
                                  baseline_p=baseline if switched else None,
                                  baseline_kind=out.get("kind") if switched
                                  else None)):
        _stamp(conn, cfg, bet, "diverged_utc", now)
        log_alert(conn, cfg, metric=spec.key, target=target, kind="divergence",
                  now=now, bucket=top["label"], p_model=top.get("p_model"),
                  kalshi_pct=top.get("market_price"), market=market,
                  local_hour=now.astimezone(ZoneInfo(cfg.station.tz)).hour)
        return 1
    return 0


def build_low_preview(cfg: AppConfig, target: str, local_hour: int,
                      manila: str, quote: dict, edge: float | None,
                      hit: float | None, ahead: float | None,
                      metric: str = "low") -> dict:
    """A metric's read at an hour BEFORE it commits. A preview, never a call.

    Measured on 1955 days, the LOW's overnight curve looks like this (LA hour):

        01  MAE 1.67  bucket 37%   98% still ahead
        02  MAE 1.05  bucket 56%   96%          <- 17:00 Manila
        03  MAE 0.98  bucket 58%   92%          <- 18:00 Manila
        04  MAE 0.89  bucket 62%   85%          <- 19:00 Manila, commits

    Hour 1 -> 2 is the largest single gain of the night, which is what makes 17:00
    worth a ping at all. But 56% is not 62%, and the gap is the whole reason the
    commit sits at hour 4 -- so the wording here has to stay a preview. The bet is
    NOT placed early and nothing about this message moves it.

    `bucket right` and `still a forecast` travel together for the same reason they
    do in `build_message`: accuracy alone climbs through the night purely because
    the answer stops being unknown.
    """
    from highcast import gating

    # Read from the gate, never duplicated -- and for the RIGHT metric: a HIGH
    # preview quoting the LOW's commit hour would send Julian to the wrong clock.
    commit_hour = gating.min_decision_hour(cfg.station.icao, metric)
    price, p = quote.get("market_price"), quote.get("p_model")
    fields = [
        {"name": "Bucket", "value": f"**{quote['label']}**", "inline": True},
        {"name": "Model", "value": f"**{p:.0%}**" if p is not None else "—",
         "inline": True},
        {"name": "Market price",
         "value": f"{price:.2f}" if price is not None else "—", "inline": True},
    ]
    if edge is not None:
        verdict = "worth placing" if edge >= cfg.staking.min_edge else "no edge"
        fields.append({"name": "Edge",
                       "value": f"**{edge:+.0%}** — {verdict}", "inline": True})
    if hit is not None:
        fields.append({"name": "Bucket right at this hour",
                       "value": f"{hit:.0%}", "inline": True})
    if ahead is not None:
        fields.append({"name": "Still a forecast", "value": f"{ahead:.0%}",
                       "inline": True})
    return {"username": "highcast", "embeds": [{
        "title": f"{metric.upper()} · {cfg.station.icao} · {target} · {manila} Manila preview",
        "description": f"Early read — the app has **not** committed yet. "
                       f"It commits at "
                       f"{_manila_label(cfg, commit_hour)} Manila.",
        "color": 0x5B6470,
        "fields": fields,
        "footer": {"text": "Preview only. Nothing is locked in until the "
                           "commit hour."},
    }]}


def build_unstable(cfg: AppConfig, target: str, events: list[str],
                   storm: bool) -> dict:
    """"Outside factors are in play — don't bother with this one."

    A WARNING, NOT A GATE, and that distinction is the whole design. We cannot
    yet justify skipping automatically: the hard days still hit 72%, and skipping
    them is only correct if the market has NOT already discounted them — a claim
    that needs price history we do not have (9 priced days). A warning costs
    nothing and cannot lose money. Julian decides; the app just tells him what it
    can see and he cannot.

    HEAT AND STORM ARE NOT EQUALLY DANGEROUS, so they do not get one message.

      HEAT   the day-ahead model carries a `heat_alert` feature, so a heat
             advisory is PARTLY priced in already. Worth knowing, not alarming.
      STORM  the model has NO precipitation input of any kind. It cannot
             represent convection, so a thunderstorm is not a number it gets
             wrong — it is a mechanism it cannot see. On 2026-08-01 that took
             KMIA from 87.98F to 75.02F in 50 minutes and destroyed a `>=81`
             low that had looked settled all day.

    So a storm warning is a much stronger signal than a heat one, and the message
    says which it is rather than averaging them into a single "unstable".
    """
    kind = "STORM" if storm else "HEAT"
    colour = 0xC0392B if storm else 0xE08B34
    if storm:
        why = ("The temperature model has **no precipitation input at all** — "
               "it cannot see convection, so this is not a number it gets "
               "slightly wrong, it is a mechanism it is blind to. A storm cut "
               "KMIA 16°F in 50 minutes on 2026-08-01 and destroyed a low that "
               "had looked settled all day.")
        advice = "**Don't bother trading this one.** Sit it out."
    else:
        why = ("The day-ahead model does carry a heat-alert feature, so this is "
               "**partly** accounted for already — it is context, not an "
               "emergency.")
        advice = ("Trade it if the read is otherwise strong, but treat a tight "
                  "call as less trustworthy than usual.")
    return {"username": "highcast", "embeds": [{
        "title": f"⚠️ {kind} · {cfg.station.icao} · {target} · unstable conditions",
        "description": f"{advice}\n\n{why}",
        "color": colour,
        "fields": [
            {"name": "Active advisory", "value": "\n".join(f"• {e}" for e in events)
             or "—", "inline": False},
        ],
        "footer": {"text": "Warning only — nothing was blocked or placed. "
                           "The call is yours."},
    }]}


def notify_unstable(conn, cfg: AppConfig, now: datetime) -> int:
    """Warn once per day per advisory kind when outside factors are in play.

    Keyed on (target date, STORM|HEAT) rather than on the individual product, so
    three consecutive Heat Advisories in one afternoon send ONE message. The
    5-minute wake cycle would otherwise send dozens — the same exactly-once
    problem `notified_utc` solves for bets.

    Reads the `alerts` table, which the wake cycle already populates. NOTE the
    storm half only works PROSPECTIVELY: `fetchers/alerts.parse_archive` maps
    VTEC codes through `_VTEC_EVENT`, which contains heat phenomena only, and the
    archive fetch requests only `HEAT_PHENOMENA` — so no thunderstorm history can
    be backfilled and this warning has no historical record to be validated
    against yet.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import db as _db
    from highcast.fetchers.alerts import STORM_EVENTS
    from highcast.timeutil import climate_day_bounds

    target_d = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    target = target_d.isoformat()
    # The station's own climate day, not a calendar day: KMIA runs 05:00Z->05:00Z
    # and KLAX 08:00Z->08:00Z, so a calendar window would attribute an evening
    # advisory to the wrong trading day.
    start, end = climate_day_bounds(target_d, cfg.station.std_utc_offset)
    rows = conn.execute(
        """SELECT DISTINCT event FROM alerts
           WHERE station=? AND onset_utc < ?
             AND (expires_utc IS NULL OR expires_utc > ?)""",
        (cfg.station.icao, end.isoformat(), start.isoformat())).fetchall()
    events = sorted({r["event"] for r in rows})

    # ── STORM: OUR OWN OBSERVATIONS, NOT `alerts` (rewired 2026-08-28) ───────
    # The `alerts` table has NEVER held a storm product for either station, so
    # `alert_stamps WHERE kind='unstable' AND key LIKE '%storm%'` was 0 rows —
    # a guard that could not fire, exactly like errors #8 and #11. `wx_thunder`
    # covers 696 KMIA days and, measured over 2,430 climate days, the low breaks
    # >=2F below what was visible on 28.6% of days after thunder onset against
    # 5.3% of quiet days. See `highcast.storm`.
    #
    # The heat half still reads `alerts`, which is the right source for it.
    from highcast import storm as _storm
    live = _storm.active(conn, cfg, now)
    storm_events = [e for e in events if e in STORM_EVENTS]
    if live:
        wx = str(live["wx"]).strip()
        # Lead with the FALL, not the phenomenon. "convection" is a label; "the
        # temperature has already dropped 11.9F" is the thing that moves a
        # bucket, and on 2026-08-24 that fall was six rungs wide.
        swing = live.get("temp_swing_f")
        label = (f"Observed convection ({wx}) — temperature already down "
                 f"{swing:.1f}°F" if isinstance(swing, (int, float)) and swing >= 1
                 else f"Observed convection ({wx})")
        storm_events = sorted({*storm_events, label})
    if not events and not storm_events:
        return 0

    sent = 0
    for storm in (True, False):
        group = (storm_events if storm
                 else [e for e in events if e not in STORM_EVENTS])
        if not group:
            continue
        key = f"{target}:{'storm' if storm else 'heat'}"
        if conn.execute("SELECT 1 FROM alert_stamps WHERE station=? AND kind=? "
                        "AND key=?", (cfg.station.icao, "unstable", key)).fetchone():
            continue
        if not send(cfg, build_unstable(cfg, target, group, storm)):
            continue          # failed send is retried next wake, never stamped
        _db.upsert_ignore(conn, "alert_stamps", [{
            "station": cfg.station.icao, "kind": "unstable", "key": key,
            "sent_utc": now.isoformat()}])
        sent += 1
    return sent


def build_missed_commit(cfg: AppConfig, m: dict) -> dict:
    """"The commit hour passed and no bet was recorded — you can still place it."

    Deliberately terse and imperative. This fires while the window is open, so
    the only thing that matters is what market, what deadline, and that the app
    did not do it.
    """
    closes = (m["window_closes"] or "")[11:16]
    tz = ZoneInfo(cfg.display_tz)
    local = ""
    if m["window_closes"]:
        local = (datetime.fromisoformat(m["window_closes"]).astimezone(tz)
                 .strftime("%H:%M"))
    return {"embeds": [{
        "title": f"⚠️ MISSED COMMIT — {m['station']} {m['metric'].upper()}",
        "description": (
            f"The **{m['commit_hour']}:00** commit for **{m['target_date']}** "
            f"passed with no bet recorded. The wake cycle was not running.\n\n"
            f"**The window is still open** — closes {closes}Z"
            + (f" ({local} {cfg.display_tz.split('/')[-1]})" if local else "")
            + ". You can still place this by hand."),
        "color": 0xE8A33D,
        "footer": {"text": "No bet exists for this market today. The app cannot "
                           "backfill one — app_bets has no late path by design."},
    }]}


def notify_missed_commit(conn, cfg: AppConfig, now: datetime) -> int:
    """Alert once per missing commit that is STILL PLACEABLE.

    WHY ONLY THE ACTIONABLE ONES. The 2026-08-08/09 outage cost four commits
    and nothing said so. Three of the four were already past their tradeable
    window by the time any message could be read — pushing those to Julian's
    phone at 1am Manila is noise he cannot act on, and noise is how a real
    alert gets muted. The wake-cycle log records every miss; Discord is
    reserved for the case he can still do something about.

    EXACTLY ONCE per (station, metric, target_date) via `alert_stamps`, the
    same pattern `notify_unstable` uses: the wake cycle fires every 5 minutes
    and an overdue commit stays overdue for hours.

    A FAILED SEND IS NEVER STAMPED, so it retries next wake — stamping before a
    confirmed send would turn one network blip into a permanently lost alert.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import db as _db

    sent = 0
    for m in appbet.missing_commits(conn, {cfg.station.icao: cfg}, now,
                                    lookback_days=0):
        if not m["still_actionable"]:
            continue
        key = f"{m['target_date']}:{m['metric']}"
        if conn.execute("SELECT 1 FROM alert_stamps WHERE station=? AND kind=? "
                        "AND key=?",
                        (cfg.station.icao, "missed_commit", key)).fetchone():
            continue
        if not send(cfg, build_missed_commit(cfg, m)):
            continue          # failed send is retried next wake, never stamped
        _db.upsert_ignore(conn, "alert_stamps", [{
            "station": cfg.station.icao, "kind": "missed_commit", "key": key,
            "sent_utc": now.isoformat()}])
        sent += 1
    return sent


def _manila_label(cfg: AppConfig, local_hour: int) -> str:
    """Render an LA-local hour as its Manila clock time.

    Computed, not hardcoded: the LA/Manila offset is 15h under PDT and 16h under
    PST, so a literal "19:00" would silently drift by an hour every November.
    """
    from datetime import date, time

    la = ZoneInfo(cfg.station.tz)
    stamp = datetime.combine(date.today(), time(hour=local_hour), tzinfo=la)
    return stamp.astimezone(ZoneInfo(cfg.display_tz)).strftime("%H:%M")


def notify_low_preview(conn, cfg: AppConfig, spec, now: datetime) -> int:
    """Ping the LOW's current read at each of PREVIEW_HOURS, once per hour.

    Only the hours BEFORE the commit are previewed. The commit hour itself is
    deliberately left out: `notify_pending` already announces the bet the moment
    it is written, and firing both would send two messages about the same hour.
    So the sequence Julian sees is 17:00 preview, 18:00 preview, 19:00 the real
    call -- three pings, no duplicates.

    Skips silently when there is no market yet, and stamps `alert_stamps` so the
    5-minute wake sends one message per hour rather than twelve.

    Station-scoped via `PREVIEW_HOURS[cfg.station.icao]`, with NO fallback: a
    station absent from that dict (anything that isn't KLAX or KMIA today)
    sends no previews at all rather than borrowing KLAX's 17:00/18:00 clock,
    which would preview at the wrong hour relative to its own commit.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import gating, signals

    hours = PREVIEW_HOURS.get(cfg.station.icao, {}).get(spec.key)
    if not hours:
        return 0
    la_hour = now.astimezone(ZoneInfo(cfg.station.tz)).hour
    if la_hour not in hours:
        return 0
    target = now.astimezone(ZoneInfo(cfg.station.tz)).date().isoformat()
    key = f"{target}:{la_hour}"
    seen = conn.execute(
        "SELECT 1 FROM alert_stamps WHERE station=? AND kind=? AND key=?",
        (cfg.station.icao, f"{spec.key}_preview", key)).fetchone()
    if seen:
        return 0
    out = signals.live_decision(conn, cfg, spec, now)
    # Same target-date guard as every other alert: a preview of tomorrow's
    # ladder labelled with today's date is the 2026-07-27 bug all over again.
    if out.get("target") != target:
        return 0
    market = out.get("market") or []
    if not market:
        return 0
    quote = max(market, key=lambda r: (r.get("p_model") or 0.0))
    price, p = quote.get("market_price"), quote.get("p_model")
    edge = None if (price is None or p is None) else p - price
    acc = gating.hour_accuracy(cfg.station.icao, spec.key, la_hour)
    msg = build_low_preview(
        cfg, target, la_hour, _manila_label(cfg, la_hour), quote, edge,
        acc["hit"] if acc else None,
        gating.extremum_still_future(cfg.station.icao, spec.key, la_hour),
        metric=spec.key)
    if not send(cfg, msg):
        return 0
    log_alert(conn, cfg, metric=spec.key, target=target, kind="preview", now=now,
              bucket=quote["label"], p_model=p, kalshi_pct=price, edge=edge,
              market=market, local_hour=la_hour)
    from highcast import db as _db
    _db.upsert_ignore(conn, "alert_stamps", [{
        "station": cfg.station.icao, "kind": f"{spec.key}_preview", "key": key,
        "sent_utc": now.isoformat()}])
    return 1


def notify_price_drop(conn, cfg: AppConfig, spec, now: datetime) -> int:
    """Fire once when the committed bucket gets cheap enough to clear the gate.

    THREE CONDITIONS, all required:

      1. The price actually FELL below what the app paid. Asked for explicitly:
         the interesting event is the market coming down, not the model drifting
         up to meet a price that never moved.
      2. The edge gate now passes at the live price -- `p_model - price >=
         min_edge`, the same threshold `staking.evaluate` uses. No second
         threshold is invented here; "worth placing" has one definition in this
         codebase and this reuses it.
      3. The bucket is still quoted on a market for the SAME target date. Once a
         window closes `live_decision` rolls to tomorrow, and tomorrow's ladder
         priced against today's locked bet is the bug that produced a bogus
         "model moved" alert on 2026-07-27. Same guard, same reason.

    Deliberately reads the LIVE model probability rather than the one frozen at
    commit time: the question being answered is "is buying this right now +EV",
    and a stale numerator would answer a question nobody asked. This does NOT
    make the bet mutable -- `app_bets` is never rewritten here, only stamped.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import signals
    from zoneinfo import ZoneInfo as _Z

    target = now.astimezone(_Z(cfg.station.tz)).date().isoformat()
    row = conn.execute(
        """SELECT * FROM app_bets WHERE station=? AND metric=? AND target_date=?
             AND price_alert_utc IS NULL AND won IS NULL""",
        (cfg.station.icao, spec.key, target)).fetchone()
    if row is None:
        return 0
    bet = dict(row)
    out = signals.live_decision(conn, cfg, spec, now)
    if out.get("target") != bet["target_date"]:
        return 0
    market = out.get("market") or []
    quote = next((m for m in market if m["label"] == bet["bucket"]), None)
    if quote is None:
        return 0
    price, p_model = quote.get("market_price"), quote.get("p_model")
    if price is None or p_model is None or price <= 0:
        return 0
    if price >= bet["price"]:
        return 0
    edge = p_model - price
    if edge < cfg.staking.min_edge:
        return 0
    if send(cfg, build_price_drop(cfg, bet, price, p_model, edge)):
        _stamp(conn, cfg, bet, "price_alert_utc", now)
        log_alert(conn, cfg, metric=spec.key, target=target, kind="price_drop",
                  now=now, bucket=bet["bucket"], p_model=p_model,
                  kalshi_pct=price, edge=edge, market=market,
                  local_hour=now.astimezone(ZoneInfo(cfg.station.tz)).hour)
        return 1
    return 0


def notify_pending(conn, cfg: AppConfig, now: datetime) -> int:
    """Alert on every committed bet that has not been announced yet.

    Driven off `notified_utc IS NULL` rather than "was placed this tick", so an
    alert missed while the webhook was down is retried on the next wake instead
    of lost.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import gating

    rows = conn.execute(
        """SELECT * FROM app_bets WHERE station=? AND notified_utc IS NULL
           ORDER BY target_date, metric""", (cfg.station.icao,)).fetchall()
    sent = 0
    for r in rows:
        bet = dict(r)
        hour = bet.get("placed_local_hour")
        acc = gating.hour_accuracy(cfg.station.icao, bet["metric"], hour)
        extra = {
            "hour_hit": acc["hit"] if acc else None,
            "still_future": gating.extremum_still_future(
                cfg.station.icao, bet["metric"], hour),
            "reason": bet.get("reason"),
        }
        # The ladder AS OF THE COMMIT, not now: the alert describes a decision
        # that was already made, and a later ladder would show the human a
        # disagreement that did not exist when they committed.
        lad, _when = marketsensor._latest_ladder(
            conn, cfg.station.icao, bet["metric"], bet["target_date"])
        extra["disagreement"] = marketsensor.disagreement(lad)
        if send(cfg, build_message(cfg, bet, extra)):
            conn.execute(
                """UPDATE app_bets SET notified_utc=?
                   WHERE station=? AND metric=? AND target_date=?""",
                (now.isoformat(), cfg.station.icao, bet["metric"],
                 bet["target_date"]))
            conn.commit()
            log_alert(conn, cfg, metric=bet["metric"], target=bet["target_date"],
                      kind="commit", now=now, bucket=bet["bucket"],
                      p_model=bet.get("p_model"), kalshi_pct=bet.get("price"),
                      edge=(None if bet.get("p_model") is None
                            or bet.get("price") is None
                            else bet["p_model"] - bet["price"]),
                      local_hour=hour)
            sent += 1
    return sent


def build_stale_feeds(cfg: AppConfig, bad: list[dict]) -> dict:
    """"A feed stopped delivering." Names every offender and its age.

    Terse and factual: this is a maintenance alert, not a trading one, and the
    only useful content is WHICH feed and HOW LONG.
    """
    lines = []
    for r in bad:
        if r.get("age_min") is None:
            lines.append(f"• **{r['label']}** — {r['reason']}")
        else:
            lines.append(
                f"• **{r['label']}** — {r['age_min'] / 60:.0f}h old "
                f"(limit {r['max_age_min'] / 60:.0f}h)"
                + (f", {r['reason']}" if r.get("reason") else ""))
    return {"embeds": [{
        "title": f"⚠️ FEED STALE — {cfg.station.icao}",
        "description": ("\n".join(lines) + "\n\nThe wake cycle is running, so "
                        "this is the feed itself, not the app."),
        "color": 0xE8A33D,
        "footer": {"text": "forecast_conv died for 9 days in July before anyone "
                           "noticed. This is the guard for that."},
    }]}


def notify_stale_feeds(conn, cfg: AppConfig, now: datetime) -> int:
    """Alert once per feed per DAY that it is stale.

    ⚠️ THE STAMP KEY INCLUDES THE LOCAL DATE, deliberately. `notify_missed_commit`
    keys on the target date because a missed commit is a one-off event; a dead
    feed is an ONGOING condition that stays true for days, and the wake cycle
    runs every five minutes. Stamping once forever would mean a feed that dies,
    is fixed, and dies again never alerts a second time. Stamping per day gives
    one reminder each morning until it is fixed — enough to stay visible, not
    enough to become noise worth muting.

    A FAILED SEND IS NEVER STAMPED, so it retries next wake.
    """
    if not cfg.discord_webhook_url:
        return 0
    from highcast import db as _db
    from highcast import staleness

    bad = staleness.stale(conn, cfg, now)
    if not bad:
        return 0
    day = now.astimezone(ZoneInfo(cfg.station.tz)).date().isoformat()
    fresh = []
    for r in bad:
        key = f"{day}:{r['table']}"
        if conn.execute("SELECT 1 FROM alert_stamps WHERE station=? AND kind=? "
                        "AND key=?",
                        (cfg.station.icao, "stale_feed", key)).fetchone():
            continue
        fresh.append((key, r))
    if not fresh:
        return 0
    if not send(cfg, build_stale_feeds(cfg, [r for _, r in fresh])):
        return 0              # failed send is retried next wake, never stamped
    _db.upsert_ignore(conn, "alert_stamps", [
        {"station": cfg.station.icao, "kind": "stale_feed", "key": k,
         "sent_utc": now.isoformat()} for k, _ in fresh])
    return len(fresh)
