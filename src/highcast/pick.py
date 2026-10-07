"""Pick of the Day — one honest verdict per market per day, or NO PICK.

WHAT THIS IS. An absolute-bar filter over the desk's existing read: take the
model's own bucket, shade its probability by the MEASURED calibration offset,
subtract Kalshi's fee, and publish only if what remains is positive by a margin.
Nothing qualifies -> NO PICK, and "skipping is part of the edge".

WHAT THIS IS NOT. It does not gate `staking.evaluate`, does not write to
`app_bets`, and does not touch the app's own committed prediction. `app_bets`
stays the immutable accuracy ledger on its flat unit stake; this is advice about
JULIAN'S money, computed alongside it. A test asserts `app_bets` is unchanged.

⚠️ READ THIS BEFORE TRUSTING IT — three things are known and unflattering.

1. IT PROTECTS AGAINST OVERPAYING, NOT AGAINST LOSING. Run over the eight bets
   committed 2026-07-27..29 it publishes 3 and goes 1W-1L-1 live, which is no
   better than betting everything (4W-2L). It declines three bets that WON, all
   priced 69-80c against calibrated reads of 39-73%. Winning a badly-priced bet
   does not make it a good bet -- but neither does n=3 prove this bar works.

2. IT PUBLISHES THE 2026-07-29 KLAX HIGH, WHICH LOST. An earlier draft applied an
   extra calibration penalty to KLAX HIGH readings >=80F and thereby declined it.
   The measurement refuted that penalty (band calib +8.6% [-0.2,+17.9] vs +8.9%
   overall -- statistically identical), so it is gone, and with it the only reason
   this bar looked good. Recorded rather than quietly re-tuned, per HANDOFF's BIAS
   DISCIPLINE: a threshold chosen because it produces a nicer table is exactly how
   a market-follower gets built one "validated" step at a time.

3. THE PRICE HALF CANNOT BE BACKTESTED. `signals` holds 7 days of prices; Kalshi
   serves ~68 days of SETTLED markets, whose prices say nothing about the price at
   our commit hour. So PRICE_CEILING and MARGIN are PRIORS, not measurements, and
   the graded log below is the only validation path that will ever exist. Hence
   the kill criterion -- it is not garnish.

WHY AN ABSOLUTE BAR rather than a best-of-four tournament: the four candidates
commit at staggered Manila hours (KMIA LOW ~17:00, KLAX LOW 19:00, KMIA HIGH
~23:00, KLAX HIGH 02:00), so a tournament could not name a winner until the last
commit and would forfeit acting on the earlier ones.
"""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

from highcast import db, fees
from highcast.buckets import ladder_lo, modal_bucket, pmf_from_buckets
from highcast.config import AppConfig
from highcast.gating import (EXTREMUM_STILL_FUTURE, HOUR_ACCURACY,
                             MIN_DECISION_HOUR, hour_calib)

# ── the two thresholds, and an honest label for each ────────────────────────
# NEITHER IS MEASURED. There is no price history to fit them on (see the module
# docstring), so they are judgement calls stated as such. Revisit them from the
# graded log, never from theory or from whether they make a table look better.
#
# PRICE_CEILING is deliberate redundancy: if `p_hat` is still overconfident in
# some regime nobody has measured, a hard ceiling refuses expensive certainty
# regardless of what the EV arithmetic says. nowcast.trade's disclosed average
# entry was 71.6c, which needs a 73.0% true hit rate to break even after fees --
# above every hit rate we have measured -- so their price point is NOT the anchor.
PRICE_CEILING = 0.75
MARGIN = 0.05

# Pre-registered kill criterion. If the published picks have not delivered
# positive realised EV after this many graded picks, the feature comes out. Stated
# here so it cannot be quietly forgotten, and chosen before any data exists.
KILL_AFTER_PICKS = 30

# Verdicts, most-blocking first. Order matters: UNBUYABLE outranks everything
# because a pick you cannot transact is not a pick, however good the arithmetic.
UNSCORED = "UNSCORED"
UNBUYABLE = "UNBUYABLE"
NO_PICK_PRICE = "NO PICK (price)"
NO_PICK_EV = "NO PICK"
NO_PICK_NO_PRICE = "NO PICK (no price)"
NO_PICK_VETOED = "NO PICK (regime veto)"
NO_PICK_TOO_EARLY = "NO PICK (before commit hour)"
NO_PICK_ABSURD_EDGE = "NO PICK (implausible edge)"
PICK = "PICK"

# An edge this large against a live book is a broken model, not an opportunity.
# HANDOFF's standing rule, learned on real money four times (+43%, +58%, +71%,
# +48%): "a 40-70 point edge against a liquid market is a trigger to investigate
# the model, and in both these cases the model was broken."
#
# FOUND BY RUNNING THIS MODULE, 2026-07-30. At KLAX local hour 1 it scored bucket
# "<=60" at p_model 52% against a 1c market -- a +51 point edge on a station whose
# lows run 69F -- and returned PICK. It also scored "<=75" at 82% against 4c for
# the HIGH. Both were garbage from a model an hour into its climate day, and both
# would have been blocked by `signals._regime_veto`, which this module was
# ignoring. A Pick of the Day that publishes those is worse than no feature.
MAX_PLAUSIBLE_EDGE = 0.25

# ── the pair ────────────────────────────────────────────────────────────────
# At most this many picks per WEATHER DAY, across both stations. Julian's call,
# 2026-07-30: "can we change it to a pair, the most confident pair, so can be low
# klax and high kmia depending on the model's pick."
#
# GROUPED BY target_date, NOT by Manila calendar day. The four candidates for one
# weather day commit across a 9-hour Manila window -- KMIA LOW ~17:00, KLAX LOW
# 19:00, KMIA HIGH ~23:00, KLAX HIGH 02:00 the NEXT Manila day. Grouping by the
# day being forecast is the only grouping under which "a pair for the day" means
# anything; grouping by Manila date would split KLAX's LOW from its own HIGH.
#
# ⚠️ THIS CAP IS DELIBERATELY CROSS-STATION, which is the exact opposite of the
# standing rule that anything keyed by metric:date alone is a latent cross-station
# bug (see the _KALSHI_CACHE fix). The rule is right and this is the documented
# exception: the pair is a constraint on JULIAN'S ATTENTION AND BANKROLL, which are
# shared across stations, not a property of either market. Everything else in this
# module stays station-scoped.
#
# SELECTED BY QUALITY, NOT BY ARRIVAL ORDER. Julian's 2026-07-30 revision: back to
# ONE pick a day, and "make sure it chooses the right one".
#
# With two slots, first-come-first-served was tolerable. With ONE it is actively
# wrong: KMIA LOW commits at 17:00 Manila and would seize the only slot nine hours
# before the last candidate exists. So `record` no longer refuses late qualifiers.
# Every candidate that clears the bar is logged as PICK -- meaning "qualified" --
# and `day_pick` names THE pick as the highest-EV qualifier available. Nothing is
# ever rewritten, so `upsert_ignore`'s first-write-wins anti-cheat contract holds;
# selection is a read over immutable rows, not an edit to them.
#
# ⚠️ WHAT THIS CANNOT DO: choose the one that WINS. At measured commit-hour hit
# rates of 51.5-62.3%, roughly four picks in ten lose by construction. Tuning this
# selector until a known day would have picked a known winner is fitting to one
# sample, and is the precise failure BIAS DISCIPLINE describes -- every such step
# looks validated and the end state is a market-follower with no edge. The selector
# maximises MEASURED expected value; it does not foresee outcomes.
MAX_PICKS_PER_DAY = 1

# Tiebreak when two candidates are within this much EV of each other. Prefer the
# market with the higher MEASURED bucket-hit rate at its commit hour -- KLAX low
# 62.3% > KMIA low 56.7% > KMIA high 56.5% > KLAX high 51.5%. That ordering is
# measured (n=565-572 per market, walk-forward), not chosen for how it scores on
# any particular day. Small window on purpose: EV is the primary criterion and
# this only settles near-ties, where EV carries no real information.
EV_TIEBREAK_WINDOW = 0.02

# A >=80F reading from the KLAX HIGH is HARDER, though NOT more overconfident:
# hit 42.0% (n=81) vs 50.6% below, MAE 1.365-2.010 vs ~1.1. That belongs on the
# display, never as an arithmetic adjustment to a probability -- see docstring
# point 2 for what happened when it was one.
HARD_BAND = {("KLAX", "high"): (80.0, 0.420, 81)}


def hard_band_note(station: str, metric: str, point: float | None) -> str | None:
    """Display-only warning when a reading sits in a measured-harder band."""
    spec = HARD_BAND.get((station, metric))
    if spec is None or point is None:
        return None
    floor, hit, n = spec
    if point < floor:
        return None
    return (f"reading >={floor:.0f}F, where measured bucket hit is {hit:.0%} "
            f"(n={n}) rather than the overall rate")


def _commit_instant(cfg: AppConfig, target: str, commit_hour: int) -> dt.datetime:
    """UTC instant of the commit hour for `target` in the station's wall clock."""
    d = dt.date.fromisoformat(target)
    return dt.datetime(d.year, d.month, d.day, commit_hour,
                       tzinfo=ZoneInfo(cfg.station.tz)).astimezone(dt.timezone.utc)


def churn_to_commit(conn, cfg: AppConfig, target: str, metric: str,
                    commit_hour: int, parity: int,
                    cutoff_utc: dt.datetime | None = None) -> int | None:
    """Distinct modal buckets the desk passed through, up to the COMMIT hour.

    ⚠️ LEAKAGE GUARD, CHRONOLOGICAL. Predictions issued after the cutoff instant
    are excluded by comparing timestamps directly, not by local hour. An earlier
    version compared `local_hour > commit_hour`, which silently LEAKED any row
    issued on a later date at a small local hour (measured: 1 such row per KLAX
    market) -- the hour wraps across midnight and the proxy cannot see the date.
    Counting a post-commit reading uses information that did not exist at the pick,
    which the freeze doc classifies as a MODEL bug, not a display one.

    `cutoff_utc` defaults to the commit instant; a preview passes its own `now` so
    churn reflects only what was visible when the preview fired.

    Scored on `parity` -- the ladder of the candidate's own bucket -- because
    `argmax_pair` is hardcoded ODD and matches no reliable ladder. This is the
    same trap that produced a wrong answer in this project's own analysis on
    2026-07-30, on a 67-68 (odd-floor) bucket.

    UNVALIDATED as a predictor. Churn has never been shown to correlate with a
    miss, and two changes that RAISED it (full pooling, parametric Student-t) also
    lost accuracy -- so churn moved with the losers, which is not evidence it
    predicts anything alone.
    """
    # Default cutoff = the END of the commit hour, not its start: the bet commits
    # at the FIRST reading at/after commit_hour (e.g. 04:05), so that reading and
    # any others within the commit hour legitimately count; only h(commit+1)+ and
    # later dates are the future. A caller (a preview) may pass an earlier cutoff.
    cutoff = cutoff_utc or (_commit_instant(cfg, target, commit_hour)
                            + dt.timedelta(hours=1))
    rows = conn.execute(
        """SELECT issued_at_utc, buckets FROM predictions
           WHERE station=? AND target_date=? AND metric=? AND kind='intraday'
             AND buckets IS NOT NULL
           ORDER BY issued_at_utc""",
        (cfg.station.icao, target, metric)).fetchall()
    seen: list[int] = []
    for r in rows:
        iss = dt.datetime.fromisoformat(r["issued_at_utc"])
        if iss > cutoff:
            continue
        pmf = pmf_from_buckets(json.loads(r["buckets"]))
        if not pmf:
            continue
        lo, _ = modal_bucket(pmf, parity)
        if not seen or seen[-1] != lo:
            seen.append(lo)
    return len(set(seen)) or None


def score(conn, cfg: AppConfig, spec, now: dt.datetime,
          out: dict | None = None) -> dict:
    """Score one market at `now`. Pure read; `record` does the persisting.

    `out` lets a caller pass an existing `signals.live_decision` result rather
    than paying for a second Kalshi round-trip.
    """
    from highcast import signals, staking
    icao, metric = cfg.station.icao, spec.key
    tz = ZoneInfo(cfg.station.tz)
    local = now.astimezone(tz)
    out = out if out is not None else signals.live_decision(conn, cfg, spec, now)
    target = out.get("target")
    commit_h = MIN_DECISION_HOUR.get(icao, {}).get(metric)
    base = {"station": icao, "metric": metric, "target_date": target,
            "issued_utc": now.isoformat(), "local_hour": local.hour,
            "commit_hour": commit_h, "verdict": None, "bucket": None,
            "p_model": None, "p_hat": None, "price": None, "contracts": None,
            "fee": None, "ev": None, "churn": None, "stall_hours": None,
            "score_with_stability": None, "score_without": None,
            "band_note": None, "reason": None}

    cal = hour_calib(icao, metric, commit_h) if commit_h is not None else None
    if cal is None:
        # NO cross-station fallback, by design. Borrowing another station's offset
        # would hand back a confident-looking number for an unmeasured market.
        return {**base, "verdict": UNSCORED,
                "reason": (f"no measured calibration for {icao} {metric} at "
                           f"h{commit_h} — cannot turn a stated probability into "
                           f"one we believe")}

    # ── REGIME VETO. Must come before anything else that looks at the number. ──
    # A veto means the model was running outside the regime it was trained and
    # validated on, so the probability itself is untrustworthy -- categorically
    # different from "no edge" or "too illiquid", which are sound numbers not worth
    # trading. `appbet` may ignore `place=False` (it records the model's honest
    # opinion) but NOTHING may ignore this.
    decision = out.get("decision")
    if decision is not None and getattr(decision, "veto", False):
        return {**base, "verdict": NO_PICK_VETOED,
                "reason": (f"the desk vetoed this read: {decision.reason} — a "
                           f"vetoed probability is not a number to price against")}

    # ── BEFORE THE COMMIT HOUR there is no pick to make. ──
    # The commit hour is where accuracy steps up because the extremum stops being
    # a pure forecast. Publishing at h1, where KLAX's LOW measures 37.1% on n=194,
    # would put the day's headline on the worst hour of the curve.
    if commit_h is not None and local.hour < commit_h:
        return {**base, "verdict": NO_PICK_TOO_EARLY,
                "reason": (f"local hour {local.hour} is before the h{commit_h} "
                           f"commit hour — too early to call, and the pre-knee "
                           f"hours are measurably the least accurate")}

    market = out.get("market")
    if not market:
        # 37% of LOW days historically had no usable price.
        return {**base, "verdict": NO_PICK_NO_PRICE,
                "reason": "no live market prices — cannot assess EV"}

    priced = [r for r in market if r.get("p_model") is not None
              and r.get("market_price") is not None]
    if not priced:
        return {**base, "verdict": NO_PICK_NO_PRICE,
                "reason": "no bucket carries both a model probability and a price"}

    # The model's OWN most-backed bucket. We judge the desk's genuine read; we do
    # not shop the ladder for whichever bucket happens to price well, which would
    # be fishing a mispriced tail rather than expressing a forecast.
    top = max(priced, key=lambda r: r["p_model"])
    p_model, price = float(top["p_model"]), float(top["market_price"])
    # Prefer the ASK for cost when the book gives one: `market_price` has been
    # observed OUTSIDE its own bid/ask (70-71 at 0.25 against bid 0.28/ask 0.36),
    # and paying the ask is what actually happens.
    ask = top.get("yes_ask")
    cost = float(ask) if isinstance(ask, (int, float)) and ask > 0 else price
    p_hat = p_model - cal

    sz = staking.size(p_model, cost, cfg.staking)
    contracts = int(sz.stake // cost) if cost > 0 else 0
    fee = fees.fee_per_contract(max(contracts, 1), cost)
    ev = p_hat - cost - fee

    point = (out.get("pred") or {}).get("point_f")
    band = hard_band_note(icao, metric, point)
    # Parity of the LIVE ladder, taken from the market's own floor rather than
    # assumed. Kalshi centres each day's ladder on the consensus forecast, so the
    # parity is a coin flip per day -- hardcoding either one is wrong ~half the
    # time, which is the ladder-parity bug this project already had to unwind.
    parity = (int(top["lo"]) % 2) if top.get("lo") is not None else 0
    # Cutoff = the earlier of now and the commit instant. A preview fired before
    # the commit sees only churn so far; a verdict at/after commit sees the whole
    # pre-commit window. Never counts a reading issued after this pick was made.
    # End of the commit hour, but never past `now` — a preview cannot see readings
    # that have not happened yet, a verdict must not count past its own hour.
    commit_end = (_commit_instant(cfg, target, commit_h) + dt.timedelta(hours=1)
                  if (target and commit_h is not None) else None)
    cutoff = min(now, commit_end) if commit_end else now
    churn = (churn_to_commit(conn, cfg, target, metric, commit_h, parity,
                             cutoff_utc=cutoff)
             if (target and commit_h is not None) else None)
    stall = None
    if target:
        try:
            from highcast import board as _board
            stall = _board.row(conn, cfg, metric, now=now).get("stall_hours")
        except Exception:
            stall = None  # advisory context only; never fail the pick for it

    common = {**base, "bucket": top.get("label"), "p_model": p_model,
              "p_hat": p_hat, "price": cost, "contracts": contracts,
              "fee": fee, "ev": ev, "churn": churn, "band_note": band,
              "stall_hours": stall,
              # BOTH scores stored so the provisional stability term MEASURES
              # ITSELF: after enough picks we can ask retroactively whether it
              # improved selection, instead of leaving it an open question.
              "score_without": ev,
              "score_with_stability": ev - (0.01 * ((churn or 1) - 1))}

    # ── IMPLAUSIBLE EDGE. Checked before affordability, because "the model is
    # broken" outranks "you cannot afford it" as a reason to say no.
    if (p_hat - cost) >= MAX_PLAUSIBLE_EDGE:
        return {**common, "verdict": NO_PICK_ABSURD_EDGE,
                "reason": (f"{top.get('label')}: shaded {p_hat:.0%} against a "
                           f"{cost:.2f} market is a {(p_hat - cost):+.0%} edge. "
                           f"An edge this large against a live book has meant a "
                           f"broken model every time it has appeared here — "
                           f"refusing rather than pricing it")}
    if contracts < 1:
        want = cost / cfg.staking.bankroll if cfg.staking.bankroll else None
        hint = (f" — raise staking.max_stake_frac to ~{want:.2f} to afford one"
                if want else "")
        return {**common, "verdict": UNBUYABLE,
                "reason": (f"{top.get('label')} at {cost:.2f} buys 0 whole "
                           f"contracts on a {cfg.staking.bankroll:.2f} bankroll "
                           f"capped at {cfg.staking.max_stake_frac:.0%}{hint}")}
    if cost > PRICE_CEILING:
        return {**common, "verdict": NO_PICK_PRICE,
                "reason": (f"{cost:.2f} is above the {PRICE_CEILING:.2f} ceiling "
                           f"— expensive certainty, refused regardless of EV")}
    if ev <= MARGIN:
        return {**common, "verdict": NO_PICK_EV,
                "reason": (f"{top.get('label')}: p_model {p_model:.0%} shaded to "
                           f"{p_hat:.0%} (calib {cal:+.1%}) vs {cost:.2f} + "
                           f"{fee:.3f} fee = EV {ev:+.3f}, under the "
                           f"{MARGIN:+.2f} bar")}
    return {**common, "verdict": PICK,
            "reason": (f"{top.get('label')}: p_model {p_model:.0%} shaded to "
                       f"{p_hat:.0%} (calib {cal:+.1%}) vs {cost:.2f} + "
                       f"{fee:.3f} fee = EV {ev:+.3f} on {contracts} contract(s)")}


def published_for_target(conn, target_date: str) -> list[dict]:
    """PICKs already published for this weather day, across BOTH stations.

    Cross-station on purpose -- see MAX_PICKS_PER_DAY. Only `kind='verdict'`
    counts: a provisional preview must never consume a slot, or a preview that
    later disagrees would have spent the pair on a read we no longer hold.
    """
    return [dict(r) for r in conn.execute(
        """SELECT station, metric, bucket, ev, price FROM picks
           WHERE target_date=? AND kind='verdict' AND verdict=?
           ORDER BY issued_utc""", (target_date, PICK))]


def _reliability(station: str, metric: str) -> float:
    """Measured bucket-hit rate at this market's commit hour. Tiebreak only."""
    h = MIN_DECISION_HOUR.get(station, {}).get(metric)
    if h is None:
        return 0.0
    got = HOUR_ACCURACY.get(station, {}).get(metric, {}).get(h)
    return float(got[1]) if got else 0.0


def day_pick(conn, target_date: str) -> dict | None:
    """THE pick for this weather day: the highest-EV qualifier, or None.

    Selection is a READ over immutable rows, never an edit to them -- so a better
    candidate arriving at 02:00 supersedes a 17:00 one without rewriting history,
    and the superseded row stays on record with its own EV.

    Ties within EV_TIEBREAK_WINDOW go to the market with the higher measured hit
    rate at its commit hour. Where EV genuinely cannot separate two candidates,
    "which market is historically more often right" is the honest next question.
    """
    rows = [dict(r) for r in conn.execute(
        """SELECT * FROM picks WHERE target_date=? AND kind='verdict'
             AND verdict=? AND ev IS NOT NULL AND contracts >= 1""",
        (target_date, PICK))]
    if not rows:
        return None
    best = max(rows, key=lambda r: r["ev"])
    near = [r for r in rows if best["ev"] - r["ev"] <= EV_TIEBREAK_WINDOW]
    if len(near) > 1:
        best = max(near, key=lambda r: (_reliability(r["station"], r["metric"]),
                                        r["ev"]))
    return {**best, "reliability": _reliability(best["station"], best["metric"]),
            "superseded": [
                {"station": r["station"], "metric": r["metric"],
                 "bucket": r["bucket"], "ev": r["ev"]}
                for r in sorted(rows, key=lambda r: -r["ev"])
                if not (r["station"] == best["station"]
                        and r["metric"] == best["metric"])]}


def record(conn, cfg: AppConfig, spec, now: dt.datetime, kind: str,
           out: dict | None = None) -> dict:
    """Score and persist. `kind` is 'preview' (provisional) or 'verdict'.

    EVERY verdict is written, NO PICK and UNBUYABLE included. Without the declines
    on record we could never discover that the bar is refusing good trades — and
    on the only sample that exists it declined three winners out of six.

    `upsert_ignore`: first write wins, same anti-cheat contract as `app_bets`.
    """
    row = score(conn, cfg, spec, now, out=out)
    if row.get("target_date") is None:
        return row
    # NO ARRIVAL-ORDER CAP. A qualifier is logged as PICK meaning "qualified";
    # `day_pick` decides which one is THE pick. Refusing late qualifiers here
    # would hand the single slot to whichever market happens to commit first,
    # which is 17:00 Manila's KMIA LOW every single day.
    db.upsert_ignore(conn, "picks", [{
        "station": row["station"], "metric": row["metric"],
        "target_date": row["target_date"], "kind": kind,
        "issued_utc": row["issued_utc"], "local_hour": row["local_hour"],
        "verdict": row["verdict"], "bucket": row["bucket"],
        "p_model": row["p_model"], "p_hat": row["p_hat"], "price": row["price"],
        "contracts": row["contracts"], "fee": row["fee"], "ev": row["ev"],
        "score_with_stability": row["score_with_stability"],
        "score_without": row["score_without"], "churn": row["churn"],
        "stall_hours": row["stall_hours"], "reason": row["reason"],
        "actual": None, "won": None, "settled_utc": None,
    }])
    return row


def settle(conn, cfg: AppConfig, now: dt.datetime) -> int:
    """Grade published picks against the settled CLI value. Wins AND losses.

    ⚠️ NEVER grades against a PRELIMINARY CLI product. One issued before its
    climate day closed physically cannot see the rest of the day -- KMIA
    2026-08-01 is the case that made this necessary: the only product IEM
    ever issued for that date was filed 8.5h early and settlement graded a
    >=81 bet WON against it when the true low (74F) meant it had LOST. See
    `scripts/migrate_prelim_settlement.py`.

    `cli_reports.cli_row_is_final` is the single, shared definition of
    "final" (also used by `appbet.settle` and `backfill._yesterday_cli`). A
    pick whose only stored product is preliminary is left with `won` still
    NULL -- the `WHERE p.won IS NULL` filter below means it is retried on
    every later call, never silently dropped.
    """
    from highcast import signals
    from highcast.fetchers import cli_reports
    n = 0
    rows = conn.execute(
        """SELECT p.* FROM picks p WHERE p.station=? AND p.won IS NULL
             AND p.verdict=? AND p.bucket IS NOT NULL""",
        (cfg.station.icao, PICK)).fetchall()
    for r in rows:
        col = "high_f" if r["metric"] == "high" else "low_f"
        got = conn.execute(
            f"SELECT {col} AS v, raw FROM cli_reports WHERE station=? AND date=?",
            (cfg.station.icao, r["target_date"])).fetchone()
        if got is None or got["v"] is None:
            continue
        product = None
        if got["raw"]:
            try:
                product = json.loads(got["raw"]).get("product")
            except (TypeError, ValueError):
                product = None
        if not cli_reports.cli_row_is_final(
                product, r["target_date"], cfg.station.std_utc_offset):
            continue          # preliminary (or unknown) -- retry on a later call
        actual = int(got["v"])
        signals.check_settlement_against_market(
            conn, cfg, r["metric"], r["target_date"], actual)
        lo = int(str(r["bucket"]).replace("–", "-").split("-")[0])
        won = 1 if lo <= actual <= lo + 1 else 0
        conn.execute(
            """UPDATE picks SET actual=?, won=?, settled_utc=?
               WHERE station=? AND metric=? AND target_date=? AND kind=?""",
            (actual, won, now.isoformat(), r["station"], r["metric"],
             r["target_date"], r["kind"]))
        n += 1
    conn.commit()
    return n


def graded_record(conn, cfg: AppConfig) -> dict:
    """The falsifiable promise: wins, losses, realised EV, and the kill check.

    Realised EV per contract is (won ? 1 - price - fee : -price - fee), which is
    the number the bar claims to be maximising -- not the win rate. A bar can post
    a fine win rate and still lose money, which is the whole critique of
    nowcast.trade's 85%.
    """
    rows = [dict(r) for r in conn.execute(
        """SELECT * FROM picks WHERE station=? AND verdict=? AND won IS NOT NULL""",
        (cfg.station.icao, PICK))]
    wins = sum(1 for r in rows if r["won"] == 1)
    losses = sum(1 for r in rows if r["won"] == 0)
    realised = 0.0
    for r in rows:
        price, fee = r["price"] or 0.0, r["fee"] or 0.0
        realised += (1.0 - price - fee) if r["won"] == 1 else (-price - fee)
    declines = conn.execute(
        """SELECT verdict, COUNT(*) n FROM picks WHERE station=? AND verdict<>?
           GROUP BY verdict""", (cfg.station.icao, PICK)).fetchall()
    # How often did a LATER candidate beat the one that qualified first? This is
    # the number that justifies selecting on quality rather than arrival -- if it
    # is always zero, the simpler rule would have been fine.
    superseded_days = 0
    for row_t in conn.execute(
            """SELECT DISTINCT target_date FROM picks WHERE kind='verdict'"""):
        target = row_t[0]
        chosen = day_pick(conn, target)
        if chosen is None:
            continue
        first = published_for_target(conn, target)
        if first and (first[0]["station"], first[0]["metric"]) != (
                chosen["station"], chosen["metric"]):
            superseded_days += 1
    n = len(rows)
    return {
        "graded": n, "wins": wins, "losses": losses,
        "picks_per_day": MAX_PICKS_PER_DAY,
        "days_a_later_candidate_won_the_slot": superseded_days,
        "realised_ev": round(realised, 4),
        "realised_ev_per_pick": round(realised / n, 4) if n else None,
        "declines": {r["verdict"]: r["n"] for r in declines},
        "kill_after": KILL_AFTER_PICKS,
        # The pre-registered verdict. None until there is enough to judge.
        "kill_triggered": (n >= KILL_AFTER_PICKS and realised <= 0.0
                           if n >= KILL_AFTER_PICKS else None),
        "note": ("PRICE_CEILING and MARGIN are priors, not measurements — there "
                 "is no price history to fit them on. This record is the only "
                 "validation this feature will ever have."),
    }
