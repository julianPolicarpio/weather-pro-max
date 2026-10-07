"""THE MARKET AS AN OBSERVATION SENSOR — reading the ladder we already store.

`signals.market_json` holds the full Kalshi ladder every ~5 minutes. When the
market concentrates on one rung it is reporting an observation of the settling
quantity, often one we cannot see yet.

⚠️ THIS IS NOT AND CANNOT BE A TRADING SIGNAL. It is the counterparty's own
opinion, arriving after they have priced it. Reading it makes us
CONTEMPORANEOUS with the market, never ahead. QUEUE item 1: the market already
beats our model at 78 of 81 hours, so feeding this to the model would launder
the market's view into ours and call it an edge. Everything here runs for
MONITORING and GRADING only.

⚠️ THE OBVIOUS VERSION WAS MEASURED AND REJECTED. "The market knows before we
do" is false on the typical day. Over 56 day-markets, comparing when the market
first reaches 90% confidence against when `latentmax.visible_running_extremum`
first reaches the settled value:

    median lead -0.8h        our ratchet is usually FIRST
    market first on 20/56    36% of days
    ratchet never reached it  0 days

The 2026-08-14 KMIA case (market 5h46m ahead) is a TAIL, not the norm. A
real-time "the market knows something" alert would be wrong-headed and noisy.

WHAT SURVIVED MEASUREMENT, and why each earns its place:

1. DIVERGENCE — the market implies a value our observations cannot yet reach.
   Fires on 14/56 = 25% of day-markets, gaps of 1-2F. This is the blindspot
   (QUEUE item 13) observable IN REAL TIME rather than in hindsight.

2. GRADE CROSS-CHECK — the market's confident call versus the grade we recorded.
   It agreed 56/56 in the sample, so it has caught nothing yet. It is built
   anyway because MEMORY.md records that a PRELIMINARY CLI has mis-graded a bet
   TWICE, and a false WIN is the dangerous direction. A guard with no positives
   against a failure that has actually happened is worth its keep.
"""
from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime
from zoneinfo import ZoneInfo

from highcast import latentmax
from highcast.config import AppConfig
from highcast.metric import MetricSpec
from highcast.timeutil import round_half_up

# The market is RIGHT on 56/56 day-markets once it reaches this. Lowering it
# trades that accuracy for earliness and has not been measured — do not.
CONF = 0.90

#: A rung the market prices below this is not worth reporting a disagreement
#: about -- see `disagreement`. Display hygiene; measured inert on the signal.
MIN_MARKET = 0.10


def implied(ladder: list[dict], conf: float = CONF) -> dict | None:
    """The rung the market has concentrated on, or None if it has not decided.

    Asks are NORMALISED first. A Kalshi ladder's asks sum to ~1.09 (the house
    spread), so comparing raw asks to a probability threshold would call the
    market decided about 9% too eagerly.
    """
    asks = []
    for x in ladder:
        a = x.get("yes_ask")
        if a is None:
            return None
        asks.append(float(a))
    total = sum(asks)
    if total <= 0:
        return None
    probs = [a / total for a in asks]
    top = max(range(len(probs)), key=lambda i: probs[i])
    if probs[top] < conf:
        return None
    x = ladder[top]
    return {"label": x.get("label"), "lo": x.get("lo"), "hi": x.get("hi"),
            "prob": probs[top]}


def divergence(conn, cfg: AppConfig, spec: MetricSpec, day: _date,
               ladder: list[dict], asof_utc: str) -> dict | None:
    """How far AHEAD of our own ratchet the market's confident call sits.

    Returns None when the market has not decided, when we have no ratchet yet,
    or when the two agree. A positive `gap_f` means the market is reporting an
    extreme our observations cannot yet reach — we are blind, right now.
    """
    imp = implied(ladder)
    if imp is None:
        return None
    vis = latentmax.visible_running_extremum(conn, cfg, day, spec.extremum,
                                             asof_utc=asof_utc)
    if vis is None:
        return None
    if spec.extremum == "max":
        if imp["lo"] is None:
            return None                       # open bucket has no lower edge
        seen = round_half_up(vis[0])
        gap = imp["lo"] - seen
    else:
        if imp["hi"] is None:
            return None
        seen = round_half_up(vis[1])
        gap = seen - imp["hi"]
    if gap <= 0:
        return None
    return {"label": imp["label"], "prob": imp["prob"], "seen": seen,
            "gap_f": gap, "asof": asof_utc}


def _latest_ladder(conn, icao: str, metric: str, target_date: str):
    r = conn.execute(
        """SELECT created_utc, market_json FROM signals WHERE station=?
             AND metric=? AND target_date=? AND market_json IS NOT NULL
             AND market_json!='' ORDER BY created_utc DESC LIMIT 1""",
        (icao, metric, target_date)).fetchone()
    if not r:
        return None, None
    try:
        return json.loads(r["market_json"]), r["created_utc"]
    except Exception:                          # noqa: BLE001 - malformed row
        return None, None


def grade_conflicts(conn, cfg: AppConfig) -> list[dict]:
    """Settled bets whose recorded outcome contradicts the market's final call.

    ⚠️ THE POINT IS THE FALSE WIN. `MEMORY.md`: a preliminary CLI mis-graded a
    bet twice, and "a false WIN is the dangerous direction" — the desk records a win
    it did not earn and nothing downstream disagrees, because the audit
    and the grade read the same corrupted row. The market is an INDEPENDENT
    witness, right 56/56 when confident.

    Only bets already graded (`won IS NOT NULL`) are checked; an ungraded row is
    not a conflict, it is just pending.
    """
    out = []
    for b in conn.execute(
            """SELECT target_date, metric, bucket, actual, won FROM app_bets
               WHERE station=? AND won IS NOT NULL AND actual IS NOT NULL""",
            (cfg.station.icao,)):
        ladder, when = _latest_ladder(conn, cfg.station.icao, b["metric"],
                                      b["target_date"])
        if not ladder:
            continue
        imp = implied(ladder)
        if imp is None:
            continue
        y = int(b["actual"])
        consistent = ((imp["lo"] is None or y >= imp["lo"])
                      and (imp["hi"] is None or y <= imp["hi"]))
        if not consistent:
            out.append({"target_date": b["target_date"], "metric": b["metric"],
                        "recorded": y, "market_says": imp["label"],
                        "prob": imp["prob"], "bucket": b["bucket"],
                        "won": bool(b["won"]), "as_of": when})
    return out


def disagreement(ladder: list[dict] | None) -> dict | None:
    """WHERE THE COUNTERPARTY OUTBIDS US MOST — one number, for the human.

    Returns the rung with the largest (normalised market price - our p_model),
    i.e. the bucket Kalshi is paying for that we are not. Asks are normalised
    (overround ~1.09) before comparing, as in the HANDOFF section 0 Brier work.

    WHY THIS EXISTS. On 2026-08-18 the KMIA high settled 99-100. At commit our
    model gave that rung **2.6%** while Kalshi paid **36.5%** -- and the day
    grade said a clean B, because every INPUT was fine. Nothing on the desk
    showed the one number that was screaming.

    MEASURED 2026-08-19 over 79 settled day-markets, D taken at the commit:

        D band      n   P(our bet lost)
        < 10%      30        17%
        10-20%     25        12%
        20-30%     13        31%
        >= 30%     11        91%

        corr(D, LOST) = +0.571, 95% CI [+0.400, +0.703]

    ⚠️ This was PREDICTED BEFORE IT WAS MEASURED, which is why it is here at
    all. Section 0 measured corr(edge, won) = -0.555 where edge = model - market
    on OUR rung; D is the mirror of that quantity, so a positive correlation
    with losing is the same fact seen from the other side -- not a new discovery
    dredged out of one vivid day.

    ⚠️⚠️ AND IT IS NOT A VETO. THREE REASONS, ALL MEASURED:

    1. D IS MOSTLY PRICE. corr(D, our ask) = -0.798 and corr(ask, LOST) =
       -0.526. Split within price bands, D separates +60% below 40c but -8% at
       60-80c and +0% above 80c. Nearly everything it knows, the ask already
       knew, and the ask is free.
    2. A D VETO IS A CHEAP-CONTRACT VETO, and QUEUE item 1 measured that on
       MARGIN (hit minus ask) the cheap PLACE bets are AHEAD, +10.8% vs +8.5%.
       Filtering them out is what the confidence gate did when it inverted P&L.
    3. The trigger question is ALREADY pre-registered and accumulating, on a
       near-identical quantity, in `scripts/salvage_alert_gate.py` (decidable
       ~2026-09-10). Adding a second correlated rule now would double-count the
       same evidence and spend the out-of-sample window twice.

    So: DISPLAY ONLY. It goes on the commit alert so the number is in front of
    the human at the moment of the decision, with no threshold and no gate. It
    survives conditioning on `stated` (separating in all four bands) which is
    more than four dead ideas managed -- and that still does not make it a rule.
    """
    priced = [x for x in (ladder or [])
              if isinstance(x.get("p_model"), (int, float))
              and isinstance(x.get("yes_ask"), (int, float))]
    tot = sum(x["yes_ask"] for x in priced)
    if not priced or tot <= 0:
        return None
    # ⚠️ THE MARKET HAS TO ACTUALLY BE PAYING FOR THE RUNG. Without this the
    # answer is routinely a 1c rung where the model says 0% and the market says
    # 1% -- a "+1% disagreement" between two numbers that are both nothing. That
    # is what the dashboard showed on its first render.
    #
    # This is DISPLAY HYGIENE, NOT A THRESHOLD ON THE SIGNAL, and it was checked
    # rather than assumed: sweeping the floor over the same 79 settled
    # day-markets moves corr(D, LOST) from +0.571 (no floor) to +0.569 at 0.10,
    # and the 11 days with D >= 30% are IDENTICAL at every floor from 0.00 to
    # 0.20. It deletes noise and no information.
    cand = [x for x in priced if x["yes_ask"] / tot >= MIN_MARKET] or priced
    best = max(cand, key=lambda x: (x["yes_ask"] / tot) - x["p_model"])
    mkt = best["yes_ask"] / tot
    return {"label": best["label"], "market": mkt, "model": best["p_model"],
            "gap": mkt - best["p_model"]}


def wake_line(conn, cfg: AppConfig, now: datetime) -> str | None:
    """One line ONLY when something is worth saying.

    Deliberately silent on a normal day. Divergence fires on ~25% of
    day-markets, which is far too often for Discord but fine for the log; a
    grade conflict should never fire at all and is loud when it does.
    """
    from highcast.metric import get_metric

    bits = []
    # STATION-LOCAL date. At 05:10Z KMIA is on the new day while KLAX is still
    # on the previous one — the trap that once produced a bogus "0 of 4 missed".
    today = now.astimezone(ZoneInfo(cfg.station.tz)).date()
    for mkey in ("high", "low"):
        spec = get_metric(cfg, mkey)
        ladder, when = _latest_ladder(conn, cfg.station.icao, mkey,
                                      today.isoformat())
        if not ladder:
            continue
        d = divergence(conn, cfg, spec, today, ladder, when)
        if d:
            bits.append(f"{mkey} market says {d['label']} "
                        f"({d['prob']:.0%}) but we only see {d['seen']}F "
                        f"— blind by {d['gap_f']:.0f}F")
    conflicts = grade_conflicts(conn, cfg)
    if conflicts:
        c = conflicts[0]
        bits.append(f"⚠️ GRADE CONFLICT {c['target_date']} {c['metric']}: "
                    f"recorded {c['recorded']}F, market says {c['market_says']}"
                    + (f" — and it is booked as a WIN" if c["won"] else ""))
    return ("  market sensor: " + " | ".join(bits)) if bits else None
