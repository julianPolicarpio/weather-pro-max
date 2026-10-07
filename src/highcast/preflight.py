"""Is today's read one worth paying for?

WHAT THIS ANSWERS. The reliability table says which MARKET to trust and gives
the same answer every day. This says whether TODAY's instance of that market is
one of its good days or one of its bad ones. Julian asked for it as "a way to at
least know what is a good prediction to trade so I don't waste money on unstable
predictions" -- the competitor ships something under that name, but theirs is a
static per-city climatological grade with no published validation, which is to
say it is our reliability table with a letter on it. This is the daily signal
that one is not.

THE FACTOR: `bound_gap`, the distance between the ratchet bound (the coldest
reading so far) and the day-ahead guidance. It is large exactly when the two
disagree about where the day is going.

⚠️ THIS IS THE SAME COLUMN THAT FAILED AS A MODEL FEATURE, AND THAT IS NOT A
CONTRADICTION. `scripts/klax_low_gap_eval.py` asked "does giving the tree this
column raise the hit rate" and the answer was no (GAP1 +1.0%, CI
[-1.1%,+3.1%]). This asks "does this column tell me when to DISBELIEVE the
model", which is a different question with a different answer: a tree averaging
a weak signal over every day gains little, while a bettor who SITS OUT the worst
quarter of days keeps his money. Both results stand.

WHAT WAS MEASURED, and against what. The outcome is SHORTFALL -- stated
probability minus realised hit -- not accuracy. A hard day is fine to trade if
the model says it is hard; what costs money is the model staying confident while
being wrong. See `scripts/preflight_eval.py` for the pre-registered gate and
`scripts/preflight_robustness.py` for the three ways it was attacked afterwards.

  KLAX LOW h4, walk-forward, test >= 2025-01-01, n=573:

    band     gap          n    stated     hit   shortfall
    CLEAR    <= 1.62    287     66.1%   69.5%      -3.4%
    FAIR     1.62-2.53  143     63.4%   64.7%      -1.2%
    AVOID    > 2.53     143     57.2%   46.2%     +11.0%

  Pre-registered gate on the median split: Dshortfall +8.3%, 98.75% CI
  [+1.5%,+15.5%] (Bonferroni-corrected for four markets) -> SHIP.

  Then, before shipping:
    * NOT merely restating the model's own confidence. Within all three
      stated-probability terciles the flagged days still do worse (+16.0%,
      +9.7%, +2.2%). In the LOWEST tercile the model says ~50% either way while
      the truth is 46% flagged against 66% clean -- a 20-point spread it cannot
      see at all.
    * HOLDS OUT OF SAMPLE IN TIME. Threshold fit on 2025-01-01..2025-10-15 and
      applied untouched to 2025-10-16..2026-07-29: flagged 51.3% vs clean 70.9%,
      Dshortfall +12.5%, 95% CI [+3.0%,+22.7%].

⚠️ KLAX LOW ONLY, and the other three markets say so rather than guessing.

    KLAX HIGH  Dshortfall -3.6%, and the effect REVERSES -- clean days are the
               worse ones. Robustness agrees: 1/3 bins, fails out of sample.
    KMIA LOW   +7.5%, CI [-0.3%,+15.1%] -- misses the corrected gate by 0.3pt.
               Same direction, near-identical threshold (1.73 vs 1.62), same
               Q4-concentrated damage. It is the strongest corroboration the
               KLAX LOW result has, and it still does not ship, because a gate
               that bends for a near miss is not a gate.
    KMIA HIGH  +2.3%, CI [-5.0%,+9.5%]. Nothing.

  The split is not arbitrary: the LOWs have it and the HIGHs do not. The ratchet
  bound only becomes informative once the extremum has HAPPENED, and at the
  HIGH's h11 commit 88% of it is still ahead on every single day -- so there are
  no two regimes for the gap to tell apart. The signal appears exactly where the
  mechanism says it should, which is why this is not curve-fitting.

FREEZE STATUS: display only. It reads a stored column and returns words. It
computes no prediction and must never be given a vote in `staking.evaluate`,
`appbet`, or the commit decision -- if the app started skipping bets on this,
the scorecard would no longer be measuring the model.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from highcast.config import AppConfig
from highcast.gating import MIN_DECISION_HOUR

# (upper bound on |gap|, band name, measured hit, measured stated, n).
# The last band's bound is None, meaning "everything above".
#
# ⚠️ THESE ARE MEASURED CONSTANTS AND THEY WILL GO STALE, exactly like
# `gating.HOUR_ACCURACY`. The threshold already drifted across the test period
# (median 1.59 -> 1.66, Q3 2.45 -> 2.68 between halves), which is small enough
# to ship a fixed cut on and large enough that it must be re-measured rather
# than trusted forever. Re-run `scripts/preflight_eval.py` when the reliability
# table is next refreshed.
BANDS: dict[tuple[str, str], tuple] = {
    # ⚠️ EMPTY SINCE 2026-08-01, DELIBERATELY. The one shipped entry
    # (("KLAX","low")) was RETIRED, not lost.
    #
    # It was measured at h4 and cleared its gate convincingly: flagged stratum
    # shortfall +6.4%, clean −2.9%, Δ+9.3% CI [+1.6%, +15.9%], and it survived
    # three robustness attacks. Then the commit hours moved on 2026-08-01
    # (KLAX low h4 -> h6) and it was re-measured at the hour it would actually
    # be used:
    #
    #     flagged (gap > 1.19)  n=287  stated 65.1%  hit 65.3%  shortfall -0.3%
    #     clean   (gap <= 1.19) n=287  stated 69.6%  hit 70.6%  shortfall -1.0%
    #     Dshortfall +0.7%  CI [-5.7%, +6.8%]     GATE -> DO NOT SHIP
    #
    # BOTH strata now show NEGATIVE shortfall: the model under-states itself in
    # each. A preflight exists to catch days where the model stays CONFIDENT
    # WHILE BEING WRONG, and at h6 that behaviour has stopped -- the same reason
    # no HIGH preflight was ever buildable. The later hour is honest, so it IS
    # the preflight.
    #
    # THE DANGEROUS PART, and why this is empty rather than stale: this dict is
    # keyed (station, metric) with NO HOUR, while `grade()` reads the commit hour
    # from `gating.MIN_DECISION_HOUR`. So after the move it silently applied
    # h4-derived thresholds to h6 readings and kept printing CLEAR/FAIR/AVOID on
    # the dashboard. Nothing failed; the verdict just stopped meaning anything.
    # Re-measure with `scripts/preflight_eval.py` before adding any entry back,
    # and add the hour to the key if a future band ever ships.
}

# Plain-language reading of each band, written for the person deciding whether
# to pay a price -- not a description of the statistic.
BAND_NOTE = {
    "CLEAR": "The ratchet and the guidance agree. On days like this the model "
             "has hit its bucket 69.5% of the time — its best state.",
    "FAIR":  "Mild disagreement between the ratchet and the guidance. Roughly "
             "the market's average day at 64.7%.",
    "AVOID": "The ratchet and the guidance disagree sharply. Days like this hit "
             "46.2% while the model claims 57.2% — its confidence is worth "
             "about 11 points less than it says.",
}


def band_for(station: str, metric: str, gap: float | None) -> tuple | None:
    """The measured band containing `gap`, or None where nothing was measured.

    No cross-market fallback: a market without its own entry gets None, never a
    neighbour's numbers. Three of the four markets genuinely have no effect
    here, and one of them has it REVERSED, so borrowing would not merely be
    unmeasured -- it would be wrong in a known direction.
    """
    bands = BANDS.get((station, metric))
    if bands is None or gap is None:
        return None
    g = abs(gap)
    for upper, name, hit, stated, n in bands:
        if upper is None or g <= upper:
            return (name, hit, stated, n)
    return None


def stored_gap(conn, cfg: AppConfig, metric: str, target_date: str,
               commit_hour: int | None = None) -> dict | None:
    """The `bound_gap` recorded with a prediction for `target_date`.

    Prefers the prediction issued DURING the commit hour, because that is the
    only hour the bands were measured at -- the gap's distribution at h20 is not
    the one these thresholds were fitted on. Falls back to the most recent
    intraday prediction, flagged `at_commit=False` so the caller can label the
    verdict provisional rather than quietly presenting an off-hour reading as
    the measured one.

    Returns None when no prediction carries a gap at all, which is the normal
    state for any day predicted before the column shipped.
    """
    rows = conn.execute(
        """SELECT issued_at_utc, bound_gap FROM predictions
           WHERE station=? AND target_date=? AND metric=? AND kind='intraday'
             AND bound_gap IS NOT NULL
           ORDER BY issued_at_utc DESC""",
        (cfg.station.icao, target_date, metric)).fetchall()
    if not rows:
        return None
    if commit_hour is not None:
        tz = ZoneInfo(cfg.station.tz)
        for r in rows:
            local = dt.datetime.fromisoformat(r["issued_at_utc"]).astimezone(tz)
            if local.hour == commit_hour:
                return {"gap": r["bound_gap"], "at_commit": True,
                        "issued_at_utc": r["issued_at_utc"]}
    return {"gap": rows[0]["bound_gap"], "at_commit": False,
            "issued_at_utc": rows[0]["issued_at_utc"]}


def grade(conn, cfg: AppConfig, metric: str, target_date: str) -> dict:
    """Today's trade-quality verdict for one market. Pure read; display only.

    `status` is what the caller should branch on:
        graded       a measured band applies
        provisional  a band applies but the gap is from an off-commit hour
        pending      this market is measured, but today has no gap recorded yet
        unmeasured   no band exists for this market, and none is invented
    """
    icao = cfg.station.icao
    commit_hour = MIN_DECISION_HOUR.get(icao, {}).get(metric)
    out = {"station": icao, "metric": metric, "target_date": target_date,
           "commit_hour": commit_hour, "gap": None, "band": None,
           "measured_hit": None, "measured_stated": None, "n": None,
           "note": None, "status": "unmeasured"}
    if (icao, metric) not in BANDS:
        out["note"] = ("No daily signal has been measured for this market — the "
                       "bound/guidance gap was tested here and did not separate "
                       "good days from bad. Use the market's static record.")
        return out
    rec = stored_gap(conn, cfg, metric, target_date, commit_hour)
    if rec is None:
        out["status"] = "pending"
        out["note"] = ("No gap recorded for this day yet — the verdict lands "
                       f"with the h{commit_hour} prediction.")
        return out
    b = band_for(icao, metric, rec["gap"])
    if b is None:
        out["status"] = "pending"
        out["note"] = "Gap recorded but outside every measured band."
        return out
    name, hit, stated, n = b
    out.update({"gap": rec["gap"], "band": name, "measured_hit": hit,
                "measured_stated": stated, "n": n,
                "issued_at_utc": rec["issued_at_utc"],
                "status": "graded" if rec["at_commit"] else "provisional",
                "note": BAND_NOTE[name]})
    if not rec["at_commit"]:
        out["note"] += (f" Provisional: this gap is not from the h{commit_hour} "
                        "commit hour, which is the only hour the bands were "
                        "measured at.")
    return out
