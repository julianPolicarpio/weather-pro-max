"""SHOULD A COMMIT HOUR MOVE EARLIER ON PRICE? The gate the accuracy gate lacks.

⚠️⚠️ READ `scripts/earlier_high_eval.py` FIRST — the Phase 0 replay ledger.
Julian ran it 2026-08-27 (status VERIFIED) and it answers the ACCURACY half of
"can the HIGH commit earlier" far more rigorously than this file does. Its
verdict, across 40 station x hour x bundle cells at n~595-601 and ~100% coverage:

    "No station/hour/bundle combination cleared every pre-registered gate."

Every cell REJECT. Four METAR feature bundles (B1 humidity/dewpoint, B2 wind,
B3 rain, B4 station-specific marine/convective) at h9-h13, both stations. And at
KMIA h13 the replay arms are SIGNIFICANTLY WORSE than the shipped model --
dHIT vs shipped -0.057 [-0.097,-0.015] for B1, -0.057 [-0.098,-0.014] for B3.

**So the accuracy route to an earlier hour is closed.** No feature set recovers
what is lost by committing before the extremum forms.

THIS FILE IS NOT REDUNDANT, AND THE REASON IS NARROW: Phase 0 contains **zero
price terms** (`grep -c 'yes_ask|market_price|margin'` -> 0). It scores dHIT,
CRPS and calibration gap. The open question it leaves is whether the CHEAPER ASK
at an earlier hour could pay for accuracy Phase 0 has now shown is unrecoverable.
That is a different question with a different answer, and it is this file's only
justification for existing.

⚠️ WHERE PHASE 0 IS STRONGER, and this file should be read in its shadow: it
uses a THREE-SEED random control family with `noise_delta_hit_p975` and a
`random_family_contaminated` flag, Holm-adjusted p across the whole family, an
n>=400 and coverage>=0.80 floor, a CRPS guard, a calibration-gap guard, and
agreement between full-history and warm-season slices. This file has one random
arm and one bootstrap CI. If the two ever disagree, believe Phase 0.

WHY THIS EXISTS. `scripts/decision_hour_move_eval.py` moved every hour LATER on
2026-07-31 and its own comment names its blind spot: **the gate has no price
term**. That is why KMIA HIGH's h14 -> h13 move had to be argued OUTSIDE it —
h14 was more accurate (79.3% vs 68.0%) and still lost money, because the ask was
0.82 against 0.46. A contract at price P needs a hit rate of P to break even, so
accuracy bought above its own price is not an improvement.

Julian asked (2026-08-23) whether KLAX could commit EARLIER. The accuracy curve
says no, loudly: KLAX HIGH bucket hit is h13 69.5%, h12 60.3%, h11 53.4%,
h10 49.0% (HOUR_ACCURACY, n=1,959). But the ASK climbs just as steeply, so the
question is genuinely open on margin and cannot be answered by the old gate.

⚠️⚠️ THIS IS NOT PRE-REGISTERED AND I WILL NOT PRETEND IT IS. I ran the
hit-vs-ask table BEFORE writing this file and saw h13 at -5% margin against h12
at +14% and h10 at +15%. Those numbers motivated the gate, so they cannot also
validate it. Everything dated on or before 2026-08-23 is EXPLORATORY and is
printed as such; the verdict is computed only on days after `OOS_START`.

────────────────────────────────────────────────────────────────────────────
THE DECISION STATISTIC
────────────────────────────────────────────────────────────────────────────

Per day, at a given local hour, on the LIVE stored Kalshi ladder:

    margin = 1{the model's top rung contains the settled value} - that rung's ask

Both terms come from the SAME object at the SAME instant. ⚠️ Do NOT mix a
walk-forward `expected_bucket_hit` with a live ask: the first is a parity-
averaged expectation over a synthetic ladder and the second is the price of one
specific labelled rung. They are different objects and subtracting them is the
`band_calib`-across-rungs mistake in another costume (error #5).

Candidate and incumbent hours are compared PAIRED on the same days.

────────────────────────────────────────────────────────────────────────────
FOUR ARMS. Arm 0 can VOID the result; it is not a tiebreak.
────────────────────────────────────────────────────────────────────────────

  0. REPRESENTATIVE  the observed hit rate at each hour, over the priced days,
     must be consistent with the long-run `HOUR_ACCURACY` curve for that hour
     (two-sided binomial, alpha 0.05). ⚠️ THIS ARM EXISTS BECAUSE THE PILOT
     FAILED IT: the priced window showed KLAX HIGH at 81% at h12 against a
     long-run 60.3%, and 52% at h11 sandwiched between 64% at h10 and 81% at
     h12. Hit rate must rise monotonically as the max forms; it did not. A
     window that cannot reproduce the known curve cannot be trusted to price it
     either, and a margin computed on it is arithmetic on noise.

  1. MARGIN       paired mean(margin_candidate - margin_incumbent) > 0 AND the
     bootstrap CI excludes zero at Bonferroni x(number of candidate hours).

  2. STILL A FORECAST  >= 40% of the extremum still ahead at the candidate hour
     (`EXTREMUM_STILL_FUTURE`), carried over from the accuracy gate so the desk
     cannot drift into transcribing an answer it can already read.

  3. CALIBRATION EXISTS  `BAND_CALIB` and `HOUR_CALIB` must have an entry at
     (station, metric, candidate_hour). They are hour-keyed with NO cross-hour
     fallback, so moving to an unmeasured hour makes `staking.evaluate` refuse
     every bet with "unmeasured confidence". This arm cannot be satisfied by
     analysis — it is a re-measurement job — so it is reported as a SHIPPING
     PRECONDITION rather than something the data can pass.

────────────────────────────────────────────────────────────────────────────
STOPPING RULE — and the reason this gate will not answer soon
────────────────────────────────────────────────────────────────────────────

Margin is `1{hit} - ask`, so its per-day SD is ~0.46-0.53 (measured, not
assumed). Powering the paired difference at 80%:

    to detect +0.15   ~75-99 paired days
    to detect +0.10   ~168-221 paired days
    to detect +0.05   ~670-884 paired days

**N_MIN = 170**, powered for +0.10. At one KLAX HIGH day-market per day that is
roughly SIX MONTHS from `OOS_START`. That is the honest cost of asking a price
question with 24 days of price history, and it is why the pilot's +14% must not
be acted on: the same arithmetic says a +14% point estimate at n=22 is entirely
consistent with zero.

⚠️ DO NOT POOL THE FOUR MARKETS TO REACH n FASTER unless you first show the
effect is common across them. Their hours, ladders and ask curves differ; a
pooled +0.10 could be one market's +0.40 and three markets' zero.

Writes no constant. Its output is read by a human who then edits
`gating.MIN_DECISION_HOUR` — and re-measures BAND_CALIB/HOUR_CALIB first.

Usage: PYTHONPATH=src .venv/bin/python scripts/decision_hour_price_gate.py
"""
import json
import re
import sqlite3
from datetime import datetime
from math import comb, sqrt
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

import numpy as np

from highcast.config import load_config
from highcast.db import connect
from highcast.gating import (BAND_CALIB, EXTREMUM_STILL_FUTURE, HOUR_ACCURACY,
                             HOUR_CALIB, MIN_DECISION_HOUR)

# ── FROZEN 2026-08-23 ────────────────────────────────────────────────────────
OOS_START = "2026-08-24"
N_MIN = 170
N_BOOT = 4000
MIN_STILL_AHEAD = 0.40
ALPHA = 0.05
TZ = {"KLAX": "America/Los_Angeles", "KMIA": "America/New_York"}
CANDIDATES = {("KLAX", "high"): [9, 10, 11, 12],
              ("KLAX", "low"): [3, 4, 5],
              ("KMIA", "high"): [10, 11, 12],
              ("KMIA", "low"): [4, 5, 6]}
rng = np.random.default_rng(20260823)


def binom_p(k, n, p):
    """Two-sided exact binomial test of k successes in n against rate p."""
    if n == 0:
        return 1.0
    obs = comb(n, k) * p ** k * (1 - p) ** (n - k)
    tot = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * p ** i * (1 - p) ** (n - i)
        if pi <= obs * (1 + 1e-9):
            tot += pi
    return min(1.0, tot)


def contains(label, v):
    m = re.match(r"^(\d+)[–-](\d+)$", label)
    if m:
        return int(m.group(1)) <= v <= int(m.group(2))
    m = re.match(r"^[≤<]=?(\d+)$", label)
    if m:
        return v <= int(m.group(1))
    m = re.match(r"^[≥>]=?(\d+)$", label)
    if m:
        return v >= int(m.group(1))
    return None


def per_day_margin(icao, metric):
    """{(target_date, local_hour): (margin, hit)} from the LIVE stored ladder.

    First snapshot in each local hour, matching how the KMIA ask-cliff was
    measured so the two are comparable.
    """
    cfg = load_config(f"config/{icao.lower()}.yaml")
    conn = connect(cfg.db_path, migrate=False)
    conn.row_factory = sqlite3.Row
    col = "high_f" if metric == "high" else "low_f"
    truth = {r["date"]: r[col] for r in conn.execute(
        f"SELECT date,{col} FROM cli_reports WHERE station=? AND {col} IS NOT NULL",
        (icao,))}
    tz = ZoneInfo(TZ[icao])
    out = {}
    for r in conn.execute(
            """SELECT target_date, created_utc, market_json FROM signals
               WHERE station=? AND metric=? AND market_json != ''
               ORDER BY created_utc""", (icao, metric)):
        t = r["target_date"]
        if t not in truth:
            continue
        h = datetime.fromisoformat(r["created_utc"]).astimezone(tz).hour
        if (t, h) in out:
            continue
        try:
            lad = [x for x in json.loads(r["market_json"])
                   if x.get("p_model") is not None and x.get("yes_ask") is not None]
        except ValueError:
            continue
        if not lad:
            continue
        top = max(lad, key=lambda x: x["p_model"])
        hit = contains(top["label"], truth[t])
        if hit is None:
            continue
        out[(t, h)] = ((1.0 if hit else 0.0) - top["yes_ask"], 1 if hit else 0)
    return out


def report(icao, metric, oos_only):
    inc = MIN_DECISION_HOUR[icao][metric]
    cands = CANDIDATES.get((icao, metric), [])
    data = per_day_margin(icao, metric)
    if oos_only:
        data = {k: v for k, v in data.items() if k[0] >= OOS_START}
    tag = "OUT-OF-SAMPLE" if oos_only else "EXPLORATORY (already seen — no verdict)"
    print(f"\n{'='*78}\n{icao} {metric.upper()}  incumbent h{inc}   [{tag}]\n{'='*78}")

    # ── ARM 0: does this window reproduce the known accuracy curve? ─────────
    print("  arm 0 — representativeness vs HOUR_ACCURACY")
    void = False
    for h in sorted(set(cands + [inc])):
        days = [k for k in data if k[1] == h]
        if len(days) < 8:
            print(f"    h{h:<3} n={len(days):<3} too few to test")
            continue
        k = sum(data[d][1] for d in days)
        ref = HOUR_ACCURACY.get(icao, {}).get(metric, {}).get(h)
        if ref is None:
            print(f"    h{h:<3} no HOUR_ACCURACY reference")
            continue
        p_ref = ref[1]
        p = binom_p(k, len(days), p_ref)
        bad = p < ALPHA
        void = void or bad
        print(f"    h{h:<3} n={len(days):<3} observed {k/len(days):>5.0%}  "
              f"long-run {p_ref:>5.0%}  binom p={p:.3f}"
              f"{'   <-- INCONSISTENT' if bad else ''}")
    if void:
        print("    -> ⚠️ WINDOW IS NOT REPRESENTATIVE. Arm 1 below is arithmetic")
        print("       on a sample that cannot reproduce a curve measured on ~1,900")
        print("       days. The result is VOID, not 'passed' and not 'failed'.")

    # ── ARM 1: paired margin ────────────────────────────────────────────────
    print(f"\n  arm 1 — paired margin vs h{inc}"
          f"   (Bonferroni x{max(len(cands),1)})")
    a = ALPHA / max(len(cands), 1)
    pcts = [100 * a / 2, 100 * (1 - a / 2)]
    best = None
    for h in cands:
        days = sorted({t for (t, hh) in data if hh == h}
                      & {t for (t, hh) in data if hh == inc})
        if len(days) < 5:
            print(f"    h{h:<3} n={len(days)} — too few paired days")
            continue
        d = np.array([data[(t, h)][0] - data[(t, inc)][0] for t in days])
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(axis=1), pcts)
        ok = d.mean() > 0 and (lo > 0)
        ahead = EXTREMUM_STILL_FUTURE.get(icao, {}).get(metric, {}).get(h)
        cal = (((icao, metric, h) in BAND_CALIB)
               and HOUR_CALIB.get(icao, {}).get(metric, {}).get(h) is not None)
        print(f"    h{h:<3} n={len(days):<3} dMargin {d.mean():+.3f}  "
              f"CI [{lo:+.3f},{hi:+.3f}]  ahead={ahead if ahead is None else f'{ahead:.0%}'}  "
              f"calib={'yes' if cal else 'NO'}"
              f"{'   arm1 PASS' if ok else ''}")
        if ok and (best is None or d.mean() > best[1]):
            best = (h, d.mean(), ahead, cal)

    n_pairs = max((len(sorted({t for (t, hh) in data if hh == h}
                              & {t for (t, hh) in data if hh == inc}))
                   for h in cands), default=0)
    print(f"\n  VERDICT: ", end="")
    if not oos_only:
        print("none — exploratory window, by construction.")
    elif n_pairs < N_MIN:
        wk = (N_MIN - n_pairs) / 7.0
        print(f"⏳ NOT DECIDABLE — {n_pairs}/{N_MIN} paired OOS days "
              f"(~{wk:.0f} weeks to go at 1/day).")
    elif void:
        print("VOID — arm 0 failed; re-run when the window is representative.")
    elif best is None:
        print(f"KEEP h{inc} — no candidate beat it on margin.")
    else:
        h, m, ahead, cal = best
        okf = ahead is not None and ahead >= MIN_STILL_AHEAD
        print(f"h{h} beats h{inc} by {m:+.3f} on margin; "
              f"still-a-forecast {'PASS' if okf else 'FAIL'}; "
              f"calibration {'present' if cal else 'MISSING — re-measure first'}")


print(__doc__.split("Usage:")[0])
for (icao, metric) in CANDIDATES:
    report(icao, metric, oos_only=False)
    report(icao, metric, oos_only=True)
