"""DAY GRADE — was this nowcast computed from clean, complete, current inputs?

⚠️ READ THIS BEFORE USING THE GRADE FOR ANYTHING.

**THE GRADE DESCRIBES INPUT QUALITY. IT DOES NOT PREDICT ACCURACY.** A grade-A
day whose model says 55% is still a coin flip, and an A on a 96% is not more
likely to hit than the 96% already says. Conflating the two is exactly how the
confidence gate inverted P&L: it filtered to "good" days, the ask climbed from
31c to 60c to meet the confidence, and the filter ate the entire edge.

WHY IT IS NOT PREDICTIVE, measured 2026-08-15 in `scripts/day_grade_gate.py`.
Every plausible day-quality factor was tested for separation of hit rate WITHIN
stated-probability bands, over ~585 walk-forward days per market:

    factor            KLAX high     KLAX low    KMIA high        KMIA low
    ratchet pinned    +6.8,+9.5       -10.4   -13.3,-11.0,-5.5     +18.2
    guidance spread       -10.6       -15.5     +13.3 / -6.8       +13.9

**The sign flips between markets and inside them.** Pinned helps KLAX high and
hurts KMIA high, consistently, in opposite directions. That is noise being
harvested by ~40 comparisons at a +-5% threshold, where the standard error on a
typical cell is ~7 points. Two of this project's four dead ideas died the same
way, and the fourth (the near-boundary warning) died for exactly this reason:
the model ALREADY KNOWS, and `stated` alone separates 42.6% from 84.1%.

**So: `stated` is the accuracy grade. This is the plumbing grade.** They answer
different questions and must never be added together.

WHAT IT IS FOR. "Can I trust that this number was computed from what it should
have been computed from?" Every check below is a DETERMINISTIC FACT — a feed is
fresh or it is not, a six-hour group filed or it did not. Nothing here is fitted,
so nothing here can be overfit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from highcast import gating, latentmax, marketsensor, staleness
from highcast.config import AppConfig
from highcast.metric import get_metric

# Severity ladder. FAIL means the number should not be read at all; WARN means
# read it knowing something is degraded; INFO is context that never moves the
# grade — see `_LETTERS`.
FAIL, WARN, INFO, OK = "fail", "warn", "info", "ok"

_LETTERS = {
    # any FAIL -> F, any WARN -> C, only INFO flags -> B, nothing -> A
    "F": "inputs are missing or stale enough that the number is not trustworthy",
    "C": "usable, but something material is degraded",
    "B": "clean inputs, with context worth knowing",
    "A": "clean",
}


@dataclass
class Check:
    name: str
    level: str
    detail: str = ""


@dataclass
class Grade:
    letter: str
    station: str
    metric: str
    day: str
    checks: list[Check] = field(default_factory=list)

    @property
    def summary(self) -> str:
        bad = [c for c in self.checks if c.level in (FAIL, WARN)]
        tail = "; ".join(f"{c.name}: {c.detail}" for c in bad) if bad else "clean"
        return f"{self.letter} — {tail}"

    def as_dict(self) -> dict:
        return {"letter": self.letter, "station": self.station,
                "metric": self.metric, "day": self.day,
                "meaning": _LETTERS[self.letter],
                "checks": [{"name": c.name, "level": c.level,
                            "detail": c.detail} for c in self.checks],
                "disclaimer": ("Input quality only. This does NOT predict "
                               "accuracy — the model's own probability does.")}


def _letter(checks: list[Check]) -> str:
    levels = {c.level for c in checks}
    if FAIL in levels:
        return "F"
    if WARN in levels:
        return "C"
    if INFO in levels:
        return "B"
    return "A"


def grade(conn, cfg: AppConfig, metric: str, day: _date | None = None,
          now: datetime | None = None) -> Grade:
    """Grade one market's inputs for one day. Never raises."""
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.station.tz)
    day = day or now.astimezone(tz).date()
    icao = cfg.station.icao
    spec = get_metric(cfg, metric)
    hour = gating.MIN_DECISION_HOUR[icao][metric]
    checks: list[Check] = []

    def add(name, level, detail=""):
        checks.append(Check(name, level, detail))

    # ── 1. are the feeds delivering? ────────────────────────────────────────
    try:
        bad = staleness.stale(conn, cfg, now)
        if bad:
            add("feeds", FAIL,
                ", ".join(f"{b['label']} stale" for b in bad))
        else:
            add("feeds", OK)
    except Exception as e:                     # noqa: BLE001
        add("feeds", WARN, f"check failed: {type(e).__name__}")

    # ── 2. did the desk actually commit? ────────────────────────────────────
    try:
        row = conn.execute(
            """SELECT placed_local_hour FROM app_bets WHERE station=?
                 AND metric=? AND target_date=?""",
            (icao, metric, day.isoformat())).fetchone()
        local_hour = now.astimezone(tz).hour
        if row:
            add("commit", OK, f"recorded at h{row['placed_local_hour']}")
        elif local_hour < hour:
            add("commit", INFO, f"not due yet (commits at h{hour})")
        else:
            add("commit", FAIL, f"h{hour} passed with no bet recorded")
    except Exception as e:                     # noqa: BLE001
        add("commit", WARN, f"check failed: {type(e).__name__}")

    # ── 3. are we blind relative to the market RIGHT NOW? ───────────────────
    # This is QUEUE item 13 observable live: the market has priced an extreme
    # our own observations cannot yet reach.
    try:
        ladder, when = marketsensor._latest_ladder(conn, icao, metric,
                                                   day.isoformat())
        if not ladder:
            add("blindspot", INFO, "no market snapshot yet")
        else:
            d = marketsensor.divergence(conn, cfg, spec, day, ladder, when)
            if d:
                add("blindspot", WARN,
                    f"market says {d['label']}, we can only see {d['seen']}F "
                    f"— blind by {d['gap_f']:.0f}F")
            else:
                add("blindspot", OK)
    except Exception as e:                     # noqa: BLE001
        add("blindspot", WARN, f"check failed: {type(e).__name__}")

    # ── 4. did every six-hour remark group we were entitled to arrive? ──────
    # ⚠️ USE `missing_groups`, NOT a minutes subtraction. `covered_minutes` can
    # EXCEED `day_minutes` because a six-hour window overruns the as-of horizon
    # (1080 vs 720 is normal), so `day_minutes - covered_minutes` goes negative
    # and is meaningless. That mistake produced a spurious "+17.4% separation"
    # in the first draft of `scripts/day_grade_gate.py`.
    try:
        # ⚠️ AS OF min(now, commit) — NOT the commit hour unconditionally.
        #
        # Evaluating at the commit hour while the clock is still hours short of
        # it reported groups as "never filed" whose filing time had not yet
        # ARRIVED. On 2026-08-15 at 09:11Z this said KMIA high was missing the
        # 12:00Z group — two hours in the future. That reads as a fault when it
        # is a forecast of one, and it contradicted the `commit` check sitting
        # right beside it, which correctly says "not due yet".
        #
        # Before the commit hour: judge only what is actually due.
        # At or after it: freeze at the commit hour, because the grade describes
        # the information the DECISION was made with, not what arrived later.
        commit_at = datetime(day.year, day.month, day.day, hour,
                             tzinfo=tz).astimezone(timezone.utc)
        asof = min(now, commit_at)
        cov = latentmax.six_hour_coverage(conn, cfg, day, spec.extremum,
                                          asof_utc=asof.isoformat()) or {}
        missing = cov.get("missing_groups") or []
        pending = cov.get("pending_groups") or []
        early = now < commit_at
        if missing:
            when = ", ".join(m[11:16] + "Z" for m in missing[:3])
            add("six_hour_groups", WARN,
                f"{len(missing)} group(s) overdue ({when})")
        elif early and pending:
            add("six_hour_groups", INFO,
                f"{len(pending)} group(s) not due yet; "
                f"judged as of {asof.strftime('%H:%M')}Z")
        else:
            add("six_hour_groups", OK)
    except Exception as e:                     # noqa: BLE001
        add("six_hour_groups", WARN, f"check failed: {type(e).__name__}")

    # ── 5. is the stated probability inside a MEASURED calibration band? ────
    try:
        r = conn.execute(
            """SELECT p_model FROM app_bets WHERE station=? AND metric=?
                 AND target_date=?""",
            (icao, metric, day.isoformat())).fetchone()
        if r is None or r["p_model"] is None:
            add("calibrated", INFO, "no committed probability yet")
        elif gating.band_calib(icao, metric, hour, float(r["p_model"])) is None:
            add("calibrated", WARN,
                f"{float(r['p_model']):.0%} sits outside every measured band")
        else:
            add("calibrated", OK)
    except Exception as e:                     # noqa: BLE001
        add("calibrated", WARN, f"check failed: {type(e).__name__}")

    # ── 6. external factors: an active advisory ─────────────────────────────
    # ⚠️ The columns are `onset_utc` / `expires_utc`. The first version guessed
    # `effective_utc`, which does not exist — and a bare `except: add(OK)`
    # swallowed the error, so this check silently reported CLEAN on every day
    # forever. That is the same "guard that cannot fire" failure as the CLI
    # staleness check, and it is why every other branch here reports WARN on
    # exception instead of pretending success.
    try:
        rows = conn.execute(
            """SELECT event FROM alerts WHERE station=?
                 AND expires_utc > ? AND onset_utc <= ?""",
            (icao, now.isoformat(), now.isoformat())).fetchall()
        if rows:
            names = ", ".join(sorted({r["event"] for r in rows}))
            add("advisory", INFO, f"{len(rows)} active: {names}")
        else:
            add("advisory", OK)
    except Exception as e:                     # noqa: BLE001
        add("advisory", WARN, f"check failed: {type(e).__name__}")

    # ── 7. CONTEXT ONLY — never moves the grade above INFO ──────────────────
    # How much of the extreme is still ahead at commit is a property of the
    # MARKET AND HOUR, not of this day, and it was measured NOT to predict
    # accuracy within a stated band. It is reported because it explains what the
    # ratchet can and cannot know — see HANDOFF §3b.
    sf = gating.EXTREMUM_STILL_FUTURE.get(icao, {}).get(metric, {}).get(hour)
    if sf is not None and sf >= 0.60:
        add("still_ahead", INFO,
            f"{sf:.0%} of the time this extreme has not happened by h{hour} — "
            f"the ratchet cannot help here")

    return Grade(_letter(checks), icao, metric, day.isoformat(), checks)


def grade_all(conn, cfg: AppConfig, day: _date | None = None,
              now: datetime | None = None) -> list[Grade]:
    return [grade(conn, cfg, m, day, now) for m in ("high", "low")]


def wake_line(conn, cfg: AppConfig, now: datetime | None = None) -> str:
    gs = grade_all(conn, cfg, None, now)
    return "  day grade: " + "   ".join(
        f"{g.metric} {g.summary}" for g in gs)
