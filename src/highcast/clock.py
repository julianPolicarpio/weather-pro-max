from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from highcast.config import AppConfig
from highcast.timeutil import climate_day_bounds


def _fmt(dt: datetime, tz: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).strftime("%a %H:%M")


def events(cfg: AppConfig, day: date, metric: str = "high") -> list[dict]:
    """Climate-day milestones. Metric-aware because the two markets care about
    opposite ends of the day: the HIGH forms during afternoon peak heating,
    while the LOW is already set by dawn and the afternoon is irrelevant to it.
    Labelling the 12:00-16:00 block "Peak heating" in LOW mode read as though
    the low were still forming then."""
    la = ZoneInfo(cfg.station.tz)
    et = ZoneInfo("America/New_York")
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    prev = day - timedelta(days=1)
    nxt = day + timedelta(days=1)

    def at(d: date, h: int, m: int = 0, tz=la) -> datetime:
        return datetime(d.year, d.month, d.day, h, m, tzinfo=tz)

    if metric == "low":
        # The LOW's day-ahead cutoff IS the prior 20:00, so a separate "00Z runs
        # land" entry would land on the same instant; fold the two together. Same
        # for dawn vs "morning obs window opens" -- for the LOW they are one
        # moment, and showing both read as two distinct milestones.
        # No separate "overnight cooling begins" row: in winter LST == PST, so
        # at(day, 0) IS the climate-day start and the two collapse onto the same
        # tick. The climate-day-start row already marks that moment.
        specific = [
            ("00Z/06Z runs land / day-ahead sharpest", at(prev, 20)),
            ("Dawn — low typically locks in", at(day, 6)),
            ("Low decided (afternoon drives the high, not the low)", at(day, 12)),
        ]
    else:
        specific = [
            ("00Z model runs land", at(prev, 20)),
            ("Morning obs window opens", at(day, 6)),
            ("12Z model runs land / day-ahead sharpest", at(day, 8)),
            ("Peak heating begins", at(day, 12)),
            ("Peak heating ends", at(day, 16)),
        ]

    raw = [
        ("Climate day starts (midnight LST)", start),
        *specific,
        ("Preliminary climate report", at(day, 16, 35)),
        ("Market close", at(day, 23, 59)),
        ("Climate day ends", end),
        ("Settlement", at(nxt, 10, 0, tz=et)),
    ]
    raw.sort(key=lambda nt: nt[1])
    now = datetime.now(timezone.utc)
    return [{"name": n, "la": _fmt(t, cfg.station.tz),
             "manila": _fmt(t, cfg.display_tz), "done": t < now}
            for n, t in raw]


# REMOVED (2026-07-26): a hardcoded HIGH-only CRPS table from 2026-07-18, with
# the LOW left empty ("until measured"). It was stale within days and could not
# answer the question that actually matters -- WHEN TO PLACE -- because CRPS alone
# says nothing about whether the extremum has already happened. Replaced by
# `gating.timing_curve()`, which serves measured per-hour bucket accuracy AND the
# share of days the extremum is still ahead, for both metrics, from
# `scripts/hour_curve.py`. Re-measure there, not here.

TIMING_DISCLAIMER = (
    "Timing reflects model accuracy only — market prices may already reflect "
    "the same information. Trade decisions are yours."
)


def _fmt_clock(dt: datetime, tz: str) -> str:
    """12-hour lowercase wall-clock time, e.g. '4:00pm'."""
    return dt.astimezone(ZoneInfo(tz)).strftime("%-I:%M%p").lower()


def _fmt_range(lo: datetime, hi: datetime, tz: str) -> str:
    """24-hour wall-clock range, e.g. '23:00–01:00'."""
    f = "%H:%M"
    return (f"{lo.astimezone(ZoneInfo(tz)).strftime(f)}"
            f"–{hi.astimezone(ZoneInfo(tz)).strftime(f)}")


def timing_guide(cfg: AppConfig, now: datetime, metric: str = "high") -> dict:
    """Pure function describing what phase of the climate day `now` (UTC) falls
    in, station-locally, and what's diagnostically useful to know right now.

    The climate day runs local-standard-midnight to local-standard-midnight,
    which is 01:00->01:00 local during DST (see climate_day_bounds) — so the
    phase boundaries below are derived from that function rather than
    hardcoded, and hold correctly year-round.
    """
    la = ZoneInfo(cfg.station.tz)
    local_now = now.astimezone(la)
    day = local_now.date()
    start, end = climate_day_bounds(day, cfg.station.std_utc_offset)
    if now < start:
        # local wall clock has rolled to a new calendar date but the climate
        # day (which starts at 01:00 local during DST) hasn't begun yet —
        # we're still in the tail of yesterday's climate day.
        day = day - timedelta(days=1)
        start, end = climate_day_bounds(day, cfg.station.std_utc_offset)

    def at(h: int, m: int = 0) -> datetime:
        return datetime(day.year, day.month, day.day, h, m, tzinfo=la)

    b6, b9, b12, b16, b1635 = at(6), at(9), at(12), at(16), at(16, 35)

    # Placement recommendation + phase narrative. The HIGH forms in the LA
    # afternoon (Manila small hours — Julian asleep); the LOW forms around dawn
    # (~6am LA = ~9pm Manila), i.e. during Julian's evening, so he is awake while
    # it forms. Boundaries are climate-day-relative (they persist through the
    # 00:00–01:00 local tail during DST). The high branch is byte-identical to
    # the original single-metric logic.
    if metric == "low":
        if now < b6:
            recommendation = {
                "target_date": day.isoformat(),
                "window_manila": "now", "window_la": "now",
                "action": "Today's low is forming NOW — place while it's live",
                "why": ("The daily minimum sets around dawn; LA overnight is "
                        "your Manila afternoon/evening, so you can watch the "
                        "running-min ratchet drop in real time — a better fit "
                        "to your hours than the high."),
            }
        else:
            recommendation = {
                "target_date": (day + timedelta(days=1)).isoformat(),
                "window_manila": _fmt_range(at(19), at(21), cfg.display_tz),
                "window_la": _fmt_range(at(19), at(21), cfg.station.tz),
                "action": ("Today's low is set — place tomorrow's low on the "
                           "evening guidance"),
                "why": ("Tomorrow's low forms next dawn; its guidance-based "
                        "read sharpens through this evening."),
            }
        phases = [
            ("overnight", start, b6,
             "The low is forming NOW — and you're awake for it",
             "The daily minimum sets around dawn; LA overnight is your Manila "
             "afternoon/evening. The running-min ratchet is a one-way floor.",
             "Dawn — low locks in", b6),
            ("morning_obs", b6, b9,
             "Low locked in at dawn",
             "~99% of days settle at the visible running min once the sun is up.",
             "Day-ahead cutoff (9am)", b9),
            ("late_morning", b9, b12,
             "Today's low is decided",
             "Attention shifts to tomorrow's low (forms next dawn).",
             "Peak heating begins (drives the high)", b12),
            ("peak_heating", b12, b16,
             "Today's low is decided",
             "The afternoon drives the HIGH, not the low — today's low set at dawn.",
             "Peak heating ends", b16),
            ("post_peak", b16, b1635,
             "Today's low long decided",
             "The low locked at dawn; the day is about the high now.",
             "Preliminary climate report", b1635),
            ("report_out", b1635, end,
             "Preliminary report is out — today's low is decided",
             "Attention shifts to the next day's low market.",
             "Climate day starts", end),
        ]
    else:
        if now < b9:
            recommendation = {
                "target_date": day.isoformat(),
                "window_manila": _fmt_range(at(8), at(10), cfg.display_tz),
                "window_la": _fmt_range(at(8), at(10), cfg.station.tz),
                "action": "Wait for the 9am refresh, then place today positions",
                "why": ("The 9am-cutoff day-ahead (morning obs + latest guidance) "
                        "is the sharpest read that still precedes formation — "
                        "after noon LA the market converges with the thermometer."),
            }
        elif now < b1635:
            recommendation = {
                "target_date": day.isoformat(),
                "window_manila": "now",
                "window_la": "now",
                "action": ("Today's window is open — the 9am read is live and "
                           "formation begins at noon LA"),
                "why": ("Odds decay fastest during peak heating (3–7am Manila); "
                        "earlier in this window preserves them."),
            }
        else:
            recommendation = {
                "target_date": (day + timedelta(days=1)).isoformat(),
                "window_manila": _fmt_range(at(17), at(19), cfg.display_tz),
                "window_la": _fmt_range(at(17), at(19), cfg.station.tz),
                "action": ("Today is decided — place tomorrow positions on the "
                           "evening guidance forecast"),
                "why": ("Tomorrow's guidance-based forecast generates at 5pm LA; "
                        "it lacks morning obs, so it's wider than the "
                        "midnight-Manila refresh will be — earliest read, best "
                        "odds, most uncertainty."),
            }
        phases = [
            ("overnight", start, b6,
             "Early tells from overnight obs",
             "Overnight temperature reveals marine-layer strength long before the "
             "day forms; intraday nowcast already updating.",
             "Morning obs window opens", b6),
            ("morning_obs", b6, b9,
             "Morning obs sharpening the picture",
             "At 9am the day-ahead model's data cutoff lands (sharpest "
             "pre-afternoon read).",
             "Day-ahead cutoff (9am)", b9),
            ("late_morning", b9, b12,
             "Burn-off revealing itself",
             "Model uncertainty roughly halves between 8am and noon "
             "(CRPS 1.12→0.60).",
             "Peak heating begins", b12),
            ("peak_heating", b12, b16,
             "The settlement value is forming NOW",
             "Each hour shrinks uncertainty (0.60→0.33); the running max is "
             "a one-way ratchet.",
             "Peak heating ends", b16),
            ("post_peak", b16, b1635,
             "High effectively locked",
             "~99% of days settle at the current visible running max.",
             "Preliminary climate report", b1635),
            ("report_out", b1635, end,
             "Preliminary report is out — today is decided",
             "Attention shifts to the next day's market.",
             "Climate day starts", end),
        ]

    from highcast import gating

    for key, lo, hi, headline, detail_base, next_name, next_time in phases:
        if lo <= now < hi:
            window = (f"{_fmt_clock(lo, cfg.display_tz)}"
                      f"–{_fmt_clock(hi, cfg.display_tz)}")
            return {
                "phase": key,
                "headline": headline,
                "detail": f"{detail_base} (this phase: {window} Manila)",
                "next_transition": {
                    "name": next_name,
                    "la": _fmt(next_time, cfg.station.tz),
                    "manila": _fmt(next_time, cfg.display_tz),
                },
                "timing_curve": gating.timing_curve(cfg.station.icao, metric),
                "decision_hour": gating.min_decision_hour(cfg.station.icao, metric),
                "disclaimer": TIMING_DISCLAIMER,
                "recommendation": recommendation,
            }

    raise AssertionError(f"unreachable: now={now} did not match any phase "
                          f"in climate day starting {start}")
