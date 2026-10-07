from datetime import date, datetime, timedelta, timezone
from highcast.clock import events, timing_guide
from highcast.config import load_config

def test_events_are_dst_aware_and_cover_the_cycle():
    evs = events(load_config(), date(2026, 7, 17))
    names = [e["name"] for e in evs]
    assert "Peak heating begins" in names
    assert "Preliminary climate report" in names
    assert "Settlement" in names
    prelim = next(e for e in evs if e["name"] == "Preliminary climate report")
    # 16:35 PDT == 07:35+1d in Manila during July
    assert prelim["la"].endswith("16:35")
    assert prelim["manila"].endswith("07:35")

def test_events_are_metric_aware():
    """The HIGH's afternoon "peak heating" block is meaningless for the LOW,
    which is already set by dawn — labelling it that way in LOW mode read as
    though the low were still forming in the afternoon."""
    cfg = load_config()
    high = [e["name"] for e in events(cfg, date(2026, 7, 17), "high")]
    low = [e["name"] for e in events(cfg, date(2026, 7, 17), "low")]
    assert "Peak heating begins" in high
    assert not any("Peak heating" in n for n in low)
    assert any("Dawn" in n for n in low)
    assert any("Low decided" in n for n in low)
    # the metric-agnostic milestones survive in both
    for shared in ("Climate day starts (midnight LST)", "Settlement",
                   "Preliminary climate report", "Climate day ends"):
        assert shared in high and shared in low
    # default stays HIGH so existing callers are unchanged
    assert [e["name"] for e in events(cfg, date(2026, 7, 17))] == high


def test_events_are_chronological():
    """Inserting the metric-specific block must not break the timeline order
    the strip renders in."""
    # January too: the strip is built from LST-anchored and wall-clock-anchored
    # instants, and DST slides them relative to each other.
    for day in (date(2026, 7, 17), date(2026, 1, 15)):
      for metric in ("high", "low"):
        evs = events(load_config(), day, metric)
        names = [e["name"] for e in evs]
        assert len(names) == len(set(names))
        # no two milestones may render at the same moment -- a duplicated
        # instant shows up as two rows on the same tick and reads as two
        # separate events (the LOW's day-ahead cutoff IS the prior 20:00, and
        # its dawn IS the morning-obs open, so both pairs were once collapsed).
        stamps = [(e["la"], e["manila"]) for e in evs]
        assert len(stamps) == len(set(stamps)), (day, metric, stamps)
        # only milestones present in BOTH modes -- the LOW folds the morning-obs
        # open into its dawn entry, so that one is not shared
        for earlier, later in [
            ("Climate day starts (midnight LST)", "Preliminary climate report"),
            ("Preliminary climate report", "Market close"),
            ("Market close", "Climate day ends"),
            ("Climate day ends", "Settlement"),
        ]:
            assert names.index(earlier) < names.index(later), (day, metric, earlier, later)


def test_winter_shift():
    evs = events(load_config(), date(2026, 1, 15))
    prelim = next(e for e in evs if e["name"] == "Preliminary climate report")
    assert prelim["manila"].endswith("08:35")   # PST: +16h


def test_timing_guide_phase_keys():
    cfg = load_config()
    # 2026-07-18T10:00:00Z == 3am LA (PDT) -> overnight
    g = timing_guide(cfg, datetime(2026, 7, 18, 10, 0, 0, tzinfo=timezone.utc))
    assert g["phase"] == "overnight"

    # 2026-07-18T20:30:00Z == 1:30pm LA (PDT) -> peak_heating
    g = timing_guide(cfg, datetime(2026, 7, 18, 20, 30, 0, tzinfo=timezone.utc))
    assert g["phase"] == "peak_heating"

    # 2026-07-19T00:00:00Z == 5pm LA (PDT, previous local calendar day) -> report_out
    g = timing_guide(cfg, datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc))
    assert g["phase"] == "report_out"


def test_timing_guide_shape_and_next_transition():
    cfg = load_config()
    g = timing_guide(cfg, datetime(2026, 7, 18, 10, 0, 0, tzinfo=timezone.utc))
    assert set(g.keys()) == {
        "phase", "headline", "detail", "next_transition", "timing_curve",
        "decision_hour", "disclaimer", "recommendation",
    }
    nt = g["next_transition"]
    assert isinstance(nt["name"], str) and nt["name"]
    assert isinstance(nt["la"], str) and nt["la"]
    assert isinstance(nt["manila"], str) and nt["manila"]
    # `sharpness` was a hardcoded HIGH-only CRPS table, stale within days and
    # empty for the LOW. It is now `timing_curve` — measured per-hour accuracy AND
    # the share of days the extremum is still ahead, served for BOTH metrics.
    assert isinstance(g["timing_curve"], list) and len(g["timing_curve"]) > 0
    assert all({"hour", "hit", "mae", "still_future", "ready"} <= r.keys()
               for r in g["timing_curve"])
    assert g["decision_hour"] is not None


def test_timing_guide_disclaimer_is_exact():
    cfg = load_config()
    g = timing_guide(cfg, datetime(2026, 7, 18, 10, 0, 0, tzinfo=timezone.utc))
    assert g["disclaimer"] == (
        "Timing reflects model accuracy only — market prices may already reflect "
        "the same information. Trade decisions are yours."
    )


def test_recommendation_before_9am_targets_today():
    cfg = load_config()
    # 2026-07-18T10:00Z == 3am PDT -> pre-9am: wait for the refresh, today
    rec = timing_guide(
        cfg, datetime(2026, 7, 18, 10, 0, 0, tzinfo=timezone.utc))["recommendation"]
    assert rec["target_date"] == "2026-07-18"
    assert rec["action"] == "Wait for the 9am refresh, then place today positions"
    # 08:00–10:00 PDT rendered in Manila (+15h during July)
    assert rec["window_manila"] == "23:00–01:00"
    assert rec["window_la"] == "08:00–10:00"


def test_recommendation_open_window_targets_today_now():
    cfg = load_config()
    # 2026-07-18T20:30Z == 1:30pm PDT -> today's window is open
    rec = timing_guide(
        cfg, datetime(2026, 7, 18, 20, 30, 0, tzinfo=timezone.utc))["recommendation"]
    assert rec["target_date"] == "2026-07-18"
    assert rec["window_manila"] == "now"
    assert rec["window_la"] == "now"
    assert rec["action"] == ("Today's window is open — the 9am read is live "
                             "and formation begins at noon LA")
    assert rec["why"]


def test_recommendation_after_1635_targets_tomorrow():
    cfg = load_config()
    # 2026-07-19T00:00Z == 5pm PDT Jul 18 -> today decided, target tomorrow
    rec = timing_guide(
        cfg, datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc))["recommendation"]
    assert rec["target_date"] == "2026-07-19"
    assert rec["action"] == ("Today is decided — place tomorrow positions on "
                             "the evening guidance forecast")
    # 17:00–19:00 PDT rendered in Manila (+15h during July)
    assert rec["window_manila"] == "08:00–10:00"
    assert rec["window_la"] == "17:00–19:00"


def test_recommendation_climate_day_tail_still_targets_next_day():
    cfg = load_config()
    # 2026-07-19T07:30Z == 00:30 PDT Jul 19 — wall clock has rolled over but
    # Jul 18's climate day is still running (ends 08:00Z): still "tomorrow"
    # relative to the climate day, i.e. Jul 19.
    rec = timing_guide(
        cfg, datetime(2026, 7, 19, 7, 30, 0, tzinfo=timezone.utc))["recommendation"]
    assert rec["target_date"] == "2026-07-19"
    assert rec["window_manila"]


def test_timing_guide_covers_full_climate_day_no_gaps():
    # sample every 20 minutes across a full climate day; every phase should
    # appear and every sample should resolve without error.
    cfg = load_config()
    start = datetime(2026, 7, 17, 8, 0, 0, tzinfo=timezone.utc)  # climate day start (PDT)
    phases = set()
    t = start
    while t < start + timedelta(hours=24):
        phases.add(timing_guide(cfg, t)["phase"])
        t += timedelta(minutes=20)
    assert phases == {
        "overnight", "morning_obs", "late_morning", "peak_heating",
        "post_peak", "report_out",
    }


def test_timing_guide_low_narrative_and_high_unchanged():
    """The low reframes the day around dawn formation; the high is unchanged."""
    from datetime import datetime, timezone
    from highcast.config import load_config
    from highcast.clock import timing_guide
    cfg = load_config()
    pre_dawn = datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc)  # ~3am PDT
    low = timing_guide(cfg, pre_dawn, "low")
    assert low["phase"] == "overnight"
    assert "low is forming" in low["headline"].lower()
    assert low["recommendation"]["target_date"] == "2026-07-24"   # act on today's low
    # The LOW now gets a real curve too — the old table left it empty.
    assert len(low["timing_curve"]) > 0
    # Knees moved 2026-07-31: KLAX LOW 4 -> 6, KLAX HIGH 11 -> 13
    # (see gating.MIN_DECISION_HOUR).
    assert low["decision_hour"] == 6
    high = timing_guide(cfg, pre_dawn, "high")
    assert high["headline"] == "Early tells from overnight obs"    # unchanged
    assert len(high["timing_curve"]) > 0
    assert high["decision_hour"] == 13
