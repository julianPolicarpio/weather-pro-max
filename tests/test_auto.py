from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from highcast import db as hdb
from highcast import notify as highcast_notify
from highcast.cli import (
    auto_cycle, in_climate_day, should_predict_dayahead, should_predict_tomorrow,
)
from highcast.config import load_config

CFG = load_config()

def test_in_climate_day_boundaries():
    # 07:00Z on July 17 is BEFORE the climate day start (08:00Z)
    assert not in_climate_day(CFG, datetime(2026, 7, 17, 7, 0, tzinfo=timezone.utc))
    assert in_climate_day(CFG, datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc))

def test_should_predict_dayahead_once_after_9am_local(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    before9 = datetime(2026, 7, 17, 15, 30, tzinfo=timezone.utc)  # 08:30 PDT
    after9 = datetime(2026, 7, 17, 16, 30, tzinfo=timezone.utc)   # 09:30 PDT
    assert not should_predict_dayahead(conn, CFG, before9)
    assert should_predict_dayahead(conn, CFG, after9)
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-17",
        "issued_at_utc": "2026-07-17T16:20:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": "{}", "buckets": "{}"}])
    assert not should_predict_dayahead(conn, CFG, after9)   # already issued

def test_evening_before_row_does_not_block_9am_refresh(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    after9 = datetime(2026, 7, 17, 16, 30, tzinfo=timezone.utc)   # 09:30 PDT
    # issued 18:00 PDT on the 16th — an evening-before forecast for the 17th,
    # well before the 17th's 08:00 PDT cutoff (15:00Z)
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-17",
        "issued_at_utc": "2026-07-17T01:00:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": "{}", "buckets": "{}"}])
    assert should_predict_dayahead(conn, CFG, after9)   # must NOT block refresh
    # a row issued at the 08:00-local cutoff exactly DOES block
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-17",
        "issued_at_utc": "2026-07-17T15:00:00+00:00", "kind": "dayahead",
        "point_f": 84.0, "quantiles": "{}", "buckets": "{}"}])
    assert not should_predict_dayahead(conn, CFG, after9)

def test_should_predict_tomorrow_only_after_5pm_local(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    before17 = datetime(2026, 7, 17, 23, 30, tzinfo=timezone.utc)  # 16:30 PDT
    after17 = datetime(2026, 7, 18, 0, 30, tzinfo=timezone.utc)    # 17:30 PDT Jul 17
    assert not should_predict_tomorrow(conn, CFG, before17)
    assert should_predict_tomorrow(conn, CFG, after17)

def test_tomorrow_row_blocks_repredicting_tomorrow(tmp_path):
    conn = hdb.connect(tmp_path / "t.db")
    after17 = datetime(2026, 7, 18, 0, 30, tzinfo=timezone.utc)    # 17:30 PDT Jul 17
    # kind='evening', because that is what the tomorrow branch now writes.
    # Seeding 'dayahead' here would leave should_predict_tomorrow unsatisfied
    # and re-issue the evening forecast on every wake for the rest of the night.
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-18",
        "issued_at_utc": "2026-07-18T00:20:00+00:00", "kind": hdb.KIND_EVENING,
        "point_f": 83.0, "quantiles": "{}", "buckets": "{}"}])
    assert not should_predict_tomorrow(conn, CFG, after17)


def test_a_dayahead_row_does_not_satisfy_the_tomorrow_check(tmp_path):
    """The two kinds are separate populations. A morning day-ahead for tomorrow
    must NOT count as the evening forecast having been issued -- they answer
    different questions and are measured separately (evening MAE 4.42F vs the
    morning refresh's 1.45F at KLAX HIGH)."""
    conn = hdb.connect(tmp_path / "t.db")
    after17 = datetime(2026, 7, 18, 0, 30, tzinfo=timezone.utc)
    hdb.upsert(conn, "predictions", [{
        "station": CFG.station.icao, "target_date": "2026-07-18",
        "issued_at_utc": "2026-07-18T00:20:00+00:00", "kind": "dayahead",
        "point_f": 83.0, "quantiles": "{}", "buckets": "{}"}])
    assert should_predict_tomorrow(conn, CFG, after17)

@pytest.fixture(autouse=True)
def _notifications_off(monkeypatch):
    """These tests exercise auto_cycle's PLUMBING, not its alerts, but they run it
    with the LIVE config — which carries a real Discord webhook. Until 2026-08-12
    that meant the suite could POST to Julian's Discord, and once
    `notify_missed_commit` shipped it actually did: a `tmp_path` database has no
    recorded bets, so every commit hour looks missed, and the alert fires whenever
    the wall clock happens to sit inside an open window.

    Returning False is the faithful stub — it is exactly what `notify.send` does
    when no webhook is configured, which is the normal posture for a test. It also
    makes the `actions` assertion below wall-clock independent; it used to pass all
    morning and fail in the afternoon with a stray "discord_1".
    """
    monkeypatch.setattr(highcast_notify, "send", lambda *a, **k: False)


def test_auto_cycle_threads_config_path_to_cmd_predict(tmp_path):
    # auto_cycle must forward its own config_path into every cmd_predict()
    # call it makes, instead of a hardcoded "config/klax.yaml" - this test
    # freezes "now" to a moment that triggers all three branches
    # (dayahead, dayahead_tomorrow, intraday) and asserts each captured
    # args.config equals the custom path we passed in, not the default.
    conn = hdb.connect(tmp_path / "t.db")
    custom_path = "config/custom-not-default.yaml"
    calls = []

    # `datetime` is imported locally inside auto_cycle, so it can't be patched
    # on the module. Force all three branches by stubbing the predicates, then
    # capture every cmd_predict arg and assert the config path was threaded.
    with patch("highcast.cli.should_predict_dayahead", return_value=True), \
         patch("highcast.cli.should_predict_tomorrow", return_value=True), \
         patch("highcast.cli.in_climate_day", return_value=True), \
         patch("highcast.cli.cmd_predict", side_effect=lambda args: calls.append(args)):
        actions = auto_cycle(conn, CFG, custom_path)

    # day-aheads are generated for BOTH metrics (the low's were missing, which
    # left the low timing/tomorrow cards permanently empty), then the intradays.
    # "evening" replaced "dayahead_tomorrow": the label follows the kind now.
    assert actions == ["dayahead", "evening",
                       "dayahead_low", "evening_low",
                       "intraday", "intraday_low"]
    assert len(calls) == 6
    assert all(a.config == custom_path for a in calls)   # threaded, not hardcoded
    assert all(a.config != "config/klax.yaml" for a in calls)
    lows = [a for a in calls if getattr(a, "metric", "high") == "low"]
    assert len(lows) == 3                                 # 2 day-ahead + 1 intraday
    assert calls[-1].live is True and calls[-1].metric == "low"   # low intraday


# ── settlement wiring ───────────────────────────────────────────────────────
# `pick.settle` existed, was tested, and was called from nowhere for weeks. A
# unit test of settle() cannot catch that; only a test of the CALLER can. Same
# class of gap as `appbet.settle`, which IS wired -- these assert the pick
# ledger is settled by the same cycle.

def _quiet():
    """Patch out everything in auto_cycle that predicts or hits the network, so
    these tests assert the settlement wiring and nothing else."""
    from contextlib import ExitStack
    stack = ExitStack()
    for target in ("highcast.cli.should_predict_dayahead",
                   "highcast.cli.should_predict_tomorrow",
                   "highcast.cli.in_climate_day"):
        stack.enter_context(patch(target, return_value=False))
    stack.enter_context(patch("highcast.cli.cmd_predict", return_value=None))
    return stack


def test_auto_cycle_settles_published_picks(tmp_path):
    """The wake cycle must grade the picks ledger, not just app_bets/signals.

    Without this call every `kind='verdict'` row keeps `won`/`actual` NULL
    forever, so `pick.graded_record` reports nothing and the pre-registered
    KILL_AFTER_PICKS criterion -- the feature's only falsification -- can never
    fire. It is not a bug in settle(); settle() was simply never invoked.
    """
    conn = hdb.connect(tmp_path / "t.db")
    seen = []
    with _quiet() as _, \
         patch("highcast.pick.settle",
               side_effect=lambda c, cfg, now: seen.append((c, cfg, now)) or 2):
        actions = auto_cycle(conn, CFG, "config/klax.yaml")
    assert len(seen) == 1                       # called exactly once per wake
    assert seen[0][0] is conn                   # on the wake's own connection
    assert "picks_settled_2" in actions


def test_auto_cycle_survives_a_pick_settle_failure(tmp_path):
    """Fenced like every other wake-cycle call: grading must never abort the
    wake that keeps predictions fresh. The failure is REPORTED in the action
    list rather than swallowed -- a silent settle path is what produced this
    gap in the first place."""
    conn = hdb.connect(tmp_path / "t.db")
    with _quiet() as _, \
         patch("highcast.pick.settle", side_effect=RuntimeError("boom")):
        actions = auto_cycle(conn, CFG, "config/klax.yaml")
    assert "pick_settle_error:RuntimeError" in actions


def test_settle_command_grades_picks_too(tmp_path, capsys):
    """`highcast settle` is what a human runs to catch up a closed laptop. It
    settled signals only, so a manual catch-up left the pick ledger ungraded."""
    import yaml
    from highcast.cli import cmd_settle
    from highcast.config import load_config as _load

    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    hdb.connect(_load(str(p)).db_path)          # create the schema

    with patch("highcast.signals.settle_signals", return_value=0), \
         patch("highcast.pick.settle", return_value=3) as settled:
        cmd_settle(_ns(config=str(p)))
    assert settled.call_count == 1
    assert "graded 3 pick(s)" in capsys.readouterr().out


def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)
