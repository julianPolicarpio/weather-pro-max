"""The staleness guard — QUEUE item 4.

A feed has died silently twice (`nbm_nbs`, `forecast_conv`) and nobody noticed
for days. These tests pin the two behaviours that make the guard worth having:
it FIRES on a dead feed, and it stays QUIET on a healthy one. A guard that cries
wolf gets ignored, and an ignored guard is worse than none.
"""
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from highcast import db as hdb
from highcast import staleness
from highcast.config import load_config

CFG = load_config()
NOW = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)


def _conn(tmp_path):
    return hdb.connect(tmp_path / "t.db")


def _feed(conn, table, icao, fetched_at, **extra):
    row = {"station": icao, "fetched_at": fetched_at, **extra}
    hdb.upsert(conn, table, [row])


def _fresh_everything(conn, icao, now):
    """Populate every fetched_at feed so only the feed under test is stale.

    Column sets come from the real schema — `forecast_intraday` needs `run_date`
    and a `metric`, `forecast_hourly` puts `fetched_at` in its primary key, etc.
    Guessing them produced nine failures on the first run.
    """
    ts = now.isoformat()
    d = now.date().isoformat()
    _feed(conn, "obs", icao, ts, valid_utc=ts, source="metar")
    _feed(conn, "forecast_hourly", icao, ts, valid_utc=ts, model="m", temp_f=70.0)
    _feed(conn, "forecast_intraday", icao, ts, target_date=d, model="m",
          metric="high", run_date=d)
    _feed(conn, "forecasts", icao, ts, target_date=d, model="m", run_date=d,
          metric="high", max_temp_f=70.0)
    _feed(conn, "forecast_nbs", icao, ts, target_date=d, run_date=d,
          metric="high")
    _feed(conn, "forecast_conv", icao, ts, valid_utc=ts, model="m")
    _feed(conn, "forecast_ens", icao, ts, target_date=d, run_date=d,
          metric="high")
    _feed(conn, "forecast_aloft", icao, ts, target_date=d, run_date=d,
          metric="high")
    # CLI is checked SEMANTICALLY — yesterday's row must exist, station-local.
    from zoneinfo import ZoneInfo
    local = now.astimezone(ZoneInfo(CFG.station.tz)).date()
    hdb.upsert(conn, "cli_reports", [{
        "station": icao, "date": (local - timedelta(days=1)).isoformat(),
        "high_f": 80, "low_f": 60, "raw": "{}", "fetched_at": ts}])


def test_all_fresh_is_quiet(tmp_path):
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    assert staleness.stale(conn, CFG, NOW) == []
    assert "all 9 fresh" in staleness.wake_line(conn, CFG, NOW)


def test_a_dead_feed_fires(tmp_path):
    """THE CASE THIS EXISTS FOR. `forecast_conv` was dead nine days in July and
    was found by accident."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (NOW - timedelta(days=9)).isoformat(), valid_utc="x", model="m")
    bad = staleness.stale(conn, CFG, NOW)
    assert [b["label"] for b in bad] == ["convection"]
    assert "convection" in staleness.wake_line(conn, CFG, NOW)
    assert "STALE" in staleness.wake_line(conn, CFG, NOW)


def test_a_feed_inside_its_threshold_is_not_stale(tmp_path):
    """Convection legitimately gaps ~39h at p90 — that must NOT fire, or the
    guard becomes noise and gets ignored."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (NOW - timedelta(hours=39)).isoformat(), valid_utc="x", model="m")
    assert staleness.stale(conn, CFG, NOW) == []


def test_a_feed_that_never_had_a_row_is_stale(tmp_path):
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_nbs")
    bad = staleness.stale(conn, CFG, NOW)
    assert [b["reason"] for b in bad] == ["no rows ever"]


def test_another_stations_rows_do_not_count(tmp_path):
    """Two stations share one database. A live KMIA feed must not make a dead
    KLAX feed look healthy — that is the `metric:date` cross-station bug class."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_aloft")
    _feed(conn, "forecast_aloft", "KMIA", NOW.isoformat(), target_date="2026-08-13",
          run_date="2026-08-13", metric="high")
    assert [b["label"] for b in staleness.stale(conn, CFG, NOW)] == ["aloft"]


# ── the CLI special case ────────────────────────────────────────────────────

def test_cli_uses_yesterdays_row_not_the_timestamp(tmp_path):
    """⚠️ THE REGRESSION THAT MATTERS. `backfill._yesterday_cli` returns 0 without
    writing once yesterday's row is final, so `fetched_at` FREEZES during healthy
    operation — observed: 7 distinct values since June, including a 379h gap that
    was not an outage. A timestamp check here fired for 14 straight days."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    # yesterday's row present, but written a fortnight ago
    conn.execute("UPDATE cli_reports SET fetched_at=?",
                 ((NOW - timedelta(days=16)).isoformat(),))
    assert staleness.stale(conn, CFG, NOW) == [], \
        "a frozen CLI timestamp means the feed is WORKING, not dead"


def test_cli_one_missing_day_is_not_yet_an_outage(tmp_path):
    """Normal state is newest == yesterday. QUEUE item 4: one missing day is not
    an outage until two are missing."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)          # newest = yesterday
    assert staleness.stale(conn, CFG, NOW) == []
    # drop back one more day — still inside the 2-day allowance
    conn.execute("DELETE FROM cli_reports")
    hdb.upsert(conn, "cli_reports", [{
        "station": CFG.station.icao, "date": "2026-08-11", "high_f": 80,
        "low_f": 60, "raw": "{}", "fetched_at": NOW.isoformat()}])
    assert staleness.stale(conn, CFG, NOW) == []


def test_cli_two_consecutive_missing_days_fires(tmp_path):
    """⚠️ THE FIRST VERSION COULD NEVER FIRE. It measured minutes since local
    midnight (max 1,440) against a 2,880-minute threshold, so a permanently dead
    CLI feed stayed silent. A guard that cannot fire looks like coverage."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM cli_reports")
    hdb.upsert(conn, "cli_reports", [{
        "station": CFG.station.icao, "date": "2026-08-10", "high_f": 80,
        "low_f": 60, "raw": "{}", "fetched_at": NOW.isoformat()}])
    bad = staleness.stale(conn, CFG, NOW)
    assert [b["label"] for b in bad] == ["CLI settlement"]
    assert "2026-08-10" in bad[0]["reason"]


def test_cli_totally_absent_fires(tmp_path):
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM cli_reports")
    bad = staleness.stale(conn, CFG, NOW)
    assert [b["reason"] for b in bad] == ["no rows ever"]


def test_cli_yesterday_is_station_local_not_utc(tmp_path):
    """At 05:10Z KMIA is on the new day while KLAX is still on the previous one.
    Comparing against the UTC date once produced a bogus '0 of 4 missed' commit
    report; the same trap applies here."""
    conn = _conn(tmp_path)
    early = datetime(2026, 8, 13, 5, 10, tzinfo=timezone.utc)   # 22:10 Aug 12 PDT
    _fresh_everything(conn, CFG.station.icao, early)
    conn.execute("DELETE FROM cli_reports")
    hdb.upsert(conn, "cli_reports", [{
        "station": CFG.station.icao, "date": "2026-08-11", "high_f": 80,
        "low_f": 60, "raw": "{}", "fetched_at": early.isoformat()}])
    # KLAX local date is still Aug 12, so "yesterday" is Aug 11 — present.
    assert staleness.stale(conn, CFG, early) == []


# ── the guard must not fail silently ────────────────────────────────────────

def test_a_broken_check_reports_stale_rather_than_vanishing(tmp_path):
    """A guard that swallows its own exception is the failure it was built to
    prevent."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    broken = dataclasses.replace(staleness.FEEDS[0], table="no_such_table")
    original = staleness.FEEDS
    try:
        staleness.FEEDS = (broken, *original[1:])
        bad = staleness.stale(conn, CFG, NOW)
        assert len(bad) == 1 and bad[0]["reason"].startswith("check failed:")
    finally:
        staleness.FEEDS = original


def test_exemptions_are_documented_not_silent():
    """Every exemption is a claim that ABSENCE IS NORMAL. Each needs a reason so
    nobody re-adds a genuinely broken feed to the exempt list to quiet it."""
    assert set(staleness.EXEMPT) == {"alerts", "obs_1min", "twc_obs", "syn5"}
    for table, why in staleness.EXEMPT.items():
        assert len(why) > 20, f"{table} exemption needs a real reason"
    checked = {f.table for f in staleness.FEEDS}
    assert not (checked & set(staleness.EXEMPT)), \
        "a feed cannot be both checked and exempt"


# ── the Discord alert ───────────────────────────────────────────────────────

def test_alert_is_stamped_once_per_day_not_once_forever(tmp_path, monkeypatch):
    """⚠️ A DEAD FEED IS AN ONGOING CONDITION, not a one-off event.

    `notify_missed_commit` stamps on the target date because a missed commit
    happens once. A stale feed stays stale for days while the wake cycle runs
    every five minutes. Stamping once FOREVER would mean a feed that dies, is
    fixed, and dies again never alerts the second time — so the key carries the
    local date and gives one reminder per morning.
    """
    import dataclasses as dc

    from highcast import notify
    from highcast.config import AppConfig

    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (NOW - timedelta(days=9)).isoformat(), valid_utc="x", model="m")
    cfg = dc.replace(CFG, discord_webhook_url="https://discord.test/hook")

    sent = []
    monkeypatch.setattr(notify, "send", lambda c, p: sent.append(p) or True)

    assert notify.notify_stale_feeds(conn, cfg, NOW) == 1
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 0, "same day: silent"

    # Advance a day and re-freshen everything else, so convection is still the
    # ONLY stale feed. Without this the other feeds age past their own limits
    # and the count is 4 — correct behaviour, but it would not test the stamp.
    tomorrow = NOW + timedelta(days=1)
    _fresh_everything(conn, CFG.station.icao, tomorrow)
    # ...but NOT convection, which is the feed under test. The fixture refreshes
    # every feed including that one, which silently healed it on the first try.
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (NOW - timedelta(days=9)).isoformat(), valid_utc="x", model="m")
    assert [b["label"] for b in staleness.stale(conn, cfg, tomorrow)] == \
        ["convection"]
    assert notify.notify_stale_feeds(conn, cfg, tomorrow) == 1, \
        "next day: one reminder, because it is still broken"
    assert len(sent) == 2
    assert "convection" in sent[0]["embeds"][0]["description"]


def test_a_failed_send_is_not_stamped(tmp_path, monkeypatch):
    """One network blip must not permanently silence a real outage."""
    import dataclasses as dc

    from highcast import notify

    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (NOW - timedelta(days=9)).isoformat(), valid_utc="x", model="m")
    cfg = dc.replace(CFG, discord_webhook_url="https://discord.test/hook")

    monkeypatch.setattr(notify, "send", lambda c, p: False)
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 0
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 1, "retried next wake"


def test_no_webhook_is_a_silent_no_op(tmp_path):
    """⚠️ Must build the cfg with an EMPTY webhook. The first version of this
    test passed the real `CFG`, which carries a live `discord_webhook_url` from
    `config/klax.yaml` — the conftest guard caught it trying to POST."""
    import dataclasses as dc

    from highcast import notify
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    conn.execute("DELETE FROM forecast_conv")
    off = dc.replace(CFG, discord_webhook_url="")
    assert notify.notify_stale_feeds(conn, off, NOW) == 0


# ── the Discord notifier ────────────────────────────────────────────────────

def _cfg_with_webhook():
    return dataclasses.replace(
        CFG, discord_webhook_url="https://discord.com/api/webhooks/test")


def _stale_db(tmp_path, now=NOW):
    """A database where exactly one feed (convection) is dead."""
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, now)
    conn.execute("DELETE FROM forecast_conv")
    _feed(conn, "forecast_conv", CFG.station.icao,
          (now - timedelta(days=9)).isoformat(), valid_utc="x", model="m")
    return conn


def test_notifier_is_a_silent_no_op_without_a_webhook(tmp_path):
    """⚠️ The webhook must be blanked EXPLICITLY. `CFG` is the real
    `config/klax.yaml` and it carries a live Discord URL — passing it unmodified
    trips the `httpx.post` guard in conftest, which is that guard doing its job.
    """
    from highcast import notify
    conn = _stale_db(tmp_path)
    no_hook = dataclasses.replace(CFG, discord_webhook_url="")
    assert notify.notify_stale_feeds(conn, no_hook, NOW) == 0


def test_notifier_sends_once_then_stays_quiet_the_same_day(tmp_path, monkeypatch):
    """The wake cycle runs every 5 minutes and a dead feed stays dead. Without
    the stamp this would post 288 times a day."""
    from highcast import notify
    sent = []
    monkeypatch.setattr(notify, "send", lambda cfg, p: sent.append(p) or True)
    conn = _stale_db(tmp_path)
    cfg = _cfg_with_webhook()
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 1
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 0
    assert notify.notify_stale_feeds(conn, cfg, NOW + timedelta(hours=3)) == 0
    assert len(sent) == 1
    assert "convection" in sent[0]["embeds"][0]["description"]


def test_notifier_reminds_again_the_next_day(tmp_path, monkeypatch):
    """⚠️ DELIBERATELY DIFFERENT FROM `notify_missed_commit`, which stamps once
    forever. A missed commit is a one-off event; a dead feed is an ONGOING
    condition. Stamping once forever would mean a feed that dies, gets fixed and
    dies again never alerts a second time.

    The other feeds are re-freshened at the later instant on purpose: leaving
    them frozen would make them stale too (metar's limit is 3h) and the test
    would pass for the wrong reason — 4 alerts rather than the 1 it claims.
    """
    from highcast import notify
    sent = []
    monkeypatch.setattr(notify, "send", lambda cfg, p: sent.append(p) or True)
    conn = _stale_db(tmp_path)
    cfg = _cfg_with_webhook()
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 1

    tomorrow = NOW + timedelta(days=1)
    _fresh_everything(conn, CFG.station.icao, tomorrow)   # refresh the healthy ones
    conn.execute("DELETE FROM forecast_conv")             # ...conv stays dead
    _feed(conn, "forecast_conv", CFG.station.icao,
          (tomorrow - timedelta(days=10)).isoformat(), valid_utc="x", model="m")
    assert [r["label"] for r in staleness.stale(conn, CFG, tomorrow)] == ["convection"]
    assert notify.notify_stale_feeds(conn, cfg, tomorrow) == 1
    assert len(sent) == 2


def test_a_failed_send_is_never_stamped(tmp_path, monkeypatch):
    """Stamping before a confirmed send turns one network blip into a
    permanently lost alert."""
    from highcast import notify
    monkeypatch.setattr(notify, "send", lambda cfg, p: False)
    conn = _stale_db(tmp_path)
    cfg = _cfg_with_webhook()
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 0
    assert conn.execute(
        "SELECT COUNT(*) n FROM alert_stamps WHERE kind='stale_feed'"
    ).fetchone()["n"] == 0
    # ...and it retries on the next wake
    monkeypatch.setattr(notify, "send", lambda cfg, p: True)
    assert notify.notify_stale_feeds(conn, cfg, NOW) == 1


def test_healthy_feeds_send_nothing(tmp_path, monkeypatch):
    from highcast import notify
    sent = []
    monkeypatch.setattr(notify, "send", lambda cfg, p: sent.append(p) or True)
    conn = _conn(tmp_path)
    _fresh_everything(conn, CFG.station.icao, NOW)
    assert notify.notify_stale_feeds(conn, _cfg_with_webhook(), NOW) == 0
    assert sent == []
