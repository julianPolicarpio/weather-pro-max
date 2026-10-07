"""DISCORD ALERT for a missed commit that is STILL PLACEABLE.

WHY ONLY THE ACTIONABLE CASE. The 2026-08-08/09 outage cost four commits and
nothing said so. But three of the four were past their tradeable window by the
time anyone could have read a message — pushing those to Julian's phone at 1am
Manila would be noise he cannot act on, and noise is how a real alert gets
muted. The wake-cycle log already records every miss; Discord is reserved for
the case where he can still place the bet by hand.

EXACTLY ONCE per (station, metric, target_date), via `alert_stamps` — the same
pattern `notify_unstable` uses. The wake cycle fires every 5 minutes and an
overdue commit stays overdue for hours, so without the stamp this would send
dozens of identical messages and be muted within a day.

A FAILED SEND IS NEVER STAMPED, so it retries on the next wake. Stamping before
a confirmed send would turn one network blip into a permanently lost alert.
"""
import datetime as dt

import pytest

from highcast import db as hdb, notify
from highcast.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/klax.yaml")
    object.__setattr__(c, "db_path", tmp_path / "t.db")
    object.__setattr__(c, "discord_webhook_url", "https://example.invalid/hook")
    return c


@pytest.fixture
def conn(cfg):
    return hdb.connect(cfg.db_path)


NOW = dt.datetime(2026, 8, 10, 5, 10, tzinfo=dt.timezone.utc)


@pytest.fixture
def sent(monkeypatch):
    box = []
    monkeypatch.setattr(notify, "send", lambda cfg, payload: box.append(payload) or True)
    return box


def _missing(actionable, station="KLAX", metric="high", target="2026-08-09"):
    return [{"station": station, "metric": metric, "target_date": target,
             "commit_hour": 13, "still_actionable": actionable,
             "window_closes": "2026-08-10T17:30:00+00:00" if actionable else None}]


def test_alerts_when_still_placeable(conn, cfg, sent, monkeypatch):
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: _missing(True))
    assert notify.notify_missed_commit(conn, cfg, NOW) == 1
    assert len(sent) == 1
    body = str(sent[0])
    assert "KLAX" in body and "HIGH" in body


def test_silent_when_the_window_has_closed(conn, cfg, sent, monkeypatch):
    """THE POINT OF THE FILTER. Three of the four misses in the 08-08/09 outage
    were unrecoverable by the time anyone could read a message."""
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: _missing(False))
    assert notify.notify_missed_commit(conn, cfg, NOW) == 0
    assert sent == []


def test_sends_exactly_once(conn, cfg, sent, monkeypatch):
    """The wake cycle runs every 5 minutes and the commit stays overdue for
    hours. Without the stamp this is dozens of identical messages."""
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: _missing(True))
    assert notify.notify_missed_commit(conn, cfg, NOW) == 1
    for _ in range(5):
        assert notify.notify_missed_commit(conn, cfg, NOW) == 0
    assert len(sent) == 1


def test_a_failed_send_is_not_stamped(conn, cfg, monkeypatch):
    """Stamping before a confirmed send turns one network blip into a
    permanently lost alert."""
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: _missing(True))
    monkeypatch.setattr(notify, "send", lambda cfg, payload: False)
    assert notify.notify_missed_commit(conn, cfg, NOW) == 0
    box = []
    monkeypatch.setattr(notify, "send", lambda cfg, payload: box.append(payload) or True)
    assert notify.notify_missed_commit(conn, cfg, NOW) == 1, "must retry after a failure"


def test_each_market_is_its_own_alert(conn, cfg, sent, monkeypatch):
    """Two markets missing the same day are two separate things to act on."""
    rows = _missing(True, metric="high") + _missing(True, metric="low")
    monkeypatch.setattr(notify.appbet, "missing_commits", lambda *a, **k: rows)
    assert notify.notify_missed_commit(conn, cfg, NOW) == 2
    assert len(sent) == 2


def test_no_webhook_means_no_work(conn, cfg, sent, monkeypatch):
    object.__setattr__(cfg, "discord_webhook_url", "")
    called = []
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: called.append(1) or _missing(True))
    assert notify.notify_missed_commit(conn, cfg, NOW) == 0
    assert sent == []


def test_the_message_says_when_the_window_shuts(conn, cfg, sent, monkeypatch):
    """An alert that does not say how long you have is not actionable."""
    monkeypatch.setattr(notify.appbet, "missing_commits",
                        lambda *a, **k: _missing(True))
    notify.notify_missed_commit(conn, cfg, NOW)
    assert "17:30" in str(sent[0]), sent[0]
