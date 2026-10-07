"""The weatherguard warning: heat vs storm, and exactly-once.

This is a WARNING, not a gate. It blocks nothing and places nothing -- the whole
point is that Julian decides. We cannot justify an automatic skip: the
near-boundary days it would skip still hit 72%, and skipping them is only correct
if the market has not already discounted them, which needs price history we do
not have. So the app reports what it can see and stays out of the way.
"""
import dataclasses
from datetime import datetime, timezone

from highcast import db, notify
from highcast.config import load_config

CFG = load_config("config/kmia.yaml")
NOW = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)   # inside KMIA's 08-01 day


def _cfg(tmp_path):
    return dataclasses.replace(
        CFG, db_path=tmp_path / "t.db",
        discord_webhook_url="https://discord.test/hook")


def _alert(conn, cfg, event, onset, expires):
    db.upsert(conn, "alerts", [{
        "station": cfg.station.icao, "event": event, "onset_utc": onset,
        "expires_utc": expires, "fetched_at": "t"}])


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send", lambda c, p: (sent.append(p), True)[1])
    return sent


def test_storm_warning_says_do_not_bother(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Severe Thunderstorm Warning",
           "2026-08-01T17:00:00+00:00", "2026-08-01T23:00:00+00:00")
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 1
    body = str(sent[0])
    assert "STORM" in body
    assert "Don't bother" in body
    # The storm case must name the STRUCTURAL reason -- the model has no
    # precipitation input -- not merely "conditions are unstable".
    assert "no precipitation input" in body


def test_heat_advisory_is_softer_than_a_storm(tmp_path, monkeypatch):
    """Heat is PARTLY priced in (the model has a heat_alert feature), so the
    wording must not tell Julian to sit out. Collapsing both into one message
    would either overstate heat or understate storms."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-08-01T15:00:00+00:00", "2026-08-01T22:00:00+00:00")
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 1
    body = str(sent[0])
    assert "HEAT" in body
    assert "Don't bother" not in body
    assert "partly" in body.lower()


def test_heat_and_storm_send_two_separate_warnings(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-08-01T15:00:00+00:00", "2026-08-01T22:00:00+00:00")
    _alert(conn, cfg, "Tornado Warning",
           "2026-08-01T17:00:00+00:00", "2026-08-01T19:00:00+00:00")
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 2
    assert len(sent) == 2


def test_sends_once_not_once_per_wake(tmp_path, monkeypatch):
    """The wake cycle runs every 5 minutes. Without the stamp this would fire
    ~dozens of times a day -- the same failure `notified_utc` prevents for bets.
    """
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-08-01T15:00:00+00:00", "2026-08-01T22:00:00+00:00")
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 1
    for _ in range(5):
        assert notify.notify_unstable(conn, cfg, NOW) == 0
    assert len(sent) == 1


def test_three_heat_products_in_one_day_send_one_message(tmp_path, monkeypatch):
    """Keyed on the KIND, not the product: KMIA logged three Heat Advisories in
    a single afternoon on 2026-07-30."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    for onset, exp in (("2026-08-01T15:00:00+00:00", "2026-08-01T18:00:00+00:00"),
                       ("2026-08-01T16:00:00+00:00", "2026-08-01T19:00:00+00:00"),
                       ("2026-08-01T17:00:00+00:00", "2026-08-01T22:00:00+00:00")):
        _alert(conn, cfg, "Heat Advisory", onset, exp)
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 1
    assert len(sent) == 1


def test_an_expired_advisory_does_not_warn(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-07-29T15:00:00+00:00", "2026-07-29T22:00:00+00:00")
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 0
    assert not sent


def test_a_failed_send_is_retried_not_stamped(tmp_path, monkeypatch):
    """A webhook outage must not silently consume the day's only warning."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-08-01T15:00:00+00:00", "2026-08-01T22:00:00+00:00")
    monkeypatch.setattr(notify, "send", lambda c, p: False)
    assert notify.notify_unstable(conn, cfg, NOW) == 0
    sent = _capture(monkeypatch)
    assert notify.notify_unstable(conn, cfg, NOW) == 1
    assert len(sent) == 1


def test_no_webhook_configured_is_silent(tmp_path):
    cfg = dataclasses.replace(_cfg(tmp_path), discord_webhook_url="")
    conn = db.connect(cfg.db_path)
    _alert(conn, cfg, "Heat Advisory",
           "2026-08-01T15:00:00+00:00", "2026-08-01T22:00:00+00:00")
    assert notify.notify_unstable(conn, cfg, NOW) == 0
