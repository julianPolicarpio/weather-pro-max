"""Discord bet alerts: content, and the exactly-once guarantee.

The wake cycle runs every 5 minutes. Without `notified_utc` the naive "alert when
a bet exists" would fire ~200 times a day, which is the only way this feature can
really fail. These tests pin that, and pin that a FAILED send is retried rather
than lost.
"""
import dataclasses
import json
from datetime import date, datetime, timedelta, timezone

from highcast import appbet, db, notify
from highcast.config import StakingConfig, load_config
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds

CFG = load_config()
LOW = get_metric(CFG, "low")


def _final_cli_raw(cfg, d):
    """A CLI `raw` payload whose product was issued AFTER `d`'s climate day
    closed -- what `appbet.settle` requires before it will grade a bet.

    These tests are about NOTIFICATION, not about the finality guard, so they
    need a settleable row to get far enough to notify. Before the guard existed
    they wrote `raw="{}"`; with no product id `cli_row_is_final` fails toward
    "not final" (correctly -- it cannot prove the report is complete), settle
    declines to grade, and there is nothing to alert about.

    Derived from `climate_day_bounds`, never a literal: KLAX closes 08:00Z and
    KMIA 05:00Z, so a hardcoded stamp would be final for one station and
    preliminary for the other. Mirrors `tests/test_appbet.py::_final_product`.
    """
    end = climate_day_bounds(date.fromisoformat(d), cfg.station.std_utc_offset)[1]
    issued = end + timedelta(hours=1)
    return json.dumps({"product": issued.strftime("%Y%m%d%H%M")
                       + "-TEST-CDUS00-CLITEST"})


def _cfg(tmp_path, hook="https://discord.test/webhook"):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               discord_webhook_url=hook,
                               staking=StakingConfig(bankroll=8.0))


def _bet():
    return {"metric": "low", "target_date": "2026-07-27", "bucket": "68–69",
            "p_model": 0.624, "price": 0.40, "stake": 1.0, "payout_if_win": 2.5,
            "placed_manila": "19:04", "won": None}


def test_message_carries_prediction_bucket_percentage_and_reason():
    """The four things asked for, plus the pair that qualifies the percentage."""
    msg = notify.build_message(CFG, _bet(), {
        "hour_hit": 0.624, "still_future": 0.854, "reason": "PLACE 68–69: edge +5%"})
    e = msg["embeds"][0]
    assert "LOW" in e["title"] and "2026-07-27" in e["title"]
    flat = {f["name"]: f["value"] for f in e["fields"]}
    assert flat["Bucket"] == "**68–69**"
    assert flat["Model"] == "**62%**"
    assert "PLACE 68–69" in flat["Reason"]
    # accuracy must never be quoted without how much is still unknown
    assert "Bucket right at this hour" in flat and "Still a forecast" in flat
    assert flat["Still a forecast"] == "85%"


def test_no_webhook_configured_is_a_silent_no_op(tmp_path):
    cfg = _cfg(tmp_path, hook="")
    conn = db.connect(cfg.db_path)
    assert notify.send(cfg, {"x": 1}) is False
    assert notify.notify_pending(conn, cfg, datetime.now(timezone.utc)) == 0


def _commit_a_bet(conn, cfg, monkeypatch):
    from highcast import signals
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": "2026-07-27",
        "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "68–69", "lo": 68, "hi": 69, "p_model": 0.62,
                    "market_price": 0.40}],
        "decision": type("D", (), {"reason": "PLACE 68–69"})()})
    # 06:00 LA, past knee 6 (KLAX LOW moved 4 -> 6 on 2026-07-31, see
    # gating.MIN_DECISION_HOUR)
    now = datetime(2026, 7, 27, 13, 0, tzinfo=timezone.utc)
    assert appbet.place(conn, cfg, LOW, now)["placed"]
    return now


def test_alerts_exactly_once_across_repeated_wakes(tmp_path, monkeypatch):
    """THE failure mode: the wake fires every 5 minutes."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])
    assert notify.notify_pending(conn, cfg, now) == 1
    for _ in range(5):                      # five more wakes
        assert notify.notify_pending(conn, cfg, now) == 0
    assert len(calls) == 1


def test_a_failed_send_is_retried_not_lost(tmp_path, monkeypatch):
    """notified_utc is written only AFTER success, so an outage self-heals."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: False)     # webhook down
    assert notify.notify_pending(conn, cfg, now) == 0
    assert conn.execute("SELECT notified_utc FROM app_bets").fetchone()[0] is None
    monkeypatch.setattr(notify, "send", lambda c, p: True)      # recovers
    assert notify.notify_pending(conn, cfg, now) == 1
    assert conn.execute("SELECT notified_utc FROM app_bets").fetchone()[0] is not None


def test_send_never_raises_on_a_network_error(tmp_path, monkeypatch):
    """A notification failure must not break the wake that keeps forecasts fresh."""
    import highcast.notify as N
    cfg = _cfg(tmp_path)
    def boom(*a, **k):
        raise ConnectionError("nodename nor servname provided")
    monkeypatch.setattr(N.httpx, "post", boom)
    assert notify.send(cfg, {"x": 1}) is False


def test_divergence_fires_once_and_does_not_touch_the_bet(tmp_path, monkeypatch):
    """The committed pick is immutable — but "the bet is locked" and "you should
    not be told" are different claims, and only the first is true."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)          # commits 68–69

    # the model swings to a different bucket
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": "2026-07-27",          # SAME day as the bet — see the guard
        "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "70–71", "p_model": 0.71, "market_price": 0.5}],
        "decision": None})
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])
    assert notify.notify_divergence(conn, cfg, LOW, now) == 1
    for _ in range(4):                                   # further wakes: silent
        assert notify.notify_divergence(conn, cfg, LOW, now) == 0
    assert len(calls) == 1

    e = calls[0]["embeds"][0]
    assert "unchanged" in e["description"]
    assert any("70–71" in f["value"] for f in e["fields"])
    # THE BET ITSELF MUST NOT HAVE MOVED
    assert conn.execute("SELECT bucket FROM app_bets").fetchone()[0] == "68–69"


def test_no_divergence_alert_when_the_model_still_agrees(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)          # model still says 68–69
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_divergence(conn, cfg, LOW, now) == 0


def test_settlement_alert_reports_win_loss_once_and_omits_pnl(tmp_path, monkeypatch):
    from highcast import appbet
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)          # 68–69 at price 0.40
    db.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": "2026-07-27", "high_f": 80,
        "low_f": 69, "raw": _final_cli_raw(cfg, "2026-07-27"),
        "fetched_at": "t"}])                             # 69 -> inside 68–69
    appbet.settle(conn, cfg, now)
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])
    assert notify.notify_settled(conn, cfg, now) == 1
    assert notify.notify_settled(conn, cfg, now) == 0    # never repeats
    e = calls[0]["embeds"][0]
    assert "WON" in e["title"] and "69" in e["description"]
    names = [f["name"] for f in e["fields"]]
    assert names == ["Bet", "Actual"]
    # No P&L: the stake is a flat notional $1 that exists only to give the ledger
    # a unit, so a dollar figure here would read as performance when it is not.
    assert not any("P&L" in n or "pnl" in n.lower() for n in names)


def test_divergence_never_compares_two_different_days(tmp_path, monkeypatch):
    """REGRESSION (2026-07-27). Once a metric's window closes the desk rolls to
    TOMORROW, and `live_decision` then returns a different target date with its
    own strike ladder. Comparing that against today's locked bet fired a bogus
    "model moved to 66–67" when the bet was Jul 27 (69–70) and the quote was
    Jul 28 — two different markets, no disagreement at all."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_a_bet(conn, cfg, monkeypatch)          # commits 68–69 for 07-27

    # the desk has rolled: a different bucket, but for TOMORROW
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": "2026-07-28",
        "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "66–67", "p_model": 0.58, "market_price": 0.10}],
        "decision": None})
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_divergence(conn, cfg, LOW, now) == 0
    assert conn.execute("SELECT diverged_utc FROM app_bets").fetchone()[0] is None

    # ...but a genuine same-day disagreement still fires
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": "2026-07-27",
        "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "70–71", "p_model": 0.71, "market_price": 0.50}],
        "decision": None})
    assert notify.notify_divergence(conn, cfg, LOW, now) == 1


def _commit_at_80(conn, cfg, monkeypatch):
    """The real 2026-07-28 LOW: a sound pick on an expensive contract.

    model 74% vs market 80% = -6% edge, so `staking.evaluate` says do not place —
    but the app still commits its own bet (it only refuses a VETOED pick), which
    is exactly the state this alert watches.
    """
    from highcast import signals
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": "2026-07-27",
        "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "68–69", "lo": 68, "hi": 69, "p_model": 0.74,
                    "market_price": 0.80}],
        "decision": type("D", (), {"reason": "no edge: -6%"})()})
    # 06:00 LA, past knee 6 (KLAX LOW moved 4 -> 6 on 2026-07-31, see
    # gating.MIN_DECISION_HOUR)
    now = datetime(2026, 7, 27, 13, 0, tzinfo=timezone.utc)
    assert appbet.place(conn, cfg, LOW, now)["placed"]
    return now


def _quote(price, p_model=0.74, label="68–69", target="2026-07-27"):
    from highcast import signals
    return lambda c, g, s, n: {
        "target": target, "pred": {"issued_at": n.isoformat()},
        "market": [{"label": label, "p_model": p_model, "market_price": price}],
        "decision": None}


def test_price_drop_alert_fires_once_when_the_contract_gets_cheap(tmp_path,
                                                                 monkeypatch):
    """The whole point: 0.80 was 'no profit', 0.65 clears the gate at +9%."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)

    monkeypatch.setattr(signals, "live_decision", _quote(0.65))
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 1
    for _ in range(5):                        # the 5-minute wake must stay quiet
        assert notify.notify_price_drop(conn, cfg, LOW, now) == 0
    assert len(calls) == 1

    e = calls[0]["embeds"][0]
    flat = {f["name"]: f["value"] for f in e["fields"]}
    assert flat["Price"] == "0.80 → **0.65**"
    assert flat["Edge now"].startswith("**+9%**")
    assert "1.54" in flat["Pays per 1.00 staked"]        # 1/0.65, was 1/0.80=1.25
    # the committed bet is a prediction, not a position — it must not move
    assert conn.execute("SELECT bucket, price FROM app_bets").fetchone()[1] == 0.80


def test_no_price_alert_while_the_edge_is_still_negative(tmp_path, monkeypatch):
    """A price that merely ticks down is not news — 0.78 is still -4%."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)
    monkeypatch.setattr(signals, "live_decision", _quote(0.78))
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 0


def test_no_price_alert_when_the_edge_opens_because_the_model_rose(tmp_path,
                                                                  monkeypatch):
    """Asked for explicitly: alert on the MARKET coming down, not on model drift.

    price 0.85 > the 0.80 paid, so even though 0.95-0.85 = +10% clears the gate,
    the contract got MORE expensive and there is nothing to tell him about."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)
    monkeypatch.setattr(signals, "live_decision", _quote(0.85, p_model=0.95))
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 0


def test_price_alert_never_reads_tomorrows_ladder(tmp_path, monkeypatch):
    """Same guard as the divergence alert: once the window closes the desk rolls
    to TOMORROW, whose 68–69 is a different contract at a different price."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)
    monkeypatch.setattr(signals, "live_decision",
                        _quote(0.20, target="2026-07-28"))
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 0
    assert conn.execute("SELECT price_alert_utc FROM app_bets").fetchone()[0] is None


def test_price_alert_stops_once_the_bet_has_settled(tmp_path, monkeypatch):
    """A cheap contract on a decided market is not an opportunity."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)
    db.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": "2026-07-27", "high_f": 80,
        "low_f": 69, "raw": _final_cli_raw(cfg, "2026-07-27"),
        "fetched_at": "t"}])
    appbet.settle(conn, cfg, now)
    monkeypatch.setattr(signals, "live_decision", _quote(0.10))
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 0


def test_price_alert_failed_send_is_retried_not_lost(tmp_path, monkeypatch):
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    now = _commit_at_80(conn, cfg, monkeypatch)
    monkeypatch.setattr(signals, "live_decision", _quote(0.65))
    monkeypatch.setattr(notify, "send", lambda c, p: False)     # webhook down
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 0
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_price_drop(conn, cfg, LOW, now) == 1


def test_price_alert_is_a_silent_no_op_without_a_webhook(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, hook="")
    conn = db.connect(cfg.db_path)
    assert notify.notify_price_drop(conn, cfg, LOW,
                                    datetime.now(timezone.utc)) == 0


def _at(la_hour, day=27):
    """A UTC instant that is `la_hour` in Los Angeles (PDT = UTC-7)."""
    return datetime(2026, 7, day, la_hour + 7, 2, tzinfo=timezone.utc)


def _preview_market(monkeypatch, price=0.40, p_model=0.62, target="2026-07-27"):
    from highcast import signals
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "target": target, "pred": {"issued_at": n.isoformat()},
        "market": [{"label": "68–69", "p_model": p_model, "market_price": price},
                   {"label": "70–71", "p_model": 0.20, "market_price": 0.25}],
        "decision": None})


def test_low_preview_fires_at_17_and_18_manila_once_each(tmp_path, monkeypatch):
    """Three pings a night: 17:00 preview, 18:00 preview, 19:00 the real bet."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])

    for la_hour in (2, 3):                       # 17:00 and 18:00 Manila
        assert notify.notify_low_preview(conn, cfg, LOW, _at(la_hour)) == 1
        for _ in range(5):                       # the 5-minute wake stays quiet
            assert notify.notify_low_preview(conn, cfg, LOW, _at(la_hour)) == 0
    assert len(calls) == 2
    assert "17:00 Manila preview" in calls[0]["embeds"][0]["title"]
    assert "18:00 Manila preview" in calls[1]["embeds"][0]["title"]


def test_low_preview_says_it_is_not_committed_and_names_the_commit_hour(
        tmp_path, monkeypatch):
    """The gap between hour 2 (55%) and hour 6 (68%) is why this must not read
    like a call. (Knee moved 4 -> 6 on 2026-07-31, see gating.MIN_DECISION_HOUR.)"""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    calls = []
    monkeypatch.setattr(notify, "send", lambda c, p: (calls.append(p), True)[1])
    notify.notify_low_preview(conn, cfg, LOW, _at(2))

    e = calls[0]["embeds"][0]
    assert "not** committed" in e["description"]
    # 21:00 Manila = LA hour 6, the new commit hour (was 19:00 at old knee 4)
    assert "21:00 Manila" in e["description"]       # computed from the real gate
    flat = {f["name"]: f["value"] for f in e["fields"]}
    assert flat["Bucket"] == "**68–69**"
    # accuracy is never quoted without how much is still unknown
    assert flat["Bucket right at this hour"] == "55%"
    assert flat["Still a forecast"] == "96%"


def test_low_preview_is_silent_at_the_commit_hour(tmp_path, monkeypatch):
    """Hour 4 belongs to notify_pending — two messages about one hour is noise."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_low_preview(conn, cfg, LOW, _at(4)) == 0
    assert notify.notify_low_preview(conn, cfg, LOW, _at(1)) == 0   # too early
    assert notify.notify_low_preview(conn, cfg, LOW, _at(11)) == 0  # too late


def test_low_preview_never_previews_the_high(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_low_preview(conn, cfg, get_metric(CFG, "high"),
                                     _at(2)) == 0


def test_low_preview_never_shows_tomorrows_ladder(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch, target="2026-07-28")   # desk has rolled
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_low_preview(conn, cfg, LOW, _at(2)) == 0


def test_low_preview_failed_send_is_retried_not_lost(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: False)      # webhook down
    assert notify.notify_low_preview(conn, cfg, LOW, _at(2)) == 0
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    assert notify.notify_low_preview(conn, cfg, LOW, _at(2)) == 1


def test_low_preview_does_not_place_a_bet(tmp_path, monkeypatch):
    """THE guarantee: previewing at 17:00 must not commit at 17:00."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    notify.notify_low_preview(conn, cfg, LOW, _at(2))
    notify.notify_low_preview(conn, cfg, LOW, _at(3))
    assert conn.execute("SELECT COUNT(*) FROM app_bets").fetchone()[0] == 0


# ---------------------------------------------------------------- KMIA / multi-station

def test_kmia_low_previews_the_two_hours_before_commit():
    """KMIA LOW is a RELATIVE entry: "the two hours before commit", with no
    independent reason to prefer those clock hours. So it is asserted as the
    RULE, computed from the live gate -- never as a literal.

    ⚠️ THIS TEST USED TO ASSERT `== (3, 4)` WHILE ITS OWN DOCSTRING DESCRIBED
    "the two hours before commit". When the KMIA LOW commit hour moved h5 -> h7
    on 2026-07-31, the literal stopped satisfying the rule and the test kept
    passing anyway -- so it did not merely fail to catch the drift, it HELD THE
    STALE VALUE IN PLACE. Previews fired at h3/h4 against an h7 commit, 4 and 3
    hours early instead of 2 and 1, until 2026-08-01.

    A test that pins a literal cannot notice that the literal stopped meaning
    what its docstring says. Pin the relationship instead.
    """
    from highcast import gating
    knee = gating.min_decision_hour("KMIA", "low")
    assert notify.PREVIEW_HOURS["KMIA"]["low"] == (knee - 2, knee - 1)


def test_klax_preview_hours_are_absolute_and_deliberately_do_not_track_commit():
    """KLAX's entries are ABSOLUTE -- specific clock hours chosen because
    something measurable happens at them (h2 is where overnight MAE collapses
    1.67 -> 1.06). That is a fact about when information arrives, not about when
    we choose to act, so they correctly did NOT move when the commit hours did.

    Asserted as literals ON PURPOSE, and the asymmetry with the KMIA test above
    is the point: applying the relative rule here would discard a measurement.
    They now sit further ahead of commit than they used to, which is earlier
    warning rather than a stale value.
    """
    from highcast import gating
    assert notify.PREVIEW_HOURS["KLAX"]["low"] == (2, 3)
    assert gating.min_decision_hour("KLAX", "low") - 2 > 2, (
        "KLAX low previews now coincide with the relative rule — the two "
        "conventions have converged and this test no longer distinguishes them")
    # KLAX high is no longer asserted here: it is DISABLED, not absolute.
    # See test_disabled_previews_are_deliberate.


def test_an_unlisted_station_has_no_preview_entry():
    """See the negative test below for the behaviour that guarantees."""
    assert "KJFK" not in notify.PREVIEW_HOURS


def test_every_preview_hour_is_strictly_before_its_own_commit_hour():
    """THE invariant. A "preview" at or after the commit hour is not a preview --
    it either duplicates `notify_pending`'s announcement or, worse, implies the
    bet is still open when it has already been placed. Checked against the real
    gate for every (station, metric) rather than against literals, so adding a
    station or moving a knee cannot quietly break the relationship."""
    from highcast import gating
    for icao, per_metric in notify.PREVIEW_HOURS.items():
        for metric, hours in per_metric.items():
            knee = gating.min_decision_hour(icao, metric)
            assert knee is not None, f"{icao} {metric}: previews but no commit hour"
            # An EMPTY tuple is a legitimate state -- that metric's previews are
            # switched OFF. KLAX high was disabled 2026-08-01 because both of its
            # hours preceded the 09:00 refresh and so previewed the 4F-cold
            # evening forecast. Which entries are off, and why, is pinned by
            # `test_disabled_previews_are_deliberate`; this test governs only the
            # hours that are actually set.
            assert all(h < knee for h in hours), (
                f"{icao} {metric}: preview hours {hours} not all before knee {knee}")


def test_disabled_previews_are_deliberate():
    """An empty preview tuple must be a DECISION, not a typo.

    `test_every_preview_hour_is_strictly_before_its_own_commit_hour` is
    vacuously true for an empty tuple, so without this a metric could lose its
    alerts silently. Pinning the disabled set means switching one back on is a
    visible edit here, next to the reason it was switched off.
    """
    disabled = {(icao, metric)
                for icao, per in notify.PREVIEW_HOURS.items()
                for metric, hours in per.items() if not hours}
    assert disabled == {("KLAX", "high")}, (
        "the set of disabled previews changed. KLAX high is off because both "
        "its hours (6, 8) preceded the 09:00 day-ahead refresh, so every "
        "preview was built on the evening forecast — MAE 4.42F, bias -4.05F "
        "against the morning refresh's 1.45F. Re-enabling needs measured hours.")


def test_a_high_preview_is_labelled_high_and_quotes_the_high_commit_hour():
    """REGRESSION GUARD. `build_low_preview` hardcoded "LOW" in its title and
    read MIN_DECISION_HOUR["low"], so the first HIGH preview would have been
    labelled LOW and pointed Julian at the LOW's 21:00 clock instead of the
    HIGH's 04:00 one. (Knees moved 2026-07-31: LOW 4 -> 6 = 21:00 Manila,
    HIGH 11 -> 13 = 04:00 Manila; see gating.MIN_DECISION_HOUR.)"""
    from highcast import gating
    q = {"label": "80–81", "p_model": 0.46, "market_price": 0.55}
    e = notify.build_low_preview(
        CFG, "2026-07-29", 6, "21:00", q, -0.09,
        gating.hour_accuracy("KLAX", "high", 6)["hit"],
        gating.extremum_still_future("KLAX", "high", 6), metric="high")["embeds"][0]
    assert e["title"].startswith("HIGH · KLAX")
    assert "04:00 Manila" in e["description"]     # the HIGH's commit, not the LOW's
    assert "21:00" not in e["description"]


def test_low_preview_sends_nothing_for_a_station_with_no_preview_hours(
        tmp_path, monkeypatch):
    """An unlisted station (KJFK stands in -- it has no PREVIEW_HOURS entry)
    must not silently inherit KLAX's 17:00/18:00 clock. That would preview at
    an hour with no relationship to a commit hour nobody has measured for it
    either -- the same "no fallback" contract as MIN_DECISION_HOUR."""
    import dataclasses
    cfg = dataclasses.replace(_cfg(tmp_path),
                              station=dataclasses.replace(CFG.station, icao="KJFK"))
    conn = db.connect(cfg.db_path)
    _preview_market(monkeypatch)
    monkeypatch.setattr(notify, "send", lambda c, p: True)
    for la_hour in range(17):    # _at() only supports LA hours 0..16 (UTC-7)
        assert notify.notify_low_preview(conn, cfg, LOW, _at(la_hour)) == 0


def test_every_alert_title_names_the_station():
    """KLAX and KMIA post to the SAME Discord webhook (one channel), so a
    title without the station is ambiguous -- two "LOW · 2026-07-29 · model
    moved" messages an hour apart, one per station, would be indistinguishable.
    Every builder must carry cfg.station.icao in its title, not just
    build_message (which already had it)."""
    import dataclasses
    mia = dataclasses.replace(CFG, station=dataclasses.replace(
        CFG.station, icao="KMIA"))
    bet = _bet()

    assert "KMIA" in notify.build_message(mia, bet)["embeds"][0]["title"]
    assert "KMIA" in notify.build_divergence(
        mia, bet, "70–71", 0.55)["embeds"][0]["title"]
    assert "KMIA" in notify.build_price_drop(
        mia, bet, 0.30, 0.60, 0.30)["embeds"][0]["title"]
    assert "KMIA" in notify.build_settlement(
        mia, {**bet, "won": 1, "actual": 69})["embeds"][0]["title"]
    assert "KMIA" in notify.build_low_preview(
        mia, "2026-07-29", 3, "18:00",
        {"label": "68–69", "market_price": 0.40, "p_model": 0.50},
        0.10, 0.50, 0.90)["embeds"][0]["title"]
