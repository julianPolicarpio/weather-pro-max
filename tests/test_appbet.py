"""The app's committed bet, and the four rules that make it honest.

This ledger exists to be compared against Julian's own spreadsheet after a week.
That comparison is worthless if the app can advantage itself, so every integrity
rule is pinned here. If one of these tests is ever "simplified" away, the
scorecard silently stops measuring prediction and starts measuring hindsight.
"""
import dataclasses
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from highcast import appbet, db
from highcast.config import StakingConfig, load_config
from highcast.metric import get_metric
from highcast.timeutil import climate_day_bounds

CFG = load_config()
LOW = get_metric(CFG, "low")
HIGH = get_metric(CFG, "high")
TZ = ZoneInfo(CFG.station.tz)


def _final_product(cfg, d):
    """A CLI product id issued safely after `d`'s climate day closes -- what
    `appbet.settle` now requires before it will grade a bet against it."""
    end = climate_day_bounds(date.fromisoformat(d), cfg.station.std_utc_offset)[1]
    issued = end + timedelta(hours=1)
    return issued.strftime("%Y%m%d%H%M") + "-TEST-CDUS00-CLITEST"


def _early_product(cfg, d):
    """A CLI product id issued one hour BEFORE `d`'s climate day closes --
    still preliminary."""
    end = climate_day_bounds(date.fromisoformat(d), cfg.station.std_utc_offset)[1]
    issued = end - timedelta(hours=1)
    return issued.strftime("%Y%m%d%H%M") + "-TEST-CDUS00-CLITEST"


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               staking=StakingConfig(bankroll=8.0))


def _at(local_hour, day=(2026, 7, 26)):
    return datetime(*day, local_hour, 5, tzinfo=TZ).astimezone(timezone.utc)


def _fake_decision(bucket="68–69", price=0.40, p=0.62):
    return lambda conn, cfg, spec, now: {
        "pred": {"issued_at": now.isoformat()},
        "market": [{"label": bucket, "lo": 68, "hi": 69, "p_model": p,
                    "market_price": price},
                   {"label": "70–71", "lo": 70, "hi": 71, "p_model": 0.2,
                    "market_price": 0.5}]}


def _cli(conn, cfg, d, high=80, low=68, product=None):
    """`product=None` writes a productless row (raw="{}"), matching every
    call site that predates the finality guard -- fine for RULE 2 checks
    (`_settled_actual`, unaffected by the guard) but NOT sufficient to make
    `appbet.settle` grade a bet; pass a real product (see `_final_product`)
    for tests that exercise settlement."""
    raw = json.dumps({"product": product}) if product else "{}"
    db.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": d, "high_f": high, "low_f": low,
        "raw": raw, "fetched_at": "t"}])


# ---------------------------------------------------------------- rule 1

def test_the_bet_is_immutable_once_committed(tmp_path, monkeypatch):
    """RULE 1. A bet that gets rewritten as the model changes its mind is not a
    prediction — it is a running commentary that always looks right afterwards."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.40))
    first = appbet.place(conn, cfg, LOW, _at(11))
    assert first["placed"] and first["bucket"] == "68–69"

    # The model now prefers a different bucket at a different price.
    monkeypatch.setattr(signals, "live_decision", _fake_decision("72–73", 0.90))
    second = appbet.place(conn, cfg, LOW, _at(12))
    assert second["placed"] is False
    assert "already committed" in second["reason"]

    rows = list(conn.execute("SELECT bucket, price FROM app_bets"))
    assert len(rows) == 1
    assert rows[0]["bucket"] == "68–69" and rows[0]["price"] == 0.40


# ---------------------------------------------------------------- rule 2

def test_refuses_once_the_outcome_is_settled(tmp_path, monkeypatch):
    """RULE 2. Betting after the CLI value is public is not a prediction."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _cli(conn, cfg, "2026-07-26")                    # outcome already known
    monkeypatch.setattr(signals, "live_decision", _fake_decision())
    out = appbet.place(conn, cfg, LOW, _at(11))
    assert out["placed"] is False
    assert "settled" in out["reason"]
    assert conn.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"] == 0


# ---------------------------------------------------------------- rule 3

@pytest.mark.parametrize("spec,knee", [(LOW, 6), (HIGH, 13)])  # moved 4->6, 11->13 on 2026-07-31 (gating.MIN_DECISION_HOUR)
def test_refuses_before_the_decision_hour(tmp_path, monkeypatch, spec, knee):
    """RULE 3. Before the measured knee the pick is still stepping between
    hours; the desk will not act there and neither may the app."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision())
    early = appbet.place(conn, cfg, spec, _at(max(0, knee - 1)))
    assert early["placed"] is False and "decision hour" in early["reason"]
    assert appbet.place(conn, cfg, spec, _at(knee))["placed"] is True


def test_refuses_to_bet_blind_on_a_station_with_no_measured_decision_hour(
        tmp_path, monkeypatch):
    """RULE 3 (unmeasured station). `gating.MIN_DECISION_HOUR` only has "KLAX"
    and "KMIA" entries today (KMIA measured 2026-07-28). A station with no
    entry at all -- KJFK stands in here, since it never got a
    `scripts/hour_curve.py` run -- must never fall back to hour 0: this gate
    commits REAL MONEY, so "no measured knee" has to mean "refuse", not "bet
    at any hour". Assert the refusal fires regardless of local_hour, including
    hours well past KLAX's own knee."""
    import dataclasses
    from highcast import signals
    cfg = dataclasses.replace(_cfg(tmp_path),
                              station=dataclasses.replace(CFG.station, icao="KJFK"))
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision())
    for hour in (0, 4, 11, 20):
        out = appbet.place(conn, cfg, LOW, _at(hour))
        assert out["placed"] is False, hour
        assert "decision hour" in out["reason"], hour
    assert conn.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"] == 0


# ---------------------------------------------------------------- economics

def test_records_the_price_and_payout_at_placement(tmp_path, monkeypatch):
    """Kalshi contracts settle at $1, so staking S at price p returns S/p. The
    payout is frozen at placement — recomputing it later from a moved price
    would flatter or punish the bet for something it never took."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11), stake=3.0)
    r = conn.execute("SELECT * FROM app_bets").fetchone()
    assert r["stake"] == 3.0
    assert r["payout_if_win"] == pytest.approx(6.0)      # 3 / 0.50
    assert r["placed_manila"]                            # his sheet's time column


def test_picks_the_models_top_bucket_not_the_cheapest(tmp_path, monkeypatch):
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.40, 0.62))
    appbet.place(conn, cfg, LOW, _at(11))
    assert conn.execute("SELECT bucket FROM app_bets").fetchone()["bucket"] == "68–69"


# ---------------------------------------------------------------- settlement

def test_settles_win_and_loss_against_the_cli_value(tmp_path, monkeypatch):
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11))
    _cli(conn, cfg, "2026-07-26", low=69,               # inside 68–69 -> win
         product=_final_product(cfg, "2026-07-26"))
    assert appbet.settle(conn, cfg, _at(20)) == 1
    r = conn.execute("SELECT * FROM app_bets").fetchone()
    assert r["actual"] == 69 and r["won"] == 1 and r["settled_utc"]

    # settling twice must not double-count or alter the committed bet
    assert appbet.settle(conn, cfg, _at(21)) == 0
    assert conn.execute("SELECT bucket FROM app_bets").fetchone()["bucket"] == "68–69"


def test_settlement_marks_a_loss_outside_the_bucket(tmp_path, monkeypatch):
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11))
    _cli(conn, cfg, "2026-07-26", low=71,               # outside -> loss
         product=_final_product(cfg, "2026-07-26"))
    appbet.settle(conn, cfg, _at(20))
    assert conn.execute("SELECT won FROM app_bets").fetchone()["won"] == 0


def test_settle_refuses_a_preliminary_cli_product(tmp_path, monkeypatch):
    """A CLI product issued before the climate day closes must not settle a
    bet -- see scripts/migrate_prelim_settlement.py. The bet stays UNGRADED
    and is retried on the next settle() call once a final product lands;
    never silently dropped."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11))
    _cli(conn, cfg, "2026-07-26", low=82,
         product=_early_product(cfg, "2026-07-26"))
    assert appbet.settle(conn, cfg, _at(20)) == 0
    r = conn.execute("SELECT actual, won, settled_utc FROM app_bets").fetchone()
    assert r["actual"] is None and r["won"] is None and r["settled_utc"] is None

    _cli(conn, cfg, "2026-07-26", low=69,
         product=_final_product(cfg, "2026-07-26"))
    assert appbet.settle(conn, cfg, _at(21)) == 1
    r = conn.execute("SELECT actual, won FROM app_bets").fetchone()
    assert r["actual"] == 69 and r["won"] == 1


def test_kmia_2026_08_01_preliminary_product_is_not_graded_won(tmp_path, monkeypatch):
    """REGRESSION: the exact incident. IEM's only CLI product for this date,
    202608012033-KMFL-CDUS42-CLIMIA, was issued 20:33Z -- 8.5h before KMIA's
    05:00Z climate-day close -- reporting low 82 at "6:24 AM". A storm then
    dropped the station from 87.98F to 75.02F and the true low landed at
    74F. The old code graded a >=81 bet WON against that preliminary report;
    it had actually LOST. This must now stay ungraded rather than settle
    wrong. See scripts/migrate_prelim_settlement.py for the full account."""
    from highcast import signals
    kmia = dataclasses.replace(
        CFG.station, icao="KMIA", tz="America/New_York", std_utc_offset=-5)
    cfg = dataclasses.replace(_cfg(tmp_path), station=kmia)
    conn = db.connect(cfg.db_path)
    market = [{"label": "≥81", "lo": 81, "hi": 999, "p_model": 0.15,
               "market_price": 0.10}]
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "pred": {"issued_at": n.isoformat()}, "market": market})
    now = datetime(2026, 8, 1, 7, 5, tzinfo=ZoneInfo(kmia.tz)).astimezone(timezone.utc)
    placed = appbet.place(conn, cfg, LOW, now)
    assert placed["placed"] and placed["bucket"] == "≥81"

    db.upsert(conn, "cli_reports", [{
        "station": "KMIA", "date": "2026-08-01", "high_f": 94, "low_f": 82,
        "raw": json.dumps({"product": "202608012033-KMFL-CDUS42-CLIMIA",
                           "high": "94", "low": "82"}),
        "fetched_at": "t"}])
    later = now + timedelta(hours=1)
    assert appbet.settle(conn, cfg, later) == 0
    r = conn.execute("SELECT actual, won, settled_utc FROM app_bets").fetchone()
    assert r["actual"] is None and r["won"] is None and r["settled_utc"] is None


def test_settle_warns_when_the_settled_value_disagrees_with_the_market(
        tmp_path, monkeypatch, caplog):
    """CROSS-CHECK. A settled value landing in a bucket the market had
    already priced near zero is a strong signal the settlement source is
    wrong -- see the KMIA 2026-08-01 incident (settled 82 while the market
    held >=81 at 0.005)."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11))
    market = [{"label": "68–69", "lo": 68, "hi": 69, "p_model": 0.05,
               "market_price": 0.01, "yes_bid": 0.01}]
    conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
             committed, within_window, place, market_json)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, "2026-07-26", "low", "intraday",
         _at(12).isoformat(), 1, 1, 1, json.dumps(market)))
    conn.commit()
    _cli(conn, cfg, "2026-07-26", low=69, product=_final_product(cfg, "2026-07-26"))
    with caplog.at_level("WARNING"):
        assert appbet.settle(conn, cfg, _at(20)) == 1
    assert "SETTLEMENT DISAGREES WITH MARKET" in caplog.text


def test_open_tail_buckets_settle_correctly():
    assert appbet._bucket_bounds("≤63") == (-999, 63)
    assert appbet._bucket_bounds("≥72") == (72, 999)
    assert appbet._bucket_bounds("68–69") == (68, 69)
    assert appbet._bucket_bounds(None) == (None, None)


def test_ledger_counts_only_settled_bets(tmp_path, monkeypatch):
    """An unsettled bet must not count as a loss — that is the bug that would
    make the app look worse than it is during the comparison week."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    appbet.place(conn, cfg, LOW, _at(11), stake=1.0)
    led = appbet.ledger(conn, cfg)
    assert led["n"] == 1 and led["n_settled"] == 0 and led["win_rate"] is None
    _cli(conn, cfg, "2026-07-26", low=69, product=_final_product(cfg, "2026-07-26"))
    appbet.settle(conn, cfg, _at(20))
    led = appbet.ledger(conn, cfg)
    assert led["n_settled"] == 1 and led["wins"] == 1 and led["win_rate"] == 1.0
    assert led["pnl"] == pytest.approx(1.0)             # 2.0 back on 1.0 staked


def test_no_market_price_means_no_bet(tmp_path, monkeypatch):
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision",
                        lambda c, g, s, n: {"pred": {"issued_at": n.isoformat()},
                                            "market": None})
    out = appbet.place(conn, cfg, LOW, _at(11))
    assert out["placed"] is False and "market" in out["reason"]


def test_coverage_reports_missed_days(tmp_path, monkeypatch):
    """A missed day must be VISIBLE. "the app went 4-1" means something very
    different if it only bet on 5 of 7 days, and the integrity rules cannot
    guarantee the laptop was awake — only that it never bet late."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    monkeypatch.setattr(signals, "live_decision", _fake_decision("68–69", 0.50))
    # The app bets on the 25th while it is still open, then all three days
    # settle. Order matters: writing the CLI first would (correctly) trip Rule 2
    # and refuse the bet — which is what a first draft of this test did.
    assert appbet.place(conn, cfg, LOW, _at(11, (2026, 7, 25)))["placed"]
    for d in ("2026-07-24", "2026-07-25", "2026-07-26"):
        _cli(conn, cfg, d, low=69)
    cov = appbet.coverage(conn, cfg)["per_metric"]
    assert cov["low"]["expected"] == 3
    assert cov["low"]["committed"] == 1
    assert cov["low"]["missing"] == ["2026-07-26", "2026-07-24"]
    assert cov["high"]["committed"] == 0        # never bet the high at all


def test_coverage_ignores_days_that_have_not_settled(tmp_path):
    """Before a day settles there is nothing to have missed — counting it would
    show a permanent false gap for today."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    db.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": "2026-07-26", "high_f": None,
        "low_f": None, "raw": "{}", "fetched_at": "t"}])
    assert appbet.coverage(conn, cfg)["per_metric"]["low"]["expected"] == 0


def test_refuses_a_vetoed_pick_but_still_bets_when_there_is_merely_no_edge(
        tmp_path, monkeypatch):
    """RULE 5. Two kinds of place=False, and they must be treated differently.

    "No edge" is a SOUND number that just is not worth trading — this ledger
    measures accuracy, not profit, so it bets anyway (gating on edge would leave
    most days blank and make the week's comparison unreadable).

    A REGIME VETO means the app has already judged its own number untrustworthy —
    e.g. a day-ahead built with every morning feature NaN, which once produced
    73.2F against guidance at 75.6-89.1. Betting that logs a loss the app knew was
    coming, which corrupts the scorecard rather than measuring it.
    """
    from highcast import signals, staking
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    market = [{"label": "68–69", "lo": 68, "hi": 69, "p_model": 0.62,
               "market_price": 0.40}]

    # (a) merely no edge -> still bets
    no_edge = staking._no("no edge: 62% vs market 60% = +2% < 5% required")
    assert no_edge.veto is False
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "pred": {"issued_at": n.isoformat()}, "market": market,
        "decision": no_edge})
    assert appbet.place(conn, cfg, LOW, _at(11))["placed"] is True

    # (b) regime veto -> refuses
    cfg2 = _cfg(tmp_path / "b")
    conn2 = db.connect(cfg2.db_path)
    vetoed = staking.blocked("issued before any morning observation existed")
    assert vetoed.veto is True
    monkeypatch.setattr(signals, "live_decision", lambda c, g, s, n: {
        "pred": {"issued_at": n.isoformat()}, "market": market,
        "decision": vetoed})
    out = appbet.place(conn2, cfg2, LOW, _at(11))
    assert out["placed"] is False and "vetoed" in out["reason"]
    assert conn2.execute("SELECT COUNT(*) c FROM app_bets").fetchone()["c"] == 0


def test_blocked_preserves_the_veto_flag_when_carrying_numbers_through(tmp_path):
    """`blocked(reason, of=decision)` keeps the observed price/probability for the
    audit trail — it must also mark the result as vetoed, or rule 5 silently
    stops firing for exactly the case it exists for."""
    from highcast import staking
    d = staking.evaluate([{"label": "68–69", "p_model": 0.65, "market_price": 0.35}],
                         StakingConfig(bankroll=8.0))
    assert d.place is True and d.veto is False
    b = staking.blocked("out of regime", of=d)
    assert b.place is False and b.veto is True
    assert b.bucket == d.bucket and b.price == d.price   # trail preserved
