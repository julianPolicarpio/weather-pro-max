import json
from datetime import datetime, timezone

import pytest

from highcast import db, signals, staking
from highcast.config import load_config, StakingConfig
from highcast.metric import get_metric

CFG = load_config()
SPEC = get_metric(CFG, "high")
STAKE = StakingConfig(bankroll=1000.0)

MARKET = [{"label": "77-78", "lo": 77, "hi": 78, "p_model": 0.85, "market_price": 0.60},
          {"label": "79-80", "lo": 79, "hi": 80, "p_model": 0.10, "market_price": 0.30}]
GATE = {"committed": True, "within_window": True, "label": "77-78", "prob": 0.85,
        "confident": True, "stable": True, "n_readings": 3}


def _cfg(tmp_path):
    import dataclasses
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db", staking=STAKE)


def _seed_cli(conn, date, high):
    db.upsert(conn, "cli_reports", [{"station": CFG.station.icao, "date": date,
              "high_f": high, "low_f": 60, "raw": "", "fetched_at": "t"}])


def _place(cfg, conn, date, when):
    d = staking.evaluate(MARKET, cfg.staking, within_window=True)
    assert d.place  # sanity: this market is a PLACE
    return signals.record_signal(conn, cfg, SPEC, date, "intraday", GATE, MARKET,
                                 d, when)


def _t(day, hour=13):
    return datetime(2026, 7, day, hour, 0, tzinfo=timezone.utc)


def test_settle_win_and_loss_unit_pnl(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _place(cfg, conn, "2026-07-22", _t(22))   # bucket 77-78
    _place(cfg, conn, "2026-07-23", _t(23))
    _seed_cli(conn, "2026-07-22", 77)          # in 77-78 -> WIN
    _seed_cli(conn, "2026-07-23", 80)          # not in 77-78 -> LOSS
    assert signals.settle_signals(conn, cfg, _t(24)) == 2
    rows = {r["target_date"]: r for r in
            conn.execute("SELECT * FROM signals ORDER BY target_date")}
    assert rows["2026-07-22"]["won"] == 1
    assert abs(rows["2026-07-22"]["pnl_unit"] - 0.40) < 1e-9   # 1 - 0.60
    assert rows["2026-07-23"]["won"] == 0
    assert abs(rows["2026-07-23"]["pnl_unit"] + 0.60) < 1e-9   # -price


def test_settle_skips_unsettled_days(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _place(cfg, conn, "2026-07-22", _t(22))    # no cli row yet
    assert signals.settle_signals(conn, cfg, _t(24)) == 0
    assert conn.execute("SELECT settled_utc FROM signals").fetchone()[0] is None


# ── the settlement-vs-market cross-check ────────────────────────────────────

def _seed_signal(conn, cfg, target, metric, created_utc, market):
    conn.execute(
        """INSERT INTO signals (station, target_date, metric, kind, created_utc,
             committed, within_window, place, market_json)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (cfg.station.icao, target, metric, "intraday", created_utc,
         1, 1, 1, json.dumps(market)))
    conn.commit()


def test_check_settlement_against_market_warns_on_disagreement(tmp_path, caplog):
    """If a settled value lands in a bucket the market had already priced
    near zero, that is very likely OUR source being wrong -- see the KMIA
    2026-08-01 incident: settled 82 while the market held 73-74 at 0.99 and
    >=81 at 0.005 (true low 74). See scripts/migrate_prelim_settlement.py."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    market = [{"label": "73-74", "lo": 73, "hi": 74, "p_model": 0.9,
               "market_price": 0.99, "yes_bid": 0.99},
              {"label": "≥81", "lo": 81, "hi": 999, "p_model": 0.01,
               "market_price": 0.005, "yes_bid": 0.005}]
    _seed_signal(conn, cfg, "2026-08-01", "high",
                datetime(2026, 8, 1, 12, tzinfo=timezone.utc).isoformat(), market)
    with caplog.at_level("WARNING"):
        info = signals.check_settlement_against_market(
            conn, cfg, "high", "2026-08-01", 82)
    assert info is not None
    assert info["price"] == pytest.approx(0.005)
    assert info["bucket"] == "≥81"
    assert "SETTLEMENT DISAGREES WITH MARKET" in caplog.text


def test_check_settlement_against_market_is_quiet_when_they_agree(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    market = [{"label": "73-74", "lo": 73, "hi": 74, "p_model": 0.9,
               "market_price": 0.99, "yes_bid": 0.99}]
    _seed_signal(conn, cfg, "2026-08-01", "high",
                datetime(2026, 8, 1, 12, tzinfo=timezone.utc).isoformat(), market)
    with caplog.at_level("WARNING"):
        info = signals.check_settlement_against_market(
            conn, cfg, "high", "2026-08-01", 74)
    assert info is None
    assert "SETTLEMENT DISAGREES" not in caplog.text


def test_check_settlement_against_market_ignores_snapshots_after_close(tmp_path):
    """Only the LAST snapshot BEFORE the climate day closed counts. KLAX's
    climate day for 2026-08-01 closes 2026-08-02T08:00Z; a snapshot taken
    after that reflects information the CLI report could not have had
    either, so it must not be used as the cross-check."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    market = [{"label": "≥81", "lo": 81, "hi": 999, "p_model": 0.01,
               "market_price": 0.005, "yes_bid": 0.005}]
    _seed_signal(conn, cfg, "2026-08-01", "high",
                datetime(2026, 8, 2, 9, tzinfo=timezone.utc).isoformat(), market)
    info = signals.check_settlement_against_market(
        conn, cfg, "high", "2026-08-01", 82)
    assert info is None


def test_no_bet_signal_settles_to_zero_and_is_excluded(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # past the window -> DO NOT PLACE (a no-bet signal)
    d = staking.evaluate(MARKET, cfg.staking, within_window=False)
    assert not d.place
    signals.record_signal(conn, cfg, SPEC, "2026-07-22", "intraday",
                          {**GATE, "within_window": False}, MARKET, d, _t(22))
    _seed_cli(conn, "2026-07-22", 77)
    assert signals.settle_signals(conn, cfg, _t(24)) == 1
    row = conn.execute("SELECT * FROM signals").fetchone()
    assert row["won"] is None and row["pnl"] == 0.0
    rep = signals.paper_report(conn, cfg, "high")
    assert rep["n_paper_trades"] == 0   # no PLACE -> not a paper trade


def test_paper_report_aggregates_first_placed_per_day(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # two placed signals same day -> the position is taken once (first)
    _place(cfg, conn, "2026-07-22", _t(22, 12))
    _place(cfg, conn, "2026-07-22", _t(22, 13))
    _place(cfg, conn, "2026-07-23", _t(23))
    _seed_cli(conn, "2026-07-22", 77)   # win
    _seed_cli(conn, "2026-07-23", 80)   # loss
    signals.settle_signals(conn, cfg, _t(24))
    rep = signals.paper_report(conn, cfg, "high")
    assert rep["n_paper_trades"] == 2          # deduped to one per day
    assert rep["wins"] == 1 and rep["win_rate"] == 0.5
    assert abs(rep["pnl_per_contract"] - (0.40 - 0.60)) < 1e-9
    assert abs(rep["avg_edge"] - 0.25) < 1e-9  # 0.85 - 0.60


def test_bucket_bounds_fallback_parses_label():
    lo, hi = signals._bucket_bounds("81-82", "[]")   # not in (empty) snapshot
    assert (lo, hi) == (81, 82)
    lo, hi = signals._bucket_bounds("83–84", "not-json")  # en-dash + bad json
    assert (lo, hi) == (83, 84)
