import dataclasses
from datetime import datetime, timezone

from highcast import db, signals
from highcast.config import StakingConfig, load_config
from highcast.metric import get_metric

CFG = load_config()
SPEC = get_metric(CFG, "high")
MARKET = [{"label": "82-83", "lo": 82, "hi": 83, "p_model": 0.40, "market_price": 0.30},
          {"label": "84-85", "lo": 84, "hi": 85, "p_model": 0.30, "market_price": 0.25}]


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               staking=StakingConfig(bankroll=8.0))


def _t(day):
    return datetime(2026, 7, day, 13, tzinfo=timezone.utc)


def test_record_manual_logs_and_settles(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    sid = signals.record_manual(conn, cfg, SPEC, "2026-07-24", "82-83", 3, 0.30,
                                MARKET, _t(24))
    assert sid is not None
    row = conn.execute("SELECT kind, place, bucket, contracts, price, stake, edge "
                       "FROM signals WHERE id=?", (sid,)).fetchone()
    assert row["kind"] == "manual" and row["place"] == 1
    assert row["bucket"] == "82-83" and row["contracts"] == 3
    assert abs(row["price"] - 0.30) < 1e-9 and abs(row["stake"] - 0.90) < 1e-9
    assert abs(row["edge"] - (0.40 - 0.30)) < 1e-9   # p_model - price

    # settle: actual 82 lands in 82-83 -> WIN; pnl = contracts*(1-price)
    db.upsert(conn, "cli_reports", [{"station": cfg.station.icao, "date": "2026-07-24",
              "high_f": 82, "low_f": 60, "raw": "", "fetched_at": "t"}])
    signals.settle_signals(conn, cfg, _t(25))
    r = conn.execute("SELECT won, pnl FROM signals WHERE id=?", (sid,)).fetchone()
    assert r["won"] == 1 and abs(r["pnl"] - 3 * (1 - 0.30)) < 1e-9


def test_manual_bucket_must_be_in_market(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    import pytest
    with pytest.raises(ValueError):
        signals.record_manual(conn, cfg, SPEC, "2026-07-24", "99-100", 1, 0.5,
                              MARKET, _t(24))


def test_manual_separate_from_real_and_shadow_ledgers(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_manual(conn, cfg, SPEC, "2026-07-24", "82-83", 1, 0.30,
                          MARKET, _t(24))
    db.upsert(conn, "cli_reports", [{"station": cfg.station.icao, "date": "2026-07-24",
              "high_f": 82, "low_f": 60, "raw": "", "fetched_at": "t"}])
    signals.settle_signals(conn, cfg, _t(25))
    assert signals.manual_report(conn, cfg, "high")["n_paper_trades"] == 1
    # the real edge ledger and the shadow ledger must NOT count a manual bet
    assert signals.paper_report(conn, cfg, "high")["n_paper_trades"] == 0
    assert signals.shadow_report(conn, cfg, "high")["n_paper_trades"] == 0
