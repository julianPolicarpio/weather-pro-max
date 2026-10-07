import dataclasses
from datetime import datetime, timezone

from highcast import db, signals
from highcast.config import StakingConfig, load_config
from highcast.metric import get_metric

CFG = load_config()
SPEC = get_metric(CFG, "high")


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               staking=StakingConfig(bankroll=8.0))


def _seed(conn, cfg, market, actual, days):
    """Log `days` settled shadow snapshots with the given market + winning actual."""
    for i in range(days):
        date = f"2026-06-{i+1:02d}"
        now = datetime(2026, 6, i + 1, 13, tzinfo=timezone.utc)
        signals.record_shadow(conn, cfg, SPEC, date, market, now)
        db.upsert(conn, "cli_reports", [{"station": cfg.station.icao, "date": date,
                  "high_f": actual, "low_f": 60, "raw": "", "fetched_at": "t"}])
    signals.settle_signals(conn, cfg, datetime(2026, 7, 1, tzinfo=timezone.utc))


def test_edge_report_detects_model_beating_market(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # model gives the winning bucket 0.70; the (de-vigged) market only 0.45
    market = [{"label": "83-84", "lo": 83, "hi": 84, "p_model": 0.70, "market_price": 0.45},
              {"label": "85-86", "lo": 85, "hi": 86, "p_model": 0.30, "market_price": 0.55}]
    _seed(conn, cfg, market, actual=84, days=15)   # actual 84 -> 83-84 wins every day
    r = signals.edge_report(conn, cfg, "high")
    assert r["n"] == 15
    assert r["model_brier"] < r["market_brier"]     # model scored better
    assert r["ci_hi"] < 0                            # CI clears zero
    assert r["verdict"] == "MODEL BEATS MARKET (edge)"


def test_edge_report_no_edge_when_model_equals_market(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # model probs == de-vigged market probs -> zero difference -> no edge
    market = [{"label": "83-84", "lo": 83, "hi": 84, "p_model": 0.45, "market_price": 0.45},
              {"label": "85-86", "lo": 85, "hi": 86, "p_model": 0.55, "market_price": 0.55}]
    _seed(conn, cfg, market, actual=85, days=15)   # top model bucket 85-86 wins
    r = signals.edge_report(conn, cfg, "high")
    assert r["n"] == 15
    assert r["verdict"] == "no significant edge (CI straddles 0)"


def test_edge_report_empty(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    r = signals.edge_report(conn, cfg, "high")
    assert r["n"] == 0 and r["verdict"] == "no scored days yet"
