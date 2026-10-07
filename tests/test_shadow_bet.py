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


def _t(day, hour=13):
    return datetime(2026, 7, day, hour, 0, tzinfo=timezone.utc)


MARKET = [{"label": "83-84", "lo": 83, "hi": 84, "p_model": 0.50, "market_price": 0.35},
          {"label": "85-86", "lo": 85, "hi": 86, "p_model": 0.30, "market_price": 0.25}]


def test_record_shadow_bets_top_bucket_dedups_and_settles(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    sid = signals.record_shadow(conn, cfg, SPEC, "2026-07-24", MARKET, _t(24))
    assert sid is not None
    row = conn.execute("SELECT kind, place, bucket, price, contracts, p_model "
                       "FROM signals WHERE id=?", (sid,)).fetchone()
    # bet the model's TOP bucket (83-84, p_model 0.50), place=1, 1 contract
    assert row["kind"] == "shadow" and row["place"] == 1
    assert row["bucket"] == "83-84" and row["contracts"] == 1
    assert abs(row["price"] - 0.35) < 1e-9

    # dedup: one shadow per (target, metric)
    assert signals.record_shadow(conn, cfg, SPEC, "2026-07-24", MARKET, _t(24)) is None

    # settle: actual 84 is inside 83-84 -> WIN, pnl_unit = 1 - price
    db.upsert(conn, "cli_reports", [{"station": cfg.station.icao,
              "date": "2026-07-24", "high_f": 84, "low_f": 60, "raw": "",
              "fetched_at": "t"}])
    signals.settle_signals(conn, cfg, _t(25))
    r = conn.execute("SELECT won, pnl_unit FROM signals WHERE id=?", (sid,)).fetchone()
    assert r["won"] == 1 and abs(r["pnl_unit"] - (1 - 0.35)) < 1e-9


def test_shadow_kept_separate_from_real_ledger(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    signals.record_shadow(conn, cfg, SPEC, "2026-07-24", MARKET, _t(24))
    db.upsert(conn, "cli_reports", [{"station": cfg.station.icao,
              "date": "2026-07-24", "high_f": 84, "low_f": 60, "raw": "",
              "fetched_at": "t"}])
    signals.settle_signals(conn, cfg, _t(25))
    assert signals.shadow_report(conn, cfg, "high")["n_paper_trades"] == 1
    # the real edge-based ledger must NOT count the shadow bet
    assert signals.paper_report(conn, cfg, "high")["n_paper_trades"] == 0


def _pred_row(icao, date, kind, issued, point, metric="high"):
    import json as _j
    return {"station": icao, "target_date": date, "issued_at_utc": issued,
            "kind": kind, "point_f": point,
            "quantiles": _j.dumps({"0.5": point}),
            "buckets": _j.dumps({str(int(point)): 1.0}), "metric": metric}


def test_desk_prefers_dayahead_inside_the_window(tmp_path, monkeypatch):
    """The preference is METRIC-AWARE and the two markets are opposite:
      HIGH @9am: max still in the future -> day-ahead better (1.509 vs 1.620)
      LOW  @6am: min already observed    -> nowcast far better (0.806 vs 1.469)
    So day-ahead only for the HIGH in-window; nowcast otherwise."""
    from highcast import db as hdb, gating, signals as sig
    from highcast.fetchers import kalshi
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, t, metric=None: None)
    target = _t(24).astimezone(__import__("zoneinfo").ZoneInfo(cfg.station.tz)).date().isoformat()
    hdb.upsert(conn, "predictions", [
        _pred_row(cfg.station.icao, target, "dayahead", "2026-07-24T16:00:00+00:00", 79.0),
        _pred_row(cfg.station.icao, target, "intraday", "2026-07-24T17:00:00+00:00", 81.0)])

    real_gate = gating.gate_state
    for within, expected in ((True, "dayahead"), (False, "intraday")):
        monkeypatch.setattr(gating, "gate_state",
                            lambda *a, _w=within, **k: {**real_gate(*a, **k),
                                                        "within_window": _w})
        out = sig.live_decision(conn, cfg, SPEC, _t(24))
        assert out["kind"] == expected, f"within_window={within} -> {out['kind']}"

    # the LOW must keep the nowcast even INSIDE its window (opposite of HIGH)
    from highcast.metric import get_metric as _gm
    low = _gm(cfg, "low")
    hdb.upsert(conn, "predictions", [
        _pred_row(cfg.station.icao, target, "dayahead", "2026-07-24T13:00:00+00:00", 64.0, "low"),
        _pred_row(cfg.station.icao, target, "intraday", "2026-07-24T14:00:00+00:00", 66.0, "low")])
    monkeypatch.setattr(gating, "gate_state",
                        lambda *a, **k: {**real_gate(*a, **k), "within_window": True})
    assert sig.live_decision(conn, cfg, low, _t(24))["kind"] == "intraday"


def test_desk_rolls_to_tomorrow_when_todays_window_has_closed(tmp_path, monkeypatch):
    """Once today's window closes, today's contract is dead. The desk must roll
    to tomorrow when tomorrow's window is open and a forecast exists, instead of
    sitting on the dead market for the rest of the day."""
    from datetime import timedelta as _td
    from zoneinfo import ZoneInfo as _Z
    from highcast import db as hdb, gating, signals as sig
    from highcast.fetchers import kalshi
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda cfg, t, metric=None: None)
    now = _t(24)
    today = now.astimezone(_Z(cfg.station.tz)).date()
    tomorrow = (today + _td(days=1)).isoformat()
    # only TOMORROW has a forecast; today's window is closed, tomorrow's is open
    hdb.upsert(conn, "predictions", [
        _pred_row(cfg.station.icao, today.isoformat(), "intraday",
                  "2026-07-24T10:00:00+00:00", 80.0),
        _pred_row(cfg.station.icao, tomorrow, "dayahead",
                  "2026-07-24T11:00:00+00:00", 77.0)])
    real = gating.gate_state
    monkeypatch.setattr(gating, "gate_state", lambda c, cf, t, m, n: {
        **real(c, cf, t, m, n), "within_window": (t == tomorrow)})
    out = sig.live_decision(conn, cfg, SPEC, now)
    assert out["target"] == tomorrow, f"desk stayed on the dead market: {out['target']}"
    assert out["gate"]["within_window"] is True
