"""The desk must refuse to trade a day-ahead prediction built with no morning
observations.

Caught live on real money 2026-07-26: the desk rolled to tomorrow, the day-ahead
model ran with every `morn_*` feature NaN (a regime it is never trained on), and
emitted 82% on "<=78F" against a market at 22% while all nine guidance models sat
at 75.6-89.1F. The edge gate read "+60% edge" and said PLACE.
"""
import dataclasses
from datetime import datetime, timedelta, timezone

from highcast import db
from highcast.config import StakingConfig, load_config
from highcast.gating import morning_obs_ready
from highcast.metric import get_metric

CFG = load_config()
HIGH = get_metric(CFG, "high")
LOW = get_metric(CFG, "low")


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db",
                               # bankroll 50, not 8: with the edge ceiling in
                               # place a placeable market must be priced >= 60c
                               # (model ~100%), and 5% of 8 dollars cannot buy
                               # one contract at that price. The two constraints
                               # conflict; the bankroll is the one that is
                               # incidental to what these tests are about.
                               staking=StakingConfig(bankroll=50.0))


def _obs(conn, cfg, valid_utc):
    db.upsert(conn, "obs", [{
        "station": cfg.station.icao, "valid_utc": valid_utc, "source": "metar",
        "tmpf": 70.0, "tmpf_precise": 70.0, "dwpf": 60.0, "sknt": 5.0,
        "drct": 260.0, "skyc": "CLR", "six_max_c": None, "six_min_c": None,
        "raw": "x", "fetched_at": "t"}])


def test_high_with_no_obs_in_window_is_not_ready(tmp_path):
    """The exact production case: tomorrow's climate day has not started, so
    there is nothing in [climate day start, 09:00 local]."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26") is False


def test_high_becomes_ready_once_a_morning_metar_lands(tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # 2026-07-26 climate day starts 08:00Z; 09:00 PDT cutoff == 16:00Z.
    _obs(conn, cfg, "2026-07-26T14:53:00+00:00")
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26") is True


def test_obs_after_the_cutoff_does_not_count(tmp_path):
    """Leakage guard: an observation later than the feature cutoff is not what
    the model was fed, so it must not unlock the trade."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _obs(conn, cfg, "2026-07-26T18:53:00+00:00")   # 11:53 PDT, past 09:00 cutoff
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26") is False


def test_obs_before_the_climate_day_does_not_count_for_the_high(tmp_path):
    """The high's window starts at the climate-day start, so the previous
    evening's observation must not satisfy it."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _obs(conn, cfg, "2026-07-26T03:53:00+00:00")   # 20:53 PDT on the 25th
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26") is False


def test_low_uses_a_24h_lookback_so_the_evening_before_counts(tmp_path):
    """The LOW's cutoff is 20:00 the prior evening with a 24h lookback -- this is
    why the LOW's tomorrow card was NOT broken and must not be blocked."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _obs(conn, cfg, "2026-07-26T01:53:00+00:00")   # 18:53 PDT on the 25th
    assert morning_obs_ready(conn, cfg, LOW, "2026-07-26") is True


def test_as_of_is_the_issue_time_not_now(tmp_path):
    """THE REGRESSION THAT MATTERS. The first version of this gate checked
    whether morning observations exist *currently*. At 03:52 PDT on 2026-07-26
    they did, so the gate opened -- but the prediction being traded was issued
    18:45 PDT the previous evening, before any of them existed, and was still
    the all-NaN 73.2F one. The window must close at min(cutoff, issued)."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    _obs(conn, cfg, "2026-07-26T10:53:00+00:00")        # 03:53 PDT, in-window
    # asking "is it ready now?" says yes ...
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26") is True
    # ... but the prediction was issued the previous evening, before that obs
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26",
                             as_of="2026-07-26T01:45:39+00:00") is False
    # and once issued after the obs, it is genuinely ready
    assert morning_obs_ready(conn, cfg, HIGH, "2026-07-26",
                             as_of="2026-07-26T11:30:00+00:00") is True




def test_live_decision_blocks_a_placeable_but_untrained_dayahead(tmp_path, monkeypatch):
    """End-to-end: a market the staking rules WOULD place gets vetoed because the
    day-ahead prediction behind it has no morning observations."""
    import json

    from highcast import gating, signals, staking
    from highcast.fetchers import kalshi

    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    # 16:30Z == 09:30 PDT Jul 26, so live_decision's target IS 2026-07-26
    now = datetime(2026, 7, 26, 16, 30, tzinfo=timezone.utc)

    # a wildly confident day-ahead prediction for a day with no morning obs
    # per-degree pmf, str-keyed as stored: nearly all mass at/below 78F
    buckets = json.dumps({str(t): (0.82 if t == 78 else 0.0225)
                          for t in range(70, 79)})
    db.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": "2026-07-26",
        "issued_at_utc": "2026-07-26T02:00:00+00:00", "kind": "dayahead",
        "point_f": 73.2, "quantiles": json.dumps({"0.1": 66.0, "0.5": 73.2, "0.9": 81.3}),
        "buckets": buckets, "metric": "high"}])
    # ⚠️ PRICED SO THE EDGE IS PLAUSIBLE. This used to quote ≤78 at 0.22
    # against a model at 100% — a +78% edge, which `StakingConfig.max_edge`
    # (0.40, mirroring `gating.MAX_EDGE`) now refuses outright. That would make
    # the assertion below pass for the WRONG reason: the market would be
    # rejected by the edge ceiling rather than taken, and the test would no
    # longer prove the regime veto is what blocks. 0.65 keeps it a clear PLACE.
    monkeypatch.setattr(kalshi, "fetch_ranges", lambda *a, **k: [
        {"label": "≤78", "lo": None, "hi": 78, "yes_bid": 0.64, "yes_ask": 0.66,
         "last_price": 0.65},
        {"label": "79–80", "lo": 79, "hi": 80, "yes_bid": 0.29, "yes_ask": 0.31,
         "last_price": 0.30}])
    monkeypatch.setattr(gating, "gate_state", lambda *a, **k: {
        "committed": True, "within_window": True, "label": "≤78", "prob": 0.82,
        "confident": True, "stable": True, "n_readings": 3})

    out = signals.live_decision(conn, cfg, HIGH, now)
    assert out["kind"] == "dayahead"
    assert out["decision"].place is False
    assert "morn_* feature was NaN" in out["decision"].reason
    # and the veto is the reason -- not a coincidental no-edge rejection
    raw = staking.evaluate(out["market"], cfg.staking, within_window=True)
    assert raw.place is True, "market must be one the staking rules would take"
