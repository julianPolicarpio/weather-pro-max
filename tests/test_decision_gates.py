"""ONE verdict, read by every surface, and it fails closed.

WHY AN AGGREGATOR AND NOT SCATTERED `if`s. The dashboard, the terminal and
(later) the bot must reach the same answer or they drift and the desk gives
advice that disagrees with itself. `board.evidence()` is assembled in Python for
exactly this reason.

WHY FAIL-CLOSED. For an unattended caller, "I could not tell" and "no" must be
the same answer. Every branch here returns a refusal on missing data or an
exception -- matching `nbs_guidance_ready`'s existing contract.
"""
import yaml

from highcast import db as hdb, gating
from highcast.config import load_config


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _ready(conn, station="KMIA", target="2026-08-02"):
    """Populate everything the gates need so a test can knock ONE thing out."""
    hdb.upsert(conn, "forecasts", [
        {"station": station, "target_date": target, "model": "gfs",
         "run_date": "2026-08-01", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-08-01T12:00:00+00:00"},
        {"station": station, "target_date": target, "model": "nbm_nbs",
         "run_date": "2026-08-01", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-08-01T12:00:00+00:00"},
    ])


def test_all_gates_green_is_ok(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is True
    assert v.refusals == ()


def test_stale_lead_2_guidance_refuses(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [
        {"station": "KMIA", "target_date": "2026-08-02", "model": "gfs",
         "run_date": "2026-07-31", "max_temp_f": 94.0, "metric": "high",
         "fetched_at": "2026-07-31T12:00:00+00:00"}])
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is False
    assert "guidance_lead" in v.refusals


def test_an_unmeasured_confidence_band_refuses(tmp_path):
    """`band_calib` returns None above the highest measured band.

    DISCOVERY 2026-08-01: this test originally exercised KMIA HIGH h14 at
    p_model=0.999 (the region all three losing bets on 2026-07-29/30 were
    stated from). That band table has since been re-derived at the new
    decision hours and KMIA HIGH h14's top band now ships all the way to 1.01
    (n=244) -- there is no longer an unmeasured region there at all, so that
    input can no longer produce this refusal. The still-open gap today is
    KLAX LOW h6's >=80% band (n=13, too thin to ship a >=0.8 catch-all entry
    for), so this test now exercises that combination instead. Same code
    path, same assertion -- only the market that is genuinely unmeasured has
    moved.
    """
    cfg = _cfg(tmp_path, src="config/klax.yaml")
    conn = hdb.connect(cfg.db_path)
    _ready(conn, station="KLAX")
    v = gating.decision_gates(conn, cfg, metric="low", target="2026-08-02",
                              hour=6, p_model=0.90)
    assert v.ok is False
    assert "unmeasured_band" in v.refusals


def test_below_the_confidence_floor_refuses(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.55)
    assert v.ok is False
    assert "confidence_floor" in v.refusals


def test_klax_high_always_refuses_on_the_floor(tmp_path):
    """It has no admissible floor by construction. Two independent lines of
    evidence say the bot should not trade it."""
    cfg = _cfg(tmp_path, src="config/klax.yaml")
    conn = hdb.connect(cfg.db_path)
    _ready(conn, station="KLAX")
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=13, p_model=0.95)
    assert v.ok is False
    assert "confidence_floor" in v.refusals


def test_the_price_rail_refuses_a_knowingly_negative_ev_buy(tmp_path):
    """Not edge-hunting — a loss floor. Paying 0.98 for a bucket calibrated near
    0.85 is a guaranteed loss after the 2c fee."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90, price=0.98)
    assert v.ok is False
    assert "price_rail" in v.refusals


def test_the_price_rail_is_skipped_when_no_price_is_supplied(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _ready(conn)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90, price=None)
    assert "price_rail" not in v.refusals


def test_a_broken_connection_fails_closed(tmp_path):
    """An exception must become a refusal, never an exception reaching a caller
    that might treat a crash as 'no opinion' and proceed."""
    cfg = _cfg(tmp_path)

    class Boom:
        def execute(self, *a, **k):
            raise RuntimeError("db is gone")

    v = gating.decision_gates(Boom(), cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.90)
    assert v.ok is False
    assert v.refusals != ()


def test_refusals_accumulate_rather_than_short_circuit(tmp_path):
    """The desk should see EVERY reason, not just the first one."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    v = gating.decision_gates(conn, cfg, metric="high", target="2026-08-02",
                              hour=14, p_model=0.55, price=0.99)
    assert v.ok is False
    assert len(v.refusals) >= 2
