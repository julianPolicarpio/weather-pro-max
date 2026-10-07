"""Band calibration and the two refusals it enables.

WHY THIS EXISTS. The app placed three bets at a stated 81% on 2026-07-29/30 and
lost all three, and logged three more "PLACE" decisions that bought ZERO
contracts and were then graded W/L as if they were real positions. Both are
covered here, because both were live money and neither had a test.

The load-bearing tests are `test_a_probability_above_every_measured_band_is_refused`
(the 81% case) and `test_a_place_that_buys_no_contracts_is_not_a_place` (the
scorecard-poisoning case).
"""
import dataclasses

import pytest

from highcast import gating, staking
from highcast.config import StakingConfig, load_config


def _cfg(**kw):
    base = load_config("config/klax.yaml").staking
    return dataclasses.replace(base, **kw)


def _r(label, p, price, **kw):
    return {"label": label, "p_model": p, "market_price": price, **kw}


# ── the calibration table itself ────────────────────────────────────────────

def test_the_seventy_to_eighty_band_is_split_not_uniform_after_the_hour_move():
    """2026-08-01: at the OLD hours all four markets overstated in the 70-80%
    band -- the finding this whole calibration layer rested on (see the
    docstring this test used to carry). At the NEW commit hours only two still
    do (KLAX HIGH +4.72%, KMIA LOW +2.40%); the other two flipped to a small
    UNDER-statement (KLAX LOW -0.52%, KMIA HIGH -0.79%). This is the same
    headline as `HOUR_CALIB`'s: the later, honester hour removed most of the
    overconfidence the correction existed to fix. Assert the exact split so a
    further re-measurement erasing or reversing it is caught, not silently
    absorbed."""
    bands = {key: dict(v) for key, v in gating.BAND_CALIB.items()}
    assert bands[("KLAX", "high", 13)][0.8] > 0, "KLAX high should still overstate at 70-80%"
    assert bands[("KMIA", "low", 7)][0.8] > 0, "KMIA low should still overstate at 70-80%"
    assert bands[("KLAX", "low", 6)][0.8] < 0, "KLAX low should now understate at 70-80%"
    assert bands[("KMIA", "high", 14)][0.8] < 0, "KMIA high should now understate at 70-80%"


def test_bands_are_ordered_and_the_top_band_refusal_survives_where_unmeasured():
    """2026-08-01 STRUCTURAL CHANGE: bands no longer universally stop at 0.80.
    At the OLD hours the >=80% region held n<=15 records everywhere (KLAX LOW's
    n=2 hit 50%), which is why the table stopped there and every market refused
    above 0.80. At the NEW hours KMIA HIGH (n=244), KMIA LOW (n=108) and KLAX
    HIGH (n=76) each measured a real band above 0.80 and ship it -- KMIA HIGH's
    is now the best-evidenced band in the project. KLAX LOW alone still has none
    (n=13, still too thin to measure), so it is the live example of the refusal
    this test protects.

    Ordering must still hold everywhere, and the refusal contract for a
    genuinely unmeasured top band must still hold where one exists."""
    for key, bands in gating.BAND_CALIB.items():
        uppers = [u for u, _ in bands]
        assert uppers == sorted(uppers), f"{key} bands out of order"
    assert gating.band_calib("KLAX", "low", 6, 0.90) is None
    assert gating.band_calib("KMIA", "high", 14, 0.90) is not None


def test_a_probability_above_every_measured_band_is_refused():
    """THE 81% CASE. Three real losses were placed here.

    2026-08-01: three of four markets now ship a measured band above 0.80 (see
    the structural-change test above), so 0.81/0.95 no longer refuse there --
    that is the intended, measured improvement, not a regression. KLAX LOW
    still has no band above 0.80 (n=13), so it remains the live example of the
    refusal this test protects."""
    assert gating.band_calib("KLAX", "low", 6, 0.81) is None
    assert gating.band_calib("KLAX", "low", 6, 0.95) is None
    assert gating.band_calib("KLAX", "low", 6, 0.65) is not None


def test_no_fallback_for_an_unknown_market_or_hour():
    """Same no-silent-fallback contract as `hour_calib`: borrowing another
    market's curve would hand a confident number to something unmeasured."""
    assert gating.band_calib("KLAX", "low", 11, 0.65) is None   # wrong hour
    assert gating.band_calib("KSAN", "low", 4, 0.65) is None    # unknown station


def test_calibration_shades_the_band_that_overstates():
    """2026-08-01: KLAX LOW's 70-80% band flipped to a small UNDER-statement at
    its new h6 commit hour (see the split-band test above), so it no longer
    demonstrates a downward shade. KLAX HIGH's 70-80% band at its new h13 hour
    still overstates (+4.72%), so it is now the example."""
    p = gating.band_calib("KLAX", "high", 13, 0.75)
    assert p is not None and p < 0.75, "the +4.72% band must shade downward"


def test_calibration_can_raise_an_understating_band():
    """KLAX LOW's 60-70% band UNDER-states by 1.33% at its new h6 commit hour
    (was 2.9% at the old h4 -- smaller, but the same direction). Correcting
    only downward would leave real edge on the table in the band the model is
    best in."""
    p = gating.band_calib("KLAX", "low", 6, 0.65)
    assert p is not None and p > 0.65


def test_calibration_is_clamped_into_a_real_probability():
    """KMIA HIGH's <70% band offset is +0.0187, so a raw probability near zero
    would go negative without the clamp -- and Kelly sizing would consume a
    negative 'probability' without complaint.

    2026-08-01: re-hour'd from KLAX high h11 / KMIA high h11 (both now stale
    keys that return None) to h13 / h14. KLAX HIGH's <60% band flipped to a
    NEGATIVE offset (-0.0532) at its new hour, so it no longer drives p_hat
    negative for a small raw input -- KMIA HIGH is the market that actually
    demonstrates the clamp today."""
    for station, metric, hour in (("KLAX", "high", 13), ("KMIA", "high", 14)):
        for raw in (0.001, 0.01, 0.05, 0.499):
            p = gating.band_calib(station, metric, hour, raw)
            if p is not None:
                assert 0.0 < p < 1.0


# ── how `evaluate` uses it ──────────────────────────────────────────────────

def _calib(station, metric, hour):
    return lambda p: gating.band_calib(station, metric, hour, p)


def test_evaluate_refuses_an_unmeasured_band_instead_of_betting_raw():
    d = staking.evaluate([_r("67–68", 0.81, 0.45)], _cfg(),
                         calib=_calib("KLAX", "low", 6))
    assert d.place is False
    assert "unmeasured" in d.reason


def test_without_a_calib_the_old_behaviour_is_unchanged():
    """`calib=None` must not silently start refusing: the backtest scripts and
    every existing caller pass no calibration."""
    d = staking.evaluate([_r("67–68", 0.81, 0.45)], _cfg())
    assert d.place is True


def test_calibration_is_applied_after_the_pick_is_chosen():
    """Band offsets differ per band, so correcting EVERY bucket before the
    argmax could hand back a different bucket than the model's actual read.

    2026-08-01: KLAX LOW's bands at its new h6 commit hour sit closer together
    (-0.0473 / -0.0133 / -0.0052) than at the old h4, so the old fixture's 0.69
    vs 0.71 no longer straddles a boundary sharply enough to demonstrate a
    reorder -- both would keep the same ranking either way. Moved the raw reads
    to straddle the 0.6 boundary instead: raw top is 68-69 (0.61 > 0.59), but a
    per-bucket correction would promote 66-67 (0.59+0.0473=0.6373 calibrated vs
    0.61+0.0133=0.6233)."""
    ranges = [_r("66–67", 0.59, 0.30), _r("68–69", 0.61, 0.30)]
    d = staking.evaluate(ranges, _cfg(min_edge=0.0),
                         calib=_calib("KLAX", "low", 6))
    assert d.bucket == "68–69", "calibration reordered the model's pick"


def test_a_place_that_buys_no_contracts_is_not_a_place():
    """THE SCORECARD CASE. $8 x 5% = $0.40 cannot buy a 56c contract, and the
    old code returned place=True with '~0 contracts', which appbet then graded
    as a real trade."""
    d = staking.evaluate([_r("94–95", 0.64, 0.56)], _cfg(bankroll=8.0),
                         calib=_calib("KMIA", "high", 14))
    assert d.place is False
    assert "unbuyable" in d.reason
    assert d.sizing is not None and d.sizing.contracts == 0


def test_the_unbuyable_reason_names_the_bankroll_that_would_work():
    """It is a bankroll limit, not a model opinion, and the message has to say
    so or it reads as the model declining."""
    d = staking.evaluate([_r("94–95", 0.64, 0.56)], _cfg(bankroll=8.0),
                         calib=_calib("KMIA", "high", 14))
    assert "bankroll" in d.reason


def test_a_bet_that_does_buy_a_contract_still_places():
    d = staking.evaluate([_r("67–68", 0.72, 0.40)], _cfg(bankroll=8.0),
                         calib=_calib("KLAX", "low", 6))
    assert d.place is True
    assert d.sizing.contracts >= 1
    assert "calibrated" in d.reason, "the reason must show both numbers"


def test_edge_is_computed_on_the_calibrated_number():
    """A 75% raw read on KLAX HIGH is really ~70%. Against a 72c market that is
    NEGATIVE edge, and betting it on the raw number loses money.

    2026-08-01: KLAX LOW's 70-80% band flipped to a small UNDER-statement at
    its new h6 commit hour (the calibrated number now sits slightly ABOVE the
    raw one), so it can no longer demonstrate a raw read that overstates. KLAX
    HIGH's 70-80% band at its new h13 hour still overstates (+4.72%), so it is
    now the example."""
    d = staking.evaluate([_r("84–85", 0.75, 0.72)], _cfg(),
                         calib=_calib("KLAX", "high", 13))
    assert d.place is False
    assert d.edge < 0


# ── the lowered bar ─────────────────────────────────────────────────────────

def test_min_edge_clears_the_fee_but_is_not_greedy():
    """At 1 contract Kalshi's fee rounds up to 2c at EVERY price, so a 2-point
    edge nets exactly zero. The bar has to sit above that and no higher --
    Julian asked for accurate small edges, not big ones."""
    for f in ("config/klax.yaml", "config/kmia.yaml"):
        me = load_config(f).staking.min_edge
        assert 0.02 < me <= 0.04, f"{f} min_edge={me} is outside the fee floor"


def test_a_three_point_edge_is_now_tradeable():
    d = staking.evaluate([_r("67–68", 0.66, 0.30)], _cfg(bankroll=8.0),
                         calib=_calib("KLAX", "low", 6))
    assert d.place is True, "a small honest edge must survive the bar"


# ── the NBS regime guard (found live 2026-07-31) ────────────────────────────

def test_nbs_guard_refuses_when_the_feed_has_gone_silent(tmp_path):
    """CAUGHT LIVE. `nbm_nbs` stopped returning rows on 2026-07-26 and the
    day-ahead LOW -- which carries seven nbs_* columns -- drifted 5-7F cold
    against unanimous guidance, putting 35% on '<=60F' in July. The HIGH had a
    morn_* guard for exactly this failure; the LOW had none."""
    import yaml
    from highcast import db as hdb, gating
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(p))
    conn = hdb.connect(cfg.db_path)
    assert gating.nbs_guidance_ready(conn, cfg, "2026-07-31") is False
    hdb.upsert(conn, "forecasts", [{
        "station": "KLAX", "target_date": "2026-07-31", "model": "nbm_nbs",
        "run_date": "2026-07-31", "max_temp_f": 67.0,
        "fetched_at": "2026-07-31T06:50:00+00:00", "metric": "low"}])
    assert gating.nbs_guidance_ready(conn, cfg, "2026-07-31") is True


def test_nbs_guard_is_per_station(tmp_path):
    """Both stations share one SQLite file; KMIA's NBS row must not satisfy
    KLAX's guard."""
    import yaml
    from highcast import db as hdb, gating
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(p))
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [{
        "station": "KMIA", "target_date": "2026-07-31", "model": "nbm_nbs",
        "run_date": "2026-07-31", "max_temp_f": 79.0,
        "fetched_at": "2026-07-31T06:50:00+00:00", "metric": "low"}])
    assert gating.nbs_guidance_ready(conn, cfg, "2026-07-31") is False


def test_nbs_guard_accepts_a_stale_row(tmp_path):
    """Presence, not freshness. NBS refreshes a few times a day and the model
    trained on rows of exactly that age -- only a MISSING row is the regime
    change, and vetoing on staleness would refuse ordinary days."""
    import yaml
    from highcast import db as hdb, gating
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(p))
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "forecasts", [{
        "station": "KLAX", "target_date": "2026-07-31", "model": "nbm_nbs",
        "run_date": "2026-07-20", "max_temp_f": 67.0,
        "fetched_at": "2026-07-20T06:50:00+00:00", "metric": "low"}])
    assert gating.nbs_guidance_ready(conn, cfg, "2026-07-31") is True


# ── the logged column must keep its meaning ─────────────────────────────────

def test_the_logged_p_model_stays_RAW_not_calibrated():
    """`signals.record_signal` writes Decision.p_model into a column holding
    thousands of historical rows, and `band_calib_eval.py` measures calibration
    off stated-vs-outcome. If this field became the corrected number, new rows
    would be corrected and old ones raw in the SAME column, and the next
    calibration run would correct an already-corrected number."""
    d = staking.evaluate([_r("67–68", 0.75, 0.30)], _cfg(bankroll=8.0),
                         calib=_calib("KLAX", "low", 6))
    assert d.p_model == 0.75, "p_model was overwritten with the calibrated value"
    assert d.p_hat is not None and d.p_hat != d.p_model
    # and the EDGE must be built from the calibrated number, not the raw one
    assert abs(d.edge - (d.p_hat - 0.30)) < 1e-9


def test_p_hat_is_carried_on_every_refusal_too():
    """The signals trail is the only minute-by-minute record of what the model
    thought; a refusal must not erase which number the call was made on."""
    d = staking.evaluate([_r("67–68", 0.75, 0.74)], _cfg(),
                         calib=_calib("KLAX", "low", 6))
    assert d.place is False
    assert d.p_model == 0.75 and d.p_hat is not None


def test_p_hat_equals_p_model_when_no_calibration_is_supplied():
    d = staking.evaluate([_r("67–68", 0.72, 0.40)], _cfg(bankroll=8.0))
    assert d.p_hat == d.p_model == 0.72


def test_the_nbs_veto_is_scoped_to_metrics_that_read_nbs():
    """FALSE-VETO BUG, caught in audit. The day-ahead HIGH consumes ZERO nbs_*
    columns, so blocking it on a missing NBS feed would suppress a sound
    prediction over data it never reads. Derived from the feature list rather
    than hardcoded, so it follows if the HIGH ever gains one."""
    from highcast import gating
    assert gating.metric_uses_nbs("low") is True
    assert gating.metric_uses_nbs("high") is False


def test_the_high_still_trades_when_nbs_is_missing(tmp_path):
    """End-to-end version of the above, through `_regime_veto`."""
    import datetime as dt
    import yaml
    from highcast import db as hdb, signals
    from highcast.metric import get_metric
    raw = yaml.safe_load(open("config/klax.yaml").read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(p))
    conn = hdb.connect(cfg.db_path)
    issued = dt.datetime(2026, 7, 31, 18, 0, tzinfo=dt.timezone.utc).isoformat()
    pred = {"issued_at": issued, "buckets": {}}
    # No NBS row exists at all in this fresh DB.
    hi = signals._regime_veto(conn, cfg, get_metric(cfg, "high"),
                              "2026-07-31", "dayahead", pred)
    assert hi is None or "NBS" not in hi, \
        "the HIGH was vetoed over a feed it does not consume"


# ── CROSSING COST GATE ──────────────────────────────────────────────────────

def _one_bucket(p_model, price, bid, ask):
    """A single-bucket market snapshot, shaped as `staking.evaluate` expects."""
    return [{"label": "79–80", "lo": 79, "hi": 80, "p_model": p_model,
             "market_price": price, "yes_bid": bid, "yes_ask": ask}]


def test_a_spread_costlier_than_the_edge_is_refused():
    """THE RULE: if crossing the spread costs more than the edge is worth, the
    trade cannot pay, whatever the model thinks.

    `max_spread` alone is an ABSOLUTE cap (10c), and 10c means something very
    different at 90c than at 7c: of 11,656 tradeable quotes that PASS the 10c
    rule, 12.4% still had a spread exceeding 35% of mid (measured 2026-08-01).

    Half the spread is what you actually pay to cross, so it is compared
    directly against the edge. No arbitrary constant.

    SCOPE, so nobody over-credits this rail: it adds 31 refusals (0.4%) among
    quotes clearing `min_edge`. It does NOT catch the KMIA >=81 case
    (0.03/0.11) that prompted the review -- that claimed a +49% edge, which
    covers a 4c crossing cost, so this gate passes it correctly. The fault there
    was the edge itself, which is MAX_PLAUSIBLE_EDGE's job.

    FIXTURE NOTE: the edge must sit ABOVE `min_edge` and BELOW the half-spread,
    or `min_edge` refuses first and this gate is never reached. With
    max_spread=0.10 the half-spread caps at 0.05, so the usable window is an
    edge in (0.03, 0.05). A first draft of this test used edge=0.02 and passed
    for the wrong reason.
    """
    from highcast import staking
    from highcast.config import StakingConfig

    cfg = StakingConfig(bankroll=100.0, min_edge=0.03, max_spread=0.10)
    # bid/ask 0.03/0.13 -> spread 0.10 (at the absolute cap, allowed), half 0.05
    # edge = 0.12 - 0.08 = 0.04  -> clears min_edge, BELOW half-spread -> refuse
    bad = staking.evaluate(_one_bucket(0.12, 0.08, 0.03, 0.13), cfg, calib=None)
    assert bad.place is False, bad.reason
    assert "cross" in bad.reason.lower(), bad.reason

    # edge = 0.20 - 0.08 = 0.12 -> comfortably covers the 0.05 crossing cost
    ok = staking.evaluate(_one_bucket(0.20, 0.08, 0.03, 0.13), cfg, calib=None)
    assert "cross" not in (ok.reason or "").lower(), ok.reason


def test_the_crossing_gate_is_skipped_without_a_two_sided_book():
    """Last-price-only markets cannot be assessed; the gate must not fabricate a
    spread and refuse on it. Same contract the absolute `max_spread` check keeps."""
    from highcast import staking
    from highcast.config import StakingConfig

    cfg = StakingConfig(bankroll=100.0, min_edge=0.03, max_spread=0.10)
    d = staking.evaluate(_one_bucket(0.80, 0.50, None, None), cfg, calib=None)
    assert "costs more" not in (d.reason or "")
