from highcast import staking
from highcast.config import StakingConfig

CFG = StakingConfig(bankroll=1000.0)  # explicit bankroll for sizing tests


def _ranges(*pairs):
    """pairs of (label, p_model, market_price) -> range dicts."""
    return [{"label": lbl, "p_model": p, "market_price": mp}
            for lbl, p, mp in pairs]


# ---- full_kelly ----------------------------------------------------------

def test_full_kelly_basic():
    assert abs(staking.full_kelly(0.9, 0.5) - 0.8) < 1e-9   # (0.4)/(0.5)


def test_full_kelly_no_edge_is_nonpositive():
    assert staking.full_kelly(0.5, 0.5) == 0.0
    assert staking.full_kelly(0.4, 0.6) < 0.0


def test_full_kelly_degenerate_price_is_zero():
    assert staking.full_kelly(0.9, 0.0) == 0.0
    assert staking.full_kelly(0.9, 1.0) == 0.0


# ---- size ----------------------------------------------------------------

def test_size_applies_quarter_kelly_and_hard_cap():
    # kelly full = (0.85-0.60)/0.40 = 0.625; quarter = 0.15625; cap 0.05 binds
    sz = staking.size(0.85, 0.60, CFG)
    assert sz.capped is True
    assert abs(sz.stake_frac - 0.05) < 1e-9
    assert abs(sz.stake - 50.0) < 1e-9
    assert sz.contracts == int(50.0 // 0.60)  # 83


def test_size_uncapped_uses_fractional_kelly():
    cfg = StakingConfig(bankroll=1000.0, max_stake_frac=0.50)
    sz = staking.size(0.85, 0.60, cfg)          # quarter*0.625 = 0.15625 < 0.50
    assert sz.capped is False
    assert abs(sz.stake_frac - 0.15625) < 1e-9
    assert abs(sz.stake - 156.25) < 1e-9


def test_size_no_edge_is_zero():
    sz = staking.size(0.5, 0.5, CFG)
    assert sz.stake == 0.0 and sz.stake_frac == 0.0


def test_size_bankroll_unset_forces_zero_dollars_but_reports_flag():
    sz = staking.size(0.85, 0.60, StakingConfig(bankroll=0.0))
    assert sz.bankroll_unset is True
    assert sz.stake == 0.0
    assert sz.stake_frac > 0.0   # the fraction is still computed


# ---- evaluate (the ruleset) ---------------------------------------------

def test_evaluate_places_on_confident_edge():
    d = staking.evaluate(_ranges(("77-78", 0.85, 0.60), ("79-80", 0.10, 0.30)),
                         CFG)
    assert d.place is True
    assert d.bucket == "77-78"
    assert abs(d.edge - 0.25) < 1e-9
    assert d.sizing.stake == 50.0


def test_evaluate_places_at_low_confidence_when_edge():
    # EDGE-based (framework B): a 45%-confident top bucket that the market
    # underprices at 25% is a +20% edge -> PLACE, even though it's < 80%.
    d = staking.evaluate(_ranges(("77-78", 0.45, 0.25), ("79-80", 0.30, 0.30)), CFG)
    assert d.place is True and d.bucket == "77-78"
    assert abs(d.edge - 0.20) < 1e-9


def test_evaluate_blocks_outside_window():
    d = staking.evaluate(_ranges(("77-78", 0.95, 0.40)), CFG, within_window=False)
    assert d.place is False and "window" in d.reason


def test_evaluate_blocks_without_prices():
    ranges = [{"label": "77-78", "p_model": 0.95, "market_price": None}]
    d = staking.evaluate(ranges, CFG)
    assert d.place is False and "market price" in d.reason


def test_evaluate_low_confidence_no_edge_still_blocks():
    # low confidence is fine, but no edge (market prices the top bucket fairly)
    # still blocks — the gate is edge, not confidence.
    d = staking.evaluate(_ranges(("77-78", 0.70, 0.69), ("79-80", 0.20, 0.20)),
                         CFG)
    assert d.place is False and "no edge" in d.reason
    assert d.bucket == "77-78"


def test_evaluate_blocks_when_no_edge():
    # confident (85%) but market already prices it at 83% -> edge 2% < 5%
    d = staking.evaluate(_ranges(("77-78", 0.85, 0.83)), CFG)
    assert d.place is False and "no edge" in d.reason


def test_evaluate_place_but_bankroll_unset_warns():
    d = staking.evaluate(_ranges(("77-78", 0.85, 0.60)),
                         StakingConfig(bankroll=0.0))
    assert d.place is True
    assert "bankroll is unset" in d.reason
    assert d.sizing.stake == 0.0


def test_evaluate_blocks_when_spread_too_wide():
    # confident + edge, but a 30c bid/ask spread > 10c max -> slippage gate
    r = [{"label": "77-78", "p_model": 0.90, "market_price": 0.60,
          "yes_bid": 0.45, "yes_ask": 0.75}]
    d = staking.evaluate(r, CFG)
    assert d.place is False and "too thin" in d.reason
    assert d.spread == 0.30


def test_evaluate_places_when_spread_tight():
    r = [{"label": "77-78", "p_model": 0.90, "market_price": 0.60,
          "yes_bid": 0.58, "yes_ask": 0.62}]           # 4c spread
    d = staking.evaluate(r, CFG)
    assert d.place is True and d.spread == 0.04


def test_evaluate_skips_spread_gate_without_book():
    r = [{"label": "77-78", "p_model": 0.90, "market_price": 0.60}]  # no bid/ask
    d = staking.evaluate(r, CFG)
    assert d.place is True and d.spread is None
