import json

import pytest
from sandbox_helpers import make_market, three_buckets

from highcast import orderbot
from sandbox.bots import (MIN_EDGE, ForecastBot, FavoriteBot, evaluate_bucket,
                          minutes_between, normal_bucket_probs)
from sandbox.models import Snapshot

T = "2026-10-04T10:00:00+00:00"
PRE_DAY = "2026-10-04T04:00:00+00:00"      # KNYC's 2026-10-04 climate day starts 05:00Z


def snap(markets, taken_at=T):
    return Snapshot(id=7, taken_at=taken_at, markets=tuple(markets))


def test_min_edge_is_the_order_bots():
    assert MIN_EDGE == orderbot.MIN_EDGE == 0.03


def test_normal_bucket_probs_symmetric_ladder():
    p = normal_bucket_probs(three_buckets(), mu=69.5, sigma=3.0)
    t69, mid, t70 = (m.ticker for m in three_buckets())
    assert p[t69] == pytest.approx(0.3694, abs=1e-3)
    assert p[t70] == pytest.approx(0.3694, abs=1e-3)
    assert p[mid] == pytest.approx(0.2611, abs=1e-3)
    assert sum(p.values()) == pytest.approx(1.0)


def test_evaluate_bucket():
    ok, fpc, edge, why = evaluate_bucket(0.50, 0.40)   # fee 0.07*.4*.6=.0168 -> .02
    assert ok and fpc == pytest.approx(0.02) and edge == pytest.approx(0.08) and why == "edge"
    assert evaluate_bucket(0.42, 0.40)[0] is False
    assert evaluate_bucket(0.42, 0.40)[3] == "below_min_edge"
    assert evaluate_bucket(0.50, None) == (False, None, None, "no_ask")


def test_minutes_between():
    assert minutes_between(T, "2026-10-04T11:30:00+00:00") == pytest.approx(90.0)
    assert minutes_between(T, None) is None


class FixedPmf(ForecastBot):
    name = "fixed"

    def __init__(self, mu):
        self.mu = mu

    def pmf_for_event(self, markets, snapshot):
        return None if self.mu is None else normal_bucket_probs(markets, self.mu, 3.0)


def test_forecast_bot_logs_every_bucket_and_bets_only_on_edge():
    d = FixedPmf(69.5).decide(snap(three_buckets()))
    assert len(d.candidates) == 3
    placed = [c for c in d.candidates if c.placed]
    # ≤68: p .369 - ask .20 - fee .02 = .149 -> bet ; others negative
    assert [c.ticker for c in placed] == ["KXHIGHNY-26OCT04-T69"]
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T69"]
    it = d.intents[0]
    assert (it.account, it.side, it.contracts, it.price_cap, it.snapshot_id) == ("fixed", "yes", 1, 0.20, 7)
    c0 = d.candidates[0]
    assert c0.spread == pytest.approx(0.05) and c0.ask_size == 3.0
    assert c0.minutes_to_close == pytest.approx(1140.0)
    assert sum(json.loads(c0.pmf_json).values()) == pytest.approx(1.0)
    assert c0.gates_json is None and c0.regime_json is None


def test_forecast_bot_without_a_forecast_does_nothing():
    d = FixedPmf(None).decide(snap(three_buckets()))
    assert d.intents == [] and d.candidates == []


def test_favorite_buys_the_most_likely_bucket_at_the_ask():
    d = FavoriteBot().decide(snap(three_buckets(), PRE_DAY))
    assert [i.ticker for i in d.intents] == ["KXHIGHNY-26OCT04-T70"]   # mid .425 is largest
    assert d.intents[0].price_cap == 0.45 and d.intents[0].account == "favorite"
    assert d.candidates == []


# ---- F1: the control never substitutes a runner-up ------------------------------

def test_favorite_without_a_usable_ask_is_skipped_not_substituted():
    # the favorite trades at bid .99 / ask 1.00 (no usable ask); the others offer .01
    ms = [make_market("A", None, 68, bid=0.0, ask=0.01),
          make_market("B", 69, 70, bid=0.0, ask=0.01),
          make_market("C", 71, None, bid=0.99, ask=1.00)]
    assert FavoriteBot().decide(snap(ms, PRE_DAY)).intents == []


def test_favorite_with_no_ask_at_all_is_skipped():
    ms = [make_market("A", None, 68, bid=0.10, ask=0.15),
          make_market("B", 69, 70, bid=0.10, ask=0.15),
          make_market("C", 71, None, bid=0.60, ask=None)]
    assert FavoriteBot().decide(snap(ms, PRE_DAY)).intents == []


def test_favorite_skips_an_unquoted_event():
    ms = [make_market("A", None, 68, bid=None, ask=None),
          make_market("B", 69, None, bid=None, ask=None)]
    assert FavoriteBot().decide(snap(ms, PRE_DAY)).intents == []


def test_favorite_ties_go_to_ladder_order():
    ms = [make_market("A", None, 68, bid=0.30, ask=0.40),
          make_market("B", 69, 70, bid=0.30, ask=0.40),
          make_market("C", 71, None, bid=0.10, ask=0.15)]
    assert [i.ticker for i in FavoriteBot().decide(snap(ms, PRE_DAY)).intents] == ["A"]


# ---- F2: the favorite control is pre-day only ------------------------------------

def test_favorite_skips_an_event_whose_climate_day_has_started():
    s = snap(three_buckets(), "2026-10-04T04:59:00+00:00")
    assert FavoriteBot().decide(s, "2026-10-04T05:00:00+00:00").intents == []
    assert len(FavoriteBot().decide(s, "2026-10-04T04:59:59+00:00").intents) == 1


def test_favorite_skips_an_unknown_station():
    s = snap(three_buckets(station="KXXX"), PRE_DAY)
    assert FavoriteBot().decide(s).intents == []


# ---- F3: intents carry a funding priority ---------------------------------------

def test_favorite_priority_is_the_favorites_probability():
    it = FavoriteBot().decide(snap(three_buckets(), PRE_DAY)).intents[0]
    assert it.priority == pytest.approx(0.425 / (0.175 + 0.275 + 0.425))


def test_forecast_bot_priority_is_the_edge_after_fee():
    d = FixedPmf(69.5).decide(snap(three_buckets()))
    placed = next(c for c in d.candidates if c.placed)
    assert d.intents[0].priority == placed.edge_after_fee > 0


# ---- F4: the decision time, not the snapshot time, stamps the candidate ---------

def test_candidates_use_the_decision_time():
    later = "2026-10-04T10:04:00+00:00"
    d = FixedPmf(69.5).decide(snap(three_buckets()), later)
    assert {c.decided_at for c in d.candidates} == {later}
    assert d.candidates[0].minutes_to_close == pytest.approx(1136.0)   # 10:04 -> 05:00 next day
    d = FixedPmf(69.5).decide(snap(three_buckets()))                    # default: taken_at
    assert {c.decided_at for c in d.candidates} == {T}


def test_plain_forecast_bot_is_not_pre_day_gated():
    assert ForecastBot.pre_day_only is False
    d = FixedPmf(69.5).decide(snap(three_buckets()), "2026-10-04T12:00:00+00:00")
    assert len(d.candidates) == 3
