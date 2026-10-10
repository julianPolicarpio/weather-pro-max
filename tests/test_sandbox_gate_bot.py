import builtins
import json
import logging
import sys
from contextlib import closing
from dataclasses import replace

import pytest

from citycast import db as hdb
from sandbox.bots import ForecastBot, GateBot, evaluate_bucket
from sandbox_helpers import make_market
from test_sandbox_citycast_bot import Snap, _history

Q = {str(a): v for a, v in zip((0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
                                (64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76))}
TICKER = "KXHIGHNY-26OCT08-B69.5"


def _markets():
    # make_market has no event_date / close_time keywords: the date is fixed, the close is `close`
    m = make_market(TICKER, lo=69, hi=70, station="KNYC", metric="high", bid=0.40, ask=0.45,
                    close="2026-10-09T03:59:00+00:00")
    return [replace(m, event_date="2026-10-08")]


def _with_choice(tmp_path, winner="B1", margin=0.05, raw=None):
    p = _history(tmp_path)
    value = raw if raw is not None else json.dumps({"winner": winner, "margin": margin,
                                                    "model_ref": None, "decided_at": "2026-10-10T00:00:00+00:00"})
    with closing(hdb.connect(p)) as c, c:
        c.execute("UPDATE live_predictions SET quantiles_json=?", (json.dumps(Q),))
        c.execute("INSERT INTO meta VALUES ('gate_choice', ?)", (value,))
    return p


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_evaluate_bucket_margin_argument_keeps_default():
    assert evaluate_bucket(0.60, 0.50)[0] is True                  # default MIN_EDGE 0.03
    assert evaluate_bucket(0.60, 0.50, min_edge=0.10)[0] is False
    assert ForecastBot.min_edge == 0.03


def test_b1_gate_returns_model_probability_and_margin(tmp_path):
    bot = GateBot(_with_choice(tmp_path))
    probs, fj = bot.forecast_for_event(_markets(), Snap())
    assert abs(probs[TICKER] - 0.7) < 1e-9
    assert bot.min_edge == 0.05 and fj["source"] == "gate" and fj["winner"] == "B1"
    assert fj["citycast_issued_utc"] == "2026-10-08T00:00:00+00:00"


def test_no_choice_means_no_bets(tmp_path):
    bot = GateBot(_history(tmp_path))
    assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
    assert GateBot(str(tmp_path / "none.db")).forecast_for_event(_markets(), Snap()) == (None, None)


def test_b0_means_no_bets(tmp_path, caplog):
    raw = json.dumps({"winner": "B0", "margin": None, "model_ref": None, "decided_at": "x"})
    with caplog.at_level(logging.WARNING):
        bot = GateBot(_with_choice(tmp_path, raw=raw))
        assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
        assert bot.decide(_snapshot()).intents == []
    assert _warnings(caplog) == []                         # B0 is an answer, not a fault


def test_l1_fails_closed_without_importing_laya(tmp_path, monkeypatch, caplog):
    real_import = builtins.__import__
    attempted = []

    def spy(name, *args, **kwargs):
        if name.split(".")[0] == "laya":
            attempted.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    bot = GateBot(_with_choice(tmp_path, winner="L1", margin=0.02))
    with caplog.at_level(logging.WARNING):
        assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
        assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
    assert attempted == [] and "laya" not in sys.modules
    warned = _warnings(caplog)
    assert len(warned) == 1 and "Laya sidecar" in warned[0].getMessage()      # once, not per event


@pytest.mark.parametrize("raw", [
    '{"margin": 0.05}',                                    # no winner
    '{"winner": "Z9", "margin": 0.05}',                    # unknown winner
    '{"winner": "B1", "margin": null}',                    # B1 needs a margin
    '{"winner": "B2", "margin": null}',
    '{"winner": "B1", "margin": "wide"}',
    '{"winner": "B1", "margin": NaN}',
    '{"winner": "B1", "margin": -0.01}',
    '[1, 2]',                                              # not an object
    'not json',
])
def test_malformed_choice_means_no_bets_with_one_warning(tmp_path, caplog, raw):
    bot = GateBot(_with_choice(tmp_path, raw=raw))
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
    assert len(_warnings(caplog)) == 1


def test_b2_missing_pickle_means_no_bets_not_an_exception(tmp_path, caplog):
    bot = GateBot(_with_choice(tmp_path, winner="B2", margin=0.04), b2_path=str(tmp_path / "missing.pkl"))
    with caplog.at_level(logging.WARNING):
        assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
        assert bot.forecast_for_event(_markets(), Snap()) == (None, None)
    assert len(_warnings(caplog)) == 1
    assert bot.min_edge == 0.03                            # a failed load does not adopt the margin


def test_b2_scores_with_the_meta_labeler(tmp_path, monkeypatch):
    seen = {}

    class FakeModel:
        def predict(self, df):
            seen["cols"] = set(df.columns)
            return [0.61] * len(df)

    monkeypatch.setattr("citycast.meta.load_model", lambda path: seen.setdefault("path", str(path)) and FakeModel())
    bot = GateBot(_with_choice(tmp_path, winner="B2", margin=0.07), b2_path=str(tmp_path / "b2.pkl"))
    probs, fj = bot.forecast_for_event(_markets(), Snap())
    assert probs == {TICKER: pytest.approx(0.61)} and fj["winner"] == "B2"
    assert bot.min_edge == 0.07 and seen["path"].endswith("b2.pkl")
    assert {"station", "metric", "p_model", "ask", "edge", "lead_h"} <= seen["cols"]


def test_prediction_must_be_gated_and_fresh(tmp_path):
    ungated = _history(tmp_path / "u", gate=0)
    with closing(hdb.connect(ungated)) as c, c:
        c.execute("INSERT INTO meta VALUES ('gate_choice', ?)", (json.dumps({"winner": "B1", "margin": 0.05}),))
    assert GateBot(ungated).forecast_for_event(_markets(), Snap()) == (None, None)
    old = _history(tmp_path / "o", issued="2026-10-07T20:00:00+00:00")
    with closing(hdb.connect(old)) as c, c:
        c.execute("INSERT INTO meta VALUES ('gate_choice', ?)", (json.dumps({"winner": "B1", "margin": 0.05}),))
    assert GateBot(old).forecast_for_event(_markets(), Snap()) == (None, None)


def test_markets_without_an_ask_are_skipped(tmp_path):
    bot = GateBot(_with_choice(tmp_path))
    ms = _markets()
    no_ask = replace(ms[0], ticker="KXHIGHNY-26OCT08-T70", lo=71, hi=None, yes_ask=None)
    probs, _ = bot.forecast_for_event(ms + [no_ask], Snap())
    assert set(probs) == {TICKER}


def test_malformed_stored_quantiles_fail_closed_with_a_warning(tmp_path, caplog):
    p = _with_choice(tmp_path)
    with closing(hdb.connect(p)) as c, c:
        c.execute("UPDATE live_predictions SET quantiles_json='{not json'")
    with caplog.at_level(logging.WARNING):
        assert GateBot(p).forecast_for_event(_markets(), Snap()) == (None, None)
    assert _warnings(caplog)


def test_a_predict_bug_propagates_so_the_cycle_can_report_it(tmp_path):
    bot = GateBot(_with_choice(tmp_path))

    def boom(rows):
        raise RuntimeError("contender bug")

    assert bot._load()
    bot._predict = boom
    with pytest.raises(RuntimeError, match="contender bug"):
        bot.forecast_for_event(_markets(), Snap())


def test_gate_history_db_is_opened_read_only(tmp_path, monkeypatch):
    import sqlite3
    p = _with_choice(tmp_path)
    real_connect, opens = sqlite3.connect, []

    def spy(database, *args, **kwargs):
        opens.append((database, kwargs.get("uri")))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr("sandbox.bots.sqlite3.connect", spy)
    assert GateBot(p).forecast_for_event(_markets(), Snap())[0] is not None
    assert opens and all(uri is True and "mode=ro" in str(d) for d, uri in opens), opens


class _Snapshot(Snap):
    id = 1

    def events(self):
        return {"KXHIGHNY-26OCT08": _markets()}


def _snapshot():
    return _Snapshot()


def test_decide_bets_only_when_probability_clears_the_tuned_margin(tmp_path):
    # P(win) 0.70, ask 0.45, fee ~0.02 -> edge ~0.23: bets at margin 0.05 and not at 0.30
    d = GateBot(_with_choice(tmp_path / "a", margin=0.05)).decide(_snapshot())
    assert [i.ticker for i in d.intents] == [TICKER] and d.intents[0].account == "gate"
    assert d.candidates[0].bot == "gate" and d.candidates[0].reason == "edge"
    d = GateBot(_with_choice(tmp_path / "b", margin=0.30)).decide(_snapshot())
    assert d.intents == [] and d.candidates[0].reason == "below_min_edge"
