import json
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from citycast import db
from citycast.final_report import ensure_not_run, l1_key, pinned_ref, run_final

NOW = lambda: "2026-10-08T00:00:00+00:00"   # noqa: E731


def _seed(c, test_won=lambda i: int(i % 3 == 0), val_won=lambda i: int(i % 3 == 0)):
    rows = []
    for split_date, won_fn in (("2026-05-10", val_won), ("2026-08-10", test_won)):
        for i in range(300):
            won = won_fn(i)
            rows.append({"ticker": f"{split_date}-{i}", "asof_utc": f"{split_date}T12:00:00+00:00",
                         "event_ticker": "E", "station": "KNYC", "metric": "high",
                         "event_date": f"{split_date[:8]}{1 + i % 28:02d}", "lo": 1, "hi": 2,
                         "p_model": 0.5, "model_median": 1, "model_iqr": 1, "ask": 0.30, "bid": 0.28,
                         "mid": 0.29, "spread": 0.02, "market_p": 0.29, "fee": 0.02, "edge": 0.18,
                         "hours_to_close": 10, "lead_h": 10, "run_gap_to_lo": None, "run_gap_to_hi": None,
                         "month": 5, "won": won, "profit_if_bet": won - 0.32})
    with c:
        db.upsert(c, "decision_rows", rows)


def _splits(c):
    df = pd.read_sql_query("SELECT * FROM decision_rows ORDER BY ticker, asof_utc", c)
    return df[df.event_date < "2026-07-01"], df[df.event_date >= "2026-07-01"]


def _oracle(d):
    return d["won"].to_numpy().astype(float)


def _flat(d, v=0.5):
    return np.full(len(d), v)


def _third(d):
    """P(win) 0.5 on every third row (the i % 3 == 0 rows), 0 elsewhere: bets on a third of the tickers."""
    ticker_i = d["ticker"].str.rsplit("-", n=1).str[1].astype(int)
    return np.where(ticker_i % 3 == 0, 0.5, 0.0)


def _untunable(d):
    p = np.zeros(len(d))
    p[:10] = 0.9            # 10 bets at most: below MIN_BETS
    return p


def _scores(val, test, **fns):
    return {n: (None, f(val), f(test)) for n, f in fns.items()}


def _choice(c):
    return json.loads(c.execute("SELECT value FROM meta WHERE key='gate_choice'").fetchone()[0])


def test_oracle_l1_wins_and_second_run_refused(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        scores = _scores(val, test, B2=_flat, L0=_flat, L1=_oracle)
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert out["winner"] == "L1" and (tmp_path / "r.md").exists()
        assert _choice(c)["winner"] == "L1"
        with pytest.raises(RuntimeError):
            run_final(c, scores, tmp_path / "r.md", now_iso=lambda: "x")


def test_ensure_not_run_refuses_unless_forced(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        ensure_not_run(c, force=False)                       # nothing recorded yet: fine
        with c:
            c.execute("INSERT INTO meta VALUES ('final_report_run_at', 'x')")
        with pytest.raises(RuntimeError, match="already run"):
            ensure_not_run(c, force=False)
        ensure_not_run(c, force=True)


def test_force_reruns_and_overwrites_choice(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        run_final(c, _scores(val, test, B2=_flat, L0=_flat, L1=_oracle), tmp_path / "r.md", now_iso=NOW)
        run_final(c, _scores(val, test, B2=_flat, L0=_flat, L1=_flat), tmp_path / "r.md", force=True,
                  now_iso=lambda: "2026-10-09T00:00:00+00:00")
        assert _choice(c)["decided_at"] == "2026-10-09T00:00:00+00:00"


def test_untunable_l1_cannot_win_even_with_best_test_profit(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        # L1 is an oracle on test but bets < 50 times on validation, so no margin can be tuned for it
        scores = {"B2": (None, _flat(val), _flat(test)), "L0": (None, _flat(val), _flat(test)),
                  "L1": (None, _untunable(val), _oracle(test))}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert out["winner"] != "L1"
        assert "L1-B0" not in out["compare"] and out["ppd"]["L1"] is None
        text = (tmp_path / "r.md").read_text()
        assert "untunable (<50 validation bets)" in text


def test_report_marks_untunable_and_lists_b0_rows(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        scores = {"B2": (None, _flat(val), _flat(test)), "L0": (None, _untunable(val), _flat(test)),
                  "L1": (None, _oracle(val), _oracle(test))}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        text = (tmp_path / "r.md").read_text()
        assert out["winner"] == "L1"                       # an untunable L0 is skipped, not a blocker
        assert "untunable (<50 validation bets)" in text
        assert "| L1-B0 |" in text and "| B1-B0 |" in text and "| B2-B0 |" in text
        assert "L1-L0" not in out["compare"]


def test_money_losing_contender_cannot_win_even_if_it_beats_b1(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, test_won=lambda i: 0)                      # every test bet loses 0.32
        val, test = _splits(c)
        scores = {"B2": (None, _untunable(val), _untunable(test)),
                  "L0": (None, _untunable(val), _untunable(test)),
                  "L1": (None, _third(val), _third(test))}   # bets on a third: loses a third as much as B1
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        mean_vs_b1, lo_vs_b1, _ = out["compare"]["L1-B1"]
        _, _, hi_vs_b0 = out["compare"]["L1-B0"]
        assert mean_vs_b1 > 0 and lo_vs_b1 > 0 and hi_vs_b0 < 0      # beats B1, yet loses money
        assert out["winner"] == "B0"


def test_b0_wins_when_nothing_beats_it(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, test_won=lambda i: 0)
        val, test = _splits(c)
        out = run_final(c, _scores(val, test, B2=_flat, L0=_flat, L1=_flat), tmp_path / "r.md", now_iso=NOW)
        assert out["winner"] == "B0" and out["margin"] is None
        assert _choice(c) == {"winner": "B0", "margin": None, "model_ref": None,
                              "decided_at": "2026-10-08T00:00:00+00:00"}
        assert "**Winner: B0**" in (tmp_path / "r.md").read_text()


def test_b1_wins_when_it_is_the_only_eligible_contender(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, test_won=lambda i: int(i % 3 != 0))        # B1 bets everything, winning two in three
        val, test = _splits(c)
        scores = {n: (None, _untunable(val), _untunable(test)) for n in ("B2", "L0", "L1")}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert out["winner"] == "B1" and out["margin"] == pytest.approx(0.03)


def test_b2_wins_when_it_beats_b1_but_l1_does_not(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        scores = {"B2": (None, _oracle(val), _oracle(test)),
                  "L0": (None, _flat(val), _flat(test)), "L1": (None, _flat(val), _flat(test))}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert out["winner"] == "B2"


def test_l1_needs_brier_not_worse_than_b2(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)

        def idx(d):
            return d["ticker"].str.rsplit("-", n=1).str[1].astype(int).to_numpy()

        def b2(d):   # near-oracle, but wrongly bets one lost ticker in every city-day cluster
            flagged = (idx(d) % 3 != 0) & (idx(d) < 56)
            return np.where((d["won"].to_numpy() == 1) | flagged, 0.99, 0.01)

        def l1(d):   # bets only the winners (so beats B2 and B1 on profit) but is badly under-confident
            return np.where(d["won"].to_numpy() == 1, 0.36, 0.0)
        scores = {"B2": (None, b2(val), b2(test)), "L0": (None, _flat(val), _flat(test)),
                  "L1": (None, l1(val), l1(test))}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert out["compare"]["L1-B2"][1] > 0 and out["compare"]["L1-B1"][1] > 0   # profit conditions met
        assert out["brier"]["L1"] > out["brier"]["B2"]
        assert out["winner"] == "B2"


def test_score_length_mismatch_writes_nothing(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        scores = _scores(val, test, B2=_flat, L0=_flat, L1=_oracle)
        scores["L1"] = (None, scores["L1"][1], scores["L1"][2][:-1])
        with pytest.raises(ValueError):
            run_final(c, scores, tmp_path / "r.md", now_iso=NOW)
        assert c.execute("SELECT COUNT(*) FROM meta WHERE key IN ('gate_choice','final_report_run_at')"
                         ).fetchone()[0] == 0
        assert not (tmp_path / "r.md").exists()


REF = "me/laya@abcdef1"


def test_gate_choice_records_the_l1_model_ref_when_l1_wins(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        out = run_final(c, _scores(val, test, B2=_flat, L0=_flat, L1=_oracle), tmp_path / "r.md",
                        now_iso=NOW, l1_ref=REF)
        assert out["winner"] == "L1" and out["model_ref"] == REF
        assert _choice(c) == {"winner": "L1", "margin": out["margin"], "model_ref": REF,
                              "decided_at": "2026-10-08T00:00:00+00:00"}


def test_gate_choice_model_ref_is_null_for_other_winners(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        val, test = _splits(c)
        scores = {n: (None, _untunable(val), _untunable(test)) for n in ("B2", "L0", "L1")}
        out = run_final(c, scores, tmp_path / "r.md", now_iso=NOW, l1_ref=REF)
        assert out["winner"] != "L1"
        assert _choice(c)["model_ref"] is None


@pytest.mark.parametrize("ref", ["me/laya@abcdef1", "me/laya.v2@0123456789abcdef0123456789abcdef01234567"])
def test_pinned_ref_accepts_repo_at_hex_revision(ref):
    assert pinned_ref(ref) == ref
    assert l1_key(ref) == f"L1@{ref}"


@pytest.mark.parametrize("ref", ["me/laya", "me/laya@main", "me/laya@", "me/laya@abc", "laya@abcdef1",
                                 "me/laya@ABCDEF1", "me/laya@abcdef1 ", "me/laya@" + "a" * 41, "", None, 7])
def test_pinned_ref_rejects_unpinned_or_malformed(ref):
    with pytest.raises(ValueError, match="repo@"):
        pinned_ref(ref)
