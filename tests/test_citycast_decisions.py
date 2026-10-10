import json
import math
from contextlib import closing

from citycast import db, decisions
from citycast.decisions import GATE_FEATURES, build, gate_features
from citycast.stations import Station
from highcast.fees import fee_per_contract

NYC = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
              "KXHIGHNY", "KXLOWTNYC")
Q = {0.02: 44.0, 0.05: 45.0, 0.1: 46.0, 0.2: 47.0, 0.3: 48.0, 0.4: 49.0, 0.5: 50.0, 0.6: 51.0,
     0.7: 52.0, 0.8: 53.0, 0.9: 54.0, 0.95: 55.0, 0.98: 56.0}
PMF = {49: 0.25, 50: 0.5, 51: 0.25}


def test_gate_features_values():
    f = gate_features(station="KNYC", metric="high", event_date="2025-01-05", lo=50, hi=51, pmf=PMF,
                      quantiles=Q, run_ext=48.0, bid=0.40, ask=0.44, asof_iso="2025-01-05T12:00:00+00:00",
                      close_iso="2025-01-06T04:59:00+00:00", climate_end_iso="2025-01-06T05:00:00+00:00")
    assert set(GATE_FEATURES) <= set(f)
    assert f["p_model"] == 0.75 and f["model_median"] == 50.0 and f["model_iqr"] == 6.0
    assert abs(f["mid"] - 0.42) < 1e-9 and abs(f["spread"] - 0.04) < 1e-9 and f["market_p"] == f["mid"]
    assert f["fee"] == 0.02 and abs(f["edge"] - (0.75 - 0.44 - 0.02)) < 1e-9
    assert f["run_gap_to_lo"] == -2.0 and f["run_gap_to_hi"] == -3.0 and f["month"] == 1
    assert abs(f["hours_to_close"] - 16.9833) < 1e-3 and f["lead_h"] == 17.0


def test_gate_features_missing_bid_and_pre_day():
    f = gate_features(station="KNYC", metric="high", event_date="2025-01-05", lo=None, hi=49, pmf=PMF,
                      quantiles=Q, run_ext=None, bid=None, ask=0.30, asof_iso="2025-01-04T12:00:00+00:00",
                      close_iso="2025-01-06T04:59:00+00:00", climate_end_iso="2025-01-06T05:00:00+00:00")
    assert f["mid"] == 0.30 and math.isnan(f["spread"]) and math.isnan(f["run_gap_to_lo"])


def _seed(c):
    F = "2026-10-07T00:00:00+00:00"
    with c:
        c.execute("INSERT INTO model_choice (station, metric, variant, sel_pass, rep_days, gate_pass, decided_at) "
                  "VALUES ('KNYC','high','shared',1,1,1,'x')")
        db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2025-01-05", "high_f": 50, "low_f": 30,
                                    "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F}])
        base = {"station": "KNYC", "metric": "high", "event_date": "2025-01-05", "fold": "x",
                "anchor": 50.0, "run_ext": None, "quantiles_json": json.dumps(Q), "k": 1.0, "created_at": F}
        db.upsert(c, "wf_predictions", [
            {**base, "asof_utc": "2025-01-05T12:00:00+00:00", "variant": "shared", "pmf_json": json.dumps(PMF)},
            {**base, "asof_utc": "2025-01-05T12:00:00+00:00", "variant": "city", "pmf_json": json.dumps({40: 1.0})}])
        mk = {"series": "KXHIGHNY", "event_ticker": "E", "station": "KNYC", "metric": "high",
              "event_date": "2025-01-05", "open_time": "2025-01-04T15:00:00+00:00",
              "close_time": "2025-01-06T14:00:00+00:00", "result": "no", "expiration_value": "50", "raw": "{}"}
        db.upsert(c, "kalshi_markets", [{**mk, "ticker": "A", "lo": 50, "hi": 51, "result": "yes"},
                                        {**mk, "ticker": "B", "lo": 52, "hi": 53},
                                        {**mk, "ticker": "C", "lo": 48, "hi": 49},
                                        # cancelled market: no yes/no result, never a decision row
                                        {**mk, "ticker": "D", "lo": 46, "hi": 47, "result": ""}])
        db.upsert(c, "kalshi_candles", [
            {"ticker": "A", "end_utc": "2025-01-05T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.44, "volume": 1, "open_interest": 1},
            {"ticker": "B", "end_utc": "2025-01-05T09:00:00+00:00", "yes_bid": 0.10, "yes_ask": 0.12, "volume": 1, "open_interest": 1},
            {"ticker": "C", "end_utc": "2025-01-05T11:30:00+00:00", "yes_bid": None, "yes_ask": 0.995, "volume": 1, "open_interest": 1}])


def test_build_uses_chosen_variant_fresh_candles_and_outcome(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        assert build(c, {"KNYC": NYC}, progress=lambda m: None) == 1     # B stale (>2 h), C ask > 0.99
        r = c.execute("SELECT ticker, p_model, won, profit_if_bet FROM decision_rows").fetchone()
        assert r[0] == "A" and r[1] == 0.75 and r[2] == 1 and abs(r[3] - (1 - 0.44 - 0.02)) < 1e-9


def _candle(ticker, end, bid, ask):
    return {"ticker": ticker, "end_utc": end, "yes_bid": bid, "yes_ask": ask, "volume": 1, "open_interest": 1}


def test_build_skips_crossed_quotes_and_falls_back_to_valid_candle(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            # A: newest candle crossed (bid > ask), older candle in the window is valid -> use the older
            db.upsert(c, "kalshi_candles", [_candle("A", "2025-01-05T11:30:00+00:00", 0.60, 0.50)])
            # B: only candle in the window is crossed -> no row
            db.upsert(c, "kalshi_candles", [_candle("B", "2025-01-05T11:00:00+00:00", 0.30, 0.20)])
        assert build(c, {"KNYC": NYC}, progress=lambda m: None) == 1
        r = c.execute("SELECT ticker, ask, bid FROM decision_rows").fetchone()
        assert (r[0], r[1], r[2]) == ("A", 0.44, 0.40)


def _seed_many(c):
    """Several as-of points and tickers so a tiny BATCH forces multiple flushes."""
    _seed(c)
    F = "2026-10-07T00:00:00+00:00"
    with c:
        base = {"station": "KNYC", "metric": "high", "event_date": "2025-01-05", "fold": "x",
                "anchor": 50.0, "run_ext": 49.0, "quantiles_json": json.dumps(Q), "k": 1.0, "created_at": F,
                "variant": "shared", "pmf_json": json.dumps(PMF)}
        extra = ["2025-01-05T13:00:00+00:00", "2025-01-05T14:00:00+00:00", "2025-01-05T15:00:00+00:00"]
        db.upsert(c, "wf_predictions", [{**base, "asof_utc": a} for a in extra])
        db.upsert(c, "kalshi_candles", [_candle(t, a, 0.30, 0.35) for t in ("A", "B", "C") for a in extra])


def _dump(c):
    return c.execute("SELECT * FROM decision_rows ORDER BY ticker, asof_utc").fetchall()


def test_build_batches_give_identical_rows(tmp_path, monkeypatch):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed_many(c)
        n_default = build(c, {"KNYC": NYC}, progress=lambda m: None)
        want = [tuple(r) for r in _dump(c)]
        assert n_default == len(want) and n_default > 2
        monkeypatch.setattr(decisions, "BATCH", 2)
        assert build(c, {"KNYC": NYC}, progress=lambda m: None) == n_default
        assert [tuple(r) for r in _dump(c)] == want


def test_build_is_atomic_and_replaces_table(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            c.execute("INSERT INTO decision_rows (ticker, asof_utc, event_ticker, station, metric, event_date, "
                      "won, profit_if_bet) VALUES ('OLD','t','E','KNYC','high','2025-01-01',0,-0.5)")
        assert build(c, {"KNYC": NYC}, progress=lambda m: None) == 1
        assert [r[0] for r in c.execute("SELECT ticker FROM decision_rows")] == ["A"]


# --- as-of, label and population guards (one test per leakage/label condition) ---------------------------
ASOF = "2025-01-05T12:00:00+00:00"
STATIONS = {"KNYC": NYC}


def _run(c):
    build(c, STATIONS, progress=lambda m: None)
    return c.execute("SELECT ticker, asof_utc, won, profit_if_bet, ask FROM decision_rows "
                     "ORDER BY ticker, asof_utc").fetchall()


def _set_candles(c, ticker, *candles):
    """Replace every candle of `ticker` with (end_utc, bid, ask) tuples."""
    with c:
        c.execute("DELETE FROM kalshi_candles WHERE ticker=?", (ticker,))
        db.upsert(c, "kalshi_candles", [_candle(ticker, e, b, a) for e, b, a in candles])


def test_market_not_yet_open_at_asof_gives_no_row(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        assert [r[0] for r in _run(c)] == ["A"]                           # control: opened before t
        with c:
            c.execute("UPDATE kalshi_markets SET open_time='2025-01-05T12:00:01+00:00' WHERE ticker='A'")
        assert _run(c) == []                                             # opens after t -> no row
        with c:
            c.execute("UPDATE kalshi_markets SET open_time=? WHERE ticker='A'", (ASOF,))
        assert [r[0] for r in _run(c)] == ["A"]                           # opening exactly at t is fine


def test_market_closed_at_or_before_asof_gives_no_row(tmp_path):
    late = "2025-01-06T06:00:00+00:00"        # a second as-of after the climate day ended (05:00Z), so the
    with closing(db.connect(tmp_path / "h.db")) as c:     # early-close rule cannot mask the close<=t rule
        _seed(c)
        with c:
            db.upsert(c, "wf_predictions", [{
                "station": "KNYC", "metric": "high", "event_date": "2025-01-05", "fold": "x", "anchor": 50.0,
                "run_ext": None, "quantiles_json": json.dumps(Q), "k": 1.0, "created_at": "2026-10-07T00:00:00+00:00",
                "asof_utc": late, "variant": "shared", "pmf_json": json.dumps(PMF)}])
        _set_candles(c, "A", ("2025-01-05T11:00:00+00:00", 0.40, 0.44), ("2025-01-06T05:30:00+00:00", 0.40, 0.44))
        for close, want in [("2025-01-06T05:30:00+00:00", [ASOF]),          # closed before t
                            (late, [ASOF]),                                 # close_time == t
                            ("2025-01-06T06:00:01+00:00", [ASOF, late])]:   # closes after t
            with c:
                c.execute("UPDATE kalshi_markets SET close_time=? WHERE ticker='A'", (close,))
            assert [r[1] for r in _run(c)] == want, close


def test_candle_window_is_open_at_t_minus_2h_and_closed_at_t(tmp_path):
    cases = [("candle ending exactly at t is used", [("2025-01-05T12:00:00+00:00", 0.40, 0.44)], 0.44),
             ("candle ending exactly at t-2h is stale", [("2025-01-05T10:00:00+00:00", 0.40, 0.44)], None),
             ("candle ending after t is the future", [("2025-01-05T12:00:01+00:00", 0.40, 0.44)], None),
             ("future candle never beats the newest candle at or before t",
              [("2025-01-05T10:00:00+00:00", 0.20, 0.30), ("2025-01-05T12:00:00+00:00", 0.40, 0.44),
               ("2025-01-05T12:30:00+00:00", 0.50, 0.55)], 0.44)]
    for i, (why, candles, want_ask) in enumerate(cases):
        with closing(db.connect(tmp_path / f"h{i}.db")) as c:
            _seed(c)
            _set_candles(c, "A", *candles)
            got = [r[4] for r in _run(c)]
            assert got == ([] if want_ask is None else [want_ask]), why


def test_non_canonical_event_gets_no_rows(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)                                                            # event E: 4 markets -> canonical
        with c:
            db.upsert(c, "kalshi_markets", [{
                "series": "KXHIGHNY", "event_ticker": "E2", "station": "KNYC", "metric": "high",
                "event_date": "2025-01-05", "open_time": "2025-01-04T15:00:00+00:00",
                "close_time": "2025-01-06T14:00:00+00:00", "result": "yes", "expiration_value": "50", "raw": "{}",
                "ticker": "A2", "lo": 50, "hi": 51}])                       # same city-day, 1 market
        _set_candles(c, "A2", ("2025-01-05T11:00:00+00:00", 0.40, 0.44))     # fully usable apart from the event
        assert [r[0] for r in _run(c)] == ["A"]


def test_labels_follow_the_settled_result(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        _set_candles(c, "B", ("2025-01-05T11:00:00+00:00", 0.28, 0.30))     # result 'no'
        with c:
            db.upsert(c, "kalshi_markets", [{
                "series": "KXHIGHNY", "event_ticker": "E", "station": "KNYC", "metric": "high",
                "event_date": "2025-01-05", "open_time": "2025-01-04T15:00:00+00:00",
                "close_time": "2025-01-06T14:00:00+00:00", "result": "scalar", "expiration_value": "50",
                "raw": "{}", "ticker": "S", "lo": 54, "hi": 55}])
        _set_candles(c, "S", ("2025-01-05T11:00:00+00:00", 0.40, 0.44))     # priced, but no yes/no outcome
        _set_candles(c, "D", ("2025-01-05T11:00:00+00:00", 0.40, 0.44))     # cancelled (result '')
        got = {r[0]: r for r in _run(c)}
        assert set(got) == {"A", "B"}
        assert got["A"][2] == 1 and abs(got["A"][3] - (1 - 0.44 - fee_per_contract(1, 0.44))) < 1e-9
        assert got["B"][2] == 0 and abs(got["B"][3] - (-0.30 - fee_per_contract(1, 0.30))) < 1e-9


def test_only_the_variant_model_choice_picked_produces_rows(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:                                      # 'city' gets its own as-of with a usable candle
            db.upsert(c, "wf_predictions", [{
                "station": "KNYC", "metric": "high", "event_date": "2025-01-05", "fold": "x", "anchor": 50.0,
                "run_ext": None, "quantiles_json": json.dumps(Q), "k": 1.0, "created_at": "2026-10-07T00:00:00+00:00",
                "asof_utc": "2025-01-05T13:00:00+00:00", "variant": "city", "pmf_json": json.dumps({40: 1.0})}])
        _set_candles(c, "A", ("2025-01-05T11:00:00+00:00", 0.40, 0.44), ("2025-01-05T12:30:00+00:00", 0.40, 0.44))
        assert [(r[1]) for r in _run(c)] == [ASOF]                          # shared chosen: no 13:00 city row
        with c:
            c.execute("UPDATE model_choice SET variant='city'")
        assert [(r[1]) for r in _run(c)] == [ASOF, "2025-01-05T13:00:00+00:00"]   # city chosen: both city rows
        assert {r[0] for r in c.execute("SELECT p_model FROM decision_rows")} == {0.0}   # city pmf, not shared


def test_market_closed_before_its_climate_day_ended_gets_no_rows(tmp_path):
    # NYC climate day 2025-01-05 ends 2025-01-06T05:00Z; an earlier close_time is not the scheduled close
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        for close, want in [("2025-01-06T04:59:00+00:00", []), ("2025-01-05T17:00:00+00:00", []),
                            ("2025-01-06T05:00:00+00:00", ["A"]), ("2025-01-06T14:00:00+00:00", ["A"])]:
            with c:
                c.execute("UPDATE kalshi_markets SET close_time=? WHERE ticker='A'", (close,))
            assert [r[0] for r in _run(c)] == want, close
