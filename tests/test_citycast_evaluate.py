# tests/test_citycast_evaluate.py
import json
from contextlib import closing

from citycast import db
from citycast.evaluate import market_brier, select_and_gate

F = "2026-10-07T00:00:00+00:00"


def _pred(variant, d, pmf, asof="12:00"):
    return {"station": "KNYC", "metric": "high", "event_date": d, "asof_utc": f"{d}T{asof}:00+00:00",
            "variant": variant, "fold": "x", "anchor": 50.0, "run_ext": None, "quantiles_json": "{}",
            "pmf_json": json.dumps(pmf), "k": 1.0, "created_at": F}


def _seed(c, good="city", rep_days=60, rep_good=True):
    """60 selection-period days where `good` beats everything, then `rep_days` report-period days
    where the chosen variant beats mosnorm (rep_good) or loses to it."""
    preds, cli = [], []
    days = [(f"2025-0{1 + i // 28}-{1 + i % 28:02d}", True) for i in range(60)]
    days += [(f"2026-0{4 + i // 28}-{1 + i % 28:02d}", rep_good) for i in range(rep_days)]
    for d, wins in days:
        cli.append({"station": "KNYC", "date": d, "high_f": 50, "low_f": 30, "final": 1,
                    "issued_utc": None, "raw": "{}", "fetched_at": F})
        sharp, wide = {50: 0.9, 51: 0.1}, {47: 0.5, 53: 0.5}
        preds += [_pred(good, d, sharp if wins else wide),
                  _pred("shared" if good == "city" else "city", d, {49: 0.5, 51: 0.5}),
                  _pred("mosnorm", d, wide if wins else sharp)]
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "wf_predictions", preds)


def test_selects_better_variant_and_passes_gate(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["variant"] == "city" and out[0]["gate_pass"] == 1
        row = c.execute("SELECT variant, gate_pass FROM model_choice WHERE station='KNYC'").fetchone()
        assert tuple(row) == ("city", 1)
        log = c.execute("SELECT metric, report_scored FROM eval_log").fetchall()
        assert [tuple(r) for r in log] == [("high", 1)]


def test_gate_fails_when_model_loses_in_report_period(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_good=False)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["sel_pass"] == 1 and out[0]["gate_pass"] == 0
        assert out[0]["crps_rep"] > out[0]["mos_crps_rep"]


def test_gate_fails_with_too_few_report_days(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=59)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["sel_pass"] == 1 and out[0]["rep_days"] == 59 and out[0]["gate_pass"] == 0


def test_variant_choice_ignores_report_period(tmp_path):
    # city wins the selection period; in the report period shared is far better than city.
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city", rep_days=0)
        rows = []
        for i in range(60):
            d = f"2026-0{4 + i // 28}-{1 + i % 28:02d}"
            rows += [_pred("city", d, {40: 1.0}), _pred("shared", d, {50: 1.0}), _pred("mosnorm", d, {45: 1.0})]
        with c:
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": r["event_date"], "high_f": 50, "low_f": 30,
                                        "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F}
                                       for r in rows[::3]])
            db.upsert(c, "wf_predictions", rows)
        out = select_and_gate(c, "high", lambda: F)
        assert out[0]["variant"] == "city"                      # chosen on selection only
        assert out[0]["gate_pass"] == 0                         # and then fails confirmation


def test_report_warns_when_code_changed_after_report_was_scored(tmp_path):
    from citycast.evaluate import code_fingerprint, write_report
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        select_and_gate(c, "high", lambda: F)
        assert "WARNING" not in write_report(c, tmp_path / "r.md").read_text()
        with c:
            c.execute("UPDATE eval_log SET fingerprint='0000'")       # first scoring ran on other code
        assert "WARNING" in write_report(c, tmp_path / "r.md").read_text()
        assert len(code_fingerprint()) == 64


def test_canonical_events_picks_one_event_per_city_day(tmp_path):
    from citycast.evaluate import canonical_events
    base = {"series": "KXHIGHCHI", "station": "KMDW", "metric": "high", "event_date": "2024-02-28",
            "lo": 1, "hi": 2, "open_time": None, "close_time": "2024-02-29T05:59:00+00:00",
            "result": "no", "expiration_value": "", "raw": "{}"}
    with closing(db.connect(tmp_path / "h.db")) as c:
        with c:
            db.upsert(c, "kalshi_markets", [{**base, "ticker": "A1", "event_ticker": "EV-B"},
                                            {**base, "ticker": "A2", "event_ticker": "EV-B"},
                                            {**base, "ticker": "B1", "event_ticker": "EV-A"}])
        assert canonical_events(c) == {"EV-B"}                   # more markets wins


def test_market_brier_uses_fresh_candle_only(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, good="city")
        select_and_gate(c, "high", lambda: F)
        with c:
            db.upsert(c, "kalshi_markets", [{
                "ticker": "M1", "series": "KXHIGHNY", "event_ticker": "E1", "station": "KNYC",
                "metric": "high", "event_date": "2025-01-01", "lo": 50, "hi": 51, "open_time": None,
                "close_time": "2025-01-02T04:59:00+00:00", "result": "yes", "expiration_value": "50",
                "raw": "{}"}])
            db.upsert(c, "kalshi_candles", [
                {"ticker": "M1", "end_utc": "2025-01-01T11:00:00+00:00", "yes_bid": 0.40, "yes_ask": 0.50,
                 "volume": 1.0, "open_interest": 1.0},
                {"ticker": "M1", "end_utc": "2025-01-01T08:00:00+00:00", "yes_bid": 0.10, "yes_ask": 0.20,
                 "volume": 1.0, "open_interest": 1.0}])
        res = market_brier(c, "high")
        mean, lo, hi, n = res["KNYC"]
        # model p = 1.0 (all mass on 50-51), market mid 0.45 from the 11:00 candle; won = 1
        assert n == 1 and abs(mean - (0.0 - 0.3025)) < 1e-9
