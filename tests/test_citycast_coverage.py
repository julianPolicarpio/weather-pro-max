from contextlib import closing
from datetime import date, timedelta

from citycast import db
from citycast.coverage import compute, render
from citycast.stations import Station

ST = Station("KNYC", "NYC", "NY_ASOS", "OKX", 40.78, -73.97, 40.0, "America/New_York", -5,
             "KXHIGHNY", "KXLOWTNYC")
F = "2026-10-07T00:00:00+00:00"


def _seed(c, days, missing_cli=0, gap_cli=()):
    start = date(2026, 1, 1)
    cli, mos = [], []
    for i in range(days):
        d = start + timedelta(days=i)
        if i >= missing_cli and i not in gap_cli:
            cli.append({"station": "KNYC", "date": d.isoformat(), "high_f": 40, "low_f": 30,
                        "final": 1, "issued_utc": None, "raw": "{}", "fetched_at": F})
        for model in ("MEX", "NBS"):
            # run issued at 12Z the day before: before the climate day starts (05Z)
            mos.append({"station": "KNYC", "model": model,
                        "runtime_utc": f"{(d - timedelta(days=1)).isoformat()}T12:00:00+00:00",
                        "target_date": d.isoformat(), "metric": "high", "value_f": 41.0,
                        "n_points": 1, "fetched_at": F})
    with c:
        db.upsert(c, "cli_daily", cli)
        db.upsert(c, "mos_runs", mos)
        db.upsert(c, "kalshi_markets", [{
            "ticker": "KXHIGHNY-26JAN05-T40", "series": "KXHIGHNY", "event_ticker": "KXHIGHNY-26JAN05",
            "station": "KNYC", "metric": "high", "event_date": "2026-01-05", "lo": 41, "hi": None,
            "open_time": None, "close_time": "2026-01-06T04:59:00+00:00", "result": "no",
            "expiration_value": "39", "raw": "{}"}])


def test_full_coverage_passes_and_disagreement_listed(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, 100)
        rep = compute(c, {"KNYC": ST}, today=date(2026, 4, 11), since=date(2026, 1, 1))
        s = rep["stations"]["KNYC"]
        assert s["cli_final_pct"] == 1.0 and s["mos_pct"]["MEX"] == 1.0 and s["pass"]
        assert s["kalshi_markets"] == 1 and s["kalshi_usable"] == 1 and s["kalshi_with_candles"] == 0
        assert rep["disagreements"] == [{"station": "KNYC", "metric": "high",
                                         "event_date": "2026-01-05", "kalshi": 39, "cli": 40}]
        assert rep["pass"] and "KNYC" in render(rep)


def test_missing_cli_fails_the_station(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, 100, gap_cli=range(10, 15))
        rep = compute(c, {"KNYC": ST}, today=date(2026, 4, 11), since=date(2026, 1, 1))
        assert rep["stations"]["KNYC"]["cli_final_pct"] == 0.95 and not rep["pass"]


def test_cli_measured_from_its_own_start(tmp_path):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c, 100, missing_cli=30)
        rep = compute(c, {"KNYC": ST}, today=date(2026, 4, 11), since=date(2026, 1, 1))
        s = rep["stations"]["KNYC"]
        assert s["cli_final_pct"] == 1.0 and s["cli_start"] == "2026-01-31" and s["pass"]
        assert "2026-01-31" in render(rep)
