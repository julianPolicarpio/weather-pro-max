"""The multi-city survey: which Kalshi temperature market is most predictable.

The roster is the load-bearing part. Stations were taken from each Kalshi series'
`rules_primary`, not guessed, and two are counter-intuitive: New York is CENTRAL
PARK (KNYC) and Chicago is MIDWAY (KMDW). Scoring a model against KJFK or KORD
would measure a station the market does not settle on — the error would be
invisible in every test that did not pin the actual ICAO.
"""
import sqlite3

import pytest

from highcast import cities, db
from highcast.config import load_config

CFG = load_config()


def test_roster_uses_the_stations_kalshi_actually_settles_on():
    by_code = {c: icao for c, icao, *_ in cities.ROSTER}
    assert by_code["NY"] == "KNYC", "New York settles on Central Park, not an airport"
    assert by_code["CHI"] == "KMDW", "Chicago settles on Midway, not O'Hare"
    assert by_code["LAX"] == "KLAX"
    # every entry is a distinct, well-formed US ICAO
    icaos = [icao for _, icao, *_ in cities.ROSTER]
    assert len(icaos) == len(set(icaos))
    assert all(len(i) == 4 and i.startswith("K") for i in icaos)


def test_every_city_offers_at_least_one_market():
    for code, icao, label, hi, lo in cities.ROSTER:
        assert hi or lo, f"{code} has neither a high nor a low market"


def test_survey_is_empty_on_an_empty_db(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    assert cities.survey(conn) == []


def test_survey_ranks_by_corrected_hit_and_flags_the_current_station(tmp_path):
    """Rows sort most-predictable first, and the caller's own station is marked
    so a surface can highlight it without re-deriving which one it is."""
    conn = db.connect(tmp_path / "t.db")
    import datetime as dt
    start = dt.date(2022, 1, 1)
    # KAAA is easy (forecast always exact); KBBB is hard (alternating 6F error).
    for i in range(500):
        d = (start + dt.timedelta(days=i)).isoformat()
        run = (start + dt.timedelta(days=i - 1)).isoformat()
        for icao, err in (("KAAA", 0.0), ("KBBB", 6.0 if i % 2 else -6.0)):
            db.upsert(conn, "cli_reports", [{
                "station": icao, "date": d, "high_f": 80.0, "low_f": 60.0,
                "raw": "{}", "fetched_at": "t"}])
            db.upsert(conn, "forecasts", [{
                "station": icao, "target_date": d, "model": "mos_mex",
                "run_date": run, "max_temp_f": 80.0 + err, "fetched_at": "t",
                "metric": "high"}])
    orig = cities.ROSTER
    cities.ROSTER = [("AAA", "KAAA", "Easy", True, False),
                     ("BBB", "KBBB", "Hard", True, False)]
    try:
        rows = cities.survey(conn, station="KBBB")
    finally:
        cities.ROSTER = orig
    assert [r["city"] for r in rows] == ["AAA", "BBB"]      # easy ranks first
    assert rows[0]["corr_hit"] > rows[1]["corr_hit"]
    assert rows[0]["corr_mae"] < rows[1]["corr_mae"]
    assert rows[1]["is_current"] and not rows[0]["is_current"]


def test_survey_uses_only_day_ahead_legal_runs(tmp_path):
    """A MOS run issued ON or AFTER the target date is hindsight. Including it
    would make every city look near-perfect and the ranking meaningless."""
    conn = db.connect(tmp_path / "t.db")
    import datetime as dt
    start = dt.date(2022, 1, 1)
    for i in range(400):
        d = (start + dt.timedelta(days=i)).isoformat()
        db.upsert(conn, "cli_reports", [{
            "station": "KAAA", "date": d, "high_f": 80.0, "low_f": 60.0,
            "raw": "{}", "fetched_at": "t"}])
        # ONLY a same-day run exists — it must be ignored entirely.
        db.upsert(conn, "forecasts", [{
            "station": "KAAA", "target_date": d, "model": "mos_mex",
            "run_date": d, "max_temp_f": 80.0, "fetched_at": "t",
            "metric": "high"}])
    orig = cities.ROSTER
    cities.ROSTER = [("AAA", "KAAA", "Easy", True, False)]
    try:
        rows = cities.survey(conn)
    finally:
        cities.ROSTER = orig
    assert len(rows) == 1
    assert rows[0]["n_pairs"] == 0          # the hindsight run was excluded
    assert rows[0]["corr_hit"] is None      # so no accuracy is claimed


def test_live_survey_matches_the_api_shape():
    """Against the real DB: the shipped roster must produce rows with the keys
    the dashboard reads."""
    conn = db.connect(CFG.db_path)
    rows = cities.survey(conn, CFG.station.icao)
    if not rows:
        pytest.skip("city survey not backfilled in this DB")
    need = {"city", "station", "label", "metric", "is_current",
            "corr_hit", "corr_mae", "persist", "n_pairs"}
    assert need <= set(rows[0])
    hits = [r["corr_hit"] for r in rows if r["corr_hit"] is not None]
    assert hits == sorted(hits, reverse=True)     # ranked
    assert any(r["is_current"] for r in rows)     # our own station is present
