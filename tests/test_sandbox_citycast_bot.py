import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from citycast import db as hdb
from sandbox.bots import CitycastBot
from sandbox_helpers import make_market

ISSUED = "2026-10-08T00:00:00+00:00"
SNAP_AT = "2026-10-08T01:00:00+00:00"


def _history(tmp_path, gate=1, issued=ISSUED, variant="shared", choice="shared"):
    p = tmp_path / "history.db"
    with closing(hdb.connect(p)) as c, c:
        c.execute("INSERT INTO model_choice VALUES ('KNYC','high',?,1,1,-0.1,?,1,1,90,?,'x')",
                  (choice, gate, gate))
        c.execute("INSERT INTO live_predictions VALUES ('KNYC','high','2026-10-08',?,?,70,NULL,'{}',?,1.0)",
                  (issued, variant, json.dumps({"69": 0.2, "70": 0.5, "71": 0.3})))
    return str(p)


def _add_prediction(path, issued, pmf, variant="shared"):
    with closing(sqlite3.connect(path)) as c, c:
        c.execute("INSERT INTO live_predictions VALUES ('KNYC','high','2026-10-08',?,?,70,NULL,'{}',?,1.0)",
                  (issued, variant, json.dumps(pmf)))


def _markets():
    ms = [make_market("KXHIGHNY-26OCT08-B69.5", lo=69, hi=70, station="KNYC", metric="high"),
          make_market("KXHIGHNY-26OCT08-T70", lo=71, hi=None, station="KNYC", metric="high")]
    return [replace(m, event_date="2026-10-08") for m in ms]


class Snap:
    taken_at = SNAP_AT


def _ago(seconds):
    return (datetime.fromisoformat(SNAP_AT) - timedelta(seconds=seconds)).isoformat()


def test_reads_latest_gated_prediction(tmp_path):
    probs, fj = CitycastBot(_history(tmp_path)).forecast_for_event(_markets(), Snap())
    assert abs(probs["KXHIGHNY-26OCT08-B69.5"] - 0.7) < 1e-9 and fj["source"] == "citycast"


def test_ungated_stale_or_missing_db_gives_nothing(tmp_path):
    assert CitycastBot(_history(tmp_path, gate=0)).forecast_for_event(_markets(), Snap()) == (None, None)
    old = "2026-10-07T20:00:00+00:00"                          # 5 h before the snapshot
    assert CitycastBot(_history(tmp_path / "o", issued=old)).forecast_for_event(_markets(), Snap()) == (None, None)
    assert CitycastBot(str(tmp_path / "none.db")).forecast_for_event(_markets(), Snap()) == (None, None)


def test_prediction_variant_must_match_current_choice(tmp_path):
    p = _history(tmp_path, variant="city", choice="shared")
    assert CitycastBot(p).forecast_for_event(_markets(), Snap()) == (None, None)


def test_never_writes_history_db(tmp_path):
    p = _history(tmp_path)
    before = Path(p).read_bytes()
    CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert Path(p).read_bytes() == before


# ---- I1: read-only is pinned by the connection, not by the bytes surviving a SELECT ----------

def test_history_db_is_only_ever_opened_read_only(tmp_path, monkeypatch):
    p = _history(tmp_path)
    real_connect = sqlite3.connect
    opens = []

    def spy(database, *args, **kwargs):
        opens.append((database, kwargs.get("uri")))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr("sandbox.bots.sqlite3.connect", spy)
    probs, _ = CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert probs is not None                  # the read really happened
    assert opens                              # ... and went through a recorded open
    assert all(uri is True and "mode=ro" in str(database) for database, uri in opens), opens


# ---- I2: the as-of cutoff, "latest of N" and the age boundary --------------------------------

def test_prediction_issued_after_the_snapshot_is_ignored(tmp_path):
    after = "2026-10-08T01:00:01+00:00"
    only_after = _history(tmp_path / "a", issued=after)
    assert CitycastBot(only_after).forecast_for_event(_markets(), Snap()) == (None, None)

    older = "2026-10-08T00:30:00+00:00"
    with_older = _history(tmp_path / "b", issued=older)          # B69.5 mass 0.7
    _add_prediction(with_older, after, {"69": 0.5, "70": 0.4, "71": 0.1})   # would be 0.9
    probs, fj = CitycastBot(with_older).forecast_for_event(_markets(), Snap())
    assert probs["KXHIGHNY-26OCT08-B69.5"] == pytest.approx(0.7)
    assert fj["issued_utc"] == older


def test_newest_of_several_eligible_predictions_wins(tmp_path):
    older, newer = "2026-10-08T00:10:00+00:00", "2026-10-08T00:40:00+00:00"
    p = _history(tmp_path, issued=older)                          # B69.5 mass 0.7
    _add_prediction(p, newer, {"69": 0.1, "70": 0.1, "71": 0.8})  # B69.5 mass 0.2
    probs, fj = CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert probs["KXHIGHNY-26OCT08-B69.5"] == pytest.approx(0.2)
    assert probs["KXHIGHNY-26OCT08-T70"] == pytest.approx(0.8)
    assert fj["issued_utc"] == newer


def test_prediction_exactly_max_age_old_is_used(tmp_path):
    p = _history(tmp_path, issued=_ago(7200))
    probs, fj = CitycastBot(p).forecast_for_event(_markets(), Snap())
    assert probs is not None and fj["issued_utc"] == _ago(7200)


def test_prediction_one_second_over_max_age_is_ignored(tmp_path):
    p = _history(tmp_path, issued=_ago(7201))
    assert CitycastBot(p).forecast_for_event(_markets(), Snap()) == (None, None)


# ---- tables absent (live history.db before the first `citycast` run): quiet; other errors warn

def _db_without_citycast_tables(tmp_path):
    p = tmp_path / "history.db"
    with closing(sqlite3.connect(p)) as c, c:
        c.execute("CREATE TABLE kalshi_markets (ticker TEXT)")
    return str(p)


def _db_with_wrong_schema(tmp_path):                  # OperationalError, but not "no such table"
    p = tmp_path / "history.db"
    with closing(sqlite3.connect(p)) as c, c:
        c.execute("CREATE TABLE live_predictions (station TEXT)")
        c.execute("CREATE TABLE model_choice (station TEXT)")
    return str(p)


def _not_a_database(tmp_path):                        # DatabaseError
    p = tmp_path / "history.db"
    p.write_bytes(b"not a database")
    return str(p)


def test_absent_tables_give_nothing_and_do_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="sandbox.bots"):
        out = CitycastBot(_db_without_citycast_tables(tmp_path)).forecast_for_event(_markets(), Snap())
    assert out == (None, None)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("make_db", [_db_with_wrong_schema, _not_a_database])
def test_other_sqlite_errors_still_warn(tmp_path, caplog, make_db):
    with caplog.at_level(logging.WARNING, logger="sandbox.bots"):
        out = CitycastBot(make_db(tmp_path)).forecast_for_event(_markets(), Snap())
    assert out == (None, None)
    assert any("KNYC" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
