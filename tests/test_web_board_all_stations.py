"""STATION TABLE covers EVERY station, not just the server's own.

THE BUG. `/api/board` built its rows from `current()` — the config the server
was started with, which in production is KLAX. So the panel headed "STATION
TABLE — the physics behind the number" showed two of the four markets and gave
no sign the other two existed. Julian spotted it from the UI.

The panel's whole reason for existing is that a probability cannot be argued
with while "min stalled 17.9h, pressure falling 1.3 hPa" can. Showing that for
half the board is worse than showing none of it, because the half that is
missing looks like it was checked.

WHAT ELSE THIS ADDS, and why those two specifically:

* `coverage` — how much of the climate day has an admissible six-hour group
  behind it. On 2026-08-06 KMIA settled a low of 75F at 01:51 local while our
  best evidence was the 78.1F METAR two minutes later; the group that would
  have caught it was never filed. The desk had no way to know it was blind.
* `ensemble` — spread and bimodality, already fetched every wake into
  `forecast_ens` and displayed NOWHERE. On 2026-08-06 KLAX HIGH's 119 members
  spanned 78.0-92.0F with bimodality 1.19 (two competing scenarios) while KMIA
  HIGH sat in 84.0-88.4F at 0.45. That is the difference between a bucket bet
  and a coin toss, and it was invisible.

Both are DISPLAY. Nothing here votes on a bet.
"""
import pytest
from fastapi.testclient import TestClient

from highcast import db as hdb
from highcast.config import load_config
from highcast.web.app import create_app

from tests.test_paper_account import _load_archive


@pytest.fixture
def client(tmp_path, monkeypatch):
    from highcast import kalshi_client
    monkeypatch.setattr(kalshi_client, "load_credentials",
                        lambda *a, **kw: (None, None))
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    _load_archive(hdb.connect(cfg.db_path))
    return TestClient(create_app(cfg))


def test_board_covers_every_station(client):
    rows = client.get("/api/board").json()["rows"]
    assert {r["station"] for r in rows} == {"KLAX", "KMIA"}, \
        sorted({r["station"] for r in rows})


def test_board_covers_every_market(client):
    rows = client.get("/api/board").json()["rows"]
    got = {(r["station"], r["metric"]) for r in rows}
    assert got == {("KLAX", "low"), ("KLAX", "high"),
                   ("KMIA", "low"), ("KMIA", "high")}, sorted(got)


def test_rows_stay_grouped_by_station(client):
    """The panel renders in order; interleaving two stations' markets would
    make it unreadable."""
    rows = client.get("/api/board").json()["rows"]
    seen, order = set(), []
    for r in rows:
        if r["station"] not in seen:
            seen.add(r["station"]); order.append(r["station"])
    assert order == sorted(set(order), key=order.index)
    assert len(order) == len(set(order)), "a station's rows must be contiguous"


def test_every_row_carries_coverage(client):
    for r in client.get("/api/board").json()["rows"]:
        assert "coverage" in r, r["station"]
        c = r["coverage"]
        assert {"blind_minutes", "covered_minutes", "day_minutes",
                "missing_groups"} <= set(c), c


def test_every_row_carries_an_ensemble_slot_even_when_empty(client):
    """Absent data must be an explicit None, never a missing key — a renderer
    that has to guess produces 'undefined' in the UI."""
    for r in client.get("/api/board").json()["rows"]:
        assert "ensemble" in r, r["station"]
        e = r["ensemble"]
        assert e is None or {"spread", "p10", "p90", "bimodality"} <= set(e)


def test_the_server_station_is_not_privileged(client):
    """KLAX and KMIA rows must carry the same fields. The bug was that only
    one station was built at all; the regression to guard is a half-built
    second station."""
    rows = client.get("/api/board").json()["rows"]
    keys = {r["station"]: set(r.keys()) for r in rows}
    assert keys["KLAX"] == keys["KMIA"], keys["KLAX"] ^ keys["KMIA"]
