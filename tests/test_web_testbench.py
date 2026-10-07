"""THE TEST BENCH — what-if knobs for the demo account, and the wall around them.

WHAT IT IS. `GET /api/demo-account` already replays the archive under the
shipped rules. The bench lets the same replay be re-run with a different commit
hour, confidence threshold, stake fraction, opening balance and so on, so a
question like "would h13 have traded more?" is answered by looking rather than
by editing a constant and restarting.

WHAT IT IS NOT, AND THIS IS THE POINT OF THIS FILE. Every knob is an ARGUMENT
TO A SIMULATION. None of them writes config, mutates `gating.MIN_DECISION_HOUR`,
or changes what the live desk or the order bot would do on the next wake. A
browser control that could retune the live decision path is a much worse idea
than it looks: this dashboard is reachable by anything on the machine, the
values feed real staking arithmetic, and a stale tab left on an experimental
setting would silently become the shipped behaviour.

So the bench is deliberately GET-only and stateless. There is no POST. The
tests below are the wall: they assert the shipped constants are identical
before and after a bench request, and that the parameters actually reach the
simulation (a knob that silently does nothing is its own kind of lie).
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from highcast import db as hdb, gating
from highcast.config import load_config
from highcast.web.app import create_app

from tests.test_paper_account import _load_archive


def _no_credentials(monkeypatch):
    from highcast import kalshi_client
    monkeypatch.setattr(kalshi_client, "load_credentials",
                        lambda *a, **kw: (None, None))


@pytest.fixture
def client(tmp_path, monkeypatch):
    _no_credentials(monkeypatch)
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    _load_archive(hdb.connect(cfg.db_path))
    return TestClient(create_app(cfg))


def test_bench_defaults_match_the_plain_endpoint(client):
    """With no knobs set the bench must be the shipped view exactly, or the
    two surfaces drift and the comparison it exists for becomes meaningless."""
    plain = client.get("/api/demo-account").json()
    bench = client.get("/api/demo-account?bench=1").json()
    assert bench["daily_plan"]["closing_balance"] == \
        plain["daily_plan"]["closing_balance"]
    assert bench["bench"]["commit_hour"] == \
        gating.MIN_DECISION_HOUR["KMIA"]["high"]


def test_commit_hour_knob_reaches_the_variants_lens(client):
    """The hour the bench is asked for must be the hour it reports on —
    otherwise the panel shows one number and labels it another."""
    body = client.get("/api/demo-account?bench=1&commit_hour=13").json()
    assert body["bench"]["commit_hour"] == 13
    hours = {v["hour"] for v in body["commit_hour_variants"]["variants"]}
    assert 13 in hours


def test_sizing_knobs_change_the_arms(client):
    """A confidence threshold of 0 must let arm A trade days a threshold of
    0.99 refuses. If both return the same rows the knob is decorative."""
    # A balance is required for the arms to be able to buy anything at all —
    # with no credentials configured the seed falls back to $0, and at $0 every
    # arm correctly refuses, which would make this comparison vacuous.
    loose = client.get(
        "/api/demo-account?bench=1&balance=50&confidence=0.10").json()
    tight = client.get(
        "/api/demo-account?bench=1&balance=50&confidence=0.99").json()

    def trades(b):
        a = next(x for x in b["sizing_arms"]["arms"]
                 if x["arm"] == "confidence_only")
        return a["n_trades"]

    assert trades(loose) > trades(tight)


def test_opening_balance_knob_is_applied(client):
    body = client.get("/api/demo-account?bench=1&balance=100").json()
    assert body["seed"]["balance_usd"] == 100.0
    assert body["seed"]["source"] == "bench"
    assert body["unit_replay"]["opening_balance"] == 100.0


def test_bench_never_mutates_the_shipped_constants(client):
    """THE WALL. Any bench request, with every knob pushed, must leave the
    decision path byte-identical."""
    before = (dict(gating.MIN_DECISION_HOUR["KMIA"]),
              dict(gating.MIN_DECISION_HOUR["KLAX"]),
              dict(gating.CONFIDENCE_FLOOR),
              {k: list(v) for k, v in gating.BAND_CALIB.items()})
    client.get("/api/demo-account?bench=1&commit_hour=9&confidence=0.05"
               "&aggressive_frac=0.99&min_payoff=0.01&balance=1000")
    after = (dict(gating.MIN_DECISION_HOUR["KMIA"]),
             dict(gating.MIN_DECISION_HOUR["KLAX"]),
             dict(gating.CONFIDENCE_FLOOR),
             {k: list(v) for k, v in gating.BAND_CALIB.items()})
    assert before == after


def test_bench_never_writes_to_the_db(client, tmp_path):
    client.get("/api/demo-account?bench=1&commit_hour=13&balance=50")
    conn = hdb.connect(tmp_path / "t.db")
    assert conn.execute(
        "SELECT COUNT(*) c FROM paper_account_txns").fetchone()["c"] == 0


def test_bench_has_no_post_route(client):
    """Stateless by construction. A POST would make one browser tab able to
    change what another one sees."""
    assert client.post("/api/demo-account").status_code in (404, 405)


@pytest.mark.parametrize("qs,field", [
    ("commit_hour=99", "commit_hour"),
    ("commit_hour=-1", "commit_hour"),
    ("confidence=1.5", "confidence"),
    ("aggressive_frac=0", "aggressive_frac"),
    ("balance=-5", "balance"),
])
def test_out_of_range_knobs_are_rejected(client, qs, field):
    r = client.get(f"/api/demo-account?bench=1&{qs}")
    assert r.status_code == 422, (qs, r.status_code)


def test_index_has_the_bench_panel(client):
    text = client.get("/").text
    assert 'id="bench"' in text
    assert 'id="bench-commit-hour"' in text
    assert 'id="bench-apply"' in text
