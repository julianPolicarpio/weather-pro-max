"""The daily trade-quality verdict.

The load-bearing tests here are the freeze contract
(`test_nothing_in_the_decision_path_reads_preflight`) and the no-fallback rule
(`test_unmeasured_markets_get_no_band`) -- three of the four markets have no
measured effect and one has it REVERSED, so a silent fallback would not just be
unmeasured, it would be wrong in a known direction.
"""
import datetime as dt

import pytest
import yaml

from highcast import db as hdb
from highcast import preflight
from highcast.config import load_config



# ── SYNTHETIC BANDS ─────────────────────────────────────────────────────────
# `preflight.BANDS` is EMPTY as of 2026-08-01: the one shipped entry was retired
# when the commit hour moved and the effect vanished at the new hour (see the
# comment on BANDS itself). The MECHANISM — preferring the commit-hour gap,
# labelling off-hour reads provisional, refusing to leak across stations — still
# has to work the moment any band is added back, so these tests inject their own
# table instead of being deleted along with the data.
_SYNTHETIC = (
    (1.62, "CLEAR", 0.695, 0.661, 287),
    (2.53, "FAIR", 0.647, 0.634, 143),
    (None, "AVOID", 0.462, 0.572, 143),
)


@pytest.fixture
def banded(monkeypatch):
    monkeypatch.setitem(preflight.BANDS, ("KLAX", "low"), _SYNTHETIC)
    return _SYNTHETIC


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _pred(station, target, issued, gap, metric="low", kind="intraday"):
    return {"station": station, "target_date": target, "issued_at_utc": issued,
            "kind": kind, "point_f": 68.0, "quantiles": "{}", "buckets": "{}",
            "metric": metric, "bound_gap": gap}


# ── the bands themselves ────────────────────────────────────────────────────

def test_bands_are_ordered_and_cover_everything():
    """A gap must land in exactly one band, and the last must be open-ended --
    an uncovered gap would silently return 'unmeasured' on a measured market."""
    for key, bands in preflight.BANDS.items():
        uppers = [b[0] for b in bands]
        assert uppers[-1] is None, f"{key} has no open-ended top band"
        finite = uppers[:-1]
        assert finite == sorted(finite), f"{key} bands are out of order"
        for gap in (0.0, 0.5, 1.62, 1.63, 2.53, 2.54, 40.0):
            assert preflight.band_for(*key, gap) is not None


def test_the_worst_band_is_the_one_that_overstates():
    """The whole point: AVOID is not merely 'less accurate', it is the band
    where the model CLAIMS MORE THAN IT DELIVERS. If a band were less accurate
    but honest about it, sitting out would cost edge rather than save money."""
    for key in preflight.BANDS:
        graded = [preflight.band_for(*key, g) for g in (0.5, 2.0, 9.0)]
        names = [g[0] for g in graded]
        assert names == ["CLEAR", "FAIR", "AVOID"]
        for name, hit, stated, _ in graded:
            if name == "AVOID":
                assert stated > hit, "AVOID must be the overconfident band"
            else:
                assert hit >= stated, f"{name} should not overstate"


def test_gap_is_used_as_a_magnitude():
    """`cur_bound_gap` is signed (bound minus guidance) but the measurement
    stratified on its ABSOLUTE value -- disagreement is disagreement whichever
    way round it is. A sign leak here would grade half the bad days as CLEAR."""
    assert preflight.band_for("KLAX", "low", 9.0) == \
        preflight.band_for("KLAX", "low", -9.0)


def test_unmeasured_markets_get_no_band():
    """No cross-market fallback. KLAX HIGH's effect is REVERSED, so borrowing
    KLAX LOW's bands there would flag exactly the wrong days."""
    for station, metric in (("KLAX", "high"), ("KMIA", "low"), ("KMIA", "high")):
        assert preflight.band_for(station, metric, 3.0) is None


# ── grading a day ───────────────────────────────────────────────────────────

def test_gap_at_the_commit_hour_is_preferred_over_a_later_one(tmp_path, banded):
    """The bands were fitted at h4 only; the gap's distribution at h20 is a
    different population. A later prediction must not silently displace the
    commit-hour one."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    # KLAX h6 local == 13:00Z under PDT (knee moved 4 -> 6 on 2026-07-31,
    # see gating.MIN_DECISION_HOUR).
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T13:10:00+00:00", 0.4),
        _pred("KLAX", "2026-07-30", "2026-07-30T20:00:00+00:00", 8.0),
    ])
    g = preflight.grade(conn, cfg, "low", "2026-07-30")
    assert g["status"] == "graded"
    assert g["gap"] == 0.4 and g["band"] == "CLEAR"


def test_an_off_hour_gap_is_labelled_provisional(tmp_path, banded):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T20:00:00+00:00", 8.0)])
    g = preflight.grade(conn, cfg, "low", "2026-07-30")
    assert g["status"] == "provisional"
    assert g["band"] == "AVOID"
    assert "commit hour" in g["note"]


def test_a_day_with_no_recorded_gap_is_pending_not_clear(tmp_path, banded):
    """Absence of evidence must never render as the good band. Every day
    predicted before the column shipped has a NULL gap."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T11:10:00+00:00", None)])
    g = preflight.grade(conn, cfg, "low", "2026-07-30")
    assert g["status"] == "pending"
    assert g["band"] is None


def test_grade_does_not_leak_across_stations(tmp_path, banded):
    """Both stations share one SQLite file; anything keyed by metric+date alone
    is a latent cross-station bug."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KMIA", "2026-07-30", "2026-07-30T11:10:00+00:00", 0.4)])
    g = preflight.grade(conn, cfg, "low", "2026-07-30")
    assert g["status"] == "pending", "picked up KMIA's row for KLAX"


def test_unmeasured_market_says_so(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T18:10:00+00:00", 0.4,
              metric="high")])
    g = preflight.grade(conn, cfg, "high", "2026-07-30")
    assert g["status"] == "unmeasured"
    assert g["band"] is None


# ── the freeze contract ─────────────────────────────────────────────────────

def test_nothing_in_the_decision_path_reads_preflight():
    """DISPLAY ONLY. If the app began skipping bets on this verdict, its
    scorecard would stop measuring the model and start measuring the filter --
    and the app-vs-Julian comparison would be against a different opponent than
    the one it started against."""
    import pathlib
    decision = ("staking.py", "appbet.py", "gating.py", "notify.py",
                "features.py")
    src = pathlib.Path("src/highcast")
    offenders = [str(f) for f in src.rglob("*.py")
                 if f.name in decision and "preflight" in f.read_text()]
    assert not offenders, f"preflight leaked into the decision path: {offenders}"


def test_bound_gap_is_not_a_model_feature():
    """It was measured as one and REJECTED (GAP1 +1.0%, CI [-1.1%,+3.1%]).
    Storing it for display must not become a back door into the feature set."""
    from highcast.models.intraday import INTRADAY_FEATURES, RATE_FEATURES
    assert "cur_bound_gap" not in INTRADAY_FEATURES
    assert "cur_bound_gap" not in RATE_FEATURES
    from highcast.models.intraday import features_for
    for station in ("KLAX", "KMIA"):
        for metric in ("low", "high"):
            assert "cur_bound_gap" not in features_for(station, metric)


def test_the_gap_column_survives_a_reconnect(tmp_path):
    """The migration runs after `ensure_metric_column`, which may REBUILD the
    predictions table with DDL that does not carry this column."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T11:10:00+00:00", 1.25)])
    conn.close()
    conn2 = hdb.connect(cfg.db_path)
    cols = {r[1] for r in conn2.execute("PRAGMA table_info(predictions)")}
    assert "bound_gap" in cols
    row = conn2.execute("SELECT bound_gap FROM predictions").fetchone()
    assert row["bound_gap"] == 1.25


def test_no_market_ships_a_preflight_band_today():
    """THE RETIREMENT, pinned so it cannot silently come back.

    KLAX LOW's bands were measured at h4 and cleared their gate (Δshortfall
    +9.3%, CI [+1.6%, +15.9%]). When the commit hour moved to h6 on 2026-08-01
    and they were re-measured AT THE HOUR THEY WOULD BE USED, the effect was
    gone: Δ+0.7%, CI [−5.7%, +6.8%], with BOTH strata now under-stating.

    The bug that made this dangerous is structural: `BANDS` is keyed
    (station, metric) with NO HOUR, while `grade()` takes the commit hour from
    `gating.MIN_DECISION_HOUR`. So the h4 thresholds were silently applied to h6
    readings and the dashboard kept printing CLEAR/FAIR/AVOID. Nothing raised;
    the verdict just stopped meaning anything.

    If a band is ever added back, add the HOUR to the key at the same time.
    """
    assert preflight.BANDS == {}, (
        "a preflight band shipped again — re-measure it with "
        "scripts/preflight_eval.py AT THE CURRENT COMMIT HOUR, and key BANDS by "
        "hour so it cannot go stale invisibly the next time an hour moves")


def test_grade_reports_unmeasured_rather_than_guessing(tmp_path):
    """With no bands, every market must say so plainly instead of defaulting to
    a cheerful verdict. A preflight that returns CLEAR when it knows nothing is
    worse than no preflight."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    hdb.upsert(conn, "predictions", [
        _pred("KLAX", "2026-07-30", "2026-07-30T13:10:00+00:00", 0.4)])
    g = preflight.grade(conn, cfg, "low", "2026-07-30")
    assert g["band"] is None
    assert g["status"] != "graded"
