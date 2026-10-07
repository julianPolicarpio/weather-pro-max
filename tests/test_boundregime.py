"""The regime layer. Every test here is about it REFUSING to act.

The layer's job is to reshape a pmf; its safety property is that it returns the
input UNCHANGED on every unknown. A reshape that fires on a cell it has no data
for is worse than no layer, because it launders a guess as a measurement.
"""
import datetime as dt

import pytest
import yaml

from highcast import boundregime as br
from highcast import db as hdb
from highcast.config import load_config

# forecasts.fetched_at is NOT NULL but carries no logic this layer reads (see
# guidance_median's docstring: fetched_at must never appear in an availability
# filter). Tests supply a fixed placeholder so the INSERTs satisfy the schema
# without pretending to exercise freshness.
_FETCHED_AT = "2026-07-31T00:00:00+00:00"


def _cfg(tmp_path, src="config/kmia.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def test_the_scope_is_exactly_kmia_low_h7():
    """Widening scope must be a deliberate edit, not a side effect. The spec
    gates one market; the HIGHs have the opposite shape (the extremum lands
    late, so the bound is uninformative for most of the day)."""
    assert br.SCOPE == ("KMIA", "low", 7)
    assert br.SEASON_MONTHS == (6, 7, 8, 9)


def test_min_cell_n_is_pinned_at_the_pre_registered_value():
    """Pre-registered in spec 3.3. Lowering it after seeing results is the
    classic way a gate gets talked into passing, so it must be a visible edit
    to a test rather than a quiet constant change."""
    assert br.MIN_CELL_N == 40


@pytest.mark.parametrize("bound,gap,expected", [
    (78.0, 0.2, (0, 0)),
    (80.0, 1.0, (1, 1)),
    (82.14, 2.1, (2, 2)),      # 2026-08-01, the worked example in the spec
    (84.0, 3.0, (3, 3)),
    (81.0, 1.5, (2, 2)),       # band edges are inclusive-low, exclusive-high
    (83.0, 2.5, (3, 3)),
])
def test_regime_key_bands(bound, gap, expected):
    assert br.regime_key(bound, gap) == expected


@pytest.mark.parametrize("bound,gap", [
    (None, 1.0), (82.0, None), (None, None),
    (float("nan"), 1.0), (82.0, float("nan")),
])
def test_regime_key_fails_closed_on_every_unknown(bound, gap):
    """For an unattended caller, "I could not tell" and "no" are the same
    answer -- the same rule gating.confidence_floor_ok follows."""
    assert br.regime_key(bound, gap) is None


def test_guidance_median_uses_only_runs_from_BEFORE_the_target_date(tmp_path):
    """forecasts.run_date is a DATE, so a same-day row may be an 18Z run that
    did not exist at the 11Z decision. Strict inequality is the only safe
    filter, and it costs nothing: correlation with the drop is +0.281 either
    way (spec 2b)."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    day = dt.date(2026, 8, 1)
    conn.executemany(
        """INSERT INTO forecasts (station, target_date, model, run_date,
                                  max_temp_f, metric, fetched_at)
           VALUES (?,?,?,?,?,?,?)""",
        [("KMIA", "2026-08-01", "gfs",   "2026-07-31", 77.0, "low", _FETCHED_AT),
         ("KMIA", "2026-08-01", "ecmwf", "2026-07-31", 79.0, "low", _FETCHED_AT),
         # SAME-DAY run: must be excluded, it may be an 18Z run.
         ("KMIA", "2026-08-01", "nam",   "2026-08-01", 60.0, "low", _FETCHED_AT)])
    conn.commit()
    assert br.guidance_median(conn, "KMIA", day) == 78.0


def test_guidance_median_is_none_when_nothing_is_available(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    assert br.guidance_median(conn, "KMIA", dt.date(2026, 8, 1)) is None


def test_guidance_median_ignores_the_other_metric(tmp_path):
    """`max_temp_f` holds the LOW when metric='low' -- a column name that has
    caused this exact confusion before."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    conn.executemany(
        """INSERT INTO forecasts (station, target_date, model, run_date,
                                  max_temp_f, metric, fetched_at)
           VALUES (?,?,?,?,?,?,?)""",
        [("KMIA", "2026-08-01", "gfs", "2026-07-31", 77.0, "low", _FETCHED_AT),
         ("KMIA", "2026-08-01", "gfs", "2026-07-31", 95.0, "high", _FETCHED_AT)])
    conn.commit()
    assert br.guidance_median(conn, "KMIA", dt.date(2026, 8, 1)) == 77.0


# ── bound_for: THE LOOK-AHEAD BOUNDARY ──────────────────────────────────────
#
# Added after review. The brief specified `bound_for` and called its `asof`
# constraint load-bearing, then supplied no test for it — so the one function
# that decides whether the layer is measuring skill or look-ahead had zero
# coverage. The bound is still falling between h7 and h8 on 40.2% of days, so an
# `asof` an hour late does not fail loudly; it just reports a better number than
# the live system can ever have.

def _obs(conn, icao, valid_utc, tmpf):
    # `fetched_at` is NOT NULL with no default. Supplied as a fixed literal and
    # never read by anything under test — `visible_running_extremum` keys off
    # `valid_utc`, and using `fetched_at` for availability is banned outright
    # (db.upsert is INSERT OR REPLACE, so it is overwritten on every re-fetch).
    conn.execute(
        """INSERT INTO obs (station, valid_utc, source, tmpf, tmpf_precise,
                            fetched_at)
           VALUES (?,?,'metar',?,?,'2026-08-01T00:00:00+00:00')""",
        (icao, valid_utc, tmpf, tmpf))
    conn.commit()


def test_bound_for_ignores_observations_after_the_decision_hour(tmp_path):
    """THE test this function exists for. KMIA commits at h7 local = 11:00Z
    under EDT. An observation at 12:00Z is one hour PAST the decision and must
    be invisible — reading it is exactly the look-ahead that inflated an earlier
    version of this analysis by ~7 points."""
    cfg = _cfg(tmp_path, "config/kmia.yaml")
    conn = hdb.connect(cfg.db_path)
    day = dt.date(2026, 8, 1)
    _obs(conn, "KMIA", "2026-08-01T09:00:00", 82.0)   # 05:00 EDT, inside
    _obs(conn, "KMIA", "2026-08-01T12:00:00", 75.0)   # 08:00 EDT, AFTER h7

    at7 = br.bound_for(conn, cfg, day, 7)
    at9 = br.bound_for(conn, cfg, day, 9)
    assert at7 == pytest.approx(82.0, abs=0.3), (
        f"bound_for saw past the decision hour: got {at7}, expected ~82")
    assert at9 == pytest.approx(75.0, abs=0.3), (
        "the later cutoff should see the colder 12:00Z reading — if it does "
        "not, the asof is being ignored entirely rather than applied correctly")


def test_bound_for_returns_the_upper_end_of_the_interval(tmp_path):
    """Upper end, not lower: it is the tightest value consistent with what has
    been observed, and the settled low cannot exceed it. Taking the lower end
    would understate every drop by the METAR tolerance."""
    from highcast import latentmax
    cfg = _cfg(tmp_path, "config/kmia.yaml")
    conn = hdb.connect(cfg.db_path)
    day = dt.date(2026, 8, 1)
    _obs(conn, "KMIA", "2026-08-01T09:00:00", 82.0)
    iv = latentmax.visible_running_extremum(
        conn, cfg, day, "min", asof_utc="2026-08-01T11:00:00+00:00")
    assert br.bound_for(conn, cfg, day, 7) == iv[1]
    assert iv[1] > iv[0]


def test_bound_for_is_none_when_nothing_has_been_observed(tmp_path):
    """Fails closed like everything else here: no obs means no bound, which
    means `regime_key` returns None and the layer is a no-op."""
    cfg = _cfg(tmp_path, "config/kmia.yaml")
    conn = hdb.connect(cfg.db_path)
    assert br.bound_for(conn, cfg, dt.date(2026, 8, 1), 7) is None


def test_bound_for_converts_the_LOCAL_hour_not_a_utc_one(tmp_path):
    """`hour` is station-local. KMIA h7 is 11:00Z under EDT, not 07:00Z. Reading
    it as UTC would cut off four hours early and silently discard the whole
    pre-dawn window where the minimum actually forms."""
    cfg = _cfg(tmp_path, "config/kmia.yaml")
    conn = hdb.connect(cfg.db_path)
    day = dt.date(2026, 8, 1)
    _obs(conn, "KMIA", "2026-08-01T09:00:00", 82.0)   # 05:00 EDT — after 07:00Z
    assert br.bound_for(conn, cfg, day, 7) is not None, (
        "the 09:00Z observation was dropped, so `hour` was treated as UTC")
