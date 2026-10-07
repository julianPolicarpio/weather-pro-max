"""The evening forecast is a DIFFERENT OBJECT from the day-ahead.

Design: an internal design note

WHAT WENT WRONG. Two forecasts were stored under kind='dayahead': the 09:00
morning refresh, and the ~17:00 evening run whose as-of features are all NaN
because its own feature cutoff is still hours in the future. Measured:

    KLAX HIGH   evening n=19  MAE 4.42F  bias -4.05F
                morning n=14  MAE 1.45F  bias +0.19F

`gating.preferred_prediction` prefers the day-ahead for the HIGH before the
commit hour, so it served the 4F-cold object as the dashboard headline every
morning until 09:00 -- and computed a 69% "edge" from it. `signals._regime_veto`
DETECTED this correctly the whole time and was simply never consulted by the
code that picks the prediction.

THE FIX IS REPRESENTATIONAL, NOT A GUARD. Separating the kinds means
`preferred_prediction` cannot find the evening row at all, so the bug is
unrepresentable rather than caught at ~18 call sites, one of which would
eventually be missed.
"""
import datetime as dt
import importlib.util
import json

import pytest
import yaml

from highcast import db as hdb
from highcast import gating
from highcast.config import load_config
from highcast.metric import get_metric

_spec = importlib.util.spec_from_file_location(
    "mig", "scripts/migrate_evening_kind.py")
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


def _cfg(tmp_path, src="config/klax.yaml"):
    raw = yaml.safe_load(open(src).read())
    raw["db_path"] = str(tmp_path / "t.db")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(str(p))


def _store(conn, icao, target, kind, metric, issued, point):
    conn.execute(
        """INSERT INTO predictions (station, target_date, issued_at_utc, kind,
             point_f, quantiles, buckets, metric)
           VALUES (?,?,?,?,?,?,?,?)""",
        (icao, target, issued, kind, point, json.dumps({"0.5": point}),
         json.dumps({str(int(point)): 1.0}), metric))
    conn.commit()


# ── THE CORE PROPERTY ───────────────────────────────────────────────────────

def test_preferred_prediction_never_returns_an_evening_row(tmp_path):
    """THE test. Store ONLY an evening row and ask for the pick before the
    commit hour — the exact window where the day-ahead used to be preferred.
    Returning it is the 9F-wrong headline that started this."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _store(conn, "KLAX", "2026-08-01", hdb.KIND_EVENING, "high",
           "2026-07-31T17:03:00+00:00", 72.9)
    pred, kind = gating.preferred_prediction(
        conn, "KLAX", "2026-08-01", "high", True, local_hour=7)
    assert (pred, kind) == (None, None)


def test_the_intraday_is_chosen_when_only_an_evening_row_exists(tmp_path):
    """The replacement behaviour. Before 09:00 there is no morning day-ahead,
    so the nowcast is the answer — and it is the sharper one anyway (KLAX HIGH
    intraday MAE 0.74 at the commit hour vs the day-ahead's 1.35)."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _store(conn, "KLAX", "2026-08-01", hdb.KIND_EVENING, "high",
           "2026-07-31T17:03:00+00:00", 72.9)
    _store(conn, "KLAX", "2026-08-01", "intraday", "high",
           "2026-08-01T13:00:00+00:00", 81.1)
    pred, kind = gating.preferred_prediction(
        conn, "KLAX", "2026-08-01", "high", True, local_hour=7)
    assert kind == "intraday"
    assert pred["point"] == 81.1


def test_a_real_dayahead_is_still_preferred_before_the_commit_hour(tmp_path):
    """The change must not break what the walk-forward actually measured: at
    09:00 the MORNING day-ahead beat the nowcast (MAE 1.509 vs 1.620), and that
    preference still governs 09:00-13:00."""
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _store(conn, "KLAX", "2026-08-01", "dayahead", "high",
           "2026-08-01T16:05:00+00:00", 79.0)
    _store(conn, "KLAX", "2026-08-01", "intraday", "high",
           "2026-08-01T16:10:00+00:00", 81.1)
    pred, kind = gating.preferred_prediction(
        conn, "KLAX", "2026-08-01", "high", True, local_hour=10)
    assert kind == "dayahead"


# ── THE MIGRATION PREDICATE ─────────────────────────────────────────────────

def test_the_predicate_is_metric_specific(tmp_path):
    """⚠️ THE BUG THIS TEST EXISTS FOR. The first predicate used `hour < 9`,
    which is the HIGH's cutoff (09:00 on the TARGET day). The LOW's cutoff is
    20:00 on the PRIOR day, so that rule marked nearly every LOW day-ahead as
    'evening' and the migration's row count came out 52 instead of 35.

    A LOW run issued at 21:00 the previous evening is AFTER its own cutoff and
    its features are complete — it is a morning-equivalent object, not a broken
    one. Only the safeguard that refuses to run on an unexpected count caught
    this.
    """
    cfg = _cfg(tmp_path)
    tz = cfg.station.tz
    hi, lo = get_metric(cfg, "high"), get_metric(cfg, "low")

    # HIGH, issued 17:03 local the prior day -> before its 09:00 cutoff.
    assert mig.is_evening("2026-07-31T17:03:00-07:00", "2026-08-01", tz,
                          hi.cutoff_local_hour, hi.cutoff_day_offset)
    # HIGH, issued 09:05 local on the target day -> after its cutoff.
    assert not mig.is_evening("2026-08-01T09:05:00-07:00", "2026-08-01", tz,
                              hi.cutoff_local_hour, hi.cutoff_day_offset)
    # LOW, issued 17:03 the prior day -> BEFORE its 20:00 prior-day cutoff.
    assert mig.is_evening("2026-07-31T17:03:00-07:00", "2026-08-01", tz,
                          lo.cutoff_local_hour, lo.cutoff_day_offset)
    # LOW, issued 21:00 the prior day -> AFTER its cutoff. Features complete.
    # The HIGH's rule would have wrongly called this evening.
    assert not mig.is_evening("2026-07-31T21:00:00-07:00", "2026-08-01", tz,
                              lo.cutoff_local_hour, lo.cutoff_day_offset)


def test_the_predicate_mirrors_the_real_feature_cutoff(tmp_path):
    """The migration must not carry its own idea of when features exist.
    `features._cutoff_utc(cfg, d + cutoff_day_offset, cutoff_local_hour)` is
    the definition; this asserts the migration reproduces it exactly."""
    from datetime import timedelta

    from highcast import features
    cfg = _cfg(tmp_path)
    for mk in ("high", "low"):
        spec = get_metric(cfg, mk)
        d = dt.date(2026, 8, 1)
        expected = features._cutoff_utc(
            cfg, d + timedelta(days=spec.cutoff_day_offset),
            spec.cutoff_local_hour)
        got = mig.feature_cutoff_utc(d.isoformat(), cfg.station.tz,
                                     spec.cutoff_local_hour,
                                     spec.cutoff_day_offset)
        assert got == expected, f"{mk}: {got} != {expected}"


def test_the_migration_is_idempotent_and_selects_only_evening_rows(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _store(conn, "KLAX", "2026-08-01", "dayahead", "high",
           "2026-07-31T17:03:00+00:00", 72.9)      # evening -> should match
    _store(conn, "KLAX", "2026-08-02", "dayahead", "high",
           "2026-08-02T16:05:00+00:00", 79.0)      # 09:05 PDT -> should not
    rows = mig.find_evening_rows(conn, cfg)
    assert [r[1] for r in rows] == ["2026-08-01"]


# ── THE GUARD MUST NOT ROT ──────────────────────────────────────────────────

def test_the_regime_veto_still_fires_on_an_evening_row(tmp_path):
    """`_regime_veto` is now redundant for the hot path, which is exactly how a
    guard quietly becomes a no-op. It must still refuse an evening-issued
    forecast, so that any future code path reaching one is still stopped."""
    from highcast import signals
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    spec = get_metric(cfg, "high")
    pred = {"issued_at": "2026-07-31T17:03:00+00:00", "point": 72.9,
            "buckets": {"73": 1.0}}
    reason = signals._regime_veto(conn, cfg, spec, "2026-08-01", "dayahead", pred)
    assert reason is not None
    assert "morn" in reason.lower() or "morning" in reason.lower()


# ── NOTHING ELSE MAY REINTRODUCE THE MIX ────────────────────────────────────

def test_no_stored_dayahead_row_predates_its_own_feature_cutoff():
    """THE INVARIANT THIS WHOLE CHANGE BUYS, checked against the real database.

    If this fails, some call site is writing kind='dayahead' for a run issued
    before its features existed — the exact defect, reintroduced.
    """
    from highcast import db as realdb
    cfg = load_config("config/klax.yaml")
    conn = realdb.connect(cfg.db_path)
    offenders = []
    for cfgname in ("config/klax.yaml", "config/kmia.yaml"):
        c = load_config(cfgname)
        for r in mig.find_evening_rows(conn, c):
            offenders.append(r)
    assert not offenders, (
        f"{len(offenders)} kind='dayahead' rows predate their feature cutoff: "
        f"{offenders[:3]}. Run scripts/migrate_evening_kind.py --apply, and "
        f"find the call site that wrote them.")
