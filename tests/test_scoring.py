import json

from highcast import db as hdb
from highcast import scoring
from highcast.config import load_config


def _cfg(tmp_path):
    cfg = load_config()
    object.__setattr__(cfg, "db_path", tmp_path / "t.db")
    return cfg


def _pred(conn, cfg, target_date, kind, issued_at_utc, point_f, buckets, metric="high"):
    hdb.upsert(conn, "predictions", [{
        "station": cfg.station.icao, "target_date": target_date,
        "issued_at_utc": issued_at_utc, "kind": kind,
        "point_f": point_f, "quantiles": json.dumps({"0.5": point_f}),
        "buckets": json.dumps(buckets),
        "metric": metric,
    }])


def _settle(conn, cfg, target_date, high_f):
    hdb.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": target_date, "high_f": high_f,
        "low_f": 60, "raw": "{}", "fetched_at": target_date + "T23:59:00+00:00",
    }])


# ---------------------------- tail_floor ------------------------------------

def test_tail_floor_crossing_two_thirds():
    # P(>=5)=1.0, P(>=6)=0.9, P(>=7)=0.65 -- 0.65 < 2/3 so the floor lands at 6
    pmf = {5: 0.1, 6: 0.25, 7: 0.65}
    assert scoring.tail_floor(pmf) == 6


def test_tail_floor_always_defined_at_min_key():
    pmf = {40: 1.0}
    assert scoring.tail_floor(pmf) == 40


# ---------------------------- tail_cap --------------------------------------

def test_tail_cap_crossing_two_thirds():
    # mirror of test_tail_floor_crossing_two_thirds: P(<=5)=0.65 (<2/3, fails),
    # P(<=6)=0.9 (>=2/3), P(<=7)=1.0 -- so the cap lands at the lowest T
    # clearing 2/3 from the bottom, which is 6.
    pmf = {5: 0.65, 6: 0.25, 7: 0.1}
    assert scoring.tail_cap(pmf) == 6


def test_tail_cap_always_defined_at_max_key():
    pmf = {40: 1.0}
    assert scoring.tail_cap(pmf) == 40


# ---------------------------- score_day ------------------------------------

def test_score_day_bucket_hit():
    pmf = {83: 0.2, 84: 0.5, 85: 0.3}
    r = scoring.score_day(pmf, 84.4, 84)
    assert r["point"] == 84
    assert r["mode"] == 84
    assert (r["pick_lo"], r["pick_hi"]) == (83, 84)
    assert r["pick_label"] == "83-84"
    assert r["point_hit"] is True
    assert r["within1"] is True
    assert r["bucket_hit"] is True
    assert abs(r["abs_error"] - 0.4) < 1e-9


def test_score_day_bucket_miss():
    pmf = {83: 0.2, 84: 0.5, 85: 0.3}
    r = scoring.score_day(pmf, 84.4, 90)
    assert r["bucket_hit"] is False
    assert r["point_hit"] is False
    assert abs(r["abs_error"] - 5.6) < 1e-9


def test_score_day_within1_boundary_true_at_exactly_one():
    pmf = {84: 1.0}
    r = scoring.score_day(pmf, 84.0, 85)
    assert r["within1"] is True
    assert r["bucket_hit"] is False  # pick is 83-84, actual is 85


def test_score_day_within1_false_past_one():
    pmf = {84: 1.0}
    r = scoring.score_day(pmf, 85.1, 84)
    assert r["within1"] is False


def test_score_day_point_hit_exact():
    pmf = {83: 0.2, 84: 0.5, 85: 0.3}
    r = scoring.score_day(pmf, 84.0, 84)
    assert r["point_hit"] is True


def test_score_day_bucket_miss_but_tail_hit():
    # mirrors the 2026-07-21 pattern: mass concentrated high (mode picks
    # 83-84), but the actual settles just below the top bucket while still
    # clearing the 2/3-confidence tail floor
    pmf = {78: 0.05, 79: 0.05, 80: 0.20, 81: 0.05, 82: 0.05, 83: 0.29, 84: 0.31}
    r = scoring.score_day(pmf, 84.0, 81)
    assert r["pick_label"] == "83-84"
    assert r["bucket_hit"] is False
    assert r["tail_floor"] == 81
    assert r["tail_label"] == "≥81"
    assert abs(r["tail_conf"] - 0.7) < 1e-9
    assert r["tail_hit"] is True


def test_score_day_both_bucket_and_tail_miss():
    pmf = {78: 0.05, 79: 0.05, 80: 0.20, 81: 0.05, 82: 0.05, 83: 0.29, 84: 0.31}
    r = scoring.score_day(pmf, 84.0, 79)
    assert r["bucket_hit"] is False
    assert r["tail_floor"] == 81
    assert r["tail_hit"] is False


def test_score_day_floor_default_matches_explicit_floor():
    # tail_dir defaults to "floor" -- the high path must be byte-identical
    # whether or not the caller passes tail_dir explicitly.
    pmf = {78: 0.05, 79: 0.05, 80: 0.20, 81: 0.05, 82: 0.05, 83: 0.29, 84: 0.31}
    default = scoring.score_day(pmf, 84.0, 81)
    explicit = scoring.score_day(pmf, 84.0, 81, tail_dir="floor")
    assert default == explicit
    assert default["tail_label"] == "≥81"
    assert default["tail_hit"] is True


# ------------------------- score_day (cap tail) -----------------------------
# pmf below mirrors the {78..84} pmf used in the floor tests above, but with
# mass concentrated at the LOW end instead of the high end (key order
# reversed: 78<->84, 79<->83, 80<->82), so tail_cap should land at 81 just as
# tail_floor lands at 81 for the mirror-image pmf.

def test_score_day_cap_bucket_miss_but_tail_hit():
    pmf = {78: 0.31, 79: 0.29, 80: 0.05, 81: 0.05, 82: 0.20, 83: 0.05, 84: 0.05}
    r = scoring.score_day(pmf, 78.0, 81, tail_dir="cap")
    assert r["pick_label"] == "77-78"
    assert r["bucket_hit"] is False
    assert r["tail_floor"] == 81
    assert r["tail_label"] == "≤81"
    assert abs(r["tail_conf"] - 0.70) < 1e-9
    assert r["tail_hit"] is True


def test_score_day_cap_both_bucket_and_tail_miss():
    pmf = {78: 0.31, 79: 0.29, 80: 0.05, 81: 0.05, 82: 0.20, 83: 0.05, 84: 0.05}
    r = scoring.score_day(pmf, 78.0, 84, tail_dir="cap")
    assert r["bucket_hit"] is False
    assert r["tail_floor"] == 81
    assert r["tail_hit"] is False


# ------------------------ scorecard + summarize -----------------------------

def test_scorecard_uses_latest_prediction_and_excludes_unsettled(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _settle(conn, cfg, "2026-07-16", 84)
    # two dayahead predictions for 2026-07-16 -- the later issued_at must win
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-15T02:00:00+00:00", 82.0,
          {"81": 0.2, "82": 0.5, "83": 0.3})
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-16T01:00:00+00:00", 84.3,
          {"83": 0.2, "84": 0.5, "85": 0.3})
    _pred(conn, cfg, "2026-07-16", "intraday", "2026-07-16T20:00:00+00:00", 84.0,
          {"83": 0.1, "84": 0.6, "85": 0.3})
    # a date with a prediction but NOT settled (no cli_reports row) -> excluded
    _pred(conn, cfg, "2026-07-17", "dayahead", "2026-07-16T02:00:00+00:00", 80.0,
          {"79": 0.3, "80": 0.4, "81": 0.3})

    records = scoring.scorecard(conn, cfg)
    assert len(records) == 2  # dayahead + intraday for 2026-07-16 only
    by_kind = {r["kind"]: r for r in records}
    assert set(by_kind) == {"dayahead", "intraday"}
    da = by_kind["dayahead"]
    assert da["target_date"] == "2026-07-16"
    assert da["actual"] == 84
    assert da["point"] == 84  # from the LATEST prediction (84.3), not 82.0
    assert da["mode"] == 84

    ia = by_kind["intraday"]
    assert ia["actual"] == 84
    assert ia["point"] == 84


def test_scorecard_kind_filter(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _settle(conn, cfg, "2026-07-16", 84)
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-16T01:00:00+00:00", 84.3,
          {"83": 0.2, "84": 0.5, "85": 0.3})
    _pred(conn, cfg, "2026-07-16", "intraday", "2026-07-16T20:00:00+00:00", 84.0,
          {"83": 0.1, "84": 0.6, "85": 0.3})
    records = scoring.scorecard(conn, cfg, kind="dayahead")
    assert len(records) == 1
    assert records[0]["kind"] == "dayahead"


def test_scorecard_sorted_by_date_then_kind(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _settle(conn, cfg, "2026-07-16", 84)
    _settle(conn, cfg, "2026-07-17", 78)
    _pred(conn, cfg, "2026-07-17", "intraday", "2026-07-17T20:00:00+00:00", 78.0,
          {"77": 0.2, "78": 0.6, "79": 0.2})
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-16T01:00:00+00:00", 84.0,
          {"83": 0.2, "84": 0.5, "85": 0.3})
    _pred(conn, cfg, "2026-07-16", "intraday", "2026-07-16T20:00:00+00:00", 84.0,
          {"83": 0.1, "84": 0.6, "85": 0.3})
    records = scoring.scorecard(conn, cfg)
    keys = [(r["target_date"], r["kind"]) for r in records]
    assert keys == sorted(keys)


def test_summarize_rates_are_correct_fractions(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    _settle(conn, cfg, "2026-07-16", 84)
    _settle(conn, cfg, "2026-07-17", 78)
    # one point-hit, one point-miss
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-16T01:00:00+00:00", 84.0,
          {"83": 0.2, "84": 0.5, "85": 0.3})
    _pred(conn, cfg, "2026-07-17", "dayahead", "2026-07-17T01:00:00+00:00", 84.0,
          {"83": 0.2, "84": 0.5, "85": 0.3})
    records = scoring.scorecard(conn, cfg, kind="dayahead")
    summary = scoring.summarize(records)
    assert summary["dayahead"]["n"] == 2
    assert summary["dayahead"]["point_hit_rate"] == 0.5
    assert summary["dayahead"]["mae"] == (0.0 + 6.0) / 2
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["point_hit_rate"] == 0.5
    # both records: {83:0.2,84:0.5,85:0.3} -> tail_floor=83 (P(>=83)=1.0>=2/3,
    # P(>=84)=0.8>=2/3, P(>=85)=0.3<2/3 -> best=84); actual 84 clears both,
    # actual 78 (the 2026-07-17 day) misses -> one tail hit of two
    assert summary["dayahead"]["tail_hit_rate"] == 0.5
    assert summary["overall"]["tail_hit_rate"] == 0.5


# --------------------------- scorecard (metric) ------------------------------

def test_scorecard_metric_low_scores_against_low_f_with_cap_tail(tmp_path):
    cfg = _cfg(tmp_path)
    conn = hdb.connect(cfg.db_path)
    # _settle always writes low_f=60 alongside the given high_f.
    _settle(conn, cfg, "2026-07-16", 84)
    _pred(conn, cfg, "2026-07-16", "dayahead", "2026-07-16T01:00:00+00:00", 60.0,
          {"58": 0.31, "59": 0.29, "60": 0.05, "61": 0.05, "62": 0.20,
           "63": 0.05, "64": 0.05},
          metric="low")

    low_records = scoring.scorecard(conn, cfg, metric="low")
    assert len(low_records) == 1
    r = low_records[0]
    assert r["metric"] == "low"
    assert r["actual"] == 60  # low_f, not high_f=84
    assert r["tail_label"] == "≤61"
    assert r["tail_hit"] is True  # actual 60 <= cap 61

    # the only prediction on this date is metric="low" -- the default
    # metric="high" scorecard must not pick it up via an unfiltered MAX(...)
    # subquery mixing predictions across metrics.
    high_records = scoring.scorecard(conn, cfg)
    assert high_records == []
