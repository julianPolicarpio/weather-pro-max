"""Trailing guidance-bias features, and the leakage rules that make them legal.

The raw `fc_*` values carry large station-vs-gridpoint offsets (KLAX is coastal;
a grid point a few km inland runs hot). Measured vs settled CLI: ecmwf HIGH
+6.09F, hrrr LOW -3.30F. A tree can absorb a CONSTANT offset into its split
thresholds, but these DRIFT (error autocorrelation r up to 0.86), and a tree
cannot SUBTRACT one feature from another. So guidance is corrected before the
model sees it, as `fcc_*`.

THE RISK IS LEAKAGE. The correction uses settled CLI values, which is exactly
the thing a forecast must not see for its own day. Two rules make it legal:
  1. Only days STRICTLY BEFORE the target contribute.
  2. Only days whose settled value was PUBLISHED by the feature cutoff
     contribute -- and that differs per metric, because the LOW's cutoff is the
     prior evening (03:00Z) while the climate day does not settle until 08:00Z.
     The LOW therefore reaches back two days where the HIGH reaches back one.
Rule 2 is the subtle one and is the reason these tests exist.
"""
import dataclasses
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from highcast import db, features
from highcast.config import load_config
from highcast.features import (BIAS_WINDOW, _actual_available_utc,
                               _forecast_error_series, build_features,
                               model_features)
from highcast.metric import get_metric

CFG = load_config()
HIGH = get_metric(CFG, "high")
LOW = get_metric(CFG, "low")


def _cfg(tmp_path):
    return dataclasses.replace(CFG, db_path=tmp_path / "t.db")


def _cli(conn, cfg, d: str, high: float, low: float):
    db.upsert(conn, "cli_reports", [{
        "station": cfg.station.icao, "date": d, "high_f": high, "low_f": low,
        "raw": "{}", "fetched_at": "t"}])


def _fc(conn, cfg, target: str, run: str, model: str, val: float, metric: str):
    db.upsert(conn, "forecasts", [{
        "station": cfg.station.icao, "target_date": target, "model": model,
        "run_date": run, "max_temp_f": val, "fetched_at": f"{run}T00:00:00+00:00",
        "metric": metric}])


# ---------------------------------------------------------------- availability

def test_actual_is_not_available_until_after_the_climate_day_ends():
    """The climate day is 08:00Z->08:00Z and the final CLI product lands just
    after (observed KLAX stamps: 08:33Z, 08:48Z). So a day's value is usable only
    from 08:00Z+1h the FOLLOWING morning, never during the day itself."""
    avail = _actual_available_utc(CFG, date(2026, 7, 26))
    assert avail == datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
    # Strictly after the climate day it describes.
    from highcast.timeutil import climate_day_bounds
    _, end = climate_day_bounds(date(2026, 7, 26), CFG.station.std_utc_offset)
    assert avail > end


def test_availability_is_dst_invariant():
    """`climate_day_bounds` uses std_utc_offset, so the availability instant must
    be the same wall-clock UTC in January and July. `_cutoff_utc` does NOT have
    this property (open item 3) — deriving availability per-date rather than as a
    fixed day-lag is what keeps the two from drifting apart across DST."""
    jan = _actual_available_utc(CFG, date(2026, 1, 15))
    jul = _actual_available_utc(CFG, date(2026, 7, 15))
    assert (jan.hour, jan.minute) == (jul.hour, jul.minute)


# ---------------------------------------------------------------- the lag split

@pytest.mark.parametrize("metric,expected_newest_lag", [(HIGH, 1), (LOW, 2)])
def test_metric_reachback_differs_because_of_the_cutoff(tmp_path, metric,
                                                        expected_newest_lag):
    """THE load-bearing asymmetry. HIGH cutoff = 09:00 local on the target day
    (16:00Z) so yesterday's settled value IS available. LOW cutoff = 20:00 the
    prior evening (03:00Z) which precedes the 08:00Z settlement, so the newest
    usable value is from two days back. Getting this wrong leaks the LOW's own
    overnight minimum into its own forecast."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    target = date(2026, 7, 20)
    cutoff = features._cutoff_utc(
        cfg, target + timedelta(days=metric.cutoff_day_offset),
        metric.cutoff_local_hour)
    # Smallest lag = most RECENT day, so the newest usable day is the min lag
    # that clears the cutoff (a larger lag is an older day and always clears).
    newest_usable = min(
        lag for lag in range(1, 6)
        if _actual_available_utc(cfg, target - timedelta(days=lag)) <= cutoff)
    assert newest_usable == expected_newest_lag
    # And the next-newer day must genuinely NOT clear it, or the split is vacuous.
    nearer = target - timedelta(days=expected_newest_lag - 1)
    assert _actual_available_utc(cfg, nearer) > cutoff


def test_error_series_never_includes_a_day_available_after_its_own_cutoff(tmp_path):
    """End-to-end leakage assertion: for every row of a real LOW frame, every
    error that fed its bias must have been publishable by that row's cutoff."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    for i in range(40):
        d = (date(2026, 6, 1) + timedelta(days=i)).isoformat()
        _cli(conn, cfg, d, 80.0, 60.0)
        _fc(conn, cfg, d, (date(2026, 6, 1) + timedelta(days=i - 1)).isoformat(),
            "gfs", 70.0, "low")
    err = _forecast_error_series(conn, cfg, LOW)
    instants, _ = err["gfs"]
    for ds in [(date(2026, 6, 1) + timedelta(days=i)).isoformat()
               for i in range(10, 40)]:
        d = date.fromisoformat(ds)
        cutoff = features._cutoff_utc(cfg, d + timedelta(days=LOW.cutoff_day_offset),
                                      LOW.cutoff_local_hour)
        used = [t for t in instants if t <= cutoff]
        assert all(t < _actual_available_utc(cfg, d) for t in used)
        # And the day itself is never in its own window.
        assert _actual_available_utc(cfg, d) not in used


# ---------------------------------------------------------------- the arithmetic

def test_correction_subtracts_the_trailing_bias(tmp_path):
    """A model that is consistently +5F hot must have that 5F removed."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    start = date(2026, 5, 1)
    for i in range(40):
        d = start + timedelta(days=i)
        _cli(conn, cfg, d.isoformat(), 80.0, 60.0)
        _fc(conn, cfg, d.isoformat(), (d - timedelta(days=1)).isoformat(),
            "gfs", 65.0, "low")            # always +5 vs the low of 60
    target = (start + timedelta(days=39)).isoformat()
    fr = build_features(conn, cfg, LOW, [target])
    assert fr["fc_gfs"].iloc[0] == pytest.approx(65.0)
    assert fr["bias_mean"].iloc[0] == pytest.approx(5.0)
    assert fr["fcc_gfs"].iloc[0] == pytest.approx(60.0)   # 65 - 5
    assert fr["bias_n"].iloc[0] == pytest.approx(BIAS_WINDOW)


def test_window_is_capped_at_bias_window(tmp_path):
    """Only the most recent BIAS_WINDOW available days count — a drifting offset
    is the whole point, so an unbounded mean would wash it out."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    start = date(2026, 4, 1)
    for i in range(60):
        d = start + timedelta(days=i)
        _cli(conn, cfg, d.isoformat(), 80.0, 60.0)
        # +20 for the distant past, +2 for the recent window.
        val = 80.0 if i < 40 else 62.0
        _fc(conn, cfg, d.isoformat(), (d - timedelta(days=1)).isoformat(),
            "gfs", val, "low")
    target = (start + timedelta(days=59)).isoformat()
    fr = build_features(conn, cfg, LOW, [target])
    # The last BIAS_WINDOW available days are all in the +2 era.
    assert fr["bias_mean"].iloc[0] == pytest.approx(2.0)


def test_no_history_yields_nan_not_a_zero_correction(tmp_path):
    """With no usable past, the correction must be absent (NaN), not silently 0 —
    a 0 would assert 'guidance is unbiased', which is the opposite of known."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    d = date(2026, 3, 1)
    _cli(conn, cfg, d.isoformat(), 80.0, 60.0)
    _fc(conn, cfg, d.isoformat(), (d - timedelta(days=1)).isoformat(),
        "gfs", 65.0, "low")
    fr = build_features(conn, cfg, LOW, [d.isoformat()])
    assert np.isnan(fr["fcc_gfs"].iloc[0])
    assert np.isnan(fr["bias_mean"].iloc[0])


# ---------------------------------------------------------------- metric scoping

def test_bias_features_are_low_only():
    """Measured: LOW dMAE -0.091 [-0.130,-0.055] SIG; HIGH +0.005 ns and
    indistinguishable from the +RANDOM noise arm. A wash is not an improvement,
    so the HIGH does not carry 13 inert columns."""
    low_cols, high_cols = model_features("low"), model_features("high")
    assert "fcc_mean" in low_cols and "bias_mean" in low_cols
    assert "fcc_mean" not in high_cols and "bias_mean" not in high_cols
    # The HIGH is the LOW minus every LOW-only group: the bias columns, the
    # NBM-bulletin temperature channel, and the marine-layer aux features. All
    # three were measured independently and all three landed LOW-only — see
    # `model_features` for each result and its +RANDOM control.
    # ...minus the dead columns, which are dropped from BOTH and so are not part
    # of the LOW-only set even though `fcc_nws`/`fcc_nbm` are nominally bias cols.
    low_only = ((set(features.BIAS_FEATURES) | set(features.NBS_FEATURES)
                 | set(features.NBS_AUX_FEATURES))
                - set(features.DEAD_FEATURES))
    assert set(high_cols) == set(low_cols) - low_only
    assert low_only <= set(low_cols)
    assert not low_only & set(high_cols)
    # Dead columns are dropped from BOTH: fc_nws is ~0.4% non-null (open item 5)
    # and fc_nbm is superseded by the bulletin channel.
    for cols in (low_cols, high_cols):
        assert not set(features.DEAD_FEATURES) & set(cols)
        # Raw guidance survives — fc_anchor and the hybrid model read it.
        assert "fc_mean" in cols and "fc_gfs" in cols
    # ...but the dead columns must still EXIST in the frame, because
    # BaselineBlend and the fc_anchor coalesce chain scan `fc_*` on the frame.
    assert set(features.DEAD_FEATURES) <= set(features.FEATURES)


def test_frame_carries_the_columns_for_both_metrics(tmp_path):
    """`build_features` stays metric-agnostic (its docstring promises identical
    columns); the SELECTION happens in `model_features`, not the frame."""
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    for i in range(20):
        d = (date(2026, 2, 1) + timedelta(days=i)).isoformat()
        _cli(conn, cfg, d, 80.0, 60.0)
    for metric in (LOW, HIGH):
        fr = build_features(conn, cfg, metric,
                            [(date(2026, 2, 1) + timedelta(days=i)).isoformat()
                             for i in range(20)])
        for c in features.BIAS_FEATURES:
            assert c in fr.columns, (metric.key, c)


def test_one_rule_governs_both_consumers():
    """The day-ahead's feature columns and the nowcast's guidance channel were
    measured separately and agreed. They must not drift apart: if a future change
    turns the correction on for a metric's model_features but leaves its
    BaselineBlend on raw guidance (or vice versa), the two halves of the same
    metric disagree about what guidance means and nothing warns."""
    from highcast.features import uses_bias_correction
    for key in ("low", "high"):
        on = uses_bias_correction(key)
        has_cols = "fcc_mean" in model_features(key)
        assert on is has_cols, key


def test_baseline_prefers_corrected_columns_only_when_asked(tmp_path):
    """`"fcc_gfs".startswith("fc_")` is False, so the corrected columns are
    invisible to the old prefix scan — they must be opted in explicitly. And each
    guidance model must enter the blend ONCE: blending fc_gfs and fcc_gfs together
    would double-weight that model."""
    from highcast.models.baseline import BaselineBlend

    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    start = date(2026, 1, 1)
    for i in range(80):
        d = start + timedelta(days=i)
        _cli(conn, cfg, d.isoformat(), 80.0, 60.0)
        _fc(conn, cfg, d.isoformat(), (d - timedelta(days=1)).isoformat(),
            "gfs", 65.0, "low")
    fr = build_features(conn, cfg, LOW,
                        [(start + timedelta(days=i)).isoformat() for i in range(80)])

    raw = BaselineBlend(prefer_corrected=False).fit(fr)
    corr = BaselineBlend(prefer_corrected=True).fit(fr)
    assert "fc_gfs" in raw.weights and "fcc_gfs" not in raw.weights
    assert "fcc_gfs" in corr.weights and "fc_gfs" not in corr.weights

    # The corrected blend moves ~5F toward the truth; the raw one sits on the
    # offset.
    #
    # THE QUIRK THIS USED TO PIN IS NOW FIXED (2026-08-02). `fc_anchor` (a
    # coalesce of fc_mos_mex -> fc_gfs -> fc_mean, documented in features.py as
    # "Not a model input") starts with "fc_", so the prefix scan picked it up and
    # blended whichever model the chain selected a SECOND time, uncorrected --
    # which is why the corrected arm landed at ~60.05 rather than 60.0.
    #
    # The earlier A/B behind "documented, not fixed" measured the DOWNSTREAM
    # nowcast only, found ns at every hour, and stopped there. Re-measured
    # 2026-08-02 across all four live markets with the blend fitted per fold on
    # train only (`scripts/anchor_exclusion_eval.py`): still no significant
    # regression anywhere, but at the commit hour the removal is significantly
    # BETTER on KLAX HIGH (CRPS -0.007, log loss -0.010) and KMIA LOW (bucket hit
    # +0.97 pts), and every market's MAE point estimate moves the same way. ns
    # was the right reason not to ship a change that costs something; it is not a
    # reason to keep a column that is a function of two other blend inputs. The
    # invariant now has its own test -- see `test_no_derived_column_enters_the_blend`.
    assert corr.point(fr)[-1] == pytest.approx(60.0, abs=0.2)
    assert raw.point(fr)[-1] == pytest.approx(65.0, abs=0.2)
    assert corr.point(fr)[-1] < raw.point(fr)[-1] - 4.0
    assert "fc_anchor" not in corr.weights     # derived; no longer a blend input
