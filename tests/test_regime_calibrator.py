import numpy as np

from highcast.models.calibrate import RegimeCalibrator


def _synth():
    # below 80: well-calibrated (mean resid 0, MAD 1.0 -> realized sd ~1.48 ~ pred 1.5)
    below_lvl = np.linspace(72.0, 79.0, 400)
    below_res = np.tile([-1.0, 1.0], 200)
    # seam 80-84: warm overshoot (y-m0 ~ -1.5) AND under-dispersed (wide resid)
    seam_lvl = np.linspace(80.0, 84.0, 160)
    seam_res = -1.5 + np.tile([-3.0, -1.0, 1.0, 3.0], 40)
    level = np.concatenate([below_lvl, seam_lvl])
    resid = np.concatenate([below_res, seam_res])
    pspread = np.full(level.size, 1.5)          # model claims sd 1.5 everywhere
    return level, resid, pspread


def test_learns_seam_shift_and_widen():
    rc = RegimeCalibrator().fit(*_synth())
    assert rc.mean_shift(81.0) < -0.5           # pull center down at the seam
    assert abs(rc.mean_shift(75.0)) < 0.4       # ~no shift below
    assert rc.spread_mult(81.0) > 1.2           # widen at the seam
    assert abs(rc.spread_mult(75.0) - 1.0) < 0.35


def test_off_and_null_are_noops():
    rc = RegimeCalibrator().fit(*_synth())
    q = {0.1: 78.0, 0.5: 81.0, 0.9: 84.0}
    assert rc.apply(q, "off") == q
    null = RegimeCalibrator().fit([], [], [])   # below the 30-sample floor
    assert null.apply(q, "center_width") == q


def test_width_keeps_median_center_shifts_both_ordered():
    rc = RegimeCalibrator().fit(*_synth())
    q = {0.1: 78.0, 0.5: 81.0, 0.9: 84.0}
    w = rc.apply(q, "width")
    assert abs(w[0.5] - 81.0) < 1e-9            # width keeps the median
    assert w[0.1] < w[0.5] < w[0.9]            # still ordered
    assert (w[0.9] - w[0.1]) > (q[0.9] - q[0.1])  # wider
    cw = rc.apply(q, "center_width")
    assert cw[0.5] < 81.0                       # center shifted down at seam
    assert cw[0.1] < cw[0.5] < cw[0.9]
