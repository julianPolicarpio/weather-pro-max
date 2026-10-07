import numpy as np
from highcast.models.calibrate import PITCalibrator


def test_identity_when_calibrated():
    cal = PITCalibrator().fit(list(np.linspace(0.01, 0.99, 200)))
    assert abs(cal.level(0.5) - 0.5) < 0.03


def test_shifts_levels_when_biased_low():
    # outcomes always land in the top of the predicted distribution
    cal = PITCalibrator().fit(list(np.clip(np.linspace(0.5, 1.0, 200), 0, 1)))
    # to truly cover 50%, we must read a higher nominal level
    assert cal.level(0.5) > 0.6
