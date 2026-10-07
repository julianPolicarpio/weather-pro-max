import numpy as np

from highcast.backtest import slice_metrics


def _recs():
    return [
        {"point": 78.0, "y": 79, "abs_err": 1.0, "crps": 0.5,
         "quantiles": {0.5: 78.0, 0.9: 82.0}},
        {"point": 81.0, "y": 80, "abs_err": 1.0, "crps": 0.7,
         "quantiles": {0.5: 81.0, 0.9: 83.0}},
        {"point": 82.5, "y": 84, "abs_err": 1.5, "crps": 0.9,
         "quantiles": {0.5: 82.5, 0.9: 85.0}},
    ]


def test_slice_metrics_filters_by_point():
    seam = slice_metrics(_recs(), 80.0, 84.0)
    assert seam["n"] == 2
    assert abs(seam["mae"] - 1.25) < 1e-9
    below = slice_metrics(_recs(), -999.0, 80.0)
    assert below["n"] == 1
    empty = slice_metrics(_recs(), 90.0, 95.0)
    assert empty["n"] == 0 and np.isnan(empty["mae"])
