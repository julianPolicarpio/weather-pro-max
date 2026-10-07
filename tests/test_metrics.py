from highcast.metrics import brier, crps_pmf, pinball, reliability

def test_crps_zero_for_certain_correct_forecast():
    assert crps_pmf({84: 1.0}, 84) == 0.0

def test_crps_penalizes_distance():
    near = crps_pmf({83: 0.5, 84: 0.5}, 84)
    far = crps_pmf({80: 0.5, 84: 0.5}, 84)
    assert near < far

def test_brier():
    assert brier({84: 0.7, 85: 0.3}, 84, 84) == (0.7 - 1) ** 2
    assert brier({84: 0.7, 85: 0.3}, 85, 84) == 0.7 ** 2

def test_pinball():
    qs = {0.5: 83.0}
    assert pinball(qs, 85) == 0.5 * (85 - 83)

def test_crps_gap_case_exact_value():
    # gap degrees 81-83 each contribute 0.25, plus 0.25 at t=80
    assert crps_pmf({80: 0.5, 84: 0.5}, 84) == 1.0

def test_reliability_half_coverage():
    # 20 records with the 0.5 quantile at 83.0; outcome falls at/below it
    # exactly half the time -> empirical coverage 0.5
    records = []
    for i in range(10):
        records.append(({0.5: 83.0}, 83))   # y <= q  (hit)
        records.append(({0.5: 83.0}, 84))   # y > q   (miss)
    assert reliability(records, 0.5) == 0.5
