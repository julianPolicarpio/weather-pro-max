from highcast.buckets import bucket_probs

QS = {0.05: 78.0, 0.25: 81.0, 0.50: 83.0, 0.75: 85.0, 0.95: 88.0}

def test_pmf_sums_to_one_and_peaks_near_median():
    pmf = bucket_probs(QS)
    assert abs(sum(pmf.values()) - 1.0) < 1e-6
    mode = max(pmf, key=pmf.get)
    assert 82 <= mode <= 84

def test_hot_tail_is_never_zeroed():
    pmf = bucket_probs(QS)
    assert pmf.get(92, 0.0) > 0.0       # beyond the 95th percentile
    assert pmf.get(95, 0.0) > 0.0

def test_pmf_monotone_cdf():
    pmf = bucket_probs(QS)
    ts = sorted(pmf)
    cum = 0.0
    for t in ts:
        assert pmf[t] >= 0.0
        cum += pmf[t]
    assert abs(cum - 1.0) < 1e-6

def test_skewed_quantiles_far_tail_not_zeroed():
    # extreme outer levels used to be hard-zeroed by a probability-threshold
    # filter; every bucket in the support window must keep positive mass
    qs = {0.001: 80.0, 0.002: 80.4, 0.5: 83.0, 0.998: 85.6, 0.999: 86.0}
    pmf = bucket_probs(qs)
    assert abs(sum(pmf.values()) - 1.0) < 1e-6
    assert pmf.get(101, 0.0) > 0.0      # top of the +15 support window
    for t, p in pmf.items():
        assert p > 0.0, f"bucket {t} was zeroed"

def test_single_quantile_does_not_crash():
    pmf = bucket_probs({0.5: 83.0})
    assert abs(sum(pmf.values()) - 1.0) < 1e-6

def test_all_equal_values_does_not_crash():
    pmf = bucket_probs({0.25: 83.0, 0.5: 83.0, 0.75: 83.0})
    assert abs(sum(pmf.values()) - 1.0) < 1e-6
