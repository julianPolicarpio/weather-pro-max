from citycast.metrics import brier, cluster_bootstrap, normal_pmf, rps


def test_rps_zero_for_perfect_and_grows_with_distance():
    assert rps({50: 1.0}, 50) == 0.0
    assert rps({50: 1.0}, 52) == 2.0
    assert rps({49: 0.5, 51: 0.5}, 50) == 0.5


def test_brier():
    assert abs(brier(0.7, 1) - 0.09) < 1e-12
    assert abs(brier(0.7, 0) - 0.49) < 1e-12


def test_cluster_bootstrap_mean_and_ci_order():
    vals = {f"d{i}": [-1.0, -0.5] for i in range(50)}
    m, lo, hi = cluster_bootstrap(vals)
    assert m == -0.75 and lo <= m <= hi < 0


def test_normal_pmf_centered_and_normalised():
    p = normal_pmf(60.0)
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == 60
