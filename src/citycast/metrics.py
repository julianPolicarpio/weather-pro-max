"""Scores (spec §3.8): discrete CRPS (ranked probability score), Brier, cluster bootstrap."""
import math

import numpy as np

SIGMA_F = 3.0   # the sandbox mos baseline's fixed spread


def rps(pmf: dict[int, float], y: int) -> float:
    lo = min(min(pmf), y) - 1
    hi = max(max(pmf), y) + 1
    cum, total = 0.0, 0.0
    for t in range(lo, hi + 1):
        cum += pmf.get(t, 0.0)
        total += (cum - (1.0 if t >= y else 0.0)) ** 2
    return total


def brier(p: float, won: int) -> float:
    return (p - won) ** 2


def cluster_bootstrap(values_by_cluster: dict, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    sums = np.array([sum(v) for v in values_by_cluster.values()], dtype=float)
    counts = np.array([len(v) for v in values_by_cluster.values()], dtype=float)
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pmf(mu: float, sigma: float = SIGMA_F) -> dict[int, float]:
    c = round(mu)
    raw = {t: _phi((t + 0.5 - mu) / sigma) - _phi((t - 0.5 - mu) / sigma) for t in range(c - 20, c + 21)}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}
