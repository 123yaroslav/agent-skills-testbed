"""Statistics for comparing agent runs.

Agent runs are random, so every scenario is run k times. The scenario is the unit of
analysis: two versions are compared on the same scenarios (paired design).
"""

from __future__ import annotations

from math import comb, sqrt
from statistics import NormalDist


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass^k for one scenario.

    pass^k is the chance that k independent runs all succeed (Yao et al., 2024).
    The scenario was run n times and c runs succeeded.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    return comb(c, k) / comb(n, k)


def minimum_detectable_effect(
    n_scenarios: int,
    k_runs: int,
    within_var: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest change in success rate that a paired one-sided test detects.

    Both versions run k_runs times on the same n_scenarios. within_var is the mean
    Bernoulli variance p * (1 - p) over scenarios; 0.25 is the worst case. Effect
    heterogeneity across scenarios is ignored, so this is a lower bound.
    """
    if n_scenarios < 1 or k_runs < 1:
        raise ValueError("n_scenarios and k_runs must be positive")
    if not 0 < within_var <= 0.25:
        raise ValueError("within_var must be in (0, 0.25]")
    std_err = sqrt(2 * within_var / (k_runs * n_scenarios))
    z = NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) * std_err
