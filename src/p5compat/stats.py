"""Inference over independent conditioning examples (stdlib only)."""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Dict, Optional, Sequence


def mean_ci(values: Sequence[float], level: float = 0.95) -> Dict[str, Optional[float]]:
    """Normal-approximation CI for the mean over independent units."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "lo": None, "hi": None}
    mu = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": mu, "sd": None, "se": None, "lo": None, "hi": None}
    sdv = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    se = sdv / math.sqrt(n)
    zq = NormalDist().inv_cdf(0.5 + level / 2.0)
    return {"n": n, "mean": mu, "sd": sdv, "se": se, "lo": mu - zq * se, "hi": mu + zq * se}


def sign_flip_pvalue(d: Sequence[float], n_flips: int, rng: random.Random) -> float:
    """One-sided (H1: E[d] > 0) sign-flip randomisation p-value.

    Valid when each d_i is symmetric about 0 under H0 and d_i are independent
    across units. For d_i = ED(A_i, C_i) - ED(A_i, B_i) with C_i and B_i
    exchangeable given A_i under H0, symmetry holds exactly.
    """
    t_obs = sum(d)
    tol = 1e-12 * (1.0 + sum(abs(v) for v in d))
    count = 1
    n = len(d)
    for _ in range(n_flips):
        bits = rng.getrandbits(n)
        s = 0.0
        for i, v in enumerate(d):
            s += v if (bits >> i) & 1 else -v
        if s >= t_obs - tol:
            count += 1
    return count / (n_flips + 1)


def wilson_ci(k: int, n: int, level: float = 0.95) -> Dict[str, float]:
    if n == 0:
        return {"k": k, "n": n, "rate": None, "lo": None, "hi": None}
    zq = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = k / n
    den = 1.0 + zq ** 2 / n
    centre = (p + zq ** 2 / (2 * n)) / den
    half = zq * math.sqrt(p * (1 - p) / n + zq ** 2 / (4 * n * n)) / den
    return {"k": k, "n": n, "rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}
