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


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    tot = 0.0
    for j in range(0, k + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq)
    return min(tot, 1.0)


def _bisect(f, lo: float, hi: float, iters: int = 80) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, level: float = 0.95, sided: str = "two") -> Dict[str, float]:
    """Exact binomial CI. sided='two' (equal tails), 'upper' (one-sided upper bound
    at `level`), or 'lower' (one-sided lower bound at `level`)."""
    if n == 0:
        return {"k": k, "n": n, "rate": None, "lo": None, "hi": None}
    tail = (1.0 - level) / 2.0 if sided == "two" else (1.0 - level)
    # upper: smallest p with P(X <= k | p) <= tail
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) <= tail, k / n, 1.0)
    # lower: largest p with P(X >= k | p) <= tail
    lo = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p) > tail, 0.0, k / n)
    out = {"k": k, "n": n, "rate": k / n, "level": level, "sided": sided}
    if sided == "two":
        out.update({"lo": lo, "hi": hi})
    elif sided == "upper":
        out.update({"lo": 0.0, "hi": hi})
    elif sided == "lower":
        out.update({"lo": lo, "hi": 1.0})
    else:
        raise ValueError(sided)
    return out
