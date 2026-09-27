"""Sample-based discrepancy, proper-score and quality metrics (stdlib only).

Energy distance (Szekely & Rizzo) between empirical measures P_n, Q_m:

    ED_V = 2 E|X-Y| - E|X-X'| - E|Y-Y'|     (V-statistic, diagonals included)

In 1-D it equals 2 * integral (F_n - G_m)^2 dt and is computed exactly in
O((n+m) log(n+m)) from sums of pairwise absolute differences.  Under
P = Q, E[ED_V] = (1/n + 1/m) E|X-X'| > 0: this is the Monte Carlo noise
floor that the FIRST RUN measures with two independent sample sets.

CRPS is the 1-D energy score, a strictly proper scoring rule for
distributions with a finite first moment; the "fair" estimator below is
unbiased for the CRPS of the generator's distribution.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .gaussian import logpdf_1d


def _sum_abs_pairs_sorted(s: Sequence[float]) -> float:
    """sum_{i<j} |s_j - s_i| for an ascending sequence s."""
    n = len(s)
    return sum(v * (2 * k - n + 1) for k, v in enumerate(s))


def sum_abs_pairs(a: Sequence[float]) -> float:
    return _sum_abs_pairs_sorted(sorted(a))


def sum_abs_cross(a: Sequence[float], b: Sequence[float]) -> float:
    """sum_{i,j} |a_i - b_j|."""
    pooled = sorted(list(a) + list(b))
    return _sum_abs_pairs_sorted(pooled) - sum_abs_pairs(a) - sum_abs_pairs(b)


def energy_distance_1d(a: Sequence[float], b: Sequence[float], unbiased: bool = False) -> float:
    n, m = len(a), len(b)
    s_aa, s_bb = sum_abs_pairs(a), sum_abs_pairs(b)
    s_ab = _sum_abs_pairs_sorted(sorted(list(a) + list(b))) - s_aa - s_bb
    if unbiased:
        return 2.0 * s_ab / (n * m) - 2.0 * s_aa / (n * (n - 1)) - 2.0 * s_bb / (m * (m - 1))
    return 2.0 * s_ab / (n * m) - 2.0 * s_aa / (n * n) - 2.0 * s_bb / (m * m)


def energy_distance_1d_bruteforce(a: Sequence[float], b: Sequence[float]) -> float:
    n, m = len(a), len(b)
    xy = sum(abs(u - v) for u in a for v in b) / (n * m)
    xx = sum(abs(u - v) for u in a for v in a) / (n * n)
    yy = sum(abs(u - v) for u in b for v in b) / (m * m)
    return 2.0 * xy - xx - yy


def slice_directions(k: int) -> List[List[float]]:
    """k unit directions in 2-D, evenly spaced on [0, pi)."""
    return [[math.cos(math.pi * t / k), math.sin(math.pi * t / k)] for t in range(k)]


def sliced_energy_distance_2d(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
                              directions: Sequence[Sequence[float]]) -> float:
    """Average 1-D energy distance over fixed projection directions.

    Averaging over all directions is proportional to the multivariate energy
    distance (Cramer-Wold); a fixed finite set of directions is a
    pre-registered approximation and can in principle miss a difference that
    lies only between the chosen directions.
    """
    tot = 0.0
    for u0, u1 in directions:
        pa = [u0 * p[0] + u1 * p[1] for p in a]
        pb = [u0 * p[0] + u1 * p[1] for p in b]
        tot += energy_distance_1d(pa, pb)
    return tot / len(directions)


def crps_fair(samples: Sequence[float], obs: float) -> float:
    """Unbiased ensemble CRPS: E|X - obs| - 1/2 E|X - X'| (X, X' distinct draws)."""
    m = len(samples)
    if m < 2:
        raise ValueError("need at least two samples")
    t1 = sum(abs(v - obs) for v in samples) / m
    t2 = sum_abs_pairs(samples) / (m * (m - 1))
    return t1 - t2


def mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def var(v: Sequence[float]) -> float:
    mu = mean(v)
    return sum((u - mu) ** 2 for u in v) / (len(v) - 1)


def sd(v: Sequence[float]) -> float:
    return math.sqrt(max(var(v), 0.0))


def paired_index_mse(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean (a_j - b_j)^2 over matched indices. NOT a compatibility metric:
    for independent exact samplers it equals 2 Var in expectation, and it is
    minimised (to 0) by collapsed generators. Reported only to show this."""
    return sum((u - v) ** 2 for u, v in zip(a, b)) / len(a)


def mean_true_logpdf(samples: Sequence[float], mean_true: float, var_true: float) -> float:
    """Oracle 'fidelity': average true log density of generated samples.
    Maximised by collapsing onto the mode, so never reported alone."""
    return sum(logpdf_1d(v, mean_true, var_true) for v in samples) / len(samples)


def is_degenerate(v: Sequence[float], rel_tol: float = 1e-9) -> bool:
    """True if all samples are (numerically) identical, e.g. a collapsed generator.
    Round-off makes the sample variance of identical floats ~1e-32, not 0."""
    lo, hi = min(v), max(v)
    return (hi - lo) <= rel_tol * (1.0 + abs(lo) + abs(hi))


def log_sd_ratio(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """log(sd(b)/sd(a)); None when either sample set is degenerate."""
    if is_degenerate(a) or is_degenerate(b):
        return None
    return math.log(sd(b) / sd(a))
