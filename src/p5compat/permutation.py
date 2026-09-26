"""Within-example (stratified) label-permutation tests for direct vs sequential.

Design. For each conditioning example i we have n_a direct samples and n_c
sequential samples, generated independently. Under H0_i (the two branches
have the same law given x_i) and when the samples inside each branch are
i.i.d. given x_i, the pooled n_a + n_c samples of example i are
exchangeable, so every allocation of branch labels inside example i is
equally likely. Labels are never exchanged across examples, and a vector
sample (y, z) is relabelled as a unit (coordinates are never shuffled
separately).

The statistic is fixed in advance: T = mean_i ED_V(a_i, c_i), the mean over
examples of the 1-D V-statistic energy distance (or the sliced version for
vectors). The p-value re-computes T for every sampled allocation.

This is the textbook permutation-test principle applied per stratum; it adds
no new theory. Validity requires the within-branch i.i.d. assumption. If a
generator re-uses one intermediate draw for several target samples, the
target samples are not i.i.d. and individual relabelling is NOT justified;
the design used here draws a fresh intermediate for every sample.

Monte Carlo p-value: p = (1 + #{b : T_b >= T_obs - tol}) / (B + 1), with ties
counted as at least as extreme (conservative); p is never 0.
Exact p-value (small designs): p = #{g in G : T(g) >= T_obs - tol} / |G|,
where G is the full product of per-example allocations and includes the
observed one.

For equal group sizes, ED_V and ED_U of one example are both increasing
affine functions of the cross sum S_ac given the pooled multiset, with slopes
that do not depend on the example, so the permutation p-value is identical
for the V- and U-statistic aggregations (see tests).
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from typing import List, Optional, Sequence, Tuple


class PooledExample:
    """Pre-sorted pooled projections for one conditioning example.

    values: list of pooled samples (scalars, or vectors when directions are
    given). The first n_a pooled indices are the observed direct samples.
    """

    def __init__(self, a: Sequence, c: Sequence, directions: Optional[Sequence[Sequence[float]]] = None):
        self.n_a, self.n_c = len(a), len(c)
        self.n = self.n_a + self.n_c
        pooled = list(a) + list(c)
        if directions is None:
            projs = [[float(v) for v in pooled]]
        else:
            projs = [[u[0] * p[0] + u[1] * p[1] for p in pooled] for u in directions]
        self.orders: List[List[Tuple[float, int]]] = []
        self.s_pool: List[float] = []
        for pr in projs:
            order = sorted((v, k) for k, v in enumerate(pr))
            self.orders.append(order)
            n = len(order)
            self.s_pool.append(sum(v * (2 * r - n + 1) for r, (v, _) in enumerate(order)))

    def stat(self, in_a: Sequence[bool], unbiased: bool = False) -> float:
        """ED (V or U) between the allocation's a-group and c-group, averaged over projections."""
        na, nc = self.n_a, self.n_c
        tot = 0.0
        for order, s_pool in zip(self.orders, self.s_pool):
            ka = kc = 0
            cum_a = cum_c = 0.0
            s_aa = s_cc = 0.0
            for v, k in order:
                if in_a[k]:
                    s_aa += ka * v - cum_a
                    ka += 1
                    cum_a += v
                else:
                    s_cc += kc * v - cum_c
                    kc += 1
                    cum_c += v
            s_ac = s_pool - s_aa - s_cc
            if unbiased:
                tot += 2.0 * s_ac / (na * nc) - 2.0 * s_aa / (na * (na - 1)) - 2.0 * s_cc / (nc * (nc - 1))
            else:
                tot += 2.0 * s_ac / (na * nc) - 2.0 * s_aa / (na * na) - 2.0 * s_cc / (nc * nc)
        return tot / len(self.orders)

    def observed_mask(self) -> List[bool]:
        return [k < self.n_a for k in range(self.n)]

    def random_mask(self, rng: random.Random) -> List[bool]:
        mask = [False] * self.n
        for k in rng.sample(range(self.n), self.n_a):
            mask[k] = True
        return mask

    def all_masks(self):
        for chosen in itertools.combinations(range(self.n), self.n_a):
            mask = [False] * self.n
            for k in chosen:
                mask[k] = True
            yield mask


def _tol(t: float) -> float:
    return 1e-12 * (1.0 + abs(t))


def aggregate(examples: Sequence[PooledExample], masks: Sequence[Sequence[bool]], unbiased: bool = False) -> float:
    return sum(ex.stat(m, unbiased) for ex, m in zip(examples, masks)) / len(examples)


def mc_permutation_test(examples: Sequence[PooledExample], n_perm: int, rng: random.Random,
                        unbiased: bool = False) -> dict:
    """Monte Carlo stratified permutation test (upper tail)."""
    t_obs = aggregate(examples, [ex.observed_mask() for ex in examples], unbiased)
    tol = _tol(t_obs)
    count = 1  # the observed allocation
    for _ in range(n_perm):
        t_b = aggregate(examples, [ex.random_mask(rng) for ex in examples], unbiased)
        if t_b >= t_obs - tol:
            count += 1
    return {"t_obs": t_obs, "p": count / (n_perm + 1), "n_perm": n_perm}


def exact_null_values(examples: Sequence[PooledExample], unbiased: bool = False, max_size: int = 200000):
    """All values of T over the full product group G (small designs only).

    Returns (values, observed_value). The per-example statistics are
    enumerated once and the aggregate is the mean over examples, so the
    product is formed on per-example value lists.
    """
    size = 1
    for ex in examples:
        size *= math.comb(ex.n, ex.n_a)
    if size > max_size:
        raise ValueError(f"|G| = {size} exceeds max_size = {max_size}")
    per_ex = [[ex.stat(m, unbiased) for m in ex.all_masks()] for ex in examples]
    n = len(examples)
    values = [sum(combo) / n for combo in itertools.product(*per_ex)]
    t_obs = aggregate(examples, [ex.observed_mask() for ex in examples], unbiased)
    return values, t_obs


def exact_pvalue(values: Sequence[float], t_obs: float) -> float:
    tol = _tol(t_obs)
    return sum(1 for v in values if v >= t_obs - tol) / len(values)


def group_superuniformity(values: Sequence[float], alphas: Sequence[float]) -> dict:
    """For every allocation g, p(g) = #{g' : T(g') >= T(g) - tol} / |G|.

    Under exchangeability every g is equally likely, so the test has exact
    conditional size <= alpha iff #{g : p(g) <= alpha} <= alpha |G|.
    Returns the counts; used to check the implementation (incl. ties).
    """
    n = len(values)
    asc = sorted(values)
    pvals = [(n - bisect.bisect_left(asc, v - _tol(v))) / n for v in values]
    out = {"size": n}
    for a in alphas:
        cnt = sum(1 for p in pvals if p <= a + 1e-15)
        out[str(a)] = {"count": cnt, "bound": a * n, "ok": cnt <= a * n + 1e-9}
    out["all_ok"] = all(out[str(a)]["ok"] for a in alphas)
    return out
