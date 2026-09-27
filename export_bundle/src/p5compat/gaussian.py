"""Exact joint / marginal / conditional algebra for a small Gaussian.

For a joint N(mu, Sigma) over named scalar variables, the conditional of
target block T given block G is

    T | G=g  ~  N( mu_T + A (g - mu_G),  Sigma_TT - A Sigma_GT ),
    A = Sigma_TG Sigma_GG^{-1}.

We store it in affine form  mean(g) = intercept + coef @ g  so that
perturbed conditionals (FIRST RUN checker validation) can be written as
edits to (intercept, coef, cov).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from . import linalg


@dataclass
class AffineGaussianConditional:
    """q(T | G=g) = N(intercept + coef g, cov), T and G given by names.

    ``temperature`` scales the noise standard deviation at sampling time;
    temperature 0 is a deterministic (collapsed) generator that returns the
    conditional mean.
    """

    target: List[str]
    given: List[str]
    intercept: List[float]
    coef: List[List[float]]  # len(target) x len(given)
    cov: List[List[float]]  # len(target) x len(target)
    temperature: float = 1.0
    label: str = ""
    _chol: List[List[float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._chol = linalg.cholesky(self.cov)

    def mean(self, g: Sequence[float]) -> List[float]:
        if len(g) != len(self.given):
            raise ValueError(f"expected {len(self.given)} conditioning values, got {len(g)}")
        return [
            self.intercept[t] + sum(self.coef[t][j] * g[j] for j in range(len(g)))
            for t in range(len(self.target))
        ]

    def sample(self, rng: random.Random, g: Sequence[float]) -> List[float]:
        m = self.mean(g)
        if self.temperature == 0.0:
            return m
        eps = [rng.gauss(0.0, 1.0) for _ in self.target]
        low = self._chol
        return [
            m[t] + self.temperature * sum(low[t][k] * eps[k] for k in range(t + 1))
            for t in range(len(self.target))
        ]

    def sd(self) -> List[float]:
        """Marginal sd of each target coordinate (before temperature)."""
        return [math.sqrt(self.cov[t][t]) for t in range(len(self.target))]


class JointGaussian:
    """Joint Gaussian over named scalar variables."""

    def __init__(self, names: Sequence[str], mean: Sequence[float], cov: Sequence[Sequence[float]]):
        self.names = list(names)
        self.mu = [float(v) for v in mean]
        self.cov = [[float(v) for v in row] for row in cov]
        if len(self.mu) != len(self.names) or len(self.cov) != len(self.names):
            raise ValueError("dimension mismatch")
        for i in range(len(self.names)):
            for j in range(len(self.names)):
                if abs(self.cov[i][j] - self.cov[j][i]) > 1e-12:
                    raise ValueError("covariance is not symmetric")
        self._chol = linalg.cholesky(self.cov)  # raises if not PD

    @classmethod
    def from_sd_corr(cls, names, mean, sd, corr) -> "JointGaussian":
        cov = [[sd[i] * sd[j] * corr[i][j] for j in range(len(sd))] for i in range(len(sd))]
        return cls(names, mean, cov)

    def idx(self, names: Sequence[str]) -> List[int]:
        return [self.names.index(n) for n in names]

    def marginal(self, names: Sequence[str]) -> AffineGaussianConditional:
        ii = self.idx(names)
        return AffineGaussianConditional(
            target=list(names),
            given=[],
            intercept=linalg.subvector(self.mu, ii),
            coef=[[] for _ in ii],
            cov=linalg.submatrix(self.cov, ii, ii),
            label=f"p({','.join(names)})",
        )

    def conditional(self, target: Sequence[str], given: Sequence[str]) -> AffineGaussianConditional:
        if not given:
            return self.marginal(target)
        ti, gi = self.idx(target), self.idx(given)
        s_tt = linalg.submatrix(self.cov, ti, ti)
        s_tg = linalg.submatrix(self.cov, ti, gi)
        s_gg = linalg.submatrix(self.cov, gi, gi)
        s_gg_inv = linalg.spd_inverse(s_gg)
        a = linalg.matmul(s_tg, s_gg_inv)
        mu_t, mu_g = linalg.subvector(self.mu, ti), linalg.subvector(self.mu, gi)
        a_mu_g = linalg.matvec(a, mu_g)
        intercept = [mu_t[t] - a_mu_g[t] for t in range(len(ti))]
        a_s_gt = linalg.matmul(a, linalg.transpose(s_tg))
        cov = [[s_tt[r][c] - a_s_gt[r][c] for c in range(len(ti))] for r in range(len(ti))]
        # symmetrise against round-off
        cov = [[0.5 * (cov[r][c] + cov[c][r]) for c in range(len(ti))] for r in range(len(ti))]
        return AffineGaussianConditional(
            target=list(target),
            given=list(given),
            intercept=intercept,
            coef=a,
            cov=cov,
            label=f"p({','.join(target)}|{','.join(given)})",
        )

    def sample(self, rng: random.Random) -> Dict[str, float]:
        eps = [rng.gauss(0.0, 1.0) for _ in self.names]
        low = self._chol
        vals = [self.mu[i] + sum(low[i][k] * eps[k] for k in range(i + 1)) for i in range(len(self.names))]
        return dict(zip(self.names, vals))

    def partial_corr(self, a: str, b: str, given: Sequence[str]) -> float:
        c = self.conditional([a, b], list(given))
        return c.cov[0][1] / math.sqrt(c.cov[0][0] * c.cov[1][1])


def logpdf_1d(v: float, mean: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + (v - mean) ** 2 / var)
