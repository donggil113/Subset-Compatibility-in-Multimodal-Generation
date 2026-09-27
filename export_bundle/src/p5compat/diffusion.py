"""Probability-flow ODE sampler with *exact* scores for Gaussian conditionals.

Purpose: isolate numerical (solver) and guidance (CFG) effects from model
error. With an exact score there is no learning error, so any direct-vs-
sequential discrepancy produced here is caused by discretisation, prior
mismatch at sigma_max, or guidance.

VE / EDM parameterisation: x_sigma = x_0 + sigma * eps. For a Gaussian target
N(m, s^2) the noised marginal is N(m, s^2 + sigma^2) and

    score(x, sigma) = -(x - m) / (s^2 + sigma^2),
    dx/dsigma       = -sigma * score(x, sigma).

Classifier-free guidance with weight w uses the guided score

    (1 + w) * score_cond - w * score_uncond,

where the unconditional model is the target's marginal p(target) (all
conditions dropped). w = 0 is unguided sampling.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

from .gaussian import AffineGaussianConditional


def edm_sigmas(n_steps: int, sigma_min: float, sigma_max: float, rho: float) -> List[float]:
    """Karras et al. (2022) step schedule, n_steps sigmas followed by 0."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_steps == 1:
        return [sigma_max, 0.0]
    a, b = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    sig = [(a + i / (n_steps - 1) * (b - a)) ** rho for i in range(n_steps)]
    return sig + [0.0]


class GuidedPFODESampler:
    """Deterministic PF-ODE sampler for a scalar-target affine Gaussian conditional."""

    def __init__(self, cond: AffineGaussianConditional, uncond: AffineGaussianConditional,
                 n_steps: int, solver: str = "euler", guidance: float = 0.0,
                 sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
        if len(cond.target) != 1:
            raise ValueError("scalar targets only")
        if solver not in ("euler", "heun"):
            raise ValueError(f"unknown solver {solver}")
        self.cond = cond
        self.target = cond.target
        self.given = cond.given
        self.var_c = cond.cov[0][0]
        self.m_u = uncond.intercept[0]
        self.var_u = uncond.cov[0][0]
        self.w = float(guidance)
        self.solver = solver
        self.sigmas = edm_sigmas(n_steps, sigma_min, sigma_max, rho)

    def velocity(self, x: float, sigma: float, m_c: float) -> float:
        g_c = -(x - m_c) / (self.var_c + sigma * sigma)
        if self.w == 0.0:
            score = g_c
        else:
            g_u = -(x - self.m_u) / (self.var_u + sigma * sigma)
            score = (1.0 + self.w) * g_c - self.w * g_u
        return -sigma * score

    def integrate(self, x0: Sequence[float], m_c: Sequence[float]) -> List[float]:
        xs = list(x0)
        sig = self.sigmas
        for k in range(len(sig) - 1):
            s, sn = sig[k], sig[k + 1]
            h = sn - s
            v = [self.velocity(x, s, m) for x, m in zip(xs, m_c)]
            xe = [x + h * vi for x, vi in zip(xs, v)]
            if self.solver == "heun" and sn > 0.0:
                v2 = [self.velocity(x, sn, m) for x, m in zip(xe, m_c)]
                xs = [x + 0.5 * h * (a + b) for x, a, b in zip(xs, v, v2)]
            else:
                xs = xe
        return xs

    def sample_batch(self, rng: random.Random, gs):
        m_c = [self.cond.mean(g)[0] for g in gs]
        # Standard initialisation N(0, sigma_max^2); the true noised marginal is
        # N(m, s^2 + sigma_max^2), so this is a (small) prior mismatch.
        x0 = [rng.gauss(0.0, self.sigmas[0]) for _ in gs]
        return [[v] for v in self.integrate(x0, m_c)]


def euler_affine_factor(sigmas: Sequence[float], var_c: float) -> float:
    """Closed form for unguided Euler on a Gaussian target:
    x_N - m = prod_k (1 + (sigma_{k+1} - sigma_k) sigma_k / (s^2 + sigma_k^2)) * (x_0 - m).
    Used by unit tests."""
    f = 1.0
    for k in range(len(sigmas) - 1):
        s, sn = sigmas[k], sigmas[k + 1]
        f *= 1.0 + (sn - s) * s / (var_c + s * s)
    return f


def exact_ode_factor(sigma_max: float, var_c: float) -> float:
    """Exact PF-ODE map: (x - m)/sqrt(s^2 + sigma^2) is invariant."""
    return math.sqrt(var_c / (var_c + sigma_max ** 2))
