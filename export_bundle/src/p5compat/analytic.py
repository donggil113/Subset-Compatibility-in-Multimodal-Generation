"""Closed-form quantities for the Gaussian probe (no Monte Carlo).

* Energy distance between two 1-D Gaussians.
* Expectation of the v1 V-statistic contrast d = ED_V(A,C) - ED_V(A,B)
  for M samples per set:  E[d] = D(P,Q) + (E|C-C'| - E|B-B'|) / M.
  (Derivation: E[ED_V(P_n,Q_m)] = D + E|X-X'|/n + E|Y-Y'|/m because the n
  zero diagonal terms of the within-sample double sums are included.)
* Exact output law of the PF-ODE Euler/Heun sampler with (guided) exact
  Gaussian scores: every step is affine in (x_0, m_cond, m_uncond), so the
  output is x_N = g x_0 + h_c m_c + h_u m_u with x_0 ~ N(0, sigma_max^2).

These are used to (i) predict what the v1 estimator should return for each
condition and (ii) compute the population direct-vs-sequential gap that the
solver / guidance alone induce, free of Monte Carlo error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

from .diffusion import edm_sigmas

SQRT_PI = math.sqrt(math.pi)


def e_abs_normal(mu: float, s: float) -> float:
    """E|X| for X ~ N(mu, s^2)."""
    if s <= 0.0:
        return abs(mu)
    z = mu / s
    return s * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z) + mu * (1.0 - 2.0 * NormalDist().cdf(-z))


def energy_distance_gauss(m1: float, s1: float, m2: float, s2: float) -> float:
    """Population energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'|."""
    cross = e_abs_normal(m1 - m2, math.sqrt(s1 * s1 + s2 * s2))
    return 2.0 * cross - 2.0 * s1 / SQRT_PI - 2.0 * s2 / SQRT_PI


def expected_v1_contrast(d_pop: float, s_c: float, s_b: float, m: int) -> float:
    """E[ED_V(A,C) - ED_V(A,B)] for Gaussian C, B with sds s_c, s_b and M samples each."""
    return d_pop + (2.0 * s_c / SQRT_PI - 2.0 * s_b / SQRT_PI) / m


# ---------------------------------------------------------------------------
# PF-ODE affine propagation
# ---------------------------------------------------------------------------


@dataclass
class AffineMap:
    """x_N = g * x0 + h_c * m_c + h_u * m_u,  x0 ~ N(0, sigma_max^2)."""

    g: float
    h_c: float
    h_u: float
    sigma_max: float

    def mean(self, m_c: float, m_u: float) -> float:
        return self.h_c * m_c + self.h_u * m_u

    def sd(self) -> float:
        return abs(self.g) * self.sigma_max


def pfode_affine_map(var_c: float, var_u: float, guidance: float, n_steps: int, solver: str = "euler",
                     sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> AffineMap:
    """Exact coefficients of the sampler in diffusion.GuidedPFODESampler.

    velocity(x, s) = s * a(s) * x - s * (b_c(s) m_c + b_u(s) m_u) with
      a   = (1+w)/(var_c+s^2) - w/(var_u+s^2),
      b_c = (1+w)/(var_c+s^2),  b_u = -w/(var_u+s^2).
    """
    w = float(guidance)
    sig = edm_sigmas(n_steps, sigma_min, sigma_max, rho)

    def vel(coef: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
        a = (1.0 + w) / (var_c + s * s) - w / (var_u + s * s)
        b_c = (1.0 + w) / (var_c + s * s)
        b_u = -w / (var_u + s * s)
        g, hc, hu = coef
        return (s * a * g, s * a * hc - s * b_c, s * a * hu - s * b_u)

    coef = (1.0, 0.0, 0.0)
    for k in range(len(sig) - 1):
        s, sn = sig[k], sig[k + 1]
        h = sn - s
        v1 = vel(coef, s)
        xe = tuple(c + h * v for c, v in zip(coef, v1))
        if solver == "heun" and sn > 0.0:
            v2 = vel(xe, sn)
            coef = tuple(c + 0.5 * h * (a + b) for c, a, b in zip(coef, v1, v2))
        elif solver in ("euler", "heun"):
            coef = xe
        else:
            raise ValueError(f"unknown solver {solver}")
    return AffineMap(g=coef[0], h_c=coef[1], h_u=coef[2], sigma_max=sigma_max)


def pfode_branch_laws(tc, x: float, n_steps: int, solver: str, guidance: float,
                      sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> Dict[str, Tuple[float, float]]:
    """Exact (mean, sd) of the target z under the PF-ODE direct and sequential branches.

    tc: samplers.TrueConditionals. Sequential stage 2 conditions on (x, y_hat).
    """
    kw = dict(guidance=guidance, n_steps=n_steps, solver=solver, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    mu_z, var_z = tc.z.intercept[0], tc.z.cov[0][0]
    mu_y, var_y = tc.y.intercept[0], tc.y.cov[0][0]
    # direct
    md = pfode_affine_map(tc.z_x.cov[0][0], var_z, **kw)
    direct = (md.mean(tc.z_x.mean([x])[0], mu_z), md.sd())
    # sequential: y_hat = g1 x0 + h1c m_{y|x} + h1u mu_y ; z = g2 x0' + h2c (a + bx x + by y_hat) + h2u mu_z
    m1 = pfode_affine_map(tc.y_x.cov[0][0], var_y, **kw)
    m2 = pfode_affine_map(tc.z_xy.cov[0][0], var_z, **kw)
    a = tc.z_xy.intercept[0]
    bx, by = tc.z_xy.coef[0]
    y_mean = m1.mean(tc.y_x.mean([x])[0], mu_y)
    seq_mean = m2.h_c * (a + bx * x + by * y_mean) + m2.h_u * mu_z
    seq_var = (m2.h_c * by * m1.g * sigma_max) ** 2 + (m2.g * sigma_max) ** 2
    return {"direct": direct, "sequential": (seq_mean, math.sqrt(seq_var)),
            "exact": (tc.z_x.mean([x])[0], math.sqrt(tc.tau2))}


def gaussian_projection(mean2: Sequence[float], cov2: Sequence[Sequence[float]], u: Sequence[float]) -> Tuple[float, float]:
    m = u[0] * mean2[0] + u[1] * mean2[1]
    v = u[0] * u[0] * cov2[0][0] + 2 * u[0] * u[1] * cov2[0][1] + u[1] * u[1] * cov2[1][1]
    return m, math.sqrt(max(v, 0.0))


def exact_flow_branch_laws(tc, x: float, sigma_max: float = 80.0) -> Dict[str, Tuple[float, float]]:
    """Continuous-time limit (no discretisation error) of the unguided PF-ODE with the
    standard initialisation x_0 ~ N(0, sigma_max^2). The exact flow keeps
    (x - m)/sqrt(s^2 + sigma^2) invariant, so the output is m + r (x_0 - m) with
    r = s / sqrt(s^2 + sigma_max^2): the only remaining error is the prior mismatch
    at sigma_max."""
    def r(var):
        return math.sqrt(var / (var + sigma_max ** 2))

    m_z = tc.z_x.mean([x])[0]
    rz = r(tc.z_x.cov[0][0])
    direct = (m_z * (1 - rz), rz * sigma_max)
    ry, r2 = r(tc.y_x.cov[0][0]), r(tc.z_xy.cov[0][0])
    a = tc.z_xy.intercept[0]
    bx, by = tc.z_xy.coef[0]
    m_y = tc.y_x.mean([x])[0]
    seq_mean = (a + bx * x + by * m_y * (1 - ry)) * (1 - r2)
    seq_sd = math.sqrt(((1 - r2) * by * ry * sigma_max) ** 2 + (r2 * sigma_max) ** 2)
    return {"direct": direct, "sequential": (seq_mean, seq_sd), "exact": (m_z, math.sqrt(tc.tau2))}
