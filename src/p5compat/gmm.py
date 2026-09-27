"""Finite Gaussian mixtures with exact marginals and conditionals (stdlib only).

A GMM is a normalized joint density, so every conditional computed from it is
a conditional of that one joint: for any split of the variables,
    q(z|x) = int q(z|x,y) q(y|x) dy
holds exactly (the mixture weights of q(z|x,y) times q(y|x) collapse to
pi_k(x) N_k(y|x); see tests/test_gmm.py for a quadrature check). This makes
a GMM a JOINT_COHERENT_CONTROL. A fitted GMM is coherent by construction even
when it fits the data badly, which is what separates coherence from quality.

Also provides: EM fitting (closed-form updates, no autograd), exact samplers
compatible with samplers.Branch, and closed-form energy distances between 1-D
Gaussian mixtures (used as quality-to-truth; the CRPS divergence is ED / 2).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from . import linalg
from .analytic import e_abs_normal
from .gaussian import AffineGaussianConditional, JointGaussian

LOG2PI = math.log(2.0 * math.pi)


class GMM:
    def __init__(self, names: Sequence[str], weights: Sequence[float], means: Sequence[Sequence[float]],
                 covs: Sequence[Sequence[Sequence[float]]]):
        self.names = list(names)
        tot = sum(weights)
        if tot <= 0 or any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative with positive sum")
        self.weights = [w / tot for w in weights]
        self.comps = [JointGaussian(self.names, m, c) for m, c in zip(means, covs)]  # checks PD
        self.K = len(self.weights)

    @classmethod
    def from_sd_corr(cls, names, weights, means, sds, corrs):
        covs = [[[s[i] * s[j] * r[i][j] for j in range(len(s))] for i in range(len(s))] for s, r in zip(sds, corrs)]
        return cls(names, weights, means, covs)

    # ---------------------------------------------------------------- densities
    @staticmethod
    def _logpdf_gauss(v, mu, cov):
        d = [a - b for a, b in zip(v, mu)]
        sol = linalg.spd_solve(cov, d)
        quad = sum(a * b for a, b in zip(d, sol))
        return -0.5 * (len(v) * LOG2PI + linalg.spd_logdet(cov) + quad)

    def logpdf(self, v: Sequence[float]) -> float:
        terms = [math.log(w) + self._logpdf_gauss(v, c.mu, c.cov) for w, c in zip(self.weights, self.comps) if w > 0]
        mx = max(terms)
        return mx + math.log(sum(math.exp(t - mx) for t in terms))

    def marginal(self, names: Sequence[str]) -> "GMM":
        idx = [self.names.index(n) for n in names]
        return GMM(names, self.weights, [linalg.subvector(c.mu, idx) for c in self.comps],
                   [linalg.submatrix(c.cov, idx, idx) for c in self.comps])

    def conditional_components(self, target: Sequence[str], given: Sequence[str], g: Sequence[float]):
        """Posterior component weights pi_k(g) and per-component affine Gaussian conditionals."""
        if not given:
            return list(self.weights), [c.marginal(list(target)) for c in self.comps]
        gi = [self.names.index(n) for n in given]
        logs = []
        conds = []
        for w, c in zip(self.weights, self.comps):
            mu_g = linalg.subvector(c.mu, gi)
            cov_g = linalg.submatrix(c.cov, gi, gi)
            logs.append(math.log(w) + self._logpdf_gauss(list(g), mu_g, cov_g) if w > 0 else -math.inf)
            conds.append(c.conditional(list(target), list(given)))
        mx = max(logs)
        ex = [math.exp(l - mx) for l in logs]
        s = sum(ex)
        return [e / s for e in ex], conds

    def conditional_1d(self, target: str, given: Sequence[str], g: Sequence[float]) -> List[Tuple[float, float, float]]:
        """(weight, mean, sd) triples of the 1-D mixture q(target | given = g)."""
        ws, conds = self.conditional_components([target], given, g)
        return [(w, c.mean(g)[0] if given else c.intercept[0], math.sqrt(c.cov[0][0])) for w, c in zip(ws, conds)]

    # ---------------------------------------------------------------- sampling
    def sample(self, rng: random.Random) -> Dict[str, float]:
        k = _choice(rng, self.weights)
        return self.comps[k].sample(rng)

    def sample_conditional(self, rng: random.Random, target: Sequence[str], given: Sequence[str], g) -> List[float]:
        ws, conds = self.conditional_components(target, given, g)
        k = _choice(rng, ws)
        return conds[k].sample(rng, g) if given else conds[k].sample(rng, [])


def _choice(rng: random.Random, ws: Sequence[float]) -> int:
    u = rng.random()
    acc = 0.0
    for k, w in enumerate(ws):
        acc += w
        if u < acc:
            return k
    return len(ws) - 1


class GMMConditionalSampler:
    """Exact sampler for q(target | given) of a GMM; interface of samplers.ExactSampler."""

    def __init__(self, gmm: GMM, target: Sequence[str], given: Sequence[str]):
        self.gmm = gmm
        self.target = list(target)
        self.given = list(given)

    def sample_batch(self, rng, gs):
        return [self.gmm.sample_conditional(rng, self.target, self.given, g) for g in gs]


# ---------------------------------------------------------------------------
# EM fitting (closed-form updates)
# ---------------------------------------------------------------------------


def fit_gmm_em(data: Sequence[Sequence[float]], names: Sequence[str], k: int, rng: random.Random,
               max_iter: int = 500, tol: float = 1e-8, reg: float = 1e-6) -> Tuple[GMM, Dict]:
    n, d = len(data), len(data[0])
    mean_all = [sum(r[j] for r in data) / n for j in range(d)]
    cov_all = [[sum((r[a] - mean_all[a]) * (r[b] - mean_all[b]) for r in data) / n + (reg if a == b else 0.0)
                for b in range(d)] for a in range(d)]
    # k-means++-style seeding of the means with the model RNG
    centers = [list(data[rng.randrange(n)])]
    while len(centers) < k:
        d2 = [min(sum((r[j] - c[j]) ** 2 for j in range(d)) for c in centers) for r in data]
        tot = sum(d2)
        u, acc, pick = rng.random() * tot, 0.0, n - 1
        for i, v in enumerate(d2):
            acc += v
            if acc >= u:
                pick = i
                break
        centers.append(list(data[pick]))
    weights = [1.0 / k] * k
    means = centers
    covs = [[row[:] for row in cov_all] for _ in range(k)]
    trace = []
    prev = -math.inf
    converged = False
    for it in range(max_iter):
        model = GMM(names, weights, means, covs)
        # E-step
        resp, ll = [], 0.0
        chols = [(c.mu, c.cov) for c in model.comps]
        for r in data:
            logs = [math.log(w) + GMM._logpdf_gauss(r, mu, cv) for w, (mu, cv) in zip(model.weights, chols)]
            mx = max(logs)
            ex = [math.exp(l - mx) for l in logs]
            s = sum(ex)
            ll += mx + math.log(s)
            resp.append([e / s for e in ex])
        ll /= n
        trace.append(ll)
        if abs(ll - prev) <= tol * (1.0 + abs(ll)):
            converged = True
            break
        prev = ll
        # M-step
        nk = [sum(rr[j] for rr in resp) for j in range(k)]
        weights = [v / n for v in nk]
        means = [[sum(rr[j] * r[a] for rr, r in zip(resp, data)) / nk[j] for a in range(d)] for j in range(k)]
        covs = []
        for j in range(k):
            m = means[j]
            covs.append([[sum(rr[j] * (r[a] - m[a]) * (r[b] - m[b]) for rr, r in zip(resp, data)) / nk[j]
                          + (reg if a == b else 0.0) for b in range(d)] for a in range(d)])
    model = GMM(names, weights, means, covs)
    return model, {"iterations": len(trace), "converged": converged, "train_mean_loglik": trace[-1],
                   "loglik_trace_head": trace[:5], "loglik_trace_tail": trace[-5:],
                   "loglik_monotone": all(b >= a - 1e-9 for a, b in zip(trace, trace[1:]))}


# ---------------------------------------------------------------------------
# Closed-form discrepancies between 1-D mixtures
# ---------------------------------------------------------------------------


def mix_e_abs(a: Sequence[Tuple[float, float, float]], b: Sequence[Tuple[float, float, float]]) -> float:
    """E|X - Y| for independent X ~ mixture a, Y ~ mixture b; entries (w, mean, sd)."""
    return sum(wa * wb * e_abs_normal(ma - mb, math.sqrt(sa * sa + sb * sb)) for wa, ma, sa in a for wb, mb, sb in b)


def mix_energy_distance(a, b) -> float:
    return 2.0 * mix_e_abs(a, b) - mix_e_abs(a, a) - mix_e_abs(b, b)


def mix_moments(a) -> Tuple[float, float]:
    m = sum(w * mu for w, mu, _ in a)
    v = sum(w * (s * s + mu * mu) for w, mu, s in a) - m * m
    return m, v


def project_2d(gmm_cond_components, u, g):
    """1-D mixture of u^T (y, z) under a 2-D conditional given g."""
    ws, conds = gmm_cond_components
    out = []
    for w, c in zip(ws, conds):
        mu = c.mean(g) if c.given else c.intercept
        cv = c.cov
        m = u[0] * mu[0] + u[1] * mu[1]
        v = u[0] ** 2 * cv[0][0] + 2 * u[0] * u[1] * cv[0][1] + u[1] ** 2 * cv[1][1]
        out.append((w, m, math.sqrt(max(v, 0.0))))
    return out


def scale_mixture(a, s: float):
    return [(w, m / s, sd / s) for w, m, sd in a]
