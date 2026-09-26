"""Generation branches for the 3-variable (x, y, z) compatibility probe.

Roles: x = observed condition, y = intermediate modality, z = target.

Branches evaluated per conditioning example (x_i, y_obs_i, z_obs_i):

* ``direct``        z_j ~ q(z | x_i)
* ``sequential``    y_j ~ q(y | x_i), z_j ~ q(z | x_i, y_j)   (generated intermediate)
* ``chain``         y_j ~ q(y | x_i), z_j ~ q(z | y_j)        (x dropped at stage 2)
* ``observed_int``  z_j ~ q(z | x_i, y_obs_i)                 (observed intermediate)
* ``direct_joint``  (y_j, z_j) ~ q(y, z | x_i)

Only ``direct`` vs ``sequential`` is a compatibility test of the identity
q(z|x) = E_{q(y|x)} q(z|x,y). ``observed_int`` conditions on extra
information and is *expected* to differ; it is a category-error control.
``chain`` is compatible only if z is independent of x given y.

Stage-2 perturbations are parameterised relative to the true joint so that
their effect on the target marginal q(z|x) is known in closed form.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

from .gaussian import AffineGaussianConditional, JointGaussian


class Conditional(Protocol):
    target: List[str]
    given: List[str]

    def sample_batch(self, rng: random.Random, gs: Sequence[Sequence[float]]) -> List[List[float]]:
        ...


class ExactSampler:
    """Exact ancestral sampling from an affine Gaussian conditional."""

    def __init__(self, cond: AffineGaussianConditional):
        self.cond = cond
        self.target = cond.target
        self.given = cond.given

    def sample_batch(self, rng, gs):
        return [self.cond.sample(rng, g) for g in gs]


@dataclass
class Branch:
    kind: str  # direct | sequential | chain | observed_int | direct_joint
    final: Conditional  # q(z|...) or q(y,z|x)
    stage1: Optional[Conditional] = None  # q(y|x) for sequential / chain

    def sample(self, rng: random.Random, example: Dict[str, float], m: int) -> Dict[str, Optional[List[float]]]:
        x = example["x"]
        if self.kind == "direct":
            z = [v[0] for v in self.final.sample_batch(rng, [[x]] * m)]
            return {"y": None, "z": z}
        if self.kind == "direct_joint":
            out = self.final.sample_batch(rng, [[x]] * m)
            return {"y": [v[0] for v in out], "z": [v[1] for v in out]}
        if self.kind in ("sequential", "chain"):
            y = [v[0] for v in self.stage1.sample_batch(rng, [[x]] * m)]
            gs = [_given_vector(self.final.given, x, yj) for yj in y]
            z = [v[0] for v in self.final.sample_batch(rng, gs)]
            return {"y": y, "z": z}
        if self.kind == "observed_int":
            g = _given_vector(self.final.given, x, example["y_obs"])
            z = [v[0] for v in self.final.sample_batch(rng, [g] * m)]
            # The intermediate is observed data, not a generated sample.
            return {"y": None, "z": z}
        raise ValueError(f"unknown branch kind {self.kind}")


def _given_vector(given: Sequence[str], x: float, y: float) -> List[float]:
    lookup = {"x": x, "y": y}
    return [lookup[name] for name in given]


# ---------------------------------------------------------------------------
# True conditionals and controlled perturbations
# ---------------------------------------------------------------------------


@dataclass
class TrueConditionals:
    z_x: AffineGaussianConditional
    y_x: AffineGaussianConditional
    z_xy: AffineGaussianConditional
    z_y: AffineGaussianConditional
    yz_x: AffineGaussianConditional
    z: AffineGaussianConditional
    y: AffineGaussianConditional

    @property
    def tau2(self) -> float:  # Var(z | x)
        return self.z_x.cov[0][0]

    @property
    def v_y(self) -> float:  # Var(y | x)
        return self.y_x.cov[0][0]

    @property
    def b(self) -> float:  # coefficient of y in E[z | x, y]
        return self.z_xy.coef[0][self.z_xy.given.index("y")]


def true_conditionals(joint: JointGaussian) -> TrueConditionals:
    return TrueConditionals(
        z_x=joint.conditional(["z"], ["x"]),
        y_x=joint.conditional(["y"], ["x"]),
        z_xy=joint.conditional(["z"], ["x", "y"]),
        z_y=joint.conditional(["z"], ["y"]),
        yz_x=joint.conditional(["y", "z"], ["x"]),
        z=joint.marginal(["z"]),
        y=joint.marginal(["y"]),
    )


def _copy(c: AffineGaussianConditional, **kw) -> AffineGaussianConditional:
    fields = dict(
        target=list(c.target),
        given=list(c.given),
        intercept=list(c.intercept),
        coef=[list(r) for r in c.coef],
        cov=[list(r) for r in c.cov],
        temperature=c.temperature,
        label=c.label,
    )
    fields.update(kw)
    return AffineGaussianConditional(**fields)


def stage2_mean_shift(tc: TrueConditionals, delta: float) -> AffineGaussianConditional:
    """E[z|x,y] + delta * sd(z|x). Shifts q_seq(z|x) mean by delta sd."""
    c = tc.z_xy
    return _copy(c, intercept=[c.intercept[0] + delta * math.sqrt(tc.tau2)], label=f"z|x,y mean+{delta}sd")


def stage2_target_var_ratio(tc: TrueConditionals, ratio: float) -> AffineGaussianConditional:
    """Residual variance set so that Var_seq(z|x) = ratio * Var(z|x)."""
    s2 = ratio * tc.tau2 - tc.b ** 2 * tc.v_y
    if s2 <= 0.0:
        raise ValueError(f"target_var_ratio={ratio} infeasible: residual variance {s2:.4g} <= 0")
    return _copy(tc.z_xy, cov=[[s2]], label=f"z|x,y var ratio {ratio}")


def stage2_corr_scale(tc: TrueConditionals, c: float) -> AffineGaussianConditional:
    """E[z|x,y] = m_{z|x}(x) + c b (y - m_{y|x}(x)), residual var chosen so that
    Var_seq(z|x) = Var(z|x). The target marginal q_seq(z|x) is therefore
    *exactly* the true p(z|x) for every c; only the joint (y, z) | x changes.
    """
    s2 = tc.tau2 - (c * tc.b) ** 2 * tc.v_y
    if s2 <= 0.0:
        raise ValueError(f"corr_scale={c} infeasible: residual variance {s2:.4g} <= 0")
    zx, yx = tc.z_x, tc.y_x
    alpha, beta = zx.intercept[0], zx.coef[0][0]
    gamma, delta = yx.intercept[0], yx.coef[0][0]
    cb = c * tc.b
    return AffineGaussianConditional(
        target=["z"],
        given=["x", "y"],
        intercept=[alpha - cb * gamma],
        coef=[[beta - cb * delta, cb]],
        cov=[[s2]],
        label=f"z|x,y corr scale {c}",
    )


def stage1_mean_shift(tc: TrueConditionals, delta: float) -> AffineGaussianConditional:
    c = tc.y_x
    return _copy(c, intercept=[c.intercept[0] + delta * math.sqrt(tc.v_y)], label=f"y|x mean+{delta}sd")


def collapsed(c: AffineGaussianConditional) -> AffineGaussianConditional:
    return _copy(c, temperature=0.0, label=c.label + " [collapsed]")


def analytic_seq_target(tc: TrueConditionals, stage1: AffineGaussianConditional,
                        stage2: AffineGaussianConditional, x: float) -> Dict[str, float]:
    """Closed-form mean/variance of q_seq(z|x) for affine-Gaussian stages with
    stage2 given (x, y). Used by unit tests to verify the perturbations."""
    if stage2.given != ["x", "y"]:
        raise ValueError("analytic_seq_target expects stage2 given (x, y)")
    t1, t2 = stage1.temperature, stage2.temperature
    m_y = stage1.mean([x])[0]
    v_y = (t1 ** 2) * stage1.cov[0][0]
    bx, by = stage2.coef[0]
    mean = stage2.intercept[0] + bx * x + by * m_y
    var = by ** 2 * v_y + (t2 ** 2) * stage2.cov[0][0]
    return {"mean": mean, "var": var}
