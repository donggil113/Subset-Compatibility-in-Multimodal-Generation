"""Conditional flow-matching learners for P5-SYN-LEARN-01 (PyTorch adapter).

STATUS: UNTESTED. PyTorch is not installed in the environment where this file
was written, and installing it was not approved, so this code has never been
executed. It fixes the intended design so that a later, approved run does not
re-decide architecture or budget after seeing data. Importing this module
without torch raises a RuntimeError naming the blocker.

Design (fixed in configs/p5_syn_learn_01.json):
* Loss: conditional flow matching with the linear (rectified-flow) path
  x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), target velocity x1 - x0, t ~ U(0, 1).
* INDEPENDENT_CONDITIONAL_FM: one MLP per conditional q(y|x), q(z|x),
  q(z|x,y), q(y,z|x).
* SHARED_CONDITIONAL_FM (SHARED_CONDITIONAL_NET): one MLP over the full
  (x, y, z) state with an observation mask; observed coordinates are clamped
  to their values, the loss is taken on target coordinates only, and the four
  conditioning patterns are sampled uniformly. Sharing parameters does NOT
  make these conditionals those of one normalized joint.
* Sampler: Euler on the learned ODE from the base N(0, I), CFG off; the
  number of steps is chosen on dev only among the two pre-declared levels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

try:  # pragma: no cover - torch absent in the development environment
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

VARS = ["x", "y", "z"]
PATTERNS = {  # name: (given, target)
    "z|x": (["x"], ["z"]),
    "y|x": (["x"], ["y"]),
    "z|x,y": (["x", "y"], ["z"]),
    "y,z|x": (["x"], ["y", "z"]),
}


def _require_torch():
    if torch is None:
        raise RuntimeError("BLOCKER: PyTorch is not installed; installation not approved. "
                           "P5-SYN-LEARN-01 flow-matching arms are NOT_RUN.")


def mlp(d_in: int, d_out: int, width: int, depth: int):
    _require_torch()
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


class IndependentConditionalFM:
    """One velocity network per conditioning pattern."""

    def __init__(self, width: int, depth: int, seed: int):
        _require_torch()
        torch.manual_seed(seed)
        self.nets = {k: mlp(len(g) + len(t) + 1, len(t), width, depth) for k, (g, t) in PATTERNS.items()}

    def velocity(self, pattern, xt, t, cond):
        return self.nets[pattern](torch.cat([xt, cond, t], dim=1))

    def parameters(self):
        return [p for n in self.nets.values() for p in n.parameters()]


class SharedConditionalFM:
    """One mask-conditioned velocity network over the full state (x, y, z)."""

    def __init__(self, width: int, depth: int, seed: int):
        _require_torch()
        torch.manual_seed(seed)
        self.net = mlp(3 + 3 + 1, 3, width, depth)  # state, observation mask, time

    def velocity(self, pattern, xt, t, cond):
        g, tg = PATTERNS[pattern]
        n = xt.shape[0]
        state = torch.zeros(n, 3)
        mask = torch.zeros(n, 3)
        for j, v in enumerate(g):
            state[:, VARS.index(v)] = cond[:, j]
            mask[:, VARS.index(v)] = 1.0
        for j, v in enumerate(tg):
            state[:, VARS.index(v)] = xt[:, j]
        out = self.net(torch.cat([state, mask, t], dim=1))
        return out[:, [VARS.index(v) for v in tg]]

    def parameters(self):
        return list(self.net.parameters())


def fm_loss(model, pattern, batch_rows, gen):
    g, tg = PATTERNS[pattern]
    cond = torch.tensor([[r[v] for v in g] for r in batch_rows], dtype=torch.float32)
    x1 = torch.tensor([[r[v] for v in tg] for r in batch_rows], dtype=torch.float32)
    x0 = torch.randn(x1.shape, generator=gen)
    t = torch.rand(x1.shape[0], 1, generator=gen)
    xt = (1 - t) * x0 + t * x1
    return ((model.velocity(pattern, xt, t, cond) - (x1 - x0)) ** 2).mean()


def train(model, rows: Sequence[Dict[str, float]], patterns: List[str], total_updates: int, batch: int,
          lr: float, seed: int) -> Dict:
    """Round-robin over patterns for the independent arm; uniform pattern sampling
    for the shared arm is obtained by passing all patterns (each update uses one)."""
    _require_torch()
    import random as _r
    gen = torch.Generator().manual_seed(seed)
    rr = _r.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    exposure = {p: 0 for p in patterns}
    for step in range(total_updates):
        pattern = patterns[step % len(patterns)] if isinstance(model, IndependentConditionalFM) else rr.choice(patterns)
        exposure[pattern] += 1
        idx = [rr.randrange(len(rows)) for _ in range(batch)]
        loss = fm_loss(model, pattern, [rows[i] for i in idx], gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"updates": total_updates, "pattern_exposure": exposure}


class FMSampler:
    """Adapter to the samplers.Branch interface: sample_batch(rng, gs)."""

    def __init__(self, model, pattern: str, n_steps: int):
        _require_torch()
        self.model, self.pattern, self.n_steps = model, pattern, n_steps
        self.given, self.target = PATTERNS[pattern]

    def sample_batch(self, rng, gs):
        gen = torch.Generator().manual_seed(rng.getrandbits(63))  # stream derived from the branch RNG
        cond = torch.tensor(gs, dtype=torch.float32)
        x = torch.randn(len(gs), len(self.target), generator=gen)
        h = 1.0 / self.n_steps
        with torch.no_grad():
            for k in range(self.n_steps):
                t = torch.full((len(gs), 1), k * h)
                x = x + h * self.model.velocity(self.pattern, x, t, cond)
        return x.tolist()
