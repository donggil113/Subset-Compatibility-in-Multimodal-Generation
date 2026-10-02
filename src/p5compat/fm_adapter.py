"""Conditional flow-matching learners for P5-SYN-LEARN-01 (PyTorch adapter).

STATUS: executed for the first time in round 6 under a limited CPU-only
install approval; see results/env/ and tests/test_fm_adapter.py (the fixtures
are SKIPPED, not passed, when torch is absent). Importing this module without
torch is fine; constructing a model raises a RuntimeError naming the blocker.

Design (configs/p5_syn_learn_01.json; amendments A1 and A2 in configs/):
* Loss: conditional flow matching with the linear (rectified-flow) path
  x_t = (1 - t) x0 + t x1, x0 ~ N(0, I) at t = 0, data x1 at t = 1, target
  velocity x1 - x0, t ~ U(0, 1). The sampler integrates from t = 0 to t = 1.
* INDEPENDENT_CONDITIONAL_FM: one MLP per conditional q(z|x), q(y|x),
  q(z|x,y), q(y,z|x); no shared parameters and one Adam optimizer per MLP.
* SHARED_CONDITIONAL_FM (SHARED_CONDITIONAL_NET): one MLP over the full
  (x, y, z) state. Each coordinate has one of three roles, encoded by two
  masks: observed (value, obs = 1), target (x_t, tgt = 1) or absent (0, both
  masks 0). The loss is taken on target coordinates only, and the four
  conditioning patterns are sampled uniformly. Sharing parameters does NOT
  make these conditionals those of one normalized joint.
* Sampler: Euler on the learned ODE from the base N(0, I), CFG off.

Amendment A1 (static review, before any execution):
* D1: the shared network originally received only an observation mask, so an
  absent coordinate (e.g. y for z|x) and a target coordinate (y for y,z|x)
  were distinguishable only by the slot value being exactly 0; z|x, y|x and
  y,z|x had the same mask. A target mask is now part of the input.
* D2: the independent arm used one Adam optimizer over all four MLPs. That is
  not an error in general: with gradients reset to None (the default of
  zero_grad since torch 2.0) Adam skips the untouched networks. The contract
  depended on that None-vs-zero distinction, so each MLP now has its own
  optimizer and gradients are reset to None explicitly (clarification, not a
  bug fix).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence

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


def fm_path(x0, x1, t):
    """Linear path: x_t and its velocity; t = 0 is noise, t = 1 is data."""
    return (1 - t) * x0 + t * x1, x1 - x0


class IndependentConditionalFM:
    """One velocity network per conditioning pattern."""

    name = "INDEPENDENT_CONDITIONAL_FM"

    def __init__(self, width: int, depth: int, seed: int):
        _require_torch()
        torch.manual_seed(seed)
        self.nets = {k: mlp(len(g) + len(t) + 1, len(t), width, depth) for k, (g, t) in PATTERNS.items()}

    def velocity(self, pattern, xt, t, cond):
        return self.nets[pattern](torch.cat([xt, cond, t], dim=1))

    def parameters(self):
        return [p for n in self.nets.values() for p in n.parameters()]

    def optimizers(self, lr: float):
        return {k: torch.optim.Adam(n.parameters(), lr=lr) for k, n in self.nets.items()}

    def state_dict(self):
        return {k: n.state_dict() for k, n in self.nets.items()}

    def load_state_dict(self, sd):
        for k, n in self.nets.items():
            n.load_state_dict(sd[k])


class SharedConditionalFM:
    """One role-conditioned velocity network over the full state (x, y, z)."""

    name = "SHARED_CONDITIONAL_FM"

    def __init__(self, width: int, depth: int, seed: int):
        _require_torch()
        torch.manual_seed(seed)
        self.net = mlp(3 + 3 + 3 + 1, 3, width, depth)  # state, observed mask, target mask, time

    @staticmethod
    def encode(pattern, xt, t, cond):
        g, tg = PATTERNS[pattern]
        n = xt.shape[0]
        state, obs, tgt = torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, 3)
        for j, v in enumerate(g):
            state[:, VARS.index(v)] = cond[:, j]
            obs[:, VARS.index(v)] = 1.0
        for j, v in enumerate(tg):
            state[:, VARS.index(v)] = xt[:, j]
            tgt[:, VARS.index(v)] = 1.0
        return torch.cat([state, obs, tgt, t], dim=1)

    def velocity(self, pattern, xt, t, cond):
        out = self.net(self.encode(pattern, xt, t, cond))
        return out[:, [VARS.index(v) for v in PATTERNS[pattern][1]]]

    def parameters(self):
        return list(self.net.parameters())

    def optimizers(self, lr: float):
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        return {k: opt for k in PATTERNS}

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


def _cond_and_target(pattern, batch_rows):
    g, tg = PATTERNS[pattern]
    cond = torch.tensor([[r[v] for v in g] for r in batch_rows], dtype=torch.float32)
    x1 = torch.tensor([[r[v] for v in tg] for r in batch_rows], dtype=torch.float32)
    return cond, x1


def fm_loss(model, pattern, batch_rows, gen):
    cond, x1 = _cond_and_target(pattern, batch_rows)
    x0 = torch.randn(x1.shape, generator=gen)
    t = torch.rand(x1.shape[0], 1, generator=gen)
    xt, u = fm_path(x0, x1, t)
    return ((model.velocity(pattern, xt, t, cond) - u) ** 2).mean()


def held_out_fm_loss(model, pattern, rows, noise_seed: int) -> float:
    """Flow-matching loss on held-out rows with FIXED noise (x0, t) from noise_seed,
    so that two checkpoints are compared on identical draws. A diagnostic of
    optimization progress, not a generative-quality score."""
    gen = torch.Generator().manual_seed(noise_seed)
    with torch.no_grad():
        return float(fm_loss(model, pattern, rows, gen))


def train(model, rows: Sequence[Dict[str, float]], patterns: List[str], total_updates: int, batch: int,
          lr: float, seed: int, log_every: int = 1000, opts: Optional[Dict] = None,
          hook: Optional[Callable[[int, object], None]] = None, hook_at: Sequence[int] = (),
          check: Optional[Callable[[int], None]] = None) -> Dict:
    """Round-robin over patterns for the independent arm (total_updates / 4 per
    network); uniform pattern sampling for the shared arm. The final iterate is
    the model; there is no checkpoint selection. `hook(step, model)` runs after
    the updates listed in hook_at; `check(step)` runs every log_every updates
    and stops training when it returns a reason string (budget guard)."""
    _require_torch()
    import random as _r
    gen = torch.Generator().manual_seed(seed)
    rr = _r.Random(seed)
    opts = opts if opts is not None else model.optimizers(lr)
    exposure = {p: 0 for p in patterns}
    trace, window, all_finite = [], [], True
    hook_at = set(hook_at)
    w0, c0 = time.time(), time.process_time()
    done, stop_reason = 0, None
    for step in range(total_updates):
        pattern = patterns[step % len(patterns)] if isinstance(model, IndependentConditionalFM) else rr.choice(patterns)
        exposure[pattern] += 1
        idx = [rr.randrange(len(rows)) for _ in range(batch)]
        loss = fm_loss(model, pattern, [rows[i] for i in idx], gen)
        opt = opts[pattern]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        done = step + 1
        v = loss.item()
        all_finite = all_finite and math.isfinite(v)
        window.append(v)
        if done % log_every == 0:
            trace.append({"update": done, "mean_train_loss": sum(window) / len(window)})
            window = []
            if check is not None:
                stop_reason = check(done)
        if hook is not None and done in hook_at:
            hook(done, model)
        if stop_reason:
            break
    return {"updates": done, "updates_planned": total_updates, "stopped_early": stop_reason, "pattern_exposure": exposure,
            "rows_seen_per_pattern": {p: e * batch for p, e in exposure.items()},
            "n_params": n_params_model(model), "loss_trace": trace, "all_losses_finite": all_finite,
            "train_wall_seconds": time.time() - w0, "train_cpu_seconds": time.process_time() - c0,
            "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__}


def n_params_model(model) -> Dict[str, int]:
    if isinstance(model, IndependentConditionalFM):
        per = {k: n_params(n) for k, n in model.nets.items()}
        return {**per, "total": sum(per.values())}
    return {"total": n_params(model.net)}


class FMSampler:
    """Adapter to the samplers.Branch interface: sample_batch(rng, gs).

    The initial noise is drawn from a torch generator seeded from the branch
    RNG, so the draw depends on the branch stream and the call order only, not
    on the number of Euler steps (two solver levels share the same noise)."""

    def __init__(self, model, pattern: str, n_steps: int):
        _require_torch()
        self.model, self.pattern, self.n_steps = model, pattern, n_steps
        self.given, self.target = PATTERNS[pattern]

    def sample_batch(self, rng, gs):
        gen = torch.Generator().manual_seed(rng.getrandbits(63))  # stream derived from the branch RNG
        cond = torch.tensor(gs, dtype=torch.float32)
        x = torch.randn(len(gs), len(self.target), generator=gen)  # rows are independent; no batch coupling
        h = 1.0 / self.n_steps
        with torch.no_grad():
            for k in range(self.n_steps):
                t = torch.full((len(gs), 1), k * h)
                x = x + h * self.model.velocity(self.pattern, x, t, cond)
        return x.tolist()
