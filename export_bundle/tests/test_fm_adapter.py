"""Tiny fixtures for the flow-matching adapter (implementation checks only).

They check dimensions, role encoding, path direction, parameter sharing and
that one optimizer step changes exactly the intended parameters. They are not
a search for good training settings. Without torch every test is SKIPPED,
which is not a pass.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import fm_adapter as fa  # noqa: E402

HAS_TORCH = fa.torch is not None
ROWS = [{"x": 0.1 * i, "y": -0.2 * i, "z": 0.05 * i} for i in range(16)]


class TestWithoutTorch(unittest.TestCase):
    @unittest.skipIf(HAS_TORCH, "torch present")
    def test_blocker_message(self):
        with self.assertRaises(RuntimeError) as cm:
            fa.SharedConditionalFM(8, 1, 0)
        self.assertIn("BLOCKER", str(cm.exception))


@unittest.skipUnless(HAS_TORCH, "SKIPPED: torch not installed (installation not approved)")
class TestAdapterWithTorch(unittest.TestCase):
    def setUp(self):
        fa.torch.set_num_threads(1)

    def test_independent_dims_and_disjoint_parameters(self):
        m = fa.IndependentConditionalFM(64, 3, seed=0)
        for k, (g, t) in fa.PATTERNS.items():
            net = m.nets[k]
            self.assertEqual(net[0].in_features, len(t) + len(g) + 1)
            self.assertEqual(net[-1].out_features, len(t))
        ids = [id(p) for n in m.nets.values() for p in n.parameters()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(fa.n_params_model(m)["total"], 34757)

    def test_shared_parameter_count(self):
        self.assertEqual(fa.n_params_model(fa.SharedConditionalFM(128, 3, seed=0))["total"], 34819)

    def test_shared_role_encoding_distinguishes_patterns_and_observed_zero(self):
        torch = fa.torch
        one = torch.zeros(1, 1)
        enc = {k: fa.SharedConditionalFM.encode(k, torch.zeros(1, len(t)), one, torch.zeros(1, len(g)))
               for k, (g, t) in fa.PATTERNS.items()}
        roles = {k: tuple(v[0, 3:9].tolist()) for k, v in enc.items()}
        self.assertEqual(len(set(roles.values())), 4)  # all values 0, yet four distinct inputs
        # observed y = 0 (z|x,y) vs absent y (z|x): same state, different observed mask
        self.assertEqual(enc["z|x,y"][0, 1].item(), 0.0)
        self.assertNotEqual(roles["z|x,y"], roles["z|x"])

    def test_path_direction(self):
        torch = fa.torch
        x0, x1 = torch.full((2, 1), -1.0), torch.full((2, 1), 3.0)
        xt0, u = fa.fm_path(x0, x1, torch.zeros(2, 1))
        xt1, _ = fa.fm_path(x0, x1, torch.ones(2, 1))
        self.assertTrue(torch.equal(xt0, x0) and torch.equal(xt1, x1))
        self.assertTrue(torch.equal(u, x1 - x0))

    def test_euler_integrates_forward_in_time(self):
        class Const:
            def velocity(self, pattern, x, t, cond):
                return fa.torch.full_like(x, 2.0)
        s = fa.FMSampler(Const(), "z|x", 8)
        rng1, rng2 = random.Random(5), random.Random(5)
        out = s.sample_batch(rng1, [[0.0]] * 3)
        base = fa.torch.randn(3, 1, generator=fa.torch.Generator().manual_seed(rng2.getrandbits(63)))
        for a, b in zip(out, base.tolist()):
            self.assertAlmostEqual(a[0], b[0] + 2.0, places=5)

    def test_one_step_updates_only_the_trained_network(self):
        m = fa.IndependentConditionalFM(8, 1, seed=1)
        before = {k: [p.detach().clone() for p in n.parameters()] for k, n in m.nets.items()}
        info = fa.train(m, ROWS, list(fa.PATTERNS), total_updates=1, batch=4, lr=1e-2, seed=2, log_every=1)
        self.assertTrue(info["all_losses_finite"])
        first = list(fa.PATTERNS)[0]
        for k, n in m.nets.items():
            changed = any(not fa.torch.equal(a, b) for a, b in zip(before[k], n.parameters()))
            self.assertEqual(changed, k == first, k)

    def test_shared_gradients_finite_and_shared_across_patterns(self):
        torch = fa.torch
        m = fa.SharedConditionalFM(8, 1, seed=3)
        gen = torch.Generator().manual_seed(4)
        w = m.net[0].weight
        for k in fa.PATTERNS:
            w.grad = None
            loss = fa.fm_loss(m, k, ROWS[:4], gen)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(w.grad).all())
            self.assertGreater(w.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
