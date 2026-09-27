import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat.diffusion import GuidedPFODESampler, edm_sigmas, euler_affine_factor, exact_ode_factor  # noqa: E402
from p5compat.gaussian import AffineGaussianConditional  # noqa: E402


def cond(m, v):
    return AffineGaussianConditional(target=["z"], given=[], intercept=[m], coef=[[]], cov=[[v]])


class TestPFODE(unittest.TestCase):
    def test_schedule(self):
        s = edm_sigmas(8, 0.002, 80.0, 7.0)
        self.assertEqual(len(s), 9)
        self.assertAlmostEqual(s[0], 80.0)
        self.assertAlmostEqual(s[-2], 0.002)
        self.assertEqual(s[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(s, s[1:])))

    def test_euler_matches_closed_form(self):
        c = cond(1.5, 0.25)
        for steps in (2, 5, 32):
            smp = GuidedPFODESampler(c, cond(0.0, 4.0), n_steps=steps, solver="euler")
            x0 = [-3.0, 0.0, 10.0, 55.0]
            out = smp.integrate(x0, [1.5] * 4)
            f = euler_affine_factor(smp.sigmas, 0.25)
            for a, b in zip(out, x0):
                self.assertAlmostEqual(a, 1.5 + f * (b - 1.5), places=9)

    def test_heun_converges_to_exact_flow(self):
        c = cond(-0.4, 0.3)
        smp = GuidedPFODESampler(c, cond(0.0, 1.0), n_steps=256, solver="heun")
        x0 = [17.0]
        out = smp.integrate(x0, [-0.4])[0]
        exact = -0.4 + exact_ode_factor(80.0, 0.3) * (17.0 + 0.4)
        self.assertAlmostEqual(out, exact, delta=2e-3)

    def test_guidance_zero_and_trivial_guidance(self):
        c = cond(0.7, 0.5)
        base = GuidedPFODESampler(c, cond(-2.0, 3.0), n_steps=16, guidance=0.0)
        same = GuidedPFODESampler(c, cond(0.7, 0.5), n_steps=16, guidance=3.0)  # uncond == cond
        g = GuidedPFODESampler(c, cond(-2.0, 3.0), n_steps=16, guidance=3.0)
        r1, r2, r3 = random.Random(0), random.Random(0), random.Random(0)
        a = base.sample_batch(r1, [[]] * 5)
        b = same.sample_batch(r2, [[]] * 5)
        d = g.sample_batch(r3, [[]] * 5)
        for u, v in zip(a, b):
            self.assertAlmostEqual(u[0], v[0], places=9)
        self.assertTrue(any(abs(u[0] - v[0]) > 1e-3 for u, v in zip(a, d)))


if __name__ == "__main__":
    unittest.main()
