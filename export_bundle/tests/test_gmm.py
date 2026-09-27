"""Small fixtures for the GMM controls (numerical checks, not proofs)."""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import gmm as gm  # noqa: E402
from p5compat import metrics as mt  # noqa: E402
from p5compat import samplers as sp  # noqa: E402

NAMES = ["x", "y", "z"]


def fixture():
    return gm.GMM.from_sd_corr(
        NAMES, [0.5, 0.3, 0.2],
        [[-1.0, -1.0, 0.0], [1.0, 1.5, 1.0], [0.0, 0.5, -1.5]],
        [[0.6, 0.8, 0.5], [0.5, 0.6, 0.7], [0.8, 0.5, 0.6]],
        [[[1, 0.5, 0.2], [0.5, 1, 0.4], [0.2, 0.4, 1]],
         [[1, -0.3, 0.4], [-0.3, 1, 0.2], [0.4, 0.2, 1]],
         [[1, 0.2, -0.5], [0.2, 1, 0.3], [-0.5, 0.3, 1]]])


def mix_pdf(comp, v):
    return sum(w * math.exp(-0.5 * ((v - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)) for w, m, s in comp)


class TestGMMAlgebra(unittest.TestCase):
    def test_conditional_density_integrates_to_one(self):
        g = fixture()
        comp = g.conditional_1d("z", ["x", "y"], [0.3, 0.2])
        h = 0.001
        tot = sum(mix_pdf(comp, -8 + h * i) for i in range(int(16 / h))) * h
        self.assertAlmostEqual(tot, 1.0, places=4)

    def test_marginalisation_identity_by_quadrature(self):
        # q(z|x) = int q(z|x,y) q(y|x) dy, checked on a grid of z for two x values.
        g = fixture()
        for x in (-0.7, 0.9):
            qy = g.conditional_1d("y", ["x"], [x])
            direct = g.conditional_1d("z", ["x"], [x])
            h = 0.005
            ys = [-8 + h * i for i in range(int(16 / h))]
            wy = [mix_pdf(qy, y) * h for y in ys]
            conds = [g.conditional_1d("z", ["x", "y"], [x, y]) for y in ys]
            for z in (-2.0, -0.5, 0.4, 1.7):
                seq = sum(w * mix_pdf(c, z) for w, c in zip(wy, conds))
                self.assertAlmostEqual(seq, mix_pdf(direct, z), delta=1e-5)

    def test_marginal_matches_component_blocks(self):
        g = fixture()
        m = g.marginal(["x", "z"])
        self.assertEqual(m.K, 3)
        self.assertAlmostEqual(m.comps[1].cov[0][1], g.comps[1].cov[0][2], places=12)

    def test_conditional_sampler_moments(self):
        g = fixture()
        s = gm.GMMConditionalSampler(g, ["z"], ["x"])
        rng = random.Random(1)
        vals = [v[0] for v in s.sample_batch(rng, [[0.4]] * 20000)]
        m, var = gm.mix_moments(g.conditional_1d("z", ["x"], [0.4]))
        mu = sum(vals) / len(vals)
        self.assertLess(abs(mu - m), 5 * math.sqrt(var / len(vals)))
        v = sum((u - mu) ** 2 for u in vals) / (len(vals) - 1)
        self.assertLess(abs(v - var), 0.05 * var)

    def test_sequential_branch_draws_fresh_intermediates(self):
        g = fixture()
        br = sp.Branch("sequential", gm.GMMConditionalSampler(g, ["z"], ["x", "y"]), gm.GMMConditionalSampler(g, ["y"], ["x"]))
        out = br.sample(random.Random(2), {"x": 0.1}, 64)
        self.assertEqual(len(set(out["y"])), 64)


class TestMixtureDistances(unittest.TestCase):
    def test_single_component_matches_gaussian_formula(self):
        from p5compat.analytic import energy_distance_gauss
        self.assertAlmostEqual(gm.mix_energy_distance([(1.0, 0.3, 1.2)], [(1.0, -0.5, 0.7)]),
                               energy_distance_gauss(0.3, 1.2, -0.5, 0.7), places=12)

    def test_crps_divergence_is_half_energy_distance(self):
        # E_{Y~b} CRPS(a, Y) - E_{Y~b} CRPS(b, Y) = ED(a, b) / 2
        a = [(0.6, -1.0, 0.5), (0.4, 1.0, 0.8)]
        b = [(0.3, 0.0, 1.0), (0.7, 0.5, 0.4)]
        crps_ab = gm.mix_e_abs(a, b) - 0.5 * gm.mix_e_abs(a, a)
        crps_bb = gm.mix_e_abs(b, b) - 0.5 * gm.mix_e_abs(b, b)
        self.assertAlmostEqual(crps_ab - crps_bb, 0.5 * gm.mix_energy_distance(a, b), places=12)

    def test_mixture_ed_against_samples(self):
        rng = random.Random(3)
        a = [(0.6, -1.0, 0.5), (0.4, 1.0, 0.8)]
        b = [(0.3, 0.0, 1.0), (0.7, 0.5, 0.4)]

        def draw(c, n):
            out = []
            for _ in range(n):
                u, acc = rng.random(), 0.0
                for w, m, s in c:
                    acc += w
                    if u < acc:
                        out.append(rng.gauss(m, s))
                        break
                else:
                    out.append(rng.gauss(c[-1][1], c[-1][2]))
            return out

        est = mt.energy_distance_1d(draw(a, 4000), draw(b, 4000), unbiased=True)
        self.assertAlmostEqual(est, gm.mix_energy_distance(a, b), delta=0.03)


class TestEM(unittest.TestCase):
    def test_em_recovers_single_gaussian_and_is_monotone(self):
        g = fixture()
        rng = random.Random(4)
        comp = g.comps[0]
        data = [[comp.sample(rng)[n] for n in NAMES] for _ in range(1500)]
        model, diag = gm.fit_gmm_em(data, NAMES, 2, random.Random(5), max_iter=200)
        self.assertTrue(diag["loglik_monotone"])
        m = gm.mix_moments(model.marginal(["x"]).conditional_1d("x", [], []))[0]
        self.assertAlmostEqual(m, -1.0, delta=0.08)


if __name__ == "__main__":
    unittest.main()
