import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import analytic as an  # noqa: E402
from p5compat import metrics as mt  # noqa: E402
from p5compat import permutation as pm  # noqa: E402
from p5compat import samplers as sp  # noqa: E402
from p5compat import stats as st  # noqa: E402
from p5compat.diffusion import GuidedPFODESampler  # noqa: E402
from p5compat.gaussian import JointGaussian  # noqa: E402
from p5compat.seeds import seed_v2  # noqa: E402

CORR = [[1.0, 0.6, 0.1], [0.6, 1.0, 0.6], [0.1, 0.6, 1.0]]


def tc():
    return sp.true_conditionals(JointGaussian.from_sd_corr(["x", "y", "z"], [0.5, -1.0, 2.0], [1.0, 2.0, 0.5], CORR))


class TestPooledStat(unittest.TestCase):
    def test_observed_stat_matches_metrics(self):
        rng = random.Random(0)
        a = [rng.gauss(0, 1) for _ in range(7)]
        c = [rng.gauss(0.5, 2) for _ in range(9)]
        ex = pm.PooledExample(a, c)
        self.assertAlmostEqual(ex.stat(ex.observed_mask()), mt.energy_distance_1d(a, c), places=10)
        self.assertAlmostEqual(ex.stat(ex.observed_mask(), unbiased=True),
                               mt.energy_distance_1d(a, c, unbiased=True), places=10)

    def test_vector_stat_matches_sliced(self):
        rng = random.Random(1)
        a = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(6)]
        c = [[rng.gauss(0, 1), rng.gauss(1, 1)] for _ in range(6)]
        dirs = mt.slice_directions(8)
        ex = pm.PooledExample(a, c, dirs)
        self.assertAlmostEqual(ex.stat(ex.observed_mask()), mt.sliced_energy_distance_2d(a, c, dirs), places=10)

    def test_v_and_u_give_identical_permutation_pvalues(self):
        rng = random.Random(2)
        exs = [pm.PooledExample([rng.gauss(0, 1) for _ in range(3)], [rng.gauss(0.3, 1) for _ in range(3)])
               for _ in range(2)]
        vv, tv = pm.exact_null_values(exs, unbiased=False)
        vu, tu = pm.exact_null_values(exs, unbiased=True)
        self.assertAlmostEqual(pm.exact_pvalue(vv, tv), pm.exact_pvalue(vu, tu), places=12)
        r1, r2 = random.Random(5), random.Random(5)
        self.assertEqual(pm.mc_permutation_test(exs, 199, r1)["p"],
                         pm.mc_permutation_test(exs, 199, r2, unbiased=True)["p"])


class TestExactValidity(unittest.TestCase):
    def test_group_superuniformity_random_and_ties(self):
        rng = random.Random(3)
        for _ in range(5):
            exs = [pm.PooledExample([rng.gauss(0, 1) for _ in range(3)], [rng.gauss(0, 1) for _ in range(3)])
                   for _ in range(2)]
            vals, _ = pm.exact_null_values(exs)
            res = pm.group_superuniformity(vals, [0.01, 0.05, 0.1, 0.2, 0.5])
            self.assertTrue(res["all_ok"])
        tie = [pm.PooledExample([1.0] * 3, [1.0] * 3) for _ in range(2)]
        vals, t_obs = pm.exact_null_values(tie)
        self.assertEqual(pm.exact_pvalue(vals, t_obs), 1.0)
        self.assertTrue(pm.group_superuniformity(vals, [0.05])["all_ok"])

    def test_mc_pvalue_never_zero_and_agrees_with_exact(self):
        rng = random.Random(4)
        exs = [pm.PooledExample([rng.gauss(0, 1) for _ in range(4)], [rng.gauss(1.5, 1) for _ in range(4)])
               for _ in range(2)]
        vals, t_obs = pm.exact_null_values(exs)
        p_ex = pm.exact_pvalue(vals, t_obs)
        res = pm.mc_permutation_test(exs, 3999, random.Random(9))
        self.assertGreater(res["p"], 0.0)
        self.assertLess(abs(res["p"] - p_ex), 4 * math.sqrt(p_ex * (1 - p_ex) / 4000) + 1 / 4000)


class TestAnalytic(unittest.TestCase):
    def test_gauss_ed_matches_known_values(self):
        # identical laws -> 0 ; location shift small-delta approx delta^2/sqrt(pi)
        self.assertAlmostEqual(an.energy_distance_gauss(0, 1, 0, 1), 0.0, places=12)
        self.assertAlmostEqual(an.energy_distance_gauss(0, 1, 0.01, 1), 0.01 ** 2 / math.sqrt(math.pi), places=8)
        # point mass vs N(0,1): 2E|X| - E|X-X'| = 2 sqrt(2/pi) - 2/sqrt(pi)
        self.assertAlmostEqual(an.energy_distance_gauss(0, 1, 0, 0),
                               2 * math.sqrt(2 / math.pi) - 2 / math.sqrt(math.pi), places=12)

    def test_expected_v1_contrast_monte_carlo(self):
        rng = random.Random(6)
        m, reps = 32, 3000
        vals = []
        for _ in range(reps):
            a = [rng.gauss(0, 1) for _ in range(m)]
            b = [rng.gauss(0, 1) for _ in range(m)]
            c = [rng.gauss(0, 0.6) for _ in range(m)]
            vals.append(mt.energy_distance_1d(a, c) - mt.energy_distance_1d(a, b))
        ci = st.mean_ci(vals)
        pred = an.expected_v1_contrast(an.energy_distance_gauss(0, 1, 0, 0.6), 0.6, 1.0, m)
        self.assertLess(abs(ci["mean"] - pred), 5 * ci["se"])

    def test_affine_map_matches_sampler(self):
        t = tc()
        for solver, steps, w in (("euler", 4, 0.0), ("euler", 16, 2.0), ("heun", 8, 1.0)):
            smp = GuidedPFODESampler(t.z_x, t.z, n_steps=steps, solver=solver, guidance=w)
            amap = an.pfode_affine_map(t.z_x.cov[0][0], t.z.cov[0][0], w, steps, solver)
            x0 = [-30.0, 3.0, 71.0]
            m_c = [1.2, 2.0, 2.4]
            out = smp.integrate(x0, m_c)
            for o, x, mc in zip(out, x0, m_c):
                self.assertAlmostEqual(o, amap.g * x + amap.h_c * mc + amap.h_u * t.z.intercept[0], places=8)

    def test_pfode_laws_unguided_many_steps_close_to_exact(self):
        t = tc()
        laws = an.pfode_branch_laws(t, 0.7, n_steps=512, solver="heun", guidance=0.0)
        for key in ("direct", "sequential"):
            self.assertAlmostEqual(laws[key][0], laws["exact"][0], delta=0.02)
            self.assertAlmostEqual(laws[key][1], laws["exact"][1], delta=0.002)


    def test_exact_flow_limit_matches_fine_heun(self):
        t = tc()
        lim = an.exact_flow_branch_laws(t, -0.3)
        fine = an.pfode_branch_laws(t, -0.3, n_steps=2048, solver="heun", guidance=0.0)
        for key in ("direct", "sequential"):
            self.assertAlmostEqual(lim[key][0], fine[key][0], delta=1e-4)
            self.assertAlmostEqual(lim[key][1], fine[key][1], delta=1e-4)


class TestStatsAndSeeds(unittest.TestCase):
    def test_clopper_pearson_known(self):
        ci = st.clopper_pearson(0, 20)
        self.assertAlmostEqual(ci["hi"], 1 - 0.025 ** (1 / 20), places=6)
        ci = st.clopper_pearson(20, 20)
        self.assertAlmostEqual(ci["lo"], 0.025 ** (1 / 20), places=6)
        up = st.clopper_pearson(0, 20, sided="upper")
        self.assertAlmostEqual(up["hi"], 1 - 0.05 ** (1 / 20), places=6)

    def test_seed_v2_distinguishes_joint(self):
        base = dict(experiment="E", condition="c", example=0, replicate=0, branch="A")
        self.assertNotEqual(seed_v2(1, joint="markov", **base), seed_v2(1, joint="nonmarkov", **base))
        self.assertEqual(seed_v2(1, joint="markov", **base), seed_v2(1, joint="markov", **base))
        with self.assertRaises(ValueError):
            seed_v2(1, joint="markov")


if __name__ == "__main__":
    unittest.main()
