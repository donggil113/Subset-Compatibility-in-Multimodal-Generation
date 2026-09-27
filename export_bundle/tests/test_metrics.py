import math
import os
import random
import sys
import unittest
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import metrics as mt  # noqa: E402
from p5compat import stats as st  # noqa: E402


def gaussian_crps(mu, sigma, y):
    w = (y - mu) / sigma
    nd = NormalDist()
    return sigma * (w * (2 * nd.cdf(w) - 1) + 2 * nd.pdf(w) - 1 / math.sqrt(math.pi))


class TestEnergyDistance(unittest.TestCase):
    def test_sorted_matches_bruteforce(self):
        rng = random.Random(0)
        for n, m in ((5, 7), (30, 30), (50, 13)):
            a = [rng.gauss(0, 1) for _ in range(n)]
            b = [rng.gauss(0.3, 1.5) for _ in range(m)]
            self.assertAlmostEqual(mt.energy_distance_1d(a, b), mt.energy_distance_1d_bruteforce(a, b), places=10)

    def test_ties_and_identity(self):
        a = [1.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(mt.energy_distance_1d(a, list(a)), 0.0, places=12)
        self.assertAlmostEqual(mt.energy_distance_1d([2.0] * 5, [2.0] * 9), 0.0, places=12)
        self.assertAlmostEqual(mt.energy_distance_1d([0.0] * 3, [1.0] * 3), 2.0, places=12)

    def test_noise_floor_expectation(self):
        # E[ED_V] under P=Q is (1/n + 1/m) E|X-X'| ; for N(0,1), E|X-X'| = 2/sqrt(pi).
        rng = random.Random(1)
        n = m = 64
        reps = 1500
        vals = []
        for _ in range(reps):
            a = [rng.gauss(0, 1) for _ in range(n)]
            b = [rng.gauss(0, 1) for _ in range(m)]
            vals.append(mt.energy_distance_1d(a, b))
        ci = st.mean_ci(vals)
        expected = (1 / n + 1 / m) * 2 / math.sqrt(math.pi)
        self.assertLess(abs(ci["mean"] - expected), 5 * ci["se"])

    def test_unbiased_version_centred(self):
        rng = random.Random(2)
        vals = []
        for _ in range(1500):
            a = [rng.gauss(0, 1) for _ in range(40)]
            b = [rng.gauss(0, 1) for _ in range(40)]
            vals.append(mt.energy_distance_1d(a, b, unbiased=True))
        ci = st.mean_ci(vals)
        self.assertLess(abs(ci["mean"]), 5 * ci["se"])

    def test_sliced_detects_correlation_flip_but_marginals_equal(self):
        rng = random.Random(3)
        n = 400
        rho = 0.7

        def draw(r):
            out = []
            for _ in range(n):
                u = rng.gauss(0, 1)
                v = r * u + math.sqrt(1 - r * r) * rng.gauss(0, 1)
                out.append([u, v])
            return out

        a, b = draw(rho), draw(-rho)
        dirs = mt.slice_directions(8)
        axis_only = [dirs[0], dirs[4]]  # (1,0) and (0,1): marginals only
        self.assertLess(mt.sliced_energy_distance_2d(a, b, axis_only), 0.02)

        # Population value: projection on (cos t, sin t) has variance 1 +/- rho sin 2t;
        # ED(N(0,s1^2), N(0,s2^2)) = 2 sqrt(2/pi) sqrt(s1^2+s2^2) - (2/sqrt(pi)) (s1+s2).
        def ed_pop(v1, v2):
            return 2 * math.sqrt(2 / math.pi) * math.sqrt(v1 + v2) - 2 / math.sqrt(math.pi) * (
                math.sqrt(v1) + math.sqrt(v2))

        pop = sum(ed_pop(1 + rho * (2 * u0 * u1), 1 - rho * (2 * u0 * u1)) for u0, u1 in dirs) / len(dirs)
        self.assertAlmostEqual(pop, 0.0797, delta=0.001)
        got = mt.sliced_energy_distance_2d(a, b, dirs)
        # V-statistic bias under equal variances is (2/n) E|X-X'| ~ 0.006 at n=400.
        self.assertAlmostEqual(got, pop + 2 / n * 2 / math.sqrt(math.pi), delta=0.03)


class TestCRPS(unittest.TestCase):
    def test_matches_gaussian_closed_form(self):
        rng = random.Random(4)
        mu, sigma = 1.0, 2.0
        samples = [rng.gauss(mu, sigma) for _ in range(20000)]
        for y in (-1.0, 1.0, 4.0):
            self.assertAlmostEqual(mt.crps_fair(samples, y), gaussian_crps(mu, sigma, y), delta=0.03)

    def test_collapsed_crps_is_absolute_error(self):
        self.assertAlmostEqual(mt.crps_fair([2.0] * 10, 3.5), 1.5, places=12)

    def test_proper_score_prefers_truth_over_collapse(self):
        # Expected CRPS: sigma/sqrt(pi) (truth) vs sigma*sqrt(2/pi) (point mass at mean).
        rng = random.Random(5)
        truth, point = [], []
        for _ in range(3000):
            y = rng.gauss(0, 1)
            s = [rng.gauss(0, 1) for _ in range(64)]
            truth.append(mt.crps_fair(s, y))
            point.append(mt.crps_fair([0.0] * 64, y))
        self.assertLess(st.mean_ci(truth)["mean"], st.mean_ci(point)["mean"])


class TestStats(unittest.TestCase):
    def test_sign_flip_edge_cases(self):
        rng = random.Random(6)
        self.assertEqual(st.sign_flip_pvalue([0.0] * 50, 199, rng), 1.0)
        self.assertLess(st.sign_flip_pvalue([1.0 + 0.01 * i for i in range(50)], 199, rng), 0.01)

    def test_sign_flip_null_calibration(self):
        rng = random.Random(8)
        ps = []
        for _ in range(300):
            d = [rng.gauss(0, 1) for _ in range(40)]
            ps.append(st.sign_flip_pvalue(d, 199, rng))
        rate = sum(p < 0.05 for p in ps) / len(ps)
        ci = st.wilson_ci(sum(p < 0.05 for p in ps), len(ps))
        self.assertTrue(ci["lo"] <= 0.05 <= ci["hi"] or rate < 0.1)

    def test_wilson(self):
        ci = st.wilson_ci(2, 40)
        self.assertLess(ci["lo"], 0.05)
        self.assertGreater(ci["hi"], 0.05)

    def test_log_sd_ratio_degenerate(self):
        collapsed = [2.0 + 0.1 + 0.2 - 0.3] * 256  # identical floats with round-off-prone value
        self.assertIsNone(mt.log_sd_ratio([0.0, 1.0, 2.0], collapsed))
        self.assertAlmostEqual(mt.log_sd_ratio([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]), math.log(2.0), places=12)

    def test_paired_mse_is_not_a_compatibility_metric(self):
        rng = random.Random(9)
        n = 20000
        a = [rng.gauss(0, 1) for _ in range(n)]
        b = [rng.gauss(0, 1) for _ in range(n)]
        # Same distribution, independent draws: paired MSE ~ 2, not 0.
        self.assertAlmostEqual(mt.paired_index_mse(a, b), 2.0, delta=0.1)
        # Collapsed pair: paired MSE 0 although both are wrong.
        self.assertEqual(mt.paired_index_mse([0.0] * 10, [0.0] * 10), 0.0)


if __name__ == "__main__":
    unittest.main()
