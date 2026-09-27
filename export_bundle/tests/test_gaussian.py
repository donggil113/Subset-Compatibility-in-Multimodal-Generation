import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import linalg  # noqa: E402
from p5compat import samplers as sp  # noqa: E402
from p5compat.gaussian import JointGaussian  # noqa: E402

CORR_NONMARKOV = [[1.0, 0.6, 0.1], [0.6, 1.0, 0.6], [0.1, 0.6, 1.0]]
CORR_MARKOV = [[1.0, 0.6, 0.36], [0.6, 1.0, 0.6], [0.36, 0.6, 1.0]]


def joint(corr=CORR_NONMARKOV):
    return JointGaussian.from_sd_corr(["x", "y", "z"], [0.5, -1.0, 2.0], [1.0, 2.0, 0.5], corr)


class TestLinalg(unittest.TestCase):
    def test_inverse_and_solve(self):
        a = [[4.0, 1.2, 0.3], [1.2, 3.0, 0.5], [0.3, 0.5, 2.0]]
        inv = linalg.spd_inverse(a)
        eye = linalg.matmul(a, inv)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(eye[i][j], 1.0 if i == j else 0.0, places=12)
        b = [1.0, -2.0, 0.5]
        x = linalg.spd_solve(a, b)
        self.assertTrue(all(abs(u - v) < 1e-12 for u, v in zip(linalg.matvec(a, x), b)))

    def test_not_pd_raises(self):
        with self.assertRaises(ValueError):
            linalg.cholesky([[1.0, 2.0], [2.0, 1.0]])


class TestConditionals(unittest.TestCase):
    def test_conditional_matches_precision_form(self):
        # Independent derivation: for z | (x, y), Var = 1/Lambda_zz and
        # coefficients = -Lambda_zg / Lambda_zz, Lambda = Sigma^{-1}.
        j = joint()
        lam = linalg.spd_inverse(j.cov)
        c = j.conditional(["z"], ["x", "y"])
        self.assertAlmostEqual(c.cov[0][0], 1.0 / lam[2][2], places=12)
        self.assertAlmostEqual(c.coef[0][0], -lam[2][0] / lam[2][2], places=12)
        self.assertAlmostEqual(c.coef[0][1], -lam[2][1] / lam[2][2], places=12)

    def test_marginalisation_identity_true_joint(self):
        # Numerical check of the (standard) identity for the true conditionals.
        j = joint()
        tc = sp.true_conditionals(j)
        for x in (-2.0, 0.0, 0.5, 3.1):
            seq = sp.analytic_seq_target(tc, tc.y_x, tc.z_xy, x)
            self.assertAlmostEqual(seq["mean"], tc.z_x.mean([x])[0], places=12)
            self.assertAlmostEqual(seq["var"], tc.tau2, places=12)

    def test_perturbations_closed_form(self):
        tc = sp.true_conditionals(joint())
        tau = math.sqrt(tc.tau2)
        for x in (-1.0, 0.7):
            m0 = tc.z_x.mean([x])[0]
            s = sp.analytic_seq_target(tc, tc.y_x, sp.stage2_mean_shift(tc, 0.3), x)
            self.assertAlmostEqual(s["mean"], m0 + 0.3 * tau, places=12)
            self.assertAlmostEqual(s["var"], tc.tau2, places=12)
            s = sp.analytic_seq_target(tc, tc.y_x, sp.stage2_target_var_ratio(tc, 1.25), x)
            self.assertAlmostEqual(s["mean"], m0, places=12)
            self.assertAlmostEqual(s["var"], 1.25 * tc.tau2, places=12)
            for c in (-1.0, 0.0, 0.5):
                s = sp.analytic_seq_target(tc, tc.y_x, sp.stage2_corr_scale(tc, c), x)
                self.assertAlmostEqual(s["mean"], m0, places=12)
                self.assertAlmostEqual(s["var"], tc.tau2, places=12)
            s = sp.analytic_seq_target(tc, sp.stage1_mean_shift(tc, 0.5), tc.z_xy, x)
            self.assertAlmostEqual(s["mean"], m0 + tc.b * 0.5 * math.sqrt(tc.v_y), places=12)
            s = sp.analytic_seq_target(tc, sp.collapsed(tc.y_x), tc.z_xy, x)
            self.assertAlmostEqual(s["var"], tc.z_xy.cov[0][0], places=12)

    def test_corr_scale_changes_joint(self):
        tc = sp.true_conditionals(joint())
        c = sp.stage2_corr_scale(tc, -1.0)
        # Cov(y, z | x) = coef_y * Var(y | x): sign flips relative to truth.
        self.assertAlmostEqual(c.coef[0][1] * tc.v_y, -tc.yz_x.cov[0][1], places=12)

    def test_markov_chain_equivalence(self):
        jm = joint(CORR_MARKOV)
        tc = sp.true_conditionals(jm)
        self.assertAlmostEqual(jm.partial_corr("x", "z", ["y"]), 0.0, places=12)
        self.assertAlmostEqual(tc.z_xy.coef[0][0], 0.0, places=12)

    def test_infeasible_var_ratio_raises(self):
        tc = sp.true_conditionals(joint())
        with self.assertRaises(ValueError):
            sp.stage2_target_var_ratio(tc, 0.01)

    def test_exact_sampler_moments(self):
        j = joint()
        tc = sp.true_conditionals(j)
        rng = random.Random(123)
        n = 20000
        x = 1.3
        br = sp.Branch("sequential", sp.ExactSampler(tc.z_xy), sp.ExactSampler(tc.y_x))
        z = br.sample(rng, {"x": x}, n)["z"]
        mu = sum(z) / n
        v = sum((u - mu) ** 2 for u in z) / (n - 1)
        se_mu = math.sqrt(tc.tau2 / n)
        se_v = tc.tau2 * math.sqrt(2.0 / (n - 1))
        self.assertLess(abs(mu - tc.z_x.mean([x])[0]), 5 * se_mu)
        self.assertLess(abs(v - tc.tau2), 5 * se_v)

    def test_joint_sampler_moments(self):
        j = joint()
        rng = random.Random(7)
        n = 20000
        rows = [j.sample(rng) for _ in range(n)]
        for a in ("x", "y", "z"):
            for b in ("x", "y", "z"):
                ia, ib = j.names.index(a), j.names.index(b)
                ma = sum(r[a] for r in rows) / n
                mb = sum(r[b] for r in rows) / n
                cov = sum((r[a] - ma) * (r[b] - mb) for r in rows) / (n - 1)
                tol = 5 * math.sqrt((j.cov[ia][ia] * j.cov[ib][ib] + j.cov[ia][ib] ** 2) / n)
                self.assertLess(abs(cov - j.cov[ia][ib]), tol)


if __name__ == "__main__":
    unittest.main()
