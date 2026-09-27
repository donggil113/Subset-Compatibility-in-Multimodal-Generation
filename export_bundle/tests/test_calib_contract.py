"""Sampling-contract checks for the calibration code path (numerical checks, not proofs)."""

import json
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat import calib  # noqa: E402
from p5compat import samplers as sp  # noqa: E402
from p5compat.experiment import build_condition, make_joint  # noqa: E402
from p5compat.seeds import seed_v2  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, "configs", "p5_e8_calib_v3.json")))


def tiny_cfg():
    cfg = json.loads(json.dumps(CFG))
    cfg["seed"] = 7  # never the registered seed
    cfg["design"]["N_examples_per_replicate"] = 3
    cfg["design"]["M_samples_per_branch"] = 4
    cfg["design"]["B_permutations"] = 19
    return cfg


class TestSamplingContract(unittest.TestCase):
    def test_level_is_in_every_key(self):
        base = dict(experiment="P5-E8-CALIB", joint="nonmarkov", condition="exact_sequential", replicate=0, example=0)
        for br in ("rows", "direct", "sequential", "permutation"):
            self.assertNotEqual(seed_v2(1, level="target", branch=br, **base), seed_v2(1, level="joint", branch=br, **base))
        self.assertNotEqual(seed_v2(1, level="target", branch="direct", **base),
                            seed_v2(1, level="target", branch="sequential", **base))

    def test_fresh_intermediate_per_sequential_sample(self):
        tc = sp.true_conditionals(make_joint(CFG["joints"]["nonmarkov"]))
        _, seq, _, _ = build_condition(tc, {"id": "exact_sequential", "kind": "exact_sequential"})
        out = seq.sample(random.Random(3), {"x": 0.2, "y_obs": 0.0, "z_obs": 0.0}, 64)
        self.assertEqual(len(set(out["y"])), 64)

    def test_target_and_joint_tasks_use_different_rows_and_samples(self):
        cfg = tiny_cfg()
        c_t = dict(cfg["cells"][0])
        c_j = dict(cfg["cells"][1])
        r_t = calib.run_task((cfg, c_t, 0))
        r_j = calib.run_task((cfg, c_j, 0))
        self.assertNotEqual(r_t["excess_crps_seq"], r_j["excess_crps_seq"])
        self.assertEqual(r_t["min_distinct_intermediates"], 4)
        self.assertTrue(0 < r_t["p"] <= 1 and 0 < r_j["p"] <= 1)

    def test_collapse_both_is_a_compatibility_null_with_ties(self):
        cfg = tiny_cfg()
        qc = dict(cfg["quality_controls"][0])
        r = calib.run_task((cfg, qc, 0))
        self.assertEqual(r["p"], 1.0)  # identical point masses: every allocation ties
        self.assertEqual(r["sd_ratio_seq"], 0.0)

    def test_crps_gauss_closed_form(self):
        # CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi)
        self.assertAlmostEqual(calib.crps_gauss(0.0, 1.0, 0.0), 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), places=12)


if __name__ == "__main__":
    unittest.main()
