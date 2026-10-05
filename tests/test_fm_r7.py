"""Fixtures for the round-7 keyed randomness (implementation checks only; SKIPPED without torch)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import fm_adapter as fa  # noqa: E402

HAS_TORCH = fa.torch is not None


@unittest.skipUnless(HAS_TORCH, "SKIPPED: torch not installed")
class TestKeyedNoise(unittest.TestCase):
    def setUp(self):
        from p5compat import syn_learn_fm_r7 as r7
        self.r7 = r7
        fa.torch.set_num_threads(1)

    def test_noise_is_keyed_and_step_free(self):
        r7 = self.r7
        a = r7.noise_matrix(1, "test", "R1", "SHARED_CONDITIONAL_FM", "z|x", 3, "direct", "sampling:target", 8, 1)
        b = r7.noise_matrix(1, "test", "R1", "SHARED_CONDITIONAL_FM", "z|x", 3, "direct", "sampling:target", 8, 1)
        self.assertTrue(fa.torch.equal(a, b))  # deterministic; no step count enters
        for kw in (dict(split="dev"), dict(rep="R2"), dict(arm="INDEPENDENT_CONDITIONAL_FM"), dict(pattern="y|x"), dict(example=4),
                   dict(branch="sequential"), dict(purpose="sampling:joint")):
            args = dict(run_seed=1, split="test", rep="R1", arm="SHARED_CONDITIONAL_FM", pattern="z|x", example=3, branch="direct",
                        purpose="sampling:target", m=8, dim=1)
            args.update(kw)
            c = r7.noise_matrix(**args)
            self.assertFalse(fa.torch.equal(a, c), kw)
        self.assertEqual(len({tuple(r) for r in a.tolist()}), 8)  # distinct per sample_id

    def test_same_noise_across_step_counts_and_fresh_intermediates(self):
        r7 = self.r7

        class Const:
            def velocity(self, pattern, x, t, cond):
                return fa.torch.full_like(x, 1.0)
        ctx = r7.Ctx(5, "test", "R1", "SHARED_CONDITIONAL_FM", 2, "target")
        z128 = r7.draw_direct(Const(), ctx, 0.3, 16, 128)
        z512 = r7.draw_direct(Const(), ctx, 0.3, 16, 512)
        for u, v in zip(z128, z512):
            self.assertAlmostEqual(u, v, places=4)  # same initial noise, same integral of a constant field
        seq = r7.draw_sequential(Const(), ctx, 0.3, 16, 8)
        self.assertEqual(len(set(seq["y"])), 16)  # a fresh y per z sample
        direct_noise = ctx.noise("z|x", "direct", 16, 1)
        seq_noise = ctx.noise("z|x,y", "sequential", 16, 1)
        self.assertFalse(fa.torch.equal(direct_noise, seq_noise))  # branches share no stream

    def test_register_seeds_are_integers_and_distinct(self):
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "p5_syn_learn_01_fm_replicates_r7.json")
        if not os.path.exists(p):
            self.skipTest("r7 config not registered yet")
        r7 = json.load(open(p))
        vals = [v for rep in r7["replicates"].values() for arm in rep.values() for v in arm.values()]
        self.assertTrue(all(isinstance(v, int) for v in vals))
        self.assertEqual(len(vals), len(set(vals)))
        pilot = [v for arm in r7["pilot"]["pilot_model_seeds"].values() for v in arm.values()]
        self.assertFalse(set(vals) & set(pilot))
        self.assertTrue(r7["test_set"]["disjoint_from_pilot_test"])


if __name__ == "__main__":
    unittest.main()
