"""Fixtures for the R8 CPU-budgeted trainer (implementation checks only; SKIPPED without torch)."""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p5compat import fm_adapter as fa  # noqa: E402

HAS_TORCH = fa.torch is not None
ROWS = [{"x": 0.1 * i, "y": -0.2 * i, "z": 0.05 * i} for i in range(64)]


@unittest.skipUnless(HAS_TORCH, "SKIPPED: torch not installed")
class TestCpuBudgetTraining(unittest.TestCase):
    def setUp(self):
        fa.torch.set_num_threads(1)

    def test_check_every_update_stops_at_full_update_boundary(self):
        m = fa.IndependentConditionalFM(8, 1, seed=1)
        calls = []

        def check(step):
            calls.append(step)
            return "stop" if step >= 7 else None
        info = fa.train(m, ROWS, list(fa.PATTERNS), total_updates=100, batch=8, lr=1e-2, seed=2, log_every=50, check=check, check_every=1)
        self.assertEqual(info["updates"], 7)
        self.assertEqual(calls, list(range(1, 8)))  # checked after every full update
        self.assertEqual(sum(info["pattern_exposure"].values()), 7)
        self.assertEqual(sorted(info["pattern_exposure"].values()), [1, 2, 2, 2])  # balanced round-robin
        self.assertEqual(info["loss_trace"][-1]["update"], 7)  # partial window recorded

    def test_cpu_budget_stop_is_timed_with_process_time(self):
        m = fa.SharedConditionalFM(16, 2, seed=3)
        c0 = time.process_time()
        cap = 0.3

        def check(step):
            return "cpu budget" if time.process_time() - c0 >= cap else None
        info = fa.train(m, ROWS, list(fa.PATTERNS), total_updates=100000, batch=8, lr=1e-2, seed=4, log_every=1000, check=check, check_every=1)
        used = time.process_time() - c0
        self.assertTrue(info["stopped_early"])
        self.assertGreaterEqual(used, cap)
        self.assertLess(used - cap, 0.1)  # overshoot bounded by one update
        self.assertGreater(info["updates"], 10)

    def test_default_check_cadence_unchanged(self):
        m = fa.IndependentConditionalFM(8, 1, seed=5)
        calls = []
        info = fa.train(m, ROWS, list(fa.PATTERNS), total_updates=6, batch=8, lr=1e-2, seed=6, log_every=3, check=lambda s: calls.append(s))
        self.assertEqual(calls, [3, 6])  # without check_every the check runs every log_every updates, as in R6/R7
        self.assertEqual(info["updates"], 6)


if __name__ == "__main__":
    unittest.main()
