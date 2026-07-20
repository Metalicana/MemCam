import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "analysis" / "unbounded_scaling" / "plot_unbounded_scaling_estimate.py"
)
SPEC = importlib.util.spec_from_file_location("unbounded_scaling", MODULE_PATH)
SCALING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALING)


class UnboundedScalingEstimateTest(unittest.TestCase):
    def test_estimate_uses_linear_storage_and_quadratic_search(self):
        rows = SCALING.estimate_scaling([60, 600, 3600])

        self.assertEqual(rows[0]["sections"], 24)
        self.assertEqual(rows[0]["stored_frames"], 1825)
        self.assertAlmostEqual(rows[0]["memory_bank_gb"], 2.2974, places=4)
        self.assertAlmostEqual(rows[1]["memory_bank_gb"], 22.6757, places=4)
        self.assertGreater(rows[2]["rollout_latency_days"], 30)
        self.assertGreater(
            rows[2]["overlap_evaluations"],
            30 * rows[1]["overlap_evaluations"],
        )


if __name__ == "__main__":
    unittest.main()
