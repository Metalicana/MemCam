import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_geometric_coverage_budget_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "geometric_coverage_budget_sweep",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeometricCoverageBudgetSweepTest(unittest.TestCase):
    def test_validate_runs_sorts_and_extracts_budgets(self):
        pairs = MODULE.validate_runs(
            ["slam_b64_covisibility", "slam_b16_covisibility"]
        )
        self.assertEqual(
            pairs,
            [(16, "slam_b16_covisibility"), (64, "slam_b64_covisibility")],
        )

    def test_validate_runs_rejects_duplicate_budgets(self):
        with self.assertRaises(ValueError):
            MODULE.validate_runs(["slam_b32_a", "slam_b32_b"])

    def test_support_curves_are_monotonic(self):
        thresholds = np.asarray([0.0, 0.5, 0.8, 1.0])
        curves = MODULE.support_curves(np.asarray([0.4, 0.7, 0.9]), thresholds)
        np.testing.assert_allclose(curves, [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])

    def test_coverage_rows_preserve_policy_and_budget(self):
        coverage = {
            "camera": {
                "Geometric Coverage": {16: np.asarray([0.5, 0.9])},
                "Recent + anchor": {16: np.asarray([0.2, 0.8])},
            }
        }
        rows = MODULE.coverage_rows(3, "scene", 4, 10, coverage)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["policy"] for row in rows}, {"Geometric Coverage", "Recent + anchor"})
        self.assertEqual({row["budget"] for row in rows}, {16})


if __name__ == "__main__":
    unittest.main()
