import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_geometric_coverage_evictions.py"
SPEC = importlib.util.spec_from_file_location(
    "geometric_coverage_visualization",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def eviction(section, frame, observers=3):
    return {
        "event": "memory_eviction",
        "section_idx": section,
        "section_end_frame": section * 4 + 4,
        "evicted_memory_frame": frame,
        "memory_policy": "slam_covisibility",
        "eviction_score": 0.1,
        "eviction_covisible_observers": observers,
    }


class GeometricCoverageVisualizationTest(unittest.TestCase):
    def test_reconstructs_bank_from_actual_evictions(self):
        events = [
            eviction(0, 1),
            eviction(0, 2),
            eviction(1, 3),
            eviction(1, 5),
            eviction(1, 6),
            eviction(1, 7),
        ]

        snapshots = MODULE.reconstruct_geometric_snapshots(
            events,
            budget=3,
            frames_per_section=5,
        )

        self.assertEqual(snapshots[0]["prospective"], [0, 1, 2, 3, 4])
        self.assertEqual(snapshots[0]["retained"], [0, 3, 4])
        self.assertEqual(snapshots[1]["prospective"], [0, 3, 4, 5, 6, 7, 8])
        self.assertEqual(snapshots[1]["retained"], [0, 4, 8])

    def test_ignores_other_policy_evictions(self):
        other = dict(eviction(0, 2), memory_policy="fifo")
        grouped = MODULE.geometric_evictions_by_section([eviction(0, 1), other])
        self.assertEqual([row["evicted_memory_frame"] for row in grouped[0]], [1])

    def test_recent_bank_preserves_anchor_and_capacity(self):
        bank = MODULE.recent_anchor_bank(range(12), budget=4, anchor=0)
        self.assertEqual(bank, [0, 9, 10, 11])

    def test_bank_support_uses_only_selected_columns(self):
        similarity = np.asarray(
            [
                [0.0, 0.7, 0.2],
                [0.7, 0.0, 0.4],
                [0.2, 0.4, 0.0],
            ]
        )
        support = MODULE.bank_support(similarity, [10, 20, 30], [10, 30])
        np.testing.assert_allclose(support, [0.2, 0.7, 0.2])

    def test_classical_mds_returns_finite_two_dimensional_coordinates(self):
        distances = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ]
        )
        coordinates = MODULE.classical_mds(distances)
        self.assertEqual(coordinates.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(coordinates)))


if __name__ == "__main__":
    unittest.main()
