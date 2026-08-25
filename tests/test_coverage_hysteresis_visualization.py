import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_coverage_hysteresis.py"
SPEC = importlib.util.spec_from_file_location(
    "coverage_hysteresis_visualization",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_events():
    return [
        {
            "event": "coverage_hysteresis_update",
            "section_idx": 0,
            "retained_memory_frames": [0, 76],
            "rolling_only_frame": 76,
        },
        {
            "event": "coverage_hysteresis_admission",
            "section_idx": 1,
            "candidate_memory_frame": 77,
            "hysteresis_admitted": False,
            "hysteresis_nearest_reference_frame": 0,
            "hysteresis_max_view_similarity": 0.96,
        },
        {
            "event": "coverage_hysteresis_admission",
            "section_idx": 1,
            "candidate_memory_frame": 100,
            "hysteresis_admitted": True,
            "hysteresis_nearest_reference_frame": 0,
            "hysteresis_max_view_similarity": 0.40,
        },
        {
            "event": "memory_eviction",
            "section_idx": 1,
            "evicted_memory_frame": 100,
            "eviction_score": 0.1,
            "eviction_slamri_slam_norm": 0.08,
            "eviction_slamri_ri_norm": 0.16,
        },
        {
            "event": "coverage_hysteresis_update",
            "section_idx": 1,
            "retained_memory_frames": [0, 152],
            "rolling_only_frame": 152,
        },
    ]


class CoverageHysteresisVisualizationTest(unittest.TestCase):
    def test_reconstruct_section_removes_previous_transient_anchor(self):
        snapshot = MODULE.reconstruct_section(synthetic_events(), 1)

        self.assertEqual(snapshot["current_memory"], [0])
        self.assertEqual(snapshot["admitted"], [100])
        self.assertEqual(snapshot["rejected"], [77])
        self.assertEqual(snapshot["prospective"], [0, 100])
        self.assertEqual(snapshot["evicted"], [100])
        self.assertEqual(snapshot["rolling_only_frame"], 152)

    def test_duplicate_selection_prefers_strong_matches_and_diverse_sections(self):
        events = synthetic_events() + [
            {
                "event": "coverage_hysteresis_admission",
                "section_idx": 2,
                "candidate_memory_frame": 160,
                "hysteresis_admitted": False,
                "hysteresis_nearest_reference_frame": 10,
                "hysteresis_max_view_similarity": 0.99,
            },
            {
                "event": "coverage_hysteresis_admission",
                "section_idx": 2,
                "candidate_memory_frame": 161,
                "hysteresis_admitted": False,
                "hysteresis_nearest_reference_frame": 11,
                "hysteresis_max_view_similarity": 0.98,
            },
        ]

        selected = MODULE.select_duplicate_rows(events, limit=2)

        self.assertEqual(
            [row["candidate_memory_frame"] for row in selected],
            [160, 77],
        )

    def test_eviction_selection_uses_lowest_score(self):
        events = synthetic_events() + [
            {
                "event": "memory_eviction",
                "section_idx": 2,
                "evicted_memory_frame": 140,
                "eviction_score": 0.04,
                "eviction_slamri_slam_norm": 0.02,
                "eviction_slamri_ri_norm": 0.10,
            }
        ]

        selected = MODULE.select_eviction_rows(events, limit=1)

        self.assertEqual(selected[0]["evicted_memory_frame"], 140)

    def test_pca_2d_is_stable_for_small_inputs(self):
        one = MODULE.pca_2d(np.array([[1.0, 2.0, 3.0]]))
        many = MODULE.pca_2d(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

        self.assertEqual(one.shape, (1, 2))
        self.assertEqual(many.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(many)))

    def test_frame_status_prioritizes_eviction_over_admission(self):
        snapshot = {
            "rejected": [],
            "evicted": [10],
            "rolling_only_frame": None,
            "admitted": [10],
        }

        self.assertEqual(MODULE.frame_status(10, snapshot), "evicted")


if __name__ == "__main__":
    unittest.main()
