import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def make_line_c2ws(num_frames):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    c2ws[:, 0, 3] = np.arange(num_frames, dtype=np.float64)
    return c2ws


def rgb_features(values):
    return {
        frame_idx: np.full(12, value, dtype=np.float32)
        for frame_idx, value in values.items()
    }


class DensityBalancedViewCoverageTest(unittest.TestCase):
    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(
                policy="density_balanced_view_coverage"
            )

    def test_selects_distinct_region_and_conserves_mass(self):
        candidates = [0, 1, 6]
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            6: np.array([0.0, 1.0], dtype=np.float32),
        }
        masses = {0: 10.0, 1: 2.0, 6: 3.0}

        scores, details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=candidates,
                c2ws=make_line_c2ws(7),
                budget=2,
                forced_keep_frames={0},
                dino_features=dino,
                rgb_features=rgb_features({0: 0.0, 1: 0.0, 6: 1.0}),
                coverage_masses=masses,
                radius=2.0,
                return_details=True,
            )
        )

        selected = {
            frame_idx
            for frame_idx, row in details.items()
            if row["density_coverage_selected"]
        }
        self.assertEqual(selected, {0, 6})
        self.assertEqual(scores[0], float("inf"))
        self.assertLess(scores[1], 0.0)
        self.assertGreater(scores[6], 0.0)
        self.assertAlmostEqual(
            details[0]["density_coverage_assigned_mass"], 12.0
        )
        self.assertAlmostEqual(
            details[6]["density_coverage_assigned_mass"], 3.0
        )
        self.assertAlmostEqual(
            sum(
                details[index]["density_coverage_assigned_mass"]
                for index in selected
            ),
            sum(masses.values()),
        )

        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="density_balanced_view_coverage",
            budget=2,
            pinned_frames={0},
        )
        evicted = memory.update(
            candidates,
            eviction_scores=scores,
            protected_frames={6},
        )
        self.assertEqual(memory.candidates(), [0, 6])
        self.assertEqual(evicted, [1])

    def test_density_balance_upweights_sparse_view(self):
        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        _, details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=[0, 1, 2],
                c2ws=c2ws,
                budget=2,
                forced_keep_frames={0},
                dino_features=dino,
                rgb_features=rgb_features({0: 0.0, 1: 0.0, 2: 1.0}),
                dino_weight=4.0,
                rgb_weight=4.0,
                return_details=True,
            )
        )

        self.assertGreater(
            details[2]["density_coverage_demand_weight"],
            details[0]["density_coverage_demand_weight"],
        )
        selected = {
            frame_idx
            for frame_idx, row in details.items()
            if row["density_coverage_selected"]
        }
        self.assertEqual(selected, {0, 2})

    def test_alpha_zero_disables_density_reweighting(self):
        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        _, details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=[0, 1, 2],
                c2ws=c2ws,
                budget=2,
                dino_features=dino,
                rgb_features=rgb_features({0: 0.0, 1: 0.0, 2: 1.0}),
                density_alpha=0.0,
                return_details=True,
            )
        )

        for frame_idx in [0, 1, 2]:
            self.assertAlmostEqual(
                details[frame_idx]["density_coverage_demand_weight"],
                1.0 / 3.0,
            )

    def test_no_compression_preserves_each_mass(self):
        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
        }
        masses = {0: 4.0, 1: 7.0}

        _, details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=[0, 1],
                c2ws=c2ws,
                budget=2,
                forced_keep_frames={0},
                dino_features=dino,
                rgb_features=rgb_features({0: 0.0, 1: 0.0}),
                coverage_masses=masses,
                return_details=True,
            )
        )
        self.assertAlmostEqual(
            details[0]["density_coverage_assigned_mass"], 4.0
        )
        self.assertAlmostEqual(
            details[1]["density_coverage_assigned_mass"], 7.0
        )
        self.assertGreaterEqual(details[0]["density_coverage_value"], 0.0)
        self.assertLessEqual(details[0]["density_coverage_value"], 1.0)

    def test_represented_mass_survives_next_online_update(self):
        c2ws = make_line_c2ws(7)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            6: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 0.0, 6: 1.0})
        _, first_details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=[0, 1],
                c2ws=c2ws,
                budget=1,
                forced_keep_frames={0},
                dino_features=dino,
                rgb_features=rgb,
                return_details=True,
            )
        )
        retained_mass = first_details[0]["density_coverage_assigned_mass"]

        _, second_details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                memory_frame_indices=[0, 6],
                c2ws=c2ws,
                budget=2,
                forced_keep_frames={0},
                dino_features=dino,
                rgb_features=rgb,
                coverage_masses={0: retained_mass, 6: 1.0},
                radius=2.0,
                return_details=True,
            )
        )
        self.assertAlmostEqual(
            sum(
                row["density_coverage_assigned_mass"]
                for row in second_details.values()
            ),
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
