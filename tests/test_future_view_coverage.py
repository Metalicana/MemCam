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


class FutureViewCoverageTest(unittest.TestCase):
    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="future_view_coverage")

    def test_no_future_queries_matches_density_balanced_view_coverage(self):
        c2ws = make_line_c2ws(7)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            6: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 0.0, 6: 1.0})
        masses = {0: 10.0, 1: 2.0, 6: 3.0}
        shared_kwargs = dict(
            memory_frame_indices=[0, 1, 6],
            c2ws=c2ws,
            budget=2,
            forced_keep_frames={0},
            dino_features=dino,
            rgb_features=rgb,
            coverage_masses=masses,
            radius=2.0,
            return_details=True,
        )

        density_scores, density_details = (
            MEMORY_POLICIES.compute_density_balanced_view_coverage_scores(
                **shared_kwargs
            )
        )
        future_scores, future_details = (
            MEMORY_POLICIES.compute_future_view_coverage_scores(
                future_query_frame_indices=None, **shared_kwargs
            )
        )

        self.assertEqual(set(density_scores), set(future_scores))
        for frame_idx in density_scores:
            self.assertAlmostEqual(density_scores[frame_idx], future_scores[frame_idx])
            self.assertAlmostEqual(
                density_details[frame_idx]["density_coverage_assigned_mass"],
                future_details[frame_idx]["future_view_coverage_assigned_mass"],
            )

    def test_future_query_rewards_the_reachable_candidate(self):
        # 0 and 5 are far apart and visually distinct, so a self-covering
        # objective alone cannot prefer one over the other. Only 5 is close
        # to the known future viewpoint at position 6.
        c2ws = make_line_c2ws(10)
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            5: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 5: 1.0})

        _, details = MEMORY_POLICIES.compute_future_view_coverage_scores(
            memory_frame_indices=[0, 5],
            c2ws=c2ws,
            budget=2,
            dino_features=dino,
            rgb_features=rgb,
            future_query_frame_indices=[6],
            future_query_weight=5.0,
            radius=2.0,
            return_details=True,
        )

        self.assertGreater(details[5]["future_view_coverage_future_kernel_mean"], 0.0)
        self.assertEqual(details[0]["future_view_coverage_future_kernel_mean"], 0.0)
        self.assertGreater(
            details[5]["future_view_coverage_removal_loss"],
            details[0]["future_view_coverage_removal_loss"],
        )

    def test_future_query_can_flip_which_candidate_survives_eviction(self):
        # Three visually distinct, geometrically isolated candidates (no
        # self-coverage redundancy between any pair) so, with equal mass,
        # a self-covering objective alone has no basis to prefer one region
        # over another. A future query near candidate 6 should be enough to
        # make it survive a budget=2 eviction over candidate 3.
        c2ws = make_line_c2ws(10)
        dino = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            6: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 3: 0.5, 6: 1.0})

        scores, details = MEMORY_POLICIES.compute_future_view_coverage_scores(
            memory_frame_indices=[0, 3, 6],
            c2ws=c2ws,
            budget=2,
            forced_keep_frames={0},
            dino_features=dino,
            rgb_features=rgb,
            future_query_frame_indices=[7],
            future_query_weight=10.0,
            radius=2.0,
            return_details=True,
        )

        selected = {
            frame_idx
            for frame_idx, row in details.items()
            if row["future_view_coverage_selected"]
        }
        self.assertEqual(selected, {0, 6})
        self.assertLess(scores[3], 0.0)

        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="future_view_coverage",
            budget=2,
            pinned_frames={0},
        )
        evicted = memory.update([0, 3, 6], eviction_scores=scores)
        self.assertEqual(memory.candidates(), [0, 6])
        self.assertEqual(evicted, [3])


if __name__ == "__main__":
    unittest.main()
