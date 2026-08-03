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


class SlamMaxCoverageTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [0, 1, 2, 7]
        self.c2ws = make_line_c2ws(8)
        self.dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.99, 0.01], dtype=np.float32),
            2: np.array([0.98, 0.02], dtype=np.float32),
            7: np.array([0.0, 1.0], dtype=np.float32),
        }

    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="slam_max_coverage")

    def test_generic_optimizer_accepts_rectangular_affinity(self):
        affinity = np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.7, 0.6, 0.0],
            ],
            dtype=np.float64,
        )
        result = MEMORY_POLICIES.greedy_max_coverage_selection(
            affinity=affinity,
            budget=2,
            forced_candidate_indices=[0],
        )

        self.assertEqual(result["selected_cols"], [0, 2])
        expected = np.mean(np.max(affinity[:, [0, 2]], axis=1))
        self.assertAlmostEqual(result["coverage_value"], expected)

    def test_affinity_matches_slam_components_off_diagonal(self):
        affinity = MEMORY_POLICIES._slam_covisibility_affinity(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            self_similarity=1.0,
        )
        pose_similarity = np.exp(
            -MEMORY_POLICIES.pose_distances(
                self.c2ws,
                self.candidates,
                self.candidates,
            )
        )
        dino_similarity = np.maximum(
            MEMORY_POLICIES._feature_cosine_similarity(
                self.candidates,
                self.dino,
            ),
            0.0,
        )
        expected = 0.65 * pose_similarity + 0.35 * dino_similarity
        np.fill_diagonal(expected, 1.0)

        np.testing.assert_allclose(affinity, expected, atol=1e-12)

        slam_affinity = MEMORY_POLICIES._slam_covisibility_affinity(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            self_similarity=0.0,
        )
        mask = ~np.eye(len(self.candidates), dtype=bool)
        np.testing.assert_allclose(affinity[mask], slam_affinity[mask])
        np.testing.assert_allclose(np.diag(slam_affinity), 0.0)

    def test_max_coverage_selects_a_distinct_view(self):
        scores, details = (
            MEMORY_POLICIES.compute_slam_max_coverage_scores(
                memory_frame_indices=self.candidates,
                c2ws=self.c2ws,
                budget=2,
                forced_keep_frames={0},
                dino_features=self.dino,
                return_details=True,
            )
        )
        selected = {
            frame_idx
            for frame_idx, row in details.items()
            if row["slam_max_coverage_selected"]
        }

        self.assertEqual(selected, {0, 7})
        self.assertEqual(scores[0], float("inf"))
        self.assertGreater(scores[7], 0.0)
        self.assertLess(scores[1], 0.0)
        self.assertLess(scores[2], 0.0)

        affinity = MEMORY_POLICIES._slam_covisibility_affinity(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            self_similarity=1.0,
        )
        selected_cols = [self.candidates.index(frame_idx) for frame_idx in selected]
        expected_value = float(np.mean(np.max(affinity[:, selected_cols], axis=1)))
        self.assertAlmostEqual(
            details[0]["slam_max_coverage_value"],
            expected_value,
        )

        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="slam_max_coverage",
            budget=2,
            pinned_frames={0},
        )
        evicted = memory.update(
            self.candidates,
            eviction_scores=scores,
            protected_frames={7},
        )
        self.assertEqual(memory.candidates(), [0, 7])
        self.assertEqual(evicted, [1, 2])

    def test_rejects_more_forced_frames_than_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.compute_slam_max_coverage_scores(
                memory_frame_indices=self.candidates,
                c2ws=self.c2ws,
                budget=1,
                forced_keep_frames={0, 7},
                dino_features=self.dino,
            )

    def test_does_not_read_unseen_future_poses(self):
        changed_future = self.c2ws.copy()
        changed_future[3:, :3, 3] += 10000.0
        causal_candidates = [0, 1, 2]
        causal_dino = {
            frame_idx: self.dino[frame_idx] for frame_idx in causal_candidates
        }

        scores = MEMORY_POLICIES.compute_slam_max_coverage_scores(
            memory_frame_indices=causal_candidates,
            c2ws=self.c2ws,
            budget=2,
            forced_keep_frames={0},
            dino_features=causal_dino,
        )
        changed_scores = MEMORY_POLICIES.compute_slam_max_coverage_scores(
            memory_frame_indices=causal_candidates,
            c2ws=changed_future,
            budget=2,
            forced_keep_frames={0},
            dino_features=causal_dino,
        )
        self.assertEqual(scores, changed_scores)


if __name__ == "__main__":
    unittest.main()
