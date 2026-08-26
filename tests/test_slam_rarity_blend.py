import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def make_line_c2ws(positions):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    c2ws[:, 0, 3] = np.asarray(positions, dtype=np.float64)
    return c2ws


class SlamRarityBlendTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [0, 1, 2]
        self.c2ws = make_line_c2ws([0.0, 0.1, 20.0])
        self.dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        self.rarity_kwargs = {"cluster_distance_threshold": 0.05}

    def test_both_policies_require_a_budget(self):
        for policy in ("rarity_only", "slam_rarity_blend"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    MEMORY_POLICIES.FrameMemoryBuffer(policy=policy)

    def test_rarity_only_uses_inverse_cluster_density(self):
        scores, details = MEMORY_POLICIES.compute_rarity_scores(
            memory_frame_indices=self.candidates,
            dino_features=self.dino,
            return_details=True,
            **self.rarity_kwargs,
        )
        self.assertEqual(details[0]["cluster_size"], 2)
        self.assertEqual(details[1]["cluster_size"], 2)
        self.assertEqual(details[2]["cluster_size"], 1)
        self.assertEqual(scores[0], scores[1])
        self.assertGreater(scores[2], scores[0])

    def test_rarity_matches_the_rarity_term_inside_ri(self):
        rgb = {
            0: np.zeros(12, dtype=np.float32),
            1: np.full(12, 0.1, dtype=np.float32),
            2: np.ones(12, dtype=np.float32),
        }
        rarity_scores = MEMORY_POLICIES.compute_rarity_scores(
            memory_frame_indices=self.candidates,
            dino_features=self.dino,
            **self.rarity_kwargs,
        )
        _, ri_details = MEMORY_POLICIES.compute_rarity_irreplaceability_scores(
            memory_frame_indices=self.candidates,
            dino_features=self.dino,
            rgb_features=rgb,
            return_details=True,
            **self.rarity_kwargs,
        )
        for frame_idx in self.candidates:
            self.assertAlmostEqual(
                rarity_scores[frame_idx],
                ri_details[frame_idx]["rarity"],
            )

    def test_blend_endpoints_reproduce_parent_rankings(self):
        slam_scores = MEMORY_POLICIES.compute_slam_covisibility_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
        )
        rarity_scores = MEMORY_POLICIES.compute_rarity_scores(
            memory_frame_indices=self.candidates,
            dino_features=self.dino,
            **self.rarity_kwargs,
        )
        geometric_endpoint = MEMORY_POLICIES.compute_slam_rarity_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            slam_weight=1.0,
            rarity_kwargs=self.rarity_kwargs,
        )
        rarity_endpoint = MEMORY_POLICIES.compute_slam_rarity_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            slam_weight=0.0,
            rarity_kwargs=self.rarity_kwargs,
        )
        ranking = lambda scores: sorted(self.candidates, key=lambda idx: scores[idx])
        self.assertEqual(ranking(geometric_endpoint), ranking(slam_scores))
        self.assertEqual(ranking(rarity_endpoint), ranking(rarity_scores))

    def test_default_blend_is_exactly_75_percent_geometric(self):
        scores, details = MEMORY_POLICIES.compute_slam_rarity_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            forced_keep_frames={0},
            rarity_kwargs=self.rarity_kwargs,
            return_details=True,
        )
        self.assertEqual(scores[0], float("inf"))
        self.assertTrue(details[0]["slamrarity_forced_keep"])
        self.assertEqual(details[1]["slamrarity_slam_weight"], 0.75)
        self.assertEqual(details[1]["slamrarity_rarity_weight"], 0.25)
        expected = (
            0.75 * details[1]["slamrarity_slam_norm"]
            + 0.25 * details[1]["slamrarity_rarity_norm"]
        )
        self.assertAlmostEqual(scores[1], expected)

    def test_rejects_invalid_slam_weight(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.compute_slam_rarity_blend_scores(
                memory_frame_indices=self.candidates,
                c2ws=self.c2ws,
                dino_features=self.dino,
                slam_weight=1.01,
            )


if __name__ == "__main__":
    unittest.main()
