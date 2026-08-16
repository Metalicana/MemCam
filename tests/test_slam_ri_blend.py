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
    c2ws[:, 0, 3] = np.array(positions, dtype=np.float64)
    return c2ws


def rgb_features(values):
    return {
        frame_idx: np.full(12, value, dtype=np.float32)
        for frame_idx, value in values.items()
    }


class SlamRiBlendTest(unittest.TestCase):
    def setUp(self):
        # a1, a2: near-identical room-A views; b: distinct room-B view --
        # same counterexample layout used in test_mce.py.
        self.candidates = [0, 1, 2]
        self.c2ws = make_line_c2ws([0.0, 0.1, 20.0])
        self.dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.98, 0.02], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        self.rgb = rgb_features({0: 0.0, 1: 0.02, 2: 1.0})

    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="slam_ri_blend")

    def test_rejects_beta_out_of_range(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.compute_slam_ri_blend_scores(
                memory_frame_indices=self.candidates,
                c2ws=self.c2ws,
                dino_features=self.dino,
                rgb_features=self.rgb,
                beta=1.5,
            )

    def test_beta_one_reproduces_slam_ranking(self):
        # beta=1 zeroes out RI's contribution entirely, so the blend's
        # ranking must exactly match slam_covisibility's own ranking --
        # min-max normalization is monotonic and doesn't reorder anything.
        blend_scores = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=1.0,
        )
        slam_scores = MEMORY_POLICIES.compute_slam_covisibility_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
        )
        blend_order = sorted(self.candidates, key=lambda idx: blend_scores[idx])
        slam_order = sorted(self.candidates, key=lambda idx: slam_scores[idx])
        self.assertEqual(blend_order, slam_order)

    def test_beta_zero_reproduces_ri_ranking(self):
        blend_scores = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=0.0,
            ri_kwargs={"rarity_neighbors": 1},
        )
        ri_scores = MEMORY_POLICIES.compute_rarity_irreplaceability_scores(
            memory_frame_indices=self.candidates,
            dino_features=self.dino,
            rgb_features=self.rgb,
            rarity_neighbors=1,
        )
        blend_order = sorted(self.candidates, key=lambda idx: blend_scores[idx])
        ri_order = sorted(self.candidates, key=lambda idx: ri_scores[idx])
        self.assertEqual(blend_order, ri_order)

    def test_duplicate_pair_ties_and_one_is_evicted_before_the_distinct_view(self):
        # 0 and 1 are mutual nearest neighbors of each other in both RGB
        # (RI's irreplaceability) and covisibility (SLAM's redundancy), so
        # this symmetric layout gives them an *exact* tie on both raw
        # scores -- that's a genuine property of the underlying scores, not
        # a blend artifact. Ties are broken by insertion order in
        # FrameMemoryBuffer, so frame 0 (inserted first) is evicted first;
        # what matters here is that the distinct view (2) always survives
        # and only a duplicate is ever sacrificed.
        scores = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=0.5,
            ri_kwargs={"rarity_neighbors": 1},
        )
        self.assertEqual(scores[0], scores[1])
        self.assertLess(scores[0], scores[2])
        self.assertLess(scores[1], scores[2])

        memory = MEMORY_POLICIES.FrameMemoryBuffer(policy="slam_ri_blend", budget=2)
        evicted = memory.update(self.candidates, eviction_scores=scores)
        self.assertEqual(evicted, [0])
        self.assertEqual(memory.candidates(), [1, 2])

    def test_forced_frames_are_never_evicted(self):
        scores, details = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=self.candidates,
            c2ws=self.c2ws,
            forced_keep_frames={1},
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=0.5,
            ri_kwargs={"rarity_neighbors": 1},
            return_details=True,
        )
        self.assertEqual(scores[1], float("inf"))
        self.assertTrue(details[1]["slamri_forced_keep"])

        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="slam_ri_blend", budget=1, pinned_frames={1}
        )
        evicted = memory.update(
            self.candidates,
            eviction_scores=scores,
            protected_frames={1},
        )
        self.assertEqual(memory.candidates(), [1])
        self.assertEqual(set(evicted), {0, 2})

    def test_beta_interpolates_between_extremes(self):
        # Frame 2 (distinct) should outscore frame 1 (near-duplicate) across
        # the whole beta range, since both SLAM and RI independently value
        # it over the redundant view.
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            scores = MEMORY_POLICIES.compute_slam_ri_blend_scores(
                memory_frame_indices=self.candidates,
                c2ws=self.c2ws,
                dino_features=self.dino,
                rgb_features=self.rgb,
                beta=beta,
                ri_kwargs={"rarity_neighbors": 1},
            )
            self.assertGreater(scores[2], scores[1])


if __name__ == "__main__":
    unittest.main()
