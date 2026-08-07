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


class MarginalCoverageEvictionTest(unittest.TestCase):
    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="mce")

    def test_paper_table1_duplicate_view_counterexample(self):
        # a1, a2: near-identical room-A views. b: distinct room-B view.
        # Budget 2 should keep one room-A view plus b, not both room-A views.
        c2ws = make_line_c2ws([0.0, 0.1, 20.0])
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.98, 0.02], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 0.02, 2: 1.0})

        scores, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=2,
            dino_features=dino,
            rgb_features=rgb,
            radius=5.0,
            return_details=True,
        )

        selected = {frame_idx for frame_idx, row in details.items() if row["mce_selected"]}
        self.assertEqual(selected, {0, 2})
        self.assertLess(scores[1], scores[0])
        self.assertLess(scores[1], scores[2])

        memory = MEMORY_POLICIES.FrameMemoryBuffer(policy="mce", budget=2)
        evicted = memory.update([0, 1, 2], eviction_scores=scores)
        self.assertEqual(memory.candidates(), [0, 2])
        self.assertEqual(evicted, [1])

    def test_kernel_is_additive_not_multiplicative(self):
        # Two candidates: identical geometry, orthogonal appearance. Under the
        # paper's additive kernel alpha*K_geo + (1-alpha)*K_vis, K_geo=1 for
        # both means K(q,m) stays well above zero even though K_vis=0 --
        # unlike a multiplicative combination, where it would collapse to 0.
        c2ws = make_line_c2ws([0.0, 0.0])
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 1.0})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            c2ws=c2ws,
            budget=2,
            dino_features=dino,
            rgb_features=rgb,
            alpha=0.65,
            radius=5.0,
            return_details=True,
        )
        # Both survive at budget == candidate count; check the coverage value
        # reflects a nonzero geometry contribution even under zero visual
        # agreement (K_vis(q=0, m=1) calibrates orthogonal cosine to 0.5, and
        # K_geo(q=0, m=1) = 1 for identical poses, so K(q=0,m=1) = 0.65*1 +
        # 0.35*0.5 = 0.825, far from the ~0 a product kernel would give).
        self.assertGreater(details[0]["mce_coverage_value"], 0.7)

    def test_forced_frames_are_never_evicted(self):
        c2ws = make_line_c2ws([0.0, 0.05, 20.0])
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 0.0, 2: 1.0})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=1,
            forced_keep_frames={1},
            dino_features=dino,
            rgb_features=rgb,
            radius=5.0,
            return_details=True,
        )
        self.assertTrue(details[1]["mce_selected"])
        self.assertTrue(details[1]["mce_forced_keep"])
        self.assertEqual(details[1]["score"], float("inf"))

    def test_future_query_biases_selection_toward_reachable_view(self):
        # Two candidates far apart and visually distinct -- Q_hist alone has
        # no reason to prefer either. A future control query near frame 5
        # should make it survive over frame 0 at budget 1.
        c2ws = make_line_c2ws(list(range(10)))
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            5: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 5: 1.0})

        scores, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 5],
            c2ws=c2ws,
            budget=1,
            dino_features=dino,
            rgb_features=rgb,
            future_query_frame_indices=[6, 7, 8],
            lambda_hist=0.2,
            radius=2.0,
            return_details=True,
        )
        self.assertTrue(details[5]["mce_selected"])
        self.assertFalse(details[0]["mce_selected"])
        self.assertGreater(details[5]["mce_num_ctrl_queries"], 0)

    def test_no_future_queries_uses_lambda_one(self):
        c2ws = make_line_c2ws([0.0, 1.0])
        dino = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        }
        rgb = rgb_features({0: 0.0, 1: 1.0})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            c2ws=c2ws,
            budget=2,
            dino_features=dino,
            rgb_features=rgb,
            radius=5.0,
            return_details=True,
        )
        self.assertEqual(details[0]["mce_lambda"], 1.0)
        self.assertEqual(details[0]["mce_num_ctrl_queries"], 0)


if __name__ == "__main__":
    unittest.main()
