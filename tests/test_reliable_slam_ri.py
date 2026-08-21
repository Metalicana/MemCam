import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def make_c2ws(positions):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    c2ws[:, 0, 3] = np.asarray(positions, dtype=np.float64)
    return c2ws


def flat_rgb(value):
    return np.full(48, value, dtype=np.float32)


class ReliableSlamRiTest(unittest.TestCase):
    def setUp(self):
        # Frames 0 and 1 are agreeing historical references. Frame 3 agrees
        # with that revisit, frame 4 is inconsistent, and frame 5 is a novel
        # unsupported view that the gate must not reject.
        self.frames = [0, 1, 2, 3, 4, 5]
        self.references = [0, 1, 2]
        self.admissions = [3, 4, 5]
        self.c2ws = make_c2ws([0.0, 0.1, 20.0, 0.05, 0.06, 40.0])
        self.dino = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.999, 0.001, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            3: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            4: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            5: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        self.rgb = {
            0: flat_rgb(0.0),
            1: flat_rgb(0.01),
            2: flat_rgb(0.60),
            3: flat_rgb(0.0),
            4: flat_rgb(0.80),
            5: flat_rgb(1.0),
        }

    def compute(self, **overrides):
        kwargs = {
            "memory_frame_indices": self.frames,
            "c2ws": self.c2ws,
            "admission_frame_indices": self.admissions,
            "reference_frame_indices": self.references,
            "dino_features": self.dino,
            "rgb_features": self.rgb,
            "slam_weight": 0.75,
            "rarity_neighbors": 1,
            "reliability_neighbors": 2,
            "reliability_min_support": 2,
            "reliability_geometry_threshold": 0.95,
            "reliability_threshold": 0.80,
            "return_details": True,
        }
        kwargs.update(overrides)
        return MEMORY_POLICIES.compute_reliable_slam_ri_scores(**kwargs)

    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="reliable_slam_ri")

    def test_gate_rejects_inconsistent_revisit_only(self):
        _scores, details = self.compute()

        self.assertTrue(details[3]["rsri_reliability_supported"])
        self.assertFalse(details[3]["rsri_gated"])
        self.assertGreaterEqual(details[3]["rsri_reliability"], 0.80)
        self.assertTrue(details[4]["rsri_reliability_supported"])
        self.assertTrue(details[4]["rsri_gated"])
        self.assertLess(details[4]["rsri_reliability"], 0.80)
        self.assertFalse(details[5]["rsri_reliability_supported"])
        self.assertFalse(details[5]["rsri_gated"])

    def test_gate_disabled_exactly_reproduces_existing_blend(self):
        scores, _details = self.compute(reliability_threshold=0.0)
        expected = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=self.frames,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=0.75,
            ri_kwargs={"rarity_neighbors": 1},
        )
        self.assertEqual(scores, expected)

    def test_gated_frame_does_not_affect_ri_or_slam_normalization(self):
        scores, details = self.compute()
        admitted = [frame_idx for frame_idx in self.frames if frame_idx != 4]
        expected = MEMORY_POLICIES.compute_slam_ri_blend_scores(
            memory_frame_indices=admitted,
            c2ws=self.c2ws,
            dino_features=self.dino,
            rgb_features=self.rgb,
            beta=0.75,
            ri_kwargs={"rarity_neighbors": 1},
        )

        self.assertTrue(details[4]["rsri_gated"])
        self.assertIsNone(details[4]["rsri_ri_norm"])
        for frame_idx in admitted:
            self.assertEqual(scores[frame_idx], expected[frame_idx])

    def test_weight_extremes_reproduce_constituent_rankings(self):
        for weight in (0.0, 1.0):
            scores, _details = self.compute(
                slam_weight=weight,
                reliability_threshold=0.0,
            )
            expected = MEMORY_POLICIES.compute_slam_ri_blend_scores(
                memory_frame_indices=self.frames,
                c2ws=self.c2ws,
                dino_features=self.dino,
                rgb_features=self.rgb,
                beta=weight,
                ri_kwargs={"rarity_neighbors": 1},
            )
            self.assertEqual(scores, expected)

    def test_buffer_evicts_gated_candidate(self):
        scores, _details = self.compute()
        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="reliable_slam_ri",
            budget=4,
        )
        evicted = memory.update(self.frames, eviction_scores=scores)

        self.assertIn(4, evicted)
        self.assertNotIn(4, memory.candidates())

    def test_forced_frame_overrides_gate(self):
        scores, details = self.compute(forced_keep_frames={4})
        self.assertEqual(scores[4], float("inf"))
        self.assertFalse(details[4]["rsri_gated"])
        self.assertTrue(details[4]["rsri_forced_keep"])

    def test_support_uses_fixed_camera_overlap(self):
        _scores, details = self.compute()
        self.assertEqual(details[3]["rsri_fov_radius"], 50.0)
        self.assertTrue(details[3]["rsri_reliability_supported"])
        self.assertFalse(details[5]["rsri_reliability_supported"])


if __name__ == "__main__":
    unittest.main()
