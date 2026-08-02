import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_surprise_forcing",
    REPO_ROOT / "utils" / "summarize_surprise_forcing.py",
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


class SurpriseForcingScoreTest(unittest.TestCase):
    def test_dual_component_score_matches_equation_six(self):
        score = MEMORY_POLICIES.surprise_forcing_score(
            candidate_descriptor=np.array([1.0, 0.0]),
            bank_descriptors=[
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
            ],
            alpha=0.7,
        )

        self.assertAlmostEqual(score["mean_similarity"], 0.5)
        self.assertAlmostEqual(score["max_similarity"], 1.0)
        self.assertAlmostEqual(score["prediction_surprise"], 0.25)
        self.assertAlmostEqual(score["novelty_surprise"], 0.0)
        self.assertAlmostEqual(score["surprise"], 0.175)

    def test_empty_bank_is_maximally_surprising(self):
        score = MEMORY_POLICIES.surprise_forcing_score(
            np.array([2.0, 0.0]),
            [],
        )
        self.assertEqual(score["surprise"], 1.0)


class SurpriseForcingControllerTest(unittest.TestCase):
    def test_policy_requires_an_explicit_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="surprise_forcing")

    def test_warmup_bypasses_gate_then_feedback_updates_threshold(self):
        controller = MEMORY_POLICIES.SurpriseForcingMemoryController(
            capacity=2,
            warmup_sections=1,
        )
        warmup = controller.consider(
            frame_idx=1,
            descriptor=np.array([1.0, 0.0]),
            section_idx=0,
        )
        after_warmup = controller.consider(
            frame_idx=2,
            descriptor=np.array([1.0, 0.0]),
            section_idx=1,
        )

        self.assertTrue(warmup["gate_pass"])
        self.assertTrue(warmup["committed"])
        self.assertAlmostEqual(warmup["threshold_after"], 0.002)
        self.assertFalse(after_warmup["gate_pass"])
        self.assertEqual(after_warmup["rejection_reason"], "surprise_gate")
        self.assertAlmostEqual(after_warmup["threshold_after"], -0.028)

    def test_full_bank_replaces_only_lower_priority_entry(self):
        controller = MEMORY_POLICIES.SurpriseForcingMemoryController(
            capacity=2,
            warmup_sections=99,
        )
        controller.consider(1, np.array([1.0, 0.0]), section_idx=0)
        controller.consider(2, np.array([1.0, 0.0]), section_idx=0)

        novel = controller.consider(
            3,
            np.array([0.0, 1.0]),
            section_idx=0,
        )
        duplicate = controller.consider(
            4,
            np.array([0.0, 1.0]),
            section_idx=0,
        )

        self.assertTrue(novel["committed"])
        self.assertEqual(novel["evicted_frame"], 2)
        self.assertFalse(duplicate["committed"])
        self.assertEqual(duplicate["rejection_reason"], "priority")
        self.assertEqual(controller.frames(), [1, 3])

    def test_cosine_routing_updates_usage(self):
        controller = MEMORY_POLICIES.SurpriseForcingMemoryController(
            capacity=3,
            warmup_sections=99,
        )
        controller.consider(1, np.array([1.0, 0.0]), section_idx=0)
        controller.consider(2, np.array([0.0, 1.0]), section_idx=0)
        controller.consider(3, np.array([-1.0, 0.0]), section_idx=0)

        routed, similarities = controller.route(
            query_descriptor=np.array([1.0, 0.0]),
            top_k=2,
        )
        state = controller.state_snapshot(current_frame=4)

        self.assertEqual(routed, [1, 2])
        self.assertAlmostEqual(similarities[1], 1.0)
        self.assertAlmostEqual(similarities[3], -1.0)
        self.assertEqual(state[1]["usage"], 1)
        self.assertEqual(state[2]["usage"], 1)
        self.assertEqual(state[3]["usage"], 0)

    def test_shortlisting_can_defer_usage_until_actual_retrieval(self):
        controller = MEMORY_POLICIES.SurpriseForcingMemoryController(
            capacity=2,
            warmup_sections=99,
        )
        controller.consider(1, np.array([1.0, 0.0]), section_idx=0)
        controller.consider(2, np.array([0.0, 1.0]), section_idx=0)

        routed, _ = controller.route(
            query_descriptor=np.array([1.0, 0.0]),
            top_k=2,
            record_usage=False,
        )
        controller.record_usage([routed[0], routed[0]])
        state = controller.state_snapshot(current_frame=3)

        self.assertEqual(state[routed[0]]["usage"], 1)
        self.assertEqual(state[routed[1]]["usage"], 0)


class SurpriseForcingSummaryTest(unittest.TestCase):
    def test_controller_rates_are_computed_from_counts(self):
        row = {
            "scene": "scene-a",
            "evaluated": 10,
            "warmup_evaluated": 4,
            "gate_passes": 7,
            "non_warmup_gate_passes": 3,
            "commits": 5,
            "replacements": 2,
            "gate_rejections": 3,
            "priority_rejections": 2,
            "route_events": 4,
            "surprise_mean": 0.2,
            "prediction_surprise_mean": 0.25,
            "novelty_surprise_mean": 0.1,
            "final_threshold": 0.5,
            "final_bank_size": 31,
            "routed_frames_mean": 3.0,
            "actually_retrieved_frames_mean": 2.0,
        }

        summary = SUMMARY.build_summary([row], expected_videos=1)
        controller = summary["controller"]

        self.assertAlmostEqual(controller["gate_pass_rate"], 0.7)
        self.assertAlmostEqual(controller["non_warmup_gate_pass_rate"], 0.5)
        self.assertAlmostEqual(controller["commit_rate"], 0.5)
        self.assertAlmostEqual(controller["replacement_rate_per_commit"], 0.4)


if __name__ == "__main__":
    unittest.main()
