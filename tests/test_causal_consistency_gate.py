import unittest

import numpy as np

from utils.calibrate_causal_consistency_gate import (
    add_pose_distance,
    expected_value,
    fit_binned_expectation,
    gate_decision,
    pose_components,
    sample_context_pairs,
)


class CausalConsistencyGateTests(unittest.TestCase):
    def test_sample_context_pairs_uses_target_stride(self):
        selected = {
            76: {"target_frame": 76},
            77: {"target_frame": 77},
            152: {"target_frame": 152},
        }
        sampled = sample_context_pairs(selected, frame_stride=76)
        self.assertEqual([row["target_frame"] for row in sampled], [76, 152])

    def test_pose_components_separates_translation_and_rotation(self):
        c2ws = np.repeat(np.eye(4)[None], 2, axis=0)
        c2ws[1, 0, 3] = 3.0
        angle = np.pi / 2.0
        c2ws[1, :3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        values = pose_components(c2ws, 1, 0)
        self.assertAlmostEqual(values["translation"], 3.0)
        self.assertAlmostEqual(values["rotation_rad"], np.pi / 2.0)

    def test_pose_distance_reuses_training_scales(self):
        train = [
            {"x_translation": 1.0, "x_rotation_rad": 0.5},
            {"x_translation": 3.0, "x_rotation_rad": 1.5},
        ]
        scales = add_pose_distance(train, "x")
        held_out = [{"x_translation": 2.0, "x_rotation_rad": 1.0}]
        add_pose_distance(held_out, "x", scales)
        self.assertEqual(scales, {"translation": 2.0, "rotation_rad": 1.0})
        self.assertAlmostEqual(held_out[0]["x_pose_distance"], 2.0)

    def test_binned_expectation_uses_fitted_bin_means(self):
        rows = [
            {"distance": 0.0, "similarity": 0.9},
            {"distance": 0.1, "similarity": 0.7},
            {"distance": 1.0, "similarity": 0.3},
            {"distance": 1.1, "similarity": 0.1},
        ]
        model = fit_binned_expectation(rows, "distance", "similarity", bins=2)
        self.assertAlmostEqual(expected_value(model, 0.0), 0.8)
        self.assertAlmostEqual(expected_value(model, 2.0), 0.2)

    def test_gate_decision_requires_every_predeclared_check(self):
        class Args:
            min_auc = 0.70
            min_bad_precision = 0.50
            min_bad_recall = 0.20
            max_test_clean_false_reject = 0.15
            min_pose_auc_gain = 0.02
            min_bad_parent_auc = 0.60

        summary = {
            "estimator": "context_pose_residual",
            "test_quality_auc": 0.75,
            "gate_test_bad_precision": 0.60,
            "gate_test_bad_recall": 0.30,
            "gate_test_clean_false_reject_rate": 0.10,
        }
        raw = {"test_quality_auc": 0.70}
        strata = {"low_fidelity_parent": {"quality_auc": 0.65}}
        self.assertEqual(
            gate_decision(summary, raw, strata, Args())["decision"], "INJECT"
        )
        summary["gate_test_bad_precision"] = 0.40
        decision = gate_decision(summary, raw, strata, Args())
        self.assertEqual(decision["decision"], "DO_NOT_INJECT")
        self.assertIn("bad_precision", decision["failed_checks"])


if __name__ == "__main__":
    unittest.main()
