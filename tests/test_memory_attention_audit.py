import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:
    torch = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = load_module(
    "summarize_attention_utility_pilot",
    REPO_ROOT / "utils" / "summarize_attention_utility_pilot.py",
)
AUDIT = (
    load_module(
        "memory_attention_audit",
        REPO_ROOT / "diffsynth" / "pipelines" / "memory_attention_audit.py",
    )
    if torch is not None
    else None
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionRecoveryTest(unittest.TestCase):
    def test_target_value_descriptors_pool_space_per_latent_frame(self):
        collector = AUDIT.TargetValueDescriptorCollector(
            context_token_count=2,
            target_length=2,
            target_spatial=2,
        )
        value_tokens = torch.tensor(
            [
                [
                    [9.0, 9.0],
                    [9.0, 9.0],
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 4.0],
                ]
            ]
        )

        collector.capture(value_tokens)

        self.assertEqual(collector.descriptors.shape, (2, 2))
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(collector.descriptors),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            )
        )

    def test_uniform_attention_has_expected_context_mass(self):
        q = torch.zeros(1, 6, 4)
        k = torch.zeros_like(q)
        result = AUDIT.sample_target_to_context_attention(
            q,
            k,
            num_heads=2,
            context_spatial=1,
            context_frame_indices=[10, 20],
            query_count=4,
            query_chunk_size=2,
        )

        self.assertAlmostEqual(result["context_attention_mass"], 2.0 / 6.0)
        self.assertAlmostEqual(result["slot_attention"][0], 1.0 / 6.0)
        self.assertAlmostEqual(result["slot_attention"][1], 1.0 / 6.0)

    def test_attention_reduces_duplicate_slots_to_memory_item(self):
        q = torch.zeros(1, 7, 4)
        k = torch.zeros_like(q)
        result = AUDIT.sample_target_to_context_attention(
            q,
            k,
            num_heads=1,
            context_spatial=1,
            context_frame_indices=[10, 10, 20],
            query_count=4,
        )
        scores = {
            row["memory_frame"]: row for row in result["memory_scores"]
        }

        self.assertEqual(scores[10]["slot_count"], 2)
        self.assertAlmostEqual(scores[10]["attention_total"], 2.0 / 7.0)
        self.assertAlmostEqual(scores[20]["attention_total"], 1.0 / 7.0)

    def test_target_queries_identify_preferred_context_key(self):
        q = torch.zeros(1, 6, 4)
        k = torch.zeros_like(q)
        q[:, 2:, 0] = 1.0
        k[:, 0, 0] = 10.0
        k[:, 1, 0] = -10.0
        result = AUDIT.sample_target_to_context_attention(
            q,
            k,
            num_heads=1,
            context_spatial=1,
            context_frame_indices=[10, 20],
            query_count=4,
        )
        scores = {
            row["memory_frame"]: row["attention_total"]
            for row in result["memory_scores"]
        }
        self.assertGreater(scores[10], scores[20])

    def test_intervention_roles_are_distinct(self):
        rows = [
            {"memory_frame": frame, "attention_total": score}
            for frame, score in [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]
        ]
        selected = AUDIT.select_intervention_candidates(rows, seed=7)

        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["memory_frame"] for row in selected}), 3)
        self.assertEqual(selected[0]["intervention_role"], "top_attention")
        self.assertEqual(selected[1]["intervention_role"], "bottom_attention")


class AttentionSummaryTest(unittest.TestCase):
    def test_summary_compares_attention_to_retrieval_controls(self):
        rows = []
        for group in range(4):
            for role, attention, effect, slots in [
                ("bottom_attention", 0.1, 0.01, 2),
                ("random_attention", 0.5, 0.05, 1),
                ("top_attention", 0.9, 0.09, 3),
            ]:
                rows.append(
                    {
                        "source_file": f"video_{group}.jsonl",
                        "row": group,
                        "section_idx": 3,
                        "progress_id": 5,
                        "intervention_role": role,
                        "attention_total": attention,
                        "attention_per_slot": attention / slots,
                        "slot_count": slots,
                        "retrieval_overlap_mean": 0.5,
                        "memory_age_mean": 10.0,
                        "prediction_relative_l2": effect,
                        "prediction_delta_mse": effect * effect,
                    }
                )

        summary = SUMMARY.build_summary(rows, ["a", "b", "c", "d"], [])

        self.assertAlmostEqual(
            summary["predictor_correlations"]["attention_total"][
                "global_spearman"
            ],
            1.0,
        )
        self.assertEqual(summary["role_comparison"]["top_bottom_pairs"], 4)
        self.assertAlmostEqual(
            summary["role_comparison"]["top_effect_greater_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
