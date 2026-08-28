import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "utils" / "analyze_common_source_retention_selection_budget.py"
)
SPEC = importlib.util.spec_from_file_location(
    "common_source_retention_selection_budget",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def context_row(selected_frame, candidate_count):
    return {
        "event": "context_access",
        "selected": True,
        "section_idx": 1,
        "context_slot": 0,
        "target_frame": 77,
        "selected_memory_frame": selected_frame,
        "candidate_count": candidate_count,
        "selected_overlap": 0.9,
    }


class CommonSourceRetentionSelectionBudgetTest(unittest.TestCase):
    def test_describe_run_extracts_family_and_budget(self):
        self.assertEqual(MODULE.describe_run("baseline"), ("Unbounded", None))
        self.assertEqual(MODULE.describe_run("ri_b64_dino_rgb"), ("RI", 64))
        self.assertEqual(
            MODULE.describe_run("slam_b128_covisibility"),
            ("GeoCov", 128),
        )

    def test_common_source_rows_separate_retention_and_selection(self):
        item = {
            "_row": 0,
            "scene": "Synthetic",
            "start_frame": 0,
            "duration_sec": 180,
            "num_frames": 80,
            "fps": 30,
        }
        generated = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (80, 1))
        gt = generated.copy()
        generated[0] = [1.0, 0.0]
        generated[1] = [0.8, 0.6]
        generated[2] = [0.0, 1.0]
        gt[77] = [1.0, 0.0]

        events = {
            "baseline": [context_row(selected_frame=2, candidate_count=73)],
            "ri_b16_dino_rgb": [
                {
                    "event": "memory_eviction",
                    "section_idx": 0,
                    "evicted_memory_frame": 0,
                },
                context_row(selected_frame=1, candidate_count=72),
            ],
        }
        rows = MODULE.common_source_item_rows(
            item=item,
            events_by_run=events,
            generated_features=generated,
            gt_features=gt,
            runs=["baseline", "ri_b16_dino_rgb"],
            reference_run="baseline",
            content_run="baseline",
            target_stride=19,
            strict=True,
        )
        by_run = {row["run_name"]: row for row in rows}
        self.assertAlmostEqual(by_run["baseline"]["retention_gap"], 0.0)
        self.assertAlmostEqual(by_run["baseline"]["retrieval_gap"], 1.0)
        self.assertAlmostEqual(by_run["ri_b16_dino_rgb"]["retention_gap"], 0.2)
        self.assertAlmostEqual(by_run["ri_b16_dino_rgb"]["retrieval_gap"], 0.0)

    def test_summary_uses_trajectory_as_uncertainty_unit(self):
        rows = []
        for row_idx, retention in ((0, 0.1), (1, 0.3)):
            for target in (77, 96):
                rows.append(
                    {
                        "run_name": "ri_b16_dino_rgb",
                        "family": "RI",
                        "budget": 16,
                        "row": row_idx,
                        "scene": f"scene_{row_idx}",
                        "section_idx": 1,
                        "target_frame": target,
                        "retention_gap": retention,
                        "retrieval_gap": 0.4,
                        "total_oracle_gap": retention + 0.4,
                        "candidate_count": 16,
                    }
                )
        summaries, trajectory_rows = MODULE.summarize_runs(rows, min_section=1)
        self.assertEqual(len(trajectory_rows), 2)
        self.assertEqual(summaries[0]["trajectories"], 2)
        self.assertAlmostEqual(summaries[0]["retention_gap"], 0.2)
        self.assertLessEqual(
            summaries[0]["retention_gap_ci_low"],
            summaries[0]["retention_gap"],
        )
        self.assertGreaterEqual(
            summaries[0]["retention_gap_ci_high"],
            summaries[0]["retention_gap"],
        )

    def test_budget_steps_report_left_and_down(self):
        rows = [
            {"family": "GeoCov", "budget": 16, "retention_gap": 0.08, "retrieval_gap": 0.12},
            {"family": "GeoCov", "budget": 32, "retention_gap": 0.05, "retrieval_gap": 0.14},
        ]
        step = MODULE.family_budget_steps(rows)[0]
        self.assertTrue(step["moves_left"])
        self.assertFalse(step["moves_down"])
        self.assertAlmostEqual(step["retention_gap_change"], -0.03)

    def test_tradeoff_plot_writes_png_and_pdf(self):
        rows = []
        for family, budget, retention, retrieval in (
            ("Unbounded", None, 0.0, 0.22),
            ("RI", 16, 0.08, 0.16),
            ("RI", 32, 0.05, 0.17),
            ("GeoCov", 16, 0.09, 0.14),
            ("GeoCov", 32, 0.06, 0.13),
        ):
            rows.append(
                {
                    "family": family,
                    "budget": budget,
                    "retention_gap": retention,
                    "retention_gap_ci_low": max(0.0, retention - 0.01),
                    "retention_gap_ci_high": retention + 0.01,
                    "retrieval_gap": retrieval,
                    "retrieval_gap_ci_low": retrieval - 0.01,
                    "retrieval_gap_ci_high": retrieval + 0.01,
                }
            )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tradeoff.png"
            MODULE.plot_tradeoff(rows, path, "Synthetic tradeoff")
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
