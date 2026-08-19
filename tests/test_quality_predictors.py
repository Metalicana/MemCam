import json
import tempfile
import unittest
from pathlib import Path

from utils.analyze_quality_predictors import (
    aggregate_section_diagnostics,
    build_paired_deltas,
    join_video_sources,
    load_section_quality,
    spearman,
)


class QualityPredictorTests(unittest.TestCase):
    def test_aggregate_section_diagnostics_uses_late_quartile(self):
        rows = []
        for section_idx in range(8):
            rows.append(
                {
                    "run_name": "baseline",
                    "row": "3",
                    "scene": "Scene",
                    "dataset_start_frame": "100",
                    "duration_sec": "180",
                    "section_idx": str(section_idx),
                    "retrieval_gap": str(section_idx),
                    "candidate_count": str(10 + section_idx),
                }
            )
        result = aggregate_section_diagnostics(rows, duration=180, late_fraction=0.25)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["retrieval_gap_early_mean"], 0.5)
        self.assertAlmostEqual(result[0]["retrieval_gap_late_mean"], 6.5)
        self.assertAlmostEqual(result[0]["retrieval_gap_late_minus_early"], 6.0)

    def test_paired_delta_positive_means_better(self):
        common = {
            "row": "3",
            "scene": "Scene",
            "start_frame": "100",
            "duration_sec": "180",
        }
        rows = [
            {
                **common,
                "run_name": "baseline",
                "retrieval_gap_late_mean": 0.30,
                "lpips_alex": 0.70,
                "ssim": 0.20,
            },
            {
                **common,
                "run_name": "slam_b32_covisibility",
                "retrieval_gap_late_mean": 0.10,
                "lpips_alex": 0.60,
                "ssim": 0.25,
            },
        ]
        result = build_paired_deltas(
            rows,
            baseline_run="baseline",
            predictors=["retrieval_gap_late_mean"],
            outcomes=["lpips_alex", "ssim"],
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result[0]["retrieval_gap_late_mean_improvement"], 0.20
        )
        self.assertAlmostEqual(result[0]["lpips_alex_improvement"], 0.10)
        self.assertAlmostEqual(result[0]["ssim_improvement"], 0.05)

    def test_join_video_sources_matches_cut3r_and_vbench(self):
        diagnostic = {
            "run_name": "slam_b32_covisibility",
            "row": "3",
            "scene": "Scene",
            "start_frame": "100",
            "duration_sec": "60",
        }
        quality = {
            **diagnostic,
            "status": "completed",
            "output": "/tmp/seed0_Scene_0100_60s_custom.mp4",
            "lpips_alex": "0.5",
        }
        cut3r = {
            **diagnostic,
            "rotation_error_deg_mean": "4.0",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "slam_b32_covisibility"
            run_dir.mkdir()
            payload = {
                "subject_consistency": [
                    0.8,
                    [
                        {
                            "video_path": "/elsewhere/seed0_Scene_0100_60s_custom.mp4",
                            "video_results": 0.75,
                        }
                    ],
                ]
            }
            (run_dir / "results_1_eval_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            joined = join_video_sources(
                [diagnostic],
                [quality],
                [cut3r],
                Path(tmp_dir),
                ["slam_b32_covisibility"],
            )
        self.assertEqual(len(joined), 1)
        self.assertAlmostEqual(joined[0]["lpips_alex"], 0.5)
        self.assertAlmostEqual(joined[0]["rotation_error_deg_mean"], 4.0)
        self.assertAlmostEqual(joined[0]["vbench_subject_consistency"], 0.75)

    def test_frame_boundary_belongs_to_previous_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "baseline"
            run_dir.mkdir()
            rows = [
                {
                    "row": 3,
                    "scene": "Scene",
                    "duration_sec": 180,
                    "frame_index": 76,
                    "gt_frame_index": 176,
                    "lpips_alex": 0.2,
                },
                {
                    "row": 3,
                    "scene": "Scene",
                    "duration_sec": 180,
                    "frame_index": 77,
                    "gt_frame_index": 177,
                    "lpips_alex": 0.4,
                },
            ]
            (run_dir / "frame_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = load_section_quality(
                Path(tmp_dir), ["baseline"], duration=180, section_stride=76
            )
        by_section = {row["section_idx"]: row for row in result}
        self.assertAlmostEqual(by_section[0]["lpips_alex"], 0.2)
        self.assertAlmostEqual(by_section[1]["lpips_alex"], 0.4)

    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 1, 2, 3]), 1.0)


if __name__ == "__main__":
    unittest.main()
