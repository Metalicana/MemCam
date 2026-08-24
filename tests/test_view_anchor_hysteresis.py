import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "utils/analyze_view_anchor_hysteresis.py"
    spec = importlib.util.spec_from_file_location("analyze_view_anchor_hysteresis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module()


def poses(count):
    output = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    output[:, 0, 3] = np.arange(count, dtype=np.float64) * 0.01
    return output


class ViewAnchorHysteresisTest(unittest.TestCase):
    def test_pair_builder_uses_oldest_equivalent_frame(self):
        pairs = ANALYSIS.build_view_pairs(
            poses(12),
            sample_stride=4,
            candidate_stride=2,
            min_history_frames=4,
            min_temporal_gap=2,
            min_view_similarity=0.8,
        )
        self.assertEqual([row["later_frame"] for row in pairs], [4, 8])
        self.assertEqual([row["earlier_frame"] for row in pairs], [1, 1])

    def test_summary_uses_trajectory_means_and_positive_is_older_better(self):
        rows = []
        for trajectory, psnr_delta, ssim_delta in [
            (0, 2.0, 0.2),
            (1, 1.0, 0.1),
        ]:
            rows.append(
                {
                    "row": trajectory,
                    "scene": f"scene{trajectory}",
                    "older_minus_newer_psnr_db": psnr_delta,
                    "older_minus_newer_ssim": ssim_delta,
                    "view_similarity": 0.9,
                    "frame_gap": 100,
                }
            )
        trajectory_rows, summary = ANALYSIS.summarize_pairs(
            rows, bootstrap_repeats=100, seed=0
        )
        self.assertEqual(len(trajectory_rows), 2)
        self.assertAlmostEqual(summary["older_minus_newer_psnr_db_mean"], 1.5)
        self.assertEqual(summary["older_minus_newer_psnr_db_trajectory_wins"], 2)
        self.assertEqual(summary["decision"], "NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
