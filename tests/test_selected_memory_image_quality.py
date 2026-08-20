import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "utils/analyze_selected_memory_image_quality.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_selected_memory_image_quality", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module()


def section(run_name, row, section_idx, selected_psnr, selected_ssim, chunk_psnr, chunk_ssim):
    return {
        "run_name": run_name,
        "row": row,
        "section_idx": section_idx,
        "selected_weighted_psnr_db_mean": selected_psnr,
        "selected_weighted_psnr_db_p10": selected_psnr - 1,
        "selected_weighted_ssim_mean": selected_ssim,
        "selected_weighted_ssim_p10": selected_ssim - 0.1,
        "following_chunk_psnr_db_mean": chunk_psnr,
        "following_chunk_ssim_mean": chunk_ssim,
    }


class SelectedMemoryImageQualityTest(unittest.TestCase):
    def test_chunk_frame_indices_follow_section_stride(self):
        frames = ANALYSIS.requested_chunk_frames([1, 3], frame_stride=4)
        self.assertEqual(frames[1][:3], [77, 81, 85])
        self.assertEqual(frames[3][0], 229)
        self.assertEqual(frames[1][-1], 149)

    def test_paired_deltas_are_policy_minus_unbounded(self):
        rows = [
            section("baseline", 0, 1, 10.0, 0.4, 11.0, 0.45),
            section("bounded", 0, 1, 12.0, 0.5, 14.0, 0.55),
            section("baseline", 1, 1, 20.0, 0.6, 21.0, 0.65),
            section("bounded", 1, 1, 21.0, 0.7, 23.0, 0.75),
        ]
        result = ANALYSIS.paired_policy_rows(rows, "baseline")[0]

        self.assertEqual(result["paired_videos"], 2)
        self.assertAlmostEqual(
            result["selected_weighted_psnr_db_delta_mean"], 1.5
        )
        self.assertAlmostEqual(
            result["following_chunk_psnr_db_delta_mean"], 2.5
        )
        self.assertEqual(
            result["selected_weighted_psnr_db_trajectory_wins"], 2
        )


if __name__ == "__main__":
    unittest.main()
