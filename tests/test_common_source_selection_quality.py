import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "utils/analyze_common_source_selection_quality.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_common_source_selection_quality", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module()


def row(run_name, trajectory, psnr, ssim):
    return {
        "selection_run": run_name,
        "content_run": "baseline",
        "row": trajectory,
        "section_idx": 4,
        "selected_weighted_psnr_db_mean": psnr,
        "selected_weighted_ssim_mean": ssim,
    }


class CommonSourceSelectionQualityTest(unittest.TestCase):
    def test_pairing_uses_same_content_and_policy_minus_reference(self):
        rows = [
            row("baseline", 0, 10.0, 0.3),
            row("slam", 0, 14.0, 0.5),
            row("baseline", 1, 12.0, 0.4),
            row("slam", 1, 14.0, 0.5),
        ]
        result = ANALYSIS.paired_rows(rows, "baseline")[0]

        self.assertEqual(result["content_run"], "baseline")
        self.assertAlmostEqual(result["selected_psnr_db_delta_mean"], 3.0)
        self.assertAlmostEqual(result["selected_ssim_delta_mean"], 0.15)
        self.assertEqual(result["selected_psnr_db_trajectory_wins"], 2)


if __name__ == "__main__":
    unittest.main()
