import unittest

from utils.analyze_common_source_quality_strata import (
    build_strata,
    failure_prevalence,
)


class CommonSourceQualityStrataTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for trajectory in range(3):
            for query in range(8):
                anchor = query < 2
                rows.append(
                    {
                        "row": trajectory,
                        "unbounded_selected_frame": 100 + query,
                        "geocov_selected_frame": 0 if anchor else 50 + query,
                        "unbounded_overlap": 0.95,
                        "geocov_overlap": 0.70 if anchor else 0.90,
                        "unbounded_psnr": 10.0,
                        "unbounded_ssim": 0.30,
                        "geocov_psnr": 20.0 if anchor else 11.0,
                        "geocov_ssim": 0.70 if anchor else 0.32,
                        "psnr_delta": 10.0 if anchor else 1.0,
                        "ssim_delta": 0.40 if anchor else 0.02,
                    }
                )
        return rows

    def test_strata_remove_anchor_and_enforce_overlap(self):
        rows = self.make_rows()
        summary = build_strata(rows, 0.8, repeats=100, seed=2)
        self.assertEqual(summary[0]["queries"], 24)
        self.assertEqual(summary[1]["queries"], 18)
        self.assertEqual(summary[2]["queries"], 18)
        self.assertAlmostEqual(summary[3]["psnr_delta"], 1.0)

    def test_failure_prevalence_filters_on_selector_overlap(self):
        rows = self.make_rows()
        summary = failure_prevalence(rows, "geocov", 0.8, [10.0, 12.0])
        self.assertEqual(summary["queries"], 18)
        self.assertEqual(summary["psnr_le_10"], 0.0)
        self.assertEqual(summary["psnr_le_12"], 1.0)


if __name__ == "__main__":
    unittest.main()
