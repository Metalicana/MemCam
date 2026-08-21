import unittest

import numpy as np
from PIL import Image, ImageFilter

from utils.calibrate_frame_quality_estimators import (
    add_within_trajectory_labels,
    classification_metrics,
    fit_conservative_threshold,
    fit_quality_threshold,
    quality_auc,
    quality_scores_from_array,
    trajectory_split,
)


class FrameQualityEstimatorTests(unittest.TestCase):
    def test_blur_reduces_gradient_and_laplacian_scores(self):
        checkerboard = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
        sharp = np.repeat(checkerboard[..., None], 3, axis=2)
        blurred = np.asarray(
            Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(radius=3))
        )
        sharp_scores = quality_scores_from_array(sharp)
        blur_scores = quality_scores_from_array(blurred)
        self.assertGreater(
            sharp_scores["gradient_energy"], blur_scores["gradient_energy"]
        )
        self.assertGreater(
            sharp_scores["laplacian_variance"],
            blur_scores["laplacian_variance"],
        )

    def test_quality_auc_has_expected_orientation(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        bad = [True, True, False, False]
        self.assertAlmostEqual(quality_auc(scores, bad), 1.0)
        self.assertAlmostEqual(quality_auc(list(reversed(scores)), bad), 0.0)

    def test_balanced_threshold_separates_clean_and_bad(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        bad = [True, True, False, False]
        threshold = fit_quality_threshold(scores, bad)
        metrics = classification_metrics(scores, bad, threshold)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)

    def test_conservative_threshold_honors_clean_rejection_cap(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        bad = [True, True, False, True, False, False]
        threshold = fit_conservative_threshold(scores, bad, 0.0)
        metrics = classification_metrics(scores, bad, threshold)
        self.assertEqual(metrics["clean_false_reject_rate"], 0.0)
        self.assertGreater(metrics["bad_recall"], 0.0)

    def test_bad_labels_are_defined_within_run_and_trajectory(self):
        rows = []
        for run_name, offset in (("baseline", 0.0), ("slam", 100.0)):
            for frame in range(10):
                rows.append(
                    {
                        "run_name": run_name,
                        "row": 3,
                        "psnr_db": offset + frame,
                        "ssim": offset + frame,
                    }
                )
        add_within_trajectory_labels(rows, 0.2)
        for run_name in ("baseline", "slam"):
            group = [row for row in rows if row["run_name"] == run_name]
            self.assertEqual(sum(row["gt_bad_frame"] for row in group), 2)

    def test_trajectory_split_is_disjoint_and_deterministic(self):
        train_a, test_a = trajectory_split(range(15), 1.0 / 3.0, 17)
        train_b, test_b = trajectory_split(range(15), 1.0 / 3.0, 17)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertFalse(train_a & test_a)
        self.assertEqual(train_a | test_a, set(range(15)))
        self.assertEqual(len(test_a), 5)


if __name__ == "__main__":
    unittest.main()
