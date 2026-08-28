import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "analyze_fixed_history_candidate_pools.py"
SPEC = importlib.util.spec_from_file_location("fixed_history_pools", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixedHistoryCandidatePoolsTest(unittest.TestCase):
    def test_nested_pools_share_core_and_are_subsets(self):
        sizes = MODULE.parse_pool_sizes("4,6,8,all")
        pools = MODULE.nested_candidate_pools(range(12), sizes, order_seed=7)

        self.assertEqual(pools["b4"], [8, 9, 10, 11])
        self.assertEqual(len(pools["b6"]), 6)
        self.assertEqual(len(pools["b8"]), 8)
        self.assertEqual(pools["all"], list(range(12)))
        self.assertLess(set(pools["b4"]), set(pools["b6"]))
        self.assertLess(set(pools["b6"]), set(pools["b8"]))
        self.assertLess(set(pools["b8"]), set(pools["all"]))

    def test_nested_pool_order_is_reproducible(self):
        sizes = MODULE.parse_pool_sizes("4,8,all")
        first = MODULE.nested_candidate_pools(range(20), sizes, order_seed=11)
        second = MODULE.nested_candidate_pools(range(20), sizes, order_seed=11)

        self.assertEqual(first, second)

    def test_pool_winner_reuses_one_score_vector(self):
        candidates = [0, 1, 2, 3]
        scores = np.asarray([0.2, 0.9, 0.5, 0.7])

        winner_small, score_small = MODULE.select_pool_winner(
            candidates, scores, [0, 2, 3]
        )
        winner_full, score_full = MODULE.select_pool_winner(
            candidates, scores, candidates
        )

        self.assertEqual((winner_small, score_small), (3, 0.7))
        self.assertEqual((winner_full, score_full), (1, 0.9))

    def test_tied_winner_matches_first_candidate_scan_order(self):
        winner, score = MODULE.select_pool_winner(
            [4, 9, 12], np.asarray([0.8, 0.8, 0.1]), [4, 9]
        )

        self.assertEqual(winner, 4)
        self.assertEqual(score, 0.8)

    def test_batched_overlap_identical_pose_is_near_one(self):
        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
        scores = MODULE.batched_fov_overlap_scores(
            c2ws,
            target_frame=0,
            candidate_frames=[1],
            num_samples=2000,
            batch_size=1,
            seed=3,
        )

        self.assertEqual(scores.shape, (1,))
        self.assertGreater(scores[0], 0.99)

    def test_paired_pool_contrasts_use_within_trajectory_differences(self):
        rows = [
            {"row": 0, "pool": "b32", "intrinsic_psnr_db": 10.0, "intrinsic_ssim": 0.30},
            {"row": 1, "pool": "b32", "intrinsic_psnr_db": 20.0, "intrinsic_ssim": 0.40},
            {"row": 0, "pool": "all", "intrinsic_psnr_db": 11.0, "intrinsic_ssim": 0.32},
            {"row": 1, "pool": "all", "intrinsic_psnr_db": 19.0, "intrinsic_ssim": 0.41},
        ]

        contrasts = MODULE.paired_pool_contrasts(rows, "b32")

        self.assertEqual(len(contrasts), 1)
        self.assertAlmostEqual(contrasts[0]["intrinsic_psnr_db_delta_mean"], 0.0)
        self.assertAlmostEqual(contrasts[0]["intrinsic_ssim_delta_mean"], 0.015)
        self.assertEqual(contrasts[0]["intrinsic_psnr_db_wins"], 1)
        self.assertEqual(contrasts[0]["intrinsic_psnr_db_losses"], 1)

    def test_exact_sign_test(self):
        self.assertAlmostEqual(MODULE.exact_sign_test(15, 0), 2 / (2**15))


if __name__ == "__main__":
    unittest.main()
