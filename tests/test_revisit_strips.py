import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))
SPEC = importlib.util.spec_from_file_location("revisit_strips", PAPER / "make_revisit_strips.py")
revisit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(revisit)


class RevisitStripTests(unittest.TestCase):
    def options(self):
        return argparse.Namespace(pose_stride=1, position_m=.25, rotation_deg=5.,
                                  min_gap_sec=3., min_away_sec=1., min_visits=3, max_visits=4)

    def test_distinct_returns_not_stationary_loitering(self):
        poses = np.repeat(np.eye(4)[None], 140, axis=0)
        self.assertEqual(revisit.visit_groups(poses, 10, self.options()), [])
        poses[:, 0, 3] = 5
        for start in (1, 50, 100):
            poses[start:start + 5, 0, 3] = 0
        groups = revisit.visit_groups(poses, 10, self.options())
        view = next(g for g in groups if g["frames"] == [1, 50, 100])
        self.assertEqual(view["position_from_anchor_m"], [0, 0, 0])

    def test_rotation_and_departure_are_required(self):
        poses = np.repeat(np.eye(4)[None], 140, axis=0)
        # Small position jitter exits the inner tolerance, but is not a departure.
        poses[:, 0, 3] = .3
        for start in (1, 50, 100):
            poses[start:start + 5, 0, 3] = 0
        self.assertEqual(revisit.visit_groups(poses, 10, self.options()), [])
        poses[:, 0, 3] = 5
        for start in (1, 50, 100):
            poses[start:start + 5, 0, 3] = 0
        poses[100:105, :3, :3] = np.diag([-1, -1, 1])
        self.assertFalse(any(g["frames"] == [1, 50, 100] for g in revisit.visit_groups(poses, 10, self.options())))

    def test_gt_pairwise_check_and_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            clean = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
            for index in (1, 50, 100):
                Image.fromarray(clean).save(root / f"{index:04d}.png")
            item = {"gt_frames_dir": str(root), "start_frame": 0}
            good, pairs, gt = revisit.verify_gt(item, [1, 50, 100], .9)
            self.assertTrue(good)
            self.assertEqual(len(pairs), 3)
            case = {"scene": "Synthetic layout test", "frames": [1, 50, 100], "fps": 10,
                    "scores": {name: [{"psnr_db": 30, "ssim": .9}] * 3 for name in ("Unbounded", "Ours")}}
            revisit.render(case, {"Unbounded": gt, "Ours": gt}, gt, root / "example")
            self.assertTrue((root / "example.png").exists())
            self.assertTrue((root / "example_fidelity.pdf").exists())
            self.assertTrue((root / "example_ours_bare.png").exists())
            Image.new("RGB", (96, 64), "black").save(root / "0100.png")
            good, _, _ = revisit.verify_gt(item, [1, 50, 100], .9)
            self.assertFalse(good)

    def test_diagnostics_separates_pairs_from_three_visit_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.options()
            args.output, args.dataset_root = root, None
            args.min_gt_ssim, args.diagnostic_gt_checks = .9, 12
            groups = [{"anchor": 1, "frames": frames, "all_pose_visits": frames}
                      for frames in ([1, 50], [1, 50, 100])]
            def verify(item, frames, threshold):
                score = .95 if len(frames) == 2 else .85
                return score >= threshold, [{"ssim": score}], {i: Image.new("RGB", (48, 32)) for i in frames}
            with patch.object(revisit, "load_poses", return_value=None), \
                 patch.object(revisit, "visit_groups", return_value=groups), \
                 patch.object(revisit, "verify_gt", side_effect=verify), redirect_stdout(io.StringIO()):
                revisit.diagnose(args, {0: {"scene": "fixture", "fps": 10}})
            report = json.loads((root / "diagnostics.json").read_text())
            self.assertEqual(len(report["records"]), 3)
            strict = report["records"][0]
            self.assertEqual(strict["position_m"], .25)
            self.assertEqual(strict["max_visits"], 3)
            self.assertEqual(strict["gt_checks"][0]["passed_among_checked"], 1)
            self.assertEqual(strict["gt_checks"][1]["passed_among_checked"], 0)
            self.assertTrue((root / "inspection_row0_strict_3plus_gt.png").exists())
            self.assertNotIn("selected", report)


if __name__ == "__main__":
    unittest.main()
