import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))
SPEC = importlib.util.spec_from_file_location("progression", PAPER / "make_progression_strips.py")
progression = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progression)


def scores(values):
    return {label: [{"ssim": float(v), "psnr_db": 20. + v} for v in sequence]
            for label, sequence in zip(progression.LABELS, values)}


class ProgressionTests(unittest.TestCase):
    def test_matches_five_times_and_ranks_both_comparator_declines(self):
        frames = list(range(1, 102, 10))
        ours = [.8] * len(frames)
        declining = np.linspace(.8, .2, len(frames))
        good = progression.choose_sequence(frames, scores([declining, declining, ours]), 10, 4)
        fifo_improves = progression.choose_sequence(frames, scores([declining, declining[::-1], ours]), 10, 4)
        ours_declines = progression.choose_sequence(frames, scores([declining, declining, declining]), 10, 4)
        self.assertGreater(good["rank_score"], fifo_improves["rank_score"])
        self.assertGreater(good["rank_score"], ours_declines["rank_score"])
        self.assertEqual(len(good["frames"]), 5)
        self.assertEqual(len(set(np.diff(good["frames"]))), 1)
        self.assertEqual(good["declining_steps"]["Unbounded"], 4)
        self.assertEqual(good["ssim_drop"]["Ours"], 0)

    def test_flat_examples_are_inspection_candidates_not_hidden(self):
        case = progression.choose_sequence([1, 11, 21, 31, 41], scores([[.5] * 5] * 3), 10, 4)
        self.assertEqual(case["rank_score"], 0)
        self.assertEqual(case["declining_steps"]["Unbounded"], 0)
        self.assertNotIn("screen_passed", case)
        with self.assertRaises(ValueError):
            progression.choose_sequence([1, 11, 21, 31, 41], scores([[.5] * 5] * 3), 10, 5)

    def test_sample_indices_exclude_input_and_stay_in_bounds(self):
        result = progression.sample_frames({"num_frames": 1825, "fps": 30}, 2)
        self.assertEqual(result[0], 1)
        self.assertLess(result[-1], 1825)
        self.assertEqual(set(np.diff(result)), {60})

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_cpu_cli_exports_five_frames_for_all_three_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt"
            gt.mkdir()
            clean = np.random.default_rng(1).integers(30, 190, (32, 48, 3), dtype=np.uint8)
            item = {"scene": "synthetic", "start_frame": 10, "duration_sec": 20, "fps": 10,
                    "num_frames": 201, "output_prefix": "fixture_", "gt_frames_dir": str(gt)}
            for index in progression.sample_frames(item, 2):
                Image.fromarray(clean).save(gt / f"{index + 10:04d}.png")
            for label, run in progression.RUNS.items():
                target = root / run
                target.mkdir()
                factors = np.full(201, .98) if label == "Ours" else np.linspace(.98, .2, 201)
                pixels = (clean[None] * factors[:, None, None, None]).astype(np.uint8)
                subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                                "-video_size", "48x32", "-framerate", "10", "-i", "pipe:0", "-c:v", "libx264",
                                "-pix_fmt", "yuv444p", "-crf", "0", str(target / "fixture_custom.mp4")],
                               input=pixels.tobytes(), check=True, capture_output=True, timeout=30)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(item))
            output = root / "output"
            command = [sys.executable, SPEC.origin, "--root", str(root), "--manifest", str(manifest),
                       "--duration", "20", "--min-span-sec", "10", "--score-width", "48", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            report = json.loads((output / "progression_search.json").read_text())
            self.assertEqual(len(report["selected"]), 1)
            case = report["selected"][0]
            self.assertEqual(len(case["frames"]), 5)
            self.assertGreater(case["ssim_drop"]["Unbounded"], .1)
            self.assertAlmostEqual(case["ssim_drop"]["Ours"], 0, places=4)
            for suffix in (".png", ".pdf", "_fifo_bare.png", "_ours_bare.png", "_unbounded_bare.png",
                           "_gt_check_bare.png", "_fidelity.pdf"):
                self.assertTrue((output / ("progression_01_row0" + suffix)).exists())
            (root / "fifo_b32/fixture_custom.mp4").unlink()
            command[-1] = str(root / "missing")
            failed = subprocess.run(command, capture_output=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            report = json.loads((root / "missing/progression_search.json").read_text())
            self.assertEqual(report["selected"], [])
            self.assertIn("Missing video", report["coverage"][0]["error"])


if __name__ == "__main__":
    unittest.main()
