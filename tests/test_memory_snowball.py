import argparse
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

SPEC = importlib.util.spec_from_file_location(
    "snowball", Path(__file__).resolve().parents[1] / "utils/visualize_memory_snowball.py")
snowball = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snowball)


def event(section, source, target):
    return {"event": "context_access", "selected": True, "section_idx": section,
            "selected_memory_frame": source, "target_frame": target, "selected_overlap": .9}


class SnowballTests(unittest.TestCase):
    def test_trace_causality_and_overrides(self):
        good = event(3, 10, 230)
        self.assertEqual(snowball.logged_reads({(3, 230): good}, 457), [good])
        with self.assertRaisesRegex(ValueError, "Noncausal"):
            snowball.logged_reads({(3, 230): event(3, 235, 230)}, 457)
        with self.assertRaisesRegex(ValueError, "override"):
            snowball.logged_reads({(3, 230): dict(good, context_content_override=1)}, 457)

    def test_rank_requires_output_degradation_and_reuse(self):
        args = argparse.Namespace(window_sections=2, min_age_sec=5, min_overlap=.8,
                                  max_memory_psnr=12, min_before_psnr=14, min_drop_db=2,
                                  min_later_drop_db=.5, min_ssim_drop=.03, min_policy_gain=1, min_reuses=1)
        reads = [event(3, 10, 230), event(4, 240, 310)]
        quality = {10: {"psnr_db": 10, "ssim": .3}, 240: {"psnr_db": 11, "ssim": .3}}
        base = {s: {"psnr_db": 20 if s < 3 else 11 if s == 3 else 9,
                    "ssim": .8 if s < 3 else .4 if s == 3 else .3} for s in range(6)}
        policy = {s: {"psnr_db": 20, "ssim": .8} for s in range(6)}
        cases = snowball.rank_episodes(reads, quality, base, policy, 30, args)
        case = next(c for c in cases if c["section"] == 3)
        self.assertTrue(case["qualified"])
        self.assertEqual(case["immediate_drop_db"], 9)
        self.assertEqual(len(case["reuses"]), 1)
        cases = snowball.rank_episodes(reads[:1], quality, base, policy, 30, args)
        self.assertFalse(next(c for c in cases if c["section"] == 3)["qualified"])
        cases = snowball.rank_episodes(reads, quality, policy, policy, 30, args)
        self.assertFalse(any(c["qualified"] for c in cases))

    def test_sampling_and_median_representative(self):
        samples = snowball.sample_sections(160, 15)
        self.assertEqual(samples[0], [1, 16, 31, 46, 61, 76])
        self.assertEqual(samples[2], [153])
        values = {i: {"psnr_db": value} for i, value in enumerate([10, 20, 30])}
        self.assertEqual(snowball.representative([0, 1, 2], values), 1)

    def test_cpu_cli_end_to_end(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Rendering/video test requires matplotlib")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("Rendering/video test requires FFmpeg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_dir = root / "gt"
            gt_dir.mkdir()
            rng = np.random.default_rng(8)
            clean = rng.integers(40, 80, (32, 48, 3), dtype=np.uint8)
            prefix = "seed0_fixture_60s_"
            count = 457
            item = {"scene": "fixture", "start_frame": 0, "duration_sec": 60, "fps": 30,
                    "num_frames": count, "output_prefix": prefix, "gt_frames_dir": str(gt_dir)}
            for i in range(count):
                Image.fromarray(clean).save(gt_dir / f"{i:04d}.png")
            for run in ("baseline", "slam_b32_covisibility"):
                run_dir = root / run
                run_dir.mkdir()
                frames = []
                for i in range(count):
                    frame = clean.copy()
                    if run == "baseline":
                        shift = 170 if i == 10 else 90 if 229 <= i <= 304 else 130 if i >= 305 else 3
                        frame = np.clip(frame.astype(float) + shift, 0, 255).astype(np.uint8)
                    frames.append(frame)
                subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                                "-video_size", "48x32", "-framerate", "30", "-i", "pipe:0",
                                "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0",
                                str(run_dir / (prefix + "custom.mp4"))],
                               input=np.stack(frames).tobytes(), check=True, capture_output=True, timeout=30)
            traces = root / "baseline/access_traces"
            traces.mkdir()
            reads = [dict(r, scene="fixture", dataset_start_frame=0, duration_sec=60)
                     for r in (event(3, 10, 230), event(4, 240, 310))]
            (traces / (prefix + "custom.jsonl")).write_text("\n".join(map(json.dumps, reads)))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(item))
            output = root / "out"
            result = subprocess.run([sys.executable, str(SPEC.origin), "--manifest", str(manifest),
                                     "--root", str(root), "--score-width", "48", "--output", str(output)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "search.json").read_text())
            self.assertEqual(report["selected"][0]["section"], 3)
            self.assertEqual(report["coverage"][0]["status"], "scored")
            self.assertTrue((output / "episode_01.pdf").is_file())
            self.assertTrue((output / "quality_over_time.csv").is_file())
            pixels = np.asarray(Image.open(output / "episode_01.png"))
            self.assertGreater(pixels.std(), 10)
            # Inspection must work even when no candidate passed the original gates.
            report["selected"] = []
            for case in report["candidates"]:
                case["qualified"] = False
                case["criteria"]["policy_advantage"] = False
            cached = json.dumps(report)
            (output / "search.json").write_text(cached)
            shutil.rmtree(traces)
            inspected = root / "inspection"
            result = subprocess.run([sys.executable, str(SPEC.origin), "--inspect-from", str(output),
                                     "--output", str(inspected), "--top", "1"],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((inspected / "inspection_01.pdf").is_file())
            self.assertIn("policy_advantage", (inspected / "inspection.csv").read_text())
            inspection = json.loads((inspected / "inspection.json").read_text())
            self.assertFalse(inspection["selected"][0]["qualified"])
            self.assertEqual((output / "search.json").read_text(), cached)


if __name__ == "__main__":
    unittest.main()
