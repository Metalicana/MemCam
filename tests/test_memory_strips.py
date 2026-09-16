import argparse
import csv
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

SPEC = importlib.util.spec_from_file_location("memory_strips", Path(__file__).resolve().parents[1] / "paper/make_memory_strips.py")
strips = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(strips)


def event(section, source, target):
    return {"event": "context_access", "selected": True, "section_idx": section,
            "selected_memory_frame": source, "target_frame": target, "selected_overlap": .9,
            "scene": "fixture", "dataset_start_frame": 0, "duration_sec": 60}


class MemoryStripTests(unittest.TestCase):
    def test_chain_requires_logged_reuse_and_not_ours_advantage(self):
        args = argparse.Namespace(min_overlap=.8, horizon_sections=6, min_drop_db=1, min_ssim_drop=.02)
        curves = {s: {"baseline_psnr_db": 20 if s < 3 else 10,
                      "baseline_ssim": .8 if s < 3 else .3,
                      "policy_psnr_db": 1, "policy_ssim": .01} for s in range(6)}
        scores = {i: {"baseline_psnr_db": 9} for i in (10, 240)}
        events = [event(3, 10, 230), event(4, 240, 310)]
        cases = strips.find_chains(events, curves, scores, args)
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0]["screen_passed"])
        self.assertEqual(strips.find_chains(events[:1], curves, scores, args), [])
        for curve in curves.values():
            curve.update(baseline_psnr_db=20, baseline_ssim=.8)
        self.assertFalse(strips.find_chains(events, curves, scores, args)[0]["screen_passed"])

    def test_full_frame_strip_dimensions_and_bare_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strip"
            strips.strip([Image.new("RGB", (80, 40), color) for color in ("red", "green", "blue", "white")],
                         ["Target ground truth", "Unbounded", "FIFO", "Ours"], path, width=80)
            with Image.open(path.with_name("strip_bare.png")) as image:
                self.assertEqual(image.size, (344, 40))
                self.assertEqual(image.getpixel((90, 20)), (0, 128, 0))
            with Image.open(path.with_suffix(".png")) as image:
                self.assertEqual(image.size, (344, 76))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_both_modes_end_to_end_on_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt"
            gt.mkdir()
            clean = np.random.default_rng(8).integers(40, 80, (32, 48, 3), dtype=np.uint8)
            prefix = "seed0_fixture_60s_"
            item = {"scene": "fixture", "start_frame": 0, "duration_sec": 60, "fps": 30,
                    "num_frames": 457, "output_prefix": prefix, "gt_frames_dir": str(gt)}
            for i in (10, 20, 30, 230, 240, 310, 420):
                Image.fromarray(clean).save(gt / f"{i:04d}.png")
            frames = np.repeat(clean[None], 457, axis=0)
            frames[10] = 240
            frames[229:] = 170
            for label, run in strips.RUNS.items():
                target = root / run
                (target / "access_traces").mkdir(parents=True)
                if label == "Unbounded":
                    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                                    "-video_size", "48x32", "-framerate", "30", "-i", "pipe:0", "-c:v", "libx264",
                                    "-pix_fmt", "yuv444p", "-crf", "0", str(target / (prefix + "custom.mp4"))],
                                   input=frames.tobytes(), check=True, capture_output=True, timeout=30)
                else:
                    shutil.copyfile(root / "baseline" / (prefix + "custom.mp4"), target / (prefix + "custom.mp4"))
                events = [event(3, {"Unbounded": 10, "FIFO": 20, "Ours": 30}[label], 230), event(4, 240, 310)]
                (target / "access_traces" / (prefix + "custom.jsonl")).write_text("\n".join(map(json.dumps, events)))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(item))
            retrieval = root / "retrieval"
            subprocess.run([sys.executable, SPEC.origin, "retrieval", "--root", str(root), "--manifest", str(manifest),
                            "--output", str(retrieval), "--target-stride", "1", "--score-width", "48"],
                           check=True, capture_output=True, timeout=60)
            self.assertTrue((retrieval / "retrieval_01_row0_bare.png").exists())
            selected = json.loads((retrieval / "retrieval_01_row0.json").read_text())
            self.assertEqual(selected["selected"], {"Unbounded": 10, "FIFO": 20, "Ours": 30})
            self.assertEqual(selected["content_mode"], "common")
            self.assertEqual(set(selected["scores"]["Ours"]), {"target_match", "historical_fidelity", "clean_view_match"})
            (root / "fifo_b32/access_traces" / (prefix + "custom.jsonl")).unlink()
            missing = root / "missing_fifo"
            subprocess.run([sys.executable, SPEC.origin, "retrieval", "--root", str(root), "--manifest", str(manifest),
                            "--output", str(missing)], check=True, capture_output=True, timeout=60)
            missing_report = json.loads((missing / "retrieval_search.json").read_text())
            self.assertEqual(missing_report["selected"], [])
            self.assertEqual(missing_report["coverage"][0]["status"], "error")
            self.assertEqual(list(missing.glob("*.png")), [])
            cache = root / "cache"
            cache.mkdir()
            (cache / "search.json").write_text(json.dumps({"parameters": {
                "root": str(root), "manifest": str(manifest), "duration": 60, "dataset_root": None,
                "reference_run": "baseline", "policy_run": "slam_b32_covisibility"}}))
            curves = [{"row": 0, "section": s, "time_sec": (s * 76 + 1) / 30, "end_sec": (s + 1) * 76 / 30,
                       "baseline_psnr_db": 20 if s < 3 else 10, "baseline_ssim": .8 if s < 3 else .3,
                       "policy_psnr_db": 10, "policy_ssim": .3} for s in range(6)]
            scores = [{"row": 0, "frame": i, "baseline_psnr_db": 9, "baseline_ssim": .2} for i in (10, 240, 420)]
            for name, rows in (("section_quality", curves), ("frame_quality", scores)):
                with (cache / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            output = root / "snowball"
            subprocess.run([sys.executable, SPEC.origin, "snowball", "--cache", str(cache), "--output", str(output)],
                           check=True, capture_output=True, timeout=60)
            for suffix in ("unbounded_bare.png", "ours_bare.png", "gt_bare.png", "curve.pdf"):
                self.assertTrue((output / f"snowball_candidate_01_row0_{suffix}").exists())
            chain = json.loads((output / "snowball_candidate_01_row0.json").read_text())
            self.assertEqual(chain["frames"], [10, 240, 310, 420])


if __name__ == "__main__":
    unittest.main()
