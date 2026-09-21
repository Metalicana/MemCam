import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from paper import export_matched_gt_videos as exporter


def fixture(root):
    folder = root / "gt"
    folder.mkdir()
    arrays = np.random.default_rng(6).integers(0, 256, size=(4, 24, 32, 3), dtype=np.uint8)
    for frame, pixels in enumerate(arrays, 7):
        Image.fromarray(pixels).save(folder / f"{frame:04d}.png")
    item = dict(scene="example", start_frame=7, num_frames=4, fps=30, duration_sec=60,
                output_prefix="seed0_example_0007_60s_", gt_frames_dir=str(folder))
    manifest = root / "manifest.jsonl"
    manifest.write_text('{}\n\n' + json.dumps(item) + '\n')
    plan = root / "plan.json"
    plan.write_text(json.dumps(dict(manifest=str(manifest), manifest_sha256=exporter.digest(manifest),
                                    rows=[2], expected=[item["output_prefix"] + "custom.mp4"])))
    return plan, item, arrays


class GTVideoExportTests(unittest.TestCase):
    def test_cohort_and_source_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, item, _ = fixture(root)
            selected = exporter.load_items(plan, 1)
            self.assertEqual(selected[0]["manifest_row"], 2)
            self.assertEqual(selected[0]["start_frame"], 7)
            with self.assertRaisesRegex(ValueError, "distinct manifest rows"):
                exporter.load_items(plan, 15)
            config = json.loads(plan.read_text())
            config["expected"] = ["wrong.mp4"]
            plan.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "identities"):
                exporter.load_items(plan, 1)
            signature = exporter.source_signature(item)
            (Path(item["gt_frames_dir"]) / "0008.png").touch()
            self.assertNotEqual(signature, exporter.source_signature(item))
            (Path(item["gt_frames_dir"]) / "0009.png").unlink()
            with self.assertRaises(FileNotFoundError):
                exporter.source_signature(item)

    def test_no_x264_options_and_reject_missing_encoder(self):
        item = dict(gt_frames_dir="/gt", start_frame=7, fps=30, num_frames=4)
        command = exporter.encode_command("ffmpeg", item, Path("clip.mkv"), 4)
        self.assertIn("ffv1", command)
        for option in ("-crf", "-preset", "libx264rgb", "cuda"):
            self.assertNotIn(option, command)
        unavailable = subprocess.CompletedProcess([], 0, stdout="Codec ffv1 is not recognized", stderr="")
        with patch.object(exporter.shutil, "which", side_effect=lambda name: name), \
                patch.object(exporter.subprocess, "run", return_value=unavailable), \
                patch.object(exporter, "load_items") as load:
            with self.assertRaisesRegex(RuntimeError, "lacks the FFV1 RGB encoder"):
                exporter.export(Path("missing.json"), Path("missing-output"))
            load.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
    def test_real_roundtrip_alignment_resume_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, item, expected_pixels = fixture(root)
            output = root / "export"
            archive = exporter.export(plan, output, expected_videos=1, threads=1)
            video = output / (item["output_prefix"] + "gt.mkv")
            result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                    check=True, capture_output=True)
            actual_pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(expected_pixels.shape)
            np.testing.assert_array_equal(actual_pixels, expected_pixels)
            before = video.stat().st_mtime_ns
            exporter.export(plan, output, expected_videos=1, threads=1)
            self.assertEqual(before, video.stat().st_mtime_ns)
            with zipfile.ZipFile(archive) as handle:
                self.assertIsNone(handle.testzip())
                records = json.loads(handle.read("manifest.json"))
                self.assertEqual(records[0]["start_frame"], 7)
                self.assertEqual(records[0]["num_frames"], 4)
                self.assertEqual(records[0]["gt_video"], video.name)
                self.assertEqual(len(handle.namelist()), 3)
            bad_count = dict(item, num_frames=5)
            with self.assertRaisesRegex(ValueError, "frame count"):
                exporter.verify_video(shutil.which("ffprobe"), video, bad_count)
            video.write_bytes(b"interrupted")
            exporter.export(plan, output, expected_videos=1, threads=1)
            exporter.verify_video(shutil.which("ffprobe"), video, item)


if __name__ == "__main__":
    unittest.main()
