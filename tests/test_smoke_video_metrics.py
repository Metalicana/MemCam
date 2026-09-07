import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
import smoke_video_metrics as smoke
sys.path.pop(0)


class SmokeTests(unittest.TestCase):
    def test_original_cohort_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text('\n'.join(json.dumps(item) for item in [
                {"duration_sec": 10, "output_prefix": "room_10s_"},
                {"duration_sec": 60, "output_prefix": "room_60s_", "num_frames": 1797},
            ]))
            run = root / "videos/baseline"
            run.mkdir(parents=True)
            (run / "seed1_extra60s_room_60s_custom.mp4").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                smoke.select_video(manifest, run.parent, "baseline")
            original = run / "room_60s_custom.mp4"
            original.write_bytes(b"fixture")
            row, _, video = smoke.select_video(manifest, run.parent, "baseline")
            self.assertEqual(row, 1)
            self.assertEqual(video, original.resolve())
            subprocess.run([
                sys.executable, str(UTILS / "smoke_video_metrics.py"),
                "--manifest", str(manifest), "--results-root", str(run.parent),
                "--output-root", str(root / "output"), "--dry-run",
            ], check=True, capture_output=True, text=True)
            work = next((root / "output").iterdir())
            self.assertEqual((work / "input/baseline" / original.name).resolve(), video)
            self.assertTrue(smoke.load(work / "spec.json")["dry_run"])
            self.assertFalse((work / "status.json").exists())

    def test_long_aggregates_and_invalid_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            name = "room_60s_custom.mp4"
            payload = {}
            for dimension in smoke.DIMENSIONS:
                value = False if dimension == "dynamic_degree" else .8
                payload[dimension] = [.8, [{"video_path": "clip.mp4", "video_results": .8}],
                                      [{"video_path": name, "video_results": value}]]
            path = output / "fixture_eval_results.json"
            path.write_text(json.dumps(payload))
            smoke.validate_long(output, name)
            payload["subject_consistency"][2][0]["video_path"] = "room.mp4"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "grouping"):
                smoke.validate_long(output, name)
            del payload["subject_consistency"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "dimension"):
                smoke.validate_long(output, name)

    def test_cut3r_complete_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reconstruction = root / "reconstruction"
            cameras = reconstruction / "camera"
            cameras.mkdir(parents=True)
            for index in range(2):
                (cameras / f"{index}.npz").touch()
            (reconstruction / "metadata.json").write_text(json.dumps({"video_frame_indices": [0, 30]}))
            (root / "cut3r_run_status.json").write_text(json.dumps([
                {"status": "completed", "output_dir": str(reconstruction)}]))
            (root / "cut3r_camera_failures.json").write_text("[]")
            (root / "cut3r_camera_metrics.json").write_text(json.dumps([{
                "run_name": "baseline", "row": 3, "rotation_error_deg_mean": 1.,
                "translation_error_scale_only_mean": .1, "translation_error_sim3_mean": .1,
            }]))
            smoke.validate_cut3r(root, root, "baseline", 3)
            with self.assertRaisesRegex(ValueError, "different run/row"):
                smoke.validate_cut3r(root, root, "baseline", 8)
            (cameras / "1.npz").unlink()
            with self.assertRaisesRegex(ValueError, "camera count"):
                smoke.validate_cut3r(root, root, "baseline", 3)


if __name__ == "__main__":
    unittest.main()
