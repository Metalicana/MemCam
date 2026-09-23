import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
import run_keepsake_native_roundtrip as native
from camera_rotation_utils import exact_roundtrip_c2ws, roundtrip_pairs


def fixture(root):
    rows = []
    for i in range(15):
        image = root / f"scene{i:02d}.png"
        Image.new("RGB", (16, 16), (i, 20, 30)).save(image)
        pose = root / f"scene{i:02d}.json"
        pose.write_text(json.dumps({"CineCameraActor": {"0": {"position": [0, 0, 0], "rotation": [0, 0, 0]}}}))
        rows.append(dict(scene=f"scene{i:02d}", start_frame=0, input_image=str(image), pose_path=str(pose),
                         prompt="A room", duration_sec=60, output_prefix=f"seed0_scene{i:02d}_0000_60s_"))
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)) + "\n")
    for name in native.WEIGHTS.values():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint, not model weights")
    output = root / "output"
    output.mkdir()
    return argparse.Namespace(manifest=manifest, starts_manifest=None, dataset_root=None,
                              model_root=root, output=output, fvd_cache_dir=root / "i3d", plan_only=False), rows


def profile_records(case):
    sections = [dict(event="section_profile", section_idx=i, section_end_frame=(i + 1) * 76,
                     memory_policy="slam_covisibility", memory_budget=32, stored_memory_size=32)
                for i in range((case["frames"] - 1) // 76)]
    return sections + [dict(event="rollout_summary", completed=True, rollout_latency_s=1.0,
                            peak_bank_frame_bytes=2048)]


def fixture_generation(pipe, case, directory):
    attempt = directory / "attempt_synthetic"
    (attempt / "frames").mkdir(parents=True)
    for i in range(case["frames"]):
        Image.new("RGB", (8, 8), (i % 100, 20, 30)).save(attempt / "frames" / f"{i:04d}.png")
    (attempt / "video.mp4").write_bytes(b"not a real video; orchestration fixture")
    np.save(attempt / "poses.npy", exact_roundtrip_c2ws(np.eye(4), case["angle"]))
    (attempt / "profile.jsonl").write_text("\n".join(map(json.dumps, profile_records(case))) + "\n")
    (attempt / "access.jsonl").write_text('{}\n')
    record = dict(status="complete", case=case, attempt=attempt.name,
                  files={str(p.relative_to(attempt)): native.digest(p) for p in attempt.rglob("*") if p.is_file()},
                  resource=native.profile_summary(attempt / "profile.jsonl", case))
    native.save_json(directory / "generation.json", record)
    return record


class FakeFVD(native.metrics.FVDRunner):
    def __init__(self):
        for key, value in native.FVD_CONFIG.items():
            setattr(self, key, value)

    def _encode_batch(self, clips):
        return np.stack([np.mean(clip, axis=(0, 2, 3)) for clip in clips])


class NativeRoundTripTests(unittest.TestCase):
    def test_exact_poses_for_both_lengths_and_nonidentity_initial_pose(self):
        base = np.eye(4)
        base[:3, 3] = [10, 20, 30]
        base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        for angle, count in ((90, 153), (360, 609)):
            poses = exact_roundtrip_c2ws(base, angle)
            self.assertEqual(poses.shape, (count, 4, 4))
            np.testing.assert_array_equal(poses, poses[::-1])
            np.testing.assert_array_equal(poses[0], base)
            np.testing.assert_array_equal(poses[-1], base)
            np.testing.assert_array_equal(poses[:, :3, 3], np.tile(base[:3, 3], (count, 1)))
            half = count // 2
            angles = np.unwrap(np.arctan2((np.linalg.inv(base) @ poses[:half + 1])[:, 1, 0],
                                          (np.linalg.inv(base) @ poses[:half + 1])[:, 0, 0]))
            self.assertAlmostEqual(angles[-1], np.radians(angle))
            self.assertTrue(np.all(np.diff(angles) > 0))
            pairs = roundtrip_pairs(count)
            self.assertEqual(len(pairs), half - 1)
            self.assertTrue(all(0 < i < half < j < count - 1 for i, j in pairs))
            for i, j in pairs:
                np.testing.assert_array_equal(poses[i], poses[j])
        for angle in (45, 180, 720):
            with self.assertRaises(ValueError):
                exact_roundtrip_c2ws(base, angle)
        with self.assertRaises(ValueError):
            exact_roundtrip_c2ws(np.zeros((4, 4)), 90)
        with self.assertRaises(ValueError):
            roundtrip_pairs(152)

    def test_frozen_five_no_missing_input_substitution_and_no_other_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, rows = fixture(Path(temporary))
            with contextlib.redirect_stdout(io.StringIO()):
                plan, cases, _ = native.prepare(args)
            self.assertEqual(len(cases), 10)
            self.assertEqual({c["start"]["scene"] for c in cases}, {f"scene{i:02d}" for i in (0, 6, 12, 13, 14)})
            self.assertEqual(plan["generation"]["policy"], "slam_covisibility")
            self.assertEqual(plan["generation"]["budget"], 32)
            self.assertFalse(plan["limits"]["original_test_split_verified"])
            Path(rows[0]["input_image"]).unlink()
            with self.assertRaises(FileNotFoundError):
                native.select_starts(args)

    def test_explicit_starts_and_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, rows = fixture(Path(temporary))
            args.starts_manifest = args.manifest.parent / "explicit.jsonl"
            args.starts_manifest.write_text("\n".join(map(json.dumps, rows[:5])))
            with contextlib.redirect_stdout(io.StringIO()):
                plan, _, _ = native.prepare(args)
                self.assertEqual([s["scene"] for s in plan["starts"]], [r["scene"] for r in rows[:5]])
                before = (args.output / "plan.json").read_bytes()
                rows[0]["prompt"] = "Changed prompt"
                args.starts_manifest.write_text("\n".join(map(json.dumps, rows[:5])))
                with self.assertRaises(ValueError):
                    native.prepare(args)
                self.assertEqual((args.output / "plan.json").read_bytes(), before)

    def test_plan_cli_needs_no_torch_gpu_or_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = fixture(Path(temporary))
            result = subprocess.run([sys.executable, str(ROOT / "utils/run_keepsake_native_roundtrip.py"),
                                     "--manifest", str(args.manifest), "--model-root", str(args.model_root),
                                     "--output", str(args.output), "--plan-only"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.count("KEEPSAKE B32:"), 10)
            self.assertFalse((args.output / "cases").exists())

    def test_score_uses_all_generated_pairs_not_input_or_turnaround(self):
        frames = [np.zeros((12, 12, 3), np.uint8) for _ in range(153)]
        for i in range(77, 152):
            frames[i][:] = 10
        frames[0][:] = 200
        frames[76][:] = 100
        frames[152][:] = 230
        learned = Mock()
        learned.compute_batch.side_effect = lambda a, b: [{"lpips_alex": .25} for _ in a]
        rows = native.score_pairs(frames, learned)
        self.assertEqual(len(rows), 75)
        self.assertAlmostEqual(np.mean([r["psnr_db"] for r in rows]), 20 * np.log10(255 / 10))
        self.assertEqual({r["lpips"] for r in rows}, {.25})
        frames[100] = np.zeros((3, 3, 3), np.uint8)
        with self.assertRaises(ValueError):
            native.score_pairs(frames, learned)

    def test_roundtrip_fvd_clips_are_pose_aligned_not_dataset_gt(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(153)]
        a, b, clips = native.roundtrip_features(frames, FakeFVD())
        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(b.shape, (4, 3))
        self.assertEqual(len(clips), 4)
        self.assertTrue(all(i + j == 152 and 0 < i < 76 < j < 152 for clip in clips for i, j in clip))
        self.assertTrue(all(len(c) == 16 for c in clips))

    def test_receipts_reject_tampering_and_profiles_require_b32(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case = dict(id="test", angle=90, frames=153, start={})
            self.assertIsNone(native.load_completed(case, directory))
            record = fixture_generation(None, case, directory)
            self.assertEqual(native.load_completed(case, directory), record)
            (directory / record["attempt"] / "frames/0002.png").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                native.load_completed(case, directory)
            path = directory / "bad_profile.jsonl"
            records = profile_records(case)
            records[1]["stored_memory_size"] = 33
            path.write_text("\n".join(map(json.dumps, records)))
            with self.assertRaises(ValueError):
                native.profile_summary(path, case)

    def test_generation_passes_only_keepsake_and_preserves_exact_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            Image.new("RGB", (640, 352)).save(image_path)
            case = dict(id="test", angle=90, frames=153,
                        start=dict(camera={}, scene="test", start_frame=0, input_image=str(image_path), prompt="test"))
            def fake_pipe(**kwargs):
                Path(kwargs["profile_path"]).write_text("\n".join(map(json.dumps, profile_records(case))))
                Path(kwargs["access_trace_path"]).write_text('{}\n')
                return [Image.new("RGB", (640, 352))] * 153
            pipe = Mock(side_effect=fake_pipe)
            poses_module = types.SimpleNamespace(compute_c2w_matrix=lambda *a, **k: np.eye(4))
            diffsynth = types.SimpleNamespace(save_video=lambda frames, path, **k: Path(path).write_bytes(b"video"))
            torch = Mock()
            with patch.dict(sys.modules, {"dataset.poses": poses_module, "diffsynth": diffsynth, "torch": torch}), \
                    patch.object(native, "verify_video"):
                record = native.generate_case(pipe, case, root)
            torch.manual_seed.assert_called_once_with(42)
            call = pipe.call_args.kwargs
            self.assertEqual(call["memory_policy"], "slam_covisibility")
            self.assertEqual(call["memory_budget"], 32)
            self.assertEqual(call["num_inference_steps"], 50)
            self.assertEqual(call["keepsake_geometry_weight"], .65)
            np.testing.assert_array_equal(call["c2ws"], call["c2ws"][::-1])
            self.assertEqual(native.load_completed(case, root), record)

    def test_end_to_end_export_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = fixture(Path(temporary))
            learned = Mock()
            learned.compute_batch.side_effect = lambda a, b: [{"lpips_alex": .2} for _ in a]
            model_setup = Mock(return_value=Mock())
            torch = Mock()
            with patch.dict(sys.modules, {"torch": torch,
                                          "inference_memcam": types.SimpleNamespace(setup_pipeline=model_setup)}), \
                    patch.object(native, "check_device"), patch.object(native, "assert_video_writer_available"), \
                    patch.object(native, "metric_models", return_value=(learned, FakeFVD(), {"test": True})), \
                    patch.object(native, "generate_case", side_effect=fixture_generation) as generate, \
                    contextlib.redirect_stdout(io.StringIO()):
                native.execute(args)
                self.assertEqual(generate.call_count, 10)
                self.assertEqual(model_setup.call_count, 1)
                native.execute(args)
                self.assertEqual(generate.call_count, 10)
                self.assertEqual(model_setup.call_count, 1)
            self.assertTrue((args.output / "scores.csv").is_file())
            self.assertTrue((args.output / "roundtrip_fvd_diagnostic.csv").is_file())
            table = (args.output / "table.tex").read_text()
            self.assertNotIn("FVD", table)
            self.assertNotIn(r"\textbf", table)
            self.assertIn("unverified", table)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "complete")
            with self.assertRaises(ValueError):
                native.export_results(args.output, [], {}, FakeFVD())

    def test_launcher_does_not_submit_other_jobs_or_change_gpu_mask(self):
        path = ROOT / "slurm/newton_keepsake_native_roundtrip.sbatch"
        script = path.read_text()
        for forbidden in ("conda run", "conda activate", "srun ", "sbatch ", "export CUDA_VISIBLE_DEVICES", "timeout "):
            self.assertNotIn(forbidden, script)
        subprocess.run(["bash", "-n", str(path)], check=True)

    def test_interrupted_generation_resumes_only_unfinished_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = fixture(Path(temporary))
            calls = []
            def interrupt(pipe, case, directory):
                calls.append(case["id"])
                if len(calls) == 3:
                    raise RuntimeError("simulated interruption")
                return fixture_generation(pipe, case, directory)
            learned = Mock()
            learned.compute_batch.side_effect = lambda a, b: [{"lpips_alex": .2} for _ in a]
            with patch.dict(sys.modules, {"torch": Mock(),
                                          "inference_memcam": types.SimpleNamespace(setup_pipeline=Mock())}), \
                    patch.object(native, "check_device"), patch.object(native, "assert_video_writer_available"), \
                    patch.object(native, "metric_models", return_value=(learned, FakeFVD(), {})), \
                    patch.object(native, "generate_case", side_effect=interrupt), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    native.execute(args)
                self.assertFalse((args.output / "scores.csv").exists())
                self.assertEqual(len(list((args.output / "cases").glob("*/generation.json"))), 2)
                native.execute(args)
            self.assertEqual(len(calls), 11)
            self.assertEqual(calls.count(calls[0]), 1)
            self.assertEqual(calls.count(calls[1]), 1)
            self.assertEqual(calls.count(calls[2]), 2)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "complete")

    def test_ssim_dependency_failure_is_explicit_before_model_loading(self):
        with patch.object(native.metrics, "cv2", None), \
                patch.object(native.metrics, "LearnedMetricRunner") as learned:
            with self.assertRaisesRegex(RuntimeError, "no global-SSIM fallback"):
                native.metric_models(argparse.Namespace(), "cpu")
            learned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
