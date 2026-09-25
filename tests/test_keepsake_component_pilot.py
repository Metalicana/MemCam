from argparse import Namespace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from paper import run_keepsake_component_pilot as pilot


def fixture(root):
    image, pose, gt = root / "image.png", root / "poses.json", root / "gt"
    image.write_bytes(b"image")
    pose.write_text("{}")
    gt.mkdir()
    for frame in range(0, pilot.FRAMES, 30):
        (gt / f"{frame:04d}.png").write_bytes(b"frame")
    items = [dict(scene=f"scene{i}", duration_sec=60, num_frames=1825, fps=30, start_frame=0,
                  output_prefix=f"scene{i}_60s_", input_image=str(image), pose_path=str(pose),
                  gt_frames_dir=str(gt)) for i in range(15)]
    manifest = root / "source.jsonl"
    manifest.write_text("\n".join(map(json.dumps, items)))
    output = root / "output"
    output.mkdir()
    return Namespace(manifest=manifest, output=output, duration=30)


def quality_files(directory, item, run):
    directory.mkdir(parents=True)
    row = dict(status="completed", run_name=run, row=item["_row"], scene=item["scene"],
               start_frame=item["start_frame"], duration_sec=30, num_frames_expected=913,
               output=item["output_prefix"] + "custom.mp4", frames_evaluated=31,
               lpips_alex=.4, psnr_db=17., ssim=.6)
    (directory / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    pilot.common.save(directory / "summary.json", {"metric_config": dict(source_duration=30,
        eval_durations=[30], max_frames=None, frame_stride=30, learned_image_size=224, learned_metrics=["lpips"])})


class ComponentPilotTests(unittest.TestCase):
    def test_single_gpu_launcher_preserves_mask_and_exit_code(self):
        launcher = pilot.REPO / "slurm/newton_keepsake_component_pilot.sbatch"
        text = launcher.read_text()
        self.assertIn("#SBATCH --gres=gpu:nvidia_h100_pcie:1", text)
        self.assertIn("#SBATCH --time=15:00:00", text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binaries = root / "bin"
            binaries.mkdir()
            for name, script in {
                "module": "#!/bin/bash\nexit 0\n",
                "nvidia-smi": "#!/bin/bash\nexit 0\n",
                "python": '#!/bin/bash\nprintf "mask=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\nexit 7\n',
            }.items():
                path = binaries / name
                path.write_text(script)
                path.chmod(0o755)
            env = dict(os.environ, PATH=f"{binaries}:{os.environ.get('PATH', '')}", MEMCAM_ROOT=str(root),
                       MEMCAM_ENV_PATH=str(root), CUDA_VISIBLE_DEVICES="GPU-fixture", SLURM_JOB_ID="fixture")
            env.pop("BASH_ENV", None)
            result = subprocess.run(["bash", str(launcher)], text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn("mask=GPU-fixture", result.stdout)
            self.assertIn("paper/run_keepsake_component_pilot.py", result.stdout)

    def test_fixed_cohort_and_no_generation_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            items = pilot.select_cohort(args.manifest)
            self.assertEqual(items, pilot.select_cohort(args.manifest))
            self.assertEqual(len(items), 3)
            self.assertEqual(len({i["scene"] for i in items}), 3)
            self.assertTrue(all(i["duration_sec"] == 30 and i["num_frames"] == 913 for i in items))
            unused = Path(tmp) / "unused"
            result = subprocess.run([sys.executable, str(Path(pilot.__file__)), "--manifest", str(args.manifest),
                                     "--output", str(unused), "--dry-run"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["new_videos"], 15)
            self.assertFalse(unused.exists())

    def test_missing_gt_and_changed_config_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with patch.object(pilot.common, "generation_assets", return_value={}):
                pilot.prepare(args)
                pilot.prepare(args)
                config = args.output / "config.json"
                data = pilot.common.load(config)
                data["seed"] = 7
                pilot.common.save(config, data)
                with self.assertRaisesRegex(ValueError, "inputs/code changed"):
                    pilot.prepare(args)
            (Path(tmp) / "gt/0000.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing sampled GT"):
                pilot.prepare(args)

    def test_commands_hold_all_other_settings_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = pilot.select_cohort(args.manifest)[0]
            with patch.object(pilot.common, "metric_command", side_effect=lambda env, program, *xs: [program, *map(str, xs)]):
                for setting in pilot.SETTINGS:
                    cmd = pilot.generation_command(args, args.manifest, setting, item)
                    self.assertEqual(cmd[cmd.index("--keepsake_geometry_weight")+1], str(setting[1]))
                    self.assertEqual(cmd[cmd.index("--keepsake_priority_mode")+1], setting[2])
                    for flag, value in (("--durations", "30"), ("--seed", "42"), ("--num_inference_steps", "50"),
                                        ("--memory_budget", "32")):
                        self.assertEqual(cmd[cmd.index(flag)+1], value)
                cmd = pilot.quality_command(args, args.manifest, args.output, args.output, "full", item)
                self.assertEqual(cmd[cmd.index("--learned_metrics")+1], "lpips")

    def test_quality_contract_and_incomplete_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = pilot.select_cohort(args.manifest)[0]
            directory = args.output / "quality"
            quality_files(directory, item, "full")
            self.assertEqual(pilot.validate_quality(directory, item, "full")["lpips_alex"], .4)
            with self.assertRaises(ValueError):
                pilot.validate_quality(directory, item, "pose_only")
            with self.assertRaises(ValueError):
                pilot.export(args.output, [])
            path = directory / "metrics.jsonl"
            row = json.loads(path.read_text())
            row["frames_evaluated"] = 30
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                pilot.validate_quality(directory, item, "full")

    def test_full_workflow_partial_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            calls, failure = [], [True]

            def fake_run(cmd, *unused):
                cmd = list(map(str, cmd))
                if "-c" in cmd:
                    return
                def arg(flag):
                    return cmd[cmd.index(flag)+1]
                item = [json.loads(line) for line in Path(arg("--manifest")).read_text().splitlines()][int(arg("--rows"))]
                if "--output_dir" in cmd:
                    source = Path(arg("--output_dir"))
                    (source / "access_traces").mkdir(parents=True, exist_ok=True)
                    (source / (item["output_prefix"] + "custom.mp4")).write_bytes(b"generated")
                    event = dict(event="context_access", keepsake_geometry_weight=float(arg("--keepsake_geometry_weight")),
                                 keepsake_priority_mode=arg("--keepsake_priority_mode"))
                    (source / "access_traces" / (item["output_prefix"] + "custom.jsonl")).write_text(json.dumps(event))
                    calls.append((source.name, item["_row"]))
                else:
                    run = arg("--run_name")
                    if run == "pose_only" and failure[0]:
                        failure[0] = False
                        raise RuntimeError("metric failure fixture")
                    quality_files(Path(arg("--metrics_dir")) / run, item, run)

            with patch.dict(os.environ, SLURM_JOB_ID="fixture"), \
                    patch.object(pilot.common, "generation_assets", return_value={}), \
                    patch.object(pilot.common, "freeze_environment"), \
                    patch.object(pilot.common, "metric_command", side_effect=lambda env, program, *xs: [program, *xs]), \
                    patch.object(pilot.common, "run_command", side_effect=fake_run), \
                    patch.object(pilot.common, "audit_trace"), patch.object(pilot.common, "probe_video"):
                with self.assertRaisesRegex(RuntimeError, "metric failure fixture"):
                    pilot.execute(args)
                self.assertEqual(len(calls), 2)
                self.assertFalse((args.output / "ablation.csv").exists())
                pilot.execute(args)
                self.assertEqual(len(calls), 15)
                self.assertEqual(len(set(calls)), 15)
                pilot.execute(args)
                self.assertEqual(len(calls), 15, "Resume must keep completed generation")
            with (args.output / "ablation.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(int(r["videos"]) == 3 for r in rows))
            self.assertTrue(all(float(r["delta_lpips_alex_vs_full"]) == 0 for r in rows))
            self.assertTrue((args.output / "ablation.tex").exists())

    def test_rejects_wrong_priority_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = pilot.select_cohort(args.manifest)[0]
            source = args.output / "videos/full"
            (source / "access_traces").mkdir(parents=True)
            trace = source / "access_traces" / (item["output_prefix"] + "custom.jsonl")
            trace.write_text(json.dumps(dict(event="context_access", keepsake_priority_mode="full")))
            with patch.object(pilot.common, "validate_generated", return_value={}):
                with self.assertRaisesRegex(ValueError, "priority component"):
                    pilot.validate_generated(source, item, .65, "degree_only")


if __name__ == "__main__":
    unittest.main()
