import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper import keepsake_component_analysis as analysis
from paper import run_keepsake_component_study as study
from paper import submit_keepsake_component_study as submitter


def plan_fixture(root):
    root.mkdir(exist_ok=True)
    source = root / "source.jsonl"
    scene_rows = []
    for i in range(15):
        item = dict(_row=i, scene=f"scene{i}", start_frame=100, duration_sec=60,
                    num_frames=1825, fps=30, output_prefix=f"scene{i}_60s_", prompt="test")
        poses = root / f"poses_{i}.npy"
        np.save(poses, np.tile(np.eye(4), (1825, 1, 1)))
        scene_rows.append(dict(item=item, revisits=[], poses=str(poses),
                               input_hashes={str(poses): analysis.sha(poses)}))
    source.write_text("".join(json.dumps(r["item"]) + "\n" for r in scene_rows))
    detector = root / "i3d.pt"
    detector.write_bytes(b"fixture")
    plan = dict(output=str(root), manifest=str(source), source_manifest=str(source),
                code=str(root / "code"), scenes=scene_rows, code_hashes={}, shared_hashes={},
                detector=str(detector), protocol=analysis.PROTOCOL, expected_cells=75)
    analysis.save(root / "plan.json", plan)
    return plan


def frames(item, value=.5):
    return [dict(row=item["_row"], scene=item["scene"], duration_sec=60, frame_index=i,
                 gt_frame_index=item["start_frame"] + i, lpips_alex=value, psnr_db=20., ssim=.7)
            for i in range(0, 1825, 30)]


def array_plan_fixture(root):
    plan = plan_fixture(root)
    launcher = Path(plan["code"]) / "slurm/newton_keepsake_components_full.sbatch"
    launcher.parent.mkdir(parents=True)
    launcher.write_text((study.ROOT / "slurm/newton_keepsake_components_full.sbatch").read_text())
    return plan


def synthetic_trace(path, item, alpha, mode):
    bank = {0}
    rows = []
    for section in range(24):
        start = section * 76
        if section:
            eligible = bank - set(range(start - 3, start + 1))
            for slot in range(76):
                rows.append(dict(event="context_access", selected=True, target_frame=start+slot+1,
                    section_idx=section, selected_memory_frame=0, candidate_count=len(eligible),
                    stored_memory_size=32, memory_policy="slam_covisibility", memory_budget=32,
                    scene=item["scene"], dataset_start_frame=item["start_frame"], duration_sec=60,
                    keepsake_geometry_weight=alpha, keepsake_priority_mode=mode))
        bank.update(range(start, start+77))
        retained = {0, *range(start+46, start+77)}
        for victim in sorted(bank - retained):
            rows.append(dict(event="memory_eviction", section_idx=section, evicted_memory_frame=victim))
        bank = retained
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class ComponentStudyTests(unittest.TestCase):
    def test_cli_limits_threads_before_first_numpy_import(self):
        # Use fresh interpreters: importing NumPy in this test process is too late.
        check = '''
import builtins
import json
import os
import runpy
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        limits = {key: os.environ.get(key) for key in
                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        assert set(limits.values()) == {"1"}, limits
        print("THREAD_LIMITS_OK", json.dumps(limits, sort_keys=True))
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
entry = sys.argv[1]
sys.argv = [entry, "--help"]
runpy.run_path(entry, run_name="__main__")
raise AssertionError("Did not reach numerical imports")
'''
        keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        for entry in ("submit_keepsake_component_study.py", "run_keepsake_component_study.py"):
            for inherited in (None, "32"):
                with self.subTest(entry=entry, inherited=inherited):
                    env = dict(os.environ)
                    for key in keys:
                        env.pop(key, None)
                        if inherited is not None:
                            env[key] = inherited
                    result = subprocess.run([sys.executable, "-c", check, str(study.ROOT / "paper" / entry)],
                                            env=env, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("THREAD_LIMITS_OK", result.stdout)

    def test_revisits_require_departure_and_do_not_use_initial_frame(self):
        poses = np.tile(np.eye(4), (1825, 1, 1))
        self.assertEqual(analysis.revisit_queries(poses), [])
        poses[120:480, 0, 3] = 1.
        found = analysis.revisit_queries(poses)
        self.assertIn(dict(first_frame=30, target_frame=480), found)
        self.assertTrue(all(r["first_frame"] > 0 and r["target_frame"]-r["first_frame"] >= 450 for r in found))
        poses[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rotation"):
            analysis.revisit_queries(poses)

    def test_frame_windows_and_gt_mapping(self):
        item = dict(_row=0, scene="a", start_frame=500)
        rows = frames(item)
        rows[0]["lpips_alex"] = 100
        windows = analysis.reduce_frames(rows, item, [dict(first_frame=30, target_frame=480)])
        self.assertEqual([r["sampled_frames"] for r in windows], [60, 15, 1])
        self.assertEqual(windows[0]["lpips_alex"], .5)
        rows[1]["gt_frame_index"] += 1
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            analysis.reduce_frames(rows, item, [])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analysis.reduce_frames(rows[:-1], item, [])

    def test_paired_summary_preserves_worse_full_and_missing_revisits(self):
        rows = []
        for scene in range(15):
            for name, _, _ in study.SETTINGS:
                for window in ("all", "late", "revisit"):
                    rows.append(dict(scene_index=scene, setting=name, window=window,
                        sampled_frames=0 if window == "revisit" else scene+1,
                        lpips_alex=None if window == "revisit" else scene/100 + (.2 if name == "full" else 0),
                        psnr_db=None if window == "revisit" else 20., ssim=None if window == "revisit" else .5))
        with patch.dict(analysis.PROTOCOL, bootstrap_draws=25):
            summaries, contrasts = analysis.summarize(rows)
        primary = [r for r in contrasts if r["p_holm_primary"] is not None]
        self.assertEqual(len(primary), 4)
        self.assertTrue(all(abs(r["difference"]-.2) < 1e-10 for r in primary))
        full = next(r for r in summaries if r["window"] == "all" and r["metric"] == "lpips_alex" and r["run"] == "full")
        self.assertAlmostEqual(full["mean"], .27, msg="Equal scenes, not frame-count weighting")
        self.assertTrue(all(r["mean"] is None and r["videos"] == 0 for r in summaries if r["window"] == "revisit"))
        with self.assertRaisesRegex(ValueError, "all 15"):
            analysis.summarize(rows[:-1])

    def test_fvd_sampler_and_commands(self):
        self.assertEqual(len(study.fvd_indices()), 64)
        self.assertEqual(max(study.fvd_indices()), 1824)
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_fixture(Path(tmp))
            with patch.object(study.common, "metric_command", side_effect=lambda env, program, *xs: [program, *map(str, xs)]):
                for setting in study.SETTINGS:
                    cmd = study.generation_command(plan, 0, setting, Path(tmp) / setting[0])
                    self.assertNotIn("--overwrite", cmd)
                    for flag, value in (("--durations", "60"), ("--seed", "42"), ("--num_inference_steps", "50"),
                                        ("--memory_budget", "32"), ("--keepsake_priority_mode", setting[2]),
                                        ("--keepsake_geometry_weight", str(setting[1]))):
                        self.assertEqual(cmd[cmd.index(flag)+1], value)

    def test_all_cells_metric_failure_resume_and_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = plan_fixture(root)
            generated, fail = [], [True]
            def fake_run(command, *args, **kwargs):
                cmd = list(map(str, command))
                arg = lambda key: cmd[cmd.index(key)+1]
                if "--output_dir" in cmd:
                    index = int(arg("--rows"))
                    source = Path(arg("--output_dir"))
                    name = source.name
                    item = plan["scenes"][index]["item"]
                    source.mkdir(parents=True)
                    (source / (item["output_prefix"] + "custom.mp4")).write_bytes(b"video")
                    synthetic_trace(source / "access_traces" / (item["output_prefix"] + "custom.jsonl"),
                                    item, float(arg("--keepsake_geometry_weight")), arg("--keepsake_priority_mode"))
                    generated.append((index, name))
                else:
                    index, name = int(arg("--scene")), arg("--setting")
                    if index == 0 and name == "pose_only" and fail[0]:
                        fail[0] = False
                        raise RuntimeError("metric failure fixture")
                    attempt = Path(arg("--attempt"))
                    item = plan["scenes"][index]["item"]
                    values = frames(item, .5 + index/100)
                    (attempt / "frame_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))
                    features = np.random.default_rng(index).normal(size=(4, 8))
                    np.savez(attempt / "fvd_features.npz", gt=features, generated=features+.1)
                    analysis.save(attempt / "result.json", dict(windows=analysis.reduce_frames(values, item, []),
                        lpips_weights_sha256="weights", prefix16_decoded_sha256="same"))
            with patch.object(study, "run", side_effect=fake_run), \
                    patch.object(study.common, "metric_command", side_effect=lambda env, program, *xs: [program, *xs]), \
                    patch.object(study.common, "probe_video"):
                for setting in study.SETTINGS[:2]:
                    if setting[0] == "pose_only":
                        with self.assertRaisesRegex(RuntimeError, "metric failure"):
                            study.execute_cell(plan, 0, setting)
                    else:
                        study.execute_cell(plan, 0, setting)
                self.assertEqual(len(generated), 2)
                for index in range(15):
                    for setting in study.SETTINGS:
                        if index == 14 and setting[0] == "closest_only":
                            continue
                        study.execute_cell(plan, index, setting)
                self.assertEqual(len(generated), 74)
                with self.assertRaisesRegex(ValueError, "missing cells"):
                    study.report(plan)
                self.assertFalse((root / "ablation.csv").exists())
                study.execute_cell(plan, 14, study.SETTINGS[-1])
                self.assertEqual(len(generated), 75)
                study.execute_cell(plan, 14, study.SETTINGS[-1])
                self.assertEqual(len(generated), 75)
            with patch.dict(analysis.PROTOCOL, bootstrap_draws=20), \
                    patch.dict(analysis.PROTOCOL["fvd"], bootstrap_draws=20):
                study.report(plan)
            with (root / "ablation.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(int(r["videos"]) == 15 for r in rows))
            self.assertEqual(study.load(root / "status.json")["status"], "complete")
            generation = study.load(root / "cells/scene_00/full/generation.json")
            path = next(iter(generation["artifacts"]))
            Path(path).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                study.receipt(root / "cells/scene_00/full/generation.json", study.identity(plan, 0, study.SETTINGS[0]))

    def test_submit_chain_idempotence_and_failed_job_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_fixture(Path(tmp))
            commands = []
            def accepted(cmd):
                commands.append(cmd)
                return str(100 + len(commands))
            with patch.object(submitter, "accepted", side_effect=accepted), \
                    patch.object(submitter, "job_state", return_value="PENDING"):
                jobs = submitter.submit(plan)
                self.assertEqual(len(commands), 16)
                self.assertNotIn("--dependency=afterany:101", commands[0])
                self.assertIn("--dependency=afterany:101", commands[1])
                submitter.submit(plan)
                self.assertEqual(len(commands), 16)
            # All old jobs have failed; every replacement must chain, not block on newly submitted jobs.
            with patch.object(submitter, "accepted", side_effect=accepted), \
                    patch.object(submitter, "job_state", return_value="FAILED"):
                submitter.submit(plan)
            self.assertEqual(len(commands), 32)
            self.assertIn("--dependency=afterany:117", commands[17])

    def test_submission_failure_keeps_accepted_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_fixture(Path(tmp))
            with patch.object(submitter, "accepted", side_effect=["7", RuntimeError("submission failed")]):
                with self.assertRaisesRegex(RuntimeError, "submission failed"):
                    submitter.submit(plan)
            self.assertEqual(study.load(Path(tmp) / "jobs.json")["scenes"], {"0": "7"})

    def test_array_throttle_report_dependency_and_no_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = array_plan_fixture(Path(tmp))
            with patch.object(submitter, "accepted", side_effect=["200", "201"]) as accept:
                jobs = submitter.submit_array(plan, 5)
            self.assertIn("--array=0-14%5", accept.call_args_list[0].args[0])
            self.assertIn("--dependency=afterany:200", accept.call_args_list[1].args[0])
            self.assertEqual(jobs["scenes"]["14"], "200_14")
            states = {f"200_{i}": "PENDING" for i in range(15)}
            with patch.object(submitter, "array_task_states", return_value=states), \
                    patch.object(submitter, "job_state", return_value="PENDING"), \
                    patch.object(submitter, "accepted") as accept:
                submitter.submit_array(plan, 5)
                accept.assert_not_called()
            with self.assertRaisesRegex(ValueError, "uses an array"):
                submitter.submit(plan)

    def test_array_report_submission_failure_does_not_resubmit_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = array_plan_fixture(Path(tmp))
            with patch.object(submitter, "accepted", side_effect=["200", RuntimeError("report submission")]):
                with self.assertRaisesRegex(RuntimeError, "report submission"):
                    submitter.submit_array(plan, 5)
            jobs = study.load(Path(tmp) / "array_jobs.json")
            self.assertEqual(len(jobs["scenes"]), 15)
            self.assertEqual(jobs["reports"], [])
            with patch.object(submitter, "array_task_states", return_value={f"200_{i}": "PENDING" for i in range(15)}), \
                    patch.object(submitter, "accepted", return_value="201") as accept:
                submitter.submit_array(plan, 5)
            self.assertEqual(accept.call_count, 1)
            self.assertIn("--dependency=afterany:200", accept.call_args.args[0])

    def test_array_retries_only_failed_scenes_and_never_overlaps_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = array_plan_fixture(Path(tmp))
            with patch.object(submitter, "accepted", side_effect=["200", "201"]):
                submitter.submit_array(plan, 5)
            states = {f"200_{i}": "COMPLETED" for i in range(15)}
            states["200_7"] = "FAILED"
            states["200_14"] = "RUNNING"
            with patch.object(submitter, "array_task_states", return_value=states), \
                    patch.object(submitter, "job_state", return_value="PENDING"), \
                    patch.object(submitter, "scene_complete", return_value=True), \
                    patch.object(submitter, "accepted") as accept:
                submitter.submit_array(plan, 5)
                accept.assert_not_called()
            states["200_14"] = "COMPLETED"
            with patch.object(submitter, "array_task_states", return_value=states), \
                    patch.object(submitter, "job_state", return_value="FAILED"), \
                    patch.object(submitter, "scene_complete", return_value=True), \
                    patch.object(submitter, "accepted", side_effect=["202", "203"]) as accept:
                jobs = submitter.submit_array(plan, 5)
            self.assertIn("--array=7%5", accept.call_args_list[0].args[0])
            self.assertIn("--dependency=afterany:202", accept.call_args_list[1].args[0])
            self.assertEqual(jobs["scenes"]["7"], "202_7")
            self.assertEqual(jobs["scenes"]["6"], "200_6")
            self.assertEqual(len(jobs["arrays"]), 2)

    def test_array_finished_accounting_is_not_a_complete_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = array_plan_fixture(Path(tmp))
            with patch.object(submitter, "accepted", side_effect=["200", "201"]):
                submitter.submit_array(plan, 5)
            states = {f"200_{i}": "COMPLETED" for i in range(15)}
            with patch.object(submitter, "array_task_states", return_value=states), \
                    patch.object(submitter, "job_state", return_value="FAILED"), \
                    patch.object(submitter, "scene_complete", side_effect=lambda plan, i: i != 4), \
                    patch.object(submitter, "accepted", side_effect=["202", "203"]) as accept:
                submitter.submit_array(plan, 5)
            self.assertIn("--array=4%5", accept.call_args_list[0].args[0])

    def test_array_state_uses_array_ids_live_requeue_and_accounting_lag(self):
        replies = ["200_2|PENDING\n999_1|RUNNING\n", "200_1|CANCELLED by 42\n200_2|COMPLETED\n"]
        with patch.object(submitter.subprocess, "check_output", side_effect=replies) as call:
            states = submitter.array_task_states(["200_1", "200_2"])
        self.assertEqual(states, {"200_1": "CANCELLED", "200_2": "PENDING"})
        self.assertIn("--array", call.call_args_list[1].args[0])
        self.assertIn("--format=JobID%64,State%32", call.call_args_list[1].args[0])
        with patch.object(submitter.subprocess, "check_output", side_effect=["", "201|RUNNING\n"]):
            with self.assertRaisesRegex(ValueError, "Cannot establish array state"):
                submitter.array_task_states(["200_1"])
        with patch.object(submitter.subprocess, "check_output", return_value="200_1|PENDING\n") as call:
            self.assertEqual(submitter.array_task_states(["200_1"]), {"200_1": "PENDING"})
            self.assertEqual(call.call_count, 1, "Queue visibility suffices before accounting catches up")

    def test_array_refuses_legacy_serial_submission_or_frozen_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = array_plan_fixture(root)
            analysis.save(root / "jobs.json", {"scenes": {"0": "100"}})
            with self.assertRaisesRegex(ValueError, "serial submissions"):
                submitter.submit_array(plan, 5)
        with tempfile.TemporaryDirectory() as tmp:
            plan = array_plan_fixture(Path(tmp))
            (Path(plan["code"]) / "slurm/newton_keepsake_components_full.sbatch").write_text("old launcher")
            with self.assertRaisesRegex(ValueError, "predates array support"):
                submitter.submit_array(plan, 5)
            with self.assertRaisesRegex(ValueError, "1..5"):
                submitter.submit_array(plan, 6)

    def test_current_attempt_log_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.log"
            path.write_text("old success\n")
            with self.assertRaisesRegex(RuntimeError, "new failure"):
                study.run([sys.executable, "-c", "print('new failure'); raise SystemExit(7)"], path)
            self.assertIn("ATTEMPT", path.read_text())
            with self.assertRaises(TimeoutError):
                study.run([sys.executable, "-c", "import time; time.sleep(60)"], path, timeout=.05)

    def test_prepare_freezes_whole_cohort_without_quality_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            image, pose, detector = source / "image.png", source / "poses.json", source / "i3d.pt"
            image.write_bytes(b"image")
            detector.write_bytes(b"detector")
            camera = {str(i): dict(position=[0, 0, 0], rotation=[0, 0, 0]) for i in range(1825)}
            pose.write_text(json.dumps(dict(CineCameraActor=camera)))
            gt = source / "gt"
            gt.mkdir()
            for frame in sorted(set(range(0, 1825, 30)) | set(study.fvd_indices())):
                (gt / f"{frame:04d}.png").write_bytes(b"gt")
            items = [dict(scene=f"scene{i}", start_frame=0, duration_sec=60, fps=30, num_frames=1825,
                          output_prefix=f"scene{i}_", input_image=str(image), pose_path=str(pose),
                          gt_frames_dir=str(gt), prompt="test") for i in range(15)]
            manifest = source / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r)+"\n" for r in items))
            with patch.object(study.common, "generation_assets", return_value={}), \
                    patch.object(study, "tokenizer_assets", return_value={}):
                plan = study.prepare(root / "study", manifest, detector)
                self.assertEqual(len(plan["scenes"]), 15)
                self.assertEqual(plan["expected_cells"], 75)
                self.assertTrue(all(r["revisits"] == [] for r in plan["scenes"]))
                self.assertEqual(study.prepare(root / "study", manifest, detector), plan)
                (gt / "0300.png").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    study.prepare(root / "study", manifest, detector)

    def test_fvd_bootstrap_pairs_whole_scenes(self):
        rng = np.random.default_rng(3)
        gt = rng.normal(size=(15, 4, 8))
        arrays = {"gt": gt, **{name: gt.copy() for name, _, _ in study.SETTINGS}}
        arrays["full"] = gt + 1
        with patch.dict(analysis.PROTOCOL["fvd"], bootstrap_draws=20):
            summaries, contrasts = analysis.summarize_fvd(arrays)
        self.assertTrue(all(r["difference"] > 0 for r in contrasts), "Do not suppress a worse full method")
        self.assertTrue(all(r["ci_low"] <= r["ci_high"] for r in summaries))
        arrays["full"] = arrays["full"][:14]
        with self.assertRaisesRegex(ValueError, "paired"):
            analysis.summarize_fvd(arrays)

    def test_launcher_preserves_slurm_mask_and_exit(self):
        launcher = study.ROOT / "slurm/newton_keepsake_components_full.sbatch"
        self.assertIn("gpu:nvidia_h100_80gb_hbm3:1", launcher.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "bin"
            binary.mkdir()
            for name, script in {"module": "exit 0", "nvidia-smi": "exit 0",
                "python": 'printf "mask=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "arg=%s\\n" "$@"\nexit 7'}.items():
                path = binary / name
                path.write_text("#!/bin/bash\n" + script + "\n")
                path.chmod(0o755)
            env = dict(os.environ, PATH=f"{binary}:{os.environ.get('PATH', '')}",
                       MEMCAM_ENV_PATH=str(root), CUDA_VISIBLE_DEVICES="GPU-test")
            env.pop("BASH_ENV", None)
            result = subprocess.run(["bash", str(launcher)], text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn("mask=GPU-test", result.stdout)
            env.update(SLURM_ARRAY_TASK_ID="12", SLURM_ARRAY_JOB_ID="200")
            result = subprocess.run(["bash", str(launcher), "--output", str(root)],
                                    text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn("mask=GPU-test", result.stdout)
            self.assertIn("arg=--scene\narg=12\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
