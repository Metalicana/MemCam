from argparse import Namespace
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper import finish_keepsake_pose_appearance as study


def fixture(root):
    items = []
    control = root / "existing"
    (control / "access_traces").mkdir(parents=True)
    image, pose, gt = root / "image.png", root / "pose.json", root / "gt"
    image.write_bytes(b"fixture")
    pose.write_text("{}")
    gt.mkdir()
    for row in range(15):
        prefix = f"seed0_scene{row}_180s_"
        items.append(dict(duration_sec=180, num_frames=5397, fps=30, output_prefix=prefix,
                          scene=f"scene{row}", start_frame=0, input_image=str(image),
                          pose_path=str(pose), gt_frames_dir=str(gt)))
        (control / f"{prefix}custom.mp4").write_bytes(b"original")
        (control / "access_traces" / f"{prefix}custom.jsonl").write_text('{}\n')
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, items)))
    output = root / "experiment"
    output.mkdir()
    return Namespace(output=output, manifest=manifest, duration=180, control=control,
                     vbench_root=root / "VBench", worker=None)


def quality_artifacts(directory, items, run, duration):
    directory.mkdir(parents=True)
    rows = [dict(output=i["output_prefix"] + "custom.mp4", row=i["_row"], scene=i["scene"],
                 start_frame=i["start_frame"], run_name=run, duration_sec=duration,
                 status="completed", lpips_alex=.5) for i in items]
    (directory / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    config = {k: v for d in study.QUALITY_CONFIG.values() for k, v in d.items()}
    config.update(source_duration=duration, eval_durations=[duration], max_frames=None)
    summary = dict(metric_config=config, by_duration={str(duration): dict(
        completed_or_short=15, fvd_clips=60, fvd=700., fvd_detector_path="fixture.pt")})
    study.save(directory / "summary.json", summary)


class CompleteAblationTests(unittest.TestCase):
    def test_quality_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            items = study.cohort(args.manifest, 180)
            directory = args.output / "quality"
            quality_artifacts(directory, items, "control", 180)
            self.assertEqual(study.validate_quality(directory, items, "control", 180), {"LPIPS": .5, "FVD": 700.})
            summary = study.load(directory / "summary.json")
            summary["by_duration"]["180"]["fvd_clips"] = 4
            study.save(directory / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "60-clip"):
                study.validate_quality(directory, items, "control", 180)

    def test_wrong_horizon_and_duplicate_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with self.assertRaises(ValueError):
                study.cohort(args.manifest, 60)
            items = study.cohort(args.manifest, 180)
            items[-1] = items[0]
            args.manifest.write_text("\n".join(map(json.dumps, items)))
            with self.assertRaises(ValueError):
                study.cohort(args.manifest, 180)

    def test_resume_checks_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, receipt = root / "video.mp4", root / "step.json"
            artifact.write_bytes(b"original")
            study.save(receipt, dict(status="complete", inputs={"alpha": 1}, artifacts=study.fingerprint([artifact])))
            self.assertIsNotNone(study.cached_step(receipt, {"alpha": 1}))
            self.assertIsNone(study.cached_step(receipt, {"alpha": 0}))
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                study.cached_step(receipt, {"alpha": 1})

    def test_complete_workflow_generates_only_endpoints_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            generation_calls, metric_calls = [], []

            def fake_run(cmd, log, env=None, cwd=None):
                cmd = list(map(str, cmd))
                def arg(name):
                    return cmd[cmd.index(name)+1]
                if any(s.endswith("run_context_memory_batch.py") for s in cmd):
                    source = Path(arg("--output_dir"))
                    alpha, row = float(arg("--keepsake_geometry_weight")), int(arg("--rows"))
                    self.assertNotEqual(alpha, .65)
                    self.assertEqual(arg("--seed"), "42")
                    self.assertEqual(arg("--num_inference_steps"), "50")
                    generation_calls.append((source.name, row, alpha, env["CUDA_VISIBLE_DEVICES"]))
                    item = [i for i in study.cohort(Path(arg("--manifest")), 180) if i["_row"] == row][0]
                    (source / "access_traces").mkdir(parents=True, exist_ok=True)
                    (source / (item["output_prefix"] + "custom.mp4")).write_bytes(b"generated")
                    (source / "access_traces" / (item["output_prefix"] + "custom.jsonl")).write_text(json.dumps(
                        dict(event="context_access", keepsake_geometry_weight=alpha)) + "\n")
                elif any(s.endswith("evaluate_context_memory_prefix_curves.py") for s in cmd):
                    run = arg("--run_name")
                    metric_calls.append((run, "quality"))
                    items = study.cohort(Path(arg("--manifest")), 180)
                    quality_artifacts(Path(arg("--metrics_dir")) / run, items, run, 180)
                elif "--videos_path" in cmd:
                    source = Path(arg("--videos_path"))
                    metric_calls.append((source.name, "vbench"))
                    result = Path(arg("--output_path"))
                    payload = {dim: [0, [dict(video_path=str(p), video_results=50 if dim == "imaging_quality" else .5)
                                         for p in sorted(source.glob("*.mp4"))]] for dim in study.DIMENSIONS}
                    study.save(result / "fixture_eval_results.json", payload)

            class FakeWorker:
                def __init__(self, cmd, **kwargs):
                    worker_args = Namespace(**vars(args))
                    worker_args.worker = cmd[cmd.index("--worker")+1]
                    with patch.dict(os.environ, kwargs["env"], clear=True):
                        study.worker(worker_args)
                    self.returncode = 0
                def poll(self):
                    return self.returncode
                def wait(self, timeout=None):
                    return self.returncode

            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2,5"), \
                    patch.object(study, "metric_command", side_effect=lambda name, program, *cmd: [program, *map(str, cmd)]), \
                    patch.object(study, "run_command", side_effect=fake_run), \
                    patch.object(study, "probe_video", return_value={}), \
                    patch.object(study, "audit_trace", return_value={}), \
                    patch.object(study, "generation_assets", return_value={}), \
                    patch.object(study, "freeze_environment", return_value="fixture"), \
                    patch.object(study.subprocess, "Popen", FakeWorker):
                study.execute(args)
                self.assertEqual(len(generation_calls), 30)
                self.assertEqual(metric_calls[:2], [("control", "quality"), ("control", "vbench")])
                self.assertEqual(len(metric_calls), 6)
                self.assertEqual({c[3] for c in generation_calls if c[0] == "appearance_only"}, {"2"})
                self.assertEqual({c[3] for c in generation_calls if c[0] == "pose_only"}, {"5"})
                self.assertTrue((args.output / "ablation.tex").exists())
                with (args.output / "ablation.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertEqual([r["geometry_weight"] for r in rows], ["0.0", "0.65", "1.0"])
                study.execute(args)
                self.assertEqual(len(generation_calls), 30, "Resume must not regenerate completed videos")
                self.assertEqual(len(metric_calls), 6, "Resume must not recompute completed metrics")

    def test_control_failure_prevents_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"), \
                    patch.object(study, "prepare", return_value=([], args.manifest)), \
                    patch.object(study, "metric_command", return_value=["fixture"]), \
                    patch.object(study, "run_command"), patch.object(study, "freeze_environment"), \
                    patch.object(study, "evaluate", side_effect=RuntimeError("control evaluator failed")), \
                    patch.object(study.subprocess, "Popen") as start:
                with self.assertRaisesRegex(RuntimeError, "control evaluator failed"):
                    study.execute(args)
                start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
