import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
import run_budget_metric_grid as grid


class GridTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "manifest.jsonl"
        items = [{"duration_sec": 10, "output_prefix": "short_", "num_frames": 301}]
        items += [{"duration_sec": 60, "output_prefix": f"seed0_scene{i}_60s_", "num_frames": 1799}
                  for i in range(15)]
        manifest.write_text("\n".join(map(json.dumps, items)))
        names = [r["output_prefix"] + "custom.mp4" for r in items[1:]]
        for run in grid.RUNS:
            folder = root / "context_memory_60s" / run
            folder.mkdir(parents=True)
            for name in names + ["seed1_extra_60s_custom.mp4"]:
                (folder / name).write_bytes(b"fixture")
        args = argparse.Namespace(root=root, manifest=manifest, output=root / "grid",
                                  vbench_root=root / "VBench", dataset_root=None, reuse_audit=None,
                                  fresh_suites=False, submit=False, parallel=2)
        return args, names

    def quality(self, directory, run, names):
        directory.mkdir(parents=True)
        config = {"max_frames": None, **grid.QUALITY_CONFIG["LPIPS"], **grid.QUALITY_CONFIG["FVD"]}
        rows = [{"output": f"/videos/{run}/{n}", "duration_sec": 60,
                 "status": "completed", "lpips_alex": .5} for n in names]
        (directory / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        summary = {"metric_config": config, "by_duration": {"60": {
            "completed_or_short": 15, "fvd": 500., "fvd_clips": 60, "fvd_detector_path": "/cache/i3d.pt"}}}
        grid.save(directory / "summary.json", summary)
        return directory / "summary.json"

    def test_grid_and_exact_staging(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            grid.prepare(args)
            plan = grid.load(args.output / "plan.json")
            self.assertEqual(len(plan["tasks"]), 84)
            self.assertEqual(len(grid.RUNS), 21)
            self.assertEqual(plan["rows"], list(range(1, 16)))
            self.assertEqual({t["evaluator"] for t in plan["tasks"]}, set(grid.EVALUATORS))
            staged = grid.stage(plan, plan["tasks"][1], root / "stage")
            self.assertEqual(set(p.name for p in staged.iterdir()), set(names))
            self.assertTrue(all(p.is_symlink() for p in staged.iterdir()))
            with self.assertRaisesRegex(ValueError, "already exists"):
                grid.prepare(args)

    def test_quality_membership_config_and_distribution_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = [f"scene{i}.mp4" for i in range(15)]
            source = self.quality(root / "quality", "baseline", names)
            self.assertEqual(grid.quality_result(source, "LPIPS", "baseline", names, 1799)["value"], .5)
            self.assertEqual(grid.quality_result(source, "FVD", "baseline", names, 1799)["value"], 500.)
            summary = grid.load(source)
            summary["metric_config"]["max_frames"] = 301
            grid.save(source, summary)
            with self.assertRaisesRegex(ValueError, "truncated"):
                grid.quality_result(source, "FVD", "baseline", names, 1799)
            summary["metric_config"]["max_frames"] = None
            summary["by_duration"]["60"]["fvd_clips"] = 59
            grid.save(source, summary)
            with self.assertRaisesRegex(ValueError, "60 clips"):
                grid.quality_result(source, "FVD", "baseline", names, 1799)
            summary["by_duration"]["60"]["fvd_clips"] = 60
            grid.save(source, summary)
            rows = [json.loads(s) for s in source.with_name("metrics.jsonl").read_text().splitlines()]
            rows[0]["output"] = "/videos/fifo_b32/scene0.mp4"
            source.with_name("metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "Mixed policies"):
                grid.quality_result(source, "FVD", "baseline", names, 1799)

    def test_legacy_reuse_skips_quality_only_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            source = self.quality(root / "old", "baseline", names)
            record = {"source": str(source), "state": "EXACT", "covered": names}
            audit = {"expected": names, "standard": {"baseline": {"LPIPS": [record], "FVD": [record]}}}
            args.reuse_audit = root / "audit.json"
            grid.save(args.reuse_audit, audit)
            grid.prepare(args)
            plan = grid.load(args.output / "plan.json")
            task = plan["tasks"][0]
            work = grid.task_dir(plan, task)
            self.assertEqual(grid.load(work / "status.json")["status"], "complete")
            self.assertEqual(len(grid.validate_task(plan, task, work)), 2)
            self.assertFalse((grid.task_dir(plan, plan["tasks"][1]) / "status.json").exists())
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                grid.validate_task(plan, task, work)

    def test_invalid_legacy_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = [f"scene{i}.mp4" for i in range(15)]
            source = self.quality(root / "quality", "baseline", names)
            summary = grid.load(source)
            summary["metric_config"]["fvd_image_size"] = 112
            grid.save(source, summary)
            record = {"source": str(source), "state": "EXACT", "covered": names}
            audit = {"standard": {"baseline": {"LPIPS": [record], "FVD": [record]}}}
            self.assertEqual(set(grid.legacy_quality(audit, "baseline", names, 1799)), {"LPIPS"})

    def test_bench_normalization_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = [f"scene{i}.mp4" for i in range(15)]
            data = {dim: [.5, [{"video_path": f"/input/baseline/{n}",
                              "video_results": 50. if dim == "imaging_quality" else .5}
                             for n in names]] for dim in grid.DIMENSIONS}
            grid.save(root / "test_eval_results.json", data)
            result = grid.validate_bench(root, names)
            self.assertTrue(all(r["value"] == .5 for r in result))
            data["subject_consistency"][1][0]["video_path"] = "/input/baseline/scene1.mp4"
            grid.save(root / "test_eval_results.json", data)
            with self.assertRaisesRegex(ValueError, "cohort"):
                grid.validate_bench(root, names)

    def test_changed_video_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            grid.prepare(args)
            plan = grid.load(args.output / "plan.json")
            grid.check_sources(plan, "baseline")
            (root / "context_memory_60s/baseline" / names[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Video changed"):
                grid.check_sources(plan, "baseline")

    def test_worker_standard_success_and_failure_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            grid.prepare(args)
            plan_path = args.output / "plan.json"
            plan = grid.load(plan_path)

            def evaluate(command, work, cwd):
                output = Path(command[command.index("--output_path") + 1])
                payload = {dim: [.5, [{"video_path": f"/input/baseline/{n}",
                                     "video_results": 50. if dim == "imaging_quality" else .5}
                                    for n in names]] for dim in grid.DIMENSIONS}
                grid.save(output / "test_eval_results.json", payload)

            with patch.object(grid, "verify_lengths"), patch.object(grid, "freeze_environment", return_value="freeze"), \
                 patch.object(grid.subprocess, "run"), patch.object(grid, "run_logged", side_effect=evaluate):
                grid.run_task(plan_path, 1)
            status = grid.load(grid.task_dir(plan, plan["tasks"][1]) / "status.json")
            self.assertEqual(status["status"], "complete")
            self.assertEqual(len(status["scores"]), 6)
            with patch.object(grid, "run_logged") as call:
                grid.run_task(plan_path, 1)
                call.assert_not_called()
            with patch.object(grid, "verify_lengths", side_effect=ValueError("bad video")):
                with self.assertRaisesRegex(ValueError, "bad video"):
                    grid.run_task(plan_path, 5)
            self.assertEqual(grid.load(grid.task_dir(plan, plan["tasks"][5]) / "status.json")["status"], "failed")

    def test_worker_quality_reuses_completed_without_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            source = self.quality(root / "old", "baseline", names)
            record = {"source": str(source), "state": "EXACT", "covered": names}
            args.reuse_audit = root / "audit.json"
            grid.save(args.reuse_audit, {"expected": names, "standard": {
                "baseline": {"LPIPS": [record], "FVD": [record]}}})
            grid.prepare(args)
            with patch.object(grid.subprocess, "run") as call:
                grid.run_task(args.output / "plan.json", 0)
                call.assert_not_called()

    def test_frame_count_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, _ = self.fixture(root)
            grid.prepare(args)
            plan = grid.load(args.output / "plan.json")
            with patch.object(grid.subprocess, "check_output", return_value='{"streams":[{"nb_read_frames":"301"}]}'):
                with self.assertRaisesRegex(ValueError, "wrong-length"):
                    grid.verify_lengths(plan, plan["tasks"][1], args.output)

    def test_submit_retries_only_pending_and_adds_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, _ = self.fixture(root)
            grid.prepare(args)
            plan = grid.load(args.output / "plan.json")
            work = grid.task_dir(plan, plan["tasks"][0])
            work.mkdir()
            grid.save(work / "status.json", {"status": "complete"})
            with patch.object(grid.subprocess, "check_output", side_effect=["12345\n", "12346\n"]) as call:
                grid.submit(args.output / "plan.json", 2)
            command = call.call_args_list[0].args[0]
            array = command[command.index("--array") + 1]
            self.assertTrue(array.startswith("1,2,3,"))
            self.assertTrue(array.endswith("%2"))
            self.assertIn("afterany:12345", call.call_args_list[1].args[0])

    def test_missing_video_fails_before_submission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, names = self.fixture(root)
            (root / "context_memory_60s/kcenter_b128" / names[0]).unlink()
            args.submit = True
            with patch.object(grid.subprocess, "check_output") as call:
                with self.assertRaisesRegex(ValueError, "missing/empty"):
                    grid.prepare(args)
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
