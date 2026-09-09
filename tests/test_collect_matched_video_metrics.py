import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
import collect_matched_video_metrics as collector
sys.path.pop(0)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def suite(self, run, evaluator, suffix="first"):
        work = self.root / f"{run}_{evaluator}_{suffix}"
        work.mkdir()
        items = [{"duration_sec": 60, "scene": f"scene{i}", "start_frame": 0,
                  "output_prefix": f"seed0_scene{i}_60s_"} for i in range(15)]
        (work / "manifest.jsonl").write_text("\n".join(map(json.dumps, items)))
        save(work / "suite_status.json", {
            "run": run, "evaluator": evaluator, "rows": list(range(15)), "dry_run": False,
            "results": [{"row": i, "exit_code": 0} for i in range(15)]})
        for i, item in enumerate(items):
            directory = work / f"row_{i:03d}/smoke_example"
            name = item["output_prefix"] + "custom.mp4"
            save(directory / "spec.json", {"row": i, "manifest_item": item,
                                          "source_video": f"/remote/videos/{run}/{name}", "dry_run": False})
            save(directory / "status.json", {evaluator: {"status": "passed"}})
            if evaluator == "vbench-long":
                payload = {dim: [.75, [{"video_path": name, "video_results": .75}]]
                           for dim in collector.DIMENSIONS}
                payload["imaging_quality"] = [.65, [], [{"video_path": name, "video_results": 65.}]]
                save(directory / "vbench_long_results/example_eval_results.json", payload)
            else:
                save(directory / "cut3r_metrics/cut3r_camera_failures.json", [])
                save(directory / "cut3r_metrics/cut3r_camera_metrics.json", [{
                    "run_name": run, "row": i, "scene": item["scene"], "start_frame": 0,
                    "duration_sec": 60, "rotation_error_deg_mean": float(i),
                    "translation_error_scale_only_mean": 2., "translation_error_sim3_mean": 1.,
                    "path_length_ratio_scale_only": None if i == 0 else 1., "sampled_frames": 61,
                }])
        return work

    def test_full_suite_normalization_and_all_numeric_scores(self):
        for run in collector.POLICIES:
            for evaluator in collector.EVALUATORS:
                self.suite(run, evaluator)
        report = collector.collect(self.root)
        self.assertTrue(all(r["complete"] for r in report["coverage"]))
        self.assertEqual(len(report["per_video"]), 120)
        scores = {(r["run"], r["metric"]): r for r in report["summary"]}
        self.assertEqual(scores["baseline", "imaging_quality"]["mean"], .65)
        self.assertEqual(scores["baseline", "rotation_error_deg_mean"]["mean"], 7.)
        self.assertEqual(scores["baseline", "path_length_ratio_scale_only"]["n"], 14)
        self.assertNotIn(("baseline", "start_frame"), scores)
        text = collector.markdown(report)
        self.assertIn("15/15", text)
        self.assertIn("provisional", text)
        self.assertIn("(n=14)", text)
        result = subprocess.run([sys.executable, str(UTILS / "collect_matched_video_metrics.py"),
                                 "--root", str(self.root)], capture_output=True, text=True, check=True)
        self.assertIn("GeoCov-32", result.stdout)
        self.assertTrue((self.root / "report/vbench-long_per_video.csv").is_file())
        self.assertTrue((self.root / "report/cut3r_summary.csv").is_file())

    def test_latest_partial_attempt_does_not_reuse_old_success(self):
        old = self.suite("baseline", "vbench-long")
        new = self.suite("baseline", "vbench-long", "retry")
        path = new / "suite_status.json"
        data = json.loads(path.read_text())
        data["results"][-1]["exit_code"] = 1
        save(path, data)
        os.utime(old / "suite_status.json", (10, 10))
        os.utime(path, (20, 20))
        report = collector.collect(self.root)
        self.assertEqual(report["coverage"][0]["valid"], 14)
        self.assertFalse(report["summary"])
        self.assertTrue(report["warnings"])

    def test_wrong_identity_and_mismatched_cohort(self):
        work = self.suite("baseline", "vbench-long")
        path = work / "row_000/smoke_example/spec.json"
        spec = json.loads(path.read_text())
        spec["source_video"] = "/remote/videos/baseline/seed1_extra60s_custom.mp4"
        save(path, spec)
        other = self.suite("fifo_b32", "vbench-long")
        manifest = other / "manifest.jsonl"
        items = [json.loads(line) for line in manifest.read_text().splitlines()]
        items[0]["start_frame"] = 100
        manifest.write_text("\n".join(map(json.dumps, items)))
        report = collector.collect(self.root)
        self.assertEqual(report["coverage"][0]["valid"], 14)
        fifo = next(r for r in report["coverage"] if r["run"] == "fifo_b32"
                    and r["evaluator"] == "vbench-long")
        self.assertFalse(fifo["complete"])
        self.assertIn("Cohort differs", fifo["issues"][0])


if __name__ == "__main__":
    unittest.main()
