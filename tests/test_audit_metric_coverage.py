import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "audit_metric_coverage", Path(__file__).parents[1] / "utils/audit_metric_coverage.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class CoverageTests(unittest.TestCase):
    def test_partial_results_prefixes_and_split_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "context_memory_180s" / "slam_b32_covisibility"
            run.mkdir(parents=True)
            names = ["room_180s_custom.mp4", "hall_180s_custom.mp4"]
            for name in names:
                (run / name).write_bytes(b"fixture")
            rows = [{"output": str(run / names[0]), "status": "completed"},
                    {"output": str(run / names[0]), "status": "skipped"}]
            (run / "run_status.jsonl").write_text("\n".join(map(json.dumps, rows)))
            metrics = root / "eval" / run.name
            metrics.mkdir(parents=True)
            rows = [
                {"output": str(run / names[0]), "status": "completed", "duration_sec": 180, "lpips_alex": .5},
                {"output": str(run / names[1]), "status": "completed", "duration_sec": 60, "lpips_alex": .4},
                {"output": str(run / names[1]), "status": "short_video", "duration_sec": 180, "lpips_alex": .6},
            ]
            (metrics / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (metrics / "summary.json").write_text(json.dumps({
                "by_duration": {"180": {"fvd": 500, "completed_or_short": 2}}
            }))
            bench = root / "vbench_results" / run.name
            bench.mkdir(parents=True)
            payload = {dim: [.8, [{"video_path": str(run / name), "video_results": .8}
                                  for name in names]] for dim in audit.DIMENSIONS}
            payload["imaging_quality"][1][1]["video_results"] = None
            (bench / "first_eval_results.json").write_text(json.dumps(payload))
            long = root / "vbench_long_results" / run.name
            long.mkdir(parents=True)
            payload = {dim: [.8, [{"video_path": str(run / "split_clip" / names[0]),
                                  "video_results": .8}]] for dim in audit.DIMENSIONS}
            (long / "first_eval_results.json").write_text(json.dumps(payload))
            report = audit.inventory(root, 180)
            group = report["runs"][0]
            self.assertEqual(group["logged_done"], [names[0]])
            self.assertEqual(group["lpips"], [names[0]])
            self.assertEqual(group["vbench_complete"], [names[0]])
            self.assertEqual(group["vbench_long_complete"], [])
            self.assertEqual(group["fvd_groups"][0]["short_videos"], 1)
            self.assertTrue(report["warnings"])

    def test_failed_latest_generation_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "baseline"
            run.mkdir()
            video = run / "room_60s_custom.mp4"
            video.write_bytes(b"fixture")
            rows = [{"output": str(video), "status": "completed"},
                    {"output": str(video), "status": "failed"},
                    {"output": str(run / "missing_60s_custom.mp4"), "status": "completed"}]
            (run / "run_status.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n{")
            report = audit.inventory(root)
            self.assertEqual(len(report["runs"]), 1)
            self.assertEqual(report["runs"][0]["present"], [video.name])
            self.assertEqual(report["runs"][0]["logged_done"], [])
            self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()
