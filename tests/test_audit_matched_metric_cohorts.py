import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
from audit_matched_metric_cohorts import DIMENSIONS, cell, cohort_record, scan_standard


class CohortAuditTests(unittest.TestCase):
    def test_membership_not_count(self):
        expected = {"a.mp4", "b.mp4"}
        self.assertEqual(cohort_record("test", ["a.mp4", "wrong.mp4"], expected)["state"], "PARTIAL")
        self.assertEqual(cohort_record("test", ["a.mp4", "b.mp4", "c.mp4"], expected)["state"], "SUPERSET")
        self.assertEqual(cohort_record("test", ["a.mp4", "b.mp4", "b.mp4"], expected)["state"], "CHECK")
        parts = [cohort_record("one", ["a.mp4"], expected), cohort_record("two", ["b.mp4"], expected)]
        self.assertEqual(cell(parts, expected), "UNION 2/2")

    def test_standard_metrics_and_fvd_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"a_60s_custom.mp4", "b_60s_custom.mp4"}
            metrics = root / "eval"
            metrics.mkdir()
            rows = [{"output": f"/videos/baseline/{name}", "duration_sec": 60,
                     "status": "completed", "lpips_alex": .5} for name in sorted(expected)]
            path = metrics / "metrics.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            summary = {"by_duration": {"60": {"fvd": 400., "completed_or_short": 2}},
                       "metric_config": {"max_frames": None}}
            (metrics / "summary.json").write_text(json.dumps(summary))
            bench = root / "vbench_results"
            bench.mkdir()
            data = {dim: [.7, [{"video_path": f"/videos/baseline/{name}",
                               "video_results": True if dim == "dynamic_degree" else .7}
                              for name in sorted(expected | {"extra_60s_custom.mp4"})]] for dim in DIMENSIONS}
            (bench / "test_eval_results.json").write_text(json.dumps(data))
            warnings = []
            report = scan_standard(root, expected, 1825, warnings)
            self.assertFalse(warnings)
            self.assertEqual(cell(report["baseline"]["LPIPS"], expected), "EXACT")
            self.assertEqual(cell(report["baseline"]["FVD"], expected), "EXACT")
            self.assertEqual(cell(report["baseline"]["VB6"], expected), "SUPERSET")
            summary["metric_config"]["max_frames"] = 301
            (metrics / "summary.json").write_text(json.dumps(summary))
            report = scan_standard(root, expected, 1825, [])
            self.assertEqual(cell(report["baseline"]["FVD"], expected), "CHECK")
            summary["metric_config"]["max_frames"] = None
            summary["by_duration"]["60"]["completed_or_short"] = 3
            rows.append(dict(rows[0], output="/videos/fifo_b32/a_60s_custom.mp4"))
            path.write_text("\n".join(map(json.dumps, rows)))
            (metrics / "summary.json").write_text(json.dumps(summary))
            report = scan_standard(root, expected, 1825, [])
            self.assertIn("FVD duration group includes other policies", report["baseline"]["FVD"][0]["issues"])
            self.assertEqual(cell(report["baseline"]["FVD"], expected), "CHECK")

    def test_cpu_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps({"duration_sec": 60, "num_frames": 1825,
                                                     "output_prefix": f"seed0_scene{i}_60s_"}) for i in range(15)))
            output = root / "out"
            result = subprocess.run([sys.executable, str(UTILS / "audit_matched_metric_cohorts.py"),
                                     "--root", str(root), "--manifest", str(manifest), "--output", str(output)],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NOT AUDITED", result.stdout)
            self.assertTrue((output / "audit.md").is_file())
            data = json.loads((output / "audit.json").read_text())
            self.assertEqual(len(data["expected"]), 15)
            self.assertEqual(len(data["standard"]), 21)


if __name__ == "__main__":
    unittest.main()
