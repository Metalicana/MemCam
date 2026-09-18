import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("vb180", Path(__file__).resolve().parents[1] / "paper/run_vbench_180s.py")
vb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vb)


class VBench180Tests(unittest.TestCase):
    def fixture(self, root):
        items = [{"duration_sec": 180, "num_frames": 5397, "fps": 30,
                  "output_prefix": f"seed0_scene{i}_180s_"} for i in range(15)]
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(map(json.dumps, items)))
        for run in vb.RUNS:
            (root / run).mkdir()
            for item in items:
                (root / run / (item["output_prefix"] + "custom.mp4")).write_bytes(b"fixture")
            (root / run / "extra_60s_custom.mp4").write_bytes(b"extra")
        return manifest, items

    def test_cohort_excludes_extras_and_rejects_missing_or_wrong_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, items = self.fixture(root)
            selected, names = vb.cohort(manifest, root)
            self.assertEqual(selected, items)
            self.assertEqual(len(names), 15)
            missing = root / vb.RUNS[-1] / names[-1]
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "Missing/empty"):
                vb.cohort(manifest, root)
            missing.write_bytes(b"fixture")
            items[0]["duration_sec"] = 60
            manifest.write_text("\n".join(map(json.dumps, items)))
            with self.assertRaisesRegex(ValueError, "15 distinct"):
                vb.cohort(manifest, root)

    def test_probe_rejects_truncated_video(self):
        item = {"num_frames": 5397, "fps": 30}
        for frames, rate, valid in ((5397, "30/1", True), (1825, "30/1", False), (5397, "10/1", False)):
            raw = json.dumps({"streams": [{"nb_read_frames": str(frames), "r_frame_rate": rate}]})
            with patch.object(vb.subprocess, "check_output", return_value=raw):
                if valid:
                    vb.probe_video(Path("test.mp4"), item)
                else:
                    with self.assertRaises(ValueError):
                        vb.probe_video(Path("test.mp4"), item)

    def test_results_require_every_dimension_exact_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"video{i}.mp4" for i in range(15)]
            payload = {dim: [0, [{"video_path": f"/input/baseline/{name}",
                                 "video_results": 50 if dim == "imaging_quality" else .5}
                                for name in names]] for dim in vb.DIMENSIONS}
            path = root / "test_eval_results.json"
            path.write_text(json.dumps(payload))
            scores = vb.validate_bench(root, names, "baseline")
            self.assertEqual(len(scores), 6)
            self.assertTrue(all(score["value"] == .5 for score in scores))
            payload[vb.DIMENSIONS[0]][1].append(payload[vb.DIMENSIONS[0]][1][0])
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                vb.validate_bench(root, names, "baseline")

    def test_runner_exports_scores_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, items = self.fixture(root)
            args = Namespace(task=0, manifest=manifest, root=root,
                             output=root / "output", vbench_root=root / "VBench")

            def fake_evaluate(command, work, cwd):
                result = work / "results"
                result.mkdir()
                run = work.name
                payload = {dim: [0, [{"video_path": f"/input/{run}/{item['output_prefix']}custom.mp4",
                                     "video_results": 50 if dim == "imaging_quality" else .5}
                                    for item in items]] for dim in vb.DIMENSIONS}
                (result / "test_eval_results.json").write_text(json.dumps(payload))

            with patch.object(vb, "probe_video", return_value={}), \
                    patch.object(vb, "metric_command", return_value=["python"]), \
                    patch.object(vb.subprocess, "run"), \
                    patch.object(vb, "freeze_environment", return_value="fixture"), \
                    patch.object(vb, "run_logged", side_effect=fake_evaluate):
                vb.execute(args)
                work = args.output / "baseline"
                self.assertEqual(json.loads((work / "status.json").read_text())["status"], "complete")
                self.assertEqual(len((work / "scores.csv").read_text().splitlines()), 7)
                self.assertEqual(len(list((work / "input/baseline").iterdir())), 15)
                # A failed evaluator must not leave a completed task or scores.
                args.task = 1
                with patch.object(vb, "run_logged", side_effect=RuntimeError("evaluation failed")):
                    with self.assertRaises(RuntimeError):
                        vb.execute(args)
                failed = args.output / "fifo_b32"
                self.assertEqual(json.loads((failed / "status.json").read_text())["status"], "failed")
                self.assertFalse((failed / "scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
