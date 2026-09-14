import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import diagnose_vbench_comparison as diagnosis


class DiagnosisTests(unittest.TestCase):
    def test_cohort_and_imaging_scale(self):
        payload = {dim: [.7, [{"video_path": f"/fifo_b32/{name}", "video_results": value} for name, value in
                             (("a.mp4", 80 if dim == "imaging_quality" else .8),
                              ("extra.mp4", 20 if dim == "imaging_quality" else .2))]] for dim in diagnosis.DIMENSIONS}
        scores, audit = diagnosis.standard_scores(payload, "fifo_b32", {"a.mp4"})
        self.assertEqual(scores["a.mp4"]["imaging_quality"], .8)
        self.assertEqual(audit["subject_consistency"]["matched_mean"], .8)
        self.assertEqual(audit["subject_consistency"]["all_video_mean"], .5)
        self.assertEqual(audit["subject_consistency"]["extra"], ["extra.mp4"])
        payload["subject_consistency"][1].append(payload["subject_consistency"][1][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            diagnosis.standard_scores(payload, "fifo_b32", {"a.mp4"})

    def test_pairing_by_name_not_row_order(self):
        left = {"a": {"x": .5}, "b": {"x": .7}, "c": {"x": 100}}
        right = {"b": {"x": .6}, "a": {"x": .4}}
        result = diagnosis.paired(left, right, {"a", "b", "c"}, "x", 100)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean_delta"], .1)
        self.assertAlmostEqual(result["ci_low"], .1)
        self.assertEqual(result["wins"], 2)

    def test_saved_result_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [{"duration_sec": 60, "output_prefix": f"seed0_scene{i}_60s_"} for i in range(15)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(map(json.dumps, items)))
            expected = [i["output_prefix"] + "custom.mp4" for i in items]
            collected = {"coverage": [], "per_video": [], "warnings": []}
            for run in (diagnosis.GEO, "fifo_b32"):
                geo = run == diagnosis.GEO
                standard = root / "vbench_results" / run
                standard.mkdir(parents=True)
                values = [(n, .8 if geo else .75) for n in expected] + [(f"extra{i}.mp4", .4 if geo else .9) for i in range(15)]
                payload = {dim: [sum(v for _, v in values) / 30,
                                 [{"video_path": f"/{run}/{n}", "video_results": v * (100 if dim == "imaging_quality" else 1)}
                                  for n, v in values]] for dim in diagnosis.DIMENSIONS}
                (standard / "results_eval_results.json").write_text(json.dumps(payload))
                suite = root / "matched15_metrics_60s" / run
                suite.mkdir(parents=True)
                (suite / "manifest.jsonl").write_text(manifest.read_text())
                (suite / "pip_freeze.txt").write_text("torch==test\n")
                collected["coverage"].append({"run": run, "evaluator": "vbench-long", "suite": str(suite), "issues": []})
                for i, name in enumerate(expected):
                    smoke = suite / f"row_{i:03d}" / "smoke_test"
                    results = smoke / "vbench_long_results"
                    results.mkdir(parents=True)
                    source = root / "context_memory_60s" / run / name
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(b"fixture")
                    (smoke / "spec.json").write_text(json.dumps({"source_video": str(source), "source_bytes": source.stat().st_size,
                                                               "source_mtime_ns": source.stat().st_mtime_ns,
                                                               "long_grouping_adapter_sha256": "same"}))
                    (smoke / "vbench-long.log").write_text("args: Namespace(dev_flag=True, read_frame=False)\n")
                    inclip, mapped, raw = (.9, .94, .8) if geo else (.91, .96, .85)
                    final = (inclip + mapped) / 2
                    payload = {}
                    for dim in diagnosis.DIMENSIONS:
                        detail = {"video_path": f"/{run}/{name}", "video_results": final * (100 if dim == "imaging_quality" else 1)}
                        if dim in diagnosis.CONSISTENCY:
                            detail.update(inclip_score=inclip, clip2clip_score=raw, mapped_clip2clip_score=mapped)
                            payload[dim] = [final, [detail]]
                        else:
                            payload[dim] = [final, [], [detail]]
                    path = results / "results_eval_results.json"
                    path.write_text(json.dumps(payload))
                    collected["per_video"].append({"run": run, "evaluator": "vbench-long", "video": name,
                                                    "source_json": str(path), "scores": {d: final for d in diagnosis.DIMENSIONS}})
            with patch.object(diagnosis, "collect", return_value=collected):
                report = diagnosis.diagnose(root, manifest, 100)
            text = diagnosis.report_text(report)
            self.assertIn("point-estimate ranking reverses after cohort matching", text)
            self.assertFalse(any("mismatch" in s or "differ" in s for s in report["issues"]))
            values = {c["component"]: c for c in report["comparisons"] if c["dimension"] == "subject_consistency"}
            self.assertEqual(values["standard"]["n"], 15)
            self.assertAlmostEqual(values["standard"]["mean_delta"], .05)
            self.assertAlmostEqual(values["long"]["mean_delta"], -.015)
            self.assertAlmostEqual(values["inclip_score"]["mean_delta"], -.01)


if __name__ == "__main__":
    unittest.main()
