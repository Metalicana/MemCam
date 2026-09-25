import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policies = load("weight_policies", "diffsynth/pipelines/memory_policies.py")
sweep = load("weight_sweep", "paper/run_keepsake_weight_ablation.py")


class KeepsakeWeightTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.repeat(np.eye(4)[None], 4, axis=0)
        self.poses[:, 0, 3] = [0, .1, 10, 20]
        self.features = {i: np.array(v, dtype=float) for i, v in enumerate(
            ([1, 0], [0, 1], [1, 0], [-1, 0]))}
        self.kwargs = dict(memory_frame_indices=list(range(4)), c2ws=self.poses,
                           dino_features=self.features)

    def test_endpoints_and_weighted_affinity(self):
        affinity = policies._slam_covisibility_affinity
        appearance = affinity(**self.kwargs, geometry_weight=0, visual_weight=1)
        geometry = affinity(**self.kwargs, geometry_weight=1, visual_weight=0)
        self.assertEqual(appearance[0, 2], 1)
        self.assertEqual(appearance[0, 1], 0)
        changed = dict(self.kwargs, dino_features={i: np.array([0., 1.]) for i in range(4)})
        np.testing.assert_array_equal(geometry, affinity(**changed, geometry_weight=1, visual_weight=0))
        changed = dict(self.kwargs, c2ws=self.poses[::-1].copy())
        np.testing.assert_array_equal(appearance, affinity(**changed, geometry_weight=0, visual_weight=1))
        for alpha in sweep.WEIGHTS:
            np.testing.assert_allclose(affinity(**self.kwargs, geometry_weight=alpha, visual_weight=1-alpha),
                                       alpha*geometry + (1-alpha)*appearance)

    def test_default_scores_unchanged_and_pinning_preserved(self):
        original = policies.compute_slam_covisibility_scores(**self.kwargs, pinned_frames=[0], return_details=True)
        explicit = policies.compute_slam_covisibility_scores(**self.kwargs, pinned_frames=[0],
                   geometry_weight=.65, visual_weight=1-.65, return_details=True)
        self.assertEqual(original, explicit)
        for alpha in sweep.WEIGHTS:
            scores, details = policies.compute_slam_covisibility_scores(**self.kwargs, pinned_frames=[0],
                geometry_weight=alpha, visual_weight=1-alpha, return_details=True)
            self.assertTrue(np.isinf(scores[0]))
            self.assertTrue(all(d["covisibility_threshold"] == .65 for d in details.values()))

    def test_priority_components_and_default_equivalence(self):
        values = {}
        for mode in ("full", "degree_only", "closest_only"):
            scores, details = policies.compute_slam_covisibility_scores(
                **self.kwargs, priority_mode=mode, return_details=True)
            values[mode] = scores
            for index, d in details.items():
                degree = 1 - min(d["covisible_observers"] / 3, 1) + .5 / (d["covisible_observers"] + 1)
                closest = .25 * (1 - d["max_covisibility"])
                expected = degree + closest if mode == "full" else (degree if mode == "degree_only" else closest)
                self.assertEqual(scores[index], expected)
                self.assertEqual(d["priority_mode"], mode)
            pinned = policies.compute_slam_covisibility_scores(**self.kwargs, priority_mode=mode, pinned_frames=[0])
            self.assertEqual(pinned[0], float("inf"))
        for index in range(4):
            self.assertEqual(values["full"][index], values["degree_only"][index] + values["closest_only"][index])
        with self.assertRaisesRegex(ValueError, "priority mode"):
            policies.compute_slam_covisibility_scores(**self.kwargs, priority_mode="typo")

    def test_task_grid_matched_and_no_gpu_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            items = [{"duration_sec": 60, "output_prefix": "exclude"}] + [
                {"duration_sec": 180, "output_prefix": f"scene{i}_180s_"} for i in range(15)]
            manifest.write_text("\n".join(map(json.dumps, items)))
            specs = [sweep.task_spec(manifest, 180, i) for i in range(75)]
            for alpha in sweep.WEIGHTS:
                subset = [s for s in specs if s["geometry_weight"] == alpha]
                self.assertEqual([s["row"] for s in subset], list(range(1, 16)))
                self.assertTrue(all(s["geometry_weight"] + s["appearance_weight"] == 1 for s in subset))
            cmd = sweep.command(specs[0], manifest, Path(tmp), 42)
            self.assertEqual(cmd[cmd.index("--keepsake_geometry_weight")+1], "0.65")
            self.assertEqual(cmd[cmd.index("--rows")+1], "1")
            result = subprocess.run([sys.executable, str(ROOT / "paper/run_keepsake_weight_ablation.py"),
                "--task", "74", "--manifest", str(manifest), "--output", str(Path(tmp)/"unused"), "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["geometry_weight"], 1)
            self.assertFalse((Path(tmp)/"unused").exists())
            with self.assertRaises(ValueError):
                sweep.task_spec(manifest, 180, 75)

    def test_cli_rejects_invalid_weights_before_loading_models(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            result = subprocess.run([sys.executable, str(ROOT / "utils/run_context_memory_batch.py"),
                "--manifest", "missing.jsonl", "--output_dir", "unused", "--memory_policy", "slam_covisibility",
                "--keepsake_geometry_weight", value], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be in [0, 1]", result.stderr)

    def test_pipeline_wires_weight_only_to_keepsake(self):
        tree = ast.parse((ROOT / "diffsynth/pipelines/wan_video_memcam.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == "compute_slam_covisibility_scores"]
        self.assertEqual(len(calls), 1)
        keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(keywords["geometry_weight"], "keepsake_geometry_weight")
        self.assertEqual(keywords["visual_weight"], "1.0 - keepsake_geometry_weight")
        self.assertEqual(keywords["priority_mode"], "keepsake_priority_mode")
        self.assertNotIn("covisibility_threshold", keywords)


if __name__ == "__main__":
    unittest.main()
