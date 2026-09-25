from argparse import Namespace
import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper import run_keepsake_sensitivity_cpu as runner


def fixture(root):
    source, cache = root / "source", root / "cache"
    (source / "baseline/access_traces").mkdir(parents=True)
    for kind in ("gt", "baseline"):
        (cache / kind).mkdir(parents=True)
    pose = root / "poses.json"
    pose.write_text("{}")
    items = []
    for i in range(15):
        item = dict(scene=f"scene{i}", start_frame=0, num_frames=1825, duration_sec=60,
                    fps=30, output_prefix=f"scene{i}_", pose_path=str(pose), gt_frames_dir=str(root / "gt"))
        items.append(item)
        for kind in ("gt", "baseline"):
            array = cache / kind / f"scene{i}_dino.npy"
            array.write_bytes(b"fixture array: verified_cache is mocked")
            meta = dict(scene=item["scene"], dataset_start_frame=0, num_frames=1825,
                        duration_sec=60, output_prefix=item["output_prefix"], kind=kind,
                        encoder=dict(model="test"), mode="fresh")
            array.with_suffix(".json").write_text(json.dumps(meta))
        (source / "baseline/access_traces" / f"scene{i}_custom.jsonl").write_text("fixture trace")
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, items)))
    return Namespace(root=source, cache=cache, manifest=manifest, output=root / "out")


def payload(item):
    queries, updates, banks = [], [], []
    for j, name in enumerate(runner.NAMES):
        for section in range(24):
            endpoint = (section + 1) * 76
            ids = [0, *range(endpoint - 30, endpoint + 1)]
            banks.append(dict(setting=name, scene=item["scene"], section_idx=section, retained_ids=ids))
            updates.append(dict(setting=name, scene=item["scene"], section_idx=section,
                                candidate_count=77 if section == 0 else 108, retained_count=32,
                                update_ms=1, edge_density=.2, isolated_fraction=.1))
            if section:
                for slot in (0, 19, 38, 57):
                    gap = .1 + .01 * j + .001 * item["_row"]
                    queries.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                                        section_idx=section, target_frame=section * 76 + slot + 1,
                                        retention_gap=gap, bank_oracle_distance=.3 + gap,
                                        full_oracle_distance=.3, retained_count=29))
    return dict(queries=queries, updates=updates, banks=banks)


class CpuSensitivityTests(unittest.TestCase):
    def test_grid_is_seven_one_at_a_time_settings_with_isolated_beta_zero(self):
        configs = runner.configurations()
        self.assertEqual(tuple(configs), runner.NAMES)
        for name, config in configs.items():
            changed = {k for k, v in config.items() if v != configs["default"][k]}
            self.assertEqual(len(changed), 0 if name == "default" else 1)
            self.assertTrue(changed <= {"tau", "beta", "lam"})
        matrix = np.array([[0, .9, .1], [.9, 0, .8], [.1, .8, 0]])
        default = runner.replay.priority(matrix, configs["default"])
        without_floor = runner.replay.priority(matrix, configs["beta_0"])
        degrees = (matrix >= .65).sum(1)
        np.testing.assert_allclose(default - without_floor, .5 / (degrees + 1))

    def test_plan_checks_encoder_completeness_and_freezes_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            args.output.mkdir()
            items = runner.cohort(args.manifest, 60)
            original = runner.prepare(args, items)
            self.assertEqual(original, runner.prepare(args, items))
            sidecar = args.cache / "gt/scene14_dino.json"
            meta = json.loads(sidecar.read_text())
            meta["encoder"]["model"] = "different"
            sidecar.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "encoder"):
                runner.prepare(args, items)
            meta["encoder"]["model"] = "test"
            meta["mode"] = "legacy"
            sidecar.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Fresh GT"):
                runner.prepare(args, items)
            sidecar.unlink()
            with self.assertRaises(FileNotFoundError):
                runner.prepare(args, items)

    def test_real_cpu_replay_all_settings_has_no_gt_feedback(self):
        n = 1825
        rng = np.random.default_rng(31)
        features = rng.normal(size=(n, 16)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        poses = np.repeat(np.eye(4)[None], n, axis=0)
        poses[:, 0, 3] = np.sin(np.arange(n) / 80)
        item = dict(scene="synthetic", num_frames=n, _row=0)
        policies = runner.replay.load_file_module("sensitivity_test_policies", runner.ROOT / "diffsynth/pipelines/memory_policies.py")
        def run(gt):
            q, u, b = runner.replay.replay_item(item, poses, features, gt, policies,
                                               configs=runner.configurations(), budget=32)
            result = dict(queries=q, updates=u, banks=b)
            runner.validate_cell(result, item)
            return result
        first = run(features)
        second = run(features[::-1].copy())
        self.assertEqual(first["banks"], second["banks"], "GT must never influence the retained IDs")
        self.assertTrue(any(a["bank_oracle_distance"] != b["bank_oracle_distance"]
                            for a, b in zip(first["queries"], second["queries"])))

    def test_cells_reject_missing_results_nonfinite_scores_and_illegal_banks(self):
        item = dict(scene="test", _row=0, num_frames=1825)
        good = payload(item)
        runner.validate_cell(good, item)
        bad = copy.deepcopy(good)
        bad["queries"].pop()
        with self.assertRaisesRegex(ValueError, "query"):
            runner.validate_cell(bad, item)
        bad = copy.deepcopy(good)
        bad["queries"][0]["retention_gap"] = float("nan")
        with self.assertRaisesRegex(ValueError, "scores"):
            runner.validate_cell(bad, item)
        bad = copy.deepcopy(good)
        bad["banks"][0]["retained_ids"][-1] = 2000
        with self.assertRaisesRegex(ValueError, "causal"):
            runner.validate_cell(bad, item)
        bad = copy.deepcopy(good)
        bad["updates"][0]["scene"] = "wrong"
        with self.assertRaisesRegex(ValueError, "scene"):
            runner.validate_cell(bad, item)

    def test_interruption_resumes_complete_cells_and_reports_correct_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            def replay_item(item, *positional, **kwargs):
                self.assertEqual(tuple(kwargs["configs"]), runner.NAMES)
                self.assertEqual(kwargs["budget"], 32)
                result = payload(item)
                return result["queries"], result["updates"], result["banks"]
            def verify(cache, item, source):
                if item["_row"] == 1:
                    raise InterruptedError("fixture interruption")
                return {"baseline": None, "gt": None}, {}
            audit = lambda path, *args: dict(sha256=runner.replay.sha256(path))
            with patch.object(runner.replay, "verified_cache", side_effect=verify), \
                    patch.object(runner.replay, "audit_trace", side_effect=audit), \
                    patch.object(runner, "load_poses", return_value=None), \
                    patch.object(runner.replay, "replay_item", side_effect=replay_item) as replay_call:
                with self.assertRaises(InterruptedError):
                    runner.execute(args)
                self.assertEqual(replay_call.call_count, 1)
            state = json.loads((args.output / "status.json").read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["completed_trajectories"], 1)
            self.assertFalse((args.output / "summary.csv").exists())
            with patch.object(runner.replay, "verified_cache", return_value=({"baseline": None, "gt": None}, {})), \
                    patch.object(runner.replay, "audit_trace", side_effect=audit), \
                    patch.object(runner, "load_poses", return_value=None), \
                    patch.object(runner.replay, "replay_item", side_effect=replay_item) as replay_call:
                runner.execute(args)
                self.assertEqual(replay_call.call_count, 14)
                runner.execute(args)
                self.assertEqual(replay_call.call_count, 14, "All validated cells must be reused")
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "complete")
            with (args.output / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            for j, row in enumerate(rows):
                self.assertAlmostEqual(float(row["variant_minus_default"]), .01 * j)
                self.assertEqual(int(row["videos"]), 15)
            self.assertTrue((args.output / "sensitivity.tex").is_file())
            self.assertNotIn("p_two_sided", (args.output / "paired_default_minus_variant.csv").read_text())
            (args.root / "baseline/access_traces/scene0_custom.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Changed source"):
                runner.execute(args)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "failed")

    def test_cpu_batch_requests_no_gpu_and_cli_rejects_login_execution(self):
        script = runner.ROOT / "slurm/newton_keepsake_sensitivity_cpu.sbatch"
        text = script.read_text()
        self.assertNotIn("--gres", text)
        self.assertNotIn("--gpus", text)
        self.assertIn('--partition=normal', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        subprocess.run(["bash", "-n", str(script)], check=True)
        with patch.dict(runner.os.environ, {}, clear=True), patch.object(runner.sys, "argv", ["runner"]):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
