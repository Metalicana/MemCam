import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper import run_keepsake_component_proxy_cpu as runner
from tests.test_keepsake_sensitivity_cpu import fixture


def payload(item):
    queries, updates, banks = [], [], []
    for j, name in enumerate(runner.NAMES):
        for section in range(24):
            endpoint = (section + 1) * 76
            banks.append(dict(setting=name, scene=item["scene"], section_idx=section,
                              retained_ids=[0, *range(endpoint - 30, endpoint + 1)]))
            updates.append(dict(setting=name, scene=item["scene"], section_idx=section,
                                candidate_count=77 if section == 0 else 108, retained_count=32,
                                update_ms=1., edge_density=.2, isolated_fraction=.1))
            if section:
                for slot in runner.SLOTS:
                    gap = .1 + .01 * j + .001 * item["_row"]
                    queries.append(dict(setting=name, scene=item["scene"], source_row=item["_row"],
                                        section_idx=section, target_frame=76 * section + slot + 1,
                                        retention_gap=gap, bank_oracle_distance=.3 + gap,
                                        full_oracle_distance=.3, retained_count=28))
    return dict(queries=queries, updates=updates, banks=banks)


class ComponentProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = runner.replay.load_file_module("proxy_test_policies", runner.ROOT / "diffsynth/pipelines/memory_policies.py")

    def test_all_five_interventions_use_production_component_definitions(self):
        configs = runner.configurations()
        self.assertEqual(tuple(configs), runner.NAMES)
        self.assertEqual(configs["pose_only"]["geometry_weight"], 1.)
        self.assertEqual(configs["appearance_only"]["geometry_weight"], 0.)
        affinity = np.array([[0., .9, .1], [.9, 0., .8], [.1, .8, 0.]])
        with patch.object(self.policies, "_slam_covisibility_affinity", return_value=affinity):
            scores = {name: np.array(list(self.policies.compute_slam_covisibility_scores(
                [0, 1, 2], None, **config).values())) for name, config in configs.items()}
        degree = (affinity >= .65).sum(1)
        np.testing.assert_allclose(scores["degree_only"], 1 - np.minimum(degree / 3., 1) + .5 / (degree + 1))
        np.testing.assert_allclose(scores["closest_only"], .25 * (1 - affinity.max(1)))
        np.testing.assert_allclose(scores["full"], scores["degree_only"] + scores["closest_only"])
        self.assertFalse(np.allclose(scores["closest_only"], scores["full"] - .5 / (degree + 1)))

    def test_production_buffer_ties_protection_and_no_iterative_rescoring(self):
        bank = self.policies.FrameMemoryBuffer("slam_covisibility", budget=3, pinned_frames={0})
        bank.update(range(6), eviction_scores={i: 1. for i in range(6)}, protected_frames={5})
        self.assertEqual(bank.candidates(), [0, 4, 5])
        bank.update([5, 6, 7], eviction_scores={i: 1. for i in (0, 4, 5, 6, 7)}, protected_frames={7})
        self.assertEqual(bank.candidates(), [0, 6, 7], "Previous endpoint must become evictable")

    def test_real_full_length_cpu_replay_no_gt_feedback_and_legacy_full_matches(self):
        n = 1825
        rng = np.random.default_rng(41)
        features = rng.normal(size=(n, 8)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        poses = np.repeat(np.eye(4)[None], n, axis=0)
        poses[:, 0, 3] = np.sin(np.arange(n) / 80)
        item = dict(scene="synthetic", num_frames=n, _row=0)
        score = self.policies.compute_slam_covisibility_scores
        def checked(ids, current_poses, **kwargs):
            self.assertEqual(set(kwargs["dino_features"]), set(ids))
            self.assertEqual(len(current_poses), max(ids) + 1)
            for i in ids:
                np.testing.assert_array_equal(kwargs["dino_features"][i], features[i])
            return score(ids, current_poses, **kwargs)
        with patch.object(self.policies, "compute_slam_covisibility_scores", side_effect=checked) as calls:
            first = runner.replay_item(item, poses, features, features, self.policies)
            second = runner.replay_item(item, poses, features, features[::-1].copy(), self.policies)
            self.assertEqual(calls.call_count, 2 * 24 * 5)
        runner.validate_cell(first, item)
        runner.validate_cell(second, item)
        self.assertEqual(first["banks"], second["banks"], "GT cannot change any retention decision")
        self.assertTrue(any(a["retention_gap"] != b["retention_gap"] for a, b in zip(first["queries"], second["queries"])))
        old_queries, _, old_banks = runner.replay.replay_item(
            item, poses, features, features, self.policies, configs={"default": runner.replay.DEFAULT})
        full_banks = [r["retained_ids"] for r in first["banks"] if r["setting"] == "full"]
        self.assertEqual(full_banks, [r["retained_ids"] for r in old_banks])
        np.testing.assert_allclose([r["retention_gap"] for r in first["queries"] if r["setting"] == "full"],
                                   [r["retention_gap"] for r in old_queries], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "features"):
            runner.replay_item(item, poses, features * 2, features, self.policies)

    def test_validation_catches_missing_queries_nan_future_ids_and_false_oracles(self):
        item = dict(scene="test", _row=0, num_frames=1825)
        good = payload(item)
        runner.validate_cell(good, item)
        for mutate in (
            lambda p: p["queries"].pop(),
            lambda p: p["queries"][0].update(retention_gap=float("nan")),
            lambda p: p["banks"][0]["retained_ids"].__setitem__(1, 2000),
            lambda p: p["banks"][1]["retained_ids"].__setitem__(1, 1),
            lambda p: p["queries"][0].update(full_oracle_distance=.2, retention_gap=.2),
            lambda p: p["queries"][0].update(retained_count=31),
            lambda p: p["updates"][0].update(candidate_count=100),
            lambda p: p["banks"][0].update(scene="wrong"),
        ):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_cell(bad, item)

    def test_plan_freezes_all_components_and_requires_shared_fresh_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            args.output.mkdir()
            items = runner.cohort(args.manifest, 60)
            plan = runner.prepare(args, items)
            self.assertEqual(plan, runner.prepare(args, items))
            self.assertEqual(set(plan["configurations"]), set(runner.NAMES))
            path = args.cache / "gt/scene14_dino.json"
            original = path.read_text()
            for key, value in (("encoder", dict(model="different")), ("mode", "legacy"), ("scene", "wrong")):
                meta = json.loads(original)
                meta[key] = value
                path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    runner.prepare(args, items)
            path.write_text(original)
            plan["bootstrap_draws"] = 123
            runner.save(args.output / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "Frozen"):
                runner.prepare(args, items)

    def test_resume_reports_all_variants_signed_deltas_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            def verify(cache, item, source):
                if item["_row"] == 1:
                    raise InterruptedError("fixture interruption")
                return {"baseline": None, "gt": None}, {}
            audit = lambda path, *args: dict(sha256=runner.replay.sha256(path))
            with patch.object(runner.replay, "verified_cache", side_effect=verify), \
                    patch.object(runner.replay, "audit_trace", side_effect=audit), \
                    patch.object(runner, "load_poses", return_value=None), \
                    patch.object(runner, "replay_item", side_effect=lambda item, *args: payload(item)) as calls:
                with self.assertRaises(InterruptedError):
                    runner.execute(args)
                self.assertEqual(calls.call_count, 1)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "interrupted")
            self.assertFalse((args.output / "summary.csv").exists())
            with patch.object(runner.replay, "verified_cache", return_value=({"baseline": None, "gt": None}, {})), \
                    patch.object(runner.replay, "audit_trace", side_effect=audit), \
                    patch.object(runner, "load_poses", return_value=None), \
                    patch.object(runner, "replay_item", side_effect=lambda item, *args: payload(item)) as calls:
                runner.execute(args)
                self.assertEqual(calls.call_count, 14)
                runner.execute(args)
                self.assertEqual(calls.call_count, 14, "All completed cells should be reused")
            with (args.output / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            for i, row in enumerate(rows):
                self.assertAlmostEqual(float(row["variant_minus_full"]), .01 * i)
                self.assertEqual(int(row["trajectories"]), 15)
            self.assertNotIn("p_two_sided", (args.output / "paired_full_minus_variant.csv").read_text())
            self.assertIn("Post-hoc", (args.output / "component_proxy.tex").read_text())
            data = args.output / "cells/row_000/replay.json"
            data.write_text(data.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Changed source/artifact"):
                runner.execute(args)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "failed")

    def test_separate_output_cpu_script_and_login_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            args.output = args.root / "bad"
            with self.assertRaisesRegex(ValueError, "separate"):
                runner.execute(args)
            args.output = Path(temporary) / "other_study"
            args.output.mkdir()
            (args.output / "status.json").write_text('{"status":"complete"}')
            with self.assertRaisesRegex(ValueError, "overwrite"):
                runner.execute(args)
            self.assertEqual(json.loads((args.output / "status.json").read_text()), dict(status="complete"))
        script = runner.ROOT / "slurm/newton_keepsake_component_proxy_cpu.sbatch"
        text = script.read_text()
        self.assertNotIn("--gres", text)
        self.assertNotIn("--gpus", text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        subprocess.run(["bash", "-n", str(script)], check=True)
        with patch.dict(runner.os.environ, {}, clear=True), patch.object(runner.sys, "argv", ["runner"]):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
