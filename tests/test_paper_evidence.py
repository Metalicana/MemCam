from argparse import Namespace
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from paper import evidence_statistics as stats
from paper import replay_keepsake_updates as replay
from paper import run_paper_evidence as runner
from paper import submit_paper_evidence as submit
from paper.benchmark_query_latency import load_file_module
from run_budget_metric_grid import QUALITY_CONFIG
from tests.test_b32_query_latency import fixture as trace_fixture
from tests.test_compare_fvd_matched import fixture as fvd_fixture, FakeRunner


def scalar_fixture(root):
    items = [dict(scene=f"scene{i}", output_prefix=f"seed0_scene{i}_0000_60s_", start_frame=0,
                  duration_sec=60, num_frames=1825, fps=30, _row=i) for i in range(15)]
    report = []
    for j, run in enumerate(stats.RUNS):
        folder = root / run
        folder.mkdir()
        rows = [dict(status="completed", run_name=run, row=i, scene=item["scene"], start_frame=0,
                     duration_sec=60, output=str(folder / (item["output_prefix"] + "custom.mp4")),
                     lpips_alex=.3 + j*.01 + i*.001) for i, item in enumerate(items)]
        (folder / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (folder / "summary.json").write_text(json.dumps(dict(metric_config={"max_frames": None, **QUALITY_CONFIG["LPIPS"]})))
        report.append(dict(run=run, evaluator="quality", metric="LPIPS", value=np.mean([r["lpips_alex"] for r in rows]),
                           n=15, source=str(folder / "metrics.jsonl")))
        vb = {}
        for dim in stats.NORMALIZATION:
            scale = 100 if dim == "imaging_quality" else 1
            details = [dict(video_path=r["output"], video_results=(.7 + j*.01 + i*.001)*scale)
                       for i, r in enumerate(rows)]
            vb[dim] = [0., details]
            report.append(dict(run=run, evaluator="vbench", metric=dim,
                               value=np.mean([d["video_results"]/scale for d in details]), n=15,
                               source=str(folder / "test_eval_results.json")))
        (folder / "test_eval_results.json").write_text(json.dumps(vb))
    path = root / "scores.csv"
    stats.write_csv(path, report)
    return items, path


class EvidenceStatisticsTests(unittest.TestCase):
    def test_exact_paired_test_and_intervals(self):
        a, b = stats.mean_contrasts({stats.KEEP: [2, 3, 4], "other": [1, 2, 3]}, draws=100)
        self.assertEqual(b[0]["difference"], 1)
        self.assertEqual((b[0]["ci_low"], b[0]["ci_high"]), (1, 1))
        self.assertEqual(b[0]["p_two_sided"], .25)
        self.assertEqual(a[0]["videos"], 3)
        _, equal = stats.mean_contrasts({stats.KEEP: [1, 2], "other": [1, 2]})
        self.assertEqual(equal[0]["p_two_sided"], 1)
        for bad in ({stats.KEEP: [1, 2], "other": [1]}, {stats.KEEP: [1, np.nan], "other": [1, 2]}):
            with self.assertRaises(ValueError):
                stats.mean_contrasts(bad)

    def test_holm(self):
        np.testing.assert_allclose(stats.holm([.04, .01, .03]), [.06, .03, .06])
        with self.assertRaises(ValueError):
            stats.holm([np.nan])

    def test_scalar_adapter_raw_data_not_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items, path = scalar_fixture(root)
            values, sources, rows = stats.scalar_inputs(path, items)
            self.assertEqual(len(rows), 90)
            self.assertEqual(len(values), 8)
            means = {k: values[k][stats.KEEP].mean() for k in stats.NORMALIZATION}
            expected = 100 * sum(w*(means[k]-lo)/(hi-lo) for k, (lo, hi, w) in stats.NORMALIZATION.items())/5.5
            self.assertAlmostEqual(values["custom_vbench6_percent"][stats.KEEP].mean(), expected)
            self.assertIn(str(path), sources)
            output = root / "out"
            output.mkdir()
            stats.export_scalars(path, items, output, draws=30)
            with (output / "contrasts.csv").open() as handle:
                contrasts = list(csv.DictReader(handle))
            self.assertEqual(len(contrasts), 40)
            target = root / stats.KEEP / "metrics.jsonl"
            target.write_text(target.read_text().splitlines()[0])
            with self.assertRaises(ValueError):
                stats.scalar_inputs(path, items)

    def test_scalar_adapter_wrong_identity_and_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items, path = scalar_fixture(root)
            items[0]["start_frame"] = 1
            with self.assertRaisesRegex(ValueError, "identity"):
                stats.scalar_inputs(path, items)
            items[0]["start_frame"] = 0
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["value"] = 0
            stats.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                stats.scalar_inputs(path, items)

    def test_fvd_whole_scene_resampling_and_policy_pairing(self):
        gt = np.random.default_rng(1).normal(size=(3, 4, 2))
        features = {"GT": gt, **{run: gt + (j+1)*.1 for j, run in enumerate(stats.RUNS)}}
        a, b, boot = stats.fvd_statistics(features, draws=12)
        self.assertEqual(boot.shape, (12, 6))
        for j, row in enumerate(a):
            self.assertAlmostEqual(row["fvd"], 2*((j+1)*.1)**2, places=10)
            self.assertAlmostEqual(row["ci_low"], row["fvd"], places=9)
        self.assertTrue(all(0 < r["p_holm_fvd_family"] <= 1 for r in b))
        same = {key: gt for key in features}
        _, contrasts, _ = stats.fvd_statistics(same, draws=4)
        self.assertTrue(all(r["p_two_sided"] == 1 for r in contrasts))
        with self.assertRaises(ValueError):
            stats.fvd_statistics({"GT": gt})


class ReplayTests(unittest.TestCase):
    def test_default_matches_production(self):
        policies = load_file_module("test_evidence_policies", replay.ROOT / "diffsynth/pipelines/memory_policies.py")
        poses = np.repeat(np.eye(4)[None], 6, axis=0)
        poses[:, 0, 3] = np.arange(6)
        features = dict(enumerate(np.eye(6)))
        matrix = policies._slam_covisibility_affinity(list(range(6)), poses, dino_features=features)
        expected = policies.compute_slam_covisibility_scores(list(range(6)), poses, dino_features=features)
        np.testing.assert_allclose(replay.priority(matrix, replay.DEFAULT), list(expected.values()))

    def test_iteration_changes_degrees_and_frozen_scores_do_not_become_inf(self):
        example = np.array([[0, .9, .5, .5, .5], [.9, 0, .5, .9, .9], [.5, .5, 0, .5, .9],
                            [.5, .9, .5, 0, .9], [.5, .9, .9, .9, 0]])
        one = replay.retain(list(range(5)), example, 2, set(), replay.DEFAULT, {})
        iterative = replay.retain(list(range(5)), example, 2, set(), {**replay.DEFAULT, "update": "iterative"}, {})
        self.assertEqual(one, [0, 2])
        self.assertEqual(iterative, [2, 3])
        matrix = np.array([[0, .9, 0, 0], [.9, 0, .9, 0], [0, .9, 0, 0], [0, 0, 0, 0]])
        ids = [0, 1, 2, 3]
        scores = {}
        config = {**replay.DEFAULT, "update": "frozen_admission"}
        kept = replay.retain(ids, matrix, 2, {0, 3}, config, scores)
        self.assertEqual(kept, [0, 3])
        self.assertTrue(all(np.isfinite(list(scores.values()))))
        old = dict(scores)
        replay.retain(ids, np.zeros((4, 4)), 2, {0}, config, scores)
        self.assertEqual(scores, old)
        self.assertEqual(replay.retain(ids, np.zeros((4, 4)), 2, {0}, replay.DEFAULT, {}), [0, 3])

    def test_bad_affinities_ids_and_budget_rejected(self):
        for a in (np.ones((2, 2)), np.array([[0, np.nan], [np.nan, 0]]), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                replay.priority(a, replay.DEFAULT)
        with self.assertRaises(ValueError):
            replay.retain([0, 0], np.zeros((2, 2)), 1, {0}, replay.DEFAULT, {})
        with self.assertRaises(ValueError):
            replay.retain([0, 1], np.zeros((2, 2)), 1, {0, 1}, replay.DEFAULT, {})

    def test_fixed_history_replay_is_causal_and_endpoint_moves(self):
        policies = load_file_module("test_evidence_replay", replay.ROOT / "diffsynth/pipelines/memory_policies.py")
        n = 229
        poses = np.repeat(np.eye(4)[None], n, axis=0)
        f = np.ones((n, 2), dtype=np.float32) / np.sqrt(2)
        item = dict(num_frames=n, scene="fixture", _row=0)
        queries, updates, snapshots = replay.replay_item(item, poses, f, f, policies,
                                                       configs={"default": replay.DEFAULT}, budget=4)
        self.assertEqual(len(queries), 8)
        self.assertTrue(all(abs(q["retention_gap"]) < 1e-6 for q in queries))
        for snap in snapshots:
            self.assertEqual(len(snap["retained_ids"]), 4)
            self.assertIn(0, snap["retained_ids"])
            self.assertIn((snap["section_idx"]+1)*76, snap["retained_ids"])
            self.assertLessEqual(max(snap["retained_ids"]), (snap["section_idx"]+1)*76)
        self.assertNotIn(76, snapshots[-1]["retained_ids"])
        self.assertTrue(all(0 <= u["edge_density"] <= 1 for u in updates))


class EvidenceJobsTests(unittest.TestCase):
    def test_cpu_fvd_export_reconciles_reported_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "fvd_features_60s"
            work.mkdir()
            gt = np.random.default_rng(1).normal(size=(15, 4, 3))
            features = {"GT": gt, **{run: gt+(j+1)*.1 for j, run in enumerate(stats.RUNS)}}
            np.savez(work / "cohort_features.npz", **features)
            (work / "cohort.json").write_text(json.dumps(dict(runs=stats.RUNS)))
            (work / "receipt.json").write_text(json.dumps(dict(status="complete", artifacts={
                str(work / name): runner.digest(work / name) for name in ("cohort_features.npz", "cohort.json")})))
            report = root / "scores.csv"
            stats.write_csv(report, [dict(run=run, evaluator="quality", metric="FVD", n=15,
                                         value=3*((j+1)*.1)**2) for j, run in enumerate(stats.RUNS)])
            out = root / "scored"
            out.mkdir()
            args = Namespace(output=root, scores=report, fvd_draws=5)
            runner.fvd_scores(args, out, (60,))
            with (out / "reported_score_check.csv").open() as handle:
                checks = list(csv.DictReader(handle))
            self.assertEqual(len(checks), 6)
            self.assertTrue(all(r["within_numerical_tolerance"] == "True" for r in checks))
            self.assertTrue((out / "contrasts.csv").exists())

    def test_feature_extraction_adapter_resume_and_stale_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "context_memory_60s"
            videos.mkdir()
            items, _, _ = fvd_fixture(videos, count=2)
            for item in items:
                for run in stats.RUNS:
                    (videos / run).mkdir(exist_ok=True)
                    (videos / run / (item["output_prefix"]+"custom.mp4")).write_bytes(run.encode())
            detector = root / "i3d.pt"
            detector.write_bytes(b"fixture detector")
            out = root / "out"
            out.mkdir()
            args = Namespace(root=root, fvd_cache=root)
            fake = FakeRunner()
            fake.resolved_detector_path = detector
            fake.device = Namespace(type="cuda")
            with patch.object(runner, "FVDRunner", return_value=fake), \
                    patch.object(runner.fvd, "check_device"), patch.object(runner.fvd, "verify_video"):
                meta = runner.feature_grid(args, items, stats.RUNS, 60, out)
                self.assertEqual(fake.calls, 12)
                self.assertIn(str(detector), meta["sources"])
                runner.feature_grid(args, items, stats.RUNS, 60, out)
                self.assertEqual(fake.calls, 12, "Reuse feature receipts instead of decoding again")
                with np.load(out / "cohort_features.npz") as data:
                    self.assertEqual(set(data.files), {"GT", *stats.RUNS})
                    self.assertTrue(all(data[k].shape == (2, 4, 3) for k in data.files))
                (videos / stats.KEEP / (items[0]["output_prefix"]+"custom.mp4")).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Stale FVD cache"):
                    runner.feature_grid(args, items, stats.RUNS, 60, out)

    def test_replay_cache_checks_actual_gt_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = dict(scene="scene", start_frame=10, duration_sec=60, num_frames=2,
                        output_prefix="seed0_scene_0010_60s_", gt_frames_dir=str(root / "frames"))
            (root / "frames").mkdir()
            sources = []
            for i in (10, 11):
                path = root / "frames" / f"{i:04d}.png"
                path.write_bytes(str(i).encode())
                sources.append(dict(index=i, path=str(path), sha256=runner.digest(path)))
            cache = root / "cache"
            features = np.zeros((2, 768), dtype=np.float32)
            features[:, 0] = 1
            for kind in ("baseline", "gt"):
                (cache / kind).mkdir(parents=True)
                path = cache / kind / (item["output_prefix"]+"dino.npy")
                np.save(path, features)
                meta = dict(kind=kind, scene="scene", dataset_start_frame=10, duration_sec=60, num_frames=2,
                            output_prefix=item["output_prefix"], encoder={"model": "fixture"},
                            feature_sha256=runner.digest(path))
                if kind == "gt":
                    meta.update(mode="fresh", source_frames=sources)
                else:
                    (root / "baseline").mkdir()
                    video = root / "baseline" / (item["output_prefix"]+"custom.mp4")
                    video.write_bytes(b"video fixture")
                    meta.update(source_sha256=runner.digest(video))
                path.with_suffix(".json").write_text(json.dumps(meta))
            arrays, recorded = replay.verified_cache(cache, item, root)
            self.assertEqual(arrays["gt"].shape, (2, 768))
            self.assertIn(str(root / "frames/0010.png"), recorded)
            (root / "frames/0010.png").write_bytes(b"modified gt")
            with self.assertRaisesRegex(ValueError, "GT source changed"):
                replay.verified_cache(cache, item, root)

    def test_cpu_main_preserves_partial_status_and_runs_remaining_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["run", "--phase", "cpu", "--output", str(root)]
            with patch.object(runner.sys, "argv", argv), patch.dict(runner.os.environ, SLURM_JOB_ID="fixture"), \
                    patch.object(runner, "verify_plan"), patch.object(runner, "cohort", return_value=[]), \
                    patch.object(runner, "archive_profiles", side_effect=ValueError("missing trace")), \
                    patch.object(runner, "export_scalars", return_value={}), \
                    patch.object(runner, "latency", return_value={}), \
                    patch.object(runner, "run_replay", return_value={}):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 1)
            state = json.loads((root / "status_cpu.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            self.assertEqual([s["status"] for s in state["stages"]], ["failed", "complete", "complete", "complete"])

    def test_stage_failure_does_not_prevent_independent_stage_and_resume_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fail(path):
                raise ValueError("missing upstream data")
            def success(path):
                (path / "answer.csv").write_text("fixture\n")
                return dict(sources={})
            self.assertEqual(runner.stage(root, "bad", fail)["status"], "failed")
            self.assertEqual(runner.stage(root, "good", success)["status"], "complete")
            reused = runner.stage(root, "good", fail)
            self.assertTrue(reused["reused"])
            (root / "good/answer.csv").write_text("changed")
            self.assertEqual(runner.stage(root, "good", success)["status"], "failed")

    def test_archive_audit_without_profiles_exports_missing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "context_memory_60s"
            videos.mkdir()
            fixture = trace_fixture(videos, runs=[runner.RUN_SPECS[r] for r in runner.TIMING_RUNS])
            items = [json.loads(fixture.manifest.read_text())]
            out = root / "out"
            out.mkdir()
            meta = runner.archive_profiles(Namespace(root=root), items, out)
            self.assertEqual(len(meta["missing_or_invalid_profiles"]), len(runner.TIMING_RUNS))
            with (out / "archive_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(int(rows[0]["final_stored_min"]), 305)
            self.assertEqual(int(rows[1]["final_stored_min"]), 32)

    def test_profiles_allow_first_section_without_retrieval_and_missing_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "context_memory_60s"
            videos.mkdir()
            fixture = trace_fixture(videos, runs=[runner.RUN_SPECS[r] for r in runner.TIMING_RUNS])
            items = [json.loads(fixture.manifest.read_text())]
            out = root / "out"
            out.mkdir()
            path = videos / "baseline/profiles/seed0_fixture_60s_custom.jsonl"
            path.parent.mkdir()
            profiles = [dict(event="section_profile", section_idx=s, scene="fixture", dataset_start_frame=0,
                             duration_sec=60, memory_policy="unbounded", stored_memory_size=(s+1)*76+1,
                             cumulative_rollout_latency_s=s+1, bank_frame_bytes=100, bank_feature_bytes=2,
                             peak_cuda_allocated_gb=1, peak_rss_gb=None,
                             phase_latency_s={} if s == 0 else {"context_selection": 1}) for s in range(4)]
            path.write_text("\n".join(map(json.dumps, profiles)))
            runner.archive_profiles(Namespace(root=root), items, out)
            with (out / "historical_profiles.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["context_selection_s"]), 3)
            self.assertEqual(rows[0]["peak_rss_gb"], "")

    def test_submission_dependencies_partial_failure_and_no_duplicate_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(output=str(root), code=str(root / "code"), after_job="848505",
                        gpu_partition="highgpu", gpu_type="nvidia_h100_80gb_hbm3", fvd_draws=10,
                        **{k: str(root / k) for k in ("root", "manifest", "manifest_180", "scores", "cache", "fvd_cache")})
            with patch.object(submit.subprocess, "check_output", side_effect=["101\n", subprocess.CalledProcessError(1, "sbatch")]) as call:
                with self.assertRaises(subprocess.CalledProcessError):
                    submit.submit(plan)
                self.assertEqual(json.loads((root / "jobs.json").read_text()), {"cpu": "101"})
                self.assertIn("--dependency=afterany:848505", call.call_args_list[1].args[0])
                self.assertIn("--gres=gpu:nvidia_h100_80gb_hbm3:1", call.call_args_list[1].args[0])
            with patch.object(submit, "job_state", return_value="PENDING"), \
                    patch.object(submit.subprocess, "check_output", side_effect=["102\n", "103\n"]) as call:
                jobs = submit.submit(plan)
                self.assertEqual(jobs, dict(cpu="101", gpu="102", **{"fvd-score": "103"}))
                self.assertIn("--dependency=afterany:101:102", call.call_args_list[1].args[0])
            with patch.object(submit, "job_state", return_value="RUNNING"), patch.object(submit.subprocess, "check_output") as call:
                submit.submit(plan)
                call.assert_not_called()

    def test_snapshot_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("manifest", "manifest_180", "scores"):
                (root / name).write_text("fixture")
            out = root / "out"
            out.mkdir()
            args = Namespace(output=out, root=root, manifest=root/"manifest", manifest_180=root/"manifest_180",
                             scores=root/"scores", cache=root/"cache", fvd_cache=root/"fvd_cache", fvd_draws=20,
                             gpu_partition="highgpu", gpu_type="nvidia_h100_80gb_hbm3", after_job="848505")
            plan = submit.prepare(args)
            self.assertTrue((out / "code/paper/run_paper_evidence.py").exists())
            self.assertTrue((out / "code/utils/compare_fvd_matched.py").exists())
            runner.verify_plan(args)
            self.assertEqual(submit.prepare(args), plan)
            args.scores.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                submit.prepare(args)

    def test_retry_repoints_pending_score_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(output=str(root), code=str(root / "code"), after_job="848505",
                        gpu_partition="highgpu", gpu_type="nvidia_h100_80gb_hbm3", fvd_draws=10,
                        **{k: str(root / k) for k in ("root", "manifest", "manifest_180", "scores", "cache", "fvd_cache")})
            (root / "jobs.json").write_text(json.dumps(dict(cpu="101", gpu="102", **{"fvd-score": "103"})))
            with patch.object(submit, "job_state", side_effect=["FAILED", "PENDING", "PENDING"]), \
                    patch.object(submit.subprocess, "check_output", return_value="104"), \
                    patch.object(submit.subprocess, "run") as update:
                submit.submit(plan)
                update.assert_called_once_with(["scontrol", "update", "JobId=103", "Dependency=afterany:104:102"], check=True)

    def test_launchers_preserve_cuda_mask_and_request_only_one_gpu(self):
        cpu = (runner.ROOT / "slurm/newton_paper_evidence_cpu.sbatch").read_text()
        gpu = (runner.ROOT / "slurm/newton_paper_evidence_gpu.sbatch").read_text()
        self.assertNotIn("#SBATCH --gres", cpu)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1", gpu)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", gpu)
        self.assertNotIn("srun ", gpu)
        for name in ("cpu", "gpu"):
            subprocess.run(["bash", "-n", str(runner.ROOT / f"slurm/newton_paper_evidence_{name}.sbatch")], check=True)


if __name__ == "__main__":
    unittest.main()
