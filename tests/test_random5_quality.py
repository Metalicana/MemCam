import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
import evaluate_random5_quality as quality


def fixture(root):
    items = [dict(scene=f"scene{i:02d}", start_frame=100, duration_sec=60, num_frames=1825,
                  fps=30, prompt="test", output_prefix=f"seed0_scene{i:02d}_0100_60s_",
                  gt_frames_dir=str(root / f"gt{i}"), _row=i) for i in range(15)]
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, items)) + "\n")
    sources, records_by_run = [], {}
    for method_index, run in enumerate(quality.METHODS.values()):
        folder = root / "legacy" / run
        folder.mkdir(parents=True)
        records = [dict(run_name=run, scene=item["scene"], start_frame=100, duration_sec=60,
                        num_frames_expected=1825, frames_seen=1825, frames_evaluated=61,
                        frame_stride=30, status="completed", psnr_db=20 - method_index + i * .1,
                        output=str(root / run / f"{item['output_prefix']}custom.mp4"))
                   for i, item in enumerate(items)]
        path = folder / "metrics.jsonl"
        path.write_text("\n".join(map(json.dumps, records)) + "\n")
        records_by_run[run] = records
        sources.append(dict(run=run, evaluator="quality", metric="LPIPS", source=str(path)))
    scores = root / "scores.csv"
    quality.write_csv(scores, sources)
    args = argparse.Namespace(manifest=manifest, root=root / "videos", scores=scores,
                              dataset_root=None, output=root / "output", psnr_only=True,
                              saved_only=True, psnr_cohort="subset", audit_only=False,
                              fvd_cache_dir=root / "i3d")
    return args, items, records_by_run


class RandomFiveTests(unittest.TestCase):
    def test_seed_zero_draw_is_fixed_and_independent_of_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, items, _ = fixture(Path(temporary))
            expected = [items[i] for i in (0, 6, 12, 13, 14)]
            self.assertEqual(quality.select_subset(items), expected)
            self.assertEqual(quality.select_subset(list(reversed(items))), expected)
            with self.assertRaises(ValueError):
                quality.select_subset(items[:-1])
            with self.assertRaises(ValueError):
                quality.select_subset(items[:-1] + [items[0]])

    def test_saved_psnr_requires_exact_identity_length_and_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, items, records = fixture(Path(temporary))
            loaded, sources = quality.saved_psnr(args.scores, items)
            self.assertEqual(len(loaded), 90)
            self.assertEqual(len(sources), 6)
            record = records["baseline"][0]
            for key, value in (("scene", "other"), ("start_frame", 101), ("duration_sec", 180),
                               ("frames_seen", 1800), ("frames_evaluated", 60),
                               ("frame_stride", 1), ("status", "short_video"),
                               ("psnr_db", float("nan")), ("run_name", "fifo_b32")):
                self.assertFalse(quality.matching_psnr(dict(record, **{key: value}), items[0], "baseline"))
            wrong = dict(record, output="different.mp4")
            self.assertFalse(quality.matching_psnr(wrong, items[0], "baseline"))

    def test_saved_only_cli_reports_all_methods_without_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, items, _ = fixture(Path(temporary))
            completed = subprocess.run([sys.executable, str(ROOT / "utils/evaluate_random5_quality.py"),
                                       "--manifest", str(args.manifest), "--scores", str(args.scores),
                                       "--output", str(args.output), "--psnr-only", "--saved-only",
                                       "--psnr-cohort", "all15"],
                                      text=True, capture_output=True, check=True, timeout=30)
            self.assertIn("KEEPSAKE", completed.stdout)
            with (args.output / "psnr_all15.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(r["videos"]) for r in rows}, {15})
            self.assertAlmostEqual(float(rows[0]["psnr_db"]), 20.7)
            self.assertFalse((args.output / "scores.csv").exists())
            self.assertFalse((args.output / "device_diagnostics.json").exists())
            cohort = json.loads((args.output / "cohort.json").read_text())
            self.assertEqual(cohort["subset"], quality.select_subset(items))

    def test_missing_or_duplicate_psnr_is_not_silently_dropped(self):
        for duplicate in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                args, items, records = fixture(Path(temporary))
                path = args.scores.parent / "legacy/baseline/metrics.jsonl"
                changed = records["baseline"] + [records["baseline"][0]] if duplicate else records["baseline"][1:]
                path.write_text("\n".join(map(json.dumps, changed)) + "\n")
                with self.assertRaises(ValueError):
                    quality.get_psnr(args, items)

    def test_full_table_uses_same_five_and_bolds_actual_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, items, _ = fixture(Path(temporary))
            args.psnr_only = args.saved_only = False
            args.psnr_cohort = "all15"
            fvd = {run: 100.0 + i for i, run in enumerate(quality.METHODS.values())}
            with patch.object(quality, "get_fvd", return_value=fvd) as compute:
                quality.execute(args)
            self.assertEqual(compute.call_args.args[1], quality.select_subset(items))
            with (args.output / "scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(r["videos"]) for r in rows}, {5})
            self.assertEqual({int(r["fvd_clips"]) for r in rows}, {20})
            table = (args.output / "table.tex").read_text()
            self.assertIn(r"Unbounded & \textbf{100.0}", table)
            self.assertIn("Exploratory subset", table)
            self.assertNotIn(r"\textbf{105.0}", table)

    def test_launcher_is_one_job_without_conda_loop_or_cuda_override(self):
        launcher = ROOT / "slurm/newton_random5_quality.sbatch"
        text = launcher.read_text()
        self.assertNotIn("conda activate", text)
        self.assertNotIn("conda run", text)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", text)
        self.assertNotIn("srun ", text)
        self.assertNotIn("timeout ", text)
        subprocess.run(["bash", "-n", str(launcher)], check=True)

    def test_fvd_requires_twenty_clips_for_each_of_six_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, items, _ = fixture(root)
            args.output.mkdir()
            selected = quality.select_subset(items)
            for item in selected:
                gt = Path(item["gt_frames_dir"])
                gt.mkdir()
                (gt / "0100.png").write_bytes(b"fixture")
            detector = root / "detector.pt"
            detector.write_bytes(b"fixture")
            runner = Mock(device="cuda", resolved_detector_path=detector)
            runner.compute_group.return_value = (123.0, 20)
            with patch.dict(sys.modules, {"torch": Mock()}), \
                    patch.object(quality, "check_device"), \
                    patch.object(quality, "verify_video"), \
                    patch.object(quality, "clip_indices", return_value=[[0]]), \
                    patch.object(quality, "FVDRunner", return_value=runner):
                results = quality.get_fvd(args, selected)
                self.assertEqual(set(results), set(quality.METHODS.values()))
                self.assertEqual(runner.compute_group.call_count, 6)
                for call in runner.compute_group.call_args_list:
                    self.assertEqual(call.args[0], selected)
                runner.compute_group.return_value = (123.0, 19)
                with self.assertRaisesRegex(ValueError, "Incomplete FVD"):
                    quality.get_fvd(args, selected)

    def test_existing_different_cohort_is_not_replaced_or_marked_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            args.output.mkdir()
            quality.save_json(args.output / "cohort.json", {"different": "cohort"})
            quality.save_json(args.output / "status.json", {"status": "complete"})
            with self.assertRaisesRegex(ValueError, "different cohort"):
                quality.execute(args)
            self.assertEqual(json.loads((args.output / "status.json").read_text()), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
