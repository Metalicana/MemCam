from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper import run_headline_uncertainty_suite as suite


def inputs(n=3):
    items = [dict(scene=f"s{i}", start_frame=i, duration_sec=60, num_frames=1825) for i in range(n)]
    gt = np.random.default_rng(13).normal(size=(n, 4, 2))
    values = {"baseline": np.linspace(.3, .7, n), suite.KEEP: np.linspace(.2, .6, n)}
    features = {"GT": gt, "baseline": gt+.5, suite.KEEP: gt+.2}
    return items, values, features


def pair_fixture(root, duration=180):
    items, values, features = inputs()
    for item in items:
        item["duration_sec"] = duration
    if duration == 180:
        features = {k: np.concatenate([v, v], axis=1) for k, v in features.items()}
    root.mkdir(exist_ok=True)
    (root/"cells").mkdir()
    plan = dict(items=items, duration_sec=duration, config=suite.config(duration),
                lpips_stride=30 if duration == 60 else 90,
                detector_sha256="detector", code_hashes={"extract.py": "abc"})
    suite.pair.save_json(root/"plan.json", plan)
    source = root/"source.mp4"
    source.write_bytes(b"synthetic input signature")
    for index, item in enumerate(items):
        path = root/"cells"/f"pair_{index:03d}.npz"
        np.savez(path, GT=features["GT"][index], baseline=features["baseline"][index], keepsake=features[suite.KEEP][index])
        suite.pair.save_json(path.with_suffix(".json"), dict(item=item,
            inputs=dict(plan_sha256=suite.pair.digest(root/"plan.json"), environment=dict(torch="test"),
                        lpips_sha256="lpips", sources={str(source): suite.pair.digest(source)}),
            scores=dict(baseline=float(values["baseline"][index]), keepsake=float(values[suite.KEEP][index])),
            feature_sha256=suite.pair.digest(path)))
    return plan


class HeadlineSuiteTests(unittest.TestCase):
    def test_scope_and_distinct_metric_protocols(self):
        self.assertEqual(sum(2*(len(runs)-1) for _, runs in suite.GROUPS.values()), 18)
        self.assertEqual(suite.config(60)["clips_per_video"], 4)
        self.assertEqual(suite.config(180)["frame_stride"], 8)
        self.assertNotIn("ri_b32_dino_rgb", suite.GROUPS["main_60s"][1])
        args = Namespace(root=Path("/results"), detector=Path("/detector"))
        self.assertEqual(suite.pair_args(args, 180, "fifo_b32", Path("/out")).lpips_stride, 90)
        self.assertEqual(suite.pair_args(args, 60, "fifo_b32", Path("/out")).lpips_stride, 30)

    def test_constant_paired_difference_and_pooled_fvd(self):
        items, values, features = inputs()
        summary, contrasts, boot, labels = suite.statistics(items, values, features, draws=30, swaps=20)
        self.assertEqual(boot.shape, (30, 2, 2))
        lp, fvd = contrasts
        self.assertAlmostEqual(lp["difference"], -.1)
        self.assertAlmostEqual(lp["ci_low"], -.1)
        self.assertAlmostEqual(lp["ci_high"], -.1)
        self.assertAlmostEqual(fvd["difference"], -.42, places=8)
        self.assertAlmostEqual(fvd["reduction_percent"], 84., places=7)
        self.assertEqual(lp["test"], "exact_cluster_label_swap")
        self.assertAlmostEqual(lp["p_two_sided"], .25)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(labels), 3)

    def test_cluster_resampling_and_swaps_are_groupwise(self):
        items, values, features = inputs(4)
        mapping = dict(s0="a", s1="a", s2="b", s3="b")
        _, comparisons, _, _ = suite.statistics(items, values, features, 20, 100, mapping=mapping)
        self.assertEqual(comparisons[0]["groups"], 2)
        self.assertEqual(comparisons[0]["permutations"], 4)
        self.assertAlmostEqual(comparisons[0]["p_two_sided"], .5)
        with self.assertRaises(ValueError):
            suite.statistics(items, values, features, mapping=dict(s0="a"))

    def test_null_monte_carlo_and_reproducibility(self):
        items, values, features = inputs(5)
        values["baseline"] = values[suite.KEEP].copy()
        features["baseline"] = features[suite.KEEP].copy()
        a = suite.statistics(items, values, features, draws=20, swaps=10)
        b = suite.statistics(items, values, features, draws=20, swaps=10)
        self.assertTrue(np.array_equal(a[2], b[2]))
        self.assertTrue(all(r["p_two_sided"] == 1 for r in a[1]))
        self.assertEqual(a[1][0]["test"], "monte_carlo_cluster_label_swap")

    def test_reject_missing_nonfinite_and_duplicates(self):
        items, values, features = inputs()
        with self.assertRaises(ValueError):
            suite.statistics(items, values, {"GT": features["GT"]})
        values["baseline"][0] = np.nan
        with self.assertRaises(ValueError):
            suite.statistics(items, values, features)
        items, values, features = inputs()
        items[-1] = items[0]
        with self.assertRaises(ValueError):
            suite.statistics(items, values, features)

    def test_portable_pair_hashes_and_source_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_fixture(root)
            data = suite.load_pair(root, check_sources=True)
            self.assertEqual(data[2]["GT"].shape, (3, 8, 2))
            (root/"source.mp4").write_bytes(b"changed")
            suite.load_pair(root)
            with self.assertRaisesRegex(ValueError, "Changed video"):
                suite.load_pair(root, check_sources=True)
            (root/"cells/pair_000.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                suite.load_pair(root)

    def test_group_report_reuse_and_no_silent_reference_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_fixture(root/"pair")
            loaded = suite.load_pair(root/"pair")
            pairs = {(180, r): loaded for r in ("baseline", "fifo_b32")}
            args = Namespace(output=root, draws=15, swaps=10, seed=17, cluster_map=None)
            suite.export_group(args, "long_180s", pairs)
            folder = root/"long_180s"
            table = (folder/"table.tex").read_text()
            self.assertIn("FIFO B32", table)
            self.assertIn("95\\% CI", table)
            with patch.object(suite, "statistics", side_effect=AssertionError("must reuse")):
                suite.export_group(args, "long_180s", pairs)
            v = {k: a.copy() for k, a in loaded[1].items()}
            v["keepsake"][0] += .01
            pairs[(180, "fifo_b32")] = (loaded[0], v, *loaded[2:])
            with self.assertRaisesRegex(ValueError, "reference"):
                suite.export_group(args, "long_180s", pairs)

    def test_missing_comparator_does_not_export_partial_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(output=Path(tmp), draws=15, swaps=10, seed=17, cluster_map=None)
            with self.assertRaises(KeyError):
                suite.export_group(args, "main_60s", {})
            self.assertFalse((args.output/"main_60s/table.tex").exists())

    def test_full_suite_and_resume_with_optional_missing_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for duration in (60, 180):
                pair_fixture(root/str(duration), duration)
            out = root/"out"
            (out/"pair_index").mkdir(parents=True)
            args = Namespace(output=out, root=root, draws=12, swaps=10, seed=17,
                             cluster_map=None, phase="run")
            def extract(args, duration, run):
                directory = root/str(duration)
                return directory, suite.load_pair(directory)
            with patch.object(suite, "freeze", return_value=dict(sources={})), \
                    patch.object(suite, "extract_pair", side_effect=extract), \
                    patch("paper.evidence_statistics.export_scalars", side_effect=FileNotFoundError("no legacy data")):
                suite.run(args)
                status = suite.read(out/"status.json")
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["stages"][-1]["status"], "unavailable")
                import csv
                with (out/"all_quality_contrasts.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 18)
                self.assertTrue(all(float(r["p_holm_all_18"]) >= float(r["p_two_sided"]) for r in rows))
                self.assertTrue((out/"reports.zip").is_file())
                args.phase = "report"
                with patch.object(suite, "statistics", side_effect=AssertionError("must reuse")), \
                        patch.object(suite, "extract_pair", side_effect=AssertionError("no GPU extraction")):
                    suite.run(args)

    def test_failed_pair_keeps_independent_tables_and_no_global_partial_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for duration in (60, 180):
                pair_fixture(root/str(duration), duration)
            out = root/"out"
            (out/"pair_index").mkdir(parents=True)
            args = Namespace(output=out, root=root, draws=12, swaps=10, seed=17,
                             cluster_map=None, phase="run")
            def extract(args, duration, run):
                if run == "mce_b32_lambda1_pilot":
                    raise FileNotFoundError("missing scene")
                directory = root/str(duration)
                return directory, suite.load_pair(directory)
            with patch.object(suite, "freeze", return_value=dict(sources={})), \
                    patch.object(suite, "extract_pair", side_effect=extract), \
                    patch("paper.evidence_statistics.export_scalars", side_effect=FileNotFoundError("no legacy data")):
                with self.assertRaisesRegex(RuntimeError, "Some quality groups"):
                    suite.run(args)
            self.assertFalse((out/"main_60s/table.tex").exists())
            self.assertTrue((out/"budget_180s/table.tex").exists())
            self.assertTrue((out/"long_180s/table.tex").exists())
            self.assertFalse((out/"all_quality_contrasts.csv").exists())
            self.assertEqual(suite.read(out/"status.json")["status"], "incomplete")

    def test_sbatch_syntax_and_one_gpu_no_generation(self):
        script = suite.ROOT/"slurm/newton_headline_uncertainty_suite.sbatch"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        self.assertIn("--gres=gpu:nvidia_h100_80gb_hbm3:1", text)
        self.assertNotIn("--array", text)
        env = dict(__import__("os").environ)
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run([sys.executable, str(Path(suite.__file__)), "run"], env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("login node", result.stderr)


if __name__ == "__main__":
    unittest.main()
