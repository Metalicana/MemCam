import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
import compare_fvd_matched as compare
from evaluate_context_memory import FVDRunner


def fixture(root, count=30):
    items = []
    for index in range(count):
        scene = f"scene{index}"
        gt = root / "frames" / scene
        gt.mkdir(parents=True)
        item = dict(scene=scene, start_frame=100, duration_sec=60, num_frames=80,
                    fps=30, prompt="test", output_prefix=f"seed0_{scene}_0100_60s_",
                    gt_frames_dir=str(gt))
        for frame in {i for clip in compare.clip_indices(item) for i in clip}:
            Image.new("RGB", (4, 4), (frame, index, 0)).save(gt / f"{100+frame:04d}.png")
        for run in compare.RUNS.values():
            path = compare.output_path(root / run, item)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"fixture {scene} {run}".encode())
        items.append(item)
    manifest, original = root / "all.jsonl", root / "original.jsonl"
    manifest.write_text("\n".join(map(json.dumps, items)))
    original.write_text("\n".join(map(json.dumps, items[:15])))
    return items, manifest, original


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def _load_item_clips(self, item, model_output_dir, dataset_root, max_frames):
        self.calls += 1
        n = int(item["scene"].replace("scene", ""))
        gt = [np.array([n, i, n * .1]) for i in range(4)]
        offset = .2 if model_output_dir.name.startswith("slam") else .5
        return [clip + offset for clip in gt], gt

    def _append_features(self, batches, clips):
        batches.append(np.stack(clips).astype(np.float32))


class MatchedFVDTests(unittest.TestCase):
    def test_fixed_manifest_not_directory_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, manifest, original = fixture(root)
            loaded = compare.cohort_items(manifest, 30)
            indices = compare.original_indices(loaded, compare.cohort_items(original, 15))
            self.assertEqual(indices, list(range(15)))
            compare.audit_inputs(loaded, root, None)
            with self.assertRaisesRegex(ValueError, "Expected 31"):
                compare.cohort_items(manifest, 31)
            missing = compare.output_path(root / compare.RUNS["RI"], items[-1])
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "no partial-cohort"):
                compare.audit_inputs(loaded, root, None)

    def test_manifest_duplicates_wrong_original_and_missing_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, manifest, _ = fixture(root, 2)
            manifest.write_text("\n".join(map(json.dumps, [items[0], items[0]])))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                compare.cohort_items(manifest, 2)
            with self.assertRaisesRegex(ValueError, "start_frame mismatch"):
                compare.original_indices(items, [dict(items[0], start_frame=101)])
            with self.assertRaisesRegex(ValueError, "missing"):
                compare.original_indices(items, [dict(items[0], output_prefix="unknown_")])
            (Path(items[0]["gt_frames_dir"]) / "0100.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing/empty"):
                compare.audit_inputs(items, root, None)

    def test_video_probe_rejects_short_video_and_wrong_fps(self):
        stream = dict(nb_frames="1825", avg_frame_rate="30/1", width=640, height=352)
        item = dict(num_frames=1825, fps=30)
        for changes, valid in (({}, True), ({"nb_frames": "1800"}, False),
                               ({"avg_frame_rate": "10/1"}, False)):
            payload = json.dumps(dict(streams=[dict(stream, **changes)]))
            with patch.object(compare.subprocess, "check_output", return_value=payload):
                if valid:
                    self.assertEqual(compare.verify_video(Path("video.mp4"), item), stream)
                else:
                    with self.assertRaisesRegex(ValueError, "length/FPS"):
                        compare.verify_video(Path("video.mp4"), item)

    def test_fast_fvd_matches_existing_evaluator_including_singular_covariances(self):
        reference = object.__new__(FVDRunner)
        rng = np.random.default_rng(4)
        for samples, dimensions in ((20, 4), (8, 40), (8, 1)):
            gt = rng.normal(size=(samples, dimensions))
            gen = rng.normal(size=(samples + 3, dimensions)) + .2
            self.assertAlmostEqual(compare.frechet_low_rank(gt, gen),
                                   reference._frechet_distance(gt, gen), places=4)
            self.assertAlmostEqual(compare.frechet_low_rank(gt, gt), 0, places=8)
        with self.assertRaises(ValueError):
            compare.frechet_low_rank(np.ones((1, 4)), np.ones((4, 4)))
        with self.assertRaises(ValueError):
            compare.frechet_low_rank(np.full((4, 4), np.nan), np.ones((4, 4)))

    def test_bootstrap_keeps_scene_replicates_and_clips_together(self):
        items = [dict(scene=scene) for scene in ("a", "a", "b", "c", "c")]
        first = list(compare.bootstrap_indices(items, 20, 17))
        second = list(compare.bootstrap_indices(items, 20, 17))
        for indices, repeated in zip(first, second):
            np.testing.assert_array_equal(indices, repeated)
            self.assertEqual(np.count_nonzero(indices == 0), np.count_nonzero(indices == 1))
            self.assertEqual(np.count_nonzero(indices == 3), np.count_nonzero(indices == 4))
        self.assertTrue(any(len(set(indices)) < len(items) for indices in first))

    def test_paired_difference_not_per_video_fvd_average(self):
        gt = np.random.default_rng(7).normal(size=(5, 4, 3))
        features = dict(GT=gt, RI=gt + .5, KEEPSAKE=gt + .2)
        items = [dict(scene=str(i)) for i in range(5)]
        result, differences = compare.compare_features(features, items, draws=25)
        self.assertAlmostEqual(result["ri_fvd"], .75, places=6)
        self.assertAlmostEqual(result["keepsake_fvd"], .12, places=6)
        self.assertAlmostEqual(result["difference"], -.63, places=6)
        np.testing.assert_allclose(differences, -.63, atol=1e-10)
        self.assertEqual(result["clips"], 20)

    def test_extraction_resume_and_stale_feature_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, _, _ = fixture(root, 2)
            runner = FakeRunner()
            encoder = dict(detector_sha256="test")
            with patch.object(compare, "verify_video", return_value={}):
                first = compare.extract_features(items, root, None, root, runner, encoder)
                second = compare.extract_features(items, root, None, root, runner, encoder)
            self.assertEqual(runner.calls, 4)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first["GT"].shape, (2, 4, 3))
            compare.output_path(root / compare.RUNS["RI"], items[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Stale"):
                compare.extract_features(items, root, None, root, runner, encoder)

    def test_short_decoding_and_invalid_features_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, _, _ = fixture(root, 1)
            with patch.object(compare, "verify_video", return_value={}), \
                    patch.object(FakeRunner, "_load_item_clips", return_value=([], [])):
                with self.assertRaisesRegex(ValueError, "Short/undecodable"):
                    compare.extract_features(items, root, None, root, FakeRunner(), {})
            with self.assertRaisesRegex(ValueError, "Invalid"):
                compare.validate_features({key: np.ones((3, 4)) for key in ("GT", *compare.RUNS)})

    def test_complete_synthetic_cohort_export_and_audit_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, manifest, original = fixture(root)
            output = root / "output"
            subprocess.run([sys.executable, str(ROOT / "utils/compare_fvd_matched.py"),
                            "--manifest", str(manifest), "--original-manifest", str(original),
                            "--root", str(root), "--output", str(output), "--audit-only"], check=True)
            with patch.object(compare, "verify_video", return_value={}):
                features = compare.extract_features(items, root, None, output, FakeRunner(), {})
            results = compare.write_comparisons(features, items, list(range(15)), output, 10, 17)
            self.assertEqual([r["videos"] for r in results], [30, 15, 15])
            self.assertEqual([r["clips"] for r in results], [120, 60, 60])
            self.assertTrue(all(r["difference"] < 0 for r in results))
            self.assertTrue((output / "scores.csv").is_file())
            self.assertEqual(len(json.loads((output / "summary.json").read_text())["results"]), 3)


if __name__ == "__main__":
    unittest.main()
