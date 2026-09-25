from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper import paired_quality_uncertainty as ci


def fixture(root, missing=False):
    for policy in ci.LABELS:
        directory = root / policy
        directory.mkdir()
        rows = [dict(scene=f"scene{i}", start_frame=i, duration_sec=180,
                     num_frames_expected=181, frames_seen=181, frames_evaluated=3,
                     frame_stride=90, status="completed", lpips_alex=.6 + i*.01 - .02*(policy == "keepsake"))
                for i in range(3)]
        if missing and policy == "baseline":
            rows[-1]["status"] = "missing_output"
            rows[-1]["lpips_alex"] = None
        (directory / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        summary = dict(metric_config=dict(frame_stride=90, learned_image_size=224, max_frames=None,
                                         fvd_clips_per_video=8, fvd_frame_stride=8),
                       by_duration={"180": dict(completed_or_short=2 if missing and policy == "baseline" else 3,
                                                lpips_alex=.6, fvd=400.)})
        (directory / "summary.json").write_text(json.dumps(summary))
    output = root / "out"
    output.mkdir()
    return Namespace(baseline=root/"baseline", keepsake=root/"keepsake", output=output,
                     expected_videos=3, duration=180, allow_matched_subset=False,
                     draws=30, seed=17, cluster_map=None)


class PairedUncertaintyTests(unittest.TestCase):
    def test_pairing_keeps_constant_difference(self):
        items = [dict(scene=str(i)) for i in range(3)]
        values = dict(baseline=[.2, .5, .9], keepsake=[.1, .4, .8])
        rows, loo, samples = ci.paired_statistics(items, values, draws=60)
        self.assertAlmostEqual(rows[0]["difference"], -.1)
        self.assertAlmostEqual(rows[0]["ci_low"], -.1)
        self.assertAlmostEqual(rows[0]["ci_high"], -.1)
        self.assertEqual(len(loo), 3)
        self.assertEqual(samples["LPIPS"].shape, (60, 2))

    def test_clusters_travel_together(self):
        items = [dict(scene="A"), dict(scene="B"), dict(scene="C")]
        labels, groups = ci.grouping(items, dict(A="house1", B="house1", C="house2"))
        self.assertEqual(len(groups), 2)
        for ix in ci.resamples(groups, 40, 17):
            self.assertEqual(list(ix).count(0), list(ix).count(1))
        with self.assertRaises(ValueError):
            ci.grouping(items, dict(A="house1"))

    def test_pooled_fvd_not_mean_of_per_video(self):
        items = [dict(scene=str(i)) for i in range(3)]
        gt = np.random.default_rng(1).normal(size=(3, 4, 2))
        features = dict(GT=gt, baseline=gt+.5, keepsake=gt+.2)
        rows, _, samples = ci.paired_statistics(items, dict(baseline=[.4]*3, keepsake=[.3]*3), features, draws=20)
        row = rows[1]
        self.assertAlmostEqual(row["baseline"], .5, places=8)
        self.assertAlmostEqual(row["keepsake"], .08, places=8)
        self.assertAlmostEqual(row["ci_low"], -.42, places=8)
        self.assertAlmostEqual(row["ci_high"], -.42, places=8)
        self.assertEqual(samples["FVD"].shape, (20, 2))

    def test_missing_cohort_rejected_after_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), missing=True)
            with self.assertRaisesRegex(ValueError, "Only 2/3"):
                ci.legacy(args)
            self.assertTrue((args.output / "source_audit.csv").exists())
            self.assertFalse((args.output / "contrasts.csv").exists())
            args.allow_matched_subset = True
            ci.legacy(args)
            metadata = json.loads((args.output / "analysis.json").read_text())
            self.assertIn("Partial", metadata["scope"])
            self.assertEqual(metadata["videos"], 2)

    def test_mismatched_sampling_not_repaired_by_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            path = args.baseline / "summary.json"
            data = json.loads(path.read_text())
            data["metric_config"]["frame_stride"] = 30
            path.write_text(json.dumps(data))
            args.allow_matched_subset = True
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                ci.legacy(args)

    def test_duplicate_and_short_completed_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            path = args.baseline / "metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["frames_seen"] = 100
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "frame coverage"):
                ci.legacy(args)
            rows[0] = rows[1]
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ci.legacy(args)

    def test_explicit_legacy_fvd_sampler(self):
        indices = ci.sample_indices(dict(num_frames=5397), dict(clip_length=16, clips_per_video=8, frame_stride=8))
        self.assertEqual(len(indices), 128)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 5396)

    def test_nonfinite_and_unpaired_features_rejected(self):
        items = [dict(scene=str(i)) for i in range(3)]
        values = dict(baseline=[.3]*3, keepsake=[.4]*3)
        with self.assertRaises(ValueError):
            ci.paired_statistics(items, {**values, "keepsake": [np.nan]*3})
        gt = np.ones((3, 4, 2))
        with self.assertRaises(ValueError):
            ci.paired_statistics(items, values, dict(GT=gt, baseline=gt[:2], keepsake=gt))

    def test_report_from_portable_cache_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = root / "cells"
            cells.mkdir()
            items = [dict(scene=f"scene{i}", start_frame=0) for i in range(3)]
            plan = dict(items=items, duration_sec=180, expected_videos=3,
                        config=dict(clips_per_video=4))
            ci.save_json(root / "plan.json", plan)
            for i, item in enumerate(items):
                data = np.random.default_rng(i).normal(size=(4, 2))
                path = cells / f"pair_{i:03d}.npz"
                np.savez(path, GT=data, baseline=data+.5, keepsake=data+.2)
                ci.save_json(path.with_suffix(".json"), dict(item=item,
                    inputs=dict(plan_sha256=ci.digest(root / "plan.json")),
                    feature_sha256=ci.digest(path), scores=dict(baseline=.5, keepsake=.4)))
            args = Namespace(output=root, duration=180, expected_videos=3,
                             cluster_map=None, draws=10, seed=17)
            ci.report(args)
            self.assertTrue((root / "fvd_numerical_check.csv").exists())
            self.assertIn("FVD", (root / "paired_uncertainty.tex").read_text())
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ci.report(args)


if __name__ == "__main__":
    unittest.main()
