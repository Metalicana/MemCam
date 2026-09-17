import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gap_source", ROOT / "paper/cache_gap_source_features.py")
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class FakeEncoder:
    def encode_batch(self, images):
        features = np.zeros((len(images), 768), dtype=np.float32)
        for i, image in enumerate(images):
            features[i, image.getpixel((0, 0))[0]] = 1
        return features


def fixture(root):
    gt = root / "frames/test"
    gt.mkdir(parents=True)
    features = np.zeros((8, 768), dtype=np.float32)
    for i in range(8):
        Image.new("RGB", (4, 4), (i, 0, 0)).save(gt / f"{100+i:04d}.png")
        features[i, i] = 1
    old = root / "legacy/test_dino.npy"
    old.parent.mkdir()
    np.save(old, features)
    item = dict(scene="test", start_frame=100, duration_sec=60, num_frames=8,
                output_prefix="test_", gt_frames_dir=str(gt))
    return item, old, features


class GapFeatureCacheTests(unittest.TestCase):
    def test_legacy_check_uses_own_indices_and_reports_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item, old, features = fixture(root)
            actual, meta = cache.check_legacy_gt(old, item, FakeEncoder())
            np.testing.assert_array_equal(actual, features)
            self.assertEqual([r["index"] for r in meta["checked_frames"]], [0, 2, 4, 6, 7])
            self.assertFalse(meta["all_gt_frames_reencoded"])
            self.assertTrue(all(d == 0 for d in meta["cosine_distances"]))

    def test_wrong_legacy_features_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            item, old, features = fixture(Path(tmp))
            np.save(old, features[::-1])
            with self.assertRaisesRegex(ValueError, "spot check failed"):
                cache.check_legacy_gt(old, item, FakeEncoder())

    def test_cache_resumption_verifies_data_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, features = fixture(root)
            path = root / "new/features.npy"
            meta = {"num_frames": 8, "source_sha256": "source-a", "encoder": {"model": "DINO"}}
            self.assertFalse(cache.cached_match(path, meta))
            cache.save_features(path, features, meta)
            self.assertTrue(cache.cached_match(path, meta))
            with self.assertRaisesRegex(ValueError, "Stale"):
                cache.cached_match(path, {**meta, "source_sha256": "source-b"})
            np.save(path, features[::-1])
            with self.assertRaisesRegex(ValueError, "Stale"):
                cache.cached_match(path, meta)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                cache.cached_match(path, meta)

    def test_source_preflight_needs_existing_video_and_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item, old, _ = fixture(root)
            video = root / "baseline/test_custom.mp4"
            with self.assertRaises(FileNotFoundError):
                cache.prepare_sources([item], root, old.parent, None, False)
            video.parent.mkdir()
            video.write_bytes(b"source-video-is-not-decoded-by-preflight")
            sources = cache.prepare_sources([item], root, old.parent, None, False)
            self.assertEqual(len(sources), 1)
            (root / "frames/test/0103.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing GT"):
                cache.prepare_sources([item], root, old.parent, None, False)


if __name__ == "__main__":
    unittest.main()
