import json
from pathlib import Path
import tempfile
import unittest

from utils.stage_matched_video_cohort import load_cohort, stage_run


class StageMatchedVideoCohortTest(unittest.TestCase):
    def test_only_manifest_cohort_is_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            records = [
                {"duration_sec": 60, "output_prefix": "a_60s_"},
                {"duration_sec": 30, "output_prefix": "ignored_30s_"},
                {"duration_sec": 60, "output_prefix": "b_60s_"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            source = root / "source" / "baseline"
            source.mkdir(parents=True)
            for name in ("a_60s_custom.mp4", "b_60s_custom.mp4", "extra_60s_custom.mp4"):
                (source / name).write_bytes(b"video")

            cohort = load_cohort(manifest, 60)
            count = stage_run(root / "source", root / "staged", "baseline", cohort)

            self.assertEqual(count, 2)
            staged = root / "staged" / "baseline"
            self.assertEqual(sorted(path.name for path in staged.glob("*.mp4")),
                             ["a_60s_custom.mp4", "b_60s_custom.mp4"])
            self.assertTrue(all(path.is_symlink() for path in staged.glob("*.mp4")))

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                stage_run(root / "source", root / "staged", "baseline",
                          [(3, {"output_prefix": "missing_60s_"})])


if __name__ == "__main__":
    unittest.main()
