import ast
from collections import defaultdict
import os
from pathlib import Path
import sys
import tempfile
import unittest


UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
import run_vbench_long as adapter
sys.path.pop(0)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def rows(self, stem, scores):
        (self.root / (stem + ".mp4")).touch()
        return [{"video_path": str(self.root / "split_clip" / stem / f"{stem}_{i:03d}.mp4"),
                 "video_results": score} for i, score in enumerate(scores)]

    def test_full_names_and_equal_video_weighting(self):
        rows = self.rows("seed0_room_a_60s_custom", [.2, .4, .6])
        rows += self.rows("seed0_room_b_60s_custom", [.8])
        overall, details, videos = adapter.reorganize_clips_results(rows)
        self.assertIs(details, rows)
        self.assertEqual(len(videos), 2)
        self.assertAlmostEqual(overall, .6)
        self.assertEqual(Path(videos[0]["video_path"]).name, "seed0_room_a_60s_custom.mp4")

    def test_boolean_and_imaging_scaling(self):
        overall, _, _ = adapter.reorganize_clips_results(self.rows("room", [True, False]))
        self.assertEqual(overall, .5)
        overall, _, videos = adapter.reorganize_clips_results(
            self.rows("room", [40., 80.]), "imaging_quality")
        self.assertEqual(overall, .6)
        self.assertEqual(videos[0]["video_results"], 60.)

    def test_rejects_invalid_layout_missing_source_and_duplicates(self):
        for path in ("/input/filtered_clips/room_000.mp4", "/input/split_clip/room/other_000.mp4"):
            with self.assertRaisesRegex(ValueError, "layout"):
                adapter.source_video(path)
        with self.assertRaisesRegex(ValueError, "Missing source"):
            adapter.source_video(str(self.root / "split_clip/missing/missing_000.mp4"))
        rows = self.rows("room", [.8])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.reorganize_clips_results(rows + rows)
        with self.assertRaisesRegex(ValueError, "empty"):
            adapter.reorganize_clips_results([])

    def test_upstream_numeric_equivalence(self):
        path = UTILS.parents[1] / "VBench/vbench2_beta_long/utils.py"
        if not path.exists():
            self.skipTest("Sibling upstream checkout not available")
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "reorganize_clips_results")
        namespace = {"os": os, "defaultdict": defaultdict}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        rows = self.rows("room_a", [20., 40.]) + self.rows("room_b", [80.])
        # Upstream's expected legacy names/layout, using distinct no-underscore IDs.
        legacy = [{"video_path": f"/input/filtered_clips/{name}_{i:03d}.mp4", "video_results": score}
                  for name, scores in (("a", [20., 40.]), ("b", [80.]))
                  for i, score in enumerate(scores)]
        for dimension in (None, "imaging_quality"):
            expected = namespace["reorganize_clips_results"](legacy, dimension)
            actual = adapter.reorganize_clips_results(rows, dimension)
            self.assertEqual(actual[0], expected[0])
            self.assertEqual([r["video_results"] for r in actual[2]],
                             [r["video_results"] for r in expected[2]])


if __name__ == "__main__":
    unittest.main()
