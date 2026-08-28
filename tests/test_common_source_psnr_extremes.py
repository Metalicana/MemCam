import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_common_source_psnr_extremes.py"
SPEC = importlib.util.spec_from_file_location(
    "common_source_psnr_extremes",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CommonSourcePsnrExtremesTest(unittest.TestCase):
    def test_query_stride_is_relative_to_section_start(self):
        self.assertTrue(MODULE.query_is_sampled(2, 153, 4))
        self.assertTrue(MODULE.query_is_sampled(2, 157, 4))
        self.assertFalse(MODULE.query_is_sampled(2, 154, 4))
        self.assertFalse(MODULE.query_is_sampled(2, 152, 4))

    def test_load_selected_queries_checks_identity(self):
        item = {
            "scene": "SceneA",
            "start_frame": 100,
            "duration_sec": 180,
        }
        row = {
            "event": "context_access",
            "selected": True,
            "scene": "SceneA",
            "dataset_start_frame": 100,
            "duration_sec": 180,
            "section_idx": 2,
            "target_frame": 153,
            "selected_memory_frame": 20,
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = MODULE.load_selected_queries(path, item)
            self.assertEqual(loaded[(2, 153)]["selected_memory_frame"], 20)

            wrong = dict(row, scene="SceneB")
            path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                MODULE.load_selected_queries(path, item)

    def test_diverse_selection_prefers_threshold_passes(self):
        rows = [
            {
                "row": 1,
                "target_frame": 100,
                "psnr_delta": 8.0,
                "ssim_delta": 0.2,
                "unbounded_psnr": 8.0,
                "geocov_psnr": 16.0,
                "unbounded_overlap": 0.9,
                "geocov_overlap": 0.9,
            },
            {
                "row": 1,
                "target_frame": 400,
                "psnr_delta": 7.0,
                "ssim_delta": 0.2,
                "unbounded_psnr": 9.0,
                "geocov_psnr": 16.0,
                "unbounded_overlap": 0.9,
                "geocov_overlap": 0.9,
            },
            {
                "row": 2,
                "target_frame": 200,
                "psnr_delta": 6.0,
                "ssim_delta": 0.1,
                "unbounded_psnr": 10.0,
                "geocov_psnr": 16.0,
                "unbounded_overlap": 0.9,
                "geocov_overlap": 0.9,
            },
            {
                "row": 3,
                "target_frame": 300,
                "psnr_delta": 3.0,
                "ssim_delta": 0.1,
                "unbounded_psnr": 10.0,
                "geocov_psnr": 13.0,
                "unbounded_overlap": 0.9,
                "geocov_overlap": 0.9,
            },
        ]
        selected = MODULE.select_diverse_extremes(
            rows,
            max_examples=2,
            per_row=1,
            min_target_gap=152,
            min_psnr_delta=4.0,
            min_ssim_delta=0.05,
            max_unbounded_psnr=12.0,
            min_geocov_psnr=14.0,
            min_overlap=0.8,
        )
        self.assertEqual([row["row"] for row in selected], [1, 2])
        self.assertTrue(all(row["meets_extreme_thresholds"] for row in selected))

    def test_error_heatmap_uses_same_fixed_scale(self):
        ground_truth = np.zeros((4, 4, 3), dtype=np.uint8)
        small_error = np.full((4, 4, 3), 24, dtype=np.uint8)
        large_error = np.full((4, 4, 3), 96, dtype=np.uint8)
        small = np.asarray(MODULE.error_heatmap(small_error, ground_truth))
        large = np.asarray(MODULE.error_heatmap(large_error, ground_truth))
        self.assertLess(float(small[..., 0].mean()), float(large[..., 0].mean()))
        self.assertEqual(int(large[..., 0].max()), 255)

    def test_case_tiles_render_metrics_and_errors(self):
        image = Image.new("RGB", (64, 36), (80, 90, 100))
        assets = {
            "target_gt": image,
            "unbounded": image,
            "unbounded_gt": image,
            "unbounded_error": image,
            "geocov": image,
            "geocov_gt": image,
            "geocov_error": image,
        }
        case = {
            "target_frame": 1000,
            "unbounded_selected_frame": 100,
            "geocov_selected_frame": 200,
            "unbounded_memory_age": 900,
            "geocov_memory_age": 800,
            "unbounded_overlap": 0.91,
            "geocov_overlap": 0.93,
            "unbounded_psnr": 8.0,
            "unbounded_ssim": 0.2,
            "geocov_psnr": 16.0,
            "geocov_ssim": 0.5,
        }
        tiles = MODULE.case_tiles(case, assets, include_errors=True)
        self.assertEqual(len(tiles), 7)
        self.assertTrue(
            all(
                tile.size == (MODULE.TILE_WIDTH, MODULE.TILE_HEIGHT)
                for tile in tiles
            )
        )


if __name__ == "__main__":
    unittest.main()
