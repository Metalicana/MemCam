import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUALITY = load_module("plot_memcam_slide_quality", "utils/plot_memcam_slide_quality.py")
CUT3R_PLOT = load_module("plot_memcam_cut3r_budget", "utils/plot_memcam_cut3r_budget.py")
CUT3R_RUN = load_module("run_cut3r_context_memory", "utils/run_cut3r_context_memory.py")


class SlideAnalysisTest(unittest.TestCase):
    def test_prefix_num_frames(self):
        self.assertEqual(CUT3R_RUN.prefix_num_frames(5397, 180, 60), 1799)
        self.assertEqual(CUT3R_RUN.prefix_num_frames(5397, 180, None), 5397)
        with self.assertRaises(ValueError):
            CUT3R_RUN.prefix_num_frames(5397, 180, 181)

    def test_quality_plotter_reads_summaries_and_writes_both_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name, offset in (("baseline", 0.0), ("ri_b32_dino_rgb", -0.1)):
                run_dir = root / "metrics" / run_name
                run_dir.mkdir(parents=True)
                summary = {
                    "by_duration": {
                        str(duration): {
                            "lpips_alex": 0.5 + offset + duration / 1000,
                            "fvd": 700 + 2 * duration + 100 * offset,
                        }
                        for duration in (10, 20, 30, 60)
                    }
                }
                (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            rows = QUALITY.load_rows(
                [root / "metrics"],
                ["baseline", "ri_b32_dino_rgb"],
                [10, 20, 30, 60],
            )
            output_dir = root / "plots"
            output_dir.mkdir()
            for metric in QUALITY.METRICS:
                QUALITY.plot_metric(
                    rows,
                    metric,
                    [10, 20, 30, 60],
                    ["baseline", "ri_b32_dino_rgb"],
                    output_dir,
                )

            self.assertEqual(len(rows), 16)
            for spec in QUALITY.METRICS.values():
                self.assertTrue((output_dir / f"{spec['stem']}.png").is_file())
                self.assertTrue((output_dir / f"{spec['stem']}.pdf").is_file())

    def test_cut3r_budget_plot_ignores_kcenter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "cut3r_camera_summary.csv"
            fields = ["run_name", "videos", *CUT3R_PLOT.METRICS]
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for run_name, value in (
                    ("baseline", 2.0),
                    ("fifo_b32", 2.2),
                    ("slam_b32_covisibility", 1.8),
                    ("ri_b32_dino_rgb", 1.7),
                    ("kcenter_b32_dino_pose", 1.6),
                ):
                    writer.writerow(
                        {
                            "run_name": run_name,
                            "videos": 15,
                            **{metric: value for metric in CUT3R_PLOT.METRICS},
                        }
                    )

            rows = CUT3R_PLOT.load_rows(summary_path, list(CUT3R_PLOT.METRICS))
            self.assertNotIn("kcenter_b32_dino_pose", {row["run_name"] for row in rows})
            self.assertEqual({row["policy"] for row in rows}, {"Unbounded", "FIFO", "SLAM", "Ours"})


if __name__ == "__main__":
    unittest.main()
