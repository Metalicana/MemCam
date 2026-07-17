import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "plot_latency_vram_pareto.py"
SPEC = importlib.util.spec_from_file_location("plot_latency_vram_pareto", MODULE_PATH)
PLOTTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTER)


class LatencyVramParetoTest(unittest.TestCase):
    def write_profile(self, directory, name, policy, budget, device, latency_scale, vram_scale):
        path = directory / f"{name}.jsonl"
        records = []
        for section_idx, duration in enumerate((10, 20, 40)):
            records.append(
                {
                    "event": "section_profile",
                    "run_name": name,
                    "row": 0,
                    "scene": "test",
                    "memory_policy": policy,
                    "memory_budget": budget,
                    "memory_bank_device": device,
                    "section_idx": section_idx,
                    "generated_seconds": duration,
                    "latency_per_generated_second": latency_scale * duration,
                    "peak_cuda_allocated_gb": vram_scale * duration,
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_aggregate_profiles_uses_requested_prefixes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = self.write_profile(
                directory,
                "ri_gpu",
                "rarity_irreplaceability",
                32,
                "cuda",
                0.1,
                0.2,
            )
            points = PLOTTER.aggregate_profiles([path], [10, 40], {})

        self.assertEqual([point["duration_sec"] for point in points], [10, 40])
        self.assertEqual(points[0]["label"], "RI-B32-GPU")
        self.assertAlmostEqual(points[1]["latency_median"], 4.0)
        self.assertAlmostEqual(points[1]["peak_vram_median"], 8.0)

    def test_pareto_frontier_removes_dominated_points(self):
        points = [
            {"label": "fast-large", "latency_median": 1.0, "peak_vram_median": 8.0},
            {"label": "dominated", "latency_median": 2.0, "peak_vram_median": 9.0},
            {"label": "balanced", "latency_median": 3.0, "peak_vram_median": 5.0},
        ]
        frontier = PLOTTER.pareto_frontier(points)
        self.assertEqual([point["label"] for point in frontier], ["fast-large", "balanced"])

    def test_plot_writes_png_pdf_and_csv(self):
        points = [
            {
                "run_name": "unbounded_cpu",
                "label": "Unbounded-CPU",
                "memory_policy": "unbounded",
                "memory_budget": None,
                "memory_bank_device": "cpu",
                "duration_sec": 180,
                "samples": 1,
                "latency_median": 3.0,
                "latency_q25": 3.0,
                "latency_q75": 3.0,
                "peak_vram_median": 20.0,
                "peak_vram_q25": 20.0,
                "peak_vram_q75": 20.0,
            },
            {
                "run_name": "ri_b32_gpu",
                "label": "RI-B32-GPU",
                "memory_policy": "rarity_irreplaceability",
                "memory_budget": 32,
                "memory_bank_device": "cuda",
                "duration_sec": 180,
                "samples": 1,
                "latency_median": 2.0,
                "latency_q25": 2.0,
                "latency_q75": 2.0,
                "peak_vram_median": 18.0,
                "peak_vram_q25": 18.0,
                "peak_vram_q75": 18.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            PLOTTER.write_points_csv(points, output_dir / "points.csv")
            PLOTTER.plot_points(points, output_dir, "Test")
            outputs = [
                output_dir / "points.csv",
                output_dir / "latency_vs_peak_vram_pareto.png",
                output_dir / "latency_vs_peak_vram_pareto.pdf",
            ]
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
