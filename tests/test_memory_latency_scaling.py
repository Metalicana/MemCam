import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "analyze_memory_latency_scaling.py"
SPEC = importlib.util.spec_from_file_location("analyze_memory_latency_scaling", MODULE_PATH)
SCALING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALING)


class MemoryLatencyScalingTest(unittest.TestCase):
    def write_profile(self, directory, name, policy, budget, memory_slope, quadratic_latency):
        path = directory / f"{name}.jsonl"
        records = []
        for section_idx, duration in enumerate((10, 40, 60)):
            records.append(
                {
                    "event": "section_profile",
                    "run_name": name,
                    "row": 0,
                    "scene": "test",
                    "memory_policy": policy,
                    "memory_budget": budget,
                    "memory_bank_device": "cpu",
                    "section_idx": section_idx,
                    "generated_seconds": duration,
                    "section_latency_s": 100 + duration,
                    "cumulative_rollout_latency_s": 2 * duration + quadratic_latency * duration ** 2,
                    "bank_frame_gb": memory_slope * duration,
                    "bank_feature_gb": 0.0,
                    "stored_memory_size": duration * 30 if budget is None else budget,
                    "candidate_count": duration * 30 if budget is None else budget,
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return path

    def test_aggregate_and_projection_recover_scaling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            profile = self.write_profile(
                directory,
                "unbounded_cpu",
                "unbounded",
                None,
                memory_slope=0.04,
                quadratic_latency=0.01,
            )
            points = SCALING.aggregate_profiles([profile], [10, 40, 60], {})
            projections, models = SCALING.build_projections(points, [600, 3600])

        self.assertEqual(len(points), 3)
        self.assertAlmostEqual(points[-1]["memory_bank_gb_median"], 2.4)
        self.assertAlmostEqual(projections[0]["memory_bank_gb"], 24.0, places=5)
        self.assertAlmostEqual(projections[1]["average_latency_s_per_s"], 38.0, places=5)
        self.assertAlmostEqual(models[0]["memory_r2"], 1.0)
        self.assertAlmostEqual(models[0]["latency_r2"], 1.0)

    def test_bounded_memory_projection_is_constant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            profile = self.write_profile(
                directory,
                "ri_b32_cpu",
                "rarity_irreplaceability",
                32,
                memory_slope=0.0,
                quadratic_latency=0.0,
            )
            records = []
            for record in SCALING.read_profile(profile):
                record["memory_bank_gb"] = 0.05
                records.append(record)
            points = []
            for duration, record in zip((10, 40, 60), records):
                point = {
                    "run_name": "ri_b32_cpu",
                    "label": "RI-B32-CPU",
                    "memory_policy": "rarity_irreplaceability",
                    "memory_budget": 32,
                    "memory_bank_device": "cpu",
                    "duration_sec": duration,
                    "samples": 1,
                    "memory_bank_gb_median": record["memory_bank_gb"],
                    "cumulative_rollout_latency_s_median": record["cumulative_rollout_latency_s"],
                }
                points.append(point)
            projections, _ = SCALING.build_projections(points, [600])

        self.assertAlmostEqual(projections[0]["memory_bank_gb"], 0.05)

    def test_plot_and_report_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            profile = self.write_profile(
                directory,
                "unbounded_cpu",
                "unbounded",
                None,
                memory_slope=0.04,
                quadratic_latency=0.01,
            )
            points = SCALING.aggregate_profiles([profile], [10, 40, 60], {})
            projections, models = SCALING.build_projections(points, [600, 3600])
            SCALING.plot_scaling(points, projections, directory, "memory")
            SCALING.plot_scaling(points, projections, directory, "latency")
            SCALING.write_report(points, projections, models, directory / "report.md")
            expected = (
                "memory_bank_scaling.png",
                "memory_bank_scaling.pdf",
                "rollout_latency_scaling.png",
                "rollout_latency_scaling.pdf",
                "report.md",
            )
            self.assertTrue(all((directory / name).stat().st_size > 0 for name in expected))


if __name__ == "__main__":
    unittest.main()
