import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "paper/plot_retrieval_deterioration.py"
SPEC = importlib.util.spec_from_file_location("deterioration", SCRIPT)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def fixture():
    rows = []
    for trajectory in range(3):
        for section in range(8):
            for slot in range(2):
                rows.append(dict(run_name="baseline", row=trajectory, scene=f"scene{trajectory}",
                                 dataset_start_frame=100, duration_sec=180, section_idx=section,
                                 target_frame=section * 76 + slot * 19 + 1, candidate_count_mismatch=0,
                                 selected_view_mismatch=.3 - section * .01,
                                 selected_memory_corruption=.1 + section * .02 + trajectory * .01,
                                 selected_effective_mismatch=.4 + section * .03,
                                 full_oracle_effective_mismatch=.2))
    return rows


class DeteriorationTests(unittest.TestCase):
    def test_aggregation_and_direction_with_trajectory_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            rows = fixture()
            # More queries in one section must not give that section more weight.
            extra = dict(rows[0], target_frame=38)
            plotter.write_csv(path, rows + [extra])
            ids, sections, values, targets, count = plotter.load_sections(path, "baseline", 180, 3)
            self.assertEqual((len(ids), len(sections), count), (3, 8, 49))
            summary = plotter.summarize(values, targets, 30, 4, 1000, 0)
            np.testing.assert_allclose(summary["delta_mean"], [-.06, .12, .18, 0], atol=1e-12)
            self.assertEqual(summary["quarter_sections"], 2)
            self.assertEqual(summary["curve_ci"].shape, (2, 4, 4))

    def test_refuses_duplicate_nonfinite_incomplete_and_bad_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            invalid = [fixture() + [fixture()[0]], fixture()[:-2]]
            for field, value in (("selected_memory_corruption", "nan"),
                                 ("full_oracle_effective_mismatch", 10), ("candidate_count_mismatch", 1)):
                rows = fixture()
                rows[0][field] = value
                invalid.append(rows)
            for rows in invalid:
                plotter.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    plotter.load_sections(path, "baseline", 180, 3)

    def test_filters_run_duration_and_checks_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            rows = fixture()
            rows += [dict(row, run_name="fifo_b32") for row in fixture()]
            rows += [dict(row, duration_sec=60) for row in fixture()]
            plotter.write_csv(path, rows)
            self.assertEqual(plotter.load_sections(path, "baseline", 180, 3)[-1], 48)
            with self.assertRaises(ValueError):
                plotter.load_sections(path, "baseline", 180, 15)

    def test_cli_exports_figures_values_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "queries.csv", root / "figures"
            plotter.write_csv(path, fixture())
            completed = subprocess.run([sys.executable, str(SCRIPT), "--input", str(path), "--output", str(output),
                                        "--expected-videos", "3", "--bootstrap", "1000"],
                                       capture_output=True, text=True, timeout=60, check=True)
            self.assertIn("3 trajectories; 48 queries; 24 sections", completed.stdout)
            self.assertTrue((output / "retrieval_deterioration.pdf").is_file())
            self.assertTrue((output / "retrieval_deterioration.png").is_file())
            for panel in ("changes", "evidence"):
                for extension in ("png", "pdf"):
                    self.assertTrue((output / f"retrieval_deterioration_{panel}.{extension}").is_file())
            source = json.loads((output / "provenance.json").read_text())
            self.assertEqual(source["early_sections"], [0, 1])
            self.assertEqual(source["late_sections"], [6, 7])
            self.assertEqual(len(source["source_sha256"]), 64)
            with (output / "curves.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 32)

            _, _, values, targets, _ = plotter.load_sections(path, "baseline", 180, 3)
            expected = plotter.summarize(values, targets, 30, 8, 1000, 0)
            restored, n = plotter.load_saved_result(output)
            self.assertEqual(n, 3)
            for key in restored:
                np.testing.assert_array_equal(restored[key], expected[key])

            panels = root / "panels"
            subprocess.run([sys.executable, str(SCRIPT), "--from-results", str(output),
                            "--individual-only", "--output", str(panels)],
                           capture_output=True, text=True, timeout=60, check=True)
            self.assertFalse((panels / "retrieval_deterioration.png").exists())
            for panel in ("changes", "evidence"):
                self.assertTrue((panels / f"retrieval_deterioration_{panel}.png").is_file())
                self.assertTrue((panels / f"retrieval_deterioration_{panel}.pdf").is_file())
            render = json.loads((panels / "render_provenance.json").read_text())
            self.assertEqual(render["original_provenance"], source)
            self.assertEqual(len(render["inputs"]), 4)

            with (output / "curves.csv").open() as handle:
                curves = list(csv.DictReader(handle))
            for bad_curves in (curves[:-1], curves + [curves[0]],
                               [dict(curves[0], mean="nan")] + curves[1:]):
                plotter.write_csv(output / "curves.csv", bad_curves)
                with self.assertRaises(ValueError):
                    plotter.load_saved_result(output)


if __name__ == "__main__":
    unittest.main()
