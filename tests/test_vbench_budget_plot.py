import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("budget_plot", Path(__file__).resolve().parents[1] / "paper/make_vbench_budget_sweep.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class BudgetPlotTests(unittest.TestCase):
    def write_scores(self, root, n=15, duplicate=False):
        path = root / "scores.csv"
        rows = [{"run": "slam_b32_covisibility", "evaluator": "vbench",
                 "metric": dim, "value": .5, "n": n} for dim in plot.DIMENSIONS]
        if duplicate:
            rows.append(rows[0])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_partial_budget_sweep_renders_without_inventing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = plot.read_scores(self.write_scores(root))
            self.assertEqual(len(scores), 6)
            self.assertEqual(scores["vbench", "slam_b32_covisibility", "subject_consistency"], 50)
            paths = plot.render(scores, root / "figures")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))

    def test_rejects_incomplete_cohort_and_duplicate_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kwargs in ({"n": 7}, {"duplicate": True}):
                with self.assertRaises(ValueError):
                    plot.read_scores(self.write_scores(root, **kwargs))
