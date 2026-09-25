from pathlib import Path
import tempfile
import unittest

from paper import export_memory_diagnostics_tikz as exporter
from tests.test_paired_retrieval_curves import fixture_rows, write_input


class MemoryTikzTests(unittest.TestCase):
    def test_original_coordinates_and_all_budget_labels(self):
        points = exporter.original_points(exporter.ROOT / "paper/make_figures.py")
        self.assertEqual(len(points), 9)
        self.assertEqual({p["policy"] for p in points}, {"FIFO", "Unbounded", "KEEPSAKE (Ours)"})
        expected = [(16, .0669, .1037), (32, .0426, .1470),
                    (64, .0366, .1596), (128, .0298, .1783)]
        self.assertEqual([(p["budget"], p["retention_gap"], p["selection_gap"])
                          for p in points if p["policy"] == "KEEPSAKE (Ours)"], expected)
        self.assertEqual(points[-1]["selection_gap"], .2188)
        tex = exporter.landscape_tex(points)
        for b in (16, 32, 64, 128):
            self.assertEqual(tex.count("{B" + str(b) + "}"), 2)
        self.assertNotIn("GeoCov", tex)
        self.assertIn("0.1668,0.0646", tex)
        self.assertIn("KEEPSAKE\\\\(Ours)", tex)

    def test_paired_statistics_and_section_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            rows = fixture_rows()
            # Deliberately make KEEPSAKE view mismatch worse; preserve its sign.
            for row in rows:
                if row["run_name"] != "baseline":
                    row["selected_view_mismatch"] += .1
            write_input(path, rows)
            statistics, provenance = exporter.statistics(path, videos=2, bootstrap=100)
            means = {(r["policy"], r["metric"]): r["mean"] for r in statistics
                     if r["statistic"] == "rollout_mean"}
            self.assertAlmostEqual(means["Unbounded", "selected_memory_corruption"], .34)
            self.assertAlmostEqual(means["KEEPSAKE (Ours)", "selected_memory_corruption"], .29)
            contrasts = {(r["statistic"], r["metric"]): r for r in provenance["paired_contrasts"]}
            for metric, difference in (("selected_view_mismatch", .1), ("selected_memory_corruption", -.05)):
                row = contrasts["rollout_mean", metric]
                for key in ("keepsake_minus_unbounded", "ci_low", "ci_high"):
                    self.assertAlmostEqual(row[key], difference)
                self.assertAlmostEqual(contrasts["late_minus_early", metric]["keepsake_minus_unbounded"], 0.)
            self.assertEqual(provenance["early_sections"], [1, 2])
            self.assertEqual(provenance["late_sections"], [7, 8])
            tex = exporter.comparison_tex(statistics)
            self.assertIn("(0.340000,0.12)", tex)
            self.assertIn("(0.290000,-0.12)", tex)
            self.assertIn("xmin=0.000000", tex)
            self.assertIn("Late $-$ early", exporter.comparison_tex(statistics, changes=True))

    def test_rejects_mismatched_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            write_input(path, fixture_rows()[:-1])
            with self.assertRaises(ValueError):
                exporter.statistics(path, videos=2, bootstrap=100)

    def test_complete_figure_uses_one_geometry_and_distinguishes_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            write_input(path, fixture_rows())
            rows, _ = exporter.statistics(path, videos=2, bootstrap=100)
            points = exporter.original_points(exporter.ROOT / "paper/make_figures.py")
            absolute = exporter.three_panel_tex(points, rows)
            change = exporter.three_panel_tex(points, rows, changes=True)
            self.assertEqual(absolute.count(r"\begin{figure}[t]"), 1)
            self.assertEqual(absolute.count(r"\begin{subfigure}[t]{0.32\linewidth}"), 3)
            self.assertEqual(absolute.count(r"rectangle (\linewidth,52mm)"), 3)
            self.assertEqual(absolute.count("keepsake panel axes,"), 3)
            for coordinate in ("(0.5980,734.2)", "(0.6514,677.3)", "(0.5876,476.6)"):
                self.assertIn(coordinate, absolute)
            self.assertNotIn(r"\input{", absolute)
            self.assertNotIn("height=36mm", absolute)
            self.assertIn("Mean view mismatch", absolute)
            self.assertNotIn("Late-minus-early changes", absolute)
            self.assertIn("Late-minus-early changes", change)
            self.assertIn("Memory-quality change.", change)
            self.assertIn("Late $-$ early distance", change)


if __name__ == "__main__":
    unittest.main()
