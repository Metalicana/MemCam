import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from paper import plot_grouped_budget_metrics as plot


class GroupedBudgetMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = self.root / "scores.csv"
        self.coverage = self.root / "coverage.csv"
        runs = ["baseline"] + [t.format(b) for _, t, _ in plot.POLICIES for b in plot.BUDGETS]
        self.rows = [dict(run=run, evaluator=e, metric=m, value=700 if m == "FVD" else 0.6,
                          n=15, source="synthetic") for run in runs for e, m, _, _ in plot.METRICS]
        self.states = [dict(run=run, evaluator=e, status="complete") for run in runs for e in ("quality", "vbench")]
        self.write_inputs()

    def write_inputs(self):
        for path, rows in ((self.scores, self.rows), (self.coverage, self.states)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_geometry_and_style(self):
        grid = plot.load_grid(self.scores, self.coverage)
        self.assertEqual(len(grid), 63)
        fig = plot.make_figure(grid)
        self.addCleanup(plt.close, fig)
        for ax in fig.axes:
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertEqual(len(ax.patches), 20)
            self.assertEqual([p.get_alpha() for p in ax.patches[:4]], list(plot.OPACITIES))
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()], [p[0] for p in plot.POLICIES])
            self.assertEqual(len(ax.lines), 1)
        self.assertEqual(fig.axes[2].patches[0].get_height(), 60)
        self.assertEqual(fig.axes[1].patches[0].get_height(), 700)
        fig.canvas.draw()
        bounds = fig.bbox
        for text in fig.texts + [a.title for a in fig.axes]:
            box = text.get_window_extent(fig.canvas.get_renderer())
            self.assertGreaterEqual(box.x0, bounds.x0)
            self.assertLessEqual(box.x1, bounds.x1)

    def test_incomplete_or_duplicate_scores_rejected(self):
        for mutation in ("missing", "duplicate", "wrong_n", "nonfinite"):
            with self.subTest(mutation=mutation):
                original = [dict(r) for r in self.rows]
                if mutation == "missing":
                    self.rows.pop()
                elif mutation == "duplicate":
                    self.rows.append(dict(self.rows[0]))
                elif mutation == "wrong_n":
                    self.rows[0]["n"] = 14
                else:
                    self.rows[0]["value"] = "nan"
                self.write_inputs()
                with self.assertRaises(ValueError):
                    plot.load_grid(self.scores, self.coverage)
                self.rows = original

    def test_failed_task_rejected(self):
        self.states[0]["status"] = "failed"
        self.write_inputs()
        with self.assertRaises(ValueError):
            plot.load_grid(self.scores, self.coverage)

    def test_exports(self):
        plot.export(self.scores, self.coverage, self.root / "output")
        for extension in ("pdf", "png", "csv", "json"):
            self.assertGreater((self.root / "output" / f"02_grouped_budget_bars.{extension}").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
