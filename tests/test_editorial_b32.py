import csv
import tempfile
import unittest
from pathlib import Path

from paper import plot_editorial_b32 as plot


class EditorialB32Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = self.root / "scores.csv"
        self.coverage = self.root / "coverage.csv"
        rows = []
        states = []
        for index, (_, run, _) in enumerate(plot.METHODS):
            for evaluator, metric, _, _ in plot.METRICS:
                value = 700 if metric == "FVD" else 0.5
                if evaluator == "vbench" and index:
                    value += (index - 3) * 0.005
                rows.append(dict(run=run, evaluator=evaluator, metric=metric, value=value, n=15, source="synthetic"))
            states.extend(dict(run=run, evaluator=e, status="complete") for e in ("quality", "vbench"))
        for path, content in ((self.scores, rows), (self.coverage, states)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(content[0]))
                writer.writeheader()
                writer.writerows(content)

    def test_relative_percent_not_percentage_points(self):
        self.assertAlmostEqual(plot.relative_difference(0.51, 0.5), 2)
        self.assertAlmostEqual(plot.relative_difference(0.49, 0.5), -2)
        self.assertEqual(plot.relative_difference(0.5, 0.5), 0)
        with self.assertRaises(ValueError):
            plot.relative_difference(0.5, 0)

    def test_layout_and_values(self):
        grid = plot.load_grid(self.scores, self.coverage, runs=[m[1] for m in plot.METHODS], metrics=plot.METRICS)
        self.assertEqual(len(grid), 30)
        fig = plot.make_figure(grid)
        self.addCleanup(plot.plt.close, fig)
        for ax in fig.axes[:2]:
            self.assertEqual(ax.get_xlim()[0], 0)
            self.assertEqual(len(ax.patches), 6)
            self.assertTrue(ax.yaxis_inverted())
            self.assertEqual([t.get_text() for t in ax.get_yticklabels()], [m[0] for m in plot.METHODS])
            self.assertEqual(ax.texts[-1].get_fontweight(), "bold")
        self.assertEqual(len(fig.axes[2].patches), 15)
        self.assertAlmostEqual(fig.axes[2].patches[0].get_width(), -2)
        self.assertAlmostEqual(fig.axes[2].patches[4].get_width(), 2)
        self.assertEqual(list(fig.axes[2].lines[0].get_xdata()), [0, 0])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = list(fig.texts)
        for ax in fig.axes:
            texts.extend([*ax.texts, *ax.get_yticklabels(), ax.xaxis.label, ax._left_title])
        for text in texts:
            box = text.get_window_extent(renderer)
            self.assertGreaterEqual(box.x0, fig.bbox.x0)
            self.assertLessEqual(box.x1, fig.bbox.x1)
            self.assertGreaterEqual(box.y0, fig.bbox.y0)
            self.assertLessEqual(box.y1, fig.bbox.y1)

    def test_export(self):
        plot.export(self.scores, self.coverage, self.root / "output")
        for extension in ("pdf", "png", "csv", "json"):
            self.assertGreater((self.root / "output" / f"01_editorial_b32.{extension}").stat().st_size, 0)
        with (self.root / "output/01_editorial_b32.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 30)
        example = next(r for r in rows if r["policy"] == "FIFO" and r["metric"] == "subject_consistency")
        self.assertAlmostEqual(float(example["relative_difference_percent"]), -2)


if __name__ == "__main__":
    unittest.main()
