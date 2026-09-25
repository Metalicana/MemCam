import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gap_comparison", ROOT / "paper/plot_retention_selection_comparison.py")
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)
LEGACY = ROOT / "paper/configs/retention_selection_180s_reported.json"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixtures(directory):
    summaries, queries = [], []
    for run, (policy, budget) in plotter.RUNS.items():
        retention = 0 if policy == "Unbounded" else .02
        values = dict(retention_gap=retention, retrieval_gap=.1,
                      total_oracle_gap=retention + .1)
        summaries.append(dict(run_name=run, budget=budget, trajectories=15,
                              queries=15, min_section=1, **values))
        for trajectory in range(15):
            queries.append(dict(run_name=run, row=trajectory, scene=f"scene{trajectory}",
                                dataset_start_frame=100, duration_sec=60,
                                content_run="baseline", section_idx=1, target_frame=77,
                                candidate_count_mismatch=0, **values))
    summary_path, query_path = directory / "summary.csv", directory / "queries.csv"
    write_csv(summary_path, summaries)
    write_csv(query_path, queries)
    return summary_path, query_path, summaries, queries


class ComparisonTests(unittest.TestCase):
    def test_requested_policies_horizons_and_legacy_points(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, queries, _, _ = fixtures(Path(directory))
            points = plotter.load_points(summary, queries, LEGACY)
        self.assertEqual(len(points), 17)
        self.assertEqual(set(p["policy"] for p in points), set(plotter.POLICIES))
        self.assertNotIn("RI", {p["policy"] for p in points})
        for point in points:
            self.assertEqual((point["duration_sec"], point["trajectories"]),
                             (180, 13) if point["policy"] == "KEEPSAKE" else (60, 15))
            self.assertAlmostEqual(point["retention_gap"] + point["selection_gap"],
                                   point["total_gap"])

    def test_legacy_matches_original_plot_without_editing_it(self):
        tree = ast.parse((ROOT / "paper/make_figures.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "plot_retention_selection_tradeoff")
        families = next(n.value for n in function.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "families" for t in n.targets))
        geocov = next(v for k, v in zip(families.keys, families.values) if k.value == "GeoCov")
        original = next(ast.literal_eval(v) for k, v in zip(geocov.keys, geocov.values)
                        if k.value == "points")
        reported = json.loads(LEGACY.read_text())["points"]
        self.assertEqual(original, [(p["budget"], p["retention_gap"], p["selection_gap"])
                                    for p in reported])

    def test_rejects_wrong_horizon_duplicate_query_bad_mean_and_missing_run(self):
        for problem in ("horizon", "duplicate", "mean", "missing", "candidate", "total"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                summary, queries, summaries, records = fixtures(Path(directory))
                if problem == "horizon":
                    records[0]["duration_sec"] = 180
                elif problem == "duplicate":
                    records.append(records[0])
                elif problem == "mean":
                    summaries[0]["retrieval_gap"] = summaries[0]["total_oracle_gap"] = .2
                elif problem == "missing":
                    summaries.pop()
                elif problem == "candidate":
                    records[0]["candidate_count_mismatch"] = 1
                else:
                    records[0]["total_oracle_gap"] = .4
                write_csv(summary, summaries)
                write_csv(queries, records)
                with self.assertRaises(ValueError):
                    plotter.load_points(summary, queries, LEGACY)

    def test_export_and_visible_duration_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, queries, _, _ = fixtures(root)
            points = plotter.load_points(summary, queries, LEGACY)
            fig = plotter.make_figure(points)
            labels = [t.get_text() for t in fig.legends[0].get_texts()]
            self.assertEqual(labels[-1], "KEEPSAKE\n180 s, n=13")
            self.assertTrue(all("60 s, n=15" in label for label in labels[:-1]))
            self.assertEqual(len(fig.axes), 1)
            ax = fig.axes[0]
            marks = {line.get_gid(): line for line in ax.lines
                     if line.get_gid().startswith("point:")}
            self.assertEqual(len(marks), 17)
            self.assertEqual(len(set(plotter.MARKERS.values())), 4)
            for point in points:
                line = marks[f"point:{point['run']}"]
                expected = "*" if point["budget"] is None else plotter.MARKERS[point["budget"]]
                self.assertEqual(line.get_marker(), expected)
                self.assertEqual(line.get_color(), plotter.COLORS[point["policy"]])
                self.assertEqual(list(line.get_xdata()), [point["retention_gap"]])
                self.assertEqual(list(line.get_ydata()), [point["selection_gap"]])
            connections = {line.get_gid(): line for line in ax.lines
                           if line.get_gid().startswith("connection:")}
            self.assertEqual(len(connections), 4)
            self.assertEqual(connections["connection:KEEPSAKE"].get_linestyle(), "--")
            self.assertEqual([t.get_text() for t in fig.legends[1].get_texts()],
                             ["B16", "B32", "B64", "B128"])
            self.assertEqual([line.get_marker() for line in fig.legends[1].get_lines()],
                             ["o", "s", "^", "D"])
            plotter.plt.close(fig)
            output = root / "figure"
            plotter.export(summary, queries, LEGACY, output)
            for extension in (".png", ".pdf", ".csv", ".caption.txt", ".provenance.json"):
                self.assertTrue(output.with_suffix(extension).is_file())
            provenance = json.loads(output.with_suffix(".provenance.json").read_text())
            self.assertEqual(provenance["comparison"], "mixed_horizon_descriptive")
            self.assertEqual(len(provenance["points"]), 17)
            self.assertEqual(provenance["presentation"]["layout"], "single_panel")
            self.assertEqual(provenance["presentation"]["budget_markers"],
                             {str(k): v for k, v in plotter.MARKERS.items()})
            caption = output.with_suffix(".caption.txt").read_text()
            self.assertNotIn("right panel", caption)
            self.assertIn("square B32", caption)


if __name__ == "__main__":
    unittest.main()
