import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper.paired_retrieval_curves import RUNS, compare


def fixture_rows():
    rows = []
    for run, _ in RUNS:
        for trajectory in range(2):
            for section in range(1, 9):
                # Unequal query counts must not change section/trajectory weights.
                for slot in range(1 if trajectory == 0 else 3):
                    corruption = .2 + .1 * trajectory + .02 * section
                    if run != "baseline":
                        corruption -= .05
                    rows.append(dict(run_name=run, content_run="baseline",
                                     budget="" if run == "baseline" else "32", row=trajectory,
                                     scene=f"scene{trajectory}", dataset_start_frame=100, duration_sec=60,
                                     section_idx=section, target_frame=76 * section + slot,
                                     selected_memory_frame=0, candidate_count_mismatch=0,
                                     selected_view_mismatch=.1, selected_memory_corruption=corruption,
                                     selected_effective_mismatch=.5, full_oracle_effective_mismatch=.2))
    return rows


def write_input(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PairedRetrievalTests(unittest.TestCase):
    def test_equal_weights_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            write_input(path, fixture_rows())
            curves, metadata = compare(path, expected_videos=2, bins=4, bootstrap=100)
            self.assertEqual(metadata["queries_per_policy"], 32)
            means = [r["mean"] for r in curves if r["run"] == "baseline"
                     and r["metric"] == "selected_memory_corruption"]
            np.testing.assert_allclose(means, [.28, .32, .36, .40])
            summary = next(r for r in metadata["summaries"] if r["metric"] == "selected_memory_corruption")
            self.assertAlmostEqual(summary["unbounded"], .34)
            self.assertAlmostEqual(summary["keepsake"], .29)
            for key in ("paired_difference", "paired_ci_low", "paired_ci_high"):
                self.assertAlmostEqual(summary[key], -.05)
            self.assertEqual(len(curves), 2 * 4 * 4)
            self.assertIn("Not each policy's own", metadata["limitations"])
            self.assertIn("full_oracle_effective_mismatch", {r["metric"] for r in curves})

    def test_rejects_incomplete_or_incompatible_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            for change in ({"content_run": "slam_b32_covisibility"}, {"budget": "16"},
                           {"scene": "wrong"}, {"candidate_count_mismatch": 1},
                           {"selected_memory_corruption": float("nan")},
                           {"selected_memory_corruption": -1},
                           {"selected_memory_frame": 10000},
                           {"full_oracle_effective_mismatch": .7}):
                rows = fixture_rows()
                rows[-1].update(change)
                write_input(path, rows)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    compare(path, expected_videos=2, bins=4, bootstrap=100)
            for rows in (fixture_rows()[:-1], fixture_rows() + [fixture_rows()[0]]):
                write_input(path, rows)
                with self.assertRaises(ValueError):
                    compare(path, expected_videos=2, bins=4, bootstrap=100)
            write_input(path, fixture_rows())
            with self.assertRaises(ValueError):
                compare(path, expected_videos=15)


if __name__ == "__main__":
    unittest.main()
