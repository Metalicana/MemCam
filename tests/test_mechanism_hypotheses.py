import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIGNMENT = load_module("summarize_ri_alignment", "utils/summarize_ri_alignment.py")
POOL = load_module("analyze_pool_growth_scaling", "utils/analyze_pool_growth_scaling.py")
H2 = load_module("check_h2_per_trajectory", "utils/check_h2_per_trajectory.py")
REPLAY = load_module("evaluate_context_replays", "utils/evaluate_context_replays.py")


class TiedRankTest(unittest.TestCase):
    def test_average_ranks_are_used_for_ties(self):
        self.assertEqual(
            ALIGNMENT.rank_values([10, 10, 5, 1]),
            [1.5, 1.5, 3.0, 4.0],
        )

    def test_tied_monotone_values_have_unit_spearman(self):
        self.assertAlmostEqual(
            ALIGNMENT.spearman([1, 1, 2, 3], [10, 10, 20, 30]),
            1.0,
        )


class PoolGrowthTest(unittest.TestCase):
    def test_pool_trend_is_computed_per_trajectory(self):
        rows = []
        for row_id, gaps in (("1", [0.1, 0.2, 0.3, 0.4]), ("2", [0.4, 0.3, 0.2, 0.1])):
            for section_idx, gap in enumerate(gaps, start=1):
                for _ in range(2):
                    rows.append(
                        {
                            "row": row_id,
                            "scene": f"scene_{row_id}",
                            "dataset_start_frame": "0",
                            "duration_sec": "180",
                            "section_idx": str(section_idx),
                            "candidate_count": str(section_idx * 10),
                            "retrieval_gap": str(gap),
                        }
                    )

        points = POOL.section_points(rows)
        summaries = POOL.summarize_trajectories(points)

        self.assertEqual(len(points), 8)
        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(summaries[0]["spearman_pool_vs_retrieval_gap"], 1.0)
        self.assertAlmostEqual(summaries[1]["spearman_pool_vs_retrieval_gap"], -1.0)


class H2PairingTest(unittest.TestCase):
    def test_only_matched_sections_are_compared(self):
        rows = [
            self._row("ri", 1, 0.8),
            self._row("ri", 2, 0.2),
            self._row("slam", 2, 0.5),
            self._row("slam", 3, 0.1),
        ]

        paired = H2.paired_by_trajectory(rows, "ri", "slam", "retention_gap")
        values = next(iter(paired.values()))

        self.assertEqual(values["shared_sections"], 1)
        self.assertAlmostEqual(values["left"], 0.2)
        self.assertAlmostEqual(values["right"], 0.5)
        self.assertAlmostEqual(values["delta"], -0.3)

    def test_sign_test_reports_ties_separately(self):
        paired = {
            ("1",): {"delta": -0.1},
            ("2",): {"delta": 0.2},
            ("3",): {"delta": 0.0},
        }

        result = H2.sign_test(paired)

        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["pvalue"], 1.0)

    @staticmethod
    def _row(run_name, section_idx, retention_gap):
        return {
            "run_name": run_name,
            "row": "3",
            "scene": "scene",
            "dataset_start_frame": "100",
            "duration_sec": "180",
            "section_idx": str(section_idx),
            "retention_gap": str(retention_gap),
        }


class ReplayTraceValidationTest(unittest.TestCase):
    def test_first_retrieval_section_can_have_empty_history(self):
        control = {(1, 77): 4}
        swap = {(1, 77): 9}

        self.assertEqual(
            REPLAY.compare_replay_trace_maps(control, swap, section_idx=1),
            (True, True, 1),
        )

    def test_history_requires_exact_keys_and_choices(self):
        control = {(1, 77): 4, (2, 153): 10, (2, 154): 11}
        matched = {(1, 77): 4, (2, 153): 20, (2, 154): 11}
        missing_history = {(2, 153): 20, (2, 154): 11}

        self.assertEqual(
            REPLAY.compare_replay_trace_maps(control, matched, section_idx=2),
            (True, True, 1),
        )
        self.assertEqual(
            REPLAY.compare_replay_trace_maps(
                control, missing_history, section_idx=2
            ),
            (False, True, 1),
        )

    def test_target_keys_must_match(self):
        control = {(1, 77): 4, (2, 153): 10, (2, 154): 11}
        missing_target = {(1, 77): 4, (2, 153): 20}

        self.assertEqual(
            REPLAY.compare_replay_trace_maps(
                control, missing_target, section_idx=2
            ),
            (True, False, 0),
        )


if __name__ == "__main__":
    unittest.main()
