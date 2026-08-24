import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def make_line_c2ws(positions):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    c2ws[:, 0, 3] = np.asarray(positions, dtype=np.float64)
    return c2ws


class CoverageHysteresisTest(unittest.TestCase):
    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="coverage_hysteresis")

    def test_sequential_admission_rejects_existing_and_new_chunk_duplicates(self):
        # Frame 1 repeats incumbent 0. Frame 2 adds a distant view. Frame 3
        # repeats frame 2 and must see it as soon as frame 2 is admitted.
        c2ws = make_line_c2ws([0.0, 0.1, 20.0, 20.1])
        admitted, details = MEMORY_POLICIES.select_coverage_hysteresis_admissions(
            existing_frame_indices=[0],
            candidate_frame_indices=[1, 2, 3],
            c2ws=c2ws,
            view_similarity_threshold=0.90,
            return_details=True,
        )
        self.assertEqual(admitted, [2])
        self.assertEqual(details[1]["hysteresis_reason"], "covered_by_incumbent")
        self.assertEqual(details[2]["hysteresis_reason"], "novel_view")
        self.assertEqual(details[3]["hysteresis_nearest_reference_frame"], 2)
        self.assertEqual(details[3]["hysteresis_reference_count"], 2)

    def test_empty_bank_admits_first_candidate_as_anchor(self):
        c2ws = make_line_c2ws([0.0, 0.1])
        admitted = MEMORY_POLICIES.select_coverage_hysteresis_admissions(
            existing_frame_indices=[],
            candidate_frame_indices=[0, 1],
            c2ws=c2ws,
            view_similarity_threshold=0.90,
        )
        self.assertEqual(admitted, [0])

    def test_threshold_is_inclusive_for_rejection(self):
        c2ws = make_line_c2ws([0.0, 10.0])
        # With radius 50 and identical orientation, distance 10 gives exactly
        # 0.90 camera similarity. Equality means the view is already covered.
        admitted = MEMORY_POLICIES.select_coverage_hysteresis_admissions(
            existing_frame_indices=[0],
            candidate_frame_indices=[1],
            c2ws=c2ws,
            view_similarity_threshold=0.90,
        )
        self.assertEqual(admitted, [])

    def test_equal_utility_preserves_older_incumbents(self):
        memory = MEMORY_POLICIES.FrameMemoryBuffer(
            policy="coverage_hysteresis", budget=2
        )
        evicted = memory.update(
            [0, 1, 2], eviction_scores={0: 1.0, 1: 1.0, 2: 1.0}
        )
        self.assertEqual(evicted, [2])
        self.assertEqual(memory.candidates(), [0, 1])

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.select_coverage_hysteresis_admissions(
                existing_frame_indices=[],
                candidate_frame_indices=[0],
                c2ws=make_line_c2ws([0.0]),
                view_similarity_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
