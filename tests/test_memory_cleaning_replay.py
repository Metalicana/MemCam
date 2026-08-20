import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_module(
    "build_memory_cleaning_replay_plan",
    "utils/build_memory_cleaning_replay_plan.py",
)
RUNNER = load_module(
    "run_memory_cleaning_replay_case",
    "utils/run_memory_cleaning_replay_case.py",
)
EVAL = load_module(
    "evaluate_memory_cleaning_replays",
    "utils/evaluate_memory_cleaning_replays.py",
)


def query_row(row_idx, section_idx, corruption, selected_frame=4):
    return {
        "run_name": "baseline",
        "row": str(row_idx),
        "scene": f"Scene_{row_idx}",
        "dataset_start_frame": "100",
        "duration_sec": "180",
        "section_idx": str(section_idx),
        "selected_memory_frame": str(selected_frame),
        "selected_view_mismatch": "0.1",
        "selected_memory_corruption": str(corruption),
        "selected_effective_mismatch": str(corruption + 0.05),
        "candidate_count": "1000",
    }


class MemoryCleaningReplayTest(unittest.TestCase):
    def test_plan_selects_high_corruption_from_unique_rows(self):
        rows = [
            query_row(1, 20, 0.9),
            query_row(1, 21, 0.8),
            query_row(2, 20, 0.7),
            query_row(3, 20, 0.6),
        ]
        sections = PLAN.aggregate_sections(rows, "baseline", 180, 20, 30)
        cases = PLAN.select_cases(sections, count=2, unique_rows=True)

        self.assertEqual(
            [(case["row"], case["section_idx"]) for case in cases],
            [(1, 20), (2, 20)],
        )

    def test_clean_overrides_keep_selected_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_dir = Path(tmp_dir) / "frames"
            gt_dir.mkdir()
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(
                gt_dir / "0104.png"
            )
            item = {
                "_row": 0,
                "scene": "Scene",
                "start_frame": 100,
                "gt_frames_dir": str(gt_dir),
            }
            selections = {
                2: {
                    153: {"memory_frame": 4, "source_run": "baseline"},
                    154: {"memory_frame": 4, "source_run": "baseline"},
                }
            }

            overrides, unique_frames = RUNNER.build_clean_content_overrides(
                item,
                section_idx=2,
                selection_overrides=selections,
                width=16,
                height=12,
            )

        self.assertEqual(unique_frames, 1)
        self.assertEqual(overrides[2][153]["memory_frame"], 4)
        self.assertEqual(overrides[2][154]["source"], "ground_truth_memory")
        self.assertEqual(overrides[2][153]["image"].size, (16, 12))

    def test_trace_comparison_requires_same_selection_and_clean_content(self):
        control = {
            (1, 77): {"selected_memory_frame": 4},
            (2, 153): {"selected_memory_frame": 9},
        }
        clean = {
            (1, 77): {
                "selected_memory_frame": 4,
                "context_content_override": 0,
                "context_content_source": "generated_memory",
            },
            (2, 153): {
                "selected_memory_frame": 9,
                "context_content_override": 1,
                "context_content_source": "ground_truth_memory",
            },
        }

        self.assertEqual(
            EVAL.compare_trace_content(control, clean, section_idx=2),
            (True, 1, True),
        )

if __name__ == "__main__":
    unittest.main()
