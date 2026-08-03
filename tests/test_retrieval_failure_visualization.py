import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_retrieval_failures.py"
SPEC = importlib.util.spec_from_file_location("retrieval_visualization", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pretty_run_names():
    assert MODULE.pretty_run_name("baseline") == "Unbounded"
    assert MODULE.pretty_run_name("fifo_b32") == "FIFO (B=32)"
    assert MODULE.pretty_run_name("ri_b64_dino_rgb") == "RI (B=64)"
    assert MODULE.pretty_run_name("slam_b32_covisibility") == "SLAM-style (B=32)"


def test_select_diverse_cases_limits_each_row():
    cases = [
        {"row": 1, "target_frame": 100, "score": 10, "key": (1, 1, 100)},
        {"row": 1, "target_frame": 110, "score": 9, "key": (1, 1, 110)},
        {"row": 2, "target_frame": 200, "score": 8, "key": (2, 2, 200)},
    ]

    selected = MODULE.select_diverse_cases(cases, max_examples=2, per_row=1)

    assert len(selected) == 2
    assert {case["row"] for case in selected} == {1, 2}


def test_status_colors_distinguish_hit_and_miss():
    assert MODULE.status_color(True) != MODULE.status_color(False)
    assert MODULE.status_color(None) not in {
        MODULE.status_color(True),
        MODULE.status_color(False),
    }
