import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "utils" / "run_context_replay_case.py"
SPEC = importlib.util.spec_from_file_location("context_replay_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def write_trace(path, source_frame_offset=0):
    rows = []
    for section_idx in [1, 2]:
        for slot in range(76):
            rows.append(
                {
                    "event": "context_access",
                    "selected": True,
                    "scene": "Scene_0",
                    "dataset_start_frame": 100,
                    "duration_sec": 180,
                    "section_idx": section_idx,
                    "context_slot": slot,
                    "target_frame": section_idx * 76 + 1 + slot,
                    "selected_memory_frame": source_frame_offset + slot,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_trace_overrides_requires_and_loads_full_section(tmp_path):
    trace = tmp_path / "trace.jsonl"
    write_trace(trace)
    item = {
        "scene": "Scene_0",
        "start_frame": 100,
        "duration_sec": 180,
    }

    overrides = RUNNER.load_trace_overrides(
        trace, item, sections=[2], source_run="slam_b32_covisibility"
    )

    assert list(overrides) == [2]
    assert len(overrides[2]) == 76
    assert overrides[2][153] == {
        "memory_frame": 0,
        "source_run": "slam_b32_covisibility",
    }


def test_merge_overrides_changes_only_replacement_section(tmp_path):
    baseline_trace = tmp_path / "baseline.jsonl"
    policy_trace = tmp_path / "policy.jsonl"
    write_trace(baseline_trace, source_frame_offset=0)
    write_trace(policy_trace, source_frame_offset=100)
    item = {
        "scene": "Scene_0",
        "start_frame": 100,
        "duration_sec": 180,
    }
    baseline = RUNNER.load_trace_overrides(
        baseline_trace, item, sections=[1, 2], source_run="baseline"
    )
    replacement = RUNNER.load_trace_overrides(
        policy_trace, item, sections=[2], source_run="slam"
    )

    merged = RUNNER.merge_overrides(baseline, replacement)

    assert merged[1] == baseline[1]
    assert merged[2] == replacement[2]
    assert baseline[2] != replacement[2]


def test_case_directory_name_is_stable():
    case = {"case_index": "3", "row": "18", "section_idx": "55"}

    assert RUNNER.case_directory_name(case) == "case_03_row18_section55"
