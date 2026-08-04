import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "analyze_retrieval_quality_decomposition.py"
SPEC = importlib.util.spec_from_file_location("retrieval_decomposition", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reconstruct_unbounded_bank_matches_pipeline_exclusion():
    banks = MODULE.reconstruct_candidate_banks([], max_section=2, num_frames=229)

    assert banks[1] == list(range(73))
    assert banks[2] == list(range(149))


def test_reconstruct_budgeted_bank_applies_previous_section_evictions():
    events = [
        {
            "event": "memory_eviction",
            "section_idx": 0,
            "evicted_memory_frame": frame_idx,
        }
        for frame_idx in range(45)
    ]

    banks = MODULE.reconstruct_candidate_banks(events, max_section=1, num_frames=153)

    # The 32-frame post-section bank is 45..76. Frames 73..76 are anchors and
    # therefore absent from the section-1 candidate set.
    assert banks[1] == list(range(45, 73))


def test_query_decomposition_separates_retention_and_retrieval_gap():
    # Generated frame 0 is a perfect target match, frame 1 is close, and frame 2
    # is orthogonal. The bank retained 1 and 2, while retrieval selected 2.
    generated = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gt = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    row = MODULE.compute_query_decomposition(
        generated_features=generated,
        gt_features=gt,
        bank_candidates=[1, 2],
        history_candidates=[0, 1, 2],
        selected_frame=2,
        target_frame=3,
    )

    assert row["full_oracle_frame"] == 0
    assert row["bank_oracle_frame"] == 1
    assert np.isclose(row["retention_gap"], 0.2)
    assert np.isclose(row["retrieval_gap"], 0.8)
    assert np.isclose(row["total_oracle_gap"], 1.0)


def test_selected_frame_must_belong_to_reconstructed_bank():
    features = np.eye(3, dtype=np.float32)

    try:
        MODULE.compute_query_decomposition(
            generated_features=features,
            gt_features=features,
            bank_candidates=[0],
            history_candidates=[0, 1],
            selected_frame=1,
            target_frame=2,
        )
    except ValueError as exc:
        assert "absent from the reconstructed bank" in str(exc)
    else:
        raise AssertionError("Expected an invalid selected-frame error")


def test_replay_selection_requires_both_effective_and_retrieval_improvement():
    candidates = [
        {
            "row": 1,
            "section_idx": 30,
            "selected_effective_improvement": 0.2,
            "retrieval_gap_improvement": 0.1,
            "replay_score": 0.25,
        },
        {
            "row": 2,
            "section_idx": 31,
            "selected_effective_improvement": 0.3,
            "retrieval_gap_improvement": -0.1,
            "replay_score": 0.25,
        },
        {
            "row": 1,
            "section_idx": 40,
            "selected_effective_improvement": 0.4,
            "retrieval_gap_improvement": 0.2,
            "replay_score": 0.5,
        },
    ]

    selected = MODULE.select_replay_cases(
        candidates, count=4, min_section=20, unique_rows=True
    )

    assert len(selected) == 1
    assert selected[0]["section_idx"] == 40
    assert selected[0]["case_index"] == 0
