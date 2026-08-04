import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "analyze_unbounded_retrieval_instability.py"
SPEC = importlib.util.spec_from_file_location("retrieval_instability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unbounded_candidates_match_pipeline_exclusion():
    candidates = MODULE.unbounded_candidates(1)

    assert len(candidates) == 73
    assert candidates[0] == 0
    assert candidates[-1] == 72


def test_parse_pool_sizes_and_recent_cap():
    pools = MODULE.parse_pool_sizes("32,128,all")

    assert pools == [("recent_32", 32), ("recent_128", 128), ("all", None)]
    assert MODULE.capped_candidates(list(range(100)), 3) == [97, 98, 99]


def test_stable_score_matrix_has_one_winner():
    scores = np.asarray(
        [
            [0.2, 0.8, 0.3],
            [0.1, 0.7, 0.4],
            [0.2, 0.9, 0.3],
        ]
    )

    summary = MODULE.summarize_score_matrix(
        scores,
        candidate_frames=[10, 20, 30],
        target_frame=100,
    )

    assert summary["unique_winner_count"] == 1
    assert summary["modal_winner_share"] == 1.0
    assert summary["winner_age_span"] == 0
    assert summary["cross_section_winner_switch"] == 0


def test_distant_winners_are_reported_as_temporal_switch():
    scores = np.asarray(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )

    summary = MODULE.summarize_score_matrix(
        scores,
        candidate_frames=[0, 200],
        target_frame=300,
    )

    assert summary["unique_winner_count"] == 2
    assert summary["modal_winner_share"] == 0.5
    assert summary["winner_age_span"] == 200
    assert summary["cross_section_winner_switch"] == 1


def test_auto_sections_excludes_initial_section():
    assert MODULE.auto_sections(total_sections=1, count=5) == []
    assert MODULE.auto_sections(total_sections=3, count=5) == [1, 2]


def test_label_fields_measure_winner_agreement():
    summary = {
        "winner_frames": "10;20;20;30",
        "reference_winner_frame": 20,
    }

    output = MODULE.add_label_fields(
        summary,
        labeled_overlaps={20, 40},
        candidate_frames=[10, 20, 30],
    )

    assert output["candidate_labeled_overlap_count"] == 1
    assert output["winner_labeled_overlap_rate"] == 0.5
    assert output["reference_winner_labeled_overlap"] == 1


def test_query_keys_do_not_depend_on_manifest_row_number():
    item = {
        "scene": "ClothingStore_1",
        "start_frame": 228,
        "duration_sec": 60,
        "_row": 18,
    }
    trace = {
        "scene": "ClothingStore_1",
        "dataset_start_frame": 228,
        "duration_sec": 60,
        "row": 999,
        "section_idx": 23,
        "target_frame": 1768,
    }

    assert MODULE.manifest_query_key(item, 23, 1768) == MODULE.trace_query_key(trace)


def test_same_numeric_row_cannot_join_different_scenes():
    item = {
        "scene": "ClothingStore_1",
        "start_frame": 228,
        "duration_sec": 60,
        "_row": 18,
    }
    wrong_trace = {
        "scene": "AncientTempleEnv_5",
        "dataset_start_frame": 684,
        "duration_sec": 60,
        "row": 18,
        "section_idx": 23,
        "target_frame": 1768,
    }

    assert MODULE.manifest_query_key(item, 23, 1768) != MODULE.trace_query_key(
        wrong_trace
    )
