import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "analyze_fixed_history_candidate_pools.py"
SPEC = importlib.util.spec_from_file_location("fixed_history_pools", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_nested_pools_share_core_and_are_subsets():
    sizes = MODULE.parse_pool_sizes("4,6,8,all")
    pools = MODULE.nested_candidate_pools(range(12), sizes, order_seed=7)

    assert pools["b4"] == [8, 9, 10, 11]
    assert len(pools["b6"]) == 6
    assert len(pools["b8"]) == 8
    assert pools["all"] == list(range(12))
    assert set(pools["b4"]) < set(pools["b6"]) < set(pools["b8"])
    assert set(pools["b8"]) < set(pools["all"])


def test_nested_pool_order_is_reproducible():
    sizes = MODULE.parse_pool_sizes("4,8,all")
    first = MODULE.nested_candidate_pools(range(20), sizes, order_seed=11)
    second = MODULE.nested_candidate_pools(range(20), sizes, order_seed=11)

    assert first == second


def test_pool_winner_reuses_one_score_vector():
    candidates = [0, 1, 2, 3]
    scores = np.asarray([0.2, 0.9, 0.5, 0.7])

    winner_small, score_small = MODULE.select_pool_winner(
        candidates, scores, [0, 2, 3]
    )
    winner_full, score_full = MODULE.select_pool_winner(
        candidates, scores, candidates
    )

    assert (winner_small, score_small) == (3, 0.7)
    assert (winner_full, score_full) == (1, 0.9)


def test_tied_winner_matches_first_candidate_scan_order():
    winner, score = MODULE.select_pool_winner(
        [4, 9, 12], np.asarray([0.8, 0.8, 0.1]), [4, 9]
    )

    assert winner == 4
    assert score == 0.8


def test_batched_overlap_identical_pose_is_near_one():
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
    scores = MODULE.batched_fov_overlap_scores(
        c2ws,
        target_frame=0,
        candidate_frames=[1],
        num_samples=2000,
        batch_size=1,
        seed=3,
    )

    assert scores.shape == (1,)
    assert scores[0] > 0.99
