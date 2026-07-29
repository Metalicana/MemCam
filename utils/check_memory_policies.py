import importlib.util
from pathlib import Path

import numpy as np


MEMORY_POLICIES_PATH = Path(__file__).resolve().parents[1] / "diffsynth" / "pipelines" / "memory_policies.py"
spec = importlib.util.spec_from_file_location("memory_policies", MEMORY_POLICIES_PATH)
memory_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_policies)
FrameMemoryBuffer = memory_policies.FrameMemoryBuffer
camera_trajectory_similarity = memory_policies.camera_trajectory_similarity
compute_facility_coreset_scores = memory_policies.compute_facility_coreset_scores
compute_h2o_heavy_hitter_scores = memory_policies.compute_h2o_heavy_hitter_scores
compute_kcenter_coreset_scores = memory_policies.compute_kcenter_coreset_scores
compute_rarity_irreplaceability_scores = memory_policies.compute_rarity_irreplaceability_scores
compute_slam_covisibility_scores = memory_policies.compute_slam_covisibility_scores
compute_trajectory_coverage_scores = memory_policies.compute_trajectory_coverage_scores


def check_unbounded():
    memory = FrameMemoryBuffer(policy="unbounded")
    memory.update(range(5))
    assert memory.candidates() == [0, 1, 2, 3, 4]
    assert memory.candidates(exclude_frames={1, 3}) == [0, 2, 4]


def check_fifo():
    memory = FrameMemoryBuffer(policy="fifo", budget=3)
    evicted = memory.update(range(5))
    assert evicted == [0, 1]
    assert memory.candidates() == [2, 3, 4]
    evicted = memory.add(5)
    assert evicted == [2]
    assert memory.candidates() == [3, 4, 5]
    assert memory.candidates(exclude_frames={4}) == [3, 5]


def check_fifo_requires_budget():
    try:
        FrameMemoryBuffer(policy="fifo")
    except ValueError:
        return
    raise AssertionError("FIFO without a budget should fail")


def make_line_c2ws(num_frames):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    c2ws[:, 0, 3] = np.arange(num_frames, dtype=np.float64)
    return c2ws


def check_rarity_irreplaceability_budgeting():
    memory = FrameMemoryBuffer(
        policy="rarity_irreplaceability",
        budget=3,
        pinned_frames={0},
    )
    scores = {0: float("inf"), 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    evicted = memory.update(range(5), eviction_scores=scores, protected_frames={4})
    assert evicted == [1, 2]
    assert memory.candidates() == [0, 3, 4]


def check_rarity_irreplaceability_requires_budget():
    try:
        FrameMemoryBuffer(policy="rarity_irreplaceability")
    except ValueError:
        return
    raise AssertionError("rarity_irreplaceability without a budget should fail")


def check_slam_covisibility_requires_budget():
    try:
        FrameMemoryBuffer(policy="slam_covisibility")
    except ValueError:
        return
    raise AssertionError("slam_covisibility without a budget should fail")


def check_facility_coreset_requires_budget():
    try:
        FrameMemoryBuffer(policy="facility_coreset")
    except ValueError:
        return
    raise AssertionError("facility_coreset without a budget should fail")


def check_kcenter_coreset_requires_budget():
    try:
        FrameMemoryBuffer(policy="kcenter_coreset")
    except ValueError:
        return
    raise AssertionError("kcenter_coreset without a budget should fail")


def check_trajectory_coverage_requires_budget():
    try:
        FrameMemoryBuffer(policy="trajectory_coverage")
    except ValueError:
        return
    raise AssertionError("trajectory_coverage without a budget should fail")


def check_h2o_heavy_hitter_requires_budget():
    try:
        FrameMemoryBuffer(policy="h2o_heavy_hitter")
    except ValueError:
        return
    raise AssertionError("h2o_heavy_hitter without a budget should fail")


def check_rarity_irreplaceability_scores():
    memory = FrameMemoryBuffer(
        policy="rarity_irreplaceability",
        budget=4,
        pinned_frames={0},
    )
    memory.update([0, 1, 2, 5], eviction_scores={idx: 0.0 for idx in [0, 1, 2, 5]})
    dino_features = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.99, 0.01], dtype=np.float32),
        2: np.array([0.98, 0.02], dtype=np.float32),
        5: np.array([0.0, 1.0], dtype=np.float32),
    }
    rgb_features = {
        0: np.zeros(12, dtype=np.float32),
        1: np.full(12, 0.01, dtype=np.float32),
        2: np.full(12, 0.02, dtype=np.float32),
        5: np.ones(12, dtype=np.float32),
    }

    scores, details = compute_rarity_irreplaceability_scores(
        memory_frame_indices=memory.candidates(),
        pinned_frames={0},
        dino_features=dino_features,
        rgb_features=rgb_features,
        return_details=True,
    )
    assert set(scores) == {0, 1, 2, 5}
    assert scores[0] == float("inf")
    assert scores[5] > scores[1]
    assert details[5]["rarity"] > details[1]["rarity"]
    assert details[5]["irreplaceability"] > details[1]["irreplaceability"]

    before = scores[2]
    memory.record_selection(2, 0.8)
    after_scores = compute_rarity_irreplaceability_scores(
        memory_frame_indices=memory.candidates(),
        pinned_frames={0},
        dino_features=dino_features,
        rgb_features=rgb_features,
    )
    assert after_scores[2] == before


def check_slam_covisibility_scores():
    c2ws = make_line_c2ws(8)
    dino_features = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.99, 0.01], dtype=np.float32),
        2: np.array([0.98, 0.02], dtype=np.float32),
        7: np.array([0.0, 1.0], dtype=np.float32),
    }
    scores, details = compute_slam_covisibility_scores(
        memory_frame_indices=[0, 1, 2, 7],
        c2ws=c2ws,
        pinned_frames={0},
        dino_features=dino_features,
        covisibility_threshold=0.5,
        n_other_observers=2,
        return_details=True,
    )
    assert scores[0] == float("inf")
    assert scores[7] > scores[1]
    assert details[1]["covisible_observers"] >= 2
    assert details[7]["covisible_observers"] == 0


def check_facility_coreset_scores():
    c2ws = make_line_c2ws(8)
    dino_features = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.99, 0.01], dtype=np.float32),
        2: np.array([0.98, 0.02], dtype=np.float32),
        7: np.array([0.0, 1.0], dtype=np.float32),
    }
    quality = {frame_idx: 1.0 for frame_idx in dino_features}
    scores, details = compute_facility_coreset_scores(
        memory_frame_indices=[0, 1, 2, 7],
        archive_frame_indices=[0, 1, 2, 7],
        c2ws=c2ws,
        budget=2,
        dino_features=dino_features,
        frame_quality=quality,
        return_details=True,
    )
    selected = {frame_idx for frame_idx, row in details.items() if row["coreset_selected"]}
    assert len(selected) == 2
    assert 7 in selected
    assert selected & {0, 1, 2}
    assert all(scores[frame_idx] > 0 for frame_idx in selected)
    assert all(scores[frame_idx] < 0 for frame_idx in set(scores) - selected)

    forced_scores, forced_details = compute_facility_coreset_scores(
        memory_frame_indices=[0, 1, 2, 7],
        archive_frame_indices=[0, 1, 2, 7],
        c2ws=c2ws,
        budget=2,
        forced_keep_frames={0},
        dino_features=dino_features,
        frame_quality=quality,
        return_details=True,
    )
    forced_selected = {
        frame_idx for frame_idx, row in forced_details.items() if row["coreset_selected"]
    }
    assert 0 in forced_selected
    assert forced_scores[0] == float("inf")


def check_kcenter_coreset_scores():
    c2ws = make_line_c2ws(8)
    dino_features = {
        frame_idx: np.array([float(frame_idx), 1.0], dtype=np.float32)
        for frame_idx in range(8)
    }
    scores, details = compute_kcenter_coreset_scores(
        memory_frame_indices=[0, 1, 2, 7],
        archive_frame_indices=list(range(8)),
        c2ws=c2ws,
        budget=2,
        forced_keep_frames={7},
        dino_features=dino_features,
        visual_weight=0.0,
        pose_weight=1.0,
        return_details=True,
    )
    selected = {frame_idx for frame_idx, row in details.items() if row["kcenter_selected"]}
    assert selected == {0, 7}
    assert scores[7] == float("inf")
    assert scores[0] > 0
    assert scores[1] < 0 and scores[2] < 0
    assert details[0]["kcenter_radius"] >= 0


def check_camera_trajectory_similarity():
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    yaw_180 = np.diag([-1.0, -1.0, 1.0])
    c2ws[1, :3, :3] = yaw_180
    c2ws[2, 0, 3] = 101.0

    similarity = camera_trajectory_similarity(
        c2ws=c2ws,
        query_frame_indices=[0],
        memory_frame_indices=[0, 1, 2],
        radius=50.0,
    )
    np.testing.assert_allclose(similarity[0], [1.0, 0.0, 0.0], atol=1e-8)


def check_trajectory_coverage_scores():
    c2ws = make_line_c2ws(8)
    c2ws[3:, 0, 3] += 20.0
    candidates = [0, 1, 6, 7]
    archive = list(range(8))

    scores, details = compute_trajectory_coverage_scores(
        memory_frame_indices=candidates,
        archive_frame_indices=archive,
        c2ws=c2ws,
        budget=2,
        forced_keep_frames={0},
        radius=5.0,
        return_details=True,
    )
    selected = {
        frame_idx
        for frame_idx, row in details.items()
        if row["trajectory_selected"]
    }
    assert 0 in selected
    assert len(selected) == 2
    assert selected & {6, 7}
    assert not ({0, 1} <= selected)
    assert scores[0] == float("inf")
    assert details[0]["trajectory_archive_size"] == len(archive)
    assert 0.0 <= details[0]["trajectory_value"] <= 1.0

    changed_future = c2ws.copy()
    changed_future[6:, :3, 3] += 1000.0
    causal_scores, causal_details = compute_trajectory_coverage_scores(
        memory_frame_indices=[0, 1, 4, 5],
        archive_frame_indices=list(range(6)),
        c2ws=c2ws,
        budget=2,
        forced_keep_frames={0},
        radius=5.0,
        return_details=True,
    )
    changed_scores, changed_details = compute_trajectory_coverage_scores(
        memory_frame_indices=[0, 1, 4, 5],
        archive_frame_indices=list(range(6)),
        c2ws=changed_future,
        budget=2,
        forced_keep_frames={0},
        radius=5.0,
        return_details=True,
    )
    assert causal_scores == changed_scores
    assert {
        frame_idx
        for frame_idx, row in causal_details.items()
        if row["trajectory_selected"]
    } == {
        frame_idx
        for frame_idx, row in changed_details.items()
        if row["trajectory_selected"]
    }


def check_h2o_heavy_hitter_scores():
    stats = {
        0: {"selection_overlap_sum": 0.0, "selected_count": 0, "best_selection_overlap": 0.0},
        1: {"selection_overlap_sum": 6.0, "selected_count": 6, "best_selection_overlap": 0.9},
        2: {"selection_overlap_sum": 0.1, "selected_count": 1, "best_selection_overlap": 0.1},
        3: {"selection_overlap_sum": 0.0, "selected_count": 0, "best_selection_overlap": 0.0},
        4: {"selection_overlap_sum": 0.0, "selected_count": 0, "best_selection_overlap": 0.0},
    }
    scores, details = compute_h2o_heavy_hitter_scores(
        memory_frame_indices=[0, 1, 2, 3, 4],
        memory_stats=stats,
        budget=3,
        forced_keep_frames={4},
        recency_fraction=1 / 3,
        return_details=True,
    )
    memory = FrameMemoryBuffer(policy="h2o_heavy_hitter", budget=3)
    evicted = memory.update(
        [0, 1, 2, 3, 4],
        eviction_scores=scores,
        protected_frames={4},
    )
    kept = set(memory.candidates())
    assert 4 in kept
    assert 1 in kept
    assert len(kept) == 3
    assert 0 in evicted or 2 in evicted or 3 in evicted
    assert details[1]["h2o_read_weight"] == 6.0
    assert details[4]["h2o_recent_keep"]


if __name__ == "__main__":
    check_unbounded()
    check_fifo()
    check_fifo_requires_budget()
    check_rarity_irreplaceability_budgeting()
    check_rarity_irreplaceability_requires_budget()
    check_rarity_irreplaceability_scores()
    check_slam_covisibility_requires_budget()
    check_slam_covisibility_scores()
    check_facility_coreset_requires_budget()
    check_facility_coreset_scores()
    check_kcenter_coreset_requires_budget()
    check_kcenter_coreset_scores()
    check_trajectory_coverage_requires_budget()
    check_camera_trajectory_similarity()
    check_trajectory_coverage_scores()
    check_h2o_heavy_hitter_requires_budget()
    check_h2o_heavy_hitter_scores()
    print("memory policy checks passed")
