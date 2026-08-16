import importlib.util
from pathlib import Path

import numpy as np


def load_analysis_module():
    module_path = Path(__file__).with_name("analyze_memory_policies.py")
    spec = importlib.util.spec_from_file_location("analyze_memory_policies", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summarize_module():
    module_path = Path(__file__).with_name("summarize_ri_alignment.py")
    spec = importlib.util.spec_from_file_location("summarize_ri_alignment", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_scored_policy_scene(num_frames=153):
    # 0, 10, 76 are geometrically/visually distinct from everything else,
    # which stays tightly clustered near the origin -- so RI's rarity and
    # SLAM's non-redundancy signals both have something real to act on, and
    # {0, 10, 76} are exactly the frames overlap_map marks as truly useful
    # for section-1 targets (compare against ground truth, not each other).
    item = {
        "_row": 1,
        "scene": "synthetic_scored",
        "start_frame": 0,
        "duration_sec": 10,
        "num_frames": num_frames,
    }
    overlap_map = {frame_idx: set() for frame_idx in range(num_frames)}
    for target_frame in range(77, num_frames):
        overlap_map[target_frame] = {0, 10, 76}

    distinct_frames = [0, 10, 76]
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    for frame_idx in range(num_frames):
        if frame_idx in distinct_frames:
            c2ws[frame_idx, 0, 3] = 100.0 * (1 + distinct_frames.index(frame_idx))
        else:
            c2ws[frame_idx, 0, 3] = 0.01 * frame_idx

    dino_features = {}
    rgb_features = {}
    for frame_idx in range(num_frames):
        dino_vector = np.full(8, 0.1, dtype=np.float32)
        rgb_value = 0.1
        if frame_idx in distinct_frames:
            dino_vector = np.zeros(8, dtype=np.float32)
            dino_vector[distinct_frames.index(frame_idx)] = 1.0
            rgb_value = 0.9
        dino_features[frame_idx] = dino_vector
        rgb_features[frame_idx] = np.full(12, rgb_value, dtype=np.float32)

    return item, overlap_map, c2ws, dino_features, rgb_features


def main():
    analysis = load_analysis_module()
    item = {
        "_row": 0,
        "scene": "synthetic",
        "start_frame": 0,
        "duration_sec": 10,
        "num_frames": 153,
    }
    overlap_map = {frame_idx: set() for frame_idx in range(153)}
    for target_frame in range(77, 153):
        overlap_map[target_frame] = {0, 10, 76}

    fifo_summary, _, _ = analysis.simulate_row(
        item=item,
        policy="fifo",
        budget=2,
        overlap_map=overlap_map,
    )
    belady_summary, _, _ = analysis.simulate_row(
        item=item,
        policy="belady",
        budget=2,
        overlap_map=overlap_map,
    )
    coverage_summary, _, _ = analysis.simulate_row(
        item=item,
        policy="coverage_oracle",
        budget=2,
        overlap_map=overlap_map,
    )

    assert belady_summary["coverage"] == 1.0
    assert coverage_summary["coverage"] == 1.0
    assert belady_summary["coverage"] > fifo_summary["coverage"]
    assert belady_summary["oracle_recall"] >= fifo_summary["oracle_recall"]

    usefulness = analysis.compute_frame_usefulness_rows(item, overlap_map)
    assert usefulness[0]["future_use_count"] == 76
    assert usefulness[10]["future_use_count"] == 76

    # -- ri / slam / slam_ri_blend: real scoring functions, real GT-overlap
    # oracle, budget respected, score_rows carry the right column names.
    scored_item, scored_overlap_map, c2ws, dino_features, rgb_features = (
        build_scored_policy_scene()
    )
    budget = 8

    for policy in ("ri", "slam", "slam_ri_blend"):
        try:
            analysis.simulate_row(
                item=scored_item,
                policy=policy,
                budget=1,
                overlap_map=scored_overlap_map,
                dino_features=dino_features,
                rgb_features=rgb_features,
                c2ws=c2ws,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{policy} with budget=1 should have raised")

    ri_summary, _, ri_scores = analysis.simulate_row(
        item=scored_item,
        policy="ri",
        budget=budget,
        overlap_map=scored_overlap_map,
        dino_features=dino_features,
        rgb_features=rgb_features,
    )
    slam_summary, _, slam_scores = analysis.simulate_row(
        item=scored_item,
        policy="slam",
        budget=budget,
        overlap_map=scored_overlap_map,
        dino_features=dino_features,
        rgb_features=rgb_features,
        c2ws=c2ws,
    )
    blend_summary, _, blend_scores = analysis.simulate_row(
        item=scored_item,
        policy="slam_ri_blend",
        budget=budget,
        overlap_map=scored_overlap_map,
        dino_features=dino_features,
        rgb_features=rgb_features,
        c2ws=c2ws,
        slamri_beta=0.5,
    )

    assert ri_summary["final_memory_size"] <= budget
    assert slam_summary["final_memory_size"] <= budget
    assert blend_summary["final_memory_size"] <= budget
    assert ri_scores and all("ri_score" in row for row in ri_scores)
    assert slam_scores and all("slam_score" in row for row in slam_scores)
    assert blend_scores and all("blend_score" in row for row in blend_scores)
    # All three real, budgeted policies should keep at least one of the
    # truly-distinctive frames over the redundant cluster -- not asserting
    # they keep *all* of {0, 10, 76} (eviction order/ties can vary), just
    # that the GT-useful set isn't wiped out entirely.
    for scores in (ri_scores, slam_scores, blend_scores):
        kept_distinct = {
            row["frame_idx"] for row in scores if row["kept_after"] and row["frame_idx"] in (0, 10, 76)
        }
        assert kept_distinct, "expected at least one truly-useful frame to survive"

    # summarize_ri_alignment.py: default score_key stays exactly "ri_score"
    # (backward compatible with its documented ri_frame_scores.jsonl usage),
    # and a non-default score_key (slam_score) produces its own prefix.
    summarize = load_summarize_module()
    ri_grouped = summarize.group_rows(ri_scores)
    ri_decisions = [
        summarize.summarize_decision(key, group, topk=4)
        for key, group in ri_grouped.items()
    ]
    assert ri_decisions and "spearman_ri_vs_gt_future" in ri_decisions[0]

    slam_grouped = summarize.group_rows(slam_scores)
    slam_decisions = [
        summarize.summarize_decision(key, group, topk=4, score_key="slam_score")
        for key, group in slam_grouped.items()
    ]
    assert slam_decisions and "spearman_slam_vs_gt_future" in slam_decisions[0]
    assert "spearman_ri_vs_gt_future" not in slam_decisions[0]

    print("memory policy analysis checks passed")


if __name__ == "__main__":
    main()
