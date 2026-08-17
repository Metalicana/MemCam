import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MEMORY_POLICIES_PATH = REPO_ROOT / "diffsynth" / "pipelines" / "memory_policies.py"
spec = importlib.util.spec_from_file_location("memory_policies", MEMORY_POLICIES_PATH)
memory_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_policies)
FrameMemoryBuffer = memory_policies.FrameMemoryBuffer
VisualMemoryFeatureExtractor = memory_policies.VisualMemoryFeatureExtractor
compute_rarity_irreplaceability_scores = memory_policies.compute_rarity_irreplaceability_scores
compute_slam_covisibility_scores = memory_policies.compute_slam_covisibility_scores
compute_slam_ri_blend_scores = memory_policies.compute_slam_ri_blend_scores


# Policies whose per-frame eviction score is a real, reusable function (not a
# GT-overlap oracle like belady/coverage_oracle) -- these are the ones this
# script's "is the score ad hoc" correlation check applies to.
SCORED_POLICIES = ("ri", "slam", "slam_ri_blend")


FRAMES_PER_SECTION = 77
PREDICT_FRAMES = 76


def load_manifest(manifest_path):
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_row"] = row_index
            rows.append(item)
    return rows


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_rows(value):
    if not value:
        return None

    rows = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            rows.update(range(int(start_text), int(end_text) + 1))
        else:
            rows.add(int(part))
    return rows


def select_rows(items, row_filter, durations, limit):
    selected = []
    duration_filter = set(durations) if durations else None
    for item in items:
        if row_filter is not None and item["_row"] not in row_filter:
            continue
        if duration_filter is not None and int(item["duration_sec"]) not in duration_filter:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def resolve_overlap_dir(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "overlap_labels" / item["scene"]
    return Path(item["overlap_dir"])


def resolve_gt_frames_dir(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "frames" / item["scene"]
    return Path(item["gt_frames_dir"])


def resolve_pose_path(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "jsons" / f"{item['scene']}.json"
    return Path(item["pose_path"])


def extract_overlap_indices(data):
    if isinstance(data, dict):
        for key in ("overlapping_frames", "overlap_frames", "frames", "indices"):
            if key in data:
                return extract_overlap_indices(data[key])
        if "frame_idx" in data:
            return [int(data["frame_idx"])]
        if "index" in data:
            return [int(data["index"])]
        return []

    if isinstance(data, list):
        indices = []
        for item in data:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str) and item.lstrip("-").isdigit():
                indices.append(int(item))
            elif isinstance(item, dict):
                indices.extend(extract_overlap_indices(item))
        return indices

    return []


def load_overlap_map(overlap_dir, start_frame, num_frames):
    overlap_map = {}
    for local_frame_idx in range(num_frames):
        global_frame_idx = start_frame + local_frame_idx
        path = overlap_dir / f"{global_frame_idx}.json"
        if not path.exists():
            path = overlap_dir / f"{local_frame_idx}.json"
        if not path.exists():
            overlap_map[local_frame_idx] = set()
            continue

        with path.open("r", encoding="utf-8") as handle:
            raw_indices = extract_overlap_indices(json.load(handle))

        local_indices = set()
        for frame_idx in raw_indices:
            local_idx = int(frame_idx) - start_frame
            if 0 <= local_idx < num_frames:
                local_indices.add(local_idx)

        # Fallback for datasets whose overlap labels are already local.
        if not local_indices:
            for frame_idx in raw_indices:
                local_idx = int(frame_idx)
                if 0 <= local_idx < num_frames:
                    local_indices.add(local_idx)

        overlap_map[local_frame_idx] = local_indices
    return overlap_map


def section_ranges(section_idx):
    section_start = section_idx * (FRAMES_PER_SECTION - 1)
    if section_idx == 0:
        anchor_range = [section_start]
    else:
        anchor_range = list(range(section_start - 3, section_start + 1))
    predict_range = list(range(section_start + 1, section_start + FRAMES_PER_SECTION))
    return section_start, anchor_range, predict_range


def available_useful_frames(target_frame, overlap_map, generated_until, exclude_frames):
    return {
        frame_idx
        for frame_idx in overlap_map.get(target_frame, set())
        if 0 <= frame_idx <= generated_until and frame_idx not in exclude_frames
    }


def next_use_distance(frame_idx, future_targets, useful_by_target):
    for target_frame in future_targets:
        if frame_idx in useful_by_target.get(target_frame, set()):
            return target_frame
    return math.inf


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def frame_future_use(frame_idx, target_frames, overlap_map):
    target_frames = list(target_frames)
    hits = [target for target in target_frames if frame_idx in overlap_map.get(target, set())]
    if hits:
        next_use = hits[0]
        next_use_distance_value = next_use - frame_idx
        last_use = hits[-1]
    else:
        next_use = None
        next_use_distance_value = math.inf
        last_use = None

    return {
        "future_use_count": len(hits),
        "future_use_fraction": safe_div(len(hits), len(target_frames)),
        "next_use_frame": next_use,
        "next_use_distance": next_use_distance_value,
        "last_use_frame": last_use,
    }


def compute_frame_usefulness_rows(item, overlap_map):
    num_frames = int(item["num_frames"])
    rows = []
    for frame_idx in range(num_frames):
        future_targets = range(frame_idx + 1, num_frames)
        usefulness = frame_future_use(frame_idx, future_targets, overlap_map)
        rows.append(
            {
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "frame_idx": frame_idx,
                "global_frame_idx": int(item["start_frame"]) + frame_idx,
                "section_idx": frame_idx // PREDICT_FRAMES,
                **usefulness,
            }
        )
    return rows


def belady_evict(memory, budget, protected_frames, future_targets, useful_by_target):
    evicted = []
    protected_frames = set(protected_frames or [])
    while len(memory) > budget:
        evictable = [frame_idx for frame_idx in sorted(memory) if frame_idx not in protected_frames]
        if not evictable:
            break
        evict_idx = max(
            evictable,
            key=lambda frame_idx: (
                next_use_distance(frame_idx, future_targets, useful_by_target),
                -frame_idx,
            ),
        )
        memory.remove(evict_idx)
        evicted.append(evict_idx)
    return evicted


def coverage_oracle_evict(memory, budget, protected_frames, future_targets, useful_by_target):
    protected_frames = set(protected_frames or []) & memory
    selected = set(protected_frames)
    uncovered_targets = set(future_targets)

    for target_frame in list(uncovered_targets):
        if useful_by_target.get(target_frame, set()) & selected:
            uncovered_targets.remove(target_frame)

    while len(selected) < budget:
        candidates = sorted(memory - selected)
        if not candidates:
            break

        def candidate_key(frame_idx):
            covered = {
                target
                for target in uncovered_targets
                if frame_idx in useful_by_target.get(target, set())
            }
            total_future_use = sum(
                1 for target in future_targets if frame_idx in useful_by_target.get(target, set())
            )
            return (len(covered), total_future_use, -frame_idx)

        best_frame = max(candidates, key=candidate_key)
        if candidate_key(best_frame)[0] == 0 and len(selected) >= len(protected_frames):
            # Fill spare budget with highest total-use frames even if all remaining targets are covered.
            best_frame = max(
                candidates,
                key=lambda frame_idx: (
                    sum(
                        1
                        for target in future_targets
                        if frame_idx in useful_by_target.get(target, set())
                    ),
                    -frame_idx,
                ),
            )
        selected.add(best_frame)
        for target_frame in list(uncovered_targets):
            if best_frame in useful_by_target.get(target_frame, set()):
                uncovered_targets.remove(target_frame)

    # If protected frames already exceeded the budget, keep them and accept overflow.
    if len(selected) > budget:
        selected = set(protected_frames)

    evicted = sorted(memory - selected)
    memory.intersection_update(selected)
    return evicted


def fifo_evict(memory, budget, protected_frames):
    evicted = []
    protected_frames = set(protected_frames or [])
    while len(memory) > budget:
        evictable = [frame_idx for frame_idx in sorted(memory) if frame_idx not in protected_frames]
        if not evictable:
            break
        evict_idx = evictable[0]
        memory.remove(evict_idx)
        evicted.append(evict_idx)
    return evicted


def ri_evict(memory_buffer, dino_features, rgb_features, protected_frames, pinned_frames):
    memory_frame_indices = memory_buffer.candidates()
    scores = compute_rarity_irreplaceability_scores(
        memory_frame_indices=memory_frame_indices,
        pinned_frames=pinned_frames,
        dino_features=dino_features,
        rgb_features=rgb_features,
    )
    return memory_buffer.evict_to_budget(protected_frames=protected_frames), scores


def evaluate_memory_for_section(memory, predict_range, overlap_map, generated_until, exclude_frames):
    possible_targets = 0
    covered_targets = 0
    retained_useful = 0
    available_useful = 0
    best_possible = 0

    for target_frame in predict_range:
        useful = available_useful_frames(
            target_frame=target_frame,
            overlap_map=overlap_map,
            generated_until=generated_until,
            exclude_frames=exclude_frames,
        )
        if useful:
            possible_targets += 1
            best_possible += 1
        retained = useful & memory
        if retained:
            covered_targets += 1
        retained_useful += len(retained)
        available_useful += len(useful)

    return {
        "targets": len(predict_range),
        "possible_targets": possible_targets,
        "covered_targets": covered_targets,
        "coverage": covered_targets / len(predict_range) if predict_range else 0.0,
        "possible_coverage": (
            covered_targets / possible_targets if possible_targets else 0.0
        ),
        "oracle_recall": (
            retained_useful / available_useful if available_useful else 0.0
        ),
        "retained_useful": retained_useful,
        "available_useful": available_useful,
        "best_possible_coverage": (
            best_possible / len(predict_range) if predict_range else 0.0
        ),
        # How many future targets get served per occupied memory slot -- the
        # direct "is budget wasted on redundant anchors" measure SLAM's
        # non-redundancy design targets, distinct from oracle_recall (which
        # normalizes by available_useful, not by memory spent).
        "retained_memory_size": len(memory),
    }


def rank_desc(values_by_key):
    sorted_keys = sorted(
        values_by_key,
        key=lambda key: (
            values_by_key[key] if values_by_key[key] is not None else -math.inf,
            -key,
        ),
        reverse=True,
    )
    return {key: rank + 1 for rank, key in enumerate(sorted_keys)}


def make_score_rows(
    item,
    policy,
    section_idx,
    decision_frame,
    budget,
    memory_before,
    new_frames,
    evicted_frames,
    memory_after,
    protected_frames,
    pinned_frames,
    scores,
    score_details,
    overlap_map,
    score_field,
    rank_field,
    detail_fields,
):
    """Per-frame rows pairing a policy's real eviction score against ground-truth
    future reuse -- the strongest available oracle for "was keeping this frame
    worth it": actual dataset view-overlap, not a learned embedding proxy.

    detail_fields maps output column name -> key into score_details[frame_idx],
    so each policy exposes its own native score components without three
    near-duplicate row builders. score_field/rank_field name the score and its
    rank column (kept distinct per policy, e.g. "ri_score" vs "slam_score", so
    existing consumers of one policy's schema -- summarize_ri_alignment.py in
    particular -- keep working unchanged).
    """
    num_frames = int(item["num_frames"])
    future_targets = list(range(decision_frame + 1, num_frames))
    horizon_targets = list(range(decision_frame + 1, min(num_frames, decision_frame + 1 + 2 * PREDICT_FRAMES)))
    gt_scores = {
        frame_idx: frame_future_use(frame_idx, future_targets, overlap_map)["future_use_count"]
        for frame_idx in memory_before
    }
    score_ranks = rank_desc(scores)
    gt_ranks = rank_desc(gt_scores)

    rows = []
    for frame_idx in sorted(memory_before):
        future_use = frame_future_use(frame_idx, future_targets, overlap_map)
        horizon_use = frame_future_use(frame_idx, horizon_targets, overlap_map)
        detail = score_details.get(frame_idx) or {}
        row = {
            "row": item["_row"],
            "scene": item["scene"],
            "start_frame": item["start_frame"],
            "duration_sec": item["duration_sec"],
            "policy": policy,
            "budget": budget,
            "section_idx": section_idx,
            "decision_frame": decision_frame,
            "decision_global_frame": int(item["start_frame"]) + decision_frame,
            "frame_idx": frame_idx,
            "global_frame_idx": int(item["start_frame"]) + frame_idx,
            "age": decision_frame - frame_idx,
            "is_new": frame_idx in new_frames,
            "is_protected": frame_idx in protected_frames,
            "is_pinned": frame_idx in pinned_frames,
            "kept_after": frame_idx in memory_after,
            "evicted": frame_idx in evicted_frames,
            score_field: scores.get(frame_idx),
            rank_field: score_ranks.get(frame_idx),
            "gt_future_rank": gt_ranks.get(frame_idx),
            "gt_future_use_count": future_use["future_use_count"],
            "gt_future_use_fraction": future_use["future_use_fraction"],
            "gt_next_use_frame": future_use["next_use_frame"],
            "gt_next_use_distance": future_use["next_use_distance"],
            "gt_last_use_frame": future_use["last_use_frame"],
            "gt_horizon_use_count": horizon_use["future_use_count"],
            "gt_horizon_use_fraction": horizon_use["future_use_fraction"],
        }
        for column, detail_key in detail_fields.items():
            row[column] = detail.get(detail_key)
        rows.append(row)
    return rows


def make_ri_score_rows(
    item,
    section_idx,
    decision_frame,
    budget,
    memory_before,
    new_frames,
    evicted_frames,
    memory_after,
    protected_frames,
    pinned_frames,
    ri_scores,
    ri_score_details,
    overlap_map,
):
    return make_score_rows(
        item=item,
        policy="ri",
        section_idx=section_idx,
        decision_frame=decision_frame,
        budget=budget,
        memory_before=memory_before,
        new_frames=new_frames,
        evicted_frames=evicted_frames,
        memory_after=memory_after,
        protected_frames=protected_frames,
        pinned_frames=pinned_frames,
        scores=ri_scores,
        score_details=ri_score_details,
        overlap_map=overlap_map,
        score_field="ri_score",
        rank_field="ri_rank",
        detail_fields={
            "ri_rarity": "rarity",
            "ri_irreplaceability": "irreplaceability",
            "ri_cluster_id": "cluster_id",
            "ri_cluster_size": "cluster_size",
            "ri_dino_cluster_threshold": "cluster_threshold",
            "ri_rgb_nearest_frame": "rgb_nearest_frame",
            "ri_rgb_nearest_distance": "rgb_nearest_distance",
        },
    )


def make_slam_score_rows(
    item,
    section_idx,
    decision_frame,
    budget,
    memory_before,
    new_frames,
    evicted_frames,
    memory_after,
    protected_frames,
    pinned_frames,
    slam_scores,
    slam_score_details,
    overlap_map,
):
    return make_score_rows(
        item=item,
        policy="slam",
        section_idx=section_idx,
        decision_frame=decision_frame,
        budget=budget,
        memory_before=memory_before,
        new_frames=new_frames,
        evicted_frames=evicted_frames,
        memory_after=memory_after,
        protected_frames=protected_frames,
        pinned_frames=pinned_frames,
        scores=slam_scores,
        score_details=slam_score_details,
        overlap_map=overlap_map,
        score_field="slam_score",
        rank_field="slam_rank",
        detail_fields={
            "slam_redundancy_ratio": "redundancy_ratio",
            "slam_covisible_observers": "covisible_observers",
            "slam_max_covisibility": "max_covisibility",
            "slam_marginal_contribution": "marginal_contribution",
            "slam_unique_bonus": "unique_bonus",
        },
    )


def make_slam_ri_blend_score_rows(
    item,
    section_idx,
    decision_frame,
    budget,
    memory_before,
    new_frames,
    evicted_frames,
    memory_after,
    protected_frames,
    pinned_frames,
    blend_scores,
    blend_score_details,
    overlap_map,
):
    return make_score_rows(
        item=item,
        policy="slam_ri_blend",
        section_idx=section_idx,
        decision_frame=decision_frame,
        budget=budget,
        memory_before=memory_before,
        new_frames=new_frames,
        evicted_frames=evicted_frames,
        memory_after=memory_after,
        protected_frames=protected_frames,
        pinned_frames=pinned_frames,
        scores=blend_scores,
        score_details=blend_score_details,
        overlap_map=overlap_map,
        score_field="blend_score",
        rank_field="blend_rank",
        detail_fields={
            "blend_beta": "slamri_beta",
            "blend_slam_raw": "slamri_slam_raw",
            "blend_slam_norm": "slamri_slam_norm",
            "blend_ri_raw": "slamri_ri_raw",
            "blend_ri_norm": "slamri_ri_norm",
            "blend_ri_rarity": "slamri_ri_rarity",
            "blend_ri_irreplaceability": "slamri_ri_irreplaceability",
            "blend_slam_redundancy_ratio": "slamri_slam_redundancy_ratio",
            "blend_slam_unique_bonus": "slamri_slam_unique_bonus",
        },
    )


def sum_metrics(metric_rows):
    keys = [
        "targets",
        "possible_targets",
        "covered_targets",
        "retained_useful",
        "available_useful",
        "retained_memory_size",
    ]
    summary = {
        key: sum(row.get(key, 0) for row in metric_rows) for key in keys
    }
    targets = summary["targets"]
    possible_targets = summary["possible_targets"]
    available_useful = summary["available_useful"]
    retained_memory_size = summary["retained_memory_size"]
    summary["coverage"] = summary["covered_targets"] / targets if targets else 0.0
    summary["possible_coverage"] = (
        summary["covered_targets"] / possible_targets if possible_targets else 0.0
    )
    summary["oracle_recall"] = (
        summary["retained_useful"] / available_useful if available_useful else 0.0
    )
    summary["best_possible_coverage"] = (
        possible_targets / targets if targets else 0.0
    )
    summary["coverage_efficiency"] = (
        summary["covered_targets"] / retained_memory_size if retained_memory_size else 0.0
    )
    return summary


def simulate_row(
    item,
    policy,
    budget,
    overlap_map,
    dino_features=None,
    rgb_features=None,
    c2ws=None,
    ri_rarity_neighbors=3,
    slamri_beta=0.5,
    slamri_rarity_neighbors=3,
):
    total_frames = int(item["num_frames"])
    total_sections = (total_frames - 1) // PREDICT_FRAMES
    if policy != "unbounded" and budget is None:
        raise ValueError(f"{policy} requires budget")

    memory = set()
    pinned_frames = {0} if policy in SCORED_POLICIES else set()
    memory_buffer = None
    if policy in SCORED_POLICIES:
        if budget < 2:
            raise ValueError(f"{policy} requires budget >= 2")
        internal_policy = {
            "ri": "rarity_irreplaceability",
            "slam": "slam_covisibility",
            "slam_ri_blend": "slam_ri_blend",
        }[policy]
        memory_buffer = FrameMemoryBuffer(
            policy=internal_policy,
            budget=budget,
            pinned_frames=pinned_frames,
        )
        if policy in {"slam", "slam_ri_blend"} and c2ws is None:
            raise ValueError(f"{policy} requires c2ws poses")

    section_metrics = []
    trace_rows = []
    score_rows = []

    for section_idx in range(total_sections):
        section_start, anchor_range, predict_range = section_ranges(section_idx)
        section_end = min(total_frames - 1, section_start + PREDICT_FRAMES)

        if section_idx > 0:
            exclude_frames = set(anchor_range) | set(predict_range)
            generated_until = section_start
            metrics = evaluate_memory_for_section(
                memory=memory,
                predict_range=predict_range,
                overlap_map=overlap_map,
                generated_until=generated_until,
                exclude_frames=exclude_frames,
            )
            metrics["section_idx"] = section_idx
            section_metrics.append(metrics)
            trace_rows.append(
                {
                    "row": item["_row"],
                    "scene": item["scene"],
                    "start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "policy": policy,
                    "budget": budget,
                    "section_idx": section_idx,
                    "memory_size": len(memory),
                    **metrics,
                }
            )

        new_frames = set(range(section_start, section_end + 1))
        protected_frames = {section_end}
        if policy == "unbounded":
            memory.update(new_frames)
            evicted = []
        elif policy == "fifo":
            memory.update(new_frames)
            evicted = fifo_evict(memory, budget, protected_frames=protected_frames)
        elif policy == "belady":
            memory.update(new_frames)
            future_targets = list(range(section_end + 1, total_frames))
            useful_by_target = {
                target: available_useful_frames(
                    target_frame=target,
                    overlap_map=overlap_map,
                    generated_until=section_end,
                    exclude_frames=set(),
                )
                for target in future_targets
            }
            evicted = belady_evict(
                memory=memory,
                budget=budget,
                protected_frames=protected_frames,
                future_targets=future_targets,
                useful_by_target=useful_by_target,
            )
        elif policy == "coverage_oracle":
            memory.update(new_frames)
            future_targets = list(range(section_end + 1, total_frames))
            useful_by_target = {
                target: available_useful_frames(
                    target_frame=target,
                    overlap_map=overlap_map,
                    generated_until=section_end,
                    exclude_frames=set(),
                )
                for target in future_targets
            }
            evicted = coverage_oracle_evict(
                memory=memory,
                budget=budget,
                protected_frames=protected_frames,
                future_targets=future_targets,
                useful_by_target=useful_by_target,
            )
        elif policy == "ri":
            for frame_idx in new_frames:
                memory_buffer.add(frame_idx, evict=False)
            scores, score_details = compute_rarity_irreplaceability_scores(
                memory_frame_indices=memory_buffer.candidates(),
                pinned_frames=pinned_frames,
                rarity_neighbors=ri_rarity_neighbors,
                dino_features=dino_features,
                rgb_features=rgb_features,
                return_details=True,
            )
            memory_buffer.set_scores(scores)
            memory_before_eviction = set(memory_buffer.candidates())
            evicted = memory_buffer.evict_to_budget(protected_frames=protected_frames)
            memory = set(memory_buffer.candidates())
            score_rows.extend(
                make_ri_score_rows(
                    item=item,
                    section_idx=section_idx,
                    decision_frame=section_end,
                    budget=budget,
                    memory_before=memory_before_eviction,
                    new_frames=new_frames,
                    evicted_frames=set(evicted),
                    memory_after=memory,
                    protected_frames=protected_frames,
                    pinned_frames=pinned_frames,
                    ri_scores=scores,
                    ri_score_details=score_details,
                    overlap_map=overlap_map,
                )
            )
        elif policy == "slam":
            for frame_idx in new_frames:
                memory_buffer.add(frame_idx, evict=False)
            scores, score_details = compute_slam_covisibility_scores(
                memory_frame_indices=memory_buffer.candidates(),
                c2ws=c2ws,
                pinned_frames=pinned_frames,
                dino_features=dino_features,
                rgb_features=rgb_features,
                return_details=True,
            )
            memory_buffer.set_scores(scores)
            memory_before_eviction = set(memory_buffer.candidates())
            evicted = memory_buffer.evict_to_budget(protected_frames=protected_frames)
            memory = set(memory_buffer.candidates())
            score_rows.extend(
                make_slam_score_rows(
                    item=item,
                    section_idx=section_idx,
                    decision_frame=section_end,
                    budget=budget,
                    memory_before=memory_before_eviction,
                    new_frames=new_frames,
                    evicted_frames=set(evicted),
                    memory_after=memory,
                    protected_frames=protected_frames,
                    pinned_frames=pinned_frames,
                    slam_scores=scores,
                    slam_score_details=score_details,
                    overlap_map=overlap_map,
                )
            )
        elif policy == "slam_ri_blend":
            for frame_idx in new_frames:
                memory_buffer.add(frame_idx, evict=False)
            scores, score_details = compute_slam_ri_blend_scores(
                memory_frame_indices=memory_buffer.candidates(),
                c2ws=c2ws,
                forced_keep_frames=pinned_frames,
                dino_features=dino_features,
                rgb_features=rgb_features,
                beta=slamri_beta,
                ri_kwargs={"rarity_neighbors": slamri_rarity_neighbors},
                return_details=True,
            )
            memory_buffer.set_scores(scores)
            memory_before_eviction = set(memory_buffer.candidates())
            evicted = memory_buffer.evict_to_budget(protected_frames=protected_frames)
            memory = set(memory_buffer.candidates())
            score_rows.extend(
                make_slam_ri_blend_score_rows(
                    item=item,
                    section_idx=section_idx,
                    decision_frame=section_end,
                    budget=budget,
                    memory_before=memory_before_eviction,
                    new_frames=new_frames,
                    evicted_frames=set(evicted),
                    memory_after=memory,
                    protected_frames=protected_frames,
                    pinned_frames=pinned_frames,
                    blend_scores=scores,
                    blend_score_details=score_details,
                    overlap_map=overlap_map,
                )
            )
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        if policy not in SCORED_POLICIES:
            memory.difference_update(evicted)

    summary = sum_metrics(section_metrics)
    summary.update(
        {
            "row": item["_row"],
            "scene": item["scene"],
            "start_frame": item["start_frame"],
            "duration_sec": item["duration_sec"],
            "policy": policy,
            "budget": budget if budget is not None else "unbounded",
            "sections_evaluated": len(section_metrics),
            "final_memory_size": len(memory),
        }
    )
    return summary, trace_rows, score_rows


def load_visual_feature_maps(item, dataset_root, feature_extractor):
    gt_frames_dir = resolve_gt_frames_dir(item, dataset_root)
    start_frame = int(item["start_frame"])
    num_frames = int(item["num_frames"])
    dino_features = {}
    rgb_features = {}

    for batch_start in range(0, num_frames, feature_extractor.batch_size):
        frame_indices = list(
            range(batch_start, min(num_frames, batch_start + feature_extractor.batch_size))
        )
        images = []
        for frame_idx in frame_indices:
            path = gt_frames_dir / f"{start_frame + frame_idx:04d}.png"
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())

        dino_batch, rgb_batch = feature_extractor.encode_pil_images(images)
        for batch_idx, frame_idx in enumerate(frame_indices):
            dino_features[frame_idx] = dino_batch[batch_idx]
            rgb_features[frame_idx] = rgb_batch[batch_idx]

    return dino_features, rgb_features


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["policy"], row["budget"], row["duration_sec"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (policy, budget, duration_sec), group in sorted(grouped.items(), key=lambda x: str(x[0])):
        totals = sum_metrics(group)
        aggregates.append(
            {
                "policy": policy,
                "budget": budget,
                "duration_sec": duration_sec,
                "videos": len(group),
                "sections_evaluated": sum(row["sections_evaluated"] for row in group),
                **totals,
            }
        )
    return aggregates


def main():
    parser = argparse.ArgumentParser(
        description="Offline memory-policy analysis using Context-as-Memory overlap labels."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("/data/ab575577/MemCam/analysis/context_memory"))
    parser.add_argument("--durations", type=str, default="10")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budgets", type=str, default="32")
    parser.add_argument(
        "--policies",
        type=str,
        default="unbounded,fifo,ri,belady,coverage_oracle",
        help=(
            "Comma-separated: unbounded, fifo, belady (oracle), coverage_oracle "
            "(oracle), ri, slam, slam_ri_blend. The last three reuse the real "
            "rarity_irreplaceability/slam_covisibility/slam_ri_blend scoring "
            "functions and additionally write a *_frame_scores.jsonl with each "
            "score alongside true GT future-need (for summarize_ri_alignment.py)."
        ),
    )
    parser.add_argument("--ri_dino_model", type=str, default="facebook/dinov2-base")
    parser.add_argument(
        "--ri_feature_device",
        type=str,
        default="cuda",
        help="Device for ri/slam/slam_ri_blend's DINO+RGB feature extraction. "
        "This whole script is otherwise CPU-only; pass 'cpu' here too to run "
        "without a GPU allocation (slower, still tractable at analysis scale).",
    )
    parser.add_argument("--ri_feature_batch_size", type=int, default=16)
    parser.add_argument("--ri_rgb_image_size", type=int, default=64)
    parser.add_argument(
        "--ri_rarity_neighbors",
        type=int,
        default=3,
        help="Explicit rarity-clustering k for the RI policy.",
    )
    parser.add_argument(
        "--slamri_beta",
        type=float,
        default=0.5,
        help="slam_ri_blend mix weight, same meaning as the real pipeline's "
        "--slamri_beta: beta*norm(SLAM) + (1-beta)*norm(RI).",
    )
    parser.add_argument(
        "--slamri_rarity_neighbors",
        type=int,
        default=3,
        help="Explicit rarity-clustering k for the RI half of slam_ri_blend.",
    )
    args = parser.parse_args()

    if args.ri_rarity_neighbors < 1 or args.slamri_rarity_neighbors < 1:
        raise ValueError("RI rarity-neighbor counts must be at least 1")

    items = load_manifest(args.manifest)
    selected = select_rows(
        items=items,
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not selected:
        raise RuntimeError("No manifest rows selected.")

    policies = parse_str_list(args.policies)
    budgets = parse_int_list(args.budgets) or []
    needs_visual_features = bool(set(SCORED_POLICIES) & set(policies))
    needs_poses = bool({"slam", "slam_ri_blend"} & set(policies))
    feature_extractor = None
    if needs_visual_features:
        feature_extractor = VisualMemoryFeatureExtractor(
            dino_model_name=args.ri_dino_model,
            device=args.ri_feature_device,
            batch_size=args.ri_feature_batch_size,
            rgb_image_size=args.ri_rgb_image_size,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    trace_rows = []
    frame_usefulness_rows = []
    score_rows = []
    for item in selected:
        print(
            f"[analysis row {item['_row']}] {item['scene']} "
            f"start={item['start_frame']} duration={item['duration_sec']}s"
        )
        overlap_map = load_overlap_map(
            overlap_dir=resolve_overlap_dir(item, args.dataset_root),
            start_frame=int(item["start_frame"]),
            num_frames=int(item["num_frames"]),
        )
        frame_usefulness_rows.extend(compute_frame_usefulness_rows(item, overlap_map))
        dino_features = None
        rgb_features = None
        if needs_visual_features:
            dino_features, rgb_features = load_visual_feature_maps(
                item=item,
                dataset_root=args.dataset_root,
                feature_extractor=feature_extractor,
            )
        c2ws = None
        if needs_poses:
            from dataset.poses import load_c2ws_from_json  # noqa: E402 -- only needed for slam/slam_ri_blend

            c2ws = load_c2ws_from_json(
                json_path=resolve_pose_path(item, args.dataset_root),
                start_frame=int(item["start_frame"]),
                num_frames=int(item["num_frames"]),
            )

        for policy in policies:
            policy_budgets = [None] if policy == "unbounded" else budgets
            for budget in policy_budgets:
                summary, traces, scores = simulate_row(
                    item=item,
                    policy=policy,
                    budget=budget,
                    overlap_map=overlap_map,
                    dino_features=dino_features,
                    rgb_features=rgb_features,
                    c2ws=c2ws,
                    ri_rarity_neighbors=args.ri_rarity_neighbors,
                    slamri_beta=args.slamri_beta,
                    slamri_rarity_neighbors=args.slamri_rarity_neighbors,
                )
                summary_rows.append(summary)
                trace_rows.extend(traces)
                score_rows.extend(scores)
                print(
                    f"  {policy} b={summary['budget']} "
                    f"coverage={summary['coverage']:.4f} "
                    f"possible={summary['possible_coverage']:.4f} "
                    f"recall={summary['oracle_recall']:.4f}"
                )

    aggregate_summary_rows = aggregate_rows(summary_rows)
    write_csv(args.output_dir / "policy_summary.csv", summary_rows)
    write_csv(args.output_dir / "policy_aggregate.csv", aggregate_summary_rows)
    write_csv(args.output_dir / "frame_usefulness.csv", frame_usefulness_rows)
    write_jsonl(args.output_dir / "policy_traces.jsonl", trace_rows)

    with (args.output_dir / "policy_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary_rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote: {args.output_dir / 'policy_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_aggregate.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_aggregate.json'}")
    print(f"Wrote: {args.output_dir / 'frame_usefulness.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_traces.jsonl'}")

    # One file per scored policy (not a single merged file) -- each has its
    # own score column name (ri_score / slam_score / blend_score), and
    # summarize_ri_alignment.py's documented default path expects
    # ri_frame_scores.jsonl to contain only ri rows.
    score_filenames = {
        "ri": "ri_frame_scores.jsonl",
        "slam": "slam_frame_scores.jsonl",
        "slam_ri_blend": "slam_ri_blend_frame_scores.jsonl",
    }
    for policy_name, filename in score_filenames.items():
        policy_score_rows = [row for row in score_rows if row["policy"] == policy_name]
        if not policy_score_rows:
            continue
        write_jsonl(args.output_dir / filename, policy_score_rows)
        print(f"Wrote: {args.output_dir / filename}")


if __name__ == "__main__":
    main()
