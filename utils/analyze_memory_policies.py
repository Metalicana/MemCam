import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

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
    num_frames = int(item["num_frames"])
    future_targets = list(range(decision_frame + 1, num_frames))
    horizon_targets = list(range(decision_frame + 1, min(num_frames, decision_frame + 1 + 2 * PREDICT_FRAMES)))
    gt_scores = {
        frame_idx: frame_future_use(frame_idx, future_targets, overlap_map)["future_use_count"]
        for frame_idx in memory_before
    }
    ri_ranks = rank_desc(ri_scores)
    gt_ranks = rank_desc(gt_scores)

    rows = []
    for frame_idx in sorted(memory_before):
        future_use = frame_future_use(frame_idx, future_targets, overlap_map)
        horizon_use = frame_future_use(frame_idx, horizon_targets, overlap_map)
        ri_score = ri_scores.get(frame_idx)
        rows.append(
            {
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "policy": "ri",
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
                "ri_score": ri_score,
                "ri_rarity": (ri_score_details.get(frame_idx) or {}).get("rarity"),
                "ri_irreplaceability": (
                    ri_score_details.get(frame_idx) or {}
                ).get("irreplaceability"),
                "ri_cluster_id": (ri_score_details.get(frame_idx) or {}).get("cluster_id"),
                "ri_cluster_size": (ri_score_details.get(frame_idx) or {}).get("cluster_size"),
                "ri_dino_cluster_threshold": (
                    ri_score_details.get(frame_idx) or {}
                ).get("cluster_threshold"),
                "ri_rgb_nearest_frame": (
                    ri_score_details.get(frame_idx) or {}
                ).get("rgb_nearest_frame"),
                "ri_rgb_nearest_distance": (
                    ri_score_details.get(frame_idx) or {}
                ).get("rgb_nearest_distance"),
                "ri_rank": ri_ranks.get(frame_idx),
                "gt_future_rank": gt_ranks.get(frame_idx),
                "gt_future_use_count": future_use["future_use_count"],
                "gt_future_use_fraction": future_use["future_use_fraction"],
                "gt_next_use_frame": future_use["next_use_frame"],
                "gt_next_use_distance": future_use["next_use_distance"],
                "gt_last_use_frame": future_use["last_use_frame"],
                "gt_horizon_use_count": horizon_use["future_use_count"],
                "gt_horizon_use_fraction": horizon_use["future_use_fraction"],
            }
        )
    return rows


def sum_metrics(metric_rows):
    keys = [
        "targets",
        "possible_targets",
        "covered_targets",
        "retained_useful",
        "available_useful",
    ]
    summary = {key: sum(row[key] for row in metric_rows) for key in keys}
    targets = summary["targets"]
    possible_targets = summary["possible_targets"]
    available_useful = summary["available_useful"]
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
    return summary


def simulate_row(item, policy, budget, overlap_map, dino_features=None, rgb_features=None):
    total_frames = int(item["num_frames"])
    total_sections = (total_frames - 1) // PREDICT_FRAMES
    if policy != "unbounded" and budget is None:
        raise ValueError(f"{policy} requires budget")

    memory = set()
    pinned_frames = {0} if policy == "ri" else set()
    memory_buffer = None
    if policy == "ri":
        if budget < 2:
            raise ValueError("ri requires budget >= 2")
        memory_buffer = FrameMemoryBuffer(
            policy="rarity_irreplaceability",
            budget=budget,
            pinned_frames=pinned_frames,
        )

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
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        if policy != "ri":
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
    parser.add_argument("--policies", type=str, default="unbounded,fifo,ri,belady,coverage_oracle")
    parser.add_argument("--ri_dino_model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--ri_feature_device", type=str, default="cuda")
    parser.add_argument("--ri_feature_batch_size", type=int, default=16)
    parser.add_argument("--ri_rgb_image_size", type=int, default=64)
    args = parser.parse_args()

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
    feature_extractor = None
    if "ri" in policies:
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
        if "ri" in policies:
            dino_features, rgb_features = load_visual_feature_maps(
                item=item,
                dataset_root=args.dataset_root,
                feature_extractor=feature_extractor,
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
    write_jsonl(args.output_dir / "ri_frame_scores.jsonl", score_rows)

    with (args.output_dir / "policy_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary_rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote: {args.output_dir / 'policy_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_aggregate.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_aggregate.json'}")
    print(f"Wrote: {args.output_dir / 'frame_usefulness.csv'}")
    print(f"Wrote: {args.output_dir / 'policy_traces.jsonl'}")
    print(f"Wrote: {args.output_dir / 'ri_frame_scores.jsonl'}")


if __name__ == "__main__":
    main()
