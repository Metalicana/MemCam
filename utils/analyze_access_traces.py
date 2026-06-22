import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


AGE_BINS = [
    ("0_75", 0, 75),
    ("76_151", 76, 151),
    ("152_303", 152, 303),
    ("304_plus", 304, None),
]

NUMERIC_EVICTION_FIELDS = [
    "eviction_score",
    "eviction_rarity",
    "eviction_irreplaceability",
    "eviction_rgb_nearest_distance",
    "eviction_redundancy_ratio",
    "eviction_covisible_observers",
    "eviction_max_covisibility",
    "eviction_marginal_contribution",
    "eviction_unique_bonus",
    "eviction_coreset_rank",
    "eviction_coreset_marginal_gain",
    "eviction_coreset_candidate_gain",
    "eviction_coreset_removal_loss",
    "eviction_coreset_archive_size",
    "eviction_coreset_facility_value",
    "eviction_coreset_quality",
    "eviction_coreset_similarity_mean",
    "eviction_coreset_similarity_max",
]


def parse_list(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def gini(values):
    values = sorted(value for value in values if value is not None)
    total = sum(values)
    if not values or total == 0:
        return None
    n = len(values)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def safe_round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def common_value(rows, key):
    counts = Counter(row.get(key) for row in rows if row.get(key) is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def age_bin(age):
    for name, low, high in AGE_BINS:
        if age >= low and (high is None or age <= high):
            return name
    return "unknown"


def entropy_stats(counts):
    total = sum(counts)
    if total <= 0:
        return {
            "selection_entropy": None,
            "selection_entropy_norm": None,
            "effective_selected_frames": 0.0,
            "top1_selected_frac": None,
        }

    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)

    return {
        "selection_entropy": safe_round(entropy),
        "selection_entropy_norm": safe_round(entropy / math.log(total)) if total > 1 else 0.0,
        "effective_selected_frames": safe_round(math.exp(entropy)),
        "top1_selected_frac": safe_round(max(counts) / total),
    }


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p10": safe_round(percentile(values, 0.1)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
        f"{prefix}_min": safe_round(min(values)) if values else None,
        f"{prefix}_max": safe_round(max(values)) if values else None,
    }


def discover_trace_dirs(root, runs, trace_dir_name, strict):
    if root.name == trace_dir_name and root.exists():
        run_name = root.parent.name
        return [(run_name, root)]

    if runs is None:
        run_names = [
            path.name
            for path in sorted(root.iterdir())
            if path.is_dir() and (path / trace_dir_name).exists()
        ]
    else:
        run_names = runs

    trace_dirs = []
    missing = []
    for run_name in run_names:
        trace_dir = root / run_name / trace_dir_name
        if trace_dir.exists():
            trace_dirs.append((run_name, trace_dir))
        else:
            missing.append((run_name, trace_dir))

    if missing:
        message = "\n".join(f"{run}: missing {path}" for run, path in missing)
        if strict:
            raise FileNotFoundError(message)
        print(message)

    return trace_dirs


def iter_trace_rows(run_name, trace_dir):
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_run_name"] = run_name
                row["_trace_file"] = path.name
                row["_line_number"] = line_number
                yield row


def context_rows(rows, post_initial=False):
    output = [row for row in rows if row.get("event") == "context_access"]
    if post_initial:
        output = [
            row
            for row in output
            if row.get("section_idx") is not None and int(row["section_idx"]) > 0
        ]
    return output


def selected_rows(rows, post_initial=False):
    return [row for row in context_rows(rows, post_initial=post_initial) if row.get("selected")]


def summarize_context_scope(rows, prefix=""):
    access = context_rows(rows, post_initial=False)
    retrieval = context_rows(rows, post_initial=True)
    selected = selected_rows(rows, post_initial=True)
    fallback = [row for row in retrieval if not row.get("selected")]

    ages = [to_float(row.get("memory_age")) for row in selected]
    overlaps = [to_float(row.get("selected_overlap")) for row in selected]
    candidates = [to_float(row.get("candidate_count")) for row in retrieval]
    stored = [to_float(row.get("stored_memory_size")) for row in retrieval]
    selected_frame_counts = Counter(
        (row.get("_trace_file"), row.get("selected_memory_frame")) for row in selected
    )
    reuse_counts = list(selected_frame_counts.values())
    bin_counts = Counter(age_bin(int(age)) for age in ages if age is not None)

    result = {
        f"{prefix}context_queries_total": len(access),
        f"{prefix}retrieval_queries": len(retrieval),
        f"{prefix}selected_queries": len(selected),
        f"{prefix}fallback_queries": len(fallback),
        f"{prefix}fallback_rate": safe_round(len(fallback) / len(retrieval)) if retrieval else 0.0,
        f"{prefix}unique_selected_frames": len(selected_frame_counts),
        f"{prefix}max_reuse_count": max(reuse_counts) if reuse_counts else 0,
        f"{prefix}mean_reuse_count": safe_round(mean(reuse_counts)),
        f"{prefix}reuse_gini": safe_round(gini(reuse_counts)),
    }
    result.update(numeric_stats(f"{prefix}age", ages))
    result.update(numeric_stats(f"{prefix}overlap", overlaps))
    result.update(numeric_stats(f"{prefix}candidate_count", candidates))
    result.update(numeric_stats(f"{prefix}stored_memory_size", stored))
    for name, _, _ in AGE_BINS:
        count = bin_counts[name]
        result[f"{prefix}age_bin_{name}"] = count
        result[f"{prefix}age_bin_{name}_frac"] = (
            safe_round(count / len(ages)) if ages else 0.0
        )
    return result


def section_summary_rows(rows):
    grouped = defaultdict(list)
    for row in context_rows(rows, post_initial=False):
        key = (
            row.get("_run_name"),
            row.get("_trace_file"),
            row.get("row"),
            row.get("scene"),
            row.get("section_idx"),
        )
        grouped[key].append(row)

    output = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        run_name, trace_file, row_id, scene, section_idx = key
        selected = [row for row in group if row.get("selected")]
        counts = Counter(row.get("selected_memory_frame") for row in selected)
        count_values = list(counts.values())
        entropy = entropy_stats(count_values)
        ages = [to_float(row.get("memory_age")) for row in selected]
        overlaps = [to_float(row.get("selected_overlap")) for row in selected]
        candidates = [to_float(row.get("candidate_count")) for row in group]
        stored = [to_float(row.get("stored_memory_size")) for row in group]
        output.append(
            {
                "run_name": run_name,
                "trace_file": trace_file,
                "row": row_id,
                "scene": scene,
                "section_idx": section_idx,
                "memory_policy": common_value(group, "memory_policy"),
                "memory_budget": common_value(group, "memory_budget"),
                "duration_sec": common_value(group, "duration_sec"),
                "queries": len(group),
                "selected_queries": len(selected),
                "fallback_queries": len(group) - len(selected),
                "unique_selected_frames": len(counts),
                "max_reuse_count": max(count_values) if count_values else 0,
                **entropy,
                **numeric_stats("age", ages),
                **numeric_stats("overlap", overlaps),
                **numeric_stats("candidate_count", candidates),
                **numeric_stats("stored_memory_size", stored),
            }
        )
    return output


def video_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("_run_name"), row.get("_trace_file"))].append(row)

    output = []
    for (run_name, trace_file), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        selected = selected_rows(group, post_initial=True)
        counts = Counter(row.get("selected_memory_frame") for row in selected)
        output.append(
            {
                "run_name": run_name,
                "trace_file": trace_file,
                "row": common_value(group, "row"),
                "scene": common_value(group, "scene"),
                "memory_policy": common_value(group, "memory_policy"),
                "memory_budget": common_value(group, "memory_budget"),
                "duration_sec": common_value(group, "duration_sec"),
                "sections": len(
                    {
                        row.get("section_idx")
                        for row in context_rows(group, post_initial=False)
                        if row.get("section_idx") is not None
                    }
                ),
                **summarize_context_scope(group),
                **entropy_stats(list(counts.values())),
            }
        )
    return output


def run_summary_rows(rows, section_rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("_run_name")].append(row)

    sections_by_run = defaultdict(list)
    for row in section_rows:
        sections_by_run[row["run_name"]].append(row)

    output = []
    for run_name, group in sorted(grouped.items()):
        section_group = sections_by_run[run_name]
        retrieval_section_group = [
            row
            for row in section_group
            if row.get("section_idx") is not None and int(row["section_idx"]) > 0
        ]
        output.append(
            {
                "run_name": run_name,
                "trace_files": len({row.get("_trace_file") for row in group}),
                "memory_policy": common_value(group, "memory_policy"),
                "memory_budget": common_value(group, "memory_budget"),
                "duration_sec": common_value(group, "duration_sec"),
                **summarize_context_scope(group),
                "section_effective_selected_frames_mean": safe_round(
                    mean(row.get("effective_selected_frames") for row in retrieval_section_group)
                ),
                "section_effective_selected_frames_median": safe_round(
                    percentile(
                        [
                            row.get("effective_selected_frames")
                            for row in retrieval_section_group
                        ],
                        0.5,
                    )
                ),
                "section_entropy_norm_mean": safe_round(
                    mean(row.get("selection_entropy_norm") for row in retrieval_section_group)
                ),
                "section_top1_selected_frac_mean": safe_round(
                    mean(row.get("top1_selected_frac") for row in retrieval_section_group)
                ),
            }
        )
    return output


def top_selected_frame_rows(rows, top_k):
    grouped = defaultdict(list)
    for row in selected_rows(rows, post_initial=True):
        key = (
            row.get("_run_name"),
            row.get("_trace_file"),
            row.get("row"),
            row.get("scene"),
            row.get("selected_memory_frame"),
        )
        grouped[key].append(row)

    totals_by_video = Counter()
    for row in selected_rows(rows, post_initial=True):
        totals_by_video[(row.get("_run_name"), row.get("_trace_file"))] += 1

    output = []
    for key, group in grouped.items():
        run_name, trace_file, row_id, scene, frame_idx = key
        total = totals_by_video[(run_name, trace_file)]
        output.append(
            {
                "run_name": run_name,
                "trace_file": trace_file,
                "row": row_id,
                "scene": scene,
                "memory_policy": common_value(group, "memory_policy"),
                "memory_budget": common_value(group, "memory_budget"),
                "duration_sec": common_value(group, "duration_sec"),
                "selected_memory_frame": frame_idx,
                "selected_dataset_frame": common_value(group, "selected_dataset_frame"),
                "access_count": len(group),
                "access_frac_in_video": safe_round(len(group) / total) if total else None,
                **numeric_stats(
                    "overlap",
                    [to_float(row.get("selected_overlap")) for row in group],
                ),
                **numeric_stats("age", [to_float(row.get("memory_age")) for row in group]),
            }
        )

    output.sort(key=lambda row: (row["run_name"], row["trace_file"], -row["access_count"]))
    if top_k is None:
        return output

    limited = []
    counts = Counter()
    for row in output:
        key = (row["run_name"], row["trace_file"])
        if counts[key] >= top_k:
            continue
        limited.append(row)
        counts[key] += 1
    return limited


def eviction_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("event") == "memory_eviction":
            grouped[row.get("_run_name")].append(row)

    output = []
    for run_name, group in sorted(grouped.items()):
        ages = [to_float(row.get("memory_age_at_eviction")) for row in group]
        recent = [age for age in ages if age is not None and age <= 76]
        row = {
            "run_name": run_name,
            "trace_files": len({item.get("_trace_file") for item in group}),
            "memory_policy": common_value(group, "memory_policy"),
            "memory_budget": common_value(group, "memory_budget"),
            "duration_sec": common_value(group, "duration_sec"),
            "evictions": len(group),
            "recent_eviction_frac_age_le_76": safe_round(len(recent) / len(ages)) if ages else None,
            **numeric_stats("evicted_age", ages),
            **numeric_stats(
                "stored_memory_size_after_eviction",
                [to_float(item.get("stored_memory_size")) for item in group],
            ),
        }
        for field in NUMERIC_EVICTION_FIELDS:
            values = [to_float(item.get(field)) for item in group]
            if any(value is not None for value in values):
                row.update(numeric_stats(field, values))
        output.append(row)
    return output


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MemCam JSONL access traces across generated runs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--trace_dir_name", type=str, default="access_traces")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k_selected", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    trace_dirs = discover_trace_dirs(
        root=args.root,
        runs=parse_list(args.runs),
        trace_dir_name=args.trace_dir_name,
        strict=args.strict,
    )
    if not trace_dirs:
        raise RuntimeError(f"No trace directories found under {args.root}")

    rows = []
    for run_name, trace_dir in trace_dirs:
        run_rows = list(iter_trace_rows(run_name, trace_dir))
        print(f"{run_name}: {len(run_rows)} trace rows from {trace_dir}")
        rows.extend(run_rows)

    if not rows:
        raise RuntimeError("No trace rows loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    section_rows = section_summary_rows(rows)
    video_rows = video_summary_rows(rows)
    run_rows = run_summary_rows(rows, section_rows)
    top_rows = top_selected_frame_rows(rows, top_k=args.top_k_selected)
    eviction_rows = eviction_summary_rows(rows)

    write_csv(args.output_dir / "access_run_summary.csv", run_rows)
    write_csv(args.output_dir / "access_video_summary.csv", video_rows)
    write_csv(args.output_dir / "access_section_summary.csv", section_rows)
    write_csv(args.output_dir / "access_top_selected_frames.csv", top_rows)
    write_csv(args.output_dir / "access_eviction_summary.csv", eviction_rows)
    write_json(args.output_dir / "access_run_summary.json", run_rows)

    print(json.dumps(run_rows, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.output_dir / 'access_run_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'access_video_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'access_section_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'access_top_selected_frames.csv'}")
    print(f"Wrote: {args.output_dir / 'access_eviction_summary.csv'}")


if __name__ == "__main__":
    main()
