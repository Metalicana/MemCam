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


def parse_list(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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


def to_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


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


def age_bin(age):
    for name, low, high in AGE_BINS:
        if age >= low and (high is None or age <= high):
            return name
    return "unknown"


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p10": safe_round(percentile(values, 0.1)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
        f"{prefix}_p95": safe_round(percentile(values, 0.95)),
        f"{prefix}_min": safe_round(min(values)) if values else None,
        f"{prefix}_max": safe_round(max(values)) if values else None,
    }


def common_value(rows, key):
    counts = Counter(row.get(key) for row in rows if row.get(key) is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def discover_trace_dirs(root, runs, trace_dir_name, strict):
    if not root.exists():
        raise FileNotFoundError(f"Trace root does not exist: {root}")

    if root.name == trace_dir_name and root.exists():
        return [(root.parent.name, root)]

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


def filter_rows(rows, durations=None, row_filter=None):
    if durations is not None:
        durations = set(int(value) for value in durations)
    if row_filter is not None:
        row_filter = set(int(value) for value in row_filter)

    output = []
    for row in rows:
        if durations is not None:
            duration = row.get("duration_sec")
            if duration is None or int(duration) not in durations:
                continue
        if row_filter is not None:
            row_id = row.get("row")
            if row_id is None or int(row_id) not in row_filter:
                continue
        output.append(row)
    return output


def is_context_access(row):
    return row.get("event") == "context_access"


def is_retrieval(row):
    section_idx = to_int(row.get("section_idx"))
    return is_context_access(row) and section_idx is not None and section_idx > 0


def is_selected(row):
    return is_retrieval(row) and bool(row.get("selected"))


def video_key(row):
    row_id = row.get("row")
    scene = row.get("scene")
    duration = row.get("duration_sec")
    dataset_start = row.get("dataset_start_frame")
    if row_id is not None and scene is not None and duration is not None:
        return (row_id, scene, duration, dataset_start)
    return (row.get("_trace_file"), scene, duration, dataset_start)


def target_key(row):
    return (
        *video_key(row),
        row.get("section_idx"),
        row.get("context_slot"),
        row.get("target_frame"),
    )


def frame_key(row, frame_field="selected_memory_frame"):
    return (*video_key(row), row.get(frame_field))


def build_baseline_index(rows, baseline_run):
    baseline = {}
    duplicate_targets = 0
    for row in rows:
        if row.get("_run_name") != baseline_run or not is_selected(row):
            continue
        key = target_key(row)
        if key in baseline:
            duplicate_targets += 1
            old_overlap = to_float(baseline[key].get("selected_overlap")) or -math.inf
            new_overlap = to_float(row.get("selected_overlap")) or -math.inf
            if new_overlap <= old_overlap:
                continue
        baseline[key] = row
    return baseline, duplicate_targets


def target_alignment_rows(rows, baseline_index, baseline_run, near_frame_window):
    output = []
    for row in rows:
        if row.get("_run_name") == baseline_run or not is_retrieval(row):
            continue

        baseline = baseline_index.get(target_key(row))
        policy_overlap = to_float(row.get("selected_overlap")) if row.get("selected") else None
        baseline_overlap = to_float(baseline.get("selected_overlap")) if baseline else None
        policy_frame = to_int(row.get("selected_memory_frame")) if row.get("selected") else None
        baseline_frame = to_int(baseline.get("selected_memory_frame")) if baseline else None
        policy_age = to_float(row.get("memory_age")) if row.get("selected") else None
        baseline_age = to_float(baseline.get("memory_age")) if baseline else None
        exact_match = (
            policy_frame is not None
            and baseline_frame is not None
            and policy_frame == baseline_frame
        )
        near_match = (
            policy_frame is not None
            and baseline_frame is not None
            and abs(policy_frame - baseline_frame) <= near_frame_window
        )
        overlap_gap = (
            baseline_overlap - policy_overlap
            if baseline_overlap is not None and policy_overlap is not None
            else None
        )
        if baseline_overlap is not None and policy_overlap is not None:
            if abs(baseline_overlap) <= 1e-12:
                overlap_capture_ratio = 1.0 if abs(policy_overlap) <= 1e-12 else None
            else:
                overlap_capture_ratio = policy_overlap / baseline_overlap
        else:
            overlap_capture_ratio = None

        output.append(
            {
                "run_name": row.get("_run_name"),
                "trace_file": row.get("_trace_file"),
                "row": row.get("row"),
                "scene": row.get("scene"),
                "duration_sec": row.get("duration_sec"),
                "dataset_start_frame": row.get("dataset_start_frame"),
                "section_idx": row.get("section_idx"),
                "context_slot": row.get("context_slot"),
                "target_frame": row.get("target_frame"),
                "target_dataset_frame": row.get("target_dataset_frame"),
                "memory_policy": row.get("memory_policy"),
                "memory_budget": row.get("memory_budget"),
                "selected": bool(row.get("selected")),
                "fallback_reason": row.get("fallback_reason"),
                "policy_selected_frame": policy_frame,
                "policy_selected_dataset_frame": row.get("selected_dataset_frame"),
                "policy_overlap": safe_round(policy_overlap),
                "policy_age": safe_round(policy_age),
                "baseline_run": baseline_run,
                "has_baseline": baseline is not None,
                "baseline_selected_frame": baseline_frame,
                "baseline_selected_dataset_frame": (
                    baseline.get("selected_dataset_frame") if baseline else None
                ),
                "baseline_overlap": safe_round(baseline_overlap),
                "baseline_age": safe_round(baseline_age),
                "exact_upperbound_match": exact_match,
                "near_upperbound_match": near_match,
                "near_frame_window": near_frame_window,
                "overlap_gap": safe_round(overlap_gap),
                "overlap_capture_ratio": safe_round(overlap_capture_ratio),
                "age_delta_vs_upperbound": safe_round(
                    policy_age - baseline_age
                    if policy_age is not None and baseline_age is not None
                    else None
                ),
            }
        )
    return output


def summarize_selected_frames(rows, run_name=None):
    selected = [row for row in rows if is_selected(row)]
    if run_name is not None:
        selected = [row for row in selected if row.get("_run_name") == run_name]

    totals_by_video = Counter((row.get("_run_name"), video_key(row)) for row in selected)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row.get("_run_name"), *frame_key(row))].append(row)

    output = []
    for key, group in grouped.items():
        run, row_id, scene, duration_sec, dataset_start_frame, frame_idx = key
        total = totals_by_video[(run, (row_id, scene, duration_sec, dataset_start_frame))]
        target_frames = [to_int(row.get("target_frame")) for row in group]
        sections = {row.get("section_idx") for row in group if row.get("section_idx") is not None}
        ages = [to_float(row.get("memory_age")) for row in group]
        overlaps = [to_float(row.get("selected_overlap")) for row in group]
        output.append(
            {
                "run_name": run,
                "row": row_id,
                "scene": scene,
                "duration_sec": duration_sec,
                "dataset_start_frame": dataset_start_frame,
                "selected_memory_frame": frame_idx,
                "selected_dataset_frame": (
                    int(dataset_start_frame) + int(frame_idx)
                    if dataset_start_frame is not None and frame_idx is not None
                    else None
                ),
                "access_count": len(group),
                "access_frac_in_video": safe_round(safe_div(len(group), total)),
                "sections_used": len(sections),
                "first_target_frame": min(target_frames) if target_frames else None,
                "last_target_frame": max(target_frames) if target_frames else None,
                **numeric_stats("overlap", overlaps),
                **numeric_stats("age", ages),
            }
        )

    output.sort(
        key=lambda row: (
            str(row["run_name"]),
            str(row["row"]),
            -int(row["access_count"]),
            int(row["selected_memory_frame"])
            if row["selected_memory_frame"] is not None
            else 10**9,
        )
    )
    return output


def upperbound_frame_rows(rows, baseline_run):
    rows = summarize_selected_frames(rows, run_name=baseline_run)
    for row in rows:
        row["upperbound_access_count"] = row.pop("access_count")
        row["upperbound_access_frac_in_video"] = row.pop("access_frac_in_video")
    return rows


def frame_alignment_rows(rows, baseline_index, baseline_run):
    if not baseline_index:
        return []

    policy_runs = sorted({row.get("_run_name") for row in rows if row.get("_run_name") != baseline_run})
    baseline_by_video_frame = defaultdict(set)
    for key, baseline in baseline_index.items():
        baseline_by_video_frame[frame_key(baseline)].add(key)

    policy_by_run_video_frame = defaultdict(set)
    policy_any_counts = Counter()
    for row in rows:
        if row.get("_run_name") == baseline_run or not is_selected(row):
            continue
        key = (row.get("_run_name"), frame_key(row))
        policy_by_run_video_frame[key].add(target_key(row))
        policy_any_counts[key] += 1

    baseline_frame_summary = {
        key: len(targets) for key, targets in baseline_by_video_frame.items()
    }
    output = []
    for policy_run in policy_runs:
        for base_frame_key, baseline_targets in baseline_by_video_frame.items():
            row_id, scene, duration_sec, dataset_start_frame, frame_idx = base_frame_key
            policy_key = (policy_run, base_frame_key)
            matched_targets = baseline_targets & policy_by_run_video_frame.get(policy_key, set())
            any_policy_access = policy_any_counts.get(policy_key, 0)
            upperbound_access_count = baseline_frame_summary[base_frame_key]
            output.append(
                {
                    "run_name": policy_run,
                    "row": row_id,
                    "scene": scene,
                    "duration_sec": duration_sec,
                    "dataset_start_frame": dataset_start_frame,
                    "selected_memory_frame": frame_idx,
                    "selected_dataset_frame": (
                        int(dataset_start_frame) + int(frame_idx)
                        if dataset_start_frame is not None and frame_idx is not None
                        else None
                    ),
                    "upperbound_access_count": upperbound_access_count,
                    "policy_exact_target_match_count": len(matched_targets),
                    "policy_any_access_count": any_policy_access,
                    "upperbound_target_recall": safe_round(
                        safe_div(len(matched_targets), upperbound_access_count)
                    ),
                    "policy_any_access_vs_upperbound": safe_round(
                        safe_div(any_policy_access, upperbound_access_count)
                    ),
                }
            )

    output.sort(
        key=lambda row: (
            str(row["run_name"]),
            str(row["row"]),
            -int(row["upperbound_access_count"]),
            int(row["selected_memory_frame"])
            if row["selected_memory_frame"] is not None
            else 10**9,
        )
    )
    return output


def run_summary_rows(rows, alignment_rows, baseline_run):
    grouped = defaultdict(list)
    for row in rows:
        if is_retrieval(row):
            grouped[row.get("_run_name")].append(row)

    alignment_by_run = defaultdict(list)
    for row in alignment_rows:
        alignment_by_run[row["run_name"]].append(row)

    output = []
    for run_name, group in sorted(grouped.items()):
        selected = [row for row in group if row.get("selected")]
        fallback = [row for row in group if not row.get("selected")]
        selected_counts = Counter(
            (row.get("_trace_file"), row.get("selected_memory_frame")) for row in selected
        )
        reuse_counts = list(selected_counts.values())
        ages = [to_float(row.get("memory_age")) for row in selected]
        overlaps = [to_float(row.get("selected_overlap")) for row in selected]
        candidates = [to_float(row.get("candidate_count")) for row in group]
        stored = [to_float(row.get("stored_memory_size")) for row in group]
        bin_counts = Counter(age_bin(int(age)) for age in ages if age is not None)
        align = alignment_by_run.get(run_name, [])
        with_baseline = [row for row in align if row.get("has_baseline")]
        capture = [to_float(row.get("overlap_capture_ratio")) for row in with_baseline]
        gaps = [to_float(row.get("overlap_gap")) for row in with_baseline]
        summary = {
            "run_name": run_name,
            "is_baseline": run_name == baseline_run,
            "trace_files": len({row.get("_trace_file") for row in group}),
            "memory_policy": common_value(group, "memory_policy"),
            "memory_budget": common_value(group, "memory_budget"),
            "duration_sec": common_value(group, "duration_sec"),
            "retrieval_slots": len(group),
            "selected_slots": len(selected),
            "fallback_slots": len(fallback),
            "fallback_rate": safe_round(safe_div(len(fallback), len(group))) or 0.0,
            "unique_selected_frames": len(selected_counts),
            "max_reuse_count": max(reuse_counts) if reuse_counts else 0,
            "mean_reuse_count": safe_round(mean(reuse_counts)),
            "reuse_gini": safe_round(gini(reuse_counts)),
            **numeric_stats("selected_age", ages),
            **numeric_stats("selected_overlap", overlaps),
            **numeric_stats("candidate_count", candidates),
            **numeric_stats("stored_memory_size", stored),
            "targets_with_upperbound": len(with_baseline),
            "exact_upperbound_match_rate": safe_round(
                safe_div(sum(bool(row["exact_upperbound_match"]) for row in with_baseline), len(with_baseline))
            ),
            "near_upperbound_match_rate": safe_round(
                safe_div(sum(bool(row["near_upperbound_match"]) for row in with_baseline), len(with_baseline))
            ),
            "mean_overlap_capture_ratio": safe_round(mean(capture)),
            "median_overlap_capture_ratio": safe_round(percentile(capture, 0.5)),
            "mean_overlap_gap": safe_round(mean(gaps)),
            "p90_overlap_gap": safe_round(percentile(gaps, 0.9)),
            "p95_overlap_gap": safe_round(percentile(gaps, 0.95)),
            "gap_gt_0_05_rate": safe_round(
                safe_div(sum(gap is not None and gap > 0.05 for gap in gaps), len(gaps))
            ),
            "gap_gt_0_10_rate": safe_round(
                safe_div(sum(gap is not None and gap > 0.10 for gap in gaps), len(gaps))
            ),
            "mean_age_delta_vs_upperbound": safe_round(
                mean(to_float(row.get("age_delta_vs_upperbound")) for row in with_baseline)
            ),
        }
        for name, _, _ in AGE_BINS:
            count = bin_counts[name]
            summary[f"age_bin_{name}"] = count
            summary[f"age_bin_{name}_frac"] = (
                safe_round(safe_div(count, len(ages))) or 0.0
            )
        output.append(summary)
    return output


def section_summary_rows(rows, alignment_rows):
    grouped = defaultdict(list)
    for row in rows:
        if is_retrieval(row):
            grouped[(row.get("_run_name"), row.get("section_idx"))].append(row)

    alignment_by_group = defaultdict(list)
    for row in alignment_rows:
        alignment_by_group[(row["run_name"], row["section_idx"])].append(row)

    output = []
    for (run_name, section_idx), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        selected = [row for row in group if row.get("selected")]
        fallback = [row for row in group if not row.get("selected")]
        align = alignment_by_group.get((run_name, section_idx), [])
        align = [row for row in align if row.get("has_baseline")]
        output.append(
            {
                "run_name": run_name,
                "section_idx": section_idx,
                "videos": len({row.get("_trace_file") for row in group}),
                "retrieval_slots": len(group),
                "selected_slots": len(selected),
                "fallback_slots": len(fallback),
                "fallback_rate": safe_round(safe_div(len(fallback), len(group))) or 0.0,
                "unique_selected_frames": len(
                    {(row.get("_trace_file"), row.get("selected_memory_frame")) for row in selected}
                ),
                **numeric_stats("selected_age", [to_float(row.get("memory_age")) for row in selected]),
                **numeric_stats(
                    "selected_overlap",
                    [to_float(row.get("selected_overlap")) for row in selected],
                ),
                "exact_upperbound_match_rate": safe_round(
                    safe_div(sum(bool(row["exact_upperbound_match"]) for row in align), len(align))
                ),
                "mean_overlap_capture_ratio": safe_round(
                    mean(to_float(row.get("overlap_capture_ratio")) for row in align)
                ),
                "mean_overlap_gap": safe_round(mean(to_float(row.get("overlap_gap")) for row in align)),
            }
        )
    return output


def eviction_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("event") == "memory_eviction":
            grouped[row.get("_run_name")].append(row)

    output = []
    for run_name, group in sorted(grouped.items()):
        ages = [to_float(row.get("memory_age_at_eviction")) for row in group]
        recent = [age for age in ages if age is not None and age <= 76]
        output.append(
            {
                "run_name": run_name,
                "trace_files": len({row.get("_trace_file") for row in group}),
                "memory_policy": common_value(group, "memory_policy"),
                "memory_budget": common_value(group, "memory_budget"),
                "duration_sec": common_value(group, "duration_sec"),
                "evictions": len(group),
                "recent_eviction_frac_age_le_76": safe_round(safe_div(len(recent), len(ages))),
                **numeric_stats("evicted_age", ages),
                **numeric_stats(
                    "eviction_score",
                    [to_float(row.get("eviction_score")) for row in group],
                ),
                **numeric_stats(
                    "eviction_rarity",
                    [to_float(row.get("eviction_rarity")) for row in group],
                ),
                **numeric_stats(
                    "eviction_irreplaceability",
                    [to_float(row.get("eviction_irreplaceability")) for row in group],
                ),
                **numeric_stats(
                    "eviction_rgb_nearest_distance",
                    [to_float(row.get("eviction_rgb_nearest_distance")) for row in group],
                ),
            }
        )
    return output


def write_csv(path, rows):
    if not rows:
        return False
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
    return True


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows, columns, limit=10):
    rows = rows[:limit]
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    path,
    run_rows,
    upperbound_rows,
    policy_frame_rows,
    frame_alignment,
    target_alignment,
    baseline_run,
    duplicate_targets,
):
    comparable_runs = [row for row in run_rows if not row.get("is_baseline")]
    comparable_runs.sort(
        key=lambda row: (
            row.get("mean_overlap_capture_ratio") is None,
            0
            if row.get("mean_overlap_capture_ratio") is None
            else -row["mean_overlap_capture_ratio"],
            row["run_name"],
        )
    )

    upperbound_top = sorted(
        upperbound_rows,
        key=lambda row: (-int(row["upperbound_access_count"]), str(row["row"])),
    )
    policy_top = sorted(
        policy_frame_rows,
        key=lambda row: (str(row["run_name"]), str(row["row"]), -int(row["access_count"])),
    )
    missed = [
        row
        for row in frame_alignment
        if row.get("upperbound_access_count", 0) >= 3
        and (row.get("upperbound_target_recall") or 0) == 0
    ]
    missed.sort(
        key=lambda row: (
            str(row["run_name"]),
            -int(row["upperbound_access_count"]),
            str(row["row"]),
        )
    )
    severe_targets = [
        row
        for row in target_alignment
        if row.get("has_baseline") and (to_float(row.get("overlap_gap")) or 0) > 0.10
    ]
    severe_targets.sort(
        key=lambda row: (
            str(row["run_name"]),
            -(to_float(row.get("overlap_gap")) or 0),
        )
    )

    lines = [
        "# MemCam Trace Usefulness Report",
        "",
        "## Executive Summary",
        "",
        (
            "Usefulness here is trace-derived: a frame is useful when the generator actually "
            "retrieves it for future context. If an unbounded baseline run is present, its "
            "selected frames are treated as an upperbound retrieval proxy because that run can "
            "choose from all prior generated frames. Bounded policies are then judged by how "
            "often they select the same or near frames and how much overlap they retain."
        ),
        "",
    ]

    if duplicate_targets:
        lines.extend(
            [
                f"Note: {duplicate_targets} duplicate baseline target rows were found; "
                "the report kept the duplicate with the higher selected overlap.",
                "",
            ]
        )

    if comparable_runs:
        lines.extend(
            [
                "## Policy Alignment To Upperbound",
                "",
                markdown_table(
                    comparable_runs,
                    [
                        "run_name",
                        "trace_files",
                        "retrieval_slots",
                        "exact_upperbound_match_rate",
                        "near_upperbound_match_rate",
                        "mean_overlap_capture_ratio",
                        "mean_overlap_gap",
                        "gap_gt_0_10_rate",
                        "unique_selected_frames",
                    ],
                    limit=20,
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Policy Alignment To Upperbound",
                "",
                (
                    f"No non-{baseline_run} runs were available for upperbound alignment. "
                    "The frame tables below still describe retrieval concentration."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Frames The Upperbound Kept Coming Back To",
            "",
            markdown_table(
                upperbound_top,
                [
                    "row",
                    "scene",
                    "selected_memory_frame",
                    "selected_dataset_frame",
                    "upperbound_access_count",
                    "upperbound_access_frac_in_video",
                    "overlap_mean",
                    "age_mean",
                    "first_target_frame",
                    "last_target_frame",
                ],
                limit=25,
            ),
            "",
            "## Frames Each Run Actually Used Most",
            "",
            markdown_table(
                policy_top,
                [
                    "run_name",
                    "row",
                    "scene",
                    "selected_memory_frame",
                    "access_count",
                    "access_frac_in_video",
                    "overlap_mean",
                    "age_mean",
                ],
                limit=30,
            ),
            "",
        ]
    )

    if missed:
        lines.extend(
            [
                "## Upperbound-Useful Frames Missed By Policies",
                "",
                (
                    "These rows highlight frames that the unbounded run selected at least three "
                    "times for a video, but a bounded policy never selected for those same target slots."
                ),
                "",
                markdown_table(
                    missed,
                    [
                        "run_name",
                        "row",
                        "scene",
                        "selected_memory_frame",
                        "selected_dataset_frame",
                        "upperbound_access_count",
                        "policy_any_access_count",
                        "upperbound_target_recall",
                    ],
                    limit=30,
                ),
                "",
            ]
        )

    if severe_targets:
        lines.extend(
            [
                "## Largest Target-Level Retrieval Gaps",
                "",
                (
                    "These are individual target context slots where the bounded policy selected "
                    "a frame with much lower camera-overlap than the unbounded run."
                ),
                "",
                markdown_table(
                    severe_targets,
                    [
                        "run_name",
                        "row",
                        "scene",
                        "section_idx",
                        "target_frame",
                        "policy_selected_frame",
                        "baseline_selected_frame",
                        "policy_overlap",
                        "baseline_overlap",
                        "overlap_gap",
                    ],
                    limit=30,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## How To Read This",
            "",
            "- `exact_upperbound_match_rate` is strict: the bounded policy picked the same frame as the unbounded run for the same target slot.",
            "- `near_upperbound_match_rate` allows a small frame-index window, useful because nearby frames can encode nearly the same viewpoint.",
            "- `mean_overlap_capture_ratio` compares policy selected overlap against the unbounded selected overlap. Values near 1 mean the policy retained most of the retrieval quality.",
            "- `upperbound_access_count` is the most direct trace-based frame usefulness signal: the all-memory run repeatedly wanted that frame later.",
            "- This is not human perceptual ground truth. It is a behavioral readout of the retrieval mechanism. If overlap labels are available, join this with `utils/analyze_memory_policies.py` outputs for label-based future-use counts.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MemCam generated-run traces to identify useful frames and compare "
            "bounded policies against an unbounded upperbound run."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--durations", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--trace_dir_name", type=str, default="access_traces")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--near_frame_window", type=int, default=4)
    parser.add_argument("--require_common_targets", action="store_true")
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

    rows = filter_rows(
        rows,
        durations=parse_int_list(args.durations),
        row_filter=parse_rows(args.rows),
    )
    if not rows:
        raise RuntimeError("No trace rows loaded after filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_index, duplicate_targets = build_baseline_index(rows, args.baseline_run)
    if args.require_common_targets and not baseline_index:
        raise RuntimeError(f"Baseline run '{args.baseline_run}' has no selected retrieval rows.")

    alignment_rows = target_alignment_rows(
        rows=rows,
        baseline_index=baseline_index,
        baseline_run=args.baseline_run,
        near_frame_window=args.near_frame_window,
    )
    if args.require_common_targets:
        alignment_rows = [row for row in alignment_rows if row.get("has_baseline")]

    run_rows = run_summary_rows(rows, alignment_rows, baseline_run=args.baseline_run)
    section_rows = section_summary_rows(rows, alignment_rows)
    upperbound_rows = upperbound_frame_rows(rows, baseline_run=args.baseline_run)
    selected_frame_rows = summarize_selected_frames(rows)
    frame_alignment = frame_alignment_rows(
        rows=rows,
        baseline_index=baseline_index,
        baseline_run=args.baseline_run,
    )
    eviction_rows = eviction_summary_rows(rows)

    csv_outputs = [
        (args.output_dir / "trace_run_summary.csv", run_rows),
        (args.output_dir / "trace_section_summary.csv", section_rows),
        (args.output_dir / "trace_target_alignment.csv", alignment_rows),
        (args.output_dir / "trace_upperbound_useful_frames.csv", upperbound_rows),
        (args.output_dir / "trace_policy_selected_frames.csv", selected_frame_rows),
        (args.output_dir / "trace_frame_alignment.csv", frame_alignment),
        (args.output_dir / "trace_eviction_summary.csv", eviction_rows),
    ]
    written_paths = []
    skipped_paths = []
    for path, output_rows in csv_outputs:
        if write_csv(path, output_rows):
            written_paths.append(path)
        else:
            skipped_paths.append(path)

    run_summary_json = args.output_dir / "trace_run_summary.json"
    write_json(run_summary_json, run_rows)
    written_paths.append(run_summary_json)

    report_path = args.output_dir / "trace_usefulness_report.md"
    write_report(
        path=report_path,
        run_rows=run_rows,
        upperbound_rows=upperbound_rows,
        policy_frame_rows=selected_frame_rows,
        frame_alignment=frame_alignment,
        target_alignment=alignment_rows,
        baseline_run=args.baseline_run,
        duplicate_targets=duplicate_targets,
    )

    print(json.dumps(run_rows, indent=2, ensure_ascii=False))
    for path in written_paths:
        print(f"Wrote: {path}")
    for path in skipped_paths:
        print(f"No rows: {path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
