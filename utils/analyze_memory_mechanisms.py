import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "memcam_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "memcam_xdg_cache"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required for mechanism figures. Install it with: "
        "python -m pip install matplotlib"
    ) from exc


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
            start, end = part.split("-", 1)
            rows.update(range(int(start), int(end) + 1))
        else:
            rows.add(int(part))
    return rows


def key_value(value):
    return "" if value is None else str(value)


def to_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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


def safe_round(value, digits=6):
    value = to_float(value)
    if value is None:
        return None
    return round(value, digits)


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    value_float = to_float(value)
    if value_float is not None:
        if abs(value_float) >= 100:
            return f"{value_float:.1f}"
        return f"{value_float:.{digits}f}"
    return str(value)


def run_sort_key(run_name):
    if run_name == "baseline":
        return (0, 0, run_name)
    match = re.search(r"_b(\d+)|b(\d+)", run_name)
    budget = int(next(group for group in match.groups() if group)) if match else 9999
    if run_name.startswith("fifo"):
        family = 1
    elif run_name.startswith("ri"):
        family = 2
    elif run_name.startswith("slam"):
        family = 3
    else:
        family = 9
    return (family, budget, run_name)


def budget_from_run(run_name):
    match = re.search(r"_b(\d+)|b(\d+)", run_name)
    if not match:
        return None
    return int(next(group for group in match.groups() if group))


def color_for_run(run_name):
    if run_name == "baseline":
        return "#555555"
    if run_name.startswith("fifo"):
        return "#2f7fbc"
    if run_name.startswith("ri"):
        return "#d88c1f"
    if run_name.startswith("slam"):
        return "#4f9b62"
    return "#8a5fbf"


def setup_axis(ax, title=None):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def discover_trace_dirs(root, runs, trace_dir_name, strict):
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
    durations = set(durations or [])
    row_filter = set(row_filter or [])
    output = []
    for row in rows:
        if durations:
            duration = to_int(row.get("duration_sec"))
            if duration not in durations:
                continue
        if row_filter:
            row_id = to_int(row.get("row"))
            if row_id not in row_filter:
                continue
        output.append(row)
    return output


def video_key(row):
    row_id = row.get("row")
    scene = row.get("scene")
    duration = row.get("duration_sec")
    dataset_start = row.get("dataset_start_frame")
    if row_id is not None and scene is not None and duration is not None:
        return (
            key_value(row_id),
            key_value(scene),
            key_value(duration),
            key_value(dataset_start),
        )
    return (
        key_value(row.get("_trace_file")),
        key_value(scene),
        key_value(duration),
        key_value(dataset_start),
    )


def target_key(row):
    return (
        *video_key(row),
        key_value(row.get("section_idx")),
        key_value(row.get("context_slot")),
        key_value(row.get("target_frame")),
    )


def is_context_access(row):
    return row.get("event") == "context_access"


def is_retrieval(row):
    section_idx = to_int(row.get("section_idx"))
    return is_context_access(row) and section_idx is not None and section_idx > 0


def is_selected(row):
    return is_retrieval(row) and bool(row.get("selected"))


def section_start_frame(section_idx, section_stride):
    return int(section_idx) * int(section_stride)


def section_end_frame(section_idx, section_stride, frames_per_section):
    return section_start_frame(section_idx, section_stride) + int(frames_per_section) - 1


def generated_frame_range(section_idx, section_stride, frames_per_section):
    start = section_start_frame(section_idx, section_stride)
    return range(start, start + int(frames_per_section))


def excluded_frames_for_section(section_idx, section_stride, frames_per_section):
    start = section_start_frame(section_idx, section_stride)
    if section_idx == 0:
        anchor = {start}
        predict = set(range(start + 1, start + frames_per_section))
    else:
        anchor = set(range(start - 3, start + 1))
        predict = set(range(start + 1, start + frames_per_section))
    return anchor | predict


def build_baseline_index(rows, baseline_run):
    baseline = {}
    duplicates = 0
    for row in rows:
        if row.get("_run_name") != baseline_run or not is_selected(row):
            continue
        key = target_key(row)
        if key in baseline:
            duplicates += 1
            old_overlap = to_float(baseline[key].get("selected_overlap")) or -math.inf
            new_overlap = to_float(row.get("selected_overlap")) or -math.inf
            if new_overlap <= old_overlap:
                continue
        baseline[key] = row
    return baseline, duplicates


def build_policy_target_index(rows):
    index = {}
    for row in rows:
        if is_retrieval(row):
            index[(row.get("_run_name"), target_key(row))] = row
    return index


def reconstruct_memory(rows, section_stride, frames_per_section):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("_run_name"), video_key(row))].append(row)

    before = {}
    after = {}
    retained_long = []
    video_meta = {}

    for (run_name, vkey), group in groups.items():
        section_values = [
            to_int(row.get("section_idx"))
            for row in group
            if to_int(row.get("section_idx")) is not None
        ]
        if not section_values:
            continue

        meta_row = next((row for row in group if row.get("row") is not None), group[0])
        video_meta[vkey] = {
            "row": meta_row.get("row"),
            "scene": meta_row.get("scene"),
            "duration_sec": meta_row.get("duration_sec"),
            "dataset_start_frame": meta_row.get("dataset_start_frame"),
        }

        evicted_by_section = defaultdict(list)
        for row in group:
            if row.get("event") != "memory_eviction":
                continue
            section_idx = to_int(row.get("section_idx"))
            evicted = to_int(row.get("evicted_memory_frame"))
            if section_idx is not None and evicted is not None:
                evicted_by_section[section_idx].append(evicted)

        memory = {0}
        for section_idx in range(max(section_values) + 1):
            before[(run_name, vkey, section_idx)] = set(memory)

            memory.update(generated_frame_range(section_idx, section_stride, frames_per_section))
            for evicted in evicted_by_section.get(section_idx, []):
                memory.discard(evicted)

            snapshot = set(memory)
            after[(run_name, vkey, section_idx)] = snapshot
            end_frame = section_end_frame(section_idx, section_stride, frames_per_section)
            for retained in sorted(snapshot):
                retained_long.append(
                    {
                        "run_name": run_name,
                        **video_meta[vkey],
                        "section_idx": section_idx,
                        "section_end_frame": end_frame,
                        "retained_frame": retained,
                        "retained_age": end_frame - retained,
                        "retained_memory_size": len(snapshot),
                    }
                )

    return before, after, retained_long, video_meta


def availability(snapshot, frame_idx, section_idx, near_window, section_stride, frames_per_section):
    if frame_idx is None or snapshot is None:
        return None, None
    excluded = excluded_frames_for_section(section_idx, section_stride, frames_per_section)
    candidates = snapshot - excluded
    exact = frame_idx in candidates
    near = any(abs(candidate - frame_idx) <= near_window for candidate in candidates)
    return exact, near


def target_alignment_rows(
    rows,
    baseline_index,
    policy_target_index,
    memory_before,
    baseline_run,
    near_window,
    section_stride,
    frames_per_section,
):
    run_names = sorted({row.get("_run_name") for row in rows}, key=run_sort_key)
    run_names = [run for run in run_names if run != baseline_run]
    output = []

    for base_key, base_row in baseline_index.items():
        vkey = video_key(base_row)
        section_idx = to_int(base_row.get("section_idx"))
        baseline_frame = to_int(base_row.get("selected_memory_frame"))
        baseline_overlap = to_float(base_row.get("selected_overlap"))
        baseline_target = to_int(base_row.get("target_frame"))

        for run_name in run_names:
            policy_row = policy_target_index.get((run_name, base_key))
            snapshot = memory_before.get((run_name, vkey, section_idx))
            available_exact, available_near = availability(
                snapshot=snapshot,
                frame_idx=baseline_frame,
                section_idx=section_idx,
                near_window=near_window,
                section_stride=section_stride,
                frames_per_section=frames_per_section,
            )

            policy_selected = bool(policy_row and policy_row.get("selected"))
            policy_frame = to_int(policy_row.get("selected_memory_frame")) if policy_selected else None
            policy_overlap = to_float(policy_row.get("selected_overlap")) if policy_selected else None
            selected_exact = policy_frame is not None and policy_frame == baseline_frame
            selected_near = (
                policy_frame is not None
                and baseline_frame is not None
                and abs(policy_frame - baseline_frame) <= near_window
            )
            overlap_gap = (
                baseline_overlap - policy_overlap
                if baseline_overlap is not None and policy_overlap is not None
                else None
            )
            overlap_capture = (
                safe_div(policy_overlap, baseline_overlap)
                if baseline_overlap not in (None, 0.0) and policy_overlap is not None
                else None
            )

            output.append(
                {
                    "run_name": run_name,
                    "baseline_run": baseline_run,
                    "row": base_row.get("row"),
                    "scene": base_row.get("scene"),
                    "duration_sec": base_row.get("duration_sec"),
                    "dataset_start_frame": base_row.get("dataset_start_frame"),
                    "section_idx": section_idx,
                    "context_slot": base_row.get("context_slot"),
                    "target_frame": baseline_target,
                    "target_dataset_frame": base_row.get("target_dataset_frame"),
                    "baseline_selected_frame": baseline_frame,
                    "baseline_selected_dataset_frame": base_row.get("selected_dataset_frame"),
                    "baseline_overlap": safe_round(baseline_overlap),
                    "has_policy_trace": snapshot is not None,
                    "baseline_frame_available_exact": available_exact,
                    "baseline_frame_available_near": available_near,
                    "policy_selected": policy_selected,
                    "policy_selected_frame": policy_frame,
                    "policy_selected_dataset_frame": (
                        policy_row.get("selected_dataset_frame") if policy_row else None
                    ),
                    "policy_selected_exact_baseline": selected_exact,
                    "policy_selected_near_baseline": selected_near,
                    "policy_overlap": safe_round(policy_overlap),
                    "overlap_gap_vs_unbounded": safe_round(overlap_gap),
                    "overlap_capture_ratio": safe_round(overlap_capture),
                }
            )
    return output


def summarize_retrieval_overlap(alignment_rows):
    grouped = defaultdict(list)
    for row in alignment_rows:
        grouped[row["run_name"]].append(row)

    output = []
    for run_name, group in sorted(grouped.items(), key=lambda item: run_sort_key(item[0])):
        comparable = [row for row in group if row.get("has_policy_trace")]
        selected = [row for row in comparable if row.get("policy_selected")]
        gaps = [to_float(row.get("overlap_gap_vs_unbounded")) for row in comparable]
        capture = [to_float(row.get("overlap_capture_ratio")) for row in comparable]
        output.append(
            {
                "run_name": run_name,
                "baseline_targets": len(group),
                "comparable_targets": len(comparable),
                "missing_targets": len(group) - len(comparable),
                "exact_available_count": sum(
                    row.get("baseline_frame_available_exact") is True for row in comparable
                ),
                "near_available_count": sum(
                    row.get("baseline_frame_available_near") is True for row in comparable
                ),
                "exact_availability_rate": safe_round(
                    safe_div(
                        sum(row.get("baseline_frame_available_exact") is True for row in comparable),
                        len(comparable),
                    )
                ),
                "near_availability_rate": safe_round(
                    safe_div(
                        sum(row.get("baseline_frame_available_near") is True for row in comparable),
                        len(comparable),
                    )
                ),
                "selected_slots": len(selected),
                "selected_exact_match_rate": safe_round(
                    safe_div(
                        sum(row.get("policy_selected_exact_baseline") is True for row in comparable),
                        len(comparable),
                    )
                ),
                "selected_near_match_rate": safe_round(
                    safe_div(
                        sum(row.get("policy_selected_near_baseline") is True for row in comparable),
                        len(comparable),
                    )
                ),
                "mean_overlap_capture_ratio": safe_round(mean(capture)),
                "mean_overlap_gap_vs_unbounded": safe_round(mean(gaps)),
                "gap_gt_0_10_rate": safe_round(
                    safe_div(sum(gap is not None and gap > 0.10 for gap in gaps), len(gaps))
                ),
            }
        )
    return output


def choose_fifo_run(run_name, fifo_runs, explicit_fifo_run=None):
    if explicit_fifo_run:
        return explicit_fifo_run
    if not fifo_runs:
        return None
    budget = budget_from_run(run_name)
    if budget is not None:
        for fifo_run in fifo_runs:
            if budget_from_run(fifo_run) == budget:
                return fifo_run
    return sorted(fifo_runs, key=run_sort_key)[0]


def preservation_vs_fifo_rows(alignment_rows, explicit_fifo_run=None):
    by_run_target = defaultdict(dict)
    fifo_runs = sorted(
        {row["run_name"] for row in alignment_rows if row["run_name"].startswith("fifo")},
        key=run_sort_key,
    )
    for row in alignment_rows:
        target = (
            key_value(row.get("row")),
            key_value(row.get("scene")),
            key_value(row.get("duration_sec")),
            key_value(row.get("dataset_start_frame")),
            key_value(row.get("section_idx")),
            key_value(row.get("context_slot")),
            key_value(row.get("target_frame")),
        )
        by_run_target[row["run_name"]][target] = row

    output = []
    policy_runs = sorted(by_run_target.keys(), key=run_sort_key)
    for run_name in policy_runs:
        if run_name.startswith("fifo"):
            continue
        fifo_run = choose_fifo_run(run_name, fifo_runs, explicit_fifo_run=explicit_fifo_run)
        if not fifo_run or fifo_run not in by_run_target:
            continue

        policy_targets = by_run_target[run_name]
        fifo_targets = by_run_target[fifo_run]
        common_targets = sorted(set(policy_targets) & set(fifo_targets))
        comparable = [
            target
            for target in common_targets
            if policy_targets[target].get("has_policy_trace")
            and fifo_targets[target].get("has_policy_trace")
        ]

        policy_kept_fifo_dropped_exact = 0
        fifo_kept_policy_dropped_exact = 0
        both_exact = 0
        neither_exact = 0
        policy_kept_fifo_dropped_near = 0
        fifo_kept_policy_dropped_near = 0
        both_near = 0
        neither_near = 0

        for target in comparable:
            policy_exact = policy_targets[target].get("baseline_frame_available_exact") is True
            fifo_exact = fifo_targets[target].get("baseline_frame_available_exact") is True
            policy_near = policy_targets[target].get("baseline_frame_available_near") is True
            fifo_near = fifo_targets[target].get("baseline_frame_available_near") is True

            policy_kept_fifo_dropped_exact += int(policy_exact and not fifo_exact)
            fifo_kept_policy_dropped_exact += int(fifo_exact and not policy_exact)
            both_exact += int(policy_exact and fifo_exact)
            neither_exact += int(not policy_exact and not fifo_exact)
            policy_kept_fifo_dropped_near += int(policy_near and not fifo_near)
            fifo_kept_policy_dropped_near += int(fifo_near and not policy_near)
            both_near += int(policy_near and fifo_near)
            neither_near += int(not policy_near and not fifo_near)

        output.append(
            {
                "run_name": run_name,
                "fifo_run": fifo_run,
                "common_targets": len(common_targets),
                "comparable_targets": len(comparable),
                "policy_kept_fifo_dropped_exact": policy_kept_fifo_dropped_exact,
                "fifo_kept_policy_dropped_exact": fifo_kept_policy_dropped_exact,
                "net_exact_preservation_vs_fifo": (
                    policy_kept_fifo_dropped_exact - fifo_kept_policy_dropped_exact
                ),
                "policy_kept_fifo_dropped_exact_rate": safe_round(
                    safe_div(policy_kept_fifo_dropped_exact, len(comparable))
                ),
                "fifo_kept_policy_dropped_exact_rate": safe_round(
                    safe_div(fifo_kept_policy_dropped_exact, len(comparable))
                ),
                "both_exact_available": both_exact,
                "neither_exact_available": neither_exact,
                "policy_kept_fifo_dropped_near": policy_kept_fifo_dropped_near,
                "fifo_kept_policy_dropped_near": fifo_kept_policy_dropped_near,
                "net_near_preservation_vs_fifo": (
                    policy_kept_fifo_dropped_near - fifo_kept_policy_dropped_near
                ),
                "policy_kept_fifo_dropped_near_rate": safe_round(
                    safe_div(policy_kept_fifo_dropped_near, len(comparable))
                ),
                "fifo_kept_policy_dropped_near_rate": safe_round(
                    safe_div(fifo_kept_policy_dropped_near, len(comparable))
                ),
                "both_near_available": both_near,
                "neither_near_available": neither_near,
            }
        )
    return output


def baseline_future_need_index(baseline_index):
    by_video = defaultdict(list)
    for row in baseline_index.values():
        by_video[video_key(row)].append(
            {
                "section_idx": to_int(row.get("section_idx")),
                "target_frame": to_int(row.get("target_frame")),
                "selected_memory_frame": to_int(row.get("selected_memory_frame")),
                "selected_overlap": to_float(row.get("selected_overlap")),
            }
        )
    for rows in by_video.values():
        rows.sort(key=lambda row: (row["target_frame"], row["section_idx"]))
    return by_video


def eviction_regret_rows(rows, baseline_index, near_window, section_stride, frames_per_section):
    future_by_video = baseline_future_need_index(baseline_index)
    output = []
    for row in rows:
        if row.get("event") != "memory_eviction":
            continue
        evicted = to_int(row.get("evicted_memory_frame"))
        section_idx = to_int(row.get("section_idx"))
        if evicted is None or section_idx is None:
            continue
        end_frame = to_int(row.get("section_end_frame"))
        if end_frame is None:
            end_frame = section_end_frame(section_idx, section_stride, frames_per_section)

        future_rows = [
            item
            for item in future_by_video.get(video_key(row), [])
            if item["target_frame"] is not None and item["target_frame"] > end_frame
        ]
        exact_rows = [item for item in future_rows if item["selected_memory_frame"] == evicted]
        near_rows = [
            item
            for item in future_rows
            if item["selected_memory_frame"] is not None
            and abs(item["selected_memory_frame"] - evicted) <= near_window
        ]

        output.append(
            {
                "run_name": row.get("_run_name"),
                "row": row.get("row"),
                "scene": row.get("scene"),
                "duration_sec": row.get("duration_sec"),
                "dataset_start_frame": row.get("dataset_start_frame"),
                "section_idx": section_idx,
                "section_end_frame": end_frame,
                "evicted_memory_frame": evicted,
                "evicted_dataset_frame": row.get("evicted_dataset_frame"),
                "memory_age_at_eviction": row.get("memory_age_at_eviction"),
                "future_upperbound_targets": len(future_rows),
                "future_exact_upperbound_uses": len(exact_rows),
                "future_near_upperbound_uses": len(near_rows),
                "later_needed_exact": len(exact_rows) > 0,
                "later_needed_near": len(near_rows) > 0,
                "first_future_exact_target": (
                    min(item["target_frame"] for item in exact_rows) if exact_rows else None
                ),
                "first_future_near_target": (
                    min(item["target_frame"] for item in near_rows) if near_rows else None
                ),
                "eviction_score": row.get("eviction_score"),
                "eviction_rarity": row.get("eviction_rarity"),
                "eviction_irreplaceability": row.get("eviction_irreplaceability"),
                "eviction_rgb_nearest_distance": row.get("eviction_rgb_nearest_distance"),
                "eviction_covisible_observers": row.get("eviction_covisible_observers"),
                "eviction_max_covisibility": row.get("eviction_max_covisibility"),
                "eviction_marginal_contribution": row.get("eviction_marginal_contribution"),
            }
        )
    return output


def summarize_eviction_regret(regret_rows):
    grouped = defaultdict(list)
    for row in regret_rows:
        grouped[row["run_name"]].append(row)

    output = []
    for run_name, group in sorted(grouped.items(), key=lambda item: run_sort_key(item[0])):
        ages = [to_float(row.get("memory_age_at_eviction")) for row in group]
        future_exact = [to_int(row.get("future_exact_upperbound_uses")) for row in group]
        future_near = [to_int(row.get("future_near_upperbound_uses")) for row in group]
        output.append(
            {
                "run_name": run_name,
                "evictions": len(group),
                "exact_regret_evictions": sum(row.get("later_needed_exact") is True for row in group),
                "near_regret_evictions": sum(row.get("later_needed_near") is True for row in group),
                "exact_regret_rate": safe_round(
                    safe_div(sum(row.get("later_needed_exact") is True for row in group), len(group))
                ),
                "near_regret_rate": safe_round(
                    safe_div(sum(row.get("later_needed_near") is True for row in group), len(group))
                ),
                "total_future_exact_upperbound_uses": sum(value or 0 for value in future_exact),
                "total_future_near_upperbound_uses": sum(value or 0 for value in future_near),
                "mean_future_exact_upperbound_uses": safe_round(mean(future_exact)),
                "mean_future_near_upperbound_uses": safe_round(mean(future_near)),
                "mean_evicted_age": safe_round(mean(ages)),
                "median_evicted_age": safe_round(percentile(ages, 0.5)),
            }
        )
    return output


def retained_age_summary_rows(retained_long):
    grouped = defaultdict(list)
    for row in retained_long:
        grouped[row["run_name"]].append(to_float(row.get("retained_age")))

    output = []
    for run_name, ages in sorted(grouped.items(), key=lambda item: run_sort_key(item[0])):
        output.append(
            {
                "run_name": run_name,
                "retained_points": len(ages),
                "mean_retained_age": safe_round(mean(ages)),
                "median_retained_age": safe_round(percentile(ages, 0.5)),
                "p10_retained_age": safe_round(percentile(ages, 0.1)),
                "p90_retained_age": safe_round(percentile(ages, 0.9)),
                "old_retained_frac_age_gt_152": safe_round(
                    safe_div(sum(age is not None and age > 152 for age in ages), len(ages))
                ),
                "old_retained_frac_age_gt_304": safe_round(
                    safe_div(sum(age is not None and age > 304 for age in ages), len(ages))
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
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def markdown_table(rows, columns, limit=12):
    rows = rows[:limit]
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def save_retrieval_overlap_plot(summary_rows, figures_dir):
    if not summary_rows:
        return None
    runs = [row["run_name"] for row in summary_rows]
    x = list(range(len(runs)))
    width = 0.34
    exact = [to_float(row.get("exact_availability_rate")) or 0.0 for row in summary_rows]
    selected = [to_float(row.get("selected_exact_match_rate")) or 0.0 for row in summary_rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - width / 2 for i in x], exact, width=width, color="#2f7fbc", label="unbounded-picked frame still available")
    ax.bar([i + width / 2 for i in x], selected, width=width, color="#d88c1f", label="policy selected same frame")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of unbounded retrieval targets")
    setup_axis(ax, "Analysis 1: retrieval overlap with unbounded")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = figures_dir / "01_retrieval_overlap_vs_unbounded.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_preservation_vs_fifo_plot(rows, figures_dir):
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: run_sort_key(row["run_name"]))
    runs = [row["run_name"] for row in rows]
    values = [to_float(row.get("net_exact_preservation_vs_fifo")) or 0.0 for row in rows]
    colors = ["#3b8f4a" if value >= 0 else "#c95050" for value in values]

    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.45 * len(rows))))
    ax.barh(runs, values, color=colors)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_xlabel("targets where policy kept unbounded-needed frame minus FIFO kept it")
    setup_axis(ax, "Analysis 1: direct preservation advantage vs FIFO")
    fig.tight_layout()
    path = figures_dir / "02_preserved_needed_frames_vs_fifo.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_eviction_regret_plot(summary_rows, figures_dir):
    if not summary_rows:
        return None
    runs = [row["run_name"] for row in summary_rows]
    x = list(range(len(runs)))
    width = 0.34
    exact = [to_float(row.get("exact_regret_rate")) or 0.0 for row in summary_rows]
    near = [to_float(row.get("near_regret_rate")) or 0.0 for row in summary_rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - width / 2 for i in x], exact, width=width, color="#c95050", label="same frame later used by unbounded")
    ax.bar([i + width / 2 for i in x], near, width=width, color="#8a5fbf", label="near frame later used by unbounded")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of evicted frames")
    setup_axis(ax, "Analysis 2: eviction regret")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = figures_dir / "03_eviction_regret.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_retained_age_distribution(retained_long, figures_dir):
    grouped = defaultdict(list)
    for row in retained_long:
        age = to_float(row.get("retained_age"))
        if age is not None:
            grouped[row["run_name"]].append(age)
    if not grouped:
        return None

    runs = sorted(grouped.keys(), key=run_sort_key)
    values = [grouped[run] for run in runs]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    try:
        box = ax.boxplot(values, tick_labels=runs, showfliers=False, patch_artist=True)
    except TypeError:
        box = ax.boxplot(values, labels=runs, showfliers=False, patch_artist=True)
    for patch, run_name in zip(box["boxes"], runs):
        patch.set_facecolor(color_for_run(run_name))
        patch.set_alpha(0.72)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("retained age in frames at section end")
    setup_axis(ax, "Analysis 3: retained memory age distribution")
    fig.tight_layout()
    path = figures_dir / "04_retained_age_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def choose_timeline_videos(retained_long, max_videos):
    counts = Counter()
    meta = {}
    for row in retained_long:
        vkey = (
            key_value(row.get("row")),
            key_value(row.get("scene")),
            key_value(row.get("duration_sec")),
            key_value(row.get("dataset_start_frame")),
        )
        counts[vkey] += 1
        meta[vkey] = {
            "row": row.get("row"),
            "scene": row.get("scene"),
            "duration_sec": row.get("duration_sec"),
            "dataset_start_frame": row.get("dataset_start_frame"),
        }
    selected = sorted(counts, key=lambda key: (to_int(meta[key].get("row")) or 10**9, meta[key].get("scene") or ""))
    return selected[:max_videos], meta


def save_timeline_plots(
    memory_after,
    retained_long,
    runs,
    figures_dir,
    max_videos,
    section_stride,
    frames_per_section,
):
    if max_videos <= 0:
        return []
    videos, meta = choose_timeline_videos(retained_long, max_videos)
    if not videos:
        return []
    output_paths = []
    runs = sorted(runs, key=run_sort_key)

    for video in videos:
        fig, axes = plt.subplots(len(runs), 1, figsize=(11.5, max(3.0, 2.0 * len(runs))), sharex=True, sharey=True)
        if len(runs) == 1:
            axes = [axes]

        all_x = []
        all_y = []
        for ax, run_name in zip(axes, runs):
            xs = []
            ys = []
            section_keys = sorted(
                (key for key in memory_after if key[0] == run_name and key[1] == video),
                key=lambda key: key[2],
            )
            for _run, _video, section_idx in section_keys:
                snapshot = memory_after[(_run, _video, section_idx)]
                section_end = section_end_frame(section_idx, section_stride, frames_per_section)
                for frame_idx in snapshot:
                    xs.append(section_end)
                    ys.append(frame_idx)
            all_x.extend(xs)
            all_y.extend(ys)
            ax.scatter(xs, ys, s=6, alpha=0.5, color=color_for_run(run_name), linewidths=0)
            ax.plot([0, max(xs) if xs else 1], [0, max(xs) if xs else 1], color="#cccccc", linewidth=0.8)
            ax.set_ylabel(run_name, rotation=0, labelpad=52, va="center")
            setup_axis(ax)

        if all_x:
            axes[-1].set_xlim(-5, max(all_x) + 10)
        if all_y:
            axes[-1].set_ylim(-5, max(all_y) + 10)
        axes[-1].set_xlabel("time: section end frame")
        fig.supylabel("retained memory frame index", x=0.02)
        title = f"Analysis 3: retained frames over time | row {meta[video].get('row')} {meta[video].get('scene')}"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0.04, 0.02, 1, 0.96))

        safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_value(meta[video].get("scene")))[:40]
        path = figures_dir / f"05_retained_timeline_row_{meta[video].get('row')}_{safe_scene}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def write_report(
    path,
    retrieval_summary,
    preservation_rows,
    regret_summary,
    age_summary,
    duplicate_targets,
    near_window,
):
    retrieval_summary = sorted(retrieval_summary, key=lambda row: run_sort_key(row["run_name"]))
    preservation_rows = sorted(
        preservation_rows,
        key=lambda row: -(to_float(row.get("net_exact_preservation_vs_fifo")) or -10**9),
    )
    regret_summary = sorted(regret_summary, key=lambda row: run_sort_key(row["run_name"]))
    age_summary = sorted(age_summary, key=lambda row: run_sort_key(row["run_name"]))

    lines = [
        "# Memory Mechanism Analysis",
        "",
        "This report answers: when a bounded method wins on LPIPS/FVD, what memory behavior could explain it?",
        "",
        "Reference definition: the unbounded run is treated as the upperbound retrieval proxy. If unbounded selected frame X for a target, X is a frame the retrieval rule wanted when no policy had evicted it.",
        "",
    ]
    if duplicate_targets:
        lines.extend(
            [
                f"Note: {duplicate_targets} duplicate unbounded target rows were found; the higher-overlap duplicate was kept.",
                "",
            ]
        )

    lines.extend(
        [
            "## Analysis 1: Retrieval Overlap",
            "",
            "Question: when unbounded retrieves frame X, does the bounded run still have X in memory, and does it actually pick X?",
            "",
            markdown_table(
                retrieval_summary,
                [
                    "run_name",
                    "comparable_targets",
                    "exact_availability_rate",
                    "near_availability_rate",
                    "selected_exact_match_rate",
                    "mean_overlap_capture_ratio",
                    "mean_overlap_gap_vs_unbounded",
                ],
                limit=30,
            ),
            "",
            "Direct FIFO comparison: positive `net_exact_preservation_vs_fifo` means the policy kept more unbounded-needed frames than the matched FIFO run.",
            "",
            markdown_table(
                preservation_rows,
                [
                    "run_name",
                    "fifo_run",
                    "comparable_targets",
                    "policy_kept_fifo_dropped_exact",
                    "fifo_kept_policy_dropped_exact",
                    "net_exact_preservation_vs_fifo",
                    "policy_kept_fifo_dropped_near",
                    "net_near_preservation_vs_fifo",
                ],
                limit=30,
            ),
            "",
            "## Analysis 2: Eviction Regret",
            "",
            f"Question: after a policy evicted frame X, did unbounded later retrieve X or a nearby frame? Nearby means +/- {near_window} frames.",
            "",
            markdown_table(
                regret_summary,
                [
                    "run_name",
                    "evictions",
                    "exact_regret_rate",
                    "near_regret_rate",
                    "mean_future_near_upperbound_uses",
                    "mean_evicted_age",
                ],
                limit=30,
            ),
            "",
            "## Analysis 3: Temporal Distribution",
            "",
            "Question: does the retained memory look like a FIFO sliding window, or like sparse old keyframes plus recent context?",
            "",
            markdown_table(
                age_summary,
                [
                    "run_name",
                    "retained_points",
                    "median_retained_age",
                    "p90_retained_age",
                    "old_retained_frac_age_gt_152",
                    "old_retained_frac_age_gt_304",
                ],
                limit=30,
            ),
            "",
            "## Files",
            "",
            "- `figures/01_retrieval_overlap_vs_unbounded.png`",
            "- `figures/02_preserved_needed_frames_vs_fifo.png`",
            "- `figures/03_eviction_regret.png`",
            "- `figures/04_retained_age_distribution.png`",
            "- `figures/05_retained_timeline_row_*.png`",
            "- `tables/retrieval_overlap_targets.csv`",
            "- `tables/retrieval_overlap_summary.csv`",
            "- `tables/preservation_vs_fifo.csv`",
            "- `tables/eviction_regret_events.csv`",
            "- `tables/eviction_regret_summary.csv`",
            "- `tables/retained_age_summary.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Trace-backed mechanism analyses for MemCam memory policies."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--fifo_run", type=str, default=None)
    parser.add_argument("--durations", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--trace_dir_name", type=str, default="access_traces")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--near_frame_window", type=int, default=4)
    parser.add_argument("--frames_per_section", type=int, default=77)
    parser.add_argument("--section_stride", type=int, default=76)
    parser.add_argument("--max_timeline_videos", type=int, default=3)
    parser.add_argument("--write_retained_long", action="store_true")
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
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_index, duplicate_targets = build_baseline_index(rows, args.baseline_run)
    if not baseline_index:
        raise RuntimeError(f"Baseline run '{args.baseline_run}' has no selected retrieval rows.")

    policy_target_index = build_policy_target_index(rows)
    memory_before, memory_after, retained_long, _video_meta = reconstruct_memory(
        rows=rows,
        section_stride=args.section_stride,
        frames_per_section=args.frames_per_section,
    )

    alignment_rows = target_alignment_rows(
        rows=rows,
        baseline_index=baseline_index,
        policy_target_index=policy_target_index,
        memory_before=memory_before,
        baseline_run=args.baseline_run,
        near_window=args.near_frame_window,
        section_stride=args.section_stride,
        frames_per_section=args.frames_per_section,
    )
    retrieval_summary = summarize_retrieval_overlap(alignment_rows)
    preservation_rows = preservation_vs_fifo_rows(
        alignment_rows,
        explicit_fifo_run=args.fifo_run,
    )
    regret_rows = eviction_regret_rows(
        rows=rows,
        baseline_index=baseline_index,
        near_window=args.near_frame_window,
        section_stride=args.section_stride,
        frames_per_section=args.frames_per_section,
    )
    regret_summary = summarize_eviction_regret(regret_rows)
    age_summary = retained_age_summary_rows(retained_long)

    write_csv(tables_dir / "retrieval_overlap_targets.csv", alignment_rows)
    write_csv(tables_dir / "retrieval_overlap_summary.csv", retrieval_summary)
    write_csv(tables_dir / "preservation_vs_fifo.csv", preservation_rows)
    write_csv(tables_dir / "eviction_regret_events.csv", regret_rows)
    write_csv(tables_dir / "eviction_regret_summary.csv", regret_summary)
    write_csv(tables_dir / "retained_age_summary.csv", age_summary)
    if args.write_retained_long:
        write_csv(tables_dir / "retained_frames_long.csv", retained_long)

    figure_paths = [
        save_retrieval_overlap_plot(retrieval_summary, figures_dir),
        save_preservation_vs_fifo_plot(preservation_rows, figures_dir),
        save_eviction_regret_plot(regret_summary, figures_dir),
        save_retained_age_distribution(retained_long, figures_dir),
    ]
    timeline_paths = save_timeline_plots(
        memory_after=memory_after,
        retained_long=retained_long,
        runs=sorted({row.get("_run_name") for row in rows}, key=run_sort_key),
        figures_dir=figures_dir,
        max_videos=args.max_timeline_videos,
        section_stride=args.section_stride,
        frames_per_section=args.frames_per_section,
    )
    figure_paths.extend(timeline_paths)

    write_report(
        path=args.output_dir / "report.md",
        retrieval_summary=retrieval_summary,
        preservation_rows=preservation_rows,
        regret_summary=regret_summary,
        age_summary=age_summary,
        duplicate_targets=duplicate_targets,
        near_window=args.near_frame_window,
    )

    print(f"Wrote: {args.output_dir / 'report.md'}")
    for path in figure_paths:
        if path is not None:
            print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
