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
        "matplotlib is required. Install it with: python -m pip install matplotlib"
    ) from exc


LOWER_IS_BETTER = {"lpips_alex", "fvd", "dino_distance", "clip_image_distance"}
QUALITY_FIELDS = ["lpips_alex", "fvd", "dino_distance", "clip_image_distance", "psnr_db", "ssim"]


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


def to_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


def safe_round(value, digits=6):
    value = to_float(value)
    if value is None:
        return None
    return round(value, digits)


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
        if durations and to_int(row.get("duration_sec")) not in durations:
            continue
        if row_filter and to_int(row.get("row")) not in row_filter:
            continue
        output.append(row)
    return output


def key_text(value):
    return "" if value is None else str(value)


def video_key(row):
    return (
        key_text(row.get("row")),
        key_text(row.get("scene")),
        key_text(row.get("duration_sec")),
        key_text(row.get("dataset_start_frame")),
    )


def metric_video_key(row):
    return (
        key_text(row.get("row")),
        key_text(row.get("scene")),
        key_text(row.get("duration_sec")),
        key_text(row.get("start_frame")),
    )


def target_key(row):
    return (
        *video_key(row),
        key_text(row.get("section_idx")),
        key_text(row.get("context_slot")),
        key_text(row.get("target_frame")),
    )


def is_retrieval(row):
    return row.get("event") == "context_access" and (to_int(row.get("section_idx")) or 0) > 0


def is_selected(row):
    return is_retrieval(row) and bool(row.get("selected"))


def gini(values):
    values = sorted(value for value in values if value is not None)
    total = sum(values)
    if not values or total == 0:
        return None
    n = len(values)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
    }


def trace_video_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("_run_name"), video_key(row))].append(row)

    output = []
    for (run_name, _vkey), group in sorted(grouped.items(), key=lambda item: (run_sort_key(item[0][0]), item[0][1])):
        selected = [row for row in group if is_selected(row)]
        retrieval = [row for row in group if is_retrieval(row)]
        ages = [to_float(row.get("memory_age")) for row in selected]
        overlaps = [to_float(row.get("selected_overlap")) for row in selected]
        candidates = [to_float(row.get("candidate_count")) for row in retrieval]
        stored = [to_float(row.get("stored_memory_size")) for row in retrieval]
        counts = Counter(row.get("selected_memory_frame") for row in selected)
        count_values = list(counts.values())
        meta = next((row for row in group if row.get("row") is not None), group[0])
        row = {
            "run_name": run_name,
            "row": meta.get("row"),
            "scene": meta.get("scene"),
            "duration_sec": meta.get("duration_sec"),
            "dataset_start_frame": meta.get("dataset_start_frame"),
            "retrieval_slots": len(retrieval),
            "selected_slots": len(selected),
            "fallback_slots": len(retrieval) - len(selected),
            "unique_selected_frames": len(counts),
            "top1_selected_frac": safe_round(safe_div(max(count_values), len(selected))) if count_values else None,
            "reuse_gini": safe_round(gini(count_values)),
            "old_selected_frac_age_gt_152": safe_round(safe_div(sum(age is not None and age > 152 for age in ages), len(ages))),
            "old_selected_frac_age_gt_304": safe_round(safe_div(sum(age is not None and age > 304 for age in ages), len(ages))),
        }
        row.update(numeric_stats("selected_age", ages))
        row.update(numeric_stats("selected_overlap", overlaps))
        row.update(numeric_stats("candidate_count", candidates))
        row.update(numeric_stats("stored_memory_size", stored))
        output.append(row)
    return output


def trace_run_summary(video_rows):
    grouped = defaultdict(list)
    for row in video_rows:
        grouped[row["run_name"]].append(row)

    output = []
    fields = [
        "retrieval_slots",
        "selected_slots",
        "unique_selected_frames",
        "top1_selected_frac",
        "reuse_gini",
        "old_selected_frac_age_gt_152",
        "old_selected_frac_age_gt_304",
        "selected_age_mean",
        "selected_age_median",
        "selected_age_p90",
        "selected_overlap_mean",
        "selected_overlap_median",
        "selected_overlap_p90",
        "candidate_count_mean",
        "stored_memory_size_mean",
    ]
    for run_name, rows in sorted(grouped.items(), key=lambda item: run_sort_key(item[0])):
        out = {"run_name": run_name, "videos": len(rows)}
        for field in fields:
            out[field] = safe_round(mean(to_float(row.get(field)) for row in rows))
        output.append(out)
    return output


def paired_target_rows(rows, baseline_run):
    baseline = {}
    policy_rows = {}
    for row in rows:
        if not is_selected(row):
            continue
        if row.get("_run_name") == baseline_run:
            current = baseline.get(target_key(row))
            if current is None or (to_float(row.get("selected_overlap")) or -1) > (to_float(current.get("selected_overlap")) or -1):
                baseline[target_key(row)] = row
        else:
            policy_rows[(row.get("_run_name"), target_key(row))] = row

    runs = sorted({row.get("_run_name") for row in rows if row.get("_run_name") != baseline_run}, key=run_sort_key)
    output = []
    for key, base in baseline.items():
        base_age = to_float(base.get("memory_age"))
        base_overlap = to_float(base.get("selected_overlap"))
        base_candidate_count = to_float(base.get("candidate_count"))
        for run_name in runs:
            policy = policy_rows.get((run_name, key))
            if policy is None:
                continue
            policy_age = to_float(policy.get("memory_age"))
            policy_overlap = to_float(policy.get("selected_overlap"))
            policy_candidate_count = to_float(policy.get("candidate_count"))
            output.append(
                {
                    "run_name": run_name,
                    "row": base.get("row"),
                    "scene": base.get("scene"),
                    "duration_sec": base.get("duration_sec"),
                    "dataset_start_frame": base.get("dataset_start_frame"),
                    "section_idx": base.get("section_idx"),
                    "context_slot": base.get("context_slot"),
                    "target_frame": base.get("target_frame"),
                    "baseline_selected_frame": base.get("selected_memory_frame"),
                    "policy_selected_frame": policy.get("selected_memory_frame"),
                    "baseline_age": safe_round(base_age),
                    "policy_age": safe_round(policy_age),
                    "age_delta_vs_unbounded": safe_round(policy_age - base_age if policy_age is not None and base_age is not None else None),
                    "baseline_overlap": safe_round(base_overlap),
                    "policy_overlap": safe_round(policy_overlap),
                    "overlap_delta_vs_unbounded": safe_round(policy_overlap - base_overlap if policy_overlap is not None and base_overlap is not None else None),
                    "baseline_candidate_count": safe_round(base_candidate_count),
                    "policy_candidate_count": safe_round(policy_candidate_count),
                    "candidate_count_delta_vs_unbounded": safe_round(
                        policy_candidate_count - base_candidate_count
                        if policy_candidate_count is not None and base_candidate_count is not None
                        else None
                    ),
                    "baseline_old_age_gt_152": base_age is not None and base_age > 152,
                    "policy_old_age_gt_152": policy_age is not None and policy_age > 152,
                    "baseline_old_age_gt_304": base_age is not None and base_age > 304,
                    "policy_old_age_gt_304": policy_age is not None and policy_age > 304,
                }
            )
    return output


def paired_run_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)

    output = []
    for run_name, group in sorted(grouped.items(), key=lambda item: run_sort_key(item[0])):
        age_deltas = [to_float(row.get("age_delta_vs_unbounded")) for row in group]
        overlap_deltas = [to_float(row.get("overlap_delta_vs_unbounded")) for row in group]
        candidate_deltas = [to_float(row.get("candidate_count_delta_vs_unbounded")) for row in group]
        output.append(
            {
                "run_name": run_name,
                "paired_targets": len(group),
                "mean_age_delta_vs_unbounded": safe_round(mean(age_deltas)),
                "median_age_delta_vs_unbounded": safe_round(percentile(age_deltas, 0.5)),
                "policy_newer_ref_rate": safe_round(safe_div(sum(delta is not None and delta < 0 for delta in age_deltas), len(group))),
                "policy_older_ref_rate": safe_round(safe_div(sum(delta is not None and delta > 0 for delta in age_deltas), len(group))),
                "mean_overlap_delta_vs_unbounded": safe_round(mean(overlap_deltas)),
                "policy_lower_overlap_rate": safe_round(safe_div(sum(delta is not None and delta < 0 for delta in overlap_deltas), len(group))),
                "mean_candidate_count_delta_vs_unbounded": safe_round(mean(candidate_deltas)),
                "baseline_old_policy_not_old_152_rate": safe_round(
                    safe_div(
                        sum(row.get("baseline_old_age_gt_152") and not row.get("policy_old_age_gt_152") for row in group),
                        len(group),
                    )
                ),
                "policy_old_baseline_not_old_152_rate": safe_round(
                    safe_div(
                        sum(row.get("policy_old_age_gt_152") and not row.get("baseline_old_age_gt_152") for row in group),
                        len(group),
                    )
                ),
                "baseline_old_policy_not_old_304_rate": safe_round(
                    safe_div(
                        sum(row.get("baseline_old_age_gt_304") and not row.get("policy_old_age_gt_304") for row in group),
                        len(group),
                    )
                ),
                "policy_old_baseline_not_old_304_rate": safe_round(
                    safe_div(
                        sum(row.get("policy_old_age_gt_304") and not row.get("baseline_old_age_gt_304") for row in group),
                        len(group),
                    )
                ),
            }
        )
    return output


def read_csv(path):
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_quality(metrics_dir, runs, duration):
    if metrics_dir is None:
        return [], []
    video_rows = []
    summary_rows = []
    for run_name in runs:
        run_dir = metrics_dir / run_name
        for row in read_csv(run_dir / "metrics.csv"):
            if str(row.get("duration_sec")) != str(duration):
                continue
            row["run_name"] = row.get("run_name") or run_name
            video_rows.append(row)

        summary = read_json(run_dir / "summary.json")
        if summary is not None:
            source = summary.get("by_duration", {}).get(str(duration), summary.get("overall", {}))
            out = {"run_name": run_name}
            for field in QUALITY_FIELDS:
                if field in source:
                    out[field] = source[field]
            summary_rows.append(out)
    return video_rows, summary_rows


def quality_join(video_trace_rows, quality_video_rows, baseline_run):
    trace_by_run_video = {
        (row["run_name"], video_key(row)): row
        for row in video_trace_rows
    }
    quality_by_run_video = {
        (row["run_name"], metric_video_key(row)): row
        for row in quality_video_rows
    }
    baseline_trace = {
        key[1]: row for key, row in trace_by_run_video.items() if key[0] == baseline_run
    }
    baseline_quality = {
        key[1]: row for key, row in quality_by_run_video.items() if key[0] == baseline_run
    }

    output = []
    for (run_name, vkey), trace_row in sorted(trace_by_run_video.items(), key=lambda item: (run_sort_key(item[0][0]), item[0][1])):
        if run_name == baseline_run:
            continue
        base_trace = baseline_trace.get(vkey)
        metric_key = (vkey[0], vkey[1], vkey[2], vkey[3])
        quality = quality_by_run_video.get((run_name, metric_key))
        base_quality = baseline_quality.get(metric_key)
        if base_trace is None:
            continue
        out = {
            "run_name": run_name,
            "row": trace_row.get("row"),
            "scene": trace_row.get("scene"),
            "duration_sec": trace_row.get("duration_sec"),
            "dataset_start_frame": trace_row.get("dataset_start_frame"),
        }
        for field in [
            "selected_age_mean",
            "selected_age_p90",
            "old_selected_frac_age_gt_152",
            "old_selected_frac_age_gt_304",
            "selected_overlap_mean",
            "candidate_count_mean",
            "top1_selected_frac",
            "reuse_gini",
        ]:
            value = to_float(trace_row.get(field))
            base = to_float(base_trace.get(field))
            out[f"baseline_{field}"] = safe_round(base)
            out[f"policy_{field}"] = safe_round(value)
            out[f"{field}_delta_vs_unbounded"] = safe_round(value - base if value is not None and base is not None else None)
        if quality is not None and base_quality is not None:
            for field in QUALITY_FIELDS:
                value = to_float(quality.get(field))
                base = to_float(base_quality.get(field))
                if value is None or base is None:
                    continue
                improvement = base - value if field in LOWER_IS_BETTER else value - base
                out[f"baseline_{field}"] = safe_round(base)
                out[f"policy_{field}"] = safe_round(value)
                out[f"{field}_improvement_vs_unbounded"] = safe_round(improvement)
                out[f"{field}_improvement_pct_vs_unbounded"] = safe_round(100.0 * improvement / base if base else None)
        output.append(out)
    return output


def write_csv(path, rows):
    if not rows:
        return False
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
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


def save_quality_bars(summary_rows, figures_dir, baseline_run):
    if not summary_rows:
        return None
    fields = [field for field in ["lpips_alex", "fvd"] if any(to_float(row.get(field)) is not None for row in summary_rows)]
    if not fields:
        return None
    runs = [row["run_name"] for row in summary_rows]
    fig, axes = plt.subplots(1, len(fields), figsize=(5.8 * len(fields), 4.8), squeeze=False)
    axes = axes[0]
    for ax, field in zip(axes, fields):
        values = [to_float(row.get(field)) or 0 for row in summary_rows]
        ax.bar(runs, values, color=[color_for_run(run) for run in runs])
        baseline = next((to_float(row.get(field)) for row in summary_rows if row["run_name"] == baseline_run), None)
        if baseline is not None:
            ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1.1)
        ax.set_ylabel(f"{field} {'lower is better' if field in LOWER_IS_BETTER else 'higher is better'}")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        setup_axis(ax, f"{field} by run")
    fig.tight_layout()
    path = figures_dir / "01_quality_headline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_age_shift_plot(summary_rows, figures_dir):
    if not summary_rows:
        return None
    runs = [row["run_name"] for row in summary_rows]
    values = [to_float(row.get("mean_age_delta_vs_unbounded")) or 0.0 for row in summary_rows]
    colors = ["#3b8f4a" if value < 0 else "#c95050" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(runs, values, color=colors)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_ylabel("policy selected age - unbounded selected age")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    setup_axis(ax, "Reference age shift vs unbounded")
    fig.tight_layout()
    path = figures_dir / "02_reference_age_shift_vs_unbounded.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_stale_rate_plot(run_rows, figures_dir, baseline_run):
    if not run_rows:
        return None
    runs = [row["run_name"] for row in run_rows]
    x = list(range(len(runs)))
    width = 0.34
    stale152 = [to_float(row.get("old_selected_frac_age_gt_152")) or 0.0 for row in run_rows]
    stale304 = [to_float(row.get("old_selected_frac_age_gt_304")) or 0.0 for row in run_rows]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar([idx - width / 2 for idx in x], stale152, width=width, color="#d88c1f", label="selected age > 152")
    ax.bar([idx + width / 2 for idx in x], stale304, width=width, color="#8a5fbf", label="selected age > 304")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of selected references")
    setup_axis(ax, "Stale reference selection rate")
    if any(row["run_name"] == baseline_run for row in run_rows):
        ax.legend(fontsize=9)
    fig.tight_layout()
    path = figures_dir / "03_stale_reference_rate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_overlap_tradeoff_plot(paired_summary, figures_dir):
    if not paired_summary:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for row in paired_summary:
        x = to_float(row.get("mean_age_delta_vs_unbounded"))
        y = to_float(row.get("mean_overlap_delta_vs_unbounded"))
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=120, color=color_for_run(row["run_name"]), edgecolor="white", linewidth=0.9)
        ax.annotate(row["run_name"], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_xlabel("selected age shift vs unbounded")
    ax.set_ylabel("overlap shift vs unbounded")
    setup_axis(ax, "Retrieval tradeoff: age vs camera-overlap")
    fig.tight_layout()
    path = figures_dir / "04_age_overlap_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_lpips_driver_scatter(driver_rows, figures_dir):
    rows = [
        row
        for row in driver_rows
        if to_float(row.get("lpips_alex_improvement_vs_unbounded")) is not None
        and to_float(row.get("old_selected_frac_age_gt_304_delta_vs_unbounded")) is not None
    ]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    panels = [
        ("old_selected_frac_age_gt_304_delta_vs_unbounded", "change in very-old selected fraction"),
        ("candidate_count_mean_delta_vs_unbounded", "change in candidate count"),
    ]
    for ax, (field, label) in zip(axes, panels):
        for row in rows:
            x = to_float(row.get(field))
            y = to_float(row.get("lpips_alex_improvement_vs_unbounded"))
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=62, alpha=0.78, color=color_for_run(row["run_name"]), edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#222222", linewidth=1)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_xlabel(label)
        ax.set_ylabel("LPIPS improvement vs unbounded")
        setup_axis(ax)
    fig.suptitle("Per-video LPIPS wins vs retrieval behavior shift", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figures_dir / "05_lpips_driver_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(path, quality_summary, run_summary, paired_summary, driver_rows, baseline_run, metrics_available):
    best_lpips = None
    best_fvd = None
    if quality_summary:
        lpips_rows = [row for row in quality_summary if to_float(row.get("lpips_alex")) is not None]
        fvd_rows = [row for row in quality_summary if to_float(row.get("fvd")) is not None]
        if lpips_rows:
            best_lpips = min(lpips_rows, key=lambda row: to_float(row.get("lpips_alex")))
        if fvd_rows:
            best_fvd = min(fvd_rows, key=lambda row: to_float(row.get("fvd")))

    paired_sorted = sorted(
        paired_summary,
        key=lambda row: abs(to_float(row.get("mean_age_delta_vs_unbounded")) or 0),
        reverse=True,
    )
    driver_with_lpips = [
        row for row in driver_rows if to_float(row.get("lpips_alex_improvement_vs_unbounded")) is not None
    ]
    driver_with_lpips.sort(
        key=lambda row: to_float(row.get("lpips_alex_improvement_vs_unbounded")) or -1e9,
        reverse=True,
    )

    lines = [
        "# Bounded Memory Driver Analysis",
        "",
        "This replaces the earlier unbounded-as-oracle report. Here, unbounded is the comparison baseline whose failures we are trying to explain, not the teacher.",
        "",
        "The useful question is: when a bounded policy improves LPIPS/FVD, how did it change the retrieval distribution relative to unbounded?",
        "",
    ]
    if best_lpips is not None:
        lines.append(f"- Best LPIPS: `{best_lpips['run_name']}` at {fmt(best_lpips.get('lpips_alex'))}.")
    if best_fvd is not None:
        lines.append(f"- Best FVD: `{best_fvd['run_name']}` at {fmt(best_fvd.get('fvd'))}.")
    if not metrics_available:
        lines.append("- Metrics CSVs were not available, so this report only contains trace-side driver shifts.")
    lines.extend(
        [
            "",
            "## Run-Level Retrieval Behavior",
            "",
            markdown_table(
                run_summary,
                [
                    "run_name",
                    "candidate_count_mean",
                    "selected_age_mean",
                    "old_selected_frac_age_gt_304",
                    "selected_overlap_mean",
                    "top1_selected_frac",
                    "reuse_gini",
                ],
                limit=30,
            ),
            "",
            "## Paired Target Shift Vs Unbounded",
            "",
            "For the same video, section, context slot, and target frame, this compares the bounded-selected reference to the unbounded-selected reference.",
            "",
            markdown_table(
                paired_sorted,
                [
                    "run_name",
                    "paired_targets",
                    "mean_age_delta_vs_unbounded",
                    "policy_newer_ref_rate",
                    "mean_overlap_delta_vs_unbounded",
                    "policy_lower_overlap_rate",
                    "mean_candidate_count_delta_vs_unbounded",
                ],
                limit=30,
            ),
            "",
        ]
    )
    if driver_with_lpips:
        lines.extend(
            [
                "## Biggest Per-Video LPIPS Wins",
                "",
                markdown_table(
                    driver_with_lpips,
                    [
                        "run_name",
                        "row",
                        "scene",
                        "baseline_lpips_alex",
                        "policy_lpips_alex",
                        "lpips_alex_improvement_pct_vs_unbounded",
                        "selected_age_mean_delta_vs_unbounded",
                        "old_selected_frac_age_gt_304_delta_vs_unbounded",
                    ],
                    limit=20,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## How To Use This",
            "",
            "- If LPIPS wins line up with lower candidate count, lower stale-reference rate, or lower selected-age p90, the story is memory regularization.",
            "- If LPIPS wins happen despite lower camera overlap, that means overlap-only retrieval is not the full objective.",
            "- If a policy improves FVD but not LPIPS, emphasize distribution-level visual plausibility and follow with CUT3R for camera fidelity.",
            "",
            "## Files",
            "",
            "- `figures/01_quality_headline.png`",
            "- `figures/02_reference_age_shift_vs_unbounded.png`",
            "- `figures/03_stale_reference_rate.png`",
            "- `figures/04_age_overlap_tradeoff.png`",
            "- `figures/05_lpips_driver_scatter.png` if metrics are available",
            "- `tables/run_trace_summary.csv`",
            "- `tables/paired_target_deltas.csv`",
            "- `tables/paired_run_summary.csv`",
            "- `tables/video_driver_quality.csv` if metrics are available",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze why bounded MemCam memory differs from an unbounded baseline."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metrics_dir", type=Path, default=None)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--durations", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--trace_dir_name", type=str, default="access_traces")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    runs = sorted(parse_list(args.runs), key=run_sort_key)
    trace_dirs = discover_trace_dirs(
        root=args.root,
        runs=runs,
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

    duration_filter = parse_int_list(args.durations) if args.durations else [args.duration]
    rows = filter_rows(rows, durations=duration_filter, row_filter=parse_rows(args.rows))
    if not rows:
        raise RuntimeError("No trace rows loaded after filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    video_trace = trace_video_summary(rows)
    run_trace = trace_run_summary(video_trace)
    target_deltas = paired_target_rows(rows, baseline_run=args.baseline_run)
    target_summary = paired_run_summary(target_deltas)

    quality_video, quality_summary = load_quality(args.metrics_dir, runs, args.duration)
    driver_rows = quality_join(video_trace, quality_video, baseline_run=args.baseline_run)
    metrics_available = bool(quality_video or quality_summary)

    write_csv(tables_dir / "video_trace_summary.csv", video_trace)
    write_csv(tables_dir / "run_trace_summary.csv", run_trace)
    write_csv(tables_dir / "paired_target_deltas.csv", target_deltas)
    write_csv(tables_dir / "paired_run_summary.csv", target_summary)
    write_csv(tables_dir / "quality_summary.csv", quality_summary)
    write_csv(tables_dir / "video_driver_quality.csv", driver_rows)

    figure_paths = [
        save_quality_bars(quality_summary, figures_dir, args.baseline_run),
        save_age_shift_plot(target_summary, figures_dir),
        save_stale_rate_plot(run_trace, figures_dir, args.baseline_run),
        save_overlap_tradeoff_plot(target_summary, figures_dir),
        save_lpips_driver_scatter(driver_rows, figures_dir),
    ]

    write_report(
        path=args.output_dir / "report.md",
        quality_summary=quality_summary,
        run_summary=run_trace,
        paired_summary=target_summary,
        driver_rows=driver_rows,
        baseline_run=args.baseline_run,
        metrics_available=metrics_available,
    )

    print(f"Wrote: {args.output_dir / 'report.md'}")
    for path in figure_paths:
        if path is not None:
            print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
