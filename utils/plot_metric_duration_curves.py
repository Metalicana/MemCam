import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "memcam_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "memcam_xdg_cache"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required for duration-curve plots. Install it with: "
        "python -m pip install matplotlib"
    ) from exc


DEFAULT_LABELS = {
    "baseline": "Unbounded",
    "fifo_b16": "FIFO",
    "fifo_b32": "FIFO",
    "fifo_b64": "FIFO",
    "fifo_b128": "FIFO",
    "ri_b16_dino_rgb": "Mine",
    "ri_b32_dino_rgb": "Mine",
    "ri_b64_dino_rgb": "Mine",
    "ri_b128_dino_rgb": "Mine",
    "slam_b16_covisibility": "SLAM",
    "slam_b32_covisibility": "SLAM",
    "slam_b64_covisibility": "SLAM",
    "slam_b96_covisibility": "SLAM",
    "slam_b128_covisibility": "SLAM",
}

METRIC_LABELS = {
    "fvd": "FVD ↓",
    "lpips_alex": "LPIPS ↓",
    "dino_distance": "DINO distance ↓",
    "clip_image_distance": "CLIP image distance ↓",
    "psnr_db": "PSNR ↑",
    "ssim": "SSIM ↑",
    "rotation_error_deg_mean_mean": "CUT3R rotation error mean ↓",
    "rotation_error_deg_p90_mean": "CUT3R rotation error p90 ↓",
    "translation_error_scale_only_mean_mean": "CUT3R translation error mean ↓",
    "translation_error_scale_only_p90_mean": "CUT3R translation error p90 ↓",
    "translation_error_sim3_mean_mean": "CUT3R Sim3 translation error mean ↓",
    "translation_error_sim3_p90_mean": "CUT3R Sim3 translation error p90 ↓",
    "endpoint_rotation_error_deg_mean": "CUT3R endpoint rotation error ↓",
    "endpoint_translation_error_scale_only_mean": "CUT3R endpoint translation error ↓",
    "loop_endpoint_distance_error_mean": "CUT3R loop endpoint error ↓",
    "worldscore_camera_control_score_mean": "WorldScore-style camera score ↑",
}

COLORS = {
    "Unbounded": "#555555",
    "FIFO": "#2f7fbc",
    "SLAM": "#4f9b62",
    "Mine": "#d88c1f",
}


def parse_list(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value):
    return [int(part) for part in parse_list(value)]


def parse_runs(value):
    runs = []
    labels = {}
    for part in parse_list(value):
        if "=" in part:
            run_name, label = part.split("=", 1)
        elif ":" in part:
            run_name, label = part.split(":", 1)
        else:
            run_name, label = part, DEFAULT_LABELS.get(part, part)
        run_name = run_name.strip()
        label = label.strip()
        runs.append(run_name)
        labels[run_name] = label
    return runs, labels


def parse_label_overrides(value):
    labels = {}
    for part in parse_list(value):
        if "=" not in part and ":" not in part:
            raise ValueError(f"Bad label override '{part}'. Use run=Label.")
        if "=" in part:
            run_name, label = part.split("=", 1)
        else:
            run_name, label = part.split(":", 1)
        labels[run_name.strip()] = label.strip()
    return labels


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def run_sort_key(run_name):
    if run_name == "baseline":
        return (0, 0, run_name)
    match = re.search(r"_b(\d+)|b(\d+)", run_name)
    budget = int(next(group for group in match.groups() if group)) if match else 9999
    if run_name.startswith("fifo"):
        family = 1
    elif run_name.startswith("slam"):
        family = 2
    elif run_name.startswith("ri"):
        family = 3
    else:
        family = 9
    return (family, budget, run_name)


def find_summary_paths(metrics_dirs):
    paths = []
    for text in metrics_dirs:
        path = Path(text).expanduser()
        if path.name == "summary.json" and path.exists():
            paths.append(path)
        elif path.exists():
            paths.extend(sorted(path.glob("**/summary.json")))
        else:
            print(f"[warn] metrics path does not exist: {path}")
    return sorted(set(paths))


def find_cut3r_summary_paths(metrics_dirs):
    paths = []
    for text in metrics_dirs:
        path = Path(text).expanduser()
        if path.name == "cut3r_camera_summary.csv" and path.exists():
            paths.append(path)
        elif path.exists():
            paths.extend(sorted(path.glob("**/cut3r_camera_summary.csv")))
        else:
            print(f"[warn] metrics path does not exist: {path}")
    return sorted(set(paths))


def infer_run_name(summary_path):
    return summary_path.parent.name


def infer_duration_from_path(path):
    for part in [*path.parts[::-1], path.name]:
        match = re.search(r"context_memory_(\d+)s", part)
        if match:
            return int(match.group(1))
        match = re.search(r"(^|_)(\d+)s($|_)", part)
        if match:
            return int(match.group(2))
    return None


def load_summary_rows(summary_paths, runs, durations, metrics):
    run_set = set(runs)
    duration_set = set(str(duration) for duration in durations)
    rows_by_key = {}

    for summary_path in summary_paths:
        run_name = infer_run_name(summary_path)
        if run_name not in run_set:
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception as exc:
            print(f"[warn] could not read {summary_path}: {exc!r}")
            continue

        by_duration = summary.get("by_duration", {})
        for duration_text, duration_summary in by_duration.items():
            if duration_text not in duration_set:
                continue
            for metric in metrics:
                value = safe_float(duration_summary.get(metric))
                if value is None:
                    continue
                key = (run_name, int(duration_text), metric)
                candidate = {
                    "run_name": run_name,
                    "duration_sec": int(duration_text),
                    "metric": metric,
                    "value": value,
                    "videos": duration_summary.get("videos"),
                    "completed_or_short": duration_summary.get("completed_or_short"),
                    "frames_evaluated": duration_summary.get("frames_evaluated"),
                    "source_path": str(summary_path),
                    "_mtime": summary_path.stat().st_mtime,
                }
                existing = rows_by_key.get(key)
                if existing is None or candidate["_mtime"] >= existing["_mtime"]:
                    rows_by_key[key] = candidate

    rows = []
    for key in sorted(rows_by_key, key=lambda item: (item[2], run_sort_key(item[0]), item[1])):
        row = dict(rows_by_key[key])
        row.pop("_mtime", None)
        rows.append(row)
    return rows


def describe_quality_summaries(summary_paths):
    rows = []
    for summary_path in summary_paths:
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception as exc:
            rows.append(
                {
                    "source_path": str(summary_path),
                    "run_name": infer_run_name(summary_path),
                    "error": repr(exc),
                }
            )
            continue

        by_duration = summary.get("by_duration", {})
        metric_config = summary.get("metric_config", {})
        learned_metrics = metric_config.get("learned_metrics", [])
        video_distribution_metrics = metric_config.get("video_distribution_metrics", [])
        metric_names = set()
        for duration_summary in by_duration.values():
            for key, value in duration_summary.items():
                if safe_float(value) is not None:
                    metric_names.add(key)
        rows.append(
            {
                "source_path": str(summary_path),
                "run_name": infer_run_name(summary_path),
                "durations": ",".join(sorted(by_duration.keys(), key=lambda item: int(item))),
                "metrics_present": ",".join(sorted(metric_names)),
                "learned_metrics_config": ",".join(learned_metrics),
                "video_distribution_metrics_config": ",".join(video_distribution_metrics),
            }
        )
    return rows


def load_cut3r_rows(summary_paths, runs, durations, metrics, fallback_duration):
    run_set = set(runs)
    duration_set = set(int(duration) for duration in durations)
    rows_by_key = {}

    for summary_path in summary_paths:
        duration = infer_duration_from_path(summary_path)
        if duration is None:
            duration = fallback_duration
        if duration is None:
            print(f"[warn] could not infer duration for {summary_path}; pass --cut3r_duration")
            continue
        if duration not in duration_set:
            continue

        try:
            with summary_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                summary_rows = list(reader)
        except Exception as exc:
            print(f"[warn] could not read {summary_path}: {exc!r}")
            continue

        for summary_row in summary_rows:
            run_name = summary_row.get("run_name")
            if run_name not in run_set:
                continue
            for metric in metrics:
                value = safe_float(summary_row.get(metric))
                if value is None:
                    continue
                key = (run_name, duration, metric)
                candidate = {
                    "run_name": run_name,
                    "duration_sec": duration,
                    "metric": metric,
                    "value": value,
                    "videos": summary_row.get("videos"),
                    "completed_or_short": summary_row.get("videos"),
                    "frames_evaluated": summary_row.get("sampled_frames_mean"),
                    "source_path": str(summary_path),
                    "_mtime": summary_path.stat().st_mtime,
                }
                existing = rows_by_key.get(key)
                if existing is None or candidate["_mtime"] >= existing["_mtime"]:
                    rows_by_key[key] = candidate

    rows = []
    for key in sorted(rows_by_key, key=lambda item: (item[2], run_sort_key(item[0]), item[1])):
        row = dict(rows_by_key[key])
        row.pop("_mtime", None)
        rows.append(row)
    return rows


def describe_cut3r_summaries(summary_paths):
    rows = []
    for summary_path in summary_paths:
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)
        except Exception as exc:
            rows.append({"source_path": str(summary_path), "error": repr(exc)})
            continue
        metric_names = set()
        for row in csv_rows:
            for key, value in row.items():
                if key != "run_name" and safe_float(value) is not None:
                    metric_names.add(key)
        rows.append(
            {
                "source_path": str(summary_path),
                "duration_inferred": infer_duration_from_path(summary_path),
                "runs": ",".join(row.get("run_name", "") for row in csv_rows),
                "metrics_present": ",".join(sorted(metric_names)),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
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


def print_inventory(source_rows, requested_runs, requested_metrics, requested_durations):
    if not source_rows:
        print("[diagnostic] No summary files were discovered in --metrics_dirs.")
        return

    print("[diagnostic] Discovered summary files:")
    for row in source_rows:
        path = row.get("source_path", "")
        run_name = row.get("run_name") or row.get("runs") or "unknown"
        durations = row.get("durations") or row.get("duration_inferred") or ""
        metrics = row.get("metrics_present", "")
        learned = row.get("learned_metrics_config", "")
        video_metrics = row.get("video_distribution_metrics_config", "")
        suffix_parts = []
        if learned:
            suffix_parts.append(f"learned_config={learned}")
        if video_metrics:
            suffix_parts.append(f"video_config={video_metrics}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        print(f"  - run={run_name} durations={durations} metrics={metrics}{suffix}")
        print(f"    {path}")

    rows_by_run = {}
    for row in source_rows:
        run_name = row.get("run_name")
        if run_name:
            rows_by_run.setdefault(run_name, []).append(row)

    missing_runs = [run_name for run_name in requested_runs if run_name not in rows_by_run]
    if missing_runs:
        print(
            "[diagnostic] Requested runs with no discovered summary.json: "
            + ", ".join(missing_runs)
        )

    for run_name in requested_runs:
        run_rows = rows_by_run.get(run_name, [])
        if not run_rows:
            continue
        available_durations = set()
        available_metrics = set()
        for row in run_rows:
            available_durations.update(parse_list(row.get("durations", "")))
            available_metrics.update(parse_list(row.get("metrics_present", "")))
        for metric in requested_metrics:
            if metric not in available_metrics:
                print(
                    f"[diagnostic] {run_name}: missing metric '{metric}'. "
                    f"Available metrics: {', '.join(sorted(available_metrics)) or 'none'}"
                )
            missing_durations = [
                str(duration)
                for duration in requested_durations
                if str(duration) not in available_durations
            ]
            if missing_durations:
                print(
                    f"[diagnostic] {run_name}: missing requested durations "
                    f"{', '.join(missing_durations)}. "
                    f"Available durations: {', '.join(sorted(available_durations, key=int)) or 'none'}"
                )


def table_rows(rows, runs, labels, durations, metrics):
    by_key = {
        (row["run_name"], int(row["duration_sec"]), row["metric"]): row
        for row in rows
    }
    output = []
    for metric in metrics:
        for run_name in runs:
            for duration in durations:
                row = by_key.get((run_name, duration, metric))
                output.append(
                    {
                        "metric": metric,
                        "run_name": run_name,
                        "label": labels.get(run_name, run_name),
                        "duration_sec": duration,
                        "value": row.get("value") if row else None,
                        "videos": row.get("videos") if row else None,
                        "source_path": row.get("source_path") if row else None,
                    }
                )
    return output


def setup_axis(ax, title=None):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")


def save_metric_plot(rows, runs, labels, durations, metric, output_dir, title_prefix):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    any_values = False
    for run_name in runs:
        label = labels.get(run_name, run_name)
        values = []
        for duration in durations:
            value = next(
                (
                    safe_float(row.get("value"))
                    for row in rows
                    if row["run_name"] == run_name
                    and int(row["duration_sec"]) == duration
                    and row["metric"] == metric
                ),
                None,
            )
            values.append(value)
        if not any(value is not None for value in values):
            print(f"[warn] no {metric} values found for {run_name}")
            continue
        any_values = True
        ax.plot(
            durations,
            values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=label,
            color=COLORS.get(label),
        )

    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks(durations)
    setup_axis(ax, f"{title_prefix}{METRIC_LABELS.get(metric, metric)} vs duration")
    path = output_dir / f"duration_curve_{metric}.png"
    if any_values:
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        print(f"Wrote: {path}")
    plt.close(fig)
    return path if any_values else None


def save_combined_plot(rows, runs, labels, durations, metrics, output_dir, title_prefix):
    metrics = [
        metric
        for metric in metrics
        if any(row["metric"] == metric and row.get("value") is not None for row in rows)
    ]
    if not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(8.0 * len(metrics), 5.2), squeeze=False)
    axes = axes[0]
    for ax, metric in zip(axes, metrics):
        for run_name in runs:
            label = labels.get(run_name, run_name)
            values = []
            for duration in durations:
                value = next(
                    (
                        safe_float(row.get("value"))
                        for row in rows
                        if row["run_name"] == run_name
                        and int(row["duration_sec"]) == duration
                        and row["metric"] == metric
                    ),
                    None,
                )
                values.append(value)
            if any(value is not None for value in values):
                ax.plot(
                    durations,
                    values,
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    label=label,
                    color=COLORS.get(label),
                )
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(durations)
        setup_axis(ax, METRIC_LABELS.get(metric, metric))
    axes[-1].legend(frameon=False)
    fig.suptitle(f"{title_prefix}duration curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "duration_curves_combined.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Plot FVD/LPIPS curves across Context-as-Memory durations."
    )
    parser.add_argument(
        "--source",
        choices=("quality", "cut3r"),
        default="quality",
        help="quality reads evaluate_context_memory summary.json; cut3r reads cut3r_camera_summary.csv.",
    )
    parser.add_argument(
        "--metrics_dirs",
        type=str,
        required=True,
        help="Comma-separated directories to recursively search for summary files.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Comma-separated run names. Use run=Label to override legend text.",
    )
    parser.add_argument("--run_labels", type=str, default=None)
    parser.add_argument("--durations", type=str, default="10,20,40,60")
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument(
        "--cut3r_duration",
        type=int,
        default=None,
        help="Fallback duration for CUT3R summary CSVs when duration cannot be inferred from the path.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title_prefix", type=str, default="")
    args = parser.parse_args()

    runs, labels = parse_runs(args.runs)
    labels.update(parse_label_overrides(args.run_labels))
    durations = parse_int_list(args.durations)
    if args.metrics:
        metrics = parse_list(args.metrics)
    elif args.source == "cut3r":
        metrics = [
            "rotation_error_deg_mean_mean",
            "translation_error_scale_only_mean_mean",
            "translation_error_sim3_mean_mean",
            "endpoint_rotation_error_deg_mean",
            "loop_endpoint_distance_error_mean",
        ]
    else:
        metrics = ["fvd", "lpips_alex"]
    metrics_dirs = parse_list(args.metrics_dirs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "cut3r":
        summary_paths = find_cut3r_summary_paths(metrics_dirs)
        print(f"Found {len(summary_paths)} cut3r_camera_summary.csv files")
        source_rows = describe_cut3r_summaries(summary_paths)
        write_csv(args.output_dir / "discovered_summary_files.csv", source_rows)
        raw_rows = load_cut3r_rows(
            summary_paths=summary_paths,
            runs=runs,
            durations=durations,
            metrics=metrics,
            fallback_duration=args.cut3r_duration,
        )
    else:
        summary_paths = find_summary_paths(metrics_dirs)
        print(f"Found {len(summary_paths)} summary.json files")
        source_rows = describe_quality_summaries(summary_paths)
        write_csv(args.output_dir / "discovered_summary_files.csv", source_rows)
        raw_rows = load_summary_rows(
            summary_paths=summary_paths,
            runs=runs,
            durations=durations,
            metrics=metrics,
        )
    rows = table_rows(raw_rows, runs=runs, labels=labels, durations=durations, metrics=metrics)
    if source_rows:
        print(f"Wrote: {args.output_dir / 'discovered_summary_files.csv'}")
    print_inventory(source_rows, runs, metrics, durations)
    write_csv(args.output_dir / "duration_metric_values.csv", rows)
    print(f"Wrote: {args.output_dir / 'duration_metric_values.csv'}")

    title_prefix = args.title_prefix
    if title_prefix and not title_prefix.endswith(" "):
        title_prefix += " "

    for metric in metrics:
        save_metric_plot(
            rows=rows,
            runs=runs,
            labels=labels,
            durations=durations,
            metric=metric,
            output_dir=args.output_dir,
            title_prefix=title_prefix,
        )
    save_combined_plot(
        rows=rows,
        runs=runs,
        labels=labels,
        durations=durations,
        metrics=metrics,
        output_dir=args.output_dir,
        title_prefix=title_prefix,
    )


if __name__ == "__main__":
    main()
