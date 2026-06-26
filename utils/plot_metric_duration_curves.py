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


def infer_run_name(summary_path):
    return summary_path.parent.name


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
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"duration_curve_{metric}.png"
    if any_values:
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
        "--metrics_dirs",
        type=str,
        required=True,
        help="Comma-separated directories to recursively search for per-run summary.json files.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Comma-separated run names. Use run=Label to override legend text.",
    )
    parser.add_argument("--run_labels", type=str, default=None)
    parser.add_argument("--durations", type=str, default="10,20,40,60")
    parser.add_argument("--metrics", type=str, default="fvd,lpips_alex")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title_prefix", type=str, default="")
    args = parser.parse_args()

    runs, labels = parse_runs(args.runs)
    labels.update(parse_label_overrides(args.run_labels))
    durations = parse_int_list(args.durations)
    metrics = parse_list(args.metrics)
    metrics_dirs = parse_list(args.metrics_dirs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = find_summary_paths(metrics_dirs)
    print(f"Found {len(summary_paths)} summary.json files")

    raw_rows = load_summary_rows(
        summary_paths=summary_paths,
        runs=runs,
        durations=durations,
        metrics=metrics,
    )
    rows = table_rows(raw_rows, runs=runs, labels=labels, durations=durations, metrics=metrics)
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
