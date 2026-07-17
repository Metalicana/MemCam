#!/usr/bin/env python3
"""Aggregate MemCam rollout profiles and plot latency versus peak GPU memory."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required. Install it with: python -m pip install matplotlib"
    ) from exc


DEFAULT_DURATIONS = (10, 20, 40, 60, 120, 180)
DEVICE_MARKERS = {"cpu": "o", "cuda": "^"}
DEVICE_LINESTYLES = {"cpu": (0, (4, 2)), "cuda": "-"}
RI_COLORS = {16: "#9ECAE1", 32: "#6BAED6", 64: "#3182BD", 128: "#08519C"}
OTHER_BUDGET_COLORS = {16: "#FDD49E", 32: "#FDBB84", 64: "#FC8D59", 128: "#D7301F"}


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_labels(value):
    labels = {}
    if not value:
        return labels
    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad label override '{part}'. Use run_name=Label.")
        run_name, label = part.split("=", 1)
        labels[run_name.strip()] = label.strip()
    return labels


def discover_profiles(inputs):
    paths = []
    for input_text in inputs:
        path = Path(input_text).expanduser()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            print(f"[warn] profile path does not exist: {path}")
    return sorted(set(paths))


def read_profile(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "section_profile":
                records.append(record)
    return records


def default_config_label(record):
    policy = record.get("memory_policy", "unknown")
    budget = record.get("memory_budget")
    bank_device = str(record.get("memory_bank_device", "cpu"))
    device = "GPU" if bank_device == "cuda" else "CPU"
    if policy == "unbounded":
        policy_label = "Unbounded"
    elif policy == "rarity_irreplaceability":
        policy_label = "RI"
    elif policy == "slam_covisibility":
        policy_label = "SLAM"
    elif policy == "kcenter_coreset":
        policy_label = "K-center"
    else:
        policy_label = policy.replace("_", " ").title()
    if budget is not None:
        policy_label = f"{policy_label}-B{budget}"
    return f"{policy_label}-{device}"


def config_key(record):
    return (
        record.get("run_name"),
        record.get("memory_policy"),
        record.get("memory_budget"),
        record.get("memory_bank_device"),
    )


def sample_key(record, path):
    return (
        record.get("row"),
        record.get("scene"),
        record.get("output_prefix"),
        str(path),
    )


def nearest_prefix_record(records, duration):
    if not records:
        return None
    max_duration = max(float(record.get("generated_seconds", 0.0)) for record in records)
    section_step = max_duration / max(len(records), 1)
    if duration > max_duration + max(section_step / 2.0, 0.5):
        return None
    return min(
        records,
        key=lambda record: abs(float(record.get("generated_seconds", 0.0)) - duration),
    )


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return median, float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


def config_color(point):
    policy = point.get("memory_policy")
    budget = point.get("memory_budget")
    if policy == "unbounded":
        return "#4D4D4D"
    if policy == "rarity_irreplaceability":
        return RI_COLORS.get(budget, "#0072B2")
    return OTHER_BUDGET_COLORS.get(budget, "#E69F00")


def aggregate_profiles(profile_paths, durations, label_overrides):
    samples = {}
    config_records = {}
    for path in profile_paths:
        records = read_profile(path)
        if not records:
            continue
        first = records[0]
        key = config_key(first)
        samples[(key, sample_key(first, path))] = records
        config_records[key] = first

    points = []
    for key, representative in sorted(
        config_records.items(),
        key=lambda item: default_config_label(item[1]),
    ):
        run_name = representative.get("run_name")
        label = label_overrides.get(run_name, default_config_label(representative))
        device = str(representative.get("memory_bank_device", "cpu"))
        config_samples = [
            records
            for (sample_config, _), records in samples.items()
            if sample_config == key
        ]
        for duration in durations:
            selected = [
                nearest_prefix_record(records, duration)
                for records in config_samples
            ]
            selected = [record for record in selected if record is not None]
            selected = [
                record
                for record in selected
                if record.get("latency_per_generated_second") is not None
                and record.get("peak_cuda_allocated_gb") is not None
            ]
            if not selected:
                continue

            x_values = [float(record["latency_per_generated_second"]) for record in selected]
            y_values = [float(record["peak_cuda_allocated_gb"]) for record in selected]
            x_median, x_q25, x_q75 = quantiles(x_values)
            y_median, y_q25, y_q75 = quantiles(y_values)
            points.append(
                {
                    "run_name": run_name,
                    "label": label,
                    "memory_policy": representative.get("memory_policy"),
                    "memory_budget": representative.get("memory_budget"),
                    "memory_bank_device": device,
                    "duration_sec": duration,
                    "samples": len(selected),
                    "latency_median": x_median,
                    "latency_q25": x_q25,
                    "latency_q75": x_q75,
                    "peak_vram_median": y_median,
                    "peak_vram_q25": y_q25,
                    "peak_vram_q75": y_q75,
                }
            )
    return points


def pareto_frontier(points):
    frontier = []
    best_y = float("inf")
    for point in sorted(points, key=lambda item: (item["latency_median"], item["peak_vram_median"])):
        if point["peak_vram_median"] < best_y:
            frontier.append(point)
            best_y = point["peak_vram_median"]
    return frontier


def write_points_csv(points, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(points[0].keys()) if points else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(points)
    print(f"Wrote {path}")


def plot_points(points, output_dir, title):
    if not points:
        raise RuntimeError("No section profile records with latency and CUDA memory were found.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 17,
            "axes.labelsize": 12.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "text.color": "#202124",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    grouped = defaultdict(list)
    for point in points:
        grouped[point["label"]].append(point)

    labels = sorted(grouped)
    for label in labels:
        config_points = sorted(grouped[label], key=lambda item: item["duration_sec"])
        x = np.asarray([point["latency_median"] for point in config_points])
        y = np.asarray([point["peak_vram_median"] for point in config_points])
        xerr = np.asarray(
            [
                [point["latency_median"] - point["latency_q25"] for point in config_points],
                [point["latency_q75"] - point["latency_median"] for point in config_points],
            ]
        )
        yerr = np.asarray(
            [
                [point["peak_vram_median"] - point["peak_vram_q25"] for point in config_points],
                [point["peak_vram_q75"] - point["peak_vram_median"] for point in config_points],
            ]
        )
        marker = DEVICE_MARKERS.get(config_points[0]["memory_bank_device"], "o")
        linestyle = DEVICE_LINESTYLES.get(config_points[0]["memory_bank_device"], "-")
        color = config_color(config_points[0])
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=0,
            elinewidth=0.8,
            capsize=2.5,
            alpha=0.95,
            label=label,
            zorder=3,
        )
    max_duration = max(point["duration_sec"] for point in points)
    final_points = [point for point in points if point["duration_sec"] == max_duration]
    frontier = pareto_frontier(final_points)
    if len(frontier) > 1:
        ax.plot(
            [point["latency_median"] for point in frontier],
            [point["peak_vram_median"] for point in frontier],
            color="#111111",
            linestyle=(0, (3, 3)),
            linewidth=1.2,
            alpha=0.75,
            label=f"{max_duration}s Pareto frontier",
            zorder=1,
        )

    ax.set_title(title, loc="left", pad=28, weight="bold")
    ax.text(
        0.0,
        1.015,
        "Prefixes: 10, 20, 40, 60, 120, 180 s; median and IQR; lower-left is better",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#555555",
        va="bottom",
    )
    ax.set_xlabel("Rollout latency (compute-seconds per generated video-second)")
    ax.set_ylabel("Peak allocated GPU memory (GB)")
    ax.grid(color="#DADCE0", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best", ncol=2, columnspacing=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"latency_vs_peak_vram_pareto.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Profile JSONL files or directories containing them.",
    )
    parser.add_argument(
        "--durations",
        default=",".join(str(duration) for duration in DEFAULT_DURATIONS),
    )
    parser.add_argument("--labels", default=None, help="Overrides like run_name=Paper label.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", default="Latency vs. Peak GPU Memory")
    return parser.parse_args()


def main():
    args = parse_args()
    profile_paths = discover_profiles(args.profiles)
    if not profile_paths:
        raise RuntimeError("No profile JSONL files found.")
    points = aggregate_profiles(
        profile_paths,
        parse_int_list(args.durations),
        parse_labels(args.labels),
    )
    output_dir = args.output_dir.expanduser()
    write_points_csv(points, output_dir / "latency_vram_points.csv")
    plot_points(points, output_dir, args.title)


if __name__ == "__main__":
    main()
