#!/usr/bin/env python3
"""Analyze measured and projected MemCam memory/latency scaling."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GIB = 1024 ** 3
DEFAULT_OBSERVED_DURATIONS = (10, 40, 60)
DEFAULT_PROJECTION_DURATIONS = (600, 3600)
POLICY_COLORS = {
    "unbounded": "#4D4D4D",
    "rarity_irreplaceability": "#0072B2",
    "fifo": "#D55E00",
    "slam_covisibility": "#6B8E23",
    "kcenter_coreset": "#E69F00",
}
DEVICE_MARKERS = {"cpu": "o", "cuda": "^"}


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
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "section_profile":
                records.append(record)
    return derive_section_values(records)


def derive_section_values(records):
    derived = []
    previous_duration = 0.0
    for record in sorted(records, key=lambda item: float(item.get("generated_seconds", 0.0))):
        duration = float(record.get("generated_seconds", 0.0))
        delta_duration = duration - previous_duration
        item = dict(record)
        bank_frame_gb = item.get("bank_frame_gb")
        if bank_frame_gb is None:
            bank_frame_gb = float(item.get("bank_frame_bytes", 0)) / GIB
        bank_feature_gb = item.get("bank_feature_gb")
        if bank_feature_gb is None:
            bank_feature_gb = float(item.get("bank_feature_bytes", 0)) / GIB
        item["memory_bank_gb"] = float(bank_frame_gb) + float(bank_feature_gb)
        item["average_latency_s_per_s"] = (
            float(item["cumulative_rollout_latency_s"]) / duration
            if duration > 0 and item.get("cumulative_rollout_latency_s") is not None
            else item.get("latency_per_generated_second")
        )
        item["local_latency_s_per_s"] = (
            float(item["section_latency_s"]) / delta_duration
            if delta_duration > 0 and item.get("section_latency_s") is not None
            else None
        )
        derived.append(item)
        previous_duration = duration
    return derived


def default_config_label(record):
    policy = record.get("memory_policy", "unknown")
    budget = record.get("memory_budget")
    device = str(record.get("memory_bank_device", "cpu")).upper()
    names = {
        "unbounded": "Unbounded",
        "rarity_irreplaceability": "RI",
        "fifo": "FIFO",
        "slam_covisibility": "SLAM",
        "kcenter_coreset": "K-center",
    }
    label = names.get(policy, policy.replace("_", " ").title())
    if budget is not None:
        label = f"{label}-B{budget}"
    return f"{label}-{device}"


def config_key(record):
    return (
        record.get("run_name"),
        record.get("memory_policy"),
        record.get("memory_budget"),
        record.get("memory_bank_device"),
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
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


AGGREGATE_FIELDS = (
    "memory_bank_gb",
    "average_latency_s_per_s",
    "local_latency_s_per_s",
    "cumulative_rollout_latency_s",
    "stored_memory_size",
    "candidate_count",
    "rss_gb",
    "cuda_allocated_gb",
    "peak_cuda_allocated_gb",
)


def aggregate_profiles(profile_paths, durations, label_overrides):
    samples = {}
    representatives = {}
    for path in profile_paths:
        records = read_profile(path)
        if not records:
            continue
        key = config_key(records[0])
        samples[(key, str(path))] = records
        representatives[key] = records[0]

    points = []
    for key, representative in sorted(
        representatives.items(), key=lambda item: default_config_label(item[1])
    ):
        run_name = representative.get("run_name")
        label = label_overrides.get(run_name, default_config_label(representative))
        config_samples = [records for (sample_key, _), records in samples.items() if sample_key == key]
        for duration in durations:
            selected = [nearest_prefix_record(records, duration) for records in config_samples]
            selected = [record for record in selected if record is not None]
            if not selected:
                continue
            point = {
                "run_name": run_name,
                "label": label,
                "memory_policy": representative.get("memory_policy"),
                "memory_budget": representative.get("memory_budget"),
                "memory_bank_device": representative.get("memory_bank_device"),
                "duration_sec": int(duration),
                "samples": len(selected),
            }
            for field in AGGREGATE_FIELDS:
                values = [float(record[field]) for record in selected if record.get(field) is not None]
                if not values:
                    point[f"{field}_median"] = None
                    point[f"{field}_q25"] = None
                    point[f"{field}_q75"] = None
                    continue
                median, q25, q75 = quantiles(values)
                point[f"{field}_median"] = median
                point[f"{field}_q25"] = q25
                point[f"{field}_q75"] = q75
            points.append(point)
    return points


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = float(np.sum((y_true - y_pred) ** 2))
    centered = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if centered == 0:
        return 1.0 if residual < 1e-12 else 0.0
    return 1.0 - residual / centered


def nonnegative_two_term_fit(first_column, second_column, y):
    first_column = np.asarray(first_column, dtype=np.float64)
    second_column = np.asarray(second_column, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([first_column, second_column])
    unconstrained, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    candidates = []
    if np.all(unconstrained >= 0):
        candidates.append(unconstrained)
    first_only = max(float(np.dot(first_column, y) / np.dot(first_column, first_column)), 0.0)
    second_only = max(float(np.dot(second_column, y) / np.dot(second_column, second_column)), 0.0)
    candidates.extend(
        [
            np.asarray([first_only, 0.0]),
            np.asarray([0.0, second_only]),
            np.zeros(2, dtype=np.float64),
        ]
    )
    return min(candidates, key=lambda coefficients: np.sum((y - design @ coefficients) ** 2))


def fit_models(config_points):
    x = np.asarray([point["duration_sec"] for point in config_points], dtype=np.float64)
    memory = np.asarray(
        [point["memory_bank_gb_median"] for point in config_points], dtype=np.float64
    )
    cumulative_latency = np.asarray(
        [point["cumulative_rollout_latency_s_median"] for point in config_points],
        dtype=np.float64,
    )

    budget = config_points[0].get("memory_budget")
    if budget is None:
        memory_coefficients = nonnegative_two_term_fit(np.ones_like(x), x, memory)
        memory_model = "nonnegative affine: intercept + slope*T"
    else:
        memory_coefficients = np.asarray([float(np.median(memory)), 0.0])
        memory_model = "constant after budget saturation"
    memory_prediction = memory_coefficients[0] + memory_coefficients[1] * x

    latency_coefficients = nonnegative_two_term_fit(x, x ** 2, cumulative_latency)
    latency_prediction = latency_coefficients[0] * x + latency_coefficients[1] * x ** 2
    return {
        "memory_intercept_gb": float(memory_coefficients[0]),
        "memory_slope_gb_per_s": float(memory_coefficients[1]),
        "memory_r2": r_squared(memory, memory_prediction),
        "memory_model": memory_model,
        "latency_linear_s_per_s": float(latency_coefficients[0]),
        "latency_quadratic_s_per_s2": float(latency_coefficients[1]),
        "latency_r2": r_squared(cumulative_latency, latency_prediction),
        "latency_model": "nonnegative cumulative: a*T + b*T^2",
    }


def group_points(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[config_key(point)].append(point)
    return grouped


def build_projections(points, durations):
    projections = []
    models = []
    for key, config_points in group_points(points).items():
        config_points = sorted(config_points, key=lambda point: point["duration_sec"])
        model = fit_models(config_points)
        representative = config_points[0]
        models.append(
            {
                "run_name": representative.get("run_name"),
                "label": representative["label"],
                "memory_policy": representative.get("memory_policy"),
                "memory_budget": representative.get("memory_budget"),
                "memory_bank_device": representative.get("memory_bank_device"),
                **model,
            }
        )
        for duration in durations:
            total_latency = (
                model["latency_linear_s_per_s"] * duration
                + model["latency_quadratic_s_per_s2"] * duration ** 2
            )
            projections.append(
                {
                    "run_name": representative.get("run_name"),
                    "label": representative["label"],
                    "memory_policy": representative.get("memory_policy"),
                    "memory_budget": representative.get("memory_budget"),
                    "memory_bank_device": representative.get("memory_bank_device"),
                    "duration_sec": int(duration),
                    "memory_bank_gb": (
                        model["memory_intercept_gb"]
                        + model["memory_slope_gb_per_s"] * duration
                    ),
                    "average_latency_s_per_s": total_latency / duration,
                    "total_rollout_latency_s": total_latency,
                    "total_rollout_latency_hours": total_latency / 3600.0,
                    "memory_r2": model["memory_r2"],
                    "latency_r2": model["latency_r2"],
                }
            )
    return projections, models


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {path}")


def config_color(point):
    return POLICY_COLORS.get(point.get("memory_policy"), "#CC79A7")


def duration_label(duration):
    if duration < 60:
        return f"{duration}s"
    if duration == 60:
        return "60s"
    return f"{duration // 60}m"


def positive_errors(median, q25, q75):
    lower = min(max(median - q25, 0.0), median * 0.99)
    upper = max(q75 - median, 0.0)
    return [[lower], [upper]]


def plot_scaling(points, projections, output_dir, metric):
    configs = group_points(points)
    projection_configs = group_points(projections)
    metric_config = {
        "memory": {
            "observed": "memory_bank_gb_median",
            "q25": "memory_bank_gb_q25",
            "q75": "memory_bank_gb_q75",
            "projected": "memory_bank_gb",
            "title": "Memory-Bank Scaling with Video Duration",
            "ylabel": "Retained memory-bank storage (GB)",
            "filename": "memory_bank_scaling",
            "subtitle": "Exact retained tensors and features; measured medians/IQR and linear projection",
        },
        "latency": {
            "observed": "average_latency_s_per_s_median",
            "q25": "average_latency_s_per_s_q25",
            "q75": "average_latency_s_per_s_q75",
            "projected": "average_latency_s_per_s",
            "title": "Rollout Latency Scaling with Video Duration",
            "ylabel": "Average compute seconds per generated video second",
            "filename": "rollout_latency_scaling",
            "subtitle": "Measured medians/IQR; projection from cumulative a*T + b*T^2 fit",
        },
    }[metric]

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
    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_durations = set()
    for key, config_points in configs.items():
        config_points = sorted(config_points, key=lambda point: point["duration_sec"])
        representative = config_points[0]
        color = config_color(representative)
        marker = DEVICE_MARKERS.get(str(representative.get("memory_bank_device")), "o")
        x = np.asarray([point["duration_sec"] for point in config_points], dtype=np.float64)
        y = np.asarray([point[metric_config["observed"]] for point in config_points])
        yerr = np.asarray(
            [
                [
                    min(
                        max(point[metric_config["observed"]] - point[metric_config["q25"]], 0.0),
                        point[metric_config["observed"]] * 0.99,
                    )
                    for point in config_points
                ],
                [
                    max(point[metric_config["q75"]] - point[metric_config["observed"]], 0.0)
                    for point in config_points
                ],
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            capsize=3,
            label=representative["label"],
            zorder=3,
        )
        projected = sorted(projection_configs.get(key, []), key=lambda point: point["duration_sec"])
        if projected:
            projected_x = [float(x[-1])] + [point["duration_sec"] for point in projected]
            projected_y = [float(y[-1])] + [point[metric_config["projected"]] for point in projected]
            ax.plot(
                projected_x,
                projected_y,
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=2.0,
                marker=marker,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.3,
                zorder=2,
            )
            all_durations.update(point["duration_sec"] for point in projected)
        all_durations.update(point["duration_sec"] for point in config_points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(all_durations)
    ax.set_xticks(ticks)
    ax.set_xticklabels([duration_label(duration) for duration in ticks])
    ax.set_title(metric_config["title"], loc="left", pad=28, weight="bold")
    ax.text(
        0.0,
        1.015,
        f"{metric_config['subtitle']}; both axes use log scales",
        transform=ax.transAxes,
        fontsize=9.3,
        color="#555555",
        va="bottom",
    )
    ax.set_xlabel("Generated video duration")
    ax.set_ylabel(metric_config["ylabel"])
    ax.grid(which="major", color="#DADCE0", linewidth=0.8, alpha=0.8)
    ax.grid(which="minor", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{metric_config['filename']}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


def write_report(points, projections, models, path):
    point_groups = group_points(points)
    projection_groups = group_points(projections)
    model_by_key = {config_key(model): model for model in models}
    lines = [
        "# MemCam Memory and Latency Scaling",
        "",
        "Measured values are medians across profile files; figure error bars show the IQR. "
        "Memory is the exact retained frame/feature bank, regardless of whether that bank is placed on CPU or GPU. "
        "Latency covers the generation rollout and excludes initial model loading and final video encoding.",
        "",
        "The 10-minute and 60-minute values are projections, not measurements. Unbounded memory uses a "
        "nonnegative affine fit. Cumulative latency uses a nonnegative `a*T + b*T^2` fit, corresponding "
        "to a rollout whose per-second cost may grow linearly with bank size.",
        "",
        "| Configuration | Samples | 60s bank GB | 10m bank GB | 60m bank GB | 60s latency s/s | 10m latency s/s | 60m latency s/s | Projected 60m rollout hours | Memory R2 | Latency R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, config_points in sorted(point_groups.items(), key=lambda item: item[1][0]["label"]):
        config_points = sorted(config_points, key=lambda point: point["duration_sec"])
        measured = config_points[-1]
        projected_by_duration = {
            point["duration_sec"]: point for point in projection_groups.get(key, [])
        }
        ten_minute = projected_by_duration.get(600)
        sixty_minute = projected_by_duration.get(3600)
        model = model_by_key[key]
        if ten_minute is None or sixty_minute is None:
            continue
        lines.append(
            "| {label} | {samples} | {memory_60:.3f} | {memory_600:.3f} | "
            "{memory_3600:.3f} | {latency_60:.2f} | {latency_600:.2f} | "
            "{latency_3600:.2f} | {hours:.1f} | {memory_r2:.3f} | {latency_r2:.3f} |".format(
                label=measured["label"],
                samples=measured["samples"],
                memory_60=measured["memory_bank_gb_median"],
                memory_600=ten_minute["memory_bank_gb"],
                memory_3600=sixty_minute["memory_bank_gb"],
                latency_60=measured["average_latency_s_per_s_median"],
                latency_600=ten_minute["average_latency_s_per_s"],
                latency_3600=sixty_minute["average_latency_s_per_s"],
                hours=sixty_minute["total_rollout_latency_hours"],
                memory_r2=model["memory_r2"],
                latency_r2=model["latency_r2"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Profile JSONL files or directories containing them.",
    )
    parser.add_argument(
        "--observed_durations",
        default=",".join(str(value) for value in DEFAULT_OBSERVED_DURATIONS),
    )
    parser.add_argument(
        "--projection_durations",
        default=",".join(str(value) for value in DEFAULT_PROJECTION_DURATIONS),
    )
    parser.add_argument("--labels", default=None, help="Overrides like run_name=Paper label.")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    profile_paths = discover_profiles(args.profiles)
    if not profile_paths:
        raise RuntimeError("No profile JSONL files found.")
    observed_durations = parse_int_list(args.observed_durations)
    projection_durations = parse_int_list(args.projection_durations)
    if len(observed_durations) < 3:
        raise ValueError("At least three observed durations are required for scaling fits.")
    points = aggregate_profiles(profile_paths, observed_durations, parse_labels(args.labels))
    if not points:
        raise RuntimeError("No usable section_profile records found.")
    projections, models = build_projections(points, projection_durations)
    output_dir = args.output_dir.expanduser()
    write_csv(points, output_dir / "observed_scaling_points.csv")
    write_csv(projections, output_dir / "projected_scaling_points.csv")
    write_csv(models, output_dir / "scaling_models.csv")
    plot_scaling(points, projections, output_dir, "memory")
    plot_scaling(points, projections, output_dir, "latency")
    write_report(points, projections, models, output_dir / "report.md")


if __name__ == "__main__":
    main()
