#!/usr/bin/env python3
"""Plot LPIPS and FVD against video duration for every policy-budget variant."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DURATIONS = [10, 20, 40, 60, 120, 180]
FAMILIES = {
    "fifo": {
        "label": "FIFO",
        "color": "#D55E00",
        "run": "fifo_b{budget}",
    },
    "ri": {
        "label": "RI",
        "color": "#0072B2",
        "run": "ri_b{budget}_dino_rgb",
    },
    "kcenter": {
        "label": "K-center",
        "color": "#E69F00",
        "run": "kcenter_b{budget}_dino_pose",
    },
    "slam": {
        "label": "SLAM",
        "color": "#6B8E23",
        "run": "slam_b{budget}_covisibility",
    },
}
BUDGET_STYLES = {
    16: {"linestyle": "-", "marker": "o"},
    32: {"linestyle": "--", "marker": "s"},
    64: {"linestyle": "-.", "marker": "D"},
    128: {"linestyle": (0, (1, 2)), "marker": "^"},
}
METRICS = {
    "lpips_alex": {
        "label": "LPIPS",
        "filename": "lpips_vs_duration_all_variants",
        "decimals": 3,
    },
    "fvd": {
        "label": "FVD",
        "filename": "fvd_vs_duration_all_variants",
        "decimals": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing one <run>/summary.json per variant.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_summary(metrics_dir, run_name):
    path = metrics_dir / run_name / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duration_values(summary, metric):
    if summary is None:
        return [None] * len(DURATIONS)
    by_duration = summary.get("by_duration", {})
    return [by_duration.get(str(duration), {}).get(metric) for duration in DURATIONS]


def plot_metric(metric, metrics_dir, output_dir):
    config = METRICS[metric]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202124",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plotted = 0

    for family in FAMILIES.values():
        for budget, style in BUDGET_STYLES.items():
            run_name = family["run"].format(budget=budget)
            values = duration_values(load_summary(metrics_dir, run_name), metric)
            points = [(duration, value) for duration, value in zip(DURATIONS, values) if value is not None]
            if not points:
                continue
            x_values, y_values = zip(*points)
            ax.plot(
                x_values,
                y_values,
                color=family["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.9,
                alpha=0.95,
                zorder=3,
            )
            plotted += 1

    baseline_summary = load_summary(metrics_dir, "baseline")
    baseline_values = duration_values(baseline_summary, metric)
    baseline_points = [
        (duration, value)
        for duration, value in zip(DURATIONS, baseline_values)
        if value is not None
    ]
    if baseline_points:
        x_values, y_values = zip(*baseline_points)
        ax.plot(
            x_values,
            y_values,
            color="#4D4D4D",
            linestyle=(0, (2, 3)),
            linewidth=2.0,
            zorder=2,
        )

    if plotted == 0 and not baseline_points:
        raise RuntimeError(f"No {metric} duration values found under {metrics_dir}")

    ax.set_title(f"{config['label']} vs. Video Duration", loc="left", pad=18, weight="bold")
    ax.set_xlabel("Video duration (seconds)")
    ax.set_ylabel(f"{config['label']} (lower is better)")
    ax.set_xticks(DURATIONS)
    ax.set_xlim(5, 185)
    ax.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    family_handles = [
        Line2D([0], [0], color=family["color"], linewidth=2.5, label=family["label"])
        for family in FAMILIES.values()
    ]
    if baseline_points:
        family_handles.append(
            Line2D(
                [0],
                [0],
                color="#4D4D4D",
                linestyle=(0, (2, 3)),
                linewidth=2.0,
                label="Unbounded",
            )
        )
    budget_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.5,
            label=f"B={budget}",
        )
        for budget, style in BUDGET_STYLES.items()
    ]

    family_legend = ax.legend(
        handles=family_handles,
        title="Policy",
        loc="upper left",
        frameon=False,
        ncol=len(family_handles),
        columnspacing=1.4,
        handlelength=2.5,
    )
    ax.add_artist(family_legend)
    ax.legend(
        handles=budget_handles,
        title="Memory budget",
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=1.4,
        handlelength=2.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{config['filename']}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    args = parse_args()
    for metric in METRICS:
        plot_metric(metric, args.metrics_dir.expanduser(), args.output_dir.expanduser())


if __name__ == "__main__":
    main()
