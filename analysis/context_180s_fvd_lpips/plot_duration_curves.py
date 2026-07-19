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
        help="Directory containing one <run>/summary.json per variant.",
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        help="Single JSON file containing summaries keyed by run name.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_summary(metrics_dir, summaries, run_name):
    if summaries is not None:
        return summaries.get(run_name)
    if metrics_dir is None:
        return None
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


def plot_metric(metric, metrics_dir, summaries, output_dir):
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

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.78)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plotted = 0
    plotted_families = set()
    plotted_budgets = set()

    for family_key, family in FAMILIES.items():
        for budget, style in BUDGET_STYLES.items():
            run_name = family["run"].format(budget=budget)
            values = duration_values(load_summary(metrics_dir, summaries, run_name), metric)
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
            plotted_families.add(family_key)
            plotted_budgets.add(budget)

    baseline_summary = load_summary(metrics_dir, summaries, "baseline")
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
        raise RuntimeError(f"No {metric} duration values found")

    fig.suptitle(
        f"{config['label']} vs. Video Duration",
        x=0.10,
        y=0.97,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    ax.set_xlabel("Video duration (seconds)")
    ax.set_ylabel(f"{config['label']} (lower is better)")
    ax.set_xticks(DURATIONS)
    ax.set_xlim(5, 185)
    ax.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILIES[family_key]["color"],
            linewidth=2.5,
            label=FAMILIES[family_key]["label"],
        )
        for family_key in FAMILIES
        if family_key in plotted_families
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
        if budget in plotted_budgets
    ]

    family_legend = ax.legend(
        handles=family_handles,
        title="Policy",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=len(family_handles),
        columnspacing=1.4,
        handlelength=2.5,
    )
    ax.add_artist(family_legend)
    ax.legend(
        handles=budget_handles,
        title="Memory budget",
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        ncol=len(budget_handles),
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
    summaries = None
    metrics_dir = args.metrics_dir.expanduser() if args.metrics_dir else None
    data_json = args.data_json
    if metrics_dir is None and data_json is None:
        data_json = Path(__file__).resolve().with_name("duration_metrics.json")
    if data_json is not None:
        with data_json.expanduser().open("r", encoding="utf-8") as handle:
            summaries = json.load(handle)
    for metric in METRICS:
        plot_metric(metric, metrics_dir, summaries, args.output_dir.expanduser())


if __name__ == "__main__":
    main()
