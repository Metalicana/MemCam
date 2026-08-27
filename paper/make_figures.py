#!/usr/bin/env python3
"""Generate paper figures from the consolidated MemCam measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

COLORS = {
    "unbounded": "#4D4D4D",
    "fifo": "#D55E00",
    "ri": "#0072B2",
    "geo": "#4C8C4A",
    "view": "#56B4E9",
    "corruption": "#CC6677",
    "effective": "#7B61A8",
}


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202124",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


def add_box(ax, xy, width, height, title, detail, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.25,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=color,
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )


def arrow(ax, start, end, color="#555555"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
        )
    )


def plot_overview():
    fig, ax = plt.subplots(figsize=(10.4, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.02, 0.36),
        0.16,
        0.28,
        "Autoregressive\ngenerator",
        "Produces the next\nvideo chunk",
        "#333333",
    )
    add_box(
        ax,
        (0.25, 0.58),
        0.19,
        0.27,
        "Unbounded archive",
        "Every generated frame\nSize grows with time",
        COLORS["unbounded"],
    )
    add_box(
        ax,
        (0.25, 0.14),
        0.19,
        0.27,
        "Bounded curated bank",
        "At most B frames\nOnline insertion and eviction",
        COLORS["geo"],
    )
    add_box(
        ax,
        (0.52, 0.58),
        0.17,
        0.27,
        "Fixed retriever",
        "One memory per\ntarget view",
        COLORS["unbounded"],
    )
    add_box(
        ax,
        (0.52, 0.14),
        0.17,
        0.27,
        "Same retriever",
        "Only the candidate\nbank changes",
        COLORS["geo"],
    )
    add_box(
        ax,
        (0.77, 0.58),
        0.20,
        0.27,
        "Retrieved context",
        "More exposure to\ncorrupted memories",
        COLORS["corruption"],
    )
    add_box(
        ax,
        (0.77, 0.14),
        0.20,
        0.27,
        "Retrieved context",
        "Cleaner indices under\nRI and GeoCov",
        COLORS["ri"],
    )

    arrow(ax, (0.18, 0.55), (0.25, 0.70))
    arrow(ax, (0.18, 0.47), (0.25, 0.28))
    arrow(ax, (0.44, 0.715), (0.52, 0.715))
    arrow(ax, (0.44, 0.275), (0.52, 0.275))
    arrow(ax, (0.69, 0.715), (0.77, 0.715))
    arrow(ax, (0.69, 0.275), (0.77, 0.275))
    ax.text(
        0.505,
        0.96,
        "The archive is not the context: curation changes the retriever's candidate set",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    save(fig, "overview")


def plot_quality():
    labels = ["Unbounded", "FIFO-32", "RI-32", "GeoCov-32"]
    lpips = [0.597960, 0.651378, 0.593894, 0.587646]
    fvd = [734.220684, 677.281936, 550.395421, 476.617571]
    colors = [COLORS["unbounded"], COLORS["fifo"], COLORS["ri"], COLORS["geo"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), constrained_layout=True)
    for ax, values, title, ylabel, ylim, fmt in [
        (axes[0], lpips, "Perceptual error at 180 seconds", "LPIPS (lower is better)", (0.56, 0.66), "{:.3f}"),
        (axes[1], fvd, "Video distribution error at 180 seconds", "FVD (lower is better)", (400, 800), "{:.0f}"),
    ]:
        bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="white")
        ax.set_xticks(x, labels, rotation=16, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#DADCE0", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.018,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    save(fig, "quality_180s")


def plot_pool_growth():
    candidate_midpoints = np.array([377, 1023, 1707, 2372, 3018, 3683, 4367, 5013])
    selected = np.array([0.4497, 0.5149, 0.5249, 0.5517, 0.5672, 0.5926, 0.5476, 0.5653])
    hindsight = np.array([0.3075, 0.3242, 0.2907, 0.3079, 0.3056, 0.3367, 0.3091, 0.3239])
    gap = selected - hindsight

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.25), constrained_layout=True)
    ax = axes[0]
    ax.plot(candidate_midpoints, selected, marker="o", linewidth=2, color=COLORS["corruption"], label="Selected memory")
    ax.plot(candidate_midpoints, hindsight, marker="s", linewidth=2, color="#555555", label="Hindsight-best available")
    ax.set_xlabel("Mean candidate count in bin")
    ax.set_ylabel("Effective DINO mismatch")
    ax.set_title("Selected evidence worsens; best available stays flat", loc="left")
    ax.grid(axis="y", color="#DADCE0", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(candidate_midpoints, gap, marker="o", linewidth=2.2, color=COLORS["effective"])
    ax.fill_between(candidate_midpoints, 0, gap, color=COLORS["effective"], alpha=0.12)
    ax.set_xlabel("Mean candidate count in bin")
    ax.set_ylabel("Retrieval gap")
    ax.set_title("The selector leaves more quality on the table", loc="left")
    ax.grid(axis="y", color="#DADCE0", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "pool_growth")


def plot_mechanism():
    fig = plt.figure(figsize=(9.2, 3.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

    ax = fig.add_subplot(grid[0, 0])
    names = ["View\nmismatch", "Memory\ncorruption", "Effective\nmismatch"]
    values = np.array([-0.033506, 0.087348, 0.073157])
    ci_low = np.array([-0.051064, 0.023562, 0.016469])
    ci_high = np.array([-0.017066, 0.157153, 0.129560])
    errors = np.vstack([values - ci_low, ci_high - values])
    x = np.arange(3)
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=[COLORS["view"], COLORS["corruption"], COLORS["effective"]],
        edgecolor="white",
        width=0.68,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Late minus early (DINO distance)")
    ax.set_title("What deteriorates over time?", loc="left")
    ax.grid(axis="y", color="#DADCE0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    labels = ["Unbounded", "FIFO", "RI", "GeoCov"]
    colors = [COLORS["unbounded"], COLORS["fifo"], COLORS["ri"], COLORS["geo"]]
    for panel, values, ylabel, title, ylim in [
        (grid[0, 1], [11.703, 11.487, 13.478, 16.332], "PSNR (dB)", "Same-video selected PSNR", (10, 17.3)),
        (grid[0, 2], [0.3089, 0.2980, 0.3761, 0.4601], "SSIM", "Same-video selected SSIM", (0.25, 0.49)),
    ]:
        ax = fig.add_subplot(panel)
        x = np.arange(4)
        ax.bar(x, values, color=colors, edgecolor="white", width=0.7)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#DADCE0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "mechanism_evidence")


def plot_problem_scaling():
    durations = np.array([10, 40, 60, 600, 3600], dtype=float)
    duration_labels = ["10 s", "40 s", "60 s", "10 min", "60 min"]
    memory_gb = np.array([0.3839, 1.5320, 2.2974, 22.6757, 136.0477])
    retrieval_hours = np.array([15.8343, 321.4692, 740.6012, 75264.8063, 2719855.6413]) / 3600.0
    quality_duration = np.array([10, 20, 40, 60, 120, 180])
    lpips = np.array([0.488565, 0.543653, 0.575972, 0.581868, 0.592156, 0.597960])
    fvd = np.array([660.911119, 705.023332, 776.591535, 675.227025, 735.276610, 734.220684])

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)

    ax = axes[0]
    duration_x = np.arange(len(durations))
    quality_x = np.arange(len(quality_duration))

    ax.plot(duration_x, memory_gb, marker="o", linewidth=2.1, color=COLORS["unbounded"])
    ax.set_yscale("log")
    ax.set_xticks(duration_x, duration_labels, rotation=28, ha="right")
    ax.set_ylabel("Archive storage (GB)")
    ax.set_title("Linear storage growth", loc="left")

    ax = axes[1]
    ax.plot(duration_x, retrieval_hours, marker="o", linewidth=2.1, color=COLORS["corruption"])
    ax.set_yscale("log")
    ax.set_xticks(duration_x, duration_labels, rotation=28, ha="right")
    ax.set_ylabel("Retrieval time (hours)")
    ax.set_title("Quadratic search growth", loc="left")

    ax = axes[2]
    ax.plot(quality_x, lpips, marker="o", linewidth=2.1, color=COLORS["ri"])
    ax.set_xticks(quality_x, [str(int(v)) for v in quality_duration])
    ax.set_xlabel("Video duration (s)")
    ax.set_ylabel("LPIPS")
    ax.set_title("Perceptual drift", loc="left")

    ax = axes[3]
    ax.plot(quality_x, fvd, marker="o", linewidth=2.1, color=COLORS["effective"])
    ax.set_xticks(quality_x, [str(int(v)) for v in quality_duration])
    ax.set_xlabel("Video duration (s)")
    ax.set_ylabel("FVD")
    ax.set_title("Distribution error", loc="left")

    for ax in axes:
        ax.grid(axis="y", color="#DADCE0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "problem_scaling_quality")


def plot_model_architecture():
    fig, ax = plt.subplots(figsize=(11.6, 3.1))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.01, 0.57), 0.12, 0.23, "Existing bank", "At most B historical\nmemories", COLORS["geo"])
    add_box(ax, (0.01, 0.19), 0.12, 0.23, "New candidates", "Frames from the\nlatest chunk", "#333333")
    add_box(ax, (0.18, 0.35), 0.13, 0.30, "Candidate pool", "Existing + new\nmemories", COLORS["unbounded"])
    add_box(ax, (0.36, 0.35), 0.14, 0.30, "Pairwise affinity", "65% camera pose\n35% DINO", COLORS["ri"])
    add_box(ax, (0.55, 0.35), 0.14, 0.30, "Coverage utility", "Observer count +\nnearest substitute", COLORS["effective"])
    add_box(ax, (0.74, 0.35), 0.10, 0.30, "Evict low U", "Retain exactly\nB memories", COLORS["geo"])
    add_box(ax, (0.89, 0.57), 0.10, 0.23, "Retriever", "Original read\noperation", "#333333")
    add_box(ax, (0.89, 0.19), 0.10, 0.23, "Generator", "Next video\nchunk", COLORS["corruption"])

    arrow(ax, (0.13, 0.68), (0.18, 0.55))
    arrow(ax, (0.13, 0.30), (0.18, 0.45))
    arrow(ax, (0.31, 0.50), (0.36, 0.50))
    arrow(ax, (0.50, 0.50), (0.55, 0.50))
    arrow(ax, (0.69, 0.50), (0.74, 0.50))
    arrow(ax, (0.84, 0.54), (0.89, 0.65))
    arrow(ax, (0.94, 0.57), (0.94, 0.42))

    ax.text(
        0.5,
        0.94,
        "Fixed-budget Geometric Coverage memory controller",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        0.5,
        0.04,
        "Only the candidate bank changes; the retriever and video generator remain frozen",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )
    save(fig, "model_architecture")


def main():
    configure()
    plot_overview()
    plot_quality()
    plot_pool_growth()
    plot_mechanism()
    plot_problem_scaling()
    plot_model_architecture()


if __name__ == "__main__":
    main()
