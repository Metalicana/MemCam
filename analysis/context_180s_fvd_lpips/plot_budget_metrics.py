#!/usr/bin/env python3
"""Plot 180-second LPIPS and FVD results across memory budgets."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


OUTPUT_DIR = Path(__file__).resolve().parent
BUDGETS = [16, 32, 64, 128]
BASELINE = {
    "lpips": 0.600412,
    "fvd": 797.478522,
}
SERIES = {
    "FIFO": {
        "lpips": [0.648025, 0.646340, 0.643354, 0.646540],
        "fvd": [640.823891, 677.288654, 801.958919, 667.191153],
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "RI (DINO-RGB)": {
        "lpips": [0.588183, 0.588744, 0.579334, 0.605207],
        "fvd": [518.263225, 550.394408, 585.383525, 839.878275],
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "K-center (DINO-pose)": {
        "lpips": [0.601113, 0.599763, 0.598066, 0.597386],
        "fvd": [551.341180, 551.804065, 550.418946, 557.186718],
        "color": "#E69F00",
        "marker": "D",
        "linestyle": "-.",
    },
    "SLAM (covisibility)": {
        "lpips": [0.587601, 0.587646, 0.586473, 0.591313],
        "fvd": [446.138360, 476.617571, 493.919568, 515.539624],
        "color": "#6B8E23",
        "marker": "^",
        "linestyle": "-",
    },
}


def plot_metric(metric, ylabel, filename, ylim, decimals):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202124",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, values in SERIES.items():
        ax.plot(
            BUDGETS,
            values[metric],
            label=label,
            color=values["color"],
            marker=values["marker"],
            linestyle=values["linestyle"],
            linewidth=2.3,
            markersize=7.5,
            markeredgewidth=1.1,
            markeredgecolor="white",
            zorder=3,
        )

    ax.axhline(
        BASELINE[metric],
        color="#4D4D4D",
        linestyle=(0, (2, 3)),
        linewidth=1.8,
        label=f"Unbounded ({BASELINE[metric]:.{decimals}f})",
        zorder=2,
    )

    ax.set_title(f"{ylabel} vs. Batch Size (180 s)", loc="left", pad=18, weight="bold")
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel(f"{ylabel} (lower is better)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(budget) for budget in BUDGETS])
    ax.set_xlim(14, 145)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.8,
    )

    for extension in ("png", "pdf"):
        output_path = OUTPUT_DIR / f"{filename}.{extension}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {output_path}")

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric(
        metric="lpips",
        ylabel="LPIPS",
        filename="lpips_vs_memory_budget_180s",
        ylim=(0.57, 0.655),
        decimals=3,
    )
    plot_metric(
        metric="fvd",
        ylabel="FVD",
        filename="fvd_vs_memory_budget_180s",
        ylim=(420, 870),
        decimals=0,
    )


if __name__ == "__main__":
    main()
