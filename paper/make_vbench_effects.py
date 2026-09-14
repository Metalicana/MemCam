#!/usr/bin/env python3
"""Plot paired effects from diagnose_vbench_comparison.py's Markdown report.

Usage: python paper/make_vbench_effects.py --report /path/to/report.md
Uses one coherent paired audit; do not substitute unpaired fresh-run means.
"""

import argparse
import ast
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import FIGURES, configure, save


DIMENSIONS = {
    "subject_consistency": "Subject consistency",
    "background_consistency": "Background consistency",
    "motion_smoothness": "Motion smoothness",
    "dynamic_degree": "Dynamic degree",
    "aesthetic_quality": "Aesthetic quality",
    "imaging_quality": "Imaging quality",
}
COMPARATORS = {"baseline": "Unbounded", "fifo_b32": "FIFO-32"}
STYLES = {"standard": ("VBench", "#0072B2", "o", -0.17),
          "long": ("VBench-Long", "#B44B71", "D", 0.17)}


def load_report(path):
    records = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            continue
        comparator, dimension, component, n, delta, interval, _ = cells
        if comparator not in COMPARATORS or dimension not in DIMENSIONS or component not in STYLES:
            continue
        if int(n) != 15:
            raise ValueError("Expected 15 paired videos for every plotted effect")
        low, high = ast.literal_eval(interval)
        effect = float(delta)
        if not low <= effect <= high:
            raise ValueError(f"Invalid interval: {line}")
        key = (comparator, dimension, component)
        if key in records:
            raise ValueError(f"Duplicate effect: {key}")
        records[key] = [100 * value for value in (effect, low, high)]
    expected = {(c, d, s) for c in COMPARATORS for d in DIMENSIONS for s in STYLES}
    if set(records) != expected:
        raise ValueError(f"Missing effects: {expected - set(records)}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = load_report(args.report)
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.22, top=0.76, wspace=0.38)
    fig.suptitle("Where GeoCov Helps, and Where It Trades Off", y=0.98, fontsize=17, weight="bold")
    fig.text(0.5, 0.925, "15 matched 60-second videos  |  GeoCov budget: 32 frames",
             ha="center", fontsize=10, color="#555555")
    handles = [Line2D([0], [0], color=color, marker=marker, linewidth=1.6, label=label)
               for label, color, marker, _ in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=2, fontsize=10, columnspacing=3)
    for ax, (comparator, label) in zip(axes, COMPARATORS.items()):
        ax.set_xlim(-3, 10)
        ax.set_ylim(5.65, -0.65)
        ax.set_title(f"GeoCov-32 vs. {label}", fontsize=12, pad=27)
        ax.axvline(0, color="#45494d", linewidth=1)
        ax.set_xticks([-2, 0, 2, 4, 6, 8, 10])
        ax.grid(axis="x", color="#e6e8ea", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_yticks(range(6), DIMENSIONS.values(), fontsize=10)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_xlabel("GeoCov minus comparator (score points)", labelpad=10)
        ax.text(0, 1.015, "GeoCov lower", transform=ax.transAxes, fontsize=8, color="#666666")
        ax.text(1, 1.015, "GeoCov higher", ha="right", transform=ax.transAxes,
                fontsize=8, color="#666666")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        for index, dimension in enumerate(DIMENSIONS):
            if index % 2 == 0:
                ax.axhspan(index - 0.48, index + 0.48, facecolor="#f5f6f7", zorder=0)
            for component, (_, color, marker, offset) in STYLES.items():
                effect, low, high = records[comparator, dimension, component]
                ax.errorbar(effect, index + offset, xerr=[[effect - low], [high - effect]],
                            color=color, marker=marker, markersize=5, linewidth=1.6,
                            capsize=3, zorder=3)
                ax.text(1.025, index + offset, f"{effect:+.2f}" if effect else "0.00",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=color)
    fig.text(0.5, 0.105, "Points: paired mean differences. Bars: trajectory-bootstrap 95% confidence intervals.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.069, "Exploratory intervals, unadjusted for multiple comparisons; a higher dynamic-degree score means more motion.",
             ha="center", fontsize=8, color="#555555")
    fig.text(0.5, 0.034, "Source: supplied matched-cohort audit (14 September 2026). Fresh GeoCov-only rerun is not mixed into paired estimates.",
             ha="center", fontsize=8, color="#555555")
    save(fig, "vbench_paired_effects_matched15_60s")
    with (FIGURES / "vbench_paired_effects_matched15_60s.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparator", "dimension", "evaluator", "delta_points", "ci_low", "ci_high"])
        for key, values in records.items():
            writer.writerow([*key, *values])


if __name__ == "__main__":
    main()
