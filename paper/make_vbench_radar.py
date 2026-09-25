#!/usr/bin/env python3
"""Render matched VBench radar previews: python paper/make_vbench_radar.py."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import COLORS, FIGURES, configure, save


LABELS = [
    "Subject\nconsistency", "Background\nconsistency", "Motion\nsmoothness",
    "Dynamic\ndegree", "Aesthetic\nquality", "Imaging\nquality",
]
POLICIES = {"unbounded": "Unbounded", "fifo": "FIFO-32", "geo": "GeoCov-32"}
STYLES = {"unbounded": ("--", "o"), "fifo": (":", "s"), "geo": ("-", "D")}


def main():
    data = json.loads((Path(__file__).resolve().parent / "configs/vbench_radar_scores.json").read_text())
    dimensions = data["dimensions"]
    assert len(dimensions) == len(LABELS)
    assert data["cohort"] == {"videos": 15, "duration_seconds": 60, "bounded_budget": 32}
    configure()
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.19, top=0.79, wspace=0.48)
    fig.suptitle("Video Quality Under a Fixed Memory Budget", fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.925, "15 matched videos  |  60 seconds  |  bounded policies: B = 32",
             ha="center", fontsize=10, color="#555555")
    handles = []
    table_rows = []
    for index, (ax, (benchmark, scores)) in enumerate(zip(axes, data["scores"].items())):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_xticks(angles, LABELS, fontsize=10)
        ax.tick_params(axis="x", pad=14)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="#777777")
        ax.set_rlabel_position(30)
        ax.grid(color="#d8dcdf", linewidth=0.65)
        ax.spines["polar"].set_color("#bec5c9")
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, 0.89,
                 f"({'ab'[index]}) {benchmark}", ha="center", fontsize=12, weight="bold")
        for policy, label in POLICIES.items():
            values = np.asarray(scores[policy], dtype=float)
            assert values.shape == (6,) and np.isfinite(values).all()
            assert ((values >= 0) & (values <= 1)).all()
            scaled = values * 100
            line_style, marker = STYLES[policy]
            line, = ax.plot(closed_angles, np.r_[scaled, scaled[0]],
                            color=COLORS[policy], linestyle=line_style,
                            linewidth=2.1 if policy == "geo" else 1.7,
                            marker=marker, markersize=4, markerfacecolor="white",
                            markeredgewidth=1.1, label=label, zorder=3)
            ax.fill(closed_angles, np.r_[scaled, scaled[0]], color=COLORS[policy], alpha=0.045)
            if index == 0:
                handles.append(line)
            table_rows.append([benchmark, label, 15, *[f"{v:.4f}" for v in scaled]])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.079),
               ncol=3, fontsize=10, handlelength=3.3, columnspacing=3)
    fig.text(0.5, 0.065, "Scores x 100; identical 0-100 axes. VBench-Long uses final dimension scores.",
             ha="center", fontsize=8, color="#555555")
    fig.text(0.5, 0.033, "RI-32 omitted from both panels: only 7/15 Long results available. Preview from reported means.",
             ha="center", fontsize=8, color="#555555")
    save(fig, "vbench_radar_matched15_60s")
    with (FIGURES / "vbench_radar_matched15_60s.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "policy", "n", *dimensions])
        writer.writerows(table_rows)


if __name__ == "__main__":
    main()
