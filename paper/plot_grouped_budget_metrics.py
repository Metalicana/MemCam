"""Render the complete MemCam budget grid in the WorldMem grouped-bar style."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


BUDGETS = (16, 32, 64, 128)
OPACITIES = (0.4, 0.6, 0.8, 1.0)
POLICIES = (
    ("FIFO", "fifo_b{}", "#CC7972"),
    ("MCE", "mce_b{}_lambda1_pilot", "#A18BBF"),
    ("K-center", "kcenter_b{}", "#BE9B4C"),
    ("RI", "ri_b{}_dino_rgb", "#688FB7"),
    ("KEEPSAKE", "slam_b{}_covisibility", "#569773"),
)
METRICS = (
    ("quality", "LPIPS", "LPIPS $\\downarrow$", 1),
    ("quality", "FVD", "FVD $\\downarrow$", 1),
    ("vbench", "background_consistency", "VBench background consistency $\\uparrow$", 100),
)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_grid(scores_path, coverage_path, *, runs=None, metrics=METRICS):
    if runs is None:
        runs = ["baseline"] + [template.format(b) for _, template, _ in POLICIES for b in BUDGETS]
    required = {(run, evaluator, metric) for run in runs for evaluator, metric, _, _ in metrics}
    grid = {}
    for row in read_rows(scores_path):
        key = row["run"], row["evaluator"], row["metric"]
        if key not in required:
            continue
        if key in grid:
            raise ValueError(f"Duplicate score: {key}")
        value = float(row["value"])
        if int(row["n"]) != 15 or not math.isfinite(value) or value < 0:
            raise ValueError(f"Expected finite nonnegative score and N=15: {key}")
        if row["evaluator"] == "vbench" and value > 1:
            raise ValueError("VBench source scores must be on the 0-1 scale")
        grid[key] = {**row, "value": value}
    if required - grid.keys():
        raise ValueError(f"Missing scores: {sorted(required - grid.keys())}")
    required_tasks = {(run, evaluator) for run, evaluator, _ in required}
    statuses = {}
    for row in read_rows(coverage_path):
        key = row["run"], row["evaluator"]
        if key in required_tasks:
            if key in statuses:
                raise ValueError(f"Duplicate coverage: {key}")
            statuses[key] = row["status"]
    for key in required_tasks:
        if statuses.get(key) != "complete":
            raise ValueError(f"Evaluation not complete: {key}")
    return grid


def make_figure(grid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.29, top=0.77, wspace=0.24)
    fig.text(0.055, 0.955, "MemCam", fontsize=16, weight="bold", ha="left")
    fig.text(0.055, 0.895, "15 matched videos  |  60-second rollouts  |  Complete budget comparison",
             fontsize=10, color="#555555", ha="left")
    for ax, (evaluator, metric, title, scale) in zip(axes, METRICS):
        values = []
        for position, (label, template, color) in enumerate(POLICIES):
            for slot, (budget, opacity) in enumerate(zip(BUDGETS, OPACITIES)):
                value = grid[(template.format(budget), evaluator, metric)]["value"] * scale
                values.append(value)
                ax.bar(position + (slot - 1.5) * 0.18, value, width=0.16,
                       color=color, alpha=opacity, linewidth=0, zorder=3)
        baseline = grid[("baseline", evaluator, metric)]["value"] * scale
        ax.axhline(baseline, color="#363636", linewidth=1, linestyle=(0, (5, 3)), zorder=4)
        ax.set_xticks(range(len(POLICIES)), [p[0] for p in POLICIES])
        ax.set_xlim(-0.58, 4.58)
        ax.set_ylim(0, 100 if scale == 100 else max(*values, baseline) * 1.13)
        ax.set_title(title, fontsize=11, loc="left", pad=14)
        ax.tick_params(axis="x", length=0, pad=10, labelsize=9)
        ax.tick_params(axis="y", length=3, width=0.6, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#AAAAAA")
            ax.spines[side].set_linewidth(0.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E9E9E9", linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if scale == 100:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    handles = [Patch(facecolor="#666666", alpha=opacity, label=f"B{budget}")
               for budget, opacity in zip(BUDGETS, OPACITIES)]
    handles.append(Line2D([], [], color="#363636", linewidth=1,
                          linestyle=(0, (5, 3)), label="Unbounded"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.105),
               ncol=5, frameon=False, fontsize=10, handlelength=2, columnspacing=2.2)
    fig.text(0.52, 0.058, "Within each policy: 16, 32, 64, 128 frames from left to right.",
             ha="center", fontsize=9, color="#555555")
    return fig


def export(scores, coverage, output):
    grid = load_grid(scores, coverage)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "02_grouped_budget_bars"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
                         "axes.facecolor": "white"}):
        fig = make_figure(grid)
        for extension in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + extension), dpi=240, facecolor="white")
        plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        fields = ["policy", "budget", "run", "evaluator", "metric", "source_value", "plotted_value", "n", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        configurations = [("Unbounded", "", "baseline")]
        configurations += [(label, b, template.format(b)) for label, template, _ in POLICIES for b in BUDGETS]
        for policy, budget, run in configurations:
            for evaluator, metric, _, scale in METRICS:
                row = grid[(run, evaluator, metric)]
                writer.writerow(dict(policy=policy, budget=budget, run=run, evaluator=evaluator,
                                     metric=metric, source_value=row["value"], plotted_value=row["value"] * scale,
                                     n=row["n"], source=row["source"]))
    provenance = {
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in (scores, coverage)],
        "system": "MemCam", "duration_seconds": 60, "videos": 15,
        "configurations": 21, "metrics": [m[1] for m in METRICS],
        "budgets": BUDGETS, "opacities": OPACITIES,
        "verification": "N=15, complete recorded task status, unique finite scores for all 63 cells. "
                        "Matched identities are inherited from the upstream grid audit; "
                        "source videos and remote evaluator artifacts are not re-audited locally.",
        "uncertainty": "Point estimates only; no confidence intervals inferred.",
    }
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated and plotted 21 configurations x 3 metrics: {stem.with_suffix('.pdf')}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=root / "metric_results_60s/scores.csv")
    parser.add_argument("--coverage", type=Path, default=root / "metric_results_60s/coverage.csv")
    parser.add_argument("--output", type=Path, default=root / "figures/metric_bars")
    args = parser.parse_args()
    export(args.scores, args.coverage, args.output)


if __name__ == "__main__":
    main()
