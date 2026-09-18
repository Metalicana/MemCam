"""Fixed-B32 quality bars and relative VBench changes, from audited aggregates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.plot_grouped_budget_metrics import POLICIES, load_grid, plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


METHODS = (("Unbounded", "baseline", "#454545"),) + tuple(
    (label, template.format(32), color) for label, template, color in POLICIES
)
VBENCH = (
    ("subject_consistency", "Subject\nconsistency"),
    ("background_consistency", "Background\nconsistency"),
    ("aesthetic_quality", "Aesthetic\nquality"),
)
METRICS = (("quality", "LPIPS", "LPIPS", 1), ("quality", "FVD", "FVD", 1)) + tuple(
    ("vbench", metric, label, 1) for metric, label in VBENCH
)


def relative_difference(value, reference):
    if reference <= 0:
        raise ValueError("Relative differences require a positive unbounded score")
    return 100 * (value - reference) / reference


def make_figure(grid):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.77, bottom=0.24, wspace=0.7)
    fig.text(0.04, 0.95, "MemCam", fontsize=17, weight="bold")
    fig.text(0.04, 0.89, "B32 bounded methods  |  60-second rollouts  |  15 matched videos",
             fontsize=11, color="#555555")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#AAAAAA")
            ax.spines[side].set_linewidth(0.6)
        ax.grid(axis="x", color="#E9E9E9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=9, labelsize=11)
        ax.tick_params(axis="x", width=0.6, labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    for ax, metric, digits in zip(axes[:2], ("LPIPS", "FVD"), (4, 1)):
        values = [grid[(run, "quality", metric)]["value"] for _, run, _ in METHODS]
        for index, ((label, _, color), value) in enumerate(zip(METHODS, values)):
            ax.barh(index, value, height=0.59, color=color, linewidth=0, zorder=3)
            ax.annotate(f"{value:.{digits}f}", (value, index), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=11,
                        weight="bold" if label == "KEEPSAKE" else "normal", color="#333333")
        ax.set_yticks(range(len(METHODS)), [m[0] for m in METHODS])
        ax.set_ylim(len(METHODS) - 0.4, -0.6)
        ax.set_xlim(0, max(values) * 1.26)
        ax.set_title(f"{metric} $\\downarrow$", loc="left", fontsize=13, pad=16)
        ax.set_xlabel("Absolute score", fontsize=10, labelpad=10)
    ax = axes[2]
    changes = []
    for group, (metric, _) in enumerate(VBENCH):
        reference = grid[("baseline", "vbench", metric)]["value"]
        for slot, (_, run, color) in enumerate(METHODS[1:]):
            delta = relative_difference(grid[(run, "vbench", metric)]["value"], reference)
            changes.append(delta)
            ax.barh(group * 1.5 + (slot - 2) * 0.19, delta, height=0.15,
                    color=color, linewidth=0, zorder=3)
    ax.set_yticks([i * 1.5 for i in range(3)], [label for _, label in VBENCH])
    ax.set_ylim(3.65, -0.65)
    low, high = min(0, *changes), max(0, *changes)
    padding = max((high - low) * 0.13, 0.25)
    ax.set_xlim(low - padding, high + padding)
    ax.axvline(0, color="#363636", linewidth=1, zorder=4)
    ax.set_title("VBench $\\uparrow$", loc="left", fontsize=13, pad=16)
    ax.set_xlabel("Relative difference from unbounded (%)", fontsize=10, labelpad=10)
    fig.legend(handles=[Patch(facecolor=color, label=label) for label, _, color in METHODS],
               loc="lower center", bbox_to_anchor=(0.52, 0.07), frameon=False, ncol=6,
               fontsize=11, handlelength=1.6, columnspacing=2)
    return fig


def export(scores, coverage, output):
    grid = load_grid(scores, coverage, runs=[m[1] for m in METHODS], metrics=METRICS)
    # Reject zero denominators before exporting any artifacts.
    for metric, _ in VBENCH:
        relative_difference(0, grid[("baseline", "vbench", metric)]["value"])
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "01_editorial_b32"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig = make_figure(grid)
        for extension in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + extension), dpi=240, facecolor="white")
        plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        fields = ["policy", "budget", "run", "evaluator", "metric", "source_value",
                  "unbounded_value", "relative_difference_percent", "n", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, run, _ in METHODS:
            for evaluator, metric, _, _ in METRICS:
                row = grid[(run, evaluator, metric)]
                reference = grid[("baseline", evaluator, metric)]["value"]
                delta = relative_difference(row["value"], reference) if evaluator == "vbench" else ""
                writer.writerow(dict(policy=label, budget="" if run == "baseline" else 32,
                                     run=run, evaluator=evaluator, metric=metric, source_value=row["value"],
                                     unbounded_value=reference, relative_difference_percent=delta,
                                     n=row["n"], source=row["source"]))
    stem.with_suffix(".json").write_text(json.dumps({
        "system": "MemCam", "budget": 32, "duration_seconds": 60, "videos": 15,
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in (scores, coverage)],
        "vbench_formula": "100 * (bounded - unbounded) / unbounded; relative percent, not percentage points",
        "verification": "All 30 scores unique, finite, N=15, with complete recorded task status. "
                        "Matched cohort identity inherited from upstream audit; remote source artifacts not re-audited.",
        "uncertainty": "Aggregate point estimates; no confidence intervals inferred.",
    }, indent=2) + "\n")
    print(f"Validated 6 configurations x 5 metrics: {stem.with_suffix('.pdf')}")


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
