"""Plot validated matched-15 budget-grid scores, without recomputing metrics."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator


DIMENSIONS = {
    "subject_consistency": "Subject consistency",
    "background_consistency": "Background consistency",
    "motion_smoothness": "Motion smoothness",
    "dynamic_degree": "Dynamic degree",
    "aesthetic_quality": "Aesthetic quality",
    "imaging_quality": "Imaging quality",
}
BUDGETS = (16, 32, 64, 128)
POLICIES = {
    "FIFO": ("fifo_b{}", "#D58526", "s"),
    "RI": ("ri_b{}_dino_rgb", "#8973AE", "^"),
    "K-center": ("kcenter_b{}", "#389A86", "D"),
    "MCE": ("mce_b{}_lambda1_pilot", "#C36D83", "v"),
    "GeoCov": ("slam_b{}_covisibility", "#0072B2", "o"),
}
EVALUATORS = {"vbench": "VBench", "vbench-long": "VBench-Long"}
RUNS = {"baseline"} | {template.format(b) for template, _, _ in POLICIES.values() for b in BUDGETS}


def read_scores(path):
    scores = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["evaluator"] not in EVALUATORS:
                continue
            if row["run"] not in RUNS or row["metric"] not in DIMENSIONS:
                raise ValueError(f"Unknown VBench row: {row}")
            value = float(row["value"])
            if int(row["n"]) != 15 or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Expected N=15 and a finite 0-1 score: {row}")
            key = (row["evaluator"], row["run"], row["metric"])
            if key in scores:
                raise ValueError(f"Duplicate score: {key}")
            scores[key] = 100 * value
    for evaluator, run in {(e, r) for e, r, _ in scores}:
        if any((evaluator, run, dim) not in scores for dim in DIMENSIONS):
            raise ValueError(f"Incomplete six-dimension result: {evaluator}/{run}")
    if not scores:
        raise ValueError("No completed matched-15 VBench results in this CSV")
    return scores


def render(scores, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    paths = []
    for evaluator, title in EVALUATORS.items():
        completed = {run for e, run, _ in scores if e == evaluator}
        if not completed:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))
        fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.13, hspace=.48, wspace=.30)
        fig.suptitle(f"{title}: Quality Across Memory Budgets", y=.97, fontsize=17, weight="bold")
        fig.text(.5, .922, f"MemCam | 15 matched videos | 60 seconds | {len(completed)}/21 configurations complete",
                 ha="center", color="#555555")
        handles = [Line2D([], [], color=color, marker=marker, label=label, linewidth=2)
                   for label, (_, color, marker) in POLICIES.items()]
        handles.append(Line2D([], [], color="#444444", linestyle="--", label="Unbounded"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .888), ncol=6, frameon=False)
        for ax, (dim, label) in zip(axes.flat, DIMENSIONS.items()):
            for policy, (template, color, marker) in POLICIES.items():
                values = [scores.get((evaluator, template.format(b), dim), float("nan")) for b in BUDGETS]
                ax.plot(BUDGETS, values, color=color, marker=marker, markersize=5,
                        linewidth=2.3 if policy == "GeoCov" else 1.6,
                        markerfacecolor="white", markeredgewidth=1.5)
            baseline = scores.get((evaluator, "baseline", dim))
            if baseline is not None:
                ax.axhline(baseline, color="#444444", linestyle="--", linewidth=1.2)
            # Keep each metric's limits identical between standard and Long figures.
            values = [v for (_, _, d), v in scores.items() if d == dim]
            low, high = min(values), max(values)
            pad = max((high - low) * .16, .05)
            ax.set_ylim(max(0, low - pad), high + pad)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BUDGETS)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlim(14, 145)
            ax.set_title(label, loc="left", fontsize=11, weight="bold", pad=10)
            ax.set_xlabel("Retained frames")
            ax.set_ylabel("Score (0-100)")
            ax.grid(axis="y", color="#E7E9EB", linewidth=.7)
            ax.set_axisbelow(True)
        fig.text(.075, .035, "Higher scores are better. Zoomed y-axes; point estimates. Missing budgets are not connected.",
                 fontsize=9, color="#555555")
        stem = evaluator.replace("-", "_") + "_budget_sweep_matched15_60s"
        for extension in ("pdf", "png"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=200, facecolor="white")
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    for path in render(read_scores(args.scores), args.output):
        print(path)


if __name__ == "__main__":
    main()
