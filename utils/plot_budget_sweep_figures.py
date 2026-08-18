"""Experiment 1: does more retained memory always help, or does downstream
quality plateau/invert before reaching unbounded?

Reads an existing cut3r_camera_summary.csv (from evaluate_cut3r_camera_metrics.py
-- e.g. the CUT3R budget-sweep grid) and plots camera-pose accuracy against
budget, one line per policy, with unbounded as a horizontal reference. No new
generation, no new CUT3R reconstruction -- this only plots numbers already on
disk.

Optionally layers in LPIPS/FVD (from evaluate_context_memory_prefix_curves.py
summary.json files, one per budget) and VBench (from eval_results.json files,
one per budget) if those exist across the same budget sweep -- pass
--lpips_fvd_summary/--vbench_results as repeatable BUDGET=PATH pairs.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


POLICY_PATTERN = re.compile(r"^([a-z0-9]+)_b(\d+)")

POLICY_LABELS = {
    "unbounded": "Unbounded",
    "fifo": "FIFO",
    "ri": "RI",
    "slam": "SLAM",
    "kcenter": "K-Center",
    "facility": "Facility",
    "slamri": "SLAM+RI blend",
    "mce": "MCE",
    "trajectory": "Trajectory coverage",
}

# Fixed categorical order (dataviz skill palette, slots 1/2/3/4/5/7/8 + gray
# for unbounded) -- assigned by policy identity, never re-cycled per plot.
POLICY_COLORS = {
    "unbounded": "#555555",
    "fifo": "#2f7fbc",
    "ri": "#d88c1f",
    "slam": "#4f9b62",
    "kcenter": "#eda100",
    "facility": "#e87ba4",
    "slamri": "#e34948",
    "mce": "#4a3aa7",
    "trajectory": "#8a5fbf",
}


def parse_policy_and_budget(run_name):
    if run_name == "baseline":
        return "unbounded", None
    match = POLICY_PATTERN.match(run_name)
    if match:
        return match.group(1), int(match.group(2))
    return run_name, None


def policy_label(policy):
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def policy_color(policy, fallback="#8a5fbf"):
    return POLICY_COLORS.get(policy, fallback)


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def group_by_policy(cut3r_rows):
    grouped = {}
    for row in cut3r_rows:
        policy, budget = parse_policy_and_budget(row["run_name"])
        grouped.setdefault(policy, []).append((budget, row))
    return grouped


def plot_metric_panel(ax, grouped, field, ylabel, title, higher_is_better):
    plotted_any = False
    unbounded_value = None
    for policy, entries in sorted(grouped.items()):
        budgeted = sorted(
            (budget, to_float(row.get(field)))
            for budget, row in entries
            if budget is not None and to_float(row.get(field)) is not None
        )
        if policy == "unbounded":
            values = [to_float(row.get(field)) for _budget, row in entries]
            values = [value for value in values if value is not None]
            if values:
                unbounded_value = sum(values) / len(values)
            continue
        if not budgeted:
            continue
        budgets = [item[0] for item in budgeted]
        values = [item[1] for item in budgeted]
        ax.plot(
            budgets,
            values,
            color=policy_color(policy),
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            label=policy_label(policy),
            zorder=3,
        )
        plotted_any = True

    if unbounded_value is not None:
        ax.axhline(
            unbounded_value,
            color=policy_color("unbounded"),
            linewidth=1.6,
            linestyle="--",
            zorder=2,
            label=policy_label("unbounded"),
        )
        plotted_any = True

    ax.set_xscale("log", base=2)
    ax.set_xlabel("memory budget (frames)")
    ax.set_ylabel(ylabel)
    arrow = "higher is better ↑" if higher_is_better else "lower is better ↓"
    ax.set_title(f"{title}\n({arrow})", fontsize=10.5)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return plotted_any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r_summary_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cut3r_rows = load_csv_rows(args.cut3r_summary_csv)
    if not cut3r_rows:
        raise RuntimeError(f"No rows in {args.cut3r_summary_csv}")
    grouped = group_by_policy(cut3r_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))

    plot_metric_panel(
        axes[0],
        grouped,
        "worldscore_camera_control_score_mean",
        "WorldScore camera-control score (0-100)",
        "A. Overall camera-pose accuracy",
        higher_is_better=True,
    )
    plot_metric_panel(
        axes[1],
        grouped,
        "rotation_error_deg_mean",
        "mean rotation error (deg)",
        "B. Rotation error",
        higher_is_better=False,
    )
    plot_metric_panel(
        axes[2],
        grouped,
        "translation_error_sim3_mean",
        "mean translation error (Sim3-aligned)",
        "C. Translation error",
        higher_is_better=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=min(6, len(by_label)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Experiment 1 -- does more retained memory keep helping, or invert?",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.965,
        "CUT3R camera-pose recovery accuracy vs. memory budget, real generated videos, per policy.",
        ha="center",
        fontsize=8.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "budget_sweep_cut3r.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.pdf')}")

    print()
    print("Policies found in the summary CSV:")
    for policy, entries in sorted(grouped.items()):
        budgets = sorted({budget for budget, _row in entries if budget is not None})
        print(f"  {policy}: budgets={budgets or 'n/a (unbounded)'}")


if __name__ == "__main__":
    main()
