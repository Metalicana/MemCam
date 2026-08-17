"""H1 evidence figure: does retrieval_gap actually grow with candidate-pool
size, and does that hold trajectory by trajectory rather than only in a
pooled average?

Reads the three outputs analyze_pool_growth_scaling.py already writes
(pool_growth_scaling_<run>.csv, pool_growth_per_trajectory_<run>.csv,
pool_growth_summary_<run>.json) and renders a three-panel figure:

  A. binned mean retrieval_gap vs candidate_count (pooled across trajectories)
  B. per-trajectory Spearman(pool size, retrieval_gap) -- one dot per
     trajectory, sorted, with the exact sign-test result annotated
  C. per-trajectory early-quarter vs late-quarter retrieval_gap (paired
     slope chart) -- does it get worse for nearly every trajectory, or just
     a few?

CPU-only (matplotlib on CSVs already on disk); no new generation or DINO
encoding.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_retrieval_quality_decomposition import (  # noqa: E402
    pretty_run_name,
    run_color,
)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        required=True,
        help="--output_dir passed to analyze_pool_growth_scaling.py "
        "(directory containing pool_growth_*_<run>.{csv,json}).",
    )
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = args.run_name
    binned_rows = load_csv_rows(args.analysis_dir / f"pool_growth_scaling_{run}.csv")
    trajectory_rows = load_csv_rows(
        args.analysis_dir / f"pool_growth_per_trajectory_{run}.csv"
    )
    summary = json.loads(
        (args.analysis_dir / f"pool_growth_summary_{run}.json").read_text(
            encoding="utf-8"
        )
    )

    color = run_color(run)
    label = pretty_run_name(run)
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    grid_color = "#e6e6e6"

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))

    # -- Panel A: binned mean retrieval_gap vs candidate_count --
    ax = axes[0]
    centers = [
        (to_float(row["candidate_count_low"]) + to_float(row["candidate_count_high"]))
        / 2.0
        for row in binned_rows
    ]
    gaps = [to_float(row["mean_retrieval_gap"]) for row in binned_rows]
    widths = [
        0.85 * (to_float(row["candidate_count_high"]) - to_float(row["candidate_count_low"]))
        for row in binned_rows
    ]
    ax.bar(centers, gaps, width=widths, color=color, alpha=0.85, linewidth=0)
    ax.set_xlabel("candidate pool size (bin center)")
    ax.set_ylabel("mean retrieval_gap")
    ax.set_title(f"A. Pooled: does error grow with pool size?\n({label})", fontsize=10.5)
    ax.grid(axis="y", color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # -- Panel B: per-trajectory Spearman rho, sorted, with sign-test result --
    ax = axes[1]
    # Positive/negative is a polarity encoding (agrees with vs. contradicts
    # the "bigger pool -> bigger gap" hypothesis), not an identity encoding
    # -- use a fixed diverging pair rather than run_color, since run_color
    # for "baseline" is nearly the same gray as the muted ink and the
    # distinction would vanish for the one run this figure exists to show.
    positive_color = "#2a78d6"  # blue
    negative_color = "#e34948"  # red
    rhos = sorted(
        to_float(row["spearman_pool_vs_retrieval_gap"])
        for row in trajectory_rows
        if to_float(row["spearman_pool_vs_retrieval_gap"]) is not None
    )
    y_positions = np.arange(len(rhos))
    dot_colors = [positive_color if rho > 0 else negative_color for rho in rhos]
    ax.axvline(0, color=ink_secondary, linewidth=1.0, linestyle="--")
    ax.scatter(rhos, y_positions, s=42, color=dot_colors, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks([])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman(pool size, retrieval_gap) per trajectory")
    ax.set_title("B. Per-trajectory trend (one dot = one video)", fontsize=10.5)
    ax.grid(axis="x", color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    pvalue = summary.get("trajectory_sign_test_pvalue")
    pvalue_text = f"p={pvalue:.4f}" if pvalue is not None else "p=NA"
    ax.text(
        0.02,
        0.02,
        f"positive={summary.get('trajectory_positive_rho')}, "
        f"negative={summary.get('trajectory_negative_rho')}, "
        f"{pvalue_text}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=ink_secondary,
        va="bottom",
    )

    # -- Panel C: paired early-quarter vs late-quarter retrieval_gap --
    ax = axes[2]
    x_positions = [0.0, 1.0]
    for row in trajectory_rows:
        early = to_float(row["early_quartile_gap"])
        late = to_float(row["late_quartile_gap"])
        if early is None or late is None:
            continue
        line_color = positive_color if late > early else negative_color
        ax.plot(
            x_positions,
            [early, late],
            color=line_color,
            linewidth=1.1,
            alpha=0.6,
            marker="o",
            markersize=3.6,
            zorder=2,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["early quarter", "late quarter"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("retrieval_gap")
    ax.set_title("C. Per-trajectory: early vs. late in the rollout", fontsize=10.5)
    ax.grid(axis="y", color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    late_minus_early = summary.get("mean_late_minus_early_gap")
    ci_low = summary.get("mean_late_minus_early_gap_ci_low")
    ci_high = summary.get("mean_late_minus_early_gap_ci_high")
    if late_minus_early is not None:
        ax.text(
            0.02,
            0.98,
            f"mean late-early={late_minus_early:+.4f}\n95% CI [{ci_low:.4f}, {ci_high:.4f}]",
            transform=ax.transAxes,
            fontsize=8.5,
            color=ink_secondary,
            va="top",
        )

    fig.suptitle(
        "H1 -- unbounded retrieval error vs. candidate-pool growth",
        fontsize=13,
        fontweight="bold",
        color=ink_primary,
        y=0.995,
    )
    fig.text(
        0.5,
        0.925,
        "Pool size and elapsed time co-vary in a rollout -- this is a scaling screen, not causal isolation of pool size.",
        ha="center",
        fontsize=8.5,
        color=ink_secondary,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"pool_growth_evidence_{run}.png"
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
