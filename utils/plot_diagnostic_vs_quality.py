"""Experiments 2 & 3: does a cheap, computable-without-real-eval diagnostic
actually predict expensive real downstream quality (VBench/FVD/CUT3R)?

Generic scatter+trendline over a small, hand-assembled CSV -- one row per
policy, at minimum a "policy" column plus the two numeric columns you want
to correlate. Deliberately does not auto-crawl the scattered per-run
summary.json/eval_results.json files (they live in three different
directory layouts with a timestamped filename for VBench) -- for a one-time
5-6-row join, assembling that CSV by hand from numbers you already have is
more robust than a crawler that breaks on a naming quirk. Example CSV:

    policy,retrieval_gap,worldscore_camera_control_score
    unbounded,0.2267,68.0
    fifo,0.1000,55.0
    ri,0.1582,72.0
    slam,0.1338,75.0
    slamri,0.1450,70.0

Usage:
    python utils/plot_diagnostic_vs_quality.py \\
      --data_csv diagnostic_vs_cut3r.csv \\
      --x_col retrieval_gap --x_label "retrieval_gap (DINO-oracle proxy)" --x_lower_is_better \\
      --y_col worldscore_camera_control_score --y_label "CUT3R WorldScore camera-control score" \\
      --title "Experiment 2: does retrieval_gap predict CUT3R accuracy?" \\
      --output_dir figures
"""

import argparse
import csv
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from plot_budget_sweep_figures import policy_color, policy_label  # noqa: E402
from summarize_ri_alignment import pearson, spearman  # noqa: E402


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
    parser.add_argument("--data_csv", type=Path, required=True)
    parser.add_argument("--policy_col", type=str, default="policy")
    parser.add_argument("--x_col", type=str, required=True)
    parser.add_argument("--y_col", type=str, required=True)
    parser.add_argument("--x_label", type=str, default=None)
    parser.add_argument("--y_label", type=str, default=None)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--x_lower_is_better", action="store_true")
    parser.add_argument("--y_lower_is_better", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = load_csv_rows(args.data_csv)
    points = []
    for row in rows:
        policy = row.get(args.policy_col)
        x_value = to_float(row.get(args.x_col))
        y_value = to_float(row.get(args.y_col))
        if policy is None or x_value is None or y_value is None:
            print(f"  [skip] incomplete row: {row}")
            continue
        points.append((policy, x_value, y_value))

    if len(points) < 3:
        raise RuntimeError(
            f"Only {len(points)} usable rows in {args.data_csv} -- need at "
            "least 3 policies to say anything about a trend."
        )

    xs = [item[1] for item in points]
    ys = [item[2] for item in points]
    r_pearson = pearson(xs, ys)
    rho_spearman = spearman(xs, ys)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for policy, x_value, y_value in points:
        ax.scatter(
            [x_value],
            [y_value],
            s=90,
            color=policy_color(policy),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            policy_label(policy),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
            color="#0b0b0b",
        )

    if len(set(xs)) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#52514e",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
            alpha=0.7,
        )

    x_label = args.x_label or args.x_col
    y_label = args.y_label or args.y_col
    if args.x_lower_is_better:
        x_label += "  (lower is better ←)"
    if args.y_lower_is_better:
        y_label += "  (lower is better ↓)"
    else:
        y_label += "  (higher is better ↑)"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(args.title, fontsize=12.5, fontweight="bold", y=0.99)
    r_text = f"{r_pearson:.3f}" if r_pearson is not None else "NA"
    rho_text = f"{rho_spearman:.3f}" if rho_spearman is not None else "NA"
    fig.text(
        0.5,
        0.93,
        f"n={len(points)} policies -- Pearson r={r_text}, Spearman ρ={rho_text}",
        ha="center",
        fontsize=9.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or f"diagnostic_vs_quality_{args.x_col}_vs_{args.y_col}"
    out_path = args.output_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"n={len(points)} policies")
    print(f"Pearson r = {r_pearson}")
    print(f"Spearman rho = {rho_spearman}")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
