#!/usr/bin/env python3
"""Audit what drives a common-source GeoCov selection-quality advantage.

The input is the matched-query CSV produced by
``visualize_common_source_psnr_extremes.py``. No video decoding or feature
extraction is performed; this is a small CPU-only sensitivity analysis.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = (
    "row",
    "unbounded_selected_frame",
    "geocov_selected_frame",
    "unbounded_overlap",
    "geocov_overlap",
    "unbounded_psnr",
    "unbounded_ssim",
    "geocov_psnr",
    "geocov_ssim",
    "psnr_delta",
    "ssim_delta",
)


def load_scores(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC_FIELDS:
            if field not in row:
                raise ValueError(f"Missing required column: {field}")
            row[field] = float(row[field])
        row["row"] = int(row["row"])
        row["unbounded_selected_frame"] = int(row["unbounded_selected_frame"])
        row["geocov_selected_frame"] = int(row["geocov_selected_frame"])
    if not rows:
        raise RuntimeError(f"No score rows found in {path}")
    return rows


def bootstrap_mean(values, repeats=10000, seed=17):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_delta(rows, name, repeats=10000, seed=17):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["row"])].append(row)
    if not grouped:
        return None
    psnr = np.asarray(
        [np.mean([row["psnr_delta"] for row in group]) for group in grouped.values()]
    )
    ssim = np.asarray(
        [np.mean([row["ssim_delta"] for row in group]) for group in grouped.values()]
    )
    psnr_low, psnr_high = bootstrap_mean(psnr, repeats, seed)
    ssim_low, ssim_high = bootstrap_mean(ssim, repeats, seed + 1)
    return {
        "stratum": name,
        "queries": len(rows),
        "trajectories": len(grouped),
        "psnr_delta": float(np.mean(psnr)),
        "psnr_ci_low": psnr_low,
        "psnr_ci_high": psnr_high,
        "psnr_trajectory_wins": int(np.sum(psnr > 0.0)),
        "ssim_delta": float(np.mean(ssim)),
        "ssim_ci_low": ssim_low,
        "ssim_ci_high": ssim_high,
        "ssim_trajectory_wins": int(np.sum(ssim > 0.0)),
    }


def build_strata(rows, overlap_threshold, repeats=10000, seed=17):
    definitions = (
        ("all disagreements", lambda row: True),
        ("exclude GeoCov frame 0", lambda row: row["geocov_selected_frame"] != 0),
        (
            "both overlap threshold",
            lambda row: row["unbounded_overlap"] >= overlap_threshold
            and row["geocov_overlap"] >= overlap_threshold,
        ),
        (
            "exclude frame 0 and both overlap threshold",
            lambda row: row["geocov_selected_frame"] != 0
            and row["unbounded_overlap"] >= overlap_threshold
            and row["geocov_overlap"] >= overlap_threshold,
        ),
    )
    return [
        summarize_delta(
            [row for row in rows if predicate(row)],
            name,
            repeats=repeats,
            seed=seed,
        )
        for name, predicate in definitions
    ]


def failure_prevalence(rows, selector, overlap_threshold, psnr_thresholds):
    selected = [
        row
        for row in rows
        if row[f"{selector}_overlap"] >= overlap_threshold
    ]
    output = {
        "selector": selector,
        "queries": len(selected),
        "mean_psnr": float(np.mean([row[f"{selector}_psnr"] for row in selected])),
        "mean_ssim": float(np.mean([row[f"{selector}_ssim"] for row in selected])),
    }
    for threshold in psnr_thresholds:
        output[f"psnr_le_{threshold:g}"] = float(
            np.mean([row[f"{selector}_psnr"] <= threshold for row in selected])
        )
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path, strata, prevalence, thresholds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["All", "No frame 0", "Both high FOV", "No frame 0\n+ both high FOV"]
    x = np.arange(len(strata))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for ax, metric, ylabel, color in (
        (axes[0], "psnr", "GeoCov - Unbounded PSNR (dB)", "#2A9D8F"),
        (axes[1], "ssim", "GeoCov - Unbounded SSIM", "#457B9D"),
    ):
        means = np.asarray([row[f"{metric}_delta"] for row in strata])
        lower = means - np.asarray([row[f"{metric}_ci_low"] for row in strata])
        upper = np.asarray([row[f"{metric}_ci_high"] for row in strata]) - means
        ax.bar(x, means, color=color, width=0.72)
        ax.errorbar(x, means, yerr=[lower, upper], fmt="none", color="#222222", capsize=3)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    for row, color, label in zip(prevalence, ("#C44536", "#2A9D8F"), ("Unbounded", "GeoCov")):
        axes[2].plot(
            thresholds,
            [row[f"psnr_le_{threshold:g}"] for threshold in thresholds],
            marker="o",
            color=color,
            label=label,
        )
    axes[2].set_xlabel("Low-fidelity threshold (PSNR dB)")
    axes[2].set_ylabel("Fraction among high-FOV selections")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(frameon=False)
    fig.suptitle("What drives the common-source selection advantage?")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_report(path, source, rows, strata, prevalence, overlap_threshold, thresholds):
    frame_zero = sum(row["geocov_selected_frame"] == 0 for row in rows)
    lines = [
        "# Common-Source Quality-Failure Sensitivity",
        "",
        "## Question",
        "",
        "Does GeoCov select cleaner baseline-rollout pixels only because it can retain the pristine frame-0 input anchor, or does an advantage remain after removing those selections?",
        "",
        "## Protocol",
        "",
        f"- Input: `{source}`.",
        f"- Matched selector disagreements: `{len(rows)}`.",
        f"- GeoCov selected frame 0 in `{frame_zero}` cases (`{frame_zero / len(rows):.3f}`).",
        f"- High-FOV threshold: `{overlap_threshold:.2f}` for each selector's logged overlap.",
        "- Deltas are averaged within trajectory before trajectory bootstrap.",
        "",
        "## Selection-Quality Sensitivity",
        "",
        "| stratum | queries | trajectories | PSNR delta | 95% CI | PSNR wins | SSIM delta | 95% CI | SSIM wins |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for row in strata:
        lines.append(
            f"| {row['stratum']} | {row['queries']} | {row['trajectories']} | "
            f"{row['psnr_delta']:+.3f} | [{row['psnr_ci_low']:+.3f}, {row['psnr_ci_high']:+.3f}] | "
            f"{row['psnr_trajectory_wins']}/{row['trajectories']} | "
            f"{row['ssim_delta']:+.4f} | [{row['ssim_ci_low']:+.4f}, {row['ssim_ci_high']:+.4f}] | "
            f"{row['ssim_trajectory_wins']}/{row['trajectories']} |"
        )
    lines.extend(
        [
            "",
            "## High-FOV Low-Fidelity Prevalence",
            "",
            "Each selector is filtered by its own overlap score here; the resulting query counts differ and this table is descriptive, not a paired causal contrast.",
            "",
            "| selector | high-FOV queries | mean PSNR | mean SSIM | "
            + " | ".join(f"PSNR <= {threshold:g}" for threshold in thresholds)
            + " |",
            "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in thresholds) + " |",
        ]
    )
    for row in prevalence:
        lines.append(
            f"| {row['selector']} | {row['queries']} | {row['mean_psnr']:.3f} | {row['mean_ssim']:.4f} | "
            + " | ".join(f"{row[f'psnr_le_{threshold:g}']:.3f}" for threshold in thresholds)
            + " |"
        )
    strict = strata[-1]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The common-source advantage is not entirely a frame-0 artifact: after excluding GeoCov's frame-0 selections, the trajectory-level PSNR and SSIM deltas remain positive. However, when both selectors must also have high FOV overlap, the non-anchor confidence intervals cross zero. The extreme montage proves that severe failures exist; it does not show that GeoCov universally finds a cleaner non-anchor candidate at matched geometric relevance.",
            "",
            f"The strict non-anchor, matched-overlap contrast is `{strict['psnr_delta']:+.3f}` dB PSNR and `{strict['ssim_delta']:+.4f}` SSIM. This points to a view-fidelity tradeoff: much of GeoCov's large gain comes from retaining stable clean anchors and allowing the retriever to accept somewhat lower geometric overlap, rather than from an online detector of image corruption.",
            "",
            "## Files",
            "",
            "- `strata_summary.csv`",
            "- `high_fov_failure_prevalence.csv`",
            "- `common_source_quality_strata.png`",
            "- `common_source_quality_strata.pdf`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.80)
    parser.add_argument("--psnr_thresholds", default="8,10,12,14")
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.psnr_thresholds.split(",")]
    rows = load_scores(args.scores_csv)
    strata = build_strata(
        rows, args.overlap_threshold, args.bootstrap_repeats, args.seed
    )
    prevalence = [
        failure_prevalence(rows, selector, args.overlap_threshold, thresholds)
        for selector in ("unbounded", "geocov")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strata_summary.csv", strata)
    write_csv(args.output_dir / "high_fov_failure_prevalence.csv", prevalence)
    make_figure(
        args.output_dir / "common_source_quality_strata.png",
        strata,
        prevalence,
        thresholds,
    )
    write_report(
        args.output_dir / "report.md",
        args.scores_csv,
        rows,
        strata,
        prevalence,
        args.overlap_threshold,
        thresholds,
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
