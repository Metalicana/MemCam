#!/usr/bin/env python3
"""Plot slide-ready CUT3R camera metrics across MemCam memory budgets."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = {
    "rotation_error_deg_mean_mean": (
        "CUT3R Mean Rotation Error",
        "Rotation error (degrees, lower is better)",
    ),
    "translation_error_scale_only_mean_mean": (
        "CUT3R Mean Translation Error",
        "Translation error (lower is better)",
    ),
    "worldscore_camera_control_score_mean": (
        "CUT3R Camera-Control Score",
        "Camera-control score (higher is better)",
    ),
}

POLICY_STYLES = {
    "FIFO": {"color": "#D55E00", "marker": "s", "linestyle": ":"},
    "Ours": {"color": "#0072B2", "marker": "^", "linestyle": "-."},
    "SLAM": {"color": "#009E73", "marker": "D", "linestyle": "-"},
}


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def describe_run(run_name: str):
    if run_name == "baseline":
        return "Unbounded", None
    match = re.search(r"_b(\d+)", run_name)
    if match is None:
        return None
    budget = int(match.group(1))
    if run_name.startswith("fifo_"):
        return "FIFO", budget
    if run_name.startswith("ri_"):
        return "Ours", budget
    if run_name.startswith("slam_"):
        return "SLAM", budget
    return None


def load_rows(path: Path, metrics: list[str]) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            description = describe_run(source_row.get("run_name", ""))
            if description is None:
                continue
            policy, budget = description
            for metric in metrics:
                value = safe_float(source_row.get(metric))
                if value is None:
                    continue
                rows.append(
                    {
                        "run_name": source_row["run_name"],
                        "policy": policy,
                        "budget": budget,
                        "metric": metric,
                        "value": value,
                        "videos": source_row.get("videos"),
                    }
                )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["run_name", "policy", "budget", "metric", "value", "videos"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, output_dir: Path) -> None:
    metric_rows = [row for row in rows if row["metric"] == metric]
    if not metric_rows:
        print(f"[warn] no values found for {metric}")
        return

    budgets = sorted({row["budget"] for row in metric_rows if row["budget"] is not None})
    baseline_values = [row["value"] for row in metric_rows if row["policy"] == "Unbounded"]
    baseline = baseline_values[0] if baseline_values else None
    title, ylabel = METRICS[metric]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for policy in ("FIFO", "SLAM", "Ours"):
        policy_rows = {
            row["budget"]: row["value"]
            for row in metric_rows
            if row["policy"] == policy and row["budget"] is not None
        }
        xs = [budget for budget in budgets if budget in policy_rows]
        if not xs:
            continue
        ax.plot(
            xs,
            [policy_rows[budget] for budget in xs],
            label=policy,
            linewidth=2.3,
            markersize=7,
            **POLICY_STYLES[policy],
        )

    if baseline is not None:
        ax.axhline(baseline, color="#111111", linewidth=2.5, label="Unbounded")

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Memory budget (frames)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(budgets)
    ax.grid(True, color="#D7D7D7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    stem = f"memcam_cut3r_{metric}"
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    metrics = parse_list(args.metrics)
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"Unknown CUT3R slide metrics: {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summary_csv, metrics)
    if not rows:
        raise RuntimeError(f"No supported policy rows found in {args.summary_csv}")
    write_rows(args.output_dir / "memcam_cut3r_budget_values.csv", rows)
    for metric in metrics:
        plot_metric(rows, metric, args.output_dir)


if __name__ == "__main__":
    main()
