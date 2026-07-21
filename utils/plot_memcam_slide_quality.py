#!/usr/bin/env python3
"""Plot slide-ready MemCam prefix FVD and LPIPS curves from evaluator summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = (
    "baseline",
    "fifo_b16",
    "fifo_b32",
    "fifo_b64",
    "fifo_b128",
    "ri_b16_dino_rgb",
    "ri_b32_dino_rgb",
    "ri_b64_dino_rgb",
    "ri_b128_dino_rgb",
    "slam_b16_covisibility",
    "slam_b32_covisibility",
    "slam_b64_covisibility",
    "slam_b128_covisibility",
)

METRICS = {
    "lpips_alex": {
        "label": "LPIPS (lower is better)",
        "title": "MemCam 60s Prefix LPIPS",
        "stem": "memcam_lpips_prefix_60s",
    },
    "fvd": {
        "label": "FVD (lower is better)",
        "title": "MemCam 60s Prefix FVD",
        "stem": "memcam_fvd_prefix_60s",
    },
}

FAMILY_STYLES = {
    "Unbounded": {"color": "#111111", "linestyle": "-", "linewidth": 3.0, "marker": "o"},
    "FIFO": {"color": "#D55E00", "linestyle": ":", "linewidth": 1.8, "marker": "s"},
    "Ours": {"color": "#0072B2", "linestyle": "-.", "linewidth": 2.0, "marker": "^"},
    "SLAM": {"color": "#009E73", "linestyle": "-", "linewidth": 2.1, "marker": "D"},
}

BUDGET_ALPHA = {16: 1.0, 32: 0.84, 64: 0.68, 128: 0.52}


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


def describe_run(run_name: str) -> tuple[str, int | None, str]:
    if run_name == "baseline":
        return "Unbounded", None, "Unbounded"

    budget_match = re.search(r"_b(\d+)", run_name)
    budget = int(budget_match.group(1)) if budget_match else None
    if run_name.startswith("fifo_"):
        family = "FIFO"
    elif run_name.startswith("ri_"):
        family = "Ours"
    elif run_name.startswith("slam_"):
        family = "SLAM"
    else:
        raise ValueError(f"Unsupported slide run: {run_name}")
    label = family if budget is None else f"{family} b{budget}"
    return family, budget, label


def find_summaries(metrics_dirs: list[Path], run_name: str) -> list[Path]:
    candidates = []
    for metrics_dir in metrics_dirs:
        direct = metrics_dir / run_name / "summary.json"
        if direct.is_file():
            candidates.append(direct)
        candidates.extend(metrics_dir.glob(f"**/{run_name}/summary.json"))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime)
    return candidates


def load_rows(metrics_dirs: list[Path], runs: list[str], durations: list[int]) -> list[dict]:
    rows = []
    for run_name in runs:
        summary_paths = find_summaries(metrics_dirs, run_name)
        if not summary_paths:
            print(f"[warn] no summary.json found for {run_name}")
            continue
        family, budget, label = describe_run(run_name)
        run_values = {}
        for summary_path in summary_paths:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            by_duration = summary.get("by_duration", {})
            for duration in durations:
                values = by_duration.get(str(duration), {})
                for metric in METRICS:
                    value = safe_float(values.get(metric))
                    if value is not None:
                        run_values[(duration, metric)] = (value, summary_path)
        for (duration, metric), (value, summary_path) in sorted(run_values.items()):
            rows.append(
                {
                    "run_name": run_name,
                    "family": family,
                    "budget": budget,
                    "label": label,
                    "duration_sec": duration,
                    "metric": metric,
                    "value": value,
                    "source_path": str(summary_path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "family",
        "budget",
        "label",
        "duration_sec",
        "metric",
        "value",
        "source_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict],
    metric: str,
    durations: list[int],
    runs: list[str],
    output_dir: Path,
) -> None:
    spec = METRICS[metric]
    metric_rows = [row for row in rows if row["metric"] == metric]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for run_name in runs:
        run_rows = [row for row in metric_rows if row["run_name"] == run_name]
        if not run_rows:
            continue
        representative = run_rows[0]
        values_by_duration = {row["duration_sec"]: row["value"] for row in run_rows}
        xs = [duration for duration in durations if duration in values_by_duration]
        ys = [values_by_duration[duration] for duration in xs]
        style = dict(FAMILY_STYLES[representative["family"]])
        style["alpha"] = BUDGET_ALPHA.get(representative["budget"], 1.0)
        ax.plot(
            xs,
            ys,
            label=representative["label"],
            markersize=5.8,
            markeredgewidth=0.8,
            **style,
        )

    ax.set_title(spec["title"], fontsize=16, pad=12)
    ax.set_xlabel("Generated future duration (seconds)", fontsize=12)
    ax.set_ylabel(spec["label"], fontsize=12)
    ax.set_xticks(durations)
    ax.grid(True, color="#D7D7D7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Memory policy",
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"
    fig.subplots_adjust(right=0.76, bottom=0.14)

    for extension in ("png", "pdf"):
        path = output_dir / f"{spec['stem']}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dirs", required=True, help="Comma-separated evaluator output roots.")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--durations", default="10,20,30,60")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    metrics_dirs = [Path(path).expanduser() for path in parse_list(args.metrics_dirs)]
    runs = parse_list(args.runs)
    durations = [int(value) for value in parse_list(args.durations)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(metrics_dirs, runs, durations)
    if not rows:
        raise RuntimeError("No FVD or LPIPS values found in the requested summaries.")
    write_csv(args.output_dir / "memcam_prefix_quality_values.csv", rows)
    for metric in METRICS:
        plot_metric(rows, metric, durations, runs, args.output_dir)


if __name__ == "__main__":
    main()
