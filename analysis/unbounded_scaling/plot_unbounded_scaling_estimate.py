#!/usr/bin/env python3
"""Plot implementation-derived memory and latency scaling for Unbounded MemCam."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_DURATIONS = (10, 40, 60, 600, 3600)
COLOR = "#4D4D4D"
GIB = 1024 ** 3


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def estimate_scaling(
    durations,
    *,
    fps=30,
    frames_per_step=76,
    context_targets=76,
    width=640,
    height=352,
    channels=3,
    bytes_per_value=2,
    denoising_section_s=160.5,
    overlap_call_us=466.1,
):
    rows = []
    overlap_call_s = overlap_call_us / 1_000_000.0
    frame_bytes = channels * height * width * bytes_per_value
    for duration in durations:
        sections = math.ceil(fps * duration / frames_per_step)
        stored_frames = 1 + frames_per_step * sections
        # Section j (zero-based, j >= 1) searches approximately 76*j - 3
        # candidates for each of 76 context targets.
        candidate_sum = (
            frames_per_step * (sections - 1) * sections // 2
            - 3 * (sections - 1)
        )
        overlap_evaluations = context_targets * candidate_sum
        retrieval_latency_s = overlap_evaluations * overlap_call_s
        denoising_latency_s = sections * denoising_section_s
        rollout_latency_s = retrieval_latency_s + denoising_latency_s
        rows.append(
            {
                "duration_sec": int(duration),
                "sections": sections,
                "stored_frames": stored_frames,
                "memory_bank_gb": stored_frames * frame_bytes / GIB,
                "overlap_evaluations": overlap_evaluations,
                "retrieval_latency_s": retrieval_latency_s,
                "denoising_latency_s": denoising_latency_s,
                "rollout_latency_s": rollout_latency_s,
                "rollout_latency_hours": rollout_latency_s / 3600.0,
                "rollout_latency_days": rollout_latency_s / 86400.0,
            }
        )
    return rows


def duration_label(duration):
    if duration <= 60:
        return f"{duration}s"
    return f"{duration // 60}m"


def memory_label(value):
    return f"{value:.2f} GB" if value < 10 else f"{value:.1f} GB"


def latency_label(hours):
    if hours < 1 / 60:
        return f"{hours * 3600:.0f} s"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def plot_metric(rows, output_dir, metric):
    config = {
        "memory": {
            "field": "memory_bank_gb",
            "title": "Unbounded Memory Scaling",
            "subtitle": "352x640 BF16 RGB bank; one retained frame per generated frame",
            "ylabel": "Estimated retained memory-bank storage (GB)",
            "filename": "unbounded_memory_scaling_estimated",
            "formatter": memory_label,
        },
        "latency": {
            "field": "rollout_latency_hours",
            "title": "Unbounded Rollout Latency Scaling",
            "subtitle": "50-step denoising timing from the supplied run plus estimated overlap-search cost",
            "ylabel": "Estimated rollout latency (hours)",
            "filename": "unbounded_latency_scaling_estimated",
            "formatter": latency_label,
        },
    }[metric]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#202124",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.78)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    short_rows = [row for row in rows if row["duration_sec"] <= 60]
    projected_rows = [row for row in rows if row["duration_sec"] >= 60]
    ax.plot(
        [row["duration_sec"] for row in short_rows],
        [row[config["field"]] for row in short_rows],
        color=COLOR,
        linewidth=2.3,
        marker="o",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax.plot(
        [row["duration_sec"] for row in projected_rows],
        [row[config["field"]] for row in projected_rows],
        color=COLOR,
        linewidth=2.3,
        linestyle=(0, (4, 3)),
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=2,
    )

    for index, row in enumerate(rows):
        value = row[config["field"]]
        offset = (0, 12) if index % 2 == 0 else (0, -21)
        ax.annotate(
            config["formatter"](value),
            (row["duration_sec"], value),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#333333",
            weight="bold" if row["duration_sec"] >= 600 else "normal",
        )

    fig.suptitle(
        config["title"],
        x=0.11,
        y=0.97,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(0.11, 0.895, config["subtitle"], ha="left", fontsize=10, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    durations = [row["duration_sec"] for row in rows]
    ax.set_xticks(durations)
    ax.set_xticklabels([duration_label(duration) for duration in durations])
    ax.set_xlabel("Generated video duration")
    ax.set_ylabel(config["ylabel"])
    ax.grid(which="major", color="#DADCE0", linewidth=0.8, alpha=0.85)
    ax.grid(which="minor", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLOR,
                marker="o",
                linewidth=2.3,
                label="Short-horizon estimate",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR,
                marker="o",
                markerfacecolor="white",
                linestyle=(0, (4, 3)),
                linewidth=2.3,
                label="Long-horizon extrapolation",
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{config['filename']}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


def write_outputs(rows, assumptions, output_dir):
    csv_path = output_dir / "unbounded_scaling_estimates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    json_path = output_dir / "unbounded_scaling_estimates.json"
    json_path.write_text(
        json.dumps({"assumptions": assumptions, "estimates": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")

    report_lines = [
        "# Unbounded MemCam Scaling Estimate",
        "",
        "These values are implementation-derived estimates, not long-horizon measurements.",
        "",
        "| Video duration | Bank storage | Overlap evaluations | Retrieval time | Total rollout time |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {duration_label(row['duration_sec'])} | {memory_label(row['memory_bank_gb'])} | "
            f"{row['overlap_evaluations']:,} | {latency_label(row['retrieval_latency_s'] / 3600)} | "
            f"{latency_label(row['rollout_latency_hours'])} |"
        )
    report_lines.extend(
        [
            "",
            "Memory follows directly from the unbounded bank retaining every decoded BF16 RGB frame. "
            "Latency combines fixed per-section denoising time with the current exhaustive retrieval loop, "
            "which compares every candidate frame against every context target.",
            "",
            "The latency estimate excludes model loading, VAE/context encoding outside the timed denoising loop, "
            "and final video encoding. CPU overlap-call timing is hardware-sensitive.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--durations",
        default=",".join(str(duration) for duration in DEFAULT_DURATIONS),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--bytes_per_value", type=int, default=2)
    parser.add_argument("--denoising_section_s", type=float, default=160.5)
    parser.add_argument("--overlap_call_us", type=float, default=466.1)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main():
    args = parse_args()
    durations = parse_int_list(args.durations)
    assumptions = {
        "fps": args.fps,
        "frames_per_section": 76,
        "context_targets_per_section": 76,
        "width": args.width,
        "height": args.height,
        "channels": 3,
        "bytes_per_value": args.bytes_per_value,
        "denoising_seconds_per_section": args.denoising_section_s,
        "overlap_call_microseconds": args.overlap_call_us,
        "overlap_samples_per_call": 5000,
    }
    rows = estimate_scaling(
        durations,
        fps=args.fps,
        width=args.width,
        height=args.height,
        bytes_per_value=args.bytes_per_value,
        denoising_section_s=args.denoising_section_s,
        overlap_call_us=args.overlap_call_us,
    )
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(rows, assumptions, output_dir)
    plot_metric(rows, output_dir, "memory")
    plot_metric(rows, output_dir, "latency")


if __name__ == "__main__":
    main()
