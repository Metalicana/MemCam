"""Test which memory diagnostics predict downstream MemCam video quality.

This is an analysis-only join over outputs that already exist. It combines:

* retrieval decomposition ``section_summary.csv`` rows;
* per-video and optional per-frame LPIPS/PSNR/SSIM metrics;
* optional CUT3R per-video camera metrics; and
* optional VBench per-video details.

The primary test is paired: for the same trajectory, does a bounded policy's
improvement in a memory diagnostic predict its quality improvement over the
unbounded baseline? FVD is intentionally excluded because it is a
distribution-level statistic and has no defensible per-video value.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_quality_predictors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memcam_quality_predictors_xdg")


SECTION_STRIDE = 76

DIAGNOSTIC_FIELDS = (
    "retrieval_gap",
    "retention_gap",
    "selected_memory_corruption",
    "selected_view_mismatch",
    "selected_effective_mismatch",
    "total_oracle_gap",
    "candidate_count",
)

VIDEO_QUALITY_FIELDS = (
    "lpips_alex",
    "mae",
    "rmse",
    "psnr_db",
    "ssim",
    "temporal_delta_mae",
    "temporal_delta_rmse",
)

CUT3R_FIELDS = (
    "rotation_error_deg_mean",
    "rotation_error_deg_p90",
    "translation_error_scale_only_mean",
    "translation_error_scale_only_p90",
    "translation_error_sim3_mean",
    "translation_error_sim3_p90",
    "endpoint_rotation_error_deg",
    "endpoint_translation_error_scale_only",
    "loop_endpoint_distance_error",
    "worldscore_camera_control_score",
)

LOWER_IS_BETTER = {
    "lpips_alex",
    "mae",
    "rmse",
    "temporal_delta_mae",
    "temporal_delta_rmse",
    "rotation_error_deg_mean",
    "rotation_error_deg_p90",
    "translation_error_scale_only_mean",
    "translation_error_scale_only_p90",
    "translation_error_sim3_mean",
    "translation_error_sim3_p90",
    "endpoint_rotation_error_deg",
    "endpoint_translation_error_scale_only",
    "loop_endpoint_distance_error",
}

HIGHER_IS_BETTER = {
    "psnr_db",
    "ssim",
    "worldscore_camera_control_score",
}


def parse_list(value):
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def to_float(value):
    try:
        if value in (None, ""):
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def to_int(value):
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(xs, ys):
    pairs = [
        (to_float(x), to_float(y))
        for x, y in zip(xs, ys)
        if to_float(x) is not None and to_float(y) is not None
    ]
    if len(pairs) < 3:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None
    return float(np.corrcoef(average_ranks(x_values), average_ranks(y_values))[0, 1])


def bootstrap_spearman(xs, ys, repeats=4000, seed=0):
    pairs = [
        (to_float(x), to_float(y))
        for x, y in zip(xs, ys)
        if to_float(x) is not None and to_float(y) is not None
    ]
    if len(pairs) < 5:
        return None, None
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(repeats)):
        sample = [pairs[index] for index in rng.integers(0, len(pairs), len(pairs))]
        rho = spearman([pair[0] for pair in sample], [pair[1] for pair in sample])
        if rho is not None:
            values.append(rho)
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def grouped_bootstrap_spearman(rows, x_field, y_field, repeats=4000, seed=0):
    grouped = defaultdict(list)
    for row in rows:
        x_value = to_float(row.get(x_field))
        y_value = to_float(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        grouped[
            (
                str(row.get("row", "")),
                str(row.get("scene", "")),
                str(row.get("start_frame", "")),
                str(row.get("duration_sec", "")),
            )
        ].append((x_value, y_value))
    keys = list(grouped)
    if len(keys) < 5:
        return None, None
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(repeats)):
        sample = []
        for index in rng.integers(0, len(keys), len(keys)):
            sample.extend(grouped[keys[index]])
        rho = spearman(
            [pair[0] for pair in sample],
            [pair[1] for pair in sample],
        )
        if rho is not None:
            values.append(rho)
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def identity(row, start_field):
    return (
        str(row.get("row", "")),
        str(row.get("scene", "")),
        str(row.get(start_field, "")),
        str(row.get("duration_sec", "")),
    )


def aggregate_section_diagnostics(rows, duration, late_fraction):
    grouped = defaultdict(list)
    for row in rows:
        if to_int(row.get("duration_sec")) != int(duration):
            continue
        run_name = row.get("run_name")
        if not run_name:
            continue
        grouped[(run_name, identity(row, "dataset_start_frame"))].append(row)

    output = []
    for (run_name, video_id), group in sorted(grouped.items()):
        group.sort(key=lambda row: to_int(row.get("section_idx")) or -1)
        late_count = max(1, int(math.ceil(len(group) * float(late_fraction))))
        early = group[:late_count]
        late = group[-late_count:]
        out = {
            "run_name": run_name,
            "row": video_id[0],
            "scene": video_id[1],
            "start_frame": video_id[2],
            "duration_sec": video_id[3],
            "sections": len(group),
        }
        for field in DIAGNOSTIC_FIELDS:
            all_mean = mean(to_float(row.get(field)) for row in group)
            early_mean = mean(to_float(row.get(field)) for row in early)
            late_mean = mean(to_float(row.get(field)) for row in late)
            out[f"{field}_mean"] = all_mean
            out[f"{field}_early_mean"] = early_mean
            out[f"{field}_late_mean"] = late_mean
            out[f"{field}_late_minus_early"] = (
                late_mean - early_mean
                if late_mean is not None and early_mean is not None
                else None
            )
        output.append(out)
    return output


def load_video_quality(metrics_dir, runs, duration):
    rows = []
    for run_name in runs:
        path = Path(metrics_dir) / run_name / "metrics.csv"
        if not path.is_file():
            print(f"[quality missing] {path}")
            continue
        for row in read_csv(path):
            if to_int(row.get("duration_sec")) != int(duration):
                continue
            if row.get("status") not in {"completed", "short_video"}:
                continue
            row["run_name"] = run_name
            rows.append(row)
    return rows


def load_cut3r(path, duration):
    if path is None or not Path(path).is_file():
        return []
    return [
        row
        for row in read_csv(path)
        if to_int(row.get("duration_sec")) == int(duration)
    ]


def latest_vbench_result(run_dir):
    candidates = sorted(Path(run_dir).glob("*_eval_results.json"))
    return candidates[-1] if candidates else None


def load_vbench(vbench_root, runs, basename_to_id):
    rows = defaultdict(dict)
    if vbench_root is None:
        return rows
    for run_name in runs:
        path = latest_vbench_result(Path(vbench_root) / run_name)
        if path is None:
            print(f"[VBench missing] {Path(vbench_root) / run_name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dimension, result in payload.items():
            if not isinstance(result, list) or len(result) < 2:
                continue
            details = result[1]
            if not isinstance(details, list):
                continue
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                video_path = detail.get("video_path") or detail.get("video")
                score = to_float(detail.get("video_results", detail.get("score")))
                if not video_path or score is None:
                    continue
                video_id = basename_to_id.get((run_name, Path(video_path).name))
                if video_id is not None:
                    rows[(run_name, video_id)][f"vbench_{dimension}"] = score
    return rows


def join_video_sources(diagnostics, quality_rows, cut3r_rows, vbench_root, runs):
    quality_by_key = {}
    basename_to_id = {}
    for row in quality_rows:
        video_id = identity(row, "start_frame")
        key = (row["run_name"], video_id)
        quality_by_key[key] = row
        output = row.get("output")
        if output:
            basename_to_id[(row["run_name"], Path(output).name)] = video_id

    cut3r_by_key = {
        (row["run_name"], identity(row, "start_frame")): row for row in cut3r_rows
    }
    vbench_by_key = load_vbench(vbench_root, runs, basename_to_id)

    output = []
    for diagnostic in diagnostics:
        video_id = identity(diagnostic, "start_frame")
        key = (diagnostic["run_name"], video_id)
        row = dict(diagnostic)
        quality = quality_by_key.get(key, {})
        for field in VIDEO_QUALITY_FIELDS:
            value = to_float(quality.get(field))
            if value is not None:
                row[field] = value
        cut3r = cut3r_by_key.get(key, {})
        for field in CUT3R_FIELDS:
            value = to_float(cut3r.get(field))
            if value is not None:
                row[field] = value
        row.update(vbench_by_key.get(key, {}))
        output.append(row)
    return output


def quality_direction(field):
    if field in LOWER_IS_BETTER:
        return "lower"
    if field in HIGHER_IS_BETTER or field.startswith("vbench_"):
        return "higher"
    return None


def available_outcomes(rows):
    candidates = list(VIDEO_QUALITY_FIELDS) + list(CUT3R_FIELDS)
    candidates.extend(
        sorted({field for row in rows for field in row if field.startswith("vbench_")})
    )
    return [
        field
        for field in candidates
        if quality_direction(field) is not None
        and sum(to_float(row.get(field)) is not None for row in rows) >= 3
    ]


def build_paired_deltas(rows, baseline_run, predictors, outcomes):
    by_key = {
        (row["run_name"], identity(row, "start_frame")): row for row in rows
    }
    baseline = {
        key[1]: row for key, row in by_key.items() if key[0] == baseline_run
    }
    output = []
    for (run_name, video_id), policy in sorted(by_key.items()):
        if run_name == baseline_run or video_id not in baseline:
            continue
        base = baseline[video_id]
        row = {
            "run_name": run_name,
            "row": video_id[0],
            "scene": video_id[1],
            "start_frame": video_id[2],
            "duration_sec": video_id[3],
        }
        for field in predictors:
            base_value = to_float(base.get(field))
            policy_value = to_float(policy.get(field))
            if base_value is not None and policy_value is not None:
                row[f"{field}_improvement"] = base_value - policy_value
        for field in outcomes:
            base_value = to_float(base.get(field))
            policy_value = to_float(policy.get(field))
            if base_value is None or policy_value is None:
                continue
            row[f"{field}_improvement"] = (
                base_value - policy_value
                if quality_direction(field) == "lower"
                else policy_value - base_value
            )
        output.append(row)
    return output


def correlation_rows(paired_rows, predictors, outcomes, bootstrap_repeats, seed):
    groups = defaultdict(list)
    for row in paired_rows:
        groups[row["run_name"]].append(row)
    groups["ALL_PAIRED"] = list(paired_rows)

    output = []
    for run_name, rows in sorted(groups.items()):
        for predictor in predictors:
            x_field = f"{predictor}_improvement"
            for outcome in outcomes:
                y_field = f"{outcome}_improvement"
                pairs = [
                    (to_float(row.get(x_field)), to_float(row.get(y_field)))
                    for row in rows
                ]
                pairs = [pair for pair in pairs if None not in pair]
                if len(pairs) < 5:
                    continue
                xs = [pair[0] for pair in pairs]
                ys = [pair[1] for pair in pairs]
                rho = spearman(xs, ys)
                if rho is None:
                    continue
                ci_low, ci_high = grouped_bootstrap_spearman(
                    rows,
                    x_field,
                    y_field,
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                output.append(
                    {
                        "scope": run_name,
                        "predictor": predictor,
                        "outcome": outcome,
                        "pairs": len(pairs),
                        "spearman_rho": rho,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                    }
                )
    return output


def load_section_quality(metrics_dir, runs, duration, section_stride):
    grouped = defaultdict(list)
    for run_name in runs:
        path = Path(metrics_dir) / run_name / "frame_metrics.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if to_int(row.get("duration_sec")) != int(duration):
                continue
            frame_index = to_int(row.get("frame_index"))
            gt_index = to_int(row.get("gt_frame_index"))
            if frame_index is None or gt_index is None or frame_index <= 0:
                continue
            video_id = (
                str(row.get("row", "")),
                str(row.get("scene", "")),
                str(gt_index - frame_index),
                str(duration),
            )
            # Section 0 owns frames 0..76. Later section s predicts
            # s*76+1 .. (s+1)*76, so the boundary frame s*76 is its anchor,
            # not part of the newly generated section.
            section_idx = (frame_index - 1) // int(section_stride)
            grouped[(run_name, video_id, section_idx)].append(row)

    output = []
    for (run_name, video_id, section_idx), rows in sorted(grouped.items()):
        out = {
            "run_name": run_name,
            "row": video_id[0],
            "scene": video_id[1],
            "start_frame": video_id[2],
            "duration_sec": video_id[3],
            "section_idx": section_idx,
            "frames_evaluated": len(rows),
        }
        for field in VIDEO_QUALITY_FIELDS:
            value = mean(to_float(row.get(field)) for row in rows)
            if value is not None:
                out[field] = value
        output.append(out)
    return output


def section_correlations(section_rows, section_quality, predictors, outcomes):
    diagnostics = {
        (
            row["run_name"],
            identity(row, "dataset_start_frame"),
            to_int(row.get("section_idx")),
        ): row
        for row in section_rows
    }
    joined = []
    for quality in section_quality:
        key = (
            quality["run_name"],
            identity(quality, "start_frame"),
            to_int(quality.get("section_idx")),
        )
        diagnostic = diagnostics.get(key)
        if diagnostic is None:
            continue
        joined.append({**diagnostic, **quality})

    output = []
    by_run = defaultdict(list)
    for row in joined:
        by_run[row["run_name"]].append(row)
    for run_name, rows in sorted(by_run.items()):
        for predictor in predictors:
            raw_predictor = predictor.removesuffix("_late_mean")
            for outcome in outcomes:
                pairs = [
                    (to_float(row.get(raw_predictor)), to_float(row.get(outcome)))
                    for row in rows
                ]
                pairs = [pair for pair in pairs if None not in pair]
                if len(pairs) < 10:
                    continue
                rho = spearman(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                )
                if rho is None:
                    continue
                expected_rho = rho if quality_direction(outcome) == "lower" else -rho
                output.append(
                    {
                        "run_name": run_name,
                        "predictor": raw_predictor,
                        "outcome": outcome,
                        "sections": len(pairs),
                        "spearman_raw": rho,
                        "spearman_bad_predicts_bad": expected_rho,
                    }
                )
    return joined, output


def save_scatter(paired_rows, x_field, y_field, output_path, title):
    points = [
        (row["run_name"], to_float(row.get(x_field)), to_float(row.get(y_field)))
        for row in paired_rows
    ]
    points = [point for point in points if point[1] is not None and point[2] is not None]
    if len(points) < 3:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "fifo_b32": "#3478a8",
        "ri_b32_dino_rgb": "#d58b25",
        "slam_b32_covisibility": "#3f8f5f",
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for run_name, x_value, y_value in points:
        ax.scatter(
            x_value,
            y_value,
            color=palette.get(run_name, "#777777"),
            alpha=0.78,
            s=54,
            label=run_name,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel(y_field.replace("_", " "))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def fmt(value, digits=3):
    value = to_float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def markdown_table(rows, fields, limit=20):
    if not rows:
        return "_No usable rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows[:limit]:
        values = []
        for field in fields:
            value = row.get(field)
            values.append(fmt(value) if to_float(value) is not None else str(value or "NA"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, joined, paired, correlations, section_joined, section_stats):
    source_counts = defaultdict(int)
    for row in joined:
        source_counts[row["run_name"]] += 1
    strongest = sorted(
        [row for row in correlations if row["scope"] == "ALL_PAIRED"],
        key=lambda row: abs(row["spearman_rho"]),
        reverse=True,
    )
    section_strongest = sorted(
        section_stats,
        key=lambda row: abs(row["spearman_bad_predicts_bad"]),
        reverse=True,
    )
    lines = [
        "# Memory Diagnostic to Video Quality",
        "",
        "## What This Tests",
        "",
        "For matched trajectories, this asks whether a bounded policy's improvement over unbounded in a memory diagnostic predicts an improvement in downstream quality. Positive paired correlations are in the expected direction: a larger diagnostic improvement accompanies a larger quality improvement.",
        "",
        "FVD is not included because it is a distribution-level metric, not a per-video observation. DINO-derived outcomes are also omitted from the primary test because retrieval_gap already uses DINO and would create circular evidence.",
        "",
        "## Data Coverage",
        "",
    ]
    for run_name, count in sorted(source_counts.items()):
        lines.append(f"- `{run_name}`: {count} trajectory rows joined.")
    lines.extend(
        [
            "",
            "## Strongest Paired Associations",
            "",
            markdown_table(
                strongest,
                [
                    "predictor",
                    "outcome",
                    "pairs",
                    "spearman_rho",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                ],
            ),
            "",
        ]
    )
    if section_joined:
        lines.extend(
            [
                "## Section-Level Associations",
                "",
                "These use quality measured inside the same generated section as the retrieval diagnostic. Positive `spearman_bad_predicts_bad` means a worse memory diagnostic accompanies worse quality.",
                "",
                markdown_table(
                    section_strongest,
                    [
                        "run_name",
                        "predictor",
                        "outcome",
                        "sections",
                        "spearman_bad_predicts_bad",
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Section-Level Associations",
                "",
                "No `frame_metrics.jsonl` files were found. Run the LPIPS frame-metric pass to test whether retrieval error predicts quality in the immediately generated section.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Rules",
            "",
            "- A positive retrieval-gap/LPIPS-improvement correlation supports retrieval as the quality mechanism.",
            "- A positive memory-corruption/LPIPS relationship supports propagation from a damaged stored frame.",
            "- A retrieval-gap/CUT3R relationship supports camera-control degradation as a mediator.",
            "- VBench subject/background consistency is secondary: it can reward a stable but incorrect scene.",
            "- Confidence intervals spanning zero mean the current 15-trajectory sample does not resolve the relationship.",
            "",
            "## Limits",
            "",
            "This is matched observational evidence. Policies still have different generated histories. A replay that changes only retrieval under matched history is required for a causal claim.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section_csv", type=Path, required=True)
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--late_fraction", type=float, default=0.25)
    parser.add_argument("--section_stride", type=int, default=SECTION_STRIDE)
    parser.add_argument("--cut3r_csv", type=Path, default=None)
    parser.add_argument("--vbench_root", type=Path, default=None)
    parser.add_argument("--bootstrap_repeats", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < args.late_fraction <= 0.5:
        raise ValueError("--late_fraction must be in (0, 0.5]")
    runs = parse_list(args.runs)
    if args.baseline_run not in runs:
        raise ValueError("--baseline_run must be included in --runs")

    section_rows = read_csv(args.section_csv)
    diagnostics = aggregate_section_diagnostics(
        section_rows,
        duration=args.duration,
        late_fraction=args.late_fraction,
    )
    quality = load_video_quality(args.metrics_dir, runs, args.duration)
    cut3r = load_cut3r(args.cut3r_csv, args.duration)
    joined = join_video_sources(
        diagnostics,
        quality,
        cut3r,
        args.vbench_root,
        runs,
    )

    predictors = [f"{field}_late_mean" for field in DIAGNOSTIC_FIELDS]
    outcomes = available_outcomes(joined)
    paired = build_paired_deltas(
        joined,
        baseline_run=args.baseline_run,
        predictors=predictors,
        outcomes=outcomes,
    )
    correlations = correlation_rows(
        paired,
        predictors=predictors,
        outcomes=outcomes,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    section_quality = load_section_quality(
        args.metrics_dir,
        runs,
        duration=args.duration,
        section_stride=args.section_stride,
    )
    section_outcomes = [
        field
        for field in VIDEO_QUALITY_FIELDS
        if quality_direction(field) is not None
        and any(to_float(row.get(field)) is not None for row in section_quality)
    ]
    section_joined, section_stats = section_correlations(
        section_rows,
        section_quality,
        predictors=predictors,
        outcomes=section_outcomes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    write_csv(tables_dir / "video_quality_drivers.csv", joined)
    write_csv(tables_dir / "paired_quality_deltas.csv", paired)
    write_csv(tables_dir / "paired_correlations.csv", correlations)
    write_csv(tables_dir / "section_quality_drivers.csv", section_joined)
    write_csv(tables_dir / "section_correlations.csv", section_stats)

    for outcome in outcomes:
        save_scatter(
            paired,
            "retrieval_gap_late_mean_improvement",
            f"{outcome}_improvement",
            figures_dir / f"retrieval_gap_vs_{outcome}.png",
            f"Does improved retrieval predict improved {outcome}?",
        )

    write_report(
        args.output_dir / "report.md",
        joined,
        paired,
        correlations,
        section_joined,
        section_stats,
    )
    print(f"Joined trajectory rows: {len(joined)}")
    print(f"Matched bounded-vs-unbounded rows: {len(paired)}")
    print(f"Available outcomes: {', '.join(outcomes) if outcomes else 'none'}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
