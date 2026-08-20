"""Evaluate matched generated-memory versus GT-cleaned-memory replays."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_context_replays import (  # noqa: E402
    bootstrap_mean_interval,
    load_csv,
    load_manifest,
    mean,
    read_gt_frames,
    read_video_frames,
    resize_like,
    write_csv,
)


SECTION_STRIDE = 76
PREDICT_FRAMES = 76


def case_directory_name(case):
    return (
        f"case_{int(case['case_index']):02d}_row{int(case['row'])}_"
        f"section{int(case['section_idx'])}"
    )


def selected_trace_rows(path, max_section=None):
    output = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "context_access" or not row.get("selected"):
                continue
            section_idx = int(row["section_idx"])
            if max_section is not None and section_idx > int(max_section):
                continue
            output[(section_idx, int(row["target_frame"]))] = row
    return output


def compare_trace_content(control_rows, clean_rows, section_idx):
    control_map = {
        key: int(row["selected_memory_frame"])
        for key, row in control_rows.items()
    }
    clean_map = {
        key: int(row["selected_memory_frame"])
        for key, row in clean_rows.items()
    }
    selection_match = bool(control_map) and control_map == clean_map
    clean_target_rows = [
        row for key, row in clean_rows.items() if key[0] == int(section_idx)
    ]
    cleaned_slots = sum(
        int(row.get("context_content_override", 0)) for row in clean_target_rows
    )
    clean_sources_valid = bool(clean_target_rows) and all(
        row.get("context_content_source") == "ground_truth_memory"
        for row in clean_target_rows
    )
    return selection_match, cleaned_slots, clean_sources_valid


def save_case_montage(case, item, gt, control, clean, target_indices, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_clean_replay_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.linspace(0, len(target_indices) - 1, min(6, len(target_indices)))
    chosen = [target_indices[int(round(position))] for position in positions]
    rows = [
        ("Target GT", gt),
        ("Generated memory", control),
        ("GT-cleaned memory", clean),
    ]
    fig, axes = plt.subplots(3, len(chosen), figsize=(3.0 * len(chosen), 7.4))
    if len(chosen) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for row_idx, (label, frames) in enumerate(rows):
        for col, frame_idx in enumerate(chosen):
            axes[row_idx, col].imshow(frames[frame_idx])
            axes[row_idx, col].axis("off")
            if row_idx == 0:
                axes[row_idx, col].set_title(f"{frame_idx / 30.0:.1f}s", fontsize=9)
            if col == 0:
                axes[row_idx, col].set_ylabel(label, fontsize=10)
    fig.suptitle(
        f"Memory cleaning replay | row {case['row']} {item['scene']} | "
        f"section {case['section_idx']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def fmt(value, digits=5):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows, fields):
    if not rows:
        return "_No completed cases._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            values.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, case_rows, overall):
    valid = [row for row in case_rows if row["matched_intervention_valid"]]
    lines = [
        "# Generated-Memory Cleaning Replay",
        "",
        "## Intervention",
        "",
        "For each case, both outputs use the same prompt, camera path, seed, unbounded history, and recorded memory-frame selections. In the target section only, the cleaned branch replaces each selected generated memory image with the dataset GT image at the same frame index. The selected pose and frame identity do not change.",
        "",
        "This tests one causal link: whether corruption inside a retrieved memory image damages the section generated from it. GT cleaning is a diagnostic intervention, not a deployable policy.",
        "",
        "## Cases",
        "",
        markdown_table(
            case_rows,
            [
                "case_index",
                "row",
                "section_idx",
                "control_source",
                "planned_selected_memory_corruption",
                "cleaned_context_slots",
                "prefix_mae",
                "matched_intervention_valid",
                "control_lpips",
                "clean_lpips",
                "lpips_delta",
                "control_dino_distance",
                "clean_dino_distance",
                "dino_distance_delta",
            ],
        ),
        "",
        "## Aggregate",
        "",
        f"- Valid matched cases: `{len(valid)}/{len(case_rows)}`.",
        f"- Mean LPIPS change (clean minus generated): `{fmt(overall.get('lpips_delta_mean'))}`; 95% case-bootstrap CI `{fmt(overall.get('lpips_delta_ci_low'))}` to `{fmt(overall.get('lpips_delta_ci_high'))}`.",
        f"- Mean DINO-distance change: `{fmt(overall.get('dino_distance_delta_mean'))}`; 95% case-bootstrap CI `{fmt(overall.get('dino_distance_delta_ci_low'))}` to `{fmt(overall.get('dino_distance_delta_ci_high'))}`.",
        f"- LPIPS-improved cases: `{overall.get('lpips_improved_cases', 0)}/{len(valid)}`.",
        f"- DINO-improved cases: `{overall.get('dino_improved_cases', 0)}/{len(valid)}`.",
        "",
        "Negative deltas mean that cleaning only the retrieved memory content improved the generated target section. A consistently negative result supports corruption propagation through autoregressive memory reuse.",
        "",
        "## Outputs",
        "",
        "- `tables/frame_metrics.csv`",
        "- `tables/case_summary.csv`",
        "- `tables/overall_summary.json`",
        "- `figures/case_*.png`",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--replay_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--prefix_check_samples", type=int, default=16)
    parser.add_argument("--prefix_mae_tolerance", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    plan = load_csv(args.plan)
    if not plan:
        raise RuntimeError(f"Replay plan is empty: {args.plan}")

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from utils.evaluate_context_memory import LearnedMetricRunner

    metric_runner = LearnedMetricRunner(
        ["lpips", "dino"],
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    frame_rows = []
    case_rows = []

    for case in plan:
        row_idx = int(case["row"])
        section_idx = int(case["section_idx"])
        item = items[row_idx]
        case_dir = args.replay_root / case_directory_name(case)
        filename = f"{item['output_prefix']}custom.mp4"
        trace_name = f"{item['output_prefix']}custom.jsonl"

        replay_control_path = case_dir / "control" / filename
        replay_control_trace = case_dir / "control" / "access_traces" / trace_name
        if replay_control_path.is_file() and replay_control_trace.is_file():
            control_path = replay_control_path
            control_trace = replay_control_trace
            control_source = "replay_control"
        else:
            control_path = args.root / args.baseline_run / filename
            control_trace = (
                args.root / args.baseline_run / "access_traces" / trace_name
            )
            control_source = "existing_baseline"

        clean_path = case_dir / "clean_gt" / filename
        clean_trace = case_dir / "clean_gt" / "access_traces" / trace_name
        required = [control_path, control_trace, clean_path, clean_trace]
        if not all(path.is_file() for path in required):
            message = (
                f"case {case['case_index']} is incomplete: "
                f"{[str(path) for path in required if not path.is_file()]}"
            )
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue

        section_start = section_idx * SECTION_STRIDE
        target_indices = list(
            range(
                section_start + 1,
                section_start + 1 + PREDICT_FRAMES,
                args.frame_stride,
            )
        )
        prefix_indices = sorted(
            {
                int(round(value))
                for value in np.linspace(
                    0,
                    section_start,
                    min(args.prefix_check_samples, section_start + 1),
                )
            }
        )
        requested = sorted(set(target_indices + prefix_indices))
        control_frames = read_video_frames(control_path, requested)
        clean_frames = read_video_frames(clean_path, requested)
        gt_frames = read_gt_frames(
            item, target_indices, dataset_root=args.dataset_root
        )

        prefix_mae = float(
            np.mean(
                [
                    np.mean(
                        np.abs(
                            control_frames[idx].astype(np.float32)
                            - clean_frames[idx].astype(np.float32)
                        )
                    )
                    for idx in prefix_indices
                ]
            )
        )
        control_rows = selected_trace_rows(control_trace, max_section=section_idx)
        clean_rows = selected_trace_rows(clean_trace, max_section=section_idx)
        selection_match, cleaned_slots, content_sources_valid = compare_trace_content(
            control_rows, clean_rows, section_idx
        )
        matched_valid = bool(
            selection_match
            and cleaned_slots == PREDICT_FRAMES
            and content_sources_valid
            and prefix_mae <= float(args.prefix_mae_tolerance)
        )

        current_frame_rows = []
        for start in range(0, len(target_indices), args.batch_size):
            batch_indices = target_indices[start : start + args.batch_size]
            gt_batch = [gt_frames[idx] for idx in batch_indices]
            control_batch = [control_frames[idx] for idx in batch_indices]
            clean_batch = [clean_frames[idx] for idx in batch_indices]
            control_metrics = metric_runner.compute_batch(control_batch, gt_batch)
            clean_metrics = metric_runner.compute_batch(clean_batch, gt_batch)
            for idx, control_metric, clean_metric in zip(
                batch_indices, control_metrics, clean_metrics
            ):
                gt_pixels = resize_like(gt_frames[idx], control_frames[idx])
                control_mae = float(
                    np.mean(
                        np.abs(
                            control_frames[idx].astype(np.float32)
                            - gt_pixels.astype(np.float32)
                        )
                    )
                )
                clean_mae = float(
                    np.mean(
                        np.abs(
                            clean_frames[idx].astype(np.float32)
                            - gt_pixels.astype(np.float32)
                        )
                    )
                )
                current_frame_rows.append(
                    {
                        "case_index": int(case["case_index"]),
                        "row": row_idx,
                        "scene": item["scene"],
                        "section_idx": section_idx,
                        "target_frame": idx,
                        "matched_intervention_valid": int(matched_valid),
                        "control_mae": control_mae,
                        "clean_mae": clean_mae,
                        "mae_delta": clean_mae - control_mae,
                        "control_lpips": control_metric["lpips_alex"],
                        "clean_lpips": clean_metric["lpips_alex"],
                        "lpips_delta": (
                            clean_metric["lpips_alex"]
                            - control_metric["lpips_alex"]
                        ),
                        "control_dino_distance": control_metric["dino_distance"],
                        "clean_dino_distance": clean_metric["dino_distance"],
                        "dino_distance_delta": (
                            clean_metric["dino_distance"]
                            - control_metric["dino_distance"]
                        ),
                    }
                )
        frame_rows.extend(current_frame_rows)

        case_row = {
            "case_index": int(case["case_index"]),
            "row": row_idx,
            "scene": item["scene"],
            "section_idx": section_idx,
            "control_source": control_source,
            "planned_selected_memory_corruption": float(
                case["mean_selected_memory_corruption"]
            ),
            "cleaned_context_slots": cleaned_slots,
            "selection_match": int(selection_match),
            "content_sources_valid": int(content_sources_valid),
            "prefix_mae": prefix_mae,
            "matched_intervention_valid": int(matched_valid),
            "evaluated_frames": len(current_frame_rows),
        }
        for field in (
            "control_mae",
            "clean_mae",
            "mae_delta",
            "control_lpips",
            "clean_lpips",
            "lpips_delta",
            "control_dino_distance",
            "clean_dino_distance",
            "dino_distance_delta",
        ):
            case_row[field] = mean(row[field] for row in current_frame_rows)
        case_rows.append(case_row)
        save_case_montage(
            case,
            item,
            gt_frames,
            control_frames,
            clean_frames,
            target_indices,
            figures_dir / f"case_{int(case['case_index']):02d}.png",
        )
        print(
            f"case {case['case_index']}: valid={matched_valid} "
            f"LPIPS delta={case_row['lpips_delta']:+.5f} "
            f"DINO delta={case_row['dino_distance_delta']:+.5f}"
        )

    if not case_rows:
        raise RuntimeError("No complete memory-cleaning replays were evaluated")
    valid_rows = [row for row in case_rows if row["matched_intervention_valid"]]
    lpips_deltas = [row["lpips_delta"] for row in valid_rows]
    dino_deltas = [row["dino_distance_delta"] for row in valid_rows]
    lpips_low, lpips_high = bootstrap_mean_interval(lpips_deltas)
    dino_low, dino_high = bootstrap_mean_interval(dino_deltas)
    overall = {
        "completed_cases": len(case_rows),
        "valid_matched_cases": len(valid_rows),
        "lpips_delta_mean": mean(lpips_deltas),
        "lpips_delta_ci_low": lpips_low,
        "lpips_delta_ci_high": lpips_high,
        "lpips_improved_cases": sum(value < 0 for value in lpips_deltas),
        "dino_distance_delta_mean": mean(dino_deltas),
        "dino_distance_delta_ci_low": dino_low,
        "dino_distance_delta_ci_high": dino_high,
        "dino_improved_cases": sum(value < 0 for value in dino_deltas),
    }

    write_csv(tables_dir / "frame_metrics.csv", frame_rows)
    write_csv(tables_dir / "case_summary.csv", case_rows)
    (tables_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", case_rows, overall)
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
