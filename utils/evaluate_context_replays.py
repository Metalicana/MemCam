import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SECTION_STRIDE = 76
PREDICT_FRAMES = 76


def load_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_row"] = row_idx
            rows.append(item)
    return rows


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def case_directory_name(case):
    return (
        f"case_{int(case['case_index']):02d}_row{int(case['row'])}_"
        f"section{int(case['section_idx'])}"
    )


def indexed_frame_paths(frame_dir):
    import re

    paths = {}
    for path in Path(frame_dir).iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match:
            paths[int(match.group(1))] = path
    return paths


def resolve_gt_dir(item, dataset_root=None):
    if dataset_root is not None:
        candidate = Path(dataset_root) / "frames" / item["scene"]
        if candidate.is_dir():
            return candidate
    original = item.get("gt_frames_dir")
    if original and Path(original).is_dir():
        return Path(original)
    raise FileNotFoundError(
        f"Ground-truth frames are unavailable for row {item['_row']}"
    )


def read_gt_frames(item, local_indices, dataset_root=None):
    from PIL import Image

    frame_dir = resolve_gt_dir(item, dataset_root=dataset_root)
    paths = indexed_frame_paths(frame_dir)
    output = {}
    for local_idx in sorted(set(int(value) for value in local_indices)):
        dataset_idx = int(item["start_frame"]) + local_idx
        path = paths.get(dataset_idx)
        if path is None:
            raise FileNotFoundError(f"Missing GT frame {dataset_idx} in {frame_dir}")
        with Image.open(path) as image:
            output[local_idx] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return output


def read_video_frames(path, frame_indices):
    import imageio.v2 as imageio

    wanted = sorted(set(int(value) for value in frame_indices))
    if not wanted:
        return {}
    wanted_set = set(wanted)
    last = wanted[-1]
    output = {}
    reader = imageio.get_reader(str(path))
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx in wanted_set:
                output[frame_idx] = np.asarray(frame, dtype=np.uint8)
            if frame_idx >= last:
                break
    finally:
        reader.close()
    missing = sorted(wanted_set - set(output))
    if missing:
        raise RuntimeError(f"Video {path} is missing requested frames {missing[:10]}")
    return output


def resize_like(frame, reference):
    if frame.shape == reference.shape:
        return frame
    from PIL import Image

    height, width = reference.shape[:2]
    return np.asarray(
        Image.fromarray(frame).resize((width, height), resample=Image.BICUBIC),
        dtype=np.uint8,
    )


def selected_trace_map(path, max_section=None):
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
            output[(section_idx, int(row["target_frame"]))] = int(
                row["selected_memory_frame"]
            )
    return output


def compare_replay_trace_maps(control_map, swap_map, section_idx):
    section_idx = int(section_idx)
    control_history = {
        key: value for key, value in control_map.items() if key[0] < section_idx
    }
    swap_history = {
        key: value for key, value in swap_map.items() if key[0] < section_idx
    }
    control_target = {
        key: value for key, value in control_map.items() if key[0] == section_idx
    }
    swap_target = {
        key: value for key, value in swap_map.items() if key[0] == section_idx
    }
    trace_history_match = control_history == swap_history
    target_key_match = bool(control_target) and set(control_target) == set(swap_target)
    changed_target_count = (
        sum(control_target[key] != swap_target[key] for key in control_target)
        if target_key_match
        else 0
    )
    return trace_history_match, target_key_match, changed_target_count


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def bootstrap_mean_interval(values, seed=0, repeats=10000):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_case_montage(case, item, gt, control, swap, target_indices, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.linspace(0, len(target_indices) - 1, min(6, len(target_indices)))
    chosen = [target_indices[int(round(position))] for position in positions]
    fig, axes = plt.subplots(3, len(chosen), figsize=(3.0 * len(chosen), 7.4))
    if len(chosen) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    rows = [("Target GT", gt), ("Unbounded context", control), ("SLAM context swap", swap)]
    for row_idx, (label, frames) in enumerate(rows):
        for col, frame_idx in enumerate(chosen):
            axes[row_idx, col].imshow(frames[frame_idx])
            axes[row_idx, col].axis("off")
            if row_idx == 0:
                axes[row_idx, col].set_title(f"{frame_idx / 30.0:.1f}s", fontsize=9)
            if col == 0:
                axes[row_idx, col].set_ylabel(label, fontsize=10)
    fig.suptitle(
        f"Matched context replay | row {case['row']} {item['scene']} | "
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


def save_summary_figure(case_rows, output_path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("lpips_delta", "LPIPS change"),
        ("dino_distance_delta", "DINO-distance change"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    case_rows = [row for row in case_rows if row["matched_history_valid"]]
    labels = [f"case {row['case_index']}" for row in case_rows]
    for ax, (field, title) in zip(axes, metrics):
        values = [float(row[field]) for row in case_rows]
        colors = ["#3f8f5f" if value < 0 else "#c55252" for value in values]
        ax.bar(labels, values, color=colors)
        ax.axhline(0.0, color="#222222", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("swap minus control (negative is better)")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def fmt(value, digits=5):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows, fields):
    if not rows:
        return "_No completed replay pairs._"
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
    valid = [row for row in case_rows if row["matched_history_valid"]]
    lines = [
        "# Matched Context-Swap Replay",
        "",
        "## Intervention",
        "",
        "Both branches use the same model, prompt, camera path, seed, unbounded bank, and recorded unbounded contexts through every preceding section. At the target section only, the intervention branch substitutes the contexts selected by the bounded SLAM-style run.",
        "",
        "A pair is valid only when its pre-intervention trace choices match and its sampled pre-intervention output pixels match.",
        "",
        "The current replay plan deliberately contains the strongest late cases predicted by the offline DINO diagnostic. These replays test whether context selection can cause a failure in those selected cases; they do not estimate how often this happens over the full benchmark.",
        "",
        "## Cases",
        "",
        markdown_table(
            case_rows,
            [
                "case_index",
                "row",
                "section_idx",
                "changed_target_count",
                "target_key_match",
                "prefix_mae",
                "matched_history_valid",
                "control_lpips",
                "swap_lpips",
                "lpips_delta",
                "control_dino_distance",
                "swap_dino_distance",
                "dino_distance_delta",
            ],
        ),
        "",
        "## Aggregate",
        "",
        f"- Valid matched pairs: `{len(valid)}/{len(case_rows)}`.",
        f"- Mean LPIPS change (swap - control): `{fmt(overall.get('lpips_delta_mean'))}`; 95% case-bootstrap CI `{fmt(overall.get('lpips_delta_ci_low'))}` to `{fmt(overall.get('lpips_delta_ci_high'))}`.",
        f"- Mean DINO-distance change: `{fmt(overall.get('dino_distance_delta_mean'))}`; 95% case-bootstrap CI `{fmt(overall.get('dino_distance_delta_ci_low'))}` to `{fmt(overall.get('dino_distance_delta_ci_high'))}`.",
        f"- LPIPS-improved pairs: `{overall.get('lpips_improved_pairs', 0)}/{len(valid)}`.",
        f"- DINO-improved pairs: `{overall.get('dino_improved_pairs', 0)}/{len(valid)}`.",
        "",
        "Negative deltas mean that changing only the retrieved contexts improved the generated section. With several valid pairs and a consistently negative delta, this is evidence that unbounded degradation is caused by context selection rather than memory capacity itself.",
        "",
        "## Outputs",
        "",
        "- `tables/frame_metrics.csv`",
        "- `tables/case_summary.csv`",
        "- `tables/overall_summary.json`",
        "- `figures/replay_metric_deltas.png`",
        "- `figures/case_*.png`",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate paired context-only MemCam replay interventions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replay_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--prefix_check_samples", type=int, default=16)
    parser.add_argument("--prefix_mae_tolerance", type=float, default=1e-6)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1 or args.prefix_check_samples < 1:
        raise ValueError("Frame stride and prefix check samples must be positive")
    items = load_manifest(args.manifest)
    plan = load_csv(args.plan)
    if not plan:
        raise RuntimeError(f"Replay plan is empty: {args.plan}")

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
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
        control_dir = case_dir / "control"
        swap_name = f"swap_{case['intervention_run']}"
        swap_dir = case_dir / swap_name
        filename = f"{item['output_prefix']}custom.mp4"
        control_path = control_dir / filename
        swap_path = swap_dir / filename
        control_trace = control_dir / "access_traces" / f"{item['output_prefix']}custom.jsonl"
        swap_trace = swap_dir / "access_traces" / f"{item['output_prefix']}custom.jsonl"
        required = [control_path, swap_path, control_trace, swap_trace]
        if not all(path.is_file() for path in required):
            message = f"case {case['case_index']} is incomplete: {[str(path) for path in required if not path.is_file()]}"
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
        requested_indices = sorted(set(target_indices + prefix_indices))
        control_frames = read_video_frames(control_path, requested_indices)
        swap_frames = read_video_frames(swap_path, requested_indices)
        gt_frames = read_gt_frames(
            item, target_indices, dataset_root=args.dataset_root
        )

        prefix_mae = float(
            np.mean(
                [
                    np.mean(
                        np.abs(
                            control_frames[idx].astype(np.float32)
                            - swap_frames[idx].astype(np.float32)
                        )
                    )
                    for idx in prefix_indices
                ]
            )
        )
        control_map = selected_trace_map(control_trace, max_section=section_idx)
        swap_map = selected_trace_map(swap_trace, max_section=section_idx)
        trace_history_match, target_key_match, changed_target_count = (
            compare_replay_trace_maps(control_map, swap_map, section_idx)
        )
        matched_history_valid = bool(
            trace_history_match
            and target_key_match
            and prefix_mae <= float(args.prefix_mae_tolerance)
            and changed_target_count > 0
        )

        for start in range(0, len(target_indices), args.batch_size):
            batch_indices = target_indices[start : start + args.batch_size]
            gt_batch = [gt_frames[idx] for idx in batch_indices]
            control_batch = [control_frames[idx] for idx in batch_indices]
            swap_batch = [swap_frames[idx] for idx in batch_indices]
            control_metrics = metric_runner.compute_batch(control_batch, gt_batch)
            swap_metrics = metric_runner.compute_batch(swap_batch, gt_batch)
            for idx, control_metric, swap_metric in zip(
                batch_indices, control_metrics, swap_metrics
            ):
                gt_for_pixels = resize_like(gt_frames[idx], control_frames[idx])
                control_mae = float(
                    np.mean(
                        np.abs(
                            control_frames[idx].astype(np.float32)
                            - gt_for_pixels.astype(np.float32)
                        )
                    )
                )
                swap_mae = float(
                    np.mean(
                        np.abs(
                            swap_frames[idx].astype(np.float32)
                            - gt_for_pixels.astype(np.float32)
                        )
                    )
                )
                frame_rows.append(
                    {
                        "case_index": int(case["case_index"]),
                        "row": row_idx,
                        "scene": item["scene"],
                        "section_idx": section_idx,
                        "target_frame": idx,
                        "matched_history_valid": int(matched_history_valid),
                        "control_mae": control_mae,
                        "swap_mae": swap_mae,
                        "mae_delta": swap_mae - control_mae,
                        "control_lpips": control_metric["lpips_alex"],
                        "swap_lpips": swap_metric["lpips_alex"],
                        "lpips_delta": (
                            swap_metric["lpips_alex"]
                            - control_metric["lpips_alex"]
                        ),
                        "control_dino_distance": control_metric["dino_distance"],
                        "swap_dino_distance": swap_metric["dino_distance"],
                        "dino_distance_delta": (
                            swap_metric["dino_distance"]
                            - control_metric["dino_distance"]
                        ),
                    }
                )

        current_rows = [
            row
            for row in frame_rows
            if int(row["case_index"]) == int(case["case_index"])
        ]
        case_row = {
            "case_index": int(case["case_index"]),
            "row": row_idx,
            "scene": item["scene"],
            "section_idx": section_idx,
            "changed_target_count": changed_target_count,
            "trace_history_match": int(trace_history_match),
            "target_key_match": int(target_key_match),
            "prefix_mae": prefix_mae,
            "matched_history_valid": int(matched_history_valid),
            "evaluated_frames": len(current_rows),
        }
        for field in (
            "control_mae",
            "swap_mae",
            "mae_delta",
            "control_lpips",
            "swap_lpips",
            "lpips_delta",
            "control_dino_distance",
            "swap_dino_distance",
            "dino_distance_delta",
        ):
            case_row[field] = mean(row[field] for row in current_rows)
        case_rows.append(case_row)
        save_case_montage(
            case,
            item,
            gt_frames,
            control_frames,
            swap_frames,
            target_indices,
            figures_dir / f"case_{int(case['case_index']):02d}.png",
        )
        print(
            f"case {case['case_index']}: valid={matched_history_valid} "
            f"LPIPS delta={case_row['lpips_delta']:+.5f} "
            f"DINO delta={case_row['dino_distance_delta']:+.5f}"
        )

    if not case_rows:
        raise RuntimeError("No complete replay pairs were evaluated")
    valid_rows = [row for row in case_rows if row["matched_history_valid"]]
    lpips_deltas = [row["lpips_delta"] for row in valid_rows]
    dino_deltas = [row["dino_distance_delta"] for row in valid_rows]
    lpips_low, lpips_high = bootstrap_mean_interval(lpips_deltas)
    dino_low, dino_high = bootstrap_mean_interval(dino_deltas)
    overall = {
        "completed_pairs": len(case_rows),
        "valid_matched_pairs": len(valid_rows),
        "lpips_delta_mean": mean(lpips_deltas),
        "lpips_delta_ci_low": lpips_low,
        "lpips_delta_ci_high": lpips_high,
        "lpips_improved_pairs": sum(value < 0 for value in lpips_deltas),
        "dino_distance_delta_mean": mean(dino_deltas),
        "dino_distance_delta_ci_low": dino_low,
        "dino_distance_delta_ci_high": dino_high,
        "dino_improved_pairs": sum(value < 0 for value in dino_deltas),
    }

    write_csv(tables_dir / "frame_metrics.csv", frame_rows)
    write_csv(tables_dir / "case_summary.csv", case_rows)
    (tables_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )
    save_summary_figure(case_rows, figures_dir / "replay_metric_deltas.png")
    write_report(output_dir / "report.md", case_rows, overall)
    print(f"Wrote report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
