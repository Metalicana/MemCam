import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.poses import load_c2ws_from_json


SUPPORTED_METRICS = {"rgb", "gradient", "lpips", "dino"}
METRIC_FIELDS = {
    "rgb": "rgb_rmse",
    "gradient": "gradient_rmse",
    "lpips": "lpips_alex",
    "dino": "dino_distance",
}
DEFAULT_ERROR_THRESHOLDS = {
    "rgb_rmse": [0.01, 0.02, 0.03, 0.05, 0.075, 0.10],
    "gradient_rmse": [0.01, 0.02, 0.03, 0.05, 0.075, 0.10],
    "lpips_alex": [0.01, 0.025, 0.05, 0.10, 0.20],
    "dino_distance": [0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
}


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value):
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def parse_rows(value):
    if not value:
        return None
    rows = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            rows.update(range(int(start_text), int(end_text) + 1))
        else:
            rows.add(int(part))
    return rows


def load_manifest(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_row"] = row_index
            rows.append(item)
    return rows


def select_items(items, row_filter, durations, limit):
    selected = []
    duration_filter = set(durations) if durations else None
    for item in items:
        if row_filter is not None and item["_row"] not in row_filter:
            continue
        if duration_filter is not None and int(item["duration_sec"]) not in duration_filter:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def resolve_gt_frames_dir(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "frames" / item["scene"]
    return Path(item["gt_frames_dir"])


def resolve_pose_path(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "jsons" / f"{item['scene']}.json"
    return Path(item["pose_path"])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rotation_distances_deg(rotations, target_rotation):
    traces = np.einsum("nij,ij->n", rotations, target_rotation)
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def build_pixel_to_ue_matrix(width, height, horizontal_fov_deg, vertical_fov_deg):
    fx = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    fy = 0.5 * height / math.tan(math.radians(vertical_fov_deg) / 2.0)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    # UE camera coordinates are +X forward, +Y right, +Z up.
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / fx, 0.0, -cx / fx],
            [0.0, -1.0 / fy, cy / fy],
        ],
        dtype=np.float64,
    )


def rotation_homography_output_to_input(
    output_c2w,
    input_c2w,
    width,
    height,
    horizontal_fov_deg,
    vertical_fov_deg,
):
    pixel_to_ue = build_pixel_to_ue_matrix(
        width,
        height,
        horizontal_fov_deg,
        vertical_fov_deg,
    )
    output_to_input_rotation = input_c2w[:3, :3].T @ output_c2w[:3, :3]
    homography = (
        np.linalg.inv(pixel_to_ue)
        @ output_to_input_rotation
        @ pixel_to_ue
    )
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    homography[np.isclose(homography, 0.0, atol=1e-12)] = 0.0
    homography[np.isclose(homography, 1.0, atol=1e-12)] = 1.0
    homography[np.isclose(homography, -1.0, atol=1e-12)] = -1.0
    return homography


def warp_perspective(image, output_to_input_homography, output_size, resample):
    matrix = np.asarray(output_to_input_homography, dtype=np.float64)
    if abs(matrix[2, 2]) <= 1e-12:
        raise ValueError("Degenerate rotation homography.")
    matrix = matrix / matrix[2, 2]
    coefficients = tuple(matrix.reshape(-1)[:8])
    return image.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=resample,
        fillcolor=0,
    )


def erode_mask(mask, iterations=2):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return result


def align_input_to_output(
    output_image,
    input_image,
    output_c2w,
    input_c2w,
    horizontal_fov_deg,
    vertical_fov_deg,
):
    if output_image.size != input_image.size:
        raise ValueError(
            f"Frame sizes differ: output={output_image.size}, input={input_image.size}"
        )
    width, height = output_image.size
    homography = rotation_homography_output_to_input(
        output_c2w,
        input_c2w,
        width,
        height,
        horizontal_fov_deg,
        vertical_fov_deg,
    )
    aligned = warp_perspective(
        input_image,
        homography,
        output_image.size,
        Image.Resampling.BILINEAR,
    )
    source_mask = Image.new("L", input_image.size, 255)
    warped_mask = warp_perspective(
        source_mask,
        homography,
        output_image.size,
        Image.Resampling.NEAREST,
    )
    mask = erode_mask(np.asarray(warped_mask) > 0, iterations=2)
    return np.asarray(output_image.convert("RGB")), np.asarray(aligned.convert("RGB")), mask


def masked_rgb_rmse(reference, aligned, mask):
    if not np.any(mask):
        return None
    diff = reference.astype(np.float64) / 255.0 - aligned.astype(np.float64) / 255.0
    return math.sqrt(float(np.mean(diff[mask] ** 2)))


def rgb_to_gray(array):
    array = array.astype(np.float64) / 255.0
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def masked_gradient_rmse(reference, aligned, mask):
    valid = erode_mask(mask, iterations=1)
    if not np.any(valid):
        return None
    reference_y, reference_x = np.gradient(rgb_to_gray(reference))
    aligned_y, aligned_x = np.gradient(rgb_to_gray(aligned))
    diff_x = reference_x - aligned_x
    diff_y = reference_y - aligned_y
    squared = 0.5 * (diff_x * diff_x + diff_y * diff_y)
    return math.sqrt(float(np.mean(squared[valid])))


def prepare_learned_pair(reference, aligned, mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    reference_crop = reference[top:bottom, left:right].copy()
    aligned_crop = aligned[top:bottom, left:right].copy()
    crop_mask = mask[top:bottom, left:right]
    reference_crop[~crop_mask] = 127
    aligned_crop[~crop_mask] = 127
    return reference_crop, aligned_crop


def load_frame(frames_dir, start_frame, local_index):
    frame_path = frames_dir / f"{int(start_frame) + int(local_index):04d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing GT frame: {frame_path}")
    with Image.open(frame_path) as image:
        return image.convert("RGB")


def candidate_pairs_for_item(
    c2ws,
    sample_indices,
    min_gap_frames,
    position_thresholds,
    rotation_thresholds,
    candidates_per_return,
):
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    max_position = max(position_thresholds)
    max_rotation = max(rotation_thresholds)
    selected = {}

    sampled = np.asarray(sample_indices, dtype=np.int64)
    for j in sample_indices:
        earlier = sampled[sampled <= j - min_gap_frames]
        if not len(earlier):
            continue
        position_distances = np.linalg.norm(positions[earlier] - positions[j], axis=1)
        broad_mask = position_distances <= max_position
        if not np.any(broad_mask):
            continue
        earlier = earlier[broad_mask]
        position_distances = position_distances[broad_mask]
        rotation_distances = rotation_distances_deg(rotations[earlier], rotations[j])
        broad_mask = rotation_distances <= max_rotation
        earlier = earlier[broad_mask]
        position_distances = position_distances[broad_mask]
        rotation_distances = rotation_distances[broad_mask]
        if not len(earlier):
            continue

        for position_threshold in position_thresholds:
            for rotation_threshold in rotation_thresholds:
                eligible = np.flatnonzero(
                    (position_distances <= position_threshold)
                    & (rotation_distances <= rotation_threshold)
                )
                if not len(eligible):
                    continue
                scores = (
                    position_distances[eligible] / max(position_threshold, 1e-12)
                    + rotation_distances[eligible] / max(rotation_threshold, 1e-12)
                )
                order = eligible[np.argsort(scores)[:candidates_per_return]]
                for index in order:
                    key = (int(earlier[index]), int(j))
                    record = selected.setdefault(
                        key,
                        {
                            "frame_i": key[0],
                            "frame_j": key[1],
                            "position_distance": float(position_distances[index]),
                            "rotation_deg": float(rotation_distances[index]),
                            "selection_support": 0,
                        },
                    )
                    record["selection_support"] += 1
    return list(selected.values())


def choose_one_pair_per_return(rows, position_threshold, rotation_threshold, min_overlap):
    grouped = defaultdict(list)
    for row in rows:
        if float(row["position_distance"]) > position_threshold:
            continue
        if float(row["rotation_deg"]) > rotation_threshold:
            continue
        if float(row["overlap_fraction"]) < min_overlap:
            continue
        grouped[(int(row["row"]), int(row["frame_j"]))].append(row)

    selected = []
    for candidates in grouped.values():
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    float(row["position_distance"]) / max(position_threshold, 1e-12)
                    + float(row["rotation_deg"]) / max(rotation_threshold, 1e-12),
                    -float(row["overlap_fraction"]),
                    -int(row["frame_i"]),
                ),
            )
        )
    return selected


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def build_threshold_sweep(
    pair_rows,
    total_videos,
    position_thresholds,
    rotation_thresholds,
    overlap_thresholds,
    metric_fields,
):
    summaries = []
    for position_threshold in position_thresholds:
        for rotation_threshold in rotation_thresholds:
            for overlap_threshold in overlap_thresholds:
                selected = choose_one_pair_per_return(
                    pair_rows,
                    position_threshold,
                    rotation_threshold,
                    overlap_threshold,
                )
                for metric_field in metric_fields:
                    metric_rows = [row for row in selected if row.get(metric_field) is not None]
                    values = [float(row[metric_field]) for row in metric_rows]
                    videos_before = len({int(row["row"]) for row in metric_rows})
                    for error_threshold in DEFAULT_ERROR_THRESHOLDS[metric_field]:
                        accepted = [
                            row
                            for row in metric_rows
                            if float(row[metric_field]) <= error_threshold
                        ]
                        videos_after = len({int(row["row"]) for row in accepted})
                        summaries.append(
                            {
                                "oracle_metric": metric_field,
                                "oracle_error_threshold": error_threshold,
                                "position_threshold": position_threshold,
                                "rotation_threshold_deg": rotation_threshold,
                                "min_overlap": overlap_threshold,
                                "pairs_before_oracle_filter": len(metric_rows),
                                "videos_before_oracle_filter": videos_before,
                                "pairs_after_oracle_filter": len(accepted),
                                "videos_after_oracle_filter": videos_after,
                                "video_coverage_fraction": (
                                    videos_after / total_videos if total_videos else 0.0
                                ),
                                "mean_oracle_error_before": (
                                    float(np.mean(values)) if values else None
                                ),
                                "median_oracle_error_before": percentile(values, 50),
                                "p90_oracle_error_before": percentile(values, 90),
                                "max_oracle_error_before": max(values) if values else None,
                            }
                        )
    return summaries


def select_best_thresholds(sweep_rows):
    grouped = defaultdict(list)
    for row in sweep_rows:
        grouped[(row["oracle_metric"], float(row["oracle_error_threshold"]))].append(row)

    best = []
    for _, rows in sorted(grouped.items()):
        best.append(
            max(
                rows,
                key=lambda row: (
                    int(row["videos_after_oracle_filter"]),
                    int(row["pairs_after_oracle_filter"]),
                    float(row["min_overlap"]),
                    -float(row["position_threshold"]),
                    -float(row["rotation_threshold_deg"]),
                ),
            )
        )
    return best


def plot_best_coverage(best_rows, output_path):
    grouped = defaultdict(list)
    for row in best_rows:
        grouped[row["oracle_metric"]].append(row)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for metric, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["oracle_error_threshold"]))
        ax.plot(
            [float(row["oracle_error_threshold"]) for row in rows],
            [float(row["video_coverage_fraction"]) for row in rows],
            marker="o",
            linewidth=1.8,
            label=metric,
        )
    ax.set_xlabel("maximum GT oracle error")
    ax.set_ylabel("fraction of videos with >=1 accepted revisit")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate pose, rotation, FOV-overlap, and GT visual-error thresholds "
            "for revisit evaluation. Rotation-only homographies use the dataset's "
            "Unreal +X-forward camera convention."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--durations", type=str, default="180")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_stride", type=int, default=30)
    parser.add_argument("--min_gap_sec", type=float, default=15.0)
    parser.add_argument(
        "--position_thresholds",
        type=str,
        default="0.10,0.25,0.50,1.00",
    )
    parser.add_argument(
        "--rotation_thresholds_deg",
        type=str,
        default="2,5,10,20,30,45,60",
    )
    parser.add_argument(
        "--overlap_thresholds",
        type=str,
        default="0.20,0.35,0.50,0.70",
    )
    parser.add_argument("--horizontal_fov_deg", type=float, default=90.0)
    parser.add_argument("--vertical_fov_deg", type=float, default=60.0)
    parser.add_argument("--candidates_per_return", type=int, default=3)
    parser.add_argument("--metrics", type=str, default="rgb,gradient")
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=16)
    parser.add_argument("--learned_image_size", type=int, default=224)
    args = parser.parse_args()

    if args.sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1")
    if args.min_gap_sec <= 0:
        raise ValueError("--min_gap_sec must be > 0")
    if args.candidates_per_return < 1:
        raise ValueError("--candidates_per_return must be >= 1")
    if not 0.0 < args.horizontal_fov_deg < 180.0:
        raise ValueError("--horizontal_fov_deg must be between 0 and 180")
    if not 0.0 < args.vertical_fov_deg < 180.0:
        raise ValueError("--vertical_fov_deg must be between 0 and 180")

    metrics = parse_str_list(args.metrics)
    unknown_metrics = sorted(set(metrics) - SUPPORTED_METRICS)
    if unknown_metrics:
        raise ValueError(
            f"Unsupported metrics {unknown_metrics}; expected {sorted(SUPPORTED_METRICS)}"
        )
    position_thresholds = sorted(set(parse_float_list(args.position_thresholds)))
    rotation_thresholds = sorted(set(parse_float_list(args.rotation_thresholds_deg)))
    overlap_thresholds = sorted(set(parse_float_list(args.overlap_thresholds)))
    if not position_thresholds or min(position_thresholds) <= 0:
        raise ValueError("Position thresholds must be positive.")
    if not rotation_thresholds or min(rotation_thresholds) <= 0:
        raise ValueError("Rotation thresholds must be positive.")
    if not overlap_thresholds or min(overlap_thresholds) < 0 or max(overlap_thresholds) > 1:
        raise ValueError("Overlap thresholds must be in [0, 1].")

    items = select_items(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    learned_metrics = [metric for metric in metrics if metric in {"lpips", "dino"}]
    learned_runner = None
    if learned_metrics:
        from utils.evaluate_context_memory import LearnedMetricRunner

        learned_runner = LearnedMetricRunner(
            learned_metrics,
            device=args.metric_device,
            batch_size=args.metric_batch_size,
            image_size=args.learned_image_size,
        )

    pair_rows = []
    pending_learned = []

    def flush_learned():
        if not pending_learned:
            return
        references = [entry[1] for entry in pending_learned]
        aligned = [entry[2] for entry in pending_learned]
        results = learned_runner.compute_batch(references, aligned)
        for (row, _, _), result in zip(pending_learned, results):
            row.update(result)
        pending_learned.clear()

    for item_index, item in enumerate(items, start=1):
        fps = float(item["fps"])
        min_gap_frames = max(1, int(math.ceil(args.min_gap_sec * fps)))
        c2ws = load_c2ws_from_json(
            json_path=resolve_pose_path(item, args.dataset_root),
            start_frame=int(item["start_frame"]),
            num_frames=int(item["num_frames"]),
        )
        sample_indices = list(range(0, len(c2ws), args.sample_stride))
        if sample_indices[-1] != len(c2ws) - 1:
            sample_indices.append(len(c2ws) - 1)
        candidates = candidate_pairs_for_item(
            c2ws,
            sample_indices,
            min_gap_frames,
            position_thresholds,
            rotation_thresholds,
            args.candidates_per_return,
        )
        frames_dir = resolve_gt_frames_dir(item, args.dataset_root)
        frame_cache = {}

        def frame(local_index):
            if local_index not in frame_cache:
                frame_cache[local_index] = load_frame(
                    frames_dir,
                    item["start_frame"],
                    local_index,
                )
            return frame_cache[local_index]

        for candidate in candidates:
            frame_i = int(candidate["frame_i"])
            frame_j = int(candidate["frame_j"])
            reference, aligned, mask = align_input_to_output(
                frame(frame_i),
                frame(frame_j),
                c2ws[frame_i],
                c2ws[frame_j],
                args.horizontal_fov_deg,
                args.vertical_fov_deg,
            )
            overlap_fraction = float(np.mean(mask))
            row = {
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "fps": fps,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "dataset_frame_i": int(item["start_frame"]) + frame_i,
                "dataset_frame_j": int(item["start_frame"]) + frame_j,
                "time_i_sec": frame_i / fps,
                "time_j_sec": frame_j / fps,
                "gap_sec": (frame_j - frame_i) / fps,
                "position_distance": candidate["position_distance"],
                "rotation_deg": candidate["rotation_deg"],
                "overlap_fraction": overlap_fraction,
                "selection_support": candidate["selection_support"],
                "horizontal_fov_deg": args.horizontal_fov_deg,
                "vertical_fov_deg": args.vertical_fov_deg,
            }
            if "rgb" in metrics:
                row["rgb_rmse"] = masked_rgb_rmse(reference, aligned, mask)
            if "gradient" in metrics:
                row["gradient_rmse"] = masked_gradient_rmse(reference, aligned, mask)
            pair_rows.append(row)

            if learned_runner is not None:
                learned_pair = prepare_learned_pair(reference, aligned, mask)
                if learned_pair is not None:
                    pending_learned.append((row, *learned_pair))
                    if len(pending_learned) >= args.metric_batch_size:
                        flush_learned()

        print(
            f"[{item_index}/{len(items)}] row={item['_row']} scene={item['scene']} "
            f"sampled={len(sample_indices)} candidates={len(candidates)}"
        )

    if learned_runner is not None:
        flush_learned()

    metric_fields = [METRIC_FIELDS[metric] for metric in metrics]
    sweep_rows = build_threshold_sweep(
        pair_rows,
        total_videos=len(items),
        position_thresholds=position_thresholds,
        rotation_thresholds=rotation_thresholds,
        overlap_thresholds=overlap_thresholds,
        metric_fields=metric_fields,
    )
    best_rows = select_best_thresholds(sweep_rows)

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    write_csv(tables_dir / "gt_aligned_pairs.csv", pair_rows)
    write_csv(tables_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(tables_dir / "best_thresholds.csv", best_rows)
    plot_best_coverage(best_rows, figures_dir / "coverage_vs_oracle_error.png")

    config = {
        "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "selected_videos": len(items),
        "sample_stride": args.sample_stride,
        "min_gap_sec": args.min_gap_sec,
        "position_thresholds": position_thresholds,
        "rotation_thresholds_deg": rotation_thresholds,
        "overlap_thresholds": overlap_thresholds,
        "horizontal_fov_deg": args.horizontal_fov_deg,
        "vertical_fov_deg": args.vertical_fov_deg,
        "candidates_per_return": args.candidates_per_return,
        "metrics": metrics,
        "candidate_pairs": len(pair_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    report_path = args.output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# GT Revisit Alignment Calibration\n\n")
        handle.write(f"- Videos: `{len(items)}`\n")
        handle.write(f"- Candidate pairs scored: `{len(pair_rows)}`\n")
        handle.write(f"- Metrics: `{','.join(metrics)}`\n")
        handle.write(
            f"- FOV: `{args.horizontal_fov_deg}` horizontal, "
            f"`{args.vertical_fov_deg}` vertical\n\n"
        )
        handle.write(
            "`best_thresholds.csv` maximizes the number of videos with at least one "
            "GT-valid revisit at each oracle-error tolerance. The raw GT error is kept "
            "for calibration; generated revisit error should later be reported as "
            "`max(0, generated_error - gt_oracle_error)`, which is zero for GT.\n\n"
        )
        handle.write("## Files\n\n")
        handle.write("- `tables/gt_aligned_pairs.csv`\n")
        handle.write("- `tables/threshold_sweep.csv`\n")
        handle.write("- `tables/best_thresholds.csv`\n")
        handle.write("- `figures/coverage_vs_oracle_error.png`\n")

    print(f"Wrote: {tables_dir / 'gt_aligned_pairs.csv'}")
    print(f"Wrote: {tables_dir / 'threshold_sweep.csv'}")
    print(f"Wrote: {tables_dir / 'best_thresholds.csv'}")
    print(f"Wrote: {figures_dir / 'coverage_vs_oracle_error.png'}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
