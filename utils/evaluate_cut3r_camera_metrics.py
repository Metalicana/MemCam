import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.poses import load_c2ws_from_json


def parse_list(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


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


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def safe_round(value, digits=6):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
        f"{prefix}_max": safe_round(max(values)) if values else None,
    }


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predicted_poses(reconstruction_dir):
    camera_dir = reconstruction_dir / "camera"
    paths = sorted(camera_dir.glob("*.npz"))
    poses = []
    for path in paths:
        with np.load(path) as data:
            poses.append(np.asarray(data["pose"], dtype=np.float64))
    if not poses:
        raise RuntimeError(f"No CUT3R camera poses found under {camera_dir}")
    return np.stack(poses, axis=0)


def resolve_pose_path(metadata, dataset_root):
    if dataset_root is not None:
        return dataset_root / "jsons" / f"{metadata['scene']}.json"
    return Path(metadata["pose_path"])


def load_gt_poses(metadata, dataset_root):
    pose_path = resolve_pose_path(metadata, dataset_root)
    all_gt = load_c2ws_from_json(
        json_path=pose_path,
        start_frame=int(metadata["start_frame"]),
        num_frames=int(metadata["num_frames"]),
    ).astype(np.float64)

    local_indices = [int(idx) for idx in metadata["manifest_local_indices"]]
    local_indices = [min(max(idx, 0), len(all_gt) - 1) for idx in local_indices]
    return all_gt[local_indices]


def relative_poses(c2ws):
    first_inv = np.linalg.inv(c2ws[0])
    return np.stack([first_inv @ pose for pose in c2ws], axis=0)


def rotation_error_deg(pred_rot, gt_rot):
    delta = pred_rot @ np.swapaxes(gt_rot, -1, -2)
    traces = np.trace(delta, axis1=1, axis2=2)
    cos_values = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_values))


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def fit_scalar(pred_points, gt_points):
    denominator = float(np.sum(pred_points * pred_points))
    if denominator <= 1e-12:
        return 1.0
    return float(np.sum(gt_points * pred_points) / denominator)


def fit_umeyama(pred_points, gt_points):
    if len(pred_points) < 2:
        return pred_points.copy(), 1.0

    pred_mean = pred_points.mean(axis=0)
    gt_mean = gt_points.mean(axis=0)
    pred_centered = pred_points - pred_mean
    gt_centered = gt_points - gt_mean

    variance = float(np.mean(np.sum(pred_centered * pred_centered, axis=1)))
    if variance <= 1e-12:
        return pred_points.copy(), 1.0

    covariance = (gt_centered.T @ pred_centered) / len(pred_points)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u_mat) * np.linalg.det(vt_mat))
    correction = np.eye(3)
    correction[-1, -1] = sign if sign != 0 else 1.0

    rotation = u_mat @ correction @ vt_mat
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    aligned = (scale * (rotation @ pred_centered.T)).T + gt_mean
    return aligned, scale


def worldscore_camera_control_score(rotation_error_mean, translation_error_mean):
    # Mirrors WorldScore camera_error normalization: empirical max is 15 deg and 0.5 translation units.
    rotation_component = 1.0 - min(max(rotation_error_mean, 0.0), 15.0) / 15.0
    translation_component = 1.0 - min(max(translation_error_mean, 0.0), 0.5) / 0.5
    return 100.0 * math.sqrt(max(rotation_component, 0.0) * max(translation_component, 0.0))


def score_reconstruction(reconstruction_dir, metadata, dataset_root):
    pred = load_predicted_poses(reconstruction_dir)
    gt = load_gt_poses(metadata, dataset_root)

    count = min(len(pred), len(gt))
    pred = pred[:count]
    gt = gt[:count]
    if count < 2:
        raise RuntimeError(f"Need at least two poses to score {reconstruction_dir}")

    pred_rel = relative_poses(pred)
    gt_rel = relative_poses(gt)

    rot_errors = rotation_error_deg(pred_rel[:, :3, :3], gt_rel[:, :3, :3])

    pred_points = pred_rel[:, :3, 3]
    gt_points = gt_rel[:, :3, 3]

    scalar = fit_scalar(pred_points, gt_points)
    pred_scaled = scalar * pred_points
    scale_only_errors = np.linalg.norm(pred_scaled - gt_points, axis=1)

    pred_sim3, sim3_scale = fit_umeyama(pred_points, gt_points)
    sim3_errors = np.linalg.norm(pred_sim3 - gt_points, axis=1)

    rotation_mean = float(rot_errors.mean())
    translation_mean = float(scale_only_errors.mean())

    return {
        "run_name": metadata["run_name"],
        "row": metadata["manifest_row"],
        "scene": metadata["scene"],
        "start_frame": metadata["start_frame"],
        "duration_sec": metadata["duration_sec"],
        "sampled_frames": count,
        "frame_stride": metadata.get("frame_stride"),
        "total_video_frames": metadata.get("total_video_frames"),
        "rotation_error_deg_mean": safe_round(rotation_mean),
        "rotation_error_deg_median": safe_round(percentile(rot_errors, 0.5)),
        "rotation_error_deg_p90": safe_round(percentile(rot_errors, 0.9)),
        "rotation_error_deg_max": safe_round(float(rot_errors.max())),
        "translation_error_scale_only_mean": safe_round(translation_mean),
        "translation_error_scale_only_median": safe_round(percentile(scale_only_errors, 0.5)),
        "translation_error_scale_only_p90": safe_round(percentile(scale_only_errors, 0.9)),
        "translation_error_scale_only_max": safe_round(float(scale_only_errors.max())),
        "translation_error_sim3_mean": safe_round(float(sim3_errors.mean())),
        "translation_error_sim3_median": safe_round(percentile(sim3_errors, 0.5)),
        "translation_error_sim3_p90": safe_round(percentile(sim3_errors, 0.9)),
        "translation_error_sim3_max": safe_round(float(sim3_errors.max())),
        "endpoint_rotation_error_deg": safe_round(float(rot_errors[-1])),
        "endpoint_translation_error_scale_only": safe_round(float(scale_only_errors[-1])),
        "endpoint_translation_error_sim3": safe_round(float(sim3_errors[-1])),
        "gt_path_length": safe_round(path_length(gt_points)),
        "pred_path_length_scale_only": safe_round(path_length(pred_scaled)),
        "pred_path_length_sim3": safe_round(path_length(pred_sim3)),
        "path_length_ratio_scale_only": safe_round(
            path_length(pred_scaled) / path_length(gt_points)
            if path_length(gt_points) > 1e-12
            else None
        ),
        "loop_gt_endpoint_distance": safe_round(float(np.linalg.norm(gt_points[-1] - gt_points[0]))),
        "loop_pred_endpoint_distance_scale_only": safe_round(
            float(np.linalg.norm(pred_scaled[-1] - pred_scaled[0]))
        ),
        "loop_endpoint_distance_error": safe_round(
            abs(
                float(np.linalg.norm(pred_scaled[-1] - pred_scaled[0]))
                - float(np.linalg.norm(gt_points[-1] - gt_points[0]))
            )
        ),
        "scale_only_factor": safe_round(scalar),
        "sim3_scale": safe_round(sim3_scale),
        "worldscore_camera_control_score": safe_round(
            worldscore_camera_control_score(rotation_mean, translation_mean)
        ),
        "reconstruction_dir": str(reconstruction_dir),
    }


def discover_metadata_files(cut3r_dir, runs=None):
    metadata_files = sorted(cut3r_dir.glob("*/*/metadata.json"))
    if runs is None:
        return metadata_files
    run_set = set(runs)
    return [path for path in metadata_files if path.parent.parent.name in run_set]


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def summarize_by_run(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)

    output = []
    metric_fields = [
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
    ]
    for run_name, group in sorted(grouped.items()):
        row = {
            "run_name": run_name,
            "videos": len(group),
            "sampled_frames_mean": safe_round(mean(item["sampled_frames"] for item in group)),
        }
        for field in metric_fields:
            row.update(numeric_stats(field, [item.get(field) for item in group]))
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Score CUT3R camera trajectories against Context-as-Memory manifest poses."
    )
    parser.add_argument("--cut3r_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--durations", type=str, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    args = parser.parse_args()

    run_filter = parse_list(args.runs)
    row_filter = parse_rows(args.rows)
    duration_filter = set(parse_int_list(args.durations) or [])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for metadata_path in discover_metadata_files(args.cut3r_dir, runs=run_filter):
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        if row_filter is not None and int(metadata["manifest_row"]) not in row_filter:
            continue
        if duration_filter and int(metadata["duration_sec"]) not in duration_filter:
            continue
        try:
            rows.append(
                score_reconstruction(
                    reconstruction_dir=metadata_path.parent,
                    metadata=metadata,
                    dataset_root=args.dataset_root,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "metadata_path": str(metadata_path),
                    "run_name": metadata.get("run_name"),
                    "row": metadata.get("manifest_row"),
                    "error": repr(exc),
                }
            )

    if not rows:
        raise RuntimeError(f"No CUT3R reconstructions scored under {args.cut3r_dir}")

    summary = summarize_by_run(rows)

    write_csv(args.output_dir / "cut3r_camera_metrics.csv", rows)
    write_csv(args.output_dir / "cut3r_camera_summary.csv", summary)
    write_json(args.output_dir / "cut3r_camera_metrics.json", rows)
    write_json(args.output_dir / "cut3r_camera_summary.json", summary)
    write_json(args.output_dir / "cut3r_camera_failures.json", failures)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.output_dir / 'cut3r_camera_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'cut3r_camera_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'cut3r_camera_failures.json'}")


if __name__ == "__main__":
    main()
