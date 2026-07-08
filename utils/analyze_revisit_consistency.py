import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.poses import load_c2ws_from_json


def load_manifest(manifest_path):
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_row"] = row_index
            rows.append(item)
    return rows


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_list(value):
    if not value:
        return []
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


def output_path(model_root, run_name, item):
    return model_root / run_name / f"{item['output_prefix']}custom.mp4"


def rotation_distance_deg(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def forward_vectors(c2ws, axis):
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "neg_x": np.array([-1.0, 0.0, 0.0]),
        "neg_y": np.array([0.0, -1.0, 0.0]),
        "neg_z": np.array([0.0, 0.0, -1.0]),
    }
    local = axis_map[axis]
    vectors = c2ws[:, :3, :3] @ local
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def closest_points_between_rays(origin_a, direction_a, origin_b, direction_b):
    # Solve closest points on two half-lines: origin + t * direction, t >= 0.
    w0 = origin_a - origin_b
    a = float(np.dot(direction_a, direction_a))
    b = float(np.dot(direction_a, direction_b))
    c = float(np.dot(direction_b, direction_b))
    d = float(np.dot(direction_a, w0))
    e = float(np.dot(direction_b, w0))
    denom = a * c - b * b

    if abs(denom) <= 1e-12:
        t_a = 0.0
        t_b = max(0.0, e / c) if c > 1e-12 else 0.0
    else:
        t_a = (b * e - c * d) / denom
        t_b = (a * e - b * d) / denom
        if t_a < 0.0:
            t_a = 0.0
            t_b = max(0.0, e / c) if c > 1e-12 else 0.0
        if t_b < 0.0:
            t_b = 0.0
            t_a = max(0.0, -d / a) if a > 1e-12 else 0.0

    point_a = origin_a + t_a * direction_a
    point_b = origin_b + t_b * direction_b
    midpoint = 0.5 * (point_a + point_b)
    distance = float(np.linalg.norm(point_a - point_b))
    return midpoint, distance, float(t_a), float(t_b)


def component_clusters(indices, edges):
    parent = {idx: idx for idx in indices}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in edges:
        union(a, b)

    groups = defaultdict(list)
    for idx in indices:
        groups[find(idx)].append(idx)
    return [sorted(group) for group in groups.values() if len(group) >= 2]


def find_exact_pose_clusters(
    c2ws,
    sample_indices,
    min_gap,
    position_threshold,
    rotation_threshold_deg,
):
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    events = []
    edges = []
    for outer, i in enumerate(sample_indices):
        for j in sample_indices[outer + 1 :]:
            if j - i < min_gap:
                continue
            position_distance = float(np.linalg.norm(positions[i] - positions[j]))
            rotation_deg = rotation_distance_deg(rotations[i], rotations[j])
            if position_distance <= position_threshold and rotation_deg <= rotation_threshold_deg:
                edges.append((i, j))
                events.append(
                    {
                        "revisit_type": "exact_pose",
                        "frame_i": i,
                        "frame_j": j,
                        "time_i_sec": None,
                        "time_j_sec": None,
                        "position_distance": position_distance,
                        "rotation_deg": rotation_deg,
                        "ray_distance": None,
                        "ray_angle_deg": None,
                        "depth_i": None,
                        "depth_j": None,
                        "point_x": None,
                        "point_y": None,
                        "point_z": None,
                    }
                )

    clusters = []
    for cluster_id, visits in enumerate(component_clusters(sample_indices, edges)):
        center = np.mean(positions[visits], axis=0)
        clusters.append(
            {
                "revisit_type": "exact_pose",
                "cluster_id": cluster_id,
                "visit_indices": visits,
                "visit_count": len(visits),
                "point": center,
                "support_pairs": sum(
                    1 for event in events if event["frame_i"] in visits and event["frame_j"] in visits
                ),
            }
        )
    return events, clusters


def find_gaze_point_clusters(
    c2ws,
    sample_indices,
    min_gap,
    forward_axis,
    ray_distance_threshold,
    point_cluster_radius,
    min_depth,
    max_depth,
    min_ray_angle_deg,
):
    positions = c2ws[:, :3, 3]
    directions = forward_vectors(c2ws, forward_axis)
    candidates = []
    events = []

    for outer, i in enumerate(sample_indices):
        for j in sample_indices[outer + 1 :]:
            if j - i < min_gap:
                continue
            direction_dot = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
            ray_angle_deg = math.degrees(math.acos(direction_dot))
            if ray_angle_deg < min_ray_angle_deg:
                continue

            point, ray_distance, depth_i, depth_j = closest_points_between_rays(
                positions[i],
                directions[i],
                positions[j],
                directions[j],
            )
            if ray_distance > ray_distance_threshold:
                continue
            if depth_i < min_depth or depth_j < min_depth:
                continue
            if max_depth is not None and (depth_i > max_depth or depth_j > max_depth):
                continue

            event = {
                "revisit_type": "gaze_point",
                "frame_i": i,
                "frame_j": j,
                "time_i_sec": None,
                "time_j_sec": None,
                "position_distance": float(np.linalg.norm(positions[i] - positions[j])),
                "rotation_deg": rotation_distance_deg(c2ws[i, :3, :3], c2ws[j, :3, :3]),
                "ray_distance": ray_distance,
                "ray_angle_deg": ray_angle_deg,
                "depth_i": depth_i,
                "depth_j": depth_j,
                "point_x": float(point[0]),
                "point_y": float(point[1]),
                "point_z": float(point[2]),
            }
            events.append(event)
            candidates.append({"point": point, "frames": {i, j}, "events": [event]})

    clusters = []
    for candidate in candidates:
        assigned = False
        for cluster in clusters:
            if np.linalg.norm(candidate["point"] - cluster["point"]) <= point_cluster_radius:
                cluster["points"].append(candidate["point"])
                cluster["frames"].update(candidate["frames"])
                cluster["events"].extend(candidate["events"])
                cluster["point"] = np.mean(cluster["points"], axis=0)
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "point": candidate["point"].copy(),
                    "points": [candidate["point"]],
                    "frames": set(candidate["frames"]),
                    "events": list(candidate["events"]),
                }
            )

    output = []
    for cluster_id, cluster in enumerate(clusters):
        visits = sorted(cluster["frames"])
        if len(visits) < 2:
            continue
        output.append(
            {
                "revisit_type": "gaze_point",
                "cluster_id": cluster_id,
                "visit_indices": visits,
                "visit_count": len(visits),
                "point": cluster["point"],
                "support_pairs": len(cluster["events"]),
            }
        )
    return events, output


def read_gt_frames(item, dataset_root, local_indices):
    frames_dir = resolve_gt_frames_dir(item, dataset_root)
    frames = {}
    for local_index in local_indices:
        frame_index = int(item["start_frame"]) + int(local_index)
        frame_path = frames_dir / f"{frame_index:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing GT frame: {frame_path}")
        with Image.open(frame_path) as image:
            frames[local_index] = image.convert("RGB")
    return frames


def read_video_frames(video_path, local_indices):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to read generated MP4 files.") from exc

    wanted = sorted(set(int(index) for index in local_indices))
    if not wanted:
        return {}
    wanted_set = set(wanted)
    frames = {}
    reader = imageio.get_reader(str(video_path))
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index > wanted[-1]:
                break
            if frame_index in wanted_set:
                frames[frame_index] = Image.fromarray(np.asarray(frame)[..., :3]).convert("RGB")
    finally:
        reader.close()
    return frames


def center_patch(image, patch_size):
    width, height = image.size
    size = min(int(patch_size), width, height)
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def patch_metrics(patches):
    indices = sorted(patches)
    pair_rmses = []
    pair_maes = []
    for outer, i in enumerate(indices):
        arr_i = np.asarray(patches[i], dtype=np.float64)
        for j in indices[outer + 1 :]:
            arr_j = np.asarray(patches[j], dtype=np.float64)
            diff = arr_i - arr_j
            pair_rmses.append(math.sqrt(float(np.mean(diff * diff))))
            pair_maes.append(float(np.mean(np.abs(diff))))
    if not pair_rmses:
        return None
    return {
        "pair_count": len(pair_rmses),
        "mean_patch_rmse": float(np.mean(pair_rmses)),
        "worst_patch_rmse": float(np.max(pair_rmses)),
        "p90_patch_rmse": float(np.percentile(pair_rmses, 90)),
        "mean_patch_mae": float(np.mean(pair_maes)),
        "worst_patch_mae": float(np.max(pair_maes)),
        "worst_patch_rmse_0_1": float(np.max(pair_rmses) / 255.0),
    }


def patch_delta_metrics(source_patches, oracle_patches):
    indices = sorted(set(source_patches) & set(oracle_patches))
    pair_rmses = []
    pair_maes = []
    for outer, i in enumerate(indices):
        source_i = np.asarray(source_patches[i], dtype=np.float64)
        oracle_i = np.asarray(oracle_patches[i], dtype=np.float64)
        for j in indices[outer + 1 :]:
            source_j = np.asarray(source_patches[j], dtype=np.float64)
            oracle_j = np.asarray(oracle_patches[j], dtype=np.float64)
            source_delta = source_i - source_j
            oracle_delta = oracle_i - oracle_j
            diff = source_delta - oracle_delta
            pair_rmses.append(math.sqrt(float(np.mean(diff * diff))))
            pair_maes.append(float(np.mean(np.abs(diff))))
    if not pair_rmses:
        return None
    return {
        "pair_count": len(pair_rmses),
        "mean_delta_rmse": float(np.mean(pair_rmses)),
        "worst_delta_rmse": float(np.max(pair_rmses)),
        "p90_delta_rmse": float(np.percentile(pair_rmses, 90)),
        "mean_delta_mae": float(np.mean(pair_maes)),
        "worst_delta_mae": float(np.max(pair_maes)),
        "worst_delta_rmse_0_1": float(np.max(pair_rmses) / 255.0),
    }


def score_clusters_for_source(item, clusters, source_name, frames, patch_size):
    rows = []
    for cluster in clusters:
        patches = {}
        missing = []
        for local_index in cluster["visit_indices"]:
            image = frames.get(local_index)
            if image is None:
                missing.append(local_index)
                continue
            patches[local_index] = center_patch(image, patch_size)

        metrics = patch_metrics(patches)
        if metrics is None:
            continue

        point = cluster["point"]
        rows.append(
            {
                "source": source_name,
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "revisit_type": cluster["revisit_type"],
                "cluster_id": cluster["cluster_id"],
                "visit_count": cluster["visit_count"],
                "support_pairs": cluster["support_pairs"],
                "scored_visits": len(patches),
                "missing_visits": len(missing),
                "visit_indices": ",".join(str(index) for index in cluster["visit_indices"]),
                "point_x": None if point is None else float(point[0]),
                "point_y": None if point is None else float(point[1]),
                "point_z": None if point is None else float(point[2]),
                **metrics,
            }
        )
    return rows


def cluster_row_key(row):
    return (str(row["revisit_type"]), int(row["cluster_id"]))


def cluster_key(cluster):
    return (str(cluster["revisit_type"]), int(cluster["cluster_id"]))


def score_delta_clusters_against_oracle(
    item,
    clusters,
    source_name,
    source_frames,
    oracle_frames,
    patch_size,
):
    rows = []
    for cluster in clusters:
        source_patches = {}
        oracle_patches = {}
        for local_index in cluster["visit_indices"]:
            source_image = source_frames.get(local_index)
            oracle_image = oracle_frames.get(local_index)
            if source_image is None or oracle_image is None:
                continue
            source_patches[local_index] = center_patch(source_image, patch_size)
            oracle_patches[local_index] = center_patch(oracle_image, patch_size)

        metrics = patch_delta_metrics(source_patches, oracle_patches)
        if metrics is None:
            continue

        point = cluster["point"]
        rows.append(
            {
                "source": source_name,
                "oracle_source": "gt_oracle",
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "revisit_type": cluster["revisit_type"],
                "cluster_id": cluster["cluster_id"],
                "visit_count": cluster["visit_count"],
                "support_pairs": cluster["support_pairs"],
                "scored_visits": len(source_patches),
                "visit_indices": ",".join(str(index) for index in cluster["visit_indices"]),
                "point_x": None if point is None else float(point[0]),
                "point_y": None if point is None else float(point[1]),
                "point_z": None if point is None else float(point[2]),
                **metrics,
            }
        )
    return rows


def make_montage(row, frames, patch_size, output_path):
    indices = [int(part) for part in row["visit_indices"].split(",") if part]
    patches = []
    labels = []
    for index in indices[:8]:
        image = frames.get(index)
        if image is None:
            continue
        patch = center_patch(image, patch_size)
        patches.append(patch)
        labels.append(str(index))
    if len(patches) < 2:
        return

    label_height = 18
    width = sum(patch.size[0] for patch in patches)
    height = max(patch.size[1] for patch in patches) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for label, patch in zip(labels, patches):
        canvas.paste(patch, (x_offset, label_height))
        draw.text((x_offset + 4, 2), label, fill=(0, 0, 0))
        x_offset += patch.size[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_trajectory(item, c2ws, sample_indices, exact_clusters, gaze_clusters, output_path, forward_axis):
    positions = c2ws[:, :3, 3]
    directions = forward_vectors(c2ws, forward_axis)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(positions[:, 0], positions[:, 1], color="#333333", linewidth=1.6, label="camera path")
    ax.scatter(positions[0, 0], positions[0, 1], color="#2ca02c", s=55, label="start", zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], color="#d62728", s=55, label="end", zorder=5)

    stride_for_arrows = max(1, len(sample_indices) // 24)
    arrow_indices = sample_indices[::stride_for_arrows]
    arrow_scale = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1]), 1.0) * 0.035
    ax.quiver(
        positions[arrow_indices, 0],
        positions[arrow_indices, 1],
        directions[arrow_indices, 0] * arrow_scale,
        directions[arrow_indices, 1] * arrow_scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color="#777777",
        alpha=0.7,
    )

    for cluster in exact_clusters:
        visits = cluster["visit_indices"]
        ax.scatter(
            positions[visits, 0],
            positions[visits, 1],
            s=34,
            color="#1f77b4",
            alpha=0.85,
            label="exact-pose revisit" if "exact-pose revisit" not in ax.get_legend_handles_labels()[1] else None,
        )

    if gaze_clusters:
        points = np.stack([cluster["point"] for cluster in gaze_clusters])
        sizes = [22 + 10 * cluster["visit_count"] for cluster in gaze_clusters]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=sizes,
            color="#ff7f0e",
            alpha=0.78,
            marker="x",
            label="gaze-point cluster",
        )

    ax.set_title(
        f"row {item['_row']} | {item['scene']} | {item['duration_sec']}s\n"
        f"exact clusters={len(exact_clusters)}, gaze clusters={len(gaze_clusters)}"
    )
    ax.set_xlabel("world X")
    ax.set_ylabel("world Y")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(score_rows):
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[(row["source"], row["revisit_type"])].append(row)

    summaries = []
    for (source, revisit_type), group in sorted(grouped.items()):
        worst = [float(row["worst_patch_rmse"]) for row in group]
        mean = [float(row["mean_patch_rmse"]) for row in group]
        visits = [int(row["visit_count"]) for row in group]
        summaries.append(
            {
                "source": source,
                "revisit_type": revisit_type,
                "clusters": len(group),
                "videos": len({row["row"] for row in group}),
                "mean_visit_count": float(np.mean(visits)) if visits else None,
                "mean_worst_patch_rmse": float(np.mean(worst)) if worst else None,
                "median_worst_patch_rmse": float(np.percentile(worst, 50)) if worst else None,
                "p90_worst_patch_rmse": float(np.percentile(worst, 90)) if worst else None,
                "mean_patch_rmse": float(np.mean(mean)) if mean else None,
                "mean_worst_patch_rmse_0_1": float(np.mean(worst) / 255.0) if worst else None,
            }
        )
    return summaries


def summarize_delta_scores(score_rows):
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[(row["source"], row["revisit_type"])].append(row)

    summaries = []
    for (source, revisit_type), group in sorted(grouped.items()):
        worst = [float(row["worst_delta_rmse"]) for row in group]
        mean = [float(row["mean_delta_rmse"]) for row in group]
        visits = [int(row["visit_count"]) for row in group]
        summaries.append(
            {
                "source": source,
                "oracle_source": "gt_oracle",
                "revisit_type": revisit_type,
                "clusters": len(group),
                "videos": len({row["row"] for row in group}),
                "mean_visit_count": float(np.mean(visits)) if visits else None,
                "mean_worst_delta_rmse": float(np.mean(worst)) if worst else None,
                "median_worst_delta_rmse": float(np.percentile(worst, 50)) if worst else None,
                "p90_worst_delta_rmse": float(np.percentile(worst, 90)) if worst else None,
                "mean_delta_rmse": float(np.mean(mean)) if mean else None,
                "mean_worst_delta_rmse_0_1": float(np.mean(worst) / 255.0) if worst else None,
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find camera revisit events from Context-as-Memory poses and score pixel-space "
            "revisit consistency on GT frames and optional generated runs."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--model_root", type=Path, default=None)
    parser.add_argument("--runs", type=str, default="")
    parser.add_argument("--durations", type=str, default="180")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_stride", type=int, default=30)
    parser.add_argument("--min_revisit_gap", type=int, default=150)
    parser.add_argument("--exact_position_threshold", type=float, default=0.25)
    parser.add_argument("--exact_rotation_deg", type=float, default=5.0)
    parser.add_argument(
        "--forward_axis",
        choices=["x", "y", "z", "neg_x", "neg_y", "neg_z"],
        default="x",
    )
    parser.add_argument("--ray_distance_threshold", type=float, default=0.75)
    parser.add_argument("--point_cluster_radius", type=float, default=1.5)
    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--min_ray_angle_deg", type=float, default=5.0)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument(
        "--revisit_types",
        type=str,
        default="exact_pose,gaze_point",
        help="Comma list from exact_pose,gaze_point. Use exact_pose for the strict oracle-calibrated metric.",
    )
    parser.add_argument(
        "--max_gt_worst_patch_rmse",
        type=float,
        default=None,
        help=(
            "If set, only score revisit clusters whose GT-oracle self-consistency "
            "worst patch RMSE is at or below this threshold."
        ),
    )
    parser.add_argument("--montages_per_source", type=int, default=8)
    args = parser.parse_args()

    if args.sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1")
    if args.min_revisit_gap < 0:
        raise ValueError("--min_revisit_gap must be >= 0")
    if args.runs and args.model_root is None:
        raise ValueError("--model_root is required when --runs is provided")
    revisit_type_filter = set(parse_list(args.revisit_types))
    allowed_revisit_types = {"exact_pose", "gaze_point"}
    unknown_revisit_types = sorted(revisit_type_filter - allowed_revisit_types)
    if unknown_revisit_types:
        raise ValueError(
            f"Unknown --revisit_types {unknown_revisit_types}; expected {sorted(allowed_revisit_types)}"
        )
    if not revisit_type_filter:
        raise ValueError("--revisit_types selected nothing.")

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    montages_dir = args.output_dir / "montages"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    items = select_items(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    runs = parse_list(args.runs)
    all_events = []
    all_clusters = []
    all_scores = []
    all_delta_scores = []
    video_summaries = []
    filter_summaries = []
    frames_for_montage = {}

    for item in items:
        c2ws = load_c2ws_from_json(
            json_path=resolve_pose_path(item, args.dataset_root),
            start_frame=int(item["start_frame"]),
            num_frames=int(item["num_frames"]),
        )
        sample_indices = list(range(0, len(c2ws), args.sample_stride))
        if sample_indices[-1] != len(c2ws) - 1:
            sample_indices.append(len(c2ws) - 1)

        exact_events, exact_clusters = find_exact_pose_clusters(
            c2ws=c2ws,
            sample_indices=sample_indices,
            min_gap=args.min_revisit_gap,
            position_threshold=args.exact_position_threshold,
            rotation_threshold_deg=args.exact_rotation_deg,
        )
        gaze_events, gaze_clusters = find_gaze_point_clusters(
            c2ws=c2ws,
            sample_indices=sample_indices,
            min_gap=args.min_revisit_gap,
            forward_axis=args.forward_axis,
            ray_distance_threshold=args.ray_distance_threshold,
            point_cluster_radius=args.point_cluster_radius,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_ray_angle_deg=args.min_ray_angle_deg,
        )
        clusters = exact_clusters + gaze_clusters

        exact_events = [event for event in exact_events if event["revisit_type"] in revisit_type_filter]
        gaze_events = [event for event in gaze_events if event["revisit_type"] in revisit_type_filter]
        exact_clusters = [
            cluster for cluster in exact_clusters if cluster["revisit_type"] in revisit_type_filter
        ]
        gaze_clusters = [
            cluster for cluster in gaze_clusters if cluster["revisit_type"] in revisit_type_filter
        ]
        clusters = [cluster for cluster in clusters if cluster["revisit_type"] in revisit_type_filter]

        for event in exact_events + gaze_events:
            event["row"] = item["_row"]
            event["scene"] = item["scene"]
            event["start_frame"] = item["start_frame"]
            event["duration_sec"] = item["duration_sec"]
            event["time_i_sec"] = event["frame_i"] / float(item["fps"])
            event["time_j_sec"] = event["frame_j"] / float(item["fps"])
            all_events.append(event)

        for cluster in clusters:
            point = cluster["point"]
            all_clusters.append(
                {
                    "row": item["_row"],
                    "scene": item["scene"],
                    "start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "revisit_type": cluster["revisit_type"],
                    "cluster_id": cluster["cluster_id"],
                    "visit_count": cluster["visit_count"],
                    "support_pairs": cluster["support_pairs"],
                    "visit_indices": ",".join(str(index) for index in cluster["visit_indices"]),
                    "visit_times_sec": ",".join(
                        f"{index / float(item['fps']):.3f}" for index in cluster["visit_indices"]
                    ),
                    "point_x": None if point is None else float(point[0]),
                    "point_y": None if point is None else float(point[1]),
                    "point_z": None if point is None else float(point[2]),
                }
            )

        video_summaries.append(
            {
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "num_frames": item["num_frames"],
                "sampled_camera_frames": len(sample_indices),
                "exact_pose_pairs": len(exact_events),
                "exact_pose_clusters": len(exact_clusters),
                "max_exact_pose_visits": max([c["visit_count"] for c in exact_clusters] or [0]),
                "gaze_point_pairs": len(gaze_events),
                "gaze_point_clusters": len(gaze_clusters),
                "max_gaze_point_visits": max([c["visit_count"] for c in gaze_clusters] or [0]),
            }
        )

        plot_trajectory(
            item=item,
            c2ws=c2ws,
            sample_indices=sample_indices,
            exact_clusters=exact_clusters,
            gaze_clusters=gaze_clusters,
            output_path=figures_dir / f"trajectory_row_{item['_row']:03d}_{item['scene']}.png",
            forward_axis=args.forward_axis,
        )

        cluster_indices = sorted({index for cluster in clusters for index in cluster["visit_indices"]})
        if not cluster_indices:
            continue

        gt_frames = read_gt_frames(item, args.dataset_root, cluster_indices)
        gt_rows = score_clusters_for_source(
            item=item,
            clusters=clusters,
            source_name="gt_oracle",
            frames=gt_frames,
            patch_size=args.patch_size,
        )
        trusted_keys = {cluster_row_key(row) for row in gt_rows}
        if args.max_gt_worst_patch_rmse is not None:
            trusted_keys = {
                cluster_row_key(row)
                for row in gt_rows
                if float(row["worst_patch_rmse"]) <= args.max_gt_worst_patch_rmse
            }
        trusted_clusters = [cluster for cluster in clusters if cluster_key(cluster) in trusted_keys]
        trusted_gt_rows = [row for row in gt_rows if cluster_row_key(row) in trusted_keys]

        for revisit_type in sorted(revisit_type_filter):
            before = [cluster for cluster in clusters if cluster["revisit_type"] == revisit_type]
            after = [cluster for cluster in trusted_clusters if cluster["revisit_type"] == revisit_type]
            before_gt = [row for row in gt_rows if row["revisit_type"] == revisit_type]
            filter_summaries.append(
                {
                    "row": item["_row"],
                    "scene": item["scene"],
                    "start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "revisit_type": revisit_type,
                    "clusters_before_gt_filter": len(before),
                    "clusters_after_gt_filter": len(after),
                    "max_gt_worst_patch_rmse": args.max_gt_worst_patch_rmse,
                    "mean_gt_worst_patch_rmse_before": (
                        float(np.mean([float(row["worst_patch_rmse"]) for row in before_gt]))
                        if before_gt
                        else None
                    ),
                    "mean_gt_worst_patch_rmse_after": (
                        float(np.mean([float(row["worst_patch_rmse"]) for row in trusted_gt_rows if row["revisit_type"] == revisit_type]))
                        if after
                        else None
                    ),
                }
            )

        all_scores.extend(trusted_gt_rows)
        frames_for_montage.update({("gt_oracle", item["_row"]): gt_frames})
        if not trusted_clusters:
            continue
        trusted_cluster_indices = sorted(
            {index for cluster in trusted_clusters for index in cluster["visit_indices"]}
        )

        for run_name in runs:
            video_path = output_path(args.model_root, run_name, item)
            if not video_path.exists():
                continue
            gen_frames = read_video_frames(video_path, trusted_cluster_indices)
            run_rows = score_clusters_for_source(
                item=item,
                clusters=trusted_clusters,
                source_name=run_name,
                frames=gen_frames,
                patch_size=args.patch_size,
            )
            all_scores.extend(run_rows)
            all_delta_scores.extend(
                score_delta_clusters_against_oracle(
                    item=item,
                    clusters=trusted_clusters,
                    source_name=run_name,
                    source_frames=gen_frames,
                    oracle_frames=gt_frames,
                    patch_size=args.patch_size,
                )
            )
            frames_for_montage[(run_name, item["_row"])] = gen_frames

    write_csv(tables_dir / "revisit_events.csv", all_events)
    write_csv(tables_dir / "revisit_clusters.csv", all_clusters)
    write_csv(tables_dir / "revisit_gt_filter_summary.csv", filter_summaries)
    write_csv(tables_dir / "trajectory_revisit_summary.csv", video_summaries)
    write_csv(tables_dir / "revisit_scores.csv", all_scores)
    summary_rows = summarize_scores(all_scores)
    write_csv(tables_dir / "revisit_summary.csv", summary_rows)
    write_csv(tables_dir / "revisit_delta_scores_vs_gt.csv", all_delta_scores)
    delta_summary_rows = summarize_delta_scores(all_delta_scores)
    write_csv(tables_dir / "revisit_delta_summary_vs_gt.csv", delta_summary_rows)

    with (tables_dir / "revisit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
        handle.write("\n")
    with (tables_dir / "revisit_delta_summary_vs_gt.json").open("w", encoding="utf-8") as handle:
        json.dump(delta_summary_rows, handle, indent=2)
        handle.write("\n")

    for source in ["gt_oracle", *runs]:
        source_rows = [row for row in all_scores if row["source"] == source]
        source_rows.sort(key=lambda row: float(row["worst_patch_rmse"]), reverse=True)
        for montage_index, row in enumerate(source_rows[: args.montages_per_source]):
            frames = frames_for_montage.get((source, int(row["row"])), {})
            make_montage(
                row=row,
                frames=frames,
                patch_size=args.patch_size,
                output_path=montages_dir
                / source
                / f"{montage_index:02d}_row_{int(row['row']):03d}_{row['revisit_type']}_cluster_{int(row['cluster_id']):03d}.png",
            )

    report_path = args.output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Revisit Consistency Analysis\n\n")
        handle.write(
            "Lower pixel RMSE means revisited views stayed more visually consistent. "
            "`gt_oracle` scores the same revisit events on ground-truth frames. "
            "The delta-vs-GT table is the safer oracle-aware score: it measures "
            "whether pairwise revisit changes match the ground-truth pairwise changes.\n\n"
        )
        handle.write("## Self-Consistency Summary\n\n")
        if summary_rows:
            headers = list(summary_rows[0].keys())
            handle.write("| " + " | ".join(headers) + " |\n")
            handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in summary_rows:
                handle.write("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |\n")
        else:
            handle.write("No revisit clusters were scored with the current thresholds.\n")
        handle.write("\n## Delta-vs-GT Summary\n\n")
        if delta_summary_rows:
            headers = list(delta_summary_rows[0].keys())
            handle.write("| " + " | ".join(headers) + " |\n")
            handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in delta_summary_rows:
                handle.write("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |\n")
        else:
            handle.write("No generated revisit clusters were scored against GT.\n")
        handle.write("\n## Files\n\n")
        for path in [
            tables_dir / "trajectory_revisit_summary.csv",
            tables_dir / "revisit_events.csv",
            tables_dir / "revisit_clusters.csv",
            tables_dir / "revisit_gt_filter_summary.csv",
            tables_dir / "revisit_scores.csv",
            tables_dir / "revisit_summary.csv",
            tables_dir / "revisit_delta_scores_vs_gt.csv",
            tables_dir / "revisit_delta_summary_vs_gt.csv",
        ]:
            handle.write(f"- `{path.relative_to(args.output_dir)}`\n")
        handle.write("- `figures/trajectory_row_*.png`\n")
        handle.write("- `montages/*/*.png`\n")

    print(f"Wrote: {tables_dir / 'trajectory_revisit_summary.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_events.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_clusters.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_gt_filter_summary.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_scores.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_summary.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_delta_scores_vs_gt.csv'}")
    print(f"Wrote: {tables_dir / 'revisit_delta_summary_vs_gt.csv'}")
    print(f"Wrote: {report_path}")
    print(f"Wrote trajectory figures under: {figures_dir}")
    print(f"Wrote worst-cluster montages under: {montages_dir}")


if __name__ == "__main__":
    main()
