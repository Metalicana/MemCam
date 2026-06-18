import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

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


def select_rows(items, row_filter, durations, limit):
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


def rotation_distance_rad(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.acos(cosine)


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def direction_entropy(step_vectors, bins=8):
    xy = step_vectors[:, :2]
    norms = np.linalg.norm(xy, axis=1)
    xy = xy[norms > 1e-8]
    if len(xy) == 0:
        return 0.0, 0

    angles = np.arctan2(xy[:, 1], xy[:, 0])
    hist, _ = np.histogram(angles, bins=bins, range=(-math.pi, math.pi))
    probs = hist[hist > 0] / np.sum(hist)
    entropy = -float(np.sum(probs * np.log2(probs)))
    return entropy / math.log2(bins), int(np.count_nonzero(hist))


def curvature_degrees(step_vectors):
    angles = []
    for idx in range(len(step_vectors) - 1):
        a = step_vectors[idx]
        b = step_vectors[idx + 1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a <= 1e-8 or norm_b <= 1e-8:
            continue
        cosine = np.dot(a, b) / (norm_a * norm_b)
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def revisit_stats(
    c2ws,
    pair_stride,
    min_revisit_gap,
    revisit_position_threshold,
    revisit_rotation_deg,
):
    indices = list(range(0, len(c2ws), pair_stride))
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    total_pairs = 0
    revisit_pairs = 0
    min_distance = None
    min_rotation_deg = None

    for i_pos, i in enumerate(indices):
        for j in indices[i_pos + 1:]:
            if j - i < min_revisit_gap:
                continue
            total_pairs += 1
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            rotation_deg = math.degrees(rotation_distance_rad(rotations[i], rotations[j]))
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_rotation_deg = rotation_deg
            if distance <= revisit_position_threshold and rotation_deg <= revisit_rotation_deg:
                revisit_pairs += 1

    return {
        "revisit_pairs": revisit_pairs,
        "revisit_candidate_pairs": total_pairs,
        "revisit_pair_rate": revisit_pairs / total_pairs if total_pairs else 0.0,
        "min_revisit_position_distance": min_distance if min_distance is not None else 0.0,
        "rotation_at_min_revisit_distance_deg": min_rotation_deg if min_rotation_deg is not None else 0.0,
    }


def analyze_trajectory(item, pair_stride, min_revisit_gap, revisit_position_threshold, revisit_rotation_deg):
    c2ws = load_c2ws_from_json(
        json_path=item["pose_path"],
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    step_vectors = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(step_vectors, axis=1)
    path_length = float(np.sum(step_distances))
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    displacement_ratio = displacement / path_length if path_length > 1e-8 else 0.0
    tortuosity = path_length / displacement if displacement > 1e-8 else math.inf

    rotation_steps = [
        math.degrees(rotation_distance_rad(rotations[idx], rotations[idx + 1]))
        for idx in range(len(rotations) - 1)
    ]
    total_rotation_deg = float(sum(rotation_steps))
    net_rotation_deg = math.degrees(rotation_distance_rad(rotations[0], rotations[-1]))
    curvature = curvature_degrees(step_vectors)
    entropy, occupied_bins = direction_entropy(step_vectors)
    revisit = revisit_stats(
        c2ws=c2ws,
        pair_stride=pair_stride,
        min_revisit_gap=min_revisit_gap,
        revisit_position_threshold=revisit_position_threshold,
        revisit_rotation_deg=revisit_rotation_deg,
    )

    return {
        "row": item["_row"],
        "scene": item["scene"],
        "start_frame": item["start_frame"],
        "duration_sec": item["duration_sec"],
        "num_frames": item["num_frames"],
        "path_length": path_length,
        "displacement": displacement,
        "displacement_ratio": displacement_ratio,
        "tortuosity": tortuosity,
        "mean_step_distance": safe_mean(step_distances),
        "std_step_distance": safe_std(step_distances),
        "p95_step_distance": percentile(step_distances, 95),
        "total_rotation_deg": total_rotation_deg,
        "net_rotation_deg": net_rotation_deg,
        "mean_rotation_step_deg": safe_mean(rotation_steps),
        "p95_rotation_step_deg": percentile(rotation_steps, 95),
        "mean_curvature_deg": safe_mean(curvature),
        "p90_curvature_deg": percentile(curvature, 90),
        "motion_direction_entropy": entropy,
        "motion_direction_bins": occupied_bins,
        "loop_like_score": (1.0 - displacement_ratio) * min(total_rotation_deg / 360.0, 1.0),
        **revisit,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"row", "start_frame", "duration_sec", "num_frames"}
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["duration_sec"], []).append(row)

    output = []
    for duration, group in sorted(grouped.items()):
        summary = {"duration_sec": duration, "videos": len(group)}
        for key in numeric_keys:
            values = [row[key] for row in group if not (isinstance(row[key], float) and math.isinf(row[key]))]
            summary[f"mean_{key}"] = safe_mean(values)
            summary[f"median_{key}"] = percentile(values, 50)
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory diversity for manifest rows.")
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("/data/ab575577/MemCam/analysis/context_memory"))
    parser.add_argument("--durations", type=str, default="10,20,40,60,120")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pair_stride", type=int, default=5)
    parser.add_argument("--min_revisit_gap", type=int, default=76)
    parser.add_argument("--revisit_position_threshold", type=float, default=2.0)
    parser.add_argument("--revisit_rotation_deg", type=float, default=20.0)
    args = parser.parse_args()

    items = select_rows(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        print(
            f"[trajectory row {item['_row']}] {item['scene']} "
            f"start={item['start_frame']} duration={item['duration_sec']}s"
        )
        rows.append(
            analyze_trajectory(
                item=item,
                pair_stride=args.pair_stride,
                min_revisit_gap=args.min_revisit_gap,
                revisit_position_threshold=args.revisit_position_threshold,
                revisit_rotation_deg=args.revisit_rotation_deg,
            )
        )

    summary_rows = aggregate(rows)
    write_csv(args.output_dir / "trajectory_diversity.csv", rows)
    write_csv(args.output_dir / "trajectory_diversity_summary.csv", summary_rows)
    with (args.output_dir / "trajectory_diversity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
        handle.write("\n")

    print(f"Wrote: {args.output_dir / 'trajectory_diversity.csv'}")
    print(f"Wrote: {args.output_dir / 'trajectory_diversity_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'trajectory_diversity_summary.json'}")


if __name__ == "__main__":
    main()
