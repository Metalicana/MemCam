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
from utils import evaluate_context_memory as evaluator


BASE_PAIR_METRICS = [
    "mae",
    "mse",
    "rmse",
    "psnr_db",
    "ssim",
]


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


def select_rows(items, row_filter, start_row, end_row, durations, limit):
    selected = []
    duration_filter = set(durations) if durations else None

    for item in items:
        row = item["_row"]
        if row_filter is not None and row not in row_filter:
            continue
        if start_row is not None and row < start_row:
            continue
        if end_row is not None and row > end_row:
            continue
        if duration_filter is not None and int(item["duration_sec"]) not in duration_filter:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def output_path(model_output_dir, item):
    return model_output_dir / f"{item['output_prefix']}custom.mp4"


def rotation_distance_rad(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.acos(cosine)


def discover_revisit_pairs(
    c2ws,
    pair_stride,
    min_revisit_gap,
    revisit_position_threshold,
    revisit_rotation_deg,
    max_pairs_per_video,
):
    indices = list(range(0, len(c2ws), pair_stride))
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    pairs = []

    for left_position, frame_i in enumerate(indices):
        for frame_j in indices[left_position + 1 :]:
            frame_gap = frame_j - frame_i
            if frame_gap < min_revisit_gap:
                continue
            pose_distance = float(np.linalg.norm(positions[frame_i] - positions[frame_j]))
            rotation_deg = math.degrees(
                rotation_distance_rad(rotations[frame_i], rotations[frame_j])
            )
            if pose_distance > revisit_position_threshold:
                continue
            if rotation_deg > revisit_rotation_deg:
                continue
            pairs.append(
                {
                    "frame_i": frame_i,
                    "frame_j": frame_j,
                    "frame_gap": frame_gap,
                    "pose_distance": pose_distance,
                    "rotation_deg": rotation_deg,
                }
            )

    pairs.sort(
        key=lambda pair: (
            -pair["frame_gap"],
            pair["pose_distance"],
            pair["rotation_deg"],
            pair["frame_i"],
        )
    )
    if max_pairs_per_video is not None:
        pairs = pairs[:max_pairs_per_video]
    pairs.sort(key=lambda pair: (pair["frame_i"], pair["frame_j"]))
    return pairs


def read_video_frames(video_path, indices):
    if not indices:
        return {}

    imageio = evaluator.get_imageio()
    reader = imageio.get_reader(str(video_path))
    wanted = set(indices)
    last_index = max(wanted)
    frames = {}
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index > last_index:
                break
            if frame_index in wanted:
                frames[frame_index] = evaluator.normalize_video_frame(frame)
    finally:
        reader.close()
    return frames


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


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
        f"{prefix}_p95": safe_round(percentile(values, 0.95)),
        f"{prefix}_min": safe_round(min(values)) if values else None,
        f"{prefix}_max": safe_round(max(values)) if values else None,
    }


def summarize_video(item, run_name, video_path, pair_rows, status, reason=None):
    summary = {
        "run_name": run_name,
        "row": item["_row"],
        "status": status,
        "reason": reason,
        "scene": item["scene"],
        "start_frame": item["start_frame"],
        "duration_sec": item["duration_sec"],
        "num_frames_expected": item["num_frames"],
        "pairs_evaluated": len(pair_rows),
        "output": str(video_path),
        "caption_key": item.get("caption_key"),
    }

    if not pair_rows:
        return summary

    for key in ["frame_gap", "pose_distance", "rotation_deg", *BASE_PAIR_METRICS]:
        summary.update(numeric_stats(key, [row.get(key) for row in pair_rows]))

    summary["worst_revisit_psnr_db"] = safe_round(min(row["psnr_db"] for row in pair_rows))
    summary["worst_revisit_ssim"] = safe_round(min(row["ssim"] for row in pair_rows))
    summary["worst_revisit_rmse"] = safe_round(max(row["rmse"] for row in pair_rows))
    summary["worst_revisit_mae"] = safe_round(max(row["mae"] for row in pair_rows))
    return summary


def evaluate_video(
    item,
    model_output_dir,
    run_name,
    pair_stride,
    min_revisit_gap,
    revisit_position_threshold,
    revisit_rotation_deg,
    max_pairs_per_video,
):
    video_path = output_path(model_output_dir, item)
    if not video_path.exists():
        return [], summarize_video(
            item=item,
            run_name=run_name,
            video_path=video_path,
            pair_rows=[],
            status="missing_output",
            reason="missing_output",
        )

    c2ws = load_c2ws_from_json(
        json_path=item["pose_path"],
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    pairs = discover_revisit_pairs(
        c2ws=c2ws,
        pair_stride=pair_stride,
        min_revisit_gap=min_revisit_gap,
        revisit_position_threshold=revisit_position_threshold,
        revisit_rotation_deg=revisit_rotation_deg,
        max_pairs_per_video=max_pairs_per_video,
    )
    if not pairs:
        return [], summarize_video(
            item=item,
            run_name=run_name,
            video_path=video_path,
            pair_rows=[],
            status="no_revisit_pairs",
            reason="no_pose_pairs_within_thresholds",
        )

    needed_indices = sorted({pair["frame_i"] for pair in pairs} | {pair["frame_j"] for pair in pairs})
    frames = read_video_frames(video_path, needed_indices)
    pair_rows = []
    missing_pairs = 0
    for pair in pairs:
        frame_i = frames.get(pair["frame_i"])
        frame_j = frames.get(pair["frame_j"])
        if frame_i is None or frame_j is None:
            missing_pairs += 1
            continue
        metrics = evaluator.frame_metrics(frame_j, frame_i)
        pair_rows.append(
            {
                "run_name": run_name,
                "row": item["_row"],
                "scene": item["scene"],
                "start_frame": item["start_frame"],
                "duration_sec": item["duration_sec"],
                "frame_i": pair["frame_i"],
                "frame_j": pair["frame_j"],
                "dataset_frame_i": int(item["start_frame"]) + pair["frame_i"],
                "dataset_frame_j": int(item["start_frame"]) + pair["frame_j"],
                "frame_gap": pair["frame_gap"],
                "pose_distance": pair["pose_distance"],
                "rotation_deg": pair["rotation_deg"],
                **metrics,
            }
        )

    if pair_rows:
        status = "completed"
        reason = None
    else:
        status = "missing_pair_frames"
        reason = f"{missing_pairs}_pairs_missing_frames"

    return pair_rows, summarize_video(
        item=item,
        run_name=run_name,
        video_path=video_path,
        pair_rows=pair_rows,
        status=status,
        reason=reason,
    )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(video_rows):
    completed = [row for row in video_rows if row["status"] == "completed"]
    summary = {
        "videos": len(video_rows),
        "completed": len(completed),
        "missing_outputs": sum(row["status"] == "missing_output" for row in video_rows),
        "no_revisit_pairs": sum(row["status"] == "no_revisit_pairs" for row in video_rows),
        "pairs_evaluated": int(sum(row.get("pairs_evaluated") or 0 for row in completed)),
    }
    for key in [
        "mae_mean",
        "rmse_mean",
        "psnr_db_mean",
        "ssim_mean",
        "mae_p95",
        "rmse_p95",
        "psnr_db_min",
        "ssim_min",
        "worst_revisit_psnr_db",
        "worst_revisit_ssim",
        "worst_revisit_rmse",
        "worst_revisit_mae",
    ]:
        summary[key] = safe_round(mean(row.get(key) for row in completed))
    return summary


def build_run_summary(video_rows):
    by_duration = defaultdict(list)
    for row in video_rows:
        by_duration[str(row["duration_sec"])].append(row)
    return {
        "overall": summarize_group(video_rows),
        "by_duration": {
            duration: summarize_group(rows)
            for duration, rows in sorted(by_duration.items(), key=lambda item: int(item[0]))
        },
    }


def flatten_run_summary(summary, run_name):
    rows = [{"run_name": run_name, "scope": "overall", **summary["overall"]}]
    for duration, values in summary["by_duration"].items():
        rows.append({"run_name": run_name, "scope": f"duration_{duration}", **values})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Measure generated-vs-generated consistency for planned camera revisits."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--model_output_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("analysis/context_memory/revisit"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None, help="Rows like '0,2,5-9'.")
    parser.add_argument("--start_row", type=int, default=None)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--durations", type=str, default=None, help="Optional durations like '10,20'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pair_stride", type=int, default=5)
    parser.add_argument("--min_revisit_gap", type=int, default=76)
    parser.add_argument("--revisit_position_threshold", type=float, default=2.0)
    parser.add_argument("--revisit_rotation_deg", type=float, default=20.0)
    parser.add_argument("--max_pairs_per_video", type=int, default=2000)
    args = parser.parse_args()

    if args.pair_stride < 1:
        raise ValueError("--pair_stride must be >= 1")
    if args.min_revisit_gap < 1:
        raise ValueError("--min_revisit_gap must be >= 1")
    if args.max_pairs_per_video is not None and args.max_pairs_per_video < 1:
        raise ValueError("--max_pairs_per_video must be >= 1")

    run_name = args.run_name or args.model_output_dir.name
    items = select_rows(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        start_row=args.start_row,
        end_row=args.end_row,
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pair_rows = []
    video_rows = []
    for item in items:
        print(
            f"[revisit row {item['_row']}] {item['scene']} "
            f"start={item['start_frame']} duration={item['duration_sec']}s"
        )
        pair_rows, video_summary = evaluate_video(
            item=item,
            model_output_dir=args.model_output_dir,
            run_name=run_name,
            pair_stride=args.pair_stride,
            min_revisit_gap=args.min_revisit_gap,
            revisit_position_threshold=args.revisit_position_threshold,
            revisit_rotation_deg=args.revisit_rotation_deg,
            max_pairs_per_video=args.max_pairs_per_video,
        )
        all_pair_rows.extend(pair_rows)
        video_rows.append(video_summary)

    run_summary = build_run_summary(video_rows)
    write_jsonl(args.output_dir / "revisit_pair_metrics.jsonl", all_pair_rows)
    write_csv(args.output_dir / "revisit_video_summary.csv", video_rows)
    write_csv(args.output_dir / "revisit_run_summary.csv", flatten_run_summary(run_summary, run_name))
    with (args.output_dir / "revisit_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.output_dir / 'revisit_pair_metrics.jsonl'}")
    print(f"Wrote: {args.output_dir / 'revisit_video_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'revisit_run_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'revisit_run_summary.json'}")


if __name__ == "__main__":
    main()
