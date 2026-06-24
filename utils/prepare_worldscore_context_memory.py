import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


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


def parse_list(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


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
    duration_filter = set(durations) if durations else None
    selected = []
    for item in items:
        row = item["_row"]
        if row_filter is not None and row not in row_filter:
            continue
        if duration_filter is not None and int(item["duration_sec"]) not in duration_filter:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def output_path(run_dir, item):
    return run_dir / f"{item['output_prefix']}custom.mp4"


def map_video_index_to_manifest_local(video_idx, total_video_frames, manifest_num_frames):
    if total_video_frames <= 1 or manifest_num_frames <= 1:
        return min(video_idx, max(0, manifest_num_frames - 1))
    scaled = round(video_idx * (manifest_num_frames - 1) / (total_video_frames - 1))
    return int(min(max(scaled, 0), manifest_num_frames - 1))


def sampled_indices(total_frames, frame_stride, max_frames):
    indices = list(range(0, total_frames, frame_stride))
    if max_frames is not None and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = [indices[int(round(pos))] for pos in positions]
        indices = sorted(set(indices))
    return indices


def extract_frames(video_path, frames_dir, frame_stride, max_frames, manifest_num_frames):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to prepare WorldScore frames.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = sampled_indices(total_frames, frame_stride, max_frames)
    frames_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    local_indices = []
    for output_idx, video_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = frames_dir / f"{output_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        written_paths.append(frame_path)
        local_indices.append(
            map_video_index_to_manifest_local(video_idx, total_frames, manifest_num_frames)
        )

    cap.release()
    if not written_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return written_paths, local_indices, total_frames


def resolve_pose_path(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "jsons" / f"{item['scene']}.json"
    return Path(item["pose_path"])


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_instance(output_root, run_name, item, video_path, frame_stride, max_frames, dataset_root, overwrite):
    instance_name = f"row{int(item['_row']):04d}_{item['scene']}_{int(item['start_frame']):04d}_{int(item['duration_sec'])}s"
    instance_dir = output_root / "worldscore_output" / "static" / "memcam" / "context_memory" / run_name / instance_name

    if instance_dir.exists() and overwrite:
        shutil.rmtree(instance_dir)
    if (instance_dir / "image_data.json").exists() and not overwrite:
        return {"status": "skipped_existing", "instance_dir": str(instance_dir)}

    frames_dir = instance_dir / "frames"
    frames, local_indices, total_video_frames = extract_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        frame_stride=frame_stride,
        max_frames=max_frames,
        manifest_num_frames=int(item["num_frames"]),
    )

    input_image_path = instance_dir / "input_image.png"
    with Image.open(frames[0]) as image:
        image.convert("RGB").save(input_image_path)

    pose_path = resolve_pose_path(item, dataset_root)
    all_gt = load_c2ws_from_json(
        json_path=pose_path,
        start_frame=int(item["start_frame"]),
        num_frames=int(item["num_frames"]),
    )
    cameras_interp = all_gt[local_indices].astype(float).tolist()

    prompt = item.get("prompt", "")
    image_data = {
        "visual_movement": "static",
        "visual_style": "memcam",
        "scene_type": "context_memory",
        "camera_path": ["move_left"],
        "total_frames": len(frames),
        "prompt_list": [prompt, prompt],
        "num_scenes": 1,
        "anchor_frame_idx": [0, len(frames) - 1],
        "content_list": [item.get("scene", "scene"), item.get("scene", "scene")],
        "source_run": run_name,
        "source_scene": item["scene"],
        "source_row": item["_row"],
        "source_video": str(video_path),
        "source_manifest_output_prefix": item["output_prefix"],
        "video_frame_stride": frame_stride,
        "video_total_frames": total_video_frames,
        "manifest_local_indices": local_indices,
    }
    camera_data = {
        "focal_length": 500,
        "scale": 1.0,
        "cameras_interp": cameras_interp,
    }

    write_json(instance_dir / "image_data.json", image_data)
    write_json(instance_dir / "camera_data.json", camera_data)
    return {
        "status": "prepared",
        "run_name": run_name,
        "row": item["_row"],
        "frames": len(frames),
        "instance_dir": str(instance_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MemCam generated videos into WorldScore static-evaluation instance layout."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--worldscore_runs_root", type=Path, required=True)
    parser.add_argument("--durations", type=str, default="60")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    args.worldscore_runs_root.mkdir(parents=True, exist_ok=True)
    items = select_rows(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    results = []
    for run_name in parse_list(args.runs):
        run_dir = args.root / run_name
        for item in items:
            video_path = output_path(run_dir, item)
            if not video_path.exists():
                result = {
                    "status": "missing_video",
                    "run_name": run_name,
                    "row": item["_row"],
                    "video_path": str(video_path),
                }
                print(result)
                results.append(result)
                continue

            result = prepare_instance(
                output_root=args.worldscore_runs_root,
                run_name=run_name,
                item=item,
                video_path=video_path,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                dataset_root=args.dataset_root,
                overwrite=args.overwrite,
            )
            print(result)
            results.append(result)

    write_json(args.worldscore_runs_root / "memcam_worldscore_prepare_status.json", results)
    print(f"Wrote: {args.worldscore_runs_root / 'memcam_worldscore_prepare_status.json'}")


if __name__ == "__main__":
    main()
