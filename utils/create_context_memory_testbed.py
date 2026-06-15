import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_DURATIONS = [10, 20, 40, 60, 120]
FRAMES_PER_CHUNK = 76


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def duration_to_num_frames(duration_sec, fps):
    chunks = max(1, round(duration_sec * fps / FRAMES_PER_CHUNK))
    return chunks * FRAMES_PER_CHUNK + 1, chunks


def load_captions(captions_path):
    captions_by_scene = defaultdict(list)

    with captions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            key, caption = parts
            try:
                scene, segment = key.split("/", 1)
                start_text, end_text = segment.removesuffix(".mp4").split("_", 1)
                start_frame = int(start_text)
                end_frame = int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid caption key on line {line_number}: {key}") from exc

            captions_by_scene[scene].append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "caption_key": key,
                    "caption": caption,
                }
            )

    for scene_captions in captions_by_scene.values():
        scene_captions.sort(key=lambda item: item["start_frame"])

    return captions_by_scene


def count_scene_frames(frames_dir):
    return len(list(frames_dir.glob("*.png")))


def count_scene_poses(json_path):
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return len(data["CineCameraActor"])


def collect_scene_metadata(dataset_root, max_num_frames):
    frames_root = dataset_root / "frames"
    jsons_root = dataset_root / "jsons"
    captions_path = dataset_root / "captions.txt"

    captions_by_scene = load_captions(captions_path)
    scene_metadata = []

    for frames_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        scene = frames_dir.name
        json_path = jsons_root / f"{scene}.json"
        if not json_path.exists():
            continue

        frame_count = count_scene_frames(frames_dir)
        pose_count = count_scene_poses(json_path)
        usable_frames = min(frame_count, pose_count)
        valid_captions = [
            item
            for item in captions_by_scene.get(scene, [])
            if item["start_frame"] + max_num_frames <= usable_frames
        ]

        if not valid_captions:
            continue

        scene_metadata.append(
            {
                "scene": scene,
                "frame_count": frame_count,
                "pose_count": pose_count,
                "usable_frames": usable_frames,
                "valid_captions": valid_captions,
            }
        )

    return scene_metadata


def build_split(scene_metadata, seed, scenes_per_split):
    rng = random.Random(seed)
    if scenes_per_split > len(scene_metadata):
        raise ValueError(
            f"Requested {scenes_per_split} scenes, but only {len(scene_metadata)} are eligible."
        )

    selected_scenes = rng.sample(scene_metadata, scenes_per_split)
    selected_scenes.sort(key=lambda item: item["scene"])

    split_items = []
    for item in selected_scenes:
        caption = rng.choice(item["valid_captions"])
        split_items.append(
            {
                "scene": item["scene"],
                "frame_count": item["frame_count"],
                "pose_count": item["pose_count"],
                "usable_frames": item["usable_frames"],
                "start_frame": caption["start_frame"],
                "caption_key": caption["caption_key"],
                "caption": caption["caption"],
            }
        )

    return split_items


def make_manifest_rows(dataset_root, split_id, split_seed, split_items, durations, fps):
    rows = []
    for item in split_items:
        scene = item["scene"]
        for duration_sec in durations:
            num_frames, chunks = duration_to_num_frames(duration_sec, fps)
            rows.append(
                {
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "scene": scene,
                    "start_frame": item["start_frame"],
                    "duration_sec": duration_sec,
                    "actual_duration_sec": round(num_frames / fps, 4),
                    "fps": fps,
                    "chunks": chunks,
                    "num_frames": num_frames,
                    "input_image": str(
                        dataset_root / "frames" / scene / f"{item['start_frame']:04d}.png"
                    ),
                    "pose_path": str(dataset_root / "jsons" / f"{scene}.json"),
                    "gt_frames_dir": str(dataset_root / "frames" / scene),
                    "overlap_dir": str(dataset_root / "overlap_labels" / scene),
                    "caption_key": item["caption_key"],
                    "prompt": item["caption"],
                    "output_prefix": (
                        f"{split_id}_{scene}_{item['start_frame']:04d}_{duration_sec}s_"
                    ),
                }
            )
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic Context-as-Memory testbed splits and run manifests."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("testbeds/context_memory"))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--scenes_per_split", type=int, default=15)
    parser.add_argument("--durations", type=str, default=",".join(map(str, DEFAULT_DURATIONS)))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    durations = parse_int_list(args.durations)
    max_num_frames = max(duration_to_num_frames(duration, args.fps)[0] for duration in durations)

    scene_metadata = collect_scene_metadata(dataset_root, max_num_frames)
    if not scene_metadata:
        raise RuntimeError("No eligible scenes found. Check dataset path and captions.")

    all_rows = []
    split_records = {
        "dataset_root": str(dataset_root),
        "fps": args.fps,
        "durations": durations,
        "max_num_frames": max_num_frames,
        "scenes_per_split": args.scenes_per_split,
        "eligible_scene_count": len(scene_metadata),
        "splits": {},
    }

    for seed in seeds:
        split_id = f"seed{seed}"
        split_items = build_split(scene_metadata, seed, args.scenes_per_split)
        split_records["splits"][split_id] = split_items
        all_rows.extend(
            make_manifest_rows(
                dataset_root=dataset_root,
                split_id=split_id,
                split_seed=seed,
                split_items=split_items,
                durations=durations,
                fps=args.fps,
            )
        )

    splits_path = output_dir / "splits.json"
    manifest_path = output_dir / "manifest.jsonl"

    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(split_records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    write_jsonl(manifest_path, all_rows)

    print(f"Eligible scenes: {len(scene_metadata)}")
    print(f"Wrote splits: {splits_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Runs: {len(all_rows)}")


if __name__ == "__main__":
    main()
