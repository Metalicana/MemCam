import argparse
import json
import os
import random
from pathlib import Path

from create_context_memory_testbed import (
    collect_scene_metadata,
    duration_to_num_frames,
)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_row"] = row_index
            rows.append(row)
    return rows


def write_jsonl_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def write_text_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def infer_dataset_root(rows):
    for row in rows:
        gt_frames_dir = row.get("gt_frames_dir")
        if gt_frames_dir:
            return Path(gt_frames_dir).expanduser().resolve().parents[1]
        input_image = row.get("input_image")
        if input_image:
            return Path(input_image).expanduser().resolve().parents[2]
    raise ValueError("Could not infer dataset root; pass --dataset_root.")


def parse_existing_duration_rows(rows, duration):
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
        if int(row.get("duration_sec", -1)) == int(duration)
    ]


def output_prefixes(rows):
    return {row.get("output_prefix") for row in rows if row.get("output_prefix")}


def choose_extra_rows(
    dataset_root,
    existing_rows,
    duration,
    fps,
    add_count,
    seed,
    split_id,
    allow_scene_reuse,
):
    rng = random.Random(seed)
    num_frames, chunks = duration_to_num_frames(duration, fps)
    scene_metadata = collect_scene_metadata(dataset_root, max_num_frames=num_frames)

    used_scenes = {row["scene"] for row in existing_rows}
    used_prefixes = output_prefixes(existing_rows)
    candidates = []
    for scene_item in scene_metadata:
        if not allow_scene_reuse and scene_item["scene"] in used_scenes:
            continue
        candidates.append(scene_item)

    if len(candidates) < add_count:
        raise ValueError(
            f"Need {add_count} extra eligible scenes, but only {len(candidates)} remain "
            f"after excluding {len(used_scenes)} existing scenes. Pass --allow_scene_reuse "
            "if you want multiple starts from the same scene."
        )

    rng.shuffle(candidates)
    selected = []
    rows = []
    for scene_item in candidates:
        captions = list(scene_item["valid_captions"])
        rng.shuffle(captions)
        for caption in captions:
            output_prefix = (
                f"{split_id}_{scene_item['scene']}_{caption['start_frame']:04d}_{duration}s_"
            )
            if output_prefix in used_prefixes:
                continue
            rows.append(
                {
                    "split_id": split_id,
                    "split_seed": seed,
                    "scene": scene_item["scene"],
                    "start_frame": caption["start_frame"],
                    "duration_sec": int(duration),
                    "actual_duration_sec": round(num_frames / fps, 4),
                    "fps": fps,
                    "chunks": chunks,
                    "num_frames": num_frames,
                    "input_image": str(
                        dataset_root
                        / "frames"
                        / scene_item["scene"]
                        / f"{caption['start_frame']:04d}.png"
                    ),
                    "pose_path": str(dataset_root / "jsons" / f"{scene_item['scene']}.json"),
                    "gt_frames_dir": str(dataset_root / "frames" / scene_item["scene"]),
                    "overlap_dir": str(
                        dataset_root / "overlap_labels" / scene_item["scene"]
                    ),
                    "caption_key": caption["caption_key"],
                    "prompt": caption["caption"],
                    "output_prefix": output_prefix,
                }
            )
            selected.append(
                {
                    "scene": scene_item["scene"],
                    "start_frame": caption["start_frame"],
                    "caption_key": caption["caption_key"],
                    "output_prefix": output_prefix,
                }
            )
            used_prefixes.add(output_prefix)
            break
        if len(rows) >= add_count:
            break

    if len(rows) < add_count:
        raise ValueError(f"Only built {len(rows)} extra rows; requested {add_count}.")
    return rows, selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a 60s Context-as-Memory manifest containing the existing 60s rows "
            "plus additional non-overlapping scenes."
        )
    )
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("testbeds/context_memory/manifest.jsonl"),
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=Path("testbeds/context_memory/manifest_60s_30.jsonl"),
    )
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--target_count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split_id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--allow_scene_reuse", action="store_true")
    parser.add_argument("--new_rows_path", type=Path, default=None)
    parser.add_argument("--metadata_path", type=Path, default=None)
    args = parser.parse_args()

    input_rows = load_jsonl(args.input_manifest)
    existing_rows = parse_existing_duration_rows(input_rows, args.duration)
    if not existing_rows:
        raise RuntimeError(f"No {args.duration}s rows found in {args.input_manifest}")
    if len(existing_rows) > args.target_count:
        raise RuntimeError(
            f"Existing {args.duration}s rows ({len(existing_rows)}) exceed target "
            f"{args.target_count}."
        )

    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else infer_dataset_root(existing_rows)
    )
    add_count = args.target_count - len(existing_rows)
    split_id = args.split_id or f"seed{args.seed}_extra60s"

    extra_rows, selected = choose_extra_rows(
        dataset_root=dataset_root,
        existing_rows=existing_rows,
        duration=args.duration,
        fps=args.fps,
        add_count=add_count,
        seed=args.seed,
        split_id=split_id,
        allow_scene_reuse=args.allow_scene_reuse,
    )

    output_rows = existing_rows + extra_rows
    write_jsonl_atomic(args.output_manifest, output_rows)

    start_new = len(existing_rows)
    end_new = len(output_rows) - 1
    row_spec = f"{start_new}-{end_new}" if add_count else ""
    new_rows_path = args.new_rows_path or args.output_manifest.with_suffix(".new_rows.txt")
    write_text_atomic(new_rows_path, row_spec + "\n")

    metadata_path = args.metadata_path or args.output_manifest.with_suffix(".metadata.json")
    write_json_atomic(
        metadata_path,
        {
            "input_manifest": str(args.input_manifest),
            "output_manifest": str(args.output_manifest),
            "dataset_root": str(dataset_root),
            "duration": args.duration,
            "target_count": args.target_count,
            "existing_count": len(existing_rows),
            "added_count": len(extra_rows),
            "new_rows": row_spec,
            "seed": args.seed,
            "split_id": split_id,
            "selected_extra_rows": selected,
        },
    )

    print(f"Existing {args.duration}s rows: {len(existing_rows)}")
    print(f"Added rows: {len(extra_rows)}")
    print(f"Wrote manifest: {args.output_manifest}")
    print(f"Wrote new-row spec: {new_rows_path} -> {row_spec}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
