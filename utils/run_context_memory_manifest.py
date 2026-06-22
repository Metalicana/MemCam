import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from utils.run_context_memory_batch import assert_video_writer_available


def read_manifest_row(manifest_path, row_index):
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"Manifest row {row_index} not found in {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Run one Context-as-Memory manifest item.")
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/ab575577/MemCam/outputs/context_memory"),
    )
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--memory_policy",
        type=str,
        default="unbounded",
        choices=[
            "unbounded",
            "fifo",
            "rarity_irreplaceability",
            "slam_covisibility",
            "facility_coreset",
        ],
    )
    parser.add_argument("--memory_budget", type=int, default=None)
    parser.add_argument("--access_trace_dir", type=Path, default=None)
    args = parser.parse_args()

    item = read_manifest_row(args.manifest, args.row)
    num_inference_steps = 20 if args.smoke else args.num_inference_steps
    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_video_writer_available(output_dir)
    access_trace_dir = args.access_trace_dir or (output_dir / "access_traces")
    access_trace_dir.mkdir(parents=True, exist_ok=True)
    access_trace_path = access_trace_dir / f"{item['output_prefix']}custom.jsonl"

    command = [
        sys.executable,
        "inference_memcam.py",
        "--trajectory_mode",
        "custom",
        "--input_image",
        item["input_image"],
        "--pose_path",
        item["pose_path"],
        "--start_frame",
        str(item["start_frame"]),
        "--num_frames",
        str(item["num_frames"]),
        "--prompt",
        item["prompt"],
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_inference_steps",
        str(num_inference_steps),
        "--seed",
        str(args.seed),
        "--memory_policy",
        args.memory_policy,
        "--device",
        "cuda",
        "--output_dir",
        str(output_dir),
        "--output_prefix",
        item["output_prefix"],
        "--access_trace_path",
        str(access_trace_path),
    ]
    if args.memory_budget is not None:
        command.extend(["--memory_budget", str(args.memory_budget)])

    env = os.environ.copy()
    visible_gpu = env.get("CUDA_VISIBLE_DEVICES")
    if not visible_gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        visible_gpu = args.gpu

    print(f"Running manifest row {args.row} on requested GPU arg {args.gpu}")
    print(f"CUDA_VISIBLE_DEVICES: {visible_gpu}")
    print(f"Scene: {item['scene']}")
    print(f"Start frame: {item['start_frame']}")
    print(f"Frames: {item['num_frames']} ({item['actual_duration_sec']}s)")
    print(f"Steps: {num_inference_steps}")
    print(f"Memory policy: {args.memory_policy}, budget: {args.memory_budget}")
    print(f"Output dir: {output_dir}")
    print(f"Caption key: {item['caption_key']}")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
