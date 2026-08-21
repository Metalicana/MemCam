import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
            "slam_max_coverage",
            "reliable_slam_ri",
            "facility_coreset",
            "kcenter_coreset",
            "trajectory_coverage",
            "density_balanced_view_coverage",
            "h2o_heavy_hitter",
            "surprise_forcing",
        ],
    )
    parser.add_argument("--memory_budget", type=int, default=None)
    parser.add_argument("--rsri_slam_weight", type=float, default=0.75)
    parser.add_argument("--rsri_rarity_neighbors", type=int, default=3)
    parser.add_argument("--rsri_reliability_neighbors", type=int, default=3)
    parser.add_argument("--rsri_reliability_min_support", type=int, default=2)
    parser.add_argument(
        "--rsri_reliability_geometry_threshold", type=float, default=0.50
    )
    parser.add_argument("--rsri_reliability_threshold", type=float, default=0.80)
    parser.add_argument("--density_coverage_alpha", type=float, default=0.5)
    parser.add_argument("--density_coverage_dino_weight", type=float, default=0.5)
    parser.add_argument("--density_coverage_rgb_weight", type=float, default=0.25)
    parser.add_argument("--surprise_alpha", type=float, default=0.7)
    parser.add_argument("--surprise_ema_momentum", type=float, default=0.95)
    parser.add_argument("--surprise_controller_step", type=float, default=0.1)
    parser.add_argument(
        "--surprise_target_admission_ratio", type=float, default=0.3
    )
    parser.add_argument("--surprise_initial_threshold", type=float, default=0.002)
    parser.add_argument("--surprise_surprise_weight", type=float, default=1.8)
    parser.add_argument("--surprise_usage_weight", type=float, default=1.0)
    parser.add_argument("--surprise_age_weight", type=float, default=0.4)
    parser.add_argument("--surprise_route_top_k", type=int, default=3)
    parser.add_argument("--surprise_value_layer", type=int, default=15)
    parser.add_argument("--surprise_warmup_sections", type=int, default=3)
    parser.add_argument(
        "--memory_bank_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--access_trace_dir", type=Path, default=None)
    parser.add_argument("--profile_dir", type=Path, default=None)
    args = parser.parse_args()

    item = read_manifest_row(args.manifest, args.row)
    num_inference_steps = 20 if args.smoke else args.num_inference_steps
    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_video_writer_available(output_dir)
    access_trace_dir = args.access_trace_dir or (output_dir / "access_traces")
    access_trace_dir.mkdir(parents=True, exist_ok=True)
    access_trace_path = access_trace_dir / f"{item['output_prefix']}custom.jsonl"
    profile_path = None
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = args.profile_dir / f"{item['output_prefix']}custom.jsonl"

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
        "--memory_bank_device",
        args.memory_bank_device,
        "--rsri_slam_weight",
        str(args.rsri_slam_weight),
        "--rsri_rarity_neighbors",
        str(args.rsri_rarity_neighbors),
        "--rsri_reliability_neighbors",
        str(args.rsri_reliability_neighbors),
        "--rsri_reliability_min_support",
        str(args.rsri_reliability_min_support),
        "--rsri_reliability_geometry_threshold",
        str(args.rsri_reliability_geometry_threshold),
        "--rsri_reliability_threshold",
        str(args.rsri_reliability_threshold),
        "--density_coverage_alpha",
        str(args.density_coverage_alpha),
        "--density_coverage_dino_weight",
        str(args.density_coverage_dino_weight),
        "--density_coverage_rgb_weight",
        str(args.density_coverage_rgb_weight),
        "--surprise_alpha",
        str(args.surprise_alpha),
        "--surprise_ema_momentum",
        str(args.surprise_ema_momentum),
        "--surprise_controller_step",
        str(args.surprise_controller_step),
        "--surprise_target_admission_ratio",
        str(args.surprise_target_admission_ratio),
        "--surprise_initial_threshold",
        str(args.surprise_initial_threshold),
        "--surprise_surprise_weight",
        str(args.surprise_surprise_weight),
        "--surprise_usage_weight",
        str(args.surprise_usage_weight),
        "--surprise_age_weight",
        str(args.surprise_age_weight),
        "--surprise_route_top_k",
        str(args.surprise_route_top_k),
        "--surprise_value_layer",
        str(args.surprise_value_layer),
        "--surprise_warmup_sections",
        str(args.surprise_warmup_sections),
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
    if profile_path is not None:
        command.extend(["--profile_path", str(profile_path)])

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
    if args.memory_policy == "density_balanced_view_coverage":
        print(
            "Density coverage: "
            f"alpha={args.density_coverage_alpha}, "
            f"DINO weight={args.density_coverage_dino_weight}, "
            f"RGB weight={args.density_coverage_rgb_weight}"
        )
    if args.memory_policy == "surprise_forcing":
        print(
            "Surprise Forcing memory: "
            f"alpha={args.surprise_alpha}, "
            f"target write ratio={args.surprise_target_admission_ratio}, "
            f"route top-k={args.surprise_route_top_k}, "
            f"value layer={args.surprise_value_layer}"
        )
    print(f"Memory bank device: {args.memory_bank_device}")
    print(f"Profile path: {profile_path}")
    print(f"Output dir: {output_dir}")
    print(f"Caption key: {item['caption_key']}")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
