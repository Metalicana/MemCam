"""Replay one MemCam section with generated versus GT-cleaned memory content."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.run_context_replay_case import (  # noqa: E402
    NEGATIVE_PROMPT,
    append_jsonl,
    load_manifest,
    load_replay_case,
    load_trace_overrides,
    reset_random_state,
)


def case_directory_name(case):
    return (
        f"case_{int(case['case_index']):02d}_row{int(case['row'])}_"
        f"section{int(case['section_idx'])}"
    )


def item_identity(item):
    return (
        str(item["scene"]),
        int(item["start_frame"]),
        int(item["duration_sec"]),
    )


def indexed_frame_paths(frame_dir):
    import re

    paths = {}
    for path in Path(frame_dir).iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match:
            paths[int(match.group(1))] = path
    return paths


def resolve_gt_dir(item, dataset_root=None):
    if dataset_root is not None:
        candidate = Path(dataset_root) / "frames" / item["scene"]
        if candidate.is_dir():
            return candidate
    original = item.get("gt_frames_dir")
    if original and Path(original).is_dir():
        return Path(original)
    raise FileNotFoundError(
        f"Ground-truth frames are unavailable for manifest row {item['_row']}"
    )


def build_clean_content_overrides(
    item,
    section_idx,
    selection_overrides,
    width,
    height,
    dataset_root=None,
):
    from PIL import Image

    gt_dir = resolve_gt_dir(item, dataset_root=dataset_root)
    frame_paths = indexed_frame_paths(gt_dir)
    cached_images = {}
    output = {int(section_idx): {}}
    for target_frame, selection in selection_overrides[int(section_idx)].items():
        memory_frame = int(selection["memory_frame"])
        dataset_frame = int(item["start_frame"]) + memory_frame
        if memory_frame not in cached_images:
            path = frame_paths.get(dataset_frame)
            if path is None:
                raise FileNotFoundError(
                    f"Missing GT frame {dataset_frame} in {gt_dir}"
                )
            with Image.open(path) as image:
                cached_images[memory_frame] = image.convert("RGB").resize(
                    (int(width), int(height)),
                    resample=Image.BICUBIC,
                )
        output[int(section_idx)][int(target_frame)] = {
            "image": cached_images[memory_frame],
            "memory_frame": memory_frame,
            "source": "ground_truth_memory",
        }
    return output, len(cached_images)


def parse_branches(value):
    branches = [part.strip() for part in str(value).split(",") if part.strip()]
    invalid = sorted(set(branches) - {"control", "clean_gt"})
    if invalid:
        raise ValueError(f"Unknown branches: {invalid}")
    if not branches:
        raise ValueError("At least one branch is required")
    return branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case_index", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--branches", default="control,clean_gt")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    branches = parse_branches(args.branches)
    case = load_replay_case(args.plan, args.case_index)
    items = load_manifest(args.manifest)
    row_idx = int(case["row"])
    item = items[row_idx]
    if item_identity(item) != (
        str(case["scene"]),
        int(case["dataset_start_frame"]),
        int(case["duration_sec"]),
    ):
        raise ValueError(
            f"Replay plan identity does not match manifest row {row_idx}"
        )

    section_idx = int(case["section_idx"])
    baseline_trace = (
        args.root
        / args.baseline_run
        / "access_traces"
        / f"{item['output_prefix']}custom.jsonl"
    )
    if not baseline_trace.is_file():
        raise FileNotFoundError(f"Missing baseline trace: {baseline_trace}")

    selection_overrides = load_trace_overrides(
        baseline_trace,
        item,
        sections=range(1, section_idx + 1),
        source_run=args.baseline_run,
    )
    clean_overrides, unique_cleaned_frames = build_clean_content_overrides(
        item=item,
        section_idx=section_idx,
        selection_overrides=selection_overrides,
        width=args.width,
        height=args.height,
        dataset_root=args.dataset_root,
    )

    case_dir = args.output_root / case_directory_name(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    status_path = case_dir / "run_status.jsonl"
    metadata = {
        "case_index": int(args.case_index),
        "row": row_idx,
        "scene": item["scene"],
        "dataset_start_frame": int(item["start_frame"]),
        "duration_sec": int(item["duration_sec"]),
        "section_idx": section_idx,
        "cleaned_context_slots": len(clean_overrides[section_idx]),
        "unique_cleaned_memory_frames": unique_cleaned_frames,
        "baseline_run": args.baseline_run,
        "seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
        "planned_selected_memory_corruption": float(
            case["mean_selected_memory_corruption"]
        ),
    }
    (case_dir / "case.json").write_text(
        json.dumps({**case, **metadata}, indent=2) + "\n",
        encoding="utf-8",
    )

    from PIL import Image
    import torch

    from dataset.poses import load_c2ws_from_json
    from diffsynth import save_video
    from inference_memcam import setup_pipeline
    from utils.run_context_memory_batch import assert_video_writer_available

    assert_video_writer_available(case_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Memory-cleaning replay requires a CUDA GPU")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Memory-cleaning case: {json.dumps(metadata, sort_keys=True)}")

    input_image = Image.open(item["input_image"]).convert("RGB").resize(
        (args.width, args.height), resample=Image.BICUBIC
    )
    c2ws = load_c2ws_from_json(
        json_path=item["pose_path"],
        start_frame=item["start_frame"],
        num_frames=item["num_frames"],
    )
    pipe = setup_pipeline(
        dit_path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        text_encoder_path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        vae_path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        dit_ckpt_path="models/MemCam/dit_step20000.ckpt",
        device="cuda",
    )

    for branch_name in branches:
        branch_dir = case_dir / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = branch_dir / "access_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_path = branch_dir / f"{item['output_prefix']}custom.mp4"
        trace_path = trace_dir / f"{item['output_prefix']}custom.jsonl"
        if output_path.is_file() and not args.overwrite:
            print(f"[{branch_name}] skip existing: {output_path}")
            continue

        content_overrides = clean_overrides if branch_name == "clean_gt" else None
        print(f"[{branch_name}] starting")
        started = time.time()
        reset_random_state(args.seed)
        try:
            video = pipe(
                prompt=item["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                input_image=input_image.copy(),
                c2ws=c2ws,
                height=args.height,
                width=args.width,
                cfg_scale=5.0,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                memory_policy="unbounded",
                memory_budget=None,
                memory_bank_device="cpu",
                context_selection_overrides=selection_overrides,
                context_content_overrides=content_overrides,
                stop_after_section=section_idx,
                access_trace_path=str(trace_path),
                access_trace_metadata={
                    **metadata,
                    "branch": branch_name,
                    "output": str(output_path),
                    "output_prefix": item["output_prefix"],
                },
                tiled=False,
            )
            generated_frames = len(video)
            save_video(video, str(output_path), fps=item["fps"], quality=5)
            del video
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            append_jsonl(
                status_path,
                {
                    **metadata,
                    "branch": branch_name,
                    "status": "failed",
                    "error": repr(exc),
                    "time_sec": round(time.time() - started, 2),
                },
            )
            raise

        elapsed = round(time.time() - started, 2)
        append_jsonl(
            status_path,
            {
                **metadata,
                "branch": branch_name,
                "status": "completed",
                "output": str(output_path),
                "generated_frames": int(generated_frames),
                "time_sec": elapsed,
            },
        )
        print(f"[{branch_name}] completed in {elapsed}s: {output_path}")

    print(f"Replay complete: {case_dir}")


if __name__ == "__main__":
    main()
