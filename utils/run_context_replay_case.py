import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，静止不动的画面，杂乱的背景"
)
SECTION_STRIDE = 76
PREDICT_FRAMES = 76


def load_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_row"] = row_idx
            rows.append(item)
    return rows


def load_replay_case(path, case_index):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["case_index"]) == int(case_index):
            return row
    raise IndexError(f"Replay case {case_index} is not present in {path}")


def item_identity(item):
    return (
        str(item["scene"]),
        int(item["start_frame"]),
        int(item["duration_sec"]),
    )


def trace_identity(row):
    return (
        str(row.get("scene")),
        int(row.get("dataset_start_frame")),
        int(row.get("duration_sec")),
    )


def load_trace_overrides(trace_path, item, sections, source_run):
    sections = {int(section) for section in sections}
    overrides = {section: {} for section in sections}
    with Path(trace_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene") is not None and trace_identity(row) != item_identity(item):
                raise ValueError(
                    f"Trace identity mismatch in {trace_path}: "
                    f"expected {item_identity(item)}, found {trace_identity(row)}"
                )
            if row.get("event") != "context_access" or not row.get("selected"):
                continue
            section_idx = int(row["section_idx"])
            if section_idx not in sections:
                continue
            target_frame = int(row["target_frame"])
            overrides[section_idx][target_frame] = {
                "memory_frame": int(row["selected_memory_frame"]),
                "source_run": source_run,
            }

    for section_idx in sorted(sections):
        expected_targets = set(
            range(
                section_idx * SECTION_STRIDE + 1,
                section_idx * SECTION_STRIDE + 1 + PREDICT_FRAMES,
            )
        )
        actual_targets = set(overrides[section_idx])
        if actual_targets != expected_targets:
            missing = sorted(expected_targets - actual_targets)
            extra = sorted(actual_targets - expected_targets)
            raise RuntimeError(
                f"Incomplete context overrides in {trace_path} section {section_idx}: "
                f"missing={missing[:5]}, extra={extra[:5]}, "
                f"found={len(actual_targets)}/{len(expected_targets)}"
            )
    return overrides


def merge_overrides(base, replacement):
    merged = {
        int(section): {
            int(target): dict(value) for target, value in target_rows.items()
        }
        for section, target_rows in base.items()
    }
    for section, target_rows in replacement.items():
        merged.setdefault(int(section), {}).update(
            {int(target): dict(value) for target, value in target_rows.items()}
        )
    return merged


def case_directory_name(case):
    return (
        f"case_{int(case['case_index']):02d}_row{int(case['row'])}_"
        f"section{int(case['section_idx'])}"
    )


def reset_random_state(seed):
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay one late MemCam section twice with matched unbounded history: "
            "original unbounded contexts versus a bounded policy's contexts."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case_index", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
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

    case = load_replay_case(args.plan, args.case_index)
    items = load_manifest(args.manifest)
    row_idx = int(case["row"])
    if row_idx < 0 or row_idx >= len(items):
        raise IndexError(f"Manifest row {row_idx} is unavailable")
    item = items[row_idx]
    if item_identity(item) != (
        str(case["scene"]),
        int(case["dataset_start_frame"]),
        int(case["duration_sec"]),
    ):
        raise ValueError(
            f"Replay plan identity does not match manifest row {row_idx}: "
            f"plan={(case['scene'], case['dataset_start_frame'], case['duration_sec'])}, "
            f"manifest={item_identity(item)}"
        )

    section_idx = int(case["section_idx"])
    baseline_run = case["baseline_run"]
    intervention_run = case["intervention_run"]
    baseline_trace = (
        args.root
        / baseline_run
        / "access_traces"
        / f"{item['output_prefix']}custom.jsonl"
    )
    intervention_trace = (
        args.root
        / intervention_run
        / "access_traces"
        / f"{item['output_prefix']}custom.jsonl"
    )
    if not baseline_trace.is_file():
        raise FileNotFoundError(f"Missing baseline trace: {baseline_trace}")
    if not intervention_trace.is_file():
        raise FileNotFoundError(f"Missing intervention trace: {intervention_trace}")

    # Every earlier section is forced to the recorded unbounded choices in both
    # branches. Only the final target section differs.
    history_sections = range(1, section_idx + 1)
    control_overrides = load_trace_overrides(
        baseline_trace,
        item,
        sections=history_sections,
        source_run=baseline_run,
    )
    intervention_section = load_trace_overrides(
        intervention_trace,
        item,
        sections=[section_idx],
        source_run=intervention_run,
    )
    swap_overrides = merge_overrides(control_overrides, intervention_section)
    changed_targets = sum(
        control_overrides[section_idx][target]["memory_frame"]
        != swap_overrides[section_idx][target]["memory_frame"]
        for target in control_overrides[section_idx]
    )
    if changed_targets == 0:
        raise RuntimeError(
            f"Case {args.case_index} has no context changes at section {section_idx}"
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
        "changed_target_count": int(changed_targets),
        "baseline_run": baseline_run,
        "intervention_run": intervention_run,
        "seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
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
        raise RuntimeError("Matched replay requires a CUDA GPU")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Replay case: {json.dumps(metadata, sort_keys=True)}")

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

    branches = [
        ("control", control_overrides, baseline_run),
        (f"swap_{intervention_run}", swap_overrides, intervention_run),
    ]
    for branch_name, overrides, target_source_run in branches:
        branch_dir = case_dir / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = branch_dir / "access_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_path = branch_dir / f"{item['output_prefix']}custom.mp4"
        trace_path = trace_dir / f"{item['output_prefix']}custom.jsonl"
        if output_path.is_file() and not args.overwrite:
            print(f"[{branch_name}] skip existing: {output_path}")
            continue

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
                context_selection_overrides=overrides,
                stop_after_section=section_idx,
                access_trace_path=str(trace_path),
                access_trace_metadata={
                    **metadata,
                    "branch": branch_name,
                    "target_context_source_run": target_source_run,
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
