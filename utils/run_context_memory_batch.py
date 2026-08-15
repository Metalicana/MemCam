import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，静止不动的画面，杂乱的背景"
)


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


def parse_int_csv(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def select_rows(items, row_filter, start_row, end_row, durations):
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
        if duration_filter is not None and item["duration_sec"] not in duration_filter:
            continue
        selected.append(item)

    return selected


def output_path(output_dir, item):
    return output_dir / f"{item['output_prefix']}custom.mp4"


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def jsonl_has_event(path, event):
    if path is None or not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == event:
                return True
    return False


def assert_video_writer_available(output_dir):
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Video writer preflight failed before model loading. Install imageio and numpy."
        ) from exc

    probe_path = output_dir / ".ffmpeg_preflight.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        writer = imageio.get_writer(str(probe_path), format="FFMPEG", fps=1, quality=5)
        writer.append_data(frame)
        writer.close()
    except Exception as exc:
        raise RuntimeError(
            "Video writer preflight failed before model loading. On Newton run: "
            "conda activate memcam && python -m pip install --no-cache-dir "
            "'imageio[ffmpeg]' imageio-ffmpeg"
        ) from exc
    finally:
        probe_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Context-as-Memory manifest rows sequentially with one loaded model."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/ab575577/MemCam/outputs/context_memory"),
    )
    parser.add_argument("--rows", type=str, default=None, help="Rows like '0,2,5-9'.")
    parser.add_argument("--start_row", type=int, default=None)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--durations", type=str, default=None, help="Optional durations like '10,20'.")
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Run at most this many manifest items after applying all row and duration filters.",
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
            "facility_coreset",
            "kcenter_coreset",
            "trajectory_coverage",
            "density_balanced_view_coverage",
            "future_view_coverage",
            "mce",
            "h2o_heavy_hitter",
            "surprise_forcing",
        ],
    )
    parser.add_argument("--memory_budget", type=int, default=None)
    parser.add_argument("--density_coverage_alpha", type=float, default=0.5)
    parser.add_argument("--density_coverage_dino_weight", type=float, default=0.5)
    parser.add_argument("--density_coverage_rgb_weight", type=float, default=0.25)
    parser.add_argument("--future_coverage_alpha", type=float, default=0.5)
    parser.add_argument("--future_coverage_dino_weight", type=float, default=0.5)
    parser.add_argument("--future_coverage_rgb_weight", type=float, default=0.25)
    parser.add_argument(
        "--future_coverage_query_stride",
        type=int,
        default=19,
        help="Frame stride used to sample the known future camera path as coverage demand.",
    )
    parser.add_argument(
        "--future_coverage_query_weight",
        type=float,
        default=1.0,
        help="Per-query demand mass given to each sampled future viewpoint.",
    )
    parser.add_argument(
        "--mce_alpha",
        type=float,
        default=0.65,
        help="MCE kernel blend: alpha * K_geo + (1 - alpha) * K_vis.",
    )
    parser.add_argument(
        "--mce_lambda",
        type=float,
        default=None,
        help="MCE historical-vs-control query weight split. Defaults to 1.0 with no "
        "future queries, else 0.5.",
    )
    parser.add_argument(
        "--mce_gamma",
        type=float,
        default=0.25,
        help="MCE exponential horizon-decay rate for future control queries.",
    )
    parser.add_argument(
        "--mce_query_stride",
        type=int,
        default=19,
        help="Frame stride used to sample the known future camera path for MCE's Q_ctrl.",
    )
    parser.add_argument(
        "--mce_rarity_neighbors",
        type=int,
        default=3,
        help="k for MCE's Q_hist clustering threshold (estimate_cluster_threshold's "
        "k-th nearest-neighbor distance). Higher values coarsen clusters (fewer, "
        "bigger); was a dead parameter before the estimate_cluster_threshold fix.",
    )
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
        help="Device that persistently stores retained memory-frame tensors.",
    )
    parser.add_argument(
        "--access_trace_dir",
        type=Path,
        default=None,
        help="Directory for per-video JSONL memory access traces. Defaults to output_dir/access_traces.",
    )
    parser.add_argument(
        "--profile_dir",
        type=Path,
        default=None,
        help="Optional directory for per-video rollout latency/VRAM JSONL profiles.",
    )
    parser.add_argument(
        "--attention_audit_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for target-to-memory attention and controlled "
            "memory-ablation JSONL traces."
        ),
    )
    parser.add_argument(
        "--attention_probe_sections",
        type=str,
        default="3,7,11",
        help="Zero-based generation sections at which to audit attention.",
    )
    parser.add_argument(
        "--attention_probe_steps",
        type=str,
        default="5,25,45",
        help="Zero-based denoising steps at which to audit attention.",
    )
    parser.add_argument(
        "--attention_probe_layers",
        type=str,
        default="5,15,25",
        help="Zero-based DiT blocks from which to sample self-attention.",
    )
    parser.add_argument("--attention_query_count", type=int, default=128)
    parser.add_argument("--attention_query_chunk_size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    slurm_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if slurm_gpu:
        visible_gpu = slurm_gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        visible_gpu = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    num_inference_steps = 20 if args.smoke else args.num_inference_steps
    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    access_trace_dir = args.access_trace_dir or (output_dir / "access_traces")
    access_trace_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = args.profile_dir
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
    attention_audit_dir = args.attention_audit_dir
    if attention_audit_dir is not None:
        attention_audit_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.jsonl"
    policy_metadata = {
        "memory_policy": args.memory_policy,
        "memory_budget": args.memory_budget,
        "memory_bank_device": args.memory_bank_device,
        "density_coverage_alpha": args.density_coverage_alpha,
        "density_coverage_dino_weight": args.density_coverage_dino_weight,
        "density_coverage_rgb_weight": args.density_coverage_rgb_weight,
        "future_coverage_alpha": args.future_coverage_alpha,
        "future_coverage_dino_weight": args.future_coverage_dino_weight,
        "future_coverage_rgb_weight": args.future_coverage_rgb_weight,
        "future_coverage_query_stride": args.future_coverage_query_stride,
        "future_coverage_query_weight": args.future_coverage_query_weight,
        "mce_alpha": args.mce_alpha,
        "mce_lambda": args.mce_lambda,
        "mce_gamma": args.mce_gamma,
        "mce_query_stride": args.mce_query_stride,
        "mce_rarity_neighbors": args.mce_rarity_neighbors,
        "surprise_alpha": args.surprise_alpha,
        "surprise_ema_momentum": args.surprise_ema_momentum,
        "surprise_controller_step": args.surprise_controller_step,
        "surprise_target_admission_ratio": args.surprise_target_admission_ratio,
        "surprise_initial_threshold": args.surprise_initial_threshold,
        "surprise_surprise_weight": args.surprise_surprise_weight,
        "surprise_usage_weight": args.surprise_usage_weight,
        "surprise_age_weight": args.surprise_age_weight,
        "surprise_route_top_k": args.surprise_route_top_k,
        "surprise_value_layer": args.surprise_value_layer,
        "surprise_warmup_sections": args.surprise_warmup_sections,
    }

    attention_probe_sections = parse_int_csv(args.attention_probe_sections)
    attention_probe_steps = parse_int_csv(args.attention_probe_steps)
    attention_probe_layers = parse_int_csv(args.attention_probe_layers)

    items = load_manifest(args.manifest)
    row_filter = parse_rows(args.rows)
    durations = [int(part) for part in args.durations.split(",")] if args.durations else None
    selected = select_rows(items, row_filter, args.start_row, args.end_row, durations)
    if args.max_items is not None:
        if args.max_items < 1:
            raise ValueError("--max_items must be at least 1")
        selected = selected[: args.max_items]

    if not selected:
        raise RuntimeError("No manifest rows selected.")

    pending = []
    for item in selected:
        row = item["_row"]
        save_path = output_path(output_dir, item)
        attention_audit_path = (
            attention_audit_dir / f"{item['output_prefix']}custom.jsonl"
            if attention_audit_dir is not None
            else None
        )
        audit_is_complete = (
            attention_audit_dir is None
            or jsonl_has_event(attention_audit_path, "attention_audit_complete")
        )
        if save_path.exists() and not args.overwrite and audit_is_complete:
            print(f"[row {row}] skip existing: {save_path}")
            append_jsonl(
                status_path,
                {
                    "row": row,
                    "status": "skipped",
                    "output": str(save_path),
                    "reason": "exists",
                    "time_sec": 0,
                    **policy_metadata,
                },
            )
        else:
            if save_path.exists() and not args.overwrite and not audit_is_complete:
                print(
                    f"[row {row}] video exists but attention audit is incomplete; rerunning"
                )
            pending.append(item)

    if not pending:
        print("No pending rows.")
        return

    assert_video_writer_available(output_dir)

    # Import after CUDA_VISIBLE_DEVICES is set.
    from PIL import Image
    import torch

    from dataset.poses import load_c2ws_from_json
    from diffsynth import save_video
    from inference_memcam import setup_pipeline

    pipe = setup_pipeline(
        dit_path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        text_encoder_path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        vae_path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        dit_ckpt_path="models/MemCam/dit_step20000.ckpt",
        device="cuda",
    )

    print(f"Selected rows: {len(selected)}")
    print(f"Pending rows: {len(pending)}")
    print(f"Requested GPU arg: {args.gpu}")
    print(f"CUDA_VISIBLE_DEVICES: {visible_gpu}")
    print(f"Steps: {num_inference_steps}")
    print(f"Memory policy: {args.memory_policy}, budget: {args.memory_budget}")
    if args.memory_policy == "density_balanced_view_coverage":
        print(
            "Density coverage: "
            f"alpha={args.density_coverage_alpha}, "
            f"DINO weight={args.density_coverage_dino_weight}, "
            f"RGB weight={args.density_coverage_rgb_weight}"
        )
    if args.memory_policy == "future_view_coverage":
        print(
            "Future view coverage: "
            f"alpha={args.future_coverage_alpha}, "
            f"DINO weight={args.future_coverage_dino_weight}, "
            f"RGB weight={args.future_coverage_rgb_weight}, "
            f"query stride={args.future_coverage_query_stride}, "
            f"query weight={args.future_coverage_query_weight}"
        )
    if args.memory_policy == "mce":
        print(
            "MCE: "
            f"alpha={args.mce_alpha}, "
            f"lambda={args.mce_lambda}, "
            f"gamma={args.mce_gamma}, "
            f"query stride={args.mce_query_stride}, "
            f"rarity_neighbors={args.mce_rarity_neighbors}"
        )
    if args.memory_policy == "surprise_forcing":
        print(
            "Surprise Forcing: "
            f"alpha={args.surprise_alpha}, "
            f"EMA={args.surprise_ema_momentum}, "
            f"target write ratio={args.surprise_target_admission_ratio}, "
            f"route top-k={args.surprise_route_top_k}, "
            f"value layer={args.surprise_value_layer}"
        )
    print(f"Memory bank device: {args.memory_bank_device}")
    print(f"Profile dir: {profile_dir}")
    print(f"Attention audit dir: {attention_audit_dir}")
    if attention_audit_dir is not None:
        print(
            "Attention probes: "
            f"sections={attention_probe_sections}, "
            f"steps={attention_probe_steps}, "
            f"layers={attention_probe_layers}, "
            f"queries={args.attention_query_count}"
        )
    print(f"Output dir: {output_dir}")

    for item in pending:
        row = item["_row"]
        save_path = output_path(output_dir, item)
        access_trace_path = access_trace_dir / f"{item['output_prefix']}custom.jsonl"
        profile_path = (
            profile_dir / f"{item['output_prefix']}custom.jsonl"
            if profile_dir is not None
            else None
        )
        attention_audit_path = (
            attention_audit_dir / f"{item['output_prefix']}custom.jsonl"
            if attention_audit_dir is not None
            else None
        )
        print(
            f"[row {row}] {item['scene']} start={item['start_frame']} "
            f"frames={item['num_frames']} duration={item['duration_sec']}s"
        )
        start_time = time.time()
        try:
            input_image = Image.open(item["input_image"]).convert("RGB").resize(
                (args.width, args.height), resample=Image.BICUBIC
            )
            c2ws = load_c2ws_from_json(
                json_path=item["pose_path"],
                start_frame=item["start_frame"],
                num_frames=item["num_frames"],
            )

            video = pipe(
                prompt=item["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                input_image=input_image,
                c2ws=c2ws,
                height=args.height,
                width=args.width,
                cfg_scale=5.0,
                num_inference_steps=num_inference_steps,
                seed=args.seed,
                memory_policy=args.memory_policy,
                memory_budget=args.memory_budget,
                memory_bank_device=args.memory_bank_device,
                density_coverage_alpha=args.density_coverage_alpha,
                density_coverage_dino_weight=args.density_coverage_dino_weight,
                density_coverage_rgb_weight=args.density_coverage_rgb_weight,
                future_coverage_alpha=args.future_coverage_alpha,
                future_coverage_dino_weight=args.future_coverage_dino_weight,
                future_coverage_rgb_weight=args.future_coverage_rgb_weight,
                future_coverage_query_stride=args.future_coverage_query_stride,
                future_coverage_query_weight=args.future_coverage_query_weight,
                mce_alpha=args.mce_alpha,
                mce_lambda=args.mce_lambda,
                mce_gamma=args.mce_gamma,
                mce_query_stride=args.mce_query_stride,
                mce_rarity_neighbors=args.mce_rarity_neighbors,
                surprise_alpha=args.surprise_alpha,
                surprise_ema_momentum=args.surprise_ema_momentum,
                surprise_controller_step=args.surprise_controller_step,
                surprise_target_admission_ratio=(
                    args.surprise_target_admission_ratio
                ),
                surprise_initial_threshold=args.surprise_initial_threshold,
                surprise_surprise_weight=args.surprise_surprise_weight,
                surprise_usage_weight=args.surprise_usage_weight,
                surprise_age_weight=args.surprise_age_weight,
                surprise_route_top_k=args.surprise_route_top_k,
                surprise_value_layer=args.surprise_value_layer,
                surprise_warmup_sections=args.surprise_warmup_sections,
                access_trace_path=str(access_trace_path),
                access_trace_metadata={
                    "row": row,
                    "scene": item["scene"],
                    "dataset_start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "num_frames": item["num_frames"],
                    "output": str(save_path),
                    "output_prefix": item["output_prefix"],
                    "run_memory_policy": args.memory_policy,
                    "run_memory_budget": args.memory_budget,
                    "run_memory_bank_device": args.memory_bank_device,
                    "density_coverage_alpha": args.density_coverage_alpha,
                    "density_coverage_dino_weight": args.density_coverage_dino_weight,
                    "density_coverage_rgb_weight": args.density_coverage_rgb_weight,
                    "future_coverage_alpha": args.future_coverage_alpha,
                    "future_coverage_dino_weight": args.future_coverage_dino_weight,
                    "future_coverage_rgb_weight": args.future_coverage_rgb_weight,
                    "future_coverage_query_stride": args.future_coverage_query_stride,
                    "future_coverage_query_weight": args.future_coverage_query_weight,
                    "mce_alpha": args.mce_alpha,
                    "mce_lambda": args.mce_lambda,
                    "mce_gamma": args.mce_gamma,
                    "mce_query_stride": args.mce_query_stride,
                    "mce_rarity_neighbors": args.mce_rarity_neighbors,
                    "surprise_alpha": args.surprise_alpha,
                    "surprise_ema_momentum": args.surprise_ema_momentum,
                    "surprise_controller_step": args.surprise_controller_step,
                    "surprise_target_admission_ratio": (
                        args.surprise_target_admission_ratio
                    ),
                    "surprise_initial_threshold": args.surprise_initial_threshold,
                    "surprise_surprise_weight": args.surprise_surprise_weight,
                    "surprise_usage_weight": args.surprise_usage_weight,
                    "surprise_age_weight": args.surprise_age_weight,
                    "surprise_route_top_k": args.surprise_route_top_k,
                    "surprise_value_layer": args.surprise_value_layer,
                    "surprise_warmup_sections": args.surprise_warmup_sections,
                },
                attention_audit_path=(
                    None
                    if attention_audit_path is None
                    else str(attention_audit_path)
                ),
                attention_audit_metadata={
                    "row": row,
                    "scene": item["scene"],
                    "dataset_start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "num_frames": item["num_frames"],
                    "output": str(save_path),
                    "output_prefix": item["output_prefix"],
                    "run_memory_policy": args.memory_policy,
                    "run_memory_budget": args.memory_budget,
                    "run_memory_bank_device": args.memory_bank_device,
                },
                attention_probe_sections=attention_probe_sections,
                attention_probe_steps=attention_probe_steps,
                attention_probe_layers=attention_probe_layers,
                attention_query_count=args.attention_query_count,
                attention_query_chunk_size=args.attention_query_chunk_size,
                profile_path=None if profile_path is None else str(profile_path),
                profile_metadata={
                    "run_name": output_dir.name,
                    "row": row,
                    "scene": item["scene"],
                    "dataset_start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "num_frames": item["num_frames"],
                    "fps": item["fps"],
                    "output": str(save_path),
                    "output_prefix": item["output_prefix"],
                },
                tiled=False,
            )
            save_video(video, str(save_path), fps=item["fps"], quality=5)
            del video
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            elapsed = round(time.time() - start_time, 2)
            append_jsonl(
                status_path,
                {
                    "row": row,
                    "status": "failed",
                    "output": str(save_path),
                    "error": repr(exc),
                    "time_sec": elapsed,
                    **policy_metadata,
                },
            )
            raise

        elapsed = round(time.time() - start_time, 2)
        append_jsonl(
            status_path,
            {
                "row": row,
                "status": "completed",
                "output": str(save_path),
                "time_sec": elapsed,
                "num_frames": item["num_frames"],
                "duration_sec": item["duration_sec"],
                "steps": num_inference_steps,
                **policy_metadata,
            },
        )
        print(f"[row {row}] completed in {elapsed}s -> {save_path}")


if __name__ == "__main__":
    main()
