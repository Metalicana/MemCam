import argparse
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
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    num_inference_steps = 20 if args.smoke else args.num_inference_steps
    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.jsonl"

    items = load_manifest(args.manifest)
    row_filter = parse_rows(args.rows)
    durations = [int(part) for part in args.durations.split(",")] if args.durations else None
    selected = select_rows(items, row_filter, args.start_row, args.end_row, durations)

    if not selected:
        raise RuntimeError("No manifest rows selected.")

    pending = []
    for item in selected:
        row = item["_row"]
        save_path = output_path(output_dir, item)
        if save_path.exists() and not args.overwrite:
            print(f"[row {row}] skip existing: {save_path}")
            append_jsonl(
                status_path,
                {
                    "row": row,
                    "status": "skipped",
                    "output": str(save_path),
                    "reason": "exists",
                    "time_sec": 0,
                },
            )
        else:
            pending.append(item)

    if not pending:
        print("No pending rows.")
        return

    # Import after CUDA_VISIBLE_DEVICES is set.
    from PIL import Image

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
    print(f"GPU: {args.gpu}")
    print(f"Steps: {num_inference_steps}")
    print(f"Output dir: {output_dir}")

    for item in pending:
        row = item["_row"]
        save_path = output_path(output_dir, item)
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
                tiled=False,
            )
            save_video(video, str(save_path), fps=item["fps"], quality=5)
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
            },
        )
        print(f"[row {row}] completed in {elapsed}s -> {save_path}")


if __name__ == "__main__":
    main()
