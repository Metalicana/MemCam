import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def item_output_dir(output_dir, run_name, item):
    return output_dir / run_name / item["output_prefix"].rstrip("_")


def map_video_index_to_manifest_local(video_idx, total_video_frames, manifest_num_frames):
    if total_video_frames <= 1 or manifest_num_frames <= 1:
        return min(video_idx, max(0, manifest_num_frames - 1))
    scaled = round(video_idx * (manifest_num_frames - 1) / (total_video_frames - 1))
    return int(min(max(scaled, 0), manifest_num_frames - 1))


def extract_sampled_frames(video_path, frame_dir, frame_stride, max_frames, manifest_num_frames):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to extract video frames for CUT3R.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = list(range(0, total_frames, frame_stride))
    if max_frames is not None and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = [indices[int(round(pos))] for pos in positions]
        indices = sorted(set(indices))

    frame_paths = []
    manifest_local_indices = []
    frame_dir.mkdir(parents=True, exist_ok=True)

    for output_idx, video_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = frame_dir / f"{output_idx:06d}_video{video_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        manifest_local_indices.append(
            map_video_index_to_manifest_local(
                video_idx=video_idx,
                total_video_frames=total_frames,
                manifest_num_frames=manifest_num_frames,
            )
        )

    cap.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return {
        "frame_paths": frame_paths,
        "video_frame_indices": indices[: len(frame_paths)],
        "manifest_local_indices": manifest_local_indices,
        "total_video_frames": total_frames,
        "video_fps": fps,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def has_completed_cut3r_output(run_output_dir):
    metadata_path = run_output_dir / "metadata.json"
    camera_dir = run_output_dir / "camera"
    if not metadata_path.exists() or not camera_dir.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = len(metadata.get("video_frame_indices", []))
    actual = len(list(camera_dir.glob("*.npz")))
    return expected > 0 and actual >= expected


def main():
    parser = argparse.ArgumentParser(
        description="Run CUT3R pose reconstruction on Context-as-Memory generated videos."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--root", type=Path, required=True, help="Root containing run dirs like baseline/fifo_b64.")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run names.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--cut3r_root", type=Path, default=REPO_ROOT / "CUT3R")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--durations", type=str, default="60")
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if not args.model_path.exists():
        raise FileNotFoundError(f"CUT3R checkpoint not found: {args.model_path}")
    if not args.cut3r_root.exists():
        raise FileNotFoundError(f"CUT3R root not found: {args.cut3r_root}")

    output_dir = args.output_dir or (args.root / "cut3r_pose_recon")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CUT3R expects imports relative to its repository root and checkpoint dir.
    sys.path.insert(0, str(args.cut3r_root.resolve()))
    sys.path.insert(0, str((args.cut3r_root / "src" / "croco").resolve()))
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(str(args.model_path.resolve()))
    if args.device == "cuda":
        try:
            from models.curope import cuRoPE2D  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "CUT3R CUDA inference requires the compiled RoPE2D extension. "
                "Compile it on the GPU node with:\n"
                "  cd $HOME/MemCam/CUT3R/src/croco/models/curope\n"
                "  python setup.py build_ext --inplace\n"
                "Then rerun the CUT3R smoke job. Without this extension CUT3R falls "
                "back to a slow PyTorch RoPE path that crashes on CUT3R's pose-token "
                "position sentinel."
            ) from exc
    from demo import prepare_input, prepare_output
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUT3R was requested with --device cuda, but torch.cuda.is_available() is false. "
            "Run inside an allocated GPU job/srun, check nvidia-smi, and verify PyTorch can see CUDA "
            "before starting CUT3R."
        )

    print(f"Loading CUT3R model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    model.eval()

    items = select_rows(
        load_manifest(args.manifest),
        row_filter=parse_rows(args.rows),
        durations=parse_int_list(args.durations),
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("No manifest rows selected.")

    runs = parse_list(args.runs)
    status_rows = []

    for run_name in runs:
        run_dir = args.root / run_name
        for item in items:
            video_path = output_path(run_dir, item)
            run_output_dir = item_output_dir(output_dir, run_name, item)
            metadata_path = run_output_dir / "metadata.json"

            if not video_path.exists():
                status_rows.append(
                    {
                        "run_name": run_name,
                        "row": item["_row"],
                        "status": "missing_video",
                        "video_path": str(video_path),
                    }
                )
                print(f"[missing] {run_name} row={item['_row']} {video_path}")
                continue

            if not args.force and has_completed_cut3r_output(run_output_dir):
                status_rows.append(
                    {
                        "run_name": run_name,
                        "row": item["_row"],
                        "status": "skipped_existing",
                        "output_dir": str(run_output_dir),
                    }
                )
                print(f"[skip] {run_name} row={item['_row']} {run_output_dir}")
                continue

            if run_output_dir.exists() and args.force:
                shutil.rmtree(run_output_dir)
            run_output_dir.mkdir(parents=True, exist_ok=True)

            temp_dir = Path(tempfile.mkdtemp(prefix="memcam_cut3r_frames_"))
            start_time = time.time()
            try:
                sampled = extract_sampled_frames(
                    video_path=video_path,
                    frame_dir=temp_dir,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    manifest_num_frames=int(item["num_frames"]),
                )

                print(
                    f"[CUT3R] run={run_name} row={item['_row']} scene={item['scene']} "
                    f"frames={len(sampled['frame_paths'])} video={video_path.name}"
                )
                views = prepare_input(
                    img_paths=sampled["frame_paths"],
                    img_mask=[True] * len(sampled["frame_paths"]),
                    size=args.size,
                    revisit=1,
                    update=True,
                )
                with torch.inference_mode():
                    outputs, _state_args = inference(views, model, device)
                    prepare_output(outputs, str(run_output_dir), revisit=1, use_pose=True)

                elapsed = round(time.time() - start_time, 3)
                metadata = {
                    "status": "completed",
                    "run_name": run_name,
                    "manifest_row": item["_row"],
                    "scene": item["scene"],
                    "start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "num_frames": item["num_frames"],
                    "pose_path": item["pose_path"],
                    "output_prefix": item["output_prefix"],
                    "video_path": str(video_path),
                    "cut3r_model_path": str(args.model_path),
                    "cut3r_size": args.size,
                    "frame_stride": args.frame_stride,
                    "max_frames": args.max_frames,
                    "video_frame_indices": sampled["video_frame_indices"],
                    "manifest_local_indices": sampled["manifest_local_indices"],
                    "dataset_frame_indices": [
                        int(item["start_frame"]) + int(idx)
                        for idx in sampled["manifest_local_indices"]
                    ],
                    "total_video_frames": sampled["total_video_frames"],
                    "video_fps": sampled["video_fps"],
                    "time_sec": elapsed,
                }
                write_json(metadata_path, metadata)
                status_rows.append(
                    {
                        "run_name": run_name,
                        "row": item["_row"],
                        "status": "completed",
                        "output_dir": str(run_output_dir),
                        "time_sec": elapsed,
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "run_name": run_name,
                        "row": item["_row"],
                        "status": "failed",
                        "output_dir": str(run_output_dir),
                        "error": repr(exc),
                    }
                )
                write_json(
                    metadata_path,
                    {
                        "status": "failed",
                        "run_name": run_name,
                        "manifest_row": item["_row"],
                        "scene": item.get("scene"),
                        "video_path": str(video_path),
                        "error": repr(exc),
                    },
                )
                raise
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

    write_json(output_dir / "cut3r_run_status.json", status_rows)
    print(f"Wrote: {output_dir / 'cut3r_run_status.json'}")


if __name__ == "__main__":
    main()
