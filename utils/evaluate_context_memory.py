import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


BASE_METRIC_FIELDS = [
    "mae",
    "mse",
    "rmse",
    "psnr_db",
    "ssim",
    "temporal_delta_mae",
    "temporal_delta_rmse",
]
LEARNED_METRIC_FIELDS = {
    "lpips": ["lpips_alex"],
    "dino": ["dino_cosine", "dino_distance"],
    "clip": ["clip_image_cosine", "clip_image_distance"],
}
SUPPORTED_LEARNED_METRICS = tuple(LEARNED_METRIC_FIELDS.keys())


def get_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to read generated MP4 files.") from exc
    return imageio


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


def select_rows(items, row_filter, start_row, end_row, durations, limit):
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
        if duration_filter is not None and int(item["duration_sec"]) not in duration_filter:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break

    return selected


def parse_int_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def output_path(model_output_dir, item):
    return model_output_dir / f"{item['output_prefix']}custom.mp4"


def resolve_gt_frames_dir(item, dataset_root):
    if dataset_root is not None:
        return dataset_root / "frames" / item["scene"]
    return Path(item["gt_frames_dir"])


def read_gt_frame(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, resample=Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def normalize_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def resize_for_metric(frame, image_size):
    if image_size is None:
        return frame
    image = Image.fromarray(frame)
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def rgb_to_luma(frame):
    frame = frame.astype(np.float64)
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def global_ssim(x, y):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.var(x))
    y_var = float(np.var(y))
    xy_cov = float(np.mean((x - x_mean) * (y - y_mean)))

    numerator = (2 * x_mean * y_mean + c1) * (2 * xy_cov + c2)
    denominator = (x_mean**2 + y_mean**2 + c1) * (x_var + y_var + c2)
    return numerator / denominator if denominator else 1.0


def ssim_score(gen_frame, gt_frame):
    x = rgb_to_luma(gen_frame)
    y = rgb_to_luma(gt_frame)

    if cv2 is None or min(x.shape[:2]) < 11:
        return float(global_ssim(x, y))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(x, kernel, sigma)
    mu_y = cv2.GaussianBlur(y, kernel, sigma)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(x * x, kernel, sigma) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y * y, kernel, sigma) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, kernel, sigma) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def frame_metrics(gen_frame, gt_frame):
    gen = gen_frame.astype(np.float64)
    gt = gt_frame.astype(np.float64)
    diff = gen - gt
    abs_error = np.abs(diff)
    sq_error = diff * diff
    mae = float(np.mean(abs_error))
    mse = float(np.mean(sq_error))
    rmse = math.sqrt(mse)
    psnr_db = 100.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / rmse)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr_db": psnr_db,
        "ssim": ssim_score(gen_frame, gt_frame),
    }


def temporal_delta_metrics(gen_frame, gt_frame, prev_gen_frame, prev_gt_frame):
    gen_delta = gen_frame.astype(np.float64) - prev_gen_frame.astype(np.float64)
    gt_delta = gt_frame.astype(np.float64) - prev_gt_frame.astype(np.float64)
    diff = gen_delta - gt_delta
    mse = float(np.mean(diff * diff))
    return {
        "temporal_delta_mae": float(np.mean(np.abs(diff))),
        "temporal_delta_rmse": math.sqrt(mse),
    }


def parse_learned_metrics(value):
    if value is None or value.lower() in {"", "none"}:
        return []
    metrics = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(metrics) - set(SUPPORTED_LEARNED_METRICS))
    if unknown:
        raise ValueError(
            f"Unsupported learned metrics {unknown}. "
            f"Expected comma-separated values from {SUPPORTED_LEARNED_METRICS}."
        )
    return metrics


class LearnedMetricRunner:
    def __init__(self, metric_names, device="cuda", batch_size=8, image_size=224):
        self.metric_names = list(metric_names)
        self.batch_size = batch_size
        self.image_size = image_size
        self.fields = [
            field
            for metric_name in self.metric_names
            for field in LEARNED_METRIC_FIELDS[metric_name]
        ]
        self.torch = None
        self.device = None
        self.lpips_model = None
        self.dino_processor = None
        self.dino_model = None
        self.clip_processor = None
        self.clip_model = None

        if self.metric_names:
            self._setup(device)

    def _setup(self, requested_device):
        import torch

        self.torch = torch
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for metrics but unavailable; using CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        if "lpips" in self.metric_names:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError(
                    "LPIPS metric requested but package 'lpips' is not installed. "
                    "Install it in the memcam env with: pip install lpips"
                ) from exc
            self.lpips_model = lpips.LPIPS(net="alex").eval().to(self.device)

        if "dino" in self.metric_names:
            from transformers import AutoImageProcessor, AutoModel

            self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(self.device)

        if "clip" in self.metric_names:
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(self.device)

    def _pil_batch(self, frames):
        return [Image.fromarray(resize_for_metric(frame, self.image_size)) for frame in frames]

    def _lpips_tensor(self, frames):
        arrays = [resize_for_metric(frame, self.image_size) for frame in frames]
        tensor = self.torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()
        tensor = tensor / 127.5 - 1.0
        return tensor.to(self.device)

    def _encode_vision_batch(self, processor, model, frames):
        inputs = processor(images=self._pil_batch(frames), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = model(**inputs)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0]
        return self.torch.nn.functional.normalize(features.float(), dim=-1)

    def compute_batch(self, gen_frames, gt_frames):
        if not self.metric_names:
            return [{} for _ in gen_frames]

        results = [{} for _ in gen_frames]
        with self.torch.inference_mode():
            if "lpips" in self.metric_names:
                gen_tensor = self._lpips_tensor(gen_frames)
                gt_tensor = self._lpips_tensor(gt_frames)
                values = self.lpips_model(gen_tensor, gt_tensor).flatten().detach().cpu().numpy()
                for result, value in zip(results, values):
                    result["lpips_alex"] = float(value)

            if "dino" in self.metric_names:
                gen_features = self._encode_vision_batch(
                    self.dino_processor,
                    self.dino_model,
                    gen_frames,
                )
                gt_features = self._encode_vision_batch(
                    self.dino_processor,
                    self.dino_model,
                    gt_frames,
                )
                cosines = (gen_features * gt_features).sum(dim=-1).detach().cpu().numpy()
                for result, cosine in zip(results, cosines):
                    result["dino_cosine"] = float(cosine)
                    result["dino_distance"] = float(1.0 - cosine)

            if "clip" in self.metric_names:
                gen_features = self._encode_vision_batch(
                    self.clip_processor,
                    self.clip_model,
                    gen_frames,
                )
                gt_features = self._encode_vision_batch(
                    self.clip_processor,
                    self.clip_model,
                    gt_frames,
                )
                cosines = (gen_features * gt_features).sum(dim=-1).detach().cpu().numpy()
                for result, cosine in zip(results, cosines):
                    result["clip_image_cosine"] = float(cosine)
                    result["clip_image_distance"] = float(1.0 - cosine)

        return results


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def evaluate_video(
    item,
    model_output_dir,
    dataset_root,
    frame_stride,
    max_frames,
    frame_metrics_handle,
    learned_runner,
    metric_fields,
):
    row = item["_row"]
    video_path = output_path(model_output_dir, item)
    if not video_path.exists():
        return {
            "row": row,
            "status": "missing_output",
            "output": str(video_path),
            "scene": item["scene"],
            "start_frame": item["start_frame"],
            "duration_sec": item["duration_sec"],
            "num_frames_expected": item["num_frames"],
            "frames_evaluated": 0,
        }

    gt_frames_dir = resolve_gt_frames_dir(item, dataset_root)
    expected_frames = int(item["num_frames"])
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames)

    frame_values = {field: [] for field in metric_fields}
    pending_learned = []
    prev_gen_frame = None
    prev_gt_frame = None
    frames_seen = 0
    frames_evaluated = 0
    size = None

    imageio = get_imageio()
    reader = imageio.get_reader(str(video_path))

    def finalize_metrics(metrics, payload):
        for field in metric_fields:
            frame_values[field].append(metrics.get(field))

        if frame_metrics_handle is not None:
            payload = {**payload, **metrics}
            frame_metrics_handle.write(json.dumps(payload) + "\n")

    def flush_learned():
        if not pending_learned:
            return
        gen_frames = [item["gen_frame"] for item in pending_learned]
        gt_frames = [item["gt_frame"] for item in pending_learned]
        learned_results = learned_runner.compute_batch(gen_frames, gt_frames)
        for pending_item, learned_metrics in zip(pending_learned, learned_results):
            pending_item["metrics"].update(learned_metrics)
            finalize_metrics(pending_item["metrics"], pending_item["payload"])
        pending_learned.clear()

    try:
        for frame_index, gen_frame in enumerate(reader):
            if frame_index >= expected_frames:
                break
            frames_seen += 1
            if frame_index % frame_stride != 0:
                continue

            gen_frame = normalize_video_frame(gen_frame)
            height, width = gen_frame.shape[:2]
            size = (width, height)
            gt_index = int(item["start_frame"]) + frame_index
            gt_path = gt_frames_dir / f"{gt_index:04d}.png"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing ground-truth frame: {gt_path}")

            gt_frame = read_gt_frame(gt_path, size)
            metrics = frame_metrics(gen_frame, gt_frame)
            if prev_gen_frame is not None and prev_gt_frame is not None:
                metrics.update(
                    temporal_delta_metrics(gen_frame, gt_frame, prev_gen_frame, prev_gt_frame)
                )
            else:
                metrics["temporal_delta_mae"] = None
                metrics["temporal_delta_rmse"] = None

            payload = {
                "row": row,
                "scene": item["scene"],
                "duration_sec": item["duration_sec"],
                "frame_index": frame_index,
                "gt_frame_index": gt_index,
            }

            if learned_runner is None:
                finalize_metrics(metrics, payload)
            else:
                pending_learned.append(
                    {
                        "metrics": metrics,
                        "payload": payload,
                        "gen_frame": gen_frame,
                        "gt_frame": gt_frame,
                    }
                )
                if len(pending_learned) >= learned_runner.batch_size:
                    flush_learned()

            prev_gen_frame = gen_frame
            prev_gt_frame = gt_frame
            frames_evaluated += 1

        flush_learned()
    finally:
        reader.close()

    if frames_evaluated == 0:
        status = "no_frames_evaluated"
    elif frames_seen < expected_frames:
        status = "short_video"
    else:
        status = "completed"

    result = {
        "row": row,
        "status": status,
        "output": str(video_path),
        "scene": item["scene"],
        "start_frame": item["start_frame"],
        "duration_sec": item["duration_sec"],
        "caption_key": item.get("caption_key"),
        "num_frames_expected": item["num_frames"],
        "frames_seen": frames_seen,
        "frames_evaluated": frames_evaluated,
        "frame_stride": frame_stride,
        "width": size[0] if size else None,
        "height": size[1] if size else None,
    }
    for field in metric_fields:
        result[field] = mean_or_none(frame_values[field])
    return result


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows, metric_fields):
    fieldnames = [
        "run_name",
        "row",
        "status",
        "scene",
        "start_frame",
        "duration_sec",
        "num_frames_expected",
        "frames_seen",
        "frames_evaluated",
        "frame_stride",
        "width",
        "height",
        *metric_fields,
        "output",
        "caption_key",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_group(rows, metric_fields):
    completed = [row for row in rows if row["status"] in {"completed", "short_video"}]
    summary = {
        "videos": len(rows),
        "completed_or_short": len(completed),
        "missing_outputs": sum(row["status"] == "missing_output" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "frames_evaluated": int(sum(row.get("frames_evaluated") or 0 for row in completed)),
    }
    for field in metric_fields:
        summary[field] = mean_or_none([row.get(field) for row in completed])
    return summary


def build_summary(rows, metric_fields):
    by_duration = {}
    for row in rows:
        duration = str(row["duration_sec"])
        by_duration.setdefault(duration, []).append(row)

    return {
        "overall": summarize_group(rows, metric_fields),
        "by_duration": {
            duration: summarize_group(duration_rows, metric_fields)
            for duration, duration_rows in sorted(by_duration.items(), key=lambda item: int(item[0]))
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Context-as-Memory generated videos against dataset frames."
    )
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument(
        "--model_output_dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing generated MP4s from run_context_memory_batch.py.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Optional dataset root override. Useful when a manifest was moved across clusters.",
    )
    parser.add_argument("--metrics_dir", type=Path, default=Path("eval/context_memory"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None, help="Rows like '0,2,5-9'.")
    parser.add_argument("--start_row", type=int, default=None)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--durations", type=str, default=None, help="Optional durations like '10,20'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--learned_metrics",
        type=str,
        default="none",
        help="Optional comma list: lpips,dino,clip. Default: none.",
    )
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--learned_image_size", type=int, default=224)
    parser.add_argument("--write_frame_metrics", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if args.metric_batch_size < 1:
        raise ValueError("--metric_batch_size must be >= 1")

    learned_metric_names = parse_learned_metrics(args.learned_metrics)
    learned_runner = None
    if learned_metric_names:
        learned_runner = LearnedMetricRunner(
            metric_names=learned_metric_names,
            device=args.metric_device,
            batch_size=args.metric_batch_size,
            image_size=args.learned_image_size,
        )
    metric_fields = BASE_METRIC_FIELDS + (
        learned_runner.fields if learned_runner is not None else []
    )

    run_name = args.run_name or args.model_output_dir.name
    metrics_dir = args.metrics_dir / run_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.manifest)
    row_filter = parse_rows(args.rows)
    durations = parse_int_list(args.durations)
    selected = select_rows(
        items=items,
        row_filter=row_filter,
        start_row=args.start_row,
        end_row=args.end_row,
        durations=durations,
        limit=args.limit,
    )
    if not selected:
        raise RuntimeError("No manifest rows selected.")

    results = []
    frame_metrics_path = metrics_dir / "frame_metrics.jsonl"
    frame_metrics_handle = None
    if args.write_frame_metrics:
        frame_metrics_handle = frame_metrics_path.open("w", encoding="utf-8")

    try:
        for item in selected:
            row = item["_row"]
            print(
                f"[eval row {row}] {item['scene']} start={item['start_frame']} "
                f"duration={item['duration_sec']}s"
            )
            try:
                result = evaluate_video(
                    item=item,
                    model_output_dir=args.model_output_dir,
                    dataset_root=args.dataset_root,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    frame_metrics_handle=frame_metrics_handle,
                    learned_runner=learned_runner,
                    metric_fields=metric_fields,
                )
            except Exception as exc:
                if args.strict:
                    raise
                result = {
                    "row": row,
                    "status": "failed",
                    "error": repr(exc),
                    "output": str(output_path(args.model_output_dir, item)),
                    "scene": item["scene"],
                    "start_frame": item["start_frame"],
                    "duration_sec": item["duration_sec"],
                    "num_frames_expected": item["num_frames"],
                    "frames_evaluated": 0,
                }

            result["run_name"] = run_name
            results.append(result)
            status = result["status"]
            metric_text = ""
            if result.get("psnr_db") is not None:
                metric_text = (
                    f" psnr={result['psnr_db']:.3f} "
                    f"ssim={result['ssim']:.4f} frames={result['frames_evaluated']}"
                )
                if result.get("dino_cosine") is not None:
                    metric_text += f" dino={result['dino_cosine']:.4f}"
                if result.get("lpips_alex") is not None:
                    metric_text += f" lpips={result['lpips_alex']:.4f}"
            print(f"[eval row {row}] {status}{metric_text}")
    finally:
        if frame_metrics_handle is not None:
            frame_metrics_handle.close()

    metrics_jsonl = metrics_dir / "metrics.jsonl"
    metrics_csv = metrics_dir / "metrics.csv"
    summary_json = metrics_dir / "summary.json"

    write_jsonl(metrics_jsonl, results)
    write_csv(metrics_csv, results, metric_fields)
    summary = build_summary(results, metric_fields)
    summary["metric_config"] = {
        "learned_metrics": learned_metric_names,
        "metric_device": args.metric_device,
        "metric_batch_size": args.metric_batch_size,
        "learned_image_size": args.learned_image_size,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote metrics: {metrics_jsonl}")
    print(f"Wrote CSV: {metrics_csv}")
    print(f"Wrote summary: {summary_json}")
    if args.write_frame_metrics:
        print(f"Wrote frame metrics: {frame_metrics_path}")


if __name__ == "__main__":
    main()
