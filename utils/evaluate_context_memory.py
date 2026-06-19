import argparse
import csv
import json
import math
import os
import tempfile
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
VIDEO_DISTRIBUTION_METRICS = {"fvd"}
SUPPORTED_LEARNED_METRICS = tuple(LEARNED_METRIC_FIELDS.keys()) + tuple(
    sorted(VIDEO_DISTRIBUTION_METRICS)
)
FVD_I3D_DETECTOR_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
FVD_I3D_BACKENDS = {"styleganv_i3d", "i3d_torchscript"}


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
        if size is not None and image.size != size:
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


class FVDRunner:
    def __init__(
        self,
        device="cuda",
        batch_size=4,
        image_size=224,
        clip_length=16,
        clips_per_video=4,
        frame_stride=1,
        backend="styleganv_i3d",
        detector_path=None,
        detector_url=FVD_I3D_DETECTOR_URL,
        cache_dir=None,
        allow_download=True,
        pca_dim=None,
        eps=1e-6,
    ):
        import torch

        backend = "styleganv_i3d" if backend == "i3d_torchscript" else backend
        if backend not in FVD_I3D_BACKENDS and backend != "torchvision_r3d18":
            raise ValueError(f"Unsupported FVD backend: {backend}")

        if clip_length < 2:
            raise ValueError("--fvd_clip_length must be >= 2")
        if clips_per_video < 1:
            raise ValueError("--fvd_clips_per_video must be >= 1")
        if frame_stride < 1:
            raise ValueError("--fvd_frame_stride must be >= 1")
        if image_size < 1:
            raise ValueError("--fvd_image_size must be >= 1")
        if pca_dim is not None and backend in FVD_I3D_BACKENDS:
            print("Warning: --fvd_pca_dim is ignored for canonical I3D FVD.")

        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for FVD but unavailable; using CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.image_size = image_size
        self.clip_length = clip_length
        self.clips_per_video = clips_per_video
        self.frame_stride = frame_stride
        self.backend = backend
        self.detector_path = Path(detector_path).expanduser() if detector_path else None
        self.detector_url = detector_url
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else self._default_cache_dir()
        self.allow_download = allow_download
        self.pca_dim = pca_dim
        self.eps = eps
        self.resolved_detector_path = None
        self.feature_model = self._build_feature_model()
        self.r3d_mean = torch.tensor(
            [0.43216, 0.394666, 0.37645],
            device=self.device,
        ).view(1, 3, 1, 1, 1)
        self.r3d_std = torch.tensor(
            [0.22803, 0.22145, 0.216989],
            device=self.device,
        ).view(1, 3, 1, 1, 1)

    def _default_cache_dir(self):
        cache_root = os.environ.get("XDG_CACHE_HOME")
        if cache_root:
            return Path(cache_root) / "memcam"
        return Path.home() / ".cache" / "memcam"

    def _build_feature_model(self):
        if self.backend in FVD_I3D_BACKENDS:
            detector_path = self._resolve_i3d_detector_path()
            self.resolved_detector_path = detector_path
            model = self.torch.jit.load(str(detector_path), map_location=self.device)
            return model.eval().to(self.device)

        from torchvision.models.video import R3D_18_Weights, r3d_18

        print(
            "Warning: torchvision_r3d18 is a nonstandard FVD proxy. "
            "Use --fvd_backend styleganv_i3d for canonical I3D FVD."
        )
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = self.torch.nn.Identity()
        return model.eval().to(self.device)

    def _resolve_i3d_detector_path(self):
        if self.detector_path is not None:
            if not self.detector_path.exists():
                raise FileNotFoundError(f"FVD detector not found: {self.detector_path}")
            return self.detector_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        detector_path = self.cache_dir / "i3d_torchscript.pt"
        if detector_path.exists():
            return detector_path

        if not self.allow_download:
            raise FileNotFoundError(
                "FVD I3D detector is not cached. Pass --fvd_detector_path or allow "
                f"download from {self.detector_url}"
            )

        print(f"Downloading FVD I3D detector to {detector_path}")
        self._download_file(self.detector_url, detector_path)
        return detector_path

    def _download_file(self, url, destination):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Downloading the FVD detector requires the 'requests' package. "
                f"Install requests or pass --fvd_detector_path. URL: {url}"
            ) from exc

        destination = Path(destination)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp_path, destination)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the FVD I3D detector. "
                f"Download it manually from {url} and pass --fvd_detector_path."
            ) from exc

    def _sample_starts(self, num_frames):
        span = (self.clip_length - 1) * self.frame_stride + 1
        if num_frames < span:
            return []
        max_start = num_frames - span
        if self.clips_per_video == 1:
            return [max_start // 2]
        starts = np.linspace(0, max_start, self.clips_per_video)
        return sorted({int(round(start)) for start in starts})

    def _to_clip_frame(self, frame):
        frame = normalize_video_frame(frame)
        return np.transpose(frame, (2, 0, 1))

    def _read_generated_frames(self, video_path, indices):
        imageio = get_imageio()
        reader = imageio.get_reader(str(video_path))
        wanted = set(indices)
        frames = {}
        last_index = max(wanted)
        try:
            for frame_index, frame in enumerate(reader):
                if frame_index > last_index:
                    break
                if frame_index in wanted:
                    frames[frame_index] = self._to_clip_frame(frame)
        finally:
            reader.close()
        return frames

    def _read_gt_frames(self, item, dataset_root, indices):
        gt_frames_dir = resolve_gt_frames_dir(item, dataset_root)
        frames = {}
        for frame_index in indices:
            gt_index = int(item["start_frame"]) + frame_index
            gt_path = gt_frames_dir / f"{gt_index:04d}.png"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing ground-truth frame: {gt_path}")
            frames[frame_index] = self._to_clip_frame(read_gt_frame(gt_path, None))
        return frames

    def _load_item_clips(self, item, model_output_dir, dataset_root, max_frames):
        video_path = output_path(model_output_dir, item)
        if not video_path.exists():
            return [], []

        num_frames = int(item["num_frames"])
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        starts = self._sample_starts(num_frames)
        if not starts:
            return [], []

        clip_indices = [
            [start + offset * self.frame_stride for offset in range(self.clip_length)]
            for start in starts
        ]
        flat_indices = sorted({index for indices in clip_indices for index in indices})
        gen_frames = self._read_generated_frames(video_path, flat_indices)
        gt_frames = self._read_gt_frames(item, dataset_root, flat_indices)

        gen_clips = []
        gt_clips = []
        for indices in clip_indices:
            if all(index in gen_frames for index in indices):
                gen_clips.append(np.stack([gen_frames[index] for index in indices]))
                gt_clips.append(np.stack([gt_frames[index] for index in indices]))
        return gen_clips, gt_clips

    def _clip_tensor(self, clips):
        tensors = []
        for clip in clips:
            tensor = self.torch.from_numpy(np.asarray(clip)).float().to(self.device)
            tensor = tensor.permute(1, 0, 2, 3).contiguous()
            frames = tensor.permute(1, 0, 2, 3)
            frames = self.torch.nn.functional.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(frames.permute(1, 0, 2, 3))

        tensor = self.torch.stack(tensors)
        if self.backend == "torchvision_r3d18":
            tensor = tensor / 255.0
            return (tensor - self.r3d_mean) / self.r3d_std
        return tensor

    def _encode_batch(self, clips):
        tensor = self._clip_tensor(clips)
        with self.torch.inference_mode():
            if self.backend in FVD_I3D_BACKENDS:
                features = self.feature_model(
                    tensor,
                    rescale=True,
                    resize=False,
                    return_features=True,
                )
            else:
                features = self.feature_model(tensor)
        return features.float().detach().cpu().numpy()

    def _append_features(self, feature_rows, clips):
        for start in range(0, len(clips), self.batch_size):
            feature_rows.append(self._encode_batch(clips[start : start + self.batch_size]))

    def _frechet_distance(self, real_features, generated_features):
        real_features = np.asarray(real_features, dtype=np.float64)
        generated_features = np.asarray(generated_features, dtype=np.float64)
        real_mu = np.mean(real_features, axis=0)
        generated_mu = np.mean(generated_features, axis=0)
        real_sigma = np.atleast_2d(np.cov(real_features, rowvar=False))
        generated_sigma = np.atleast_2d(np.cov(generated_features, rowvar=False))
        real_sigma = self._symmetric_matrix(real_sigma)
        generated_sigma = self._symmetric_matrix(generated_sigma)

        diff = real_mu - generated_mu
        trace_covmean = self._trace_sqrt_product(real_sigma, generated_sigma)
        value = diff.dot(diff) + np.trace(real_sigma) + np.trace(generated_sigma)
        value = value - 2.0 * trace_covmean
        if value < 0.0:
            return 0.0
        return float(value)

    @staticmethod
    def _symmetric_matrix(matrix):
        return 0.5 * (matrix + matrix.T)

    def _trace_sqrt_product(self, sigma_a, sigma_b):
        sigma_a = self._symmetric_matrix(sigma_a)
        sigma_b = self._symmetric_matrix(sigma_b)
        eigvals, eigvecs = np.linalg.eigh(sigma_a)
        eigvals = np.clip(eigvals, 0.0, None)
        sigma_a_sqrt = (eigvecs * np.sqrt(eigvals)).dot(eigvecs.T)
        product = sigma_a_sqrt.dot(sigma_b).dot(sigma_a_sqrt)
        product = self._symmetric_matrix(product)
        product_eigvals = np.linalg.eigvalsh(product)
        return float(np.sum(np.sqrt(np.clip(product_eigvals, 0.0, None))))

    def compute_group(self, items, model_output_dir, dataset_root, max_frames=None):
        gen_batch = []
        gt_batch = []
        gen_features = []
        gt_features = []
        clip_count = 0

        for item in items:
            gen_clips, gt_clips = self._load_item_clips(
                item=item,
                model_output_dir=model_output_dir,
                dataset_root=dataset_root,
                max_frames=max_frames,
            )
            for gen_clip, gt_clip in zip(gen_clips, gt_clips):
                gen_batch.append(gen_clip)
                gt_batch.append(gt_clip)
                clip_count += 1
                if len(gen_batch) >= self.batch_size:
                    self._append_features(gen_features, gen_batch)
                    self._append_features(gt_features, gt_batch)
                    gen_batch.clear()
                    gt_batch.clear()

        if gen_batch:
            self._append_features(gen_features, gen_batch)
            self._append_features(gt_features, gt_batch)

        if clip_count < 2:
            return None, clip_count
        value = self._frechet_distance(np.concatenate(gt_features), np.concatenate(gen_features))
        return value, clip_count


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
        help="Optional comma list: lpips,dino,clip,fvd. Default: none.",
    )
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--learned_image_size", type=int, default=224)
    parser.add_argument("--fvd_clip_length", type=int, default=16)
    parser.add_argument("--fvd_clips_per_video", type=int, default=4)
    parser.add_argument("--fvd_frame_stride", type=int, default=1)
    parser.add_argument("--fvd_image_size", type=int, default=224)
    parser.add_argument(
        "--fvd_pca_dim",
        type=int,
        default=None,
        help="Deprecated compatibility flag. Canonical I3D FVD does not apply PCA.",
    )
    parser.add_argument("--fvd_eps", type=float, default=1e-6)
    parser.add_argument(
        "--fvd_backend",
        type=str,
        default="styleganv_i3d",
        choices=["styleganv_i3d", "i3d_torchscript", "torchvision_r3d18"],
    )
    parser.add_argument(
        "--fvd_detector_path",
        type=Path,
        default=None,
        help="Optional local i3d_torchscript.pt path for canonical FVD.",
    )
    parser.add_argument("--fvd_detector_url", type=str, default=FVD_I3D_DETECTOR_URL)
    parser.add_argument(
        "--fvd_cache_dir",
        type=Path,
        default=None,
        help="Directory used to cache the I3D detector. Defaults to ~/.cache/memcam.",
    )
    parser.add_argument(
        "--no_fvd_download",
        action="store_false",
        dest="fvd_allow_download",
        help="Do not download the I3D detector automatically.",
    )
    parser.set_defaults(fvd_allow_download=True)
    parser.add_argument("--write_frame_metrics", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if args.metric_batch_size < 1:
        raise ValueError("--metric_batch_size must be >= 1")

    learned_metric_names = parse_learned_metrics(args.learned_metrics)
    fvd_requested = "fvd" in learned_metric_names
    frame_learned_metric_names = [
        metric_name for metric_name in learned_metric_names if metric_name != "fvd"
    ]
    learned_runner = None
    if frame_learned_metric_names:
        learned_runner = LearnedMetricRunner(
            metric_names=frame_learned_metric_names,
            device=args.metric_device,
            batch_size=args.metric_batch_size,
            image_size=args.learned_image_size,
        )
    fvd_runner = None
    if fvd_requested:
        fvd_runner = FVDRunner(
            device=args.metric_device,
            batch_size=args.metric_batch_size,
            image_size=args.fvd_image_size,
            clip_length=args.fvd_clip_length,
            clips_per_video=args.fvd_clips_per_video,
            frame_stride=args.fvd_frame_stride,
            backend=args.fvd_backend,
            detector_path=args.fvd_detector_path,
            detector_url=args.fvd_detector_url,
            cache_dir=args.fvd_cache_dir,
            allow_download=args.fvd_allow_download,
            pca_dim=args.fvd_pca_dim,
            eps=args.fvd_eps,
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
    if fvd_runner is not None:
        completed_rows = {
            row["row"]
            for row in results
            if row["status"] in {"completed", "short_video"}
        }
        completed_items = [item for item in selected if item["_row"] in completed_rows]
        print("Computing FVD over completed videos...")
        fvd_value, fvd_clips = fvd_runner.compute_group(
            items=completed_items,
            model_output_dir=args.model_output_dir,
            dataset_root=args.dataset_root,
            max_frames=args.max_frames,
        )
        summary["overall"]["fvd"] = fvd_value
        summary["overall"]["fvd_clips"] = fvd_clips
        summary["overall"]["fvd_backend"] = fvd_runner.backend
        summary["overall"]["fvd_detector_path"] = (
            str(fvd_runner.resolved_detector_path)
            if fvd_runner.resolved_detector_path is not None
            else None
        )

        for duration, duration_summary in summary["by_duration"].items():
            duration_items = [
                item
                for item in completed_items
                if int(item["duration_sec"]) == int(duration)
            ]
            duration_fvd, duration_fvd_clips = fvd_runner.compute_group(
                items=duration_items,
                model_output_dir=args.model_output_dir,
                dataset_root=args.dataset_root,
                max_frames=args.max_frames,
            )
            duration_summary["fvd"] = duration_fvd
            duration_summary["fvd_clips"] = duration_fvd_clips
            duration_summary["fvd_backend"] = fvd_runner.backend
            duration_summary["fvd_detector_path"] = (
                str(fvd_runner.resolved_detector_path)
                if fvd_runner.resolved_detector_path is not None
                else None
            )

    summary["metric_config"] = {
        "learned_metrics": learned_metric_names,
        "frame_learned_metrics": frame_learned_metric_names,
        "video_distribution_metrics": ["fvd"] if fvd_requested else [],
        "metric_device": args.metric_device,
        "metric_batch_size": args.metric_batch_size,
        "learned_image_size": args.learned_image_size,
        "fvd_clip_length": args.fvd_clip_length,
        "fvd_clips_per_video": args.fvd_clips_per_video,
        "fvd_frame_stride": args.fvd_frame_stride,
        "fvd_image_size": args.fvd_image_size,
        "fvd_backend": args.fvd_backend,
        "fvd_detector_url": args.fvd_detector_url,
        "fvd_detector_path": str(args.fvd_detector_path) if args.fvd_detector_path else None,
        "fvd_cache_dir": str(args.fvd_cache_dir) if args.fvd_cache_dir else None,
        "fvd_allow_download": args.fvd_allow_download,
        "fvd_pca_dim_ignored": args.fvd_pca_dim,
        "fvd_eps": args.fvd_eps,
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
