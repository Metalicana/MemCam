"""Prepare shared GT/baseline DINO features for the 60s gap grid.

Extracts features from existing files only. Legacy GT caches require explicit
frame spot checks; those checks do not certify every historical cache entry.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.audit_gap_inputs import audit_cache
from utils.analyze_retrieval_quality_decomposition import (
    DinoFrameEncoder, indexed_frame_paths, iter_gt_images, iter_video_images,
    load_manifest, resolve_gt_dir,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_legacy_gt(cache, item, encoder, dataset_root=None):
    from PIL import Image
    n = int(item["num_frames"])
    audit_cache(cache, n)
    features = np.load(cache, mmap_mode="r", allow_pickle=False)
    indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    directory = resolve_gt_dir(item, dataset_root)
    files = indexed_frame_paths(directory)
    images, sources = [], []
    for i in indices:
        path = files[int(item["start_frame"]) + i]
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
        sources.append({"index": i, "path": str(path), "sha256": sha256(path)})
    fresh = encoder.encode_batch(images)
    distances = 1 - np.clip(np.sum(fresh * features[indices], axis=1), -1, 1)
    if not np.isfinite(distances).all() or np.max(distances) > 1e-4:
        raise ValueError(f"Legacy GT spot check failed: {cache}; distances={distances.tolist()}. "
                         "Use --fresh-gt with a new output directory to re-encode GT.")
    return np.array(features[:n], dtype=np.float32), {
        "mode": "legacy_cache_with_five_frame_spot_check", "source": str(cache),
        "source_sha256": sha256(cache), "checked_frames": sources,
        "cosine_distances": distances.tolist(), "all_gt_frames_reencoded": False,
        "limit": "Matching five frames and cache dimensions/norms is not full cache provenance verification."}


def save_features(path, features, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(features, dtype=np.float32), allow_pickle=False)
    audit_cache(temporary, metadata["num_frames"])
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps({**metadata, "feature_sha256": sha256(path)}, indent=2) + "\n")


def cached_match(path, expected):
    sidecar = path.with_suffix(".json")
    if not path.exists() and not sidecar.exists():
        return False
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"Incomplete cache pair: {path}; use a new output directory")
    saved = json.loads(sidecar.read_text())
    if any(saved.get(key) != value for key, value in expected.items()) or saved.get("feature_sha256") != sha256(path):
        raise ValueError(f"Stale/incompatible cache: {path}; use a new output directory")
    audit_cache(path, expected["num_frames"])
    return True


def prepare_sources(items, root, legacy_gt, dataset_root, fresh_gt):
    sources = []
    for item in items:
        video = root / "baseline" / f"{item['output_prefix']}custom.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(video)
        gt_dir = resolve_gt_dir(item, dataset_root)
        gt_files = indexed_frame_paths(gt_dir)
        missing = [i for i in range(int(item["start_frame"]), int(item["start_frame"]) + int(item["num_frames"]))
                   if i not in gt_files]
        if missing:
            raise ValueError(f"Missing GT frames in {gt_dir}: {missing[:5]}")
        cache = legacy_gt / f"{item['output_prefix']}dino.npy"
        if not fresh_gt:
            audit_cache(cache, int(item["num_frames"]))
        sources.append((item, video, cache, gt_dir))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--legacy-gt-cache", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fresh-gt", action="store_true", help="Re-encode all GT instead of reusing spot-checked legacy features")
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = load_manifest(args.manifest, duration=60)
    if len(items) != args.expected_videos or len({i["output_prefix"] for i in items}) != len(items):
        parser.error("Manifest must contain exactly the expected distinct 60s cohort")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.legacy_gt_cache = args.legacy_gt_cache or args.root / "analysis_fov_poisoning/feature_cache/gt"
    sources = prepare_sources(items, args.root, args.legacy_gt_cache, args.dataset_root, args.fresh_gt)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation unavailable; refusing an unintended CPU extraction run")
    model_name = "facebook/dinov2-base"
    encoder = DinoFrameEncoder(model_name, device=args.device, batch_size=args.batch_size)
    config = {"model": model_name, "model_revision": getattr(encoder.model.config, "_commit_hash", None),
              "processor": encoder.processor.to_dict(), "torch": str(torch.__version__),
              "pooling": "pooler_output_else_cls", "normalization": "float32_L2",
              "encoder_source_sha256": sha256(ROOT / "utils/analyze_retrieval_quality_decomposition.py")}
    # Compare only JSON-serializable configuration, consistently on resumes.
    config = json.loads(json.dumps(config))
    args.output.mkdir(parents=True, exist_ok=True)
    cache_manifest = {"status": "running", "videos": len(items), "duration_sec": 60, "source_run": "baseline",
                      "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
                      "gt_mode": "fresh" if args.fresh_gt else "legacy_spot_checked_not_fully_reencoded",
                      "encoder": config}
    (args.output / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2) + "\n")
    for position, (item, video, legacy, gt_dir) in enumerate(sources, 1):
        n = int(item["num_frames"])
        print(f"[{position}/{len(items)}] {item['scene']}: {n} frames", flush=True)
        common = {"row": item["_row"], "scene": item["scene"], "dataset_start_frame": item["start_frame"],
                  "duration_sec": 60, "num_frames": n, "output_prefix": item["output_prefix"], "encoder": config}
        stem = f"{item['output_prefix']}dino.npy"
        gt_path = args.output / "gt" / stem
        if args.fresh_gt:
            files = indexed_frame_paths(gt_dir)
            gt_sources = [{"index": i, "path": str(files[i]), "sha256": sha256(files[i])}
                          for i in range(int(item["start_frame"]), int(item["start_frame"]) + n)]
            gt_meta = {**common, "kind": "gt", "mode": "fresh", "source_frames": gt_sources}
            if not cached_match(gt_path, gt_meta):
                features = encoder.encode_iterable(iter_gt_images(item, args.dataset_root), n, label="GT frames")
                save_features(gt_path, features, gt_meta)
        else:
            features, check = check_legacy_gt(legacy, item, encoder, args.dataset_root)
            gt_meta = {**common, "kind": "gt", "mode": "legacy_spot_checked",
                       "legacy_source": str(legacy), "legacy_source_sha256": check["source_sha256"],
                       "checked_source_frames": check["checked_frames"]}
            if not cached_match(gt_path, gt_meta):
                save_features(gt_path, features, {**gt_meta, "legacy_validation": check})
        video_meta = {**common, "kind": "baseline", "source": str(video), "source_sha256": sha256(video)}
        video_path = args.output / "baseline" / stem
        if cached_match(video_path, video_meta):
            print("  baseline cache verified; reusing", flush=True)
        else:
            features = encoder.encode_iterable(iter_video_images(video, max_frames=n), n, label="baseline frames")
            save_features(video_path, features, video_meta)
            print("  baseline features saved", flush=True)
    cache_manifest["status"] = "complete"
    (args.output / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2) + "\n")
    print(f"Shared cache complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
