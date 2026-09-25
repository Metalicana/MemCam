"""Run standard VBench without whole-video CLIP activation batches.

Only background-consistency image encoding is microbatched. Upstream decoding,
transforms, temporal comparisons, and aggregation are unchanged; no frames are
dropped and no video is split into independently scored clips.
"""

import argparse
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import runpy
import sys

from audit_metric_coverage import DIMENSIONS


class BatchedImageEncoder:
    def __init__(self, model, batch_size):
        if batch_size < 1:
            raise ValueError("CLIP batch size must be positive")
        self.model = model
        self.batch_size = batch_size

    def encode_image(self, images):
        import torch

        if len(images) == 0:
            raise ValueError("Cannot encode an empty video")
        # The upstream function still holds its input tensor. Only transformer
        # activations are bounded here, not video decoding or host RAM usage.
        with torch.no_grad():
            return torch.cat([self.model.encode_image(batch)
                              for batch in images.split(self.batch_size)], dim=0)


def batch_background(original, batch_size):
    @wraps(original)
    def evaluate(clip_model, preprocess, video_list, device, read_frame):
        import torch

        with torch.no_grad():
            return original(BatchedImageEncoder(clip_model, batch_size), preprocess,
                            video_list, device, read_frame)

    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    args, upstream_args = parser.parse_known_args()
    if args.clip_batch_size < 1:
        parser.error("--clip-batch-size must be positive")
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--mode", required=True, choices=["custom_input"])
    check.add_argument("--dimension", nargs="+", required=True, choices=DIMENSIONS)
    check.add_argument("--output_path", type=Path, required=True)
    options, _ = check.parse_known_args(upstream_args)

    root = args.vbench_root.resolve()
    entry = root / "evaluate.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    sys.path.insert(0, str(root))
    import torch

    info = {"adapter": str(Path(__file__).resolve()), "clip_batch_size": args.clip_batch_size,
            "dimensions": options.dimension, "gradients": False,
            "frame_sampling": "unchanged upstream", "temporal_aggregation": "unchanged upstream"}
    if "background_consistency" in options.dimension:
        module = importlib.import_module("vbench.background_consistency")
        original = module.background_consistency
        if list(inspect.signature(original).parameters) != [
                "clip_model", "preprocess", "video_list", "device", "read_frame"]:
            raise ValueError("Unsupported upstream background-consistency interface")
        info["upstream_background_sha256"] = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
        module.background_consistency = batch_background(original, args.clip_batch_size)
    options.output_path.mkdir(parents=True, exist_ok=True)
    (options.output_path / "batching.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Standard VBench bounded CLIP inference: {json.dumps(info)}", flush=True)
    sys.argv = [str(entry), *upstream_args]
    try:
        with torch.no_grad():
            runpy.run_path(str(entry), run_name="__main__")
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
