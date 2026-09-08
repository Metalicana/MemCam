"""Run upstream VBench-Long with corrected custom-input clip grouping.

Only the six no-prompt dimensions on the split_clip/<stem>/<stem>_NNN.mp4
layout are supported. The upstream checkout is not modified.
"""

import argparse
from collections import defaultdict
import hashlib
import inspect
from pathlib import Path
import runpy
import sys

from audit_metric_coverage import DIMENSIONS, finite


def source_video(clip_path):
    clip = Path(clip_path)
    stem, separator, index = clip.stem.rpartition("_")
    if (clip.parent.parent.name != "split_clip" or not separator
            or not index.isascii() or not index.isdigit() or stem != clip.parent.name):
        raise ValueError(f"Unsupported VBench-Long clip layout: {clip}")
    video = clip.parent.parent.parent / (stem + ".mp4")
    if not video.is_file():
        raise ValueError(f"Missing source video for clip {clip}: {video}")
    return str(video)


def reorganize_clips_results(detailed_results, dimension=None):
    if dimension is not None and dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported dimension for custom-input grouping: {dimension}")
    if not detailed_results:
        raise ValueError("Cannot aggregate empty VBench-Long clip results")
    grouped = defaultdict(list)
    seen = set()
    for result in detailed_results:
        path = result["video_path"]
        if path in seen:
            raise ValueError(f"Duplicate clip result: {path}")
        seen.add(path)
        score = result["video_results"]
        if not isinstance(score, bool) and not finite(score):
            raise ValueError(f"Invalid clip score: {path}: {score}")
        grouped[source_video(path)].append(score)

    # Preserve upstream equal-clip means, equal-video means and imaging scaling.
    averages = [{"video_path": path, "video_results": sum(scores) / len(scores)}
                for path, scores in grouped.items()]
    overall = sum(row["video_results"] for row in averages) / len(averages)
    if dimension == "imaging_quality":
        overall /= 100
    return overall, detailed_results, averages


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vbench-root", type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    # Restrict the adapter to the layout exercised by our smoke and cohort runs.
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--mode", required=True, choices=["long_custom_input"])
    check.add_argument("--dimension", nargs="+", required=True, choices=DIMENSIONS)
    check.add_argument("--use_semantic_splitting", action="store_true")
    check.add_argument("--static_filter_flag", action="store_true")
    options, _ = check.parse_known_args(upstream_args)
    if options.use_semantic_splitting or options.static_filter_flag:
        check.error("The clip grouping adapter does not support semantic/static filtering")

    root = args.vbench_root.resolve()
    sys.path.insert(0, str(root))
    from vbench2_beta_long import utils

    original_hash = hashlib.sha256(inspect.getsource(utils.reorganize_clips_results).encode()).hexdigest()
    print(f"Custom-input grouping adapter enabled; upstream function SHA256: {original_hash}", flush=True)
    utils.reorganize_clips_results = reorganize_clips_results
    entry = root / "vbench2_beta_long/eval_long.py"
    sys.argv = [str(entry), *upstream_args]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
