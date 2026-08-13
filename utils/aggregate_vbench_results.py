"""Aggregate VBench *_eval_results.json files across runs into one table.

VBench's evaluate.py writes results_dict[dimension] = [mean_score,
[per-video detail, ...]] to <output_path>/<name>_eval_results.json (see
vbench/__init__.py:evaluate() and e.g. vbench/subject_consistency.py's
compute_subject_consistency, which both return (all_results, video_results)
tuples). This script picks the most recent *_eval_results.json per run
directory, pulls out just the means, and prints one run x dimension table.

Usage:
    python utils/aggregate_vbench_results.py \
        --root ~/memcam_results/vbench_results
"""

import argparse
import json
from pathlib import Path


def load_latest_eval_results(run_dir):
    candidates = sorted(run_dir.glob("*_eval_results.json"))
    if not candidates:
        return None, None
    # Filenames are results_<timestamp>_eval_results.json -- lexical sort on
    # the ISO-ish timestamp is chronological, so the last one is newest.
    latest = candidates[-1]
    with latest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return latest, payload


def count_videos_scored(payload):
    for dimension, value in payload.items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):
            return len(value[1])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated run dir names (default: all subdirectories of --root).")
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated dimensions to show (default: whatever's in the files).")
    args = parser.parse_args()

    if args.runs:
        run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    else:
        run_names = sorted(p.name for p in args.root.iterdir() if p.is_dir())

    rows = {}
    all_dimensions = []
    for run_name in run_names:
        run_dir = args.root / run_name
        if not run_dir.is_dir():
            print(f"[skip] {run_name}: directory not found at {run_dir}")
            continue
        source_path, payload = load_latest_eval_results(run_dir)
        if payload is None:
            print(f"[skip] {run_name}: no *_eval_results.json under {run_dir}")
            continue

        means = {}
        for dimension, value in payload.items():
            if isinstance(value, list) and len(value) == 2:
                means[dimension] = float(value[0])
            elif isinstance(value, (int, float)):
                means[dimension] = float(value)
            else:
                means[dimension] = None
            if dimension not in all_dimensions:
                all_dimensions.append(dimension)

        rows[run_name] = {
            "means": means,
            "videos_scored": count_videos_scored(payload),
            "source": source_path.name,
        }

    dimensions = (
        [d.strip() for d in args.dimensions.split(",") if d.strip()]
        if args.dimensions
        else all_dimensions
    )

    name_width = max([len("run")] + [len(name) for name in rows]) + 2
    col_width = 11
    header = f"{'run':<{name_width}}{'n_vid':>7}  " + "".join(f"{dim[:col_width-1]:>{col_width}}" for dim in dimensions)
    print(header)
    print("-" * len(header))
    for run_name, data in rows.items():
        n_vid = data["videos_scored"]
        n_vid_str = str(n_vid) if n_vid is not None else "?"
        line = f"{run_name:<{name_width}}{n_vid_str:>7}  "
        for dim in dimensions:
            value = data["means"].get(dim)
            line += f"{value:>{col_width}.4f}" if value is not None else f"{'--':>{col_width}}"
        print(line)

    print(f"\n({len(rows)} runs, source file per run noted below if you need to double-check freshness)")
    for run_name, data in rows.items():
        print(f"  {run_name}: {data['source']}")


if __name__ == "__main__":
    main()
