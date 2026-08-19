"""Cross-system check of WorldMem's finding: when geometric FOV overlap
finds nothing (no candidate genuinely overlaps the target), what does the
retriever fall back to?

For unbounded/FIFO-like policies on WorldMem's data, the fallback is almost
always something brand new (the pool always has a fresh, redundant frame
sitting around to hand back) -- which tells the model almost nothing it
doesn't already know. SLAM's redundancy-driven eviction breaks that: it
doesn't protect recent frames just for being recent, so its fallback is
forced to be something genuinely different, because nothing lazy survives
in its pool. Two separate levers, not one: (1) how often a real match gets
found at all, (2) when it doesn't, how informative the fallback is.

This needs zero recomputation on MemCam's side -- every context_access
trace event already logs selected_overlap (the winning IoU) and memory_age
(target_frame - selected_memory_frame). Pure trace-reading, no geometry, no
DINO, no GPU, not even a bank reconstruction. Cheapest of every diagnostic
built this session, and directly comparable number-for-number to what
WorldMem just reported on their own data.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import sys

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_retrieval_quality_decomposition import (  # noqa: E402
    load_manifest,
    read_trace,
    run_identity,
    selected_context_rows,
)


def parse_rows(value):
    if not value:
        return None
    rows = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            rows.update(range(int(start), int(end) + 1))
        else:
            rows.add(int(part))
    return rows


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100))


def analyze_run(items, root, run_name, overlap_threshold):
    hit_ages = []
    miss_ages = []
    total = 0

    for item in items:
        trace_path = root / run_name / "access_traces" / f"{item['output_prefix']}custom.jsonl"
        if not trace_path.is_file():
            continue
        events = read_trace(trace_path, expected_identity=run_identity(item))
        selected = selected_context_rows(events)
        for trace_row in selected.values():
            overlap = trace_row.get("selected_overlap")
            age = trace_row.get("memory_age")
            if overlap is None or age is None:
                continue
            total += 1
            if float(overlap) > overlap_threshold:
                hit_ages.append(int(age))
            else:
                miss_ages.append(int(age))

    if total == 0:
        return None

    hit_rate = len(hit_ages) / total
    return {
        "run_name": run_name,
        "queries": total,
        "hit_rate": hit_rate,
        "miss_rate": 1.0 - hit_rate,
        "zero_overlap_age_median": percentile(miss_ages, 0.5),
        "zero_overlap_age_p90": percentile(miss_ages, 0.9),
        "zero_overlap_age_max": (max(miss_ages) if miss_ages else None),
        "hit_age_median": percentile(hit_ages, 0.5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Comma-separated run names to compare side by side, e.g. "
        "baseline,fifo_b32,slam_b32_covisibility,mce_b32_lambda1_pilot",
    )
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.01,
        help="Overlap at or below this counts as 'no real geometric match'.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    items = load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=parse_rows(args.rows),
        max_rows=args.max_rows,
    )
    if not items:
        raise RuntimeError("No manifest rows selected")

    run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    results = []
    for run_name in run_names:
        result = analyze_run(items, args.root, run_name, args.overlap_threshold)
        if result is None:
            print(f"[skip] {run_name}: no access traces found")
            continue
        results.append(result)

    if not results:
        raise RuntimeError("No runs produced results -- check --runs and --root")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "zero_overlap_fallback.csv"
    fields = list(results[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("How often does geometry find a real match, and what happens when it doesn't:")
    print()
    header = f"{'run':<28}{'hit rate':>10}{'zero-ov age (med)':>20}{'zero-ov age (p90)':>20}{'zero-ov age (max)':>20}"
    print(header)
    for row in results:
        print(
            f"{row['run_name']:<28}"
            f"{row['hit_rate']:>10.1%}"
            f"{row['zero_overlap_age_median']:>20.1f}"
            f"{row['zero_overlap_age_p90']:>20.1f}"
            f"{row['zero_overlap_age_max']:>20.0f}"
        )
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
