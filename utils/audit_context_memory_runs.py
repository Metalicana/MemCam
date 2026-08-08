"""Audit which (policy run, manifest row, duration) combinations already have
a generated video, and which are still missing -- so you know exactly what's
left to submit before queuing more sbatch jobs.

Re-derives each row's expected output filename the same way
run_context_memory_batch.py does (via load_manifest/output_path), then just
checks whether that file exists under each run directory. This avoids trying
to reverse-parse row indices out of already-generated filenames, which don't
encode the manifest row directly.

Usage:
    python utils/audit_context_memory_runs.py \
        --manifest testbeds/context_memory/manifest.jsonl \
        --root ~/memcam_results/context_memory_60s

    # Only check specific runs:
    python utils/audit_context_memory_runs.py \
        --manifest testbeds/context_memory/manifest.jsonl \
        --root ~/memcam_results/context_memory_60s \
        --runs mce_b32_lambda1_pilot,ri_b32_dino_rgb,slam_b32_covisibility
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.run_context_memory_batch import load_manifest, output_path, parse_int_csv  # noqa: E402

BUDGET_PATTERN = re.compile(r"_b(\d+)(?:_|$)")


def guess_budget(run_name):
    match = BUDGET_PATTERN.search(run_name)
    return int(match.group(1)) if match else None


def format_row_ranges(rows):
    """Compact 'a-b,c,d-e' formatting for a sorted list of ints."""
    if not rows:
        return "(none)"
    rows = sorted(rows)
    ranges = []
    start = prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = row
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing one subdirectory per policy run.")
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated run directory names to check (default: all subdirectories of --root).")
    parser.add_argument("--durations", type=str, default=None, help="Comma-separated duration_sec values to include (default: all present in manifest).")
    args = parser.parse_args()

    manifest_items = load_manifest(args.manifest)
    duration_filter = set(parse_int_csv(args.durations)) if args.durations else None
    if duration_filter:
        manifest_items = [item for item in manifest_items if item["duration_sec"] in duration_filter]

    durations_present = sorted({item["duration_sec"] for item in manifest_items})
    total_expected = len(manifest_items)
    all_rows = sorted({item["_row"] for item in manifest_items})

    print(f"Manifest: {args.manifest}")
    print(f"Rows in manifest (after duration filter): {len(all_rows)}  ({format_row_ranges(all_rows)})")
    print(f"Durations covered: {durations_present}")
    print(f"Expected videos per fully-completed run: {total_expected}\n")

    if args.runs:
        run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    else:
        run_names = sorted(p.name for p in args.root.iterdir() if p.is_dir())

    header = f"{'run':40s} {'budget':7s} {'done':>6s} {'/':1s} {'total':>6s} {'%':>6s}  missing rows"
    print(header)
    print("-" * len(header))

    for run_name in run_names:
        run_dir = args.root / run_name
        if not run_dir.is_dir():
            print(f"{run_name:40s}  -- directory not found: {run_dir}")
            continue

        budget = guess_budget(run_name)
        done_rows = []
        missing_rows = []
        for item in manifest_items:
            path = output_path(run_dir, item)
            if path.is_file():
                done_rows.append(item["_row"])
            else:
                missing_rows.append(item["_row"])

        pct = 100.0 * len(done_rows) / total_expected if total_expected else 0.0
        budget_str = str(budget) if budget is not None else "?"
        print(
            f"{run_name:40s} {budget_str:7s} {len(done_rows):6d} / {total_expected:6d} {pct:5.1f}%  "
            f"{format_row_ranges(missing_rows) if missing_rows else '(complete)'}"
        )


if __name__ == "__main__":
    main()
