"""Diagnostic: does one memory frame win the per-target argmax for most/all
of a section's ~76 targets, quietly collapsing the model's context for that
whole predicted span down to a handful of repeats?

Cheaper than the occlusion-poisoning diagnostic: needs no geometry
recomputation and no DINO at all. The real retriever already ran during
generation and already recorded, in the access trace, which memory frame it
picked for every single target -- this just counts. Pure CPU, pure
bookkeeping, no GPU, no new computation of any kind.

For each (row, section): gather every target's selected_memory_frame from
the access trace, then report:
  - distinct_frames_selected: how many different memory frames actually got
    used across the ~76 targets (76 = perfect diversity, 1 = total collapse)
  - max_frame_share: the single most-picked frame's share of all targets in
    the section -- the direct "does frame X win most of the 76" number
  - effective_num_frames: 1 / sum(p_i^2) (inverse Herfindahl index) -- a
    smooth diversity measure between the two extremes above
  - candidate_pool_size: the real reconstructed bank size for that section,
    to test whether collapse gets worse as the pool grows (an H1-style
    check, specific to this mechanism)
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_retrieval_quality_decomposition import (  # noqa: E402
    load_manifest,
    read_trace,
    reconstruct_candidate_banks,
    run_identity,
    selected_context_rows,
)
from summarize_ri_alignment import spearman  # noqa: E402


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


def section_diversity_stats(targets_in_section):
    """targets_in_section: list of selected_memory_frame values, one per
    target, for a single section."""
    counts = Counter(targets_in_section)
    total = len(targets_in_section)
    shares = [count / total for count in counts.values()]
    effective_num_frames = 1.0 / sum(share * share for share in shares)
    max_frame, max_count = counts.most_common(1)[0]
    return {
        "targets": total,
        "distinct_frames_selected": len(counts),
        "dominant_frame": int(max_frame),
        "dominant_frame_count": max_count,
        "max_frame_share": max_count / total,
        "effective_num_frames": effective_num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    section_rows = []

    for item_position, item in enumerate(items, start=1):
        row_idx = int(item["_row"])
        expected_frames = int(item["num_frames"])
        print(f"[{item_position}/{len(items)}] row {row_idx} {item['scene']}")

        trace_path = args.root / args.run_name / "access_traces" / f"{item['output_prefix']}custom.jsonl"
        if not trace_path.is_file():
            print(f"  [skip] no access trace at {trace_path}")
            continue
        events = read_trace(trace_path, expected_identity=run_identity(item))
        selected = selected_context_rows(events)
        if not selected:
            print(f"  [skip] no selected context rows in {trace_path}")
            continue

        by_section = defaultdict(list)
        for (section_idx, _target_frame), trace_row in selected.items():
            by_section[int(section_idx)].append(int(trace_row["selected_memory_frame"]))

        max_section = max(by_section)
        banks = reconstruct_candidate_banks(events, max_section=max_section, num_frames=expected_frames)

        for section_idx, selected_frames in sorted(by_section.items()):
            if len(selected_frames) < 2:
                continue
            stats = section_diversity_stats(selected_frames)
            candidate_pool_size = len(banks.get(section_idx, []))
            section_rows.append(
                {
                    "row": row_idx,
                    "scene": item["scene"],
                    "dataset_start_frame": int(item["start_frame"]),
                    "duration_sec": int(item["duration_sec"]),
                    "section_idx": section_idx,
                    "candidate_pool_size": candidate_pool_size,
                    **stats,
                }
            )

    if not section_rows:
        raise RuntimeError("No sections produced -- check --run_name and access-trace paths")

    fields = list(section_rows[0].keys())
    out_csv = args.output_dir / f"context_diversity_{args.run_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(section_rows)

    max_shares = [row["max_frame_share"] for row in section_rows]
    effective_counts = [row["effective_num_frames"] for row in section_rows]
    pool_sizes = [row["candidate_pool_size"] for row in section_rows]
    rho_pool_vs_share = spearman(pool_sizes, max_shares)
    targets_per_section = section_rows[0]["targets"]

    print()
    print(f"Run: {args.run_name}  Sections analyzed: {len(section_rows)}  (targets/section: ~{targets_per_section})")
    print(f"Mean max_frame_share: {sum(max_shares)/len(max_shares):.3f}  (1/{targets_per_section}={1/targets_per_section:.3f} = perfectly uniform, 1.0 = total collapse)")
    print(f"Mean effective_num_frames: {sum(effective_counts)/len(effective_counts):.2f}  (out of ~{targets_per_section} possible)")
    print(f"Sections with max_frame_share > 0.5 (one frame won more than half the section's targets): "
          f"{sum(1 for s in max_shares if s > 0.5)}/{len(max_shares)}")
    print(f"Spearman(candidate_pool_size, max_frame_share) = {rho_pool_vs_share}  (does collapse get worse as the pool grows?)")
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
