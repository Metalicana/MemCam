"""Diagnostic: is the retriever's pick stable, or does it flip depending on
which random points the Monte Carlo FOV-overlap check happened to sample?

The real retriever estimates FOV overlap by scattering thousands of random
points and checking visibility -- it's an estimate, not an exact number.
With more candidates in the pool, you're taking an argmax over more noisy
estimates, and the closer the true top few are to tied, the more that
argmax can land on "whichever one got lucky with its random points" rather
than the genuinely best match. This measures that directly: recompute the
same query's IoUs at a much higher sample count (closer to the true value)
and see whether the winner changes, and whether the higher-precision winner
actually looks more like the target when it does.

CPU-only, no GPU -- calculate_overlap_from_c2w hardcodes torch.device('cpu')
regardless of --device (that flag here only controls DINO encoding for the
optional appearance comparison, reusing the same cache the occlusion
diagnostic already built).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_retrieval_quality_decomposition import (  # noqa: E402
    DinoFrameEncoder,
    cached_features,
    cosine_distance,
    iter_gt_images,
    load_manifest,
    read_trace,
    reconstruct_candidate_banks,
    resolve_gt_dir,
    run_identity,
    selected_context_rows,
)
from summarize_ri_alignment import spearman  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECTION_STRIDE = 76


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


def pairwise_iou(target_c2w, candidate_c2ws, num_samples, fov_half_h, fov_half_v, radius):
    """Same helper as diagnose_fov_occlusion_poisoning.py -- imported lazily
    since calculate_overlap_from_c2w needs torch, pulled in via diffsynth's
    package __init__, and --help shouldn't require the full stack."""
    from diffsynth.models.wan_video_overlap import calculate_overlap_from_c2w

    ious = np.empty(len(candidate_c2ws), dtype=np.float64)
    for index, candidate_c2w in enumerate(candidate_c2ws):
        ious[index] = calculate_overlap_from_c2w(
            target_c2w,
            candidate_c2w,
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
            num_samples=num_samples,
            radius=radius,
            return_details=False,
        )
    return ious


def diagnose_query(
    target_frame,
    candidate_frames,
    c2ws,
    gt_features,
    low_samples,
    high_samples,
    fov_half_h,
    fov_half_v,
    radius,
    max_candidates,
    rng,
):
    candidate_frames = np.asarray(sorted(candidate_frames), dtype=np.int64)
    if candidate_frames.size > max_candidates:
        candidate_frames = rng.choice(candidate_frames, size=max_candidates, replace=False)
        candidate_frames.sort()
    if candidate_frames.size < 2:
        return None

    target_c2w = c2ws[target_frame]
    candidate_c2ws = c2ws[candidate_frames]

    ious_low = pairwise_iou(target_c2w, candidate_c2ws, low_samples, fov_half_h, fov_half_v, radius)
    ious_high = pairwise_iou(target_c2w, candidate_c2ws, high_samples, fov_half_h, fov_half_v, radius)

    winner_low_pos = int(np.argmax(ious_low))
    winner_high_pos = int(np.argmax(ious_high))
    flipped = winner_low_pos != winner_high_pos

    sorted_low = np.sort(ious_low)[::-1]
    top2_gap_low = float(sorted_low[0] - sorted_low[1]) if len(sorted_low) > 1 else None

    result = {
        "target_frame": int(target_frame),
        "num_candidates": int(len(candidate_frames)),
        "low_samples": int(low_samples),
        "high_samples": int(high_samples),
        "winner_low_frame": int(candidate_frames[winner_low_pos]),
        "winner_high_frame": int(candidate_frames[winner_high_pos]),
        "flipped": bool(flipped),
        "top2_iou_gap_low": top2_gap_low,
    }

    if gt_features is not None:
        target_feature = gt_features[target_frame]
        result["winner_low_appearance_distance"] = float(
            cosine_distance(gt_features[candidate_frames[winner_low_pos]], target_feature)
        )
        result["winner_high_appearance_distance"] = float(
            cosine_distance(gt_features[candidate_frames[winner_high_pos]], target_feature)
        )
        result["appearance_cost_of_low_precision"] = (
            result["winner_low_appearance_distance"] - result["winner_high_appearance_distance"]
        )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--target_stride", type=int, default=19)
    parser.add_argument(
        "--low_samples",
        type=int,
        default=5000,
        help="Matches FOV_SAMPLES in wan_video_memcam.py -- what the real retriever actually uses.",
    )
    parser.add_argument(
        "--high_samples",
        type=int,
        default=50000,
        help="10x the real retriever's precision, used as a closer-to-true reference.",
    )
    parser.add_argument("--iou_fov_half_h", type=float, default=45.0)
    parser.add_argument("--iou_fov_half_v", type=float, default=30.0)
    parser.add_argument("--iou_radius", type=float, default=50.0)
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=200,
        help="Random subsample cap per query -- this test computes IoU twice "
        "per candidate (low + high sample count), so it's roughly 2x the "
        "cost of the occlusion diagnostic at the same cap.",
    )
    parser.add_argument(
        "--skip_appearance",
        action="store_true",
        help="Skip DINO/GT loading entirely -- pure geometry-vs-geometry, "
        "no appearance cost computed. Faster, and doesn't need --dataset_root.",
    )
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    encoder = None
    cache_dir = None
    if not args.skip_appearance:
        cache_dir = args.feature_cache_dir or (output_dir / "feature_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoder = DinoFrameEncoder(
            model_name=args.dino_model, device=args.device, batch_size=args.batch_size
        )

    query_rows = []
    for item_position, item in enumerate(items, start=1):
        row_idx = int(item["_row"])
        expected_frames = int(item["num_frames"])
        print(f"[{item_position}/{len(items)}] row {row_idx} {item['scene']} ({expected_frames} frames)")

        gt_features = None
        if encoder is not None:
            gt_cache = cache_dir / "gt" / f"{item['output_prefix']}dino.npy"
            gt_features = cached_features(
                gt_cache,
                encoder,
                lambda item=item: iter_gt_images(item, dataset_root=args.dataset_root),
                expected_frames,
                label=f"GT frames for row {row_idx}",
            )

        from dataset.poses import load_c2ws_from_json

        c2ws = load_c2ws_from_json(
            json_path=item["pose_path"],
            start_frame=int(item["start_frame"]),
            num_frames=expected_frames,
        )

        trace_path = args.root / args.run_name / "access_traces" / f"{item['output_prefix']}custom.jsonl"
        if not trace_path.is_file():
            print(f"  [skip] no access trace at {trace_path}")
            continue
        events = read_trace(trace_path, expected_identity=run_identity(item))
        selected = selected_context_rows(events)
        if not selected:
            print(f"  [skip] no selected context rows in {trace_path}")
            continue
        max_section = max(section for section, _target in selected)
        banks = reconstruct_candidate_banks(events, max_section=max_section, num_frames=expected_frames)

        for (section_idx, target_frame), trace_row in sorted(selected.items()):
            context_slot = int(trace_row.get("context_slot", target_frame % SECTION_STRIDE))
            if context_slot % args.target_stride != 0:
                continue
            bank_candidates = banks.get(section_idx, [])
            result = diagnose_query(
                target_frame=target_frame,
                candidate_frames=bank_candidates,
                c2ws=c2ws,
                gt_features=gt_features,
                low_samples=args.low_samples,
                high_samples=args.high_samples,
                fov_half_h=args.iou_fov_half_h,
                fov_half_v=args.iou_fov_half_v,
                radius=args.iou_radius,
                max_candidates=args.max_candidates,
                rng=rng,
            )
            if result is None:
                continue
            query_rows.append(
                {
                    "row": row_idx,
                    "scene": item["scene"],
                    "dataset_start_frame": int(item["start_frame"]),
                    "duration_sec": int(item["duration_sec"]),
                    "section_idx": int(section_idx),
                    **result,
                }
            )

    if not query_rows:
        raise RuntimeError("No queries produced -- check --run_name and access-trace paths")

    fields = list(query_rows[0].keys())
    out_csv = output_dir / f"iou_estimation_noise_{args.run_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_rows)

    flips = [row["flipped"] for row in query_rows]
    flip_rate = sum(flips) / len(flips)
    pool_sizes = [row["num_candidates"] for row in query_rows]
    gaps = [row["top2_iou_gap_low"] for row in query_rows if row["top2_iou_gap_low"] is not None]
    rho_pool_vs_flip = spearman(pool_sizes, [1.0 if f else 0.0 for f in flips])

    print()
    print(f"Run: {args.run_name}  Queries analyzed: {len(query_rows)}")
    print(f"How often does re-measuring more precisely change the winner: {flip_rate:.2%}")
    print(f"Mean gap between the top and 2nd-best candidate at low precision: {sum(gaps)/len(gaps):.4f}  (small = fragile/tie-prone)")
    print(f"Spearman(pool size, flip happened) = {rho_pool_vs_flip}  (does instability grow with pool size?)")

    if "appearance_cost_of_low_precision" in query_rows[0]:
        flipped_rows = [row for row in query_rows if row["flipped"]]
        if flipped_rows:
            costs = [row["appearance_cost_of_low_precision"] for row in flipped_rows]
            print(
                f"When the winner flips, mean appearance cost of trusting the low-precision pick: "
                f"{sum(costs)/len(costs):.4f}  (positive = the higher-precision pick actually looked more like the target)"
            )

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
