"""Diagnostic: does the geometry-only FOV-overlap retriever get systematically
"poisoned" -- i.e. does its whole geometric shortlist agree with each other
but disagree with the true target appearance, because they're all occluded
by the same obstruction (a building, a wall) rather than genuinely showing
the target view?

This needs NO generated video and NO GPU: only real camera poses (from the
dataset's own pose JSONs) and real ground-truth frame appearance (DINO on
GT frames, cached once). calculate_overlap_from_c2w is CPU-only regardless
of device (hardcoded torch.device('cpu') inside it), and DINO on a modest
GT-frame count is CPU-tractable. This is strictly more portable than
retrieval_gap in analyze_retrieval_quality_decomposition.py, which needs a
real generated video to exist first -- this diagnostic could in principle
run before ever generating anything.

For each sampled query (target frame, real historical candidate pool
reconstructed from an existing access trace):
  1. Recompute IoU(target, candidate) for every candidate -- purely
     geometric, from c2w poses, exactly what the real retriever computes.
  2. Look up DINO appearance distance from each candidate's GT frame to the
     target's GT frame -- purely from the dataset, independent of any
     generated video.
  3. "Geometric winner" = the candidate the real retriever would pick
     (argmax IoU). "Appearance oracle" = the candidate that actually looks
     most like the target (argmin appearance distance).
  4. The key number: appearance_oracle_iou_percentile -- where does the
     TRUE best-appearance match rank in the geometric IoU ordering? 0 = it's
     also the geometric top pick (no conflict). Close to 1 = the true best
     match is geometrically buried at the bottom of the ranking, while
     everything the retriever actually considers (its top-K by IoU) is a
     geometrically-agreeing but visually-wrong cluster -- exactly the
     "poisoned by a shared occluder" signature.
  5. topk_std_appearance_distance being LOW while topk_mean is HIGH is the
     "consistent wrongness" signature of a shared occluder, as opposed to
     just noisy-but-roughly-right FOV imprecision (which would show HIGH
     variance instead).
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
FRAMES_PER_SECTION = 77


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
    """IoU of one target pose against many candidate poses. Always CPU --
    calculate_overlap_from_c2w hardcodes torch.device('cpu') internally.
    Imported lazily (needs torch, pulled in via diffsynth's package
    __init__) so --help and argument validation don't require it."""
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
    num_samples,
    fov_half_h,
    fov_half_v,
    radius,
    top_k,
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
    ious = pairwise_iou(
        target_c2w, candidate_c2ws, num_samples, fov_half_h, fov_half_v, radius
    )

    target_feature = gt_features[target_frame]
    candidate_features = gt_features[candidate_frames]
    distances = np.array(
        [cosine_distance(candidate_features[i], target_feature) for i in range(len(candidate_frames))],
        dtype=np.float64,
    )

    iou_order = np.argsort(-ious)  # best geometric match first
    geometric_winner_pos = iou_order[0]
    appearance_oracle_pos = int(np.argmin(distances))
    appearance_oracle_iou_rank = int(np.where(iou_order == appearance_oracle_pos)[0][0])
    denom = max(len(candidate_frames) - 1, 1)

    top_k_actual = min(top_k, len(candidate_frames))
    top_k_positions = iou_order[:top_k_actual]
    top_k_distances = distances[top_k_positions]

    return {
        "target_frame": int(target_frame),
        "num_candidates": int(len(candidate_frames)),
        "geometric_winner_frame": int(candidate_frames[geometric_winner_pos]),
        "geometric_winner_iou": float(ious[geometric_winner_pos]),
        "geometric_winner_appearance_distance": float(distances[geometric_winner_pos]),
        "appearance_oracle_frame": int(candidate_frames[appearance_oracle_pos]),
        "appearance_oracle_iou": float(ious[appearance_oracle_pos]),
        "appearance_oracle_distance": float(distances[appearance_oracle_pos]),
        "appearance_oracle_iou_rank": appearance_oracle_iou_rank,
        "appearance_oracle_iou_percentile": float(appearance_oracle_iou_rank / denom),
        "poisoning_gap": float(
            distances[geometric_winner_pos] - distances[appearance_oracle_pos]
        ),
        "top_k": int(top_k_actual),
        "topk_mean_appearance_distance": float(np.mean(top_k_distances)),
        "topk_std_appearance_distance": float(np.std(top_k_distances)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline",
        help="Which run's access trace to reconstruct real candidate banks "
        "from (bank composition differs by policy; poses/appearance don't).",
    )
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--target_stride", type=int, default=19)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--iou_num_samples", type=int, default=2000)
    parser.add_argument("--iou_fov_half_h", type=float, default=45.0)
    parser.add_argument("--iou_fov_half_v", type=float, default=30.0)
    parser.add_argument("--iou_radius", type=float, default=50.0)
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=200,
        help="Random subsample cap per query -- unbounded's pool can reach "
        "into the thousands late in a rollout, and IoU is O(candidates x "
        "iou_num_samples) per query with no shortcut.",
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

    from dataset.poses import load_c2ws_from_json

    output_dir = args.output_dir
    cache_dir = args.feature_cache_dir or (output_dir / "feature_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = DinoFrameEncoder(
        model_name=args.dino_model, device=args.device, batch_size=args.batch_size
    )
    rng = np.random.default_rng(args.seed)

    query_rows = []
    for item_position, item in enumerate(items, start=1):
        row_idx = int(item["_row"])
        expected_frames = int(item["num_frames"])
        print(f"[{item_position}/{len(items)}] row {row_idx} {item['scene']} ({expected_frames} frames)")

        gt_cache = cache_dir / "gt" / f"{item['output_prefix']}dino.npy"
        gt_features = cached_features(
            gt_cache,
            encoder,
            lambda item=item: iter_gt_images(item, dataset_root=args.dataset_root),
            expected_frames,
            label=f"GT frames for row {row_idx}",
        )

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
                num_samples=args.iou_num_samples,
                fov_half_h=args.iou_fov_half_h,
                fov_half_v=args.iou_fov_half_v,
                radius=args.iou_radius,
                top_k=args.top_k,
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
    out_csv = output_dir / f"fov_occlusion_poisoning_{args.run_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_rows)

    gaps = [row["poisoning_gap"] for row in query_rows]
    percentiles = [row["appearance_oracle_iou_percentile"] for row in query_rows]
    pool_sizes = [row["num_candidates"] for row in query_rows]
    rho_pool_vs_gap = spearman(pool_sizes, gaps)
    conflict_rate = sum(1 for g in gaps if g > 1e-6) / len(gaps)

    print()
    print(f"Run: {args.run_name}  Queries analyzed: {len(query_rows)}")
    print(f"Mean poisoning_gap: {np.mean(gaps):.4f}  (0 = geometric winner is also the best appearance match)")
    print(f"Fraction of queries with any conflict (poisoning_gap > 0): {conflict_rate:.2%}")
    print(f"Mean appearance_oracle_iou_percentile: {np.mean(percentiles):.4f}  (0=best-ranked geometrically, 1=worst)")
    print(f"Spearman(candidate pool size, poisoning_gap) = {rho_pool_vs_gap}  (does poisoning risk grow with pool size?)")
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
