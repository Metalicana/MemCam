"""Offline replay of a retriever-conditioned coverage objective (U_sel), as
opposed to MCE's abstract oracle coverage.

Motivation: MCE's noisy-OR objective is monotone (adding memory can never
decrease coverage), but the actual observed failure mode this whole project
has been chasing (unbounded memory's large retrieval_gap in
unbounded_failure_decomposition_180s: retention_gap=0.0000, retrieval_gap=
0.2267 -- the correct frame was always retained, the frozen top-1 selector
just picked wrong as the pool grew) is a case where adding memory DOES hurt
real quality. MCE's objective has no mechanism to represent that risk,
because it never asks "what would the real selector actually pick from this
set." A real-generation test of the estimate_cluster_threshold fix (which
did meaningfully improve MCE's write-path anchor-persistence, final_age_p90
161->825) still lost to RI/SLAM on real LPIPS/SSIM at matched n=3 -- evidence
that the coverage-side fix alone isn't sufficient, motivating this axis.

U_sel(M) = sum_q w_q * K(q, r(q;M)), where
    r(q;M) = argmax_{m in M} s_native(q,m)
is EXACTLY what MemCam's real context-selection step would pick -- this
reuses diffsynth.models.wan_video_overlap.calculate_overlap_from_c2w
directly (the same function, same FOV_HALF_H/V, same radius, called the same
way as wan_video_memcam.py's real "Selecting context frames" block). Not an
approximation of the real selector: the real selector.

TWO MODES, because of a mathematical fact caught by hand-testing before this
was used for anything:

  Phase 1 (pose-only, K = s_native): the same geometric FOV-overlap function
  used for both selection AND value. This makes U_sel mathematically
  IDENTICAL in structure to the facility-location objective
  (sum_q w_q max_{m in M} K(q,m)) -- and max-reductions are monotone by
  construction: removing a candidate can only decrease or maintain each
  query's max, never increase it. Verified directly: delta =
  U_sel(M-i) - U_sel(M) is always <= 0. So Phase 1 is, like MCE, monotone --
  it does NOT and CANNOT represent the "distractor" phenomenon (a candidate
  winning the selector's argmax while making outcomes worse), because that
  requires the selector's own criterion (s_native) to genuinely disagree
  with true quality (K), i.e. K != s_native. Still useful as a more faithful
  MCE alternative in its own right: real per-target-frame queries and the
  real selector's exact scoring function, instead of MCE's medoid-compressed
  Q_hist and approximate kernel (closer to WorldMem's diagnosed "comparison
  breadth" gap in MCE).

  Phase 2 (--manifest/--row, K = ground-truth content similarity): r(q;M)
  is still computed from s_native (the real, pose-only selector -- it has no
  other signal available), but VALUED against K(q,m) = DINO similarity
  between ground-truth appearance at m and ground-truth appearance at the
  query's target position -- genuinely independent of s_native, using no
  generated video at all (reuses iter_gt_images/resolve_gt_dir from
  analyze_retrieval_quality_decomposition.py, the same oracle methodology
  behind unbounded_failure_decomposition_180s's retention/retrieval gaps).
  THIS mode can show positive delta_i (removal that increases U_sel, in the
  eviction loop's delta = U_sel(after) - U_sel(before) convention): a
  candidate can win the pose-only argmax for some query while genuinely
  being a poor content match (a real distractor -- e.g. similar camera
  angle, different scene state), so removing it lets a truer-but-lower-IOU
  candidate win instead. positive_marginal_fraction > 0 here is the actual
  signal this tool exists to find -- verified against a hand-built
  distractor case (see test in the commit history) before trusting it on
  real data.

Usage:
    # Phase 1 -- pose only, no manifest, no GPU:
    python utils/selector_aware_offline_sweep.py \
        --pose_json /path/to/scene.json --start_frame 1368 --num_frames 1825 --budget 32

    # Phase 2 -- ground-truth content quality, needs a manifest row + GPU for DINO:
    python utils/selector_aware_offline_sweep.py \
        --manifest testbeds/context_memory/manifest.jsonl --row 3 --budget 32 \
        --dataset_root ~/data/Context-as-Memory-Dataset/Context-as-Memory-Dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffsynth.models.wan_video_overlap import calculate_overlap_from_c2w  # noqa: E402
from utils.analyze_retrieval_quality_decomposition import (  # noqa: E402
    load_manifest,
    iter_gt_images,
)

FRAMES_PER_SECTION = 77
PREDICT_FRAMES = 76
FOV_HALF_H = 45.0
FOV_HALF_V = 30.0
FOV_RADIUS = 50.0


def section_boundaries(total_frames):
    total_sections = (total_frames - 1) // PREDICT_FRAMES
    sections = []
    for section_idx in range(total_sections):
        if section_idx == 0:
            new_frame_indices = list(range(0, FRAMES_PER_SECTION))
        else:
            start = section_idx * PREDICT_FRAMES + 1
            new_frame_indices = list(range(start, start + PREDICT_FRAMES))
        sections.append(new_frame_indices)
    return sections


def build_pose_matrix(c2ws, query_frame_indices, candidate_frame_indices, num_samples):
    """s_native(q, m) for every (query, candidate) pair -- exactly the real
    selector's per-pair computation, cached once per section so the reverse-
    deletion loop below never re-touches the (stochastic, Monte-Carlo) FOV
    overlap function again, only cheap numpy operations over it."""
    num_queries = len(query_frame_indices)
    num_candidates = len(candidate_frame_indices)
    matrix = np.zeros((num_queries, num_candidates), dtype=np.float64)
    for qi, query_frame in enumerate(query_frame_indices):
        target_c2w = c2ws[query_frame]
        for ci, candidate_frame in enumerate(candidate_frame_indices):
            matrix[qi, ci] = calculate_overlap_from_c2w(
                target_c2w,
                c2ws[candidate_frame],
                fov_half_h=FOV_HALF_H,
                fov_half_v=FOV_HALF_V,
                num_samples=num_samples,
                radius=FOV_RADIUS,
                return_details=False,
            )
    return matrix


def build_content_matrix(gt_dino_features, query_frame_indices, candidate_frame_indices):
    """K(q, m) = calibrated DINO cosine similarity between ground-truth
    appearance at m and ground-truth appearance at q -- a genuine quality
    signal independent of s_native (pose-only), needed for Phase 2."""
    query_features = np.stack([gt_dino_features[f] for f in query_frame_indices])
    candidate_features = np.stack([gt_dino_features[f] for f in candidate_frame_indices])
    cosine = query_features @ candidate_features.T
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)


def load_gt_content_features(manifest_item, dataset_root, device):
    """DINO features over every ground-truth frame the manifest row covers --
    no generated video needed, this is purely dataset content."""
    from diffsynth.pipelines.memory_policies import VisualMemoryFeatureExtractor

    images = list(iter_gt_images(manifest_item, dataset_root=dataset_root))
    extractor = VisualMemoryFeatureExtractor(device=device)
    dino_batch, _ = extractor.encode_pil_images(images)
    return {i: dino_batch[i] for i in range(len(images))}


def u_sel(s_matrix, k_matrix, weights, active_mask):
    """r(q;M) = argmax over active columns of s_matrix (what the real
    selector would pick); value it under k_matrix (which may be s_matrix
    itself in Phase 1, or a genuinely different quality signal in Phase 2)."""
    if not np.any(active_mask):
        return 0.0
    active_indices = np.flatnonzero(active_mask)
    winners = active_indices[np.argmax(s_matrix[:, active_mask], axis=1)]
    values = k_matrix[np.arange(k_matrix.shape[0]), winners]
    return float(np.sum(weights * values))


def reverse_delete_selector_aware(s_matrix, k_matrix, weights, candidate_frame_indices, budget, forced_keep_frames):
    """Same reverse-deletion shape as MCE's Algorithm 1, but scored under
    U_sel instead of noisy-OR coverage. Evicts argmax delta_i, where
    delta_i = U_sel(active - i) - U_sel(active) -- i.e. whichever single
    removal leaves U_sel highest (equivalently: least harmful, or in Phase 2,
    possibly outright beneficial if i was a distractor). NOTE: this is
    argmax, not argmin, because delta is defined as the value AFTER removal
    minus before -- the opposite sign convention from MCE's Delta_i (loss
    from removal), which is why MCE evicts argmin. Getting this backwards
    was caught by hand-testing a synthetic distractor case before this tool
    was used for anything: with argmin, it would have preferentially evicted
    the MOST valuable candidates first. delta_i can be negative in Phase 2
    (evicting i is beneficial -- i was a distractor), impossible in Phase 1
    (k_matrix is s_matrix, so this reduces to monotone facility location --
    see module docstring)."""
    num_candidates = len(candidate_frame_indices)
    active_mask = np.ones(num_candidates, dtype=bool)
    forced_positions = {
        position for position, frame_idx in enumerate(candidate_frame_indices)
        if frame_idx in forced_keep_frames
    }
    removal_order = []

    selected_limit = min(int(budget), num_candidates)
    while int(active_mask.sum()) > selected_limit:
        base_value = u_sel(s_matrix, k_matrix, weights, active_mask)
        removable = [p for p in range(num_candidates) if active_mask[p] and p not in forced_positions]
        if not removable:
            break
        best_delta = None
        best_position = None
        for position in removable:
            trial_mask = active_mask.copy()
            trial_mask[position] = False
            delta = u_sel(s_matrix, k_matrix, weights, trial_mask) - base_value
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_position = position
        active_mask[best_position] = False
        removal_order.append((candidate_frame_indices[best_position], best_delta))

    retained = [candidate_frame_indices[p] for p in range(num_candidates) if active_mask[p]]
    return retained, removal_order


def simulate_selector_aware(c2ws, total_frames, budget, query_stride, num_samples, gt_dino_features=None):
    phase2 = gt_dino_features is not None
    sections = section_boundaries(total_frames)
    memory = []
    admitted = set()
    alive_since = {}
    survival_records = []
    positive_marginal_evictions = 0
    total_evictions = 0
    final_ages = None

    for section_idx, new_frame_indices in enumerate(sections):
        section_end_frame = new_frame_indices[-1]
        prospective_memory = memory + [f for f in new_frame_indices if f not in memory]
        for f in new_frame_indices:
            admitted.add(f)
            alive_since.setdefault(f, section_idx)

        query_frame_indices = new_frame_indices[::max(1, int(query_stride))]
        weights = np.full(len(query_frame_indices), 1.0 / len(query_frame_indices))

        s_matrix = build_pose_matrix(c2ws, query_frame_indices, prospective_memory, num_samples)
        if phase2:
            k_matrix = build_content_matrix(gt_dino_features, query_frame_indices, prospective_memory)
        else:
            k_matrix = s_matrix

        forced_keep_frames = {0, section_end_frame}
        retained, removal_order = reverse_delete_selector_aware(
            s_matrix, k_matrix, weights, prospective_memory, budget, forced_keep_frames
        )

        evicted = set(prospective_memory) - set(retained)
        for frame_idx, delta in removal_order:
            total_evictions += 1
            if delta > 0:
                positive_marginal_evictions += 1
        for frame_idx in evicted:
            survival_records.append((frame_idx, section_idx - alive_since[frame_idx] + 1))

        memory = retained
        if section_idx == len(sections) - 1:
            final_ages = [section_end_frame - f for f in memory]

    last_section_idx = len(sections) - 1
    for frame_idx in memory:
        survival_records.append((frame_idx, last_section_idx - alive_since[frame_idx] + 1))

    survival = np.array([s for _, s in survival_records], dtype=np.float64)
    return {
        "mode": "phase2_gt_content" if phase2 else "phase1_pose_only",
        "retained_frames": sorted(memory),
        "final_age_median": float(np.median(final_ages)) if final_ages else None,
        "final_age_p90": float(np.percentile(final_ages, 90)) if final_ages else None,
        "survival_sections_mean": float(survival.mean()) if survival.size else None,
        "survival_sections_p90": float(np.percentile(survival, 90)) if survival.size else None,
        "survival_gini": gini(survival) if survival.size else None,
        "unique_frames_ever_admitted": len(admitted),
        "num_sections": len(sections),
        "total_evictions": total_evictions,
        "positive_marginal_evictions": positive_marginal_evictions,
        "positive_marginal_fraction": (
            positive_marginal_evictions / total_evictions if total_evictions else None
        ),
    }


def gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    if n == 0 or values.sum() <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose_json", type=Path, default=None, help="Phase 1 only.")
    parser.add_argument("--start_frame", type=int, default=None, help="Phase 1 only.")
    parser.add_argument("--num_frames", type=int, default=None, help="Phase 1 only.")
    parser.add_argument("--manifest", type=Path, default=None, help="Phase 2: enables ground-truth content quality.")
    parser.add_argument("--row", type=int, default=None, help="Phase 2: manifest row index.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Phase 2: passed to resolve_gt_dir.")
    parser.add_argument("--gt_device", type=str, default="cuda", help="Phase 2: device for DINO feature extraction.")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument(
        "--query_stride", type=int, default=4,
        help="Subsample the 76 real target-frame queries per section by this "
             "stride, to keep the Monte-Carlo FOV overlap cost down (it's the "
             "same stochastic function the real system calls, just re-run here "
             "offline).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000,
        help="Monte-Carlo sample count per FOV-overlap call (real system default "
             "is 5000; lower trades fidelity for speed in this offline sweep).",
    )
    args = parser.parse_args()

    from dataset.poses import load_c2ws_from_json

    gt_dino_features = None
    if args.manifest is not None:
        if args.row is None:
            parser.error("--manifest requires --row")
        items = load_manifest(args.manifest, selected_rows={args.row})
        if not items:
            parser.error(f"row {args.row} not found in {args.manifest}")
        item = items[0]
        pose_json = Path(item["pose_path"])
        start_frame = int(item["start_frame"])
        num_frames = int(item["num_frames"])
        print(f"Phase 2: manifest row {args.row}, scene={item.get('scene')}, "
              f"loading ground-truth frames for DINO content quality...")
        gt_dino_features = load_gt_content_features(item, args.dataset_root, args.gt_device)
        print(f"Loaded {len(gt_dino_features)} ground-truth DINO features.")
    else:
        if args.pose_json is None or args.start_frame is None or args.num_frames is None:
            parser.error("Phase 1 requires --pose_json, --start_frame, and --num_frames")
        pose_json = args.pose_json
        start_frame = args.start_frame
        num_frames = args.num_frames
        print("Phase 1: pose-only mode -- no video features, no GPU needed.")

    c2ws = load_c2ws_from_json(json_path=pose_json, start_frame=start_frame, num_frames=num_frames)
    print(f"Loaded {len(c2ws)} poses from {pose_json}")
    print(f"Budget: {args.budget}  Query stride: {args.query_stride}  Monte-Carlo samples: {args.num_samples}")

    result = simulate_selector_aware(
        c2ws, total_frames=len(c2ws), budget=args.budget,
        query_stride=args.query_stride, num_samples=args.num_samples,
        gt_dino_features=gt_dino_features,
    )

    for key, value in result.items():
        if key == "retained_frames":
            continue
        print(f"  {key}: {value}")

    print(
        "\nCompare final_age_p90/survival_gini against the same axis from "
        "mce_offline_selection_sweep.py (SLAM/RI target: age p90 ~1100-1600)."
    )
    if result["mode"] == "phase1_pose_only":
        if result["positive_marginal_fraction"] not in (0.0, None):
            print(
                f"\nWARNING: positive_marginal_fraction={result['positive_marginal_fraction']:.4f} "
                "should be mathematically impossible in Phase 1 (K=s_native, monotone by "
                "construction). A nonzero value means there's a bug -- do not trust these "
                "numbers until found."
            )
        else:
            print("\npositive_marginal_fraction=0.0, as expected in Phase 1 (correctness check, not a finding).")
    else:
        frac = result["positive_marginal_fraction"]
        print(
            f"\npositive_marginal_fraction={frac:.4f} in Phase 2 -- this IS the real signal. "
            "A value near 0 means this dataset's confined/orbiting trajectories rarely let "
            "pose-only selection get fooled into picking a genuine content distractor at this "
            "budget (a real, if less exciting, finding). A value clearly above 0 is direct, "
            "quantified evidence that the retrieval-gap mechanism can be corrected by an "
            "eviction rule that accounts for what the real selector will do -- the core "
            "hypothesis this whole diagnostic chain has been chasing."
        )


if __name__ == "__main__":
    main()
