"""Offline replay of a retriever-conditioned coverage objective (U_sel), as
opposed to MCE's abstract oracle coverage -- no video generation, no video
features, no GPU required.

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
that the coverage-side fix alone isn't sufficient, motivating this different
axis.

U_sel(M) = sum_q w_q * K(q, r(q;M)), where
    r(q;M) = argmax_{m in M} s_native(q,m)
is EXACTLY what MemCam's real context-selection step would pick -- this
reuses diffsynth.models.wan_video_overlap.calculate_overlap_from_c2w
directly (the same function, same FOV_HALF_H/V, same radius, called the same
way as wan_video_memcam.py's real "Selecting context frames" block). Not an
approximation of the real selector: the real selector.

IMPORTANT CORRECTION (caught by hand-testing before this was used for
anything): this first version sets K = s_native (the same geometric
FOV-overlap function used for selection). That makes U_sel mathematically
IDENTICAL in structure to the facility-location objective
(sum_q w_q max_{m in M} K(q,m)) -- and max-reductions are monotone by
construction: removing a candidate can only decrease or maintain each
query's max, never increase it. Verified directly: for any matrix and any
removed column, delta = U_sel(M-i) - U_sel(M) is always <= 0, never
positive. So THIS version is, like MCE, monotone -- it does NOT and
CANNOT represent the "distractor" phenomenon (a candidate winning the
selector's argmax while making outcomes worse), because that requires the
selector's own criterion (s_native) to genuinely disagree with true quality
(K) -- i.e. K != s_native. With K = s_native, whatever wins the argmax is,
by definition, "best" according to the only quality signal in the
objective, so there is nothing for reverse deletion to correct.

What THIS version still is: a legitimate, more faithful alternative to MCE
-- it evaluates coverage through the real selector's exact query structure
(actual future target-frame poses) and exact scoring function
(calculate_overlap_from_c2w itself), instead of MCE's medoid-compressed
Q_hist. That's closer to WorldMem's diagnosed "comparison breadth" gap in
MCE (medoid-only comparison loses resolution vs. RI/SLAM's full-pairwise
scoring). Worth keeping as a candidate baseline in its own right.

To actually test the distractor/retrieval-gap hypothesis this tool was
originally meant to test, K needs to be a genuine ground-truth quality
signal DIFFERENT from s_native (e.g. DINO distance to the true target view,
matching unbounded_failure_decomposition_180s's oracle methodology) -- not
built in this version. That's the real next step, not an optional
extension.

Usage:
    python utils/selector_aware_offline_sweep.py \
        --pose_json /path/to/scene.json \
        --start_frame 1368 \
        --num_frames 1825 \
        --budget 32
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffsynth.models.wan_video_overlap import calculate_overlap_from_c2w  # noqa: E402

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


def build_score_matrix(c2ws, query_frame_indices, candidate_frame_indices, num_samples):
    """s_native(q, m) for every (query, candidate) pair -- exactly the real
    selector's per-pair computation, cached once per section so the reverse-
    deletion loop below never re-touches the (stochastic, Monte-Carlo) FOV
    overlap function again, only cheap numpy max-reductions over it."""
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


def u_sel(matrix, weights, active_mask):
    if not np.any(active_mask):
        return 0.0
    active_max = matrix[:, active_mask].max(axis=1)
    return float(np.sum(weights * active_max))


def reverse_delete_selector_aware(matrix, weights, candidate_frame_indices, budget, forced_keep_frames):
    """Same reverse-deletion shape as MCE's Algorithm 1, but scored under
    U_sel instead of noisy-OR coverage. Evicts argmin delta_i, where
    delta_i = U_sel(active) - U_sel(active - i); delta_i can be negative
    (evicting i is free or beneficial) since U_sel is not monotone."""
    num_candidates = len(candidate_frame_indices)
    active_mask = np.ones(num_candidates, dtype=bool)
    forced_positions = {
        position for position, frame_idx in enumerate(candidate_frame_indices)
        if frame_idx in forced_keep_frames
    }
    removal_order = []

    selected_limit = min(int(budget), num_candidates)
    while int(active_mask.sum()) > selected_limit:
        base_value = u_sel(matrix, weights, active_mask)
        removable = [p for p in range(num_candidates) if active_mask[p] and p not in forced_positions]
        if not removable:
            break
        best_delta = None
        best_position = None
        for position in removable:
            trial_mask = active_mask.copy()
            trial_mask[position] = False
            delta = u_sel(matrix, weights, trial_mask) - base_value
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_position = position
        active_mask[best_position] = False
        removal_order.append((candidate_frame_indices[best_position], best_delta))

    retained = [candidate_frame_indices[p] for p in range(num_candidates) if active_mask[p]]
    return retained, removal_order


def simulate_selector_aware(c2ws, total_frames, budget, query_stride, num_samples):
    sections = section_boundaries(total_frames)
    memory = []  # list of frame indices currently retained
    admitted = set()
    alive_since = {}
    survival_records = []
    negative_marginal_evictions = 0
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

        matrix = build_score_matrix(c2ws, query_frame_indices, prospective_memory, num_samples)
        forced_keep_frames = {0, section_end_frame}
        retained, removal_order = reverse_delete_selector_aware(
            matrix, weights, prospective_memory, budget, forced_keep_frames
        )

        evicted = set(prospective_memory) - set(retained)
        for frame_idx, delta in removal_order:
            total_evictions += 1
            if delta < 0:
                negative_marginal_evictions += 1
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
        "retained_frames": sorted(memory),
        "final_age_median": float(np.median(final_ages)) if final_ages else None,
        "final_age_p90": float(np.percentile(final_ages, 90)) if final_ages else None,
        "survival_sections_mean": float(survival.mean()) if survival.size else None,
        "survival_sections_p90": float(np.percentile(survival, 90)) if survival.size else None,
        "survival_gini": gini(survival) if survival.size else None,
        "unique_frames_ever_admitted": len(admitted),
        "num_sections": len(sections),
        "total_evictions": total_evictions,
        "negative_marginal_evictions": negative_marginal_evictions,
        "negative_marginal_fraction": (
            negative_marginal_evictions / total_evictions if total_evictions else None
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
    parser.add_argument("--pose_json", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument(
        "--query_stride", type=int, default=4,
        help="Subsample the 76 real target-frame queries per section by this "
             "stride, to keep the Monte-Carlo FOV overlap cost down (it's the "
             "same stochastic function the real system calls, just re-run here "
             "offline -- no video features or GPU needed, but it's not free).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000,
        help="Monte-Carlo sample count per FOV-overlap call (real system default "
             "is 5000; lower trades fidelity for speed in this offline sweep).",
    )
    args = parser.parse_args()

    from dataset.poses import load_c2ws_from_json

    c2ws = load_c2ws_from_json(
        json_path=args.pose_json, start_frame=args.start_frame, num_frames=args.num_frames
    )
    print(f"Loaded {len(c2ws)} poses from {args.pose_json}")
    print(f"Budget: {args.budget}  Query stride: {args.query_stride}  Monte-Carlo samples: {args.num_samples}")
    print("(Pose-only -- no video features, no GPU needed for this objective's Phase-1 scope.)")

    result = simulate_selector_aware(
        c2ws, total_frames=len(c2ws), budget=args.budget,
        query_stride=args.query_stride, num_samples=args.num_samples,
    )

    for key, value in result.items():
        if key == "retained_frames":
            continue
        print(f"  {key}: {value}")

    print(
        "\nCompare final_age_p90/survival_gini against the same axis from "
        "mce_offline_selection_sweep.py (SLAM/RI target: age p90 ~1100-1600)."
    )
    if result["negative_marginal_fraction"] not in (0.0, None):
        print(
            f"\nWARNING: negative_marginal_fraction={result['negative_marginal_fraction']:.4f} "
            "should be mathematically impossible under K=s_native (max-reduction is "
            "monotone -- see the module docstring's correction). A nonzero value here "
            "means there's a bug in this implementation, not a real finding -- do not "
            "trust these numbers until that's found."
        )
    else:
        print(
            "\nnegative_marginal_fraction=0.0, as it must be under K=s_native (monotone "
            "by construction -- this is a correctness check, not a finding; see the "
            "module docstring for why this version can't test the distractor hypothesis)."
        )


if __name__ == "__main__":
    main()
