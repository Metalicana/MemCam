"""Diagnose (and optionally fix) a fixed camera-axis-convention offset between
CUT3R's predicted rotations and this project's UE-convention ground truth.

Motivation: every run in the CUT3R pilot showed rotation_error_deg_mean ~78-83
degrees -- close to the ~90 degree expected value for two *uncorrelated*
random rotations -- uniformly across every memory policy, including baseline
and unbounded. That uniformity (not policy-dependent) is the signature of a
systematic axis-convention mismatch, not a real per-policy difference:
evaluate_cut3r_camera_metrics.py's rotation_error_deg() compares predicted vs
GT rotation matrices with no correction for a fixed local-camera-axis
relabeling between CUT3R's convention (commonly OpenCV: x=right, y=down,
z=forward) and this project's UE convention (x=forward, y=right, z=up; see
dataset/poses.py:compute_c2w_matrix). translation_error_sim3 doesn't show
this problem because its Umeyama fit includes a rotation term that can absorb
a *global* axis difference -- but rotation_error_deg has no such correction.

This script does NOT assume a specific convention. It empirically searches
the 24 proper (det=+1) signed-permutation matrices -- i.e. every possible
"this axis becomes that axis, possibly flipped" relabeling -- for the one
that best explains the observed pred-vs-GT rotation mismatch, pooled across
every already-scored CUT3R reconstruction. If a single fixed correction
collapses the error from ~80 degrees to something small and consistent, that
confirms (and fixes) a convention bug. If no candidate helps, the error is
likely a genuine reconstruction/alignment problem, not a convention bug, and
evaluate_cut3r_camera_metrics.py's rotation numbers should stay flagged as
unreliable rather than "corrected" with a guess.

No new CUT3R runs are needed -- this reuses whatever reconstructions already
exist under --cut3r_dir.

Usage:
    python utils/diagnose_cut3r_rotation_convention.py \
        --cut3r_dir ~/memcam_results/context_memory_60s/cut3r_pose_recon_pilot \
        --runs baseline,fifo_b32,slam_b32_covisibility,ri_b32_dino_rgb,future_view_coverage_b32_pilot,mce_b32_pilot,mce_b32_lambda08_pilot,mce_b32_lambda1_pilot
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_cut3r_camera_metrics import (
    discover_metadata_files,
    load_gt_poses,
    load_json,
    load_predicted_poses,
    relative_poses,
    rotation_error_deg,
)


def signed_permutation_rotations():
    """All 24 proper (det=+1) signed 3x3 permutation matrices -- every way to
    relabel/flip the three axes that still preserves handedness of a
    right-handed frame's *internal* structure (the rotation group of the
    cube). Covers exactly the "which local axis is which, with what sign"
    ambiguities that differ between camera conventions like UE vs OpenCV."""
    matrices = []
    for perm in itertools.permutations(range(3)):
        base = np.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            candidate = base * np.array(signs)[:, None]
            if abs(np.linalg.det(candidate) - 1.0) < 1e-9:
                matrices.append(candidate)
    assert len(matrices) == 24, f"expected 24 proper signed permutations, got {len(matrices)}"
    return matrices


def pool_relative_rotations(cut3r_dir, runs, rows, durations, dataset_root):
    pred_rots = []
    gt_rots = []
    per_reconstruction_counts = []
    per_reconstruction_labels = []
    run_filter = runs.split(",") if runs else None
    row_filter = {int(r) for r in rows.split(",")} if rows else None
    duration_filter = {int(d) for d in durations.split(",")} if durations else None

    for metadata_path in discover_metadata_files(cut3r_dir, runs=run_filter):
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        if row_filter is not None and int(metadata["manifest_row"]) not in row_filter:
            continue
        if duration_filter and int(metadata["duration_sec"]) not in duration_filter:
            continue
        try:
            pred = load_predicted_poses(metadata_path.parent)
            gt = load_gt_poses(metadata, dataset_root)
        except Exception as exc:
            print(f"  skip {metadata_path}: {exc!r}")
            continue
        count = min(len(pred), len(gt))
        if count < 2:
            continue
        pred_rel = relative_poses(pred[:count])
        gt_rel = relative_poses(gt[:count])
        pred_rots.append(pred_rel[:, :3, :3])
        gt_rots.append(gt_rel[:, :3, :3])
        per_reconstruction_counts.append(count)
        per_reconstruction_labels.append(f"{metadata.get('run_name')}/row{metadata.get('manifest_row')}")

    if not pred_rots:
        raise RuntimeError(f"No scoreable CUT3R reconstructions found under {cut3r_dir}")

    return (
        np.concatenate(pred_rots, axis=0),
        np.concatenate(gt_rots, axis=0),
        per_reconstruction_counts,
        per_reconstruction_labels,
    )


def per_reconstruction_breakdown(errors, counts, labels):
    """Split a pooled per-frame-pair error array back out by reconstruction
    and report each one's mean/p90, sorted worst-first -- distinguishes "a
    few bad videos are dragging up the mean" from "every video has a
    consistent modest residual"."""
    rows = []
    start = 0
    for count, label in zip(counts, labels):
        chunk = errors[start:start + count]
        start += count
        rows.append((label, float(chunk.mean()), float(np.percentile(chunk, 90)), count))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def axis_angle_vector(rot):
    """v = [R32-R23, R13-R31, R21-R12] = 2*sin(theta)*axis for each rotation
    in a stack. This is linear in the conjugation model: if pred = C.T @ gt @
    C for a fixed C, then axis_angle_vector(pred) = C.T @ axis_angle_vector(gt)
    exactly (conjugation preserves rotation angle and rotates the axis by
    C.T), which turns "fit a fixed conjugating rotation" into a standard
    Wahba/orthogonal-Procrustes problem solvable in closed form via SVD --
    the same rotation-fitting math as fit_umeyama() in
    evaluate_cut3r_camera_metrics.py, just applied to axis vectors instead of
    3D points, and with no translation/scale term since these are directions.
    """
    return np.stack(
        [
            rot[:, 2, 1] - rot[:, 1, 2],
            rot[:, 0, 2] - rot[:, 2, 0],
            rot[:, 1, 0] - rot[:, 0, 1],
        ],
        axis=1,
    )


def fit_continuous_conjugation(pred_rot, gt_rot):
    """Closed-form fit for the conjugation model pred ~= C.T @ gt @ C, via
    Wahba's problem on axis-angle vectors (v_pred ~= C.T @ v_gt exactly under
    that model). Returns M = C.T -- feed this directly as `matrix` to
    evaluate_candidate(..., mode="conjugate"), which computes M.T @ pred @ M
    = C @ pred @ C.T = gt when the fit is exact. (Returning C itself here and
    conjugating by C, instead of C.T, silently undoes nothing -- verified by
    round-tripping a synthetic injected rotation before trusting this.)"""
    v_pred = axis_angle_vector(pred_rot)
    v_gt = axis_angle_vector(gt_rot)

    # Solve for M=C.T minimizing sum ||v_pred - M @ v_gt||^2 (standard Wahba/
    # orthogonal Procrustes, no centering -- these are directions, not points).
    h_mat = v_pred.T @ v_gt
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    sign = np.sign(np.linalg.det(u_mat) * np.linalg.det(vt_mat))
    correction = np.eye(3)
    correction[-1, -1] = sign if sign != 0 else 1.0
    m_matrix = u_mat @ correction @ vt_mat
    return m_matrix, v_pred, v_gt


def evaluate_candidate(pred_rot, gt_rot, mode, matrix):
    if mode == "right":  # local-camera-axis relabeling: R_pred @ M
        corrected = pred_rot @ matrix[None, :, :]
    elif mode == "left":  # world-frame relabeling: M @ R_pred
        corrected = matrix[None, :, :] @ pred_rot
    elif mode == "conjugate":  # both: M.T @ R_pred @ M
        corrected = matrix.T[None, :, :] @ pred_rot @ matrix[None, :, :]
    else:
        raise ValueError(mode)
    errors = rotation_error_deg(corrected, gt_rot)
    return float(errors.mean()), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut3r_dir", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--durations", type=str, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    args = parser.parse_args()

    print(f"Pooling relative rotations from {args.cut3r_dir} ...")
    pred_rot, gt_rot, counts, labels = pool_relative_rotations(
        args.cut3r_dir, args.runs, args.rows, args.durations, args.dataset_root
    )
    print(f"Pooled {pred_rot.shape[0]} frame-pairs from {len(counts)} reconstructions "
          f"(per-reconstruction counts: {counts})")

    baseline_errors = rotation_error_deg(pred_rot, gt_rot)
    print(f"\nUncorrected rotation_error_deg: mean={baseline_errors.mean():.2f}  "
          f"median={np.median(baseline_errors):.2f}  p90={np.percentile(baseline_errors, 90):.2f}")
    print("(For reference: two *uncorrelated* random 3D rotations average ~90 deg apart, "
          "so a mean this close to 90 is consistent with -- but doesn't prove -- a convention bug.)")

    candidates = signed_permutation_rotations()
    results = []
    for mode in ("right", "left", "conjugate"):
        for matrix in candidates:
            mean_err, _ = evaluate_candidate(pred_rot, gt_rot, mode, matrix)
            results.append((mean_err, mode, matrix))
    results.sort(key=lambda item: item[0])

    print(f"\nSearched {len(results)} (mode, axis-relabeling) candidates. Top 5:")
    for mean_err, mode, matrix in results[:5]:
        print(f"  mode={mode:9s} mean_rotation_error_deg={mean_err:6.2f}  matrix=\n{matrix}")

    best_err, best_mode, best_matrix = results[0]
    print(f"\nBest discrete candidate: mode={best_mode}, mean error {baseline_errors.mean():.2f} -> {best_err:.2f} deg")

    print("\nFitting a continuous conjugating rotation C (pred ~= C.T @ gt @ C) via "
          "closed-form Wahba/Procrustes on axis-angle vectors ...")
    c_matrix, v_pred, v_gt = fit_continuous_conjugation(pred_rot, gt_rot)
    continuous_err, continuous_errors = evaluate_candidate(pred_rot, gt_rot, "conjugate", c_matrix)
    print(f"Continuous conjugation: mean error {baseline_errors.mean():.2f} -> {continuous_err:.2f} deg "
          f"(median={np.median(continuous_errors):.2f}, p90={np.percentile(continuous_errors, 90):.2f})")
    print(f"Fitted C:\n{c_matrix}")

    print("\nPer-reconstruction breakdown after the continuous correction (worst first):")
    breakdown = per_reconstruction_breakdown(continuous_errors, counts, labels)
    for label, rec_mean, rec_p90, count in breakdown:
        print(f"  {label:45s} mean={rec_mean:6.2f}  p90={rec_p90:6.2f}  frames={count}")
    worst_mean = breakdown[0][1]
    best_rec_mean = breakdown[-1][1]
    if worst_mean > 3.0 * max(best_rec_mean, 1.0):
        print(
            f"\nThe corrected error is concentrated in specific reconstructions ({breakdown[0][0]} "
            f"and similar, mean={worst_mean:.1f}) rather than spread evenly (best reconstruction "
            f"mean={best_rec_mean:.1f}). That points to per-video tracking drift/failure in those "
            "specific CUT3R reconstructions, not a residual convention problem -- worth inspecting "
            "those videos/reconstructions directly rather than refining the correction further."
        )
    else:
        print(
            "\nCorrected error is roughly even across reconstructions -- consistent with a real, "
            "if imperfect, residual (e.g. per-frame estimation noise) rather than a few broken videos."
        )

    final_err = min(best_err, continuous_err)
    final_label = "discrete axis-relabeling" if best_err <= continuous_err else "continuous conjugation"
    final_matrix = best_matrix if best_err <= continuous_err else c_matrix
    final_mode = best_mode if best_err <= continuous_err else "conjugate"

    if final_err < 15.0:
        print(
            f"\nThis looks like a real convention fix ({final_label}, error collapses to a "
            "small, sane value). Recommend baking this correction into "
            "evaluate_cut3r_camera_metrics.py's rotation_error_deg() call as a fixed one-time "
            "calibration, then re-scoring. Winning matrix printed above under that label."
        )
    elif final_err < baseline_errors.mean() - 20.0:
        print(
            f"\nPartial improvement only (best: {final_label}, {baseline_errors.mean():.2f} -> "
            f"{final_err:.2f} deg). A single fixed rotation doesn't fully explain the mismatch -- "
            "this could mean the offset varies across runs/rows (not truly fixed), or there is a "
            "second-order effect (e.g. frame-index drift) on top of the convention bug. Worth "
            "checking whether per-run (rather than pooled) continuous fits are consistent with "
            "each other before trusting a single global correction."
        )
    else:
        print(
            "\nNeither the discrete search nor the continuous conjugation fit meaningfully helps. "
            "This is NOT a simple convention-relabeling bug -- treat the large rotation error as a "
            "genuine reconstruction/alignment issue (or a bug elsewhere, e.g. frame-index "
            "misalignment between predicted and GT poses) and keep rotation_error_deg/worldscore "
            "flagged unreliable until that's found."
        )


if __name__ == "__main__":
    main()
