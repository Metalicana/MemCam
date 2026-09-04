#!/usr/bin/env python3
"""Validate a pose-scheduled, locally matched anchor chain offline.

The proposed online signal uses only generated images and known camera poses.
An established anchor is kept while the camera view overlaps it. Before that
overlap disappears, a newer frame is compared with the anchor using local SIFT
matches and the known relative camera motion. Frames with no predicted overlap
are explicitly outside the gate's scope; they are not labeled unreliable.

Exact-index dataset PSNR/SSIM are used only as held-out evaluation labels. This
script does not modify generation or inject a memory policy.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.poses import load_c2ws_from_json  # noqa: E402
from utils.analyze_view_anchor_hysteresis import (  # noqa: E402
    camera_trajectory_similarity,
)
from utils.calibrate_causal_consistency_gate import (  # noqa: E402
    load_manifest,
    read_gt_frames,
    read_video_frames,
)
from utils.calibrate_frame_quality_estimators import (  # noqa: E402
    add_within_trajectory_labels,
    estimator_rows,
    estimator_run_rows,
    fmt,
    quality_auc,
    trajectory_split,
)
from utils.evaluate_context_memory import frame_metrics  # noqa: E402


ESTIMATOR_FIELDS = (
    "match_support",
    "descriptor_consistency",
    "geometric_inlier_fraction",
    "geometric_support",
    "anchor_chain_score",
)

SEVERITY_QUANTILES = (0.05, 0.10, 0.20)

# Dataset c2w rotations use Unreal coordinates: X forward, Y right, Z up.
# Image geometry uses OpenCV coordinates: x right, y down, z forward.
UE_FROM_CV = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def camera_intrinsics(width, height, fov_half_h=45.0, fov_half_v=30.0):
    """Return a pinhole K matching MemCam's nominal horizontal/vertical FOV."""
    width = float(width)
    height = float(height)
    fx = 0.5 * width / math.tan(math.radians(float(fov_half_h)))
    fy = 0.5 * height / math.tan(math.radians(float(fov_half_v)))
    return np.asarray(
        [[fx, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def cv_camera_rotation(c2w):
    """Convert a dataset camera-to-world rotation to OpenCV camera axes."""
    return np.asarray(c2w, dtype=np.float64)[:3, :3] @ UE_FROM_CV


def relative_camera_motion(anchor_c2w, current_c2w):
    """Transform points from the anchor camera into the current camera."""
    anchor = np.asarray(anchor_c2w, dtype=np.float64)
    current = np.asarray(current_c2w, dtype=np.float64)
    r_anchor = cv_camera_rotation(anchor)
    r_current = cv_camera_rotation(current)
    rotation = r_current.T @ r_anchor
    translation = r_current.T @ (anchor[:3, 3] - current[:3, 3])
    return rotation, translation


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def transform_points(homography, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    valid = np.abs(mapped[:, 2]) > 1e-12
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = mapped[valid, :2] / mapped[valid, 2:3]
    return output


def rotation_transfer_errors(anchor_points, current_points, rotation, intrinsics):
    homography = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    predicted = transform_points(homography, anchor_points)
    return np.linalg.norm(predicted - np.asarray(current_points), axis=1)


def epipolar_errors(anchor_points, current_points, rotation, translation, intrinsics):
    """Symmetric point-to-epipolar-line distance in pixels."""
    inverse_k = np.linalg.inv(intrinsics)
    fundamental = inverse_k.T @ skew(translation) @ rotation @ inverse_k
    first = np.column_stack([anchor_points, np.ones(len(anchor_points))])
    second = np.column_stack([current_points, np.ones(len(current_points))])
    lines_second = (fundamental @ first.T).T
    lines_first = (fundamental.T @ second.T).T
    residual = np.abs(np.sum(second * lines_second, axis=1))
    denom_second = np.linalg.norm(lines_second[:, :2], axis=1)
    denom_first = np.linalg.norm(lines_first[:, :2], axis=1)
    valid = (denom_second > 1e-12) & (denom_first > 1e-12)
    errors = np.full(len(first), np.inf, dtype=np.float64)
    errors[valid] = 0.5 * residual[valid] * (
        1.0 / denom_second[valid] + 1.0 / denom_first[valid]
    )
    return errors


def correspondence_errors(
    anchor_points,
    current_points,
    anchor_c2w,
    current_c2w,
    intrinsics,
    pure_rotation_translation=1e-3,
):
    rotation, translation = relative_camera_motion(anchor_c2w, current_c2w)
    if np.linalg.norm(translation) <= float(pure_rotation_translation):
        return (
            rotation_transfer_errors(
                anchor_points, current_points, rotation, intrinsics
            ),
            "rotation",
        )
    return (
        epipolar_errors(
            anchor_points,
            current_points,
            rotation,
            translation,
            intrinsics,
        ),
        "epipolar",
    )


def pair_view_similarity(c2ws, first, second):
    return float(
        camera_trajectory_similarity(c2ws, [int(first)], [int(second)])[0, 0]
    )


def build_pose_anchor_schedule(
    c2ws,
    frame_stride=19,
    handoff_overlap=0.50,
    min_pair_overlap=0.15,
):
    """Build anchor/current pairs without looking at generated image quality.

    A handoff is scheduled while the current frame still overlaps the active
    anchor. If the sample stride skips past that window, the previous sampled
    frame is used as an emergency bridge. A true discontinuity becomes a
    provisional anchor and is excluded from reliability scoring.
    """
    c2ws = np.asarray(c2ws, dtype=np.float64)
    if len(c2ws) < 2:
        return []
    active_anchor = 0
    previous_sample = 0
    output = []
    for current in range(int(frame_stride), len(c2ws), int(frame_stride)):
        anchor = active_anchor
        overlap = pair_view_similarity(c2ws, anchor, current)
        emergency_bridge = False
        if overlap < float(min_pair_overlap) and previous_sample != anchor:
            bridge_overlap = pair_view_similarity(c2ws, previous_sample, current)
            if bridge_overlap >= float(min_pair_overlap):
                anchor = previous_sample
                overlap = bridge_overlap
                active_anchor = anchor
                emergency_bridge = True

        evaluable = overlap >= float(min_pair_overlap)
        handoff = evaluable and overlap <= float(handoff_overlap)
        output.append(
            {
                "anchor_frame": int(anchor),
                "current_frame": int(current),
                "pose_overlap": float(overlap),
                "evaluable": bool(evaluable),
                "scheduled_handoff": bool(handoff),
                "emergency_bridge": bool(emergency_bridge),
                "provisional_restart": bool(not evaluable),
            }
        )
        if handoff or not evaluable:
            active_anchor = current
        previous_sample = current
    return output


def create_sift(max_features):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install the repository requirement "
            "`opencv-contrib-python` in the active environment."
        ) from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT_create")
    return cv2, cv2.SIFT_create(nfeatures=int(max_features))


def extract_sift(cv2, extractor, image):
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = extractor.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return points, np.asarray(descriptors, dtype=np.float32)


def ratio_matches(cv2, first, second, ratio_threshold):
    if len(first) < 2 or len(second) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    output = {}
    for neighbors in matcher.knnMatch(first, second, k=2):
        if len(neighbors) != 2:
            continue
        best, runner_up = neighbors
        if runner_up.distance <= 1e-12:
            continue
        ratio = float(best.distance / runner_up.distance)
        if ratio < float(ratio_threshold):
            output[int(best.queryIdx)] = (int(best.trainIdx), ratio)
    return output


def mutual_ratio_matches(cv2, anchor_descriptors, current_descriptors, ratio_threshold):
    forward = ratio_matches(
        cv2, anchor_descriptors, current_descriptors, ratio_threshold
    )
    backward = ratio_matches(
        cv2, current_descriptors, anchor_descriptors, ratio_threshold
    )
    matches = []
    for anchor_idx, (current_idx, forward_ratio) in forward.items():
        reverse = backward.get(current_idx)
        if reverse is None or reverse[0] != anchor_idx:
            continue
        matches.append(
            (anchor_idx, current_idx, 0.5 * (forward_ratio + reverse[1]))
        )
    return matches


def grid_coverage(points, mask, width, height, grid_size=4):
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    selected = points[mask]
    xs = np.clip((selected[:, 0] / max(float(width), 1.0) * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip((selected[:, 1] / max(float(height), 1.0) * grid_size).astype(int), 0, grid_size - 1)
    occupied = len(set(zip(xs.tolist(), ys.tolist())))
    return float(occupied / float(grid_size * grid_size))


def score_anchor_pair(
    cv2,
    anchor_features,
    current_features,
    anchor_c2w,
    current_c2w,
    image_shape,
    ratio_threshold=0.80,
    inlier_threshold_px=4.0,
    min_support_matches=12,
    fov_half_h=45.0,
    fov_half_v=30.0,
):
    anchor_points, anchor_descriptors = anchor_features
    current_points, current_descriptors = current_features
    matches = mutual_ratio_matches(
        cv2, anchor_descriptors, current_descriptors, ratio_threshold
    )
    match_count = len(matches)
    support = min(float(match_count) / max(int(min_support_matches), 1), 1.0)
    if not matches:
        return {
            "match_count": 0,
            "match_support": 0.0,
            "descriptor_consistency": 0.0,
            "geometry_mode": "unavailable",
            "geometric_inlier_count": 0,
            "geometric_inlier_fraction": 0.0,
            "geometric_median_error_px": None,
            "geometric_spatial_coverage": 0.0,
            "geometric_support": 0.0,
            "anchor_chain_score": 0.0,
        }

    anchor_indices = [match[0] for match in matches]
    current_indices = [match[1] for match in matches]
    ratios = np.asarray([match[2] for match in matches], dtype=np.float64)
    matched_anchor = anchor_points[anchor_indices]
    matched_current = current_points[current_indices]
    height, width = image_shape[:2]
    intrinsics = camera_intrinsics(width, height, fov_half_h, fov_half_v)
    errors, mode = correspondence_errors(
        matched_anchor,
        matched_current,
        anchor_c2w,
        current_c2w,
        intrinsics,
    )
    finite = np.isfinite(errors)
    inliers = finite & (errors <= float(inlier_threshold_px))
    inlier_count = int(np.sum(inliers))
    inlier_fraction = float(inlier_count / match_count)
    spatial_coverage = grid_coverage(
        matched_anchor, inliers, width=width, height=height
    )
    descriptor_consistency = float(np.mean(1.0 - ratios))
    geometric_support = float(inlier_fraction * support)
    # Match support prevents a tiny accidental correspondence set from earning
    # a perfect score. Spatial coverage discounts matches confined to one patch.
    chain_score = float(
        geometric_support
        * max(descriptor_consistency, 0.0)
        * (0.5 + 0.5 * spatial_coverage)
    )
    return {
        "match_count": match_count,
        "match_support": support,
        "descriptor_consistency": descriptor_consistency,
        "geometry_mode": mode,
        "geometric_inlier_count": inlier_count,
        "geometric_inlier_fraction": inlier_fraction,
        "geometric_median_error_px": (
            float(np.median(errors[finite])) if np.any(finite) else None
        ),
        "geometric_spatial_coverage": spatial_coverage,
        "geometric_support": geometric_support,
        "anchor_chain_score": chain_score,
    }


def add_transition_labels(rows, bad_quantile):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["row"]))].append(row)
    for group in grouped.values():
        psnr_delta = np.asarray(
            [row["current_psnr_db"] - row["anchor_psnr_db"] for row in group]
        )
        ssim_delta = np.asarray(
            [row["current_ssim"] - row["anchor_ssim"] for row in group]
        )
        psnr_order = np.argsort(np.argsort(psnr_delta, kind="stable"), kind="stable")
        ssim_order = np.argsort(np.argsort(ssim_delta, kind="stable"), kind="stable")
        denominator = max(len(group) - 1, 1)
        percentile = 0.5 * (psnr_order + ssim_order) / denominator
        cutoff = float(np.quantile(percentile, float(bad_quantile)))
        for row, value in zip(group, percentile):
            row["gt_transition_quality_percentile"] = float(value)
            row["gt_bad_transition"] = bool(value <= cutoff)


def add_current_quality_aliases(rows):
    """Expose current-frame fidelity under the shared estimator schema."""
    for row in rows:
        row["psnr_db"] = float(row["current_psnr_db"])
        row["ssim"] = float(row["current_ssim"])


def add_anchor_fidelity_labels(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["row"]))].append(row)
    for group in grouped.values():
        values = np.asarray(
            [row["anchor_psnr_db"] + 20.0 * row["anchor_ssim"] for row in group]
        )
        cutoff = float(np.median(values))
        for row, value in zip(group, values):
            row["anchor_low_fidelity"] = bool(value <= cutoff)


def summaries_for_transition(
    rows, train_rows, test_rows, args
):
    transformed = []
    for row in rows:
        item = copy.deepcopy(row)
        item["gt_bad_frame"] = bool(item["gt_bad_transition"])
        item["gt_quality_percentile"] = float(
            item["gt_transition_quality_percentile"]
        )
        transformed.append(item)
    return estimator_rows(
        transformed,
        ESTIMATOR_FIELDS,
        train_rows,
        test_rows,
        args.bootstrap_repeats,
        args.split_seed,
        args.max_train_clean_false_reject,
    )


def failure_severity_sweep(rows, train_rows, test_rows, args):
    """Measure the same held-out gate against increasingly severe failures.

    This is diagnostic only. The predeclared injection decision still uses
    ``args.bad_quantile`` and is not selected from this sweep.
    """
    output = []
    for bad_quantile in SEVERITY_QUANTILES:
        labeled = copy.deepcopy(rows)
        add_within_trajectory_labels(labeled, bad_quantile)
        add_transition_labels(labeled, bad_quantile)
        absolute = estimator_rows(
            labeled,
            ESTIMATOR_FIELDS,
            train_rows,
            test_rows,
            args.bootstrap_repeats,
            args.split_seed,
            args.max_train_clean_false_reject,
        )
        transition = summaries_for_transition(
            labeled, train_rows, test_rows, args
        )
        transition_by_name = {
            row["estimator"]: row for row in transition
        }
        for row in absolute:
            transition_row = transition_by_name.get(row["estimator"])
            output.append(
                {
                    "bad_quantile": bad_quantile,
                    "estimator": row["estimator"],
                    "test_frames": row["test_frames"],
                    "absolute_auc": row["test_quality_auc"],
                    "transition_auc": (
                        transition_row["test_quality_auc"]
                        if transition_row is not None
                        else None
                    ),
                    "gate_bad_precision": row["gate_test_bad_precision"],
                    "gate_bad_recall": row["gate_test_bad_recall"],
                    "gate_clean_reject": row[
                        "gate_test_clean_false_reject_rate"
                    ],
                }
            )
    return sorted(output, key=lambda row: (row["bad_quantile"], row["estimator"]))


def anchor_strata_auc(rows, test_rows, estimator):
    output = {}
    for label, low in (("low_fidelity_anchor", True), ("high_fidelity_anchor", False)):
        selected = [
            row
            for row in rows
            if int(row["row"]) in test_rows
            and bool(row["anchor_low_fidelity"]) == low
        ]
        output[label] = {
            "samples": len(selected),
            "quality_auc": quality_auc(
                [row[estimator] for row in selected],
                [row["gt_bad_frame"] for row in selected],
            ),
        }
    return output


def make_decision(absolute, transition, descriptor, strata, coverage, args):
    checks = {
        "absolute_auc": absolute["test_quality_auc"] >= args.min_auc,
        "bad_precision": (
            absolute["gate_test_bad_precision"] is not None
            and absolute["gate_test_bad_precision"] >= args.min_bad_precision
        ),
        "bad_recall": absolute["gate_test_bad_recall"] >= args.min_bad_recall,
        "clean_reject": (
            absolute["gate_test_clean_false_reject_rate"]
            <= args.max_test_clean_false_reject
        ),
        "descriptor_gain": (
            absolute["test_quality_auc"] - descriptor["test_quality_auc"]
            >= args.min_descriptor_auc_gain
        ),
        "transition_auc": transition["test_quality_auc"] >= args.min_transition_auc,
        "low_fidelity_anchor_auc": (
            strata["low_fidelity_anchor"]["quality_auc"] is not None
            and strata["low_fidelity_anchor"]["quality_auc"]
            >= args.min_bad_anchor_auc
        ),
        "pose_coverage": coverage >= args.min_pose_coverage,
    }
    return {
        "decision": "INJECT" if all(checks.values()) else "DO_NOT_INJECT",
        "estimator": "anchor_chain_score",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def summary_by_name(rows, name):
    for row in rows:
        if row["estimator"] == name:
            return row
    raise RuntimeError(f"Estimator summary missing {name}")


def make_plot(path, absolute, transition):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_anchor_chain_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["estimator"] for row in absolute]
    absolute_by_name = {row["estimator"]: row for row in absolute}
    transition_by_name = {row["estimator"]: row for row in transition}
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        x - 0.19,
        [absolute_by_name[name]["test_quality_auc"] for name in names],
        width=0.38,
        color="#287271",
        label="absolute fidelity",
    )
    ax.bar(
        x + 0.19,
        [transition_by_name[name]["test_quality_auc"] for name in names],
        width=0.38,
        color="#E9C46A",
        label="quality drop from anchor",
    )
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(0.7, color="#9C2F2F", linestyle=":", linewidth=1)
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out quality AUC")
    ax.set_title("Can an anchor link identify a corrupted write?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path,
    absolute,
    transition,
    by_run,
    strata,
    severity_sweep,
    decision,
    args,
    train_rows,
    test_rows,
    total_scheduled,
    total_evaluable,
):
    absolute_by_name = {row["estimator"]: row for row in absolute}
    transition_by_name = {row["estimator"]: row for row in transition}
    severity_proposed = [
        row
        for row in severity_sweep
        if row["estimator"] == "anchor_chain_score"
    ]
    coverage = total_evaluable / total_scheduled if total_scheduled else 0.0
    lines = [
        "# Verified Anchor-Chain Calibration",
        "",
        "## Question",
        "",
        "Can local image correspondences that obey known camera motion identify a damaged generated frame before it becomes persistent memory?",
        "",
        "## Proposed Runtime Behavior",
        "",
        "An active anchor remains trusted while camera geometry predicts overlap. A newer frame is eligible to inherit that trust only when local feature matches agree with the known relative pose. Handoffs are scheduled before overlap disappears. Camera discontinuities abstain and start a provisional anchor; they are not treated as visual failures.",
        "",
        "## Leakage Boundary",
        "",
        "The score uses generated RGB frames, camera poses, nominal camera intrinsics, and SIFT correspondences. Exact-index GT supplies PSNR/SSIM labels only. The anchor schedule is determined from poses before image quality is inspected.",
        "",
        "## Protocol",
        "",
        f"- Runs: `{','.join(args.runs)}`.",
        f"- Scheduled links: `{total_scheduled}`; pose-evaluable links: `{total_evaluable}` (`{coverage:.3f}`).",
        f"- Frame stride: `{args.frame_stride}`.",
        f"- Handoff/minimum overlap: `{args.handoff_overlap:.2f}` / `{args.min_pair_overlap:.2f}`.",
        f"- Train trajectories: `{','.join(map(str, sorted(train_rows)))}`.",
        f"- Held-out trajectories: `{','.join(map(str, sorted(test_rows)))}`.",
        f"- Bad prevalence: bottom `{100 * args.bad_quantile:.0f}%` within each run and trajectory.",
        "",
        "## Held-Out Results",
        "",
        "| estimator | absolute AUC | transition AUC | gate precision | gate recall | clean reject |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ESTIMATOR_FIELDS:
        if name not in absolute_by_name or name not in transition_by_name:
            continue
        row = absolute_by_name[name]
        lines.append(
            f"| {name} | {fmt(row['test_quality_auc'])} | "
            f"{fmt(transition_by_name[name]['test_quality_auc'])} | "
            f"{fmt(row['gate_test_bad_precision'])} | "
            f"{fmt(row['gate_test_bad_recall'])} | "
            f"{fmt(row['gate_test_clean_false_reject_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Corrupted-Anchor Stress Test",
            "",
            f"- Low-fidelity anchors: AUC `{fmt(strata['low_fidelity_anchor']['quality_auc'])}` over `{strata['low_fidelity_anchor']['samples']}` held-out links.",
            f"- High-fidelity anchors: AUC `{fmt(strata['high_fidelity_anchor']['quality_auc'])}` over `{strata['high_fidelity_anchor']['samples']}` held-out links.",
            "",
            "## Failure-Severity Sweep",
            "",
            "This diagnostic asks whether anchor consistency detects only the most catastrophic failures even if it is weak over the broad bottom-20% label. It does not change the predeclared injection decision.",
            "",
            "| bad-frame tail | absolute AUC | transition AUC | gate precision | gate recall | clean reject |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in severity_proposed:
        lines.append(
            f"| bottom {100 * row['bad_quantile']:.0f}% | "
            f"{fmt(row['absolute_auc'])} | "
            f"{fmt(row['transition_auc'])} | "
            f"{fmt(row['gate_bad_precision'])} | "
            f"{fmt(row['gate_bad_recall'])} | "
            f"{fmt(row['gate_clean_reject'])} |"
        )
    lines.extend(
        [
            "",
            "## Injection Decision",
            "",
            f"**{decision['decision']}**",
            "",
            f"Failed checks: `{','.join(decision['failed_checks']) if decision['failed_checks'] else 'none'}`.",
            "",
            "Passing requires useful absolute-fidelity discrimination, rejection precision, recall, clean-frame preservation, improvement over descriptor matching alone, transition detection, robustness to a weak anchor, and sufficient pose coverage. A transition-only signal is not enough to inject a persistent-memory gate.",
            "",
            "## Files",
            "",
            "- `pair_scores.csv`",
            "- `estimator_summary_absolute.csv`",
            "- `estimator_summary_transition.csv`",
            "- `estimator_by_run.csv`",
            "- `estimator_severity_sweep.csv`",
            "- `decision.json`",
            "- `anchor_chain_validation.png`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=parse_list,
        default=parse_list("baseline,slam_b32_covisibility"),
    )
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--frame_stride", type=int, default=19)
    parser.add_argument("--max_samples_per_video", type=int, default=None)
    parser.add_argument("--handoff_overlap", type=float, default=0.50)
    parser.add_argument("--min_pair_overlap", type=float, default=0.15)
    parser.add_argument("--max_features", type=int, default=1200)
    parser.add_argument("--ratio_threshold", type=float, default=0.80)
    parser.add_argument("--inlier_threshold_px", type=float, default=4.0)
    parser.add_argument("--min_support_matches", type=int, default=12)
    parser.add_argument("--fov_half_h", type=float, default=45.0)
    parser.add_argument("--fov_half_v", type=float, default=30.0)
    parser.add_argument("--bad_quantile", type=float, default=0.20)
    parser.add_argument("--max_train_clean_false_reject", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--bootstrap_repeats", type=int, default=4000)
    parser.add_argument("--min_auc", type=float, default=0.70)
    parser.add_argument("--min_bad_precision", type=float, default=0.50)
    parser.add_argument("--min_bad_recall", type=float, default=0.20)
    parser.add_argument("--max_test_clean_false_reject", type=float, default=0.15)
    parser.add_argument("--min_descriptor_auc_gain", type=float, default=0.02)
    parser.add_argument("--min_transition_auc", type=float, default=0.65)
    parser.add_argument("--min_bad_anchor_auc", type=float, default=0.60)
    parser.add_argument("--min_pose_coverage", type=float, default=0.80)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride <= 0 or args.max_features <= 0:
        parser.error("frame_stride and max_features must be positive")
    if not 0.0 <= args.min_pair_overlap < args.handoff_overlap <= 1.0:
        parser.error("Require 0 <= min_pair_overlap < handoff_overlap <= 1")
    if not 0.0 < args.ratio_threshold < 1.0:
        parser.error("ratio_threshold must be between zero and one")
    if args.inlier_threshold_px <= 0 or args.min_support_matches <= 0:
        parser.error("inlier threshold and minimum support must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"strict", "output_dir"}
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("Output directory contains a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    items = load_manifest(args.manifest, args.duration)
    if not items:
        raise RuntimeError(f"No {args.duration}s manifest rows found")
    requested_rows = {int(item["_row"]) for item in items}
    train_rows, test_rows = trajectory_split(
        requested_rows, args.test_fraction, args.split_seed
    )
    cv2, extractor = create_sift(args.max_features)

    partial_path = args.output_dir / "pair_scores.partial.jsonl"
    pair_rows = load_jsonl(partial_path)
    completed = {
        (row["run_name"], int(row["row"]))
        for row in pair_rows
        if row.get("record_type") == "completed_video"
    }
    pair_rows = [row for row in pair_rows if row.get("record_type") != "completed_video"]
    total_scheduled = 0
    total_evaluable = 0

    for run_name in args.runs:
        for item in items:
            row_id = int(item["_row"])
            c2ws = load_c2ws_from_json(
                item["pose_path"],
                start_frame=int(item["start_frame"]),
                num_frames=int(item["num_frames"]),
            )
            schedule = build_pose_anchor_schedule(
                c2ws,
                frame_stride=args.frame_stride,
                handoff_overlap=args.handoff_overlap,
                min_pair_overlap=args.min_pair_overlap,
            )
            if args.max_samples_per_video is not None:
                schedule = schedule[: int(args.max_samples_per_video)]
            total_scheduled += len(schedule)
            total_evaluable += sum(row["evaluable"] for row in schedule)
            if (run_name, row_id) in completed:
                continue

            run_dir = args.root / run_name
            video_path = run_dir / f"{item['output_prefix']}custom.mp4"
            if not video_path.is_file():
                message = f"Missing video for {run_name} row {row_id}: {video_path}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue

            evaluable = [row for row in schedule if row["evaluable"]]
            requested = sorted(
                {
                    int(frame)
                    for row in evaluable
                    for frame in (row["anchor_frame"], row["current_frame"])
                }
            )
            if not requested:
                print(f"[skip] {run_name} row {row_id}: no pose-evaluable links")
                continue
            generated = read_video_frames(video_path, requested)
            reference_shape = generated[requested[0]].shape
            gt = read_gt_frames(item, requested, reference_shape)
            features = {
                frame: extract_sift(cv2, extractor, generated[frame])
                for frame in requested
            }
            metrics = {
                frame: frame_metrics(generated[frame], gt[frame])
                for frame in requested
            }

            video_rows = []
            for link in evaluable:
                anchor = int(link["anchor_frame"])
                current = int(link["current_frame"])
                scores = score_anchor_pair(
                    cv2,
                    features[anchor],
                    features[current],
                    c2ws[anchor],
                    c2ws[current],
                    reference_shape,
                    ratio_threshold=args.ratio_threshold,
                    inlier_threshold_px=args.inlier_threshold_px,
                    min_support_matches=args.min_support_matches,
                    fov_half_h=args.fov_half_h,
                    fov_half_v=args.fov_half_v,
                )
                video_rows.append(
                    {
                        "run_name": run_name,
                        "row": row_id,
                        "scene": item["scene"],
                        "start_frame": int(item["start_frame"]),
                        "duration_sec": int(item["duration_sec"]),
                        "anchor_frame": anchor,
                        "current_frame": current,
                        "time_sec": float(current / float(item["fps"])),
                        **link,
                        "anchor_psnr_db": float(metrics[anchor]["psnr_db"]),
                        "anchor_ssim": float(metrics[anchor]["ssim"]),
                        "current_psnr_db": float(metrics[current]["psnr_db"]),
                        "current_ssim": float(metrics[current]["ssim"]),
                        "psnr_delta_from_anchor": float(
                            metrics[current]["psnr_db"] - metrics[anchor]["psnr_db"]
                        ),
                        "ssim_delta_from_anchor": float(
                            metrics[current]["ssim"] - metrics[anchor]["ssim"]
                        ),
                        **scores,
                    }
                )
            pair_rows.extend(video_rows)
            completed.add((run_name, row_id))
            serialized = list(pair_rows) + [
                {
                    "record_type": "completed_video",
                    "run_name": completed_run,
                    "row": completed_row,
                }
                for completed_run, completed_row in sorted(completed)
            ]
            atomic_write_jsonl(partial_path, serialized)
            print(
                f"[{run_name}] row {row_id}: {len(video_rows)} links; "
                f"mean matches={np.mean([row['match_count'] for row in video_rows]):.1f}"
            )

    if not pair_rows:
        raise RuntimeError("No pose-evaluable anchor links were produced")
    add_current_quality_aliases(pair_rows)
    add_within_trajectory_labels(pair_rows, args.bad_quantile)
    add_transition_labels(pair_rows, args.bad_quantile)
    add_anchor_fidelity_labels(pair_rows)

    absolute = estimator_rows(
        pair_rows,
        ESTIMATOR_FIELDS,
        train_rows,
        test_rows,
        args.bootstrap_repeats,
        args.split_seed,
        args.max_train_clean_false_reject,
    )
    transition = summaries_for_transition(pair_rows, train_rows, test_rows, args)
    severity_sweep = failure_severity_sweep(
        pair_rows, train_rows, test_rows, args
    )
    by_run = estimator_run_rows(pair_rows, absolute, test_rows)
    proposed = summary_by_name(absolute, "anchor_chain_score")
    transition_proposed = summary_by_name(transition, "anchor_chain_score")
    descriptor = summary_by_name(absolute, "descriptor_consistency")
    strata = anchor_strata_auc(
        pair_rows, test_rows, estimator="anchor_chain_score"
    )
    coverage = total_evaluable / total_scheduled if total_scheduled else 0.0
    decision = make_decision(
        proposed, transition_proposed, descriptor, strata, coverage, args
    )

    write_csv(args.output_dir / "pair_scores.csv", pair_rows)
    write_csv(args.output_dir / "estimator_summary_absolute.csv", absolute)
    write_csv(args.output_dir / "estimator_summary_transition.csv", transition)
    write_csv(args.output_dir / "estimator_by_run.csv", by_run)
    write_csv(args.output_dir / "estimator_severity_sweep.csv", severity_sweep)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    make_plot(args.output_dir / "anchor_chain_validation.png", absolute, transition)
    write_report(
        args.output_dir / "report.md",
        absolute,
        transition,
        by_run,
        strata,
        severity_sweep,
        decision,
        args,
        train_rows,
        test_rows,
        total_scheduled,
        total_evaluable,
    )
    print(f"Decision: {decision['decision']}")
    print(
        "Anchor-chain AUC: "
        f"absolute={proposed['test_quality_auc']:.3f} "
        f"transition={transition_proposed['test_quality_auc']:.3f}"
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
