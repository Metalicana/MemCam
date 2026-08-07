"""Cheap revisit-structure check on ground-truth camera poses alone.

For a manifest row, finds frame pairs that are geometrically close (small
camera-center distance, small forward-direction angle) but temporally far
apart (gap > min_gap frames). If a row has none of these, no eviction policy
-- real or offline -- can show anchor persistence on it, because there is
nothing to revisit. This is a coarse pose-only proxy (no occlusion/visibility
check, unlike the fuller calibrate_revisit_alignment.py machinery) meant to
quickly triage which manifest rows are worth spending real generation time on.

Usage:
    python utils/check_revisit_structure.py \
        --pose_json /path/to/scene.json \
        --start_frame 1368 \
        --num_frames 1825
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_json", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--min_gap", type=int, default=304, help="Minimum frame gap to count as a revisit, not local loitering.")
    parser.add_argument("--position_threshold", type=float, default=5.0, help="Max camera-center distance (scene units) to count as 'close'.")
    parser.add_argument("--forward_angle_deg", type=float, default=30.0, help="Max forward-direction angle difference to count as 'facing similarly'.")
    args = parser.parse_args()

    from dataset.poses import load_c2ws_from_json

    c2ws = load_c2ws_from_json(
        json_path=args.pose_json, start_frame=args.start_frame, num_frames=args.num_frames
    )
    positions = c2ws[:, :3, 3]
    forward = c2ws[:, :3, 0]
    forward = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-12)

    n = len(positions)
    print(f"Frames: {n}")
    print(f"Position range: min={positions.min(axis=0)}, max={positions.max(axis=0)}")
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    print(f"Total path length: {path_length:.2f}")

    revisit_pairs = []
    for t in range(n):
        candidates = np.arange(n)
        gap = np.abs(candidates - t)
        far_enough = gap > args.min_gap
        if not np.any(far_enough):
            continue
        dist = np.linalg.norm(positions[far_enough] - positions[t], axis=1)
        close_enough = dist < args.position_threshold
        if not np.any(close_enough):
            continue
        far_candidates = candidates[far_enough][close_enough]
        cos_angle = np.clip(forward[far_candidates] @ forward[t], -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        facing_similarly = angle_deg < args.forward_angle_deg
        for other in far_candidates[facing_similarly]:
            if other > t:  # dedupe symmetric pairs
                revisit_pairs.append((t, int(other), float(np.linalg.norm(positions[other] - positions[t]))))

    print(f"\nRevisit pairs found (gap > {args.min_gap}, position < {args.position_threshold}, "
          f"forward angle < {args.forward_angle_deg} deg): {len(revisit_pairs)}")
    if revisit_pairs:
        gaps = [abs(b - a) for a, b, _ in revisit_pairs]
        print(f"Gap distribution: min={min(gaps)}, median={int(np.median(gaps))}, max={max(gaps)}")
        print("Sample pairs (frame_a, frame_b, position_distance):")
        for pair in revisit_pairs[:10]:
            print(f"  {pair}")
    else:
        print("No revisit structure found at these thresholds -- this row cannot "
              "demonstrate anchor-persistence benefits regardless of policy. Try "
              "a looser --position_threshold/--forward_angle_deg, or a different row.")


if __name__ == "__main__":
    main()
