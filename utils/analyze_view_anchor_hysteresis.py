"""Test whether older same-view anchors are cleaner than later rewrites.

This is a CPU-only analysis over existing videos. For sampled later frames, it
finds an earlier frame from the same rollout with high camera-FOV similarity.
Both frames are compared with their own exact-index dataset ground truth. The
test therefore asks whether repeated observations of an already-covered view
become less faithful as autoregressive generation proceeds.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def load_manifest(path, duration, max_rows=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["duration_sec"]) != int(duration):
                continue
            item["_row"] = row_idx
            rows.append(item)
    return rows if max_rows is None else rows[: int(max_rows)]


def camera_trajectory_similarity(
    c2ws,
    query_frame_indices,
    memory_frame_indices,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
):
    """Vectorized copy of MemCam's fixed-scale camera-FOV affinity."""
    query_poses = np.asarray(c2ws[query_frame_indices], dtype=np.float64)
    memory_poses = np.asarray(c2ws[memory_frame_indices], dtype=np.float64)
    query_positions = query_poses[:, :3, 3]
    memory_positions = memory_poses[:, :3, 3]
    position_distance = np.linalg.norm(
        query_positions[:, None, :] - memory_positions[None, :, :], axis=-1
    )
    position_similarity = np.clip(
        1.0 - position_distance / (2.0 * float(radius)), 0.0, 1.0
    )

    query_forward = query_poses[:, :3, 0]
    memory_forward = memory_poses[:, :3, 0]
    query_forward /= np.maximum(
        np.linalg.norm(query_forward, axis=1, keepdims=True), 1e-12
    )
    memory_forward /= np.maximum(
        np.linalg.norm(memory_forward, axis=1, keepdims=True), 1e-12
    )
    query_yaw = np.arctan2(query_forward[:, 1], query_forward[:, 0])
    memory_yaw = np.arctan2(memory_forward[:, 1], memory_forward[:, 0])
    yaw_distance = np.abs(query_yaw[:, None] - memory_yaw[None, :])
    yaw_distance = np.minimum(yaw_distance, 2.0 * np.pi - yaw_distance)
    query_pitch = np.arctan2(
        query_forward[:, 2], np.linalg.norm(query_forward[:, :2], axis=1)
    )
    memory_pitch = np.arctan2(
        memory_forward[:, 2], np.linalg.norm(memory_forward[:, :2], axis=1)
    )
    pitch_distance = np.abs(query_pitch[:, None] - memory_pitch[None, :])
    horizontal_similarity = np.clip(
        1.0 - yaw_distance / (2.0 * np.deg2rad(float(fov_half_h))), 0.0, 1.0
    )
    vertical_similarity = np.clip(
        1.0 - pitch_distance / (2.0 * np.deg2rad(float(fov_half_v))), 0.0, 1.0
    )
    return position_similarity * horizontal_similarity * vertical_similarity


def build_view_pairs(
    c2ws,
    sample_stride=76,
    candidate_stride=4,
    min_history_frames=304,
    min_temporal_gap=152,
    min_view_similarity=0.85,
    earliest_candidate_frame=1,
):
    """Match each sampled later frame to its oldest equivalent earlier view."""
    c2ws = np.asarray(c2ws, dtype=np.float64)
    pairs = []
    for later_frame in range(int(min_history_frames), len(c2ws), int(sample_stride)):
        stop = later_frame - int(min_temporal_gap)
        if stop <= 0:
            continue
        earlier_frames = np.arange(
            int(earliest_candidate_frame),
            stop,
            int(candidate_stride),
            dtype=np.int64,
        )
        if earlier_frames.size == 0:
            continue
        similarities = camera_trajectory_similarity(
            c2ws,
            [later_frame],
            earlier_frames.tolist(),
        )[0]
        eligible = np.flatnonzero(similarities >= float(min_view_similarity))
        if eligible.size == 0:
            continue
        # The persistent-memory hypothesis is specifically that an established
        # representative should not be overwritten by a later equivalent view.
        earlier_position = int(eligible[0])
        pairs.append(
            {
                "earlier_frame": int(earlier_frames[earlier_position]),
                "later_frame": int(later_frame),
                "frame_gap": int(later_frame - earlier_frames[earlier_position]),
                "view_similarity": float(similarities[earlier_position]),
                "best_available_view_similarity": float(np.max(similarities)),
            }
        )
    return pairs


def read_gt_frame(item, local_frame, reference_shape):
    dataset_frame = int(item["start_frame"]) + int(local_frame)
    path = Path(item["gt_frames_dir"]) / f"{dataset_frame:04d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT frame: {path}")
    height, width = reference_shape[:2]
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def measure_requested_frames(video_path, item, requested_frames):
    import imageio.v2 as imageio
    from utils.evaluate_context_memory import frame_metrics

    requested_frames = {int(frame) for frame in requested_frames}
    output = {}
    if not requested_frames:
        return output
    reader = imageio.get_reader(str(video_path))
    try:
        last_frame = max(requested_frames)
        for frame_idx, generated in enumerate(reader):
            if frame_idx > last_frame:
                break
            if frame_idx not in requested_frames:
                continue
            generated = np.asarray(generated, dtype=np.uint8)
            gt = read_gt_frame(item, frame_idx, generated.shape)
            output[frame_idx] = frame_metrics(generated, gt)
    finally:
        reader.close()
    missing = sorted(requested_frames - set(output))
    if missing:
        raise RuntimeError(f"Video {video_path} is missing frames {missing[:10]}")
    return output


def enrich_pairs(pairs, item, metrics):
    output = []
    for pair in pairs:
        earlier = metrics[pair["earlier_frame"]]
        later = metrics[pair["later_frame"]]
        output.append(
            {
                "row": int(item["_row"]),
                "scene": item["scene"],
                "start_frame": int(item["start_frame"]),
                "duration_sec": int(item["duration_sec"]),
                **pair,
                "later_time_sec": float(pair["later_frame"] / item["fps"]),
                "earlier_psnr_db": float(earlier["psnr_db"]),
                "later_psnr_db": float(later["psnr_db"]),
                "older_minus_newer_psnr_db": float(
                    earlier["psnr_db"] - later["psnr_db"]
                ),
                "earlier_ssim": float(earlier["ssim"]),
                "later_ssim": float(later["ssim"]),
                "older_minus_newer_ssim": float(earlier["ssim"] - later["ssim"]),
            }
        )
    return output


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def bootstrap_mean_interval(values, repeats=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def exact_two_sided_sign_pvalue(wins, losses):
    trials = int(wins) + int(losses)
    if trials == 0:
        return None
    smaller = min(int(wins), int(losses))
    tail = sum(math.comb(trials, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def summarize_pairs(rows, bootstrap_repeats=10000, seed=0):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["row"])].append(row)

    trajectory_rows = []
    for trajectory, values in sorted(grouped.items()):
        trajectory_rows.append(
            {
                "row": trajectory,
                "scene": values[0]["scene"],
                "pairs": len(values),
                "older_minus_newer_psnr_db": mean(
                    row["older_minus_newer_psnr_db"] for row in values
                ),
                "older_minus_newer_ssim": mean(
                    row["older_minus_newer_ssim"] for row in values
                ),
                "view_similarity_mean": mean(row["view_similarity"] for row in values),
                "frame_gap_mean": mean(row["frame_gap"] for row in values),
            }
        )

    summary = {
        "pairs": len(rows),
        "trajectories": len(trajectory_rows),
    }
    for metric in ("psnr_db", "ssim"):
        field = f"older_minus_newer_{metric}"
        trajectory_values = [row[field] for row in trajectory_rows]
        ci_low, ci_high = bootstrap_mean_interval(
            trajectory_values, repeats=bootstrap_repeats, seed=seed
        )
        wins = sum(value > 0 for value in trajectory_values)
        losses = sum(value < 0 for value in trajectory_values)
        summary.update(
            {
                f"{field}_mean": mean(trajectory_values),
                f"{field}_ci_low": ci_low,
                f"{field}_ci_high": ci_high,
                f"{field}_trajectory_wins": wins,
                f"{field}_trajectory_losses": losses,
                f"{field}_sign_pvalue": exact_two_sided_sign_pvalue(wins, losses),
            }
        )

    summary["decision"] = (
        "SUPPORTS_HYSTERESIS"
        if summary["trajectories"] >= 10
        and summary["older_minus_newer_psnr_db_ci_low"] > 0
        and summary["older_minus_newer_ssim_ci_low"] > 0
        else "NOT_SUPPORTED"
    )
    return trajectory_rows, summary


def summarize_age_bins(rows, num_bins=4):
    if not rows:
        return []
    later_frames = np.asarray([row["later_frame"] for row in rows], dtype=np.float64)
    edges = np.quantile(later_frames, np.linspace(0.0, 1.0, int(num_bins) + 1))
    edges = np.maximum.accumulate(edges)
    output = []
    for bin_idx in range(int(num_bins)):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        members = [
            row
            for row in rows
            if row["later_frame"] >= lower
            and (row["later_frame"] <= upper if bin_idx == num_bins - 1 else row["later_frame"] < upper)
        ]
        if not members:
            continue
        output.append(
            {
                "age_bin": bin_idx,
                "later_frame_min": int(min(row["later_frame"] for row in members)),
                "later_frame_max": int(max(row["later_frame"] for row in members)),
                "pairs": len(members),
                "older_minus_newer_psnr_db": mean(
                    row["older_minus_newer_psnr_db"] for row in members
                ),
                "older_minus_newer_ssim": mean(
                    row["older_minus_newer_ssim"] for row in members
                ),
            }
        )
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_report(path, summary, age_rows, args):
    lines = [
        "# View-Anchor Hysteresis Test",
        "",
        "## Question",
        "",
        "When the camera revisits an already-covered view, is its older generated representative cleaner than a later autoregressive rewrite? Each image is compared with dataset ground truth at its own exact frame index. Positive deltas mean the older frame is cleaner.",
        "",
        "## Protocol",
        "",
        f"- Run: `{args.run_name}`; duration: `{args.duration}s`.",
        f"- Later-frame stride: `{args.sample_stride}`; earlier-candidate stride: `{args.candidate_stride}`.",
        f"- Minimum temporal gap: `{args.min_temporal_gap}` frames.",
        f"- Earliest candidate frame: `{args.earliest_candidate_frame}` (frame 0, the clean input anchor, is excluded by default).",
        f"- Minimum camera-FOV similarity: `{args.min_view_similarity}`.",
        "- The oldest earlier frame clearing the view threshold is treated as the incumbent anchor.",
        "- Confidence intervals resample trajectory-level means.",
        "",
        "## Result",
        "",
        f"- Decision: **{summary['decision']}**.",
        f"- Matched pairs: `{summary['pairs']}` across `{summary['trajectories']}` trajectories.",
        f"- Older-minus-newer PSNR: `{fmt(summary['older_minus_newer_psnr_db_mean'], 3)}` dB; 95% CI `[{fmt(summary['older_minus_newer_psnr_db_ci_low'], 3)}, {fmt(summary['older_minus_newer_psnr_db_ci_high'], 3)}]`; trajectory wins/losses `{summary['older_minus_newer_psnr_db_trajectory_wins']}/{summary['older_minus_newer_psnr_db_trajectory_losses']}`.",
        f"- Older-minus-newer SSIM: `{fmt(summary['older_minus_newer_ssim_mean'])}`; 95% CI `[{fmt(summary['older_minus_newer_ssim_ci_low'])}, {fmt(summary['older_minus_newer_ssim_ci_high'])}]`; trajectory wins/losses `{summary['older_minus_newer_ssim_trajectory_wins']}/{summary['older_minus_newer_ssim_trajectory_losses']}`.",
        "",
        "## By Video Age",
        "",
        "| bin | later-frame range | pairs | PSNR delta | SSIM delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in age_rows:
        lines.append(
            f"| {row['age_bin']} | {row['later_frame_min']}--{row['later_frame_max']} | {row['pairs']} | {fmt(row['older_minus_newer_psnr_db'], 3)} | {fmt(row['older_minus_newer_ssim'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`SUPPORTS_HYSTERESIS` requires the trajectory-bootstrap lower bound to be positive for both PSNR and SSIM. This supports conservative promotion of redundant views; it does not prove that every later frame is corrupted or that genuinely novel views should be rejected.",
            "",
            "## Files",
            "",
            "- `view_pairs.csv`",
            "- `trajectory_summary.csv`",
            "- `age_bin_summary.csv`",
            "- `summary.json`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--sample_stride", type=int, default=76)
    parser.add_argument("--candidate_stride", type=int, default=4)
    parser.add_argument("--min_history_frames", type=int, default=304)
    parser.add_argument("--min_temporal_gap", type=int, default=152)
    parser.add_argument("--earliest_candidate_frame", type=int, default=1)
    parser.add_argument("--min_view_similarity", type=float, default=0.85)
    parser.add_argument("--age_bins", type=int, default=4)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    for name in ("sample_stride", "candidate_stride", "min_history_frames", "min_temporal_gap", "age_bins"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.earliest_candidate_frame < 1:
        raise ValueError("--earliest_candidate_frame must exclude the clean frame-0 input")
    if not 0.0 <= args.min_view_similarity <= 1.0:
        raise ValueError("--min_view_similarity must be in [0, 1]")

    items = load_manifest(args.manifest, args.duration, max_rows=args.max_rows)
    from dataset.poses import load_c2ws_from_json

    all_rows = []
    for item in items:
        video_path = args.root / args.run_name / f"{item['output_prefix']}custom.mp4"
        if not video_path.is_file():
            message = f"missing video for row {item['_row']}: {video_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue
        c2ws = load_c2ws_from_json(
            item["pose_path"],
            start_frame=item["start_frame"],
            num_frames=item["num_frames"],
        )
        pairs = build_view_pairs(
            c2ws,
            sample_stride=args.sample_stride,
            candidate_stride=args.candidate_stride,
            min_history_frames=args.min_history_frames,
            min_temporal_gap=args.min_temporal_gap,
            min_view_similarity=args.min_view_similarity,
            earliest_candidate_frame=args.earliest_candidate_frame,
        )
        requested = {
            frame
            for pair in pairs
            for frame in (pair["earlier_frame"], pair["later_frame"])
        }
        metrics = measure_requested_frames(video_path, item, requested)
        rows = enrich_pairs(pairs, item, metrics)
        all_rows.extend(rows)
        print(f"row {item['_row']}: {len(rows)} matched same-view pairs")

    if not all_rows:
        raise RuntimeError("No same-view frame pairs were found")
    trajectory_rows, summary = summarize_pairs(
        all_rows,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    age_rows = summarize_age_bins(all_rows, num_bins=args.age_bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "view_pairs.csv", all_rows)
    write_csv(args.output_dir / "trajectory_summary.csv", trajectory_rows)
    write_csv(args.output_dir / "age_bin_summary.csv", age_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summary, age_rows, args)

    print(f"\nDecision: {summary['decision']}")
    print(
        "Older-minus-newer: "
        f"PSNR={summary['older_minus_newer_psnr_db_mean']:+.3f} dB "
        f"SSIM={summary['older_minus_newer_ssim_mean']:+.4f}"
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
