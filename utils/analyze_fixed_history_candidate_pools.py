"""Test candidate-pool growth on one frozen generated history.

For each fixed query, this analysis computes MemCam's camera-FOV overlap score
for every candidate exactly once. It then admits candidates through nested
pools while reusing that same score vector. The generated rollout, query,
candidate pixels, and retrieval scores are therefore fixed; only candidate
membership changes.

The smallest pool is a shared recent-history core. Larger pools add older
frames in a fixed random order, repeated with several orders. Selected
generated frames are compared with exact-index dataset ground truth to measure
stored-image fidelity. This is CPU-only and does not generate video.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_STRIDE = 76
PREDICT_FRAMES = 76

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.analyze_selected_memory_image_quality import read_gt_frame  # noqa: E402
from utils.evaluate_context_memory import frame_metrics  # noqa: E402
from utils.visualize_geometric_coverage_evictions import (  # noqa: E402
    load_video_frames_single_pass,
)


def parse_int_ranges(value):
    if value in (None, ""):
        return None
    output = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        else:
            output.add(int(part))
    return sorted(output)


def parse_pool_sizes(value):
    output = []
    previous = 0
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"all", "unbounded"}:
            if any(size is None for _, size in output):
                raise ValueError("The full-history pool may appear only once")
            output.append(("all", None))
            continue
        size = int(part)
        if size <= previous:
            raise ValueError("Finite pool sizes must be strictly increasing")
        previous = size
        output.append((f"b{size}", size))
    if not output or output[-1][1] is not None:
        raise ValueError("--pool_sizes must end with all")
    return output


def load_manifest(path, duration, selected_rows=None, max_rows=None):
    selected_rows = set(selected_rows) if selected_rows is not None else None
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("duration_sec", -1)) != int(duration):
                continue
            if selected_rows is not None and row_idx not in selected_rows:
                continue
            item["_row"] = row_idx
            rows.append(item)
    return rows if max_rows is None else rows[: int(max_rows)]


def resolve_pose_path(item, dataset_root=None):
    path = Path(item["pose_path"])
    if path.is_file():
        return path
    if dataset_root is not None:
        path = Path(dataset_root) / "jsons" / f"{item['scene']}.json"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Pose file not found for row {item['_row']}: {item['pose_path']}"
    )


def resolve_video_path(root, run_name, item):
    return Path(root) / run_name / f"{item['output_prefix']}custom.mp4"


def unbounded_candidates(section_idx):
    if int(section_idx) <= 0:
        return []
    section_start = int(section_idx) * SECTION_STRIDE
    return list(range(section_start - 3))


def nested_candidate_pools(candidates, pool_sizes, order_seed):
    """Return nested pools with a shared recent core.

    The first finite size defines the core. Every larger finite pool appends a
    prefix of one permutation of the older history. The full pool contains all
    candidates. Membership changes, but the core and expansion order do not.
    """
    candidates = sorted(set(int(frame) for frame in candidates))
    finite_sizes = [size for _, size in pool_sizes if size is not None]
    if not finite_sizes:
        raise ValueError("At least one finite pool is required")
    core_size = min(finite_sizes[0], len(candidates))
    core = candidates[-core_size:]
    older = np.asarray(candidates[:-core_size], dtype=np.int64)
    rng = np.random.default_rng(int(order_seed))
    expansion = rng.permutation(older).tolist() if len(older) else []

    output = {}
    for label, requested_size in pool_sizes:
        if requested_size is None or requested_size >= len(candidates):
            members = list(candidates)
        else:
            added = max(int(requested_size) - core_size, 0)
            members = sorted(core + expansion[:added])
        output[label] = members
    return output


def extract_pose_parameters(c2ws):
    c2ws = np.asarray(c2ws, dtype=np.float64)
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]
    sy = np.sqrt(rotations[:, 0, 0] ** 2 + rotations[:, 1, 0] ** 2)
    singular = sy < 1e-6
    roll = np.where(
        singular,
        np.arctan2(-rotations[:, 1, 2], rotations[:, 1, 1]),
        np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2]),
    )
    pitch = np.arctan2(-rotations[:, 2, 0], sy)
    yaw = np.where(
        singular,
        0.0,
        np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]),
    )
    return positions, np.degrees(pitch), np.degrees(roll), np.degrees(yaw)


def batched_fov_overlap_scores(
    c2ws,
    target_frame,
    candidate_frames,
    num_samples=5000,
    batch_size=64,
    fov_half_h=45.0,
    fov_half_v=30.0,
    radius=50.0,
    seed=0,
):
    """Vectorized form of MemCam's stochastic FOV-IoU scorer."""
    import torch

    candidates = np.asarray(candidate_frames, dtype=np.int64)
    if candidates.size == 0:
        return np.empty(0, dtype=np.float64)
    positions, pitch, _roll, yaw = extract_pose_parameters(c2ws)
    target_position = positions[int(target_frame)]
    target_pitch = float(pitch[int(target_frame)])
    target_yaw = float(yaw[int(target_frame)])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scores = []
    for start in range(0, len(candidates), int(batch_size)):
        batch_frames = candidates[start : start + int(batch_size)]
        count = len(batch_frames)
        candidate_positions = torch.as_tensor(
            positions[batch_frames], dtype=torch.float32
        )
        target_position_t = torch.as_tensor(target_position, dtype=torch.float32)
        midpoint = (candidate_positions + target_position_t[None, :]) / 2.0

        samples_r = torch.rand(
            (count, int(num_samples)), generator=generator, dtype=torch.float32
        )
        samples_phi = torch.rand(
            (count, int(num_samples)), generator=generator, dtype=torch.float32
        )
        samples_u = torch.rand(
            (count, int(num_samples)), generator=generator, dtype=torch.float32
        )
        sampled_radius = float(radius) * torch.pow(samples_r, 1.0 / 3.0)
        phi_t = 2.0 * math.pi * samples_phi
        theta_t = torch.acos(1.0 - 2.0 * samples_u)
        points = torch.stack(
            (
                sampled_radius * torch.sin(theta_t) * torch.cos(phi_t),
                sampled_radius * torch.sin(theta_t) * torch.sin(phi_t),
                sampled_radius * torch.cos(theta_t),
            ),
            dim=-1,
        )
        points = points + midpoint[:, None, :]

        def inside(center, center_pitch, center_yaw):
            vectors = points - center
            x = vectors[..., 0]
            y = vectors[..., 1]
            z = vectors[..., 2]
            azimuth = torch.atan2(y, x) * (180.0 / math.pi)
            elevation = torch.atan2(z, torch.sqrt(x * x + y * y)) * (
                180.0 / math.pi
            )
            diff_azimuth = torch.remainder(
                torch.abs(azimuth - center_yaw), 360.0
            )
            diff_elevation = torch.remainder(
                torch.abs(elevation - center_pitch), 360.0
            )
            diff_azimuth = torch.where(
                diff_azimuth > 180.0, 360.0 - diff_azimuth, diff_azimuth
            )
            diff_elevation = torch.where(
                diff_elevation > 180.0, 360.0 - diff_elevation, diff_elevation
            )
            return (diff_azimuth < float(fov_half_h)) & (
                diff_elevation < float(fov_half_v)
            )

        target_inside = inside(
            target_position_t[None, None, :], target_pitch, target_yaw
        )
        candidate_inside = inside(
            candidate_positions[:, None, :],
            torch.as_tensor(pitch[batch_frames], dtype=torch.float32)[:, None],
            torch.as_tensor(yaw[batch_frames], dtype=torch.float32)[:, None],
        )
        intersection = (target_inside & candidate_inside).sum(dim=1)
        union = (target_inside | candidate_inside).sum(dim=1)
        batch_scores = torch.where(
            union > 0,
            intersection.float() / union.float(),
            torch.zeros_like(union, dtype=torch.float32),
        )
        scores.append(batch_scores.numpy().astype(np.float64))
    return np.concatenate(scores)


def select_pool_winner(candidate_frames, scores, pool_members):
    candidates = np.asarray(candidate_frames, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(candidates) != len(scores):
        raise ValueError("candidate_frames and scores must have equal length")
    allowed = np.isin(candidates, np.asarray(pool_members, dtype=np.int64))
    positions = np.flatnonzero(allowed)
    if not len(positions):
        raise ValueError("Pool has no candidates")
    # np.argmax returns the first maximum, matching the runtime's strict > tie.
    winner_position = int(positions[int(np.argmax(scores[positions]))])
    return int(candidates[winner_position]), float(scores[winner_position])


def bootstrap_interval(values, repeats=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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


def summarize_rows(rows, pool_sizes):
    query_groups = defaultdict(list)
    for row in rows:
        key = (row["row"], row["section_idx"], row["target_frame"], row["pool"])
        query_groups[key].append(row)

    fields = (
        "selected_overlap",
        "selected_age",
        "intrinsic_psnr_db",
        "intrinsic_ssim",
        "view_psnr_db",
        "view_ssim",
        "effective_psnr_db",
        "effective_ssim",
        "winner_changed_from_core",
    )
    query_rows = []
    for key, group in query_groups.items():
        row = {
            "row": key[0],
            "section_idx": key[1],
            "target_frame": key[2],
            "pool": key[3],
            "candidate_count": float(np.mean([item["candidate_count"] for item in group])),
            "orders": len(group),
        }
        for field in fields:
            row[field] = float(np.mean([item[field] for item in group]))
        query_rows.append(row)

    trajectory_groups = defaultdict(list)
    for row in query_rows:
        trajectory_groups[(row["row"], row["pool"])].append(row)
    trajectory_rows = []
    for (row_idx, pool), group in trajectory_groups.items():
        summary = {
            "row": row_idx,
            "pool": pool,
            "queries": len(group),
            "candidate_count": float(np.mean([item["candidate_count"] for item in group])),
        }
        for field in fields:
            summary[field] = float(np.mean([item[field] for item in group]))
        trajectory_rows.append(summary)

    label_order = [label for label, _ in pool_sizes]
    pool_rows = []
    for pool in label_order:
        group = [row for row in trajectory_rows if row["pool"] == pool]
        if not group:
            continue
        summary = {
            "pool": pool,
            "trajectories": len(group),
            "candidate_count_mean": float(np.mean([row["candidate_count"] for row in group])),
        }
        for field in fields:
            values = [row[field] for row in group]
            low, high = bootstrap_interval(values, seed=1701)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_ci_low"] = low
            summary[f"{field}_ci_high"] = high
        pool_rows.append(summary)
    return query_rows, trajectory_rows, pool_rows


def save_figure(pool_rows, output_dir):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_fixed_pool_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["candidate_count_mean"] for row in pool_rows]
    panels = [
        ("intrinsic_psnr_db", "Selected-frame fidelity", "PSNR to exact-index GT (dB)"),
        ("intrinsic_ssim", "Selected-frame fidelity", "SSIM to exact-index GT"),
        ("winner_changed_from_core", "Selection instability", "winner differs from B32"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (field, title, ylabel) in zip(axes, panels):
        y = [row[f"{field}_mean"] for row in pool_rows]
        low = [row[f"{field}_ci_low"] for row in pool_rows]
        high = [row[f"{field}_ci_high"] for row in pool_rows]
        ax.plot(x, y, marker="o", color="#205493", linewidth=2)
        ax.fill_between(x, low, high, color="#9fc2df", alpha=0.45)
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.set_xlabel("admitted candidates")
        ax.set_ylabel(ylabel)
        ax.grid(color="#e4e4e4", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fixed_history_pool_curve.png", dpi=200)
    fig.savefig(output_dir / "fixed_history_pool_curve.pdf")
    plt.close(fig)


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_report(path, pool_rows, args):
    core = pool_rows[0]
    full = pool_rows[-1]
    psnr_delta = full["intrinsic_psnr_db_mean"] - core["intrinsic_psnr_db_mean"]
    ssim_delta = full["intrinsic_ssim_mean"] - core["intrinsic_ssim_mean"]
    lines = [
        "# Fixed-History Candidate-Pool Intervention",
        "",
        "## Question",
        "",
        "When the generated history, target query, candidate pixels, and retrieval scores are frozen, does admitting more historical candidates make the real FOV-overlap selector choose lower-fidelity memory?",
        "",
        "## Intervention",
        "",
        "Every pool contains the same recent B32 core. Older frames are added in nested random order. Each query's FOV-overlap score vector is computed once over full history and reused at every pool size, so score noise cannot change between pools. The selected generated frame is compared with dataset ground truth at its own exact trajectory index. Higher PSNR/SSIM is better.",
        "",
        f"- Trajectories: `{full['trajectories']}`.",
        f"- Nested expansion orders per query: `{args.nested_repeats}`.",
        f"- FOV samples per candidate: `{args.num_samples}`.",
        f"- Sections: `{args.sections}`; target offsets: `{args.target_offsets}`.",
        "",
        "## Result",
        "",
        "| pool | mean candidates | selected PSNR | 95% CI | selected SSIM | 95% CI | winner changed from B32 |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for row in pool_rows:
        lines.append(
            f"| {row['pool']} | {row['candidate_count_mean']:.1f} | "
            f"{fmt(row['intrinsic_psnr_db_mean'], 3)} | "
            f"[{fmt(row['intrinsic_psnr_db_ci_low'], 3)}, {fmt(row['intrinsic_psnr_db_ci_high'], 3)}] | "
            f"{fmt(row['intrinsic_ssim_mean'])} | "
            f"[{fmt(row['intrinsic_ssim_ci_low'])}, {fmt(row['intrinsic_ssim_ci_high'])}] | "
            f"{fmt(row['winner_changed_from_core_mean'], 3)} |"
        )
    lines.extend(
        [
            "",
            f"Full history minus B32: `{psnr_delta:+.3f}` dB PSNR and `{ssim_delta:+.4f}` SSIM. Negative values mean that admitting candidates selected less faithful stored images.",
            "",
            "## What This Establishes",
            "",
            "This is a fixed-history candidate-admission intervention. A downward fidelity curve demonstrates that adding candidates can harm the frozen selector without autoregressive age, rollout quality, or Monte Carlo score noise changing between pool sizes. It does not establish downstream video causality; the separate GT-memory-cleaning replay tests whether corrupted selected content propagates into the next generated chunk.",
            "",
            "## Files",
            "",
            "- `tables/selection_rows.csv`",
            "- `tables/query_summary.csv`",
            "- `tables/trajectory_summary.csv`",
            "- `tables/pool_summary.csv`",
            "- `figures/fixed_history_pool_curve.png`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--sections", default="35,50,65")
    parser.add_argument("--target_offsets", default="19,57")
    parser.add_argument("--pool_sizes", default="32,64,128,256,512,1024,all")
    parser.add_argument("--nested_repeats", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--fov_half_h", type=float, default=45.0)
    parser.add_argument("--fov_half_v", type=float, default=30.0)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--torch_threads", type=int, default=8)
    parser.add_argument("--decode_timeout_sec", type=int, default=600)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.nested_repeats < 1 or args.num_samples < 1:
        raise ValueError("nested repeats and FOV samples must be positive")
    sections = parse_int_ranges(args.sections) or []
    target_offsets = parse_int_ranges(args.target_offsets) or []
    if not sections or not target_offsets:
        raise ValueError("sections and target offsets cannot be empty")
    if any(offset < 1 or offset > PREDICT_FRAMES for offset in target_offsets):
        raise ValueError("target offsets must be in [1, 76]")
    pool_sizes = parse_pool_sizes(args.pool_sizes)
    items = load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=parse_int_ranges(args.rows),
        max_rows=args.max_rows,
    )
    if not items:
        raise RuntimeError("No manifest rows matched the requested filters")

    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))
    overlap_path = REPO_ROOT / "diffsynth" / "models" / "wan_video_overlap.py"
    spec = importlib.util.spec_from_file_location("fixed_pool_overlap", overlap_path)
    overlap_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlap_module)

    selection_rows = []
    for item_position, item in enumerate(items):
        row_idx = int(item["_row"])
        video_path = resolve_video_path(args.root, args.run_name, item)
        if not video_path.is_file():
            message = f"Missing baseline video for row {row_idx}: {video_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}", flush=True)
            continue
        pose_path = resolve_pose_path(item, dataset_root=args.dataset_root)
        all_c2ws, _ = overlap_module.load_poses_from_json(str(pose_path))
        start_frame = int(item["start_frame"])
        num_frames = int(item["num_frames"])
        c2ws = all_c2ws[start_frame : start_frame + num_frames]
        valid_sections = [
            section
            for section in sections
            if 0 < section and section * SECTION_STRIDE + PREDICT_FRAMES < num_frames
        ]
        print(
            f"row {row_idx} {item['scene']}: sections={valid_sections}", flush=True
        )

        pending = []
        selected_frames = set()
        for section_idx in valid_sections:
            candidates = unbounded_candidates(section_idx)
            for target_offset in target_offsets:
                target_frame = section_idx * SECTION_STRIDE + int(target_offset)
                query_seed = (
                    int(args.seed)
                    + item_position * 10_000_019
                    + section_idx * 100_003
                    + target_frame * 101
                )
                scores = batched_fov_overlap_scores(
                    c2ws=c2ws,
                    target_frame=target_frame,
                    candidate_frames=candidates,
                    num_samples=args.num_samples,
                    batch_size=args.score_batch_size,
                    fov_half_h=args.fov_half_h,
                    fov_half_v=args.fov_half_v,
                    radius=args.radius,
                    seed=query_seed,
                )
                core_winner = None
                for repeat_idx in range(args.nested_repeats):
                    pools = nested_candidate_pools(
                        candidates,
                        pool_sizes,
                        order_seed=query_seed + repeat_idx * 1_000_003,
                    )
                    for pool_label, _requested_size in pool_sizes:
                        winner, overlap = select_pool_winner(
                            candidates, scores, pools[pool_label]
                        )
                        if pool_label == pool_sizes[0][0]:
                            core_winner = winner
                        pending.append(
                            {
                                "row": row_idx,
                                "scene": item["scene"],
                                "section_idx": section_idx,
                                "target_frame": target_frame,
                                "repeat": repeat_idx,
                                "pool": pool_label,
                                "candidate_count": len(pools[pool_label]),
                                "selected_frame": winner,
                                "selected_overlap": overlap,
                                "selected_age": target_frame - winner,
                                "core_winner": core_winner,
                            }
                        )
                        selected_frames.add(winner)
                print(
                    f"  section={section_idx} target={target_frame}: "
                    f"scored={len(candidates)} candidates",
                    flush=True,
                )

        if not pending:
            continue
        print(
            f"  decoding {len(selected_frames)} selected frames from {video_path.name}",
            flush=True,
        )
        images = load_video_frames_single_pass(
            video_path,
            selected_frames,
            timeout_sec=args.decode_timeout_sec,
        )
        for row in pending:
            generated = np.asarray(images[row["selected_frame"]], dtype=np.uint8)
            selected_gt = read_gt_frame(item, row["selected_frame"], generated.shape)
            target_gt = read_gt_frame(item, row["target_frame"], generated.shape)
            intrinsic = frame_metrics(generated, selected_gt)
            view = frame_metrics(selected_gt, target_gt)
            effective = frame_metrics(generated, target_gt)
            row.update(
                {
                    "winner_changed_from_core": int(
                        row["selected_frame"] != row["core_winner"]
                    ),
                    "intrinsic_psnr_db": intrinsic["psnr_db"],
                    "intrinsic_ssim": intrinsic["ssim"],
                    "view_psnr_db": view["psnr_db"],
                    "view_ssim": view["ssim"],
                    "effective_psnr_db": effective["psnr_db"],
                    "effective_ssim": effective["ssim"],
                }
            )
            selection_rows.append(row)

    if not selection_rows:
        raise RuntimeError("No fixed-history selection rows were produced")
    query_rows, trajectory_rows, pool_rows = summarize_rows(
        selection_rows, pool_sizes
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tables" / "selection_rows.csv", selection_rows)
    write_csv(args.output_dir / "tables" / "query_summary.csv", query_rows)
    write_csv(args.output_dir / "tables" / "trajectory_summary.csv", trajectory_rows)
    write_csv(args.output_dir / "tables" / "pool_summary.csv", pool_rows)
    save_figure(pool_rows, args.output_dir / "figures")
    write_report(args.output_dir / "report.md", pool_rows, args)
    print(f"Wrote: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
