"""Show the same camera view on separate visits, with one GT reference.

CPU-only selection uses camera poses and GT agreement, never policy scores.
No arbitrary temporal snapshots or connected-component pose drift are used.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.analyze_selected_memory_image_quality import load_manifest, read_gt_frame
from utils.evaluate_context_memory import frame_metrics
from utils.visualize_common_source_psnr_extremes import parse_int_ranges, remap_gt_dir
from utils.visualize_geometric_coverage_evictions import load_video_frames_single_pass


def distances(poses, anchor):
    position = np.linalg.norm(poses[:, :3, 3] - poses[anchor, :3, 3], axis=1)
    relative = np.einsum("ij,njk->nik", poses[anchor, :3, :3].T, poses[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return position, angle


def runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def visit_groups(poses, fps, args):
    output, seen = [], set()
    for anchor in range(1, len(poses), args.pose_stride):
        position, angle = distances(poses, anchor)
        near = (position <= args.position_m) & (angle <= args.rotation_deg)
        near[0] = False  # The supplied conditioning image is not generated output.
        far = (position > 2 * args.position_m) | (angle > 2 * args.rotation_deg)
        cost = position / args.position_m + angle / args.rotation_deg
        visits = []
        for interval in runs(near):
            best = int(interval[np.argmin(cost[interval])])
            if visits:
                if best - visits[-1] < fps * args.min_gap_sec:
                    continue
                departure = runs(far[visits[-1] + 1:best])
                if not departure or max(map(len, departure)) / fps < args.min_away_sec:
                    continue
            visits.append(best)
        if len(visits) < args.min_visits:
            continue
        # Only deduplicate identical frames. Coarse time bins can discard a
        # better-aligned anchor before ground-truth agreement is checked.
        key = tuple(visits)
        if key in seen:
            continue
        seen.add(key)
        chosen = [visits[i] for i in np.linspace(0, len(visits) - 1, min(args.max_visits, len(visits)), dtype=int)]
        output.append({"anchor": anchor, "frames": chosen, "all_pose_visits": visits,
                       "position_from_anchor_m": [float(position[i]) for i in chosen],
                       "rotation_from_anchor_deg": [float(angle[i]) for i in chosen],
                       "span_sec": (chosen[-1] - chosen[0]) / fps})
    return sorted(output, key=lambda c: (-len(c["frames"]), -c["span_sec"], c["anchor"]))


def pose_path(item, dataset_root):
    return dataset_root / "jsons" / f"{item['scene']}.json" if dataset_root else Path(item["pose_path"])


def load_poses(item, dataset_root):
    # Reuse the generator's coordinate convention, but refuse ambiguous GT indexing.
    from dataset.poses import load_c2ws_from_json
    path = pose_path(item, dataset_root)
    data = json.loads(path.read_text())["CineCameraActor"]
    keys = sorted(map(int, data))
    start, count = int(item["start_frame"]), int(item["num_frames"])
    if keys[start:start + count] != list(range(start, start + count)):
        raise ValueError("Pose keys and trajectory-indexed GT do not align; verify indexing before plotting")
    return load_c2ws_from_json(path, start_frame=start, num_frames=count)


def gt_image(item, frame):
    path = Path(item["gt_frames_dir"]) / f"{int(item['start_frame']) + frame:04d}.png"
    with Image.open(path) as image:
        return image.convert("RGB")


def verify_gt(item, frames, threshold):
    images = {i: gt_image(item, i) for i in frames}
    reference = images[frames[0]]
    size = (256, round(256 * reference.height / reference.width))
    pixels = {i: np.asarray(image.resize(size, Image.Resampling.BICUBIC)) for i, image in images.items()}
    # Pairwise agreement avoids treating two different sides of the pose tolerance as one view.
    pairs = [{"a": a, "b": b, "ssim": frame_metrics(pixels[a], pixels[b])["ssim"]}
             for n, a in enumerate(frames) for b in frames[n + 1:]]
    return min(pair["ssim"] for pair in pairs) >= threshold, pairs, images


def diagnose(args, items):
    """Audit view availability without decoding generated videos or tuning on scores."""
    from make_memory_strips import strip
    profiles = [("strict", args.position_m, args.rotation_deg),
                ("nearby", max(.5, args.position_m), max(10., args.rotation_deg)),
                ("wider", max(1., args.position_m), max(15., args.rotation_deg))]
    records, errors = [], []
    for row, item in items.items():
        try:
            poses = load_poses(item, args.dataset_root)
            parts = []
            for name, position, rotation in profiles:
                options = argparse.Namespace(**vars(args))
                options.position_m, options.rotation_deg = position, rotation
                options.min_visits = 2
                groups = visit_groups(poses, float(item["fps"]), options)
                result = {"row": row, "scene": item["scene"], "profile": name,
                          "position_m": position, "rotation_deg": rotation,
                          "pose_groups": len(groups),
                          "max_visits": max((len(g["all_pose_visits"]) for g in groups), default=0),
                          "gt_checks": []}
                summaries = []
                for category, pool in (("2visits", [g for g in groups if len(g["frames"]) == 2]),
                                       ("3plus", [g for g in groups if len(g["frames"]) >= 3])):
                    best, best_images = None, None
                    checked = []
                    # Spread checks across the pose-only ordering rather than
                    # inspecting only nearly identical top-ranked anchors.
                    indexes = np.linspace(0, len(pool) - 1, min(len(pool), args.diagnostic_gt_checks), dtype=int)
                    for index in indexes:
                        group = pool[index]
                        good, pairs, images = verify_gt(item, group["frames"], args.min_gt_ssim)
                        record = dict(group, gt_pairs=pairs, min_gt_ssim=min(p["ssim"] for p in pairs),
                                      gt_passed=good)
                        checked.append(record)
                        if best is None or record["min_gt_ssim"] > best["min_gt_ssim"]:
                            best, best_images = record, images
                    result["gt_checks"].append({"category": category, "available_groups": len(pool),
                        "checked_groups": len(checked), "passed_among_checked": sum(c["gt_passed"] for c in checked),
                        "best": best, "checked": checked})
                    if best:
                        frames = best["frames"]
                        stem = args.output / f"inspection_row{row}_{name}_{category}_gt"
                        strip([best_images[i] for i in frames],
                              [f"GT | {i / item['fps']:.1f}s" for i in frames], stem)
                        stem.with_suffix(".json").write_text(json.dumps({
                            "row": row, "profile": name, "position_m": position, "rotation_deg": rotation,
                            "candidate": best, "status": "GT-only inspection; not a confirmed same-view figure",
                        }, indent=2) + "\n")
                    summaries.append(f"{category}={best['min_gt_ssim']:.3f}" if best else f"{category}=none")
                parts.append(f"{name}: max_visits={result['max_visits']}, best_checked_GT " + ", ".join(summaries))
                records.append(result)
            print(f"row {row} {item['scene']}\n  " + "\n  ".join(parts), flush=True)
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"row": row, "error": str(exc)})
            print(errors[-1], flush=True)
    (args.output / "diagnostics.json").write_text(json.dumps({
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": records, "errors": errors,
        "note": "No generated pixels used. Strict retains the requested pose tolerance; nearby/wider are separately labeled sensitivity checks, not automatic relaxations. GT checks sample each visit-count category; a failed sample does not prove no suitable group exists. Previews are GT-only for checking whether the same place/view is visible."
    }, indent=2) + "\n")
    print(f"Diagnostics and GT-only previews: {args.output}")


def render(case, images, ground_truth, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames, fps = case["frames"], case["fps"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.1 * (len(frames) + 1), 4.8))
    grid = fig.add_gridspec(2, len(frames) + 1, left=.025, right=.995, top=.84, bottom=.03,
                          wspace=.08, hspace=.07)
    ref = fig.add_subplot(grid[:, 0])
    ref.imshow(ground_truth[frames[0]])
    ref.set_title("Ground truth", fontsize=12)
    ref.axis("off")
    for row, label in enumerate(("Unbounded", "Ours")):
        for col, frame in enumerate(frames, 1):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(images[label][frame])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 1:
                ax.set_ylabel(label, fontsize=11, weight="bold")
            if row == 0:
                ax.set_title(f"Visit {col} | {frame / fps:.1f}s", fontsize=11)
    fig.suptitle(f"{case['scene']} | Returning to the same view", fontsize=15, y=.97)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=180, facecolor="white")
    plt.close(fig)
    # Bare strips for manuscript layout, plus a separate full-GT audit strip.
    from make_memory_strips import strip
    for label in ("Unbounded", "Ours"):
        strip([images[label][i] for i in frames], [f"Visit {n + 1} | {i / fps:.1f}s" for n, i in enumerate(frames)],
              output.with_name(output.name + "_" + label.lower()))
    strip([ground_truth[i] for i in frames], [f"GT | {i / fps:.1f}s" for i in frames],
          output.with_name(output.name + "_gt_alignment_check"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("psnr_db", "ssim"), ("Exact-index PSNR (dB)", "Exact-index SSIM")):
        for label, color in (("Unbounded", "#B93838"), ("Ours", "#0072B2")):
            ax.plot([i / fps for i in frames], [s[metric] for s in case["scores"][label]],
                    marker="o", label=label, color=color)
        ax.set(xlabel="Visit time (s)", ylabel=ylabel)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_name(output.name + "_fidelity").with_suffix("." + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ours-run", default="slam_b32_covisibility")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument("--max-visits", type=int, default=4)
    parser.add_argument("--position-m", type=float, default=.25)
    parser.add_argument("--rotation-deg", type=float, default=5.)
    parser.add_argument("--min-gap-sec", type=float, default=3.)
    parser.add_argument("--min-away-sec", type=float, default=1.)
    parser.add_argument("--min-gt-ssim", type=float, default=.9)
    parser.add_argument("--pose-stride", type=int, default=15)
    parser.add_argument("--decode-timeout", type=int, default=600)
    parser.add_argument("--diagnose", action="store_true", help="Audit 2+ returns and GT agreement; no generated video decoding")
    parser.add_argument("--diagnostic-gt-checks", type=int, default=12, help="Max GT groups per visit-count category and pose profile")
    args = parser.parse_args()
    if (args.min_visits < 2 or args.max_visits < args.min_visits or args.top < 1 or args.pose_stride < 1
            or min(args.position_m, args.rotation_deg, args.min_gap_sec, args.min_away_sec, args.decode_timeout) <= 0
            or args.diagnostic_gt_checks < 1 or not 0 <= args.min_gt_ssim <= 1):
        parser.error("Invalid thresholds or counts")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    selected_rows = parse_int_ranges(args.rows)
    candidates, coverage, gt_cache = [], [], {}
    items = {i["_row"]: remap_gt_dir(i, args.dataset_root) for i in load_manifest(args.manifest, args.duration)
             if selected_rows is None or i["_row"] in selected_rows}
    if not items:
        parser.error("No matching trajectories")
    if args.diagnose:
        diagnose(args, items)
        return
    for row, item in items.items():
        try:
            groups = visit_groups(load_poses(item, args.dataset_root), float(item["fps"]), args)
            accepted = None
            for group in groups:
                good, pairs, gt = verify_gt(item, group["frames"], args.min_gt_ssim)
                if good:
                    accepted = dict(group, row=row, scene=item["scene"], fps=float(item["fps"]), gt_pairs=pairs)
                    candidates.append(accepted)
                    gt_cache[row] = gt
                    break
            coverage.append({"row": row, "scene": item["scene"], "pose_groups": len(groups),
                             "status": "matched visits" if accepted else "no repeated GT-verified view"})
        except (OSError, ValueError, KeyError) as exc:
            coverage.append({"row": row, "status": "error", "error": str(exc)})
        print(coverage[-1], flush=True)
    candidates.sort(key=lambda c: (-len(c["frames"]), -c["span_sec"], c["row"]))
    selected = []
    for case in candidates:
        if len(selected) >= args.top:
            break
        item, frames = items[case["row"]], case["frames"]
        try:
            paths = {label: args.root / run / (item["output_prefix"] + "custom.mp4")
                     for label, run in (("Unbounded", "baseline"), ("Ours", args.ours_run))}
            images = {label: load_video_frames_single_pass(path, frames, args.decode_timeout) for label, path in paths.items()}
            case["scores"] = {label: [frame_metrics(np.asarray(img[i]), read_gt_frame(item, i, np.asarray(img[i]).shape))
                                      for i in frames] for label, img in images.items()}
            case["source_videos"] = {label: str(path) for label, path in paths.items()}
            output = args.output / f"revisit_{len(selected) + 1:02d}_row{case['row']}"
            render(case, images, gt_cache[case["row"]], output)
            output.with_suffix(".json").write_text(json.dumps(case, indent=2) + "\n")
            selected.append(case)
        except (OSError, ValueError, RuntimeError) as exc:
            coverage.append({"row": case["row"], "status": "render error", "error": str(exc)})
    (args.output / "revisit_search.json").write_text(json.dumps({
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "coverage": coverage, "selected": selected,
        "selection": "Pose/GT only, one group per trajectory; prioritize visit count then temporal span. No policy-quality filter.",
        "note": "Separate returns require a sustained departure beyond twice the pose tolerance. Same requested view, not proof the generated camera followed it. A single GT reference represents pairwise-GT-verified visits; exact-index GT is used for scores. This shows revisit behavior, not causal memory propagation."
    }, indent=2) + "\n")
    print(f"{len(selected)} same-view revisit figures saved to {args.output}")
    if not selected:
        print("No qualifying repeated views. No arbitrary-time fallback was substituted.")


if __name__ == "__main__":
    main()
