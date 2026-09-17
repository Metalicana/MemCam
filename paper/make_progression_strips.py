"""CPU-only, matched-time Unbounded / FIFO / Ours progression candidates.

Post-hoc ranking uses exact-index SSIM, which includes wrong-view errors and
does not measure visual artifacts alone. Outputs require visual review.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from make_memory_strips import (
    REPO, RUNS, decode, load_manifest, parse_int_ranges, read_gt_frame,
    remap_gt_dir, frame_metrics, save_json, strip, video,
)


LABELS = ("Unbounded", "FIFO", "Ours")
NOTE = (
    "Exploratory, outcome-selected examples, not representative cohort results. "
    "All policies use the same five equally spaced sample positions. "
    "SSIM/PSNR measure exact-index fidelity, including viewpoint and scene errors, "
    "not artifact quality alone. Inspect pixels for progressive deterioration. "
    "Neither the ranking nor these frames establish causal memory propagation."
)


def sample_frames(item, interval_sec):
    count, fps = int(item["num_frames"]), float(item["fps"])
    if count < 6 or fps <= 0:
        raise ValueError("Need at least five generated frames and a positive FPS")
    frames = np.arange(1, count, max(1, round(interval_sec * fps)), dtype=int).tolist()
    if len(frames) < 5:
        raise ValueError("Fewer than five samples; reduce --sample-sec")
    return frames


def choose_sequence(frames, scores, fps, min_span_sec):
    """Search uniform five-frame sequences; never select different times per policy."""
    best = None
    values = {label: np.asarray([v["ssim"] for v in scores[label]]) for label in LABELS}
    if not all(len(v) == len(frames) and np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("Missing or nonfinite SSIM scores")
    for step in range(1, (len(frames) - 1) // 4 + 1):
        for start in range(len(frames) - 4 * step):
            positions = [start + step * i for i in range(5)]
            chosen = [frames[i] for i in positions]
            if (chosen[-1] - chosen[0]) / fps < min_span_sec:
                continue
            selected = {label: values[label][positions] for label in LABELS}
            changes = {label: float(v[0] - v[-1]) for label, v in selected.items()}
            # Reward declines in BOTH comparators; penalize their recoveries and
            # any instability in Ours. No sequence is automatically called a success.
            decline = min(changes[label] - float(np.maximum(np.diff(selected[label]), 0).sum())
                          for label in LABELS[:2])
            ours_variation = float(np.abs(np.diff(selected["Ours"])).sum())
            final_margin = min(float(selected["Ours"][-1] - selected[label][-1]) for label in LABELS[:2])
            rank = decline - ours_variation + min(final_margin, 0.)
            case = {"frames": chosen, "sample_positions": positions, "rank_score": rank,
                    "ssim_drop": changes,
                    "declining_steps": {label: int(np.sum(np.diff(v) < 0)) for label, v in selected.items()},
                    "ours_ssim_range": float(np.ptp(selected["Ours"])),
                    "ours_final_margin": final_margin,
                    "display_scores": {label: [scores[label][i] for i in positions] for label in LABELS}}
            if best is None or case["rank_score"] > best["rank_score"]:
                best = case
    if best is None:
        raise ValueError("No five-frame sequence spans --min-span-sec; reduce it or --sample-sec")
    return best


def render(case, images, gt, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, fps = case["frames"], case["fps"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    aspect = images["Unbounded"][frames[0]].height / images["Unbounded"][frames[0]].width
    fig, axes = plt.subplots(3, 5, figsize=(15, 9 * aspect + .6))
    fig.subplots_adjust(left=.07, right=.997, bottom=.015, top=.92, wspace=.025, hspace=.04)
    for row, label in enumerate(LABELS):
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            ax.imshow(images[label][frame])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(f"{frame / fps:.1f} s", fontsize=12)
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8, fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=180, facecolor="white")
    plt.close(fig)
    for label in LABELS:
        strip([images[label][i] for i in frames], [f"{i / fps:.1f} s" for i in frames],
              output.with_name(output.name + "_" + label.lower()))
    strip([gt[i] for i in frames], [f"GT | {i / fps:.1f} s" for i in frames],
          output.with_name(output.name + "_gt_check"))
    # Plot all sampled scores, not just the five selected points.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("psnr_db", "ssim"), ("Exact-index PSNR (dB)", "Exact-index SSIM")):
        for label, color in zip(LABELS, ("#B93838", "#D18322", "#0072B2")):
            ax.plot(np.asarray(case["sample_frames"]) / fps,
                    [s[metric] for s in case["scores"][label]], label=label, color=color)
            ax.scatter(np.asarray(frames) / fps, [s[metric] for s in case["display_scores"][label]],
                       color=color, s=18)
        ax.set(xlabel="Video time (s)", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_name(output.name + "_fidelity").with_suffix("." + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--sample-sec", type=float, default=2.)
    parser.add_argument("--min-span-sec", type=float, default=20.)
    parser.add_argument("--score-width", type=int, default=256)
    parser.add_argument("--decode-timeout", type=int, default=600)
    parser.add_argument("--fifo-run", default="fifo_b32")
    parser.add_argument("--ours-run", default="slam_b32_covisibility")
    args = parser.parse_args()
    if (min(args.sample_sec, args.min_span_sec, args.top, args.decode_timeout) <= 0
            or args.score_width < 16):
        parser.error("Positive counts/times and score width >=16 are required")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    rows = parse_int_ranges(args.rows)
    items = {i["_row"]: remap_gt_dir(i, args.dataset_root) for i in load_manifest(args.manifest, args.duration)
             if rows is None or i["_row"] in rows}
    if not items:
        parser.error("No matching trajectories")
    runs = dict(RUNS, FIFO=args.fifo_run, Ours=args.ours_run)
    candidates, coverage = [], []
    for row, item in items.items():
        print(f"[row {row}] {item['scene']}: scoring Unbounded / FIFO / Ours", flush=True)
        try:
            frames = sample_frames(item, args.sample_sec)
            paths = {label: video(args.root, run, item) for label, run in runs.items()}
            for path in paths.values():
                if not path.is_file():
                    raise ValueError(f"Missing video: {path}")
            scores = {}
            for label, path in paths.items():
                images = decode(path, frames, args, small=True)
                scores[label] = [frame_metrics(np.asarray(images[i]), read_gt_frame(item, i, np.asarray(images[i]).shape))
                                 for i in frames]
            case = dict(choose_sequence(frames, scores, float(item["fps"]), args.min_span_sec),
                        row=row, scene=item["scene"], fps=float(item["fps"]), sample_frames=frames,
                        scores=scores, source_videos={label: str(p) for label, p in paths.items()},
                        source_manifest_item=item, note=NOTE)
            candidates.append(case)
            save_json(args.output / f"scan_row{row}.json", case)
            coverage.append({"row": row, "status": "scored"})
            print(f"  SSIM start-end drop: " + ", ".join(f"{label}={case['ssim_drop'][label]:+.3f}" for label in LABELS), flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            coverage.append({"row": row, "status": "error", "error": str(exc)})
            print(coverage[-1], flush=True)
    candidates.sort(key=lambda c: (-c["rank_score"], c["row"]))
    selected = []
    for case in candidates:
        if len(selected) >= args.top:
            break
        try:
            images = {label: decode(Path(path), case["frames"], args) for label, path in case["source_videos"].items()}
            gt = {i: Image.fromarray(read_gt_frame(items[case["row"]], i, np.asarray(images["Unbounded"][i]).shape))
                  for i in case["frames"]}
            stem = args.output / f"progression_{len(selected) + 1:02d}_row{case['row']}"
            render(case, images, gt, stem)
            save_json(stem.with_suffix(".json"), case)
            selected.append(case)
            print(f"Rendered {stem.name}: {case['scene']}", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            coverage.append({"row": case["row"], "status": "render error", "error": str(exc)})
            print(coverage[-1], flush=True)
    save_json(args.output / "progression_search.json", {
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "coverage": coverage, "candidates": candidates, "selected": selected, "note": NOTE,
        "ranking": "Minimum baseline net SSIM decline minus recovery, minus Ours total SSIM variation, penalized if Ours finishes below either baseline. No pass/fail cutoff.",
    })
    with (args.output / "ranking.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "row", "scene", "score", "frames", "unbounded_ssim_drop", "fifo_ssim_drop", "ours_ssim_drop"])
        for n, case in enumerate(candidates, 1):
            writer.writerow([n, case["row"], case["scene"], case["rank_score"], ";".join(map(str, case["frames"])),
                             *(case["ssim_drop"][label] for label in LABELS)])
    print(f"{len(selected)} progression candidates saved to {args.output}. Visual review required.")
    if not selected:
        raise SystemExit("No figures rendered; see progression_search.json for errors.")


if __name__ == "__main__":
    main()
