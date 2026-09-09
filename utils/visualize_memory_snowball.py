"""CPU-only search for video degradation following low-fidelity memory reads.

Ranks observed episodes, not causal effects. Outputs exact-index PSNR/SSIM
curves, before/after/later images, actual retrieved memories, and trace evidence.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.analyze_selected_memory_image_quality import load_manifest, read_gt_frame
from utils.evaluate_context_memory import frame_metrics
from utils.visualize_common_source_psnr_extremes import load_selected_queries, parse_int_ranges, remap_gt_dir
from utils.visualize_geometric_coverage_evictions import load_video_frames_single_pass

SECTION = 76
RED, GREEN = "#b93838", "#177b58"


def logged_reads(queries, frame_count):
    output = []
    for (section, target), event in sorted(queries.items()):
        source = int(event["selected_memory_frame"])
        if not 0 <= source < section * SECTION + 1 <= target < frame_count:
            raise ValueError(f"Noncausal or out-of-range retrieval: {section}, {target}, {source}")
        if target > (section + 1) * SECTION:
            raise ValueError("Trace does not match the 76-frame section layout")
        if (event.get("context_content_override")
                or event.get("context_content_source", "generated_memory") != "generated_memory"
                or event.get("selection_source", "retriever") != "retriever"):
            raise ValueError("Replay/override traces cannot be used as ordinary rollout evidence")
        output.append(event)
    return output


def sample_sections(frame_count, stride):
    return {section: list(range(start, min(start + SECTION, frame_count), stride))
            for section, start in enumerate(range(1, frame_count, SECTION))}


def decode_small(video, indices, width, timeout):
    decoded = load_video_frames_single_pass(video, indices, timeout_sec=timeout)
    small = {}
    for i in sorted(indices):
        image = decoded.pop(i)
        height = max(16, round(image.height * width / image.width))
        small[i] = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC))
    return small


def section_scores(samples, quality):
    return {s: {metric: float(np.mean([quality[i][metric] for i in indices]))
                for metric in ("psnr_db", "ssim")} for s, indices in samples.items()}


def good_overlap(event, threshold):
    overlap = event.get("selected_overlap")
    return overlap is not None and np.isfinite(float(overlap)) and float(overlap) >= threshold


def rank_episodes(reads, base_quality, base_sections, policy_sections, fps, args):
    candidates = []
    for section in sorted(base_sections):
        previous = list(range(section - args.window_sections, section))
        later = list(range(section + 1, section + 1 + args.window_sections))
        if any(s not in base_sections for s in previous + later):
            continue
        eligible = [r for r in reads if int(r["section_idx"]) == section
                    and int(r["selected_memory_frame"]) > 0
                    and int(r["target_frame"]) - int(r["selected_memory_frame"]) >= fps * args.min_age_sec
                    and good_overlap(r, args.min_overlap)]
        if not eligible:
            continue
        event = min(eligible, key=lambda r: (base_quality[int(r["selected_memory_frame"])]["psnr_db"],
                                            int(r["target_frame"])))
        memory = base_quality[int(event["selected_memory_frame"])]
        before = {m: float(np.mean([base_sections[s][m] for s in previous])) for m in ("psnr_db", "ssim")}
        after = base_sections[section]
        late = {m: float(np.mean([base_sections[s][m] for s in later])) for m in before}
        geo_late = {m: float(np.mean([policy_sections[s][m] for s in later])) for m in before}
        # These are real reads of newly generated evidence from the affected section.
        reuses = [r for r in reads if int(r["section_idx"]) in later
                  and section * SECTION < int(r["selected_memory_frame"]) <= (section + 1) * SECTION
                  and good_overlap(r, args.min_overlap)
                  and base_quality[int(r["selected_memory_frame"])]["psnr_db"] <= args.max_memory_psnr]
        drop = before["psnr_db"] - after["psnr_db"]
        late_drop = before["psnr_db"] - late["psnr_db"]
        gain = geo_late["psnr_db"] - late["psnr_db"]
        tests = {
            "corrupted_memory": memory["psnr_db"] <= args.max_memory_psnr,
            "good_before": before["psnr_db"] >= args.min_before_psnr,
            "immediate_drop": drop >= args.min_drop_db,
            "persistent_drop": late_drop >= args.min_drop_db,
            "further_worsening": after["psnr_db"] - late["psnr_db"] >= args.min_later_drop_db,
            "ssim_drop": min(before["ssim"] - after["ssim"], before["ssim"] - late["ssim"]) >= args.min_ssim_drop,
            "policy_advantage": gain >= args.min_policy_gain,
            "downstream_reuse": len(reuses) >= args.min_reuses,
        }
        candidates.append({"section": section, "event": event, "reuses": reuses,
                           "memory": memory, "before": before, "after": after, "later": late,
                           "immediate_drop_db": drop, "persistent_drop_db": late_drop,
                           "late_policy_gain_db": gain, "qualified": all(tests.values()), "criteria": tests,
                           "rank_score": min(drop, late_drop)})
    return sorted(candidates, key=lambda c: (-c["rank_score"], -c["late_policy_gain_db"], c["section"]))


def representative(indices, quality):
    center = float(np.median([quality[i]["psnr_db"] for i in indices]))
    center_time = float(np.median(indices))
    return min(indices, key=lambda i: (abs(quality[i]["psnr_db"] - center), abs(i - center_time), i))


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    return plt


def quality_axes(axes, curve, policy_label):
    for ax, metric, label in zip(axes, ("psnr_db", "ssim"), ("PSNR (dB)", "SSIM")):
        for run, color, name in (("baseline", RED, "Unbounded"), ("policy", GREEN, policy_label)):
            times = [r["time_sec"] for r in curve] + [curve[-1]["end_sec"]]
            values = [r[f"{run}_{metric}"] for r in curve] + [curve[-1][f"{run}_{metric}"]]
            ax.step(times, values, where="post", color=color, label=name, linewidth=1.8)
        ax.set_ylabel(label + " (higher is better)")
        ax.set_xlabel("Generated video time (seconds)")
        ax.grid(alpha=.18)
    axes[0].legend(loc="best", frameon=False)


def render_episode(case, item, args, directory):
    plt = plotting()
    frames = case["display_frames"]
    source = int(case["event"]["selected_memory_frame"])
    name = item["output_prefix"] + "custom.mp4"
    base = load_video_frames_single_pass(args.root / args.reference_run / name, set(frames + [source]), args.decode_timeout)
    geo = load_video_frames_single_pass(args.root / args.policy_run / name, frames, args.decode_timeout)
    gt = {i: read_gt_frame(item, i, (base[i].height, base[i].width, 3)) for i in set(frames + [source])}
    fig = plt.figure(figsize=(15, 11), layout="constrained")
    grid = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1.05, 1.05, 1.05], width_ratios=[.86, 1, 1, 1])
    axes = [fig.add_subplot(grid[i, :]) for i in range(2)]
    quality_axes(axes, case["curve"], args.policy_label)
    start = (case["section"] * SECTION + 1) / item["fps"]
    end = ((case["section"] + 1) * SECTION) / item["fps"]
    for ax in axes:
        ax.axvspan(start, end, color=RED, alpha=.1)
        ax.axvline(start, color=RED, linestyle="--", linewidth=1)
        for reuse in sorted({int(r["section_idx"]) for r in case["reuses"]}):
            ax.axvline((reuse * SECTION + 1) / item["fps"], color="#666666", linestyle=":", linewidth=1)
    axes[0].set_title("Dashed: section receives displayed memory. Dotted: low-fidelity frames from that section retrieved again.", fontsize=10)
    for row, (images, label, color) in enumerate(((gt, "Ground truth", "#333333"),
                                                (base, "Unbounded", RED), (geo, args.policy_label, GREEN))):
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(grid[row + 2, col + 1])
            ax.imshow(images[frame])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(label, color=color, fontweight="bold")
            if row == 0:
                ax.set_title(f"{('Before read', 'After read', 'Later')[col]} | {frame / item['fps']:.1f}s (f{frame})", fontsize=10)
            else:
                metric = case["display_quality"]["baseline" if row == 1 else "policy"][col]
                ax.set_xlabel(f"{metric['psnr_db']:.2f} dB | SSIM {metric['ssim']:.3f}", color=color)
    for row, image, title in ((2, gt[source], f"GT at memory index f{source}"),
                              (3, base[source], f"Actual retrieved memory f{source}")):
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    ax = fig.add_subplot(grid[4, 0])
    ax.axis("off")
    ax.text(0, .98, f"Logged read into section {case['section']}\nTarget f{case['event']['target_frame']}\n"
            f"FOV overlap {case['event']['selected_overlap']:.3f}\nMemory PSNR {case['memory']['psnr_db']:.2f} dB\n\n"
            f"Output drop: {case['immediate_drop_db']:.2f} dB\nLater drop: {case['persistent_drop_db']:.2f} dB\n"
            f"Later reuse slots: {len(case['reuses'])}", va="top", fontsize=10, linespacing=1.5)
    inspection = getattr(args, "inspection", False)
    heading = "Inspection candidate: memory read and subsequent output" if inspection else "Video degradation following a corrupted-memory read"
    failed = [name for name, passed in case["criteria"].items() if not passed]
    suffix = f"\nOriginal filters: {len(failed)} failed; see inspection.csv" if inspection else ""
    fig.suptitle(f"{heading} | {item['scene']}{suffix}", fontsize=14)
    notice = "Unconfirmed inspection candidate. " if inspection else "Selected observational example, not causal proof. "
    fig.supxlabel(notice + "Exact-index GT; section means at "
                  f"{args.score_width}px width. Images chosen near median frame PSNR in each window.", fontsize=9)
    fig.savefig(directory.with_suffix(".png"), dpi=160)
    fig.savefig(directory.with_suffix(".pdf"))
    plt.close(fig)


def render_overall(curves, output, policy_label):
    plt = plotting()
    # Use the common temporal support so the cohort cannot change along the curve.
    shared = sorted(set.intersection(*(set(r["section"] for r in curve) for curve in curves)))
    aggregate = []
    for section in shared:
        rows = [next(r for r in curve if r["section"] == section) for curve in curves]
        times = [r["time_sec"] for r in rows]
        if not np.allclose(times, times[0]):
            raise ValueError("Cannot aggregate trajectories with different section times/FPS")
        aggregate.append({"section": section, "time_sec": times[0], "end_sec": min(r["end_sec"] for r in rows), "n": len(rows),
                          **{key: float(np.mean([r[key] for r in rows])) for key in
                             ("baseline_psnr_db", "baseline_ssim", "policy_psnr_db", "policy_ssim")}})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    quality_axes(axes, aggregate, policy_label)
    fig.suptitle(f"Generated-video quality over time | {len(curves)} matched trajectories")
    fig.supxlabel("Equal-trajectory means over all successfully analyzed inputs; not conditioned on selected extreme cases.", fontsize=9)
    fig.savefig(output / "quality_over_time.png", dpi=170)
    fig.savefig(output / "quality_over_time.pdf")
    plt.close(fig)
    return aggregate


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_cached(cache, output, top):
    """Render ranked candidates without changing thresholds or recomputing metrics."""
    report = json.loads((cache / "search.json").read_text())
    args = argparse.Namespace(**report["parameters"])
    for key in ("root", "manifest", "dataset_root"):
        value = getattr(args, key)
        setattr(args, key, Path(value) if value is not None else None)
    args.inspection = True
    items = {item["_row"]: remap_gt_dir(item, args.dataset_root)
             for item in load_manifest(args.manifest, args.duration)}
    curves = {}
    with (cache / "section_quality.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            row = {k: int(v) if k in ("row", "section") else float(v) for k, v in row.items()}
            curves.setdefault(row["row"], []).append(row)
    # Baseline-only ranking; GeoCov advantage is neither a gate nor a tie-break.
    ranked = sorted(report["candidates"], key=lambda c: (
        -min(c["immediate_drop_db"], c["persistent_drop_db"]), c["row"], c["section"]))
    selected, seen = [], set()
    for case in ranked:
        if case["row"] in seen:
            continue
        selected.append(case)
        seen.add(case["row"])
        if len(selected) >= top:
            break
    rows = []
    for index, case in enumerate(selected, 1):
        stem = f"inspection_{index:02d}"
        failed = [name for name, passed in case["criteria"].items() if not passed]
        rows.append({"figure": stem + ".png", "row": case["row"], "scene": case["scene"],
                     "section": case["section"], "original_qualified": case["qualified"],
                     "immediate_drop_db": case["immediate_drop_db"],
                     "persistent_drop_db": case["persistent_drop_db"],
                     "reuse_slots": len(case["reuses"]), "failed_criteria": ",".join(failed)})
        print(f"Rendering {stem}: {case['scene']}; failed checks: {','.join(failed) or 'none'}", flush=True)
        render_episode(dict(case, curve=sorted(curves[case["row"]], key=lambda r: r["section"])),
                       items[case["row"]], args, output / stem)
    write_csv(output / "inspection.csv", rows)
    (output / "inspection.json").write_text(json.dumps({
        "source_search": str((cache / "search.json").resolve()),
        "original_parameters": report["parameters"], "selected": selected,
        "ranking": "min(immediate_drop_db, persistent_drop_db), descending; one per trajectory; no policy-advantage filter",
    }, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Inspection candidates, NOT newly qualified snowball examples. Original thresholds and decisions are unchanged.\n"
        "Ranked by the smaller of immediate and persistent baseline PSNR drops, at most one per trajectory.\n"
        "No GeoCov advantage is required. The candidate pool still inherits the original age/overlap/window restrictions.\n"
        "Scores and frame choices are reused from the cached search; only illustration frames are decoded again.\n"
        "See inspection.csv for failed checks. Check whether the SAME visible error recurs in memory and output.\n"
        "Whole-frame PSNR/SSIM cannot distinguish rendering defects from incorrect scene content or establish causality.\n")
    print(f"Output: {output}; {len(selected)} inspection figures; no metric recomputation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reference-run", default="baseline")
    parser.add_argument("--policy-run", default="slam_b32_covisibility")
    parser.add_argument("--policy-label", default="GeoCov-32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-from", type=Path,
                        help="Render unconfirmed candidates from an existing search directory without rescoring")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--score-width", type=int, default=256)
    parser.add_argument("--window-sections", type=int, default=2)
    parser.add_argument("--min-age-sec", type=float, default=5.)
    parser.add_argument("--min-overlap", type=float, default=.8)
    parser.add_argument("--max-memory-psnr", type=float, default=12.)
    parser.add_argument("--min-before-psnr", type=float, default=14.)
    parser.add_argument("--min-drop-db", type=float, default=2.)
    parser.add_argument("--min-later-drop-db", type=float, default=.5)
    parser.add_argument("--min-ssim-drop", type=float, default=.03)
    parser.add_argument("--min-policy-gain", type=float, default=1.)
    parser.add_argument("--min-reuses", type=int, default=1)
    parser.add_argument("--decode-timeout", type=int, default=600)
    args = parser.parse_args()
    if (args.top < 1 or not 1 <= args.frame_stride <= SECTION or args.score_width < 16
            or args.window_sections < 1 or args.min_reuses < 0 or args.min_age_sec < 0
            or not 0 <= args.min_overlap <= 1 or args.decode_timeout < 1):
        parser.error("Invalid search/sampling arguments")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Use an empty output directory to avoid mixing searches")
    if args.inspect_from is not None:
        inspect_cached(args.inspect_from, args.output, args.top)
        return
    row_filter = parse_int_ranges(args.rows)
    items = [remap_gt_dir(item, args.dataset_root) for item in load_manifest(args.manifest, args.duration)
             if row_filter is None or item["_row"] in row_filter]
    if not items:
        parser.error("No matching manifest items")
    coverage, all_cases, curves, frame_rows = [], [], [], []
    for item in items:
        record = {"row": item["_row"], "scene": item["scene"]}
        coverage.append(record)
        print(f"[row {item['_row']}] {item['scene']}", flush=True)
        try:
            name = item["output_prefix"] + "custom"
            trace = args.root / args.reference_run / "access_traces" / (name + ".jsonl")
            reads = logged_reads(load_selected_queries(trace, item, strict=True), int(item["num_frames"]))
            samples = sample_sections(int(item["num_frames"]), args.frame_stride)
            sampled = {i for indices in samples.values() for i in indices}
            requested = sampled | {int(r["selected_memory_frame"]) for r in reads}
            print(f"  {len(reads)} logged reads; decoding/scoring {len(requested)} baseline and {len(sampled)} policy frames", flush=True)
            base_images = decode_small(args.root / args.reference_run / (name + ".mp4"), requested, args.score_width, args.decode_timeout)
            gt = {i: read_gt_frame(item, i, base_images[i].shape) for i in requested}
            base_quality = {i: frame_metrics(base_images[i], gt[i]) for i in requested}
            shape = next(iter(base_images.values())).shape
            del base_images
            policy_images = decode_small(args.root / args.policy_run / (name + ".mp4"), sampled, args.score_width, args.decode_timeout)
            if any(image.shape != shape for image in policy_images.values()):
                raise ValueError("Policy video resolution/aspect ratio differs")
            policy_quality = {i: frame_metrics(policy_images[i], gt[i]) for i in sampled}
            del policy_images, gt
            base_sections, policy_sections = section_scores(samples, base_quality), section_scores(samples, policy_quality)
            curve = [{"row": item["_row"], "section": s, "time_sec": (s * SECTION + 1) / item["fps"],
                      "end_sec": min((s + 1) * SECTION + 1, int(item["num_frames"])) / item["fps"],
                      **{f"baseline_{m}": base_sections[s][m] for m in ("psnr_db", "ssim")},
                      **{f"policy_{m}": policy_sections[s][m] for m in ("psnr_db", "ssim")}}
                     for s in samples]
            for i in sorted(requested):
                frame_rows.append({"row": item["_row"], "frame": i, "sampled_for_curve": i in sampled,
                                   "baseline_psnr_db": base_quality[i]["psnr_db"], "baseline_ssim": base_quality[i]["ssim"],
                                   "policy_psnr_db": policy_quality.get(i, {}).get("psnr_db"),
                                   "policy_ssim": policy_quality.get(i, {}).get("ssim")})
            cases = rank_episodes(reads, base_quality, base_sections, policy_sections, float(item["fps"]), args)
            for case in cases:
                s = case["section"]
                windows = [[i for sec in range(s - args.window_sections, s) for i in samples[sec]],
                           samples[s], [i for sec in range(s + 1, s + 1 + args.window_sections) for i in samples[sec]]]
                frames = [representative(window, base_quality) for window in windows]
                case.update(row=item["_row"], scene=item["scene"], trace=str(trace), display_frames=frames,
                            display_quality={"baseline": [base_quality[i] for i in frames],
                                             "policy": [policy_quality[i] for i in frames]})
            all_cases.extend(cases)
            curves.append(curve)
            record.update(status="scored", reads=len(reads), candidates=len(cases), qualified=sum(c["qualified"] for c in cases))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            record.update(status="error", error=str(exc))
        print(record, flush=True)
    ranked = sorted(all_cases, key=lambda c: (-c["rank_score"], -c["late_policy_gain_db"], c["row"], c["section"]))
    chosen, seen = [], set()
    for case in ranked:
        if case["qualified"] and case["row"] not in seen:
            chosen.append(case)
            seen.add(case["row"])
        if len(chosen) == args.top:
            break
    report = {"parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "coverage": coverage, "selected": chosen, "candidates": ranked}
    (args.output / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output / "frame_quality.csv", frame_rows)
    write_csv(args.output / "section_quality.csv", [r for curve in curves for r in curve])
    write_csv(args.output / "ranked_episodes.csv", [{"row": c["row"], "scene": c["scene"], "section": c["section"],
              "qualified": c["qualified"], "memory_frame": c["event"]["selected_memory_frame"],
              "memory_psnr": c["memory"]["psnr_db"], "immediate_drop_db": c["immediate_drop_db"],
              "persistent_drop_db": c["persistent_drop_db"], "late_policy_gain_db": c["late_policy_gain_db"],
              "reuse_slots": len(c["reuses"]), "failed_criteria": ",".join(k for k, v in c["criteria"].items() if not v)} for c in ranked])
    if curves:
        write_csv(args.output / "quality_over_time.csv", render_overall(curves, args.output, args.policy_label))
    for i, case in enumerate(chosen, 1):
        item = next(item for item in items if item["_row"] == case["row"])
        case = dict(case, curve=next(c for c in curves if c[0]["row"] == case["row"]))
        render_episode(case, item, args, args.output / f"episode_{i:02d}")
    (args.output / "README.txt").write_text(
        f"{len(chosen)} qualifying examples. {len(curves)}/{len(items)} trajectories analyzed. See search.json for errors and thresholds.\n"
        "Overall curves use ALL analyzed trajectories, not only selected examples. They may or may not show degradation.\n"
        "Episode ranking: smaller of immediate and persistent PSNR drops, then late comparator advantage; at most one per trajectory.\n"
        "Default filters require a poor old memory, high FOV overlap, good preceding output, PSNR/SSIM deterioration and later reuse of corrupted output.\n"
        "Before/later windows span two sections by default. Display frames are nearest median unbounded frame PSNR in each window.\n"
        "Curves are step plots over generation-section boundaries, avoiding interpolated drops before a logged read.\n"
        "Metrics use exact-index ground truth, CPU PSNR/SSIM at configured width; no learned models.\n"
        "These are selected temporal associations, not proof that the read caused the drop or that archive cardinality caused it.\n"
        "Inspect the images for a persistent concrete defect; metrics alone cannot establish that the same object error was copied.\n")
    print(f"Output: {args.output}; {len(chosen)} qualifying episodes; {len(curves)}/{len(items)} trajectories analyzed", flush=True)
    if not chosen:
        print("No example met every threshold. No fallback figure fabricated; inspect ranked_episodes.csv.")
    raise SystemExit(0 if chosen and len(curves) == len(items) else 2)


if __name__ == "__main__":
    main()
