"""CPU-only retrieval and read/reuse strips, with evidence in JSON sidecars.

Retrieval defaults to common baseline pixels at each policy's logged index.
Snowball candidates use a cached CPU scan and actual read/reuse links; they
are observational examples for visual inspection, not causal demonstrations.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.analyze_selected_memory_image_quality import load_manifest, read_gt_frame
from utils.evaluate_context_memory import frame_metrics
from utils.visualize_common_source_psnr_extremes import (
    fit_image, fitting_font, load_selected_queries, parse_int_ranges, remap_gt_dir,
)
from utils.visualize_geometric_coverage_evictions import load_video_frames_single_pass
from utils.visualize_memory_snowball import SECTION, good_overlap, logged_reads


RUNS = {"Unbounded": "baseline", "FIFO": "fifo_b32", "Ours": "slam_b32_covisibility"}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def strip(images, labels, path, width=384):
    """Keep full images, separated by white gutters; also export a label-free strip."""
    if not images or len(images) != len(labels):
        raise ValueError("One label is required per image")
    height = round(width * images[0].height / images[0].width)
    gap, header = 8, 36
    raw = Image.new("RGB", (len(images) * width + (len(images) - 1) * gap, height), "white")
    for i, image in enumerate(images):
        raw.paste(fit_image(image, width, height), (i * (width + gap), 0))
    raw.save(path.with_name(path.name + "_bare").with_suffix(".png"))
    labeled = Image.new("RGB", (raw.width, height + header), "white")
    labeled.paste(raw, (0, header))
    draw = ImageDraw.Draw(labeled)
    for i, label in enumerate(labels):
        draw.text((i * (width + gap) + width // 2, header // 2), label,
                  anchor="mm", fill="#222222",
                  font=fitting_font(draw, label, 18, width - 12, bold=True))
    labeled.save(path.with_suffix(".png"))
    labeled.save(path.with_suffix(".pdf"), resolution=150)


def video(root, run, item):
    return root / run / (item["output_prefix"] + "custom.mp4")


def reads(root, run, item):
    path = root / run / "access_traces" / (item["output_prefix"] + "custom.jsonl")
    queries = load_selected_queries(path, item, strict=True)
    logged_reads(queries, int(item["num_frames"]))
    return queries


def decode(path, indices, args, small=False):
    images = load_video_frames_single_pass(path, sorted(set(indices)), args.decode_timeout)
    if small:
        images = {i: image.resize((args.score_width, round(args.score_width * image.height / image.width)),
                                 Image.Resampling.BICUBIC) for i, image in images.items()}
    return images


def diverse(cases, top):
    selected, seen = [], set()
    for case in cases:
        if case["row"] not in seen:
            selected.append(case)
            seen.add(case["row"])
        if len(selected) == top:
            break
    return selected


def retrieval(args, items):
    runs = dict(RUNS, FIFO=args.fifo_run, Ours=args.ours_run)
    requested = None
    if args.cases:
        with args.cases.open(newline="") as handle:
            requested = defaultdict(set)
            for row in csv.DictReader(handle):
                requested[int(row["row"])].add((int(row["section_idx"]), int(row["target_frame"])))
    cases, coverage = [], []
    for item in items.values():
        row = item["_row"]
        if requested is not None and row not in requested:
            continue
        try:
            traces = {label: reads(args.root, run, item) for label, run in runs.items()}
            shared = set.intersection(*(set(t) for t in traces.values()))
            if requested is not None:
                if not requested[row] <= shared:
                    raise ValueError("Requested examples do not have matching Unbounded/FIFO/Ours trace queries")
                keys = sorted(requested[row])
            else:
                keys = [key for key in sorted(shared)
                        if (key[1] - (key[0] * SECTION + 1)) % args.target_stride == 0
                        and traces["Unbounded"][key]["selected_memory_frame"] != traces["Ours"][key]["selected_memory_frame"]]
            if not keys:
                coverage.append({"row": row, "status": "no matched disagreements"})
                continue
            indices = {label: {int(t[key]["selected_memory_frame"]) for key in keys} for label, t in traces.items()}
            if args.content == "common":
                common = decode(video(args.root, "baseline", item), set.union(*indices.values()), args, small=True)
                images = {label: common for label in runs}
            else:
                images = {label: decode(video(args.root, run, item), indices[label], args, small=True)
                          for label, run in runs.items()}
            for key in keys:
                selected = {label: int(t[key]["selected_memory_frame"]) for label, t in traces.items()}
                target = read_gt_frame(item, key[1], np.asarray(images["Unbounded"][selected["Unbounded"]]).shape)
                scores = {}
                for label, index in selected.items():
                    pixels = np.asarray(images[label][index])
                    own_gt = read_gt_frame(item, index, pixels.shape)
                    # Separate effective target mismatch, stored corruption, and clean view mismatch.
                    scores[label] = {
                        "target_match": frame_metrics(pixels, target),
                        "historical_fidelity": frame_metrics(pixels, own_gt),
                        "clean_view_match": frame_metrics(own_gt, target),
                    }
                delta = scores["Ours"]["target_match"]["ssim"] - scores["Unbounded"]["target_match"]["ssim"]
                cases.append({"row": row, "scene": item["scene"], "section": key[0], "target": key[1],
                              "selected": selected, "scores": scores, "rank_target_ssim_gain": delta,
                              "events": {label: t[key] for label, t in traces.items()}})
            coverage.append({"row": row, "status": "scored", "queries": len(keys)})
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            coverage.append({"row": row, "status": "error", "error": str(exc)})
        print(coverage[-1], flush=True)
    cases.sort(key=lambda c: (-c["rank_target_ssim_gain"], c["row"], c["target"]))
    chosen = diverse(cases if requested is not None else [c for c in cases if c["rank_target_ssim_gain"] >= args.min_target_gain], args.top)
    for number, case in enumerate(chosen, 1):
        item = items[case["row"]]
        if args.content == "common":
            common = decode(video(args.root, "baseline", item), case["selected"].values(), args)
            selected_images = [common[index] for index in case["selected"].values()]
        else:
            selected_images = [decode(video(args.root, runs[label], item), [index], args)[index]
                               for label, index in case["selected"].items()]
        gt = Image.fromarray(read_gt_frame(item, case["target"], np.asarray(selected_images[0]).shape))
        stem = args.output / f"retrieval_{number:02d}_row{case['row']}"
        strip([gt, *selected_images], ["Target ground truth", "Unbounded", "FIFO", "Ours"], stem)
        save_json(stem.with_suffix(".json"), dict(case, content_mode=args.content,
                  content_videos={label: str(video(args.root, "baseline" if args.content == "common" else run, item))
                                  for label, run in runs.items()}))
    save_json(args.output / "retrieval_search.json", {"coverage": coverage, "selected": chosen, "candidates": cases,
        "content_mode": args.content, "ranking": "Ours minus Unbounded target-match SSIM; one example per trajectory",
        "note": "Target match includes viewpoint mismatch, not only image quality. Common mode replaces content with baseline pixels; it is not actual policy-conditioned content."})
    print(f"{len(chosen)} retrieval strips; missing policies are never substituted.", flush=True)


def find_chains(events, curves, frame_scores, args):
    """Find actual two-read chains; do not require a comparator win or monotonic decline."""
    grouped = defaultdict(list)
    for event in events:
        grouped[int(event["section_idx"])].append(event)
    cases = []
    for section, group in sorted(grouped.items()):
        before_sections = list(range(section - 2, section))
        if any(s not in curves for s in before_sections + [section]):
            continue
        eligible = [e for e in group if int(e["selected_memory_frame"]) in frame_scores
                    and good_overlap(e, args.min_overlap)]
        if not eligible:
            continue
        first = min(eligible, key=lambda e: (frame_scores[int(e["selected_memory_frame"])]["baseline_psnr_db"], int(e["target_frame"])))
        before = {m: float(np.mean([curves[s][f"baseline_{m}"] for s in before_sections])) for m in ("psnr_db", "ssim")}
        for reuse_section in range(section + 1, section + args.horizon_sections + 1):
            if reuse_section not in curves or reuse_section + 1 not in curves:
                continue
            linked = [e for e in grouped[reuse_section]
                      if section * SECTION < int(e["selected_memory_frame"]) <= (section + 1) * SECTION
                      and good_overlap(e, args.min_overlap)]
            if not linked:
                continue
            second = min(linked, key=lambda e: (frame_scores[int(e["selected_memory_frame"])]["baseline_psnr_db"], int(e["target_frame"])))
            after = curves[section]
            later = {m: float(np.mean([curves[s][f"baseline_{m}"] for s in (reuse_section, reuse_section + 1)])) for m in before}
            drop = min(before["psnr_db"] - after["baseline_psnr_db"], before["psnr_db"] - later["psnr_db"])
            ssim_drop = min(before["ssim"] - after["baseline_ssim"], before["ssim"] - later["ssim"])
            memory = frame_scores[int(first["selected_memory_frame"])]
            reused = frame_scores[int(second["selected_memory_frame"])]
            checks = {"sustained_psnr_drop": drop >= args.min_drop_db,
                      "sustained_ssim_drop": ssim_drop >= args.min_ssim_drop,
                      "memory_below_pre_read_fidelity": memory["baseline_psnr_db"] < before["psnr_db"],
                      "reused_output_below_pre_read_fidelity": reused["baseline_psnr_db"] < before["psnr_db"]}
            cases.append({"section": section, "reuse_section": reuse_section, "first_read": first,
                          "second_read": second, "before": before, "after_reuse": later,
                          "sustained_drop_db": drop, "sustained_ssim_drop": ssim_drop,
                          "checks": checks, "screen_passed": all(checks.values())})
    return sorted(cases, key=lambda c: (-c["sustained_drop_db"], c["section"], c["reuse_section"]))


def plot_curve(curve, case, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
    rows = [curve[s] for s in sorted(curve)]
    for ax, metric, label in zip(axes, ("psnr_db", "ssim"), ("PSNR (dB)", "SSIM")):
        for run, name, color in (("baseline", "Unbounded", "#B93838"), ("policy", "Ours", "#0072B2")):
            ax.step([r["time_sec"] for r in rows] + [rows[-1]["end_sec"]],
                    [r[f"{run}_{metric}"] for r in rows] + [rows[-1][f"{run}_{metric}"]],
                    where="post", label=name, color=color)
        for section, style in ((case["section"], "--"), (case["reuse_section"], ":")):
            ax.axvline((section * SECTION + 1) / fps, color="#555555", linestyle=style, linewidth=1)
        ax.set(xlabel="Video time (s)", ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def snowball(args):
    report = json.loads((args.cache / "search.json").read_text())
    parameters = report["parameters"]
    root = args.root or Path(parameters["root"])
    manifest = args.manifest or Path(parameters["manifest"])
    dataset = args.dataset_root or parameters.get("dataset_root")
    items = {i["_row"]: remap_gt_dir(i, dataset) for i in load_manifest(manifest, parameters["duration"])}
    curves, scores = defaultdict(dict), defaultdict(dict)
    for filename, destination, key in (("section_quality.csv", curves, "section"), ("frame_quality.csv", scores, "frame")):
        with (args.cache / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                destination[int(row["row"])][int(row[key])] = {
                    k: float(v) for k, v in row.items() if k not in ("row", "section", "frame", "sampled_for_curve") and v != ""}
    candidates, coverage = [], []
    selected_rows = parse_int_ranges(args.rows)
    for row in sorted(curves):
        if selected_rows is not None and row not in selected_rows:
            continue
        try:
            events = list(reads(root, parameters["reference_run"], items[row]).values())
            cases = find_chains(events, curves[row], scores[row], args)
            candidates.extend(dict(c, row=row, scene=items[row]["scene"]) for c in cases)
            coverage.append({"row": row, "chains": len(cases), "passed": sum(c["screen_passed"] for c in cases)})
        except (OSError, ValueError, KeyError) as exc:
            coverage.append({"row": row, "error": str(exc)})
        print(coverage[-1], flush=True)
    candidates.sort(key=lambda c: (-c["sustained_drop_db"], c["row"], c["section"]))
    passed = diverse([c for c in candidates if c["screen_passed"]], args.top)
    inspection = diverse([c for c in candidates if not c["screen_passed"]], args.top) if not passed else []
    for number, case in enumerate(passed or inspection, 1):
        item = items[case["row"]]
        late_section = case["reuse_section"] + 1
        late_frames = [i for i in scores[case["row"]] if late_section * SECTION < i <= (late_section + 1) * SECTION]
        late = min(late_frames, key=lambda i: abs(i - (late_section * SECTION + SECTION // 2)))
        frames = [int(case["first_read"]["selected_memory_frame"]),
                  int(case["second_read"]["selected_memory_frame"]), int(case["second_read"]["target_frame"]), late]
        prefix = "snowball_candidate" if case["screen_passed"] else "inspection_only"
        stem = args.output / f"{prefix}_{number:02d}_row{case['row']}"
        base = decode(video(root, parameters["reference_run"], item), frames, args)
        ours = decode(video(root, parameters["policy_run"], item), frames, args)
        gt = {i: Image.fromarray(read_gt_frame(item, i, np.asarray(base[i]).shape)) for i in frames}
        times = [f"{i / item['fps']:.1f}s" for i in frames]
        stages = ("Retrieved memory", "Output retrieved again", "After reuse", "Later")
        strip([base[i] for i in frames], [f"{s} | {t}" for s, t in zip(stages, times)], stem.with_name(stem.name + "_unbounded"))
        strip([ours[i] for i in frames], [f"Ours | {t}" for t in times], stem.with_name(stem.name + "_ours"))
        strip([gt[i] for i in frames], [f"Ground truth | {t}" for t in times], stem.with_name(stem.name + "_gt"))
        plot_curve(curves[case["row"]], case, item["fps"], stem.with_name(stem.name + "_curve"))
        save_json(stem.with_suffix(".json"), dict(case, frames=frames,
            frame_scores={str(i): frame_metrics(np.asarray(base[i]), np.asarray(gt[i])) for i in frames},
            reference_video=str(video(root, parameters["reference_run"], item)),
            ours_video=str(video(root, parameters["policy_run"], item)),
            note="Ours shows output at the same indices, not its retrieved context. The first read conditions the section producing column 2; the second read reuses column 2 to condition the section containing column 3. Confirm the same visible error manually."))
    save_json(args.output / "snowball_search.json", {"cache": str(args.cache), "original_parameters": parameters,
        "screen_parameters": {"horizon_sections": args.horizon_sections, "min_overlap": args.min_overlap,
                              "min_drop_db": args.min_drop_db, "min_ssim_drop": args.min_ssim_drop},
        "coverage": coverage, "candidates": candidates, "selected": passed, "inspection_only": inspection,
        "note": "New exploratory screen over cached scores, not the original qualification rule. No comparator-advantage filter. PSNR/SSIM measure exact-index fidelity, not artifact identity or statistical significance. Cache source pixels were not reverified for the full curves. No causal claim."})
    print(f"{len(passed)} screened chains; {len(inspection)} inspection-only chains. Visual confirmation is still required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("retrieval", "snowball"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--decode-timeout", type=int, default=600)
    parser.add_argument("--score-width", type=int, default=256)
    parser.add_argument("--target-stride", type=int, default=16)
    parser.add_argument("--cases", type=Path, help="Existing selected_examples.csv; recover FIFO from matching traces")
    parser.add_argument("--content", choices=("common", "own"), default="common")
    parser.add_argument("--fifo-run", default="fifo_b32")
    parser.add_argument("--ours-run", default="slam_b32_covisibility")
    parser.add_argument("--min-target-gain", type=float, default=.05)
    parser.add_argument("--cache", type=Path, help="Existing visualize_memory_snowball.py output directory")
    parser.add_argument("--horizon-sections", type=int, default=6)
    parser.add_argument("--min-overlap", type=float, default=.8)
    parser.add_argument("--min-drop-db", type=float, default=1.)
    parser.add_argument("--min-ssim-drop", type=float, default=.02)
    args = parser.parse_args()
    if min(args.top, args.decode_timeout, args.target_stride, args.horizon_sections) < 1 or args.score_width < 16:
        parser.error("Counts, timeouts and strides must be positive; score width must be >=16")
    if not 0 <= args.min_overlap <= 1:
        parser.error("Overlap must be between 0 and 1")
    if args.mode == "snowball" and args.cache is None:
        parser.error("snowball requires --cache")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    if args.mode == "snowball":
        snowball(args)
    else:
        args.root = args.root or Path.home() / "memcam_results/context_memory_60s"
        args.manifest = args.manifest or REPO / "testbeds/context_memory/manifest.jsonl"
        rows = parse_int_ranges(args.rows)
        items = {i["_row"]: remap_gt_dir(i, args.dataset_root) for i in load_manifest(args.manifest, args.duration)
                 if rows is None or i["_row"] in rows}
        if not items:
            parser.error("No matching trajectories")
        retrieval(args, items)


if __name__ == "__main__":
    main()
