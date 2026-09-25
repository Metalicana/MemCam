"""Fixed random-five FVD comparison and matched PSNR reporting. No generation."""

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from compare_fvd_matched import CONFIG, check_device, clip_indices, cohort_items, digest, save_json, verify_video
from evaluate_context_memory import FVDRunner, evaluate_video, output_path, resolve_gt_frames_dir


METHODS = {
    "Unbounded": "baseline", "FIFO": "fifo_b32", "MCE": "mce_b32_lambda1_pilot",
    "K-center": "kcenter_b32", "RI": "ri_b32_dino_rgb", "KEEPSAKE": "slam_b32_covisibility",
}
STRIDE = 30


def select_subset(items):
    ordered = sorted(items, key=lambda item: item["output_prefix"])
    if len(ordered) != 15 or len({i["output_prefix"] for i in ordered}) != 15:
        raise ValueError("Expected fifteen unique source videos")
    # Fix this draw before loading any scores; neither quality nor file availability enters it.
    return sorted(random.Random(0).sample(ordered, 5), key=lambda item: item["output_prefix"])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matching_psnr(record, item, run):
    expected_samples = len(range(0, int(item["num_frames"]), STRIDE))
    expected = dict(run_name=run, scene=item["scene"], start_frame=item["start_frame"],
                    duration_sec=60, num_frames_expected=item["num_frames"],
                    frames_seen=item["num_frames"], frames_evaluated=expected_samples,
                    frame_stride=STRIDE, status="completed")
    value = record.get("psnr_db")
    return (all(record.get(key) == expected_value for key, expected_value in expected.items())
            and Path(record.get("output", "")).name == f"{item['output_prefix']}custom.mp4"
            and isinstance(value, (int, float)) and math.isfinite(value))


def saved_psnr(scores, items):
    """Read per-video PSNR, never the PSNR/FVD of an unmatched aggregate."""
    if not scores.is_file():
        return {}, {}
    with scores.open(newline="") as handle:
        sources = {(r["run"], str(Path(r["source"]).parent / "metrics.jsonl"))
                   for r in csv.DictReader(handle)
                   if r["run"] in METHODS.values() and r["evaluator"] == "quality"}
    results, receipts = {}, {}
    for run, source in sorted(sources):
        path = Path(source)
        if not path.is_file():
            continue
        records = read_jsonl(path)
        for item in items:
            matches = [r for r in records if matching_psnr(r, item, run)]
            if len(matches) > 1:
                raise ValueError(f"Duplicate PSNR records for {run}/{item['output_prefix']}: {path}")
            if not matches:
                continue
            key = (run, item["output_prefix"])
            value = float(matches[0]["psnr_db"])
            if key in results and not math.isclose(results[key]["psnr_db"], value, abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Conflicting saved PSNR values for {key}")
            results[key] = dict(matches[0], source=str(path), source_kind="recorded_identity_only")
            receipts[str(path)] = digest(path)
    return results, receipts


def write_table(path, rows, title, include_fvd):
    columns = "lrr" if include_fvd else "lr"
    heading = r"Method & FVD $\downarrow$ & PSNR (dB) $\uparrow$ \\" if include_fvd else r"Method & PSNR (dB) $\uparrow$ \\"
    best_psnr = max(r["psnr_db"] for r in rows)
    best_fvd = min(r["fvd"] for r in rows) if include_fvd else None
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{title}}}", r"\small",
             f"\\begin{{tabular}}{{{columns}}}", r"\toprule", heading, r"\midrule"]
    for row in rows:
        psnr = f"{row['psnr_db']:.2f}"
        if row["psnr_db"] == best_psnr:
            psnr = f"\\textbf{{{psnr}}}"
        cells = [row["method"]]
        if include_fvd:
            fvd = f"{row['fvd']:.1f}"
            cells.append(f"\\textbf{{{fvd}}}" if row["fvd"] == best_fvd else fvd)
        lines.append(" & ".join([*cells, psnr]) + r" \\")
    path.write_text("\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def psnr_summary(results, items):
    return [dict(method=label, run=run, duration_sec=60, videos=len(items),
                 psnr_db=float(np.mean([results[run, item["output_prefix"]]["psnr_db"] for item in items])))
            for label, run in METHODS.items()]


def get_psnr(args, items):
    results, sources = saved_psnr(args.scores, items)
    missing = [(label, run, item) for label, run in METHODS.items() for item in items
               if (run, item["output_prefix"]) not in results]
    if args.saved_only and missing:
        raise ValueError("Missing compatible saved PSNR (no recomputation requested):\n" +
                         "\n".join(f"{label}: {item['output_prefix']}" for label, _, item in missing))
    for label, run, item in missing:
        video = output_path(args.root / run, item)
        print(f"Computing PSNR: {label}, {item['scene']}", flush=True)
        verify_video(video, item)
        result = evaluate_video(item, args.root / run, args.dataset_root, STRIDE,
                                None, None, None, ["psnr_db"])
        result["run_name"] = run
        if not matching_psnr(result, item, run):
            raise ValueError(f"Incomplete or invalid PSNR: {label}, {item['output_prefix']}")
        results[run, item["output_prefix"]] = dict(result, source=str(video), source_kind="recomputed")
    records = [dict(method=label, run=run, output_prefix=item["output_prefix"],
                    psnr_db=results[run, item["output_prefix"]]["psnr_db"],
                    source=results[run, item["output_prefix"]]["source"],
                    source_kind=results[run, item["output_prefix"]]["source_kind"])
               for label, run in METHODS.items() for item in items]
    write_csv(args.output / "psnr_per_video.csv", records)
    save_json(args.output / "psnr_provenance.json", dict(
        source_artifact_sha256=sources, frame_stride=STRIDE,
        definition="Mean per-frame RGB PSNR, peak 255; then equal-video mean. "
                   "GT index=start_frame+generated index; GT bicubic-resized only if sizes differ. "
                   "Sample indices 0,30,60,... include the initial frame; production evaluator caps exact matches at 100 dB.",
        limits="Saved scores are checked by recorded identity/length/sampling, not historical video hashes."))
    return results


def get_fvd(args, items):
    # Reject incomplete inputs before importing Torch; never shrink the selected cohort.
    for item in items:
        for run in METHODS.values():
            verify_video(output_path(args.root / run, item), item)
        gt = resolve_gt_frames_dir(item, args.dataset_root)
        for i in {i for clip in clip_indices(item) for i in clip}:
            if not (gt / f"{int(item['start_frame']) + i:04d}.png").is_file():
                raise FileNotFoundError(f"Missing GT: {gt}, relative frame {i}")
    import torch
    check_device(torch, "cuda", args.output)
    runner = FVDRunner(device="cuda", batch_size=4, cache_dir=args.fvd_cache_dir, **CONFIG)
    if str(runner.device).split(":")[0] != "cuda":
        raise RuntimeError("FVD switched away from the requested GPU")
    save_json(args.output / "fvd_provenance.json", dict(
        config=CONFIG, detector_sha256=digest(runner.resolved_detector_path),
        evaluator_sha256=digest(Path(__file__).with_name("evaluate_context_memory.py")),
        scope="One fixed random-five subset; FVD recomputed from pooled clips, not averaged per-video FVD."))
    values = {}
    for label, run in METHODS.items():
        print(f"Computing FVD: {label}, {len(items)} matched videos", flush=True)
        value, clips = runner.compute_group(items, args.root / run, args.dataset_root, None)
        if clips != 4 * len(items) or value is None or not math.isfinite(value):
            raise ValueError(f"Incomplete FVD for {label}: {clips} clips, score={value}")
        values[run] = float(value)
        save_json(args.output / "fvd_progress.json", values)
        print(f"{label}: FVD={value:.3f}, clips={clips}", flush=True)
    return values


def execute(args):
    items = cohort_items(args.manifest, 15)
    subset = select_subset(items)
    plan = dict(source_manifest=str(args.manifest.resolve()), source_manifest_sha256=digest(args.manifest),
                selection="random.Random(0).sample(sorted output_prefix identities, 5), without replacement",
                seed=0, pool_videos=15, subset=subset, methods=METHODS, duration_sec=60, bounded_budget=32)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "cohort.json"
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError("Output contains a different cohort; refusing to replace its selection")
    save_json(existing, plan)
    print("Fixed random-five subset (seed 0, from the original 15):", flush=True)
    for item in subset:
        print(f"  {item['scene']} start={item['start_frame']}", flush=True)
    if args.audit_only:
        return
    try:
        run_metrics(args, items, subset)
    except Exception as exc:
        save_json(args.output / "status.json", dict(status="failed", error=str(exc)))
        raise


def run_metrics(args, items, subset):
    save_json(args.output / "status.json", dict(status="running", psnr_only=args.psnr_only))
    psnr_items = items if args.psnr_cohort == "all15" else subset
    results = get_psnr(args, psnr_items)
    rows = psnr_summary(results, psnr_items)
    stem = "psnr_all15" if args.psnr_cohort == "all15" else "psnr_random5"
    write_csv(args.output / f"{stem}.csv", rows)
    write_table(args.output / f"{stem}.tex", rows,
                f"MemCam PSNR on {len(psnr_items)} matched 60-second rollouts. "
                + ("Random-five subset, seed 0. " if len(psnr_items) == 5 else "")
                + "Bounded methods use $B=32$; RGB PSNR is sampled every 30 frames.", False)
    for row in rows:
        print(f"{row['method']:12s} PSNR={row['psnr_db']:.4f} dB  N={row['videos']}", flush=True)
    if not args.psnr_only:
        values = get_fvd(args, subset)
        rows = [dict(row, fvd=values[row["run"]], fvd_clips=20) for row in psnr_summary(results, subset)]
        write_csv(args.output / "scores.csv", rows)
        write_table(args.output / "table.tex", rows,
                    "MemCam on five randomly selected matched 60-second rollouts (seed 0). "
                    "All bounded methods use $B=32$. Exploratory subset; FVD uses 20 clips per method.", True)
    save_json(args.output / "status.json", dict(status="psnr_complete" if args.psnr_only else "complete"))
    print(f"Results: {args.output}", flush=True)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--scores", type=Path, default=repo / "paper/results/metric_results_60s/scores.csv")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "memcam_results/random5_quality_60s_seed0")
    parser.add_argument("--psnr-only", action="store_true", help="No GPU or FVD required")
    parser.add_argument("--saved-only", action="store_true", help="Only read saved PSNR; no video decoding")
    parser.add_argument("--psnr-cohort", choices=("subset", "all15"), default="subset")
    parser.add_argument("--audit-only", action="store_true", help="Freeze and print the cohort without scoring")
    parser.add_argument("--fvd-cache-dir", type=Path, default=Path.home() / "hf_cache/memcam_fvd")
    args = parser.parse_args()
    if args.saved_only and not args.psnr_only:
        parser.error("--saved-only requires --psnr-only")
    execute(args)


if __name__ == "__main__":
    main()
