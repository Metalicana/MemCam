"""CPU replay of MemCam's production FOV retrieval on real logged banks.

No generation, decoded pixels, DINO extraction, CUDA, or fitted cost model.
Both policies replay identical sampled target queries, on one process/machine.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.analyze_retrieval_quality_decomposition import (
    load_manifest, read_trace, reconstruct_candidate_banks, run_identity,
)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sample_queries(rows, duration, fps, per_window):
    bins = defaultdict(list)
    for row in sorted(rows, key=lambda r: int(r["target_frame"])):
        window = min(3, int(int(row["target_frame"]) / fps / (duration / 4)))
        bins[window].append(row)
    result = []
    for window in range(4):
        candidates = bins[window]
        if not candidates or (per_window and len(candidates) < per_window):
            raise ValueError(f"Too few queries in window {window}")
        n = per_window or len(candidates)
        # Midpoints of equal-count strata, selected without looking at timings/quality.
        result.extend((window, candidates[(2 * k + 1) * len(candidates) // (2 * n)])
                      for k in range(n))
    return result


def trace_banks(path, item, queries, expected_policy, budget):
    events = read_trace(path, expected_identity=run_identity(item))
    selected = {}
    for event in events:
        if event.get("event") != "context_access" or not event.get("selected"):
            continue
        key = (int(event["section_idx"]), int(event["target_frame"]))
        if key in selected:
            raise ValueError(f"Duplicate read in {path}: {key}")
        selected[key] = event
    banks = reconstruct_candidate_banks(events, max(int(q["section_idx"]) for q in queries),
                                        num_frames=int(item["num_frames"]))
    for query in queries:
        key = (int(query["section_idx"]), int(query["target_frame"]))
        if key not in selected:
            raise ValueError(f"Missing read in {path}: {key}")
        event, candidates = selected[key], banks[key[0]]
        policy = event.get("memory_policy", event.get("run_memory_policy"))
        recorded_budget = event.get("memory_budget", event.get("run_memory_budget"))
        if policy != expected_policy or (budget and int(recorded_budget or -1) != budget):
            raise ValueError(f"Wrong policy/budget in {path}: {policy}/{recorded_budget}")
        if event.get("selection_source", "retriever") != "retriever" or event.get("fallback_reason"):
            raise ValueError(f"Override/fallback is not ordinary retrieval: {path}, {key}")
        if len(candidates) != int(event.get("candidate_count", -1)):
            raise ValueError(f"Reconstructed/logged candidate counts differ: {path}, {key}")
        if not candidates or int(event["selected_memory_frame"]) not in candidates:
            raise ValueError(f"Selected frame missing from bank: {path}, {key}")
        if budget and (len(candidates) > budget or int(event.get("stored_memory_size", budget)) > budget):
            raise ValueError(f"Budget violation: {path}, {key}")
        if min(candidates) < 0 or max(candidates) >= key[0] * 76 - 3:
            raise ValueError(f"Future or continuation frame in bank: {path}, {key}")
        if expected_policy == "unbounded":
            if candidates != list(range(key[0] * 76 - 3)):
                raise ValueError(f"Incomplete unbounded bank: {path}, {key}")
            if len(candidates) != int(query["candidate_count"]):
                raise ValueError(f"CSV/trace count mismatch: {path}, {key}")
    return banks


def retrieve(poses, target, candidates, overlap):
    # Same exhaustive candidate loop and strict > tie rule as wan_video_memcam.py.
    target_pose = poses[target]
    winner, best = None, -1
    for index in candidates:
        score = overlap(target_pose, poses[index], fov_half_h=45.0, fov_half_v=30.0,
                        num_samples=5000, radius=50.0, return_details=False)
        if score > best:
            winner, best = index, score
    return winner, best


def summarize(records, duration):
    repeats = defaultdict(list)
    for r in records:
        if not math.isfinite(r["query_ms"]) or r["query_ms"] <= 0:
            raise ValueError("Invalid latency")
        repeats[(r["row"], r["window"], r["policy"], r["target_frame"])].append(r["query_ms"])
    trajectories = defaultdict(list)
    for (row, window, policy, _), values in repeats.items():
        trajectories[(window, policy, row)].append(mean(values))
    output = []
    for window in range(4):
        row = {"system": "MemCam", "duration_sec": duration,
               "start_sec": window * duration / 4, "end_sec": (window + 1) * duration / 4}
        for policy in ("unbounded", "ours"):
            values = [mean(v) for (w, p, _), v in trajectories.items() if (w, p) == (window, policy)]
            if not values:
                raise ValueError("Incomplete latency windows")
            row[f"{policy}_query_ms"] = mean(values)
            row[f"{policy}_videos"] = len(values)
        output.append(row)
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    table = load_file_module("lookup_table", ROOT / "paper/build_lookup_work_table.py")
    table.memcam_counts(args.queries, args.baseline_run, args.duration, args.fps, args.expected_videos)
    with args.queries.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["run_name"] == args.baseline_run and int(r["duration_sec"]) == args.duration]
    by_row = defaultdict(list)
    for row in rows:
        by_row[int(row["row"])].append(row)
    manifest = {i["_row"]: i for i in load_manifest(args.manifest, duration=args.duration)}
    poses_module = load_file_module("latency_poses", ROOT / "dataset/poses.py")
    cases, pose_sets, sources = [], {}, []
    for row_id, queries in sorted(by_row.items()):
        item = manifest[row_id]
        if any((q["scene"], int(q["dataset_start_frame"]), int(q["duration_sec"])) != run_identity(item)
               for q in queries):
            raise ValueError(f"CSV/manifest identity mismatch for row {row_id}")
        if float(item.get("fps", args.fps)) != args.fps:
            raise ValueError(f"FPS mismatch for row {row_id}")
        pose_path = (args.dataset_root / "jsons" / f"{item['scene']}.json"
                     if args.dataset_root else Path(item["pose_path"]))
        keys = sorted(map(int, json.loads(pose_path.read_text())["CineCameraActor"]))
        start, count = int(item["start_frame"]), int(item["num_frames"])
        if keys[start:start + count] != list(range(start, start + count)):
            raise ValueError(f"Pose indexing/coverage mismatch: {pose_path}")
        poses = poses_module.load_c2ws_from_json(pose_path, start_frame=start, num_frames=count)
        pose_sets[row_id] = poses
        sources.append({"path": str(pose_path), "sha256": digest(pose_path)})
        banks = {}
        for label, run, policy, budget in (("unbounded", args.baseline_run, "unbounded", None),
                                            ("ours", args.ours_run, "slam_covisibility", 32)):
            path = args.root / run / "access_traces" / f"{item['output_prefix']}custom.jsonl"
            banks[label] = trace_banks(path, item, queries, policy, budget)
            sources.append({"path": str(path), "sha256": digest(path)})
        for window, query in sample_queries(queries, args.duration, args.fps, args.queries_per_window):
            section, target = int(query["section_idx"]), int(query["target_frame"])
            if target >= len(poses):
                raise ValueError("Target outside pose sequence")
            cases.append({"row": row_id, "scene": item["scene"], "dataset_start_frame": start, "window": window,
                          "section_idx": section, "target_frame": target,
                          "candidates": {p: b[section] for p, b in banks.items()}})
        print(f"Validated row {row_id}: {item['scene']}", flush=True)
    return cases, pose_sets, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path.home() / "memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory_180s/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_180s")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--baseline-run", default="baseline")
    parser.add_argument("--ours-run", default="slam_b32_covisibility")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--queries-per-window", type=int, default=6, help="Per trajectory; 0 uses every diagnostic query.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queries_per_window < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("Invalid sampling, repeat or thread count")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cases, poses, sources = prepare(args)
    scorer_path = ROOT / "diffsynth/models/wan_video_overlap.py"
    overlap = load_file_module("production_overlap", scorer_path).calculate_overlap_from_c2w
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"status": "validated" if args.check_only else "running", "system": "MemCam",
                  "duration_sec": args.duration, "videos": args.expected_videos,
                  "benchmark": "isolated_cpu_retrieval_replay", "query_unit": "one target-frame query",
                  "scope": "Production FOV scoring and argmax; no generation, encoding, RGB transfer, bank updates or log I/O inside timer.",
                  "aggregation": "Arithmetic mean over repetitions per query, queries per trajectory/window, then equal-weight trajectories.",
                  "sampling": "Equal-count stratum midpoints within each time quarter; diagnostic CSV queries only.",
                  "limitations": "Not end-to-end generation latency. Production scorer uses CPU. RNG reset outside timer; selected identity need not match original Monte Carlo run. Historical trace code revisions and historical hardware are not certified.",
                  "host": platform.node(), "platform": platform.platform(), "cpu": platform.processor(),
                  "cpu_info": Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else None,
                  "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  "torch": torch.__version__, "threads": torch.get_num_threads(),
                  "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "sources": sources + [{"path": str(p), "sha256": digest(p)} for p in
                                         (args.queries, args.manifest, scorer_path, Path(__file__),
                                          ROOT / "dataset/poses.py", ROOT / "utils/analyze_retrieval_quality_decomposition.py",
                                          ROOT / "diffsynth/pipelines/wan_video_memcam.py")],
                  "fov": {"samples": 5000, "half_h": 45, "half_v": 30, "radius": 50},
                  "case_count": len(cases),
                  "cohort": [dict(row=row_id, scene=next(c["scene"] for c in cases if c["row"] == row_id),
                                  dataset_start_frame=next(c["dataset_start_frame"] for c in cases if c["row"] == row_id))
                             for row_id in sorted(poses)]}
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "query_plan.json").write_text(json.dumps(cases) + "\n")
    print(f"{len(cases)} paired queries; {args.repeats} repeats; CPU threads={args.threads}", flush=True)
    if args.check_only:
        print(f"Validation only; no timing. Output: {args.output}")
        return
    records = []
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    raw_path = args.output / "query_timings.jsonl"
    with raw_path.open("w") as raw:
        for index, case in enumerate(cases):
            c2ws, target = poses[case["row"]], case["target_frame"]
            for policy in ("unbounded", "ours"):
                torch.manual_seed(args.seed + index)
                retrieve(c2ws, target, case["candidates"][policy], overlap)  # Untimed warmup.
            first_policy = rng.randrange(2)
            for repeat in range(args.repeats):
                order = ("unbounded", "ours") if (first_policy + repeat) % 2 == 0 else ("ours", "unbounded")
                for policy in order:
                    torch.manual_seed(args.seed + index * args.repeats + repeat)
                    before = time.perf_counter_ns()
                    winner, score = retrieve(c2ws, target, case["candidates"][policy], overlap)
                    elapsed = (time.perf_counter_ns() - before) / 1e6
                    record = {k: v for k, v in case.items() if k != "candidates"}
                    record.update(policy=policy, repeat=repeat, query_ms=elapsed,
                                  candidate_count=len(case["candidates"][policy]), winner=winner, overlap=score)
                    records.append(record)
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
            print(f"[{index + 1}/{len(cases)}] row {case['row']} frame {target}: "
                  f"unbounded={mean(r['query_ms'] for r in records[-2*args.repeats:] if r['policy']=='unbounded'):.2f} ms; "
                  f"ours={mean(r['query_ms'] for r in records[-2*args.repeats:] if r['policy']=='ours'):.2f} ms", flush=True)
    summary = summarize(records, args.duration)
    write_csv(args.output / "latency_summary.csv", summary)
    provenance.update(status="complete", timing_records=len(records),
                      summary_sha256=digest(args.output / "latency_summary.csv"))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    for row in summary:
        print(f"{row['start_sec']:g}-{row['end_sec']:g}s: Unbounded {row['unbounded_query_ms']:.2f} ms; Ours {row['ours_query_ms']:.2f} ms")
    print(f"Completed: {args.output}")


if __name__ == "__main__":
    main()
