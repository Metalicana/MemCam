"""Six-policy, matched-60s CPU retrieval replay; no generation or feature caches."""

import argparse
from collections import defaultdict
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
from paper.audit_gap_inputs import audit_trace
from paper.benchmark_query_latency import digest, load_file_module, retrieve, write_csv
from utils.analyze_retrieval_quality_decomposition import load_manifest, read_trace, reconstruct_candidate_banks, run_identity


RUNS = (
    ("Unbounded", "baseline", "unbounded", None),
    ("FIFO", "fifo_b32", "fifo", 32),
    ("MCE", "mce_b32_lambda1_pilot", "mce", 32),
    ("K-center", "kcenter_b32", "kcenter_coreset", 32),
    ("RI", "ri_b32_dino_rgb", "rarity_irreplaceability", 32),
    ("KEEPSAKE", "slam_b32_covisibility", "slam_covisibility", 32),
)


def final_bank_size(events, count, budget):
    evictions = defaultdict(list)
    sections = (count - 1) // 76
    for event in events:
        if event.get("event") == "memory_eviction":
            section = int(event["section_idx"])
            if not 0 <= section < sections:
                raise ValueError("Eviction outside rollout")
            evictions[section].append(int(event["evicted_memory_frame"]))
    bank = {0}
    for section in range(sections):
        bank.update(range(section * 76, section * 76 + 77))
        for index in evictions[section]:
            if index not in bank:
                raise ValueError("Eviction of an absent frame")
            bank.remove(index)
        expected = min(budget, section * 76 + 77) if budget else section * 76 + 77
        if len(bank) != expected:
            raise ValueError(f"Post-update bank has {len(bank)} frames; expected {expected}")
    return len(bank)


def prepare(args):
    items = load_manifest(args.manifest, duration=60)
    if len(items) != args.expected_videos or len({i["output_prefix"] for i in items}) != len(items):
        raise ValueError("Expected complete distinct matched manifest cohort")
    pose_module = load_file_module("latency_grid_poses", ROOT / "dataset/poses.py")
    cases, pose_sets, sources, archives = [], {}, [], []
    for item in items:
        row_id, count = int(item["_row"]), int(item["num_frames"])
        if float(item["fps"]) != 30:
            raise ValueError("Expected 30 FPS MemCam manifest")
        pose_path = (args.dataset_root / "jsons" / f"{item['scene']}.json"
                     if args.dataset_root else Path(item["pose_path"]))
        keys = sorted(map(int, json.loads(pose_path.read_text())["CineCameraActor"]))
        start = int(item["start_frame"])
        if keys[start:start + count] != list(range(start, start + count)):
            raise ValueError(f"Pose index coverage mismatch: {pose_path}")
        poses = pose_module.load_c2ws_from_json(pose_path, start_frame=start, num_frames=count)
        import numpy as np
        if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(f"Invalid poses: {pose_path}")
        pose_sets[row_id] = poses
        sources.append({"path": str(pose_path), "sha256": digest(pose_path)})
        sections = (count - 1) // 76
        banks = {}
        for label, run, policy, budget in RUNS:
            path = args.root / run / "access_traces" / f"{item['output_prefix']}custom.jsonl"
            audit = audit_trace(path, item, policy, budget)
            events = read_trace(path, expected_identity=run_identity(item))
            if digest(path) != audit["sha256"]:
                raise ValueError(f"Trace changed during audit: {path}")
            banks[run] = reconstruct_candidate_banks(events, sections - 1, num_frames=count)
            archives.append(dict(row=row_id, scene=item["scene"], run=run, policy=label,
                                 final_stored_frames=final_bank_size(events, count, budget)))
            sources.append({"path": str(path), "sha256": audit["sha256"]})
        # Every retrieved section contributes the same number of target slots.
        # All 76 actual reads per section were validated above, across all policies.
        for section in range(1, sections):
            for k in range(args.queries_per_section):
                slot = (2 * k + 1) * 76 // (2 * args.queries_per_section)
                cases.append(dict(row=row_id, scene=item["scene"], dataset_start_frame=start,
                                  section_idx=section, target_frame=section * 76 + slot + 1,
                                  candidates={run: bank[section] for run, bank in banks.items()}))
        print(f"Validated {item['scene']}: six traces, {sections - 1} retrieved sections", flush=True)
    return items, cases, pose_sets, sources, archives


def summarize(records, cases, archives, repeats):
    expected = {(c["row"], c["section_idx"], c["target_frame"], run, r)
                for c in cases for _, run, _, _ in RUNS for r in range(repeats)}
    by_query, seen = defaultdict(list), set()
    for record in records:
        key = tuple(record[k] for k in ("row", "section_idx", "target_frame", "run", "repeat"))
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate timing record")
        if not math.isfinite(record["query_ms"]) or record["query_ms"] <= 0:
            raise ValueError("Invalid elapsed time")
        seen.add(key)
        by_query[key[:-1]].append(record["query_ms"])
    if seen != expected:
        raise ValueError("Incomplete timings; no final summary can be produced")
    by_trajectory = defaultdict(list)
    for (row, _, _, run), values in by_query.items():
        by_trajectory[(row, run)].append(mean(values))
    trajectories = [dict(row=row, run=run, queries=len(values), query_ms=mean(values))
                    for (row, run), values in sorted(by_trajectory.items())]
    summary = []
    for label, run, _, budget in RUNS:
        values = [r for r in trajectories if r["run"] == run]
        counts = [r["final_stored_frames"] for r in archives if r["run"] == run]
        if len(counts) != len(values) or len(set(counts)) != 1:
            raise ValueError("Incomplete or inconsistent final archive counts")
        summary.append(dict(system="MemCam", policy=label, run=run, budget=budget or "unbounded",
                            duration_sec=60, videos=len(values), final_stored_frames=counts[0],
                            sampled_queries=sum(r["queries"] for r in values), repeats=repeats,
                            query_ms=mean(r["query_ms"] for r in values)))
    return summary, trajectories


def measure(cases, poses, repeats, seed, overlap, torch, output):
    rng = random.Random(seed)
    order = list(cases)
    rng.shuffle(order)
    runs = [r[1] for r in RUNS]
    records = []
    with (output / "query_timings.jsonl").open("w") as raw:
        for index, case in enumerate(order):
            policies = list(runs)
            rng.shuffle(policies)
            for run in policies:
                torch.manual_seed(seed + index)
                retrieve(poses[case["row"]], case["target_frame"], case["candidates"][run], overlap)
            for repeat in range(repeats):
                # Rotate the shuffled policy order to reduce ordering bias.
                shift = repeat % len(policies)
                for run in policies[shift:] + policies[:shift]:
                    torch.manual_seed(seed + index * repeats + repeat)
                    before = time.perf_counter_ns()
                    winner, score = retrieve(poses[case["row"]], case["target_frame"], case["candidates"][run], overlap)
                    elapsed = (time.perf_counter_ns() - before) / 1e6
                    if winner not in case["candidates"][run] or not math.isfinite(float(score)):
                        raise ValueError("Invalid retrieval result")
                    record = {k: v for k, v in case.items() if k != "candidates"}
                    record.update(run=run, repeat=repeat, query_ms=elapsed, candidate_count=len(case["candidates"][run]),
                                  winner=int(winner), overlap=float(score))
                    records.append(record)
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
            print(f"[{index + 1}/{len(order)}] row {case['row']}, target {case['target_frame']}: six policies timed", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--queries-per-section", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.queries_per_section <= 76 or min(args.repeats, args.threads, args.expected_videos) < 1:
        parser.error("Invalid sampling, repeat, thread or cohort count")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(status="validating", system="MemCam", duration_sec=60, videos=args.expected_videos,
                      parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    path = args.output / "provenance.json"
    try:
        path.write_text(json.dumps(provenance, indent=2) + "\n")
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        items, cases, poses, sources, archives = prepare(args)
        scorer = ROOT / "diffsynth/models/wan_video_overlap.py"
        provenance.update(
            status="validated" if args.check_only else "running", benchmark="isolated_cpu_retrieval_replay_b32",
            query_unit="One target-frame query; one selected memory. Not per-chunk latency.",
            scope="Production CPU FOV candidate scoring and argmax. Excludes generation, RGB transfer, encoding, descriptor extraction, bank reconstruction and eviction.",
            sampling="Two equal-count stratum midpoints per 76-query section by default; no quality/timing-based selection. Initial section has no historical read.",
            aggregation="Mean repeats per query, sampled queries per trajectory, then equal trajectory means.",
            limitations="Replay on current hardware/code, not original rollout wall time. Monte Carlo RNG is reset outside timers; historical winners need not recur. Historical code revisions not certified.",
            cohort=items, case_count=len(cases), host=platform.node(), platform=platform.platform(),
            cpu=platform.processor(), cpu_info=Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else None,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            torch=str(torch.__version__), threads=torch.get_num_threads(),
            fov=dict(samples=5000, half_h=45, half_v=30, radius=50),
            sources=sources + [{"path": str(p), "sha256": digest(p)} for p in
                              (args.manifest, scorer, Path(__file__), ROOT / "paper/benchmark_query_latency.py",
                               ROOT / "paper/audit_gap_inputs.py", ROOT / "dataset/poses.py",
                               ROOT / "utils/analyze_retrieval_quality_decomposition.py", ROOT / "diffsynth/pipelines/wan_video_memcam.py")])
        path.write_text(json.dumps(provenance, indent=2) + "\n")
        (args.output / "query_plan.json").write_text(json.dumps(cases) + "\n")
        write_csv(args.output / "archive_counts.csv", archives)
        print(f"{len(cases)} target queries x six policies x {args.repeats} timed repeats; CPU threads={args.threads}", flush=True)
        if args.check_only:
            return
        overlap = load_file_module("b32_production_overlap", scorer).calculate_overlap_from_c2w
        records = measure(cases, poses, args.repeats, args.seed, overlap, torch, args.output)
        summary, trajectories = summarize(records, cases, archives, args.repeats)
        write_csv(args.output / "trajectory_latency.csv", trajectories)
        write_csv(args.output / "latency_summary.csv", summary)
        provenance.update(status="complete", timing_records=len(records), summary_sha256=digest(args.output / "latency_summary.csv"))
        for row in summary:
            print(f"{row['policy']:12s} N={row['videos']} frames={row['final_stored_frames']} query={row['query_ms']:.3f} ms")
        print(f"Completed: {args.output}")
    except Exception as exc:
        provenance.update(status="failed", error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
