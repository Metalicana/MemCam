"""Independent, resumable evidence jobs using existing videos only."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from paper.evidence_statistics import RUNS, KEEP, write_csv, export_scalars, fvd_statistics, holm
from paper.finish_keepsake_pose_appearance import cohort
from paper.benchmark_b32_query_latency import RUN_SPECS, final_bank_size
from paper.audit_gap_inputs import audit_trace
from paper.replay_keepsake_updates import run_replay
import compare_fvd_matched as fvd
from run_budget_metric_grid import digest, save
from evaluate_context_memory import FVDRunner, resolve_gt_frames_dir, output_path

TIMING_RUNS = (*RUNS, "slam_b16_covisibility", "slam_b64_covisibility", "slam_b128_covisibility")


def sources_unchanged(sources):
    for path, expected in sources.items():
        if isinstance(expected, str) and len(expected) == 64:
            if digest(Path(path)) != expected:
                raise ValueError(f"Source changed: {path}")


def stage(output, name, function):
    """A failed stage cannot stop independent evidence tasks or look complete."""
    directory = output / name
    directory.mkdir(exist_ok=True)
    receipt = directory / "receipt.json"
    try:
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record.get("status") == "complete":
                sources_unchanged(record["artifacts"])
                sources_unchanged(record["provenance"].get("sources", {}))
                print(f"REUSE {name}", flush=True)
                return dict(stage=name, status="complete", reused=True)
        print(f"START {name}", flush=True)
        metadata = function(directory)
        artifacts = {str(p): digest(p) for p in sorted(directory.rglob("*"))
                     if p.is_file() and p != receipt and p.name != "error.txt"}
        save(receipt, dict(status="complete", provenance=metadata, artifacts=artifacts))
        return dict(stage=name, status="complete", reused=False)
    except Exception as exc:
        error = traceback.format_exc()
        (directory / "error.txt").write_text(error)
        print(f"FAILED {name}: {exc}", flush=True)
        # Preserve a previous complete receipt when its validation fails.
        return dict(stage=name, status="failed", error=str(exc))


def latency(args, directory):
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=directory))
    command = [sys.executable, "-u", str(ROOT / "paper/benchmark_b32_query_latency.py"),
        "--manifest", str(args.manifest), "--root", str(args.root / "context_memory_60s"),
        "--duration", "60", "--first-video", "--runs", *TIMING_RUNS,
        "--sample-sections", "8", "--queries-per-section", "1", "--repeats", "3", "--threads", "1",
        "--output", str(attempt)]
    with (attempt / "launcher.log").open("w") as log:
        # Benchmark requires an empty output folder; put its outputs in a child.
        command[-1] = str(attempt / "results")
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    results = attempt / "results"
    for filename in ("latency_summary.csv", "archive_counts.csv", "trajectory_latency.csv"):
        (directory / filename).write_bytes((results / filename).read_bytes())
    record = json.loads((results / "provenance.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Incomplete timing run")
    return dict(benchmark=record, sources={r["path"]: r["sha256"] for r in record["sources"]})


def archive_profiles(args, items, directory):
    counts, profiles, coverage, sources = [], [], [], {}
    for run in TIMING_RUNS:
        label, _, policy, budget = RUN_SPECS[run]
        for item in items:
            path = args.root / "context_memory_60s" / run / "access_traces" / (item["output_prefix"] + "custom.jsonl")
            audit = audit_trace(path, item, policy, budget)
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            count = final_bank_size(events, int(item["num_frames"]), budget)
            counts.append(dict(run=run, policy=label, scene=item["scene"], duration_sec=60,
                               num_frames=item["num_frames"], final_stored_frames=count))
            sources[str(path)] = audit["sha256"]
            profile = path.parent.parent / "profiles" / path.name
            status = dict(run=run, scene=item["scene"], path=str(profile))
            try:
                records = [json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
                records = [r for r in records if r.get("event") == "section_profile"]
                if sorted(int(r["section_idx"]) for r in records) != list(range((int(item["num_frames"])-1)//76)):
                    raise ValueError("Incomplete/duplicate section profiles")
                for r in records:
                    if (r["scene"] != item["scene"] or int(r["dataset_start_frame"]) != int(item["start_frame"])
                            or int(r["duration_sec"]) != 60 or r["memory_policy"] != policy
                            or (budget and int(r["memory_budget"]) != budget)):
                        raise ValueError("Profile identity/policy mismatch")
                final = max(records, key=lambda r: int(r["section_idx"]))
                if int(final["stored_memory_size"]) != count:
                    raise ValueError("Profile and trace archive counts disagree")
                keys = ("cumulative_rollout_latency_s", "bank_frame_bytes", "bank_feature_bytes", "peak_cuda_allocated_gb", "peak_rss_gb")
                for key in keys:
                    if key == "peak_rss_gb" and final.get(key) is None:
                        continue
                    if not math.isfinite(float(final[key])) or float(final[key]) < 0:
                        raise ValueError(f"Invalid profile value: {key}")
                selection_times = []
                for r in records:
                    phases = r["phase_latency_s"]
                    # Section zero conditions on the input image; no memory query occurs.
                    seconds = float(phases.get("context_selection", 0) if int(r["section_idx"]) == 0
                                    else phases["context_selection"])
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError("Invalid context-selection timing")
                    selection_times.append(seconds)
                profiles.append(dict(run=run, scene=item["scene"], duration_sec=60,
                                     **{key: final.get(key) for key in keys},
                                     context_selection_s=sum(selection_times)))
                status.update(status="valid", error="")
                sources[str(profile)] = digest(profile)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status.update(status="missing" if not profile.exists() else "invalid", error=str(exc))
            coverage.append(status)
    write_csv(directory / "archive_counts.csv", counts)
    write_csv(directory / "profile_coverage.csv", coverage)
    if profiles:
        write_csv(directory / "historical_profiles.csv", profiles)
    summary = []
    for run in TIMING_RUNS:
        subset = [r for r in counts if r["run"] == run]
        summary.append(dict(run=run, videos=len(subset), duration_sec=60,
                            final_stored_min=min(r["final_stored_frames"] for r in subset),
                            final_stored_max=max(r["final_stored_frames"] for r in subset),
                            valid_historical_profiles=sum(r["run"] == run and r["status"] == "valid" for r in coverage)))
    write_csv(directory / "archive_summary.csv", summary)
    return dict(sources=sources, missing_or_invalid_profiles=[r for r in coverage if r["status"] != "valid"],
        scope="Counts audited across all 15. Profiles are historical telemetry, not matched-hardware speed or VRAM comparisons.",
        warning="A complete archive audit does not mean all historical profiles exist; inspect profile_coverage.csv.")


def feature_grid(args, items, runs, duration, directory):
    if len({i["scene"] for i in items}) != len(items):
        raise ValueError("FVD inference requires one rollout per independent scene")
    root = args.root / ("context_memory_60s" if duration == 60 else "context_180s")
    # Audit every required video and GT clip before loading a detector.
    for item in items:
        gt = resolve_gt_frames_dir(item, None)
        required = [output_path(root / run, item) for run in runs]
        required += [gt / f"{int(item['start_frame']) + i:04d}.png"
                     for i in {i for clip in fvd.clip_indices(item) for i in clip}]
        missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
        if missing:
            raise FileNotFoundError("Missing FVD inputs: " + ", ".join(missing[:8]))
    import torch
    fvd.check_device(torch, "cuda", directory)
    runner = FVDRunner(device="cuda", batch_size=4, image_size=224, clip_length=16,
                       clips_per_video=4, frame_stride=4, cache_dir=args.fvd_cache, allow_download=False)
    if runner.device.type != "cuda":
        raise RuntimeError("No silent CPU fallback")
    encoder = dict(path=str(runner.resolved_detector_path), sha256=digest(runner.resolved_detector_path),
                   torch=str(torch.__version__), config=fvd.CONFIG)
    folder = directory / "features"
    folder.mkdir(exist_ok=True)
    arrays = {key: [] for key in ("GT", *runs)}
    sources = {str(runner.resolved_detector_path): encoder["sha256"]}
    for index, item in enumerate(items):
        print(f"FVD {duration}s {index+1}/{len(items)}: {item['scene']}", flush=True)
        gt = resolve_gt_frames_dir(item, None)
        gt_sources = {str(gt / f"{int(item['start_frame'])+i:04d}.png"): digest(gt / f"{int(item['start_frame'])+i:04d}.png")
                      for i in {i for clip in fvd.clip_indices(item) for i in clip}}
        sources.update(gt_sources)
        gt_features = None
        for run in runs:
            video = output_path(root / run, item)
            sources[str(video)] = digest(video)
            inputs = dict(item=item, encoder=encoder, video_sha256=sources[str(video)], gt=gt_sources)
            path = folder / f"row{index:03d}_{run}.npz"
            receipt = path.with_suffix(".json")
            if path.exists() and receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["inputs"] != inputs or saved["sha256"] != digest(path):
                    raise ValueError(f"Stale FVD cache: {path}")
                with np.load(path, allow_pickle=False) as data:
                    generated, reference = data["generated"], data["GT"]
            else:
                fvd.verify_video(video, item)
                clips, gt_clips = runner._load_item_clips(item, root / run, None, None)
                if len(clips) != 4 or len(gt_clips) != 4:
                    raise ValueError("Expected four production-sampled clips")
                batches = []
                runner._append_features(batches, clips)
                generated = np.concatenate(batches)
                if gt_features is None:
                    batches = []
                    runner._append_features(batches, gt_clips)
                    reference = np.concatenate(batches)
                else:
                    reference = gt_features
                temporary = path.with_suffix(".tmp.npz")
                np.savez(temporary, generated=generated, GT=reference)
                temporary.replace(path)
                save(receipt, dict(inputs=inputs, sha256=digest(path)))
            if (generated.shape != reference.shape or generated.ndim != 2 or generated.shape[0] != 4
                    or not np.isfinite(generated).all() or not np.isfinite(reference).all()):
                raise ValueError("Invalid FVD feature array")
            if gt_features is not None and not np.array_equal(gt_features, reference):
                raise ValueError("GT features differ across policies")
            gt_features = reference
            arrays[run].append(generated)
        arrays["GT"].append(gt_features)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    sources_unchanged(sources)
    np.savez(directory / "cohort_features.npz", **arrays)
    save(directory / "cohort.json", dict(items=items, runs=runs, duration_sec=duration, encoder=encoder))
    return dict(sources=sources, encoder=encoder, duration_sec=duration, runs=runs,
                scope="Existing videos only; four production I3D clips per trajectory, exact matched cohort.")


def fvd_scores(args, directory, durations=(60, 180)):
    summaries, contrasts, sources = [], [], {}
    for duration in durations:
        work = args.output / f"fvd_features_{duration}s"
        record = json.loads((work / "receipt.json").read_text())
        if record["status"] != "complete":
            raise ValueError("FVD feature stage incomplete")
        sources_unchanged(record["artifacts"])
        path = work / "cohort_features.npz"
        config = json.loads((work / "cohort.json").read_text())
        with np.load(path, allow_pickle=False) as data:
            features = {k: data[k] for k in data.files}
        # Confirm low-rank algebra against the production covariance implementation.
        reference = object.__new__(FVDRunner)
        for run in config["runs"]:
            gt, gen = (features[k].reshape(-1, features[k].shape[-1]) for k in ("GT", run))
            original = reference._frechet_distance(gt, gen)
            if not np.isclose(original, fvd.frechet_low_rank(gt, gen), atol=1e-3, rtol=1e-6):
                raise ValueError("Fast FVD disagrees with production evaluator")
        print(f"Scoring FVD {duration}s: {args.fvd_draws} paired draws", flush=True)
        a, b, boot = fvd_statistics(features, draws=args.fvd_draws, runs=config["runs"])
        summaries.extend(dict(duration_sec=duration, **r) for r in a)
        contrasts.extend(dict(duration_sec=duration, **r) for r in b)
        np.save(directory / f"bootstrap_{duration}s.npy", boot)
        sources[str(path)] = digest(path)
        sources[str(work / "cohort.json")] = digest(work / "cohort.json")
    for row, p in zip(contrasts, holm([r["p_two_sided"] for r in contrasts])):
        row["p_holm_both_horizons"] = p
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "contrasts.csv", contrasts)
    if 60 in durations:
        with args.scores.open() as handle:
            reported = [r for r in csv.DictReader(handle)
                        if r["evaluator"] == "quality" and r["metric"] == "FVD" and r["run"] in RUNS]
        if len(reported) != len(RUNS) or len({r["run"] for r in reported}) != len(RUNS):
            raise ValueError("Missing/duplicate FVD reference rows")
        checks = []
        for r in reported:
            fresh = next(s["fvd"] for s in summaries if s["run"] == r["run"] and s["duration_sec"] == 60)
            old = float(r["value"])
            if int(r["n"]) != 15 or not math.isfinite(old):
                raise ValueError("Invalid FVD reference cohort/value")
            checks.append(dict(run=r["run"], reported_fvd=old, recomputed_fvd=fresh,
                               difference=fresh-old, within_numerical_tolerance=bool(np.isclose(fresh, old, atol=1e-3, rtol=1e-6))))
        write_csv(directory / "reported_score_check.csv", checks)
        sources[str(args.scores)] = digest(args.scores)
    return dict(sources=sources, draws=args.fvd_draws, seed=17,
        interval="Paired whole-scene percentile bootstrap; FVD recomputed for every resampled cohort, all clips stay together.",
        test="Two-sided paired policy-label swaps by whole scene, Monte Carlo plus-one correction; exchangeability assumption.",
        limitations="One observed rollout per scene, not seed variance or proof of equivalence; finite-sample FVD bias remains. "
                    "Intervals belong to the fresh feature-derived point estimates; inspect reported_score_check.csv before replacing historical table values.")


def joint_tests(output, directory):
    rows, sources = [], {}
    for name in ("scalar_statistics", "fvd_statistics_60s", "fvd_statistics_180s"):
        record = json.loads((output / name / "receipt.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Incomplete {name}")
        sources_unchanged(record["artifacts"])
        path = output / name / "contrasts.csv"
        sources[str(path)] = digest(path)
        with path.open() as handle:
            for r in csv.DictReader(handle):
                rows.append(dict(metric=r.get("metric", "FVD"), duration_sec=r.get("duration_sec", 60),
                                 reference=r["reference"], compared=r["compared"], p_two_sided=float(r["p_two_sided"])))
    for row, p in zip(rows, holm([r["p_two_sided"] for r in rows])):
        row["p_holm_all_evidence"] = p
    write_csv(directory / "joint_multiplicity.csv", rows)
    return dict(sources=sources, family="All scalar and FVD contrasts across both horizons, Holm FWER")


def verify_plan(args):
    plan = json.loads((args.output / "plan.json").read_text())
    for key in ("root", "manifest", "manifest_180", "scores", "cache", "fvd_cache"):
        if str(getattr(args, key).resolve()) != plan[key]:
            raise ValueError(f"Plan changed: {key}")
    if args.fvd_draws != plan["fvd_draws"]:
        raise ValueError("FVD draw count changed")
    sources_unchanged(plan["code_hashes"])
    sources_unchanged(plan["input_hashes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "gpu", "fvd-score"), required=True)
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--manifest-180", type=Path, default=ROOT / "testbeds/context_memory_180s/manifest.jsonl")
    parser.add_argument("--scores", type=Path, default=Path.home() / "memcam_results/budget_metrics_60s_820776/scores.csv")
    parser.add_argument("--cache", type=Path, default=Path.home() / "memcam_results/context_memory_60s/gap_feature_cache_fresh")
    parser.add_argument("--fvd-cache", type=Path, default=Path.home() / "hf_cache/memcam_fvd")
    parser.add_argument("--fvd-draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run through the submitted batch jobs, not on a login node")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f".{args.phase}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = args.output / f"status_{args.phase}.json"
        state = dict(status="running", job=os.environ.get("SLURM_JOB_ID"), phase=args.phase, stages=[],
                     python=sys.version, numpy=np.__version__, host=platform.node(), platform=platform.platform(),
                     environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_JOB_NODELIST")})
        save(status_path, state)
        try:
            verify_plan(args)
            if args.phase == "cpu":
                items = cohort(args.manifest, 60)
                tasks = [("archive_audit", lambda p: archive_profiles(args, items, p)),
                         ("scalar_statistics", lambda p: export_scalars(args.scores, items, p)),
                         ("latency_60s", lambda p: latency(args, p)),
                         ("update_replay", lambda p: run_replay(items, args.root / "context_memory_60s", args.cache, p))]
            elif args.phase == "gpu":
                tasks = [(f"fvd_features_{d}s", lambda p, d=d, m=m, runs=runs:
                          feature_grid(args, cohort(m, d), runs, d, p))
                         for d, m, runs in ((60, args.manifest, RUNS), (180, args.manifest_180, ("baseline", "fifo_b32", KEEP)))]
            else:
                tasks = [(f"fvd_statistics_{d}s", lambda p, d=d: fvd_scores(args, p, (d,))) for d in (60, 180)]
                tasks.append(("joint_tests", lambda p: joint_tests(args.output, p)))
            for name, function in tasks:
                state["stages"].append(stage(args.output, name, function))
                save(status_path, state)
            state["status"] = "complete" if all(s["status"] == "complete" for s in state["stages"]) else "incomplete"
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            save(status_path, state)
        print(json.dumps(state, indent=2), flush=True)
        if state["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
