"""One resumable experiment: two generated endpoints, one existing control, metrics and tables."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
from statistics import mean
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "utils"))
from paper.audit_gap_inputs import audit_trace
from paper.run_vbench_180s import probe_video
from audit_metric_coverage import bench_details, valid_bench_score
from metric_environment import metric_command
from run_budget_metric_grid import DIMENSIONS, QUALITY_CONFIG, digest, freeze_environment, save, validate_bench
from smoke_video_metrics import cuda_check_code

SETTINGS = (("appearance_only", 0.0), ("control", 0.65), ("pose_only", 1.0))
LABELS = {"appearance_only": "Appearance only", "control": "KEEPSAKE", "pose_only": "Pose only"}
CLIP_BATCH_SIZE = 16
VBENCH_ADAPTER = REPO / "utils/run_vbench_batched.py"
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0, 1.0),
    "background_consistency": (0.2615, 1.0, 1.0),
    "motion_smoothness": (0.706, 0.9975, 1.0),
    "dynamic_degree": (0.0, 1.0, 0.5),
    "aesthetic_quality": (0.0, 1.0, 1.0),
    "imaging_quality": (0.0, 1.0, 1.0),
}


def load(path):
    return json.loads(Path(path).read_text())


def cohort(manifest, duration):
    items = [dict(json.loads(line), _row=row) for row, line in enumerate(manifest.read_text().splitlines()) if line.strip()]
    items = [item for item in items if int(item["duration_sec"]) == duration]
    if len(items) != 15 or len({i["output_prefix"] for i in items}) != 15:
        raise ValueError("Expected exactly 15 distinct matched trajectories")
    expected_frames = {60: 1825, 180: 5397}[duration]
    if any(int(i["num_frames"]) != expected_frames or float(i["fps"]) != 30 for i in items):
        raise ValueError("Unexpected frame count/FPS in manifest")
    return items


def start_log(handle, command):
    offset = handle.tell()
    header = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "job": os.environ.get("SLURM_JOB_ID"), "step": os.environ.get("SLURM_STEP_ID"),
              "command": list(map(str, command))}
    handle.write(("\n=== ATTEMPT " + json.dumps(header) + " ===\n").encode())
    handle.flush()
    return offset


def current_log_tail(log, offset):
    with log.open("rb") as handle:
        handle.seek(max(offset, log.stat().st_size - 16384))
        return "\n".join(handle.read().decode(errors="replace").splitlines()[-20:])


def run_command(command, log, env=None, cwd=REPO):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {log}: {' '.join(map(str, command))}", flush=True)
    started = time.monotonic()
    with log.open("ab") as handle:
        offset = start_log(handle, command)
        with subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                              stdout=handle, stderr=subprocess.STDOUT) as process:
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"RUNNING {log.name}: {(time.monotonic()-started)/60:.0f} min; log: {log}", flush=True)
            if code:
                tail = current_log_tail(log, offset)
                raise RuntimeError(f"Command failed ({code}); {log}\n{tail}")


def fingerprint(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def generation_assets():
    paths = [REPO / "models" / p for p in (
        "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", "MemCam/dit_step20000.ckpt")]
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing generation checkpoint: {path}")
    return fingerprint(paths)


def cached_step(path, inputs):
    if not path.exists():
        return None
    record = load(path)
    if record.get("status") != "complete" or record.get("inputs") != inputs:
        return None
    for artifact, expected in record.get("artifacts", {}).items():
        if not Path(artifact).is_file() or digest(artifact) != expected:
            raise ValueError(f"Completed artifact changed: {artifact}")
    return record


def validate_quality(directory, items, run, duration):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    expected = {i["output_prefix"] + "custom.mp4": i for i in items}
    names = [Path(r["output"]).name for r in rows]
    if len(rows) != 15 or set(names) != set(expected):
        raise ValueError("Quality evaluation has a different or duplicate cohort")
    for row, name in zip(rows, names):
        item = expected[name]
        if (row["status"] != "completed" or row["run_name"] != run
                or int(row["duration_sec"]) != duration or int(row["row"]) != item["_row"]
                or row["scene"] != item["scene"] or int(row["start_frame"]) != int(item["start_frame"])
                or not math.isfinite(float(row.get("lpips_alex", float("nan"))))):
            raise ValueError(f"Incomplete/mismatched LPIPS record: {name}")
    summary = load(directory / "summary.json")
    config = summary["metric_config"]
    for settings in QUALITY_CONFIG.values():
        for key, value in settings.items():
            if config.get(key) != value:
                raise ValueError(f"Quality configuration mismatch: {key}")
    if (config.get("source_duration") != duration or config.get("eval_durations") != [duration]
            or "max_frames" not in config or config["max_frames"] is not None):
        raise ValueError("Wrong or truncated quality-evaluation horizon")
    group = summary["by_duration"][str(duration)]
    if (group.get("completed_or_short") != 15 or group.get("fvd_clips") != 60
            or not group.get("fvd_detector_path") or not math.isfinite(float(group.get("fvd", float("nan"))))):
        raise ValueError("FVD requires the complete 15-video / 60-clip cohort")
    return {"LPIPS": mean(float(r["lpips_alex"]) for r in rows), "FVD": float(group["fvd"])}


def quality_command(args, manifest, source, attempt, run, items):
    cmd = metric_command("memcam", "python", REPO / "utils/evaluate_context_memory_prefix_curves.py",
        "--manifest", manifest, "--model_output_dir", source, "--metrics_dir", attempt,
        "--run_name", run, "--rows", ",".join(str(i["_row"]) for i in items),
        "--source_duration", args.duration, "--eval_durations", args.duration,
        "--learned_metrics", "lpips,fvd", "--frame_stride", 30, "--learned_image_size", 224,
        "--metric_batch_size", 8, "--fvd_clip_length", 16, "--fvd_clips_per_video", 4,
        "--fvd_frame_stride", 4, "--fvd_image_size", 224, "--fvd_backend", "styleganv_i3d",
        "--fvd_eps", "1e-6", "--fvd_cache_dir", Path.home() / "hf_cache/memcam_fvd", "--strict")
    return cmd


def reuse_control_quality(args, inputs, items):
    previous = getattr(args, "reuse_control_metrics", None)
    if previous is None or not (previous / "metrics/control/quality.json").is_file():
        return None
    try:
        config = load(previous / "config.json")
        if config["manifest_sha256"] != digest(args.output / "manifest.jsonl"):
            raise ValueError("different manifest")
        # Permit a new orchestration/CLIP adapter, not changes to the old quality
        # evaluator, its helpers, source files, or the metric environment.
        quality_script = str(REPO / "utils/evaluate_context_memory_prefix_curves.py")
        if quality_script not in config["code"]:
            raise ValueError("missing quality code identity")
        for path, expected in config["code"].items():
            if Path(path).resolve() != Path(__file__).resolve() and digest(path) != expected:
                raise ValueError(f"changed experiment dependency: {path}")
        if load(previous / "environment_memcam.json") != load(args.output / "environment_memcam.json"):
            raise ValueError("changed metric environment")
        record = cached_step(previous / "metrics/control/quality.json", inputs)
        if record is None:
            raise ValueError("different video inputs or incomplete receipt")
        summaries = [Path(p) for p in record["artifacts"] if Path(p).name == "summary.json"]
        if len(summaries) != 1:
            raise ValueError("missing quality summary")
        details = str(summaries[0].parent / "metrics.jsonl")
        if details not in record["artifacts"]:
            raise ValueError("missing hashed per-video records")
        if validate_quality(summaries[0].parent, items, "control", args.duration) != record["scores"]:
            raise ValueError("inconsistent quality scores")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Control quality reuse rejected ({exc}); recomputing metrics only.", flush=True)
        return None
    print(f"REUSE verified control LPIPS/FVD: {previous}", flush=True)
    return {**record, "reused_from": str(previous)}


def validate_vbench_dimension(directory, dimension, videos, run):
    files = list(directory.glob("*_eval_results.json"))
    if len(files) != 1:
        raise ValueError(f"{dimension}: expected one result file")
    payload = load(files[0])
    if set(payload) != {dimension}:
        raise ValueError(f"{dimension}: wrong result dimensions")
    rows = bench_details(payload[dimension])
    expected = {p.name for p in videos}
    if (rows is None or len(rows) != 15 or len(expected) != 15
            or {Path(r["video_path"]).name for r in rows} != expected
            or any(Path(r["video_path"]).parent.name != run for r in rows)
            or not all(valid_bench_score(dimension, r.get("video_results")) for r in rows)):
        raise ValueError(f"{dimension}: incomplete/invalid matched cohort")
    return payload[dimension], files[0]


def evaluate_vbench(args, run, source, work, videos, inputs, env):
    merged, artifacts = {}, {}
    for dimension in DIMENSIONS:
        receipt = work / f"vbench_{dimension}.json"
        identity = {**inputs, "dimension": dimension, "adapter_sha256": digest(VBENCH_ADAPTER),
                    "clip_batch_size": CLIP_BATCH_SIZE}
        record = cached_step(receipt, identity)
        if record is None:
            attempt = Path(tempfile.mkdtemp(prefix=f"vbench_{dimension}_", dir=work))
            cmd = metric_command("vbench", "python", VBENCH_ADAPTER,
                                 "--vbench-root", args.vbench_root, "--clip-batch-size", CLIP_BATCH_SIZE,
                                 "--videos_path", source, "--output_path", attempt,
                                 "--mode", "custom_input", "--dimension", dimension)
            run_command(cmd, work / f"vbench_{dimension}.log", env, args.vbench_root)
            payload, path = validate_vbench_dimension(attempt, dimension, videos, run)
            record = {"status": "complete", "inputs": identity, "payload": payload,
                      "artifacts": fingerprint([path, attempt / "batching.json"])}
            save(receipt, record)
        merged[dimension] = record["payload"]
        artifacts.update(record["artifacts"])
        print(f"VBENCH {run}: {dimension} complete (15/15)", flush=True)
    combined = work / "vbench_combined"
    combined.mkdir(exist_ok=True)
    path = combined / "combined_eval_results.json"
    save(path, merged)
    values = validate_bench(combined, [p.name for p in videos], run)
    scores = {r["metric"]: r["value"] for r in values}
    save(work / "vbench.json", {"status": "complete", "inputs": inputs, "scores": scores,
                               "artifacts": {**artifacts, **fingerprint([path])},
                               "adapter_sha256": digest(VBENCH_ADAPTER), "clip_batch_size": CLIP_BATCH_SIZE})
    return scores


def evaluate(args, run, items, manifest, env):
    source = args.output / "videos" / run
    work = args.output / "metrics" / run
    work.mkdir(parents=True, exist_ok=True)
    videos = [source / (i["output_prefix"] + "custom.mp4") for i in items]
    inputs = fingerprint(videos)
    inputs["duration"] = args.duration
    quality_path = work / "quality.json"
    record = cached_step(quality_path, inputs)
    if record is None and run == "control":
        record = reuse_control_quality(args, inputs, items)
        if record is not None:
            save(quality_path, record)
    if record is None:
        attempt = Path(tempfile.mkdtemp(prefix="quality_", dir=work))
        run_command(quality_command(args, manifest, source, attempt, run, items), work / "quality.log", env)
        directory = attempt / run
        scores = validate_quality(directory, items, run, args.duration)
        artifacts = fingerprint([directory / "metrics.jsonl", directory / "summary.json"])
        record = {"status": "complete", "inputs": inputs, "scores": scores, "artifacts": artifacts}
        save(quality_path, record)
    scores = dict(record["scores"])
    scores.update(evaluate_vbench(args, run, source, work, videos, inputs, env))
    if fingerprint(videos) != {k: v for k, v in inputs.items() if k != "duration"}:
        raise ValueError("Video sources changed during evaluation")
    save(work / "scores.json", scores)
    print(f"EVALUATED {run}: {scores}", flush=True)
    return scores


def generation_command(args, manifest, run, alpha, item):
    return metric_command("memcam", "python", REPO / "utils/run_context_memory_batch.py",
        "--manifest", manifest, "--rows", item["_row"], "--durations", args.duration,
        "--output_dir", args.output / "videos" / run, "--memory_policy", "slam_covisibility",
        "--memory_budget", 32, "--keepsake_geometry_weight", alpha,
        "--num_inference_steps", 50, "--seed", 42, "--overwrite")


def validate_generated(source, item, alpha):
    name = item["output_prefix"] + "custom"
    video = source / (name + ".mp4")
    trace = source / "access_traces" / (name + ".jsonl")
    audit_trace(trace, item, "slam_covisibility", 32)
    observed = False
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "context_access":
            if event.get("keepsake_geometry_weight") != alpha:
                raise ValueError("Generated trace has the wrong affinity weight")
            observed = True
    if not observed:
        raise ValueError("No generated reads")
    probe_video(video, item)
    return fingerprint([video, trace])


def generate(args, run, alpha, items, manifest, env):
    work = args.output / "generation" / run
    work.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items):
        state = work / f"row_{item['_row']:03d}.json"
        inputs = {"row": item["_row"], "alpha": alpha, "duration": args.duration,
                  "seed": 42, "budget": 32, "manifest": digest(manifest)}
        if cached_step(state, inputs):
            print(f"KEEP {run} [{index+1}/15]: {item['scene']}", flush=True)
            continue
        print(f"GENERATE {run} [{index+1}/15]: {item['scene']}", flush=True)
        run_command(generation_command(args, manifest, run, alpha, item),
                    work / f"row_{item['_row']:03d}.log", env)
        artifacts = validate_generated(args.output / "videos" / run, item, alpha)
        save(state, {"status": "complete", "inputs": inputs, "artifacts": artifacts})


def export(args):
    rows = []
    for run, alpha in SETTINGS:
        scores = load(args.output / "metrics" / run / "scores.json")
        aggregate = 100 * sum(w * (scores[k]-lo)/(hi-lo) for k, (lo, hi, w) in NORMALIZATION.items()) / 5.5
        rows.append(dict(setting=LABELS[run], geometry_weight=alpha, appearance_weight=1-alpha,
                         duration_sec=args.duration, videos=15, frames=32,
                         FVD=scores["FVD"], LPIPS=scores["LPIPS"],
                         **{dim: 100*scores[dim] for dim in DIMENSIONS}, vbench_weighted6=aggregate))
    with (args.output / "ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{table}[t]", r"\centering\small",
             rf"\caption{{Pose--appearance ablation at B32 on 15 matched {args.duration}-second MemCam videos. VBench-6 is the normalized six-dimension weighted aggregate (\%).}}",
             r"\label{tab:pose-appearance-ablation}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Setting & Pose & Appearance & FVD $\downarrow$ & LPIPS $\downarrow$ & VBench-6 $\uparrow$ \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['setting']} & {row['geometry_weight']:.2f} & {row['appearance_weight']:.2f} & "
                     f"{row['FVD']:.1f} & {row['LPIPS']:.4f} & {row['vbench_weighted6']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.output / "ablation.tex").write_text("\n".join(lines) + "\n")
    print((args.output / "ablation.csv").read_text(), flush=True)


def prepare(args):
    items = cohort(args.manifest, args.duration)
    for item in items:
        for key in ("input_image", "pose_path", "gt_frames_dir"):
            if not Path(item[key]).exists():
                raise FileNotFoundError(item[key])
    frozen = args.output / "manifest.jsonl"
    paths = list((REPO / "utils").glob("*.py")) + [Path(__file__), REPO / "paper/audit_gap_inputs.py",
            REPO / "paper/run_vbench_180s.py", REPO / "diffsynth/pipelines/wan_video_memcam.py",
            REPO / "diffsynth/pipelines/memory_policies.py", REPO / "inference_memcam.py", REPO / "dataset/poses.py"]
    config = {"manifest_sha256": digest(args.manifest), "code": fingerprint(paths),
              "generation_checkpoints": generation_assets(),
              "duration": args.duration, "control": str(args.control), "settings": list(SETTINGS),
              "seed": 42, "steps": 50, "budget": 32, "height": 352, "width": 640,
              "control_seed_provenance": "Legacy seed is not logged; 42 is the original batch runner default, not independently verified.",
              "control_weight_provenance": "Legacy missing weight metadata uses the original fixed 0.65/0.35 implementation."}
    encoded = json.dumps(config, sort_keys=True, indent=2)
    config_path = args.output / "config.json"
    if config_path.exists():
        if config_path.read_text() != encoded or digest(frozen) != config["manifest_sha256"]:
            raise ValueError("Experiment code/config changed; use a different output directory")
    else:
        frozen.write_bytes(args.manifest.read_bytes())
        config_path.write_text(encoded)
    staged = args.output / "videos/control"
    staged.mkdir(parents=True, exist_ok=True)
    validation = args.output / "control_validation.json"
    sources = []
    for item in items:
        name = item["output_prefix"] + "custom"
        video = args.control / (name + ".mp4")
        trace = args.control / "access_traces" / (name + ".jsonl")
        sources += [video, trace]
    inputs = fingerprint(sources)
    if not cached_step(validation, inputs):
        for item in items:
            name = item["output_prefix"] + "custom"
            video = args.control / (name + ".mp4")
            trace = args.control / "access_traces" / (name + ".jsonl")
            audit_trace(trace, item, "slam_covisibility", 32)
            for line in trace.read_text().splitlines():
                event = json.loads(line)
                if "keepsake_geometry_weight" in event and event["keepsake_geometry_weight"] != .65:
                    raise ValueError(f"Control has a non-default weight: {trace}")
            probe_video(video, item)
        save(validation, {"status": "complete", "inputs": inputs, "artifacts": inputs})
    for item in items:
        name = item["output_prefix"] + "custom.mp4"
        video = (args.control / name).resolve()
        dest = staged / name
        if not dest.is_symlink():
            dest.symlink_to(video)
        elif dest.resolve() != video:
            raise ValueError(f"Wrong staged control: {dest}")
    if sorted(p.name for p in staged.iterdir()) != sorted(i["output_prefix"] + "custom.mp4" for i in items):
        raise ValueError("Control staging contains extra videos")
    return items, frozen


def step_command(args, run):
    # Slurm owns GPU assignment and cgroup-local device numbering for each step.
    command = ["srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=96G", "--gres=gpu:nvidia_h100_pcie:1",
            "--kill-on-bad-exit=1", "--export=ALL",
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker", run,
            "--output", str(args.output), "--duration", str(args.duration),
            "--vbench-root", str(args.vbench_root)]
    if getattr(args, "reuse_control_metrics", None) is not None:
        command += ["--reuse-control-metrics", str(args.reuse_control_metrics)]
    return command


def preflight(args, run):
    step = os.environ.get("SLURM_STEP_ID")
    if not step or step in ("batch", "extern"):
        raise RuntimeError("GPU workers must run inside a resource-assigned srun step")
    allocation = {name: os.environ.get(name) for name in (
        "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    print(f"WORKER {run}: {json.dumps(allocation, sort_keys=True)}", flush=True)
    check = (cuda_check_code()
             + "; print('Visible CUDA devices:', torch.cuda.device_count(), flush=True)"
             + "; assert torch.cuda.device_count() == 1, 'Expected one GPU in this Slurm step'")
    for name in ("memcam", "vbench"):
        run_command(metric_command(name, "python", "-c", check),
                    args.output / f"preflight_{run}_{name}.log", os.environ.copy())


def worker(args):
    run = args.worker
    if run not in dict(SETTINGS):
        raise ValueError(f"Unknown setting: {run}")
    items = cohort(args.output / "manifest.jsonl", args.duration)
    alpha = dict(SETTINGS)[run]
    path = args.output / f"status_{run}.json"
    state = {"status": "running", "setting": run, "phase": "preflight",
             "job": os.environ.get("SLURM_JOB_ID"), "step": os.environ.get("SLURM_STEP_ID")}
    save(path, state)
    try:
        preflight(args, run)
        if run != "control":
            state["phase"] = "generation"
            save(path, state)
            generate(args, run, alpha, items, args.output / "manifest.jsonl", os.environ.copy())
        state["phase"] = "evaluation"
        save(path, state)
        evaluate(args, run, items, args.output / "manifest.jsonl", os.environ.copy())
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        save(path, state)


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute(args):
    if not os.environ.get("SLURM_JOB_ID") or not shutil.which("srun"):
        raise RuntimeError("Submit the two-GPU batch launcher; this driver requires Slurm srun")
    prepare(args)
    for name in ("memcam", "vbench"):
        freeze_environment({"output": str(args.output), "vbench_root": str(args.vbench_root)}, name)
    # Finish real control evaluation before spending compute on either endpoint.
    run_command(step_command(args, "control"), args.output / "control.log", os.environ.copy())
    children = []
    handles = []
    try:
        for run in ("appearance_only", "pose_only"):
            handle = (args.output / f"{run}.log").open("ab")
            handles.append(handle)
            cmd = step_command(args, run)
            start_log(handle, cmd)
            children.append(subprocess.Popen(cmd, cwd=REPO, env=os.environ.copy(), stdout=handle,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        heartbeat = 0
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(f"Endpoint failed; inspect {args.output}/status_*.json and endpoint logs")
            if time.monotonic() >= heartbeat:
                for run in ("appearance_only", "pose_only"):
                    done = len(list((args.output / "generation" / run).glob("row_*.json")))
                    state = args.output / f"status_{run}.json"
                    phase = load(state).get("phase", "starting") if state.exists() else "starting"
                    print(f"PROGRESS {run}: {done}/15 videos validated; {phase}. Log: {args.output / (run + '.log')}", flush=True)
                heartbeat = time.monotonic() + 60
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError(f"Endpoint failed; inspect {args.output}/status_*.json")
    finally:
        stop_workers(children)
        for handle in handles:
            handle.close()
    for name in ("memcam", "vbench"):
        freeze_environment({"output": str(args.output), "vbench_root": str(args.vbench_root)}, name)
    export(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, choices=(60, 180), default=180)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    parser.add_argument("--reuse-control-metrics", type=Path,
                        help="Previous experiment root; reuse only audited control LPIPS/FVD, never its VBench results")
    parser.add_argument("--worker", choices=("appearance_only", "control", "pose_only"))
    args = parser.parse_args()
    args.manifest = (args.manifest or REPO / ("testbeds/context_memory_180s/manifest.jsonl" if args.duration == 180
                                            else "testbeds/context_memory/manifest.jsonl")).resolve()
    args.control = (args.control or Path.home() / "memcam_results" /
                    ("context_180s" if args.duration == 180 else "context_memory_60s") / "slam_b32_covisibility").resolve()
    args.output, args.vbench_root = args.output.resolve(), args.vbench_root.resolve()
    if args.reuse_control_metrics is not None:
        args.reuse_control_metrics = args.reuse_control_metrics.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    state = {"status": "running", "job": os.environ.get("SLURM_JOB_ID"), "duration": args.duration,
             "planned_new_videos": 30, "reused_control_videos": 15}
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(args.output / "status.json", state)
        try:
            execute(args)
            state["status"] = "complete"
            print(f"COMPLETE: {args.output / 'ablation.csv'} and {args.output / 'ablation.tex'}", flush=True)
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            save(args.output / "status.json", state)


if __name__ == "__main__":
    main()
