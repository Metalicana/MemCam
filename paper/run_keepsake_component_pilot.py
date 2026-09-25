"""One-H100 component pilot: five settings x three matched 30-second videos.

This is not the 15-video main evaluation. All controls are generated afresh.
LPIPS, PSNR and SSIM are evaluated per video; no small-cohort FVD is reported.
"""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paper import finish_keepsake_pose_appearance as common

SETTINGS = (
    ("full", .65, "full"),
    ("pose_only", 1., "full"),
    ("appearance_only", 0., "full"),
    ("degree_only", .65, "degree_only"),
    ("closest_only", .65, "closest_only"),
)
LABELS = {"full": "KEEPSAKE", "pose_only": "Without appearance", "appearance_only": "Without pose",
          "degree_only": "Without closest-substitute term", "closest_only": "Without degree term"}
METRICS = ("lpips_alex", "psnr_db", "ssim")
DURATION, FRAMES, VIDEOS, SEED = 30, 913, 3, 42


def select_cohort(path):
    source = [dict(json.loads(line), source_row=i) for i, line in enumerate(path.read_text().splitlines())
              if line.strip()]
    source = [item for item in source if int(item["duration_sec"]) == 60]
    if len(source) != 15 or len({i["scene"] for i in source}) != 15:
        raise ValueError("Pilot selection requires the fixed 15-distinct-scene 60s manifest")
    selected = sorted(random.Random(0).sample(source, VIDEOS), key=lambda i: i["source_row"])
    for row, item in enumerate(selected):
        if float(item["fps"]) != 30 or int(item["num_frames"]) < FRAMES:
            raise ValueError("Expected 30-FPS source trajectories covering the pilot horizon")
        item.update(source_duration_sec=60, duration_sec=DURATION, num_frames=FRAMES, _row=row,
                    output_prefix=f"component_row{item['source_row']:03d}_{item['scene']}_"
                                  f"{int(item['start_frame']):04d}_30s_")
    return selected


def prepare(args):
    items = select_cohort(args.manifest)
    source_files = []
    for item in items:
        for key in ("input_image", "pose_path", "gt_frames_dir"):
            if not Path(item[key]).exists():
                raise FileNotFoundError(item[key])
        source_files += [Path(item["input_image"]), Path(item["pose_path"])]
        for frame in range(0, FRAMES, 30):
            gt = Path(item["gt_frames_dir"]) / f"{int(item['start_frame']) + frame:04d}.png"
            if not gt.is_file():
                raise FileNotFoundError(f"Missing sampled GT before generation: {gt}")
            source_files.append(gt)
    paths = list((REPO / "utils").glob("*.py")) + [Path(__file__), Path(common.__file__),
             REPO / "paper/audit_gap_inputs.py", REPO / "paper/run_vbench_180s.py",
             REPO / "diffsynth/pipelines/wan_video_memcam.py", REPO / "diffsynth/pipelines/memory_policies.py",
             REPO / "dataset/poses.py", REPO / "inference_memcam.py"]
    config = {"schema": 1, "source_manifest_sha256": common.digest(args.manifest),
              "code": common.fingerprint(paths), "checkpoints": common.generation_assets(),
              "source_files": common.fingerprint(source_files),
              "items": items, "settings": SETTINGS, "duration_sec": DURATION, "num_frames": FRAMES,
              "videos_per_setting": VIDEOS, "selection_seed": 0, "generation_seed": SEED,
              "budget": 32, "steps": 50, "resolution": [352, 640],
              "metrics": METRICS, "frame_stride": 30, "learned_image_size": 224,
              "scope": "small matched component pilot, not main evaluation or parameter robustness"}
    encoded = json.dumps(config, indent=2, sort_keys=True)
    path = args.output / "config.json"
    frozen = args.output / "manifest.jsonl"
    manifest_text = "\n".join(json.dumps(i, sort_keys=True) for i in items) + "\n"
    if path.exists():
        if path.read_text() != encoded or frozen.read_text() != manifest_text:
            raise ValueError("Pilot inputs/code changed; choose a new output directory")
    else:
        path.write_text(encoded)
        frozen.write_text(manifest_text)
    return items, frozen


def generation_command(args, manifest, setting, item):
    run, alpha, mode = setting
    command = common.generation_command(args, manifest, run, alpha, item)
    return command + ["--keepsake_priority_mode", mode]


def validate_generated(source, item, alpha, mode):
    artifacts = common.validate_generated(source, item, alpha)
    trace = source / "access_traces" / (item["output_prefix"] + "custom.jsonl")
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    reads = [event for event in events if event.get("event") == "context_access"]
    if not reads or any(event.get("keepsake_priority_mode") != mode for event in reads):
        raise ValueError("Generated trace has the wrong priority component")
    return artifacts


def quality_command(args, manifest, source, attempt, run, item):
    command = common.quality_command(args, manifest, source, attempt, run, [item])
    command[command.index("--learned_metrics") + 1] = "lpips"
    return command


def validate_quality(directory, item, run):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected exactly one per-video metric record")
    row = rows[0]
    if (row["status"] != "completed" or row["run_name"] != run or int(row["row"]) != item["_row"]
            or row["scene"] != item["scene"] or int(row["start_frame"]) != int(item["start_frame"])
            or int(row["duration_sec"]) != DURATION or int(row["num_frames_expected"]) != FRAMES
            or Path(row["output"]).name != item["output_prefix"] + "custom.mp4"
            or int(row["frames_evaluated"]) != len(range(0, FRAMES, 30))
            or not all(math.isfinite(float(row.get(k, float("nan")))) for k in METRICS)):
        raise ValueError("Incomplete or mismatched pilot metrics")
    config = common.load(directory / "summary.json")["metric_config"]
    expected = dict(source_duration=DURATION, eval_durations=[DURATION], max_frames=None,
                    frame_stride=30, learned_image_size=224, learned_metrics=["lpips"])
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Wrong pilot metric configuration")
    return {key: float(row[key]) for key in METRICS}


def execute_cell(args, manifest, setting, item):
    run, alpha, mode = setting
    work = args.output / "cells" / run / f"row_{item['_row']:03d}"
    work.mkdir(parents=True, exist_ok=True)
    source = args.output / "videos" / run
    inputs = {"manifest_sha256": common.digest(manifest), "row": item["_row"], "alpha": alpha,
              "priority_mode": mode, "duration": DURATION, "seed": SEED, "budget": 32}
    receipt = work / "generation.json"
    if common.cached_step(receipt, inputs) is None:
        common.run_command(generation_command(args, manifest, setting, item), work / "generation.log", os.environ.copy())
        artifacts = validate_generated(source, item, alpha, mode)
        common.save(receipt, {"status": "complete", "inputs": inputs, "artifacts": artifacts})
    video = source / (item["output_prefix"] + "custom.mp4")
    metric_inputs = {**inputs, "video_sha256": common.digest(video)}
    record = common.cached_step(work / "quality.json", metric_inputs)
    if record is None:
        attempt = Path(tempfile.mkdtemp(prefix="quality_", dir=work))
        common.run_command(quality_command(args, manifest, source, attempt, run, item), work / "quality.log", os.environ.copy())
        directory = attempt / run
        scores = validate_quality(directory, item, run)
        record = {"status": "complete", "inputs": metric_inputs, "scores": scores,
                  "artifacts": common.fingerprint([directory / "metrics.jsonl", directory / "summary.json"])}
        common.save(work / "quality.json", record)
    return dict(setting=run, geometry_weight=alpha, priority_mode=mode, source_row=item["source_row"],
                scene=item["scene"], start_frame=item["start_frame"], duration_sec=DURATION,
                frames=FRAMES, budget=32, seed=SEED, **record["scores"])


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export(output, rows):
    expected = {(run, row["source_row"]) for run, _, _ in SETTINGS for row in rows if row["setting"] == "full"}
    observed = {(row["setting"], row["source_row"]) for row in rows}
    if len(rows) != len(observed) or len(expected) != 15 or observed != expected:
        raise ValueError("Cannot export a complete ablation from missing or duplicate cells")
    controls = {r["source_row"]: r for r in rows if r["setting"] == "full"}
    paired = [{**r, **{f"delta_{k}_vs_full": r[k] - controls[r["source_row"]][k] for k in METRICS}} for r in rows]
    write_csv(output / "per_video.csv", paired)
    summary = []
    for run, alpha, mode in SETTINGS:
        subset = [r for r in paired if r["setting"] == run]
        summary.append(dict(setting=LABELS[run], geometry_weight=alpha, priority_mode=mode,
                            videos=VIDEOS, duration_sec=DURATION, budget=32,
                            **{k: mean(r[k] for r in subset) for k in METRICS},
                            **{f"delta_{k}_vs_full": mean(r[f"delta_{k}_vs_full"] for r in subset) for k in METRICS}))
    write_csv(output / "ablation.csv", summary)
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Component pilot on three matched 30-second MemCam trajectories, $B=32$.}",
             r"\label{tab:keepsake-components-pilot}", r"\begin{tabular}{lrrr}\toprule",
             r"Variant & LPIPS $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ \\", r"\midrule"]
    for row in summary:
        lines.append(f"{row['setting']} & {row['lpips_alex']:.4f} & {row['psnr_db']:.2f} & {row['ssim']:.4f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (output / "ablation.tex").write_text("\n".join(lines) + "\n")


def execute(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit the one-H100 pilot launcher; this is not a login-node job")
    items, manifest = prepare(args)
    common.freeze_environment({"output": str(args.output)}, "memcam")
    check = common.cuda_check_code() + "; assert torch.cuda.device_count() == 1"
    common.run_command(common.metric_command("memcam", "python", "-c", check), args.output / "cuda_preflight.log")
    metric_check = (f"import sys; sys.path.insert(0, {str(REPO / 'utils')!r}); "
                    "import torch; from skimage.metrics import structural_similarity; "
                    "from evaluate_context_memory import LearnedMetricRunner; "
                    "assert torch.cuda.is_available(); m=LearnedMetricRunner(['lpips']); "
                    "x=torch.zeros(1,3,224,224,device='cuda'); "
                    "assert torch.isfinite(m.lpips_model(x,x)).all(); print('LPIPS/SSIM preflight OK')")
    common.run_command(common.metric_command("memcam", "python", "-c", metric_check), args.output / "metric_preflight.log")
    rows, started = [], time.monotonic()
    for item in items:
        for setting in SETTINGS:
            print(f"CELL {len(rows)+1}/15: {setting[0]} / {item['scene']}", flush=True)
            rows.append(execute_cell(args, manifest, setting, item))
            write_csv(args.output / "partial_per_video.csv", rows)
            elapsed = (time.monotonic() - started) / 3600
            common.save(args.output / "progress.json", dict(completed_cells=len(rows), planned_cells=15,
                        elapsed_hours=elapsed, last_setting=setting[0], last_scene=item["scene"]))
            print(f"COMPLETE {len(rows)}/15 cells; elapsed {elapsed:.2f} h", flush=True)
    common.freeze_environment({"output": str(args.output)}, "memcam")
    config = common.load(args.output / "config.json")
    if common.fingerprint(config["source_files"]) != config["source_files"]:
        raise ValueError("Pilot inputs changed during execution")
    export(args.output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.manifest, args.output, args.duration = args.manifest.resolve(), args.output.resolve(), DURATION
    if args.dry_run:
        print(json.dumps(dict(items=select_cohort(args.manifest), settings=SETTINGS,
                              new_videos=15, metrics=METRICS, duration_sec=DURATION), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status="running", job=os.environ.get("SLURM_JOB_ID"), planned_new_videos=15,
                     videos_per_setting=VIDEOS, duration_sec=DURATION, pilot=True)
        common.save(args.output / "status.json", state)
        try:
            execute(args)
            state["status"] = "complete"
            print(f"COMPLETE: {args.output / 'ablation.csv'}", flush=True)
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            common.save(args.output / "status.json", state)


if __name__ == "__main__":
    main()
