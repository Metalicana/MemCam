"""Standard VBench on the four matched 180-second headline configurations."""

import argparse
import csv
import fcntl
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "utils"))
from run_budget_metric_grid import (DIMENSIONS, digest, freeze_environment,
                                   run_logged, save, validate_bench)
from metric_environment import metric_command
from smoke_video_metrics import cuda_check_code

RUNS = ("baseline", "fifo_b32", "ri_b32_dino_rgb", "slam_b32_covisibility")


def cohort(manifest, root):
    items = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    items = [item for item in items if int(item["duration_sec"]) == 180]
    names = [item["output_prefix"] + "custom.mp4" for item in items]
    if len(items) != 15 or len(set(names)) != 15:
        raise ValueError("Expected exactly 15 distinct 180-second manifest videos")
    for item in items:
        if int(item["num_frames"]) != 5397 or float(item["fps"]) != 30:
            raise ValueError("Headline cohort must have 5,397 frames at 30 FPS")
    for run in RUNS:
        for name in names:
            path = root / run / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty matched video: {path}")
    return items, names


def probe_video(video, item):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
        "-of", "json", str(video)], text=True))
    stream = probe["streams"][0]
    if (int(stream["nb_read_frames"]) != int(item["num_frames"])
            or abs(float(Fraction(stream["r_frame_rate"])) - float(item["fps"])) > 1e-6):
        raise ValueError(f"Wrong frame count or FPS: {video}")
    return probe


def execute(args):
    run = RUNS[args.task]
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / run
    # Never overwrite or reuse an incomplete result without explicit inspection.
    work.mkdir()
    status = {"status": "validating", "run": run, "duration_sec": 180,
              "videos": 15, "evaluator": "standard-vbench", "job": os.getenv("SLURM_JOB_ID")}
    save(work / "status.json", status)
    try:
        manifest_bytes = args.manifest.read_bytes()
        items, names = cohort(args.manifest, args.root)
        if args.manifest.read_bytes() != manifest_bytes:
            raise ValueError("Manifest changed during validation")
        (work / "manifest.jsonl").write_bytes(manifest_bytes)
        identity = {"manifest_sha256": digest(work / "manifest.jsonl"),
                    "names": names, "root": str(args.root.resolve())}
        with (args.output / "cohort.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            shared = args.output / "cohort.json"
            if shared.exists() and json.loads(shared.read_text()) != identity:
                raise ValueError("Cohort differs from earlier tasks in this array")
            if not shared.exists():
                save(shared, identity)
        staged = work / "input" / run
        staged.mkdir(parents=True)
        sources = []
        for item, name in zip(items, names):
            video = (args.root / run / name).resolve()
            probe = probe_video(video, item)
            sources.append({"path": str(video), "sha256": digest(video), "probe": probe})
            (staged / name).symlink_to(video)
        save(work / "sources.json", sources)
        print(f"Validated {run}: 15 matched videos, 5,397 frames each; extras excluded", flush=True)
        subprocess.run(metric_command("vbench", "python", "-c", cuda_check_code()), check=True)
        freeze = freeze_environment({"output": str(args.output),
                                     "vbench_root": str(args.vbench_root)}, "vbench")
        (work / "pip_freeze.txt").write_text(freeze)
        result = work / "results"
        command = metric_command("vbench", "python", args.vbench_root / "evaluate.py",
                                 "--videos_path", staged, "--mode", "custom_input",
                                 "--output_path", result, "--dimension", *DIMENSIONS)
        status.update(status="running", command=command, manifest_sha256=digest(work / "manifest.jsonl"))
        save(work / "status.json", status)
        run_logged(command, work, args.vbench_root)
        scores = validate_bench(result, names, run)
        for source in sources:
            if digest(source["path"]) != source["sha256"]:
                raise ValueError(f"Video changed during evaluation: {source['path']}")
        freeze_environment({"output": str(args.output), "vbench_root": str(args.vbench_root)}, "vbench")
        with (work / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("run", "duration_sec", "metric", "value", "n", "source"))
            writer.writeheader()
            for score in scores:
                writer.writerow({"run": run, "duration_sec": 180, **score})
                print(f"{run:26s} {score['metric']:24s} {score['value']:.6f} N=15", flush=True)
        status.update(status="complete", scores=str(work / "scores.csv"))
        print(f"DONE: {work / 'scores.csv'}", flush=True)
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        save(work / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(4), required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory_180s/manifest.jsonl")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_180s")
    parser.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("manifest", "root", "vbench_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    execute(args)


if __name__ == "__main__":
    main()
