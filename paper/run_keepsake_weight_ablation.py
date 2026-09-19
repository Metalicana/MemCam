"""Run one matched generation cell of the Keepsake affinity-weight ablation."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
# Current setting first so task 0 tests the existing method's execution path.
WEIGHTS = (0.65, 0.0, 0.5, 0.8, 1.0)


def task_spec(manifest, duration, task):
    items = [(row, json.loads(line)) for row, line in enumerate(manifest.read_text().splitlines()) if line.strip()]
    items = [(row, item) for row, item in items if int(item["duration_sec"]) == duration]
    if len(items) != 15 or len({item["output_prefix"] for _, item in items}) != 15:
        raise ValueError("Expected exactly 15 distinct matched videos for this duration")
    if not 0 <= task < len(WEIGHTS) * len(items):
        raise ValueError("Task must be in 0..74")
    weight = WEIGHTS[task // 15]
    row, item = items[task % 15]
    return dict(task=task, row=row, item=item, geometry_weight=weight,
                appearance_weight=1.0-weight, budget=32, duration_sec=duration,
                run=f"keepsake_b32_g{round(100*weight):03d}")


def command(spec, manifest, output, seed):
    return [sys.executable, "-u", str(REPO / "utils/run_context_memory_batch.py"),
            "--manifest", str(manifest), "--rows", str(spec["row"]),
            "--durations", str(spec["duration_sec"]), "--output_dir", str(output / spec["run"]),
            "--memory_policy", "slam_covisibility", "--memory_budget", "32",
            "--keepsake_geometry_weight", str(spec["geometry_weight"]),
            "--num_inference_steps", "50", "--seed", str(seed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--duration", type=int, choices=(60, 180), default=180)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest or REPO / ("testbeds/context_memory_180s/manifest.jsonl" if args.duration == 180
                                        else "testbeds/context_memory/manifest.jsonl")
    manifest = manifest.resolve()
    output = args.output.resolve()
    spec = task_spec(manifest, args.duration, args.task)
    if args.dry_run:
        print(json.dumps({**spec, "command": command(spec, manifest, output, args.seed)}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), REPO / "utils/run_context_memory_batch.py",
            REPO / "diffsynth/pipelines/wan_video_memcam.py", REPO / "diffsynth/pipelines/memory_policies.py"]
    frozen = output / "manifest.jsonl"
    identity = dict(duration=args.duration, seed=args.seed, weights=WEIGHTS,
                    code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    # All array cells must use the same manifest, code and generation seed.
    encoded = json.dumps(identity, sort_keys=True)
    with (output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = output / "config.json"
        if config.exists():
            if config.read_text() != encoded or frozen.read_bytes() != manifest.read_bytes():
                raise ValueError("Sweep inputs changed; choose a new output directory")
        else:
            frozen.write_bytes(manifest.read_bytes())
            config.write_text(encoded)
    spec = task_spec(frozen, args.duration, args.task)
    for key in ("pose_path", "input_image"):
        if not Path(spec["item"][key]).is_file():
            raise FileNotFoundError(spec["item"][key])
    task_dir = output / "tasks" / f"{args.task:03d}"
    task_dir.mkdir(parents=True, exist_ok=False)
    state = dict(**spec, seed=args.seed, status="running")
    status = task_dir / "status.json"
    status.write_text(json.dumps(state, indent=2))
    try:
        cmd = command(spec, frozen, output, args.seed)
        state["command"] = cmd
        print(json.dumps(state, indent=2), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        # Existing-file skips are not proof that this coefficient was evaluated.
        run_dir = output / spec["run"]
        records = [json.loads(line) for line in (run_dir / "run_status.jsonl").read_text().splitlines() if line.strip()]
        record = [r for r in records if r["row"] == spec["row"]][-1]
        if record["status"] != "completed" or record.get("keepsake_geometry_weight") != spec["geometry_weight"]:
            raise ValueError("No completed generation with the requested coefficient")
        video = run_dir / (spec["item"]["output_prefix"] + "custom.mp4")
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError("Generation did not produce a video")
        state.update(status="complete", video=str(video))
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        status.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
