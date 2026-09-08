"""One original 60s video, two independent GPU evaluator smoke tests.

Run on an allocated GPU node. No generation, package installation, or batch
submission is performed. Passing CUT3R is a runtime check, not calibration.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score


REPO = Path(__file__).resolve().parents[1]


def long_import_check():
    modules = ["moviepy.editor", "av", "dreamsim"] + [
        f"vbench2_beta_long.{dimension}" for dimension in DIMENSIONS
    ]
    # Import all dimensions before preprocessing, collecting independent failures.
    return f"""import importlib
errors = []
for name in {modules!r}:
    try:
        importlib.import_module(name)
        print('IMPORT OK:', name, flush=True)
    except Exception as exc:
        errors.append(name)
        print('IMPORT FAILED:', name, type(exc).__name__, str(exc), flush=True)
if errors:
    raise SystemExit('VBench-Long dependency preflight failed: ' + ', '.join(errors))
"""


def load(path):
    return json.loads(path.read_text())


def select_video(manifest, root, run, row=None):
    items = [(i, json.loads(line)) for i, line in enumerate(manifest.read_text().splitlines())
             if line.strip()]
    selected = [(i, item) for i, item in items
                if int(item["duration_sec"]) == 60 and (row is None or i == row)]
    if not selected:
        raise ValueError("No matching 60s manifest row")
    index, item = selected[0]
    video = root / run / (item["output_prefix"] + "custom.mp4")
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"Selected original-cohort video is missing/empty: {video}")
    return index, item, video.resolve()


def validate_long(output, video_name):
    paths = sorted(output.glob("*_eval_results.json"))
    if not paths:
        raise ValueError("VBench-Long produced no final result JSON")
    payload = load(paths[-1])
    for dimension in DIMENSIONS:
        value = payload.get(dimension)
        details = bench_details(value)
        if details is None or not finite(value[0]):
            raise ValueError(f"Missing/invalid VBench-Long dimension: {dimension}")
        rows = [r for r in details if isinstance(r, dict) and "video_path" in r]
        if (len(rows) != 1 or Path(rows[0]["video_path"]).name != video_name
                or not valid_bench_score(dimension, rows[0].get("video_results"))):
            raise ValueError(
                f"{dimension}: expected one aggregate for {video_name}; got "
                f"{[r.get('video_path') for r in rows]}. Check Long's clip-to-video grouping."
            )


def validate_cut3r(output, metrics, run, row):
    status = load(output / "cut3r_run_status.json")
    if len(status) != 1 or status[0].get("status") != "completed":
        raise ValueError(f"CUT3R did not complete the selected reconstruction: {status}")
    reconstruction = Path(status[0]["output_dir"])
    metadata = load(reconstruction / "metadata.json")
    expected = len(metadata.get("video_frame_indices", []))
    cameras = list((reconstruction / "camera").glob("*.npz"))
    if expected < 2 or len(cameras) != expected:
        raise ValueError(f"CUT3R camera count: expected {expected}, got {len(cameras)}")
    failures = load(metrics / "cut3r_camera_failures.json")
    results = load(metrics / "cut3r_camera_metrics.json")
    if failures or len(results) != 1:
        raise ValueError(f"CUT3R scoring failed/partial: {failures}")
    result = results[0]
    if result.get("run_name") != run or result.get("row") != row:
        raise ValueError("CUT3R scored a different run/row")
    for key in ("rotation_error_deg_mean", "translation_error_scale_only_mean",
                "translation_error_sim3_mean"):
        if not finite(result.get(key)):
            raise ValueError(f"Non-finite CUT3R result: {key}")


def execute(command, cwd, log):
    print("+ " + shlex.join(map(str, command)), flush=True)
    log.write("+ " + shlex.join(map(str, command)) + "\n")
    log.flush()
    with subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--results-root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "CUT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "memcam_results/metric_smoke_60s")
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--row", type=int, help="Default: first 60s row in original manifest")
    parser.add_argument("--only", choices=("both", "vbench-long", "cut3r"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    row, item, video = select_video(args.manifest, args.results_root, args.run, args.row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="smoke_", dir=args.output_root)).resolve()
    staged = work / "input" / args.run
    staged.mkdir(parents=True)
    (staged / video.name).symlink_to(video)
    spec = {
        "row": row, "manifest_item": item, "source_video": str(video),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_bytes": video.stat().st_size, "source_mtime_ns": video.stat().st_mtime_ns,
        "checkpoint": str(args.checkpoint), "vbench_root": str(args.vbench_root),
        "long_grouping_adapter_sha256": hashlib.sha256(
            (REPO / "utils/run_vbench_long.py").read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dry_run": args.dry_run,
    }
    (work / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Smoke directory: {work}\nRun: {args.run}; manifest row: {row}\nVideo: {video}", flush=True)
    if args.dry_run:
        print("Dry run: staged only; no GPU evaluators invoked.")
        return

    # Probe the same selected file before either evaluator is allowed to run.
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(video)
    ], text=True)
    (work / "video_probe.json").write_text(probe)
    frames = int(json.loads(probe)["streams"][0]["nb_read_frames"])
    if frames != int(item["num_frames"]):
        raise ValueError(f"Video length mismatch: {frames} frames vs manifest {item['num_frames']}")

    long_out = work / "vbench_long_results"
    recon = work / "cut3r_reconstruction"
    scores = work / "cut3r_metrics"
    cuda_check = "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
    def conda(env, *command):
        return ["conda", "run", "--no-capture-output", "-n", env, *command]

    stages = {
        "vbench-long": [
            (conda("vbench", "python", "-c", cuda_check), REPO),
            (conda("vbench", "python", "-c", long_import_check()), args.vbench_root),
            (conda("vbench", "python", REPO / "utils/run_vbench_long.py",
                   "--vbench-root", args.vbench_root.resolve(),
                   "--videos_path", staged, "--dimension", *DIMENSIONS,
                   "--mode", "long_custom_input", "--dev_flag", "--output_path", long_out), args.vbench_root),
        ],
        "cut3r": [
            (conda("memcam", "python", "-c", cuda_check), REPO),
            (conda("memcam", "python", "utils/run_cut3r_context_memory.py",
                   "--manifest", args.manifest.resolve(), "--root", args.results_root.resolve(),
                   "--runs", args.run, "--rows", str(row), "--durations", "60",
                   "--output_dir", recon, "--cut3r_root", REPO / "CUT3R",
                   "--model_path", args.checkpoint.resolve(), "--size", "512", "--device", "cuda",
                   "--frame_stride", "30", "--max_frames", "120"), REPO),
            (conda("memcam", "python", "utils/evaluate_cut3r_camera_metrics.py",
                   "--cut3r_dir", recon, "--output_dir", scores,
                   "--runs", args.run, "--rows", str(row), "--durations", "60"), REPO),
        ],
    }
    status = {}
    for name, commands in stages.items():
        if args.only not in ("both", name):
            continue
        with (work / f"{name}.log").open("w") as log:
            try:
                for command, cwd in commands:
                    execute(command, cwd, log)
                if name == "vbench-long":
                    validate_long(long_out, video.name)
                else:
                    validate_cut3r(recon, scores, args.run, row)
                status[name] = {"status": "passed"}
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
                status[name] = {"status": "failed", "error": str(exc)}
                print(f"{name} FAILED: {exc}", flush=True)
                log.write(f"VALIDATION/EXECUTION FAILURE: {exc}\n")
        (work / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    print(f"Logs and artifacts: {work}")
    print("CUT3R passing means executable, not GT camera-calibrated. No batch job was submitted.")
    raise SystemExit(1 if any(s["status"] == "failed" for s in status.values()) else 0)


if __name__ == "__main__":
    main()
