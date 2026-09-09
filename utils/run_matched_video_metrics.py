"""Evaluate exactly the original 15 matched 60s videos with a tested evaluator.

Each video uses the smoke runner's isolation and validation. Failed videos do
not prevent remaining videos from running; any failure makes the task fail.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from collect_matched_video_metrics import read_video
from smoke_video_metrics import REPO, cuda_check_code, load, select_video


def latest_suite(root, run, evaluator):
    paths = []
    for path in root.glob("*/suite_status.json"):
        status = load(path)
        if (status.get("run") == run and status.get("evaluator") == evaluator
                and not status.get("dry_run")):
            paths.append(path)
    if not paths:
        raise ValueError(f"No existing suite to resume for {run}/{evaluator}")
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent.resolve()


def resume_plan(work, manifest, root, run, evaluator, rows):
    status = load(work / "suite_status.json")
    if (status.get("dry_run") or status.get("run") != run
            or status.get("evaluator") != evaluator or status.get("rows") != rows
            or (work / "manifest.jsonl").read_bytes() != manifest.read_bytes()):
        raise ValueError("Resume suite identity/manifest does not match the requested cohort")
    valid = set()
    for row in rows:
        _, item, video = select_video(manifest, root, run, row)
        try:
            read_video(work, run, evaluator, row, item)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        spec = load(next((work / f"row_{row:03d}").glob("smoke_*/spec.json")))
        if Path(spec["source_video"]).resolve() != video:
            raise ValueError(f"Row {row}: resume source path changed")
        if (spec.get("source_bytes") != video.stat().st_size
                or spec.get("source_mtime_ns") != video.stat().st_mtime_ns):
            raise ValueError(f"Row {row}: source file changed; use a fresh suite")
        if evaluator == "vbench-long":
            digest = hashlib.sha256((REPO / "utils/run_vbench_long.py").read_bytes()).hexdigest()
            if spec.get("long_grouping_adapter_sha256") != digest:
                raise ValueError("Grouping adapter changed; use a fresh suite")
        valid.add(row)
    return valid


def archive_row(work, row):
    output = work / f"row_{row:03d}"
    if output.exists():
        archive = work / "failed_attempts"
        archive.mkdir(exist_ok=True)
        output.rename(archive / f"row_{row:03d}_{uuid.uuid4().hex}")


def save_status(work, summary):
    temp = work / "suite_status.json.tmp"
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(work / "suite_status.json")


def matched_rows(manifest, root, run):
    rows = [i for i, line in enumerate(manifest.read_text().splitlines())
            if line.strip() and int(json.loads(line)["duration_sec"]) == 60]
    if len(rows) != 15:
        raise ValueError(f"Expected exactly 15 original 60s manifest entries, got {len(rows)}")
    videos = [select_video(manifest, root, run, row)[2] for row in rows]
    if len(set(videos)) != 15:
        raise ValueError("Original cohort contains duplicate source videos")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--only", choices=("vbench-long", "cut3r"), required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--results-root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "memcam_results/matched15_metrics_60s")
    parser.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    parser.add_argument("--dry-run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-suite", type=Path)
    resume.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    rows = matched_rows(args.manifest, args.results_root, args.run)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resuming = args.resume_suite is not None or args.resume_latest
    if resuming:
        work = (args.resume_suite.resolve() if args.resume_suite is not None else
                latest_suite(args.output_root, args.run, args.only))
    else:
        work = Path(tempfile.mkdtemp(prefix=f"{args.run}_{args.only}_", dir=args.output_root)).resolve()
    # Hold the lock until process exit, including all evaluator subprocesses.
    lock = (work / ".runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another runner is using {work}")
    valid = resume_plan(work, args.manifest, args.results_root, args.run, args.only, rows) if resuming else set()
    if resuming:
        print(f"Resume {work}: keep {len(valid)}/15; pending rows: {[r for r in rows if r not in valid]}", flush=True)
        if args.dry_run or len(valid) == 15:
            return
    manifest = work / "manifest.jsonl"
    if not resuming:
        manifest.write_bytes(args.manifest.read_bytes())
    summary = {"run": args.run, "evaluator": args.only, "rows": rows,
               "dry_run": args.dry_run, "cut3r_calibrated": False, "results": []}
    if not args.dry_run:
        env = "vbench" if args.only == "vbench-long" else "memcam"
        subprocess.run(["conda", "run", "--no-capture-output", "-n", env,
                        "python", "-c", cuda_check_code()], check=True, timeout=90)
        freeze = subprocess.check_output(["conda", "run", "--no-capture-output", "-n", env,
                                          "python", "-m", "pip", "freeze"], text=True)
        if resuming and sorted((work / "pip_freeze.txt").read_text().splitlines()) != sorted(freeze.splitlines()):
            raise ValueError("Evaluator environment changed; use a fresh suite rather than mixing versions")
        if not resuming:
            (work / "pip_freeze.txt").write_text(freeze)
    if resuming:
        (work / f"suite_status_before_resume_{uuid.uuid4().hex}.json").write_bytes(
            (work / "suite_status.json").read_bytes())
        summary["results"] = [{"row": row, "exit_code": 0, "output_root": str(work / f"row_{row:03d}")}
                              for row in rows if row in valid]
        save_status(work, summary)
    print(f"Matched-15 output: {work}", flush=True)
    for row in rows:
        if row in valid:
            print(f"[keep] validated row {row}", flush=True)
            continue
        if resuming:
            archive_row(work, row)
        output = work / f"row_{row:03d}"
        command = [sys.executable, str(REPO / "utils/smoke_video_metrics.py"),
                   "--manifest", str(manifest), "--results-root", str(args.results_root.resolve()),
                   "--run", args.run, "--row", str(row), "--only", args.only,
                   "--output-root", str(output), "--vbench-root", str(args.vbench_root.resolve())]
        if args.dry_run:
            command.append("--dry-run")
        code = subprocess.run(command).returncode
        summary["results"].append({"row": row, "exit_code": code, "output_root": str(output)})
        summary["results"].sort(key=lambda r: r["row"])
        save_status(work, summary)
    failures = [r["row"] for r in summary["results"] if r["exit_code"] != 0]
    print(f"Completed {15 - len(failures)}/15; failed rows: {failures}; results: {work}", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
