"""Evaluate exactly the original 15 matched 60s videos with a tested evaluator.

Each video uses the smoke runner's isolation and validation. Failed videos do
not prevent remaining videos from running; any failure makes the task fail.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from smoke_video_metrics import REPO, select_video


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
    args = parser.parse_args()
    rows = matched_rows(args.manifest, args.results_root, args.run)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{args.run}_{args.only}_", dir=args.output_root)).resolve()
    manifest = work / "manifest.jsonl"
    manifest.write_bytes(args.manifest.read_bytes())
    summary = {"run": args.run, "evaluator": args.only, "rows": rows,
               "dry_run": args.dry_run, "cut3r_calibrated": False, "results": []}
    if not args.dry_run:
        env = "vbench" if args.only == "vbench-long" else "memcam"
        with (work / "pip_freeze.txt").open("w") as log:
            subprocess.run(["conda", "run", "--no-capture-output", "-n", env,
                            "python", "-m", "pip", "freeze"], stdout=log, check=True)
    print(f"Matched-15 output: {work}", flush=True)
    for row in rows:
        output = work / f"row_{row:03d}"
        command = [sys.executable, str(REPO / "utils/smoke_video_metrics.py"),
                   "--manifest", str(manifest), "--results-root", str(args.results_root.resolve()),
                   "--run", args.run, "--row", str(row), "--only", args.only,
                   "--output-root", str(output), "--vbench-root", str(args.vbench_root.resolve())]
        if args.dry_run:
            command.append("--dry-run")
        code = subprocess.run(command).returncode
        summary["results"].append({"row": row, "exit_code": code, "output_root": str(output)})
        (work / "suite_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    failures = [r["row"] for r in summary["results"] if r["exit_code"] != 0]
    print(f"Completed {15 - len(failures)}/15; failed rows: {failures}; results: {work}", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
