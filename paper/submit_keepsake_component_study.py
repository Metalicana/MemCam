"""Freeze and submit resumable one-H100 scene jobs, serially or as an array."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __name__ == "__main__":
    # Login-node limits apply before sbatch; inherited BLAS defaults can be 32+.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper import run_keepsake_component_study as study
from paper.keepsake_component_analysis import save
from paper.submit_paper_evidence import job_state

ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def accepted(command):
    response = subprocess.check_output(command, text=True).strip()
    job = response.split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError(f"Unexpected sbatch response {response!r}; inspect queue before retrying")
    return job


def scene_complete(plan, index):
    for setting in study.SETTINGS:
        folder = Path(plan["output"]) / "cells" / f"scene_{index:02d}" / setting[0]
        identity = study.identity(plan, index, setting)
        generated = study.receipt(folder / "generation.json", identity)
        measured = study.receipt(folder / "quality.json", {**identity,
            "generation_artifacts": generated["artifacts"]}) if generated else None
        if measured is None:
            return False
    return True


def array_task_states(tasks):
    """Use array IDs, not JobIDRaw's unrelated per-element numeric IDs."""
    expected = set(tasks)
    if not expected:
        return {}
    if any(not re.fullmatch(r"\d+_\d+", task) for task in expected):
        raise ValueError("Invalid recorded array task ID")
    live_text = subprocess.check_output(
        ["squeue", "--me", "--array", "--noheader", "--format=%i|%T"], text=True)

    def parse(raw):
        states = {}
        for line in raw.splitlines():
            fields = [f.strip() for f in line.split("|")]
            if len(fields) < 2 or fields[0] not in expected:
                continue
            state = fields[1].split()[0].rstrip("+")
            if fields[0] in states and states[fields[0]] != state:
                raise ValueError(f"Conflicting scheduler states for {fields[0]}")
            states[fields[0]] = state
        return states

    live, states = parse(live_text), {}
    for root in sorted({task.split("_")[0] for task in expected}):
        if all(task in live for task in expected if task.startswith(root + "_")):
            continue
        raw = subprocess.check_output(["sacct", "--array", "-j", root, "-X", "-n", "-P",
                                       "--format=JobID%64,State%32"], text=True)
        states.update(parse(raw))
    # A task can have been requeued since the last accounting update.
    states.update(live)
    if expected - states.keys():
        raise ValueError(f"Cannot establish array state for {sorted(expected-states.keys())}; "
                         "retry after accounting catches up, not with a new output directory")
    return states


def submit_array(plan, parallel):
    if parallel not in range(1, 6):
        raise ValueError("Array concurrency must be 1..5 one-H100 tasks")
    output, code = Path(plan["output"]), Path(plan["code"])
    if (output / "jobs.json").exists():
        raise ValueError("This study already has serial submissions. Do not duplicate them with an array; "
                         "inspect their state before changing scheduling.")
    launcher = code / "slurm/newton_keepsake_components_full.sbatch"
    if "SLURM_ARRAY_TASK_ID" not in launcher.read_text():
        raise ValueError("Frozen launcher predates array support. Preserve this study; "
                         "use serial submission or a fresh output directory, not an edited snapshot.")
    path = output / "array_jobs.json"
    jobs = study.load(path) if path.exists() else {"scenes": {}, "arrays": [], "reports": []}
    states = array_task_states(jobs["scenes"].values())
    active = {task for task, state in states.items() if state not in TERMINAL}
    report_state = job_state(jobs["reports"][-1]) if jobs["reports"] else None
    needed = []
    for index in range(15):
        old = jobs["scenes"].get(str(index))
        if old in active:
            continue
        if old and states[old] == "COMPLETED" and scene_complete(plan, index):
            continue
        needed.append(index)
    if needed and active:
        print(f"KEEP active array. Scenes {needed} need retry after it finishes; "
              "no overlapping second array submitted.", flush=True)
    elif needed:
        if report_state is not None and report_state not in TERMINAL:
            raise RuntimeError("The previous report is still active; retry when it exits")
        indices = "0-14" if needed == list(range(15)) else ",".join(map(str, needed))
        command = ["sbatch", "--parsable", "--chdir", str(code), f"--array={indices}%{parallel}",
            "--output", str(output / "scene_%A_%a.out"), "--error", str(output / "scene_%A_%a.err"),
            str(launcher), "--output", str(output)]
        root = accepted(command)
        for index in needed:
            jobs["scenes"][str(index)] = f"{root}_{index}"
        jobs["arrays"].append(dict(job=root, scenes=needed, parallel=parallel))
        save(path, jobs)
        active = {f"{root}_{i}" for i in needed}
        print(f"SUBMITTED array {root}: {indices}%{parallel}; ONE H100 per task", flush=True)

    dependencies = [jobs["scenes"][str(i)] for i in range(15)]
    if jobs["reports"] and jobs.get("report_dependencies") == dependencies:
        if report_state not in TERMINAL or report_state == "COMPLETED":
            print(f"KEEP report: {jobs['reports'][-1]}", flush=True)
            return jobs
    if report_state is not None and report_state not in TERMINAL:
        raise RuntimeError("Existing report is active; no duplicate report submitted")
    command = ["sbatch", "--parsable", "--chdir", str(code),
        "--output", str(output / "report_%j.out"), "--error", str(output / "report_%j.err")]
    if active:
        roots = sorted({task.split("_")[0] for task in active})
        command += ["--dependency=afterany:" + ":".join(roots)]
    command += [str(code / "slurm/newton_keepsake_components_report.sbatch"), "--output", str(output)]
    job = accepted(command)
    jobs["reports"].append(job)
    jobs["report_dependencies"] = dependencies
    save(path, jobs)
    print(f"SUBMITTED report: {job}", flush=True)
    return jobs


def submit(plan):
    output, code = Path(plan["output"]), Path(plan["code"])
    if (output / "array_jobs.json").exists():
        raise ValueError("This study uses an array; resume with --array-parallel 5")
    path = output / "jobs.json"
    jobs = json.loads(path.read_text()) if path.exists() else {"scenes": {}, "reports": []}
    existing_states = {job: job_state(job) for job in
                       list(jobs["scenes"].values()) + jobs["reports"][-1:]}
    retry_needed = any(existing_states[job] not in ACTIVE | {"COMPLETED"} for job in jobs["scenes"].values())
    if retry_needed and any(state in ACTIVE for state in existing_states.values()):
        raise RuntimeError("A retry is needed, but other study jobs are active. Let this chain finish, then resubmit.")
    previous = None
    for index in range(15):
        key = str(index)
        old = jobs["scenes"].get(key)
        if old:
            state = existing_states[old]
            if state in ACTIVE:
                previous = old
                print(f"KEEP scene {index}: {old} ({state})", flush=True)
                continue
            if state == "COMPLETED":
                # A successful scheduler exit alone is insufficient evidence.
                if not scene_complete(plan, index):
                    raise ValueError(f"Job {old} completed without valid artifacts; inspect scene {index}")
                print(f"KEEP complete scene {index}: {old}", flush=True)
                continue
        command = ["sbatch", "--parsable", "--chdir", str(code),
            "--output", str(output / f"scene{index:02d}_%j.out"),
            "--error", str(output / f"scene{index:02d}_%j.err")]
        if previous:
            command += [f"--dependency=afterany:{previous}"]
        command += [str(code / "slurm/newton_keepsake_components_full.sbatch"), "--output", str(output), "--scene", str(index)]
        job = accepted(command)
        jobs["scenes"][key] = job
        jobs.setdefault("attempts", []).append(dict(scene=index, job=job))
        save(path, jobs)
        previous = job
        print(f"SUBMITTED scene {index}: {job}", flush=True)
    dependencies = [jobs["scenes"][str(i)] for i in range(15)]
    if jobs["reports"] and jobs.get("report_dependencies") == dependencies:
        if existing_states[jobs["reports"][-1]] in ACTIVE | {"COMPLETED"}:
            print(f"KEEP report: {jobs['reports'][-1]}", flush=True)
            return jobs
    if jobs["reports"] and existing_states[jobs["reports"][-1]] in ACTIVE:
        raise RuntimeError("Existing report is active; do not submit a second report")
    command = ["sbatch", "--parsable", "--chdir", str(code), "--dependency=afterany:" + ":".join(dependencies),
        "--output", str(output / "report_%j.out"), "--error", str(output / "report_%j.err"),
        str(code / "slurm/newton_keepsake_components_report.sbatch"), "--output", str(output)]
    job = accepted(command)
    jobs["reports"].append(job)
    jobs["report_dependencies"] = dependencies
    save(path, jobs)
    print(f"SUBMITTED report: {job}", flush=True)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--detector", type=Path, default=Path.home() / "hf_cache/memcam_fvd/i3d_torchscript.pt")
    parser.add_argument("--output", type=Path, default=Path.home() / "memcam_results/keepsake_components_60s_n15")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--array-parallel", type=int, choices=range(1, 6),
                        help="Use an array with this many concurrent one-H100 tasks (recommended: 5)")
    args = parser.parse_args()
    if not args.prepare_only and shutil.which("sbatch") is None:
        parser.error("Run submission on Newton, where sbatch is available")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = study.prepare(args.output, args.manifest.resolve(), args.detector.resolve())
        if not args.prepare_only:
            if args.array_parallel is not None:
                submit_array(plan, args.array_parallel)
            else:
                submit(plan)
    scheduling = f"array capped at {args.array_parallel} one-H100 tasks" if args.array_parallel else "serial one-H100 jobs"
    print(f"Study: {args.output}\n75 videos; {scheduling}; prior 30s pilot is untouched.")


if __name__ == "__main__":
    main()
