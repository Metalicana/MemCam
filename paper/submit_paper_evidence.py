"""Freeze analysis code and submit existing-data evidence, without new generation."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def prepare(args):
    settings = {key: str(getattr(args, key).resolve()) for key in
                ("root", "manifest", "manifest_180", "scores", "cache", "fvd_cache")}
    settings.update(fvd_draws=args.fvd_draws, gpu_partition=args.gpu_partition,
                    gpu_type=args.gpu_type, after_job=args.after_job)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if any(plan.get(k) != v for k, v in settings.items()):
            raise ValueError("Existing evidence plan differs; use a new output directory")
        for path, expected in {**plan["code_hashes"], **plan["input_hashes"]}.items():
            if digest(path) != expected:
                raise ValueError(f"Frozen evidence input changed: {path}")
        return plan
    for path in (args.manifest, args.manifest_180, args.scores):
        if not path.is_file():
            raise FileNotFoundError(path)
    code = args.output / "code"
    if code.exists():
        raise ValueError("Incomplete snapshot already exists; use a fresh output directory")
    for folder in ("paper", "utils", "dataset", "diffsynth"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            target = code / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("newton_paper_evidence_cpu.sbatch", "newton_paper_evidence_gpu.sbatch"):
        target = code / "slurm" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "slurm" / name, target)
    inputs = {str(p.resolve()): digest(p) for p in (args.manifest, args.manifest_180, args.scores)}
    hashes = {str(p): digest(p) for p in sorted(code.rglob("*")) if p.is_file()}
    plan = dict(schema=1, **settings, code=str(code), output=str(args.output),
                code_hashes=hashes, input_hashes=inputs,
                scope="No generation. CPU evidence plus one GPU feature job AFTER the component job, then CPU FVD inference.")
    save(plan_path, plan)
    return plan


def job_state(job):
    result = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "--format=JobIDRaw,State"],
                            text=True, capture_output=True, check=True)
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    states = [r[1].split()[0].rstrip("+") for r in rows if r[0] == job]
    if len(states) != 1:
        raise ValueError(f"Cannot establish job {job} state; refusing a duplicate submission")
    return states[0]


def submit(plan):
    output, code = Path(plan["output"]), Path(plan["code"])
    submitted = output / "jobs.json"
    jobs = json.loads(submitted.read_text()) if submitted.exists() else {}
    arguments = []
    for key in ("root", "manifest", "manifest_180", "scores", "cache", "fvd_cache", "fvd_draws", "output"):
        arguments += ["--" + key.replace("_", "-"), str(plan[key])]
    active_or_done = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "COMPLETED", "REQUEUED"}
    dependencies_path = output / "score_dependencies.json"
    for phase in ("cpu", "gpu", "fvd-score"):
        desired_dependencies = [jobs.get("cpu"), jobs.get("gpu")]
        recorded_dependencies = json.loads(dependencies_path.read_text()) if dependencies_path.exists() else None
        upstream_changed = desired_dependencies != recorded_dependencies
        if phase in jobs:
            status = job_state(jobs[phase])
            if status in active_or_done:
                if phase == "fvd-score" and upstream_changed:
                    if status == "PENDING":
                        subprocess.run(["scontrol", "update", f"JobId={jobs[phase]}",
                                        f"Dependency=afterany:{jobs['cpu']}:{jobs['gpu']}"], check=True)
                        save(dependencies_path, desired_dependencies)
                    elif status != "COMPLETED":
                        raise RuntimeError("FVD scoring is active while upstream jobs were retried. "
                                           "No duplicate scoring submitted; rerun this command after it finishes.")
                if not (phase == "fvd-score" and upstream_changed and status == "COMPLETED"):
                    print(f"Existing {phase}: {jobs[phase]} ({status})", flush=True)
                    continue
        gpu = phase == "gpu"
        command = ["sbatch", "--parsable", "--chdir", str(code),
                   "--output", str(output / f"{phase}_%j.out"), "--error", str(output / f"{phase}_%j.err")]
        if gpu:
            command += [f"--partition={plan['gpu_partition']}", f"--gres=gpu:{plan['gpu_type']}:1",
                        f"--dependency=afterany:{plan['after_job']}"]
        elif phase == "fvd-score":
            command += [f"--dependency=afterany:{jobs['cpu']}:{jobs['gpu']}"]
        command += [str(code / "slurm" / f"newton_paper_evidence_{'gpu' if gpu else 'cpu'}.sbatch"),
                    "--phase", phase, *arguments]
        result = subprocess.check_output(command, text=True).strip()
        job = result.split(";")[0]
        if not re.fullmatch(r"\d+", job):
            raise ValueError(f"Unexpected sbatch response: {result!r}; inspect scheduler before retrying")
        jobs[phase] = job
        save(submitted, jobs)
        if phase == "fvd-score":
            save(dependencies_path, desired_dependencies)
        print(f"Submitted {phase}: {job}", flush=True)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", required=True, help="Existing component GPU job; never launches another component run")
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--manifest-180", type=Path, default=ROOT / "testbeds/context_memory_180s/manifest.jsonl")
    parser.add_argument("--scores", type=Path, default=Path.home() / "memcam_results/budget_metrics_60s_820776/scores.csv")
    parser.add_argument("--cache", type=Path, default=Path.home() / "memcam_results/context_memory_60s/gap_feature_cache_fresh")
    parser.add_argument("--fvd-cache", type=Path, default=Path.home() / "hf_cache/memcam_fvd")
    parser.add_argument("--fvd-draws", type=int, default=2000)
    parser.add_argument("--gpu-partition", default="highgpu")
    parser.add_argument("--gpu-type", default="nvidia_h100_80gb_hbm3")
    parser.add_argument("--output", type=Path, default=Path.home() / "memcam_results/paper_evidence_20260925")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+", args.after_job) or args.fvd_draws < 1:
        parser.error("Expected numeric dependency job and positive FVD draw count")
    if not args.prepare_only and not shutil.which("sbatch"):
        parser.error("Run this command on Newton; sbatch is unavailable here")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        if not args.prepare_only:
            submit(plan)
    print(f"Evidence output: {args.output}")


if __name__ == "__main__":
    main()
