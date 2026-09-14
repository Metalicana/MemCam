"""Plan, run, resume, and collect the matched MemCam 60s metric budget grid.

No video generation. FVD is always computed on a whole policy cohort.
Legacy quality reuse is opt-in and verifies recorded identities/settings,
not historical video hashes or evaluator environments.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
from statistics import mean
import subprocess
import sys
import tempfile

from audit_matched_metric_cohorts import RUNS, identity
from audit_metric_coverage import DIMENSIONS, bench_details, finite, valid_bench_score
from collect_matched_video_metrics import read_video
from run_matched_video_metrics import latest_suite, matched_rows, resume_plan
from smoke_video_metrics import REPO, cuda_check_code, select_video


EVALUATORS = ("quality", "vbench", "vbench-long", "cut3r")
QUALITY_CONFIG = {
    "LPIPS": {"frame_stride": 30, "learned_image_size": 224},
    "FVD": {"fvd_clip_length": 16, "fvd_clips_per_video": 4,
            "fvd_frame_stride": 4, "fvd_image_size": 224,
            "fvd_backend": "styleganv_i3d", "fvd_eps": 1e-6},
}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def quality_result(source, metric, run, expected, frames):
    source = Path(source)
    directory = source.parent
    summary = load(directory / "summary.json")
    config = summary.get("metric_config", {})
    if "max_frames" not in config or (config["max_frames"] not in (None, 0)
            and int(config["max_frames"]) < frames):
        raise ValueError("Unknown or truncated evaluation length")
    for key, value in QUALITY_CONFIG[metric].items():
        if config.get(key) != value:
            raise ValueError(f"{metric}: incompatible {key}={config.get(key)!r}; expected {value!r}")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if str(r.get("duration_sec")) == "60"]
    names = [identity(r)[1] for r in rows]
    if len(names) != 15 or set(names) != set(expected):
        raise ValueError("Quality results must contain exactly the canonical 15 videos")
    if any(identity(r)[0] != run or r.get("status") != "completed" for r in rows):
        raise ValueError("Mixed policies, failed or short videos in quality cohort")
    if metric == "LPIPS":
        if not all(finite(r.get("lpips_alex")) for r in rows):
            raise ValueError("Missing LPIPS values")
        value = mean(r["lpips_alex"] for r in rows)
    else:
        group = summary.get("by_duration", {}).get("60", {})
        if group.get("completed_or_short") != 15 or not finite(group.get("fvd")):
            raise ValueError("Missing or incorrectly grouped FVD")
        if group.get("fvd_clips") != 60:
            raise ValueError("FVD must use exactly 60 clips (4 per video)")
        if not group.get("fvd_detector_path"):
            raise ValueError("FVD resolved detector identity missing")
        value = group["fvd"]
    return {"metric": metric, "value": value, "n": 15, "source": str(source),
            "metric_config": config, "historical_hashes_verified": False}


def legacy_quality(audit, run, expected, frames):
    selected = {}
    for metric in QUALITY_CONFIG:
        candidates = audit.get("standard", {}).get(run, {}).get(metric, [])
        for candidate in sorted(candidates, key=lambda r: r["source"]):
            if candidate.get("state") != "EXACT" or set(candidate.get("covered", [])) != set(expected):
                continue
            try:
                result = quality_result(candidate["source"], metric, run, expected, frames)
                parent = Path(result["source"]).parent
                result["artifact_hashes"] = {str(parent / name): digest(parent / name)
                                             for name in ("summary.json", "metrics.jsonl")}
                selected[metric] = result
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return selected


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output already exists. Use submit/report on its plan.json to resume.")
    manifest = args.manifest.resolve()
    videos_root = args.root.resolve() / "context_memory_60s"
    sources, rows = {}, None
    for run in RUNS:
        current = matched_rows(manifest, videos_root, run)
        if rows is not None and current != rows:
            raise ValueError("Policy cohorts differ")
        rows = current
        sources[run] = [source_info(select_video(manifest, videos_root, run, row)[2]) for row in rows]
    items = [select_video(manifest, videos_root, "baseline", row)[1] for row in rows]
    expected = [item["output_prefix"] + "custom.mp4" for item in items]
    frames = max(int(item["num_frames"]) for item in items)
    audit = load(args.reuse_audit) if args.reuse_audit else {}
    if audit and set(audit.get("expected", [])) != set(expected):
        raise ValueError("Reuse audit has a different cohort")
    output.mkdir(parents=True)
    (output / "manifest.jsonl").write_bytes(manifest.read_bytes())
    plan = {"schema": 1, "root": str(args.root.resolve()), "output": str(output),
            "videos_root": str(videos_root), "manifest": str(output / "manifest.jsonl"),
            "manifest_sha256": digest(manifest),
            "vbench_root": str(args.vbench_root.resolve()), "rows": rows,
            "expected": expected, "frames": frames, "sources": sources,
            "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
            "reuse_audit": str(args.reuse_audit.resolve()) if args.reuse_audit else None,
            "cut3r_calibrated": False, "tasks": []}
    # Freeze local evaluator code across queued tasks, not just the launcher.
    plan["code_hashes"] = {str(p): digest(p) for p in sorted((REPO / "utils").glob("*.py"))}
    for run in RUNS:
        reuse = legacy_quality(audit, run, expected, frames) if audit else {}
        for evaluator in EVALUATORS:
            task = {"id": len(plan["tasks"]), "run": run, "evaluator": evaluator}
            if evaluator == "quality":
                task["legacy"] = reuse
            if evaluator in ("vbench-long", "cut3r") and not args.fresh_suites:
                try:
                    task["suite"] = str(latest_suite(args.root / "matched15_metrics_60s", run, evaluator))
                except ValueError:
                    pass
            plan["tasks"].append(task)
    save(output / "plan.json", plan)
    for task in plan["tasks"]:
        if set(task.get("legacy", {})) == set(QUALITY_CONFIG):
            work = task_dir(plan, task)
            work.mkdir()
            save(work / "selection.json", {m: r["source"] for m, r in task["legacy"].items()})
            save(work / "status.json", {**task, "status": "complete", "reuse": "legacy-recorded-identity/config",
                                        "historical_video_hashes_verified": False})
    print(f"Validated {len(RUNS)} configurations x 15 videos (extras excluded).")
    print(f"{len(plan['tasks'])} task slots: quality (LPIPS + FVD), VBench, Long, CUT3R.")
    reused = sum(len(t.get("legacy", {})) for t in plan["tasks"])
    print(f"Legacy LPIPS/FVD cells accepted by recorded identity/config: {reused}/42.")
    print("Legacy reuse does NOT verify historical video hashes or environments.")
    print(f"Plan: {output / 'plan.json'}")
    if args.submit:
        submit(output / "plan.json", args.parallel)


def task_dir(plan, task):
    return Path(plan["output"]) / f"task_{task['id']:03d}_{task['run']}_{task['evaluator']}"


def submit(path, parallel):
    plan = load(path)
    pending = []
    for task in plan["tasks"]:
        status_path = task_dir(plan, task) / "status.json"
        if not status_path.exists() or load(status_path).get("status") != "complete":
            pending.append(task["id"])
    if not pending:
        print("No pending tasks; run report to validate the completed grid.")
        return
    (REPO / "logs").mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "--chdir", str(REPO),
               "--array", ",".join(map(str, pending)) + f"%{parallel}",
               str(REPO / "slurm/newton_budget_metric_grid.sbatch"), str(Path(path).resolve())]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    print(f"Submitted {job}: {len(pending)} task slots, at most {parallel} concurrent GPUs.")
    report_job = subprocess.check_output([
        "sbatch", "--parsable", "--chdir", str(REPO), "--dependency", f"afterany:{job}",
        str(REPO / "slurm/newton_budget_metric_report.sbatch"), str(Path(path).resolve())
    ], text=True).strip()
    save(Path(plan["output"]) / f"submission_{job}.json", {"job": job, "report_job": report_job,
                                                           "task_ids": pending})
    print(f"CPU report job: {report_job}")


def check_sources(plan, run):
    if digest(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest changed; make a fresh plan")
    for original in plan["sources"][run]:
        if source_info(original["path"]) != original:
            raise ValueError(f"Video changed after planning: {original['path']}; make a fresh plan")
    for path, expected in plan["code_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Evaluator code changed after planning: {path}; make a fresh plan")


def verify_lengths(plan, task, work):
    manifest = Path(plan["manifest"])
    root = Path(plan["videos_root"])
    probes = []
    for row in plan["rows"]:
        _, item, video = select_video(manifest, root, task["run"], row)
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
            "-of", "json", str(video)], text=True)
        probe = json.loads(raw)
        if int(probe["streams"][0]["nb_read_frames"]) != int(item["num_frames"]):
            raise ValueError(f"Truncated/wrong-length video: {video}")
        probes.append({"video": str(video), "probe": probe})
    save(work / "video_probes.json", probes)


def conda(env, *command):
    return ["conda", "run", "--no-capture-output", "-n", env, *map(str, command)]


def freeze_environment(plan, env):
    root = Path(plan["output"])
    freeze = subprocess.check_output(conda(env, "python", "-m", "pip", "freeze"), text=True)
    runtime = subprocess.check_output(conda(env, "python", "-c",
        "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)"), text=True)
    fingerprint = {"pip": sorted(freeze.splitlines()), "runtime": runtime}
    if env == "vbench":
        vb = Path(plan["vbench_root"])
        fingerprint["vbench_sources"] = {str(p.relative_to(vb)): digest(p)
            for pattern in ("*.py", "vbench/**/*.py", "vbench2_beta_long/**/*.py",
                            "vbench2_beta_long/configs/*.yaml", "vbench2_beta_long/configs/*.json")
            for p in sorted(vb.glob(pattern))}
    with (root / f"environment_{env}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / f"environment_{env}.json"
        if path.exists() and load(path) != fingerprint:
            raise ValueError(f"{env} environment/code differs from earlier grid tasks; use a fresh plan")
        if not path.exists():
            save(path, fingerprint)
    return freeze


def stage(plan, task, work):
    folder = work / "input" / task["run"]
    folder.mkdir(parents=True, exist_ok=True)
    for source in plan["sources"][task["run"]]:
        path = Path(source["path"])
        dest = folder / path.name
        if dest.is_symlink() and dest.resolve() == path:
            continue
        dest.symlink_to(path)
    if sorted(p.name for p in folder.iterdir()) != sorted(plan["expected"]):
        raise ValueError("Unexpected staged inputs")
    return folder


def validate_bench(output, expected, run=None):
    files = list(Path(output).glob("*_eval_results.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one standard VBench result file")
    payload = load(files[0])
    results = []
    for dim in DIMENSIONS:
        details = bench_details(payload.get(dim))
        if details is None:
            raise ValueError(f"Missing {dim}")
        names = [Path(r["video_path"]).name for r in details]
        if run is not None and any(Path(r["video_path"]).parent.name != run for r in details):
            raise ValueError(f"{dim}: wrong policy input")
        if len(names) != 15 or set(names) != set(expected):
            raise ValueError(f"{dim}: different or duplicate cohort")
        if not all(valid_bench_score(dim, r.get("video_results")) for r in details):
            raise ValueError(f"{dim}: invalid values")
        scale = 100 if dim == "imaging_quality" else 1
        results.append({"metric": dim, "value": mean(float(r["video_results"]) / scale for r in details),
                        "n": 15, "source": str(files[0])})
    return results


def run_logged(command, work, cwd=REPO):
    print("+ " + shlex.join(map(str, command)), flush=True)
    with (work / "evaluate.log").open("a") as log:
        log.write("\n+ " + shlex.join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def suite_result(plan, task, suite):
    rows = plan["rows"]
    valid = resume_plan(Path(suite), Path(plan["manifest"]), Path(plan["videos_root"]),
                        task["run"], task["evaluator"], rows)
    if set(rows) != valid:
        raise ValueError(f"Only {len(valid)}/15 valid videos")
    videos = []
    for row in rows:
        item = select_video(Path(plan["manifest"]), Path(plan["videos_root"]), task["run"], row)[1]
        videos.append(read_video(Path(suite), task["run"], task["evaluator"], row, item))
    keys = DIMENSIONS if task["evaluator"] == "vbench-long" else (
        "rotation_error_deg_mean", "translation_error_scale_only_mean", "translation_error_sim3_mean")
    return [{"metric": key, "value": mean(v["scores"][key] for v in videos), "n": 15,
             "source": str(suite), "calibrated": False if task["evaluator"] == "cut3r" else None}
            for key in keys]


def validate_task(plan, task, work):
    selection = load(work / "selection.json")
    if task["evaluator"] == "quality":
        if set(selection) != set(QUALITY_CONFIG):
            raise ValueError("Both LPIPS and FVD must be complete")
        for record in task.get("legacy", {}).values():
            for path, expected in record["artifact_hashes"].items():
                if digest(path) != expected:
                    raise ValueError(f"Legacy artifact changed: {path}")
        return [quality_result(source, metric, task["run"], plan["expected"], plan["frames"])
                for metric, source in selection.items()]
    if task["evaluator"] == "vbench":
        return validate_bench(selection["output"], plan["expected"], task["run"])
    return suite_result(plan, task, selection["suite"])


def run_task(plan_path, task_id):
    plan = load(plan_path)
    if not 0 <= task_id < len(plan["tasks"]):
        raise ValueError("Invalid task ID")
    task = plan["tasks"][task_id]
    work = task_dir(plan, task)
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_sources(plan, task["run"])
        status_path = work / "status.json"
        if status_path.exists() and load(status_path).get("status") == "complete":
            validate_task(plan, task, work)
            print(f"KEEP: {task['run']} {task['evaluator']}")
            return
        status = {**task, "status": "running", "cut3r_calibrated": False,
                  "job": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")}
        save(status_path, status)
        try:
            evaluator = task["evaluator"]
            legacy = task.get("legacy", {})
            for record in legacy.values():
                for path, expected in record["artifact_hashes"].items():
                    if digest(path) != expected:
                        raise ValueError(f"Legacy artifact changed: {path}; make a fresh plan")
            needs_gpu = evaluator != "quality" or set(legacy) != set(QUALITY_CONFIG)
            if needs_gpu:
                verify_lengths(plan, task, work)
                env = "vbench" if evaluator in ("vbench", "vbench-long") else "memcam"
                subprocess.run(conda(env, "python", "-c", cuda_check_code()), check=True, timeout=90)
                freeze = freeze_environment(plan, env)
                (work / "pip_freeze.txt").write_text(freeze)
            if evaluator == "quality":
                selection = {metric: record["source"] for metric, record in legacy.items()}
                missing = [m for m in QUALITY_CONFIG if m not in selection]
                if missing:
                    attempt = Path(tempfile.mkdtemp(prefix="quality_", dir=work))
                    command = conda("memcam", "python", REPO / "utils/evaluate_context_memory_prefix_curves.py",
                        "--manifest", plan["manifest"], "--model_output_dir", Path(plan["videos_root"]) / task["run"],
                        "--metrics_dir", attempt, "--run_name", task["run"], "--source_duration", 60,
                        "--eval_durations", "60", "--rows", ",".join(map(str, plan["rows"])),
                        "--learned_metrics", ",".join(m.lower() for m in missing),
                        "--frame_stride", 30, "--learned_image_size", 224, "--metric_batch_size", 8,
                        "--fvd_clip_length", 16, "--fvd_clips_per_video", 4, "--fvd_frame_stride", 4,
                        "--fvd_image_size", 224, "--fvd_backend", "styleganv_i3d", "--fvd_eps", "1e-6",
                        "--fvd_cache_dir", Path.home() / "hf_cache/memcam_fvd", "--strict")
                    if plan["dataset_root"]:
                        command += ["--dataset_root", plan["dataset_root"]]
                    run_logged(command, work)
                    for metric in missing:
                        selection[metric] = str(attempt / task["run"] / "summary.json")
                save(work / "selection.json", selection)
            elif evaluator == "vbench":
                staged = stage(plan, task, work)
                attempt = Path(tempfile.mkdtemp(prefix="vbench_", dir=work))
                run_logged(conda("vbench", "python", Path(plan["vbench_root"]) / "evaluate.py",
                                 "--videos_path", staged, "--mode", "custom_input",
                                 "--output_path", attempt, "--dimension", *DIMENSIONS),
                           work, Path(plan["vbench_root"]))
                save(work / "selection.json", {"output": str(attempt)})
            else:
                suite = task.get("suite")
                if (work / "selection.json").exists():
                    suite = load(work / "selection.json")["suite"]
                if suite is None:
                    try:
                        suite = str(latest_suite(work, task["run"], evaluator))
                    except ValueError:
                        pass
                if suite and sorted((Path(suite) / "pip_freeze.txt").read_text().splitlines()) != sorted(freeze.splitlines()):
                    raise ValueError("Legacy suite environment differs. Prepare with --fresh-suites to avoid mixing versions.")
                command = [sys.executable, str(REPO / "utils/run_matched_video_metrics.py"),
                           "--run", task["run"], "--only", evaluator, "--manifest", plan["manifest"],
                           "--results-root", plan["videos_root"], "--output-root", str(work),
                           "--vbench-root", plan["vbench_root"]]
                if suite:
                    command += ["--resume-suite", suite]
                try:
                    run_logged(command, work)
                finally:
                    if suite is None:
                        try:
                            suite = str(latest_suite(work, task["run"], evaluator))
                        except ValueError:
                            pass
                    if suite:
                        save(work / "selection.json", {"suite": suite})
            scores = validate_task(plan, task, work)
            status.update(status="complete", scores=scores)
            save(status_path, status)
            print(f"DONE: {task['run']} {evaluator}; validated 15/15", flush=True)
        except Exception as exc:
            status.update(status="failed", error=str(exc), log=str(work / "evaluate.log"))
            save(status_path, status)
            raise


def report(plan_path):
    plan = load(plan_path)
    rows, scores = [], []
    for task in plan["tasks"]:
        work = task_dir(plan, task)
        state, error = "pending", ""
        if (work / "status.json").exists():
            status = load(work / "status.json")
            state, error = status["status"], status.get("error", "")
            if state == "complete":
                try:
                    check_sources(plan, task["run"])
                    values = validate_task(plan, task, work)
                    scores.extend({"run": task["run"], "evaluator": task["evaluator"], **v} for v in values)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    state, error = "invalid", str(exc)
        rows.append({"task": task["id"], "run": task["run"], "evaluator": task["evaluator"],
                     "status": state, "error": error, "log": str(work / "evaluate.log"),
                     "legacy_quality": ",".join(sorted(task.get("legacy", {})))})
    out = Path(plan["output"])
    save(out / "report.json", {"coverage": rows, "scores": scores, "cut3r_calibrated": False})
    with (out / "scores.csv").open("w", newline="") as handle:
        fields = ["run", "evaluator", "metric", "value", "n", "source", "calibrated"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores)
    with (out / "coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'RUN':30s} {'LPIPS/FVD':12s} {'VB6':12s} {'VBL6':12s} CUT3R")
    for run in RUNS:
        states = [r["status"] for r in rows if r["run"] == run]
        print(f"{run:30s} " + " ".join(f"{s:12s}" for s in states))
    print("CUT3R complete means execution/cohort complete, NOT camera calibration validated.")
    print(f"Full scores: {out / 'scores.csv'}\nErrors and logs: {out / 'coverage.csv'}")
    return any(r["status"] != "complete" for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--root", type=Path, default=Path.home() / "memcam_results")
    p.add_argument("--manifest", type=Path, default=REPO / "testbeds/context_memory/manifest.jsonl")
    p.add_argument("--vbench-root", type=Path, default=Path.home() / "VBench")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-audit", type=Path, help="Opt in to recorded-identity/config reuse of exact LPIPS/FVD files")
    p.add_argument("--fresh-suites", action="store_true", help="Do not resume historical Long/CUT3R suites")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--parallel", type=int, default=2)
    for action in ("submit", "run", "report"):
        sub = commands.add_parser(action)
        sub.add_argument("--plan", type=Path, required=True)
        if action == "run":
            sub.add_argument("--task", type=int, required=True)
        if action == "submit":
            sub.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    if hasattr(args, "parallel") and args.parallel < 1:
        parser.error("--parallel must be positive")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "submit":
        submit(args.plan, args.parallel)
    elif args.action == "run":
        run_task(args.plan, args.task)
    else:
        raise SystemExit(report(args.plan))


if __name__ == "__main__":
    main()
