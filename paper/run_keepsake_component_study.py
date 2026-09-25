"""Full matched 15-scene component study; independent resumable scene workers."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

if __name__ == "__main__":
    # Direct prepare/report calls also need the limit before importing NumPy.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
import numpy as np
from paper import finish_keepsake_pose_appearance as common
from paper import keepsake_component_analysis as analysis
from paper.run_keepsake_component_pilot import SETTINGS, LABELS, METRICS, validate_generated


def load(path):
    return json.loads(Path(path).read_text())


def check_hashes(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or analysis.sha(path) != expected:
            raise ValueError(f"Frozen artifact missing or changed: {path}")


def code_snapshot(output):
    code = output / "code"
    if code.exists():
        raise ValueError("An unfinished code snapshot exists; choose a fresh output directory")
    for folder in ("paper", "utils", "dataset", "diffsynth"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            target = code / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in (ROOT / "inference_memcam.py", ROOT / "slurm/newton_keepsake_components_full.sbatch",
                   ROOT / "slurm/newton_keepsake_components_report.sbatch"):
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (code / "models").symlink_to(ROOT / "models", target_is_directory=True)
    return code


def fvd_indices():
    starts = [int(round(x)) for x in np.linspace(0, 1825 - 61, 4)]
    return sorted({start + 4 * offset for start in starts for offset in range(16)})


def tokenizer_assets():
    folder = ROOT / "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    paths = [p for p in folder.rglob("*") if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"Missing production tokenizer assets: {folder}")
    return common.fingerprint(paths)


def prepare(output, manifest, detector):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = load(plan_path)
        if (plan["source_manifest"] != str(manifest) or plan["detector"] != str(detector)
                or plan["protocol"] != analysis.PROTOCOL):
            raise ValueError("Existing study has a different protocol or inputs")
        check_hashes(plan["code_hashes"])
        check_hashes(plan["shared_hashes"])
        for row in plan["scenes"]:
            check_hashes(row["input_hashes"])
        return plan
    from dataset.poses import load_c2ws_from_json
    items = common.cohort(manifest, 60)
    if len({r["scene"] for r in items}) != 15:
        raise ValueError("Expected 15 distinct scenes, not repeated trajectories from one scene")
    if not detector.is_file():
        raise FileNotFoundError(f"Production I3D must be cached before submission: {detector}")
    weights = {**common.generation_assets(), **tokenizer_assets()}
    scenes = []
    for index, original in enumerate(items):
        item = dict(original, source_row=original["_row"], _row=index)
        item["output_prefix"] = f"component60_row{index:02d}_{original['output_prefix']}"
        if not item.get("prompt"):
            raise ValueError(f"Missing prompt: {item['scene']}")
        pose = Path(item["pose_path"])
        start = int(item["start_frame"])
        keys = sorted(map(int, load(pose)["CineCameraActor"]))
        if keys[start:start+1825] != list(range(start, start+1825)):
            raise ValueError(f"Pose/dataset index mapping is not contiguous: {pose}")
        poses = np.asarray(load_c2ws_from_json(pose, start_frame=start, num_frames=1825))
        revisits = analysis.revisit_queries(poses)
        paths = [pose, Path(item["input_image"])]
        paths += [Path(item["gt_frames_dir"]) / f"{start+i:04d}.png"
                  for i in sorted(set(range(0, 1825, 30)) | set(fvd_indices()))]
        hashes = common.fingerprint(paths)
        poses_path = output / f"poses_{index:02d}.npy"
        np.save(poses_path, poses, allow_pickle=False)
        hashes[str(poses_path)] = analysis.sha(poses_path)
        scenes.append(dict(item=item, poses=str(poses_path), revisits=revisits, input_hashes=hashes))
        print(f"AUDIT {index+1}/15 {item['scene']}: {len(revisits)} pose-defined revisit queries", flush=True)
    frozen_manifest = output / "manifest.jsonl"
    frozen_manifest.write_text("".join(json.dumps(r["item"]) + "\n" for r in scenes))
    code = code_snapshot(output)
    hashes = {str(p): analysis.sha(p) for folder in ("paper", "utils", "dataset", "diffsynth", "slurm")
              for p in sorted((code / folder).rglob("*")) if p.is_file() and p.suffix in (".py", ".sbatch")}
    hashes[str(code / "inference_memcam.py")] = analysis.sha(code / "inference_memcam.py")
    plan = dict(schema=1, output=str(output), code=str(code), manifest=str(frozen_manifest),
                source_manifest=str(manifest), detector=str(detector), protocol=analysis.PROTOCOL,
                settings=[list(s) for s in SETTINGS], scenes=scenes, code_hashes=hashes,
                shared_hashes={**weights, **common.fingerprint([manifest, frozen_manifest, detector])},
                expected_cells=75, reuse_legacy=False)
    analysis.save(plan_path, plan)
    analysis.statistics.write_csv(output / "revisit_coverage.csv", [dict(scene_index=i,
        scene=r["item"]["scene"], queries=len(r["revisits"]), included_in_whole_rollout=True)
        for i, r in enumerate(scenes)])
    return plan


def receipt(path, inputs):
    if not path.exists():
        return None
    record = load(path)
    if record.get("status") != "complete" or record.get("inputs") != inputs or not record.get("artifacts"):
        raise ValueError(f"Invalid or incompatible receipt: {path}")
    check_hashes(record["artifacts"])
    return record


def run(command, log, timeout=None):
    """Terminate only our child process group on timeout or allocation termination."""
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    with log.open("ab") as handle:
        offset = common.start_log(handle, command)
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        started = time.monotonic()
        try:
            while process.poll() is None:
                try:
                    remaining = max(.01, timeout - (time.monotonic()-started)) if timeout is not None else 30
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"RUNNING {log}: {elapsed/60:.1f} min", flush=True)
                    if timeout is not None and elapsed >= timeout:
                        raise TimeoutError(f"Timeout after {timeout}s: {log}")
            if process.returncode:
                raise RuntimeError(f"Command failed ({process.returncode}): {log}\n{common.current_log_tail(log, offset)}")
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def identity(plan, scene, setting):
    return dict(plan_sha256=analysis.sha(Path(plan["output"]) / "plan.json"),
                scene_index=scene, setting=list(setting))


def generation_command(plan, index, setting, attempt):
    item = plan["scenes"][index]["item"]
    args = argparse.Namespace(duration=60, output=attempt)
    name, alpha, mode = setting
    cmd = common.generation_command(args, Path(plan["manifest"]), name, alpha, item)
    cmd.remove("--overwrite")
    return cmd + ["--keepsake_priority_mode", mode, "--memory_bank_device", "cpu"]


def execute_cell(plan, index, setting):
    name, alpha, mode = setting
    output = Path(plan["output"])
    work = output / "cells" / f"scene_{index:02d}" / name
    work.mkdir(parents=True, exist_ok=True)
    inputs = identity(plan, index, setting)
    item = plan["scenes"][index]["item"]
    generated = receipt(work / "generation.json", inputs)
    if generated is None:
        attempt = Path(tempfile.mkdtemp(prefix="generation_", dir=work))
        run(generation_command(plan, index, setting, attempt), attempt / "generation.log")
        source = attempt / "videos" / name
        artifacts = validate_generated(source, item, alpha, mode)
        generated = dict(status="complete", inputs=inputs, artifacts=artifacts, source=str(source))
        analysis.save(work / "generation.json", generated)
    measured_inputs = {**inputs, "generation_artifacts": generated["artifacts"]}
    measured = receipt(work / "quality.json", measured_inputs)
    if measured is None:
        attempt = Path(tempfile.mkdtemp(prefix="metrics_", dir=work))
        run(common.metric_command("memcam", "python", Path(__file__), "measure", "--output", output,
            "--scene", index, "--setting", name, "--source", generated["source"], "--attempt", attempt),
            attempt / "metrics.log")
        result = load(attempt / "result.json")
        # Validate every frame and scope before accepting the metric checkpoint.
        frame_rows = [json.loads(line) for line in (attempt / "frame_metrics.jsonl").read_text().splitlines()]
        if result["windows"] != analysis.reduce_frames(frame_rows, item, plan["scenes"][index]["revisits"]):
            raise ValueError("Metric checkpoint disagrees with per-frame results")
        with np.load(attempt / "fvd_features.npz", allow_pickle=False) as features:
            if (set(features.files) != {"gt", "generated"} or features["gt"].shape[0] != 4
                    or features["gt"].shape != features["generated"].shape
                    or not all(np.isfinite(features[k]).all() for k in features.files)):
                raise ValueError("Invalid FVD feature checkpoint")
        measured = dict(status="complete", inputs=measured_inputs, result=result, directory=str(attempt),
                        artifacts=common.fingerprint([attempt / n for n in
                            ("result.json", "frame_metrics.jsonl", "fvd_features.npz")]))
        analysis.save(work / "quality.json", measured)
    return generated, measured


def measure(plan, index, name, source, attempt):
    import torch
    import imageio.v2 as imageio
    from evaluate_context_memory import LearnedMetricRunner, FVDRunner, evaluate_video
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One allocated CUDA device required; no CPU fallback")
    item = plan["scenes"][index]["item"]
    runner = LearnedMetricRunner(["lpips"], device="cuda", batch_size=8, image_size=224)
    with (attempt / "frame_metrics.jsonl").open("w") as handle:
        metrics = evaluate_video(item, source, None, 30, None, handle, runner, METRICS)
    if metrics["status"] != "completed" or metrics["frames_seen"] != 1825:
        raise ValueError("Truncated metric input")
    frame_rows = [json.loads(line) for line in (attempt / "frame_metrics.jsonl").read_text().splitlines()]
    windows = analysis.reduce_frames(frame_rows, item, plan["scenes"][index]["revisits"])
    lpips_weights = hashlib.sha256()
    for key, value in sorted(runner.lpips_model.state_dict().items()):
        lpips_weights.update(key.encode())
        lpips_weights.update(value.detach().cpu().contiguous().numpy().tobytes())
    del runner
    torch.cuda.empty_cache()
    fvd = FVDRunner(device="cuda", batch_size=4, image_size=224, clip_length=16,
                    clips_per_video=4, frame_stride=4, backend="styleganv_i3d",
                    detector_path=Path(plan["detector"]), allow_download=False)
    generated, gt = fvd._load_item_clips(item, source, None, None)
    if len(generated) != 4 or len(gt) != 4:
        raise ValueError("FVD requires four complete clips per trajectory")
    np.savez(attempt / "fvd_features.npz", gt=fvd._encode_batch(gt), generated=fvd._encode_batch(generated))
    digest = hashlib.sha256()
    video = source / (item["output_prefix"] + "custom.mp4")
    with imageio.get_reader(str(video)) as reader:
        for i, frame in enumerate(reader):
            if i == 16:
                break
            digest.update(np.ascontiguousarray(frame).tobytes())
    analysis.save(attempt / "result.json", dict(windows=windows, lpips_weights_sha256=lpips_weights.hexdigest(),
        prefix16_decoded_sha256=digest.hexdigest(), prefix_note="Diagnostic only: compressed pixels before first eviction",
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, torch_cuda=torch.version.cuda,
        job=os.environ.get("SLURM_JOB_ID"), metric_frame_count=61, generated_metric_frame_count=60))


def model_preflight(plan, index):
    import torch
    from PIL import Image
    from diffsynth.pipelines.memory_policies import VisualMemoryFeatureExtractor
    from evaluate_context_memory import LearnedMetricRunner, FVDRunner
    from run_context_memory_batch import assert_video_writer_available
    torch.cuda.init()
    assert torch.cuda.device_count() == 1
    work = Path(plan["output"]) / "cells" / f"scene_{index:02d}"
    assert_video_writer_available(work)
    extractor = VisualMemoryFeatureExtractor(device="cuda", batch_size=1)
    features, _ = extractor.encode_pil_images([Image.new("RGB", (224, 224))])
    if not np.isfinite(features).all() or features.shape != (1, 768):
        raise ValueError("Production DINO smoke failed")
    weights = hashlib.sha256()
    for key, value in sorted(extractor.model.state_dict().items()):
        weights.update(key.encode())
        weights.update(value.detach().cpu().contiguous().numpy().tobytes())
    encoder = dict(model_sha256=weights.hexdigest(), processor=extractor.processor.to_dict(),
                   revision=getattr(extractor.model.config, "_commit_hash", None))
    with (Path(plan["output"]) / ".encoder.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        analysis.freeze(Path(plan["output"]) / "encoder.json", encoder)
    del extractor
    metric = LearnedMetricRunner(["lpips"], device="cuda", batch_size=1)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    if not np.isfinite(metric.compute_batch([frame], [frame])[0]["lpips_alex"]):
        raise ValueError("LPIPS smoke failed")
    del metric
    torch.cuda.empty_cache()
    detector = FVDRunner(device="cuda", batch_size=1, detector_path=Path(plan["detector"]), allow_download=False)
    encoded = detector._encode_batch([np.zeros((16, 3, 224, 224), dtype=np.uint8)])
    if encoded.shape[0] != 1 or not np.isfinite(encoded).all():
        raise ValueError("Production I3D smoke failed")
    print("DINO, LPIPS, I3D and video-writer preflight passed", flush=True)


def worker(plan, index):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the Slurm launcher, not a login-node GPU process")
    if index not in range(15):
        raise ValueError("Scene index must be 0..14")
    work = Path(plan["output"]) / "cells" / f"scene_{index:02d}"
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status="running", scene_index=index, completed_cells=0,
                     job=os.environ.get("SLURM_JOB_ID"), planned_cells=5)
        analysis.save(work / "status.json", state)
        try:
            check_hashes(plan["code_hashes"])
            check_hashes(plan["shared_hashes"])
            check_hashes(plan["scenes"][index]["input_hashes"])
            common.freeze_environment({"output": plan["output"]}, "memcam")
            run(common.metric_command("memcam", "python", "-c", common.cuda_check_code() +
                "; assert torch.cuda.device_count() == 1"), work / "preflight.log", timeout=180)
            run(common.metric_command("memcam", "python", Path(__file__), "preflight", "--output", plan["output"],
                "--scene", index), work / "model_preflight.log", timeout=900)
            # The real per-video metric process is isolated from generation to release GPU memory.
            for setting in SETTINGS:
                print(f"SCENE {index+1}/15: {setting[0]} / {plan['scenes'][index]['item']['scene']}", flush=True)
                execute_cell(plan, index, setting)
                state["completed_cells"] += 1
                analysis.save(work / "status.json", state)
            check_hashes(plan["scenes"][index]["input_hashes"])
            common.freeze_environment({"output": plan["output"]}, "memcam")
            state["status"] = "complete"
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            analysis.save(work / "status.json", state)


def report(plan):
    output = Path(plan["output"])
    check_hashes(plan["code_hashes"])
    for scene in plan["scenes"]:
        check_hashes(scene["input_hashes"])
    coverage, rows, diagnostics, prefix, features = [], [], [], [], {name: [] for name, _, _ in SETTINGS}
    features["gt"] = []
    lpips_hashes = set()
    for index, scene in enumerate(plan["scenes"]):
        loaded = {}
        for setting in SETTINGS:
            name = setting[0]
            work = output / "cells" / f"scene_{index:02d}" / name
            try:
                gen = receipt(work / "generation.json", identity(plan, index, setting))
                if gen is None:
                    raise ValueError("Missing complete generation receipt")
                quality = receipt(work / "quality.json", {**identity(plan, index, setting),
                                  "generation_artifacts": gen["artifacts"]})
                if quality is None:
                    raise ValueError("Missing complete metric receipt")
                directory = Path(quality["directory"])
                frame_rows = [json.loads(line) for line in (directory / "frame_metrics.jsonl").read_text().splitlines()]
                if (quality["result"] != load(directory / "result.json")
                        or quality["result"]["windows"] != analysis.reduce_frames(frame_rows, scene["item"], scene["revisits"])):
                    raise ValueError("Metric receipt disagrees with raw per-frame results")
                loaded[name] = (gen, quality)
                coverage.append(dict(scene_index=index, scene=scene["item"]["scene"], setting=name, status="complete", error=""))
            except (OSError, ValueError, KeyError) as exc:
                coverage.append(dict(scene_index=index, scene=scene["item"]["scene"], setting=name, status="incomplete", error=str(exc)))
        if len(loaded) != 5:
            continue
        poses = np.load(scene["poses"], allow_pickle=False)
        trace_name = scene["item"]["output_prefix"] + "custom.jsonl"
        full_trace = Path(loaded["full"][0]["source"]) / "access_traces" / trace_name
        gt_reference = None
        for name, _, _ in SETTINGS:
            gen, quality = loaded[name]
            lpips_hashes.add(quality["result"]["lpips_weights_sha256"])
            for row in quality["result"]["windows"]:
                rows.append(dict(scene_index=index, scene=scene["item"]["scene"], setting=name, **row))
            diagnostics.append(dict(scene_index=index, scene=scene["item"]["scene"], setting=name,
                **analysis.trace_comparison(full_trace, Path(gen["source"]) / "access_traces" / trace_name,
                                             poses, scene["revisits"])))
            prefix.append(dict(scene_index=index, setting=name,
                prefix16_equal_full=quality["result"]["prefix16_decoded_sha256"] ==
                                    loaded["full"][1]["result"]["prefix16_decoded_sha256"]))
            with np.load(Path(quality["directory"]) / "fvd_features.npz", allow_pickle=False) as f:
                gt = f["gt"].copy()
                if gt_reference is not None and not np.allclose(gt, gt_reference, atol=1e-5, rtol=1e-5):
                    raise ValueError("Same-scene GT I3D features differ across settings")
                if gt_reference is None:
                    gt_reference = gt
                features[name].append(f["generated"].copy())
        features["gt"].append(gt_reference)
    analysis.statistics.write_csv(output / "coverage.csv", coverage)
    if rows:
        analysis.statistics.write_csv(output / "partial_per_video.csv", rows)
    if any(r["status"] != "complete" for r in coverage):
        analysis.save(output / "status.json", dict(status="incomplete",
            completed_cells=sum(r["status"] == "complete" for r in coverage), expected_cells=75))
        raise ValueError("Full table withheld: missing cells; inspect coverage.csv and resume")
    if len(lpips_hashes) != 1:
        raise ValueError("LPIPS weights differ across study cells")
    summaries, contrasts = analysis.summarize(rows)
    for filename, values in (("per_video.csv", rows), ("summary.csv", summaries), ("contrasts.csv", contrasts),
                             ("trace_diagnostics.csv", diagnostics), ("prefix_pairing.csv", prefix)):
        analysis.statistics.write_csv(output / filename, values)
    fvd_summary, fvd_contrasts = analysis.summarize_fvd(features)
    analysis.statistics.write_csv(output / "fvd_summary.csv", fvd_summary)
    analysis.statistics.write_csv(output / "fvd_contrasts.csv", fvd_contrasts)
    scores = {r["setting"]: r for r in fvd_summary}
    ablation = []
    for name, alpha, mode in SETTINGS:
        row = dict(setting=LABELS[name], geometry_weight=alpha, priority_mode=mode,
                   videos=15, duration_sec=60, budget=32, fvd=scores[name]["fvd"],
                   fvd_ci_low=scores[name]["ci_low"], fvd_ci_high=scores[name]["ci_high"])
        for metric in METRICS:
            r = next(r for r in summaries if r["run"] == name and r["window"] == "all" and r["metric"] == metric)
            row.update({metric: r["mean"], metric+"_ci_low": r["ci_low"], metric+"_ci_high": r["ci_high"]})
        ablation.append(row)
    analysis.statistics.write_csv(output / "ablation.csv", ablation)
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Component ablation on fifteen matched 60-second MemCam trajectories, $B=32$.}",
             r"\label{tab:keepsake-components}", r"\begin{tabular}{lrrrr}\toprule",
             r"Variant & LPIPS $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & FVD $\downarrow$ \\", r"\midrule"]
    for r in ablation:
        lines.append(f"{r['setting']} & {r['lpips_alex']:.4f} & {r['psnr_db']:.2f} & {r['ssim']:.4f} & {r['fvd']:.1f} " + r"\\")
    (output / "ablation.tex").write_text("\n".join(lines + [r"\bottomrule\end{tabular}", r"\end{table}"]) + "\n")
    analysis.save(output / "status.json", dict(status="complete", completed_cells=75,
        prefix_mismatches=sum(not r["prefix16_equal_full"] for r in prefix),
        note="Inspect prefix_pairing.csv before attributing differences solely to eviction. FVD is a cohort point estimate, not mean per-video FVD."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "run", "measure", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--detector", type=Path, default=Path.home() / "hf_cache/memcam_fvd/i3d_torchscript.pt")
    parser.add_argument("--scene", type=int)
    parser.add_argument("--setting", choices=[s[0] for s in SETTINGS])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase == "prepare":
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args.output, args.manifest.resolve(), args.detector.resolve())
    else:
        plan = load(args.output / "plan.json")
        if args.phase == "run":
            worker(plan, args.scene)
        elif args.phase == "preflight":
            model_preflight(plan, args.scene)
        elif args.phase == "measure":
            measure(plan, args.scene, args.setting, args.source, args.attempt)
        else:
            with (args.output / ".report.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                analysis.save(args.output / "status.json", dict(status="running_report"))
                try:
                    report(plan)
                except BaseException as exc:
                    state = load(args.output / "status.json")
                    if state.get("status") != "incomplete":
                        state = dict(status="failed_report")
                    analysis.save(args.output / "status.json", {**state, "error": str(exc)})
                    raise
    print(f"{args.phase.upper()} COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
