"""One resumable KEEPSAKE-B32 run: native-style 90/360 round trips and metrics."""

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera_rotation_utils import exact_roundtrip_c2ws, roundtrip_pairs
from compare_fvd_matched import CONFIG as FVD_CONFIG, check_device, digest, save_json, verify_video
import evaluate_context_memory as metrics
from run_context_memory_batch import NEGATIVE_PROMPT, assert_video_writer_available


GENERATION = dict(policy="slam_covisibility", budget=32, geometry_weight=0.65,
                  seed=42, steps=50, cfg_scale=5.0, height=352, width=640, fps=30,
                  bank_device="cpu", pose_scale=100.0, reset_rng_each_video=True)
WEIGHTS = dict(
    dit_path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    text_encoder_path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    vae_path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    dit_ckpt_path="models/MemCam/dit_step20000.ckpt",
)
LIMITS = {
    "original_test_split_verified": False,
    "original_evaluator_verified": False,
    "claim": "Native-style round-trip reproduction, not verified reproduction of published table values.",
    "fvd": "Round-trip FVD compares generated outward clips with pose-aligned reversed return clips. "
           "It is not verified as the paper's FVD reference and is exported separately.",
}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_starts(args):
    path = args.starts_manifest or args.manifest
    rows = metrics.load_manifest(path)
    if args.starts_manifest:
        chosen = rows
        selection = "All supplied starting views; original split identity not independently verified"
    else:
        pool = [r for r in rows if r.get("duration_sec") == 60]
        if len(pool) != 15 or len({r["output_prefix"] for r in pool}) != 15:
            raise ValueError("Default selection requires the existing fifteen unique 60s manifest rows")
        chosen = random.Random(0).sample(sorted(pool, key=lambda r: r["output_prefix"]), 5)
        selection = "Five of our existing fifteen 60s starts; sorted output_prefix, random.Random(0)"
    if len(chosen) != 5 or len({r["scene"] for r in chosen}) != 5:
        raise ValueError("Expected five starting views from five distinct scenes")
    starts = []
    for r in sorted(chosen, key=lambda x: (x["scene"], x["start_frame"])):
        scene = r["scene"]
        if not scene or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in scene):
            raise ValueError(f"Invalid scene name: {scene!r}")
        start = r["start_frame"]
        if not isinstance(start, int) or start < 0 or not r["prompt"].strip():
            raise ValueError(f"Invalid starting view: {r}")
        image_path, pose_path = Path(r["input_image"]).resolve(), Path(r["pose_path"]).resolve()
        if args.dataset_root:
            image_path = args.dataset_root / "frames" / scene / f"{start:04d}.png"
            pose_path = args.dataset_root / "jsons" / f"{scene}.json"
        with Image.open(image_path) as image:
            image.verify()
        camera_data = json.loads(pose_path.read_text())["CineCameraActor"]
        keys = sorted(camera_data, key=int)
        if start >= len(keys) or int(keys[start]) != start:
            raise ValueError(f"Pose index is not the requested dataset frame: {pose_path}, {start}")
        camera = camera_data[keys[start]]
        for key in ("position", "rotation"):
            values = np.asarray(camera[key], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid camera {key}: {pose_path}")
        starts.append(dict(scene=scene, start_frame=start, prompt=r["prompt"], camera=camera,
                           input_image=str(image_path), pose_path=str(pose_path),
                           image_sha256=digest(image_path), pose_sha256=digest(pose_path)))
    return starts, dict(path=str(path.resolve()), sha256=digest(path), selection=selection)


def prepare(args):
    starts, source = select_starts(args)
    weights = {key: str((args.model_root / value).resolve()) for key, value in WEIGHTS.items()}
    for path in weights.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    print("Auditing checkpoint hashes and freezing the ten-video plan", flush=True)
    code = ["inference_memcam.py", "utils/camera_rotation_utils.py",
            "utils/run_keepsake_native_roundtrip.py", "utils/run_context_memory_batch.py",
            "dataset/poses.py", "diffsynth/pipelines/wan_video_memcam.py",
            "diffsynth/pipelines/memory_policies.py", "diffsynth/pipelines/memory_profiling.py",
            "diffsynth/models/wan_video_overlap.py"]
    plan = dict(version=1, source=source, starts=starts, generation=GENERATION,
                checkpoint_sha256={key: digest(path) for key, path in weights.items()},
                code_sha256={path: digest(ROOT / path) for path in code},
                tests={"90": 153, "360": 609}, limits=LIMITS)
    path = args.output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing plan differs in inputs/code/checkpoints; use a new output directory")
    save_json(path, plan)
    cases = [dict(id=f"{s['scene']}_{s['start_frame']:04d}_{angle}deg", start=s,
                  angle=angle, frames=153 if angle == 90 else 609, plan_sha256=signature(plan))
             for s in starts for angle in (90, 360)]
    for case in cases:
        print(f"  KEEPSAKE B32: {case['id']} ({case['frames']} frames)", flush=True)
    return plan, cases, weights


def load_completed(case, directory):
    path = directory / "generation.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("case") != case or record.get("status") != "complete":
        raise ValueError(f"Invalid generation receipt: {path}")
    expected = {f"frames/{i:04d}.png" for i in range(case["frames"])}
    expected.update({"video.mp4", "poses.npy", "profile.jsonl", "access.jsonl"})
    if set(record["files"]) != expected:
        raise ValueError(f"Incomplete generation receipt: {path}")
    attempt = directory / record["attempt"]
    for name, checksum in record["files"].items():
        if not (attempt / name).is_file() or digest(attempt / name) != checksum:
            raise ValueError(f"Saved output changed or missing: {attempt / name}; refusing silent reuse")
    return record


def profile_summary(path, case):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sections = [r for r in records if r["event"] == "section_profile"]
    summaries = [r for r in records if r["event"] == "rollout_summary" and r.get("completed")]
    count = (case["frames"] - 1) // 76
    if len(summaries) != 1 or [r["section_idx"] for r in sections] != list(range(count)):
        raise ValueError(f"Incomplete generation profile: {path}")
    if any(r.get("memory_policy") != "slam_covisibility" or r.get("memory_budget") != 32
           or not 1 <= r["stored_memory_size"] <= 32 for r in sections):
        raise ValueError(f"Incorrect policy or archive exceeds B32: {path}")
    if sections[-1]["section_end_frame"] != case["frames"] - 1:
        raise ValueError(f"Wrong final generated frame: {path}")
    return dict(stored_items=sections[-1]["stored_memory_size"],
                rollout_seconds=summaries[0]["rollout_latency_s"],
                peak_bank_frame_bytes=summaries[0]["peak_bank_frame_bytes"])


def generate_case(pipe, case, directory):
    import torch
    from dataset.poses import compute_c2w_matrix
    from diffsynth import save_video

    start = case["start"]
    base = compute_c2w_matrix(start["camera"], scale=GENERATION["pose_scale"])
    poses = exact_roundtrip_c2ws(base, case["angle"])
    pairs = roundtrip_pairs(case["frames"])
    if not np.array_equal(poses[[i for i, _ in pairs]], poses[[j for _, j in pairs]]):
        raise ValueError("Outward/return camera poses do not match")
    attempt = directory / f"attempt_{time.time_ns()}"
    (attempt / "frames").mkdir(parents=True)
    np.save(attempt / "poses.npy", poses)
    with Image.open(start["input_image"]) as source:
        initial = source.convert("RGB").resize((640, 352), Image.Resampling.BICUBIC)
    metadata = dict(run_name="keepsake_b32_native_roundtrip", scene=start["scene"],
                    dataset_start_frame=start["start_frame"], angle=case["angle"],
                    num_frames=case["frames"], duration_sec=(case["frames"] - 1) / 30,
                    output=str(attempt / "video.mp4"))
    # Overlap sampling uses the global Torch RNG, separately from diffusion noise.
    random.seed(GENERATION["seed"])
    np.random.seed(GENERATION["seed"])
    torch.manual_seed(GENERATION["seed"])
    frames = pipe(prompt=start["prompt"], negative_prompt=NEGATIVE_PROMPT,
                  input_image=initial, c2ws=poses, height=352, width=640,
                  cfg_scale=5.0, num_inference_steps=50, seed=42,
                  memory_policy="slam_covisibility", memory_budget=32,
                  keepsake_geometry_weight=0.65, memory_bank_device="cpu",
                  access_trace_path=str(attempt / "access.jsonl"), access_trace_metadata=metadata,
                  profile_path=str(attempt / "profile.jsonl"), profile_metadata=metadata, tiled=False)
    if len(frames) != case["frames"]:
        raise ValueError(f"Wrong generated length: {len(frames)} instead of {case['frames']}")
    for i, frame in enumerate(frames):
        frame = frame.convert("RGB")
        if frame.size != (640, 352):
            raise ValueError(f"Wrong generated resolution: {frame.size}")
        frame.save(attempt / "frames" / f"{i:04d}.png")
    save_video(frames, str(attempt / "video.mp4"), fps=30, quality=5)
    verify_video(attempt / "video.mp4", dict(num_frames=case["frames"], fps=30))
    resource = profile_summary(attempt / "profile.jsonl", case)
    files = {str(p.relative_to(attempt)): digest(p) for p in sorted(attempt.rglob("*")) if p.is_file()}
    receipt = dict(status="complete", case=case, attempt=attempt.name, files=files, resource=resource)
    save_json(directory / "generation.json", receipt)
    return receipt


def load_frames(attempt, count):
    frames = []
    for i in range(count):
        with Image.open(attempt / "frames" / f"{i:04d}.png") as image:
            frames.append(np.asarray(image.convert("RGB")))
    return frames


def score_pairs(frames, learned, batch_size=8):
    pairs = roundtrip_pairs(len(frames))
    rows = []
    for offset in range(0, len(pairs), batch_size):
        selected = pairs[offset:offset + batch_size]
        left, right = [frames[i] for i, _ in selected], [frames[j] for _, j in selected]
        perceptual = learned.compute_batch(left, right)
        if len(perceptual) != len(selected):
            raise ValueError("Incomplete LPIPS batch")
        for (i, j), a, b, learned_row in zip(selected, left, right, perceptual):
            values = metrics.frame_metrics(a, b)
            row = dict(outward_frame=i, return_frame=j, psnr_db=values["psnr_db"],
                       ssim=values["ssim"], lpips=learned_row["lpips_alex"])
            if not all(np.isfinite(row[name]) for name in ("psnr_db", "ssim", "lpips")):
                raise ValueError("Nonfinite paired-frame score")
            rows.append(row)
    return rows


def roundtrip_features(frames, runner):
    pairs = roundtrip_pairs(len(frames))
    starts = runner._sample_starts(len(pairs))
    if len(starts) != 4:
        raise ValueError("Expected four clips per round-trip leg")
    clip_pairs = [[pairs[start + k * runner.frame_stride] for k in range(runner.clip_length)]
                  for start in starts]
    features = []
    for side in (0, 1):
        clips = [np.stack([runner._to_clip_frame(frames[pair[side]]) for pair in clip])
                 for clip in clip_pairs]
        features.append(runner._encode_batch(clips))
    a, b = (np.asarray(f) for f in features)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 4 or not np.isfinite([a, b]).all():
        raise ValueError("Incomplete round-trip I3D features")
    return a, b, clip_pairs


def metric_models(args, device):
    if metrics.cv2 is None:
        raise RuntimeError("OpenCV is required for windowed SSIM; no global-SSIM fallback")
    print(f"Loading LPIPS and production I3D on {device}", flush=True)
    learned = metrics.LearnedMetricRunner(["lpips"], device=device, batch_size=8, image_size=None)
    fvd = metrics.FVDRunner(device=device, batch_size=4, cache_dir=args.fvd_cache_dir, **FVD_CONFIG)
    if str(learned.device) != device or str(fvd.device) != device:
        raise RuntimeError("Metric device changed unexpectedly")
    lpips_hash = hashlib.sha256()
    for name, value in sorted(learned.lpips_model.state_dict().items()):
        lpips_hash.update(name.encode())
        lpips_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    identity = dict(evaluator_sha256=digest(ROOT / "utils/evaluate_context_memory.py"),
                    lpips_weights_sha256=lpips_hash.hexdigest(), i3d_sha256=digest(fvd.resolved_detector_path),
                    opencv_version=metrics.cv2.__version__, torch_version=learned.torch.__version__,
                    psnr="RGB peak=255; mean frame dB; exact matches capped at 100 dB",
                    ssim="Luma; OpenCV Gaussian 11x11, sigma=1.5, production border handling",
                    lpips="AlexNet, full 640x352 resolution, RGB normalized to [-1,1]",
                    pixels="Lossless generated RGB PNG, not decoded MP4",
                    pairing="(i,N-1-i), i=1..N//2-1; exclude initial-frame pair and turnaround",
                    aggregation="Mean pairs within video; equal mean of five videos per angle",
                    fvd_config=FVD_CONFIG, limits=LIMITS)
    return learned, fvd, identity


def evaluate_case(case, directory, generation, learned, fvd, identity):
    receipt_path = directory / "metrics.json"
    key = dict(generation_sha256=signature(generation), metric_identity=identity)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("key") == key and all((directory / name).is_file() and digest(directory / name) == value
                                         for name, value in old["files"].items()):
            return old
    frames = load_frames(directory / generation["attempt"], case["frames"])
    pairs = score_pairs(frames, learned)
    a, b, clip_pairs = roundtrip_features(frames, fvd)
    write_csv(directory / "pairs.csv", pairs)
    np.savez(directory / "roundtrip_features.npz", outward=a, returning=b)
    summary = dict(case_id=case["id"], angle=case["angle"], scene=case["start"]["scene"],
                   start_frame=case["start"]["start_frame"], pairs=len(pairs),
                   **{name: float(np.mean([r[name] for r in pairs])) for name in ("psnr_db", "ssim", "lpips")},
                   **generation["resource"])
    result = dict(key=key, summary=summary, fvd_clip_pairs=clip_pairs,
                  files={name: digest(directory / name) for name in ("pairs.csv", "roundtrip_features.npz")})
    save_json(receipt_path, result)
    return result


def export_results(output, cases, results, fvd):
    if set(results) != {c["id"] for c in cases} or len(cases) != 10:
        raise ValueError("All ten outputs required; refusing a partial-cohort table")
    per_video = [results[c["id"]]["summary"] for c in cases]
    rows, fvd_rows = [], []
    for angle in (90, 360):
        group = [r for r in per_video if r["angle"] == angle]
        if len(group) != 5 or len({r["scene"] for r in group}) != 5:
            raise ValueError(f"Incomplete matched scene coverage for {angle} degrees")
        row = dict(method="MemCam + KEEPSAKE", budget=32, angle_deg=angle, videos=5,
                   frames=153 if angle == 90 else 609,
                   **{name: float(np.mean([r[name] for r in group])) for name in ("psnr_db", "ssim", "lpips")},
                   stored_items_min=min(r["stored_items"] for r in group),
                   stored_items_max=max(r["stored_items"] for r in group))
        features = []
        for r in group:
            with np.load(output / "cases" / r["case_id"] / "roundtrip_features.npz") as data:
                features.append((data["outward"], data["returning"]))
        value = float(fvd._frechet_distance(np.concatenate([x[0] for x in features]),
                                          np.concatenate([x[1] for x in features])))
        if not np.isfinite(value):
            raise ValueError("Nonfinite aggregate round-trip FVD")
        fvd_rows.append(dict(angle_deg=angle, videos=5, clips_per_leg=20, roundtrip_fvd=value,
                             reference="generated outward leg", compared="pose-aligned reversed return leg",
                             published_fvd_comparability="unverified"))
        rows.append(row)
    write_csv(output / "per_video.csv", per_video)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "roundtrip_fvd_diagnostic.csv", fvd_rows)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{KEEPSAKE B32 on five fixed starting views with native-style MemCam round trips. "
             r"Generated outward/return views are compared. Original test split and evaluator are unverified.}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Round trip & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Budget \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['angle_deg']}$^\\circ$ & {row['psnr_db']:.2f} & {row['ssim']:.3f} & "
                     f"{row['lpips']:.3f} & 32 " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "table.tex").write_text("\n".join(lines) + "\n")
    wide = "MemCam + KEEPSAKE (B32) & " + " & ".join(
        f"{r['psnr_db']:.2f} & {r['ssim']:.3f} & {r['lpips']:.3f}" for r in rows) + r" \\" + "\n"
    (output / "row.tex").write_text("% Measured reproduction; no published-FVD columns or cross-source winner bolding.\n" + wide)
    for row in rows:
        print(f"{row['angle_deg']} deg: PSNR={row['psnr_db']:.3f}, SSIM={row['ssim']:.4f}, "
              f"LPIPS={row['lpips']:.4f}, N=5", flush=True)


def execute(args):
    plan, cases, weights = prepare(args)
    if args.plan_only:
        print(f"Plan only: {args.output / 'plan.json'}; no generation or metrics launched", flush=True)
        return
    save_json(args.output / "status.json", dict(status="running", phase="setup", videos=10))
    records = {}
    coverage = {c["id"]: dict(case_id=c["id"], angle=c["angle"], status="pending") for c in cases}
    for case in cases:
        directory = args.output / "cases" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        record = load_completed(case, directory)
        if record:
            records[case["id"]] = record
            coverage[case["id"]]["status"] = "generated"
    write_csv(args.output / "coverage.csv", list(coverage.values()))
    print("Importing Torch; no import timeout and no GPU-mask changes", flush=True)
    import torch
    check_device(torch, "cuda", args.output)
    assert_video_writer_available(args.output)
    # Resolve metric dependencies/weights before expensive generation, using CPU memory.
    learned, fvd, identity = metric_models(args, "cpu")
    save_json(args.output / "metric_protocol.json", identity)
    del learned, fvd
    gc.collect()
    pending = [c for c in cases if c["id"] not in records]
    if pending:
        from inference_memcam import setup_pipeline
        print(f"Loading MemCam once for {len(pending)} pending KEEPSAKE-B32 videos", flush=True)
        pipe = setup_pipeline(**weights, device="cuda")
        for index, case in enumerate(pending, 1):
            save_json(args.output / "status.json", dict(status="running", phase="generation", case=case["id"]))
            print(f"[{index}/{len(pending)}] Generating {case['id']}", flush=True)
            records[case["id"]] = generate_case(pipe, case, args.output / "cases" / case["id"])
            coverage[case["id"]]["status"] = "generated"
            write_csv(args.output / "coverage.csv", list(coverage.values()))
            gc.collect()
            torch.cuda.empty_cache()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    learned, fvd, measured_identity = metric_models(args, "cuda")
    if identity != measured_identity:
        raise ValueError("Metric identity changed between setup and scoring")
    results = {}
    for case in cases:
        save_json(args.output / "status.json", dict(status="running", phase="metrics", case=case["id"]))
        print(f"Scoring {case['id']}", flush=True)
        results[case["id"]] = evaluate_case(case, args.output / "cases" / case["id"],
                                           records[case["id"]], learned, fvd, identity)
        coverage[case["id"]]["status"] = "complete"
        write_csv(args.output / "coverage.csv", list(coverage.values()))
    export_results(args.output, cases, results, fvd)
    save_json(args.output / "status.json", dict(status="complete", generated_videos=10,
                                               original_paper_comparability="unverified"))
    print(f"COMPLETE: {args.output / 'scores.csv'}; table.tex and row.tex", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "testbeds/context_memory/manifest.jsonl")
    parser.add_argument("--starts-manifest", type=Path, help="Five explicit scene/start/image/pose/prompt JSONL rows")
    parser.add_argument("--dataset-root", type=Path, help="Remap input images and pose JSONs to another dataset root")
    parser.add_argument("--model-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path.home() / "memcam_results/keepsake_native_roundtrip_b32")
    parser.add_argument("--fvd-cache-dir", type=Path, default=Path.home() / "hf_cache/memcam_fvd")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another job is already writing this output directory") from exc
        try:
            execute(args)
        except Exception as exc:
            save_json(args.output / "last_error.json", dict(error_type=type(exc).__name__, error=str(exc),
                                                           job=os.environ.get("SLURM_JOB_ID")))
            # Preserve previously completed results if a changed plan was rejected.
            status = args.output / "status.json"
            if status.exists():
                current = json.loads(status.read_text())
                if current.get("status") == "running":
                    save_json(status, dict(current, status="failed", error=str(exc)))
            raise


if __name__ == "__main__":
    main()
