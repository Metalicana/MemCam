"""Compare RI and KEEPSAKE FVD on the fixed expanded 60s cohort. No generation."""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate_context_memory import FVDRunner, load_manifest, output_path, resolve_gt_frames_dir


RUNS = {"RI": "ri_b32_dino_rgb", "KEEPSAKE": "slam_b32_covisibility"}
CONFIG = dict(clip_length=16, clips_per_video=4, frame_stride=4,
              image_size=224, backend="styleganv_i3d", eps=1e-6)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_device(torch, device, output):
    """Keep the actual CUDA initialization/kernel error instead of a boolean check."""
    record = dict(
        python=sys.executable, torch=str(torch.__version__), torch_path=str(torch.__file__),
        torch_cuda_build=torch.version.cuda, requested_device=device,
        environment={key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CONDA_PREFIX", "LD_LIBRARY_PATH",
            "PYTORCH_NVML_BASED_CUDA_CHECK")},
        status="checking",
    )
    path = output / "device_diagnostics.json"
    save_json(path, record)
    print("Device diagnostics: " + json.dumps(record), flush=True)
    if not device.startswith("cuda"):
        record["status"] = "explicit_non_cuda"
        save_json(path, record)
        return record
    try:
        print("Initializing CUDA", flush=True)
        torch.cuda.init()
        print(f"Checking tensor allocation and kernel on {device}", flush=True)
        value = torch.ones(1, device=device)
        value.mul_(2)
        torch.cuda.synchronize(device)
        if value.item() != 2:
            raise RuntimeError("CUDA kernel check returned an incorrect value")
        record.update(status="passed", device_count=torch.cuda.device_count(),
                      gpu_name=torch.cuda.get_device_name(device))
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save_json(path, record)
        save_json(output / "status.json", dict(status="failed", phase="cuda_setup",
                                               error=str(exc), diagnostics=str(path)))
        raise RuntimeError(
            f"CUDA setup failed before FVD: {type(exc).__name__}: {exc}. "
            f"Environment details: {path}. No CPU fallback or metric scoring was performed."
        ) from exc
    save_json(path, record)
    print(f"CUDA kernel OK: {record['gpu_name']}", flush=True)
    return record


def cohort_items(manifest, expected):
    items = [r for r in load_manifest(manifest) if r["duration_sec"] == 60]
    if len(items) != expected:
        raise ValueError(f"Expected {expected} 60s entries in {manifest}; found {len(items)}")
    names = [r["output_prefix"] for r in items]
    if len(set(names)) != expected:
        raise ValueError("Duplicate video identities in manifest")
    for item in items:
        if (Path(item["output_prefix"]).name != item["output_prefix"]
                or int(item["start_frame"]) < 0 or int(item["num_frames"]) < 61
                or Fraction(str(item["fps"])) <= 0):
            raise ValueError(f"Invalid cohort item: {item}")
    return items


def original_indices(items, original):
    by_name = {r["output_prefix"]: (i, r) for i, r in enumerate(items)}
    indices = []
    for old in original:
        match = by_name.get(old["output_prefix"])
        if match is None:
            raise ValueError(f"Original cohort item missing: {old['output_prefix']}")
        index, new = match
        for key in ("scene", "start_frame", "duration_sec", "num_frames", "fps", "prompt"):
            if new.get(key) != old.get(key):
                raise ValueError(f"Original cohort {key} mismatch: {old['output_prefix']}")
        indices.append(index)
    return indices


def clip_indices(item):
    # Use the production sampler, including its rounding and full-rollout span.
    sampler = object.__new__(FVDRunner)
    for key, value in CONFIG.items():
        setattr(sampler, key, value)
    starts = sampler._sample_starts(int(item["num_frames"]))
    if len(starts) != CONFIG["clips_per_video"]:
        raise ValueError(f"Insufficient distinct clips: {item['output_prefix']}")
    return [[s + k * CONFIG["frame_stride"] for k in range(CONFIG["clip_length"])]
            for s in starts]


def audit_inputs(items, root, dataset_root):
    missing = []
    for item in items:
        paths = [output_path(root / run, item) for run in RUNS.values()]
        gt = resolve_gt_frames_dir(item, dataset_root)
        paths += [gt / f"{int(item['start_frame']) + i:04d}.png"
                  for i in sorted({i for clip in clip_indices(item) for i in clip})]
        missing.extend(str(p) for p in paths if not p.is_file() or p.stat().st_size == 0)
    if missing:
        raise ValueError(f"Missing/empty inputs ({len(missing)}); no partial-cohort scoring:\n"
                         + "\n".join(missing[:30]))


def verify_video(path, item):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=nb_frames,nb_read_frames,avg_frame_rate,width,height", "-of", "json"]
    stream = json.loads(subprocess.check_output([*command, str(path)], text=True))["streams"][0]
    count = stream.get("nb_frames", "N/A")
    if count == "N/A":
        stream = json.loads(subprocess.check_output(
            [*command, "-count_frames", str(path)], text=True))["streams"][0]
        count = stream.get("nb_read_frames", "0")
    if (int(count) != int(item["num_frames"])
            or Fraction(stream["avg_frame_rate"]) != Fraction(str(item["fps"]))):
        raise ValueError(f"Wrong video length/FPS: {path}: {stream}")
    return stream


def feature_inputs(item, root, dataset_root, encoder):
    indices = clip_indices(item)
    gt = resolve_gt_frames_dir(item, dataset_root)
    gt_hashes = [(i, digest(gt / f"{int(item['start_frame']) + i:04d}.png"))
                 for i in sorted({i for clip in indices for i in clip})]
    return dict(item=item, config=CONFIG, encoder=encoder, clip_indices=indices,
                gt_dir=str(gt.resolve()), gt_sample_sha256=gt_hashes,
                videos={label: {"path": str(output_path(root / run, item).resolve()),
                                "sha256": digest(output_path(root / run, item))}
                        for label, run in RUNS.items()})


def validate_features(arrays):
    shapes = {array.shape for array in arrays.values()}
    if (set(arrays) != {"GT", *RUNS} or len(shapes) != 1
            or any(a.ndim != 2 or a.shape[0] != CONFIG["clips_per_video"]
                   or a.shape[1] < 1 or not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError("Invalid or incomplete per-video feature arrays")


def cached_features(path, inputs):
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    # JSON round-trip canonicalizes tuples in input signatures.
    if saved["inputs"] != json.loads(json.dumps(inputs)) or saved["sha256"] != digest(path):
        raise ValueError(f"Stale feature cache: {path}; choose a fresh output directory")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_features(arrays)
    return arrays


def extract_features(items, root, dataset_root, output, runner, encoder):
    folder = output / "features"
    folder.mkdir(exist_ok=True)
    collected = {key: [] for key in ("GT", *RUNS)}
    for number, item in enumerate(items, 1):
        print(f"[{number}/{len(items)}] {item['output_prefix']}", flush=True)
        inputs = feature_inputs(item, root, dataset_root, encoder)
        path = folder / (item["output_prefix"] + "i3d.npz")
        arrays = cached_features(path, inputs)
        if arrays is None:
            arrays, streams = {}, {}
            for label, run in RUNS.items():
                video = output_path(root / run, item)
                streams[label] = verify_video(video, item)
                gen, gt = runner._load_item_clips(item, root / run, dataset_root, None)
                if len(gen) != CONFIG["clips_per_video"] or len(gt) != len(gen):
                    raise ValueError(f"Short/undecodable video: {video}; expected four clips")
                batches = []
                runner._append_features(batches, gen)
                arrays[label] = np.concatenate(batches)
                if "GT" not in arrays:
                    batches = []
                    runner._append_features(batches, gt)
                    arrays["GT"] = np.concatenate(batches)
            validate_features(arrays)
            if inputs != feature_inputs(item, root, dataset_root, encoder):
                raise ValueError("Source changed during extraction")
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **arrays)
            temporary.replace(path)
            save_json(path.with_suffix(".json"), dict(inputs=inputs, streams=streams, sha256=digest(path)))
            print("  saved GT, RI and KEEPSAKE features", flush=True)
        else:
            print("  reused verified features", flush=True)
        for key in collected:
            collected[key].append(arrays[key])
    return {key: np.stack(value) for key, value in collected.items()}


def frechet_low_rank(real, generated):
    """Same sample-covariance FVD, evaluated by a smaller exact SVD (no PCA)."""
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if (real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]
            or min(len(real), len(generated)) < 2
            or not np.isfinite(real).all() or not np.isfinite(generated).all()):
        raise ValueError("FVD requires finite feature matrices with at least two samples")
    delta = real.mean(0) - generated.mean(0)
    a = (real - real.mean(0)) / np.sqrt(len(real) - 1)
    b = (generated - generated.mean(0)) / np.sqrt(len(generated) - 1)
    # If covariances are A^T A and B^T B, their fidelity is ||A B^T||_*.
    covariance_term = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    value = delta @ delta + np.sum(a * a) + np.sum(b * b) - 2 * covariance_term
    return float(max(0, value))


def bootstrap_indices(items, draws, seed):
    # Keep repeated seeds/starts from one scene together, and pair both policies and GT.
    scenes = sorted({item["scene"] for item in items})
    if len(scenes) < 2:
        raise ValueError("Need at least two scenes for cluster bootstrap")
    groups = [np.array([i for i, item in enumerate(items) if item["scene"] == scene])
              for scene in scenes]
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        yield np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])


def compare_features(features, items, draws=2000, seed=17):
    if draws < 1 or any(a.shape[0] != len(items) for a in features.values()):
        raise ValueError("Invalid bootstrap count or feature/cohort length")
    flattened = {key: array.reshape(-1, array.shape[-1]) for key, array in features.items()}
    # Point estimates use the existing evaluator; verify the fast bootstrap algebra against it.
    reference = object.__new__(FVDRunner)
    values = {}
    for label in RUNS:
        values[label] = reference._frechet_distance(flattened["GT"], flattened[label])
        fast = frechet_low_rank(flattened["GT"], flattened[label])
        if not np.isclose(fast, values[label], atol=1e-3, rtol=1e-6):
            raise ValueError(f"FVD numerical implementations disagree: {fast} vs {values[label]}")
    differences = []
    for indices in bootstrap_indices(items, draws, seed):
        sample = {key: array[indices].reshape(-1, array.shape[-1]) for key, array in features.items()}
        differences.append(frechet_low_rank(sample["GT"], sample["KEEPSAKE"])
                           - frechet_low_rank(sample["GT"], sample["RI"]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return dict(videos=len(items), scenes=len({i["scene"] for i in items}),
                clips=len(flattened["GT"]), ri_fvd=values["RI"], keepsake_fvd=values["KEEPSAKE"],
                difference=values["KEEPSAKE"] - values["RI"],
                difference_ci_low=float(low), difference_ci_high=float(high)), differences


def write_comparisons(features, items, original, output, draws, seed):
    cohorts = {"all30": list(range(len(items))), "original15": original,
               "additional15": [i for i in range(len(items)) if i not in original]}
    results = []
    for label, indices in cohorts.items():
        print(f"Scoring {label}; paired scene bootstrap ({draws} draws)", flush=True)
        selected = {key: array[indices] for key, array in features.items()}
        summary, differences = compare_features(selected, [items[i] for i in indices], draws, seed)
        results.append(dict(cohort=label, **summary))
        np.save(output / f"{label}_bootstrap_differences.npy", differences)
        print(f"{label}: RI={summary['ri_fvd']:.3f}; KEEPSAKE={summary['keepsake_fvd']:.3f}; "
              f"Keepsake-RI={summary['difference']:+.3f}; 95% interval "
              f"[{summary['difference_ci_low']:+.3f}, {summary['difference_ci_high']:+.3f}]", flush=True)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_json(output / "summary.json", dict(
        results=results, bootstrap_draws=draws, bootstrap_seed=seed,
        bootstrap_unit="scene; all videos, clips, both policies and paired GT kept together",
        interval="paired percentile bootstrap sensitivity interval; not a proof of equivalence",
        difference="KEEPSAKE minus RI; negative favors KEEPSAKE",
        note="Each FVD is recomputed on the entire named cohort, not averaged over video FVDs. "
             "Compare methods within a cohort; sample-size changes also affect FVD."))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest_60s_30.jsonl"))
    parser.add_argument("--original-manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--root", type=Path, default=Path.home() / "memcam_results/context_memory_60s")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--detector-path", type=Path)
    parser.add_argument("--fvd-cache-dir", type=Path, default=Path.home() / "hf_cache/memcam_fvd")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_draws < 1:
        parser.error("batch-size and bootstrap-draws must be positive")
    items = cohort_items(args.manifest, 30)
    original = original_indices(items, cohort_items(args.original_manifest, 15))
    audit_inputs(items, args.root, args.dataset_root)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "status.json", dict(status="audited" if args.audit_only else "running"))
    save_json(args.output / "cohort.json", dict(items=items, original_indices=original, runs=RUNS))
    print(f"Matched 30/30 videos per policy; {len(set(i['scene'] for i in items))} scenes; "
          "original 15 present. File/GT audit passed.", flush=True)
    if args.audit_only:
        return
    print(f"Importing torch with {sys.executable}", flush=True)
    import torch

    device_info = check_device(torch, args.device, args.output)
    print("Loading the production I3D detector", flush=True)
    runner = FVDRunner(device=args.device, batch_size=args.batch_size,
                       detector_path=args.detector_path, cache_dir=args.fvd_cache_dir, **CONFIG)
    encoder = dict(detector_sha256=digest(runner.resolved_detector_path),
                   evaluator_sha256=digest(Path(__file__).with_name("evaluate_context_memory.py")),
                   comparison_sha256=digest(Path(__file__)), torch=str(torch.__version__),
                   numpy=np.__version__, device=str(runner.device))
    save_json(args.output / "provenance.json", dict(
        manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
        original_manifest_sha256=digest(args.original_manifest), config=CONFIG, encoder=encoder,
        device_diagnostics=device_info,
        runs=RUNS, budget=32, duration_sec=60, videos_per_policy=30,
        identity="Fixed manifest filenames and GT mapping; per-video content hashes in feature receipts. "
                 "Does not independently certify historical generation checkpoint/settings."))
    features = extract_features(items, args.root, args.dataset_root, args.output, runner, encoder)
    write_comparisons(features, items, original, args.output, args.bootstrap_draws, args.seed)
    save_json(args.output / "status.json", dict(status="complete"))
    print(f"Results: {args.output / 'scores.csv'}", flush=True)


if __name__ == "__main__":
    main()
