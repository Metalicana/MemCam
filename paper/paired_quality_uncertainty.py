"""Audit legacy means or estimate paired uncertainty from existing MemCam videos.

No generation. Legacy partial cohorts require explicit opt-in. Fresh extraction
requires the entire manifest cohort and caches one matched pair at a time.
"""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from compare_fvd_matched import digest, save_json, frechet_low_rank, verify_video, check_device
from evaluate_context_memory import FVDRunner, load_manifest, output_path, resolve_gt_frames_dir

LABELS = ("baseline", "keepsake")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def identity(row):
    return (row["scene"], int(row["start_frame"]), int(row["duration_sec"]),
            int(row.get("num_frames_expected", row.get("num_frames", 0))))


def grouping(items, mapping=None):
    if mapping is not None and any(item["scene"] not in mapping for item in items):
        raise ValueError("Cluster map must cover every scene; no guessed environment identities")
    labels = [str(mapping[i["scene"]] if mapping is not None else i["scene"]) for i in items]
    groups = [np.flatnonzero(np.array(labels) == label) for label in sorted(set(labels))]
    if len(groups) < 2:
        raise ValueError("Need at least two independent resampling groups")
    return labels, groups


def resamples(groups, draws, seed):
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        yield np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])


def paired_statistics(items, values, features=None, *, groups=None, draws=2000, seed=17):
    n = len(items)
    if draws < 1 or set(values) != set(LABELS):
        raise ValueError("Need positive draws and both policies")
    values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite or unpaired LPIPS values")
    if features is not None:
        if (set(features) != {"GT", *LABELS} or len({a.shape for a in features.values()}) != 1
                or any(a.ndim != 3 or a.shape[0] != n or min(a.shape[1:]) < 1
                       or not np.isfinite(a).all() for a in features.values())):
            raise ValueError("Need paired [trajectory, clip, feature] arrays for GT and both policies")
    groups = grouping(items)[1] if groups is None else groups
    if len(groups) < 2 or sorted(np.concatenate(groups).tolist()) != list(range(n)):
        raise ValueError("Groups must partition the cohort")

    def evaluate(ix):
        result = {"LPIPS": [float(values[k][ix].mean()) for k in LABELS]}
        if features is not None:
            flat = {k: a[ix].reshape(-1, a.shape[-1]) for k, a in features.items()}
            result["FVD"] = [frechet_low_rank(flat["GT"], flat[k]) for k in LABELS]
        return result

    points = evaluate(np.arange(n))
    samples = {metric: [] for metric in points}
    for ix in resamples(groups, draws, seed):
        for metric, scores in evaluate(ix).items():
            samples[metric].append(scores)
    contrasts = []
    for metric, point in points.items():
        samples[metric] = np.array(samples[metric])
        delta = samples[metric][:, 1] - samples[metric][:, 0]
        low, high = np.percentile(delta, [2.5, 97.5])
        contrasts.append(dict(metric=metric, videos=n, resampling_groups=len(groups),
            baseline=point[0], keepsake=point[1], difference=point[1]-point[0],
            ci_low=float(low), ci_high=float(high), draws=draws, seed=seed))
    leave_out = []
    for group in groups:
        for metric, scores in evaluate(np.setdiff1d(np.arange(n), group)).items():
            leave_out.append(dict(omitted_scenes=";".join(sorted({items[i]["scene"] for i in group})),
                metric=metric, remaining_videos=n-len(group), difference=scores[1]-scores[0]))
    return contrasts, leave_out, samples


def export(output, items, values, features, args, scope):
    mapping = json.loads(args.cluster_map.read_text()) if args.cluster_map else None
    labels, groups = grouping(items, mapping)
    rows, leave_out, samples = paired_statistics(items, values, features, groups=groups,
                                                draws=args.draws, seed=args.seed)
    for row in rows:
        row.update(duration_sec=args.duration, scope=scope)
    per_video = [dict(scene=item["scene"], start_frame=item["start_frame"], cluster=labels[j],
                      baseline_lpips=values["baseline"][j], keepsake_lpips=values["keepsake"][j],
                      difference=values["keepsake"][j]-values["baseline"][j]) for j, item in enumerate(items)]
    write_csv(output / "per_video.csv", per_video)
    write_csv(output / "contrasts.csv", rows)
    write_csv(output / "leave_one_group_out.csv", leave_out)
    np.savez(output / "bootstrap.npz", **samples)
    lines = [r"% Requires booktabs. Differences are KEEPSAKE minus unbounded; negative is better.",
             "% " + scope, r"\begin{tabular}{lrrr}", r"\toprule",
             r"\multicolumn{4}{l}{" + f"MemCam, {args.duration} s; $N={len(items)}$"
             + (" (partial cohort)" if scope.startswith("Partial") else "") + r"} \\",
             r"Metric & Unbounded & KEEPSAKE & Difference [95\% CI] \\", r"\midrule"]
    for row in rows:
        precision = 4 if row["metric"] == "LPIPS" else 1
        fmt = lambda x: f"{x:.{precision}f}"
        lines.append(f"{row['metric']} & {fmt(row['baseline'])} & {fmt(row['keepsake'])} & "
                     f"{fmt(row['difference'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "paired_uncertainty.tex").write_text("\n".join(lines) + "\n")
    save_json(output / "analysis.json", dict(scope=scope, videos=len(items), groups=labels,
        draws=args.draws, seed=args.seed, cluster_map=mapping,
        interval="Paired whole-group percentile bootstrap; all clips and policies stay together. "
                 "Equal-trajectory LPIPS mean and pooled FVD recomputed on every draw.",
        limitations="Approximate, conditional on existing outputs; not generation-seed variance, "
                    "not held-out generalization, not a correction for finite-sample FVD bias. "
                    "Without a cluster map, scene identifiers define independence; shared environments "
                    "need an explicit map. Leave-one-group-out results are sensitivity checks, not selection rules."))
    for row in rows:
        print(f"{row['metric']}: {row['baseline']:.6f} -> {row['keepsake']:.6f}; "
              f"delta {row['difference']:+.6f}, 95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]", flush=True)


def legacy(args):
    all_rows, configs, audit, sources = {}, {}, [], {}
    for label in LABELS:
        directory = getattr(args, label)
        summary_path, raw_path = directory / "summary.json", directory / "metrics.jsonl"
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if int(r["duration_sec"]) == args.duration]
        keyed = {identity(r): r for r in rows}
        if len(keyed) != len(rows) or len(rows) != args.expected_videos:
            raise ValueError(f"Wrong-size or duplicate intended cohort: {label}")
        all_rows[label] = keyed
        configs[label] = summary["metric_config"]
        completed = [r for r in rows if r["status"] == "completed"]
        block = summary["by_duration"][str(args.duration)]
        audit.append(dict(policy=label, intended_videos=len(rows), completed_videos=len(completed),
            reported_completed=block["completed_or_short"], reported_lpips=block["lpips_alex"],
            completed_lpips_mean=float(np.mean([r["lpips_alex"] for r in completed])) if completed else None,
            reported_fvd=block.get("fvd"), lpips_frame_stride=configs[label]["frame_stride"],
            fvd_clips_per_video=configs[label].get("fvd_clips_per_video"),
            fvd_frame_stride=configs[label].get("fvd_frame_stride"), source=str(summary_path)))
        sources.update({str(p): digest(p) for p in (summary_path, raw_path)})
    write_csv(args.output / "source_audit.csv", audit)
    save_json(args.output / "sources.json", sources)
    if set(all_rows["baseline"]) != set(all_rows["keepsake"]):
        raise ValueError("Intended cohorts differ; cannot silently intersect different evaluations")
    keys = sorted(all_rows["baseline"])
    matched = [k for k in keys if all(all_rows[p][k]["status"] == "completed" for p in LABELS)]
    write_csv(args.output / "coverage.csv", [dict(scene=k[0], start_frame=k[1], duration_sec=k[2],
        **{p: all_rows[p][k]["status"] for p in LABELS}, matched=k in matched) for k in keys])
    for key in ("frame_stride", "learned_image_size", "max_frames"):
        if key not in configs["baseline"] or configs["baseline"][key] != configs["keepsake"].get(key):
            raise ValueError(f"LPIPS protocol mismatch: {key}; audit exported, no paired interval")
    if configs["baseline"]["max_frames"] is not None:
        raise ValueError("Truncated metrics cannot be described as whole-rollout quality")
    if len(matched) != args.expected_videos and not args.allow_matched_subset:
        raise ValueError(f"Only {len(matched)}/{args.expected_videos} complete pairs. Audit exported; "
                         "use --allow-matched-subset only for an explicitly partial diagnostic.")
    for key in matched:
        for label in LABELS:
            row = all_rows[label][key]
            stride = int(configs[label]["frame_stride"])
            if (int(row["frames_seen"]) != key[3] or int(row["frame_stride"]) != stride
                    or int(row["frames_evaluated"]) != len(range(0, key[3], stride))):
                raise ValueError("Completed row has incompatible frame coverage")
    items = [dict(scene=k[0], start_frame=k[1]) for k in matched]
    values = {p: np.array([all_rows[p][k]["lpips_alex"] for k in matched]) for p in LABELS}
    scope = (f"Partial matched-cohort diagnostic ({len(matched)}/{args.expected_videos}); NOT the headline comparison"
             if len(matched) != args.expected_videos else "Matched legacy LPIPS; FVD requires clip features")
    export(args.output, items, values, None, args, scope)


def load_protocol(args):
    config = dict(clip_length=16, clips_per_video=args.clips, frame_stride=args.clip_stride,
                  image_size=224, backend="styleganv_i3d", eps=1e-6)
    if min(args.clips, args.clip_stride, args.lpips_stride) < 1:
        raise ValueError("Sampling strides and clip count must be positive")
    items = [i for i in load_manifest(args.manifest) if int(i["duration_sec"]) == args.duration]
    if len(items) != args.expected_videos or len({identity(i) for i in items}) != len(items):
        raise ValueError("Manifest must contain the full unique expected cohort")
    for item in items:
        prefix = item["output_prefix"]
        if Path(prefix).name != prefix or not prefix or int(item["start_frame"]) < 0:
            raise ValueError("Unsafe prefix or invalid start frame")
    return items, config


def sample_indices(item, config):
    sampler = object.__new__(FVDRunner)
    for key, value in config.items():
        setattr(sampler, key, value)
    starts = sampler._sample_starts(int(item["num_frames"]))
    if len(starts) != config["clips_per_video"]:
        raise ValueError("Insufficient distinct production clips")
    return sorted({s + j * config["frame_stride"] for s in starts for j in range(config["clip_length"])})


def extract(args):
    items, config = load_protocol(args)
    code = [Path(__file__), ROOT / "utils/evaluate_context_memory.py", ROOT / "utils/compare_fvd_matched.py"]
    plan = dict(items=items, duration_sec=args.duration, config=config, lpips_stride=args.lpips_stride,
                expected_videos=args.expected_videos, learned_image_size=224,
                baseline=str(args.baseline.resolve()), keepsake=str(args.keepsake.resolve()),
                detector_sha256=digest(args.detector), code_hashes={str(p): digest(p) for p in code},
                manifest_sha256=digest(args.manifest), scope="Fresh matched evaluation of existing videos; not generation")
    path = args.output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Extraction protocol or code changed; choose a new output directory")
    save_json(path, plan)
    # Fail before CUDA work if any policy or sampled GT input is absent.
    signatures = []
    for item in items:
        gt = resolve_gt_frames_dir(item, None)
        indices = sorted(set(sample_indices(item, config)) | set(range(0, int(item["num_frames"]), args.lpips_stride)))
        paths = [output_path(getattr(args, p), item) for p in LABELS]
        paths += [gt / f"{int(item['start_frame']) + i:04d}.png" for i in indices]
        signatures.append({str(p): digest(p) for p in paths})
        for policy in LABELS:
            stream = verify_video(output_path(getattr(args, policy), item), item)
            if (stream["width"], stream["height"]) != (640, 352):
                raise ValueError("Expected production 640x352 videos")
    import torch
    from evaluate_context_memory import LearnedMetricRunner, evaluate_video
    check_device(torch, "cuda", args.output)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one Slurm-allocated visible GPU")
    runtime = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0))
    runtime_path = args.output / "environment.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError("Runtime/hardware differs from earlier extraction; use a new output directory")
    save_json(runtime_path, runtime)
    learned = LearnedMetricRunner(["lpips"], device="cuda", batch_size=8, image_size=224)
    runner = FVDRunner(device="cuda", batch_size=4, detector_path=args.detector,
                       allow_download=False, **config)
    import hashlib
    lpips_hash = hashlib.sha256()
    for key, value in sorted(learned.lpips_model.state_dict().items()):
        lpips_hash.update(key.encode())
        lpips_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    cells = args.output / "cells"
    cells.mkdir(exist_ok=True)
    for index, item in enumerate(items):
        path = cells / f"pair_{index:03d}.json"
        feature_path = path.with_suffix(".npz")
        inputs = dict(plan_sha256=digest(args.output / "plan.json"), sources=signatures[index],
                      environment=runtime, lpips_sha256=lpips_hash.hexdigest())
        if path.exists():
            old = json.loads(path.read_text())
            if old["inputs"] != inputs or old["feature_sha256"] != digest(feature_path):
                raise ValueError(f"Stale pair cache: {path}")
            print(f"REUSE {index+1}/{len(items)} {item['scene']}", flush=True)
            continue
        scores, arrays = {}, {}
        for policy in LABELS:
            print(f"SCORE {index+1}/{len(items)} {item['scene']} {policy}", flush=True)
            row = evaluate_video(item, getattr(args, policy), None, args.lpips_stride, None,
                                 None, learned, ["lpips_alex"])
            if row["status"] != "completed" or not np.isfinite(row["lpips_alex"]):
                raise ValueError("Incomplete LPIPS evaluation")
            scores[policy] = row["lpips_alex"]
            clips, gt_clips = runner._load_item_clips(item, getattr(args, policy), None, None)
            if len(clips) != args.clips or len(gt_clips) != args.clips:
                raise ValueError("Incomplete FVD clips")
            batches = []
            runner._append_features(batches, clips)
            arrays[policy] = np.concatenate(batches)
            if "GT" not in arrays:
                batches = []
                runner._append_features(batches, gt_clips)
                arrays["GT"] = np.concatenate(batches)
        for source, expected in signatures[index].items():
            if digest(Path(source)) != expected:
                raise ValueError("Video/GT changed during evaluation")
        temporary = feature_path.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(feature_path)
        save_json(path, dict(inputs=inputs, item=item, scores=scores, feature_sha256=digest(feature_path)))
    for source, expected in plan["code_hashes"].items():
        if digest(Path(source)) != expected:
            raise ValueError("Analysis code changed during extraction")


def report(args):
    plan = json.loads((args.output / "plan.json").read_text())
    if args.duration != plan["duration_sec"] or args.expected_videos != plan["expected_videos"]:
        raise ValueError("Report duration/count differs from extraction plan")
    items = plan["items"]
    values, features = {k: [] for k in LABELS}, {k: [] for k in ("GT", *LABELS)}
    for index, item in enumerate(items):
        path = args.output / "cells" / f"pair_{index:03d}.json"
        pair = json.loads(path.read_text())
        if pair["item"] != item or pair["inputs"]["plan_sha256"] != digest(args.output / "plan.json"):
            raise ValueError("Wrong extraction receipt")
        if digest(path.with_suffix(".npz")) != pair["feature_sha256"]:
            raise ValueError("Feature checksum mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            if set(data.files) != set(features):
                raise ValueError("Wrong feature keys")
            for key in features:
                features[key].append(data[key])
        for key in values:
            values[key].append(pair["scores"][key])
    features = {k: np.stack(v) for k, v in features.items()}
    if features["GT"].shape[1] != plan["config"]["clips_per_video"]:
        raise ValueError("Clip count differs from extraction protocol")
    # Record numerical agreement with the original covariance-based evaluator.
    evaluator = object.__new__(FVDRunner)
    checks = []
    flat = {k: v.reshape(-1, v.shape[-1]) for k, v in features.items()}
    for key in LABELS:
        original = evaluator._frechet_distance(flat["GT"], flat[key])
        fast = frechet_low_rank(flat["GT"], flat[key])
        checks.append(dict(policy=key, covariance_fvd=original, low_rank_fvd=fast, difference=fast-original))
        if not np.isclose(original, fast, atol=.05, rtol=1e-6):
            raise ValueError("Numerical FVD implementations disagree materially")
    write_csv(args.output / "fvd_numerical_check.csv", checks)
    export(args.output, items, values, features, args,
           "Corrected full matched cohort; new point estimates, not intervals attached to unmatched historical means")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("legacy", "extract", "report"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--keepsake", type=Path)
    parser.add_argument("--duration", type=int, choices=(60, 180), required=True)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-matched-subset", action="store_true")
    parser.add_argument("--cluster-map", type=Path, help="JSON mapping exact scene IDs to independent environment IDs")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--detector", type=Path, default=Path.home() / "hf_cache/memcam_fvd/i3d_torchscript.pt")
    parser.add_argument("--lpips-stride", type=int)
    parser.add_argument("--clips", type=int)
    parser.add_argument("--clip-stride", type=int)
    args = parser.parse_args()
    if args.phase != "report" and (args.baseline is None or args.keepsake is None):
        parser.error("--baseline and --keepsake are required")
    if args.phase == "extract" and any(getattr(args, k) is None for k in ("manifest", "lpips_stride", "clips", "clip_stride")):
        parser.error("Extraction requires manifest and explicit LPIPS/clip sampling settings")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {"legacy": legacy, "extract": extract, "report": report}[args.phase](args)
        except Exception as exc:
            save_json(args.output / f"status_{args.phase}.json", dict(status="failed", error=str(exc)))
            raise
        save_json(args.output / f"status_{args.phase}.json", dict(status="complete", phase=args.phase))


if __name__ == "__main__":
    main()
