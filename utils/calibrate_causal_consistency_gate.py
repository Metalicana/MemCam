"""Validate a pose-calibrated causal consistency gate before deployment.

For each sampled generated target frame, this script recovers the historical
frame that MemCam actually used to condition generation. It measures DINO
agreement between the generated target and that generated context, then
subtracts the expected agreement for the same camera displacement. The
expectation and gate threshold are fitted on training trajectories only.

Exact-index GT is used only to fit the pose expectation and to evaluate whether
the score predicts candidate fidelity. It is never an input to the proposed
online score. The script also evaluates a clean-input-anchor control and checks
whether the context score collapses when its generated parent is corrupted.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.poses import load_c2ws_from_json  # noqa: E402
from utils.calibrate_frame_quality_estimators import (  # noqa: E402
    add_within_trajectory_labels,
    estimator_rows,
    estimator_run_rows,
    fmt,
    quality_auc,
    trajectory_split,
    write_csv,
)
from utils.evaluate_context_memory import frame_metrics  # noqa: E402


ESTIMATOR_FIELDS = (
    "context_raw_similarity",
    "context_pose_residual",
    "context_overlap_residual",
    "anchor_raw_similarity",
    "anchor_pose_residual",
    "context_anchor_mean_residual",
)


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_manifest(path, duration):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["duration_sec"]) != int(duration):
                continue
            item["_row"] = row_idx
            rows.append(item)
    return rows


def load_selected_contexts(path):
    selected = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "context_access" or not row.get("selected"):
                continue
            target = int(row["target_frame"])
            value = {
                "section_idx": int(row["section_idx"]),
                "target_frame": target,
                "context_frame": int(row["selected_memory_frame"]),
                "selected_overlap": float(row["selected_overlap"]),
            }
            previous = selected.get(target)
            if previous is not None and previous != value:
                raise RuntimeError(f"Conflicting trace selections for target {target}")
            selected[target] = value
    return selected


def sample_context_pairs(selected, frame_stride, max_samples=None):
    rows = [
        value
        for target, value in sorted(selected.items())
        if int(target) % int(frame_stride) == 0
    ]
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    return rows


def rotation_distance_rad(first, second):
    relative = np.asarray(first[:3, :3]).T @ np.asarray(second[:3, :3])
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def pose_components(c2ws, target_frame, reference_frame):
    target = np.asarray(c2ws[int(target_frame)], dtype=np.float64)
    reference = np.asarray(c2ws[int(reference_frame)], dtype=np.float64)
    return {
        "translation": float(np.linalg.norm(target[:3, 3] - reference[:3, 3])),
        "rotation_rad": rotation_distance_rad(reference, target),
    }


def read_video_frames(path, requested):
    import imageio.v2 as imageio

    requested = sorted(set(int(value) for value in requested))
    needed = set(requested)
    frames = {}
    reader = imageio.get_reader(str(path))
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx in needed:
                frames[frame_idx] = np.asarray(frame, dtype=np.uint8)
                if len(frames) == len(needed):
                    break
    finally:
        reader.close()
    missing = sorted(needed - set(frames))
    if missing:
        raise RuntimeError(f"Video {path} is missing requested frames: {missing[:10]}")
    return frames


def read_gt_frames(item, requested, reference_shape):
    height, width = reference_shape[:2]
    output = {}
    for local_frame in sorted(set(int(value) for value in requested)):
        dataset_frame = int(item["start_frame"]) + local_frame
        path = Path(item["gt_frames_dir"]) / f"{dataset_frame:04d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing GT frame: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), resample=Image.BICUBIC)
            output[local_frame] = np.asarray(image, dtype=np.uint8)
    return output


class DinoEncoder:
    def __init__(self, model_name, device, batch_size):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for DINO but torch reports no GPU")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    def encode(self, arrays):
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(arrays), self.batch_size):
                images = [
                    Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
                    for array in arrays[start : start + self.batch_size]
                ]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {
                    key: value.to(self.device) for key, value in inputs.items()
                }
                outputs = self.model(**inputs)
                batch = getattr(outputs, "pooler_output", None)
                if batch is None:
                    batch = outputs.last_hidden_state[:, 0]
                batch = self.torch.nn.functional.normalize(batch.float(), dim=-1)
                features.append(batch.cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)


def save_feature_cache(path, frame_indices, generated_features, gt_features, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            generated_features=np.asarray(generated_features, dtype=np.float32),
            gt_features=np.asarray(gt_features, dtype=np.float32),
            psnr_db=np.asarray([metrics[idx]["psnr_db"] for idx in frame_indices]),
            ssim=np.asarray([metrics[idx]["ssim"] for idx in frame_indices]),
        )
    temporary.replace(path)


def load_feature_cache(path):
    with np.load(path) as payload:
        frame_indices = payload["frame_indices"].astype(np.int64).tolist()
        generated = payload["generated_features"].astype(np.float32)
        gt = payload["gt_features"].astype(np.float32)
        psnr = payload["psnr_db"].astype(np.float64)
        ssim = payload["ssim"].astype(np.float64)
    return {
        int(frame): {
            "generated_feature": generated[index],
            "gt_feature": gt[index],
            "psnr_db": float(psnr[index]),
            "ssim": float(ssim[index]),
        }
        for index, frame in enumerate(frame_indices)
    }


def cosine(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def median_positive(values, fallback=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1e-9)]
    return float(np.median(values)) if len(values) else float(fallback)


def add_pose_distance(rows, prefix, scales=None):
    if scales is None:
        scales = {
            "translation": median_positive(
                [row[f"{prefix}_translation"] for row in rows]
            ),
            "rotation_rad": median_positive(
                [row[f"{prefix}_rotation_rad"] for row in rows]
            ),
        }
    for row in rows:
        row[f"{prefix}_pose_distance"] = (
            float(row[f"{prefix}_translation"]) / scales["translation"]
            + float(row[f"{prefix}_rotation_rad"]) / scales["rotation_rad"]
        )
    return scales


def fit_binned_expectation(rows, distance_field, value_field, bins):
    pairs = [
        (float(row[distance_field]), float(row[value_field]))
        for row in rows
        if row.get(distance_field) is not None and row.get(value_field) is not None
        and math.isfinite(float(row[distance_field]))
        and math.isfinite(float(row[value_field]))
    ]
    if len(pairs) < int(bins) * 2:
        raise ValueError(f"Too few rows to fit {bins} expectation bins")
    distances = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.unique(np.quantile(distances, quantiles)).tolist()
    assignments = np.searchsorted(edges, distances, side="right")
    global_mean = float(np.mean(values))
    means = []
    counts = []
    for bin_idx in range(len(edges) + 1):
        selected = values[assignments == bin_idx]
        means.append(float(np.mean(selected)) if len(selected) else global_mean)
        counts.append(int(len(selected)))
    return {
        "distance_field": distance_field,
        "value_field": value_field,
        "edges": [float(value) for value in edges],
        "means": means,
        "counts": counts,
    }


def expected_value(model, distance):
    bin_idx = int(np.searchsorted(model["edges"], float(distance), side="right"))
    return float(model["means"][bin_idx])


def add_parent_quality_labels(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["row"]))].append(row)
    for group in grouped.values():
        values = np.asarray(
            [row["context_generated_gt_similarity"] for row in group],
            dtype=np.float64,
        )
        threshold = float(np.median(values))
        for row, value in zip(group, values):
            row["context_parent_low_fidelity"] = bool(value <= threshold)


def summarize_parent_strata(rows, test_rows, estimator):
    output = {}
    for label, low_fidelity in (("low_fidelity_parent", True), ("high_fidelity_parent", False)):
        selected = [
            row for row in rows
            if int(row["row"]) in test_rows
            and bool(row["context_parent_low_fidelity"]) == low_fidelity
        ]
        output[label] = {
            "samples": len(selected),
            "quality_auc": quality_auc(
                [row[estimator] for row in selected],
                [row["gt_bad_frame"] for row in selected],
            ),
        }
    return output


def gate_decision(summary, raw_summary, parent_strata, args):
    reasons = []
    checks = {
        "auc": summary["test_quality_auc"] >= args.min_auc,
        "bad_precision": (
            summary["gate_test_bad_precision"] is not None
            and summary["gate_test_bad_precision"] >= args.min_bad_precision
        ),
        "bad_recall": summary["gate_test_bad_recall"] >= args.min_bad_recall,
        "clean_reject": (
            summary["gate_test_clean_false_reject_rate"]
            <= args.max_test_clean_false_reject
        ),
        "pose_calibration_gain": (
            summary["test_quality_auc"] - raw_summary["test_quality_auc"]
            >= args.min_pose_auc_gain
        ),
        "low_fidelity_parent_auc": (
            parent_strata["low_fidelity_parent"]["quality_auc"] is not None
            and parent_strata["low_fidelity_parent"]["quality_auc"]
            >= args.min_bad_parent_auc
        ),
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return {
        "decision": "INJECT" if all(checks.values()) else "DO_NOT_INJECT",
        "estimator": summary["estimator"],
        "checks": checks,
        "failed_checks": reasons,
    }


def markdown_expectation(model):
    lines = ["| bin | distance interval | train pairs | expected GT cosine |", "| ---: | --- | ---: | ---: |"]
    edges = [-math.inf] + list(model["edges"]) + [math.inf]
    for index, (count, mean) in enumerate(zip(model["counts"], model["means"])):
        low = "-inf" if not math.isfinite(edges[index]) else f"{edges[index]:.3f}"
        high = "inf" if not math.isfinite(edges[index + 1]) else f"{edges[index + 1]:.3f}"
        lines.append(f"| {index} | [{low}, {high}) | {count} | {mean:.4f} |")
    return lines


def write_report(path, summaries, parent_strata, decision, models, args, train_rows, test_rows, pair_count):
    lines = [
        "# Causal Consistency Gate Calibration",
        "",
        "## Question",
        "",
        "Can agreement with the frame that actually conditioned generation predict exact-index candidate fidelity after accounting for camera displacement?",
        "",
        "## Leakage Boundary",
        "",
        "GT is used only for training-trajectory pose expectations and held-out labels. The deployable score uses generated DINO features, causal trace identity, and known camera geometry. All thresholds remain fixed on held-out trajectories.",
        "",
        "## Protocol",
        "",
        f"- Runs: `{','.join(args.runs)}`.",
        f"- Pairs: `{pair_count}` sampled every `{args.frame_stride}` target frames.",
        f"- Train trajectories: `{','.join(map(str, sorted(train_rows)))}`.",
        f"- Held-out trajectories: `{','.join(map(str, sorted(test_rows)))}`.",
        f"- Bad candidate prevalence: bottom `{100 * args.bad_quantile:.0f}%` by within-run/trajectory PSNR--SSIM rank.",
        f"- Deployment threshold allows at most `{100 * args.max_train_clean_false_reject:.0f}%` clean rejection on training trajectories.",
        "- `context_pose_residual` is the proposed gate. Raw context similarity and clean-input-anchor scores are controls.",
        "",
        "## Held-Out Results",
        "",
        "| estimator | AUC | balanced | gate precision | gate recall | clean reject | within-trajectory rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['estimator']} | {fmt(row['test_quality_auc'])} | "
            f"{fmt(row['test_balanced_accuracy'])} | "
            f"{fmt(row['gate_test_bad_precision'])} | "
            f"{fmt(row['gate_test_bad_recall'])} | "
            f"{fmt(row['gate_test_clean_false_reject_rate'])} | "
            f"{fmt(row['test_mean_within_trajectory_spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Corrupted-Parent Stress Test",
            "",
            "The context frame is split at its within-trajectory median generated-to-GT DINO fidelity. This is diagnostic only; the split is not available online.",
            "",
            f"- AUC with low-fidelity parents: `{fmt(parent_strata['low_fidelity_parent']['quality_auc'])}` over `{parent_strata['low_fidelity_parent']['samples']}` pairs.",
            f"- AUC with high-fidelity parents: `{fmt(parent_strata['high_fidelity_parent']['quality_auc'])}` over `{parent_strata['high_fidelity_parent']['samples']}` pairs.",
            "",
            "## Pose Expectation Learned on Training Trajectories",
            "",
        ]
    )
    lines.extend(markdown_expectation(models["context_pose"]))
    lines.extend(
        [
            "",
            "## Injection Decision",
            "",
            f"**{decision['decision']}**",
            "",
            f"Failed checks: `{','.join(decision['failed_checks']) if decision['failed_checks'] else 'none'}`.",
            "",
            "The gate is injected only if the pose-calibrated context score clears every predeclared check: held-out AUC, rejection precision, bad-frame recall, clean-frame rejection, improvement over raw similarity, and robustness when the conditioning frame is itself corrupted.",
            "",
            "## Files",
            "",
            "- `pair_scores.csv`",
            "- `estimator_summary.csv`",
            "- `estimator_by_run.csv`",
            "- `calibration.json`",
            "- `decision.json`",
            "- `heldout_gate_validation.png`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path, summaries):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_causal_gate_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["estimator"].replace("_", " ") for row in summaries]
    aucs = [row["test_quality_auc"] for row in summaries]
    precisions = [row["gate_test_bad_precision"] or 0.0 for row in summaries]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(labels))))
    ax.barh(y - 0.18, aucs, height=0.34, color="#287271", label="held-out AUC")
    ax.barh(y + 0.18, precisions, height=0.34, color="#E76F51", label="gate precision")
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("score")
    ax.set_title("Causal consistency must predict fidelity before injection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=parse_list, default=parse_list("baseline,slam_b32_covisibility"))
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--frame_stride", type=int, default=76)
    parser.add_argument("--max_samples_per_video", type=int, default=None)
    parser.add_argument("--dino_model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pose_bins", type=int, default=4)
    parser.add_argument("--bad_quantile", type=float, default=0.20)
    parser.add_argument("--max_train_clean_false_reject", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--bootstrap_repeats", type=int, default=4000)
    parser.add_argument("--min_auc", type=float, default=0.70)
    parser.add_argument("--min_bad_precision", type=float, default=0.50)
    parser.add_argument("--min_bad_recall", type=float, default=0.20)
    parser.add_argument("--max_test_clean_false_reject", type=float, default=0.15)
    parser.add_argument("--min_pose_auc_gain", type=float, default=0.02)
    parser.add_argument("--min_bad_parent_auc", type=float, default=0.60)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride <= 0 or args.pose_bins < 2:
        parser.error("frame_stride must be positive and pose_bins must be at least 2")
    if not 0.0 < args.bad_quantile < 0.5:
        parser.error("bad_quantile must be between 0 and 0.5")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("test_fraction must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runs": args.runs,
        "duration": args.duration,
        "frame_stride": args.frame_stride,
        "max_samples_per_video": args.max_samples_per_video,
        "dino_model": args.dino_model,
    }
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("Output directory contains a different configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    items = load_manifest(args.manifest, args.duration)
    if not items:
        raise RuntimeError(f"No {args.duration}s manifest rows found")
    requested_rows = {int(item["_row"]) for item in items}
    train_rows, test_rows = trajectory_split(
        requested_rows, args.test_fraction, args.split_seed
    )

    encoder = None
    pair_rows = []
    for run_name in args.runs:
        for item in items:
            row_id = int(item["_row"])
            run_dir = args.root / run_name
            video_path = run_dir / f"{item['output_prefix']}custom.mp4"
            trace_path = run_dir / "access_traces" / f"{item['output_prefix']}custom.jsonl"
            if not video_path.is_file() or not trace_path.is_file():
                message = f"Missing video or trace for {run_name} row {row_id}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue
            selected = load_selected_contexts(trace_path)
            pairs = sample_context_pairs(
                selected, args.frame_stride, args.max_samples_per_video
            )
            if not pairs:
                message = f"No sampled context pairs for {run_name} row {row_id}"
                if args.strict:
                    raise RuntimeError(message)
                print(f"[skip] {message}")
                continue

            requested = {0}
            for pair in pairs:
                requested.add(int(pair["target_frame"]))
                requested.add(int(pair["context_frame"]))
            cache_path = (
                args.output_dir / "feature_cache" / run_name / f"row_{row_id:03d}.npz"
            )
            if cache_path.is_file():
                feature_rows = load_feature_cache(cache_path)
                if not requested <= set(feature_rows):
                    raise RuntimeError(f"Incomplete feature cache: {cache_path}")
            else:
                generated = read_video_frames(video_path, requested)
                reference_shape = generated[min(generated)].shape
                gt = read_gt_frames(item, requested, reference_shape)
                frame_indices = sorted(requested)
                if encoder is None:
                    encoder = DinoEncoder(args.dino_model, args.device, args.batch_size)
                arrays = [generated[idx] for idx in frame_indices] + [gt[idx] for idx in frame_indices]
                features = encoder.encode(arrays)
                split = len(frame_indices)
                metrics = {
                    idx: frame_metrics(generated[idx], gt[idx]) for idx in frame_indices
                }
                save_feature_cache(
                    cache_path,
                    frame_indices,
                    features[:split],
                    features[split:],
                    metrics,
                )
                feature_rows = load_feature_cache(cache_path)
                print(f"[features] {run_name} row {row_id}: {len(frame_indices)} frames")

            c2ws = load_c2ws_from_json(
                item["pose_path"],
                start_frame=int(item["start_frame"]),
                num_frames=int(item["num_frames"]),
            )
            for pair in pairs:
                target = int(pair["target_frame"])
                context = int(pair["context_frame"])
                target_features = feature_rows[target]
                context_features = feature_rows[context]
                anchor_features = feature_rows[0]
                context_pose = pose_components(c2ws, target, context)
                anchor_pose = pose_components(c2ws, target, 0)
                pair_rows.append(
                    {
                        "run_name": run_name,
                        "row": row_id,
                        "scene": item["scene"],
                        "start_frame": int(item["start_frame"]),
                        "duration_sec": int(item["duration_sec"]),
                        "section_idx": int(pair["section_idx"]),
                        "target_frame": target,
                        "context_frame": context,
                        "selected_overlap": float(pair["selected_overlap"]),
                        "psnr_db": target_features["psnr_db"],
                        "ssim": target_features["ssim"],
                        "target_generated_gt_similarity": cosine(
                            target_features["generated_feature"],
                            target_features["gt_feature"],
                        ),
                        "context_generated_gt_similarity": cosine(
                            context_features["generated_feature"],
                            context_features["gt_feature"],
                        ),
                        "context_raw_similarity": cosine(
                            target_features["generated_feature"],
                            context_features["generated_feature"],
                        ),
                        "context_gt_similarity": cosine(
                            target_features["gt_feature"],
                            context_features["gt_feature"],
                        ),
                        "context_translation": context_pose["translation"],
                        "context_rotation_rad": context_pose["rotation_rad"],
                        "context_overlap_distance": 1.0 - float(pair["selected_overlap"]),
                        "anchor_raw_similarity": cosine(
                            target_features["generated_feature"],
                            anchor_features["generated_feature"],
                        ),
                        "anchor_gt_similarity": cosine(
                            target_features["gt_feature"],
                            anchor_features["gt_feature"],
                        ),
                        "anchor_translation": anchor_pose["translation"],
                        "anchor_rotation_rad": anchor_pose["rotation_rad"],
                    }
                )

    if not pair_rows:
        raise RuntimeError("No causal context pairs were produced")
    add_within_trajectory_labels(pair_rows, args.bad_quantile)
    add_parent_quality_labels(pair_rows)
    training = [row for row in pair_rows if int(row["row"]) in train_rows]

    context_scales = add_pose_distance(training, "context")
    add_pose_distance(pair_rows, "context", context_scales)
    anchor_scales = add_pose_distance(training, "anchor")
    add_pose_distance(pair_rows, "anchor", anchor_scales)

    models = {
        "context_pose": fit_binned_expectation(
            training, "context_pose_distance", "context_gt_similarity", args.pose_bins
        ),
        "context_overlap": fit_binned_expectation(
            training, "context_overlap_distance", "context_gt_similarity", args.pose_bins
        ),
        "anchor_pose": fit_binned_expectation(
            training, "anchor_pose_distance", "anchor_gt_similarity", args.pose_bins
        ),
    }
    for row in pair_rows:
        row["context_pose_residual"] = row["context_raw_similarity"] - expected_value(
            models["context_pose"], row["context_pose_distance"]
        )
        row["context_overlap_residual"] = row["context_raw_similarity"] - expected_value(
            models["context_overlap"], row["context_overlap_distance"]
        )
        row["anchor_pose_residual"] = row["anchor_raw_similarity"] - expected_value(
            models["anchor_pose"], row["anchor_pose_distance"]
        )
        row["context_anchor_mean_residual"] = 0.5 * (
            row["context_pose_residual"] + row["anchor_pose_residual"]
        )

    summaries = estimator_rows(
        pair_rows,
        ESTIMATOR_FIELDS,
        train_rows,
        test_rows,
        args.bootstrap_repeats,
        args.split_seed,
        args.max_train_clean_false_reject,
    )
    run_summaries = estimator_run_rows(pair_rows, summaries, test_rows)
    summary_by_name = {row["estimator"]: row for row in summaries}
    proposed = summary_by_name["context_pose_residual"]
    raw = summary_by_name["context_raw_similarity"]
    parent_strata = summarize_parent_strata(
        pair_rows, test_rows, "context_pose_residual"
    )
    decision = gate_decision(proposed, raw, parent_strata, args)

    calibration = {
        "context_scales": context_scales,
        "anchor_scales": anchor_scales,
        "models": models,
        "train_rows": sorted(train_rows),
        "test_rows": sorted(test_rows),
    }
    write_csv(args.output_dir / "pair_scores.csv", pair_rows)
    write_csv(args.output_dir / "estimator_summary.csv", summaries)
    write_csv(args.output_dir / "estimator_by_run.csv", run_summaries)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "report.md",
        summaries,
        parent_strata,
        decision,
        models,
        args,
        train_rows,
        test_rows,
        len(pair_rows),
    )
    make_plot(args.output_dir / "heldout_gate_validation.png", summaries)

    print(f"Pairs: {len(pair_rows)}")
    print(f"Decision: {decision['decision']}")
    print(f"Failed checks: {','.join(decision['failed_checks']) or 'none'}")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
