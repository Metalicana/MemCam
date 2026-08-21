"""Validate causal, no-reference frame-quality estimators against exact GT.

The estimators only receive a generated frame. PSNR and SSIM against the
dataset frame at the same trajectory index are evaluation labels and are never
available to the estimator or to the eventual online memory policy.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_context_memory import frame_metrics  # noqa: E402


CPU_ESTIMATORS = (
    "sharpness_contrast",
    "laplacian_variance",
    "gradient_energy",
    "contrast",
    "entropy",
    "unclipped_fraction",
)


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


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


def read_json(path, default):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_jsonl_by_key(path):
    rows = {}
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["run_name"], int(row["row"]), int(row["frame_index"]))
            rows[key] = row
    return rows


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resize_rgb(frame, size):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    if size is not None:
        image = image.resize((int(size), int(size)), resample=Image.BICUBIC)
    return image


def read_gt_frame(item, local_frame, reference_shape):
    dataset_frame = int(item["start_frame"]) + int(local_frame)
    path = Path(item["gt_frames_dir"]) / f"{dataset_frame:04d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT frame: {path}")
    height, width = reference_shape[:2]
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def quality_scores_from_array(frame):
    """Cheap no-reference scores; every returned value is higher-is-better."""
    rgb = np.asarray(frame, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if min(gray.shape) < 3:
        raise ValueError("Quality scoring requires images at least 3x3")

    grad_y, grad_x = np.gradient(gray)
    gradient_energy = float(np.mean(grad_x * grad_x + grad_y * grad_y))
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    laplacian_variance = float(np.var(laplacian))
    contrast = float(np.std(gray))
    histogram = np.histogram(gray, bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    entropy /= math.log2(64.0)
    clipped = np.mean((rgb <= 1.0 / 255.0) | (rgb >= 254.0 / 255.0))
    sharpness_contrast = (gradient_energy / (gradient_energy + 0.002)) * (
        contrast / (contrast + 0.05)
    )

    return {
        "sharpness_contrast": float(np.clip(sharpness_contrast, 0.0, 1.0)),
        "laplacian_variance": laplacian_variance,
        "gradient_energy": gradient_energy,
        "contrast": contrast,
        "entropy": entropy,
        "unclipped_fraction": float(1.0 - clipped),
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 2 or np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def quality_auc(scores, bad_labels):
    """AUC where clean is positive and larger scores predict clean."""
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    clean = ~bad[valid]
    n_clean = int(clean.sum())
    n_bad = int((~clean).sum())
    if n_clean == 0 or n_bad == 0:
        return None
    ranks = average_ranks(scores) + 1.0
    clean_rank_sum = float(ranks[clean].sum())
    return (clean_rank_sum - n_clean * (n_clean + 1) / 2.0) / (
        n_clean * n_bad
    )


def classification_metrics(scores, bad_labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    bad = bad[valid]
    predicted_bad = scores < float(threshold)
    tp = int(np.sum(predicted_bad & bad))
    fn = int(np.sum((~predicted_bad) & bad))
    fp = int(np.sum(predicted_bad & (~bad)))
    tn = int(np.sum((~predicted_bad) & (~bad)))
    bad_recall = tp / (tp + fn) if tp + fn else None
    clean_recall = tn / (tn + fp) if tn + fp else None
    balanced = (
        0.5 * (bad_recall + clean_recall)
        if bad_recall is not None and clean_recall is not None
        else None
    )
    rejected = tp + fp
    bad_prevalence = (tp + fn) / len(scores) if len(scores) else None
    bad_precision = tp / rejected if rejected else None
    return {
        "samples": len(scores),
        "bad_frames": int(bad.sum()),
        "clean_frames": int((~bad).sum()),
        "rejected_frames": rejected,
        "rejected_fraction": rejected / len(scores) if len(scores) else None,
        "bad_prevalence": bad_prevalence,
        "bad_precision": bad_precision,
        "rejection_enrichment": (
            bad_precision / bad_prevalence
            if bad_precision is not None and bad_prevalence
            else None
        ),
        "bad_recall": bad_recall,
        "clean_false_reject_rate": fp / (fp + tn) if fp + tn else None,
        "balanced_accuracy": balanced,
    }


def fit_quality_threshold(scores, bad_labels):
    """Fit a low-score rejection threshold by training balanced accuracy."""
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    bad = bad[valid]
    if len(scores) == 0 or not np.any(bad) or np.all(bad):
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_bad = bad[order]
    total_bad = int(sorted_bad.sum())
    total_clean = len(sorted_bad) - total_bad
    cumulative_bad = np.concatenate([[0], np.cumsum(sorted_bad)])
    cumulative_clean = np.arange(len(sorted_bad) + 1) - cumulative_bad
    boundaries = [0]
    boundaries.extend(
        idx for idx in range(1, len(sorted_scores))
        if sorted_scores[idx - 1] < sorted_scores[idx]
    )
    boundaries.append(len(sorted_scores))

    best = None
    for count in boundaries:
        tp = int(cumulative_bad[count])
        fp = int(cumulative_clean[count])
        bad_recall = tp / total_bad
        clean_fpr = fp / total_clean
        balanced = 0.5 * (bad_recall + 1.0 - clean_fpr)
        if count == 0:
            threshold = float("-inf")
        elif count == len(sorted_scores):
            threshold = float("inf")
        else:
            threshold = float(
                0.5 * (sorted_scores[count - 1] + sorted_scores[count])
            )
        candidate = (balanced, -clean_fpr, bad_recall, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3]


def fit_conservative_threshold(scores, bad_labels, max_clean_false_reject):
    """Maximize bad-frame recall subject to a clean-frame rejection cap."""
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    bad = bad[valid]
    if len(scores) == 0 or not np.any(bad) or np.all(bad):
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_bad = bad[order]
    total_bad = int(sorted_bad.sum())
    total_clean = len(sorted_bad) - total_bad
    cumulative_bad = np.concatenate([[0], np.cumsum(sorted_bad)])
    cumulative_clean = np.arange(len(sorted_bad) + 1) - cumulative_bad
    boundaries = [0]
    boundaries.extend(
        idx for idx in range(1, len(sorted_scores))
        if sorted_scores[idx - 1] < sorted_scores[idx]
    )
    boundaries.append(len(sorted_scores))
    best = None
    for count in boundaries:
        bad_recall = float(cumulative_bad[count]) / total_bad
        clean_fpr = float(cumulative_clean[count]) / total_clean
        if clean_fpr > float(max_clean_false_reject) + 1e-12:
            continue
        if count == 0:
            threshold = float("-inf")
        elif count == len(sorted_scores):
            threshold = float("inf")
        else:
            threshold = float(
                0.5 * (sorted_scores[count - 1] + sorted_scores[count])
            )
        candidate = (bad_recall, -clean_fpr, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else float("-inf")


def trajectory_split(row_ids, test_fraction, seed):
    row_ids = sorted(set(int(row) for row in row_ids))
    if len(row_ids) < 2:
        raise ValueError("At least two trajectories are required for a held-out split")
    rng = np.random.default_rng(int(seed))
    shuffled = list(row_ids)
    rng.shuffle(shuffled)
    test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_fraction)))
    test_rows = set(shuffled[:test_count])
    train_rows = set(shuffled[test_count:])
    return train_rows, test_rows


def add_within_trajectory_labels(rows, bad_quantile):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["row"]))].append(row)
    for group_rows in grouped.values():
        psnr_ranks = average_ranks([row["psnr_db"] for row in group_rows])
        ssim_ranks = average_ranks([row["ssim"] for row in group_rows])
        denominator = max(len(group_rows) - 1, 1)
        composite = 0.5 * (psnr_ranks + ssim_ranks) / denominator
        cutoff = float(np.quantile(composite, float(bad_quantile)))
        for row, value in zip(group_rows, composite):
            row["gt_quality_percentile"] = float(value)
            row["gt_bad_frame"] = bool(value <= cutoff)


def bootstrap_group_mean(values_by_group, repeats=4000, seed=0):
    values = np.asarray(
        [value for value in values_by_group if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def estimator_rows(
    scored_rows,
    estimator_fields,
    train_rows,
    test_rows,
    repeats,
    seed,
    max_train_clean_false_reject,
):
    output = []
    train = [row for row in scored_rows if int(row["row"]) in train_rows]
    test = [row for row in scored_rows if int(row["row"]) in test_rows]
    for estimator in estimator_fields:
        usable_train = [row for row in train if row.get(estimator) is not None]
        usable_test = [row for row in test if row.get(estimator) is not None]
        if not usable_train or not usable_test:
            continue
        threshold = fit_quality_threshold(
            [row[estimator] for row in usable_train],
            [row["gt_bad_frame"] for row in usable_train],
        )
        if threshold is None:
            continue
        gate_threshold = fit_conservative_threshold(
            [row[estimator] for row in usable_train],
            [row["gt_bad_frame"] for row in usable_train],
            max_train_clean_false_reject,
        )
        test_metrics = classification_metrics(
            [row[estimator] for row in usable_test],
            [row["gt_bad_frame"] for row in usable_test],
            threshold,
        )
        gate_test_metrics = classification_metrics(
            [row[estimator] for row in usable_test],
            [row["gt_bad_frame"] for row in usable_test],
            gate_threshold,
        )
        per_trajectory_rhos = []
        for trajectory in sorted(test_rows):
            run_rhos = []
            for run_name in sorted({row["run_name"] for row in usable_test}):
                group = [
                    row for row in usable_test
                    if int(row["row"]) == trajectory
                    and row["run_name"] == run_name
                ]
                rho = spearman(
                    [row[estimator] for row in group],
                    [row["gt_quality_percentile"] for row in group],
                )
                if rho is not None:
                    run_rhos.append(rho)
            per_trajectory_rhos.append(
                float(np.mean(run_rhos)) if run_rhos else None
            )
        ci_low, ci_high = bootstrap_group_mean(
            per_trajectory_rhos, repeats=repeats, seed=seed
        )
        output.append(
            {
                "estimator": estimator,
                "train_trajectories": len(train_rows),
                "test_trajectories": len(test_rows),
                "train_frames": len(usable_train),
                "test_frames": len(usable_test),
                "threshold": threshold,
                "gate_threshold": gate_threshold,
                "max_train_clean_false_reject": max_train_clean_false_reject,
                "test_quality_auc": quality_auc(
                    [row[estimator] for row in usable_test],
                    [row["gt_bad_frame"] for row in usable_test],
                ),
                "test_spearman_psnr": spearman(
                    [row[estimator] for row in usable_test],
                    [row["psnr_db"] for row in usable_test],
                ),
                "test_spearman_ssim": spearman(
                    [row[estimator] for row in usable_test],
                    [row["ssim"] for row in usable_test],
                ),
                "test_mean_within_trajectory_spearman": float(
                    np.mean([value for value in per_trajectory_rhos if value is not None])
                ),
                "test_within_trajectory_spearman_ci_low": ci_low,
                "test_within_trajectory_spearman_ci_high": ci_high,
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{
                    f"gate_test_{key}": value
                    for key, value in gate_test_metrics.items()
                },
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["test_balanced_accuracy"] if row["test_balanced_accuracy"] is not None else -1,
            row["test_quality_auc"] if row["test_quality_auc"] is not None else -1,
        ),
        reverse=True,
    )


def estimator_run_rows(scored_rows, summaries, test_rows):
    output = []
    for summary in summaries:
        estimator = summary["estimator"]
        for run_name in sorted({row["run_name"] for row in scored_rows}):
            rows = [
                row for row in scored_rows
                if int(row["row"]) in test_rows
                and row["run_name"] == run_name
                and row.get(estimator) is not None
            ]
            if not rows:
                continue
            metrics = classification_metrics(
                [row[estimator] for row in rows],
                [row["gt_bad_frame"] for row in rows],
                summary["gate_threshold"],
            )
            output.append(
                {
                    "estimator": estimator,
                    "run_name": run_name,
                    "test_frames": len(rows),
                    "quality_auc": quality_auc(
                        [row[estimator] for row in rows],
                        [row["gt_bad_frame"] for row in rows],
                    ),
                    "spearman_psnr": spearman(
                        [row[estimator] for row in rows],
                        [row["psnr_db"] for row in rows],
                    ),
                    "spearman_ssim": spearman(
                        [row[estimator] for row in rows],
                        [row["ssim"] for row in rows],
                    ),
                    **{f"gate_{key}": value for key, value in metrics.items()},
                }
            )
    return output


def score_cpu_frames(args, items, cache, state):
    import imageio.v2 as imageio

    for run_name in args.runs:
        for item in items:
            video_key = f"{run_name}:{item['_row']}"
            if video_key in state["cpu_completed"]:
                continue
            video_path = args.root / run_name / f"{item['output_prefix']}custom.mp4"
            if not video_path.is_file():
                message = f"Missing video: {video_path}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue
            frame_dir = args.frame_cache_dir / run_name / f"row_{item['_row']:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            reader = imageio.get_reader(str(video_path))
            sampled = 0
            try:
                for frame_index, generated in enumerate(reader):
                    if frame_index % args.frame_stride != 0:
                        continue
                    if args.max_frames_per_video and sampled >= args.max_frames_per_video:
                        break
                    generated = np.asarray(generated, dtype=np.uint8)
                    gt = read_gt_frame(item, frame_index, generated.shape)
                    metric_values = frame_metrics(generated, gt)
                    quality_image = resize_rgb(generated, args.quality_size)
                    quality_array = np.asarray(quality_image, dtype=np.uint8)
                    estimates = quality_scores_from_array(quality_array)
                    key = (run_name, int(item["_row"]), int(frame_index))
                    cache[key] = {
                        "run_name": run_name,
                        "row": int(item["_row"]),
                        "scene": item["scene"],
                        "start_frame": int(item["start_frame"]),
                        "duration_sec": int(item["duration_sec"]),
                        "frame_index": int(frame_index),
                        "dataset_frame": int(item["start_frame"]) + int(frame_index),
                        "time_sec": float(frame_index / float(item["fps"])),
                        "psnr_db": float(metric_values["psnr_db"]),
                        "ssim": float(metric_values["ssim"]),
                        **estimates,
                    }
                    quality_image.save(frame_dir / f"{frame_index:05d}.png")
                    sampled += 1
            finally:
                reader.close()
            state["cpu_completed"].append(video_key)
            atomic_write_jsonl(args.cache_jsonl, cache)
            atomic_write_json(args.state_json, state)
            print(f"[cpu] {video_key}: {sampled} sampled frames")


def load_cached_images(args, rows):
    images = []
    for row in rows:
        path = (
            args.frame_cache_dir
            / row["run_name"]
            / f"row_{int(row['row']):03d}"
            / f"{int(row['frame_index']):05d}.png"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing frame cache image: {path}")
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return images


def model_scores(model, arrays, torch, device, batch_size):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(arrays), int(batch_size)):
            batch_arrays = np.stack(arrays[start : start + int(batch_size)])
            tensor = torch.from_numpy(batch_arrays).permute(0, 3, 1, 2).float()
            tensor = tensor.div_(255.0).to(device)
            try:
                result = model(tensor)
                values = result.detach().float().reshape(-1).cpu().tolist()
                if len(values) != len(batch_arrays):
                    raise RuntimeError("metric returned a non-batched result")
            except Exception:
                values = [
                    float(model(sample.unsqueeze(0)).detach().float().item())
                    for sample in tensor
                ]
            scores.extend(values)
    return scores


def score_pyiqa_frames(args, items, cache, state):
    if not args.iqa_metrics:
        return {}
    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Learned IQA requested but pyiqa is unavailable. Install it in the "
            "active environment with `python -m pip install pyiqa`."
        ) from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA IQA requested but torch reports no available GPU")
    device = torch.device(args.device)
    directions = {}
    item_by_row = {int(item["_row"]): item for item in items}
    for metric_name in args.iqa_metrics:
        field = f"iqa_{safe_name(metric_name)}"
        print(f"[iqa] loading {metric_name} as {field}")
        try:
            model = pyiqa.create_metric(metric_name, device=str(device))
        except TypeError:
            model = pyiqa.create_metric(metric_name).to(device)
        model.eval()
        lower_better = bool(getattr(model, "lower_better", False))
        directions[field] = "lower" if lower_better else "higher"
        completed = set(state["iqa_completed"].get(field, []))
        for run_name in args.runs:
            for row_id in sorted(item_by_row):
                video_key = f"{run_name}:{row_id}"
                if video_key in completed:
                    continue
                frame_rows = [
                    row for key, row in cache.items()
                    if key[0] == run_name and key[1] == row_id
                ]
                frame_rows.sort(key=lambda row: int(row["frame_index"]))
                if not frame_rows:
                    continue
                arrays = load_cached_images(args, frame_rows)
                raw_scores = model_scores(
                    model, arrays, torch, device, args.batch_size
                )
                for row, raw_score in zip(frame_rows, raw_scores):
                    row[field] = float(-raw_score if lower_better else raw_score)
                    row[f"{field}_raw"] = float(raw_score)
                completed.add(video_key)
                state["iqa_completed"][field] = sorted(completed)
                state["iqa_directions"][field] = directions[field]
                atomic_write_jsonl(args.cache_jsonl, cache)
                atomic_write_json(args.state_json, state)
                print(f"[iqa] {metric_name} {video_key}: {len(frame_rows)} frames")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return directions


def fmt(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    path, summaries, run_summaries, args, train_rows, test_rows, frame_count
):
    lines = [
        "# Frame Quality Estimator Calibration",
        "",
        "## Question",
        "",
        "Can a score computed from the generated frame alone identify corrupted frames before they enter memory? Exact-index dataset PSNR/SSIM are hidden evaluation labels; they are not estimator inputs.",
        "",
        "## Protocol",
        "",
        f"- Runs: `{','.join(args.runs)}`.",
        f"- Sampled frames: `{frame_count}` at stride `{args.frame_stride}`.",
        f"- Train trajectories: `{','.join(map(str, sorted(train_rows)))}`.",
        f"- Held-out trajectories: `{','.join(map(str, sorted(test_rows)))}`.",
        f"- A bad frame is in the bottom `{100 * args.bad_quantile:.0f}%` of combined PSNR/SSIM rank within its own run and trajectory.",
        "- Thresholds maximize balanced accuracy on train trajectories only, then remain fixed on held-out trajectories.",
        "- Every estimator is oriented so that a larger value means better predicted quality.",
        "",
        "## Held-Out Results",
        "",
        "| estimator | quality AUC | balanced accuracy | deployable bad precision | deployable bad recall | deployable clean reject | rho PSNR | rho SSIM | within-trajectory rho | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['estimator']} | {fmt(row['test_quality_auc'])} | "
            f"{fmt(row['test_balanced_accuracy'])} | "
            f"{fmt(row['gate_test_bad_precision'])} | "
            f"{fmt(row['gate_test_bad_recall'])} | "
            f"{fmt(row['gate_test_clean_false_reject_rate'])} | "
            f"{fmt(row['test_spearman_psnr'])} | {fmt(row['test_spearman_ssim'])} | "
            f"{fmt(row['test_mean_within_trajectory_spearman'])} | "
            f"[{fmt(row['test_within_trajectory_spearman_ci_low'])}, "
            f"{fmt(row['test_within_trajectory_spearman_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "The deployable threshold is selected on training trajectories while allowing at most "
            f"`{100 * args.max_train_clean_false_reject:.0f}%` clean-frame rejection there. Its held-out rejection rates are shown above.",
            "",
            "## Per-Run Held-Out Check",
            "",
            "| estimator | run | quality AUC | bad recall | clean false reject | rho PSNR | rho SSIM |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_summaries:
        lines.append(
            f"| {row['estimator']} | {row['run_name']} | "
            f"{fmt(row['quality_auc'])} | {fmt(row['gate_bad_recall'])} | "
            f"{fmt(row['gate_clean_false_reject_rate'])} | "
            f"{fmt(row['spearman_psnr'])} | {fmt(row['spearman_ssim'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A useful gate should have held-out AUC and balanced accuracy clearly above 0.5, positive within-trajectory correlation whose interval does not collapse around zero, and an acceptable clean-frame false-rejection rate. Failure here means the estimator should not be put into the memory policy, regardless of its reputation on generic IQA benchmarks.",
            "",
            "PSNR/SSIM include identity and alignment errors as well as blur or artifacts. A no-reference estimator may therefore detect visible corruption but miss a plausible-looking wrong view. This experiment tests the estimator's actual usefulness for MemCam; it does not redefine full-reference fidelity as available online information.",
            "",
            "## Files",
            "",
            "- `frame_scores.jsonl`",
            "- `estimator_summary.csv`",
            "- `estimator_by_run.csv`",
            "- `state.json`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path, summaries):
    if not summaries:
        return
    import matplotlib.pyplot as plt

    labels = [row["estimator"].replace("iqa_", "") for row in summaries]
    auc = [row["test_quality_auc"] for row in summaries]
    balanced = [row["test_balanced_accuracy"] for row in summaries]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.48 * len(labels))))
    ax.barh(y - 0.18, auc, height=0.34, label="Quality AUC", color="#287271")
    ax.barh(y + 0.18, balanced, height=0.34, label="Balanced accuracy", color="#E76F51")
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Held-out score (0.5 = chance)")
    ax.set_title("Can no-reference quality scores detect corrupted MemCam frames?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=parse_list,
        default=parse_list("baseline,fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility"),
    )
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--quality_size", type=int, default=224)
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    parser.add_argument(
        "--iqa_metrics",
        type=parse_list,
        default=parse_list("musiq-paq2piq,clipiqa+,topiq_nr,niqe"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bad_quantile", type=float, default=0.20)
    parser.add_argument("--max_train_clean_false_reject", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--bootstrap_repeats", type=int, default=4000)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_cache_dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride <= 0 or args.quality_size <= 0:
        parser.error("frame_stride and quality_size must be positive")
    if not 0.0 < args.bad_quantile < 0.5:
        parser.error("bad_quantile must be between 0 and 0.5")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("test_fraction must be between 0 and 1")
    if not 0.0 <= args.max_train_clean_false_reject < 1.0:
        parser.error("max_train_clean_false_reject must be in [0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_cache_dir is None:
        args.frame_cache_dir = args.output_dir / "frame_cache"
    args.frame_cache_dir.mkdir(parents=True, exist_ok=True)
    args.cache_jsonl = args.output_dir / "frame_scores.jsonl"
    args.state_json = args.output_dir / "state.json"
    state = read_json(
        args.state_json,
        {"cpu_completed": [], "iqa_completed": {}, "iqa_directions": {}},
    )
    state.setdefault("cpu_completed", [])
    state.setdefault("iqa_completed", {})
    state.setdefault("iqa_directions", {})
    config = {
        "runs": args.runs,
        "duration": args.duration,
        "frame_stride": args.frame_stride,
        "quality_size": args.quality_size,
        "max_frames_per_video": args.max_frames_per_video,
    }
    if state.get("config") not in (None, config):
        raise RuntimeError(
            "Output directory contains a different sampling configuration. "
            "Use a new --output_dir or restore the original arguments."
        )
    state["config"] = config
    cache = load_jsonl_by_key(args.cache_jsonl)
    items = load_manifest(args.manifest, args.duration)
    if not items:
        raise RuntimeError(f"No {args.duration}s rows in {args.manifest}")

    print(f"Trajectories: {len(items)}; runs: {len(args.runs)}")
    print(f"CPU estimators: {','.join(CPU_ESTIMATORS)}")
    print(f"Learned IQA: {','.join(args.iqa_metrics) if args.iqa_metrics else 'none'}")
    score_cpu_frames(args, items, cache, state)
    score_pyiqa_frames(args, items, cache, state)

    requested_rows = {int(item["_row"]) for item in items}
    scored_rows = [
        row for row in cache.values()
        if row["run_name"] in args.runs and int(row["row"]) in requested_rows
    ]
    if not scored_rows:
        raise RuntimeError("No frame scores were produced")
    add_within_trajectory_labels(scored_rows, args.bad_quantile)
    train_rows, test_rows = trajectory_split(
        requested_rows, args.test_fraction, args.split_seed
    )
    estimator_fields = list(CPU_ESTIMATORS)
    estimator_fields.extend(
        sorted(
            field for field in state["iqa_directions"]
            if all(row.get(field) is not None for row in scored_rows)
        )
    )
    summaries = estimator_rows(
        scored_rows,
        estimator_fields,
        train_rows,
        test_rows,
        args.bootstrap_repeats,
        args.split_seed,
        args.max_train_clean_false_reject,
    )
    run_summaries = estimator_run_rows(scored_rows, summaries, test_rows)
    atomic_write_jsonl(args.cache_jsonl, cache)
    write_csv(args.output_dir / "estimator_summary.csv", summaries)
    write_csv(args.output_dir / "estimator_by_run.csv", run_summaries)
    write_report(
        args.output_dir / "report.md",
        summaries,
        run_summaries,
        args,
        train_rows,
        test_rows,
        len(scored_rows),
    )
    make_plot(args.output_dir / "heldout_estimator_comparison.png", summaries)

    print("\nHeld-out estimator ranking:")
    for row in summaries:
        print(
            f"{row['estimator']}: AUC={fmt(row['test_quality_auc'])} "
            f"balanced={fmt(row['test_balanced_accuracy'])} "
            f"gate_precision={fmt(row['gate_test_bad_precision'])} "
            f"gate_recall={fmt(row['gate_test_bad_recall'])} "
            f"gate_clean_reject={fmt(row['gate_test_clean_false_reject_rate'])}"
        )
    print(f"\nWrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
