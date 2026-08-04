import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SECTION_STRIDE = 76
FRAMES_PER_SECTION = 77


def parse_list(value):
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_rows(value):
    if not value:
        return None
    rows = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            rows.update(range(int(start), int(end) + 1))
        else:
            rows.add(int(part))
    return rows


def run_identity(item):
    return (
        str(item["scene"]),
        int(item["start_frame"]),
        int(item["duration_sec"]),
    )


def trace_identity(row):
    return (
        str(row.get("scene")),
        int(row.get("dataset_start_frame")),
        int(row.get("duration_sec")),
    )


def load_manifest(path, duration=None, selected_rows=None, max_rows=None):
    selected_rows = set(selected_rows) if selected_rows is not None else None
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_row"] = row_idx
            if duration is not None and int(item["duration_sec"]) != int(duration):
                continue
            if selected_rows is not None and row_idx not in selected_rows:
                continue
            rows.append(item)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    return rows


def read_trace(path, expected_identity=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if expected_identity is not None and row.get("scene") is not None:
                if trace_identity(row) != expected_identity:
                    raise ValueError(
                        f"Trace identity mismatch in {path}: "
                        f"expected {expected_identity}, found {trace_identity(row)}"
                    )
            rows.append(row)
    return rows


def selected_context_rows(events):
    rows = {}
    for row in events:
        if row.get("event") != "context_access" or not row.get("selected"):
            continue
        key = (int(row["section_idx"]), int(row["target_frame"]))
        rows[key] = row
    return rows


def reconstruct_candidate_banks(
    events,
    max_section,
    num_frames=None,
    section_stride=SECTION_STRIDE,
    frames_per_section=FRAMES_PER_SECTION,
):
    """Reconstruct the exact pre-retrieval bank from insertion and eviction logs."""
    evictions = defaultdict(list)
    for row in events:
        if row.get("event") != "memory_eviction":
            continue
        evictions[int(row["section_idx"])].append(
            int(row["evicted_memory_frame"])
        )

    bank = {0}
    banks = {0: []}
    for section_idx in range(int(max_section) + 1):
        section_start = section_idx * int(section_stride)
        if section_idx > 0:
            anchor_frames = set(range(section_start - 3, section_start + 1))
            banks[section_idx] = sorted(bank - anchor_frames)

        section_end = section_start + int(frames_per_section) - 1
        if num_frames is not None:
            section_end = min(section_end, int(num_frames) - 1)
        if section_start <= section_end:
            bank.update(range(section_start, section_end + 1))
        for frame_idx in evictions.get(section_idx, []):
            bank.discard(frame_idx)
    return banks


def cosine_distance(left, right):
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


def best_effective_match(
    generated_features,
    target_feature,
    candidates,
    effective_similarities=None,
):
    candidates = np.asarray(list(candidates), dtype=np.int64)
    if candidates.size == 0:
        return None, None
    if effective_similarities is None:
        similarities = generated_features[candidates] @ target_feature
    else:
        similarities = np.asarray(effective_similarities)[candidates]
    column = int(np.argmax(similarities))
    return int(candidates[column]), float(1.0 - np.clip(similarities[column], -1.0, 1.0))


def frame_match_components(generated_features, gt_features, memory_frame, target_frame):
    return {
        "view_mismatch": cosine_distance(
            gt_features[int(memory_frame)], gt_features[int(target_frame)]
        ),
        "memory_corruption": cosine_distance(
            generated_features[int(memory_frame)], gt_features[int(memory_frame)]
        ),
        "effective_mismatch": cosine_distance(
            generated_features[int(memory_frame)], gt_features[int(target_frame)]
        ),
    }


def compute_query_decomposition(
    generated_features,
    gt_features,
    bank_candidates,
    history_candidates,
    selected_frame,
    target_frame,
    effective_similarities=None,
):
    target_frame = int(target_frame)
    selected_frame = int(selected_frame)
    feature_count = min(len(generated_features), len(gt_features))
    if not 0 <= target_frame < feature_count:
        raise IndexError(f"Target frame {target_frame} is outside {feature_count} features")
    if not 0 <= selected_frame < feature_count:
        raise IndexError(
            f"Selected memory frame {selected_frame} is outside {feature_count} features"
        )

    bank_candidates = [
        int(frame) for frame in bank_candidates if 0 <= int(frame) < feature_count
    ]
    history_candidates = [
        int(frame) for frame in history_candidates if 0 <= int(frame) < feature_count
    ]
    if selected_frame not in bank_candidates:
        raise ValueError(
            f"Selected frame {selected_frame} is absent from the reconstructed bank"
        )

    target_feature = gt_features[target_frame]
    bank_oracle_frame, bank_oracle_distance = best_effective_match(
        generated_features,
        target_feature,
        bank_candidates,
        effective_similarities=effective_similarities,
    )
    full_oracle_frame, full_oracle_distance = best_effective_match(
        generated_features,
        target_feature,
        history_candidates,
        effective_similarities=effective_similarities,
    )
    selected = frame_match_components(
        generated_features, gt_features, selected_frame, target_frame
    )
    bank_oracle = frame_match_components(
        generated_features, gt_features, bank_oracle_frame, target_frame
    )
    full_oracle = frame_match_components(
        generated_features, gt_features, full_oracle_frame, target_frame
    )

    retention_gap = max(0.0, float(bank_oracle_distance - full_oracle_distance))
    retrieval_gap = max(0.0, float(selected["effective_mismatch"] - bank_oracle_distance))
    return {
        "selected_memory_frame": selected_frame,
        "selected_view_mismatch": selected["view_mismatch"],
        "selected_memory_corruption": selected["memory_corruption"],
        "selected_effective_mismatch": selected["effective_mismatch"],
        "bank_oracle_frame": bank_oracle_frame,
        "bank_oracle_view_mismatch": bank_oracle["view_mismatch"],
        "bank_oracle_memory_corruption": bank_oracle["memory_corruption"],
        "bank_oracle_effective_mismatch": bank_oracle_distance,
        "full_oracle_frame": full_oracle_frame,
        "full_oracle_view_mismatch": full_oracle["view_mismatch"],
        "full_oracle_memory_corruption": full_oracle["memory_corruption"],
        "full_oracle_effective_mismatch": full_oracle_distance,
        "retention_gap": retention_gap,
        "retrieval_gap": retrieval_gap,
        "total_oracle_gap": retention_gap + retrieval_gap,
    }


class DinoFrameEncoder:
    def __init__(self, model_name, device="cuda", batch_size=64):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for DINO but unavailable; using CPU.")
            device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    def encode_batch(self, images):
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        with self.torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = outputs.last_hidden_state[:, 0]
            features = self.torch.nn.functional.normalize(features.float(), dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_iterable(self, images, expected_count=None, label="frames"):
        features = []
        batch = []
        count = 0
        for image in images:
            batch.append(image)
            if len(batch) >= self.batch_size:
                features.append(self.encode_batch(batch))
                count += len(batch)
                batch = []
                if count % (self.batch_size * 10) == 0:
                    print(f"  encoded {count} {label}")
        if batch:
            features.append(self.encode_batch(batch))
            count += len(batch)
        if expected_count is not None and count < int(expected_count):
            raise RuntimeError(
                f"Encoded only {count}/{expected_count} {label}; input is incomplete"
            )
        return np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)

    def cross_similarity(self, left_features, right_features, batch_size=512):
        """Compute a normalized feature cross-product on the encoder device."""
        left_features = np.asarray(left_features, dtype=np.float32)
        right_features = np.asarray(right_features, dtype=np.float32)
        right = self.torch.from_numpy(right_features).to(self.device)
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(left_features), int(batch_size)):
                left = self.torch.from_numpy(
                    left_features[start : start + int(batch_size)]
                ).to(self.device)
                outputs.append((left @ right.T).detach().cpu().numpy())
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.zeros((0, len(right_features)), dtype=np.float32)
        )


def iter_video_images(path, max_frames=None):
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(path))
    try:
        for frame_idx, array in enumerate(reader):
            if max_frames is not None and frame_idx >= int(max_frames):
                break
            yield Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
    finally:
        reader.close()


def resolve_gt_dir(item, dataset_root=None):
    if dataset_root is not None:
        candidate = Path(dataset_root) / "frames" / item["scene"]
        if candidate.is_dir():
            return candidate
    original = item.get("gt_frames_dir")
    if original and Path(original).is_dir():
        return Path(original)
    raise FileNotFoundError(
        f"Ground-truth frames are unavailable for {run_identity(item)}. "
        "Pass --dataset_root to remap the manifest."
    )


def indexed_frame_paths(frame_dir):
    paths = {}
    for path in Path(frame_dir).iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match:
            paths[int(match.group(1))] = path
    return paths


def iter_gt_images(item, dataset_root=None):
    from PIL import Image

    frame_dir = resolve_gt_dir(item, dataset_root=dataset_root)
    paths = indexed_frame_paths(frame_dir)
    start = int(item["start_frame"])
    for local_frame in range(int(item["num_frames"])):
        dataset_frame = start + local_frame
        path = paths.get(dataset_frame)
        if path is None:
            raise FileNotFoundError(
                f"Missing GT frame {dataset_frame} in {frame_dir}"
            )
        with Image.open(path) as image:
            yield image.convert("RGB")


def cached_features(cache_path, encoder, image_iter_factory, expected_count, label):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        features = np.load(cache_path)
        if len(features) >= int(expected_count):
            print(f"  cache hit: {cache_path}")
            return np.asarray(features[: int(expected_count)], dtype=np.float32)
        print(f"  ignoring short cache {cache_path}: {len(features)}/{expected_count}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = encoder.encode_iterable(
        image_iter_factory(),
        expected_count=expected_count,
        label=label,
    )
    features = np.asarray(features[: int(expected_count)], dtype=np.float32)
    np.save(cache_path, features)
    return features


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


SUMMARY_METRICS = (
    "selected_view_mismatch",
    "selected_memory_corruption",
    "selected_effective_mismatch",
    "full_oracle_effective_mismatch",
    "bank_oracle_effective_mismatch",
    "retention_gap",
    "retrieval_gap",
    "total_oracle_gap",
)


def aggregate_query_rows(query_rows, group_fields):
    grouped = defaultdict(list)
    for row in query_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = dict(zip(group_fields, key))
        summary["queries"] = len(rows)
        summary["videos"] = len({int(row["row"]) for row in rows})
        for field in SUMMARY_METRICS:
            summary[field] = mean(row.get(field) for row in rows)
        summary["candidate_count"] = mean(row.get("candidate_count") for row in rows)
        summary["candidate_count_mismatch_rate"] = mean(
            row.get("candidate_count_mismatch") for row in rows
        )
        output.append(summary)
    return output


def build_replay_candidates(section_rows, baseline_run, intervention_run):
    by_key = {
        (row["run_name"], int(row["row"]), int(row["section_idx"])): row
        for row in section_rows
    }
    output = []
    for (run_name, row_idx, section_idx), policy in by_key.items():
        if run_name != intervention_run:
            continue
        baseline = by_key.get((baseline_run, row_idx, section_idx))
        if baseline is None:
            continue
        selected_improvement = (
            float(baseline["selected_effective_mismatch"])
            - float(policy["selected_effective_mismatch"])
        )
        retrieval_improvement = (
            float(baseline["retrieval_gap"]) - float(policy["retrieval_gap"])
        )
        output.append(
            {
                "row": row_idx,
                "scene": policy["scene"],
                "dataset_start_frame": policy["dataset_start_frame"],
                "duration_sec": policy["duration_sec"],
                "section_idx": section_idx,
                "baseline_run": baseline_run,
                "intervention_run": intervention_run,
                "queries": min(int(baseline["queries"]), int(policy["queries"])),
                "baseline_selected_effective_mismatch": baseline[
                    "selected_effective_mismatch"
                ],
                "intervention_selected_effective_mismatch": policy[
                    "selected_effective_mismatch"
                ],
                "selected_effective_improvement": selected_improvement,
                "baseline_retrieval_gap": baseline["retrieval_gap"],
                "intervention_retrieval_gap": policy["retrieval_gap"],
                "retrieval_gap_improvement": retrieval_improvement,
                "replay_score": selected_improvement + 0.5 * retrieval_improvement,
            }
        )
    return output


def select_replay_cases(candidates, count, min_section=1, unique_rows=True):
    ranked = sorted(
        (
            row
            for row in candidates
            if int(row["section_idx"]) >= int(min_section)
            and float(row["selected_effective_improvement"]) > 0.0
            and float(row["retrieval_gap_improvement"]) > 0.0
        ),
        key=lambda row: (float(row["replay_score"]), int(row["section_idx"])),
        reverse=True,
    )
    selected = []
    used_rows = set()
    for row in ranked:
        if unique_rows and int(row["row"]) in used_rows:
            continue
        selected.append(dict(row, case_index=len(selected)))
        used_rows.add(int(row["row"]))
        if len(selected) >= int(count):
            break
    return selected


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pretty_run_name(run_name):
    if run_name == "baseline":
        return "Unbounded"
    if run_name.startswith("fifo"):
        return "FIFO"
    if run_name.startswith("ri"):
        return "RI"
    if run_name.startswith("slam"):
        return "SLAM-style"
    return run_name.replace("_", " ")


def run_color(run_name):
    if run_name == "baseline":
        return "#555555"
    if run_name.startswith("fifo"):
        return "#2f7fbc"
    if run_name.startswith("ri"):
        return "#d88c1f"
    if run_name.startswith("slam"):
        return "#4f9b62"
    return "#8a5fbf"


def save_gap_figures(run_summary, section_summary, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [row["run_name"] for row in run_summary]
    oracle = np.asarray(
        [row["full_oracle_effective_mismatch"] for row in run_summary]
    )
    retention = np.asarray([row["retention_gap"] for row in run_summary])
    retrieval = np.asarray([row["retrieval_gap"] for row in run_summary])
    colors = [run_color(run) for run in runs]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(runs))
    ax.bar(x, oracle, color="#d9d9d9", label="Best available in full history")
    ax.bar(x, retention, bottom=oracle, color="#d89a45", label="Retention gap")
    ax.bar(
        x,
        retrieval,
        bottom=oracle + retention,
        color=colors,
        label="Retrieval gap",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_run_name(run) for run in runs])
    ax.set_ylabel("DINO distance to target GT (lower is better)")
    ax.set_title("Where selected-context error comes from")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    decomposition_path = output_dir / "retrieval_error_decomposition.png"
    fig.savefig(decomposition_path, dpi=200)
    fig.savefig(decomposition_path.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    for run in runs:
        points = [row for row in section_summary if row["run_name"] == run]
        points.sort(key=lambda row: int(row["section_idx"]))
        x_values = [int(row["section_idx"]) * SECTION_STRIDE / 30.0 for row in points]
        axes[0].plot(
            x_values,
            [row["retrieval_gap"] for row in points],
            color=run_color(run),
            linewidth=2,
            label=pretty_run_name(run),
        )
        axes[1].plot(
            x_values,
            [row["retention_gap"] for row in points],
            color=run_color(run),
            linewidth=2,
            label=pretty_run_name(run),
        )
    for ax, title, ylabel in (
        (axes[0], "Retriever leaves a better retained frame unused", "retrieval gap"),
        (axes[1], "Eviction removes the best historical frame", "retention gap"),
    ):
        ax.set_title(title)
        ax.set_xlabel("generated time (seconds)")
        ax.set_ylabel(f"DINO {ylabel}")
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    temporal_path = output_dir / "gaps_over_time.png"
    fig.savefig(temporal_path, dpi=200)
    fig.savefig(temporal_path.with_suffix(".pdf"))
    plt.close(fig)
    return decomposition_path, temporal_path


def load_video_frame(path, frame_idx, cache):
    from PIL import Image
    import imageio.v2 as imageio

    key = str(path), int(frame_idx)
    if key in cache:
        return cache[key].copy()
    reader = imageio.get_reader(str(path))
    try:
        array = reader.get_data(int(frame_idx))
    finally:
        reader.close()
    image = Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
    cache[key] = image
    return image.copy()


def load_gt_frame(item, local_frame, dataset_root=None):
    from PIL import Image

    frame_dir = resolve_gt_dir(item, dataset_root=dataset_root)
    paths = indexed_frame_paths(frame_dir)
    path = paths[int(item["start_frame"]) + int(local_frame)]
    with Image.open(path) as image:
        return image.convert("RGB")


def save_failure_montage(
    query_rows,
    items_by_row,
    root,
    baseline_run,
    intervention_run,
    dataset_root,
    output_path,
    max_examples=6,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keyed = {
        (row["run_name"], int(row["row"]), int(row["target_frame"])): row
        for row in query_rows
    }
    candidates = []
    for key, baseline in keyed.items():
        if key[0] != baseline_run:
            continue
        policy = keyed.get((intervention_run, key[1], key[2]))
        if policy is None:
            continue
        improvement = (
            float(baseline["selected_effective_mismatch"])
            - float(policy["selected_effective_mismatch"])
        )
        if improvement > 0 and float(baseline["retrieval_gap"]) > 0:
            candidates.append((improvement, baseline, policy))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[: int(max_examples)]
    if not candidates:
        return None

    frame_cache = {}
    fig, axes = plt.subplots(4, len(candidates), figsize=(3.2 * len(candidates), 10.2))
    if len(candidates) == 1:
        axes = np.asarray(axes).reshape(4, 1)
    row_labels = [
        "Target GT",
        "Unbounded selected",
        "Best frame already available",
        f"{pretty_run_name(intervention_run)} selected",
    ]
    for col, (improvement, baseline, policy) in enumerate(candidates):
        item = items_by_row[int(baseline["row"])]
        target_frame = int(baseline["target_frame"])
        baseline_video = Path(root) / baseline_run / f"{item['output_prefix']}custom.mp4"
        policy_video = Path(root) / intervention_run / f"{item['output_prefix']}custom.mp4"
        images = [
            load_gt_frame(item, target_frame, dataset_root=dataset_root),
            load_video_frame(
                baseline_video, int(baseline["selected_memory_frame"]), frame_cache
            ),
            load_video_frame(
                baseline_video, int(baseline["bank_oracle_frame"]), frame_cache
            ),
            load_video_frame(
                policy_video, int(policy["selected_memory_frame"]), frame_cache
            ),
        ]
        for row_idx, image in enumerate(images):
            axes[row_idx, col].imshow(image)
            axes[row_idx, col].axis("off")
            if col == 0:
                axes[row_idx, col].set_ylabel(row_labels[row_idx], fontsize=10)
        axes[0, col].set_title(
            f"row {baseline['row']} | {target_frame / 30.0:.1f}s\n"
            f"selected improvement {improvement:.3f}",
            fontsize=9,
        )
    fig.suptitle(
        "Unbounded retrieval failures: a better frame was already in memory",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows, fields):
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            values.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, run_summary, replay_cases, args):
    baseline = next(
        (row for row in run_summary if row["run_name"] == args.baseline_run), None
    )
    lines = [
        "# Retrieval Quality Decomposition",
        "",
        "## Question",
        "",
        "Why can unbounded MemCam be worse even though it retains every past frame?",
        "",
        "For each real retrieval query, DINO distance to the target ground-truth view is split into:",
        "",
        "- **Full-history oracle:** best generated frame among every eligible past frame.",
        "- **Retention gap:** best retained frame minus the full-history oracle.",
        "- **Retrieval gap:** the frame MemCam selected minus the best retained frame.",
        "",
        "The three terms sum to the selected frame's excess distance. Unbounded has zero retention gap by construction; any avoidable error therefore appears as retrieval gap.",
        "",
        "## Overall",
        "",
        markdown_table(
            run_summary,
            [
                "run_name",
                "queries",
                "selected_effective_mismatch",
                "full_oracle_effective_mismatch",
                "retention_gap",
                "retrieval_gap",
            ],
        ),
        "",
    ]
    if baseline is not None:
        lines.extend(
            [
                "## Unbounded Readout",
                "",
                f"- Mean selected-context DINO distance: `{fmt(baseline['selected_effective_mismatch'])}`.",
                f"- Mean full-history oracle distance: `{fmt(baseline['full_oracle_effective_mismatch'])}`.",
                f"- Mean retrieval gap despite retaining the oracle candidate: `{fmt(baseline['retrieval_gap'])}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Matched Replay Plan",
            "",
            "These cases are hypotheses selected from the offline decomposition. The replay job uses the original unbounded choices for every preceding section in both branches, then changes only the target section to the intervention policy's recorded choices.",
            "",
            markdown_table(
                replay_cases,
                [
                    "case_index",
                    "row",
                    "scene",
                    "section_idx",
                    "selected_effective_improvement",
                    "retrieval_gap_improvement",
                ],
            ),
            "",
            "## Limits",
            "",
            "This first stage is diagnostic, not causal: DINO defines the oracle and each policy has its own generated history. The paired replay is the causal test. A positive replay result means replacing only the context choice improves the generated target section under matched history and noise.",
            "",
            "## Outputs",
            "",
            "- `tables/query_decomposition.csv`",
            "- `tables/run_summary.csv`",
            "- `tables/section_summary.csv`",
            "- `tables/replay_plan.csv`",
            "- `figures/retrieval_error_decomposition.png`",
            "- `figures/gaps_over_time.png`",
            "- `figures/retrieval_failure_montage.png`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Separate memory-retention failures from retrieval failures using "
            "real MemCam traces and DINO view matching."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--runs",
        default="baseline,fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility",
    )
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--intervention_run", default="slam_b32_covisibility")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--target_stride", type=int, default=19)
    parser.add_argument("--min_replay_section", type=int, default=20)
    parser.add_argument("--max_replay_cases", type=int, default=4)
    parser.add_argument("--allow_repeat_replay_rows", action="store_true")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dino_model", default="facebook/dinov2-base")
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.target_stride < 1:
        raise ValueError("--target_stride must be positive")
    runs = parse_list(args.runs)
    if args.baseline_run not in runs:
        raise ValueError("--baseline_run must be included in --runs")
    if args.intervention_run not in runs:
        raise ValueError("--intervention_run must be included in --runs")

    items = load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=parse_rows(args.rows),
        max_rows=args.max_rows,
    )
    if not items:
        raise RuntimeError("No manifest rows selected")

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    cache_dir = args.feature_cache_dir or (output_dir / "feature_cache")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs = []
    for item in items:
        for run_name in runs:
            run_dir = args.root / run_name
            expected = [
                run_dir / f"{item['output_prefix']}custom.mp4",
                run_dir
                / "access_traces"
                / f"{item['output_prefix']}custom.jsonl",
            ]
            missing_inputs.extend(path for path in expected if not path.is_file())
    if missing_inputs and args.strict:
        preview = "\n".join(str(path) for path in missing_inputs[:20])
        raise FileNotFoundError(
            f"Missing {len(missing_inputs)} required videos/traces before analysis:\n"
            f"{preview}"
        )
    for item in items:
        resolve_gt_dir(item, dataset_root=args.dataset_root)

    encoder = DinoFrameEncoder(
        model_name=args.dino_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    query_rows = []
    items_by_row = {int(item["_row"]): item for item in items}

    for item_position, item in enumerate(items, start=1):
        row_idx = int(item["_row"])
        expected_frames = int(item["num_frames"])
        print(
            f"[{item_position}/{len(items)}] row {row_idx} {item['scene']} "
            f"({expected_frames} frames)"
        )
        gt_cache = cache_dir / "gt" / f"{item['output_prefix']}dino.npy"
        gt_features = cached_features(
            gt_cache,
            encoder,
            lambda item=item: iter_gt_images(item, dataset_root=args.dataset_root),
            expected_frames,
            label=f"GT frames for row {row_idx}",
        )

        for run_name in runs:
            run_dir = args.root / run_name
            video_path = run_dir / f"{item['output_prefix']}custom.mp4"
            trace_path = run_dir / "access_traces" / f"{item['output_prefix']}custom.jsonl"
            if not video_path.is_file() or not trace_path.is_file():
                message = (
                    f"row {row_idx} {run_name}: missing "
                    f"{'video' if not video_path.is_file() else 'trace'}"
                )
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"  [skip] {message}")
                continue

            print(f"  run {run_name}")
            events = read_trace(trace_path, expected_identity=run_identity(item))
            selected = selected_context_rows(events)
            if not selected:
                message = f"row {row_idx} {run_name}: no selected context rows"
                if args.strict:
                    raise RuntimeError(message)
                print(f"  [skip] {message}")
                continue
            max_section = max(section for section, _target in selected)
            banks = reconstruct_candidate_banks(
                events,
                max_section=max_section,
                num_frames=expected_frames,
            )
            generated_cache = cache_dir / run_name / f"{item['output_prefix']}dino.npy"
            generated_features = cached_features(
                generated_cache,
                encoder,
                lambda path=video_path: iter_video_images(path, max_frames=expected_frames),
                expected_frames,
                label=f"{run_name} frames for row {row_idx}",
            )

            sampled_queries = []
            for (section_idx, target_frame), trace_row in sorted(selected.items()):
                context_slot = int(trace_row.get("context_slot", target_frame % SECTION_STRIDE))
                if context_slot % args.target_stride != 0:
                    continue
                sampled_queries.append(
                    (section_idx, target_frame, context_slot, trace_row)
                )
            effective_similarities = encoder.cross_similarity(
                gt_features[
                    [int(target_frame) for _section, target_frame, _slot, _row in sampled_queries]
                ],
                generated_features,
            )

            for query_idx, (
                section_idx,
                target_frame,
                context_slot,
                trace_row,
            ) in enumerate(sampled_queries):
                section_start = int(section_idx) * SECTION_STRIDE
                history_candidates = range(0, max(0, section_start - 3))
                bank_candidates = banks.get(section_idx, [])
                try:
                    decomposition = compute_query_decomposition(
                        generated_features=generated_features,
                        gt_features=gt_features,
                        bank_candidates=bank_candidates,
                        history_candidates=history_candidates,
                        selected_frame=int(trace_row["selected_memory_frame"]),
                        target_frame=target_frame,
                        effective_similarities=effective_similarities[query_idx],
                    )
                except (IndexError, ValueError) as exc:
                    if args.strict:
                        raise
                    print(
                        f"  [skip] row {row_idx} {run_name} section {section_idx} "
                        f"target {target_frame}: {exc}"
                    )
                    continue
                traced_candidate_count = int(trace_row.get("candidate_count", -1))
                candidate_count_mismatch = int(
                    traced_candidate_count >= 0
                    and traced_candidate_count != len(bank_candidates)
                )
                if args.strict and candidate_count_mismatch:
                    raise RuntimeError(
                        f"Bank reconstruction mismatch for row {row_idx} {run_name} "
                        f"section {section_idx}: reconstructed={len(bank_candidates)}, "
                        f"trace={traced_candidate_count}"
                    )
                query_rows.append(
                    {
                        "run_name": run_name,
                        "row": row_idx,
                        "scene": item["scene"],
                        "dataset_start_frame": int(item["start_frame"]),
                        "duration_sec": int(item["duration_sec"]),
                        "section_idx": int(section_idx),
                        "context_slot": context_slot,
                        "target_frame": int(target_frame),
                        "target_dataset_frame": int(item["start_frame"]) + int(target_frame),
                        "candidate_count": len(bank_candidates),
                        "traced_candidate_count": traced_candidate_count,
                        "candidate_count_mismatch": candidate_count_mismatch,
                        "selected_overlap": trace_row.get("selected_overlap"),
                        **decomposition,
                    }
                )

    if not query_rows:
        raise RuntimeError("No retrieval queries were analyzed")

    run_summary = aggregate_query_rows(query_rows, ["run_name"])
    section_summary = aggregate_query_rows(
        query_rows,
        [
            "run_name",
            "row",
            "scene",
            "dataset_start_frame",
            "duration_sec",
            "section_idx",
        ],
    )
    time_summary = aggregate_query_rows(query_rows, ["run_name", "section_idx"])
    replay_candidates = build_replay_candidates(
        section_summary,
        baseline_run=args.baseline_run,
        intervention_run=args.intervention_run,
    )
    replay_cases = select_replay_cases(
        replay_candidates,
        count=args.max_replay_cases,
        min_section=args.min_replay_section,
        unique_rows=not args.allow_repeat_replay_rows,
    )

    write_csv(tables_dir / "query_decomposition.csv", query_rows)
    write_csv(tables_dir / "run_summary.csv", run_summary)
    write_csv(tables_dir / "section_summary.csv", section_summary)
    write_csv(tables_dir / "time_summary.csv", time_summary)
    write_csv(tables_dir / "replay_candidates.csv", replay_candidates)
    write_csv(tables_dir / "replay_plan.csv", replay_cases)
    save_gap_figures(run_summary, time_summary, figures_dir)
    save_failure_montage(
        query_rows=query_rows,
        items_by_row=items_by_row,
        root=args.root,
        baseline_run=args.baseline_run,
        intervention_run=args.intervention_run,
        dataset_root=args.dataset_root,
        output_path=figures_dir / "retrieval_failure_montage.png",
    )
    write_report(output_dir / "report.md", run_summary, replay_cases, args)

    print(f"Wrote report: {output_dir / 'report.md'}")
    print(f"Replay cases selected: {len(replay_cases)}")
    for row in replay_cases:
        print(
            f"  case {row['case_index']}: row {row['row']} "
            f"section {row['section_idx']} score={row['replay_score']:.4f}"
        )


if __name__ == "__main__":
    main()
