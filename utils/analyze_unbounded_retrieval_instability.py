import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_STRIDE = 76
PREDICT_FRAMES = 76


def parse_int_ranges(value):
    if not value:
        return None
    values = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return sorted(values)


def parse_pool_sizes(value):
    pools = []
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"all", "unbounded"}:
            pools.append(("all", None))
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("Pool sizes must be positive")
        pools.append((f"recent_{size}", size))
    if not pools:
        raise ValueError("At least one candidate pool size is required")
    return pools


def run_identity(scene, start_frame, duration_sec):
    if scene in (None, "") or start_frame in (None, "") or duration_sec in (None, ""):
        return None
    try:
        return str(scene), int(start_frame), int(float(duration_sec))
    except (TypeError, ValueError):
        return None


def manifest_identity(item):
    return run_identity(
        item.get("scene"),
        item.get("start_frame"),
        item.get("duration_sec"),
    )


def trace_identity(row):
    return run_identity(
        row.get("scene"),
        row.get("dataset_start_frame"),
        row.get("duration_sec"),
    )


def manifest_query_key(item, section_idx, target_frame):
    identity = manifest_identity(item)
    if identity is None:
        return None
    return (*identity, int(section_idx), int(target_frame))


def trace_query_key(row):
    identity = trace_identity(row)
    if identity is None:
        return None
    try:
        return (
            *identity,
            int(row["section_idx"]),
            int(row["target_frame"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def auto_sections(total_sections, count=5):
    valid = list(range(1, int(total_sections)))
    if not valid:
        return []
    count = max(1, min(int(count), len(valid)))
    positions = np.linspace(0, len(valid) - 1, count)
    return sorted({valid[int(round(position))] for position in positions})


def unbounded_candidates(section_idx):
    if section_idx <= 0:
        return []
    section_start = int(section_idx) * SECTION_STRIDE
    # Before section s, unbounded memory contains frames [0, section_start].
    # MemCam excludes the four overlapping anchor frames
    # [section_start - 3, section_start] from retrieval.
    return list(range(0, section_start - 3))


def capped_candidates(candidates, size):
    if size is None or len(candidates) <= int(size):
        return list(candidates)
    return list(candidates[-int(size) :])


def normalized_entropy(values):
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return entropy / math.log(len(counts))


def summarize_score_matrix(scores, candidate_frames, target_frame):
    scores = np.asarray(scores, dtype=np.float64)
    candidate_frames = np.asarray(candidate_frames, dtype=np.int64)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [repeats, candidates]")
    if scores.shape[1] != len(candidate_frames):
        raise ValueError("score columns must match candidate_frames")
    if scores.shape[0] < 2:
        raise ValueError("At least two repeats are required")
    if scores.shape[1] == 0:
        raise ValueError("At least one candidate is required")

    repeats = scores.shape[0]
    winner_columns = np.argmax(scores, axis=1)
    winner_frames = candidate_frames[winner_columns]
    winner_ages = int(target_frame) - winner_frames
    counts = Counter(int(frame) for frame in winner_frames)
    modal_frame, modal_count = counts.most_common(1)[0]

    mean_scores = scores.mean(axis=0)
    reference_column = int(np.argmax(mean_scores))
    reference_frame = int(candidate_frames[reference_column])
    sorted_reference = np.sort(mean_scores)
    reference_margin = (
        float(sorted_reference[-1] - sorted_reference[-2])
        if scores.shape[1] > 1
        else float("nan")
    )

    top1_margins = []
    optimism = []
    leave_one_out_regret = []
    score_sums = scores.sum(axis=0)
    for repeat_idx, winner_column in enumerate(winner_columns):
        row = scores[repeat_idx]
        if len(row) > 1:
            top_two = np.partition(row, -2)[-2:]
            top1_margins.append(float(top_two.max() - top_two.min()))
        else:
            top1_margins.append(float("nan"))

        other_mean = (score_sums - row) / float(repeats - 1)
        optimism.append(float(row[winner_column] - other_mean[winner_column]))
        leave_one_out_regret.append(
            float(other_mean.max() - other_mean[winner_column])
        )

    winner_frame_min = int(winner_frames.min())
    winner_frame_max = int(winner_frames.max())
    winner_age_min = int(winner_ages.min())
    winner_age_max = int(winner_ages.max())
    return {
        "repeat_count": int(repeats),
        "candidate_count": int(scores.shape[1]),
        "unique_winner_count": int(len(counts)),
        "winner_entropy": float(normalized_entropy(winner_frames.tolist())),
        "modal_winner_frame": int(modal_frame),
        "modal_winner_share": float(modal_count / repeats),
        "reference_winner_frame": reference_frame,
        "reference_winner_age": int(target_frame) - reference_frame,
        "reference_winner_share": float(np.mean(winner_frames == reference_frame)),
        "reference_overlap_mean": float(mean_scores[reference_column]),
        "reference_top1_margin": reference_margin,
        "noisy_top1_margin_mean": float(np.nanmean(top1_margins)),
        "noisy_max_optimism_mean": float(np.mean(optimism)),
        "leave_one_out_regret_mean": float(np.mean(leave_one_out_regret)),
        "winner_frame_min": winner_frame_min,
        "winner_frame_max": winner_frame_max,
        "winner_frame_span": winner_frame_max - winner_frame_min,
        "winner_age_mean": float(np.mean(winner_ages)),
        "winner_age_std": float(np.std(winner_ages)),
        "winner_age_min": winner_age_min,
        "winner_age_max": winner_age_max,
        "winner_age_span": winner_age_max - winner_age_min,
        "cross_section_winner_switch": int(
            winner_age_max - winner_age_min >= SECTION_STRIDE
        ),
        "winner_frames": ";".join(str(int(frame)) for frame in winner_frames),
        "winner_counts": ";".join(
            f"{frame}:{count}" for frame, count in sorted(counts.items())
        ),
        "_reference_scores": mean_scores,
    }


def add_label_fields(summary, labeled_overlaps, candidate_frames):
    output = dict(summary)
    if labeled_overlaps is None:
        output.update(
            {
                "overlap_label_available": 0,
                "candidate_labeled_overlap_count": None,
                "winner_labeled_overlap_rate": None,
                "reference_winner_labeled_overlap": None,
            }
        )
        return output

    candidates = set(int(frame) for frame in candidate_frames)
    winner_frames = [
        int(frame)
        for frame in str(output.get("winner_frames", "")).split(";")
        if frame != ""
    ]
    output.update(
        {
            "overlap_label_available": 1,
            "candidate_labeled_overlap_count": len(candidates & labeled_overlaps),
            "winner_labeled_overlap_rate": (
                float(np.mean([frame in labeled_overlaps for frame in winner_frames]))
                if winner_frames
                else None
            ),
            "reference_winner_labeled_overlap": int(
                int(output["reference_winner_frame"]) in labeled_overlaps
            ),
        }
    )
    return output


def add_actual_trace_fields(
    summary,
    actual_row,
    candidate_frames,
    labeled_overlaps=None,
):
    output = dict(summary)
    reference_scores = output.pop("_reference_scores", None)
    output.update(
        {
            "actual_trace_available": int(actual_row is not None),
            "actual_selected_frame": None,
            "actual_selected_age": None,
            "actual_selected_overlap": None,
            "actual_reference_overlap_mean": None,
            "actual_reference_regret": None,
            "actual_reference_rank": None,
            "actual_is_reference_winner": None,
            "actual_selected_labeled_overlap": None,
        }
    )
    if actual_row is None or reference_scores is None:
        return output

    actual_frame = actual_row.get("selected_memory_frame")
    if actual_frame is None:
        return output
    actual_frame = int(actual_frame)
    candidate_to_column = {
        int(frame): column for column, frame in enumerate(candidate_frames)
    }
    actual_column = candidate_to_column.get(actual_frame)
    if actual_column is None:
        return output

    actual_score = float(reference_scores[actual_column])
    descending = np.argsort(-reference_scores)
    rank = int(np.where(descending == actual_column)[0][0]) + 1
    output.update(
        {
            "actual_selected_frame": actual_frame,
            "actual_selected_age": int(actual_row.get("memory_age", 0)),
            "actual_selected_overlap": float(
                actual_row.get("selected_overlap", float("nan"))
            ),
            "actual_reference_overlap_mean": actual_score,
            "actual_reference_regret": float(reference_scores.max() - actual_score),
            "actual_reference_rank": rank,
            "actual_is_reference_winner": int(
                actual_frame == output["reference_winner_frame"]
            ),
            "actual_selected_labeled_overlap": (
                int(actual_frame in labeled_overlaps)
                if labeled_overlaps is not None
                else None
            ),
        }
    )
    return output


def load_overlap_module():
    path = REPO_ROOT / "diffsynth" / "models" / "wan_video_overlap.py"
    spec = importlib.util.spec_from_file_location("memcam_overlap_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load overlap module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest(path, duration=None, selected_rows=None, max_rows=None):
    rows = []
    selected_rows = set(selected_rows) if selected_rows is not None else None
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_row"] = row_idx
            if duration is not None and int(item.get("duration_sec", -1)) != int(duration):
                continue
            if selected_rows is not None and row_idx not in selected_rows:
                continue
            rows.append(item)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    return rows


def resolve_pose_path(item, dataset_root=None):
    original = Path(item["pose_path"])
    if original.is_file():
        return original
    if dataset_root is not None:
        remapped = Path(dataset_root) / "jsons" / f"{item['scene']}.json"
        if remapped.is_file():
            return remapped
    raise FileNotFoundError(
        f"Pose file not found for row {item['_row']}: {original}. "
        "Use --dataset_root to remap the manifest to this machine."
    )


def resolve_overlap_dir(item, dataset_root=None):
    if dataset_root is not None:
        remapped = Path(dataset_root) / "overlap_labels" / item["scene"]
        if remapped.is_dir():
            return remapped
    original = item.get("overlap_dir")
    if original and Path(original).is_dir():
        return Path(original)
    return None


def extract_overlap_indices(data):
    if isinstance(data, dict):
        for key in ("overlapping_frames", "overlap_frames", "frames", "indices"):
            if key in data:
                return extract_overlap_indices(data[key])
        for key in ("frame_idx", "index"):
            if key in data:
                return [int(data[key])]
        return []
    if isinstance(data, list):
        indices = []
        for item in data:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str) and item.lstrip("-").isdigit():
                indices.append(int(item))
            elif isinstance(item, dict):
                indices.extend(extract_overlap_indices(item))
        return indices
    return []


def load_labeled_overlaps(overlap_dir, target_frame, start_frame, num_frames):
    if overlap_dir is None:
        return None
    global_target = int(start_frame) + int(target_frame)
    path = Path(overlap_dir) / f"{global_target}.json"
    if not path.is_file():
        path = Path(overlap_dir) / f"{int(target_frame)}.json"
    if not path.is_file():
        return None

    raw = extract_overlap_indices(json.loads(path.read_text(encoding="utf-8")))
    local = {
        int(frame) - int(start_frame)
        for frame in raw
        if 0 <= int(frame) - int(start_frame) < int(num_frames)
    }
    if not local:
        local = {
            int(frame) for frame in raw if 0 <= int(frame) < int(num_frames)
        }
    return local


def resolve_trace_dir(path):
    if path is None:
        return None
    path = Path(path)
    if path.name == "access_traces":
        return path
    nested = path / "access_traces"
    return nested if nested.is_dir() else path


def load_actual_trace_rows(trace_dir):
    if trace_dir is None:
        return {}
    trace_dir = resolve_trace_dir(trace_dir)
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {trace_dir}")

    rows = {}
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "context_access" or not row.get("selected"):
                    continue
                key = trace_query_key(row)
                if key is None:
                    continue
                rows[key] = row
    return rows


def collect_score_matrix(
    overlap_module,
    c2ws,
    target_frame,
    candidate_frames,
    repeats,
    num_samples,
    fov_half_h,
    fov_half_v,
    radius,
    seed,
):
    import torch

    scores = np.empty((int(repeats), len(candidate_frames)), dtype=np.float64)
    target_c2w = c2ws[int(target_frame)]
    for repeat_idx in range(int(repeats)):
        torch.manual_seed(int(seed) + repeat_idx)
        for candidate_column, candidate_frame in enumerate(candidate_frames):
            scores[repeat_idx, candidate_column] = (
                overlap_module.calculate_overlap_from_c2w(
                    target_c2w,
                    c2ws[int(candidate_frame)],
                    fov_half_h=float(fov_half_h),
                    fov_half_v=float(fov_half_v),
                    num_samples=int(num_samples),
                    radius=float(radius),
                    return_details=False,
                )
            )
    return scores


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def aggregate_rows(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    metric_fields = [
        "candidate_count",
        "unique_winner_count",
        "winner_entropy",
        "modal_winner_share",
        "reference_winner_share",
        "reference_top1_margin",
        "noisy_top1_margin_mean",
        "noisy_max_optimism_mean",
        "leave_one_out_regret_mean",
        "winner_age_mean",
        "winner_age_std",
        "winner_age_span",
        "cross_section_winner_switch",
        "actual_reference_regret",
        "actual_reference_rank",
        "actual_is_reference_winner",
        "candidate_labeled_overlap_count",
        "winner_labeled_overlap_rate",
        "reference_winner_labeled_overlap",
        "actual_selected_labeled_overlap",
    ]
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        summary = dict(zip(group_fields, key))
        summary["queries"] = len(group)
        for field in metric_fields:
            summary[f"{field}_mean"] = mean_or_none(
                [row.get(field) for row in group]
            )
        output.append(summary)
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field.startswith("_") or field in fields:
                continue
            fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: value
                for field, value in row.items()
                if not field.startswith("_")
            }
            for row in rows
        )


def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows, fields):
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value if value is not None else "NA"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_figure(section_summary, output_dir):
    if not section_summary:
        return None
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = []
    for row in section_summary:
        if row["pool"] not in pools:
            pools.append(row["pool"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("modal_winner_share_mean", "Top-1 stability", "modal winner share"),
        (
            "noisy_max_optimism_mean_mean",
            "Noisy maximum optimism",
            "estimated overlap inflation",
        ),
        ("winner_age_span_mean", "Temporal ambiguity", "winner age span (frames)"),
    ]
    for ax, (field, title, ylabel) in zip(axes, panels):
        for pool in pools:
            points = [row for row in section_summary if row["pool"] == pool]
            points.sort(key=lambda row: int(row["section_idx"]))
            ax.plot(
                [int(row["section_idx"]) for row in points],
                [row.get(field) for row in points],
                marker="o",
                linewidth=2,
                label=pool,
            )
        ax.set_title(title)
        ax.set_xlabel("generation section")
        ax.set_ylabel(ylabel)
        ax.grid(color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "retrieval_instability.png"
    pdf = output_dir / "retrieval_instability.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def write_report(path, query_rows, overall_summary, args):
    all_rows = [row for row in query_rows if row["pool"] == "all"]
    cross_switch = mean_or_none(
        [row["cross_section_winner_switch"] for row in all_rows]
    )
    modal_share = mean_or_none([row["modal_winner_share"] for row in all_rows])
    optimism = mean_or_none(
        [row["noisy_max_optimism_mean"] for row in all_rows]
    )
    actual_regret = mean_or_none(
        [row.get("actual_reference_regret") for row in all_rows]
    )
    labeled_hit_rate = mean_or_none(
        [row.get("winner_labeled_overlap_rate") for row in all_rows]
    )
    actual_labeled_hit_rate = mean_or_none(
        [row.get("actual_selected_labeled_overlap") for row in all_rows]
    )

    lines = [
        "# Unbounded Retrieval Instability Audit",
        "",
        "## What This Tests",
        "",
        "MemCam always retrieves one context frame per target. Unbounded memory does not enlarge the context tensor; it enlarges the candidate search pool. The current retriever estimates camera-FOV IoU with fresh Monte Carlo samples for every candidate and takes the largest estimate.",
        "",
        "This audit holds poses and candidate frames fixed, repeats that exact stochastic argmax, and measures whether the selected memory changes. It is a retrieval diagnostic, not yet evidence that instability causes FVD or LPIPS degradation.",
        "",
        "## Pilot Readout",
        "",
        f"- Audited target queries: {len(all_rows)} across {len(set(row['row'] for row in query_rows))} manifest rows.",
        f"- Candidate-pool evaluations: {len(query_rows)}.",
        f"- Mean modal-winner share for the full unbounded pool: {fmt(modal_share)}. One means perfectly stable.",
        f"- Full-pool queries whose repeated winners differ by at least one 76-frame section: {fmt(cross_switch)}.",
        f"- Mean noisy-maximum optimism for the full pool: {fmt(optimism, 6)} IoU.",
        f"- Mean robust-score regret of the winner recorded in the real trace: {fmt(actual_regret, 6)} IoU.",
        f"- Replayed full-pool winners accepted by the dataset overlap labels: {fmt(labeled_hit_rate)}.",
        f"- Real traced winners accepted by the dataset overlap labels: {fmt(actual_labeled_hit_rate)}.",
        "",
        "## Overall By Candidate Pool",
        "",
        markdown_table(
            overall_summary,
            [
                "pool",
                "queries",
                "candidate_count_mean",
                "modal_winner_share_mean",
                "noisy_max_optimism_mean_mean",
                "winner_age_span_mean",
                "cross_section_winner_switch_mean",
                "winner_labeled_overlap_rate_mean",
                "actual_reference_regret_mean",
                "actual_selected_labeled_overlap_mean",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Falling top-1 stability as the pool grows supports retrieval ambiguity or estimator noise.",
        "- Compare noisy-maximum optimism across pools before claiming a multiple-comparisons effect; an increase is required for that interpretation.",
        "- A large winner-age span means the noise changes which temporal episode is retrieved, rather than merely swapping adjacent frames.",
        "- Dataset-label agreement tests whether the stochastic retriever selects a frame that training itself regarded as valid context. The labels are a reference criterion, not a guarantee of visual quality.",
        "- None of these alone proves quality causality. The causal follow-up is a matched rollout using deterministic common overlap samples while leaving the unbounded bank unchanged.",
        "",
        "## Configuration",
        "",
        f"- Repeats: `{args.repeats}`",
        f"- Samples per pair: `{args.num_samples}` (MemCam default is 5000)",
        f"- Candidate pools: `{args.pool_sizes}`",
        f"- Target stride: `{args.target_stride}`",
        f"- Sections: `{args.sections}`",
        "",
        "## Outputs",
        "",
        "- `tables/query_audit.csv`",
        "- `tables/section_summary.csv`",
        "- `tables/overall_summary.csv`",
        "- `figures/retrieval_instability.png`",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay MemCam's stochastic overlap argmax to test whether an "
            "unbounded candidate pool makes retrieval unstable."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace_dir", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--sections", type=str, default="auto")
    parser.add_argument("--section_count", type=int, default=5)
    parser.add_argument("--target_stride", type=int, default=19)
    parser.add_argument("--pool_sizes", type=str, default="32,128,all")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--fov_half_h", type=float, default=45.0)
    parser.add_argument("--fov_half_v", type=float, default=30.0)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("--repeats must be at least two")
    if args.target_stride <= 0:
        raise ValueError("--target_stride must be positive")

    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))
    overlap_module = load_overlap_module()
    pools = parse_pool_sizes(args.pool_sizes)
    manifest_rows = load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=parse_int_ranges(args.rows),
        max_rows=args.max_rows,
    )
    if not manifest_rows:
        raise RuntimeError("No manifest rows matched the requested filters")
    actual_traces = load_actual_trace_rows(args.trace_dir)

    query_rows = []
    for manifest_position, item in enumerate(manifest_rows):
        row_idx = int(item["_row"])
        total_sections = (int(item["num_frames"]) - 1) // SECTION_STRIDE
        if args.sections.strip().lower() == "auto":
            sections = auto_sections(total_sections, args.section_count)
        else:
            sections = [
                section
                for section in parse_int_ranges(args.sections)
                if 0 < section < total_sections
            ]
        pose_path = resolve_pose_path(item, dataset_root=args.dataset_root)
        overlap_dir = resolve_overlap_dir(item, dataset_root=args.dataset_root)
        all_c2ws, _ = overlap_module.load_poses_from_json(str(pose_path))
        start_frame = int(item["start_frame"])
        num_frames = int(item["num_frames"])
        c2ws = all_c2ws[start_frame : start_frame + num_frames]
        if len(c2ws) != num_frames:
            raise RuntimeError(
                f"Row {row_idx} expected {num_frames} poses, found {len(c2ws)}"
            )

        print(
            f"row {row_idx} {item['scene']}: sections={sections}, "
            f"frames={num_frames}"
        )
        for section_idx in sections:
            section_start = section_idx * SECTION_STRIDE
            base_candidates = unbounded_candidates(section_idx)
            target_frames = range(
                section_start + 1,
                section_start + PREDICT_FRAMES + 1,
                int(args.target_stride),
            )
            for target_frame in target_frames:
                labeled_overlaps = load_labeled_overlaps(
                    overlap_dir=overlap_dir,
                    target_frame=target_frame,
                    start_frame=start_frame,
                    num_frames=num_frames,
                )
                for pool_position, (pool_label, pool_size) in enumerate(pools):
                    candidates = capped_candidates(base_candidates, pool_size)
                    query_seed = (
                        int(args.seed)
                        + manifest_position * 10_000_019
                        + section_idx * 100_003
                        + target_frame * 101
                        + pool_position * 1_009
                    )
                    scores = collect_score_matrix(
                        overlap_module=overlap_module,
                        c2ws=c2ws,
                        target_frame=target_frame,
                        candidate_frames=candidates,
                        repeats=args.repeats,
                        num_samples=args.num_samples,
                        fov_half_h=args.fov_half_h,
                        fov_half_v=args.fov_half_v,
                        radius=args.radius,
                        seed=query_seed,
                    )
                    summary = summarize_score_matrix(
                        scores,
                        candidate_frames=candidates,
                        target_frame=target_frame,
                    )
                    summary = add_label_fields(
                        summary,
                        labeled_overlaps=labeled_overlaps,
                        candidate_frames=candidates,
                    )
                    actual_row = None
                    if pool_label == "all":
                        actual_row = actual_traces.get(
                            manifest_query_key(item, section_idx, target_frame)
                        )
                    summary = add_actual_trace_fields(
                        summary,
                        actual_row=actual_row,
                        candidate_frames=candidates,
                        labeled_overlaps=labeled_overlaps,
                    )
                    summary = {
                        "row": row_idx,
                        "scene": item["scene"],
                        "dataset_start_frame": start_frame,
                        "duration_sec": int(item["duration_sec"]),
                        "output_prefix": item.get("output_prefix"),
                        "section_idx": section_idx,
                        "target_frame": target_frame,
                        "target_dataset_frame": start_frame + target_frame,
                        "pool": pool_label,
                        "requested_pool_size": pool_size,
                        **summary,
                    }
                    query_rows.append(summary)
                    print(
                        f"  section={section_idx} target={target_frame} "
                        f"pool={pool_label} candidates={len(candidates)} "
                        f"modal_share={summary['modal_winner_share']:.2f} "
                        f"age_span={summary['winner_age_span']}"
                    )

    section_summary = aggregate_rows(query_rows, ["pool", "section_idx"])
    overall_summary = aggregate_rows(query_rows, ["pool"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    write_csv(tables / "query_audit.csv", query_rows)
    write_csv(tables / "section_summary.csv", section_summary)
    write_csv(tables / "overall_summary.csv", overall_summary)
    save_figure(section_summary, figures)
    write_report(
        args.output_dir / "report.md",
        query_rows=query_rows,
        overall_summary=overall_summary,
        args=args,
    )
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
