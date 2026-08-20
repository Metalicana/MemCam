"""Measure retrieved-memory image quality and following-section quality.

This is CPU-only. For each real context retrieval, the selected generated
memory frame is compared with the dataset frame at the same trajectory index.
Repeated selections remain repeated in the primary statistics because they
represent repeated exposure to that memory content.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_context_memory import frame_metrics  # noqa: E402


SECTION_STRIDE = 76
PREDICT_FRAMES = 76


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


def load_selected_frames_by_section(path):
    output = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "context_access" or not row.get("selected"):
                continue
            output[int(row["section_idx"])].append(
                int(row["selected_memory_frame"])
            )
    return output


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


def requested_chunk_frames(section_indices, frame_stride):
    output = defaultdict(list)
    for section_idx in section_indices:
        start = int(section_idx) * SECTION_STRIDE + 1
        output[int(section_idx)] = list(
            range(start, start + PREDICT_FRAMES, int(frame_stride))
        )
    return output


def compute_requested_frame_metrics(video_path, item, requested_frames):
    import imageio.v2 as imageio

    requested_frames = {int(frame) for frame in requested_frames}
    if not requested_frames:
        return {}
    last_frame = max(requested_frames)
    output = {}
    reader = imageio.get_reader(str(video_path))
    try:
        for frame_idx, generated in enumerate(reader):
            if frame_idx > last_frame:
                break
            if frame_idx not in requested_frames:
                continue
            generated = np.asarray(generated, dtype=np.uint8)
            gt = read_gt_frame(item, frame_idx, generated.shape)
            output[frame_idx] = frame_metrics(generated, gt)
    finally:
        reader.close()
    missing = sorted(requested_frames - set(output))
    if missing:
        raise RuntimeError(
            f"Video {video_path} is missing requested frames {missing[:10]}"
        )
    return output


def percentile(values, q):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(values, q)) if values else None


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def bootstrap_mean_interval(values, repeats=10000, seed=0):
    values = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
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


def exact_two_sided_sign_pvalue(wins, losses):
    trials = int(wins) + int(losses)
    if trials == 0:
        return None
    smaller = min(int(wins), int(losses))
    tail = sum(math.comb(trials, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def metric_summary(prefix, metrics):
    output = {}
    for field in ("psnr_db", "ssim"):
        values = [row[field] for row in metrics]
        output[f"{prefix}_{field}_mean"] = mean(values)
        output[f"{prefix}_{field}_median"] = percentile(values, 0.5)
        output[f"{prefix}_{field}_p10"] = percentile(values, 0.1)
    return output


def build_section_row(
    run_name,
    item,
    section_idx,
    selected_frames,
    chunk_frames,
    frame_metric_map,
):
    selected_metrics = [frame_metric_map[frame] for frame in selected_frames]
    unique_selected = sorted(set(selected_frames))
    unique_metrics = [frame_metric_map[frame] for frame in unique_selected]
    chunk_metrics = [frame_metric_map[frame] for frame in chunk_frames]
    return {
        "run_name": run_name,
        "row": int(item["_row"]),
        "scene": item["scene"],
        "start_frame": int(item["start_frame"]),
        "duration_sec": int(item["duration_sec"]),
        "section_idx": int(section_idx),
        "section_time_sec": float(section_idx * SECTION_STRIDE / item["fps"]),
        "retrievals": len(selected_frames),
        "unique_selected_frames": len(unique_selected),
        "chunk_frames_evaluated": len(chunk_frames),
        **metric_summary("selected_weighted", selected_metrics),
        **metric_summary("selected_unique", unique_metrics),
        **metric_summary("following_chunk", chunk_metrics),
    }


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


def aggregate_run_rows(section_rows):
    grouped = defaultdict(list)
    for row in section_rows:
        grouped[row["run_name"]].append(row)
    output = []
    for run_name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["row"], row["section_idx"]))
        late_start = np.quantile(
            [row["section_idx"] for row in rows], 0.75
        )
        late_rows = [row for row in rows if row["section_idx"] >= late_start]
        output.append(
            {
                "run_name": run_name,
                "videos": len({row["row"] for row in rows}),
                "sections": len(rows),
                "selected_psnr_mean": mean(
                    row["selected_weighted_psnr_db_mean"] for row in rows
                ),
                "selected_psnr_p10_mean": mean(
                    row["selected_weighted_psnr_db_p10"] for row in rows
                ),
                "selected_ssim_mean": mean(
                    row["selected_weighted_ssim_mean"] for row in rows
                ),
                "selected_ssim_p10_mean": mean(
                    row["selected_weighted_ssim_p10"] for row in rows
                ),
                "late_selected_psnr_mean": mean(
                    row["selected_weighted_psnr_db_mean"] for row in late_rows
                ),
                "late_selected_ssim_mean": mean(
                    row["selected_weighted_ssim_mean"] for row in late_rows
                ),
                "following_chunk_psnr_mean": mean(
                    row["following_chunk_psnr_db_mean"] for row in rows
                ),
                "following_chunk_ssim_mean": mean(
                    row["following_chunk_ssim_mean"] for row in rows
                ),
                "late_following_chunk_psnr_mean": mean(
                    row["following_chunk_psnr_db_mean"] for row in late_rows
                ),
                "late_following_chunk_ssim_mean": mean(
                    row["following_chunk_ssim_mean"] for row in late_rows
                ),
            }
        )
    return output


def paired_policy_rows(section_rows, baseline_run):
    by_key = {
        (row["run_name"], int(row["row"]), int(row["section_idx"])): row
        for row in section_rows
    }
    run_names = sorted({row["run_name"] for row in section_rows})
    fields = (
        "selected_weighted_psnr_db_mean",
        "selected_weighted_ssim_mean",
        "following_chunk_psnr_db_mean",
        "following_chunk_ssim_mean",
    )
    output = []
    for run_name in run_names:
        if run_name == baseline_run:
            continue
        shared = []
        for (candidate_run, row_idx, section_idx), policy_row in by_key.items():
            if candidate_run != run_name:
                continue
            baseline_row = by_key.get((baseline_run, row_idx, section_idx))
            if baseline_row is not None:
                shared.append((policy_row, baseline_row))
        if not shared:
            continue

        result = {
            "run_name": run_name,
            "baseline_run": baseline_run,
            "paired_sections": len(shared),
            "paired_videos": len({row[0]["row"] for row in shared}),
        }
        for field in fields:
            deltas = [
                float(policy[field]) - float(baseline[field])
                for policy, baseline in shared
            ]
            trajectory_deltas = defaultdict(list)
            for (policy, _baseline), delta in zip(shared, deltas):
                trajectory_deltas[int(policy["row"])].append(delta)
            trajectory_means = [mean(values) for values in trajectory_deltas.values()]
            wins = sum(value > 1e-12 for value in trajectory_means)
            losses = sum(value < -1e-12 for value in trajectory_means)
            ties = len(trajectory_means) - wins - losses
            ci_low, ci_high = bootstrap_mean_interval(trajectory_means)
            prefix = field.removesuffix("_mean")
            result[f"{prefix}_delta_mean"] = mean(trajectory_means)
            result[f"{prefix}_delta_ci_low"] = ci_low
            result[f"{prefix}_delta_ci_high"] = ci_high
            result[f"{prefix}_trajectory_wins"] = wins
            result[f"{prefix}_trajectory_losses"] = losses
            result[f"{prefix}_trajectory_ties"] = ties
            result[f"{prefix}_sign_pvalue"] = exact_two_sided_sign_pvalue(
                wins, losses
            )
        output.append(result)
    return output


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_report(path, run_rows, paired_rows, baseline_run, chunk_frame_stride):
    lines = [
        "# Retrieved Memory Image Quality",
        "",
        "## Question",
        "",
        "Does unbounded memory retrieve more corrupted generated images than bounded memory, and is the following generated chunk also worse?",
        "",
        "Each retrieved generated frame is compared with the dataset ground-truth frame at the same trajectory index. Repeated retrievals are counted repeatedly because they are repeated conditioning exposures. Higher PSNR and SSIM are better.",
        "",
        "The following-chunk metrics compare generated output with dataset ground truth every "
        f"{int(chunk_frame_stride)} frames in the section conditioned on those memories.",
        "",
        "## Run Summary",
        "",
        "| run | videos | sections | selected PSNR | selected SSIM | late selected PSNR | late selected SSIM | late next-chunk PSNR | late next-chunk SSIM |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['run_name']} | {row['videos']} | {row['sections']} | "
            f"{fmt(row['selected_psnr_mean'], 3)} | {fmt(row['selected_ssim_mean'])} | "
            f"{fmt(row['late_selected_psnr_mean'], 3)} | {fmt(row['late_selected_ssim_mean'])} | "
            f"{fmt(row['late_following_chunk_psnr_mean'], 3)} | "
            f"{fmt(row['late_following_chunk_ssim_mean'])} |"
        )

    lines.extend(
        [
            "",
            f"## Paired Difference Versus {baseline_run}",
            "",
            "Deltas are bounded minus unbounded on matched trajectory-sections. Positive is better for both PSNR and SSIM. Confidence intervals resample trajectories, not individual retrievals.",
            "",
            "| run | trajectories | selected PSNR delta | 95% CI | selected SSIM delta | 95% CI | next-chunk PSNR delta | 95% CI | next-chunk SSIM delta | 95% CI |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['run_name']} | {row['paired_videos']} | "
            f"{fmt(row['selected_weighted_psnr_db_delta_mean'], 3)} | "
            f"[{fmt(row['selected_weighted_psnr_db_delta_ci_low'], 3)}, {fmt(row['selected_weighted_psnr_db_delta_ci_high'], 3)}] | "
            f"{fmt(row['selected_weighted_ssim_delta_mean'])} | "
            f"[{fmt(row['selected_weighted_ssim_delta_ci_low'])}, {fmt(row['selected_weighted_ssim_delta_ci_high'])}] | "
            f"{fmt(row['following_chunk_psnr_db_delta_mean'], 3)} | "
            f"[{fmt(row['following_chunk_psnr_db_delta_ci_low'], 3)}, {fmt(row['following_chunk_psnr_db_delta_ci_high'], 3)}] | "
            f"{fmt(row['following_chunk_ssim_delta_mean'])} | "
            f"[{fmt(row['following_chunk_ssim_delta_ci_low'])}, {fmt(row['following_chunk_ssim_delta_ci_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Cleaner selected memories in a bounded run support the memory-corruption explanation.",
            "- Cleaner selected memories together with cleaner following chunks support, but do not by themselves prove, corruption propagation.",
            "- The matched GT-cleaning replay is the causal test: it keeps selected frame identities fixed and changes only their image content.",
            "- PSNR and SSIM are valid here because the dataset supplies the same camera trajectory and exact frame index, but they remain pixel-level measures and can penalize plausible appearance changes.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--chunk_frame_stride", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.chunk_frame_stride < 1:
        raise ValueError("--chunk_frame_stride must be positive")
    runs = parse_list(args.runs)
    items = load_manifest(args.manifest, args.duration)
    section_rows = []

    for run_name in runs:
        print(f"=== {run_name} ===")
        for item in items:
            video_path = args.root / run_name / f"{item['output_prefix']}custom.mp4"
            trace_path = (
                args.root
                / run_name
                / "access_traces"
                / f"{item['output_prefix']}custom.jsonl"
            )
            if not video_path.is_file() or not trace_path.is_file():
                message = f"missing video or trace for {run_name} row {item['_row']}"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue

            selected_by_section = load_selected_frames_by_section(trace_path)
            if not selected_by_section:
                print(f"[skip] no selected contexts for {run_name} row {item['_row']}")
                continue
            chunk_by_section = requested_chunk_frames(
                selected_by_section, args.chunk_frame_stride
            )
            requested = {
                frame
                for frames in selected_by_section.values()
                for frame in frames
            }
            requested.update(
                frame for frames in chunk_by_section.values() for frame in frames
            )
            frame_metric_map = compute_requested_frame_metrics(
                video_path, item, requested
            )
            for section_idx in sorted(selected_by_section):
                section_rows.append(
                    build_section_row(
                        run_name,
                        item,
                        section_idx,
                        selected_by_section[section_idx],
                        chunk_by_section[section_idx],
                        frame_metric_map,
                    )
                )
            print(
                f"row {item['_row']}: {len(selected_by_section)} sections, "
                f"{len(frame_metric_map)} unique frames measured"
            )

    if not section_rows:
        raise RuntimeError("No selected-memory quality rows were produced")
    run_rows = aggregate_run_rows(section_rows)
    paired_rows = paired_policy_rows(section_rows, args.baseline_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "section_quality.csv", section_rows)
    write_csv(args.output_dir / "run_summary.csv", run_rows)
    write_csv(args.output_dir / "paired_vs_unbounded.csv", paired_rows)
    write_report(
        args.output_dir / "report.md",
        run_rows,
        paired_rows,
        baseline_run=args.baseline_run,
        chunk_frame_stride=args.chunk_frame_stride,
    )

    print("\nHigher PSNR/SSIM is better.")
    for row in run_rows:
        print(
            f"{row['run_name']}: "
            f"selected PSNR={row['selected_psnr_mean']:.3f} "
            f"SSIM={row['selected_ssim_mean']:.4f}; "
            f"late selected PSNR={row['late_selected_psnr_mean']:.3f} "
            f"SSIM={row['late_selected_ssim_mean']:.4f}; "
            f"late next-chunk PSNR={row['late_following_chunk_psnr_mean']:.3f} "
            f"SSIM={row['late_following_chunk_ssim_mean']:.4f}"
        )
    print(f"\nWrote: {args.output_dir / 'section_quality.csv'}")
    print(f"Wrote: {args.output_dir / 'run_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'paired_vs_unbounded.csv'}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
