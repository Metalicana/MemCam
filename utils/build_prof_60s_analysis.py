import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "memcam_matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "memcam_xdg_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required for the professor analysis figures. "
        "Install it in the memcam env with: python -m pip install matplotlib"
    ) from exc


QUALITY_FIELDS = [
    ("lpips_alex", "LPIPS ↓"),
    ("fvd", "FVD ↓"),
    ("dino_distance", "DINO distance ↓"),
    ("clip_image_distance", "CLIP image distance ↓"),
    ("psnr_db", "PSNR ↑"),
    ("ssim", "SSIM ↑"),
]

LOWER_IS_BETTER = {
    "lpips_alex",
    "fvd",
    "dino_distance",
    "clip_image_distance",
    "mae",
    "mse",
    "rmse",
    "temporal_delta_mae",
    "temporal_delta_rmse",
}

MEMORY_FIELDS = [
    ("candidate_count_mean", "candidate count"),
    ("stored_memory_size_mean", "stored memory size"),
    ("selected_age_mean", "selected age"),
    ("age_mean", "selected age"),
    ("selected_overlap_mean", "selected overlap"),
    ("overlap_mean", "selected overlap"),
    ("reuse_gini", "reuse gini"),
    ("section_effective_selected_frames_mean", "effective frames / section"),
    ("section_top1_selected_frac_mean", "top-1 retrieval fraction"),
    ("mean_overlap_capture_ratio", "overlap capture ratio"),
    ("mean_overlap_gap", "overlap gap vs unbounded"),
]


def parse_list(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_round(value, digits=4):
    value = safe_float(value)
    if value is None:
        return None
    return round(value, digits)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def run_sort_key(run_name):
    if run_name == "baseline":
        return (0, 0, run_name)
    match = re.search(r"_b(\d+)|b(\d+)", run_name)
    budget = int(next(group for group in match.groups() if group)) if match else 9999
    if run_name.startswith("fifo"):
        family = 1
    elif run_name.startswith("ri"):
        family = 2
    elif run_name.startswith("slam"):
        family = 3
    else:
        family = 9
    return (family, budget, run_name)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(manifest_path):
    rows = {}
    if manifest_path is None or not manifest_path.exists():
        return rows
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_row"] = index
            rows[index] = item
    return rows


def load_quality(metrics_dir, runs, duration):
    summary_rows = []
    video_rows = []
    frame_rows_by_run = {}

    for run_name in runs:
        run_dir = metrics_dir / run_name
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        frame_metrics_path = run_dir / "frame_metrics.jsonl"

        if not summary_path.exists():
            print(f"[warn] missing summary: {summary_path}")
            continue

        summary = read_json(summary_path)
        source = summary.get("by_duration", {}).get(str(duration), summary.get("overall", {}))
        row = {"run_name": run_name}
        for key, value in source.items():
            if isinstance(value, (int, float, str)) or value is None:
                row[key] = value
        summary_rows.append(row)

        for video_row in read_csv(metrics_path):
            if str(video_row.get("duration_sec")) != str(duration):
                continue
            video_row["run_name"] = video_row.get("run_name") or run_name
            video_rows.append(video_row)

        frame_rows = []
        if frame_metrics_path.exists():
            with frame_metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if str(payload.get("duration_sec")) == str(duration):
                        payload["run_name"] = run_name
                        frame_rows.append(payload)
        frame_rows_by_run[run_name] = frame_rows

    return summary_rows, video_rows, frame_rows_by_run


def choose_existing(*paths):
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def load_memory(root):
    access_path = choose_existing(
        root / "access_trace_analysis_complete15" / "access_run_summary.csv",
        root / "access_trace_analysis" / "access_run_summary.csv",
    )
    trace_path = choose_existing(
        root / "trace_usefulness_analysis" / "trace_run_summary.csv",
    )
    access = {row["run_name"]: row for row in read_csv(access_path)}
    trace = {row["run_name"]: row for row in read_csv(trace_path)}
    return access, trace, access_path, trace_path


def merge_quality_memory(summary_rows, access_rows, trace_rows):
    merged = []
    for row in summary_rows:
        run_name = row["run_name"]
        combined = {**row}
        for prefix, source in (("access", access_rows.get(run_name, {})), ("trace", trace_rows.get(run_name, {}))):
            for key, value in source.items():
                if key == "run_name":
                    continue
                target_key = key if key not in combined else f"{prefix}_{key}"
                combined[target_key] = value
        merged.append(combined)
    return sorted(merged, key=lambda item: run_sort_key(item["run_name"]))


def color_for_run(run_name):
    if run_name == "baseline":
        return "#555555"
    if run_name.startswith("fifo"):
        return "#2f7fbc"
    if run_name.startswith("ri"):
        return "#d88c1f"
    if run_name.startswith("slam"):
        return "#4f9b62"
    return "#8a5fbf"


def setup_axis(ax, title=None):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def save_bar_quality(summary_rows, figures_dir):
    available = [
        (field, label)
        for field, label in QUALITY_FIELDS
        if any(safe_float(row.get(field)) is not None for row in summary_rows)
    ]
    if not available:
        return None

    fields = [field for field, _label in available[:4]]
    labels = [label for _field, label in available[:4]]
    runs = [row["run_name"] for row in summary_rows]

    fig, axes = plt.subplots(1, len(fields), figsize=(5.4 * len(fields), 4.6), squeeze=False)
    axes = axes[0]
    for ax, field, label in zip(axes, fields, labels):
        values = [safe_float(row.get(field)) for row in summary_rows]
        ax.bar(runs, [value if value is not None else 0 for value in values], color=[color_for_run(run) for run in runs])
        ax.set_ylabel(label)
        setup_axis(ax, label)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        baseline = next((safe_float(row.get(field)) for row in summary_rows if row["run_name"] == "baseline"), None)
        if baseline is not None:
            ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1.1)
    fig.tight_layout()
    path = figures_dir / "01_main_quality_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_delta_vs_baseline(summary_rows, figures_dir, baseline_run):
    baseline = next((row for row in summary_rows if row["run_name"] == baseline_run), None)
    if baseline is None:
        return None
    fields = [
        field
        for field, _label in QUALITY_FIELDS
        if field in LOWER_IS_BETTER
        and safe_float(baseline.get(field)) not in (None, 0.0)
        and any(row["run_name"] != baseline_run and safe_float(row.get(field)) is not None for row in summary_rows)
    ]
    if not fields:
        return None

    fig, axes = plt.subplots(len(fields), 1, figsize=(10, 3.2 * len(fields)), squeeze=False)
    axes = axes[:, 0]
    runs = [row["run_name"] for row in summary_rows if row["run_name"] != baseline_run]
    for ax, field in zip(axes, fields):
        base = safe_float(baseline.get(field))
        deltas = []
        for run in runs:
            value = next((safe_float(row.get(field)) for row in summary_rows if row["run_name"] == run), None)
            deltas.append(100.0 * (base - value) / base if value is not None else None)
        colors = ["#3b8f4a" if value is not None and value >= 0 else "#c95050" for value in deltas]
        ax.bar(runs, [value if value is not None else 0 for value in deltas], color=colors)
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_ylabel("% improvement vs baseline")
        ax.set_title(f"{field}: positive means better than unbounded baseline", fontsize=12, fontweight="bold")
        setup_axis(ax)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.tight_layout()
    path = figures_dir / "02_percent_improvement_vs_unbounded.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def memory_value(row, candidates):
    for key in candidates:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def save_memory_quality_scatter(merged_rows, figures_dir):
    y_field = "lpips_alex" if any(safe_float(row.get("lpips_alex")) is not None for row in merged_rows) else "fvd"
    if not any(safe_float(row.get(y_field)) is not None for row in merged_rows):
        return None

    panels = [
        (["candidate_count_mean", "access_candidate_count_mean", "trace_candidate_count_mean"], "retrieval candidates"),
        (["selected_age_mean", "age_mean", "access_age_mean", "trace_selected_age_mean"], "selected memory age"),
        (["selected_overlap_mean", "overlap_mean", "access_overlap_mean", "trace_selected_overlap_mean"], "selected overlap"),
        (["reuse_gini", "access_reuse_gini", "trace_reuse_gini"], "reuse concentration"),
        (["section_effective_selected_frames_mean"], "effective frames / section"),
        (["section_top1_selected_frac_mean"], "top-1 retrieval fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes = axes.flatten()
    y_label = "LPIPS ↓" if y_field == "lpips_alex" else "FVD ↓"

    for ax, (keys, label) in zip(axes, panels):
        points = []
        for row in merged_rows:
            x = memory_value(row, keys)
            y = safe_float(row.get(y_field))
            if x is not None and y is not None:
                points.append((row["run_name"], x, y))
        if not points:
            ax.text(0.5, 0.5, "missing data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=12, fontweight="bold")
            continue
        for run_name, x, y in points:
            ax.scatter(x, y, s=90, color=color_for_run(run_name), edgecolor="white", linewidth=0.8)
            ax.annotate(run_name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel(y_label)
        setup_axis(ax, label)

    fig.suptitle("Memory behavior vs visual quality", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = figures_dir / "03_memory_behavior_vs_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def frame_bucket(frame_index, duration_sec):
    if duration_sec <= 0:
        return None
    fps = 30.0
    t = frame_index / fps
    if t < duration_sec / 3:
        return "early"
    if t < 2 * duration_sec / 3:
        return "middle"
    return "late"


def bucket_frame_metrics(frame_rows_by_run, duration):
    output = []
    for run_name, rows in frame_rows_by_run.items():
        grouped = defaultdict(list)
        for row in rows:
            metric = safe_float(row.get("lpips_alex"))
            frame_index = safe_int(row.get("frame_index"))
            if metric is None or frame_index is None:
                continue
            bucket = frame_bucket(frame_index, duration)
            if bucket:
                grouped[bucket].append(metric)
        for bucket in ["early", "middle", "late"]:
            if grouped[bucket]:
                output.append(
                    {
                        "run_name": run_name,
                        "bucket": bucket,
                        "lpips_alex": mean(grouped[bucket]),
                        "frames": len(grouped[bucket]),
                    }
                )
    return output


def save_time_bucket_plot(bucket_rows, figures_dir):
    if not bucket_rows:
        return None
    runs = sorted({row["run_name"] for row in bucket_rows}, key=run_sort_key)
    buckets = ["early", "middle", "late"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for run_name in runs:
        values = []
        for bucket in buckets:
            value = next(
                (
                    safe_float(row.get("lpips_alex"))
                    for row in bucket_rows
                    if row["run_name"] == run_name and row["bucket"] == bucket
                ),
                None,
            )
            values.append(value)
        if any(value is not None for value in values):
            ax.plot(
                buckets,
                values,
                marker="o",
                linewidth=2,
                label=run_name,
                color=color_for_run(run_name),
            )
    ax.set_ylabel("LPIPS ↓")
    setup_axis(ax, "LPIPS by video third")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    path = figures_dir / "04_lpips_by_time_bucket.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def video_key(row):
    return (str(row.get("row")), row.get("scene"), str(row.get("start_frame")), str(row.get("duration_sec")))


def compute_biggest_wins(video_rows, baseline_run, metric="lpips_alex", top_k=8):
    baseline = {
        video_key(row): safe_float(row.get(metric))
        for row in video_rows
        if row.get("run_name") == baseline_run and safe_float(row.get(metric)) is not None
    }
    wins = []
    for row in video_rows:
        run_name = row.get("run_name")
        if run_name == baseline_run:
            continue
        value = safe_float(row.get(metric))
        base = baseline.get(video_key(row))
        if value is None or base is None:
            continue
        improvement = base - value if metric in LOWER_IS_BETTER else value - base
        pct = 100.0 * improvement / base if base not in (None, 0.0) else None
        wins.append(
            {
                "run_name": run_name,
                "row": row.get("row"),
                "scene": row.get("scene"),
                "start_frame": row.get("start_frame"),
                "duration_sec": row.get("duration_sec"),
                f"baseline_{metric}": base,
                f"policy_{metric}": value,
                f"{metric}_improvement": improvement,
                f"{metric}_improvement_pct": pct,
                "output": row.get("output"),
            }
        )
    wins.sort(key=lambda row: safe_float(row.get(f"{metric}_improvement")) or -1e9, reverse=True)
    return wins[:top_k]


def read_video_frame(video_path, frame_index):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None
    try:
        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(int(frame_index))
        reader.close()
    except Exception:
        return None
    return Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).convert("RGB")


def resolve_gt_frame(manifest_rows, win, frame_index, dataset_root):
    item = manifest_rows.get(safe_int(win.get("row")))
    if item is None:
        return None
    if dataset_root is not None:
        frames_dir = dataset_root / "frames" / item["scene"]
    else:
        frames_dir = Path(item["gt_frames_dir"])
    gt_idx = int(item["start_frame"]) + int(frame_index)
    path = frames_dir / f"{gt_idx:04d}.png"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def make_labeled_thumb(image, label, size=(240, 135)):
    if image is None:
        image = Image.new("RGB", size, color=(235, 235, 235))
    else:
        image = image.resize(size, Image.BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + 28), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, size[1] + 7), label[:42], fill=(20, 20, 20))
    return canvas


def save_contact_sheet(wins, video_rows, manifest_rows, root, baseline_run, dataset_root, figures_dir):
    if not wins:
        return None
    baseline_outputs = {
        video_key(row): row.get("output")
        for row in video_rows
        if row.get("run_name") == baseline_run
    }
    rows = []
    for win in wins[:4]:
        item = manifest_rows.get(safe_int(win.get("row")))
        if item is None:
            continue
        frame_index = min(max(int(item["num_frames"]) // 2, 0), int(item["num_frames"]) - 1)
        key = (str(win.get("row")), win.get("scene"), str(win.get("start_frame")), str(win.get("duration_sec")))
        baseline_video = baseline_outputs.get(key)
        policy_video = win.get("output")
        gt = resolve_gt_frame(manifest_rows, win, frame_index, dataset_root)
        baseline_img = read_video_frame(baseline_video, frame_index) if baseline_video else None
        policy_img = read_video_frame(policy_video, frame_index) if policy_video else None
        title = f"row {win.get('row')} {win.get('scene')} | {win.get('run_name')} LPIPS win {safe_round(win.get('lpips_alex_improvement_pct'), 1)}%"
        rows.append((title, [("GT", gt), (baseline_run, baseline_img), (win.get("run_name"), policy_img)]))

    if not rows:
        return None

    thumb_size = (260, 146)
    title_h = 28
    row_h = thumb_size[1] + 28 + title_h + 10
    width = 3 * thumb_size[0]
    height = len(rows) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    y = 0
    for title, images in rows:
        draw.text((6, y + 6), title[:110], fill=(0, 0, 0))
        y += title_h
        x = 0
        for label, image in images:
            thumb = make_labeled_thumb(image, label, size=thumb_size)
            sheet.paste(thumb, (x, y))
            x += thumb_size[0]
        y += thumb_size[1] + 28 + 10

    path = figures_dir / "05_biggest_lpips_wins_contact_sheet.png"
    sheet.save(path)
    return path


def format_value(value, digits=4):
    value = safe_float(value)
    if value is None:
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def best_runs(summary_rows, metric, baseline_run):
    rows = [row for row in summary_rows if safe_float(row.get(metric)) is not None]
    if not rows:
        return None, None, None
    reverse = metric not in LOWER_IS_BETTER
    best = sorted(rows, key=lambda row: safe_float(row.get(metric)), reverse=reverse)[0]
    baseline = next((row for row in rows if row["run_name"] == baseline_run), None)
    improvement = None
    if baseline is not None and best["run_name"] != baseline_run:
        base = safe_float(baseline.get(metric))
        value = safe_float(best.get(metric))
        if base not in (None, 0.0) and value is not None:
            improvement = 100.0 * ((base - value) / base if metric in LOWER_IS_BETTER else (value - base) / base)
    return best, baseline, improvement


def write_report(path, summary_rows, merged_rows, wins, figure_paths, access_path, trace_path, baseline_run):
    lpips_best, lpips_baseline, lpips_improvement = best_runs(summary_rows, "lpips_alex", baseline_run)
    fvd_best, fvd_baseline, fvd_improvement = best_runs(summary_rows, "fvd", baseline_run)

    lines = [
        "# 60s Memory Policy Analysis",
        "",
        "## Main Claim",
        "",
        "Bounded memory can beat the unbounded run on visual metrics because the generator is not a perfect consumer of all retrieved history. More memory increases candidate count and long-range evidence, but it can also inject stale or conflicting context. A bounded policy can act like memory regularization.",
        "",
        "## Quality Readout",
        "",
    ]

    if lpips_best is not None:
        text = f"- Best LPIPS: `{lpips_best['run_name']}` at {format_value(lpips_best.get('lpips_alex'))}"
        if lpips_baseline is not None:
            text += f" vs `{baseline_run}` at {format_value(lpips_baseline.get('lpips_alex'))}"
        if lpips_improvement is not None:
            text += f" ({lpips_improvement:.1f}% lower)."
        lines.append(text)
    if fvd_best is not None:
        text = f"- Best FVD: `{fvd_best['run_name']}` at {format_value(fvd_best.get('fvd'))}"
        if fvd_baseline is not None:
            text += f" vs `{baseline_run}` at {format_value(fvd_baseline.get('fvd'))}"
        if fvd_improvement is not None:
            text += f" ({fvd_improvement:.1f}% lower)."
        lines.append(text)

    lines.extend(
        [
            "",
            "## How To Interpret The Figures",
            "",
            "- `01_main_quality_metrics.png`: headline quality table as bars. Dashed line is the unbounded baseline.",
            "- `02_percent_improvement_vs_unbounded.png`: positive bars mean bounded policy improves over unbounded.",
            "- `03_memory_behavior_vs_quality.png`: connects visual quality to memory behavior such as candidate count, selected age, overlap, reuse concentration, and section diversity.",
            "- `04_lpips_by_time_bucket.png`: if frame metrics were written, this shows whether the advantage appears early, middle, or late in the 60s video.",
            "- `05_biggest_lpips_wins_contact_sheet.png`: visual examples for the biggest LPIPS wins, with GT, unbounded, and bounded frames side by side.",
            "",
            "## Evidence Tables",
            "",
            "- `tables/main_quality_table.csv`: one row per run, quality metrics.",
            "- `tables/joined_memory_quality.csv`: quality joined with memory behavior.",
            "- `tables/top_lpips_wins.csv`: video-level bounded wins versus unbounded.",
            "- `tables/lpips_time_buckets.csv`: frame-level LPIPS by video third when available.",
            "",
            "## Source Files Used",
            "",
            f"- Access summary: `{access_path}`" if access_path else "- Access summary: missing",
            f"- Trace usefulness summary: `{trace_path}`" if trace_path else "- Trace usefulness summary: missing",
            "",
            "## Caveat",
            "",
            "This report explains why the current bounded runs can look better on LPIPS/FVD. It does not prove the policy is geometrically better. CUT3R camera metrics should be the follow-up for camera-control fidelity.",
            "",
        ]
    )

    if wins:
        lines.extend(["## Biggest LPIPS Wins", ""])
        for win in wins[:8]:
            lines.append(
                f"- `{win['run_name']}` row {win['row']} `{win['scene']}`: "
                f"{format_value(win.get('baseline_lpips_alex'))} -> {format_value(win.get('policy_lpips_alex'))} "
                f"({format_value(win.get('lpips_alex_improvement_pct'), 1)}% lower)"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build professor-ready 60s figures explaining bounded vs unbounded MemCam quality."
    )
    parser.add_argument("--root", type=Path, required=True, help="Generated 60s root containing run folders and trace analyses.")
    parser.add_argument("--metrics_dir", type=Path, required=True, help="Directory containing per-run evaluate_context_memory outputs.")
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("testbeds/context_memory/manifest.jsonl"))
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--no_contact_sheet", action="store_true")
    args = parser.parse_args()

    runs = sorted(parse_list(args.runs), key=run_sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    summary_rows, video_rows, frame_rows_by_run = load_quality(args.metrics_dir, runs, args.duration)
    if not summary_rows:
        raise RuntimeError(
            f"No quality summaries found under {args.metrics_dir}. "
            "Run utils/evaluate_context_memory.py for each policy first."
        )
    summary_rows = sorted(summary_rows, key=lambda row: run_sort_key(row["run_name"]))
    access_rows, trace_rows, access_path, trace_path = load_memory(args.root)
    merged_rows = merge_quality_memory(summary_rows, access_rows, trace_rows)
    bucket_rows = bucket_frame_metrics(frame_rows_by_run, args.duration)
    wins = compute_biggest_wins(video_rows, args.baseline_run, metric="lpips_alex", top_k=args.top_k)
    manifest_rows = load_manifest(args.manifest)

    write_csv(tables_dir / "main_quality_table.csv", summary_rows)
    write_csv(tables_dir / "joined_memory_quality.csv", merged_rows)
    write_csv(tables_dir / "lpips_time_buckets.csv", bucket_rows)
    write_csv(tables_dir / "top_lpips_wins.csv", wins)

    figure_paths = []
    for maker in (
        lambda: save_bar_quality(summary_rows, figures_dir),
        lambda: save_delta_vs_baseline(summary_rows, figures_dir, args.baseline_run),
        lambda: save_memory_quality_scatter(merged_rows, figures_dir),
        lambda: save_time_bucket_plot(bucket_rows, figures_dir),
    ):
        path = maker()
        if path is not None:
            figure_paths.append(path)

    if not args.no_contact_sheet:
        path = save_contact_sheet(
            wins=wins,
            video_rows=video_rows,
            manifest_rows=manifest_rows,
            root=args.root,
            baseline_run=args.baseline_run,
            dataset_root=args.dataset_root,
            figures_dir=figures_dir,
        )
        if path is not None:
            figure_paths.append(path)

    write_report(
        args.output_dir / "report.md",
        summary_rows=summary_rows,
        merged_rows=merged_rows,
        wins=wins,
        figure_paths=figure_paths,
        access_path=access_path,
        trace_path=trace_path,
        baseline_run=args.baseline_run,
    )

    print(f"Wrote report folder: {args.output_dir}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")
    print(f"Wrote report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
