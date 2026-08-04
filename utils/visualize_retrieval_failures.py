import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from analyze_unbounded_retrieval_instability import (  # noqa: E402
    load_labeled_overlaps,
    load_manifest,
    manifest_identity,
    manifest_query_key,
    resolve_overlap_dir,
    trace_identity,
    trace_query_key,
)


TILE_WIDTH = 320
IMAGE_HEIGHT = 180
HEADER_HEIGHT = 46
FOOTER_HEIGHT = 32
TILE_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT


def parse_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


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


def safe_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_bool(value):
    if value in (True, 1, "1", "true", "True"):
        return True
    if value in (False, 0, "0", "false", "False"):
        return False
    return None


def discover_trace_dir(root, run_name):
    path = Path(root) / run_name / "access_traces"
    return path if path.is_dir() else None


def load_selected_trace_rows(trace_dir, wanted_keys=None, wanted_identity=None):
    rows = {}
    if trace_dir is None:
        return rows
    wanted_keys = set(wanted_keys) if wanted_keys is not None else None
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
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
                if wanted_keys is not None and key not in wanted_keys:
                    continue
                if wanted_identity is not None and key[:3] != tuple(wanted_identity):
                    continue
                row["_trace_file"] = str(path)
                rows[key] = row
    return rows


def load_audit_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("pool") == "all"
        and safe_int(row.get("actual_trace_available")) == 1
    ]


def pretty_run_name(run_name):
    if run_name == "baseline":
        return "Unbounded"
    budget_match = re.search(r"_b(\d+)|b(\d+)", run_name)
    budget = None
    if budget_match:
        budget = next(group for group in budget_match.groups() if group)
    suffix = f" (B={budget})" if budget else ""
    if run_name.startswith("fifo"):
        return f"FIFO{suffix}"
    if run_name.startswith("ri"):
        return f"RI{suffix}"
    if run_name.startswith("slammax"):
        return f"SLAM-Max{suffix}"
    if run_name.startswith("slam"):
        return f"SLAM-style{suffix}"
    if run_name.startswith("density"):
        return f"DBVC{suffix}"
    return run_name.replace("_", " ")


def resolve_gt_dir(item, dataset_root=None):
    if dataset_root is not None:
        remapped = Path(dataset_root) / "frames" / item["scene"]
        if remapped.is_dir():
            return remapped
    original = item.get("gt_frames_dir")
    if original and Path(original).is_dir():
        return Path(original)
    return None


def load_gt_frame(item, local_frame, dataset_root=None):
    gt_dir = resolve_gt_dir(item, dataset_root=dataset_root)
    if gt_dir is None:
        return None
    dataset_frame = int(item["start_frame"]) + int(local_frame)
    for suffix in (".png", ".jpg", ".jpeg"):
        path = gt_dir / f"{dataset_frame:04d}{suffix}"
        if path.is_file():
            return Image.open(path).convert("RGB")
    return None


def resolve_video_path(root, run_name, item, trace_row):
    if trace_row is not None and trace_identity(trace_row) != manifest_identity(item):
        raise ValueError(
            "Trace/video identity mismatch: "
            f"manifest={manifest_identity(item)}, trace={trace_identity(trace_row)}"
        )
    expected_name = f"{item['output_prefix']}custom.mp4"
    traced_prefix = trace_row.get("output_prefix") if trace_row else None
    if traced_prefix and str(traced_prefix) != str(item["output_prefix"]):
        raise ValueError(
            "Trace output-prefix mismatch: "
            f"manifest={item['output_prefix']}, trace={traced_prefix}"
        )

    candidate = Path(root) / run_name / expected_name
    if candidate.is_file():
        return candidate

    traced = trace_row.get("output") if trace_row else None
    if traced and Path(traced).is_file():
        traced_path = Path(traced)
        if traced_path.name != expected_name:
            raise ValueError(
                "Trace output filename mismatch: "
                f"expected={expected_name}, trace={traced_path.name}"
            )
        return traced_path
    return None


def load_video_frame(video_path, frame_idx, cache):
    if video_path is None:
        return None
    key = str(video_path), int(frame_idx)
    if key in cache:
        return cache[key].copy() if cache[key] is not None else None
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(video_path))
        try:
            array = reader.get_data(int(frame_idx))
        finally:
            reader.close()
        image = Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
    except Exception as exc:
        print(f"[warn] failed to read {video_path} frame {frame_idx}: {exc}")
        image = None
    cache[key] = image.copy() if image is not None else None
    return image


def build_failure_cases(
    audit_rows,
    manifest_by_row,
    trace_rows_by_run,
    runs,
    baseline_run,
    dataset_root=None,
):
    cases = []
    for audit_row in audit_rows:
        row_idx = safe_int(audit_row.get("row"))
        section_idx = safe_int(audit_row.get("section_idx"))
        target_frame = safe_int(audit_row.get("target_frame"))
        if None in (row_idx, section_idx, target_frame):
            continue
        if row_idx not in manifest_by_row:
            continue
        item = manifest_by_row[row_idx]
        query_key = manifest_query_key(item, section_idx, target_frame)
        if query_key is None:
            continue
        labeled = load_labeled_overlaps(
            overlap_dir=resolve_overlap_dir(item, dataset_root=dataset_root),
            target_frame=target_frame,
            start_frame=int(item["start_frame"]),
            num_frames=int(item["num_frames"]),
        )

        selections = {}
        for run_name in runs:
            trace_row = trace_rows_by_run.get(run_name, {}).get(query_key)
            if trace_row is None:
                continue
            memory_frame = safe_int(trace_row.get("selected_memory_frame"))
            if memory_frame is None:
                continue
            selections[run_name] = {
                "trace": trace_row,
                "memory_frame": memory_frame,
                "age": safe_int(trace_row.get("memory_age")),
                "overlap": safe_float(trace_row.get("selected_overlap")),
                "label_hit": (
                    memory_frame in labeled if labeled is not None else None
                ),
            }
        baseline = selections.get(baseline_run)
        if baseline is None:
            continue

        comparison_hits = [
            selection["label_hit"]
            for run_name, selection in selections.items()
            if run_name != baseline_run and selection["label_hit"] is not None
        ]
        policy_valid = any(comparison_hits)
        baseline_invalid = baseline["label_hit"] is False
        if baseline_invalid and policy_valid:
            category = "Unbounded MISS, bounded HIT"
            category_score = 1000.0
        elif baseline_invalid:
            category = "Unbounded overlap-label MISS"
            category_score = 700.0
        elif policy_valid:
            category = "Different valid retrieval choices"
            category_score = 200.0
        else:
            category = "High-instability retrieval"
            category_score = 0.0

        policy_frames = [
            selection["memory_frame"]
            for run_name, selection in selections.items()
            if run_name != baseline_run
        ]
        disagreement = max(
            [abs(frame - baseline["memory_frame"]) for frame in policy_frames]
            or [0]
        )
        age_span = safe_float(audit_row.get("winner_age_span")) or 0.0
        score = (
            category_score
            + min(age_span, 1000.0) / 10.0
            + min(disagreement, 1000) / 20.0
        )
        cases.append(
            {
                "key": (row_idx, section_idx, target_frame),
                "row": row_idx,
                "section_idx": section_idx,
                "target_frame": target_frame,
                "scene": item["scene"],
                "category": category,
                "score": score,
                "labeled_overlap_count": len(labeled) if labeled is not None else None,
                "selections": selections,
                "audit": audit_row,
                "item": item,
            }
        )
    return cases


def select_diverse_cases(cases, max_examples, per_row=2):
    ranked = sorted(cases, key=lambda case: (-case["score"], case["key"]))
    selected = []
    row_counts = Counter()
    selected_targets = defaultdict(list)
    for case in ranked:
        row_idx = case["row"]
        if row_counts[row_idx] >= int(per_row):
            continue
        if any(
            abs(case["target_frame"] - target) < 76
            for target in selected_targets[row_idx]
        ):
            continue
        selected.append(case)
        row_counts[row_idx] += 1
        selected_targets[row_idx].append(case["target_frame"])
        if len(selected) >= int(max_examples):
            return selected

    for case in ranked:
        if case in selected:
            continue
        selected.append(case)
        if len(selected) >= int(max_examples):
            break
    return selected


def load_font(size, bold=False):
    candidates = (
        ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fitting_font(draw, text, initial_size, max_width, bold=False, minimum_size=9):
    for size in range(int(initial_size), int(minimum_size) - 1, -1):
        font = load_font(size, bold=bold)
        box = draw.textbbox((0, 0), str(text), font=font)
        if box[2] - box[0] <= int(max_width):
            return font
    return load_font(minimum_size, bold=bold)


def fit_image(image, width, height, background=(24, 24, 24)):
    canvas = Image.new("RGB", (int(width), int(height)), background)
    if image is None:
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (width // 2, height // 2),
            "missing frame",
            fill=(220, 220, 220),
            anchor="mm",
            font=load_font(15),
        )
        return canvas
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def status_color(label_hit):
    if label_hit is True:
        return (45, 170, 85)
    if label_hit is False:
        return (220, 65, 65)
    return (125, 130, 140)


def make_tile(title, image, footer, border_color, inset=None):
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (248, 248, 248))
    draw = ImageDraw.Draw(tile)
    draw.rectangle(
        (0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1),
        outline=border_color,
        width=6,
    )
    draw.text(
        (TILE_WIDTH // 2, HEADER_HEIGHT // 2),
        title,
        fill=(20, 20, 20),
        anchor="mm",
        font=fitting_font(draw, title, 16, TILE_WIDTH - 24, bold=True),
    )
    fitted = fit_image(image, TILE_WIDTH - 12, IMAGE_HEIGHT)
    image_x = 6
    image_y = HEADER_HEIGHT
    tile.paste(fitted, (image_x, image_y))

    if inset is not None:
        inset_width, inset_height = 104, 66
        inset_image = fit_image(inset, inset_width, inset_height)
        x = TILE_WIDTH - inset_width - 14
        y = HEADER_HEIGHT + IMAGE_HEIGHT - inset_height - 8
        draw.rectangle(
            (x - 3, y - 19, x + inset_width + 3, y + inset_height + 3),
            fill=(255, 255, 255),
        )
        draw.text(
            (x + inset_width // 2, y - 9),
            "same-frame GT",
            fill=(20, 20, 20),
            anchor="mm",
            font=load_font(11, bold=True),
        )
        tile.paste(inset_image, (x, y))

    draw.text(
        (TILE_WIDTH // 2, HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT // 2),
        footer,
        fill=(35, 35, 35),
        anchor="mm",
        font=fitting_font(draw, footer, 12, TILE_WIDTH - 18),
    )
    return tile


def case_to_csv_row(case, runs):
    row = {
        "row": case["row"],
        "scene": case["scene"],
        "section_idx": case["section_idx"],
        "target_frame": case["target_frame"],
        "category": case["category"],
        "failure_score": case["score"],
        "labeled_overlap_count": case["labeled_overlap_count"],
    }
    for run_name in runs:
        selection = case["selections"].get(run_name)
        row[f"{run_name}_frame"] = selection["memory_frame"] if selection else None
        row[f"{run_name}_age"] = selection["age"] if selection else None
        row[f"{run_name}_overlap"] = selection["overlap"] if selection else None
        row[f"{run_name}_label_hit"] = selection["label_hit"] if selection else None
    return row


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


def render_montage(cases, runs, root, dataset_root, output_dir):
    if not cases:
        return None
    title_height = 100
    width = TILE_WIDTH * (len(runs) + 1)
    height = title_height + TILE_HEIGHT * len(cases)
    montage = Image.new("RGB", (width, height), (236, 238, 241))
    draw = ImageDraw.Draw(montage)
    draw.text(
        (width // 2, 24),
        "What did the memory retriever actually choose?",
        fill=(20, 24, 30),
        anchor="mm",
        font=load_font(32, bold=True),
    )
    draw.text(
        (width // 2, 68),
        "Green = dataset-valid overlap   Red = overlap-label miss   Inset = GT at retrieved frame",
        fill=(60, 64, 72),
        anchor="mm",
        font=load_font(16),
    )

    video_cache = {}
    output_dir = Path(output_dir)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for case_idx, case in enumerate(cases):
        y = title_height + case_idx * TILE_HEIGHT
        item = case["item"]
        target_gt = load_gt_frame(
            item,
            case["target_frame"],
            dataset_root=dataset_root,
        )
        query_title = f"Target GT | row {case['row']}"
        query_footer = (
            f"{case['scene']}  q={case['target_frame']}  {case['category']}"
        )
        query_tile = make_tile(
            query_title,
            target_gt,
            query_footer,
            border_color=(45, 105, 190),
        )
        montage.paste(query_tile, (0, y))
        row_strip = Image.new("RGB", (width, TILE_HEIGHT), (236, 238, 241))
        row_strip.paste(query_tile, (0, 0))

        for run_position, run_name in enumerate(runs, start=1):
            selection = case["selections"].get(run_name)
            if selection is None:
                tile = make_tile(
                    pretty_run_name(run_name),
                    None,
                    "trace unavailable",
                    status_color(None),
                )
            else:
                memory_frame = selection["memory_frame"]
                video_path = resolve_video_path(
                    root,
                    run_name,
                    item,
                    selection["trace"],
                )
                generated = load_video_frame(video_path, memory_frame, video_cache)
                memory_gt = load_gt_frame(
                    item,
                    memory_frame,
                    dataset_root=dataset_root,
                )
                hit = selection["label_hit"]
                hit_text = "HIT" if hit is True else "MISS" if hit is False else "unknown"
                overlap = selection["overlap"]
                overlap_text = f"{overlap:.3f}" if overlap is not None else "NA"
                footer = (
                    f"m={memory_frame}  age={selection['age']}  "
                    f"IoU={overlap_text}  {hit_text}"
                )
                tile = make_tile(
                    pretty_run_name(run_name),
                    generated,
                    footer,
                    status_color(hit),
                    inset=memory_gt,
                )
            x = run_position * TILE_WIDTH
            montage.paste(tile, (x, y))
            row_strip.paste(tile, (x, 0))

        example_path = (
            examples_dir
            / f"row{case['row']}_section{case['section_idx']}_target{case['target_frame']}.png"
        )
        row_strip.save(example_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "retrieval_failure_montage.png"
    pdf = output_dir / "retrieval_failure_montage.pdf"
    montage.save(png)
    montage.save(pdf, "PDF", resolution=150.0)
    return png


def load_timeline_rows(root, runs, item):
    output = {}
    identity = manifest_identity(item)
    for run_name in runs:
        trace_dir = discover_trace_dir(root, run_name)
        output[run_name] = list(
            load_selected_trace_rows(
                trace_dir,
                wanted_identity=identity,
            ).values()
        )
    return output


def save_retrieval_map(
    row_idx,
    item,
    timeline_rows,
    runs,
    dataset_root,
    output_dir,
):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/memcam_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    rows_count = int(math.ceil(len(runs) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(13, 5.4 * rows_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    overlap_dir = resolve_overlap_dir(item, dataset_root=dataset_root)
    label_cache = {}
    max_frame = int(item["num_frames"]) - 1

    for ax, run_name in zip(axes, runs):
        points = timeline_rows.get(run_name, [])
        x_values = []
        y_values = []
        colors = []
        ages = []
        hits = []
        for row in points:
            target = safe_int(row.get("target_frame"))
            memory = safe_int(row.get("selected_memory_frame"))
            if target is None or memory is None:
                continue
            if target not in label_cache:
                label_cache[target] = load_labeled_overlaps(
                    overlap_dir=overlap_dir,
                    target_frame=target,
                    start_frame=int(item["start_frame"]),
                    num_frames=int(item["num_frames"]),
                )
            labeled = label_cache[target]
            hit = memory in labeled if labeled is not None else None
            color = "#2ca25f" if hit is True else "#de4d4d" if hit is False else "#777777"
            x_values.append(target)
            y_values.append(memory)
            colors.append(color)
            ages.append(target - memory)
            if hit is not None:
                hits.append(hit)

        ax.scatter(x_values, y_values, c=colors, s=8, alpha=0.62, linewidths=0)
        ax.plot([0, max_frame], [0, max_frame], color="#222222", linestyle="--", linewidth=1)
        hit_rate = float(np.mean(hits)) if hits else None
        median_age = float(np.median(ages)) if ages else None
        subtitle = f"label hit={hit_rate:.1%}" if hit_rate is not None else "labels unavailable"
        if median_age is not None:
            subtitle += f" | median age={median_age:.0f}f"
        ax.set_title(f"{pretty_run_name(run_name)}\n{subtitle}", fontweight="bold")
        ax.set_xlim(0, max_frame)
        ax.set_ylim(0, max_frame)
        ax.set_xlabel("target frame")
        ax.set_ylabel("retrieved memory frame")
        ax.grid(color="#ececec", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(runs) :]:
        ax.axis("off")
    fig.suptitle(
        f"Retrieval map: row {row_idx}, {item['scene']}\n"
        "Green = dataset-valid overlap, red = overlap-label miss",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"retrieval_map_row{row_idx}.png"
    pdf = output_dir / f"retrieval_map_row{row_idx}.pdf"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser(
        description="Create frame montages and retrieval maps for MemCam failures."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=str,
        default="baseline,fifo_b32,ri_b32_dino_rgb,slam_b32_covisibility",
    )
    parser.add_argument("--baseline_run", type=str, default="baseline")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=15)
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--per_row", type=int, default=2)
    parser.add_argument("--timeline_row", type=int, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    runs = parse_list(args.runs)
    if args.baseline_run not in runs:
        runs.insert(0, args.baseline_run)
    manifest_rows = load_manifest(
        args.manifest,
        duration=args.duration,
        selected_rows=parse_int_ranges(args.rows),
        max_rows=args.max_rows,
    )
    manifest_by_row = {int(item["_row"]): item for item in manifest_rows}
    audit_rows = [
        row
        for row in load_audit_rows(args.audit_csv)
        if safe_int(row.get("row")) in manifest_by_row
    ]
    if not audit_rows:
        raise RuntimeError("No full-pool traced queries were found in the audit CSV")
    wanted_keys = set()
    for row in audit_rows:
        row_idx = safe_int(row.get("row"))
        section_idx = safe_int(row.get("section_idx"))
        target_frame = safe_int(row.get("target_frame"))
        item = manifest_by_row.get(row_idx)
        if item is None or None in (section_idx, target_frame):
            continue
        wanted_keys.add(manifest_query_key(item, section_idx, target_frame))
    wanted_keys.discard(None)

    trace_rows_by_run = {}
    available_runs = []
    for run_name in runs:
        trace_dir = discover_trace_dir(args.root, run_name)
        if trace_dir is None:
            print(f"[warn] missing access traces for {run_name}; skipping")
            continue
        selected = load_selected_trace_rows(trace_dir, wanted_keys=wanted_keys)
        if not selected:
            print(f"[warn] no matched trace rows for {run_name}; skipping")
            continue
        trace_rows_by_run[run_name] = selected
        available_runs.append(run_name)
        print(f"{run_name}: {len(selected)} matched retrieval rows")
    runs = available_runs
    if args.baseline_run not in runs:
        raise RuntimeError(f"Baseline traces unavailable: {args.baseline_run}")

    cases = build_failure_cases(
        audit_rows=audit_rows,
        manifest_by_row=manifest_by_row,
        trace_rows_by_run=trace_rows_by_run,
        runs=runs,
        baseline_run=args.baseline_run,
        dataset_root=args.dataset_root,
    )
    selected_cases = select_diverse_cases(
        cases,
        max_examples=args.max_examples,
        per_row=args.per_row,
    )
    if not selected_cases:
        raise RuntimeError("No paired retrieval cases were available to visualize")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "retrieval_failure_examples.csv",
        [case_to_csv_row(case, runs) for case in selected_cases],
    )
    montage = render_montage(
        selected_cases,
        runs=runs,
        root=args.root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )

    if args.timeline_row is not None:
        timeline_row = int(args.timeline_row)
    else:
        row_scores = defaultdict(float)
        for case in cases:
            row_scores[case["row"]] += case["score"]
        timeline_row = max(row_scores, key=row_scores.get)
    timeline_item = manifest_by_row[timeline_row]
    timeline_rows = load_timeline_rows(args.root, runs, timeline_item)
    retrieval_map = save_retrieval_map(
        row_idx=timeline_row,
        item=timeline_item,
        timeline_rows=timeline_rows,
        runs=runs,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )

    print(f"Wrote: {montage}")
    print(f"Wrote: {retrieval_map}")
    print(f"Wrote: {args.output_dir / 'retrieval_failure_examples.csv'}")


if __name__ == "__main__":
    main()
