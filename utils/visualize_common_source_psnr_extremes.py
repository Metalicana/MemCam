"""Find visual examples where GeoCov selects cleaner historical evidence.

The selector identities come from each policy's real access trace, but every
selected image is read from one common rollout (normally ``baseline``). This
removes the cleaner-history advantage of a bounded policy and isolates which
historical indices its candidate bank made available to the frozen retriever.
PSNR and SSIM compare each selected image with dataset ground truth at that
same historical index.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.analyze_selected_memory_image_quality import (  # noqa: E402
    load_manifest,
    read_gt_frame,
)
from utils.evaluate_context_memory import frame_metrics  # noqa: E402
from utils.visualize_geometric_coverage_evictions import (  # noqa: E402
    load_video_frames_single_pass,
)


SECTION_STRIDE = 76
TILE_WIDTH = 292
IMAGE_HEIGHT = 164
HEADER_HEIGHT = 42
FOOTER_HEIGHT = 54
TILE_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT

COLORS = {
    "target": (45, 104, 176),
    "unbounded": (194, 61, 56),
    "geocov": (39, 145, 83),
    "ground_truth": (90, 96, 106),
    "error": (151, 78, 31),
}


def parse_int_ranges(value):
    if not value:
        return None
    output = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        else:
            output.add(int(part))
    return sorted(output)


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


def trace_identity(row):
    try:
        return (
            str(row["scene"]),
            int(row["dataset_start_frame"]),
            int(row["duration_sec"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def manifest_identity(item):
    return (
        str(item["scene"]),
        int(item["start_frame"]),
        int(item["duration_sec"]),
    )


def load_selected_queries(path, item, strict=True):
    """Load one selected context-access record per (section, target)."""
    expected_identity = manifest_identity(item)
    output = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "context_access" or not row.get("selected"):
                continue
            identity = trace_identity(row)
            if identity != expected_identity:
                message = (
                    f"Trace identity mismatch in {path}: expected "
                    f"{expected_identity}, found {identity}"
                )
                if strict:
                    raise ValueError(message)
                continue
            section_idx = safe_int(row.get("section_idx"))
            target_frame = safe_int(row.get("target_frame"))
            selected_frame = safe_int(row.get("selected_memory_frame"))
            if None in (section_idx, target_frame, selected_frame):
                continue
            key = (section_idx, target_frame)
            if strict and key in output:
                raise ValueError(f"Duplicate selected trace query {key} in {path}")
            output[key] = row
    return output


def query_is_sampled(section_idx, target_frame, target_stride):
    first_target = int(section_idx) * SECTION_STRIDE + 1
    offset = int(target_frame) - first_target
    return offset >= 0 and offset % int(target_stride) == 0


def remap_gt_dir(item, dataset_root):
    if dataset_root is None:
        return item
    candidate = Path(dataset_root) / "frames" / item["scene"]
    if not candidate.is_dir():
        raise FileNotFoundError(f"Remapped GT directory does not exist: {candidate}")
    output = dict(item)
    output["gt_frames_dir"] = str(candidate)
    return output


def trace_value(row, name):
    if name in {"memory_age"}:
        return safe_int(row.get(name))
    return safe_float(row.get(name))


def build_query_row(item, key, reference_trace, policy_trace, images):
    section_idx, target_frame = key
    reference_frame = int(reference_trace["selected_memory_frame"])
    policy_frame = int(policy_trace["selected_memory_frame"])
    reference_image = np.asarray(images[reference_frame], dtype=np.uint8)
    policy_image = np.asarray(images[policy_frame], dtype=np.uint8)
    reference_gt = read_gt_frame(item, reference_frame, reference_image.shape)
    policy_gt = read_gt_frame(item, policy_frame, policy_image.shape)
    reference_metrics = frame_metrics(reference_image, reference_gt)
    policy_metrics = frame_metrics(policy_image, policy_gt)
    return {
        "row": int(item["_row"]),
        "scene": item["scene"],
        "dataset_start_frame": int(item["start_frame"]),
        "duration_sec": int(item["duration_sec"]),
        "section_idx": int(section_idx),
        "section_time_sec": float(section_idx * SECTION_STRIDE / float(item["fps"])),
        "target_frame": int(target_frame),
        "unbounded_selected_frame": reference_frame,
        "geocov_selected_frame": policy_frame,
        "unbounded_memory_age": trace_value(reference_trace, "memory_age"),
        "geocov_memory_age": trace_value(policy_trace, "memory_age"),
        "unbounded_overlap": trace_value(reference_trace, "selected_overlap"),
        "geocov_overlap": trace_value(policy_trace, "selected_overlap"),
        "unbounded_psnr": reference_metrics["psnr_db"],
        "unbounded_ssim": reference_metrics["ssim"],
        "geocov_psnr": policy_metrics["psnr_db"],
        "geocov_ssim": policy_metrics["ssim"],
        "psnr_delta": policy_metrics["psnr_db"] - reference_metrics["psnr_db"],
        "ssim_delta": policy_metrics["ssim"] - reference_metrics["ssim"],
    }


def meets_thresholds(
    row,
    min_psnr_delta,
    min_ssim_delta,
    max_unbounded_psnr,
    min_geocov_psnr,
    min_overlap,
):
    reference_overlap = row.get("unbounded_overlap")
    policy_overlap = row.get("geocov_overlap")
    return (
        float(row["psnr_delta"]) >= float(min_psnr_delta)
        and float(row["ssim_delta"]) >= float(min_ssim_delta)
        and (
            max_unbounded_psnr is None
            or float(row["unbounded_psnr"]) <= float(max_unbounded_psnr)
        )
        and (
            min_geocov_psnr is None
            or float(row["geocov_psnr"]) >= float(min_geocov_psnr)
        )
        and (
            min_overlap is None
            or (
                reference_overlap is not None
                and policy_overlap is not None
                and float(reference_overlap) >= float(min_overlap)
                and float(policy_overlap) >= float(min_overlap)
            )
        )
    )


def rank_key(row):
    return (
        -float(row["psnr_delta"]),
        -float(row["ssim_delta"]),
        float(row["unbounded_psnr"]),
        int(row["row"]),
        int(row["target_frame"]),
    )


def select_diverse_extremes(
    rows,
    max_examples,
    per_row,
    min_target_gap,
    min_psnr_delta,
    min_ssim_delta,
    max_unbounded_psnr,
    min_geocov_psnr,
    min_overlap,
    strict_thresholds=False,
):
    """Select ranked examples while limiting scene and temporal repetition."""
    for row in rows:
        row["meets_extreme_thresholds"] = meets_thresholds(
            row,
            min_psnr_delta,
            min_ssim_delta,
            max_unbounded_psnr,
            min_geocov_psnr,
            min_overlap,
        )
    eligible = sorted(
        [row for row in rows if row["meets_extreme_thresholds"]],
        key=rank_key,
    )
    fallback = [] if strict_thresholds else sorted(
        [row for row in rows if not row["meets_extreme_thresholds"]],
        key=rank_key,
    )

    selected = []
    row_counts = Counter()
    row_targets = defaultdict(list)
    for row in eligible + fallback:
        row_idx = int(row["row"])
        if row_counts[row_idx] >= int(per_row):
            continue
        if any(
            abs(int(row["target_frame"]) - target) < int(min_target_gap)
            for target in row_targets[row_idx]
        ):
            continue
        selected.append(row)
        row_counts[row_idx] += 1
        row_targets[row_idx].append(int(row["target_frame"]))
        if len(selected) >= int(max_examples):
            break
    return selected


def load_font(size, bold=False):
    names = (
        ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def fitting_font(draw, text, initial_size, max_width, bold=False, minimum_size=9):
    for size in range(int(initial_size), int(minimum_size) - 1, -1):
        font = load_font(size, bold=bold)
        bounds = draw.textbbox((0, 0), str(text), font=font)
        if bounds[2] - bounds[0] <= int(max_width):
            return font
    return load_font(minimum_size, bold=bold)


def fit_image(image, width, height):
    canvas = Image.new("RGB", (int(width), int(height)), (18, 18, 18))
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def make_tile(title, image, footer, border_color):
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (248, 248, 248))
    draw = ImageDraw.Draw(tile)
    draw.rectangle(
        (0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1),
        outline=border_color,
        width=5,
    )
    draw.text(
        (TILE_WIDTH // 2, HEADER_HEIGHT // 2),
        title,
        fill=(22, 22, 24),
        anchor="mm",
        font=fitting_font(draw, title, 16, TILE_WIDTH - 20, bold=True),
    )
    tile.paste(fit_image(image, TILE_WIDTH - 10, IMAGE_HEIGHT), (5, HEADER_HEIGHT))
    draw.multiline_text(
        (TILE_WIDTH // 2, HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT // 2),
        footer,
        fill=(35, 35, 38),
        anchor="mm",
        align="center",
        spacing=2,
        font=fitting_font(draw, footer, 12, TILE_WIDTH - 16),
    )
    return tile


def error_heatmap(image, ground_truth, maximum_error=96.0):
    image = np.asarray(image, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    error = np.mean(np.abs(image - ground_truth), axis=2)
    scaled = np.clip(error / float(maximum_error), 0.0, 1.0)
    heatmap = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.asarray(255.0 * scaled, dtype=np.uint8)
    heatmap[..., 1] = np.asarray(80.0 * scaled, dtype=np.uint8)
    heatmap[..., 2] = np.asarray(24.0 * scaled, dtype=np.uint8)
    return Image.fromarray(heatmap)


def load_case_assets(case, item, content_video, timeout_sec):
    reference_frame = int(case["unbounded_selected_frame"])
    policy_frame = int(case["geocov_selected_frame"])
    decoded = load_video_frames_single_pass(
        content_video,
        [reference_frame, policy_frame],
        timeout_sec=timeout_sec,
    )
    reference_image = decoded[reference_frame]
    policy_image = decoded[policy_frame]
    reference_array = np.asarray(reference_image, dtype=np.uint8)
    policy_array = np.asarray(policy_image, dtype=np.uint8)
    reference_gt = Image.fromarray(
        read_gt_frame(item, reference_frame, reference_array.shape)
    )
    policy_gt = Image.fromarray(read_gt_frame(item, policy_frame, policy_array.shape))
    target_gt = Image.fromarray(
        read_gt_frame(item, int(case["target_frame"]), reference_array.shape)
    )
    return {
        "target_gt": target_gt,
        "unbounded": reference_image,
        "unbounded_gt": reference_gt,
        "unbounded_error": error_heatmap(reference_image, reference_gt),
        "geocov": policy_image,
        "geocov_gt": policy_gt,
        "geocov_error": error_heatmap(policy_image, policy_gt),
    }


def case_tiles(case, assets, include_errors=False):
    u_frame = int(case["unbounded_selected_frame"])
    g_frame = int(case["geocov_selected_frame"])
    u_age = int(case["unbounded_memory_age"])
    g_age = int(case["geocov_memory_age"])
    u_overlap = float(case["unbounded_overlap"])
    g_overlap = float(case["geocov_overlap"])
    tiles = [
        make_tile(
            "Target ground truth",
            assets["target_gt"],
            f"query frame {int(case['target_frame'])}",
            COLORS["target"],
        ),
        make_tile(
            "Unbounded selected",
            assets["unbounded"],
            f"m={u_frame}  age={u_age}  IoU={u_overlap:.3f}\n"
            f"PSNR {case['unbounded_psnr']:.2f} | SSIM {case['unbounded_ssim']:.3f}",
            COLORS["unbounded"],
        ),
        make_tile(
            "GT at unbounded index",
            assets["unbounded_gt"],
            f"exact trajectory frame {u_frame}",
            COLORS["ground_truth"],
        ),
        make_tile(
            "GeoCov selected",
            assets["geocov"],
            f"m={g_frame}  age={g_age}  IoU={g_overlap:.3f}\n"
            f"PSNR {case['geocov_psnr']:.2f} | SSIM {case['geocov_ssim']:.3f}",
            COLORS["geocov"],
        ),
        make_tile(
            "GT at GeoCov index",
            assets["geocov_gt"],
            f"exact trajectory frame {g_frame}",
            COLORS["ground_truth"],
        ),
    ]
    if include_errors:
        tiles.extend(
            [
                make_tile(
                    "Unbounded error",
                    assets["unbounded_error"],
                    "black=match, red=large error",
                    COLORS["unbounded"],
                ),
                make_tile(
                    "GeoCov error",
                    assets["geocov_error"],
                    "same fixed error scale",
                    COLORS["geocov"],
                ),
            ]
        )
    return tiles


def render_figures(cases, items_by_row, root, content_run, output_dir, timeout_sec):
    figures_dir = Path(output_dir) / "figures"
    cases_dir = figures_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    title_height = 96
    overview = Image.new(
        "RGB",
        (TILE_WIDTH * 5, title_height + TILE_HEIGHT * len(cases)),
        (233, 235, 238),
    )
    draw = ImageDraw.Draw(overview)
    draw.text(
        (overview.width // 2, 27),
        "Extreme common-source retrieval examples",
        fill=(18, 23, 29),
        anchor="mm",
        font=load_font(29, bold=True),
    )
    draw.text(
        (overview.width // 2, 66),
        "Both selectors read pixels from the same unbounded rollout; PSNR/SSIM use exact-index GT",
        fill=(56, 61, 69),
        anchor="mm",
        font=load_font(15),
    )

    for case_index, case in enumerate(cases):
        item = items_by_row[int(case["row"])]
        video = Path(root) / content_run / f"{item['output_prefix']}custom.mp4"
        assets = load_case_assets(case, item, video, timeout_sec)
        tiles = case_tiles(case, assets, include_errors=False)
        y = title_height + case_index * TILE_HEIGHT
        for column, tile in enumerate(tiles):
            overview.paste(tile, (column * TILE_WIDTH, y))

        detail_tiles = case_tiles(case, assets, include_errors=True)
        detail_title = 82
        detail = Image.new(
            "RGB",
            (TILE_WIDTH * len(detail_tiles), detail_title + TILE_HEIGHT),
            (233, 235, 238),
        )
        detail_draw = ImageDraw.Draw(detail)
        detail_draw.text(
            (detail.width // 2, 24),
            f"{case['scene']} | section {case['section_idx']} | query {case['target_frame']}",
            fill=(18, 23, 29),
            anchor="mm",
            font=load_font(24, bold=True),
        )
        detail_draw.text(
            (detail.width // 2, 56),
            f"GeoCov - Unbounded: {case['psnr_delta']:+.2f} dB PSNR, {case['ssim_delta']:+.3f} SSIM",
            fill=(39, 95, 60),
            anchor="mm",
            font=load_font(16, bold=True),
        )
        for column, tile in enumerate(detail_tiles):
            detail.paste(tile, (column * TILE_WIDTH, detail_title))
        detail.save(
            cases_dir
            / (
                f"case_{case_index:02d}_row{case['row']}_section"
                f"{case['section_idx']}_target{case['target_frame']}.png"
            )
        )

    overview_path = figures_dir / "01_extreme_common_source_examples.png"
    overview.save(overview_path)
    overview.convert("RGB").save(
        figures_dir / "01_extreme_common_source_examples.pdf",
        "PDF",
        resolution=150.0,
    )
    return overview_path


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


def write_report(
    path,
    rows,
    cases,
    reference_run,
    policy_run,
    content_run,
    thresholds,
):
    qualifying = [row for row in rows if row.get("meets_extreme_thresholds")]
    lines = [
        "# Extreme Common-Source Retrieval Examples",
        "",
        "## What Is Held Fixed",
        "",
        f"`{reference_run}` and `{policy_run}` supply only selected historical frame indices. Every selected image is read from the same `{content_run}` rollout. PSNR and SSIM compare that generated image with dataset ground truth at the exact same historical index.",
        "",
        "This isolates index selection from policy-specific generated-history quality. It does not isolate candidate-set size from candidate-set composition.",
        "",
        "## Scan",
        "",
        f"- Matched selector disagreements scored: `{len(rows)}`.",
        f"- Queries clearing the display thresholds: `{len(qualifying)}`.",
        f"- Display threshold: PSNR delta >= `{thresholds['min_psnr_delta']:.2f}` dB, SSIM delta >= `{thresholds['min_ssim_delta']:.3f}`, unbounded PSNR <= `{thresholds['max_unbounded_psnr']}`, GeoCov PSNR >= `{thresholds['min_geocov_psnr']}`, and both logged FOV overlaps >= `{thresholds['min_overlap']}`.",
        f"- Diverse examples selected: `{len(cases)}`.",
        "",
        "## Selected Examples",
        "",
        "| case | row | scene | section | query | unbounded frame | unbounded PSNR | GeoCov frame | GeoCov PSNR | PSNR delta | SSIM delta | threshold pass |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(cases):
        lines.append(
            f"| {index} | {row['row']} | {row['scene']} | {row['section_idx']} | "
            f"{row['target_frame']} | {row['unbounded_selected_frame']} | "
            f"{row['unbounded_psnr']:.2f} | {row['geocov_selected_frame']} | "
            f"{row['geocov_psnr']:.2f} | {row['psnr_delta']:+.2f} | "
            f"{row['ssim_delta']:+.3f} | {bool(row['meets_extreme_thresholds'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading the Figure",
            "",
            "Each selected generated frame is shown beside its own exact-index ground truth. This matters because Unbounded and GeoCov may select different camera moments. A cleaner GeoCov image is therefore visible as a closer match to the GT immediately beside it, not by comparing it directly with the Unbounded image.",
            "",
            "These are deliberately ranked extreme examples for qualitative explanation. They are not an estimate of how often GeoCov wins. The aggregate common-source experiment, its trajectory bootstrap intervals, and its 15/15 trajectory wins provide the population-level evidence.",
            "",
            "## Files",
            "",
            "- `figures/01_extreme_common_source_examples.png`",
            "- `figures/01_extreme_common_source_examples.pdf`",
            "- `figures/cases/case_*.png`",
            "- `tables/all_matched_query_scores.csv`",
            "- `tables/selected_examples.csv`",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference_run", default="baseline")
    parser.add_argument("--policy_run", default="slam_b32_covisibility")
    parser.add_argument("--content_run", default="baseline")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--min_section", type=int, default=35)
    parser.add_argument("--max_section", type=int, default=None)
    parser.add_argument("--target_stride", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=6)
    parser.add_argument("--per_row", type=int, default=1)
    parser.add_argument("--min_target_gap", type=int, default=152)
    parser.add_argument("--min_psnr_delta", type=float, default=4.0)
    parser.add_argument("--min_ssim_delta", type=float, default=0.05)
    parser.add_argument("--max_unbounded_psnr", type=float, default=12.0)
    parser.add_argument("--min_geocov_psnr", type=float, default=14.0)
    parser.add_argument("--min_overlap", type=float, default=0.8)
    parser.add_argument("--decode_timeout_sec", type=int, default=600)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--strict_thresholds", action="store_true")
    args = parser.parse_args()

    if args.target_stride <= 0:
        raise ValueError("--target_stride must be positive")
    if args.max_examples <= 0 or args.per_row <= 0:
        raise ValueError("--max_examples and --per_row must be positive")

    selected_rows = set(parse_int_ranges(args.rows) or [])
    items = load_manifest(args.manifest, args.duration)
    if selected_rows:
        items = [item for item in items if int(item["_row"]) in selected_rows]
    items = [remap_gt_dir(item, args.dataset_root) for item in items]
    items_by_row = {int(item["_row"]): item for item in items}

    all_rows = []
    for item in items:
        trace_name = f"{item['output_prefix']}custom.jsonl"
        reference_path = args.root / args.reference_run / "access_traces" / trace_name
        policy_path = args.root / args.policy_run / "access_traces" / trace_name
        content_video = args.root / args.content_run / f"{item['output_prefix']}custom.mp4"
        missing = [
            path
            for path in (reference_path, policy_path, content_video)
            if not path.is_file()
        ]
        if missing:
            message = f"row {item['_row']} is missing: {', '.join(map(str, missing))}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue

        reference = load_selected_queries(reference_path, item, strict=args.strict)
        policy = load_selected_queries(policy_path, item, strict=args.strict)
        shared = []
        for key in sorted(set(reference) & set(policy)):
            section_idx, target_frame = key
            if section_idx < args.min_section:
                continue
            if args.max_section is not None and section_idx > args.max_section:
                continue
            if not query_is_sampled(section_idx, target_frame, args.target_stride):
                continue
            reference_frame = safe_int(reference[key].get("selected_memory_frame"))
            policy_frame = safe_int(policy[key].get("selected_memory_frame"))
            if reference_frame is None or policy_frame is None:
                continue
            if reference_frame == policy_frame:
                continue
            shared.append(key)

        requested_frames = {
            int(trace["selected_memory_frame"])
            for key in shared
            for trace in (reference[key], policy[key])
        }
        if not requested_frames:
            print(f"row {item['_row']}: no sampled selector disagreements")
            continue
        try:
            images = load_video_frames_single_pass(
                content_video,
                requested_frames,
                timeout_sec=args.decode_timeout_sec,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"[skip] row {item['_row']} decode failed: {exc}")
            continue

        for key in shared:
            all_rows.append(
                build_query_row(item, key, reference[key], policy[key], images)
            )
        print(
            f"row {item['_row']}: {len(shared)} matched disagreements, "
            f"{len(requested_frames)} unique common-source frames"
        )

    if not all_rows:
        raise RuntimeError("No matched common-source selector disagreements were scored")

    cases = select_diverse_extremes(
        all_rows,
        max_examples=args.max_examples,
        per_row=args.per_row,
        min_target_gap=args.min_target_gap,
        min_psnr_delta=args.min_psnr_delta,
        min_ssim_delta=args.min_ssim_delta,
        max_unbounded_psnr=args.max_unbounded_psnr,
        min_geocov_psnr=args.min_geocov_psnr,
        min_overlap=args.min_overlap,
        strict_thresholds=args.strict_thresholds,
    )
    if not cases:
        raise RuntimeError("No examples cleared the requested diversity and threshold rules")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tables" / "all_matched_query_scores.csv", all_rows)
    write_csv(args.output_dir / "tables" / "selected_examples.csv", cases)
    figure = render_figures(
        cases,
        items_by_row,
        args.root,
        args.content_run,
        args.output_dir,
        args.decode_timeout_sec,
    )
    thresholds = {
        "min_psnr_delta": args.min_psnr_delta,
        "min_ssim_delta": args.min_ssim_delta,
        "max_unbounded_psnr": args.max_unbounded_psnr,
        "min_geocov_psnr": args.min_geocov_psnr,
        "min_overlap": args.min_overlap,
    }
    write_report(
        args.output_dir / "report.md",
        all_rows,
        cases,
        args.reference_run,
        args.policy_run,
        args.content_run,
        thresholds,
    )
    print(f"Wrote: {figure}")
    print(f"Wrote: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
