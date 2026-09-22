"""Build a data-backed draw.io teaser: two charts above a full-width sample grid.

Charts, labels and confidence bands are vector objects. Real generated and GT
frames are embedded PNGs, decoded from videos or extracted from saved strips.
The original draw.io document is never modified. Use draw.io to export PDF/PNG.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zlib

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.replace_method_frame_assets import extract_frames
from paper.paired_retrieval_curves import compare as compare_retrieval
from paper.plot_retrieval_deterioration import write_csv

INK, MUTED, GRID = "#24282C", "#687078", "#E4E7E9"
UNBOUNDED, KEEPSAKE, ORACLE = "#BC5147", "#277A60", "#397BA5"
WIDTH, HEIGHT, COMPACT_HEIGHT = 2180, 1580, 1304
SELECTED = "selected_effective_mismatch"
BEST = "full_oracle_effective_mismatch"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_curves(directory):
    with (directory / "curves.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    series = {}
    for metric in (SELECTED, BEST):
        values = [{key: float(row[key]) for key in ("time_sec", "mean", "ci_low", "ci_high")}
                  for row in rows if row["metric"] == metric]
        values.sort(key=lambda row: row["time_sec"])
        if len(values) < 2 or any(
            not all(math.isfinite(v) for v in row.values())
            or not 0 <= row["ci_low"] <= row["mean"] <= row["ci_high"] <= 2
            for row in values
        ):
            raise ValueError(f"Invalid curve: {metric}")
        times = [row["time_sec"] for row in values]
        if len(set(times)) != len(times):
            raise ValueError(f"Duplicate time bins: {metric}")
        series[metric] = values
    if [r["time_sec"] for r in series[SELECTED]] != [r["time_sec"] for r in series[BEST]]:
        raise ValueError("The selected and oracle curves must use the same bins")
    if any(a["mean"] + 1e-5 < b["mean"] for a, b in zip(series[SELECTED], series[BEST])):
        raise ValueError("Eligible-history oracle cannot be worse than selection")
    return series


class Diagram:
    def __init__(self, height=HEIGHT, width=WIDTH):
        self.width = width
        self.file = ET.Element("mxfile", host="app.diagrams.net")
        page = ET.SubElement(self.file, "diagram", id="keepsake-teaser", name="Two-row teaser")
        model = ET.SubElement(page, "mxGraphModel", dx=str(width), dy=str(height), grid="1",
                              gridSize="10", page="1", pageScale="1", pageWidth=str(width),
                              pageHeight=str(height), background="#FFFFFF", math="0", shadow="0")
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.index = 0

    def cell(self, value, x, y, w, h, style, parent="1", name=None):
        self.index += 1
        identity = name or f"element-{self.index}"
        node = ET.SubElement(self.root, "mxCell", id=identity, value=value, style=style,
                             vertex="1", parent=parent)
        ET.SubElement(node, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h),
                      attrib={"as": "geometry"})
        return identity

    def label(self, value, x, y, w, h=32, size=24, color=INK, bold=False,
              align="left", parent="1", extra=""):
        return self.cell(value, x, y, w, h,
                         f"text;html=1;strokeColor=none;fillColor=none;align={align};"
                         f"verticalAlign=middle;whiteSpace=wrap;rounded=0;spacing=0;"
                         f"fontFamily=Arial;fontSize={size};fontColor={color};"
                         f"fontStyle={int(bold)};{extra}", parent)

    def rect(self, x, y, w, h, color, parent="1", extra=""):
        return self.cell("", x, y, w, h,
                         f"rounded=0;strokeColor=none;fillColor={color};{extra}", parent)

    def line(self, points, color, width=1.5, parent="1", extra=""):
        self.index += 1
        node = ET.SubElement(self.root, "mxCell", id=f"element-{self.index}", value="",
                             style=f"endArrow=none;startArrow=none;html=1;rounded=0;"
                                   f"strokeColor={color};strokeWidth={width};{extra}",
                             edge="1", parent=parent)
        geometry = ET.SubElement(node, "mxGeometry", relative="1", attrib={"as": "geometry"})
        for point, name in ((points[0], "sourcePoint"), (points[-1], "targetPoint")):
            ET.SubElement(geometry, "mxPoint", x=str(point[0]), y=str(point[1]), attrib={"as": name})
        if len(points) > 2:
            array = ET.SubElement(geometry, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def band(self, points, color, parent):
        # draw.io custom stencils keep the confidence band vector-based in PDF.
        x0, y0 = min(p[0] for p in points), min(p[1] for p in points)
        w, h = max(p[0] for p in points) - x0, max(p[1] for p in points) - y0
        shape = ET.Element("shape", name="Confidence band", w="100", h="100", aspect="variable")
        foreground = ET.SubElement(shape, "foreground")
        path = ET.SubElement(foreground, "path")
        for i, (x, y) in enumerate(points):
            ET.SubElement(path, "move" if i == 0 else "line", x=str(100 * (x - x0) / w),
                          y=str(100 * (y - y0) / h))
        ET.SubElement(path, "close")
        ET.SubElement(foreground, "fill")
        encoded = quote(ET.tostring(shape, encoding="unicode"), safe="~()*!.'-").encode()
        compressor = zlib.compressobj(wbits=-15)
        packed = base64.b64encode(compressor.compress(encoded) + compressor.flush()).decode()
        self.cell("", x0, y0, w, h, f"shape=stencil({packed});fillColor={color};"
                  "fillOpacity=13;strokeColor=none;", parent)

    def picture(self, im, x, y, w, h, parent):
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        self.cell("", x, y, w, h, "shape=image;imageAspect=0;aspect=fixed;"
                  f"image=data:image/png,{data};", parent)

    def save(self, path):
        ET.indent(self.file)
        ET.ElementTree(self.file).write(path, encoding="utf-8", xml_declaration=True)


def panel(diagram, name, x, width, title, subtitle, y=22, height=636):
    group = diagram.cell("", x, y, width, height, "group;", name=name)
    diagram.label(title, 0, 0, width, 44, size=35, bold=True, parent=group)
    if subtitle:
        diagram.label(subtitle, 0, 48, width, 32, size=25, color=MUTED, parent=group)
    return group


def compact_panel_geometry(width):
    left_width = round((width - 2 * 26 - 32) * .474)
    right_x = 26 + left_width + 32
    return left_width, right_x, width - 26 - right_x


def efficiency_panel(diagram, data, compact=False):
    panel_width = compact_panel_geometry(diagram.width)[0] if compact else 970
    group = panel(diagram, "efficiency", 26, panel_width, "(a) Smaller memory, faster lookup",
                  None if compact else "MemCam / 180-second rollout", y=18 if compact else 22,
                  height=388 if compact else 636)
    records = {r["policy"]: r for r in data["records"]}
    if set(records) != {"Unbounded", "KEEPSAKE"}:
        raise ValueError("Efficiency requires Unbounded and KEEPSAKE records")
    if (data["duration_sec"], data["videos"], data["sampled_queries"], data["repeats"], data["threads"]) != (180, 1, 8, 1, 1):
        raise ValueError("Update the timing labels/caption to match the input protocol")
    for key in ("final_stored_frames", "query_ms"):
        if any(not math.isfinite(float(r[key])) or r[key] <= 0 for r in records.values()):
            raise ValueError("Resource measurements must be finite and positive")
    for top, key, title, maximum, ticks in (
        (53 if compact else 115, "final_stored_frames", "Retained frames", 6000, (0, 2000, 4000, 6000)),
        (225 if compact else 340, "query_ms", "CPU lookup time (ms/query)", 4500, (0, 1500, 3000, 4500)),
    ):
        diagram.label(title, 0, top, 580, size=29, bold=True, parent=group)
        x, plot_w = 177, panel_width - 262 if compact else 680
        bottom = top + (127 if compact else 136)
        for tick in ticks:
            tx = x + plot_w * tick / maximum
            diagram.line([(tx, top + 44), (tx, bottom)], GRID, 1, group)
            diagram.label(f"{tick:,}", tx - 40, bottom + 5, 80, 29, size=23,
                          color=MUTED, align="center", parent=group)
        for row, (name, color) in enumerate((("Unbounded", UNBOUNDED), ("KEEPSAKE", KEEPSAKE))):
            value = records[name][key]
            y, bar_w = top + (44 + row * 43 if compact else 46 + row * 57), plot_w * value / maximum
            diagram.label(name, 0, y - 2 if compact else y, 167, 32, size=27, color=color,
                          bold=name == "KEEPSAKE", parent=group)
            diagram.rect(x, y, bar_w, 29 if compact else 31, color, group)
            label = f"{int(value):,}" if key == "final_stored_frames" else f"{value:,.1f}"
            diagram.label(label, x + bar_w + 10, y - 2, 115, 32 if compact else 36, size=30,
                          bold=True, color=color, parent=group)
        diagram.line([(x, bottom), (x + plot_w, bottom)], MUTED, 1, group)
    if not compact:
        diagram.label("1 CPU thread / 8 queries / 1 trajectory", 0, 607, 970, 28,
                      size=24, color=MUTED, parent=group)


def retrieval_panel(diagram, directory):
    series = load_curves(directory)
    metadata = json.loads((directory / "provenance.json").read_text())
    params = metadata["parameters"]
    if params["duration"] != 180 or params["expected_videos"] != 15 or params["run"] != "baseline":
        raise ValueError("The diagnostic must be the 15-trajectory, 180 s unbounded run")
    group = panel(diagram, "retrieval", 1148, 1000, "(b) Retrieval quality", "Unbounded / 180 s / 15 trajectories")
    for y, name, color in ((105, "Selected memory", UNBOUNDED), (143, "Best available in archive", ORACLE)):
        diagram.line([(74, y + 15), (108, y + 15)], color, 3, group)
        diagram.label(name, 120, y, 480, 32, size=27, color=color, parent=group)
    diagram.label("DINO distance to target GT &#8595;", 74, 192, 568,
                  size=27, parent=group)
    x, y, w, h = 74, 244, 840, 272
    ymin, ymax = .2, .7
    px = lambda t: x + w * t / 180
    py = lambda v: y + h * (ymax - v) / (ymax - ymin)
    if any(not ymin <= row[key] <= ymax for values in series.values()
           for row in values for key in ("ci_low", "ci_high")):
        raise ValueError("Diagnostic axis would clip a confidence interval")
    for value in (.2, .3, .4, .5, .6, .7):
        yy = py(value)
        diagram.line([(x, yy), (x + w, yy)], GRID, 1, group)
        diagram.label(f"{value:.1f}", 14, yy - 14, 48, 28, size=25,
                      align="right", color=MUTED, parent=group)
    for metric, color in ((SELECTED, UNBOUNDED), (BEST, ORACLE)):
        rows = series[metric]
        points = [(px(r["time_sec"]), py(r["ci_low"])) for r in rows]
        points += [(px(r["time_sec"]), py(r["ci_high"])) for r in reversed(rows)]
        diagram.band(points, color, group)
        means = [(px(r["time_sec"]), py(r["mean"])) for r in rows]
        diagram.line(means, color, 3, group)
        for xx, yy in means:
            diagram.cell("", xx - 3.5, yy - 3.5, 7, 7,
                         f"ellipse;strokeColor=none;fillColor={color};", group)
    diagram.line([(x, y), (x, y + h), (x + w, y + h)], MUTED, 1.5, group)
    for time in (0, 60, 120, 180):
        diagram.label(str(time), px(time) - 29, y + h + 10, 58, 29, size=25,
                      align="center", color=MUTED, parent=group)
    diagram.label("Generated time (s)", x, y + h + 47, w, 31, size=27,
                  align="center", parent=group)
    diagram.label("Mean and 95% trajectory-bootstrap intervals", 0, 607, 1000, 28,
                  size=24, color=MUTED, parent=group)
    return metadata


def paired_retrieval_panel(diagram, spec, output, compact=False):
    curves, metadata = compare_retrieval(ROOT / spec["source"], **spec["parameters"])
    metric = metadata["displayed_metric"]
    selected = [r for r in curves if r["metric"] == metric]
    duration = metadata["parameters"]["duration"]
    n = metadata["parameters"]["expected_videos"]
    _, panel_x, panel_width = compact_panel_geometry(diagram.width) if compact else (0, 1148, 1000)
    group = panel(diagram, "retrieval", panel_x, panel_width,
                  "(b) Higher selected-memory fidelity",
                  None if compact else f"Shared-source test / {duration} s / {n} matched trajectories",
                  y=18 if compact else 22, height=388 if compact else 636)
    legend = ((74, 49, "Unbounded", UNBOUNDED), (390, 49, "KEEPSAKE", KEEPSAKE)) if compact else (
        (74, 105, "Unbounded selection", UNBOUNDED), (74, 143, "KEEPSAKE selection (B32)", KEEPSAKE))
    for xx, yy, name, color in legend:
        diagram.line([(xx, yy + 15), (xx + 34, yy + 15)], color, 3, group)
        diagram.label(name, xx + 46, yy, (240 if xx == 74 else 320) if compact else 800,
                      32, size=27, color=color, parent=group)
    diagram.label("DINO cosine distance to reference &#8595;", 74, 83 if compact else 192,
                  panel_width - 100 if compact else 900, size=26 if compact else 27, parent=group)
    x, y, w, h = (74, 124, panel_width - 124, 202) if compact else (74, 244, 840, 272)
    minimum, maximum = spec.get("y_limits", [0, max(.1, math.ceil(max(r["ci_high"] for r in selected) * 10) / 10)])
    if (not all(math.isfinite(v) for v in (minimum, maximum)) or not 0 <= minimum < maximum <= 2
            or any(not minimum <= r[key] <= maximum for r in selected for key in ("ci_low", "mean", "ci_high"))):
        raise ValueError("Retrieval axis limits would clip a mean or confidence interval")
    metadata["presentation"] = dict(y_limits=[minimum, maximum], axis="DINO cosine distance to reference",
                                   reference="GT at each selected memory's own historical index",
                                   zero_origin=minimum == 0, compact=compact)
    px = lambda t: x + w * t / duration
    py = lambda value: y + h * (maximum - value) / (maximum - minimum)
    for tick in range(math.ceil(minimum * 10), math.floor(maximum * 10) + 1):
        value = tick / 10
        yy = py(value)
        diagram.line([(x, yy), (x + w, yy)], GRID, 1, group)
        diagram.label(f"{value:.1f}", 14, yy - 14, 48, 28, size=25,
                      align="right", color=MUTED, parent=group)
    for run, color in (("baseline", UNBOUNDED), ("slam_b32_covisibility", KEEPSAKE)):
        rows = [r for r in selected if r["run"] == run]
        if any(not 0 <= r["time_sec"] <= duration for r in rows):
            raise ValueError("Time-bin means lie outside the plotted horizon")
        if any(r["ci_high"] > r["ci_low"] for r in rows):
            points = [(px(r["time_sec"]), py(r["ci_low"])) for r in rows]
            points += [(px(r["time_sec"]), py(r["ci_high"])) for r in reversed(rows)]
            diagram.band(points, color, group)
        points = [(px(r["time_sec"]), py(r["mean"])) for r in rows]
        diagram.line(points, color, 3, group)
        for xx, yy in points:
            diagram.cell("", xx - 3.5, yy - 3.5, 7, 7,
                         f"ellipse;strokeColor=none;fillColor={color};", group)
    diagram.line([(x, y), (x, y + h), (x + w, y + h)], MUTED, 1.5, group)
    for time in (0, duration / 4, duration / 2, 3 * duration / 4, duration):
        diagram.label(f"{time:g}", px(time) - 29, y + h + (4 if compact else 10),
                      58, 25 if compact else 29, size=25,
                      align="center", color=MUTED, parent=group)
    diagram.label("Generated time (s)", x, y + h + (32 if compact else 47), w,
                  28 if compact else 31, size=27,
                  align="center", parent=group)
    if not compact:
        diagram.label("Shared source images / GT at each memory's historical index", 0, 607, 1000, 28,
                      size=23, color=MUTED, parent=group)
    write_csv(output.with_suffix(".retrieval_comparison.csv"), curves)
    return metadata


def sample_panel(diagram, root, specifications):
    if len(specifications) != 5:
        raise ValueError("This layout requires five timestamps from one scene")
    if (len({s["case"] for s in specifications}) != 1
            or len({s["display_name"] for s in specifications}) != 1):
        raise ValueError("All columns must come from the same trajectory")
    case_path = root / (specifications[0]["case"] + ".json")
    case = json.loads(case_path.read_text())
    manifest = case["source_manifest_item"]
    positions = [spec["position"] for spec in specifications]
    if (manifest["duration_sec"] != 60 or not math.isfinite(case["fps"]) or case["fps"] <= 0
            or any(not isinstance(p, int) or not 0 <= p < len(case["frames"]) for p in positions)):
        raise ValueError("Invalid sample duration, FPS or strip position")
    frames = [case["frames"][p] for p in positions]
    if (any(a >= b for a, b in zip(positions, positions[1:]))
            or any(a >= b for a, b in zip(frames, frames[1:]))):
        raise ValueError("Sample timestamps must be distinct and ordered")
    group = panel(diagram, "samples", 26, 2128, "(c) Scene comparison",
                  f"MemCam / {specifications[0]['display_name']} / selected sequence from a 60 s rollout",
                  y=710, height=848)
    evidence = []
    image_height = None
    rows = (("Ground truth", "Ground<br>truth", "gt_check", INK, 128),
            ("Unbounded", "Unbounded", "unbounded", UNBOUNDED, 353),
            ("KEEPSAKE-32", "KEEPSAKE<br>B32", "ours", KEEPSAKE, 578))
    for col, spec in enumerate(specifications):
        position, frame = positions[col], frames[col]
        x, display_w = 160 + col * 394, 384
        diagram.label(f"{frame / case['fps']:.1f} s", x, 88,
                      display_w, 32, size=29, align="center", parent=group)
        for name, label, suffix, color, yy in rows:
            path = root / f"{spec['case']}_{suffix}_bare.png"
            # make_memory_strips.strip fixes tile width at 384 and gutters at 8.
            tile_w, gutter = 384, 8
            with Image.open(path) as im:
                expected = len(case["frames"]) * tile_w + (len(case["frames"]) - 1) * gutter
                if im.width != expected:
                    raise ValueError(f"Unexpected saved strip layout: {path}")
                if image_height is None:
                    image_height = im.height
                if im.height != image_height:
                    raise ValueError("GT and generated strips must have the same image dimensions")
                start = position * (tile_w + gutter)
                tile = im.crop((start, 0, start + tile_w, im.height))
                display_h = display_w * tile.height / tile.width
                if display_h > 211:
                    raise ValueError("Sample aspect ratio does not fit the three-row layout")
                if col == 0:
                    diagram.label(label, 0, yy, 142, display_h, size=25, bold=True,
                                  align="right", color=color, parent=group)
                diagram.picture(tile, x, yy, display_w, display_h, group)
            source_kind = "ground_truth" if suffix == "gt_check" else "generated_video"
            dataset_frame = manifest["start_frame"] + frame
            source_path = (str(Path(manifest["gt_frames_dir"]) / f"{dataset_frame:04d}.png") if suffix == "gt_check"
                           else case["source_videos"]["Ours" if suffix == "ours" else "Unbounded"])
            evidence.append(dict(policy=name, scene=case["scene"], row=case["row"],
                                 frame=frame, time_sec=frame / case["fps"],
                                 dataset_start_frame=manifest["start_frame"],
                                 dataset_frame=dataset_frame,
                                 seed=manifest["split_seed"], duration_sec=60,
                                 source_kind=source_kind, source_path=source_path,
                                 local_strip=str(path.relative_to(ROOT)), strip_sha256=digest(path),
                                 crop_box=[start, 0, start + tile_w, tile.height],
                                 metadata_sha256=digest(case_path)))
    diagram.label("Each column uses the same frame index for ground truth and both methods", 0, 806, 2128, 28,
                  size=24, color=MUTED, parent=group)
    return evidence


def annotated_sample_panel(diagram, root, spec, assets):
    """Show a local content contrast without asserting whole-frame GT agreement."""
    case_path = root / (spec["case"] + ".json")
    case = json.loads(case_path.read_text())
    item = case["source_manifest_item"]
    position = spec["position"]
    if (type(position) is not int or not 0 <= position < len(case["frames"])
            or item["duration_sec"] != 60 or case["fps"] != item["fps"]):
        raise ValueError("Invalid matched-frame comparison identity")
    frame = case["frames"][position]
    if frame != spec["frame"] or not 76 < frame < item["num_frames"]:
        raise ValueError("Expected a verified post-initial-chunk frame")
    box = spec["highlight_box"]
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError("Highlight box must lie inside the full image")
    assets.mkdir(parents=True, exist_ok=True)
    gt_path = root / f"{spec['case']}_gt_check_bare.png"
    with Image.open(gt_path) as strip:
        if strip.width != len(case["frames"]) * 392 - 8:
            raise ValueError("Unexpected GT strip layout")
        crop = [position * 392, 0, position * 392 + 384, strip.height]
        gt = strip.crop(crop).convert("RGB")
        gt.save(assets / "target_gt.png")
    evidence = [dict(policy="Ground truth", displayed_in_main=False,
                     local_strip=str(gt_path), strip_sha256=digest(gt_path), crop_box=crop,
                     local_image=str(assets / "target_gt.png"),
                     image_sha256=digest(assets / "target_gt.png"),
                     source_kind="ground_truth",
                     source_path=str(Path(item["gt_frames_dir"]) / f"{item['start_frame'] + frame:04d}.png"))]
    group = panel(diagram, "samples", 26, 2128, "(c) Generated scene content",
                  f"MemCam / {spec['display_name']} / same target frame at {frame / case['fps']:.2f} s",
                  y=710, height=848)
    reference = Diagram()
    model = reference.file.find(".//mxGraphModel")
    model.set("pageHeight", "650")
    reference.rect(0, 0, WIDTH, 650, "#FFFFFF")
    reference.label("GT check: local content contrast, not an exact scene reconstruction", 26, 18,
                    2128, 42, size=32, bold=True)
    reference.label(f"{case['scene']} / generated frame {frame} / GT dataset frame {item['start_frame'] + frame}",
                    26, 64, 2128, size=25, color=MUTED)
    pictures = [("Ground truth", gt, INK)]
    size = None
    for col, (policy, folder, color) in enumerate((
            ("KEEPSAKE (Ours), B32", "KEEPSAKE_B32", KEEPSAKE),
            ("Unbounded", "Unbounded", UNBOUNDED))):
        video = Path(spec["videos_directory"]).expanduser() / folder / (item["output_prefix"] + "custom.mp4")
        image_path = extract_frames(video, [frame], assets / folder)[frame]
        with Image.open(image_path) as image:
            im = image.convert("RGB")
        if size is not None and im.size != size:
            raise ValueError("Generated videos have different frame dimensions")
        size = im.size
        x, w, yy = col * 1084, 1044, 145
        h = w * im.height / im.width
        if h > 580:
            raise ValueError("Frame aspect ratio does not fit the paired layout")
        diagram.label(policy, x, 97, w, 40, size=32, color=color, bold=True, parent=group)
        diagram.picture(im, x, yy, w, h, group)
        diagram.cell("", x + box[0] * w, yy + box[1] * h,
                     (box[2] - box[0]) * w, (box[3] - box[1]) * h,
                     f"rounded=0;fillColor=none;strokeColor={color};strokeWidth=3;", group)
        diagram.label(spec["annotations"][folder], x, 740, w, 40,
                      size=30, color=color, bold=True, parent=group)
        pictures.append((policy, im, color))
        evidence.append(dict(policy=folder, displayed_in_main=True, source_kind="generated_video",
                             source_path=str(video), video_sha256=digest(video),
                             local_image=str(image_path), image_sha256=digest(image_path),
                             highlight_box=box, annotation=spec["annotations"][folder]))
    for col, (label, im, color) in enumerate(pictures):
        x, w, yy = 26 + col * 716, 696, 162
        h = w * im.height / im.width
        reference.label(label, x, 112, w, 42, size=30, color=color, bold=True)
        reference.picture(im, x, yy, w, h, "1")
        reference.cell("", x + box[0] * w, yy + box[1] * h,
                       (box[2] - box[0]) * w, (box[3] - box[1]) * h,
                       f"rounded=0;fillColor=none;strokeColor={color};strokeWidth=3;")
    reference.label(spec["gt_check_note"],
                    26, 582, 2128, 40, size=28, color=MUTED)
    reference.save(assets / "gt_check.drawio")
    diagram.label("Boxes mark the open gap in GT and the corresponding region in each generated frame.",
                  0, 803, 2128, 32, size=25, color=MUTED, parent=group)
    for row in evidence:
        row.update(scene=case["scene"], row=case["row"], frame=frame,
                   time_sec=frame / case["fps"], dataset_frame=item["start_frame"] + frame,
                   dataset_start_frame=item["start_frame"], duration_sec=60,
                   seed=item["split_seed"], metadata_sha256=digest(case_path))
    return evidence


def brighten_for_display(image, gamma):
    """Apply one fixed RGB lookup table, never a per-frame normalization."""
    if not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Display gamma must be finite and in (0, 1]")
    lut = [round(255 * (value / 255) ** gamma) for value in range(256)]
    return image.convert("RGB").point(lut * 3)


def right_crop_bounds(size, fraction):
    """Return one fixed right-side crop in source-pixel coordinates."""
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Right crop fraction must be finite and in [0, 1)")
    width, height = size
    retained_width = round(width * (1 - fraction))
    if retained_width < 1:
        raise ValueError("Right crop removes the entire image")
    return (0, 0, retained_width, height)


def revisit_sample_panel(diagram, root, spec, assets, compact=False):
    """Embed a matched visit/intervening-view/nearby-return sequence."""
    case_path = root / (spec["case"] + ".json")
    case = json.loads(case_path.read_text())
    item, frames = case["source_manifest_item"], spec["frames"]
    if (len(frames) != 3 or any(type(n) is not int or not 76 < n < item["num_frames"] for n in frames)
            or not frames[0] < frames[1] < frames[2]
            or item["duration_sec"] != 60 or case["fps"] != item["fps"]
            or not math.isfinite(case["fps"]) or case["fps"] <= 0):
        raise ValueError("Revisit needs three ordered, matched post-initial-chunk frames")
    gamma = spec.get("display_gamma", 1)
    brighten_for_display(Image.new("RGB", (1, 1)), gamma)
    crop_fraction = spec.get("display_right_crop_fraction", 0)
    right_crop_bounds((640, 352), crop_fraction)
    box = spec.get("highlight_box")
    if box is not None and (len(box) != 4 or not all(math.isfinite(v) for v in box)
                            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= .84):
        raise ValueError("Highlight box must fit the full image above the annotation strip")
    if box is not None and box[2] > 1 - crop_fraction:
        raise ValueError("Right crop would remove part of the highlighted region")
    diagnostic_path = ROOT / spec["diagnostics"]
    diagnostic = json.loads(diagnostic_path.read_text())
    matches = [(record, group) for record in diagnostic["records"]
               if (record["row"], record["scene"], record["profile"]) ==
                  (case["row"], case["scene"], spec["profile"])
               for category in record["gt_checks"] for group in category.get("checked", [])
               if group["anchor"] == spec["anchor"]
               and frames[0] in group["frames"] and frames[-1] in group["frames"]]
    if not matches:
        raise ValueError("Endpoints are not in the specified saved pose-return group")
    record, checked = matches[0]
    pair = next((p for p in checked["gt_pairs"] if (p["a"], p["b"]) == (frames[0], frames[-1])), None)
    if pair is None:
        raise ValueError("Missing endpoint GT check in the saved diagnostic")
    audit = dict(source=str(diagnostic_path), sha256=digest(diagnostic_path),
                 parameters=diagnostic["parameters"], profile=record["profile"],
                 position_tolerance_from_anchor_m=record["position_m"],
                 rotation_tolerance_from_anchor_deg=record["rotation_deg"],
                 checked_group=checked, endpoint_gt_ssim=pair["ssim"],
                 interpretation="Nearby pose return, not identical-view or GT-reconstruction evidence. "
                 "The middle frame was selected visually; it is not a measured disappearance interval.")
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "revisit_check.json").write_text(json.dumps(audit, indent=2) + "\n")
    group = panel(diagram, "samples", 26, diagram.width - 52 if compact else 2128, "(c) More consistent scene recall",
                  None if compact else spec.get("visual_note", f"MemCam / {spec['display_name']} / same timestamps for both methods / 60 s rollout"),
                  y=424 if compact else 710, height=862 if compact else 848)
    full_width = 680 if compact else 630
    w = full_width * (1 - crop_fraction)
    xs = tuple(56 + col * (w + 14) for col in range(3)) if compact else (180, 834, 1488)
    stages = ("Initial view", "Intervening view", "Revisit")
    for x, frame, stage in zip(xs, frames, stages):
        diagram.label(f"{stage} / {frame / case['fps']:.1f} s", x, 50 if compact else 88,
                      w, 30 if compact else 36,
                      size=29, align="center", bold=True, parent=group)
    evidence, size = [], None
    for policy, label, color, yy in (("Unbounded", "Unbounded", UNBOUNDED, 88 if compact else 138),
                                     ("KEEPSAKE_B32", "KEEPSAKE" if compact else "KEEPSAKE<br>(Ours, B32)", KEEPSAKE,
                                      474 if compact else 498)):
        video = Path(spec["videos_directory"]).expanduser() / policy / (item["output_prefix"] + "custom.mp4")
        decoded = extract_frames(video, frames, assets / policy)
        video_hash = digest(video)
        diagram.label(label, 0, yy, 40 if compact else 163, 374 if compact else 330,
                      size=27, color=color, bold=True,
                      align="center" if compact else "right", parent=group,
                      extra="horizontal=0;" if compact else "")
        for x, frame, stage in zip(xs, frames, stages):
            path = decoded[frame]
            with Image.open(path) as image:
                native_size = image.size
                crop_bounds = right_crop_bounds(native_size, crop_fraction)
                im = brighten_for_display(image.crop(crop_bounds), gamma)
            display_path = path.with_name(f"display_{path.name}")
            im.save(display_path)
            if size is not None and native_size != size:
                raise ValueError("Matched frames have different dimensions")
            size = native_size
            display_width = full_width * im.width / native_size[0]
            h = display_width * im.height / im.width
            if h > (374 if compact else 348):
                raise ValueError("Frame aspect ratio does not fit the revisit layout")
            diagram.picture(im, x, yy, display_width, h, group)
            outlined = box is not None and frame in (frames[0], frames[-1])
            display_box = None
            if outlined:
                display_box = [box[0] * native_size[0] / im.width, box[1],
                               box[2] * native_size[0] / im.width, box[3]]
                if display_box[2] > 1:
                    raise ValueError("Rounded crop boundary clips the highlighted region")
                bx, by = x + display_box[0] * display_width, yy + display_box[1] * h
                bw, bh = (display_box[2] - display_box[0]) * display_width, (box[3] - box[1]) * h
                diagram.cell("", bx, by, bw, bh,
                             f"rounded=0;fillColor=none;strokeColor={color};strokeWidth=6;dashed=0;",
                             group)
                if frame == frames[-1]:
                    diagram.label(spec["revisit_annotations"][policy], bx, by + bh + 7,
                                  bw, 37, size=27, bold=True, color=color, align="center", parent=group,
                                  extra="fillColor=#FFFFFF;spacingLeft=6;spacingRight=6;")
            evidence.append(dict(policy=policy, scene=case["scene"], row=case["row"],
                                 stage=stage, frame=frame, time_sec=frame / case["fps"],
                                 dataset_start_frame=item["start_frame"],
                                 dataset_frame=item["start_frame"] + frame,
                                 seed=item["split_seed"], duration_sec=60,
                                 source_kind="generated_video", source_path=str(video),
                                 video_sha256=video_hash, local_image=str(path),
                                 image_sha256=digest(path), native_dimensions=list(native_size),
                                 crop_box=list(crop_bounds), display_dimensions=list(im.size),
                                 display_image=str(display_path), display_image_sha256=digest(display_path),
                                 display_transform=dict(gamma=gamma, formula="round(255 * (channel / 255) ** gamma)",
                                                        right_crop_fraction=crop_fraction,
                                                        scope="Identical right-side crop and per-channel transform for every frame and policy; display only"),
                                 highlight_box=box if outlined else None,
                                 display_highlight_box=display_box,
                                 annotation=spec["revisit_annotations"][policy] if outlined and frame == frames[-1] else None,
                                 image_processing="Exact-index decode; fixed right-side display crop; uniform display gamma; full original pixels preserved separately. Editable vector outlines/labels are not baked into the frame.",
                                 metadata_sha256=digest(case_path)))
    return evidence, audit


CAPTION = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/ICLR27_Motivation_two_row.pdf}
\caption{\textbf{Memory efficiency, retrieval quality, and generated outputs.}
(a) In MemCam, KEEPSAKE retains 32 rather than 5,397 frames at 180 seconds.
CPU lookup time averages eight sampled queries from one matched trajectory,
using one thread and one timed repeat; timing covers FOV scoring and selection,
not generation or memory maintenance.
(b) In 15 complete-retention 180-second rollouts, the actual selected memory
increasingly differs from target ground truth, while more relevant generated
observations remain available in the eligible archive. Both curves describe
the unbounded archive; the blue curve is an offline ground-truth-assisted
oracle, not KEEPSAKE. Bands are pointwise 95\% trajectory-bootstrap intervals.
(c) One selected trajectory from the separate 60-second suite, shown at five
timestamps. Each column aligns the target ground truth with the corresponding
Unbounded and KEEPSAKE-32 outputs. Ground truth provides a reference for the
intended viewpoint and scene content, including errors that remain in both
outputs. The timestamps need not depict the same camera view; this is not
a verified revisit or a controlled causal error-propagation experiment.}
\label{fig:motivation-two-row}
\end{figure*}
"""

def paired_caption(spec, samples):
    description = (f"(c) A selected {spec['display_name']} frame at "
                   f"{samples[0]['time_sec']:.2f} seconds from the separate 60-second suite. "
                   + spec["caption_detail"])
    return CAPTION[:CAPTION.index("(c) One selected")] + description + r"""
Outlines identify a local difference, not a crop or edit of the images.
The corresponding GT comparison is exported alongside the figure for inspection.
This example is not a verified revisit or a causal error-propagation test.}
\label{fig:motivation-two-row}
\end{figure*}
"""


def revisit_caption(spec, samples):
    times = ", ".join(f"{s['time_sec']:.1f}" for s in samples[:3])
    description = (f"(c) A selected {spec['display_name']} sequence at {times} seconds "
                   "from the separate 60-second suite. Rows use identical frame indices. "
                   + spec["caption_detail"])
    display_note = (f"All six displayed frames receive the same gamma adjustment "
                    f"($\\gamma={spec.get('display_gamma', 1):g}$) for visibility; "
                    "this changes neither source videos nor metric measurements. ")
    if spec.get("highlight_box"):
        display_note += ("Red and green outlines mark the same image region in the first and return frames, "
                         "not a tracked or geometrically registered correspondence. ")
    display_note += display_crop_note(spec) + " "
    return CAPTION[:CAPTION.index("(c) One selected")] + description + " " + display_note + r"""
The endpoint poses belong to a nearby-return group, not an identical camera view;
this is a selected qualitative comparison, not a GT-fidelity score or a causal
error-propagation test. Full decoded originals are preserved separately;
only the stated display adjustments and vector annotations are applied.}
\label{fig:motivation-two-row}
\end{figure*}
"""


def comparison_caption(metadata):
    parameters = metadata["parameters"]
    effective = next(r for r in metadata["summaries"] if r["metric"] == SELECTED)
    return (f"(b) Unbounded versus KEEPSAKE-B32 selection on "
            f"{parameters['expected_videos']} matched {parameters['duration']}-second trajectories "
            f"({metadata['queries_per_policy']:,} reads per policy). "
            "Both policies' actual selected historical indices are scored using the same "
            "unbounded source video, against ground truth at each selected memory's own "
            "historical index. Lower DINO cosine distance means higher selected-source fidelity; "
            "GT is a scoring reference, not a plotted method. Curves average queries within "
            f"sections and equally weight sections within {parameters['bins']} time bins, then trajectories. "
            r"Bands are pointwise 95\% trajectory-bootstrap intervals. "
            "This controlled-source diagnostic does not compare each policy's own memory "
            "pixels or imply better alignment to the current target view; the latter's "
            f"whole-rollout mean DINO mismatches are {effective['unbounded']:.3f} (Unbounded) "
            f"and {effective['keepsake']:.3f} (KEEPSAKE). All component curves accompany the figure.\n")


def display_crop_note(spec):
    fraction = spec.get("display_right_crop_fraction", 0)
    return (f"The rightmost {100 * fraction:g}\\% is cropped identically from all six frames."
            if fraction else "Full frames are shown.")


def compact_caption(spec, metadata, samples):
    params = metadata["parameters"]
    times = ", ".join(f"{s['time_sec']:.1f}" for s in samples[:3])
    return (r"\begin{figure*}[t]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=\textwidth]{figures/ICLR27_Motivation_two_row.pdf}" "\n"
            r"\caption{\textbf{Memory efficiency, selected-memory fidelity, and scene recall.} "
            "All panels use MemCam; KEEPSAKE uses a 32-frame budget throughout. "
            "(a) At 180 seconds, KEEPSAKE retains 32 rather than 5,397 frames. "
            "CPU lookup timings cover FOV scoring and selection: eight queries, one trajectory, "
            "one thread and one repeat; generation and memory maintenance are excluded. "
            f"(b) Actual selections from {params['expected_videos']} matched {params['duration']}-second "
            "trajectories are evaluated using shared unbounded-source pixels against GT at "
            "each selected memory's historical index. Lower DINO cosine distance indicates "
            "higher selected-memory fidelity, not better current-query alignment. "
            r"Bands are pointwise 95\% trajectory-bootstrap intervals; the y-axis is zoomed. "
            f"(c) A selected {spec['display_name']} sequence ({times} seconds) shows a closer "
            "return in KEEPSAKE after an intervening view, while Unbounded changes the scene composition. "
            "Both rows use identical timestamps "
            f"and the same display gamma ($\\gamma={spec.get('display_gamma', 1):g}$). "
            + display_crop_note(spec) + " Boxes mark the same image region, not registered correspondences. "
            "Match labels describe visual recurrence relative to the initial generated view. "
            "This nearby-pose return illustrates generated appearance consistency, not exact "
            "GT reconstruction or causal error propagation.}\n"
            r"\label{fig:motivation-two-row}" "\n"
            r"\end{figure*}" "\n")


def build(config_path, output):
    config = json.loads(config_path.read_text())
    compact = config.get("layout") == "compact"
    if compact and not ("retrieval_comparison" in config and "revisit_triplet" in config):
        raise ValueError("Compact layout requires the paired retrieval plot and revisit triplet")
    width = WIDTH
    if compact:
        crop_fraction = config["revisit_triplet"].get("display_right_crop_fraction", 0)
        right_crop_bounds((640, 352), crop_fraction)
        width = round(WIDTH - 3 * 680 * crop_fraction)
        if width < 1600:
            raise ValueError("Right crop leaves insufficient canvas width for the chart labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    height = COMPACT_HEIGHT if compact else HEIGHT
    diagram = Diagram(height=height, width=width)
    diagram.rect(0, 0, width, height, "#FFFFFF")
    separator_x = compact_panel_geometry(width)[1] - 16 if compact else 1076
    diagram.line([(separator_x, 24), (separator_x, 398 if compact else 648)], GRID, 1)
    divider = 410 if compact else 684
    diagram.line([(26, divider), (width - 26, divider)], GRID, 1)
    efficiency_panel(diagram, config["efficiency"], compact=compact)
    comparison = "retrieval_comparison" in config
    if comparison:
        metadata = paired_retrieval_panel(diagram, config["retrieval_comparison"], output, compact=compact)
        retrieval_provenance = dict(retrieval_source=metadata["source"],
                                    retrieval_source_sha256=metadata["source_sha256"],
                                    retrieval_comparison=metadata)
    else:
        directory = ROOT / config["retrieval_directory"]
        metadata = retrieval_panel(diagram, directory)
        retrieval_provenance = dict(retrieval_source=str(directory / "curves.csv"),
                                    retrieval_source_sha256=digest(directory / "curves.csv"),
                                    retrieval_provenance_sha256=digest(directory / "provenance.json"))
    paired = "sample_comparison" in config
    revisit = "revisit_triplet" in config
    if revisit and paired:
        raise ValueError("Choose one sample layout")
    if revisit:
        samples, audit = revisit_sample_panel(diagram, ROOT / config["sample_directory"],
                                             config["revisit_triplet"],
                                             output.parent / (output.stem + "_sample_assets"), compact=compact)
        caption = revisit_caption(config["revisit_triplet"], samples)
    elif paired:
        samples = annotated_sample_panel(diagram, ROOT / config["sample_directory"],
                                         config["sample_comparison"],
                                         output.parent / (output.stem + "_sample_assets"))
        caption = paired_caption(config["sample_comparison"], samples)
    else:
        samples = sample_panel(diagram, ROOT / config["sample_directory"], config["samples"])
        caption = CAPTION
    if comparison:
        caption = (caption[:caption.index("(b)")] + comparison_caption(metadata)
                   + caption[caption.index("(c)"):])
    if compact:
        caption = compact_caption(config["revisit_triplet"], metadata, samples)
    diagram.save(output)
    output.with_suffix(".tex").write_text(caption)
    provenance = dict(configuration=str(config_path), configuration_sha256=digest(config_path),
                      builder_sha256=digest(Path(__file__)), efficiency=config["efficiency"],
                      layout=dict(name="compact_two_row" if compact else "two_row", width=width, height=height),
                      **retrieval_provenance,
                      retrieval_parameters=metadata["parameters"], samples=samples,
                      sample_note=config["sample_note"],
                      editability="Native draw.io chart marks, axes, text, confidence-band shapes and embedded PNG frames",
                      limitations="Evidence scopes are stated separately for each panel. No new generation or causal tests; generated-image recurrence does not establish GT correctness.")
    if revisit:
        provenance["revisit_check"] = audit
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Editable diagram: {output}")
    print(f"Caption: {output.with_suffix('.tex')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper/motivation_teaser_inputs.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/ICLR27_Motivation_two_row.drawio")
    args = parser.parse_args()
    build(args.config, args.output)


if __name__ == "__main__":
    main()
