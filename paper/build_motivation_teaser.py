"""Build a data-backed draw.io teaser: two charts above a full-width sample grid.

Charts, labels and confidence bands are vector objects. Real generated and GT
frames are embedded PNGs, extracted intact from existing label-free strips.
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
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zlib

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#24282C", "#687078", "#E4E7E9"
UNBOUNDED, KEEPSAKE, ORACLE = "#BC5147", "#277A60", "#397BA5"
WIDTH, HEIGHT = 2180, 1580
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
    def __init__(self):
        self.file = ET.Element("mxfile", host="app.diagrams.net")
        page = ET.SubElement(self.file, "diagram", id="keepsake-teaser", name="Two-row teaser")
        model = ET.SubElement(page, "mxGraphModel", dx=str(WIDTH), dy=str(HEIGHT), grid="1",
                              gridSize="10", page="1", pageScale="1", pageWidth=str(WIDTH),
                              pageHeight=str(HEIGHT), background="#FFFFFF", math="0", shadow="0")
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
    diagram.label(subtitle, 0, 48, width, 32, size=25, color=MUTED, parent=group)
    return group


def efficiency_panel(diagram, data):
    group = panel(diagram, "efficiency", 26, 970, "(a) Efficiency", "MemCam / 180-second rollout")
    records = {r["policy"]: r for r in data["records"]}
    if set(records) != {"Unbounded", "KEEPSAKE"}:
        raise ValueError("Efficiency requires Unbounded and KEEPSAKE records")
    if (data["duration_sec"], data["videos"], data["sampled_queries"], data["repeats"], data["threads"]) != (180, 1, 8, 1, 1):
        raise ValueError("Update the timing labels/caption to match the input protocol")
    for key in ("final_stored_frames", "query_ms"):
        if any(not math.isfinite(float(r[key])) or r[key] <= 0 for r in records.values()):
            raise ValueError("Resource measurements must be finite and positive")
    for top, key, title, maximum, ticks in (
        (115, "final_stored_frames", "Retained frames", 6000, (0, 2000, 4000, 6000)),
        (340, "query_ms", "CPU lookup time (ms/query)", 4500, (0, 1500, 3000, 4500)),
    ):
        diagram.label(title, 0, top, 580, size=29, bold=True, parent=group)
        x, plot_w = 177, 680
        for tick in ticks:
            tx = x + plot_w * tick / maximum
            diagram.line([(tx, top + 44), (tx, top + 136)], GRID, 1, group)
            diagram.label(f"{tick:,}", tx - 40, top + 145, 80, 29, size=23,
                          color=MUTED, align="center", parent=group)
        for row, (name, color) in enumerate((("Unbounded", UNBOUNDED), ("KEEPSAKE", KEEPSAKE))):
            value = records[name][key]
            y, bar_w = top + 46 + row * 57, plot_w * value / maximum
            diagram.label(name, 0, y, 167, 32, size=27, color=color,
                          bold=name == "KEEPSAKE", parent=group)
            diagram.rect(x, y, bar_w, 31, color, group)
            label = f"{int(value):,}" if key == "final_stored_frames" else f"{value:,.1f}"
            diagram.label(label, x + bar_w + 10, y - 2, 115, 36, size=30,
                          bold=True, color=color, parent=group)
        diagram.line([(x, top + 136), (x + plot_w, top + 136)], MUTED, 1, group)
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


def build(config_path, output):
    config = json.loads(config_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    diagram = Diagram()
    diagram.rect(0, 0, WIDTH, HEIGHT, "#FFFFFF")
    diagram.line([(1076, 24), (1076, 648)], GRID, 1)
    diagram.line([(26, 684), (2154, 684)], GRID, 1)
    efficiency_panel(diagram, config["efficiency"])
    directory = ROOT / config["retrieval_directory"]
    metadata = retrieval_panel(diagram, directory)
    samples = sample_panel(diagram, ROOT / config["sample_directory"], config["samples"])
    diagram.save(output)
    output.with_suffix(".tex").write_text(CAPTION)
    provenance = dict(configuration=str(config_path), configuration_sha256=digest(config_path),
                      builder_sha256=digest(Path(__file__)), efficiency=config["efficiency"],
                      retrieval_source=str(directory / "curves.csv"),
                      retrieval_source_sha256=digest(directory / "curves.csv"),
                      retrieval_provenance_sha256=digest(directory / "provenance.json"),
                      retrieval_parameters=metadata["parameters"], samples=samples,
                      sample_note=config["sample_note"],
                      editability="Native draw.io chart marks, axes, text, confidence-band shapes and embedded PNG frames",
                      limitations="Different evidence scopes: single-trajectory CPU timing, 15-trajectory 180 s retrieval diagnostic, five selected timestamps from one 60 s rollout with GT. No new experiments or causal tests.")
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
