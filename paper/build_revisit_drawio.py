"""Build editable, self-contained three-policy revisit figures from video frames."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build_motivation_teaser import Diagram
from paper.compare_local_rollouts import probe
from paper.make_180s_gt_comparisons import decode_cached, digest, save_json

METHODS = (("Unbounded", "#454545"), ("FIFO", "#BF665D"), ("KEEPSAKE", "#277A60"))
BOX_COLORS = {"Unbounded": "#DC4747", "FIFO": "#DC4747", "KEEPSAKE": "#24A05A"}


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_highlights(record):
    seen = set()
    width, height = record["metadata"]["width"], record["metadata"]["height"]
    for box in record.get("highlights", []):
        key = (box["method"], box["frame"], box["landmark"])
        if (key in seen or box["method"] not in BOX_COLORS
                or box["frame"] not in (record["frames"][0], record["frames"][-1])):
            raise ValueError("Highlights must identify distinct landmarks in first/return frames")
        x, y, w, h = box["xywh"]
        if (not all(math.isfinite(v) for v in (x, y, w, h))
                or not (0 <= x < x + w <= width and 0 <= y < y + h <= height)):
            raise ValueError("Highlight rectangle must lie within the source frame")
        seen.add(key)
    for method, _, landmark in seen:
        if not all((method, frame, landmark) in seen for frame in (record["frames"][0], record["frames"][-1])):
            raise ValueError("Each highlighted landmark needs both first and return boxes")


def prepare_case(spec, assets):
    frames = spec["frames"]
    timeline_fps = float(spec["timeline_fps"])
    if not math.isfinite(timeline_fps) or timeline_fps <= 0:
        raise ValueError("A positive rollout FPS is required separately from video playback FPS")
    if len(frames) != 3 or any(type(i) is not int for i in frames) or frames != sorted(set(frames)):
        raise ValueError("Expected three distinct, increasing integer frame indices")
    videos = {method: resolve(spec["videos"][method]) for method, _ in METHODS}
    metadata = {method: probe(path) for method, path in videos.items()}
    info = metadata["Unbounded"]
    if any(value != info for value in metadata.values()) or info != spec["expected_video"]:
        raise ValueError(f"Video count, FPS or geometry mismatch: {metadata}")
    if frames[0] < 0 or frames[-1] >= info["frames"]:
        raise ValueError("Frame index outside video")
    if len({path.name for path in videos.values()}) != 1:
        raise ValueError("The three policies must use the same trajectory filename")

    evidence = spec["revisit_evidence"]
    source = resolve(evidence["csv"])
    with source.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if all(row.get(k) == str(v) for k, v in evidence["match"].items())]
    if len(matches) != 1:
        raise ValueError("Expected one recorded near-pose return for this trajectory")
    event = matches[0]
    if (event["revisit_type"] != "exact_pose"
            or event["scene"] != spec["scene"]
            or int(event["start_frame"]) != spec["start_frame"]
            or int(event["duration_sec"]) != spec["duration_sec"]
            or [int(event["frame_i"]), int(event["frame_j"])] != [frames[0], frames[-1]]
            or not 0 <= float(event["position_distance"]) <= 0.25
            or not 0 <= float(event["rotation_deg"]) <= 5):
        raise ValueError("Revisit event does not match the plotted trajectory and endpoints")
    for frame, key in ((frames[0], "time_i_sec"), (frames[-1], "time_j_sec")):
        if abs(float(event[key]) - frame / timeline_fps) > 1e-8:
            raise ValueError("Revisit record uses a different frame/time origin")
    record = dict(spec, metadata=info, frame_paths={}, source_sha256={},
                  source_videos={k: str(p) for k, p in videos.items()},
                  event=event, event_source=str(source), event_source_sha256=digest(source))
    for method, path in videos.items():
        paths, receipt = decode_cached(path, frames, assets / method)
        record["frame_paths"][method] = {str(i): str(p.resolve()) for i, p in paths.items()}
        record["source_sha256"][method] = receipt["identity"]["source_sha256"]
    return record


def make_diagram(record):
    validate_highlights(record)
    tile_w = 640
    tile_h = tile_w * record["metadata"]["height"] / record["metadata"]["width"]
    left, right, gap, top, bottom = 238, 12, 12, 144, 12
    width = left + 3 * tile_w + 2 * gap + right
    height = top + 3 * tile_h + 2 * gap + bottom
    diagram = Diagram(width=width, height=height)
    page = diagram.file.find("diagram")
    page.set("id", record["stem"])
    page.set("name", f"{record['system']} - {record['scene']}")
    diagram.label(f"{record['system']}, {record['duration_sec']} s", left, 8, 950, 50,
                  size=42, bold=True)
    diagram.label(record["scene"], width - 760, 8, 748, 50, size=34, align="right")
    for col, (phase, frame) in enumerate(zip(("First visit", "Intervening view", "Revisit"), record["frames"])):
        x = left + col * (tile_w + gap)
        diagram.label(phase, x, 65, tile_w, 42, size=36, align="center")
        seconds = f"{frame / record['timeline_fps']:.1f}".removesuffix(".0")
        diagram.label(f"{seconds} s", x, 108, tile_w, 29, size=28, align="center", color="#555555")
        for row, (method, color) in enumerate(METHODS):
            y = top + row * (tile_h + gap)
            with Image.open(record["frame_paths"][method][str(frame)]) as image:
                diagram.picture(image.convert("RGB"), x, y, tile_w, tile_h, "1")
            cell = list(diagram.root)[-1]
            cell.set("id", f"frame-{method.lower()}-{frame}")
            cell.set("value", "")
            scale = tile_w / record["metadata"]["width"]
            for number, box in enumerate(record.get("highlights", [])):
                if box["method"] != method or box["frame"] != frame:
                    continue
                bx, by, bw, bh = (v * scale for v in box["xywh"])
                diagram.cell("", x + bx, y + by, bw, bh,
                             "shape=rectangle;rounded=0;fillColor=none;"
                             f"strokeColor={BOX_COLORS[method]};strokeWidth=4;"
                             "dashed=0;shadow=0;", name=f"highlight-{method.lower()}-{frame}-{number}")
            if col == 0:
                diagram.label(method, 6, y + tile_h / 2 - 26, left - 20, 52,
                              size=34, color=color, align="right", bold=method == "KEEPSAKE")
    return diagram


def build(config, output):
    output.mkdir(parents=True, exist_ok=True)
    specs = json.loads(config.read_text())
    if len({s["stem"] for s in specs}) != len(specs):
        raise ValueError("Duplicate figure stem")
    combined = ET.Element("mxfile", host="app.diagrams.net")
    for spec in specs:
        if Path(spec["stem"]).name != spec["stem"]:
            raise ValueError("Invalid figure stem")
        record = prepare_case(spec, output / "frames" / spec["stem"])
        diagram = make_diagram(record)
        diagram.save(output / f"{spec['stem']}.drawio")
        combined.append(diagram.file.find("diagram"))
        record.update(
            selection="Manually selected qualitative example; all policies use identical video-relative frame indices.",
            display="Original decoded full frames with separate editable rectangle overlays; no brightness, color, crop, alignment or generative edits.",
            interpretation="Near-view return, not GT reconstruction. The middle frame is an intervening view, not a measured disappearance interval.",
            config_sha256=digest(config), renderer_sha256=digest(Path(__file__)))
        save_json(output / f"{spec['stem']}.provenance.json", record)
        (output / f"{spec['stem']}.caption.txt").write_text(
            f"Revisit comparison for {spec['system']} ({spec['duration_sec']}-second rollout). "
            "Unbounded, FIFO B32 and KEEPSAKE B32 are shown at identical timestamps. "
            f"{spec['description']} {spec.get('highlight_description', '')}\n")
        print(f"Created {output / (spec['stem'] + '.drawio')}", flush=True)
    ET.indent(combined)
    ET.ElementTree(combined).write(output / "revisit_comparisons.drawio", encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper/configs/revisit_drawio_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/revisit_editable")
    args = parser.parse_args()
    build(args.config, args.output)


if __name__ == "__main__":
    main()
