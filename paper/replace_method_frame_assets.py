"""Replace draw.io placeholders with trace-checked frames from one MemCam update."""

import argparse
import base64
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
VIDEO = "seed0_Warehouse_0_3496_60s_custom.mp4"
SECTION = 12
STRIDE = 76
CELL_FRAMES = {
    40: 913, 42: 928, 44: 943, 46: 958, 48: 973, 50: 988,
    92: 0, 95: 147, 98: 304, 101: 454,
    107: 935, 110: 936, 113: 987, 116: 988,
    124: 0, 127: 147, 130: 935, 133: 987,
    163: 41, 165: 147, 167: 0, 169: 42, 171: 304, 173: 454,
    175: 68, 177: 30, 179: 935, 181: 988, 183: 987,
    217: 0, 228: 304, 234: 935, 240: 41, 247: 42, 254: 987,
    272: 0, 280: 147, 283: 304, 286: 454, 289: 935, 292: 988,
}
OLD_CELLS = (92, 95, 98, 101)
NEW_CELLS = (107, 110, 113, 116)
KEPT_CELLS = (217, 228, 234, 272, 280, 283, 286, 289, 292)
EVICTED_CELLS = (240, 247, 254)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_bank(events, section):
    evictions = defaultdict(list)
    for row in events:
        if row.get("event") == "memory_eviction":
            evictions[int(row["section_idx"])].append(row)
    bank = {0}
    for s in range(section + 1):
        before = bank.copy()
        candidates = before | set(range(s * STRIDE, (s + 1) * STRIDE + 1))
        removed = [int(e["evicted_memory_frame"]) for e in evictions[s]]
        if len(removed) != len(set(removed)) or not set(removed) <= candidates:
            raise ValueError(f"Invalid eviction IDs in section {s}")
        bank = candidates - set(removed)
        if len(bank) != 32 or any(e["stored_memory_size"] != 32 for e in evictions[s]):
            raise ValueError(f"Incomplete B32 trace at section {s}")
    return before, candidates - before, bank, {e["evicted_memory_frame"]: e for e in evictions[section]}


def validate_roles(old, new, kept, evicted):
    for cells, available, name in ((OLD_CELLS, old, "existing"), (NEW_CELLS, new, "new"),
                                  (KEPT_CELLS, kept, "retained"),
                                  (EVICTED_CELLS, evicted, "evicted")):
        for cell in cells:
            if CELL_FRAMES[cell] not in available:
                raise ValueError(f"Cell {cell}: frame {CELL_FRAMES[cell]} is not {name}")
    if not set(CELL_FRAMES.values()) <= old | new:
        raise ValueError("Illustrated frame is outside this update's candidate bank")
    if evicted[987]["eviction_nearest_covisible_frame"] != 988 or 988 not in kept:
        raise ValueError("The illustrated closest substitute must be logged and retained")


def extract_frames(video, frames, output):
    """Decode by zero-based frame index, without seek rounding or image retouching."""
    frames = sorted(set(frames))
    output.mkdir(parents=True, exist_ok=True)
    select = "+".join(f"eq(n\\,{n})" for n in frames)
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(video),
                    "-vf", f"select={select}", "-fps_mode", "vfr", "-start_number", "0",
                    "-frames:v", str(len(frames)), "-y", str(output / "decoded_%03d.png")], check=True)
    result = {}
    for i, frame in enumerate(frames):
        path = output / f"decoded_{i:03d}.png"
        if not path.is_file():
            raise ValueError(f"Video did not contain requested frame {frame}")
        dest = output / f"frame_{frame:04d}.png"
        path.replace(dest)
        result[frame] = dest
    return result


def set_style(cell, **updates):
    tokens = cell.get("style", "").strip(";").split(";")
    tokens = [t for t in tokens if t.split("=", 1)[0] not in updates]
    cell.set("style", ";".join(tokens + [f"{key}={value}" for key, value in updates.items()]) + ";")


def add_frame_label(root, cell, frame):
    g = cell.find("mxGeometry")
    height, width = float(g.get("height")), float(g.get("width"))
    if height / width > .65:
        height -= 12
        g.set("height", str(height))
    label = ET.SubElement(root, "mxCell", id=f"real-frame-label-{cell.get('id')}",
                          value=f"f{frame}", parent=cell.get("parent"), vertex="1",
                          style="text;html=1;align=center;verticalAlign=middle;spacing=0;"
                                "fontFamily=Georgia;fontSize=11;fontColor=#34424B;"
                                "strokeColor=none;fillColor=#FFFFFF;")
    ET.SubElement(label, "mxGeometry", x=g.get("x", "0"),
                  y=str(float(g.get("y", 0)) + height),
                  width=g.get("width"), height="12", attrib={"as": "geometry"})


def rewrite_diagram(template, output, assets, evicted):
    tree = ET.parse(template)
    root = tree.find(".//mxGraphModel/root")
    cells = {int(c.get("id").rsplit("-", 1)[-1]): c for c in root
             if c.get("id", "").rsplit("-", 1)[-1].isdigit()}
    image_cells = {key for key, cell in cells.items() if "image=" in cell.get("style", "")}
    if image_cells != set(CELL_FRAMES):
        raise ValueError(f"Template images changed: {image_cells ^ set(CELL_FRAMES)}")
    for key, frame in CELL_FRAMES.items():
        cell = cells[key]
        data = base64.b64encode(assets[frame].read_bytes()).decode()
        set_style(cell, image=f"data:image/png,{data}", imageAspect="1")
        cell.set("data-frame-index", str(frame))
        cell.set("data-source-video", VIDEO)
        if key not in (40, 42, 44, 46, 48, 50):
            add_frame_label(root, cell, frame)

    # Preserve the overview while making the numerical example trace-backed.
    cells[51].set("value", "Generated chunk <i>N</i><sub>t</sub> (30.4--32.9 s)")
    set_style(cells[51], fontSize="16")
    cells[306].set("value", "KEEPSAKE update: Warehouse / 108 candidates to 32 retained")
    set_style(cells[306], fontSize="27")
    cells[143].set("value", "Similar views connect (graph schematic).")
    cells[137].set("value", "108 candidates before eviction")
    cells[215].set("value", "Retain or evict")
    cells[268].set("value", "Six of the 32 retained frames shown.")
    cells[296].set("value", "|<i>M</i><sub>t+1</sub>| = 32")
    cells[308].set("value", "Real MemCam rollout frames, logged membership and eviction scores; graph layout is schematic. Frame 0 is the initial input.")
    for key, label in ((220, "protected"), (231, "retained"), (237, "retained")):
        cells[key].set("value", label)
        cells[key].find("mxGeometry").set("x", "105")
        cells[key].find("mxGeometry").set("width", "182")
        set_style(cells[key], fontSize="19", fontColor="#277A60")
    for key in (218, 219, 229, 230, 235, 236):
        root.remove(cells[key])
    for frame, label, bar in ((41, 243, 242), (42, 250, 249), (987, 257, 256)):
        score = evicted[frame]["eviction_score"]
        cells[label].set("value", f"{score:.4f}")
        set_style(cells[label], fontSize="19")
        cells[bar].find("mxGeometry").set("width", str(116 * score / .05))
    cells[262].set("value", "Evicted frames: logged utility.<br>Lower scores are removed first.")
    set_style(cells[262], fontSize="18")
    for key, color in ((233, "#B87624"), (239, "#2F6FA9"), (246, "#2F6FA9")):
        set_style(cells[key], strokeColor=color)
    for key in (131, 134):
        set_style(cells[key], strokeColor="#B87624")
    # The protected graph node is f0, not the neighboring f68.
    lock = cells[184].find("mxGeometry")
    lock.set("x", "250")
    lock.set("y", "113")
    # The chunk endpoint is also protected by the production update rule.
    for group, x, y, suffix in ((184, 354, 325, "graph-endpoint"), (274, 332, 295, "archive-endpoint")):
        old_id = cells[group].get("id")
        new_id = f"protected-{suffix}"
        for original in (cells[group], *(c for c in cells.values() if c.get("parent") == old_id)):
            clone = copy.deepcopy(original)
            clone.set("id", new_id if original is cells[group] else new_id + "-" + original.get("id"))
            if original is cells[group]:
                clone.find("mxGeometry").set("x", str(x))
                clone.find("mxGeometry").set("y", str(y))
            else:
                clone.set("parent", new_id)
            root.append(clone)
    edge = ET.SubElement(root, "mxCell", id="memory-feedback", edge="1", value="",
        parent=cells[73].get("parent"), source=cells[73].get("id"), target=cells[72].get("id"),
        style="edgeStyle=orthogonalEdgeStyle;rounded=1;endArrow=block;endFill=1;"
              "strokeColor=#607083;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=1;")
    geometry = ET.SubElement(edge, "mxGeometry", relative="1", attrib={"as": "geometry"})
    points = ET.SubElement(geometry, "Array", attrib={"as": "points"})
    ET.SubElement(points, "mxPoint", x="1402", y="337")
    ET.SubElement(points, "mxPoint", x="397", y="337")
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def build(args):
    video, trace = args.videos / VIDEO, args.traces / Path(VIDEO).with_suffix(".jsonl")
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if not events or any((e.get("scene"), e.get("dataset_start_frame"), e.get("duration_sec"),
                         e.get("run_memory_policy"), e.get("run_memory_budget"))
                        != ("Warehouse_0", 3496, 60, "slam_covisibility", 32) for e in events):
        raise ValueError("Trace is not the matching Warehouse KEEPSAKE B32 rollout")
    old, new, kept, evicted = update_bank(events, SECTION)
    validate_roles(old, new, kept, evicted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backup = args.output.parent / "archive/method_previous" / args.template.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(args.template, backup)
    template = backup if args.template.resolve() == args.output.resolve() else args.template
    assets = extract_frames(video, CELL_FRAMES.values(), args.output.parent / "method_frame_assets")
    rewrite_diagram(template, args.output, assets, evicted)
    provenance = dict(source_video=str(video), source_video_sha256=digest(video),
        source_trace=str(trace), source_trace_sha256=digest(trace), section=SECTION,
        scene="Warehouse_0", dataset_start_frame=3496, old_bank=sorted(old),
        new_frames=sorted(new), retained_bank=sorted(kept), evicted_frames=sorted(evicted),
        shown_evictions={str(f): evicted[f] for f in (41, 42, 987)},
        frame_assets={str(n): dict(path=str(p), sha256=digest(p)) for n, p in assets.items()},
        cell_frames=CELL_FRAMES, builder_sha256=digest(Path(__file__)),
        template_sha256=digest(template),
        extraction="Exact zero-based decoded frame indices; full frame, no cropping or retouching.",
        limitations="Graph edges/layout are schematic, except the logged nearest pair f987--f988. "
                    "Retained utility values are not logged and are not invented. "
                    "Only a subset of the 108 candidates / 32 retained items is displayed.")
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Replaced {len(CELL_FRAMES)} thumbnails using {len(assets)} real frames: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "paper/figures/ICLR27 Method Figure.drawio.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/ICLR27 Method Figure.drawio.xml")
    parser.add_argument("--videos", type=Path, default=Path.home() / "Downloads/memcam_editing/KEEPSAKE_B32")
    parser.add_argument("--traces", type=Path, default=Path.home() / "Downloads/context/slam_b32_covisibility/access_traces")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
