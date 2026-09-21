"""Enlarge the original four-stage draw.io figure without replacing its structure."""

import argparse
import base64
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.replace_method_frame_assets import digest, extract_frames, set_style, update_bank

PREFIX = "taUKAWZvAetxOZZTuqRX-"
VIDEO = "seed0_ChemicalPlantEnv_5_3648_60s_custom.mp4"
FRAMES = {
    40: 153, 42: 168, 44: 183, 46: 198, 48: 213, 50: 228,
    92: 0, 95: 41, 98: 125, 101: 145,
    107: 176, 110: 184, 113: 227, 116: 228,
    124: 0, 127: 41, 130: 176, 133: 227,
    163: 146, 165: 41, 167: 0, 169: 183, 171: 125, 173: 145,
    175: 120, 177: 58, 179: 176, 181: 228, 183: 227,
    217: 0, 228: 125, 234: 176, 240: 146, 247: 183, 254: 227,
    272: 0, 280: 41, 283: 125, 286: 145, 289: 176, 292: 228,
}
PANEL_TOP = 385
PANEL_HEIGHT = 625


def prepare(trace):
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    expected = ("ChemicalPlantEnv_5", 3648, 60, "slam_covisibility", 32)
    keys = ("scene", "dataset_start_frame", "duration_sec", "run_memory_policy", "run_memory_budget")
    if not events or any(tuple(e.get(k) for k in keys) != expected for e in events):
        raise ValueError("Wrong rollout identity")
    old, new, kept, evicted = update_bank(events, 2)
    for indices, available in (((92, 95, 98, 101), old), ((107, 110, 113, 116), new),
            ((217, 228, 234, 272, 280, 283, 286, 289, 292), kept), ((240, 247, 254), evicted)):
        if not {FRAMES[n] for n in indices} <= set(available):
            raise ValueError("Illustrated membership does not match the update")
    if not set(FRAMES.values()) <= old | new:
        raise ValueError("A depicted frame was not an update candidate")
    if evicted[227]["eviction_nearest_covisible_frame"] != 228 or 228 not in kept:
        raise ValueError("Closest-substitute edge is not logged")
    return old, new, kept, evicted


def rewrite(template, output, assets, evicted):
    tree = ET.parse(template)
    root = tree.find(".//mxGraphModel/root")
    cells = {c.get("id"): c for c in root}

    def cell(n):
        return cells[PREFIX + str(n)] if isinstance(n, int) else cells[n]

    def geom(n, x=None, y=None, w=None, h=None):
        g = cell(n).find("mxGeometry")
        for key, value in (("x", x), ("y", y), ("width", w), ("height", h)):
            if value is not None:
                g.set(key, str(value))

    def label(n, value=None, size=None):
        if value is not None:
            cell(n).set("value", value)
        if size is not None:
            set_style(cell(n), fontSize=str(size))

    def edge(n, points):
        g = cell(n).find("mxGeometry")
        for child in list(g):
            g.remove(child)
        ET.SubElement(g, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), attrib={"as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), attrib={"as": "targetPoint"})
        if len(points) > 2:
            a = ET.SubElement(g, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(a, "mxPoint", x=str(x), y=str(y))

    # Frame IDs belong in provenance/caption, not tiny labels on every thumbnail.
    for c in list(root):
        if c.get("id", "").startswith("real-frame-label-"):
            root.remove(c)
        elif c.get("value") and c.get("vertex"):
            match = re.search(r"fontSize=([\d.]+)", c.get("style", ""))
            size = float(match[1]) if match else 20
            set_style(c, fontSize=str(max(24, round(size * 1.2))))
    actual_images = {c.get("id") for c in root if "image=" in c.get("style", "")}
    if actual_images != {PREFIX + str(n) for n in FRAMES}:
        raise ValueError("Original four-stage template has changed")
    for n, frame in FRAMES.items():
        c = cell(n)
        data = base64.b64encode(assets[frame].read_bytes()).decode()
        set_style(c, image=f"data:image/png,{data}", imageAspect="1")
        c.set("data-frame-index", str(frame))
        c.set("data-source-video", VIDEO)

    # Preserve the original generation loop, arrows, cameras and image stack.
    label(303, "Long-horizon generation loop (KEEPSAKE)", 38)
    label(304, "<b>Plug-and-play:</b> generator and retriever unchanged.", 27)
    geom(304, h=53, y=6)
    geom(3, y=6, h=53)
    geom(8, w=245, h=150)
    geom(9, w=245, h=150)
    geom(10, x=3, w=239, h=65, y=8)
    label(10, size=24)
    label(27, size=26)
    label(55, size=26)
    label(35, "Generator<br>(unchanged)", 28)
    geom(35, y=3, h=57)
    geom(36, y=67, h=62)
    geom(37, y=71, h=53)
    label(37, size=26)
    label(51, "Generated chunk <i>N</i><sub>t</sub>", 27)
    geom(51, h=46, y=98)
    label(52, "<b>Update memory<br>(KEEPSAKE)</b>", 26)
    geom(72, w=225, x=284)
    geom(73, w=225, x=1289)
    root.remove(cell(305))
    if "memory-feedback" in cells:
        root.remove(cell("memory-feedback"))
    geom(4, y=324, h=704)
    label(306, "KEEPSAKE: geometry-aware memory update", 38)
    geom(306, y=332, h=47, w=1720)
    for c in root:
        if c.get("parent") == PREFIX + "7":
            g = c.find("mxGeometry")
            if g is not None and c.get("vertex"):
                g.set("y", str(float(g.get("y", 0)) - 25))
            elif g is not None and c.get("edge"):
                for point in g.iter("mxPoint"):
                    if point.get("y") is not None:
                        point.set("y", str(float(point.get("y")) - 25))
    edge(5, [(1330, 184), (1330, 307), (270, 324)])
    edge(6, [(1474, 184), (1474, 307), (1667, 324)])
    for panel, background, band, number, title, subtitle, width in (
            (83, 84, 85, 86, 87, 88, 470), (138, 139, 140, 141, 142, 143, 460),
            (209, 210, 211, 212, 213, 214, 372), (263, 264, 265, 266, 267, 268, 416)):
        geom(panel, y=PANEL_TOP, h=PANEL_HEIGHT)
        geom(background, h=PANEL_HEIGHT)
        geom(band, h=130)
        geom(number, y=17, w=52, h=52)
        label(number, size=34)
        geom(title, x=77, y=10, w=width - 89, h=67)
        label(title, size=30)
        geom(subtitle, x=17, y=79, w=width - 34, h=47)
        label(subtitle, size=24)
    label(88, "Merge existing memory<br>and new observations.")
    label(143, "Connect views using<br>pose and appearance.")
    label(214, "Compute utility;<br>evict low-utility views.")
    label(268, "Six of 32 retained<br>frames shown.")

    # Restore horizontal filmstrips so the four-stage figure stays landscape.
    for box, title, groups, images, borders, top in (
            (89, 90, (91, 94, 97, 100), (92, 95, 98, 101), (93, 96, 99, 102), 144),
            (104, 105, (106, 109, 112, 115), (107, 110, 113, 116), (108, 111, 114, 117), 285),
            (121, 122, (123, 126, 129, 132), (124, 127, 130, 133), (125, 128, 131, 134), 456)):
        geom(box, y=top, h=119)
        geom(title, y=top + 5, h=36)
        label(title, size=28)
        for i, (group, image, border) in enumerate(zip(groups, images, borders)):
            geom(group, x=18 + i * 110, y=top + 49, w=100, h=55)
            geom(image, x=0, y=0, w=100, h=55)
            geom(border, x=-1, y=-1, w=102, h=57)
    for n in (103, 118, 135):
        root.remove(cell(n))
    label(122, "<i>C</i><sub>t</sub> = <i>M</i><sub>t</sub> &cup; <i>N</i><sub>t</sub>", 32)
    geom(122, x=26, w=414)
    edge(119, [(455, 217), (462, 217), (462, 434), (236, 434)])
    edge(120, [(46, 410), (46, 434), (236, 434), (236, 452)])
    label(137, "108 candidates", 32)
    geom(137, x=30, y=581, w=410, h=39)
    set_style(cell(137), align="center")

    # Keep all eleven original graph nodes and all original connections.
    graph_positions = ((162, 12, 249), (164, 99, 196), (166, 201, 154),
                       (168, 71, 300), (170, 177, 267), (172, 17, 377),
                       (174, 135, 377), (176, 343, 156), (178, 282, 263),
                       (180, 307, 366), (182, 378, 333))
    for node, x, y in graph_positions:
        geom(node, x=x, y=y, w=72, h=72)
        geom(node + 1, x=x + 4, y=y + 18.4, w=64, h=35.2)
    set_style(cell(168), strokeColor="#B87624")
    geom(146, x=9, y=205, w=280, h=250)
    geom(144, x=18, y=140, w=177, h=52)
    geom(145, x=22, y=140, w=169, h=52)
    label(145, "neighbor<br>support", 24)
    edge(190, [(58, 194), (68, 219), (77, 250)])
    geom(184, x=257, y=148)
    geom("protected-graph-endpoint", x=362, y=361)
    geom(189, x=361, y=230, w=30, h=40)
    geom(191, x=179, y=465, w=264, h=38)
    geom(192, x=183, y=465, w=256, h=38)
    label(192, size=26)
    edge(193, [(427, 465), (444, 441), (425, 406)])
    geom(194, y=514, h=104)
    edge(195, [(33, 532), (76, 532)])
    geom(196, x=87, y=516, w=185, h=32)
    label(196, size=24)
    edge(197, [(33, 565), (76, 565)])
    geom(198, x=87, y=549, w=260, h=32)
    label(198, size=24)
    geom(199, x=284, y=521)
    geom(204, x=307, y=516, w=130, h=32)
    label(204, size=24)
    geom(205, x=48, y=592)
    geom(206, x=76, y=584, w=133, h=32)
    geom(207, x=263, y=592)
    geom(208, x=291, y=584, w=133, h=32)

    # Retain the score list, utility bars, checks and crosses from the original.
    geom(215, x=20, y=138, w=332, h=34)
    label(215, size=28)
    rows = ((216, 217, 220, 221, 178), (227, 228, 231, 232, 245),
            (233, 234, 237, 238, 312), (239, 240, 243, 244, 431),
            (246, 247, 250, 251, 493), (253, 254, 257, 258, 555))
    for index, (circle, image, text, mark, y) in enumerate(rows):
        geom(circle, x=23, y=y, w=60, h=60)
        geom(image, x=26, y=y + 15.15, w=54, h=29.7)
        if index < 3:
            geom(text, x=106, y=y + 11, w=192, h=38)
            label(text, size=28)
            edge(mark, [(318, y + 28), (328, y + 38), (346, y + 17)])
        else:
            geom(text, x=212, y=y + 11, w=99, h=38)
            label(text, size=26)
            edge(mark, [(324, y + 20), (345, y + 41)])
            edge(mark + 1, [(324, y + 41), (345, y + 20)])
    set_style(cell(246), strokeColor="#B87624")
    for frame, bar, background, text, y in ((146, 242, 241, 243, 451),
            (183, 249, 248, 250, 513), (227, 256, 255, 257, 575)):
        score = evicted[frame]["eviction_score"]
        geom(background, x=104, y=y, w=92, h=20)
        geom(bar, x=104, y=y, w=92 * score / .05, h=20)
        label(text, f"{score:.4f}")
    geom(222, x=72, y=174)
    geom(260, x=47, y=386, w=25, h=31)
    geom(261, x=96, y=382, w=259, h=42)
    geom(262, x=105, y=385, w=241, h=36)
    label(262, "Evict lowest utility", 24)

    # The same six retained items, enlarged to a two-column, three-row bank.
    geom(269, y=145, h=391)
    geom(270, y=151, h=62)
    label(270, "Updated memory<br><i>M</i><sub>t+1</sub>", 28)
    for i, (group, image, border) in enumerate(((271, 272, 273), (279, 280, 281),
            (282, 283, 284), (285, 286, 287), (288, 289, 290), (291, 292, 293))):
        geom(group, x=36 + i % 2 * 183, y=219 + i // 2 * 107, w=156, h=85.8)
        geom(image, x=0, y=0, w=156, h=85.8)
        geom(border, x=-1, y=-1, w=158, h=87.8)
    geom(274, x=168, y=212)
    geom("protected-archive-endpoint", x=350, y=426)
    root.remove(cell(294))
    geom(296, x=36, y=546, w=344, h=42)
    label(296, "|<i>M</i><sub>t+1</sub>| = 32", 36)
    geom(297, y=593, h=28)
    geom(298, y=593, h=28)
    label(298, "Bounded archive", 26)
    for n, start, end in ((299, 505, 520), (300, 984, 998), (301, 1374, 1388)):
        edge(n, [(start, 706), (end, 706)])
    geom(307, x=35, y=1033, w=1785, h=35)
    label(307, "Preserve views with few geometric and visual substitutes.", 32)
    root.remove(cell(308))
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def build(args):
    trace = args.traces / Path(VIDEO).with_suffix(".jsonl")
    video = args.videos / VIDEO
    old, new, kept, evicted = prepare(trace)
    assets = extract_frames(video, FRAMES.values(), args.output.parent / "method_daylight_frames")
    # Keep the rejected redesign available, without making it the source of this edit.
    rejected = args.output.parent / "method_previous/rejected_three_stage.drawio.xml"
    if args.output.exists() and not rejected.exists():
        shutil.copy2(args.output, rejected)
    rewrite(args.template, args.output, assets, evicted)
    provenance = dict(source_video=str(video), source_video_sha256=digest(video),
        source_trace=str(trace), source_trace_sha256=digest(trace), section_idx=2,
        old_bank=sorted(old), new_frames=sorted(new), retained_bank=sorted(kept),
        evicted_frames=sorted(evicted), cell_frames=FRAMES,
        shown_evictions={str(n): evicted[n] for n in (146, 183, 227)},
        original_layout=str(args.template), original_layout_sha256=digest(args.template),
        builder_sha256=digest(Path(__file__)),
        changes="Original four stages, eleven graph nodes, score list and generation flow preserved. "
                "Return arrow and its next-chunk label removed. Original landscape proportions; "
                "larger text and thumbnails; daylight frames.",
        limitations="Graph layout/edges schematic except the logged nearest pair f227--f228. "
                    "Only a subset of candidates and retained frames is displayed.",
        extraction="Exact decoded zero-based frame indices; full frames, no image enhancement.")
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    caption = r"""\caption{\textbf{KEEPSAKE maintains a bounded archive inside the generation loop.}
The original retriever and generator are unchanged. Existing memory and new
observations are merged, connected using camera pose and DINO appearance, scored
for redundancy, and pruned to the fixed budget. The example uses real frames
from one ChemicalPlant update with 108 candidates, 76 evictions and 32 retained
items. Eviction scores and membership are logged; graph layout is schematic.
The closest-substitute pair is frame 227 and retained frame 228. The initial
input and current chunk endpoint are protected. Only selected items are shown.}
"""
    (args.output.parent / "ICLR27_Method_caption.tex").write_text(caption)
    print(f"Restored and enlarged: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path,
                        default=ROOT / "paper/figures/method_previous/warehouse_real_frames.drawio.xml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/ICLR27 Method Figure.drawio.xml")
    parser.add_argument("--videos", type=Path,
                        default=Path.home() / "Downloads/memcam_editing/KEEPSAKE_B32")
    parser.add_argument("--traces", type=Path,
                        default=Path.home() / "Downloads/context/slam_b32_covisibility/access_traces")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
