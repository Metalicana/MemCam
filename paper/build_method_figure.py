"""Build the trace-grounded method figure with an expanded graph/scoring panel."""

import argparse
import base64
import json
import math
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
PANEL_HEIGHT = 585
PANELS = {83: (34, 230), 138: (282, 1268), 263: (1570, 236)}
RED, BLUE, GREEN = "#B24C55", "#2F6FA9", "#277A60"


def scoring_example(evicted):
    record = evicted[227]
    count = int(record["eviction_covisible_observers"])
    maximum = float(record["eviction_max_covisibility"])
    score = float(record["eviction_score"])
    if count < 1 or not 0.65 <= maximum <= 1 or record["eviction_nearest_covisible_frame"] != 228:
        raise ValueError("Invalid logged graph example")
    expected = 1 - min(count / 3, 1) + 0.5 / (count + 1) + 0.25 * (1 - maximum)
    if not math.isclose(expected, score, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Logged utility disagrees with the illustrated scoring rule")
    return {"frame": 227, "closest_frame": 228, "neighbors": count,
            "max_affinity": maximum, "utility": score}


def draw_update(root, assets, example):
    """Native draw.io shapes; only the highlighted pair/statistics are measured."""
    layer = PREFIX + "82"
    graph = PREFIX + "138"
    nodes = {}
    text_color = "#151D2A"

    def vertex(name, value, x, y, w, h, parent=layer, *, fill="none",
               stroke="none", size=26, bold=False, shape="rectangle", color=text_color):
        c = ET.SubElement(root, "mxCell", id=name, parent=parent, value=value, vertex="1")
        set_style(c, shape=shape, html="1", whiteSpace="wrap", fontFamily="Georgia",
                  fontSize=str(size), fontStyle="1" if bold else "0", fontColor=color,
                  fillColor=fill, strokeColor=stroke, strokeWidth="1.5",
                  align="center", verticalAlign="middle", spacing="0", rounded="0")
        ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h),
                      attrib={"as": "geometry"})
        nodes[name] = c
        return name

    def line(name, points, parent=layer, *, color="#607083", width=2,
             arrow=True, dashed=False):
        c = ET.SubElement(root, "mxCell", id=name, parent=parent, value="", edge="1")
        set_style(c, edgeStyle="none", rounded="0", html="1", strokeColor=color,
                  strokeWidth=str(width), endArrow="block" if arrow else "none",
                  startArrow="none", endSize="7", dashed="1" if dashed else "0")
        g = ET.SubElement(c, "mxGeometry", relative="1", attrib={"as": "geometry"})
        ET.SubElement(g, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), attrib={"as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), attrib={"as": "targetPoint"})
        if len(points) > 2:
            bends = ET.SubElement(g, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(bends, "mxPoint", x=str(x), y=str(y))
        nodes[name] = c
        return c

    def photo(name, frame, x, y, w, parent, role, stroke="#607083"):
        h = w * 352 / 640
        vertex(name, "", x, y, w, h, parent, stroke=stroke)
        c = nodes[name]
        set_style(c, shape="image", imageAspect="1",
                  image="data:image/png," + base64.b64encode(assets[frame].read_bytes()).decode())
        c.set("data-frame-index", str(frame))
        c.set("data-source-video", VIDEO)
        c.set("data-role", role)

    def decision(name, x, y, accepted):
        color = GREEN if accepted else RED
        vertex(name, "", x, y, 30, 30, graph, fill="#FFFFFF", stroke=color, shape="ellipse")
        nodes[name].set("data-role", "link-decision")
        if accepted:
            line(name + "-tick", [(x + 7, y + 15), (x + 13, y + 21), (x + 23, y + 8)],
                 graph, color=color, width=3, arrow=False)
        else:
            for suffix, points in (("a", [(x + 9, y + 9), (x + 21, y + 21)]),
                                   ("b", [(x + 9, y + 21), (x + 21, y + 9)])):
                line(name + "-cross-" + suffix, points, graph, color=color, width=3, arrow=False)

    def panel(n, title, fill, stroke):
        x, w = PANELS[n]
        name = PREFIX + str(n)
        vertex(name, "", x, PANEL_TOP, w, PANEL_HEIGHT, fill="#FFFFFF", stroke=stroke)
        vertex(name + "-header", "", 1, 1, w - 2, 89, name, fill=fill)
        vertex(name + "-title", title, 12, 8, w - 24, 74, name, bold=True,
               size=29 if n == 138 else 27)
        return name

    bank = panel(83, "Candidate<br>bank", "#FBEFF0", RED)
    panel(138, "Pose&ndash;appearance graph and budgeted update", "#EDF5FC", BLUE)
    archive = panel(263, "Retained<br>archive", "#EDF7F2", GREEN)

    for title, frames, y, color, role in (("Existing <i>M</i><sub>t</sub>", (0, 41, 125, 145), 108, RED, "existing"),
            ("New <i>N</i><sub>t</sub>", (176, 184, 227, 228), 288, RED, "new")):
        vertex(role + "-title", title, 12, y, 206, 38, bank, size=27, bold=True)
        for k, frame in enumerate(frames):
            photo(role + f"-{frame}", frame, 13 + 106 * (k % 2), y + 48 + 62 * (k // 2),
                  98, bank, role, color)
    line("candidate-merge", [(115, 456), (115, 479)], bank)
    vertex("candidate-union", "<i>C</i><sub>t</sub> = <i>M</i><sub>t</sub> &cup; <i>N</i><sub>t</sub>",
           8, 484, 214, 42, bank, size=28)
    vertex("candidate-count", "108 candidates", 8, 538, 214, 36, bank, size=27)

    # Pairwise inputs split into pose and appearance branches, then recombine.
    vertex("pair-title", "Pairwise similarity", 16, 104, 318, 38, graph, bold=True, size=27)
    for frame, x, symbol in ((227, 35, "i"), (228, 203, "j")):
        photo("pair-" + symbol, frame, x, 175, 112, graph, "affinity_pair")
        vertex("pair-" + symbol + "-label", f"<i>{symbol}</i>", x, 141, 112, 30, graph, size=26)
    line("pair-join-i", [(91, 237), (91, 249), (259, 249), (259, 237)], graph, arrow=False)
    line("pair-pose", [(91, 249), (91, 268)], graph)
    line("pair-appearance", [(259, 249), (259, 268)], graph)
    vertex("pose-affinity", "<b>Pose</b><br><i>P</i><sub>ij</sub> = e<sup>&minus;d<sub>p</sub>(i,j)</sup>",
           16, 268, 150, 85, graph, fill="#EDF5FC", stroke="#8EB4D8", size=24)
    vertex("appearance-affinity", "<b>Appearance</b><br><i>A</i><sub>ij</sub> = [z<sub>i</sub><sup>T</sup>z<sub>j</sub>]<sub>+</sub>",
           186, 268, 150, 85, graph, fill="#EDF5FC", stroke="#8EB4D8", size=22)
    line("affinity-merge", [(91, 353), (91, 371), (261, 371), (261, 353)], graph, arrow=False)
    line("affinity-merge-output", [(176, 371), (176, 390)], graph)
    vertex("combined-affinity", "<i>K</i><sub>ij</sub> = &alpha;<i>P</i><sub>ij</sub> + (1 &minus; &alpha;)<i>A</i><sub>ij</sub>",
           16, 390, 320, 58, graph, fill="#EDF5FC", stroke=BLUE, size=25)

    vertex("edge-rule-title", "Add undirected links", 368, 104, 426, 38, graph, bold=True, size=27)
    vertex("edge-rule", "<i>i</i> &ne; <i>j</i>, &nbsp;<i>K</i><sub>ij</sub> &ge; &tau;",
           368, 146, 426, 42, graph, size=26)
    line("affinity-to-graph", [(336, 419), (383, 419), (383, 345), (421, 345)], graph)

    # Anonymous nodes show thresholded links schematically, without inventing frame IDs.
    for name, points in (("schematic-link-1", [(467, 309), (445, 245)]),
                         ("schematic-link-3", [(461, 381), (464, 409)])):
        c = line(name, points, graph, arrow=False, color="#8FA6AD", width=2.3)
        c.set("data-link-origin", "schematic")
    c = line("logged-closest-link", [(496, 318), (668, 269)], graph, arrow=False, color=GREEN, width=3.5)
    c.set("data-link-origin", "logged nearest affinity")
    c.set("data-frame-pair", "227,228")
    c.set("data-affinity", str(example["max_affinity"]))
    c.set("source", "graph-i")
    c.set("target", "graph-j")
    set_style(c, exitX="0.85", exitY="0.2", entryX="0", entryY="0.6",
              exitPerimeter="1", entryPerimeter="1")
    for name, x, y in (("neighbor-1", 419, 206), ("neighbor-3", 440, 406)):
        vertex(name, "<i>j</i>", x, y, 44, 44, graph, fill="#F6F9FA", stroke="#8FA6AD", shape="ellipse", size=24)
    vertex("graph-i", "", 416, 298, 92, 92, graph, fill="#EDF5FC", stroke=BLUE, shape="ellipse")
    photo("graph-i-image", 227, 422, 321, 80, graph, "scored_node", BLUE)
    vertex("graph-i-label", "<i>i</i> = 227", 346, 266, 92, 28, graph, size=24)
    vertex("graph-j", "", 666, 214, 92, 92, graph, fill="#EDF7F2", stroke=GREEN, shape="ellipse")
    photo("graph-j-image", 228, 672, 237, 80, graph, "closest_node", GREEN)
    vertex("graph-j-label", "<i>j</i>* = 228", 647, 184, 134, 30, graph, size=25)
    vertex("high-affinity-label", "High score", 526, 224, 137, 30, graph, color=GREEN, size=23)
    vertex("graph-edge-value", f"{example['max_affinity']:.4f}", 542, 263, 117, 31, graph,
           fill="#FFFFFF", color=GREEN, size=25)
    vertex("connect-decision", "Connect", 542, 312, 117, 26, graph, color=GREEN, size=23, bold=True)
    decision("matched-decision", 744, 213, True)

    # A real candidate thumbnail illustrates the rejection rule, not a measured low affinity.
    vertex("unlinked-node", "", 666, 356, 92, 92, graph,
           fill="#FBEFF0", stroke=RED, shape="ellipse")
    nodes["unlinked-node"].set("data-link-origin", "schematic below-threshold example")
    photo("graph-k-image", 41, 672, 379, 80, graph, "illustrative_mismatch", RED)
    vertex("graph-k-label", "<i>k</i>", 687, 324, 52, 30, graph, size=25)
    vertex("low-affinity-label", "Low score", 526, 348, 137, 28, graph, color=RED, size=23)
    vertex("no-edge-rule", "<i>K</i><sub>ik</sub> &lt; &tau;", 528, 378, 128, 30, graph,
           color=RED, size=24)
    vertex("no-link-decision", "No link", 528, 416, 128, 28, graph, color=RED, size=23, bold=True)
    decision("mismatched-decision", 744, 354, False)

    vertex("statistics-title", "Node statistics", 844, 104, 392, 38, graph, bold=True, size=27)
    vertex("neighbor-count", "<b>Neighbor count</b><br><i>c</i><sub>i</sub> = |{j &ne; i : K<sub>ij</sub> &ge; &tau;}|",
           844, 159, 392, 93, graph, fill="#EDF5FC", stroke="#8EB4D8", size=25)
    vertex("neighbor-value", f"<i>c</i><sub>227</sub> = {example['neighbors']}", 844, 252, 392, 39, graph, size=27, color=BLUE)
    vertex("strongest-affinity", "<b>Closest substitute</b><br><i>k</i><sub>i</sub><sup>max</sup> = max<sub>j &ne; i</sub> K<sub>ij</sub>",
           844, 311, 392, 93, graph, fill="#EDF5FC", stroke="#8EB4D8", size=25)
    vertex("strongest-value", f"<i>k</i><sub>227</sub><sup>max</sup> = {example['max_affinity']:.4f}",
           844, 404, 392, 42, graph, size=27, color=BLUE)
    line("graph-to-statistics", [(796, 301), (836, 301)], graph)
    line("statistics-to-priority", [(1040, 448), (1040, 462), (482, 462), (482, 473)], graph)
    vertex("priority-band", "", 16, 477, 932, 93, graph, fill="#F4F8FD", stroke="#8EB4D8")
    vertex("priority-title", "Retention priority", 27, 479, 910, 32, graph, bold=True, size=27)
    vertex("priority-formula",
           "<i>u</i><sub>i</sub> = 1 &minus; min(c<sub>i</sub>/3, 1) + 0.5/(c<sub>i</sub> + 1) + 0.25(1 &minus; k<sub>i</sub><sup>max</sup>)",
           27, 521, 695, 37, graph, size=24)
    vertex("priority-value", f"<i>u</i><sub>227</sub> = {example['utility']:.4f}",
           732, 521, 206, 37, graph, bold=True, size=25, color=BLUE)

    # Eviction is the final operation inside the graph/scoring update, not a separate stage.
    eviction = PREFIX + "209"
    vertex(eviction, "", 1000, 477, 252, 93, graph, fill="#E4EFFB", stroke=BLUE)
    vertex("evict-title", "Evict lowest<br>priority", 6, 4, 240, 53, eviction, bold=True, size=26)
    vertex("evict-count", "108 &rarr; 32", 6, 61, 240, 28, eviction, size=26)
    line("priority-to-eviction", [(951, 523), (992, 523)], graph, color=BLUE, width=2.5)
    for k, frame in enumerate((0, 125, 228)):
        photo("retained-" + str(frame), frame, 21, 166 + 123 * k, 194, archive, "retained", GREEN)
    vertex("retained-title", "<i>M</i><sub>t+1</sub>", 14, 99, 208, 36, archive, bold=True, size=30)
    vertex("retained-count", "|<i>M</i><sub>t+1</sub>| = 32", 6, 538, 224, 38, archive, size=29)
    line("candidates-to-graph", [(264, PANEL_TOP + 318), (280, PANEL_TOP + 318)], width=2.5)
    line("eviction-to-archive", [(1536, PANEL_TOP + 523), (1568, PANEL_TOP + 523)], width=2.5, color=GREEN)


def prepare(trace):
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    expected = ("ChemicalPlantEnv_5", 3648, 60, "slam_covisibility", 32)
    keys = ("scene", "dataset_start_frame", "duration_sec", "run_memory_policy", "run_memory_budget")
    if not events or any(tuple(e.get(k) for k in keys) != expected for e in events):
        raise ValueError("Wrong rollout identity")
    for event in events:
        for key, value in (("keepsake_geometry_weight", .65), ("keepsake_appearance_weight", .35)):
            if key in event and not math.isclose(float(event[key]), value, abs_tol=1e-12):
                raise ValueError("Trace affinity weights differ from the method figure")
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

    def connect(n, source, target, start, end):
        c = cell(n)
        c.set("source", cell(source).get("id"))
        c.set("target", cell(target).get("id"))
        set_style(c, exitX=str(start[0]), exitY=str(start[1]),
                  entryX=str(end[0]), entryY=str(end[1]),
                  exitDx="0", exitDy="0", entryDx="0", entryDy="0",
                  exitPerimeter="1", entryPerimeter="1")
        g = c.find("mxGeometry")
        for child in list(g):
            g.remove(child)

    def conditioning_input(name, title, x, width, entry_x):
        c = ET.SubElement(root, "mxCell", id=name, vertex="1",
                          parent=PREFIX + "7", value=title)
        c.set("style", cell(72).get("style"))
        set_style(c, fontSize="24", fillColor="#F4F7FB", strokeColor="#607083")
        ET.SubElement(c, "mxGeometry", x=str(x), y="247", width=str(width),
                      height="45", attrib={"as": "geometry"})
        cells[name] = c
        arrow = ET.SubElement(root, "mxCell", id=name + "-arrow", edge="1",
                              parent=PREFIX + "7", style=cell(79).get("style"))
        ET.SubElement(arrow, "mxGeometry", relative="1", attrib={"as": "geometry"})
        cells[name + "-arrow"] = arrow
        connect(name + "-arrow", name, 28, (0.5, 0), (entry_x, 1))

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
    geom(303, x=34, w=1772)
    set_style(cell(303), align="center")
    geom(8, w=245, h=150)
    geom(9, w=245, h=150)
    geom(10, x=3, w=239, h=65, y=8)
    label(10, size=24)
    label(27, size=26)
    label(55, size=26)
    label(35, "Generator", 28)
    geom(35, y=8, h=39)
    geom(36, y=58, h=70)
    geom(37, y=64, h=58)
    label(37, size=26)
    label(51, "Generated chunk <i>N</i><sub>t</sub>", 27)
    geom(51, h=46, y=98)
    label(52, "<b>Update memory<br>(KEEPSAKE)</b>", 26)
    root.remove(cell(305))
    if "memory-feedback" in cells:
        root.remove(cell("memory-feedback"))
    geom(4, y=324, h=PANEL_TOP + PANEL_HEIGHT + 18 - 324)
    set_style(cell(4), strokeColor="#B24C55", strokeWidth="2.3")
    label(306, "KEEPSAKE: geometry-aware memory update", 38)
    geom(306, x=34, y=332, h=47, w=1772)
    set_style(cell(306), align="center")
    # Equal 40-unit gaps and shared centerline keep the retrieval arrows symmetric.
    for n, x, y, width, height in ((8, 30, 72, 245, 150),
            (27, 315, 95, 208, 104), (28, 563, 76.5, 332, 141),
            (38, 935, 76.5, 294, 141), (52, 1269, 100.5, 232, 93),
            (53, 1541, 72, 243, 150)):
        geom(n, x=x, y=y, w=width, h=height)
    for n, source, target in ((74, 8, 27), (75, 27, 28), (76, 28, 38),
                               (77, 38, 52), (78, 52, 53)):
        connect(n, source, target, (1, 0.5), (0, 0.5))
    geom(72, x=306.5, y=247, w=225)
    geom(73, x=1272.5, y=247, w=225)
    set_style(cell(73), fillColor="#EDF7F2", strokeColor=GREEN)
    connect(79, 72, 27, (0.5, 0), (0.5, 1))
    connect(80, 52, 73, (0.5, 1), (0.5, 0))
    conditioning_input("caption-input", "Initial caption", 563, 204, 102 / 332)
    conditioning_input("noise-input", "Noise", 789, 106, 279 / 332)

    # Expansion guides, not data-flow arrows; route outside the updated-memory box.
    edge(5, [(1269, 193.5), (1247, 218), (1247, 301), (270, 324)])
    edge(6, [(1501, 193.5), (1523, 218), (1523, 301), (1667, 324)])
    for n in (5, 6):
        set_style(cell(n), strokeColor="#B24C55", strokeWidth="3", dashed="1",
                  dashPattern="4 3", opacity="100", endArrow="none", startArrow="none")
    obsolete = {PREFIX + str(n) for n in (*range(83, 302), 3, 304, 307, 308)}
    obsolete.update(("protected-graph-endpoint", "protected-archive-endpoint"))
    while True:
        descendants = {c.get("id") for c in root if c.get("parent") in obsolete}
        if descendants <= obsolete:
            break
        obsolete.update(descendants)
    for c in list(root):
        if c.get("id") in obsolete:
            root.remove(c)
    draw_update(root, assets, scoring_example(evicted))
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def build(args):
    trace = args.traces / Path(VIDEO).with_suffix(".jsonl")
    video = args.videos / VIDEO
    old, new, kept, evicted = prepare(trace)
    assets = extract_frames(video, FRAMES.values(), args.output.parent / "method_daylight_frames")
    # Keep the rejected redesign available, without making it the source of this edit.
    rejected = args.output.parent / "archive/method_previous/rejected_three_stage.drawio.xml"
    if args.output.exists() and not rejected.exists():
        rejected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, rejected)
    rewrite(args.template, args.output, assets, evicted)
    example = scoring_example(evicted)
    rendered_cells = list(ET.parse(args.output).iter("mxCell"))
    provenance = dict(source_video=str(video), source_video_sha256=digest(video),
        source_trace=str(trace), source_trace_sha256=digest(trace), section_idx=2,
        old_bank=sorted(old), new_frames=sorted(new), retained_bank=sorted(kept),
        evicted_frames=sorted(evicted),
        cell_frames={c.get("id"): int(c.get("data-frame-index")) for c in rendered_cells
                     if c.get("data-frame-index") is not None},
        scoring_example=example, shown_evictions={"227": evicted[227]},
        protection={"permanent": [0], "temporary_endpoint": 228,
                    "counts_toward_budget": True, "participates_in_affinity": True,
                    "meaning": "Excluded from eviction, not a separate category from retained."},
        original_layout=str(args.template), original_layout_sha256=digest(args.template),
        builder_sha256=digest(Path(__file__)),
        changes="Generation loop preserved. Red candidate bank, expanded blue graph/scoring/update "
                "panel with integrated budgeted eviction, green retained archive. Centered titles "
                "without step numbers. Symbolic alpha and one-minus-alpha affinity weights; "
                "pose/appearance labels and no numeric covisibility threshold. "
                "Protection annotations omitted from the visual explanation. "
                "Green tick and connect label on the logged high-affinity pair; a real frame-41 "
                "thumbnail with a red cross illustrates the below-threshold no-link case.",
        rejected_link_example=dict(source_frame=227, illustrated_candidate_frame=41,
            origin="schematic decision with a real candidate thumbnail",
            affinity=None, threshold_relation="K_ik < tau (illustrative, not measured)",
            limitation="The trace does not log this pair's affinity; no numeric low score is asserted."),
        scoring_sources=["diffsynth/pipelines/memory_policies.py:_slam_covisibility_affinity",
                         "diffsynth/pipelines/memory_policies.py:compute_slam_covisibility_scores",
                         "diffsynth/pipelines/memory_policies.py:FrameMemoryBuffer.evict_to_budget"],
        conditioning_sources=["inference_memcam.py:run_generation",
                              "diffsynth/pipelines/wan_video_memcam.py:WanVideoMemCamPipeline.__call__"],
        limitations="Anonymous graph nodes/links illustrate the threshold rule, not measured IDs. "
                    "The highlighted 227--228 pair and the count/maximum/utility are logged. "
                    "Neighbor count covers all 108 candidates, not only the drawn subset. "
                    "The frame-41 red-cross example illustrates the rejection rule, not a measured nonedge.",
        extraction="Exact decoded zero-based frame indices; full frames, no image enhancement.")
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    caption = r"""\caption{\textbf{KEEPSAKE maintains a bounded archive inside the generation loop.}
The original retriever and generator are unchanged. Dark dashed guides expand
the memory-update block; they are not data-flow arrows. Merge existing and
new observations. For every distinct pair, combine pose proximity with
nonnegative cosine similarity of normalized DINO descriptors, where
$K_{ij}=\alpha P_{ij}+(1-\alpha)A_{ij}$ and $[s]_+=\max(s,0)$.
Pose distance combines median-normalized translation
and rotation angle with weight two.
Add an undirected link when the combined affinity meets the linking criterion:
the green check marks a connected pair; the red cross illustrates a below-threshold
pair with no link. These decisions use pairwise similarity, not retention priority. Count these
neighbors and find the strongest affinity over all other candidates to compute
the displayed retention priority. Within the same update, evict the lowest-scored
eligible items using fixed scores, then keep the bounded archive.
"""
    caption += (f"The ChemicalPlant example has {len(old | new)} candidates and retains {len(kept)}. "
                f"Frame {example['frame']} has {example['neighbors']} thresholded neighbors; "
                f"its strongest match is frame {example['closest_frame']} with affinity "
                f"{example['max_affinity']:.4f}, yielding priority {example['utility']:.4f}. "
                "This pair, its statistics, and archive membership are logged. The red-cross example "
                "uses the real frame-41 thumbnail to illustrate the rejection rule; its low pairwise "
                "score is symbolic, not measured. Anonymous nodes "
                "and links illustrate the rule schematically; the count is over the full "
                "candidate bank, not just the displayed subset.}\n")
    (args.output.parent / "ICLR27_Method_caption.tex").write_text(caption)
    print(f"Updated method figure: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path,
                        default=ROOT / "paper/figures/archive/method_previous/warehouse_real_frames.drawio.xml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/ICLR27 Method Figure.drawio.xml")
    parser.add_argument("--videos", type=Path,
                        default=Path.home() / "Downloads/memcam_editing/KEEPSAKE_B32")
    parser.add_argument("--traces", type=Path,
                        default=Path.home() / "Downloads/context/slam_b32_covisibility/access_traces")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
