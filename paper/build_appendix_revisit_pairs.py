"""Pair existing audited revisit triplets in the user's compact appendix layout."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build_motivation_teaser import Diagram
from paper.build_revisit_drawio import METHODS
from paper.make_180s_gt_comparisons import digest

CATALOG = ROOT / "paper/figures/revisit_appendix/examples"
OUT = ROOT / "paper/figures/appendix_revisit_pairs"

PAIRS = [
    ("memcam_03_ClothingStore_6", "worldmem_trajectory_05"),
    ("memcam_07_DragonRise_1", "worldmem_trajectory_06"),
    ("memcam_04_ContainerYard_3", "memcam_10_Warehouse_0"),
]


def load_case(stem):
    path = CATALOG / f"{stem}.provenance.json"
    record = json.loads(path.read_text())
    assert record["stem"] == stem
    assert record["frames"] == sorted(set(record["frames"])) and len(record["frames"]) == 3
    assert record["timeline_fps"] == (30 if record["system"] == "MemCam" else 10)
    frame_hashes = {}
    for method, _ in METHODS:
        paths = record["frame_paths"][method]
        assert set(paths) == {str(f) for f in record["frames"]}
        receipt = json.loads((Path(next(iter(paths.values()))).parent / "source.json").read_text())
        assert receipt["identity"]["source_sha256"] == record["source_sha256"][method]
        assert digest(Path(record["source_videos"][method])) == record["source_sha256"][method]
        for frame, name in paths.items():
            p = Path(name)
            frame_hashes[f"{method}/{frame}"] = digest(p)
            assert receipt["frames"][p.name] == frame_hashes[f"{method}/{frame}"]
            with Image.open(p) as im:
                assert im.size == (record["metadata"]["width"], record["metadata"]["height"])
    record["highlights"] = []
    record["verified_frame_sha256"] = frame_hashes
    record["parent_provenance_sha256"] = digest(path)
    return record


def panel(d, record, offset):
    tile, gap, top, row_height = 640, 12, 132, 360
    width = 3*tile + 2*gap
    d.label(f"{record['system']}, {record['duration_sec']} s", offset, 0, 700, 48, size=42, bold=True)
    d.label(record["scene"], offset+920, 6, width-920, 40, size=27, align="right", color="#555555")
    for col, (role, frame) in enumerate(zip(("First visit", "Intervening view", "Revisit"), record["frames"])):
        x = offset + col*(tile+gap)
        d.label(role, x, 56, tile, 40, size=36, align="center")
        seconds = f"{frame/record['timeline_fps']:g}"
        d.label(f"{seconds} s", x, 98, tile, 30, size=28, align="center", color="#555555")
        for row, (method, color) in enumerate(METHODS):
            height = tile*record["metadata"]["height"]/record["metadata"]["width"]
            y = top+row*(row_height+gap)
            with Image.open(record["frame_paths"][method][str(frame)]) as im:
                d.picture(im.convert("RGB"), x, y, tile, height, "1")
            cell = list(d.root)[-1]
            cell.set("data-example", record["stem"])
            cell.set("data-policy", method)
            cell.set("data-frame-index", str(frame))
            cell.set("data-timeline-fps", str(record["timeline_fps"]))
            cell.set("data-frame-sha256", record["verified_frame_sha256"][f"{method}/{frame}"])
            if col == 0:
                label_width = {"Unbounded": 198, "FIFO": 82, "KEEPSAKE": 198}[method]
                d.label(method, x+7, y+7, label_width, 38, size=34, color=color, bold=True,
                        extra="fillColor=#FFFFFF;spacingLeft=2;spacingRight=2;")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    combined = ET.Element("mxfile", host="app.diagrams.net")
    captions, all_records = [], []
    for i, pair in enumerate(PAIRS, 1):
        print(f"Verifying pair {i}: {pair}", flush=True)
        records = [load_case(stem) for stem in pair]
        d = Diagram(width=3912, height=1236)
        page = d.file.find("diagram")
        page.set("id", f"pair-{i:02d}")
        page.set("name", f"Pair {i:02d}")
        for j, record in enumerate(records):
            panel(d, record, j*1968)
        images = [c for c in d.root if "shape=image;" in c.get("style", "")]
        assert len(images) == 18
        assert len({c.get("id") for c in d.root}) == len(d.root)
        for c in d.root:
            geo = c.find("mxGeometry")
            if geo is not None:
                assert float(geo.get("x")) >= 0 and float(geo.get("y")) >= 0
                assert float(geo.get("x"))+float(geo.get("width")) <= 3912
                assert float(geo.get("y"))+float(geo.get("height")) <= 1236
        d.save(OUT / f"pair_{i:02d}.drawio")
        combined.append(page)
        caption = (f"Pair {i:02d}. Left: {records[0]['scene']}; right: {records[1]['scene']}. "
                   "Rows: Unbounded, FIFO B32, KEEPSAKE B32. All policies use identical frame indices within each example. "
                   "These manually chosen examples include mixed outcomes and do not estimate aggregate performance.\n" +
                   "\n".join(r["caption"] for r in records))
        (OUT / f"pair_{i:02d}.caption.txt").write_text(caption+"\n")
        captions.append(caption)
        all_records.append(dict(pair=i, records=records))
    ET.indent(combined)
    ET.ElementTree(combined).write(OUT / "appendix_pairs.drawio", encoding="utf-8", xml_declaration=True)
    (OUT / "provenance.json").write_text(json.dumps(all_records, indent=2)+"\n")
    (OUT / "captions.txt").write_text("\n\n".join(captions)+"\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    build()
