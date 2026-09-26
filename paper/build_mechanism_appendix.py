"""Editable bank, logged-reader, and full-resolution common-source figures."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build_motivation_teaser import Diagram
from paper.make_180s_gt_comparisons import decode_cached, digest
from paper.replace_method_frame_assets import update_bank
from utils.visualize_common_source_psnr_extremes import fit_image

DOWNLOADS = Path.home() / "Downloads"
RUNS = DOWNLOADS / "context_180s"
BUNDLE = DOWNLOADS / "memcam_edit_180s"
OUT = ROOT / "paper/figures/mechanism_appendix"
ARMS = (("Unbounded", "baseline", "#454545"), ("FIFO B32", "fifo_b32", "#BF665D"),
        ("KEEPSAKE B32", "slam_b32_covisibility", "#277A60"))
AUDIT = []


def image(d, path, x, y, w, h, **metadata):
    with Image.open(path) as im:
        scale = min(w / im.width, h / im.height)
        dw, dh = im.width * scale, im.height * scale
        d.picture(im.convert("RGB"), x+(w-dw)/2, y+(h-dh)/2, dw, dh, "1")
    node = list(d.root)[-1]
    for k, v in dict(metadata, source_sha256=digest(Path(path))).items():
        node.set("data-"+k.replace("_", "-"), str(v))


def trace(item, run, target):
    path = RUNS / run / "access_traces" / (item["output_prefix"]+"custom.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get("scene") is not None:
            assert (r["scene"], r["dataset_start_frame"], r["duration_sec"]) == (
                item["scene"], item["start_frame"], 180)
    matches = [r for r in rows if r.get("event") == "context_access" and r.get("target_frame") == target]
    assert len(matches) == 1 and matches[0]["selected"]
    e = matches[0]
    assert e["selected_memory_frame"] < e["section_idx"]*76-3
    if run != "baseline":
        bank = update_bank(rows, e["section_idx"]-1)[2]
        eligible = bank-set(range(e["section_idx"]*76-3, e["section_idx"]*76+1))
        assert len(bank) == e["stored_memory_size"] == 32
        assert len(eligible) == e["candidate_count"] and e["selected_memory_frame"] in eligible
    else:
        bank = set(range(e["section_idx"]*76+1))
        assert len(bank) == e["stored_memory_size"]
    AUDIT.append(dict(kind="trace", path=str(path), sha256=digest(path), query=e, resident_bank=sorted(bank)))
    return e, sorted(bank)


def decode(item, run, frames, tag):
    video = RUNS / run / (item["output_prefix"]+"custom.mp4")
    paths, receipt = decode_cached(video, sorted(set(frames)), OUT / "assets" / tag / run)
    AUDIT.append(dict(kind="video", path=str(video), receipt=receipt))
    return paths


def save(d, stem, name):
    page = d.file.find("diagram")
    page.set("id", stem)
    page.set("name", name)
    ids = [c.get("id") for c in d.root]
    assert len(ids) == len(set(ids))
    model = page.find("mxGraphModel")
    for cell in d.root:
        g = cell.find("mxGeometry")
        if g is not None and cell.get("vertex") == "1":
            assert float(g.get("x"))+float(g.get("width")) <= float(model.get("pageWidth"))
            assert float(g.get("y"))+float(g.get("height")) <= float(model.get("pageHeight"))
    d.save(OUT / f"{stem}.drawio")
    return page


def reader(item, target, number):
    d = Diagram(width=2020, height=1390)
    d.label(f"Recorded memory reads | {item['scene']}", 18, 6, 1980, 50, size=38, bold=True)
    d.label(f"MemCam, 180 s rollout | Target: frame {target}, {target/30:g} s", 18, 59, 1980, 34, size=26)
    for x, text in ((18, "Target ground truth"), (684, "Actually selected memory"), (1350, "Generated target frame")):
        d.label(text, x, 103, 640, 38, size=30, bold=True, align="center")
    sample = next(s for s in item["samples"] if s["frame_index"] == target)
    gt = BUNDLE / sample["gt_file"]
    for i, (label, run, color) in enumerate(ARMS):
        e, _ = trace(item, run, target)
        memory = e["selected_memory_frame"]
        paths = decode(item, run, [memory, target], f"reader_{number}")
        y = 148 + i*400
        for x, path, role, frame in ((18, gt, "gt", target), (684, paths[memory], "memory", memory),
                                    (1350, paths[target], "output", target)):
            image(d, path, x, y, 640, 352, role=role, frame=frame, policy=label)
        d.label(label, 27, y+8, 275, 40, size=30, bold=True, color=color, extra="fillColor=#FFFFFF;")
        d.label(f"Memory f{memory} ({memory/30:.2f} s) | Age {(target-memory)/30:.2f} s | FOV overlap {e['selected_overlap']:.3f}",
                684, y+355, 1306, 36, size=25)
    d.label("Each row uses its own rollout pixels. One logged read is shown from the chunk's multi-frame context; this is not a single-frame causal intervention.",
            18, 1350, 1980, 36, size=22)
    return save(d, f"reader_{number:02d}", item["scene"])


def bank(item, target):
    info = [(label, run, color, *trace(item, run, target)) for label, run, color in ARMS[1:]]
    indices = sorted(set(f for _, _, _, _, bank in info for f in bank))
    paths = decode(item, "baseline", indices, "bank_common_source")
    d = Diagram(width=2056, height=1590)
    d.label(f"What remains in memory | {item['scene']}", 16, 4, 2024, 48, size=38, bold=True)
    d.label(f"Before chunk {info[0][3]['section_idx']} (zero-based), query f{target} at {target/30:g} s | All 32 resident IDs per bank", 16, 57, 2024, 34, size=25)
    d.label("Common-source display: recorded bank IDs, all thumbnails from the same Unbounded video. Not a counterfactual policy replay.",
            16, 94, 2024, 34, size=24)
    for col, (label, run, color, e, frames) in enumerate(info):
        x0 = 16+col*1024
        d.label(label, x0, 140, 1000, 38, size=32, color=color, bold=True)
        d.label(f"History span: {frames[0]/30:.2f}-{frames[-1]/30:.2f} s | {e['candidate_count']} reader-eligible", x0, 180, 1000, 32, size=24)
        for n, f in enumerate(frames):
            x, y = x0+(n%4)*252, 226+(n//4)*166
            image(d, paths[f], x, y, 240, 132, frame=f, bank_policy=label, pixels_from="Unbounded")
            d.label(f"f{f} | {f/30:.2f} s", x, y+134, 240, 26, size=21, align="center")
    d.label("Chronological order; no retained IDs omitted. Recent anchor frames remain resident but are excluded from the reader's candidate set.",
            16, 1557, 2024, 30, size=22)
    save(d, "retained_banks", "Retained banks")


def extremes(items):
    source = DOWNLOADS / "extreme_unbounded_vs_geocov"
    csv_path = source / "tables/selected_examples.csv"
    cases = list(csv.DictReader(csv_path.open()))
    assert len(cases) == 5
    AUDIT.append(dict(kind="extreme_selection", path=str(csv_path), sha256=digest(csv_path)))
    d = Diagram(width=1680, height=2610)
    d.label("Extreme common-source retrieval examples", 16, 4, 1648, 48, size=36, bold=True)
    d.label("Selected memory pixels come from one Unbounded video. Scores use GT at each memory's own historical index, not target-view GT.",
            16, 56, 1648, 56, size=24)
    for x, w, label in ((16, 320, "Target GT"), (352, 640, "Unbounded-selected index"), (1008, 640, "KEEPSAKE-selected index")):
        d.label(label, x, 124, w, 38, size=29, bold=True, align="center")
    for i, case in enumerate(cases):
        item = next(r for r in items if r["scene"] == case["scene"] and r["start_frame"] == int(case["dataset_start_frame"]))
        q = int(case["target_frame"])
        ids = [int(case[k]) for k in ("unbounded_selected_frame", "geocov_selected_frame")]
        for run, frame in zip(("baseline", "slam_b32_covisibility"), ids):
            e, _ = trace(item, run, q)
            assert e["selected_memory_frame"] == frame and e["section_idx"] == int(case["section_idx"])
        paths = decode(item, "baseline", ids, f"extreme_{i}")
        old_path = source / "figures/cases" / f"case_{i:02d}_row{case['row']}_section{case['section_idx']}_target{q}.png"
        with Image.open(old_path) as old:
            # Recover the existing GT tile only; full-resolution GT is not in this download.
            assert old.width == 7*292 and old.height == 342
            gt = old.crop((5, 124, 287, 288)).convert("RGB")
            gt_path = OUT / "assets" / f"extreme_{i}_saved_target_gt.png"
            gt.save(gt_path)
            for frame, old_col in zip(ids, (1, 3)):
                with Image.open(paths[frame]) as native:
                    reference = np.asarray(fit_image(native, 282, 164), dtype=float)
                saved = np.asarray(old.crop((old_col*292+5,124,old_col*292+287,288)), dtype=float)
                error = float(np.abs(reference-saved).mean())
                assert error < 2, (old_path, frame, error)
        AUDIT.append(dict(kind="extreme_case", record=case, source_montage=str(old_path), sha256=digest(old_path),
                          target_gt_resolution="282x164 saved tile; original GT absent"))
        y = 176+i*478
        d.label(f"{item['scene']} | Query f{q}, {q/30:.2f} s", 16, y, 1648, 36, size=28, bold=True)
        image(d, gt_path, 16, y+105, 320, 164*320/282, role="saved_target_gt", frame=q)
        d.label("Saved GT reference", 16, y+309, 320, 32, size=23, align="center")
        for col, (prefix, frame) in enumerate(zip(("unbounded", "geocov"), ids)):
            x = 352+col*656
            image(d, paths[frame], x, y+40, 640, 352, frame=frame, pixels_from="Unbounded", selected_by=prefix)
            origin = " | conditioning image" if frame == 0 else ""
            d.label(f"f{frame}, {frame/30:.2f} s{origin}", x, y+397, 640, 30, size=25, align="center")
            d.label(f"PSNR {float(case[prefix+'_psnr']):.2f} dB | SSIM {float(case[prefix+'_ssim']):.3f}",
                    x, y+430, 640, 30, size=25, align="center")
    d.label("Ranked extreme cases, not representative wins. Four KEEPSAKE indices are frame 0 (the conditioning image). No sharpening or synthesis.",
            16, 2570, 1648, 36, size=22)
    save(d, "extreme_common_source", "Extreme common-source examples")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads((BUNDLE / "manifest.json").read_text())
    combined = ET.Element("mxfile", host="app.diagrams.net")
    # Available exact GT timestamps; all three logged FOV overlaps exceed .8,
    # and KEEPSAKE's selected frame is at least fifteen seconds old.
    for number, (scene, target) in enumerate((("AnimeCitySuburbs_5", 3600), ("ClothingStore_6", 1800)), 1):
        print(f"Reader {number}: {scene}", flush=True)
        item = next(r for r in items if r["scene"] == scene)
        combined.append(reader(item, target, number))
        if number == 1:
            print("Retained-bank comparison", flush=True)
            bank(item, target)
    ET.indent(combined)
    ET.ElementTree(combined).write(OUT / "reader_examples.drawio", encoding="utf-8", xml_declaration=True)
    print("Full-resolution extreme examples", flush=True)
    extremes(items)
    (OUT / "provenance.json").write_text(json.dumps(AUDIT, indent=2)+"\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
