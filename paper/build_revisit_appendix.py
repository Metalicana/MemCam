"""Build an outcome-independent, editable appendix of saved near-pose returns.

One page per eligible MemCam scene and per WorldMem trajectory. Selection never
uses generated pixels or quality scores; original full frames are embedded.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper import build_revisit_drawio as memcam
from paper import build_worldmem_revisit as worldmem
from paper.make_180s_gt_comparisons import digest, load_items, save_json
from paper.make_revisit_comparisons import load_events


MEMCAM_RULE = "longest_return_per_scene"
WORLDMEM_RULE = "first_geometry_rank_per_trajectory"


def select_memcam(records):
    # Prefer long separations, with a deterministic endpoint tie break.
    selected = {}
    for record in sorted(records, key=lambda r: (
            -(r["frames"][-1] - r["frames"][0]), r["frames"][0], r["frames"][-1])):
        selected.setdefault((record["scene"], int(record["start_frame"])), record)
    return [selected[key] for key in sorted(selected)]


def select_worldmem(candidates):
    selected = {}
    for candidate in sorted(candidates, key=lambda c: (c["trajectory"], c["rank"])):
        selected.setdefault(candidate["trajectory"], candidate)
    return [selected[key] for key in sorted(selected)]


def make_specs(config):
    if (config["memcam_selection"] != MEMCAM_RULE
            or config["worldmem_selection"] != WORLDMEM_RULE):
        raise ValueError("Unknown selection rule; do not silently change the appendix sample")
    bundle = memcam.resolve(config["memcam_bundle"])
    events = memcam.resolve(config["memcam_events"])
    items = load_items(bundle)
    records = load_events(events, items)
    sources = {(p["scene"], int(p["start_frame"])): p for p in items}
    specs = []
    for number, record in enumerate(select_memcam(records), 1):
        item = sources[record["scene"], int(record["start_frame"])]
        filename = Path(item["videos"]["unbounded"]).name
        spec = dict(system="MemCam", stem=f"memcam_{number:02d}_{record['scene']}",
                    example_id=f"M{number:02d}", scene=record["scene"],
                    start_frame=int(record["start_frame"]), duration_sec=180,
                    timeline_fps=30, frames=record["frames"], selection_rule=MEMCAM_RULE,
                    expected_video=dict(width=640, height=352, frames=5397, fps=30.),
                    videos={"Unbounded": str(bundle / item["videos"]["unbounded"]),
                            "FIFO": str(memcam.resolve(config["memcam_fifo"]) / filename),
                            "KEEPSAKE": str(bundle / item["videos"]["keepsake_b32"])},
                    revisit_evidence=dict(csv=str(events), match={key: record["event"][key]
                        for key in ("scene", "start_frame", "revisit_type", "frame_i", "frame_j")}))
        specs.append(spec)
    candidates, _ = worldmem.load_bundle(memcam.resolve(config["worldmem_bundle"]))
    for candidate in select_worldmem(candidates):
        t = candidate["trajectory"]
        specs.append(dict(system="WorldMem", stem=f"worldmem_trajectory_{t:02d}",
                          example_id=f"W{t + 1:02d}", display_scene=f"Trajectory {t:02d}",
                          trajectory_id=t, candidate_rank=candidate["rank"],
                          frames=candidate["frames"], selection_rule=WORLDMEM_RULE))
    stems = [s["stem"] for s in specs]
    if not specs or len(stems) != len(set(stems)):
        raise ValueError("Expected nonempty, distinct example pages")
    audit = dict(memcam_candidate_count=len(records), worldmem_candidate_count=len(candidates),
                 memcam_scenes_without_saved_events=[p["scene"] for p in items
                     if (p["scene"], int(p["start_frame"])) not in
                     {(r["scene"], int(r["start_frame"])) for r in records}],
                 selection="One example from every eligible scene/trajectory; no pixel or quality-score filtering.",
                 memcam_rule="Maximum endpoint time separation; ties use earliest first, then return frame. Middle is the arithmetic midpoint, not a verified departure.",
                 worldmem_rule="Lowest exported geometry rank per trajectory, including its verified-departure middle frame.",
                 scope="Coverage of these saved candidate files, not all possible revisits and not an aggregate performance estimate.")
    return specs, audit


def caption(record):
    first, middle, last = [frame / record["timeline_fps"] for frame in record["frames"]]
    base = (f"{record['example_id']}: {record['system']}, {record['scene']}, "
            f"{record['duration_sec']}-second rollout. Unbounded, FIFO B32 and KEEPSAKE B32 "
            f"at identical generated times {first:g}, {middle:g} and {last:g} seconds. ")
    event = record["event"]
    if record["system"] == "MemCam":
        return (base + f"Endpoint pose difference {float(event['position_distance']):.3f} m "
                f"and {float(event['rotation_deg']):.2f} degrees. Selected by longest "
                "saved near-pose return per scene. The middle is an intervening midpoint, "
                "not a verified departure; 'First visit' means the earlier sampled endpoint, "
                "not necessarily the first-ever visit. This is not a GT reconstruction comparison.")
    return (base + f"Requested endpoint pose difference {float(event['endpoint_position_distance']):.3f} "
            f"block-coordinate units and {float(event['endpoint_rotation_deg']):.2f} degrees, "
            "with a verified departure in the requested trajectory. Selected by best exported "
            "geometry rank per trajectory, not output quality. Times use the 10-FPS trajectory "
            "clock, not 15-FPS MP4 playback. Actual post-retry dataset identity was not logged "
            "upstream; these are requested-trajectory matches, not verified GT matches.")


def appendix_diagram(record):
    diagram = memcam.make_diagram(record)
    page = diagram.file.find("diagram")
    page.set("name", f"{record['example_id']} | {record['system']} | {record['scene']}")
    for cell in diagram.root.findall("mxCell"):
        value = cell.get("value")
        if value == f"{record['system']}, {record['duration_sec']} s":
            cell.set("value", f"{record['example_id']} &nbsp; {value}")
        if value in ("FIFO", "KEEPSAKE"):
            label = "FIFO<br>B32" if value == "FIFO" else "KEEPSAKE<br>(Ours, B32)"
            cell.set("value", label)
            cell.set("style", cell.get("style").replace("fontSize=34;", "fontSize=30;"))
            geometry = cell.find("mxGeometry")
            geometry.set("y", str(float(geometry.get("y")) - 16))
            geometry.set("height", "84")
        identity = cell.get("id", "")
        if identity.startswith("frame-"):
            method_slug, frame = identity.removeprefix("frame-").rsplit("-", 1)
            method = {m.lower(): m for m, _ in memcam.METHODS}[method_slug]
            cell.set("data-frame-index", frame)
            cell.set("data-timeline-fps", str(record["timeline_fps"]))
            cell.set("data-source-sha256", record["source_sha256"][method])
    return diagram


def write_shortlist(config, combined, rows, output, config_sha256):
    entries = config.get("shortlist", [])
    if not entries:
        return
    ids = [entry["example_id"] for entry in entries]
    by_id = {row["example_id"]: row for row in rows}
    if len(ids) != len(set(ids)) or any(identity not in by_id for identity in ids):
        raise ValueError("Shortlist must name distinct existing examples")
    pages = {page.get("id"): page for page in combined.findall("diagram")}
    document = ET.Element("mxfile", host="app.diagrams.net")
    selected = []
    for number, entry in enumerate(entries, 1):
        row = by_id[entry["example_id"]]
        document.append(pages[row["stem"]])
        selected.append(dict(shortlist_page=number, catalog_page=row["page"],
                             example_id=row["example_id"], stem=row["stem"], note=entry["note"]))
    ET.indent(document)
    ET.ElementTree(document).write(output / "appendix_shortlist.drawio", encoding="utf-8", xml_declaration=True)
    save_json(output / "shortlist.json", dict(examples=selected,
        selection="Manual visual selection for inspectable scene structure from the complete geometry/time-selected collection. Includes mixed outcomes; not a representative performance sample.",
        config_sha256=config_sha256))


def build(config_path, output):
    config = json.loads(config_path.read_text())
    specs, audit = make_specs(config)
    output.mkdir(parents=True, exist_ok=True)
    individual = output / "examples"
    individual.mkdir(exist_ok=True)
    combined = ET.Element("mxfile", host="app.diagrams.net")
    rows = []
    renderer_hashes = {name: digest(ROOT / "paper" / name) for name in (
        "build_revisit_appendix.py", "build_revisit_drawio.py", "build_worldmem_revisit.py",
        "build_motivation_teaser.py", "replace_method_frame_assets.py")}
    for page_number, spec in enumerate(specs, 1):
        print(f"[{page_number}/{len(specs)}] {spec['stem']}: {spec['frames']}", flush=True)
        if spec["system"] == "MemCam":
            record = memcam.prepare_case(spec, output / "frames" / spec["stem"])
        else:
            record = worldmem.prepare_case(memcam.resolve(config["worldmem_bundle"]),
                         memcam.resolve(config["worldmem_videos"]), spec, output)
        record.update(config_sha256=digest(config_path), renderer_hashes=renderer_hashes,
                      selection_audit=audit,
                      display="Full original decoded frames; no crop, warping, color correction, enhancement or synthesis. Images and labels remain individually editable.")
        record["caption"] = caption(record)
        diagram = appendix_diagram(record)
        diagram.save(individual / f"{spec['stem']}.drawio")
        combined.append(diagram.file.find("diagram"))
        save_json(individual / f"{spec['stem']}.provenance.json", record)
        (individual / f"{spec['stem']}.caption.txt").write_text(record["caption"] + "\n")
        event = record["event"]
        row = dict(page=page_number, example_id=spec["example_id"], system=record["system"],
                   scene=record["scene"], stem=record["stem"],
                   selection_rule=record["selection_rule"], timeline_fps=record["timeline_fps"],
                   playback_fps=record["metadata"]["fps"],
                   endpoint_position_distance=event.get("position_distance", event.get("endpoint_position_distance")),
                   endpoint_rotation_deg=event.get("rotation_deg", event.get("endpoint_rotation_deg")),
                   position_unit="m" if record["system"] == "MemCam" else "block-coordinate unit",
                   verified_departure=record["system"] == "WorldMem",
                   caption=record["caption"])
        for role, frame in zip(worldmem.ROLES, record["frames"]):
            row[f"{role}_frame"] = frame
            row[f"{role}_time_sec"] = frame / record["timeline_fps"]
        rows.append(row)
    ET.indent(combined)
    ET.ElementTree(combined).write(output / "revisit_appendix.drawio", encoding="utf-8", xml_declaration=True)
    with (output / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "selection.json", dict(audit=audit, examples=specs,
              config=config, config_sha256=digest(config_path), renderer_hashes=renderer_hashes))
    (output / "captions.txt").write_text("\n\n".join(row["caption"] for row in rows) + "\n")
    write_shortlist(config, combined, rows, output, digest(config_path))
    lines = ["# Revisit Appendix", "", f"{len(rows)} independent editable pages; nine embedded original frames per page.",
             "Open `revisit_appendix.drawio` and switch pages using the bottom tabs.",
             "Individual documents are in `examples/`; source frames in `frames/`.", "",
             "Selection is geometry/time based, not policy-performance based. This includes mixed outcomes.",
             "MemCam middle frames are temporal midpoints, not verified departures. WorldMem uses requested",
             "trajectory poses and a 10-FPS simulation clock; actual post-retry dataset identity is unverified.",
             "Consult `captions.txt`, `index.csv` and the individual provenance JSON files before publication.", "",
             "No object boxes are asserted automatically. Every image and label is independently editable.", "",
             "`appendix_shortlist.drawio` is a smaller manual visual selection for inspectable scene structure.",
             "It includes mixed outcomes and is not a representative performance sample; see `shortlist.json`.", "",
             "| PDF page | Example | Scene / trajectory | Times (s) |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['page']} | {r['example_id']} / {r['system']} | {r['scene']} | "
              f"{r['first_time_sec']:g}, {r['middle_time_sec']:g}, {r['revisit_time_sec']:g} |" for r in rows]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Created {output / 'revisit_appendix.drawio'} ({len(rows)} pages)", flush=True)


def render_previews(output):
    """Rasterize the actual draw.io PDF export, not a second diagram renderer."""
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    with (output / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # pdftoppm pads page numbers to the number of digits in the document length.
    digits = len(str(len(rows)))
    subprocess.run(["pdftoppm", "-png", "-scale-to", "1600", "-r", "96",
                    str(output / "revisit_appendix.pdf"), str(previews / "page")], check=True)
    contact_sheets(rows, previews, output, "overview", digits)
    shortlist = output / "shortlist.json"
    if shortlist.exists():
        by_id = {row["example_id"]: row for row in rows}
        selected = [by_id[entry["example_id"]] for entry in json.loads(shortlist.read_text())["examples"]]
        contact_sheets(selected, previews, output, "shortlist_overview", digits)
    print(f"Rendered {len(rows)} page previews and overview sheets", flush=True)


def contact_sheets(rows, previews, output, prefix, digits):
    for offset in range(0, len(rows), 6):
        group = rows[offset:offset + 6]
        canvas = Image.new("RGB", (1620, 470 * ((len(group) + 1) // 2)), "#EEEEEE")
        for n, row in enumerate(group):
            path = previews / f"page-{int(row['page']):0{digits}d}.png"
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((800, 455), Image.Resampling.LANCZOS)
                canvas.paste(image, (5 + (n % 2) * 810, 5 + (n // 2) * 470))
        canvas.save(output / f"{prefix}_{offset // 6 + 1:02d}.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper/configs/revisit_appendix_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/revisit_appendix")
    parser.add_argument("--preview-only", action="store_true",
                        help="Render PNGs and overview sheets from an existing draw.io PDF export")
    args = parser.parse_args()
    if args.preview_only:
        render_previews(args.output)
    else:
        build(args.config, args.output)


if __name__ == "__main__":
    main()
