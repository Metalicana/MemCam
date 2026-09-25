"""Review exported WorldMem pose revisits and build an editable three-policy figure."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build_revisit_drawio import make_diagram
from paper.compare_local_rollouts import font, probe
from paper.make_180s_gt_comparisons import bundle_path, decode_cached, digest, save_json

SOURCES = {"Unbounded": "unbounded", "FIFO": "fifo_b32", "KEEPSAKE": "keepsake_b32"}
DIRECTORIES = {"Unbounded": "Unbounded", "FIFO": "FIFO_B32", "KEEPSAKE": "KEEPSAKE_B32"}
ROLES = ("first", "middle", "revisit")
DEFAULT_BUNDLE = ROOT / "paper/figures/revisit_editable/worldmem_source/revisit_qualitative_60s_n15"


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_bundle(folder):
    provenance = json.loads((folder / "provenance.json").read_text())
    events = csv_rows(folder / "revisit_candidates.csv")
    mappings = csv_rows(folder / "trajectory_mapping.csv")
    clock = provenance["clock_and_index_mapping"]
    if (clock["trajectory_fps"] != 10 or clock["expected_mp4_frame_count"] != 600
            or clock["output_mp4_contains_initial_context"] or clock["output_frame_index_base"] != 0):
        raise ValueError("Unexpected WorldMem output clock/index contract")
    by_trajectory = {int(m["trajectory_id"]): m for m in mappings}
    if len(by_trajectory) != 15 or len(mappings) != 15 or set(by_trajectory) != set(range(15)):
        raise ValueError("Expected the fifteen distinct matched trajectories")
    previews = defaultdict(dict)
    for p in csv_rows(folder / "preview_manifest.csv"):
        key = (int(p["trajectory_id"]), int(p["candidate_rank"]))
        slot = (p["source"], p["role"])
        allowed = {"exported", "exported_unchecked"} if p["source"] == "ground_truth" else {"exported"}
        if slot in previews[key] or p["status"] not in allowed:
            raise ValueError("Duplicate or unavailable preview")
        suffix = Path(p["path"]).relative_to(Path(provenance["outputs"]["preview_root"]))
        path = bundle_path(folder, str(Path("previews") / suffix))
        if int(p["source_video_frame"]) != int(p["output_frame"]) + clock["source_start_offset"] + clock["context_frames"]:
            raise ValueError("Preview source-frame mapping mismatch")
        previews[key][slot] = dict(path=str(path), frame=int(p["output_frame"]))
    candidates, seen = [], set()
    for event in events:
        t = int(event["trajectory_id"])
        frames = [int(event[f"{role}_frame"]) for role in ROLES]
        mapping = by_trajectory[t]
        if (frames != sorted(set(frames)) or not 0 <= frames[0] < frames[-1] < 600
                or event["video_filename"] != mapping["video_filename"]
                or mapping["same_requested_trajectory"] != "True"
                or float(mapping["trajectory_fps"]) != 10 or int(mapping["output_frames"]) != 600):
            raise ValueError("Candidate identity/clock/frame mismatch")
        if (event["pose_match"] != "True" or event["away_verified"] != "True"
                or not 0 <= float(event["endpoint_position_distance"]) <= .750001
                or not 0 <= float(event["endpoint_rotation_deg"]) <= 15.000001
                or float(event["away_duration_sec"]) < 1
                or (frames[-1] - frames[0]) / 10 < 5
                or not (float(event["middle_position_distance"]) >= 2 - 1e-6
                        or float(event["middle_rotation_deg"]) >= 45 - 1e-6)):
            raise ValueError("Candidate fails the exported pose/departure thresholds")
        if any(abs(float(event[f"{role}_time_sec"]) - frame / 10) > 1e-6 for role, frame in zip(ROLES, frames)):
            raise ValueError("Timestamps must use the 10-FPS trajectory clock")
        matches = [(key, values) for key, values in previews.items() if key[0] == t
                   and [values.get(("unbounded", role), {}).get("frame") for role in ROLES] == frames]
        if len(matches) != 1 or matches[0][0] in seen:
            raise ValueError("Candidate must map to one distinct preview triplet")
        key, values = matches[0]
        expected_slots = {(source, role) for source in (*SOURCES.values(), "ground_truth") for role in ROLES}
        if set(values) != expected_slots or any(values[source, role]["frame"] != frame
                for source in (*SOURCES.values(), "ground_truth") for role, frame in zip(ROLES, frames)):
            raise ValueError("Preview policy/GT frame alignment mismatch")
        seen.add(key)
        candidates.append(dict(trajectory=t, rank=key[1], frames=frames, event=event,
                               mapping=mapping, previews=values))
    if len(seen) != len(previews) or len(candidates) != sum(provenance["search_and_sampling"]["selected_candidate_counts"].values()):
        raise ValueError("Incomplete candidate coverage")
    return sorted(candidates, key=lambda c: (c["trajectory"], c["rank"])), provenance


def review(folder, output):
    candidates, _ = load_bundle(folder)
    output.mkdir(parents=True, exist_ok=True)
    tw, th, left, gap, top, row_gap = 192, 108, 132, 4, 55, 25
    for trajectory in sorted({c["trajectory"] for c in candidates}):
        group = [c for c in candidates if c["trajectory"] == trajectory]
        canvas = Image.new("RGB", (left + 8 * (tw + gap), top + len(group) * (th + row_gap)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Trajectory {trajectory}", font=font(16, True), fill="black")
        for k, name in enumerate(("GT reference", *SOURCES)):
            draw.text((left + 2 * k * (tw + gap), 7), name + "  First / Return", font=font(18, True), fill="black")
        for row, candidate in enumerate(group):
            y = top + row * (th + row_gap)
            first, middle, last = candidate["frames"]
            draw.multiline_text((8, y + 8), f"Rank {candidate['rank']}\n{first} / {last}\n{float(candidate['event']['endpoint_rotation_deg']):g} deg",
                                font=font(17), fill="black")
            for k, source in enumerate(("ground_truth", *SOURCES.values())):
                for col, role in enumerate(("first", "revisit")):
                    with Image.open(candidate["previews"][source, role]["path"]) as image:
                        canvas.paste(image.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS),
                                     (left + (2 * k + col) * (tw + gap), y))
        canvas.save(output / f"trajectory_{trajectory:02d}.png")
    save_json(output / "review.json", dict(candidate_count=len(candidates),
              candidate_csv_sha256=digest(folder / "revisit_candidates.csv"),
              method="All exported candidates in geometry-rank order; endpoint thumbnails only, no pixel edits."))
    print(f"Reviewed-sheet export: {len(candidates)} candidates in {output}", flush=True)


def prepare_case(folder, videos, spec, output):
    """Decode and verify one selection without writing or replacing diagram pages."""
    candidates, provenance = load_bundle(folder)
    selected = [c for c in candidates if c["trajectory"] == spec["trajectory_id"] and c["rank"] == spec["candidate_rank"]]
    if len(selected) != 1 or selected[0]["frames"] != spec["frames"]:
        raise ValueError("Selection does not identify one exported candidate")
    c = selected[0]
    output.mkdir(parents=True, exist_ok=True)
    stem = spec["stem"]
    if Path(stem).name != stem:
        raise ValueError("Invalid figure stem")
    record = dict(spec, system="WorldMem", scene=spec.get("display_scene", f"Trajectory {c['trajectory']}"), duration_sec=60,
                  timeline_fps=10., metadata=dict(width=640, height=360, frames=600, fps=15.),
                  frame_paths={}, source_sha256={}, pixel_checks={}, event=c["event"], mapping=c["mapping"],
                  source_videos={}, remote_provenance=provenance,
                  source_files_sha256={name: digest(folder / name) for name in
                      ("revisit_candidates.csv", "trajectory_mapping.csv", "preview_manifest.csv", "provenance.json")},
                  renderer_sha256=digest(Path(__file__)),
                  diagram_renderer_sha256=digest(ROOT / "paper/build_revisit_drawio.py"),
                  pixel_check_tolerance=dict(max_channel_error=3, mean_channel_error=1,
                      units="8-bit RGB levels", reason="Allow small decoder RGB-conversion rounding differences; not a perceptual similarity test."))
    for method, source in SOURCES.items():
        video = videos / DIRECTORIES[method] / c["mapping"]["video_filename"]
        remote = provenance["video_metadata"][str(c["trajectory"])][source]
        if probe(video) != record["metadata"] or video.stat().st_size != remote["size_bytes"]:
            raise ValueError(f"Local video differs from exporter metadata: {video}")
        frames, receipt = decode_cached(video, c["frames"], output / "frames" / stem / method)
        record["frame_paths"][method] = {str(f): str(p.resolve()) for f, p in frames.items()}
        record["source_sha256"][method] = receipt["identity"]["source_sha256"]
        record["source_videos"][method] = str(video.resolve())
        for role, frame in zip(ROLES, c["frames"]):
            preview = Path(c["previews"][source, role]["path"])
            with Image.open(preview) as remote_image, Image.open(frames[frame]) as local_image:
                a, b = np.asarray(remote_image.convert("RGB"), dtype=np.int16), np.asarray(local_image.convert("RGB"), dtype=np.int16)
                if a.shape != b.shape:
                    raise ValueError("Local/preview geometry differs")
                error = np.abs(a - b)
                if error.max() > 3 or error.mean() > 1:
                    raise ValueError(f"Local video pixels differ from exported preview: {method}, {frame}")
            record["pixel_checks"][f"{method}:{frame}"] = dict(max_error=int(error.max()), mean_error=float(error.mean()),
                                                               preview_sha256=digest(preview))
    return record


def build(folder, videos, config, output):
    spec = json.loads(config.read_text())
    record = prepare_case(folder, videos, spec, output)
    record["config_sha256"] = digest(config)
    stem = spec["stem"]
    diagram = make_diagram(record)
    diagram.save(output / f"{stem}.drawio")
    record.update(selection="Manually selected qualitative illustration after contact-sheet review; not an aggregate policy result.",
                  display="Full original video frames; no crop, exposure correction, image warping or generative editing. Boxes are separate editable annotations.",
                  interpretation="Near-return in the requested trajectory, with a verified departure. Actual post-retry dataset identity was not logged upstream.")
    save_json(output / f"{stem}.provenance.json", record)
    (output / f"{stem}.caption.txt").write_text(spec["caption"] + "\n")
    combined = ET.Element("mxfile", host="app.diagrams.net")
    memcam = output / "memcam_180s_revisit.drawio"
    if memcam.exists():
        for page in ET.parse(memcam).getroot().findall("diagram"):
            combined.append(page)
    combined.append(diagram.file.find("diagram"))
    ET.indent(combined)
    ET.ElementTree(combined).write(output / "revisit_comparisons.drawio", encoding="utf-8", xml_declaration=True)
    print(f"Created {output / (stem + '.drawio')}; all nine source frames checked against remote previews", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "build"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--videos", type=Path, default=Path.home() / "Downloads/worldmem_editing")
    parser.add_argument("--config", type=Path, default=ROOT / "paper/configs/worldmem_revisit_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/revisit_editable")
    args = parser.parse_args()
    if args.mode == "review":
        review(args.bundle, args.output / "worldmem_review")
    else:
        build(args.bundle, args.videos, args.config, args.output)


if __name__ == "__main__":
    main()
