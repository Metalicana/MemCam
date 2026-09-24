"""Render pose-checked revisit comparisons from local 180-second videos."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.make_180s_gt_comparisons import bundle_path, decode_cached, digest, load_items, save_json
from paper.compare_local_rollouts import probe

METHODS = (("unbounded", "Unbounded", "#B45C5C"),
           ("keepsake_b32", "KEEPSAKE", "#277A60"))


def load_events(path, items):
    identities = {(p["scene"], int(p["start_frame"])): p for p in items}
    records, seen = [], set()
    with path.open(newline="") as handle:
        for event in csv.DictReader(handle):
            if event["revisit_type"] != "exact_pose":
                continue
            item = identities[(event["scene"], int(event["start_frame"]))]
            first, last = int(event["frame_i"]), int(event["frame_j"])
            position, rotation = float(event["position_distance"]), float(event["rotation_deg"])
            key = (item["scene"], first, last)
            if (key in seen or int(event["duration_sec"]) != 180
                    or not 0 <= first < last < int(item["num_frames"])
                    or last - first < 150 or not 0 <= position <= 0.25
                    or not 0 <= rotation <= 5):
                raise ValueError(f"Invalid or duplicate pose-return event: {event}")
            for frame, field in ((first, "time_i_sec"), (last, "time_j_sec")):
                if abs(float(event[field]) - frame / item["fps"]) > 1e-8:
                    raise ValueError(f"Frame/time mismatch: {event}")
            seen.add(key)
            records.append(dict(scene=item["scene"], start_frame=item["start_frame"],
                                frames=[first, (first + last) // 2, last],
                                position_distance_m=position, rotation_difference_deg=rotation,
                                stem=f"{item['scene']}_{first:04d}_{last:04d}",
                                event=event, fps=item["fps"], frame_paths={}))
    if not records:
        raise ValueError("No eligible pose-return events")
    return records


def make_figure(record):
    methods = (("gt", "GT", "#454545"),) + METHODS if "gt" in record["frame_paths"] else METHODS
    dpi, tile_w, tile_h = 160, 640, 360 if "gt" in record["frame_paths"] else 352
    left, top, gap, bottom = 195, 103, 10, 12
    width = left + 3 * tile_w + 2 * gap + 12
    height = top + len(methods) * tile_h + (len(methods) - 1) * gap + bottom
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    fig.text(left / width, 1 - 17 / height, record["scene"], fontsize=14,
             weight="bold", va="top", color="#242424")
    for column, (phase, frame) in enumerate(zip(("First visit", "Between visits", "Return"), record["frames"])):
        x = left + column * (tile_w + gap)
        seconds = f"{frame / record['fps']:.1f}".removesuffix(".0")
        fig.text((x + tile_w / 2) / width, 1 - 71 / height,
                 f"{phase}  |  {seconds} s", ha="center", va="center", fontsize=12)
        for row, (method, label, color) in enumerate(methods):
            y = height - top - (row + 1) * tile_h - row * gap
            ax = fig.add_axes([x / width, y / height, tile_w / width, tile_h / height])
            with Image.open(record["frame_paths"][method][str(frame)]) as image:
                ax.imshow(image.convert("RGB"), aspect="equal", interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(column != 1)
                spine.set_linewidth(1.4)
                spine.set_color(color)
            if column == 0:
                fig.text((left - 13) / width, (y + tile_h / 2) / height, label,
                         ha="right", va="center", fontsize=12, color=color,
                         weight="bold" if method == "keepsake_b32" else "normal")
    return fig


def render(folder, events_path, output, selection=None):
    items = load_items(folder)
    records = load_events(events_path, items)
    spec = json.loads(selection.read_text()) if selection else []
    for entry in spec:
        if entry.get("basis") != "visual_gt":
            continue
        item = next(p for p in items if p["scene"] == entry["scene"])
        frames = entry["frames"]
        samples = {p["frame_index"]: p for p in item["samples"]}
        if len(frames) != 3 or frames != sorted(set(frames)) or any(i not in samples for i in frames):
            raise ValueError("Visual-GT sequence requires three ordered, exported GT frames")
        paths = {str(i): str(bundle_path(folder, samples[i]["gt_file"])) for i in frames}
        records.append(dict(scene=item["scene"], start_frame=item["start_frame"],
                            frames=frames, stem=entry["stem"], fps=item["fps"],
                            frame_paths={"gt": paths}, gt_sha256={i: digest(Path(p)) for i, p in paths.items()},
                            gt_dataset_indices=[item["start_frame"] + i for i in frames],
                            return_basis="Visual GT correspondence; numerical pose distance not available for these endpoints."))
    output.mkdir(parents=True, exist_ok=True)
    for item in items:
        group = [r for r in records if r["scene"] == item["scene"]]
        if not group:
            continue
        indices = sorted({frame for r in group for frame in r["frames"]})
        print(f"{item['scene']}: {len(group)} returns, {len(indices)} frames/method", flush=True)
        for method, _, _ in METHODS:
            video = bundle_path(folder, item["videos"][method])
            metadata = probe(video)
            if metadata != {"width": 640, "height": 352, "frames": item["num_frames"], "fps": item["fps"]}:
                raise ValueError(f"Video metadata mismatch: {video}: {metadata}")
            paths, receipt = decode_cached(video, indices, output / "frames" / item["scene"] / method)
            for record in group:
                record["frame_paths"][method] = {str(i): str(paths[i].resolve()) for i in record["frames"]}
                record.setdefault("videos", {})[method] = str(video)
                record.setdefault("video_sha256", {})[method] = receipt["identity"]["source_sha256"]
    previews = []
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        for record in records:
            fig = make_figure(record)
            fig.savefig(output / f"{record['stem']}.png", dpi=160)
            plt.close(fig)
            with Image.open(output / f"{record['stem']}.png") as image:
                preview = image.convert("RGB")
                preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                previews.append(preview)
        for offset in range(0, len(previews), 3):
            group = previews[offset:offset + 3]
            canvas = Image.new("RGB", (1280, sum(p.height for p in group) + 20 * (len(group) - 1)), "white")
            y = 0
            for preview in group:
                canvas.paste(preview, (0, y))
                y += preview.height + 20
            canvas.save(output / f"review_{offset // 3 + 1:02d}.jpg", quality=95)
        if selection:
            selected = []
            by_stem = {r["stem"]: r for r in records}
            for entry in spec:
                record = dict(by_stem[entry["stem"]], visual_review=entry["description"])
                selected.append(record)
                fig = make_figure(record)
                fig.savefig(output / f"{record['stem']}.pdf")
                plt.close(fig)
                caption = (f"{record['scene']}, MemCam 180-second rollout: Unbounded and KEEPSAKE B32. "
                           f"Identical frames {record['frames']} are used across methods. "
                           + entry["description"] + " Full frames and original colors are shown. ")
                if "position_distance_m" in record:
                    caption += (f"Endpoint cameras differ by {record['position_distance_m']:.3f} m and "
                                f"{record['rotation_difference_deg']:.2f} degrees in the saved pose analysis. "
                                "This is a near-view return, not an identical-pose or GT reconstruction comparison.")
                else:
                    caption += "GT shows the wall-and-column view before and after the intervening view; numerical pose proximity is not established."
                (output / f"{record['stem']}.caption.txt").write_text(caption + "\n")
            with PdfPages(output / "selected_revisits.pdf") as document:
                for record in selected:
                    fig = make_figure(record)
                    document.savefig(fig)
                    plt.close(fig)
            save_json(output / "selected_revisits.json", selected)
    save_json(output / "manifest.json", dict(
        records=records, event_source=str(events_path), event_sha256=digest(events_path),
        bundle_manifest_sha256=digest(folder / "manifest.json"),
        renderer_sha256=digest(Path(__file__)),
        selection="All saved near-pose events reviewed; final examples chosen by visual inspection, not a random sample.",
        middle_frame="Pose-event examples use arithmetic midpoints; visual-GT examples use specified exported GT times. Identical indices across methods; no measured disappearance duration.",
        display="Full original frames and colors. No enhancement, cropping, image synthesis or alignment warping.",
        limits="Near-matched pose endpoints from saved analysis, not identical camera poses or a GT-fidelity score. Original pose files are not local."))
    print(f"Saved {len(records)} revisit comparisons: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads/memcam_edit_180s")
    parser.add_argument("--events", type=Path, default=Path.home() / "Downloads/context_180s/revisit_consistency_b32_oracle30/tables/revisit_events.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/revisit_comparisons_180s")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    render(args.input, args.events, args.output, args.selection)


if __name__ == "__main__":
    main()
