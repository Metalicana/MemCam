"""Review matched local videos and export unretouched five-frame comparison strips."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.replace_method_frame_assets import extract_frames


METHODS = {"Unbounded": "Unbounded", "KEEPSAKE": "KEEPSAKE_B32", "FIFO": "FIFO_B32"}
COLORS = {"Unbounded": "#42464B", "KEEPSAKE": "#277A60", "FIFO": "#BF665D"}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def font(size, bold=False):
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(path),
    ], text=True))["streams"][0]
    return dict(width=int(data["width"]), height=int(data["height"]),
                frames=int(data["nb_frames"]), fps=float(Fraction(data["avg_frame_rate"])))


def title(system, name):
    if system == "MemCam":
        return name.removeprefix("seed0_").rsplit("_", 3)[0]
    return "Sequence " + str(int(name.split("batch")[1].split("_")[0]))


def sources(folder):
    names = {method: {p.name for p in (folder / directory).glob("*.mp4")}
             for method, directory in METHODS.items()}
    if not names["Unbounded"] or any(s != names["Unbounded"] for s in names.values()):
        raise ValueError(f"Incomplete matching policy folders: {folder}")
    return [{method: folder / directory / name for method, directory in METHODS.items()}
            for name in sorted(names["Unbounded"])]


def decode(source, frames, output):
    expected = {i: output / f"frame_{i:04d}.png" for i in frames}
    identity = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "frames": frames}
    manifest = output / "source.json"
    if (not all(p.is_file() for p in expected.values()) or not manifest.is_file()
            or json.loads(manifest.read_text()) != identity):
        extracted = extract_frames(source, frames, output)
        save_json(manifest, identity)
        return extracted
    return expected


def review_card(record, frames, cell_width=216):
    aspect = record["metadata"]["height"] / record["metadata"]["width"]
    cell_height = round(cell_width * aspect)
    left, gap, top = 112, 3, 50
    canvas = Image.new("RGB", (left + len(frames) * (cell_width + gap),
                               top + 3 * (cell_height + gap) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 2), f"{record['system']} | {record['title']}", font=font(20, True), fill="#151515")
    for column, index in enumerate(frames):
        draw.text((left + column * (cell_width + gap), 28), f"Frame {index}", font=font(14), fill="#555555")
    for row, method in enumerate(METHODS):
        y = top + row * (cell_height + gap)
        draw.text((6, y + cell_height // 2 - 9), method, font=font(16, True), fill=COLORS[method])
        for column, index in enumerate(frames):
            image = Image.open(record["frame_paths"][method][str(index)]).convert("RGB")
            canvas.paste(image.resize((cell_width, cell_height), Image.Resampling.LANCZOS),
                         (left + column * (cell_width + gap), y))
    return canvas


def scan(downloads, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for system, directory in (("MemCam", "memcam_editing"), ("WorldMem", "worldmem_editing")):
        cards = []
        for number, paths in enumerate(sources(downloads / directory), 1):
            name = paths["Unbounded"].name
            print(f"{system} [{number}/15]: {name}", flush=True)
            infos = {method: probe(path) for method, path in paths.items()}
            info = infos["Unbounded"]
            if any(data != info for data in infos.values()):
                raise ValueError(f"Frame geometry/count/FPS differ: {name}: {infos}")
            # Fixed overview positions for every video, independent of visual quality.
            frames = np.linspace(0, info["frames"] - 1, 9).round().astype(int).tolist()
            record = dict(system=system, filename=name, title=title(system, name), metadata=info,
                          sampled_frames=frames, source_videos={k: str(v) for k, v in paths.items()},
                          source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                          frame_paths={})
            for method, path in paths.items():
                folder = output / "review_frames" / system / Path(name).stem / method
                extracted = decode(path, frames, folder)
                record["frame_paths"][method] = {str(i): str(p) for i, p in extracted.items()}
            card = review_card(record, frames[::2])
            path = output / f"review_{system.lower()}_{number:02d}.png"
            card.save(path)
            record["review_png"] = str(path)
            cards.append(card)
            records.append(record)
        for page in range(0, len(cards), 3):
            selected = cards[page:page+3]
            atlas = Image.new("RGB", (max(im.width for im in selected), sum(im.height for im in selected)), "white")
            y = 0
            for im in selected:
                atlas.paste(im, (0, y))
                y += im.height
            atlas.save(output / f"atlas_{system.lower()}_{page // 3 + 1:02d}.jpg", quality=95)
    save_json(output / "review_manifest.json", records)
    print(f"Review: {output}", flush=True)


def render(selected_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 12})
    choices = json.loads(selected_path.read_text())
    records = json.loads((output / "review_manifest.json").read_text())
    for choice in choices:
        record = next(r for r in records if r["system"] == choice["system"] and r["filename"] == choice["filename"])
        frames = choice["frames"]
        if len(frames) != 5 or sorted(set(frames)) != frames or frames[0] < 0 or frames[-1] >= record["metadata"]["frames"]:
            raise ValueError("Expected five increasing valid frame indices")
        paths = {}
        for method, path in record["source_videos"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record["source_sha256"][method]:
                raise ValueError(f"Source changed since visual review: {path}")
            folder = output / "selected_frames" / choice["stem"] / method
            paths[method] = decode(Path(path), frames, folder)
        aspect = record["metadata"]["height"] / record["metadata"]["width"]
        fig = plt.figure(figsize=(16, 3 * 2.9 * aspect + .72), facecolor="white")
        left, right, top, bottom, gap = .086, .995, .875, .015, .003
        width = (right - left - 4 * gap) / 5
        height = (top - bottom - 2 * .009) / 3
        for row, method in enumerate(METHODS):
            y = top - (row + 1) * height - row * .009
            fig.text(left - .009, y + height / 2, method, ha="right", va="center",
                     fontsize=11.5, weight="bold" if method == "KEEPSAKE" else "normal", color=COLORS[method])
            for column, index in enumerate(frames):
                x = left + column * (width + gap)
                ax = fig.add_axes([x, y, width, height])
                ax.imshow(Image.open(paths[method][index]), interpolation="none")
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"Frame {index}", fontsize=12, pad=5)
        fig.text(.012, .984, f"{record['system']} | {choice.get('display_title', record['title'])}",
                 va="top", fontsize=17, weight="bold")
        stem = output / choice["stem"]
        fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        save_json(stem.with_suffix(".json"), dict(
            **choice, source_videos=record["source_videos"], source_sha256=record["source_sha256"],
            metadata=record["metadata"], frame_indexing="zero-based; identical indices across policies",
            frame_paths={k: {str(i): str(p) for i, p in v.items()} for k, v in paths.items()},
            selection="Manually selected qualitative illustration after viewing all 15 matched triplets per system.",
            display="Original decoded pixels; full-frame aspect preserved; no gamma, contrast, crop or retouching.",
            limits="No local GT in these input folders; visible consistency/artifact differences are not proof of correct GT content or causal snowballing."))
        print(f"Rendered {stem}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/rollout_comparisons_20260923")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.selection:
        render(args.selection, args.output)
    else:
        scan(args.downloads, args.output)


if __name__ == "__main__":
    main()
