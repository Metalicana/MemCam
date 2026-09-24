"""Render fixed-time GT/Unbounded/KEEPSAKE comparisons from the downloaded bundle."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.compare_local_rollouts import probe
from paper.replace_method_frame_assets import extract_frames


METHODS = (("gt", "GT", "#454545"), ("unbounded", "Unbounded", "#B45C5C"),
           ("keepsake_b32", "KEEPSAKE", "#277A60"))
TIMES = (1, 60, 120, 180)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bundle_path(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside downloaded bundle: {name}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def load_items(folder, expected_videos=15):
    items = json.loads((folder / "manifest.json").read_text())
    if len(items) != expected_videos or len({p["output_prefix"] for p in items}) != len(items):
        raise ValueError("Expected the complete distinct matched cohort")
    for item in items:
        prefix = item["output_prefix"]
        if (Path(prefix).name != prefix or int(item["duration_sec"]) != 180
                or int(item["num_frames"]) != 5397 or float(item["fps"]) != 30
                or int(item["start_frame"]) < 0):
            raise ValueError(f"Wrong 180-second video identity: {prefix}")
        for method in ("unbounded", "keepsake_b32"):
            name = item["videos"][method]
            if name != f"{method}/{prefix}custom.mp4":
                raise ValueError(f"Mismatched policy video: {name}")
            bundle_path(folder, name)
        if tuple(p["requested_sec"] for p in item["samples"]) != TIMES:
            raise ValueError(f"Expected the four fixed times: {prefix}")
        for sample in item["samples"]:
            frame = min(round(sample["requested_sec"] * 30), 5396)
            if (sample["frame_index"] != frame
                    or not math.isclose(sample["actual_sec"], frame / 30, abs_tol=1e-9)
                    or sample["gt_file"] != f"gt/{prefix}frame_{frame:05d}.png"):
                raise ValueError(f"GT/time alignment mismatch: {prefix}")
            with Image.open(bundle_path(folder, sample["gt_file"])) as image:
                image.verify()
    return items


def decode_cached(video, indices, folder):
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha256=digest(video), frames=indices,
                    decoder_sha256=digest(ROOT / "paper/replace_method_frame_assets.py"))
    paths = {i: folder / f"frame_{i:04d}.png" for i in indices}
    receipt_path = folder / "source.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if (receipt.get("identity") == identity
            and all(p.is_file() and receipt.get("frames", {}).get(p.name) == digest(p)
                    for p in paths.values())):
        return paths, receipt
    # Fresh staging prevents an old decoded PNG from masking a truncated source video.
    with tempfile.TemporaryDirectory(dir=folder, prefix="decode_") as temporary:
        extracted = extract_frames(video, indices, Path(temporary))
        for index, path in extracted.items():
            path.replace(paths[index])
    receipt = dict(identity=identity, frames={p.name: digest(p) for p in paths.values()})
    save_json(receipt_path, receipt)
    return paths, receipt


def make_figure(record):
    # Layout dimensions are pixels at export DPI; source image aspect ratios are preserved.
    dpi, tile_width, tile_height = 200, 640, 360
    left, right, top, bottom, gap = 230, 12, 110, 12, 10
    width = left + 4 * tile_width + 3 * gap + right
    height = top + 3 * tile_height + 2 * gap + bottom
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    fig.text(left / width, 1 - 18 / height, record["scene"], fontsize=14,
             weight="bold", va="top", color="#242424")
    for column, sample in enumerate(record["samples"]):
        x = left + column * (tile_width + gap)
        seconds = sample["actual_sec"]
        label = f"{seconds:g} s" if seconds == int(seconds) else f"{seconds:.1f} s"
        fig.text((x + tile_width / 2) / width, 1 - 79 / height, label,
                 ha="center", va="center", fontsize=13, color="#242424")
        for row, (key, label, color) in enumerate(METHODS):
            y = height - top - (row + 1) * tile_height - row * gap
            ax = fig.add_axes([x / width, y / height, tile_width / width, tile_height / height])
            with Image.open(record["frame_paths"][key][str(sample["frame_index"])]).convert("RGB") as image:
                ax.imshow(image, interpolation="none", aspect="equal")
            ax.axis("off")
            if column == 0:
                fig.text((left - 17) / width, (y + tile_height / 2) / height,
                         label, ha="right", va="center", fontsize=12, color=color,
                         weight="bold" if key == "keepsake_b32" else "normal")
    return fig


def render(folder, output, expected_videos=15):
    items = load_items(folder, expected_videos)
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = digest(folder / "manifest.json")
    metadata = {}
    for item in items:
        infos = {key: probe(bundle_path(folder, item["videos"][key]))
                 for key in ("unbounded", "keepsake_b32")}
        info = infos["unbounded"]
        if (infos["keepsake_b32"] != info or info["frames"] != item["num_frames"]
                or info["fps"] != float(item["fps"])):
            raise ValueError(f"Generated video count/FPS/geometry mismatch: {item['output_prefix']}")
        metadata[item["output_prefix"]] = info
    print(f"Validated {len(items)} video pairs and four GT stills per scene", flush=True)
    records, previews = [], []
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {item['scene']}: fixed frames 30, 1800, 3600, 5396", flush=True)
        stem = f"comparison_{index:02d}_{item['scene']}"
        indices = [s["frame_index"] for s in item["samples"]]
        record = dict(scene=item["scene"], stem=stem, start_frame=item["start_frame"],
                      budget=32, metadata=metadata[item["output_prefix"]],
                      samples=item["samples"], frame_paths={}, source_videos={}, source_sha256={})
        for key in ("unbounded", "keepsake_b32"):
            source = bundle_path(folder, item["videos"][key])
            paths, receipt = decode_cached(source, indices, output / "frames" / stem / key)
            record["frame_paths"][key] = {str(i): str(p.resolve()) for i, p in paths.items()}
            record["source_videos"][key] = str(source)
            record["source_sha256"][key] = receipt["identity"]["source_sha256"]
        record["frame_paths"]["gt"] = {str(s["frame_index"]): str(bundle_path(folder, s["gt_file"]))
                                           for s in item["samples"]}
        record["gt_sha256"] = {i: digest(Path(p)) for i, p in record["frame_paths"]["gt"].items()}
        record["gt_dataset_indices"] = [int(item["start_frame"]) + i for i in indices]
        record["manifest_sha256"] = manifest_hash
        record["display"] = "Full frames, native aspect ratios; no cropping, gamma, brightness or other retouching."
        record["selection"] = "All manifest scenes at fixed requested times; no timestamp search or image-quality filtering."
        record["limits"] = "GT mapping follows the downloaded manifest; original dataset and generating environments not re-audited."
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
            fig = make_figure(record)
            fig.savefig(output / f"{stem}.png", dpi=200, facecolor="white")
            fig.savefig(output / f"{stem}.pdf", facecolor="white")
            plt.close(fig)
        caption = (f"{item['scene']}: ground truth, Unbounded MemCam and KEEPSAKE B32 at "
                   "1, 60, 120 and 179.87 seconds of the nominal 180-second rollout. "
                   "Columns use identical zero-based frame indices (30, 1800, 3600, 5396); "
                   "GT uses dataset start_frame plus the corresponding index. Full frames are shown "
                   "without enhancement or cropping; source aspect ratios are preserved. "
                   "Fixed-time qualitative comparison, not evidence by itself of causal error propagation.\n")
        (output / f"{stem}.caption.txt").write_text(caption)
        save_json(output / f"{stem}.json", record)
        records.append(record)
        with Image.open(output / f"{stem}.png") as image:
            preview = image.convert("RGB")
            preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            previews.append(preview)
    for start in range(0, len(previews), 3):
        group = previews[start:start + 3]
        canvas = Image.new("RGB", (max(im.width for im in group), sum(im.height for im in group) + 24 * (len(group) - 1)), "white")
        y = 0
        for preview in group:
            canvas.paste(preview, (0, y))
            y += preview.height + 24
        canvas.save(output / f"review_{start // 3 + 1:02d}.jpg", quality=95)
    save_json(output / "manifest.json", records)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}), \
            PdfPages(output / "all_scenes.pdf") as document:
        for record in records:
            fig = make_figure(record)
            document.savefig(fig, facecolor="white")
            plt.close(fig)
    print(f"Saved {len(records)} matched PNG/PDF figures and review sheets: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads/memcam_edit_180s")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/rollout_comparisons_180s")
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
