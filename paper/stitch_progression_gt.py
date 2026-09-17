"""Stack an existing progression's GT and policy strips without changing pixels."""

import argparse
import json
from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path, help="Progression filename without extension")
    args = parser.parse_args()
    stem = args.stem
    case = json.loads(stem.with_suffix(".json").read_text())
    labels = ("Ground truth", "Unbounded", "FIFO", "Ours")
    rows = []
    for suffix in ("gt_check", "unbounded", "fifo", "ours"):
        with Image.open(stem.with_name(f"{stem.name}_{suffix}_bare.png")) as image:
            rows.append(image.convert("RGB"))
    if len({row.size for row in rows}) != 1:
        raise ValueError("All strips must have identical dimensions")
    width, height = rows[0].size
    count, gutter = len(case["frames"]), 8
    tile_width, remainder = divmod(width - (count - 1) * gutter, count)
    if remainder:
        raise ValueError("Strip width does not match the original eight-pixel gutters")
    margin, header, gap = 150, 42, 14
    canvas = Image.new("RGB", (margin + width, header + 4 * height + 3 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 19)
    for col, frame in enumerate(case["frames"]):
        draw.text((margin + col * (tile_width + gutter) + tile_width // 2, header // 2),
                  f"{frame / case['fps']:.1f} s", font=font, fill="black", anchor="mm")
    for index, (label, row) in enumerate(zip(labels, rows)):
        y = header + index * (height + gap)
        canvas.paste(row, (margin, y))
        draw.text((margin - 12, y + height // 2), label, font=font, fill="black", anchor="rm")
    output = stem.with_name(stem.name + "_with_gt")
    canvas.save(output.with_suffix(".png"))
    canvas.save(output.with_suffix(".pdf"), resolution=150)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
