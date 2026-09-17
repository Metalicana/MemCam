"""Remove historical-GT columns from the existing five-column raster figure.

Only layout and text headers change; retained image and metric pixels are copied.
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def fitting_font(draw, text, size, width, bold=False):
    mac_font = Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf")
    name = str(mac_font) if mac_font.exists() else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    for point_size in range(size, 8, -1):
        font = ImageFont.truetype(name, point_size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= width:
            return font
    return font


def simplify(source, destination):
    with Image.open(source) as original:
        original = original.convert("RGB")
    tile_width, tile_height, title_height = 292, 260, 96
    if original.width != 5 * tile_width or (original.height - title_height) % tile_height:
        raise ValueError("Expected the original five-column, 292px-per-column figure")
    output = Image.new("RGB", (3 * tile_width, original.height), (233, 235, 238))
    for new, old in enumerate((0, 1, 3)):
        output.paste(original.crop((old * tile_width, title_height,
                                    (old + 1) * tile_width, original.height)),
                     (new * tile_width, title_height))
    draw = ImageDraw.Draw(output)
    for text, y, size, bold in (
        ("Extreme common-source retrieval examples", 27, 25, True),
        ("Both selectors read the same unbounded rollout; PSNR/SSIM use exact-index GT", 66, 14, False),
    ):
        draw.text((output.width // 2, y), text, anchor="mm", fill=(22, 22, 24),
                  font=fitting_font(draw, text, size, output.width - 30, bold=bold))
    rows = (original.height - title_height) // tile_height
    for row in range(rows):
        y = title_height + row * tile_height
        x = 2 * tile_width
        draw.rectangle((x + 5, y + 5, x + tile_width - 6, y + 41), fill=(248, 248, 248))
        draw.text((x + tile_width // 2, y + 21), "Ours", anchor="mm", fill=(22, 22, 24),
                  font=fitting_font(draw, "Ours", 16, tile_width - 20, bold=True))
        for new, old in enumerate((0, 1, 3)):
            # Check exact preservation below each header, including all scores.
            expected = original.crop((old * tile_width, y + 42, (old + 1) * tile_width, y + tile_height))
            actual = output.crop((new * tile_width, y + 42, (new + 1) * tile_width, y + tile_height))
            assert actual.tobytes() == expected.tobytes()
    output.save(destination)
    print(f"Wrote {destination}: {output.width}x{output.height}; all {rows * 3} retained image/score regions pixel-identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    simplify(args.source, args.destination)
