import base64
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from paper import replace_method_frame_assets as method


class MethodFrameTests(unittest.TestCase):
    def test_bank_reconstruction_uses_real_evictions(self):
        events = [dict(event="memory_eviction", section_idx=s, evicted_memory_frame=f,
                       stored_memory_size=32)
                  for s, frames in ((0, range(31, 76)), (1, range(76, 152))) for f in frames]
        old, new, kept, evicted = method.update_bank(events, 1)
        self.assertEqual(old, set(range(31)) | {76})
        self.assertEqual(new, set(range(77, 153)))
        self.assertEqual(kept, set(range(31)) | {152})
        self.assertEqual(len(old | new), 108)
        for bad in (events[:-1], events + [events[-1]], events + [dict(events[-1], evicted_memory_frame=999)]):
            with self.assertRaises(ValueError):
                method.update_bank(bad, 1)

    def test_roles_reject_wrong_kept_and_missing_substitute(self):
        old = {f for f in method.CELL_FRAMES.values() if f <= 912}
        new = set(method.CELL_FRAMES.values()) - old
        evicted = {f: {"eviction_nearest_covisible_frame": 988} for f in (41, 42, 987)}
        kept = (old | new) - evicted.keys()
        method.validate_roles(old, new, kept, evicted)
        with self.assertRaisesRegex(ValueError, "not retained"):
            method.validate_roles(old, new, kept - {935}, evicted)
        evicted[987]["eviction_nearest_covisible_frame"] = 999
        with self.assertRaisesRegex(ValueError, "substitute"):
            method.validate_roles(old, new, kept, evicted)

    def test_all_images_replaced_and_scores_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = ET.Element("mxfile")
            page = ET.SubElement(document, "diagram")
            model = ET.SubElement(page, "mxGraphModel")
            cells = ET.SubElement(model, "root")
            for key in range(309):
                cell = ET.SubElement(cells, "mxCell", id=f"cell-{key}", parent="cell-1", value="",
                    style="shape=image;image=placeholder;" if key in method.CELL_FRAMES else "shape=text;")
                ET.SubElement(cell, "mxGeometry", x="0", y="0", width="46", height="46", attrib={"as": "geometry"})
            template = root / "input.drawio"
            ET.ElementTree(document).write(template)
            assets = {}
            for n in set(method.CELL_FRAMES.values()):
                path = root / f"frame_{n}.png"
                Image.new("RGB", (64, 35), (n % 256, 20, 30)).save(path)
                assets[n] = path
            output = root / "output.drawio"
            evictions = {f: {"eviction_score": v} for f, v in ((41, .03875), (42, .03875), (987, .03522))}
            method.rewrite_diagram(template, output, assets, evictions)
            result = list(ET.parse(output).iter("mxCell"))
            images = [c for c in result if c.get("data-frame-index") is not None]
            self.assertEqual(len(images), 41)
            for c in images:
                n = int(c.get("data-frame-index"))
                encoded = next(t[6:] for t in c.get("style").split(";") if t.startswith("image="))
                with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as image:
                    self.assertEqual(image.getpixel((0, 0)), (n % 256, 20, 30))
                    self.assertEqual(image.size, (64, 35))
                label = next((r for r in result if r.get("id") == "real-frame-label-" + c.get("id")), None)
                if label is not None:
                    g, l = c.find("mxGeometry"), label.find("mxGeometry")
                    self.assertGreaterEqual(float(l.get("y")), float(g.get("y")) + float(g.get("height")))
            self.assertEqual(len(result), len({c.get("id") for c in result}))
            values = [c.get("value") for c in result]
            self.assertIn("protected", values)
            self.assertIn("0.0387", values)
            self.assertIn("0.0352", values)
            self.assertNotIn("0.92", values)
            self.assertTrue(any(c.get("id") == "memory-feedback" for c in result))

    @unittest.skipUnless(shutil.which("ffmpeg"), "Requires FFmpeg")
    def test_decoder_preserves_zero_based_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                Image.new("RGB", (32, 18), (i * 40, 20, 30)).save(root / f"input_{i:03d}.png")
            video = root / "test.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-framerate", "30", "-i", str(root / "input_%03d.png"),
                            "-c:v", "ffv1", str(video)], check=True)
            assets = method.extract_frames(video, [3, 1], root / "assets")
            self.assertEqual(list(assets), [1, 3])
            for n, path in assets.items():
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), (n * 40, 20, 30))
            with self.assertRaises(ValueError):
                method.extract_frames(video, [0, 5], root / "missing")


if __name__ == "__main__":
    unittest.main()
