import base64
from pathlib import Path
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from paper import build_method_figure as method


class EnlargedMethodTests(unittest.TestCase):
    def test_preserves_four_stage_structure_and_frame_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            doc = ET.Element("mxfile")
            page = ET.SubElement(doc, "diagram")
            model = ET.SubElement(page, "mxGraphModel")
            root = ET.SubElement(model, "root")
            edge_ids = (*range(147, 162), 119, 120, 190, 193, 195, 197,
                        221, 232, 238, 244, 245, 251, 252, 258, 259, 299, 300, 301)
            for n in range(309):
                attrs = dict(id=method.PREFIX + str(n), parent="0", value="",
                             style="fontSize=18;" + ("image=placeholder;" if n in method.FRAMES else ""))
                if n in edge_ids:
                    attrs.update(edge="1", source=method.PREFIX + "162", target=method.PREFIX + "164")
                else:
                    attrs["vertex"] = "1"
                cell = ET.SubElement(root, "mxCell", **attrs)
                ET.SubElement(cell, "mxGeometry", x="0", y="0", width="50", height="50", attrib={"as": "geometry"})
            for name in ("protected-graph-endpoint", "protected-archive-endpoint"):
                cell = ET.SubElement(root, "mxCell", id=name, parent="0", vertex="1")
                ET.SubElement(cell, "mxGeometry", attrib={"as": "geometry"})
            feedback = ET.SubElement(root, "mxCell", id="memory-feedback", parent="0", edge="1")
            ET.SubElement(feedback, "mxGeometry", relative="1", attrib={"as": "geometry"})
            template = path / "before.drawio"
            ET.ElementTree(doc).write(template)
            frames = {}
            for n in set(method.FRAMES.values()):
                frames[n] = path / f"frame{n}.png"
                Image.new("RGB", (640, 352), (n, 22, 33)).save(frames[n])
            out = path / "after.drawio"
            evicted = {n: {"eviction_score": s} for n, s in ((146, .025112), (183, .024982), (227, .019780))}
            method.rewrite(template, out, frames, evicted)
            cells = {c.get("id"): c for c in ET.parse(out).iter("mxCell")}
            self.assertNotIn("memory-feedback", cells)
            self.assertNotIn(method.PREFIX + "305", cells)
            for n in edge_ids:
                self.assertEqual(cells[method.PREFIX + str(n)].get("source"), method.PREFIX + "162")
                self.assertEqual(cells[method.PREFIX + str(n)].get("target"), method.PREFIX + "164")
            for n in (83, 138, 209, 263):
                g = cells[method.PREFIX + str(n)].find("mxGeometry")
                self.assertEqual(float(g.get("height")), method.PANEL_HEIGHT)
                self.assertEqual(float(g.get("y")), method.PANEL_TOP)
                self.assertLessEqual(float(g.get("height")), 650)
            for n in range(162, 184, 2):
                self.assertEqual(cells[method.PREFIX + str(n)].find("mxGeometry").get("width"), "72")
            images = [c for c in cells.values() if c.get("data-frame-index")]
            self.assertEqual(len(images), 41)
            for c in images:
                payload = re.search(r"image=data:image/png,([^;]+)", c.get("style"))[1]
                frame = int(c.get("data-frame-index"))
                self.assertEqual(base64.b64decode(payload), frames[frame].read_bytes())
            self.assertEqual(cells[method.PREFIX + "92"].find("mxGeometry").get("width"), "100")
            self.assertEqual(cells[method.PREFIX + "217"].find("mxGeometry").get("width"), "54")
            self.assertEqual(cells[method.PREFIX + "272"].find("mxGeometry").get("width"), "156")
            footer = cells[method.PREFIX + "307"].find("mxGeometry")
            layout_height = float(footer.get("y")) + float(footer.get("height")) - 6
            self.assertGreaterEqual(1808 / layout_height, 1.65)
            self.assertLessEqual(1808 / layout_height, 1.80)
            self.assertEqual(cells[method.PREFIX + "243"].get("value"), "0.0251")
            self.assertEqual(cells[method.PREFIX + "250"].get("value"), "0.0250")
            self.assertEqual(cells[method.PREFIX + "257"].get("value"), "0.0198")


if __name__ == "__main__":
    unittest.main()
