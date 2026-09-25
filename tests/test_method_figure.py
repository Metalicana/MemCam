import base64
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from paper import build_method_figure as method


def example_record():
    return {227: {"eviction_covisible_observers": 31,
                  "eviction_max_covisibility": 0.9833801374815905,
                  "eviction_nearest_covisible_frame": 228,
                  "eviction_score": 0.01977996562960238}}


class MethodFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name)
        doc = ET.Element("mxfile")
        page = ET.SubElement(doc, "diagram")
        model = ET.SubElement(page, "mxGraphModel")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        for n in range(309):
            attrs = dict(id=method.PREFIX + str(n), parent="0", value="",
                         style="fontSize=18;" + ("image=placeholder;" if n in method.FRAMES else ""))
            attrs["edge" if n in (5, 6, *range(74, 81)) else "vertex"] = "1"
            c = ET.SubElement(root, "mxCell", **attrs)
            ET.SubElement(c, "mxGeometry", x="0", y="0", width="50", height="50", attrib={"as": "geometry"})
        for name in ("protected-graph-endpoint", "protected-archive-endpoint"):
            c = ET.SubElement(root, "mxCell", id=name, parent=method.PREFIX + "138", vertex="1")
            ET.SubElement(c, "mxGeometry", attrib={"as": "geometry"})
            c = ET.SubElement(root, "mxCell", id=name + "-child", parent=name, vertex="1")
            ET.SubElement(c, "mxGeometry", attrib={"as": "geometry"})
        c = ET.SubElement(root, "mxCell", id="memory-feedback", parent="0", edge="1")
        ET.SubElement(c, "mxGeometry", relative="1", attrib={"as": "geometry"})
        template = path / "before.drawio"
        ET.ElementTree(doc).write(template)
        self.frames = {}
        for n in set(method.FRAMES.values()):
            self.frames[n] = path / f"frame{n}.png"
            Image.new("RGB", (640, 352), (n, 22, 33)).save(self.frames[n])
        output = path / "after.drawio"
        method.rewrite(template, output, self.frames, example_record())
        self.rendered = list(ET.parse(output).iter("mxCell"))
        self.cells = {c.get("id"): c for c in self.rendered}

    def cell(self, n):
        return self.cells[method.PREFIX + str(n) if isinstance(n, int) else n]

    def test_backup_creates_archive_parents_and_keeps_first_version(self):
        root = Path(self.tmp.name)
        output = root / "after.drawio"
        original = output.read_bytes()
        args = SimpleNamespace(traces=root, videos=root, output=output, template=root / "before.drawio")
        backup = root / "archive/method_previous/rejected_three_stage.drawio.xml"
        with patch.object(method, "prepare", return_value=(set(), set(), set(), example_record())), \
             patch.object(method, "extract_frames", return_value=self.frames), \
             patch.object(method, "rewrite", side_effect=RuntimeError("stop after backup")):
            for contents in (original, b"later figure"):
                output.write_bytes(contents)
                with self.assertRaisesRegex(RuntimeError, "stop after backup"):
                    method.build(args)
                self.assertEqual(backup.read_bytes(), original)

    def test_narrow_banks_dominant_graph_and_integrated_eviction(self):
        graph_width = float(self.cell(138).find("mxGeometry").get("width"))
        self.assertGreater(graph_width / 1808, .60)
        for n in (83, 263):
            g = self.cell(n).find("mxGeometry")
            self.assertLess(float(g.get("width")), 250)
            self.assertEqual(float(g.get("height")), method.PANEL_HEIGHT)
        self.assertEqual(self.cell(209).get("parent"), method.PREFIX + "138")
        eviction = self.cell(209).find("mxGeometry")
        self.assertLess(float(eviction.get("width")), graph_width / 4)
        self.assertLess(float(eviction.get("height")), method.PANEL_HEIGHT / 2)
        self.assertLess(float(eviction.get("x")) + float(eviction.get("width")), graph_width)
        self.assertEqual(eviction.get("y"), self.cell("priority-band").find("mxGeometry").get("y"))
        self.assertIn("priority-to-eviction", self.cells)
        self.assertNotIn("graph-to-eviction", self.cells)
        for n in (83, 138, 263):
            self.assertEqual(float(self.cell(n).find("mxGeometry").get("y")), method.PANEL_TOP)

    def test_images_preserve_source_pixels_and_native_aspect(self):
        images = [c for c in self.rendered if c.get("data-frame-index") is not None]
        self.assertEqual(len(images), 21)
        for c in images:
            payload = re.search(r"image=data:image/png,([^;]+)", c.get("style"))[1]
            frame = int(c.get("data-frame-index"))
            self.assertEqual(base64.b64decode(payload), self.frames[frame].read_bytes())
            if not c.get("id").startswith(method.PREFIX):
                g = c.find("mxGeometry")
                self.assertAlmostEqual(float(g.get("width")) / float(g.get("height")), 640 / 352)
        retained = {int(c.get("data-frame-index")) for c in images if c.get("data-role") == "retained"}
        self.assertEqual(retained, {0, 125, 228})

    def test_graph_construction_and_scoring_are_explicit(self):
        formula = self.cell("combined-affinity").get("value")
        self.assertIn("&alpha;<i>P</i>", formula)
        self.assertIn("(1 &minus; &alpha;)<i>A</i>", formula)
        self.assertIn("&ge;", self.cell("edge-rule").get("value"))
        self.assertIn("i</i> &ne;", self.cell("edge-rule").get("value"))
        self.assertIn("&tau;", self.cell("edge-rule").get("value"))
        self.assertIn("Appearance", self.cell("appearance-affinity").get("value"))
        self.assertIn("Pose", self.cell("pose-affinity").get("value"))
        for c in self.rendered:
            self.assertNotIn("0.65", c.get("value", ""))
            self.assertNotIn("0.35", c.get("value", ""))
            self.assertNotIn("DINO", c.get("value", ""))
        self.assertIn("31", self.cell("neighbor-value").get("value"))
        self.assertIn("0.9834", self.cell("strongest-value").get("value"))
        self.assertIn("0.0198", self.cell("priority-value").get("value"))
        for term in ("min(c<sub>i</sub>/3, 1)", "0.5/(c<sub>i</sub> + 1)", "0.25(1 &minus;"):
            self.assertIn(term, self.cell("priority-formula").get("value"))
        edge = self.cell("logged-closest-link")
        self.assertEqual(edge.get("data-frame-pair"), "227,228")
        self.assertIn("endArrow=none;", edge.get("style"))
        self.assertAlmostEqual(float(edge.get("data-affinity")), example_record()[227]["eviction_max_covisibility"])
        for i in range(1, 4):
            self.assertEqual(self.cell(f"schematic-link-{i}").get("data-link-origin"), "schematic")

    def test_centered_unnumbered_headers_and_stage_colors(self):
        for n, color in ((83, method.RED), (138, method.BLUE), (263, method.GREEN)):
            self.assertIn(f"strokeColor={color};", self.cell(n).get("style"))
            panel = self.cell(n).find("mxGeometry")
            header = self.cell(method.PREFIX + str(n) + "-title")
            g = header.find("mxGeometry")
            self.assertEqual(float(g.get("x")) + float(g.get("width")) / 2,
                             float(panel.get("width")) / 2)
            self.assertIn("align=center;", header.get("style"))
            self.assertNotIn(method.PREFIX + str(n) + "-number", self.cells)
        self.assertNotIn("evict-number", self.cells)
        for n in (303, 306):
            self.assertIn("align=center;", self.cell(n).get("style"))

    def test_preserves_generation_inputs_spacing_and_visible_expansion(self):
        flow = (8, 27, 28, 38, 52, 53)
        blocks = [self.cell(n).find("mxGeometry") for n in flow]
        for left, right in zip(blocks, blocks[1:]):
            self.assertEqual(float(right.get("x")) - float(left.get("x")) - float(left.get("width")), 40)
            self.assertEqual(float(left.get("y")) + float(left.get("height")) / 2, 147)
        for n, source, target in zip(range(74, 79), flow, flow[1:]):
            self.assertEqual(self.cell(n).get("source"), method.PREFIX + str(source))
            self.assertEqual(self.cell(n).get("target"), method.PREFIX + str(target))
        for n in (5, 6):
            style = self.cell(n).get("style")
            for expected in ("strokeColor=#B24C55;", "strokeWidth=3;", "dashed=1;", "endArrow=none;"):
                self.assertIn(expected, style)
        for name in ("caption-input", "noise-input"):
            self.assertEqual(self.cell(name + "-arrow").get("source"), name)
            self.assertEqual(self.cell(name + "-arrow").get("target"), method.PREFIX + "28")

    def test_no_orphans_subtitles_or_protection_annotations(self):
        self.assertEqual(len(self.cells), len(self.rendered))
        for c in self.rendered:
            for key in ("parent", "source", "target"):
                if c.get(key):
                    self.assertIn(c.get(key), self.cells)
            self.assertNotIn("protected", c.get("value", "").lower())
            self.assertNotIn("unlocked", c.get("value", "").lower())
            self.assertNotIn("protected", c.get("id", ""))
        self.assertNotIn("memory-feedback", self.cells)
        for n in (3, 88, 143, 214, 268, 304, 305, 307):
            self.assertNotIn(method.PREFIX + str(n), self.cells)

    def test_scoring_example_validation(self):
        example = method.scoring_example(example_record())
        self.assertAlmostEqual(example["utility"], .01977996562960238)
        for key, value in (("eviction_score", 0.1), ("eviction_score", float("nan")),
                           ("eviction_covisible_observers", -1),
                           ("eviction_max_covisibility", 1.1),
                           ("eviction_nearest_covisible_frame", 145)):
            record = example_record()
            record[227][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                method.scoring_example(record)


if __name__ == "__main__":
    unittest.main()
