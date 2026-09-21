import base64
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zlib

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teaser", ROOT / "paper/build_motivation_teaser.py")
teaser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teaser)


def write_curves(root, bad=None):
    rows = [dict(time_sec=t, metric=metric, mean=mean, ci_low=mean-.05, ci_high=mean+.05)
            for t in (15, 160) for metric, mean in ((teaser.SELECTED, .5), (teaser.BEST, .3))]
    if bad == "oracle":
        rows[1].update(mean=.8, ci_low=.75, ci_high=.85)
    elif bad == "time":
        rows[1]["time_sec"] = 30
    elif bad == "nan":
        rows[0]["mean"] = float("nan")
    elif bad == "duplicate":
        rows.append(rows[0])
    with (root / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "provenance.json").write_text(json.dumps({"parameters": {
        "duration": 180, "expected_videos": 15, "run": "baseline"}}))


class MotivationTeaserTests(unittest.TestCase):
    def test_curve_data_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_curves(root)
            curves = teaser.load_curves(root)
            self.assertEqual(curves[teaser.SELECTED][0]["mean"], .5)
            self.assertEqual(curves[teaser.BEST][1]["mean"], .3)
            for bad in ("oracle", "time", "nan", "duplicate"):
                write_curves(root, bad)
                with self.assertRaises(ValueError):
                    teaser.load_curves(root)

    def test_confidence_band_is_a_vector_stencil(self):
        diagram = teaser.Diagram()
        diagram.band([(0, 10), (10, 20), (10, 5), (0, 0)], "#CC0000", "1")
        cell = list(diagram.root)[-1]
        packed = cell.attrib["style"].split("stencil(")[1].split(")")[0]
        shape = ET.fromstring(unquote(zlib.decompress(base64.b64decode(packed), -15).decode()))
        self.assertEqual(shape.tag, "shape")
        self.assertIsNotNone(shape.find("foreground/path/close"))
        self.assertIsNotNone(shape.find("foreground/fill"))

    def test_resource_bars_are_linear_and_scope_is_checked(self):
        data = json.loads((ROOT / "paper/motivation_teaser_inputs.json").read_text())["efficiency"]
        diagram = teaser.Diagram()
        teaser.efficiency_panel(diagram, data)
        bars = [c for c in diagram.root if c.attrib.get("style", "").startswith("rounded=0;")]
        widths = [float(c.find("mxGeometry").attrib["width"]) for c in bars]
        self.assertAlmostEqual(widths[0] / widths[1], 5397 / 32)
        self.assertAlmostEqual(widths[2] / widths[3], 3781.9190497500003 / 43.771138)
        data["videos"] = 15
        with self.assertRaises(ValueError):
            teaser.efficiency_panel(teaser.Diagram(), data)

    def test_end_to_end_embeds_matched_frames_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_curves(root)
            config = json.loads((ROOT / "paper/motivation_teaser_inputs.json").read_text())
            config.update(retrieval_directory=".", sample_directory=".")
            for spec in config["samples"]:
                case = dict(frames=[1, 301, 601, 901, 1201], fps=30, scene=spec["case"], row=1,
                            source_manifest_item=dict(duration_sec=60, start_frame=100, split_seed=0,
                                                      gt_frames_dir="/dataset/frames/scene"),
                            source_videos={"Unbounded": "baseline.mp4", "Ours": "ours.mp4"})
                (root / (spec["case"] + ".json")).write_text(json.dumps(case))
                for policy in ("ours", "unbounded", "gt_check"):
                    im = Image.new("RGB", (1952, 211), "white")
                    for i in range(5):
                        im.paste((20 * (i + 1), 40, 60), (392*i, 0, 392*i+384, 211))
                    im.save(root / (spec["case"] + "_" + policy + "_bare.png"))
            path = root / "input.json"
            path.write_text(json.dumps(config))
            output = root / "figure.drawio"
            with patch.object(teaser, "ROOT", root):
                teaser.build(path, output)
            xml = ET.parse(output)
            images = [c for c in xml.iter("mxCell") if "shape=image;" in c.attrib.get("style", "")]
            self.assertEqual(len(images), 15)
            self.assertEqual(len({c.attrib["id"] for c in xml.iter("mxCell")}),
                             len(list(xml.iter("mxCell"))))
            provenance = json.loads(output.with_suffix(".provenance.json").read_text())
            self.assertEqual(provenance["efficiency"]["videos"], 1)
            self.assertEqual(provenance["retrieval_parameters"]["expected_videos"], 15)
            for col, position in enumerate(range(5)):
                triplet = provenance["samples"][3*col:3*col+3]
                self.assertEqual(len({r["frame"] for r in triplet}), 1)
                self.assertEqual(triplet[0]["crop_box"], [392*position, 0, 392*position+384, 211])
                self.assertEqual(triplet[0]["duration_sec"], 60)
                self.assertEqual(triplet[0]["source_kind"], "ground_truth")
                self.assertTrue(triplet[0]["source_path"].endswith(f"/{101+position*300:04d}.png"))
                self.assertEqual(triplet[1]["source_kind"], "generated_video")
                for cell in images[3*col:3*col+3]:
                    encoded = cell.attrib["style"].split("image=data:image/png,")[1].split(";")[0]
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
                        self.assertEqual(im.size, (384, 211))
                        self.assertEqual(im.getpixel((10, 10)), (20*(position+1), 40, 60))
            labels = [c.attrib.get("value", "") for c in xml.iter("mxCell")]
            self.assertIn("Ground<br>truth", labels)
            self.assertIn("KEEPSAKE<br>B32", labels)
            self.assertIn("not KEEPSAKE", output.with_suffix(".tex").read_text())
            for positions in ((4, 3, 2, 1, 0), (0, 1, 1, 3, 4), (-1, 1, 2, 3, 4), (0, 1, 2, 3, 6)):
                invalid = [dict(s, position=p) for s, p in zip(config["samples"], positions)]
                with self.assertRaises(ValueError):
                    teaser.sample_panel(teaser.Diagram(), root, invalid)
            invalid = [dict(s) for s in config["samples"]]
            invalid[-1]["case"] = "another-scene"
            with self.assertRaises(ValueError):
                teaser.sample_panel(teaser.Diagram(), root, invalid)
            (root / (config["samples"][0]["case"] + "_gt_check_bare.png")).unlink()
            with self.assertRaises(FileNotFoundError):
                teaser.sample_panel(teaser.Diagram(), root, config["samples"])


if __name__ == "__main__":
    unittest.main()
