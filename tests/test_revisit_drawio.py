import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from paper import build_revisit_drawio as plotter


class EditableRevisitTests(unittest.TestCase):
    def fixture(self, root):
        event = dict(scene="Scene", start_frame=2128, duration_sec=180, revisit_type="exact_pose",
                     frame_i=4230, frame_j=4650, time_i_sec=141, time_j_sec=155,
                     position_distance=0.236, rotation_deg=2.46)
        source = root / "events.csv"
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(event))
            writer.writeheader()
            writer.writerow(event)
        spec = dict(stem="figure", system="MemCam", scene="Scene", start_frame=2128,
                    duration_sec=180, timeline_fps=30, frames=[4230, 4440, 4650], videos={},
                    expected_video=dict(width=640, height=352, frames=5397, fps=30.),
                    revisit_evidence=dict(csv=str(source), match=dict(scene="Scene")),
                    description="Test caption.")
        for method, _ in plotter.METHODS:
            folder = root / method
            folder.mkdir()
            video = folder / "same_trajectory.mp4"
            video.write_bytes(method.encode())
            spec["videos"][method] = str(video)
        return spec

    def decoder(self, video, frames, folder):
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for frame in frames:
            path = folder / f"frame_{frame:04d}.png"
            Image.new("RGB", (640, 352), (frame % 256, 90, 120)).save(path)
            paths[frame] = path
        return paths, dict(identity=dict(source_sha256=plotter.digest(video)))

    def test_embedded_pixels_and_editable_elements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.fixture(root)
            spec["highlights"] = [dict(method=method, frame=frame, landmark="facade", xywh=[100, 40, 210, 250])
                                  for method, _ in plotter.METHODS for frame in (4230, 4650)]
            config = root / "config.json"
            config.write_text(json.dumps([spec]))
            with patch.object(plotter, "probe", return_value=spec["expected_video"]), \
                    patch.object(plotter, "decode_cached", side_effect=self.decoder):
                plotter.build(config, root / "out")
            tree = ET.parse(root / "out/figure.drawio")
            cells = list(tree.iter("mxCell"))
            self.assertEqual(len(cells), len({c.get("id") for c in cells}))
            pictures = [c for c in cells if c.get("style", "").startswith("shape=image;")]
            self.assertEqual(len(pictures), 9)
            boxes = [c for c in cells if c.get("id", "").startswith("highlight-")]
            self.assertEqual(len(boxes), 6)
            for cell in boxes:
                self.assertIn("fillColor=none;", cell.get("style"))
                self.assertIn("dashed=0;", cell.get("style"))
                method = "KEEPSAKE" if "keepsake" in cell.get("id") else "FIFO"
                self.assertIn(f"strokeColor={plotter.BOX_COLORS[method]};", cell.get("style"))
            for cell in pictures:
                data = cell.get("style").split("image=data:image/png,", 1)[1].split(";", 1)[0]
                with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                    frame = int(cell.get("id").rsplit("-", 1)[1])
                    self.assertEqual(image.size, (640, 352))
                    self.assertEqual(image.getpixel((0, 0)), (frame % 256, 90, 120))
                geometry = cell.find("mxGeometry")
                self.assertAlmostEqual(float(geometry.get("width")) / float(geometry.get("height")), 640 / 352)
            text = {c.get("value") for c in cells}
            for label in ("Unbounded", "FIFO", "KEEPSAKE", "First visit", "Intervening view", "Revisit", "141 s", "148 s", "155 s"):
                self.assertIn(label, text)
            self.assertTrue((root / "out/revisit_comparisons.drawio").is_file())
            self.assertTrue((root / "out/figure.provenance.json").is_file())

    def test_highlights_reject_invalid_bounds_or_unpaired_objects(self):
        record = dict(metadata=dict(width=640, height=352), frames=[4230, 4440, 4650])
        pair = [dict(method="KEEPSAKE", frame=f, landmark="facade", xywh=[100, 40, 200, 250])
                for f in (4230, 4650)]
        plotter.validate_highlights(dict(record, highlights=pair))
        invalid = ([pair[0]], [pair[0], pair[0]], [pair[0], dict(pair[1], frame=4440)],
                   [pair[0], dict(pair[1], xywh=[100, 40, 600, 250])],
                   [pair[0], dict(pair[1], xywh=[100, float("nan"), 200, 250])])
        for boxes in invalid:
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                plotter.validate_highlights(dict(record, highlights=boxes))

    def test_rejects_wrong_indices_and_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.fixture(root)
            with patch.object(plotter, "probe", return_value=spec["expected_video"]):
                for change in (dict(frames=[4230, 4440, 4700]), dict(frames=[4230, 4230, 4650]),
                               dict(scene="Other"), dict(start_frame=0), dict(timeline_fps=15)):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        plotter.prepare_case(dict(spec, **change), root / "out")


if __name__ == "__main__":
    unittest.main()
