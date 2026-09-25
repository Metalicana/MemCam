import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from paper import build_revisit_appendix as appendix


class RevisitAppendixTests(unittest.TestCase):
    def test_memcam_covers_all_scenes_and_uses_time_not_quality(self):
        rows = [dict(scene="B", start_frame=10, frames=[10, 20, 30], quality=999),
                dict(scene="A", start_frame=0, frames=[10, 50, 90], quality=-999),
                dict(scene="A", start_frame=0, frames=[0, 40, 80], quality=0),
                dict(scene="B", start_frame=10, frames=[0, 30, 60], quality=-999)]
        selected = appendix.select_memcam(rows)
        self.assertEqual([(r["scene"], r["frames"]) for r in selected],
                         [("A", [0, 40, 80]), ("B", [0, 30, 60])])
        self.assertEqual(selected, appendix.select_memcam(list(reversed(rows))))

    def test_worldmem_selects_one_best_geometry_rank_for_every_trajectory(self):
        rows = [dict(trajectory=4, rank=2, quality=999),
                dict(trajectory=4, rank=1, quality=-999),
                dict(trajectory=1, rank=1, quality=0)]
        self.assertEqual([(r["trajectory"], r["rank"]) for r in appendix.select_worldmem(rows)],
                         [(1, 1), (4, 1)])

    def test_unknown_selection_rule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown selection"):
            appendix.make_specs(dict(memcam_selection="best_lpips", worldmem_selection=appendix.WORLDMEM_RULE))

    def test_shortlist_preserves_catalog_and_records_manual_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = ET.fromstring('<mxfile><diagram id="m"/><diagram id="w"/></mxfile>')
            rows = [dict(example_id="M01", page=1, stem="m"), dict(example_id="W01", page=2, stem="w")]
            config = dict(shortlist=[dict(example_id="W01", note="Mixed outcome")])
            appendix.write_shortlist(config, catalog, rows, root, "config-hash")
            self.assertEqual([p.get("id") for p in catalog], ["m", "w"])
            selected = ET.parse(root / "appendix_shortlist.drawio").getroot()
            self.assertEqual([p.get("id") for p in selected], ["w"])
            manifest = json.loads((root / "shortlist.json").read_text())
            self.assertIn("Manual visual selection", manifest["selection"])
            self.assertEqual(manifest["examples"][0]["catalog_page"], 2)
            self.assertEqual(manifest["config_sha256"], "config-hash")
            for ids in (("M01", "M01"), ("missing",)):
                config = dict(shortlist=[dict(example_id=i, note="Test") for i in ids])
                with self.assertRaises(ValueError):
                    appendix.write_shortlist(config, catalog, rows, root, "config-hash")

    def record(self, root, system):
        frames = [10, 40, 70]
        paths = {}
        for method, _ in appendix.memcam.METHODS:
            paths[method] = {}
            for frame in frames:
                path = root / f"{method}_{frame}.png"
                Image.new("RGB", (640, 360), (frame, 90, 120)).save(path)
                paths[method][str(frame)] = str(path)
        return dict(system=system, scene="Test scene", stem=system.lower(),
                    example_id="M01" if system == "MemCam" else "W01",
                    duration_sec=180 if system == "MemCam" else 60,
                    selection_rule=appendix.MEMCAM_RULE if system == "MemCam" else appendix.WORLDMEM_RULE,
                    frames=frames, timeline_fps=30 if system == "MemCam" else 10,
                    metadata=dict(width=640, height=360, frames=600, fps=15),
                    frame_paths=paths, source_sha256={m: "hash-" + m for m in paths},
                    event=dict(position_distance="0.1", rotation_deg="2",
                               endpoint_position_distance="0.4", endpoint_rotation_deg="10"))

    def test_frames_are_embedded_unmodified_and_labels_remain_editable(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.record(Path(folder), "WorldMem")
            diagram = appendix.appendix_diagram(record)
            cells = diagram.root.findall("mxCell")
            identities = [c.get("id") for c in cells]
            self.assertEqual(len(identities), len(set(identities)))
            labels = {c.get("value") for c in cells}
            self.assertTrue({"First visit", "Intervening view", "Revisit", "1 s", "4 s", "7 s",
                             "FIFO<br>B32", "KEEPSAKE<br>(Ours, B32)"} <= labels)
            images = [c for c in cells if c.get("id", "").startswith("frame-")]
            self.assertEqual(len(images), 9)
            for cell in images:
                data = cell.get("style").split("image=data:image/png,", 1)[1].split(";", 1)[0]
                with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                    frame = int(cell.get("data-frame-index"))
                    self.assertEqual(image.size, (640, 360))
                    self.assertEqual(image.getpixel((100, 100)), (frame, 90, 120))
                self.assertEqual(cell.get("data-timeline-fps"), "10")
                self.assertTrue(cell.get("data-source-sha256").startswith("hash-"))

    def test_captions_preserve_limits_and_correct_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.record(Path(folder), "WorldMem")
            text = appendix.caption(record)
            self.assertIn("1, 4 and 7 seconds", text)
            self.assertIn("10-FPS trajectory", text)
            self.assertIn("Actual post-retry dataset identity was not logged", text)
            record = self.record(Path(folder), "MemCam")
            text = appendix.caption(record)
            self.assertIn("not a verified departure", text)
            self.assertIn("not necessarily the first-ever visit", text)
            self.assertIn("not a GT reconstruction", text)

    def test_build_keeps_every_page_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [self.record(root, system) for system in ("MemCam", "WorldMem")]
            config = root / "config.json"
            config.write_text(json.dumps(dict(worldmem_bundle=str(root), worldmem_videos=str(root))))
            output = root / "output"
            with patch.object(appendix, "make_specs", return_value=(copy.deepcopy(records), {"test": True})), \
                    patch.object(appendix.memcam, "prepare_case", return_value=records[0]), \
                    patch.object(appendix.worldmem, "prepare_case", return_value=records[1]):
                appendix.build(config, output)
            pages = ET.parse(output / "revisit_appendix.drawio").getroot().findall("diagram")
            self.assertEqual([p.get("id") for p in pages], ["memcam", "worldmem"])
            self.assertEqual(sum("shape=image" in c.get("style", "") for p in pages
                                 for c in p.iter("mxCell")), 18)
            for record in records:
                provenance = json.loads((output / "examples" / f"{record['stem']}.provenance.json").read_text())
                self.assertEqual(provenance["selection_audit"], {"test": True})
                self.assertIn("renderer_hashes", provenance)
                self.assertTrue((output / "examples" / f"{record['stem']}.drawio").is_file())
            self.assertTrue((output / "index.csv").is_file())
            self.assertTrue((output / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
