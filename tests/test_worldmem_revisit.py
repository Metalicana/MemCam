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

from paper import build_worldmem_revisit as figure


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class WorldMemRevisitTests(unittest.TestCase):
    def fixture(self, root):
        clock = dict(trajectory_fps=10, expected_mp4_frame_count=600,
                     output_mp4_contains_initial_context=False, output_frame_index_base=0,
                     source_start_offset=100, context_frames=600)
        metadata = dict(width=640, height=360, frames=600, fps=15.)
        video_name = "video_batch00000_0_rank0.mp4"
        mappings = [dict(trajectory_id=t, video_filename=video_name if t == 0 else f"video{t}.mp4",
                         same_requested_trajectory="True", actual_post_retry_source_verified="False",
                         trajectory_fps=10, output_frames=600) for t in range(15)]
        event = dict(trajectory_id=0, video_filename=video_name, first_frame=10, middle_frame=40,
                     revisit_frame=70, first_time_sec=1, middle_time_sec=4, revisit_time_sec=7,
                     pose_match="True", away_verified="True", endpoint_position_distance=.25,
                     endpoint_rotation_deg=15, middle_position_distance=0, middle_rotation_deg=90,
                     away_duration_sec=2, gt_checked="False")
        rows = []
        for source in (*figure.SOURCES.values(), "ground_truth"):
            for role, frame in zip(figure.ROLES, (10, 40, 70)):
                name = f"{source}/{role}.png"
                path = root / "previews" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (640, 360), (frame, 90, 120)).save(path)
                rows.append(dict(trajectory_id=0, candidate_rank=1, source=source, role=role,
                                 output_frame=frame, source_video_frame=frame + 700,
                                 path="/remote/previews/" + name,
                                 status="exported_unchecked" if source == "ground_truth" else "exported"))
        write_csv(root / "preview_manifest.csv", rows)
        write_csv(root / "trajectory_mapping.csv", mappings)
        write_csv(root / "revisit_candidates.csv", [event])
        remote = {}
        for method, source in figure.SOURCES.items():
            path = root / "videos" / figure.DIRECTORIES[method] / video_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(method.encode())
            remote[source] = dict(size_bytes=path.stat().st_size)
        provenance = dict(clock_and_index_mapping=clock,
                          outputs=dict(preview_root="/remote/previews"),
                          search_and_sampling=dict(selected_candidate_counts={"0": 1}),
                          video_metadata={"0": remote})
        figure.save_json(root / "provenance.json", provenance)
        config = root / "selection.json"
        figure.save_json(config, dict(trajectory_id=0, candidate_rank=1, frames=[10, 40, 70],
                                     stem="worldmem", caption="Test caption."))
        return metadata, config

    def decoder(self, video, frames, folder):
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for frame in frames:
            path = folder / f"frame{frame}.png"
            Image.new("RGB", (640, 360), (frame, 90, 120)).save(path)
            paths[frame] = path
        return paths, dict(identity=dict(source_sha256=figure.digest(video)))

    def test_all_sources_align_and_unchecked_gt_remains_unchecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            candidates, _ = figure.load_bundle(root)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(candidates[0]["previews"]), 12)
            self.assertEqual(candidates[0]["event"]["gt_checked"], "False")
            self.assertEqual(candidates[0]["mapping"]["actual_post_retry_source_verified"], "False")

    def test_rejects_bad_times_geometry_and_away_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "revisit_candidates.csv"
            event = figure.csv_rows(path)[0]
            invalid = [dict(first_time_sec=10 / 15), dict(endpoint_rotation_deg=16),
                       dict(endpoint_position_distance=.8), dict(away_verified="False"),
                       dict(middle_frame=70), dict(middle_rotation_deg=20),
                       dict(away_duration_sec=.5), dict(video_filename="different.mp4")]
            for change in invalid:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    write_csv(path, [dict(event, **change)])
                    figure.load_bundle(root)

    def test_rejects_duplicate_missing_or_misaligned_previews(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "preview_manifest.csv"
            rows = figure.csv_rows(path)
            for changed in (rows + [rows[0]], rows[:-1],
                            [dict(rows[0], status="exported_unchecked")] + rows[1:],
                            [dict(rows[0], source_video_frame=10)] + rows[1:],
                            [dict(rows[0], output_frame=11, source_video_frame=711)] + rows[1:]):
                with self.assertRaises(ValueError):
                    write_csv(path, changed)
                    figure.load_bundle(root)

    def test_rejects_wrong_cohort_and_playback_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "trajectory_mapping.csv"
            rows = figure.csv_rows(path)
            for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], trajectory_fps=15)] + rows[1:]):
                with self.assertRaises(ValueError):
                    write_csv(path, changed)
                    figure.load_bundle(root)

    def test_export_pixels_timestamps_and_preserve_memcam_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, config = self.fixture(root)
            output = root / "out"
            output.mkdir()
            memcam = output / "memcam_180s_revisit.drawio"
            memcam.write_text('<mxfile><diagram id="memcam" name="Preserve me"/></mxfile>')
            before = memcam.read_bytes()
            with patch.object(figure, "probe", return_value=metadata), \
                    patch.object(figure, "decode_cached", side_effect=self.decoder):
                figure.build(root, root / "videos", config, output)
            self.assertEqual(memcam.read_bytes(), before)
            pages = ET.parse(output / "revisit_comparisons.drawio").getroot().findall("diagram")
            self.assertEqual([p.get("id") for p in pages], ["memcam", "worldmem"])
            cells = list(pages[1].iter("mxCell"))
            self.assertTrue({"1 s", "4 s", "7 s"} <= {c.get("value") for c in cells})
            images = [c for c in cells if c.get("id", "").startswith("frame-")]
            self.assertEqual(len(images), 9)
            for cell in images:
                data = cell.get("style").split("image=data:image/png,", 1)[1].split(";", 1)[0]
                with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                    frame = int(cell.get("id").rsplit("-", 1)[1])
                    self.assertEqual(image.getpixel((0, 0)), (frame, 90, 120))
                    self.assertEqual(image.size, (640, 360))
            record = json.loads((output / "worldmem.provenance.json").read_text())
            self.assertEqual(len(record["pixel_checks"]), 9)
            self.assertEqual(record["mapping"]["actual_post_retry_source_verified"], "False")

    def test_rejects_wrong_local_video_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, config = self.fixture(root)
            Image.new("RGB", (640, 360), (200, 0, 0)).save(root / "previews/unbounded/first.png")
            with patch.object(figure, "probe", return_value=metadata), \
                    patch.object(figure, "decode_cached", side_effect=self.decoder), self.assertRaisesRegex(ValueError, "pixels differ"):
                figure.build(root, root / "videos", config, root / "out")


if __name__ == "__main__":
    unittest.main()
