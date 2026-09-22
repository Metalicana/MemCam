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
from tests.test_paired_retrieval_curves import fixture_rows, write_input


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
            config.pop("layout", None)
            config.pop("sample_comparison", None)
            config.pop("revisit_triplet", None)
            config.pop("retrieval_comparison", None)
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

    def test_paired_frames_are_intact_and_keep_hidden_gt_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(case="pair", display_name="Chemical plant", position=2, frame=841,
                        videos_directory=str(root / "videos"), highlight_box=[.43, .06, .7, .67],
                        gt_check_note="Both images have other errors.",
                        caption_detail="A local contrast, not an exact-view reconstruction.",
                        annotations={"KEEPSAKE_B32": "Open gap", "Unbounded": "Extra vessel"})
            case = dict(frames=[1, 421, 841, 1261, 1681], fps=30,
                        scene="ChemicalPlantEnv_5", row=13,
                        source_manifest_item=dict(duration_sec=60, start_frame=3648,
                                                  split_seed=0, fps=30, num_frames=1825,
                                                  output_prefix="seed0_test_", gt_frames_dir="/gt"))
            (root / (spec["case"] + ".json")).write_text(json.dumps(case))
            Image.new("RGB", (1952, 211), "blue").save(
                root / (spec["case"] + "_gt_check_bare.png"))
            for folder in ("KEEPSAKE_B32", "Unbounded"):
                path = root / "videos" / folder / "seed0_test_custom.mp4"
                path.parent.mkdir(parents=True)
                path.write_bytes(folder.encode())

            def decode(video, frames, out):
                self.assertEqual(frames, [841])
                out.mkdir(parents=True)
                image = Image.new("RGB", (640, 352), "green")
                image.putpixel((0, 0), (255, 0, 0))
                image.putpixel((639, 351), (255, 255, 0))
                path = out / "frame_0841.png"
                image.save(path)
                return {841: path}

            diagram = teaser.Diagram()
            with patch.object(teaser, "extract_frames", side_effect=decode) as extraction:
                rows = teaser.annotated_sample_panel(diagram, root, spec, root / "assets")
            self.assertEqual(extraction.call_count, 2)
            self.assertEqual([r["policy"] for r in rows], ["Ground truth", "KEEPSAKE_B32", "Unbounded"])
            self.assertEqual({r["dataset_frame"] for r in rows}, {4489})
            self.assertFalse(rows[0]["displayed_in_main"])
            self.assertEqual(rows[0]["crop_box"], [784, 0, 1168, 211])
            self.assertTrue(rows[0]["source_path"].endswith("4489.png"))
            caption = teaser.paired_caption(spec, rows)
            self.assertIn("28.03 seconds", caption)
            self.assertIn("not an exact-view reconstruction", caption)
            self.assertIn("not KEEPSAKE", caption)
            images = [c for c in diagram.root if "shape=image;" in c.get("style", "")]
            self.assertEqual(len(images), 2)
            for c in images:
                g = c.find("mxGeometry")
                self.assertAlmostEqual(float(g.get("width")) / float(g.get("height")), 640 / 352)
                encoded = c.get("style").split("image=data:image/png,")[1].split(";")[0]
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
                    self.assertEqual(im.size, (640, 352))
                    self.assertEqual(im.getpixel((0, 0)), (255, 0, 0))
                    self.assertEqual(im.getpixel((639, 351)), (255, 255, 0))
            proof = ET.parse(root / "assets/gt_check.drawio")
            self.assertEqual(sum("shape=image;" in c.get("style", "") for c in proof.iter("mxCell")), 3)
            for changes in ({"position": -1}, {"frame": 900},
                            {"highlight_box": [-.1, 0, .5, .5]},
                            {"highlight_box": [.4, 0, .3, .5]}):
                with self.assertRaises(ValueError):
                    teaser.annotated_sample_panel(teaser.Diagram(), root, dict(spec, **changes), root / "invalid")

    def test_display_gamma_is_uniform_monotonic_and_preserves_source(self):
        image = Image.new("RGB", (256, 1))
        image.putdata([(v, v, v) for v in range(256)])
        original = image.tobytes()
        adjusted = teaser.brighten_for_display(image, .65)
        self.assertEqual(image.tobytes(), original)
        expected = [round(255 * (v / 255) ** .65) for v in range(256)]
        self.assertEqual(list(adjusted.getdata()), [(v, v, v) for v in expected])
        self.assertEqual(expected, sorted(expected))
        self.assertEqual((expected[0], expected[-1]), (0, 255))
        self.assertGreater(expected[64], 64)
        self.assertEqual(teaser.brighten_for_display(image, 1).tobytes(), original)
        for value in (0, -1, 2, float("nan"), "auto"):
            with self.assertRaises(ValueError):
                teaser.brighten_for_display(image, value)

    def test_right_crop_bounds_are_fixed_and_validated(self):
        self.assertEqual(teaser.right_crop_bounds((640, 352), .25), (0, 0, 480, 352))
        self.assertEqual(teaser.right_crop_bounds((640, 352), 0), (0, 0, 640, 352))
        for fraction in (-.1, 1, float("nan"), "auto", .99999):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                teaser.right_crop_bounds((640, 352), fraction)

    def test_revisit_brightens_six_frames_equally_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_curves(root)
            config = json.loads((ROOT / "paper/motivation_teaser_inputs.json").read_text())
            config.update(retrieval_directory=".", sample_directory=".")
            write_input(root / "queries.csv", fixture_rows())
            config["retrieval_comparison"] = dict(source="queries.csv", y_limits=[.1, .6], parameters=dict(
                duration=60, expected_videos=2, bins=4, bootstrap=100, seed=0, fps=30))
            spec = config["revisit_triplet"]
            spec.update(videos_directory=str(root / "videos"), diagnostics="diagnostics.json")
            frames = spec["frames"]
            case = dict(fps=30, row=3, scene="AncientTempleEnv_5",
                        source_manifest_item=dict(fps=30, num_frames=1825, duration_sec=60,
                                                  start_frame=1368, split_seed=0,
                                                  output_prefix="seed0_test_"))
            (root / (spec["case"] + ".json")).write_text(json.dumps(case))
            group = dict(anchor=spec["anchor"], frames=sorted(set([frames[0], spec["anchor"], frames[-1]])),
                         gt_passed=False, gt_pairs=[dict(a=frames[0], b=frames[-1], ssim=.356)])
            diagnostic = dict(parameters={"min_gt_ssim": .9}, records=[dict(
                row=3, scene="AncientTempleEnv_5", profile=spec["profile"], position_m=.5,
                rotation_deg=10., gt_checks=[dict(checked=[group])])])
            (root / "diagnostics.json").write_text(json.dumps(diagnostic))
            for policy in ("Unbounded", "KEEPSAKE_B32"):
                video = root / "videos" / policy / "seed0_test_custom.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(policy.encode())

            def decode(video, requested, out):
                self.assertEqual(requested, frames)
                out.mkdir(parents=True, exist_ok=True)
                result = {}
                for i, frame in enumerate(requested):
                    image = Image.new("RGB", (640, 352), (i * 50, 80, 100))
                    image.putpixel((0, 0), (255, 0, 0))
                    image.putpixel((479, 351), (0, 255, 0))
                    image.putpixel((639, 351), (0, 0, 255))
                    path = out / f"frame_{frame:04d}.png"
                    image.save(path)
                    result[frame] = path
                return result

            config_path = root / "input.json"
            config_path.write_text(json.dumps(config))
            output = root / "figure.drawio"
            with patch.object(teaser, "ROOT", root), patch.object(teaser, "extract_frames", side_effect=decode) as decode_mock:
                teaser.build(config_path, output)
            self.assertEqual(decode_mock.call_count, 2)
            provenance = json.loads(output.with_suffix(".provenance.json").read_text())
            rows = provenance["samples"]
            self.assertEqual([r["frame"] for r in rows], frames * 2)
            self.assertEqual([r["policy"] for r in rows], ["Unbounded"] * 3 + ["KEEPSAKE_B32"] * 3)
            self.assertEqual([r["dataset_frame"] for r in rows[:3]], [1368 + f for f in frames])
            self.assertFalse(provenance["revisit_check"]["checked_group"]["gt_passed"])
            self.assertEqual(provenance["revisit_check"]["endpoint_gt_ssim"], .356)
            self.assertIn("not identical-view", provenance["revisit_check"]["interpretation"])
            caption = output.with_suffix(".tex").read_text()
            self.assertIn(", ".join(f"{f / 30:.1f}" for f in frames) + " seconds", caption)
            self.assertIn("not exact GT reconstruction", caption)
            self.assertIn("shared unbounded-source pixels", caption)
            self.assertIn("gamma=0.65", caption)
            self.assertIn("same image region", caption)
            self.assertIn("All panels use MemCam", caption)
            self.assertIn("32-frame budget throughout", caption)
            self.assertIn("one thread and one repeat", caption)
            self.assertIn("2 matched 60-second", caption)
            self.assertIn("Ancient temple", caption)
            self.assertIn("Unbounded changes the scene composition", caption)
            self.assertIn("Match labels describe visual recurrence relative to the initial generated view", caption)
            self.assertIn("rightmost 25\\% is cropped identically from all six frames", caption)
            self.assertNotIn("Full frames are shown", caption)
            xml = ET.parse(output)
            self.assertEqual(int(xml.find(".//mxGraphModel").get("pageHeight")), teaser.COMPACT_HEIGHT)
            page_width = int(xml.find(".//mxGraphModel").get("pageWidth"))
            self.assertEqual(page_width, 1670)
            self.assertEqual(provenance["layout"]["width"], page_width)
            self.assertEqual(provenance["layout"]["name"], "compact_two_row")
            cells = list(xml.iter("mxCell"))
            self.assertEqual(len(cells), len({c.get("id") for c in cells}))
            images = [c for c in cells if "shape=image;" in c.get("style", "")]
            self.assertEqual(len(images), 6)
            outlines = [c for c in cells if "fillColor=none;strokeColor=" in c.get("style", "")
                        and c.get("parent") == "samples"]
            self.assertEqual(len(outlines), 4)
            self.assertTrue(all("strokeWidth=6;dashed=0;" in c.get("style") for c in outlines))
            self.assertTrue(all("strokeColor=#FFFFFF;" not in c.get("style") for c in outlines))
            self.assertEqual(sum(f"strokeColor={teaser.UNBOUNDED};" in c.get("style") for c in outlines), 2)
            self.assertEqual(sum(f"strokeColor={teaser.KEEPSAKE};" in c.get("style") for c in outlines), 2)
            labels = [c.get("value", "") for c in cells]
            for removed in ("View changes", "Closer revisit", "MemCam / 180-second rollout",
                            "1 CPU thread / 8 queries / 1 trajectory", spec["visual_note"],
                            "Shared-source test / 60 s / 2 matched trajectories",
                            "KEEPSAKE (B32)", "KEEPSAKE<br>(Ours, B32)"):
                self.assertNotIn(removed, labels)
            self.assertIn("KEEPSAKE", labels)
            self.assertEqual([r["annotation"] for r in rows],
                             [None, None, "View mismatch", None, None, "View matched"])
            for text, color in (("View mismatch", teaser.UNBOUNDED), ("View matched", teaser.KEEPSAKE)):
                annotations = [c for c in cells if c.get("value") == text]
                self.assertEqual(len(annotations), 1)
                annotation = annotations[0]
                self.assertIn(f"fontColor={color};", annotation.get("style"))
                geometry = annotation.find("mxGeometry")
                box_geometry = [c.find("mxGeometry") for c in outlines
                                if f"strokeColor={color};" in c.get("style")][-1]
                self.assertEqual(geometry.get("x"), box_geometry.get("x"))
                self.assertEqual(geometry.get("width"), box_geometry.get("width"))
                self.assertGreater(float(geometry.get("y")),
                                   float(box_geometry.get("y")) + float(box_geometry.get("height")))
            groups = {c.get("id"): c.find("mxGeometry") for c in cells if c.get("style") == "group;"}
            efficiency_right = float(groups["efficiency"].get("x")) + float(groups["efficiency"].get("width"))
            self.assertEqual(float(groups["retrieval"].get("x")) - efficiency_right, 32)
            self.assertEqual(float(groups["retrieval"].get("x")) + float(groups["retrieval"].get("width")), page_width - 26)
            row_labels = [c for c in cells if c.get("parent") == "samples"
                          and c.get("value") in ("Unbounded", "KEEPSAKE")]
            self.assertEqual(len(row_labels), 2)
            for label in row_labels:
                self.assertIn("horizontal=0;", label.get("style"))
                self.assertLessEqual(float(label.find("mxGeometry").get("width")), 40)
            content_bottoms = {}
            for name in ("efficiency", "retrieval"):
                content_bottoms[name] = max(
                    float(c.find("mxGeometry").get("y", 0)) + float(c.find("mxGeometry").get("height"))
                    for c in cells if c.get("parent") == name and c.get("vertex") == "1")
            self.assertAlmostEqual(content_bottoms["efficiency"], content_bottoms["retrieval"])
            for cell in cells:
                if cell.get("parent") not in groups or cell.get("vertex") != "1":
                    continue
                geometry, parent = cell.find("mxGeometry"), groups[cell.get("parent")]
                for coord, extent in (("x", "width"), ("y", "height")):
                    end = float(geometry.get(coord, 0)) + float(geometry.get(extent))
                    self.assertLessEqual(end, float(parent.get(extent)), cell.get("value"))
            for group_geometry in groups.values():
                self.assertLessEqual(float(group_geometry.get("y")) + float(group_geometry.get("height")),
                                     teaser.COMPACT_HEIGHT)
                self.assertLessEqual(float(group_geometry.get("x")) + float(group_geometry.get("width")),
                                     page_width)
            for i, cell in enumerate(images):
                geometry = cell.find("mxGeometry")
                w, h = float(geometry.get("width")), float(geometry.get("height"))
                self.assertAlmostEqual(w / h, 480 / 352)
                self.assertAlmostEqual(w, 510)
                self.assertAlmostEqual(h, 374)
                self.assertLessEqual(float(geometry.get("x")) + w, page_width - 52)
                self.assertLessEqual(float(geometry.get("y")) + h, 862)
                encoded = cell.get("style").split("image=data:image/png,")[1].split(";")[0]
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    self.assertEqual(image.size, (480, 352))
                    self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
                    self.assertEqual(image.getpixel((479, 351)), (0, 255, 0))
                    raw_pixel = ((i % 3) * 50, 80, 100)
                    self.assertEqual(image.getpixel((100, 100)),
                                     tuple(round(255 * (v / 255) ** .65) for v in raw_pixel))
                    with Image.open(rows[i]["local_image"]) as original:
                        self.assertEqual(original.size, (640, 352))
                        self.assertEqual(original.getpixel((639, 351)), (0, 0, 255))
                        self.assertEqual(original.getpixel((100, 100)), raw_pixel)
                        expected = teaser.brighten_for_display(original.crop((0, 0, 480, 352)), .65)
                        self.assertEqual(image.tobytes(), expected.tobytes())
                    with Image.open(rows[i]["display_image"]) as displayed:
                        self.assertEqual(displayed.tobytes(), image.tobytes())
                self.assertEqual(rows[i]["display_transform"]["gamma"], .65)
                self.assertEqual(rows[i]["display_transform"]["right_crop_fraction"], .25)
                self.assertEqual(rows[i]["native_dimensions"], [640, 352])
                self.assertEqual(rows[i]["display_dimensions"], [480, 352])
                self.assertEqual(rows[i]["crop_box"], [0, 0, 480, 352])
                self.assertEqual(rows[i]["highlight_box"], spec["highlight_box"] if i % 3 != 1 else None)
                if i % 3 != 1:
                    source_box, display_box = rows[i]["highlight_box"], rows[i]["display_highlight_box"]
                    self.assertAlmostEqual(display_box[0] * w, source_box[0] * 680)
                    self.assertAlmostEqual(display_box[2] * w, source_box[2] * 680)
                    self.assertEqual(display_box[1::2], source_box[1::2])
            first, middle, last = frames
            for changes in ({"frames": [first, first - 1, last]}, {"frames": [first, middle, middle]},
                            {"frames": [1, middle, last]}, {"frames": [first, middle, 1825]},
                            {"frames": [first + 1, middle, last]}, {"anchor": 42}, {"profile": "strict"},
                            {"display_gamma": 0}, {"highlight_box": [0, 0, 2, .8]},
                            {"display_right_crop_fraction": -1}, {"display_right_crop_fraction": .8}):
                with patch.object(teaser, "ROOT", root), self.assertRaises(ValueError):
                    teaser.revisit_sample_panel(teaser.Diagram(), root, dict(spec, **changes), root / "invalid")

    def test_paired_panel_exports_actual_policies_without_oracle_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_input(root / "queries.csv", fixture_rows())
            config = json.loads((ROOT / "paper/motivation_teaser_inputs.json").read_text())
            config.pop("layout", None)
            config.pop("revisit_triplet")
            config["retrieval_comparison"] = dict(source="queries.csv", parameters=dict(
                duration=60, expected_videos=2, bins=4, bootstrap=100, seed=0, fps=30))
            config_path = root / "input.json"
            config_path.write_text(json.dumps(config))
            output = root / "figure.drawio"
            with patch.object(teaser, "ROOT", root), patch.object(teaser, "sample_panel", return_value=[]):
                teaser.build(config_path, output)
            provenance = json.loads(output.with_suffix(".provenance.json").read_text())
            self.assertEqual(provenance["retrieval_parameters"]["duration"], 60)
            self.assertEqual(provenance["retrieval_comparison"]["queries_per_policy"], 32)
            labels = [c.get("value", "") for c in ET.parse(output).iter("mxCell")
                      if c.get("parent") == "retrieval"]
            self.assertIn("Unbounded selection", labels)
            self.assertIn("KEEPSAKE selection (B32)", labels)
            self.assertNotIn("Best available in archive", labels)
            self.assertIn("DINO cosine distance to reference &#8595;", labels)
            caption = output.with_suffix(".tex").read_text()
            self.assertIn("2 matched 60-second trajectories (32 reads per policy)", caption)
            self.assertIn("same unbounded source video", caption)
            self.assertNotIn("blue curve", caption)
            with output.with_suffix(".retrieval_comparison.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            self.assertEqual({r["run"] for r in rows}, {"baseline", "slam_b32_covisibility"})

    def test_zoom_changes_only_coordinates_and_rejects_clipped_bands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_input(root / "queries.csv", fixture_rows())
            spec = dict(source="queries.csv", parameters=dict(
                duration=60, expected_videos=2, bins=4, bootstrap=100, seed=0, fps=30))
            with patch.object(teaser, "ROOT", root):
                full = teaser.paired_retrieval_panel(teaser.Diagram(), spec, root / "full.drawio")
                zoom = teaser.paired_retrieval_panel(teaser.Diagram(), dict(spec, y_limits=[.1, .6]),
                                                    root / "zoom.drawio", compact=True)
                self.assertEqual((root / "full.retrieval_comparison.csv").read_bytes(),
                                 (root / "zoom.retrieval_comparison.csv").read_bytes())
                self.assertEqual(full["summaries"], zoom["summaries"])
                self.assertEqual(zoom["presentation"]["y_limits"], [.1, .6])
                self.assertFalse(zoom["presentation"]["zero_origin"])
                for limits in ([.3, .4], [.6, .1], [float("nan"), .6], [0, 3]):
                    with self.subTest(limits=limits), self.assertRaises(ValueError):
                        teaser.paired_retrieval_panel(teaser.Diagram(), dict(spec, y_limits=limits),
                                                      root / "invalid.drawio")


if __name__ == "__main__":
    unittest.main()
