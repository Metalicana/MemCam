import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from paper import make_180s_gt_comparisons as plotter


def fixture(folder):
    prefix = "seed0_test_0684_180s_"
    item = dict(output_prefix=prefix, duration_sec=180, num_frames=5397, fps=30,
                scene="Test", start_frame=684, videos={}, samples=[])
    for method in ("unbounded", "keepsake_b32"):
        relative = f"{method}/{prefix}custom.mp4"
        path = folder / relative
        path.parent.mkdir()
        path.write_bytes(b"fixture video")
        item["videos"][method] = relative
    (folder / "gt").mkdir()
    for seconds in plotter.TIMES:
        frame = min(seconds * 30, 5396)
        relative = f"gt/{prefix}frame_{frame:05d}.png"
        Image.new("RGB", (64, 36), (40, 60, 80)).save(folder / relative)
        item["samples"].append(dict(requested_sec=seconds, actual_sec=frame / 30,
                                    frame_index=frame, gt_file=relative))
    (folder / "manifest.json").write_text(json.dumps([item]))
    return item


class ComparisonTests(unittest.TestCase):
    def test_mapping_and_input_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = fixture(root)
            self.assertEqual(plotter.load_items(root, 1)[0]["samples"][-1]["frame_index"], 5396)
            for change in ("time", "gt", "policy", "duplicate", "count", "missing"):
                with self.subTest(change=change):
                    item = copy.deepcopy(original)
                    if change == "time":
                        item["samples"][-1]["actual_sec"] = 180
                    elif change == "gt":
                        item["samples"][0]["gt_file"] = item["samples"][1]["gt_file"]
                    elif change == "policy":
                        item["videos"]["keepsake_b32"] = item["videos"]["unbounded"]
                    elif change == "count":
                        item["num_frames"] = 5401
                    elif change == "missing":
                        item["output_prefix"] = "missing_"
                        item["videos"]["unbounded"] = "unbounded/missing_custom.mp4"
                    items = [item, item] if change == "duplicate" else [item]
                    (root / "manifest.json").write_text(json.dumps(items))
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        plotter.load_items(root, 1)

    def test_cache_detects_tampering_and_uses_fresh_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            output = root / "cache"

            def extract(video, frames, folder):
                paths = {}
                for frame in frames:
                    paths[frame] = folder / f"frame_{frame:04d}.png"
                    Image.new("RGB", (64, 35), (frame, 20, 30)).save(paths[frame])
                return paths

            with patch.object(plotter, "extract_frames", side_effect=extract) as decoder:
                paths, _ = plotter.decode_cached(source, [1, 3], output)
                plotter.decode_cached(source, [1, 3], output)
                self.assertEqual(decoder.call_count, 1)
                paths[1].write_bytes(b"bad image")
                plotter.decode_cached(source, [1, 3], output)
                self.assertEqual(decoder.call_count, 2)
                self.assertNotEqual(decoder.call_args.args[2], output)

    def test_end_to_end_exports_and_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            output = root / "output"

            def extract(video, frames, folder):
                paths = {}
                for frame in frames:
                    path = folder / f"frame_{frame:04d}.png"
                    Image.new("RGB", (64, 35), (frame % 256, 90, 120)).save(path)
                    paths[frame] = path
                return paths

            info = dict(frames=5397, fps=30., width=64, height=35)
            with patch.object(plotter, "probe", return_value=info), patch.object(plotter, "extract_frames", side_effect=extract):
                plotter.render(root, output, 1)
            record = json.loads((output / "manifest.json").read_text())[0]
            self.assertEqual(record["gt_dataset_indices"], [714, 2484, 4284, 6080])
            for extension in ("png", "pdf", "json", "caption.txt"):
                self.assertTrue((output / f"{record['stem']}.{extension}").is_file())
            self.assertTrue((output / "review_01.jpg").is_file())
            self.assertTrue((output / "all_scenes.pdf").is_file())
            fig = plotter.make_figure(record)
            self.assertEqual(len(fig.axes), 12)
            labels = [t.get_text() for t in fig.texts]
            for label in ("GT", "Unbounded", "KEEPSAKE", "1 s", "60 s", "120 s", "179.9 s"):
                self.assertIn(label, labels)
            self.assertEqual(fig.axes[0].images[0].get_array().shape, (36, 64, 3))
            self.assertEqual(fig.axes[1].images[0].get_array().shape, (35, 64, 3))
            fig.canvas.draw()
            for text in fig.texts:
                box = text.get_window_extent(fig.canvas.get_renderer())
                self.assertGreaterEqual(box.x0, 0)
                self.assertGreaterEqual(box.y0, 0)
                self.assertLessEqual(box.x1, fig.bbox.width)
                self.assertLessEqual(box.y1, fig.bbox.height)
            plotter.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
