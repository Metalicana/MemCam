import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from paper import make_revisit_comparisons as plotter


class RevisitComparisonTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(scene="Scene", start_frame=684, num_frames=5397, fps=30)
        self.event = dict(scene="Scene", start_frame=684, duration_sec=180,
                          revisit_type="exact_pose", frame_i=210, frame_j=1500,
                          time_i_sec=7, time_j_sec=50, position_distance=0.16,
                          rotation_deg=0.05)

    def load(self, root, events):
        path = root / "events.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.event))
            writer.writeheader()
            writer.writerows(events)
        return plotter.load_events(path, [self.item])

    def test_pose_event_identity_and_midpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.load(root, [self.event])[0]
            self.assertEqual(record["frames"], [210, 855, 1500])
            self.assertEqual(record["position_distance_m"], 0.16)
            for change in (dict(frame_j=6000), dict(start_frame=0), dict(time_j_sec=49),
                           dict(position_distance=0.26), dict(rotation_deg=float("nan")),
                           dict(revisit_type="gaze_point"), dict(frame_i=1490)):
                with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                    self.load(root, [dict(self.event, **change)])
            with self.assertRaises(ValueError):
                self.load(root, [self.event, self.event])

    def test_pixels_labels_and_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.load(root, [self.event])[0]
            for method, _, _ in plotter.METHODS:
                record["frame_paths"][method] = {}
                for frame in record["frames"]:
                    path = root / f"{method}_{frame}.png"
                    Image.new("RGB", (640, 352), (frame % 256, 70, 90)).save(path)
                    record["frame_paths"][method][str(frame)] = str(path)
            fig = plotter.make_figure(record)
            self.assertEqual(len(fig.axes), 6)
            for col, frame in enumerate(record["frames"]):
                for row in range(2):
                    arr = fig.axes[2 * col + row].images[0].get_array()
                    self.assertEqual(arr.shape, (352, 640, 3))
                    self.assertEqual(tuple(arr[0, 0]), (frame % 256, 70, 90))
            labels = [t.get_text() for t in fig.texts]
            self.assertIn("First visit  |  7 s", labels)
            self.assertIn("Between visits  |  28.5 s", labels)
            self.assertIn("Return  |  50 s", labels)
            fig.canvas.draw()
            for text in fig.texts:
                box = text.get_window_extent(fig.canvas.get_renderer())
                self.assertGreaterEqual(box.x0, 0)
                self.assertGreaterEqual(box.y0, 0)
                self.assertLessEqual(box.x1, fig.bbox.width)
                self.assertLessEqual(box.y1, fig.bbox.height)
            plotter.plt.close(fig)
            record["frame_paths"]["gt"] = {}
            for frame in record["frames"]:
                path = root / f"gt_{frame}.png"
                Image.new("RGB", (640, 360), (frame % 256, 10, 20)).save(path)
                record["frame_paths"]["gt"][str(frame)] = str(path)
            fig = plotter.make_figure(record)
            self.assertEqual(len(fig.axes), 9)
            self.assertEqual(fig.axes[0].images[0].get_array().shape, (360, 640, 3))
            self.assertEqual(fig.axes[1].images[0].get_array().shape, (352, 640, 3))
            plotter.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
