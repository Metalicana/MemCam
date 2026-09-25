import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from run_vbench_batched import BatchedImageEncoder, batch_background


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is needed for tensor equivalence checks")
class BatchedVBenchTests(unittest.TestCase):
    def test_encoding_is_bounded_ordered_and_gradient_free(self):
        import torch

        class Encoder:
            def __init__(self):
                self.calls = []

            def encode_image(self, images):
                self.calls.append((len(images), torch.is_grad_enabled()))
                return images * 2 + 1

        model = Encoder()
        images = torch.arange(67 * 3, dtype=torch.float32).reshape(67, 3).requires_grad_()
        output = BatchedImageEncoder(model, 16).encode_image(images)
        torch.testing.assert_close(output, images.detach() * 2 + 1)
        self.assertEqual(model.calls, [(16, False)] * 4 + [(3, False)])
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.is_grad_enabled())

    def test_temporal_score_preserves_first_frame_and_cross_batch_pairs(self):
        import torch
        import torch.nn.functional as F

        generator = torch.Generator().manual_seed(2)
        videos = [torch.randn(n, 7, generator=generator) for n in (35, 67)]

        class Encoder:
            def encode_image(self, images):
                return images.sin() + images * .2

        # Upstream compares each frame to both the previous frame and the
        # video's first frame, then weights videos by transition counts.
        def upstream(model, preprocess, video_list, device, read_frame):
            total, count, details = 0., 0, []
            for images in video_list:
                features = F.normalize(model.encode_image(images), dim=-1, p=2)
                score = 0.
                for index in range(1, len(features)):
                    prev = max(0., F.cosine_similarity(features[index-1:index], features[index:index+1]).item())
                    first = max(0., F.cosine_similarity(features[:1], features[index:index+1]).item())
                    score += (prev + first) / 2
                total += score
                count += len(features) - 1
                details.append(score / (len(features) - 1))
            return total / count, details

        expected = upstream(Encoder(), None, videos, "cpu", False)
        for size in (1, 16, 128):
            with self.subTest(batch_size=size):
                self.assertEqual(batch_background(upstream, size)(Encoder(), None, videos, "cpu", False), expected)

    def test_invalid_batch_and_empty_input_rejected(self):
        import torch

        with self.assertRaises(ValueError):
            BatchedImageEncoder(None, 0)
        with self.assertRaises(ValueError):
            BatchedImageEncoder(None, 16).encode_image(torch.empty(0, 3))

    def test_against_installed_upstream_background_function(self):
        import torch
        import torch.nn.functional as F

        default = Path(__file__).resolve().parents[2] / "VBench"
        path = Path(os.environ.get("VBENCH_ROOT", default)) / "vbench/background_consistency.py"
        if not path.is_file():
            self.skipTest("Set VBENCH_ROOT for the upstream-source equivalence test")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "background_consistency")
        generator = torch.Generator().manual_seed(8)
        frames = {name: torch.randn(n, 7, generator=generator) for name, n in (("a", 35), ("b", 67))}
        namespace = {"torch": torch, "F": F, "get_rank": lambda: 0,
                     "tqdm": lambda xs, **kw: xs, "load_video": frames.__getitem__,
                     "clip_transform": lambda size: lambda images: images}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        upstream = namespace["background_consistency"]

        class Encoder:
            def encode_image(self, images):
                return images.sin() + .2 * images

        expected = upstream(Encoder(), None, list(frames), "cpu", False)
        actual = batch_background(upstream, 16)(Encoder(), None, list(frames), "cpu", False)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
