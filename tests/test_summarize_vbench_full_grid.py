import json
from pathlib import Path
import tempfile
import unittest

from utils.audit_metric_coverage import DIMENSIONS
from utils.summarize_vbench_full_grid import find_exact_result, validate_result


def payload(names, long=False):
    result = {}
    for dimension in DIMENSIONS:
        rows = [{"video_path": f"/stage/baseline/{name}",
                 "video_results": 50.0 if dimension == "imaging_quality" else 0.5}
                for name in names]
        result[dimension] = [0.5, rows, rows] if long else [0.5, rows]
    return result


class SummarizeVBenchFullGridTest(unittest.TestCase):
    def test_standard_and_long_details_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = {f"video_{i}.mp4" for i in range(15)}
            for label, long in (("standard", False), ("long", True)):
                path = root / f"{label}_eval_results.json"
                path.write_text(json.dumps(payload(sorted(names), long=long)))
                scores = validate_result(path, names)
                self.assertEqual(scores["subject_consistency"]["n"], 15)
                self.assertEqual(scores["imaging_quality"]["mean"], 0.5)

    def test_superset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = {f"video_{i}.mp4" for i in range(15)}
            path = root / "bad_eval_results.json"
            path.write_text(json.dumps(payload(sorted(names | {"extra.mp4"}))))
            with self.assertRaises(ValueError):
                validate_result(path, names)
            with self.assertRaises(ValueError):
                find_exact_result(root, names)


if __name__ == "__main__":
    unittest.main()
