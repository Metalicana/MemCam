import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
from run_matched_video_metrics import matched_rows
sys.path.pop(0)


class MatchedTests(unittest.TestCase):
    def test_cohort_validation_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "videos/baseline"
            run.mkdir(parents=True)
            manifest = root / "manifest.jsonl"
            items = [{"duration_sec": 10, "output_prefix": "short_"}]
            for index in range(15):
                prefix = f"seed0_scene{index}_60s_"
                items.append({"duration_sec": 60, "output_prefix": prefix, "num_frames": 1825})
                (run / (prefix + "custom.mp4")).write_bytes(b"fixture")
            (run / "seed1_extra60s_custom.mp4").write_bytes(b"fixture")
            manifest.write_text("\n".join(map(json.dumps, items)))
            self.assertEqual(matched_rows(manifest, run.parent, "baseline"), list(range(1, 16)))
            subprocess.run([sys.executable, str(UTILS / "run_matched_video_metrics.py"),
                            "--manifest", str(manifest), "--results-root", str(run.parent),
                            "--output-root", str(root / "output"), "--run", "baseline",
                            "--only", "vbench-long", "--dry-run"], check=True, capture_output=True)
            status = json.loads(next((root / "output").glob("*/suite_status.json")).read_text())
            self.assertEqual(len(status["results"]), 15)
            self.assertTrue(all(row["exit_code"] == 0 for row in status["results"]))
            (run / "seed0_scene0_60s_custom.mp4").unlink()
            with self.assertRaisesRegex(ValueError, "missing/empty"):
                matched_rows(manifest, run.parent, "baseline")
            manifest.write_text("\n".join(map(json.dumps, items[:-1])))
            with self.assertRaisesRegex(ValueError, "exactly 15"):
                matched_rows(manifest, run.parent, "baseline")


if __name__ == "__main__":
    unittest.main()
