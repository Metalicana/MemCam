import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper/build_lookup_work_table.py"
SPEC = importlib.util.spec_from_file_location("lookup_work", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    return [dict(run_name="baseline", row=i, scene=f"scene{i}", dataset_start_frame=0,
                 duration_sec=180, section_idx=s, target_frame=target,
                 candidate_count=10 * (s + 1) + i * 2,
                 traced_candidate_count=10 * (s + 1) + i * 2,
                 candidate_count_mismatch=0)
            for i in range(2) for s, target in enumerate((30, 1350, 2700, 4050, 5400))]


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LookupWorkTests(unittest.TestCase):
    def test_means_windows_endpoint_and_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            rows = fixture()
            write(path, rows + [dict(r, run_name="fifo") for r in rows])
            result = module.memcam_counts(path, "baseline", 180, 30, 2)
            self.assertEqual(result["queries"], 10)
            self.assertEqual(result["queries_with_logged_count"], 10)
            self.assertEqual([r["candidate_mean"] for r in result["windows"]], [11, 21, 31, 46])
            self.assertEqual(result["windows"][-1]["sampled_queries"], 4)
            self.assertEqual(result["windows"][-1]["candidate_min"], 40)
            self.assertEqual(result["windows"][-1]["candidate_max"], 52)

    def test_rejects_duplicates_coverage_counts_and_timestamps(self):
        invalid = [fixture() + [fixture()[0]], fixture()[:-1]]
        for field, value in (("candidate_count", "nan"), ("traced_candidate_count", 99),
                             ("candidate_count_mismatch", 1), ("target_frame", 5401)):
            rows = fixture()
            rows[0][field] = value
            invalid.append(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            for rows in invalid:
                write(path, rows)
                with self.assertRaises(ValueError):
                    module.memcam_counts(path, "baseline", 180, 30, 2)
            write(path, fixture())
            with self.assertRaises(ValueError):
                module.memcam_counts(path, "baseline", 180, 30, 15)

    def test_reconstructed_only_counts_are_disclosed(self):
        rows = fixture()
        for row in rows:
            row["traced_candidate_count"] = -1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            write(path, rows)
            self.assertEqual(module.memcam_counts(path, "baseline", 180, 30, 2)["queries_with_logged_count"], 0)

    def test_cli_unified_artifacts_and_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, output = root / "queries.csv", root / "out"
            write(path, fixture())
            completed = subprocess.run([sys.executable, str(SCRIPT), "--input", str(path),
                                        "--expected-videos", "2", "--output", str(output)],
                                       check=True, capture_output=True, text=True)
            self.assertIn("WorldMem: user-reported", completed.stdout)
            with (output / "lookup_work.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertAlmostEqual(float(rows[-1]["growth"]), 1124.5 / 674.5)
            tex = (output / "lookup_work.tex").read_text()
            self.assertIn("135--180", tex)
            self.assertIn("45--60", tex)
            self.assertIn("not measured latency", tex)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["systems"][0]["queries"], 10)
            self.assertIn("not independently audited", provenance["systems"][1]["evidence"])


if __name__ == "__main__":
    unittest.main()
