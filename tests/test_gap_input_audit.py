import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gap_audit", ROOT / "paper/audit_gap_inputs.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def events(policy, budget):
    bank, result = {0}, []
    for section in range(3):
        if section:
            eligible = bank - set(range(section*76-3, section*76+1))
            for slot in range(76):
                result.append(dict(event="context_access", selected=True, section_idx=section,
                                   target_frame=section*76+slot+1, selected_memory_frame=0,
                                   memory_policy=policy, memory_budget=budget, stored_memory_size=len(bank),
                                   candidate_count=len(eligible), scene="test", dataset_start_frame=0,
                                   duration_sec=60))
        bank.update(range(section*76, section*76+77))
        if budget and len(bank) > budget:
            keep = {0} | set(sorted(bank - {0})[-(budget-1):])
            result += [dict(event="memory_eviction", section_idx=section, evicted_memory_frame=i)
                       for i in bank - keep]
            bank = keep
    return result


def write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


ITEM = dict(scene="test", start_frame=0, duration_sec=60, num_frames=229, output_prefix="seed0_test_60s_")


class GapInputAuditTests(unittest.TestCase):
    def test_validates_all_reads_and_bank_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = events("slam_covisibility", 32)
            write_trace(path, rows)
            result = audit.audit_trace(path, ITEM, "slam_covisibility", 32)
            self.assertEqual(result["reads"], 152)
            self.assertEqual(result["sampled_reads"], 8)
            bad = [dict(r) for r in rows]
            next(r for r in bad if r["event"] == "context_access")["selected_memory_frame"] = 900
            write_trace(path, bad)
            with self.assertRaises(ValueError):
                audit.audit_trace(path, ITEM, "slam_covisibility", 32)

    def test_missing_duplicate_identity_and_wrong_policy_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = events("unbounded", None)
            variants = [rows[:-1], rows + [rows[0]]]
            for field, value in (("scene", "other"), ("memory_policy", "fifo"), ("candidate_count", 99)):
                bad = [dict(r) for r in rows]
                bad[0][field] = value
                variants.append(bad)
            for bad in variants:
                write_trace(path, bad)
                with self.assertRaises(ValueError):
                    audit.audit_trace(path, ITEM, "unbounded", None)

    def test_feature_shapes_norms_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            features = np.zeros((230, 768), dtype=np.float32)
            features[:, 0] = 1
            np.save(path, features)
            self.assertEqual(audit.audit_cache(path, 229)["extra_frames"], 1)
            for invalid in (features[:10], features[:, :12], features * 2,
                            np.full_like(features, np.nan)):
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    audit.audit_cache(path, 229)

    def test_full_grid_cli_does_not_generate_or_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(ITEM) + "\n")
            for run, policy, budget in audit.CONFIGS:
                folder = root / "context_memory_60s" / run
                write_trace(folder / "access_traces/seed0_test_60s_custom.jsonl", events(policy, budget))
                (folder / "seed0_test_60s_custom.mp4").write_bytes(b"not decoded in this audit")
            features = np.zeros((229, 768), dtype=np.float32)
            features[:, 0] = 1
            for kind in ("gt", "baseline"):
                folder = root / "feature_cache" / kind
                folder.mkdir(parents=True)
                np.save(folder / "seed0_test_60s_dino.npy", features)
            output = root / "audit"
            result = subprocess.run([sys.executable, str(ROOT / "paper/audit_gap_inputs.py"),
                                     "--root", str(root), "--manifest", str(manifest),
                                     "--expected-videos", "1", "--output", str(output)],
                                    check=True, capture_output=True, text=True)
            self.assertIn("All policy traces valid: YES", result.stdout)
            report = json.loads((output / "audit.json").read_text())
            self.assertEqual(len(report["traces"]), 21)
            self.assertEqual(report["complete_cache_roots"], [str(root / "feature_cache")])
            self.assertEqual(len(report["caches"]), 2)
            self.assertEqual((output / "issues.csv").read_text().strip(), "kind,path,error")


if __name__ == "__main__":
    unittest.main()
