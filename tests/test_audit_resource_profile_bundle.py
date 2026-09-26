import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "bundle", Path(__file__).resolve().parents[1] / "paper/audit_resource_profile_bundle.py")
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def fixture():
    common = dict(scene="scene", dataset_start_frame=0, num_frames=153, total_sections=2,
                  memory_policy="slam_covisibility", memory_budget=32, memory_bank_device="cpu",
                  duration_sec=5, peak_rss_gb=3, peak_cuda_allocated_gb=16,
                  peak_cuda_reserved_gb=18, device_used_gb=70)
    rows = [dict(common, event="rollout_start")]
    for s in range(2):
        phases = dict(memory_policy_update=2, bank_store=.01)
        if s:
            phases["context_selection"] = 3.8
        rows.append(dict(common, event="section_profile", section_idx=s, section_end_frame=(s+1)*76,
                         bank_frame_bytes=43253760, bank_feature_bytes=1000,
                         stored_memory_size=32, phase_latency_s=phases))
    rows.append(dict(common, event="rollout_summary", sections=2, completed=True,
                     peak_bank_frame_bytes=43253760, rollout_latency_s=300))
    return rows


def elapsed_fixture(policy, budget, prefix="seed0_"):
    return [dict(output=f"/run/{prefix}scene{i}.mp4", status="completed", duration_sec=60,
                 num_frames=1825, steps=50, memory_policy=policy, memory_budget=budget,
                 time_sec=100+i) for i in range(15)]


class BundleAuditTest(unittest.TestCase):
    def test_profile_measurements_and_scope(self):
        row = bundle.summarize_profile(fixture(), "run/profiles/x.jsonl")
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["context_selection_mean_s_per_section"], 3.8)
        self.assertEqual(row["memory_policy_update_total_s"], 4)
        self.assertEqual(row["torch_allocated_rollout_peak_gib"], 16)
        self.assertNotIn("device_used_gb", row)
        self.assertNotIn("descriptor_extraction_s", row)

    def test_interrupted_run_has_no_reported_peaks(self):
        row = bundle.summarize_profile(fixture()[:-1], "run/profiles/x.jsonl")
        self.assertEqual(row["status"], "incomplete")
        self.assertNotIn("torch_allocated_rollout_peak_gib", row)

    def test_false_completion_rejected(self):
        rows = fixture()
        del rows[-2]
        with self.assertRaises(ValueError):
            bundle.summarize_profile(rows, "run/profiles/x.jsonl")

    def test_missing_phase_rejected(self):
        rows = fixture()
        del rows[1]["phase_latency_s"]["memory_policy_update"]
        with self.assertRaises(ValueError):
            bundle.summarize_profile(rows, "run/profiles/x.jsonl")

    def test_extra_cohort_not_pooled_and_hardware_not_assumed(self):
        a = elapsed_fixture("unbounded", None)
        b = elapsed_fixture("slam_covisibility", 32)
        a += elapsed_fixture("unbounded", None, "seed1_extra60s_")
        rows = bundle.match_elapsed(a, b, 60)
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(r["hardware_matching"] == "unverified" for r in rows))

    def test_duplicate_completed_output_rejected(self):
        a = elapsed_fixture("unbounded", None)
        a.append(copy.deepcopy(a[0]))
        with self.assertRaises(ValueError):
            bundle.match_elapsed(a, elapsed_fixture("slam_covisibility", 32), 60)

    def test_frame_or_cohort_mismatch_rejected(self):
        a = elapsed_fixture("unbounded", None)
        b = elapsed_fixture("slam_covisibility", 32)
        b[0]["num_frames"] = 913
        with self.assertRaises(ValueError):
            bundle.match_elapsed(a, b, 60)
        with self.assertRaises(ValueError):
            bundle.match_elapsed(a[:-1], b, 60)


if __name__ == "__main__":
    unittest.main()
