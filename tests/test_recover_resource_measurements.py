import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "paper/recover_resource_measurements.py"
SPEC = importlib.util.spec_from_file_location("resources", MODULE)
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


def profile_fixture():
    case = dict(id="scene_0000_90deg", frames=153, angle=90, start=dict(scene="scene"))
    common = dict(memory_policy="slam_covisibility", memory_budget=32, memory_bank_device="cpu",
                  scene="scene", angle=90, num_frames=153, peak_cuda_allocated_gb=16.,
                  peak_cuda_reserved_gb=18., peak_rss_gb=3., device_used_gb=70.)
    records = [dict(common, event="rollout_start")]
    for section in range(2):
        phases = dict(memory_policy_update=2., bank_store=.001)
        if section:
            phases["context_selection"] = 3.8
        records.append(dict(common, event="section_profile", section_idx=section, stored_memory_size=32,
                            section_end_frame=(section + 1) * 76, bank_frame_bytes=43253760,
                            bank_feature_bytes=1000, phase_latency_s=phases))
    records.append(dict(common, event="rollout_summary", completed=True, peak_bank_frame_bytes=43253760,
                        rollout_latency_s=300.))
    accesses = [dict(event="context_access", section_idx=s, target_frame=s * 76 + slot + 1,
                      selected=s > 0, candidate_count=31 if s else 0,
                      fallback_reason=None if s else "initial_section") for s in range(2) for slot in range(76)]
    return records, accesses, case


class ResourceRecoveryTest(unittest.TestCase):
    def test_initial_fallback_is_not_in_retrieval_denominator(self):
        result = resources.summarize_profile(*profile_fixture())
        self.assertEqual(result["retrieval_queries"], 76)
        self.assertEqual(result["retrieval_ms_per_query"], 50.)
        self.assertEqual(result["archive_rgb_peak_mib"], 41.25)
        self.assertEqual(result["descriptor_and_update_ms_per_section"], 2000.)

    def test_device_wide_memory_is_not_process_peak(self):
        result = resources.summarize_profile(*profile_fixture())
        self.assertEqual(result["torch_allocated_rollout_peak_gib"], 16.)
        self.assertFalse(any("device_used" in k for k in result))

    def test_partial_profile_rejected(self):
        records, accesses, case = profile_fixture()
        with self.assertRaises(ValueError):
            resources.summarize_profile(records[:-1], accesses, case)

    def test_duplicate_target_rejected(self):
        records, accesses, case = profile_fixture()
        accesses[-1] = copy.deepcopy(accesses[-2])
        with self.assertRaises(ValueError):
            resources.summarize_profile(records, accesses, case)

    def test_missing_timer_is_not_zero(self):
        records, accesses, case = profile_fixture()
        del records[1]["phase_latency_s"]["memory_policy_update"]
        with self.assertRaises(ValueError):
            resources.summarize_profile(records, accesses, case)

    def test_nan_or_wrong_policy_rejected(self):
        for key, value in (("peak_cuda_allocated_gb", float("nan")), ("memory_policy", "unbounded")):
            records, accesses, case = profile_fixture()
            records[1][key] = value
            with self.assertRaises(ValueError):
                resources.summarize_profile(records, accesses, case)

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text("{}")
            checksum = resources.sha(path)
            path.write_text("[]")
            with self.assertRaises(ValueError):
                resources.verify(path, checksum, {})

    def test_proxy_uses_saved_receipts_and_all_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = [dict(_row=i, scene=f"scene_{i}") for i in range(15)]
            plan = dict(study="keepsake_component_proxy_cpu_v1", items=items, configurations={"full": {}}, runtime={})
            (root / "plan.json").write_text(json.dumps(plan))
            for item in items:
                directory = root / "cells" / f"row_{item['_row']:03d}"
                directory.mkdir(parents=True)
                payload = dict(updates=[dict(setting="full", section_idx=s, scene=item["scene"], retained_count=32,
                                             update_ms=10. + item["_row"]) for s in range(24)])
                (directory / "replay.json").write_text(json.dumps(payload))
                receipt = dict(item=item, plan_sha256=resources.sha(root / "plan.json"), host="test", job="1",
                               artifacts={"/remote/cells/replay.json": resources.sha(directory / "replay.json")})
                (directory / "receipt.json").write_text(json.dumps(receipt))
            rows, summary, provenance = resources.proxy_updates(root)
            self.assertEqual(len(rows), 15)
            self.assertEqual(summary[0]["updates"], 360)
            self.assertEqual(summary[0]["mean_ms_per_update"], 17.)
            self.assertAlmostEqual(summary[0]["mean_total_s_per_history"], .408)
            self.assertEqual(len(provenance["sources"]), 31)


if __name__ == "__main__":
    unittest.main()
