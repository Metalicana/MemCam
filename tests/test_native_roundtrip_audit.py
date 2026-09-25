import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper import audit_native_roundtrip as audit


def events_for(poses, scene="synthetic"):
    bank, events = {0}, []
    for s in range((len(poses) - 1) // 76):
        metadata = dict(scene=scene, dataset_start_frame=0, duration_sec=(len(poses) - 1) / 30,
                        memory_policy="slam_covisibility", memory_budget=32, section_idx=s)
        if s:
            eligible = sorted(bank - set(range(s * 76 - 3, s * 76 + 1)))
            for q in range(s * 76 + 1, (s + 1) * 76 + 1):
                selected = eligible[int(np.argmin(audit.rotation_errors(poses, q, eligible)))]
                events.append(dict(metadata, event="context_access", selected=True, target_frame=q,
                                   selected_memory_frame=selected, candidate_count=len(eligible),
                                   stored_memory_size=32))
        candidates = bank | set(range(s * 76, (s + 1) * 76 + 1))
        bank = {0} | set(sorted(candidates - {0})[-31:])
        events += [dict(metadata, event="memory_eviction", evicted_memory_frame=i,
                        eviction_nearest_covisible_frame=i + 1) for i in sorted(candidates - bank)]
    return events


def case_fixture(root):
    poses = audit.exact_roundtrip_c2ws(np.eye(4), 90)
    folder = root / "synthetic_0000_90deg"
    attempt = folder / "attempt_1"
    attempt.mkdir(parents=True)
    np.save(attempt / "poses.npy", poses)
    (attempt / "video.mp4").write_bytes(b"not decoded in this test")
    (attempt / "profile.jsonl").write_text('{}\n')
    (attempt / "access.jsonl").write_text("\n".join(map(json.dumps, events_for(poses))) + "\n")
    case = dict(id=folder.name, frames=len(poses), angle=90, start=dict(scene="synthetic", start_frame=0))
    generation = dict(status="complete", case=case, attempt=attempt.name,
                      files={p.name: audit.digest(p) for p in attempt.iterdir()})
    (folder / "generation.json").write_text(json.dumps(generation))
    pairs = audit.roundtrip_pairs(153)
    audit.write_csv(folder / "pairs.csv", [dict(outward_frame=i, return_frame=j, psnr_db=20., ssim=.5, lpips=.25) for i, j in pairs])
    np.savez(folder / "roundtrip_features.npz", outward=np.arange(12).reshape(4, 3), returning=np.arange(12).reshape(4, 3) + 1)
    measurement = dict(key=dict(generation_sha256=hashlib.sha256(json.dumps(generation, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                                metric_identity=dict(fvd_config=dict(clip_length=16, frame_stride=4))),
                       files={name: audit.digest(folder / name) for name in ("pairs.csv", "roundtrip_features.npz")},
                       fvd_clip_pairs=[[pairs[start + k * 4] for k in range(16)] for start in (0, 5, 9, 14)])
    (folder / "metrics.json").write_text(json.dumps(measurement))
    return folder


class NativeRoundtripAuditTests(unittest.TestCase):
    def test_complete_reads_respect_continuation_exclusion_in_history_oracle(self):
        for angle in (90, 360):
            poses = audit.exact_roundtrip_c2ws(np.eye(4), angle)
            rows, banks = audit.retrieval_rows(events_for(poses), poses, angle)
            self.assertEqual(len(rows), len(poses) - 77)
            returns = [r for r in rows if r["phase"] == "return"]
            self.assertTrue(all(r["full_best_rotation_deg"] < 1e-5 for r in returns[3:]))
            self.assertTrue(all(0 < r["full_best_rotation_deg"] < 4 for r in returns[:3]))
            self.assertTrue(all(abs(r["selection_rotation_gap_deg"]) < 1e-5 for r in returns))
            self.assertGreater(max(r["retention_rotation_gap_deg"] for r in returns), 10)
            self.assertTrue(all(len(bank) <= 32 for bank in banks.values()))

    def test_rejects_duplicate_missing_and_invalid_queries(self):
        poses = audit.exact_roundtrip_c2ws(np.eye(4), 90)
        events = events_for(poses)
        index = next(i for i, e in enumerate(events) if e["event"] == "context_access")
        duplicate = events + [copy.deepcopy(events[index])]
        missing = events[:index] + events[index + 1:]
        bad_count = copy.deepcopy(events)
        bad_count[index]["candidate_count"] += 1
        future = copy.deepcopy(events)
        future[index]["selected_memory_frame"] = 140
        for bad in (duplicate, missing, bad_count, future):
            with self.assertRaises(ValueError):
                audit.retrieval_rows(bad, poses, 90)

    def test_rejects_wrong_pose_sequence(self):
        poses = audit.exact_roundtrip_c2ws(np.eye(4), 90)
        events = events_for(poses)
        poses[10] = poses[11]
        with self.assertRaises(ValueError):
            audit.retrieval_rows(events, poses, 90)

    def test_updates_count_simultaneous_neighbor_loss(self):
        poses = audit.exact_roundtrip_c2ws(np.eye(4), 360)
        updates = audit.update_rows(events_for(poses), len(poses))
        self.assertEqual(len(updates), 8)
        self.assertEqual(updates[0]["evicted"], 45)
        self.assertEqual(updates[0]["nearest_also_evicted"], 44)
        self.assertTrue(all(u["new_frames_kept"] == 31 for u in updates))
        events = events_for(poses)
        events.append(copy.deepcopy(next(e for e in events if e["event"] == "memory_eviction")))
        with self.assertRaises(ValueError):
            audit.update_rows(events, len(poses))

    def test_receipts_pairing_and_section_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = case_fixture(Path(directory))
            record = audit.load_case(folder)
            section = audit.section_rows(record)
            self.assertEqual(len(section), 1)
            self.assertEqual(section[0]["queries"], 76)
            self.assertEqual(section[0]["lpips"], .25)
            self.assertEqual(record["queries"][-1]["lpips"], "")
            (folder / "pairs.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Changed metric artifact"):
                audit.load_case(folder)

    def test_rejects_wrong_generation_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = case_fixture(Path(directory))
            path = folder / "metrics.json"
            measurement = json.loads(path.read_text())
            measurement["key"]["generation_sha256"] = "different generation"
            path.write_text(json.dumps(measurement))
            with self.assertRaisesRegex(ValueError, "different generation receipt"):
                audit.load_case(folder)

    def test_cached_fvd_math_and_no_source_overwrite(self):
        a = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=float)
        self.assertAlmostEqual(audit.frechet(a, a), 0, places=6)
        self.assertAlmostEqual(audit.frechet(a, a + [2, 3]), 13, places=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "separate"):
                audit.run(root, root / "audit", render=False)


if __name__ == "__main__":
    unittest.main()
