import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from paper import benchmark_b32_query_latency as bench


def fixture(root):
    count = 305
    pose = root / "poses.json"
    pose.write_text(json.dumps({"CineCameraActor": {str(i): dict(position=[i, 0, 0], rotation=[0, 0, 0])
                                                  for i in range(count)}}))
    item = dict(scene="fixture", start_frame=0, duration_sec=60, fps=30, num_frames=count,
                pose_path=str(pose), output_prefix="seed0_fixture_60s_")
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(item) + "\n")
    for _, run, policy, budget in bench.RUNS:
        bank, events = {0}, []
        for s in range(4):
            if s:
                candidates = sorted(bank - set(range(s * 76 - 3, s * 76 + 1)))
                for slot in range(76):
                    events.append(dict(event="context_access", selected=True, section_idx=s,
                                       target_frame=s * 76 + slot + 1, selected_memory_frame=candidates[0],
                                       candidate_count=len(candidates), stored_memory_size=len(bank),
                                       memory_policy=policy, memory_budget=budget,
                                       scene="fixture", dataset_start_frame=0, duration_sec=60))
            bank.update(range(s * 76, s * 76 + 77))
            if budget:
                keep = set(sorted(bank)[-budget:])
                for index in sorted(bank - keep):
                    events.append(dict(event="memory_eviction", section_idx=s, evicted_memory_frame=index))
                bank = keep
        path = root / run / "access_traces" / "seed0_fixture_60s_custom.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return SimpleNamespace(manifest=manifest, root=root, expected_videos=1,
                           dataset_root=None, queries_per_section=2)


class B32LatencyTests(unittest.TestCase):
    def test_complete_six_policy_trace_adapter_and_final_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            _, cases, _, _, archives = bench.prepare(args)
            self.assertEqual(len(cases), 6)
            self.assertEqual([c["target_frame"] for c in cases[:2]], [96, 134])
            self.assertEqual([r["final_stored_frames"] for r in archives], [305, 32, 32, 32, 32, 32])
            self.assertEqual(set(cases[0]["candidates"]), {r[1] for r in bench.RUNS})
            trace = args.root / "fifo_b32/access_traces/seed0_fixture_60s_custom.jsonl"
            lines = trace.read_text().splitlines()
            # Missing a real read fails even if that slot would not be timed.
            lines = [s for s in lines if json.loads(s).get("target_frame") != 77]
            trace.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "Query coverage"):
                bench.prepare(args)

    def test_aggregation_complete_records_equal_trajectory_weight(self):
        cases = [dict(row=row, section_idx=1, target_frame=t)
                 for row, targets in ((0, [1, 2, 3]), (1, [1])) for t in targets]
        archives = [dict(row=row, run=run, final_stored_frames=305 if budget is None else 32)
                    for row in (0, 1) for _, run, _, budget in bench.RUNS]
        records = [dict(**case, run=run, repeat=repeat, query_ms=2 if case["row"] == 0 else 10)
                   for case in cases for _, run, _, _ in bench.RUNS for repeat in (0, 1)]
        summary, _ = bench.summarize(records, cases, archives, 2)
        self.assertTrue(all(r["query_ms"] == 6 and r["videos"] == 2 for r in summary))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            bench.summarize(records[:-1], cases, archives, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bench.summarize(records + [records[0]], cases, archives, 2)

    def test_invalid_final_evictions(self):
        with self.assertRaises(ValueError):
            bench.final_bank_size([], 305, 32)
        with self.assertRaises(ValueError):
            bench.final_bank_size([dict(event="memory_eviction", section_idx=0, evicted_memory_frame=1000)], 305, None)

    def test_production_cpu_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = fixture(root)
            output = root / "output"
            result = subprocess.run([
                sys.executable, str(bench.ROOT / "paper/benchmark_b32_query_latency.py"),
                "--manifest", str(args.manifest), "--root", str(root), "--expected-videos", "1",
                "--queries-per-section", "1", "--repeats", "1", "--output", str(output),
            ], capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            meta = json.loads((output / "provenance.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["timing_records"], 18)
            self.assertEqual(meta["summary_sha256"], bench.digest(output / "latency_summary.csv"))
            self.assertIn("KEEPSAKE", (output / "latency_summary.csv").read_text())
            raw = [json.loads(s) for s in (output / "query_timings.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["query_ms"] > 0 for r in raw))


if __name__ == "__main__":
    unittest.main()
