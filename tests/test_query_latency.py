import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("query_latency", ROOT / "paper/benchmark_query_latency.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def fixture(root):
    pose = root / "poses.json"
    pose.write_text(json.dumps({"CineCameraActor": {str(i): {"position": [i, 0, 0], "rotation": [0, 0, 0]}
                                                     for i in range(609)}}))
    item = dict(scene="test", start_frame=0, duration_sec=8, fps=76, num_frames=609,
                pose_path=str(pose), output_prefix="seed0_test_8s_")
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(item) + "\n")
    queries = []
    for section in range(1, 8):
        queries.append(dict(run_name="baseline", row=0, scene="test", dataset_start_frame=0,
                            duration_sec=8, section_idx=section, target_frame=76*section + 1,
                            candidate_count=76*section - 3, traced_candidate_count=76*section - 3,
                            candidate_count_mismatch=0))
    query_path = root / "queries.csv"
    bench.write_csv(query_path, queries)
    for name, policy, budget in (("baseline", "unbounded", None),
                                 ("slam_b32_covisibility", "slam_covisibility", 32)):
        bank, events = {0}, []
        for s in range(8):
            if s:
                candidates = bank - set(range(s*76-3, s*76+1))
                events.append(dict(event="context_access", selected=True, section_idx=s,
                                   target_frame=s*76+1, selected_memory_frame=0,
                                   candidate_count=len(candidates), stored_memory_size=len(bank),
                                   memory_policy=policy, memory_budget=budget, scene="test",
                                   dataset_start_frame=0, duration_sec=8))
            bank.update(range(s*76, s*76+77))
            if budget:
                keep = {0} | set(sorted(bank - {0})[-31:])
                for index in sorted(bank - keep):
                    events.append(dict(event="memory_eviction", section_idx=s, evicted_memory_frame=index))
                bank = keep
        path = root / name / "access_traces" / "seed0_test_8s_custom.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return manifest, query_path, item, queries


class QueryLatencyTests(unittest.TestCase):
    def test_sampling_independent_of_metrics_and_all_mode(self):
        rows = [dict(target_frame=i) for i in range(1, 81)]
        chosen = bench.sample_queries(rows, 8, 10, 2)
        self.assertEqual(len(chosen), 8)
        self.assertEqual([w for w, _ in chosen], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(len(bench.sample_queries(rows, 8, 10, 0)), 80)
        with self.assertRaises(ValueError):
            bench.sample_queries(rows[:10], 8, 10, 2)

    def test_real_trace_reconstruction_and_mismatch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, item, queries = fixture(root)
            path = root / "slam_b32_covisibility/access_traces/seed0_test_8s_custom.jsonl"
            banks = bench.trace_banks(path, item, queries, "slam_covisibility", 32)
            self.assertEqual(banks[1], [0] + list(range(46, 73)))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            event = next(e for e in events if e["event"] == "context_access")
            event["candidate_count"] += 1
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
            with self.assertRaises(ValueError):
                bench.trace_banks(path, item, queries, "slam_covisibility", 32)

    def test_loop_uses_production_parameters_and_first_tie(self):
        calls = []
        def overlap(a, b, **kwargs):
            calls.append((a, b, kwargs))
            return .5
        winner, score = bench.retrieve(["zero", "one", "two"], 2, [0, 1], overlap)
        self.assertEqual((winner, score), (0, .5))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][2], dict(fov_half_h=45., fov_half_v=30., num_samples=5000,
                                         radius=50., return_details=False))

    def test_summary_weights_trajectories_not_query_counts(self):
        records = []
        for w in range(4):
            for p in ("ours", "unbounded"):
                for row, values in ((0, [2, 2, 2]), (1, [10])):
                    for target, value in enumerate(values):
                        for repeat in range(3):
                            records.append(dict(window=w, policy=p, row=row, target_frame=target,
                                                query_ms=value, repeat=repeat))
        self.assertEqual(bench.summarize(records, 180)[0]["ours_query_ms"], 6)

    def test_cpu_cli_and_table_import_with_production_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, query_path, _, _ = fixture(root)
            output = root / "benchmark"
            command = [sys.executable, str(ROOT / "paper/benchmark_query_latency.py"),
                       "--queries", str(query_path), "--manifest", str(manifest), "--root", str(root),
                       "--duration", "8", "--fps", "76", "--expected-videos", "1",
                       "--queries-per-window", "1", "--repeats", "1", "--output", str(output)]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            self.assertIn("4 paired queries", result.stdout)
            meta = json.loads((output / "provenance.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["timing_records"], 8)
            raw = [json.loads(line) for line in (output / "query_timings.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["query_ms"] > 0 for r in raw))
            table = bench.load_file_module("table_import", ROOT / "paper/build_lookup_work_table.py")
            memcam = table.memcam_counts(query_path, "baseline", 8, 76, 1)
            rows = table.table_rows([memcam])
            table.attach_latency(rows, output / "latency_summary.csv", memcam)
            self.assertTrue(all(r["ours_query_ms"] > 0 for r in rows))
            self.assertNotIn("& TBD & TBD", table.latex(rows))
            meta["status"] = "running"
            (output / "provenance.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                table.attach_latency(rows, output / "latency_summary.csv", memcam)


if __name__ == "__main__":
    unittest.main()
