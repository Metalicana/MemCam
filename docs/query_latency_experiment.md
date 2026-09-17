# Query Latency Experiment

## MemCam

Run `paper/benchmark_query_latency.py` in the memcam environment on a compute
node. This is a retrieval-only replay, not another generation experiment.
The production FOV implementation explicitly uses CPU tensors. No GPU request,
CUDA preflight, image decoding, model checkpoints, or generation is needed.

Default inputs:

- `~/memcam_results/context_180s/unbounded_failure_decomposition_180s/tables/query_decomposition.csv`
- `testbeds/context_memory_180s/manifest.jsonl`
- `~/memcam_results/context_180s/{baseline,slam_b32_covisibility}/access_traces/`
- Pose JSON paths recorded in the manifest. Override with `--dataset-root` if needed.

The complete 15-trajectory diagnostic cohort and both sets of banks are checked
before timing starts. Missing queries, reconstruction/count mismatches, wrong
policies/budgets, or overrides fail rather than silently shrinking the cohort.
The benchmark does not need the generated video files.

Each of four equal-duration windows contributes six deterministically spaced
queries per trajectory: 360 paired queries total. Use `--queries-per-window 0`
to time every diagnostic query instead. Sampling never uses quality or latency.
Each policy/query gets one untimed warmup and three timed repeats. Query order
is shuffled with a recorded seed and policy order alternates across repeats.
RNG resets occur outside timing. The full production candidate loop runs for
every repeat, including 5,000-point sampling per candidate and argmax selection.
The replay does not cache overlap scores, vectorize the loop differently, or
extrapolate from a single pairwise comparison.

Timing uses `perf_counter_ns` around scoring and selection only. It excludes
pose/trace loading, generation, encoding, RGB transfers, bank construction,
eviction, and feature extraction. This measures isolated warm retrieval latency,
not end-to-end speedup or a complete accounting of memory-management overhead.
One CPU thread is the default; it is a declared benchmark setting, not an
assertion about thread counts used in historical generation jobs.

The original sampled winner need not recur: production retrieval uses Monte
Carlo draws. Logged winners validate bank membership; the experiment measures
execution time, not exact historical RNG reproduction or generation quality.

Outputs include raw repeated timings, the exact sampled query/candidate plan,
and four-window `latency_summary.csv`. Average repetitions within each query,
queries within each trajectory/window, then trajectories equally. Record CPU,
affinity, Torch version, thread count, seeds, and source hashes. Partial runs
retain raw timings but do not produce a completed summary for paper import.

Submit one CPU job from the repository (no Python job submission or conda init):

```bash
sbatch slurm/newton_query_latency_cpu.sbatch
```

The batch file directly invokes `~/.conda/envs/memcam/bin/python`. It requests
one CPU, 16 GB RAM, and a four-hour ceiling; that ceiling is not a runtime
prediction. Progress reports every paired query. Do not run this on a login node.
`--check-only` validates inputs and writes the query plan without benchmarking.

After completion:

```bash
python paper/build_lookup_work_table.py \
  --latency-summary "$HOME/memcam_results/query_latency_JOBID/latency_summary.csv" \
  --output "$HOME/memcam_results/unified_lookup_timed_JOBID"
```

The importer checks completion, summary hash, cohort, windows and policy names.
WorldMem cells remain unmeasured. The replay sample is smaller than the sample
used for the candidate-count means; both sampling schemes are preserved.

## WorldMem Handoff

Implement the same experiment in the WorldMem repository using its actual
production reader. Do not copy MemCam's CPU FOV implementation into WorldMem.

Use the confirmed 600-context-frame initialization and 600 generated frames at
10 FPS. Pair Unbounded and the real B32 banks at identical generated indices.
Sample six queries per trajectory from each 15-second window, independently of
quality. Keep all 15 trajectories. Load poses and reconstruct banks before timing.

Time the complete selection of eight memories, including any eight greedy
passes, masks, ranking, and sampling in the production reader. Do not report
candidate count times 10,000 as elapsed time. Keep device, dtype, FOV settings,
thread count and execution path fixed across policies. If it uses CUDA,
synchronize immediately before and after the timed reader call; report device
and whether host-device transfers are included. Do not optimize one policy's
reader differently or reuse previously computed overlap scores.

Use warmups, alternating policy order, repeated measurements, trajectory-level
aggregation, and provenance as above. Preserve slowdowns or null differences.
Export per-query times plus `system,duration_sec,start_sec,end_sec,unbounded_query_ms,ours_query_ms,unbounded_videos,ours_videos`
in a summary CSV and record the exact timing scope. Source-audit that adapter
before extending the unified table importer to WorldMem latency.
