# Resources Without New GPU Runs

No new GPU allocation, generation, model loading or replay is needed by
`paper/recover_resource_measurements.py`. It reads completed artifacts with the
Python standard library and verifies their recorded hashes. It never modifies
the input study or pretends to reconstruct unrecorded telemetry.

## Recovered Locally

### New Profile Bundle Audit

The subsequently downloaded `memcam_resource_profiles.tar.gz` contains 318
profiles, 317 complete. The audit and tables are in
`paper/results/resources_bundle_audit/`; its README details coverage and timer
boundaries. `paper/audit_resource_profile_bundle.py` reproduces the audit without
new GPU work. The bundle also recovers original fifteen-identity Unbounded versus
KEEPSAKE-B32 **logged elapsed times**: 106.46 versus 65.46 minutes at 60 seconds,
and 587.99 versus 193.14 minutes at 180 seconds. Both arms record 50 steps.
Hardware/software equivalence is absent from these logs, so these are historical
observations, not a controlled speedup benchmark.

Actual memory and phase timers exist for Unbounded's 30-second attention audit,
KEEPSAKE B128 at 60 seconds and numerous bounded baselines. The original
Unbounded/B32 60/180-second memory profiles are not in this bundle. Do not replace
the missing pair with these different experiments or relabel B128 as B32.

### Earlier Short Round-Trip Bundle

The downloaded `keepsake_native_roundtrip_b32` study contains ten completed GPU
profiles, not just archive-size counters. Its five starting views each have a
153-frame and a 609-frame round trip. Both use KEEPSAKE B32, CPU archive,
640x352, 50 denoising steps, seed 42, H100 PCIe, Torch 2.5.1, CUDA build 12.4.
These are approximately 5.1- and 20.3-second clips, not the 60/180-second cohort.
There is no matched unbounded arm in this experiment.

| Recorded quantity | 153 frames, five runs | 609 frames, five runs |
| --- | ---: | ---: |
| Maximum retained RGB payload | 41.25 MiB | 41.25 MiB |
| Maximum retained-feature logical payload | 1.594 MiB | 1.594 MiB |
| Maximum sampled host RSS | 2.534 GiB | 3.128 GiB |
| Maximum PyTorch allocated rollout peak | 16.306 GiB | 16.307 GiB |
| Maximum PyTorch reserved rollout peak | 18.527 GiB | 18.549 GiB |
| Mean retrieval time per query | 49.012 ms | 50.769 ms |
| Mean descriptor + update time per section | 1.685 s | 1.524 s |

Memory maxima are over all five runs in each protocol. Timing means weight each
run equally. Initial-section fallback targets are NOT retrieval queries:
there are 76 actual queries per short run and 532 per long run. The small number
of short runs is not independent evidence of long-horizon scaling or a speedup.
No observations are discarded as warm-up. First-call overhead remains included.

`paper/results/resources_existing_runs/` contains the CSVs, LaTeX table and
source hashes. The sources are the downloaded generation receipts, actual
profiles, access traces, plan and device diagnostics. Hash verification covers
the telemetry used here, not a fresh decoding/verification of every video.

## Measurement Boundaries

- RGB payload is tensor element count times element size at post-update bank
  snapshots. Descriptor payload is logical array bytes plus the existing scalar
  accounting. Neither includes Python containers, allocator overhead or larger
  backing arrays retained through descriptor views. They are not total RSS.
- Host RSS was sampled at rollout/section checkpoints. It is a maximum observed
  sample, not the process lifetime high-water mark or a continuous peak.
- GPU peaks are PyTorch allocated/reserved counters reset at rollout start.
  Setup and final video encoding are not comprehensively profiled. The logged
  device-wide `device_used_gb` is deliberately not relabeled as process usage.
- `memory_policy_update` includes image conversion, DINO/RGB/quality-feature
  extraction, scoring, buffer update, eviction, and eviction trace writing.
  Retained-RGB storage is separately timed. The old logs cannot split descriptor
  extraction from the remaining policy update retrospectively.
- These profiles measure the existing instrumented implementation. They do not
  establish an uninstrumented end-to-end speedup or matched baseline comparison.

## Recover CPU Update Costs On Newton

The sensitivity summary you already pasted reports **101.272 ms per update**
for its default, averaging 24 updates in each of 15 histories (about 2.431 s per
history). This times the replay affinity construction and retention decision,
not production DINO extraction or end-to-end generation. It is a different
timer from the production scorer/buffer proxy described below. Its exact source
is `keepsake_sensitivity_cpu_60s_n15/summary.csv`, `default.update_ms =
101.27184930671419`, in attachment
`91014678-cb96-4c04-bc35-9167cfc4b8af/Pasted text.txt`. That source was supplied as
text; the underlying per-update receipts have not been downloaded/verified here.

Your completed component proxy already timed the production scorer and buffer
update, on cached descriptors. This is separate from the GPU profile above:
no DINO extraction, RGB retention/copying, actual reader or generation occurs
inside that timed region. It is a single-pass replay observation, not a repeated
isolated microbenchmark. Do not subtract it from the GPU combined timer.

After transferring this script to Newton, run from any shell. No `sbatch` and
no GPU are needed; this only reads small saved JSON artifacts:

```bash
cd "$HOME/MemCam"
REPORT="$HOME/memcam_results/resource_report_$(date +%Y%m%d_%H%M%S)"
"$HOME/.conda/envs/memcam/bin/python" paper/recover_resource_measurements.py \
  --roundtrip-root "$HOME/memcam_results/keepsake_native_roundtrip_b32" \
  --proxy-root "$HOME/memcam_results/keepsake_component_proxy_cpu_60s_n15" \
  --output "$REPORT"
cat "$REPORT/cpu_update_summary.csv"
```

Omit `--roundtrip-root` to recover only CPU update timings. The script requires
all fifteen proxy trajectories, uses receipt-validated `replay.json` files, and
exports per-trajectory host/job identity, mean update milliseconds and total
update seconds per 60-second history. It does not rerun the fifteen-minute replay.
This CPU cost has not yet been read locally because that study's raw receipts
are on Newton, not in the downloaded round-trip bundle.

## Replacement Wording

The original sentence promises a complete matched resource evaluation that we
do not have. Use narrower wording grounded in the recovered data:

> Resource diagnostics on ten short KEEPSAKE-B32 round-trip rollouts record
> retained payload size, retrieval latency, combined descriptor-extraction and
> policy-update time, sampled host RSS, and PyTorch peak allocated/reserved GPU
> memory. These diagnostics are reported separately from the 60/180-second
> quality evaluations. They do not constitute a matched resource comparison with
> unbounded retention, establish constant total process memory, or demonstrate
> an end-to-end speedup.

Separate descriptor-only GPU time and matched 60/180-second process-memory
peaks remain unmeasured. Analytical payload bounds and CPU replay measurements
must not be presented as replacements for those missing quantities.
