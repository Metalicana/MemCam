# Downloaded Resource Profile Audit

Source: `memcam_resource_profiles.tar.gz`, downloaded from Newton. All numbers
below are read from recorded telemetry, not extrapolated from an archive-size
formula. No generation, replay or GPU allocation was performed for this audit.
Input hashes are in `provenance.json`.

## Original Unbounded Versus KEEPSAKE-B32 Runs

| Horizon | Matched output identities | Mean Unbounded elapsed | Mean KEEPSAKE-B32 elapsed |
| --- | ---: | ---: | ---: |
| 60 s | 15 | 106.46 min | 65.46 min |
| 180 s | 15 | 587.99 min | 193.14 min |

These are actual `time_sec` records from `baseline/run_status.jsonl` and
`slam_b32_covisibility/run_status.jsonl` under `context_memory_60s` and
`context_180s`. Both arms match output basenames, horizon, frame count and 50
denoising steps. The logs do not record hardware, software/code revision or
bank placement. They do not establish controlled speedup or retrieval-only
latency. Current runner code times video preparation, rollout, saving and
cleanup, excluding model initialization; historical code equivalence is not
recorded. `seed0_` is an output identity prefix, not proof of generator seed 0.

There is also a separate fifteen-identity `seed1_extra60s_` cohort: 109.33 versus
66.79 minutes. It is exported separately, not pooled into the original cohort.
All 45 paired identities are in `identity_matched_elapsed.csv`.

## Recorded Memory And Phase Timers

318 profile files were audited: 317 completed rollouts across 26 complete-run
groups, plus one incomplete GPU-bank RI run (28/71 sections), excluded from
numerical summaries. `profile_summary.csv` inventories all 27 groups.

The following rows are separate experiments, NOT matched comparisons. All have
CPU archives. Memory entries are maxima across completed runs; timing entries
are equal-run means of the logged phase totals over each video.

| Profile | Horizon | n | RGB MiB | Host RSS GiB | Allocated GiB | Reserved GiB | Retrieval s/video | Descriptor + update s/video |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbounded, attention audit | 30 s | 10 | 1176.914 | 4.330 | 15.987 | 18.320 | 632.663 | 0.002509 |
| KEEPSAKE-B128 | 60 s | 12 | 165.000 | 4.692 | 16.309 | 18.514 | 357.454 | 32.142 |
| k-center B32 | 60 s | 15 | 41.250 | 4.865 | 16.309 | 18.514 | 89.766 | 29.975 |
| MCE B32, lambda=1 | 60 s | 15 | 41.250 | 4.956 | 16.309 | 18.514 | 89.127 | 25.093 |

The tiny Unbounded update timer does not measure descriptor extraction: that
policy does not compute the bounded policies' descriptors in this update path.
Do not subtract times across these different experiments to estimate a
descriptor-only cost. First-call costs are included, not removed as warm-up.

## What This Does And Does Not Recover

| Requested quantity | Recovery from this bundle |
| --- | --- |
| Archive bytes | Recorded logical RGB and descriptor payloads for the profiled arms; not the original B32-vs-Unbounded 60/180s pair |
| Retrieval latency | Actual `context_selection` phase times for the profiled arms; original B32-vs-Unbounded 60/180s pair absent |
| Descriptor extraction | Not separately timed in legacy profiles |
| Update overhead | Combined descriptor/scoring/update/eviction timer; RGB bank storage separately timed |
| Host memory | Maximum sampled RSS at rollout/section checkpoints, not continuous process high-water mark |
| GPU memory | PyTorch allocated/reserved peaks reset at rollout start; not full process lifetime or `nvidia-smi` peaks |
| Matched hardware | Unverified: generation hardware/software identities absent from these records |

RGB/descriptor bytes exclude Python containers and allocator overhead. Logical
descriptor bytes may not capture larger backing arrays. Bank payload counters
describe post-update resident payload, not transient insertion peaks. GPU
rollout peaks exclude comprehensive model-setup/final-encoding coverage.
Device-wide `device_used_gb` is deliberately not reported as per-process memory.
The attention-audit workload is instrumented differently from ordinary runs.

The original `slam_b32_covisibility` and `baseline` directories have elapsed
records but no 60/180-second profile JSONLs in this archive. This does not prove
that no other file on Newton contains those measurements. No GPU time is needed
to inspect further existing logs, but the missing telemetry cannot be recovered
by arithmetic from these records. Separate short B32 round-trip profiles are
documented in `../resources_existing_runs/`; they are not substituted here.

## Files And Reproduction

- `profiles_per_video.csv`: every profile, with incomplete status preserved.
- `profile_summary.csv`: completed-run maxima and timing means by run directory.
- `identity_matched_elapsed.csv`, `elapsed_summary.csv`: paired elapsed records.
- `elapsed.tex`, `profiles.tex`, `preview.pdf`: explicitly scoped appendix tables.
- `status.json`, `provenance.json`: coverage and source hashes.

From the repository, using a fresh destination:

```bash
python paper/audit_resource_profile_bundle.py \
  --bundle "$HOME/Downloads/memcam_resource_profiles.tar.gz" \
  --output /tmp/memcam_resource_bundle_audit
```

The parser uses only Python's standard library and never extracts archive paths.
It rejects conflicting identities, duplicate completions, false completion
markers and missing phase timers rather than filling absent values with zeros.
