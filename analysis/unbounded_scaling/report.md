# Unbounded MemCam Scaling Estimate

These values are implementation-derived estimates, not long-horizon measurements.

| Video duration | Bank storage | Overlap evaluations | Retrieval time | Total rollout time |
|---:|---:|---:|---:|---:|
| 10s | 0.38 GB | 33,972 | 16 s | 11 min |
| 40s | 1.53 GB | 689,700 | 5 min | 48 min |
| 60s | 2.30 GB | 1,588,932 | 12 min | 1.3 h |
| 10m | 22.7 GB | 161,477,808 | 20.9 h | 31.5 h |
| 60m | 136.0 GB | 5,835,347,868 | 31.5 days | 34.1 days |

Memory follows directly from the unbounded bank retaining every decoded BF16 RGB frame. Latency combines fixed per-section denoising time with the current exhaustive retrieval loop, which compares every candidate frame against every context target.

The latency estimate excludes model loading, VAE/context encoding outside the timed denoising loop, and final video encoding. CPU overlap-call timing is hardware-sensitive.
