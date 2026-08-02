# Surprise Forcing Memory Baseline

This implementation adapts the external-memory component of
[Surprise Forcing](https://arxiv.org/abs/2607.18436) to MemCam. It does not use
the paper's adaptive denoising scheduler: that scheduler assumes a distilled
four-step LongLive sampler, while MemCam uses a 50-step Wan sampler.

## Implemented controller

For candidate descriptor `d` and bank descriptors `d_i`:

```text
s_pred = 0.5 * (1 - mean_i cosine(d, d_i))
s_nov  = 0.5 * (1 - max_i cosine(d, d_i))
s      = alpha * s_pred + (1 - alpha) * s_nov
```

The implementation then applies the paper's EMA normalization, feedback-updated
admission threshold, priority replacement, and cosine top-k routing. The default
parameters match the paper: `alpha=0.7`, EMA momentum `0.95`, controller step
`0.1`, target pass ratio `0.3`, initial threshold `0.002`, priority weights
`1.8/1.0/0.4`, route `k=3`, and three warmup sections.

## MemCam adaptation

- A descriptor is the spatial mean of value tokens from DiT block 15, L2
  normalized, from the conditional branch at the final denoising pass. The paper
  does not disclose its block.
- Each of MemCam's 19 generated latent-frame units is evaluated causally after a
  section. The corresponding RGB frame is the stored payload.
- Budget `B` includes the pinned sink frame, leaving `B-1` controller-managed
  slots. The one-frame autoregressive anchor is local state outside this bank.
- Surprise content routing selects up to three global memories. MemCam's common
  camera-overlap retriever then assigns those routed memories to target frames.
  Usage increases only for shortlisted memories that actually enter that context.
- The paper says usage and age are normalized to `[0,1]` without publishing the
  operator. This implementation divides each by the current maximum.
- The paper mentions near-duplicate filtering and neighboring-chunk throttling
  without specifying them. They are omitted instead of being guessed.
- EMA state starts at mean `0` and variance `1`; the paper does not disclose the
  initialization, and its three-section warmup reduces sensitivity to this choice.

## Pilot

The Newton job runs 10 videos at 30 seconds and budget 32:

```bash
sbatch slurm/newton_memcam_h100_30s_surprise_forcing.sbatch
```

Set `BUDGET=6` to reproduce the paper's long-video bank capacity rather than the
budget-32 policy comparison used in MemCam experiments.
