# RI versus KEEPSAKE: 30-Video FVD

One evaluation job, one GPU, no generation and no other metrics:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_fvd_ri_keepsake_30.sbatch
```

The fixed input is `testbeds/context_memory/manifest_60s_30.jsonl`, previously
created by the plus-15 generation study. The runner requires all 30 listed
videos in both `context_memory_60s/ri_b32_dino_rgb` and
`context_memory_60s/slam_b32_covisibility`. It never selects a favorable subset
or silently intersects incomplete folders. The original 15 entries come from
`testbeds/context_memory/manifest.jsonl`, filtered to 60 seconds, and must be
present with unchanged frame/GT mapping. Additional directory files are ignored.

For a file/GT-availability audit without importing Torch:

```bash
"$HOME/.conda/envs/memcam/bin/python" utils/compare_fvd_matched.py \
  --audit-only --output "$HOME/memcam_results/fvd30_input_audit"
```

The evaluator uses the existing `FVDRunner`: StyleGAN-V I3D features, four
uniformly spaced 16-frame clips per video, stride four, 224-pixel inputs, and
the entire manifest frame count (typically 1,825 for the nominal 60s suite).
No LPIPS, VBench, diffusion or policy tuning is run. The I3D checkpoint and
evaluator hashes are recorded; no historical features are reused blindly.

Each matched video saves generated RI/KEEPSAKE and GT features together, with
source-video hashes and hashes of the sampled GT images. Resuming the same
output verifies those signatures and reuses complete feature files. Video FPS,
frame count and successful decoding of all sampled clips are checked before
scoring. Manifest correspondence is not independent verification of historical
generation checkpoint, inference steps or random seed; those remain a separate
generation-provenance audit. A manifest's `split_seed` is not the diffusion seed.

Outputs go to `~/memcam_results/fvd_ri_keepsake_30_JOBID/`:

- `scores.csv` and `summary.json`: RI FVD, KEEPSAKE FVD and their paired difference
  on all 30, the original 15, and the additional 15 separately.
- `*_bootstrap_differences.npy`: 2,000 paired bootstrap differences, seed 17.
- `cohort.json`, `provenance.json`, `status.json`, and per-video feature receipts.

The difference is **KEEPSAKE minus RI**; negative favors KEEPSAKE. Bootstrap
draws resample whole scenes, retaining all starts/seeds and all four clips
within each sampled scene, using identical indices for both methods and GT.
With one video per scene this is a paired video bootstrap. The 2.5/97.5
percentiles describe sampling sensitivity, not an equivalence test or correction
for finite-sample FVD bias. Including zero does not prove the methods are equal
or prove a particular observed difference is noise.

FVD is recomputed on each complete sampled feature distribution, not averaged
over per-video FVDs. Bootstrap computation uses an exact low-rank SVD identity
for the same covariance distance, without dimensionality truncation. Point
estimates use the existing production covariance implementation; the code checks
agreement with the SVD identity before resampling. Compare methods within each
cohort: a raw score change from 15 to 30 can also reflect sample-size dependence.
See the [original FVD paper](https://research.google/pubs/towards-accurate-generative-models-of-video-a-new-metric-challenges/)
for the distribution-level metric.

The Slurm launcher uses the environment's Python directly. It does not run
`conda activate`, use nested GPU steps, override `CUDA_VISIBLE_DEVICES`, or impose
an import/preflight timeout. The allocation wall-time limit is four hours, not
a runtime estimate. CUDA failures remain allocation/environment errors.

## Job 839413: CUDA Unavailable on evc27

The fixed-cohort audit passed: 30 videos per method, 30 scenes, original 15
included, sampled GT files present. The job stopped at `torch.cuda.is_available()`
before loading I3D or extracting any features. This was not a timeout or a
missing-video failure. The boolean check did not identify whether the failure
was in the driver, GPU access/mapping, or the Python installation.

The runner now records the Python executable, Torch path/version/CUDA build,
Slurm device variables and library path in `device_diagnostics.json`. It calls
CUDA initialization explicitly and exercises a tensor kernel, preserving the
underlying exception and marking `status.json` as failed at `cuda_setup` if
either fails. It does not switch to CPU or alter the GPU visibility mask.

The launcher prints NVIDIA inventory and loaded modules, and sets PCI bus
ordering following the [Slurm GRES guide](https://slurm.schedmd.com/gres.html).
Its default exclusions now include `evc27` as a temporary scheduling workaround,
not a confirmed diagnosis or repair. If CUDA fails elsewhere, use the recorded
initialization error and inventory to investigate the shared installation and
allocation with Newton support rather than repeatedly excluding more nodes.
The FVD inputs, parameters and statistical comparison are unchanged.
