# KEEPSAKE B32: One Native-Style Round-Trip Job

Run on Newton after syncing the code:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_keepsake_native_roundtrip.sbatch
```

One H100 allocation runs KEEPSAKE B32 only. No unbounded, FIFO, RI, training,
budget sweep, child jobs, Conda activation, GPU-mask rewriting or import timeout.
The twelve-hour request is a wall-time limit, not a runtime prediction.

## Scope

Ten generated videos: five starting views, each with a 90-degree round trip
(153 frames) and a 360-degree round trip (609 frames). These lengths follow
[MemCam Section IV-A](https://arxiv.org/html/2603.26193). Each path turns outward
then retraces the same camera poses. The 360 test is two full rotations in
opposite directions, not the existing inference CLI's single `360` rotation.

Defaults: 640x352, 30 FPS, 50 diffusion steps, CFG 5, generation seed 42,
released `dit_step20000.ckpt`, `slam_covisibility`, B32, pose/appearance weights
0.65/0.35, CPU archive. No tuning or selecting a favorable seed/subset.
Python, NumPy and Torch RNGs are reset to 42 for every video, including the
retriever's Monte Carlo sampling, so skipping completed videos does not advance
the random stream differently for subsequent cases. Bitwise GPU determinism
across machines/library versions is not guaranteed.

The default starting views are a seed-0 draw of five of our existing fifteen
60s manifest entries, ordered by output prefix before sampling. Only their
starting images, poses and prompts are used; the original wandering trajectory
is replaced, not replayed. Starting images are bicubic-resized as in inference.
The new helper retains the released local-Z rotation composition, but constructs
exactly symmetric paths without the legacy endpoint padding/asymmetric spacing.

The original paper's exact five test identities and evaluator have not been
verified. This is a native-style protocol reproduction, not an established
reproduction of its table values. Published numbers and these results cannot
by themselves establish a matched improvement. No automatic winner bolding or
paper-number insertion is performed. Supplying a manifest does not automatically
verify its identity against the authors' split.

To supply five explicit starting views instead of our default draw:

```bash
sbatch slurm/newton_keepsake_native_roundtrip.sbatch \
  --starts-manifest /absolute/path/to/five_starts.jsonl
```

Each JSONL row requires `scene`, integer `start_frame`, `input_image`, `pose_path`,
and `prompt`. Use five distinct scenes. `--dataset-root` can remap image and
pose paths. `--model-root` is the directory containing `models/`.

## Metrics

Save all generated RGB frames losslessly, with an MP4 for viewing. Compare
frame i with frame N-1-i for i=1..N//2-1. This gives 75 or 303 generated/generated
pairs per video; exclude the initial-image pair and the turnaround self-pair.
No matching by image similarity, pose tolerance, temporal warping or quality
threshold is used. Save the exact camera array and pair indices.

- PSNR: RGB peak 255; average frame-level dB; existing evaluator caps exact
  matches at 100 dB.
- SSIM: existing luma SSIM with OpenCV 11x11 Gaussian window, sigma 1.5 and
  its normal border handling. Require OpenCV; no global-SSIM fallback.
- LPIPS: AlexNet on full 640x352 RGB frames, normalized to [-1,1].
- First average pairs within each video, then weight the five videos equally.
- Final stored-item counts and rollout/peak frame-bank resource measurements
  are read from actual generation profiles; the post-update bank must obey B32.

These implementation details are explicit choices where the paper does not
specify its metric preprocessing, endpoint handling or aggregation. They are
recorded in `metric_protocol.json`, not asserted to be the authors' implementation.

The job also exports a **separate round-trip FVD diagnostic**: production
StyleGAN-V I3D, four 16-frame clips per leg per video, stride four, 224x224 resize.
The reference is the generated outward leg; the return leg is reversed into
matching pose order. Pool twenty clips per leg across five scenes before scoring.
This measures feature-distribution consistency, not GT fidelity. The paper's
FVD reference/clip procedure is not verified, so this value is not inserted into
the main LaTeX row or represented as its published FVD metric.

## Outputs and Resume

Default: `~/memcam_results/keepsake_native_roundtrip_b32/`

- `scores.csv`, `table.tex`, `row.tex`: both angles' PSNR, SSIM and LPIPS.
- `roundtrip_fvd_diagnostic.csv`: separately labeled FVD diagnostic.
- `per_video.csv`, per-case `pairs.csv`: detailed measurements.
- `plan.json`, `metric_protocol.json`: frozen identities, settings and hashes.
- `coverage.csv`, `status.json`, `last_error.json` if a failure occurs.
- `cases/*/attempt_*/`: MP4, lossless PNGs, poses, access traces and profiles.
- Per-case generation/metric receipts and cached I3D features.

Resubmit the same command to continue. Completed generations are verified using
input/code/checkpoint identity and output hashes, then skipped. Incomplete
attempts are preserved; only that case is regenerated in a new attempt directory.
Completed metric receipts are reused when their identities and hashes match.
Changed inputs, checkpoints or generation code require a new output directory.
An output-directory lock prevents two jobs from writing the same study.

Both full five-video groups must finish before the final table is exported.
A failed job is not reported as complete. The script cannot repair a cluster
CUDA allocation fault; it records the original CUDA initialization/kernel error.

Check progress (replace JOBID):

```bash
tail -n 40 -F keep_native32_JOBID.out keep_native32_JOBID.err
```

CPU-only planning is available via the same Python entry point with `--plan-only`.
It validates input files and checkpoint hashes but performs no model loading,
generation, or scoring. No Newton/GPU execution was performed during local tests.
