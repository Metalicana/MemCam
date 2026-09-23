# Random-Five FVD and PSNR

The primary pool is the existing fifteen matched 60-second MemCam trajectories.
The subset is selected once with `random.Random(0).sample`, without replacement,
after sorting `output_prefix`. All six methods use the same five identities:
Unbounded, FIFO, MCE, K-center, RI and KEEPSAKE, with B32 for bounded methods.
This is a custom exploratory subset, not the Context-as-Memory authors' test
split. The script neither searches for favorable subsets nor drops missing runs.

## PSNR Without a GPU

After syncing the code to Newton:

```bash
cd "$HOME/MemCam"
"$HOME/.conda/envs/memcam/bin/python" utils/evaluate_random5_quality.py \
  --psnr-only --saved-only --psnr-cohort all15
```

This reads the per-video `psnr_db` already computed alongside LPIPS, using
the source paths in `paper/metric_results_60s/scores.csv`. It checks method,
scene, start frame, duration, video filename, complete frame count, stride and
sample count. It does not establish historical video hashes. Missing, short,
duplicate or conflicting records are not averaged away.

It prints six PSNR means and exports `psnr_all15.csv` and `psnr_all15.tex`.
For the random five alone, use `--psnr-cohort subset` instead.
If compatible saved PSNR is missing, omit `--saved-only` inside an existing
CPU allocation to compute it from the videos. No nested `srun` is needed.

PSNR follows the existing evaluator: RGB at the generated video's resolution,
peak value 255, one frame every 30 frames throughout the full rollout, including
frame zero. GT uses `start_frame + generated_index`, with bicubic resizing only
if needed. Average frame-level dB within each video, then weight videos equally.
The production evaluator caps exact matches at 100 dB; these are not full-frame-rate
or luminance-only PSNR values, and are not generated-to-generated revisit scores.

## One Job for FVD and PSNR

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_random5_quality.sbatch
```

The job computes both metrics for the random five and all six methods. Add
`--psnr-cohort all15` after the script name to also export fifteen-video PSNR;
the combined FVD/PSNR table still uses only the same five videos for both metrics.

One GPU allocation, no generation, no child jobs, no Conda activation and no
import/preflight timeout. It preserves Slurm's GPU mask. The four-hour allocation
limit is not a runtime estimate. FVD features are extracted with the production
I3D detector for these six runs; previous RI/KEEPSAKE-only caches are not imported.

FVD uses four 16-frame clips per video, stride four, uniformly positioned across
the complete rollout, resized to 224 pixels. FVD is computed from all twenty
clips together, not by averaging per-video FVD scores. Each method must supply
all five videos and twenty clips. GPU failure does not trigger a CPU fallback.

Default output:
`~/memcam_results/random5_quality_60s_seed0/`

- `cohort.json`: the frozen source pool signature, selected identities and runs.
- `scores.csv`, `table.tex`: five-video FVD and PSNR for all six methods.
- `psnr_random5.csv/.tex` or `psnr_all15.csv/.tex`: selected PSNR cohort.
- `psnr_per_video.csv`, `psnr_provenance.json`: PSNR sources and definition.
- `fvd_progress.json`, `fvd_provenance.json`, `device_diagnostics.json`, `status.json`.

The table bolds the measured winners, not a predetermined method. Five-video
results remain a labeled subset analysis; they do not replace the full-suite
comparison or establish that KEEPSAKE will win. No uncertainty is inferred from
the saved aggregate scores.
