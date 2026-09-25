# Full-Cohort Component Ablation

This is the follow-up authorized on 2026-09-25: all fifteen matched 60-second
MemCam trajectories, five settings, B32, across resumable jobs. With the user's
subsequent concurrency approval, the recommended launcher is a fifteen-task array
capped at five concurrent tasks, **one H100 per task**, not a five-GPU allocation.
It does not overwrite or relabel the completed three-scene 30-second pilot.
It tests components; it does not tune alpha or promise that the full method wins.

## One Submission Command

After transferring the new code to Newton, from a login shell:

```bash
cd "$HOME/MemCam"
"$HOME/.conda/envs/memcam/bin/python" paper/submit_keepsake_component_study.py \
  --array-parallel 5
```

Default output: `~/memcam_results/keepsake_components_60s_n15/`.
CPU preparation freezes the full cohort, source hashes, pose-defined query labels
and a code snapshot before any generation. It verifies the four production model
files, sampled GT coverage and the cached production I3D detector at
`~/hf_cache/memcam_fvd/i3d_torchscript.pt`. It then submits an array with
`--array=0-14%5` and one dependent CPU reporting job. Each task runs all five
variants on one scene. Each task uses `highgpu`, one `nvidia_h100_80gb_hbm3`, eight
CPUs, 96G host memory, and a twelve-hour limit. Tasks can run on different nodes.
The `%5` cap limits concurrent tasks, not GPUs within each allocation; see
[Slurm's array documentation](https://slurm.schedmd.com/job_array.html).

There are **75 new videos**, including freshly generated full-method controls.
Scaling the completed 30-second pilot suggests roughly **55 GPU-hours plus
overhead** total, or roughly **eleven compute hours with five GPUs continuously
available**. Queue delays, scene imbalance, model loading and the final report
are additional. The user's fifteen-hour deadline is plausible, not guaranteed.
Individual scenes can take different times; the per-task twelve-hour limit is
not the wall time of the whole study.
No Newton submission or real-GPU validation has been performed from the Mac.

For preparation/audit without submission, append `--prepare-only`.
Both Python entry points cap BLAS/OpenMP/NumExpr threads at one before importing
NumPy, including when an inherited environment requests 32 threads. The batch
script's limits alone do not protect the login-node submission process.

For an older checkout that fails with `OpenBLAS blas_thread_init: pthread_create
failed` before submission, retry without changing packages:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$HOME/.conda/envs/memcam/bin/python" \
  paper/submit_keepsake_component_study.py --array-parallel 5
```

Omitting `--array-parallel` retains the older serial mode, which will not fit the
deadline. The two modes cannot submit to the same study root. A previously frozen
launcher without array support is rejected rather than silently modified; do not
duplicate already submitted serial work under a new root without first checking
its state. Nothing in this launcher cancels existing jobs.

## Frozen Protocol

| Setting | Pose weight | Appearance weight | Priority |
| --- | --- | --- | --- |
| KEEPSAKE | .65 | .35 | Full |
| Without appearance | 1 | 0 | Full |
| Without pose | 0 | 1 | Full |
| Without closest-substitute term | .65 | .35 | Degree terms only |
| Without degree term | .65 | .35 | Closest-substitute term only |

All original fifteen distinct scenes are retained in their manifest order. No
ranking or filtering by prior LPIPS, FVD, visual appearance or revisit performance
is used. There is one generation seed, 42, matched across settings; 50 denoising
steps; 640x352 resolution; CPU memory bank; original retrieval, insertion,
protection and tie-breaking behavior. Only the indicated component changes.
The fixed covisibility threshold remains .65 for all affinity removals, so this
does not compare independently tuned pose-only and appearance-only methods.

The native nominal 60-second output has 1825 frames at 30 FPS. Generated-frame
fidelity uses exactly frames 30,60,...,1800: sixty samples, excluding the initial
conditioning image. This exclusion differs from the earlier pilot and legacy
tables, which include frame zero. Each trajectory receives equal weight.

- **Primary:** whole-rollout LPIPS, four paired full-minus-removal contrasts.
  Whole-trajectory bootstrap: 5,000 draws, seed 17; two-sided paired sign
  randomization under within-scene label exchangeability; Holm across four tests.
- **Secondary:** PSNR, SSIM, late and revisit GT fidelity, cohort FVD, and trace
  diagnostics. Secondary scalar intervals are descriptive, not a second search
  for a significant primary metric. No "robust to ablations" conclusion is coded.
- **Late window:** 45 < generated time <= 60 seconds, fifteen sampled frames.
- **Revisit window:** target pose within .10 m and 5 degrees of an earlier
  generated view at least fifteen seconds old, with at least three consecutive
  one-second samples between them more than .50 m or 30 degrees from the first
  visit. Select the earliest qualifying first visit for each target. These labels
  use dataset poses only and are frozen before generation. They do not require
  any policy's generated pixels to look good. No threshold relaxation if coverage
  is poor. `revisit_coverage.csv` lists every scene, including zero-event scenes.
- **FVD:** four 16-frame clips per video, temporal stride four, size 224, production
  StyleGAN-V I3D. Pool all 60 clip features per setting against the same GT. The
  summary is one cohort FVD, not an average of per-video FVDs. Its descriptive
  intervals use 2,000 paired whole-scene bootstrap draws, including all clips
  from each sampled scene and recomputing FVD. No FVD significance test is used
  to replace the primary LPIPS test; finite-sample FVD bias remains.

LPIPS uses the existing AlexNet implementation at size 224; PSNR/SSIM use the
existing frame evaluator. Revisit scores compare generated frames to same-index
GT: they measure **revisit fidelity**, not first-visit/return self-consistency.
Uncertainty is across scenes conditional on one recorded seed, not seed variance.
These scenes were inspected previously; this is a follow-up, not a held-out test.

## Did the Ablation Change Memory?

The actual traces are audited for all 1748 target-frame reads per video, legal
selected IDs, candidate counts, B32 and the requested component settings.
`trace_diagnostics.csv` reports per scene and variant:

- Candidate-bank Jaccard overlap with full KEEPSAKE, averaged equally over sections.
- Fraction of actual selected memory IDs agreeing with the full method.
- At each predefined revisit, whether the bank contains an eligible view at least
  fifteen seconds old within the same position/rotation thresholds, and whether
  the reader selects such a view. These are camera-coverage diagnostics, not
  estimates of the retained image's visual quality.

This separates "the component barely changed decisions" from "it changed
decisions but did not improve generated quality." It is closed-loop: after
outputs diverge, banks contain each setting's own pixels, not a frozen replay.

The first sixteen decoded output frames are hashed across variants as a
pre-eviction pairing diagnostic. Differences are exported in `prefix_pairing.csv`
and counted in status; they are not hidden or called eviction effects. A mismatch
requires investigation before an exclusively retention-causal claim. Equality of
this short compressed prefix does not prove complete numerical determinism.

## Execution and Resume

Each scene job runs a timed real-CUDA kernel check, production DINO extraction,
LPIPS and I3D smoke, and a video-writer check before generating. It preserves
Slurm's GPU mask and uses the existing `memcam` environment directly. DINO weights
and processor identity and the Python environment are checked across jobs. It
does not install packages, touch VMem/DFoT, or run VBench.

Generation and scoring run in separate processes so generation VRAM is released
before evaluation. Each completed video/trace and metric checkpoint has hashes
and a separate receipt. Successful generation is reused if metrics fail. An
interrupted video has no mid-video resume here: only that unfinished cell starts
again in a new attempt directory; previous attempts remain intact.

`array_jobs.json` records the accepted array, its per-scene task IDs, and the report
job immediately. Repeating the command does not duplicate queued/running work.
Once the current array and report finish, rerun the same command to submit only
failed/incomplete scenes as a new array. Within those scenes, completed videos
and metrics are reused. No overlapping retry array is submitted while existing
tasks remain active, so retries cannot exceed the concurrency cap.

Task states use expanded array IDs in both `squeue` and `sacct` (not `JobIDRaw`).
Unknown states or accounting delays stop resubmission rather than risk duplicate
GPU work. Failed tasks do not prevent other array tasks from running. The CPU
report waits for all tasks to terminate using `afterany` on the array root ID,
then checks every artifact and refuses a complete table until all 75 cells are
valid. A failed report submission can be retried without resubmitting the array.
The legacy serial mode uses `jobs.json` instead.

Frozen code is executed from the study directory; model files are referenced and
hashed, not copied. Do not edit the snapshot, weights, dataset sources, or Conda
environment during the experiment. A changed protocol requires a new output root.

## Monitor and Download

```bash
OUT="$HOME/memcam_results/keepsake_components_60s_n15"
cat "$OUT/array_jobs.json"
squeue --me --array
find "$OUT/cells" -maxdepth 2 -name status.json -print -exec cat {} \;
```

Logs are `scene_ARRAYID_TASKID.out/.err` and `report_JOBID.out/.err` inside `$OUT`.
Per-cell attempts contain `generation.log` or `metrics.log`; per-scene preflights
are under `cells/scene_NN/`. Use the numeric array ID printed by the submitter:

```bash
sacct --array -j ARRAY_ID -X --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 40 -F "$OUT/scene_ARRAY_ID_0.out" "$OUT/scene_ARRAY_ID_0.err"
```

After the final report:

```bash
cat "$OUT/status.json"
cat "$OUT/ablation.csv"
cat "$OUT/contrasts.csv"
```

Main exports: `ablation.csv`, `ablation.tex`, `per_video.csv`, `summary.csv`,
`contrasts.csv`, `fvd_summary.csv`, `fvd_contrasts.csv`, `trace_diagnostics.csv`,
`prefix_pairing.csv`, `coverage.csv`.
An incomplete report writes coverage and explicitly partial per-video results,
not a finished ablation table. The plan and receipts remain the provenance.

From the Mac, after completion, download the small reports without all videos:

```bash
mkdir -p "$HOME/Downloads/keepsake_components_60s_n15"
scp 'ab575577@newton.ist.ucf.edu:memcam_results/keepsake_components_60s_n15/*.csv' \
  "$HOME/Downloads/keepsake_components_60s_n15/"
scp 'ab575577@newton.ist.ucf.edu:memcam_results/keepsake_components_60s_n15/*.json' \
  "$HOME/Downloads/keepsake_components_60s_n15/"
scp ab575577@newton.ist.ucf.edu:memcam_results/keepsake_components_60s_n15/ablation.tex \
  "$HOME/Downloads/keepsake_components_60s_n15/"
```
