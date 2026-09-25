# Paired Quality Uncertainty

No new videos are generated. This workflow does not alter the running component
array, its frozen protocol, or its outputs. Nothing is submitted automatically.

## Historical Source Audit

The downloaded `context_180s` artifacts do not substantiate a matched fifteen-video
comparison of FVD 734.2 versus 476.6:

- `eval_fvd_lpips_duration_curves_180s/baseline/summary.json`, duration 180:
  9 completed, 6 missing; FVD 734.220684; LPIPS 0.5979595; LPIPS stride 30.
- `eval_metrics_180s/slam_b32_covisibility/summary.json`:
  15 completed; overall FVD 476.617572; LPIPS 0.5876462; LPIPS stride 90.
- `eval_metrics_180s/baseline/summary.json`:
  8 completed, 7 missing; overall FVD 797.478520; LPIPS 0.6004122; stride 90.

All these FVD configurations use **eight 16-frame clips, temporal stride eight**,
size 224, StyleGAN-V I3D. The existing paper-evidence job's four-clip/stride-four
analysis is a different protocol and must not supply intervals for these points.
The historical covariance evaluator can differ slightly between pooled and
per-duration computations; retain the exact source field and report numerical
agreement checks, not false bitwise equivalence.

`legacy` audits both intended and completed cohorts and LPIPS sampling. It rejects
partial pairs by default. `--allow-matched-subset` explicitly produces a diagnostic
on the intersection, never a replacement labeled fifteen videos. Frame-stride
mismatches are always rejected, even with this flag. No FVD interval can be
recovered from an aggregate FVD alone.

The explicitly partial, stride-90 intersection was analyzed locally (eight
matched trajectories, 5,000 draws, seed 17, scene-ID grouping): baseline LPIPS
0.6004122, KEEPSAKE 0.6019437, difference +0.0015314, approximate 95% CI
[-0.008146, +0.010157]. This is not clear evidence of improvement or equivalence,
and must not be labeled the fifteen-video headline result. Outputs are in
`paper/results/paired_uncertainty_180s_partial8/`. A full-cohort FVD interval
has not been computed locally or submitted remotely by this implementation.

## Corrected Full-Cohort Evaluation

On Newton, after transferring the code, submit this separate one-H100 metric job:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_headline_uncertainty.sbatch
```

This is an additional GPU job, not a replacement for the ablation. Its four-hour
limit is not a measured completion estimate. Existing videos, exact-index GT and
the cached production I3D weights must all exist; missing input causes failure,
not a silently reduced cohort. The full manifest has fifteen trajectories.
Both policies are rescored at the same LPIPS stride 90, including frame zero,
and with eight I3D clips at stride eight. Revised means may differ from the old
table. Report them as a corrected matched evaluation, not as error bars added to
the old unmatched means. Model generation and its checkpoints are untouched.

Extraction caches a matched pair's LPIPS and GT/policy I3D features with hashes.
Repeat the command using the same output root to reuse valid completed pairs;
an interrupted pair is recomputed. One writer per output root is enforced.
Do not change source files, videos, GT, weights or environment during extraction.
A changed extraction protocol/code/runtime requires a new output directory.

Default output: `~/memcam_results/headline_uncertainty_180s_matched15`.
Check `status_extract.json`, `status_report.json`, `contrasts.csv`,
`per_video.csv`, `leave_one_group_out.csv`, `fvd_numerical_check.csv` and
`paired_uncertainty.tex`. The latter is a tabular fragment requiring booktabs.

The report can also run on CPU after extraction, without CUDA initialization:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$HOME/.conda/envs/memcam/bin/python" paper/paired_quality_uncertainty.py report \
  --duration 180 --draws 2000 \
  --output "$HOME/memcam_results/headline_uncertainty_180s_matched15"
```

Run CPU bootstrap work on an allocated node, not a busy login node. To run it
locally, transfer the output directory including `plan.json` and `cells/`.
Raw videos and the original source paths are not needed by the reporting phase.

## Interpretation And Grouping

Each draw samples whole paired groups with replacement, keeping every clip from
each selected trajectory and using identical multiplicities for GT and both
policies. LPIPS remains an equal-trajectory average. FVD is recomputed from pooled
features and sample covariances, never averaged from per-video FVD values.
The report uses 2,000 draws, seed 17, and the 2.5/97.5 percentiles of the paired
difference (KEEPSAKE minus unbounded). Negative differences favor KEEPSAKE.
Whole-group leave-one-out checks are exported to expose sensitivity, not to
choose a favorable subset. No automatic significance or superiority prose.

By default, identical scene IDs are grouped. Distinct names such as
`ContainerYard_3` and `ContainerYard_7` might share an environment; this must be
checked with the dataset, not assumed independent from the suffix. Supply
`--cluster-map path.json` to `report` (or as a batch argument), mapping **every**
scene ID to an independent environment/group ID. All trajectories in a group
then travel together. The number of sampled trajectories can vary under grouped
resampling, while the original point estimate retains equal trajectory weights.
Report the grouping assumption; these exploratory intervals do not measure
generation-seed variability or remove development-set reuse/FVD finite-sample bias.

The CLI also supports 60-second MemCam manifests with explicit sampling options.
Do not substitute the 60-second ablation's initial-frame exclusion for the
historical tables' inclusion. WorldMem needs its actual per-video metrics,
clip features, camera/GT mapping and metric protocol; this MemCam extractor is
not presented as a WorldMem evaluator. No WorldMem interval is fabricated.
