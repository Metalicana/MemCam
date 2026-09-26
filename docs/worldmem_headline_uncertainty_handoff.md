# WorldMem Headline Intervals: Request To The WorldMem Session

Implement and run a resumable, existing-data uncertainty analysis for the
KEEPSAKE manuscript. NO new generation, tuning, changed trajectories, or
selection based on favorable scores. Preserve all current work. This handoff
does not claim a WorldMem job has been submitted or the sources audited.

## Scope And Priority

1. The main **60-second / N=15 / B32** table: Unbounded, FIFO, MCE, K-center,
   KEEPSAKE. Keep the available RI results as a separately labeled supplementary
   comparison, not silently discarded because of rankings. Freeze its test
   family separately from the five-row manuscript table.
2. The KEEPSAKE **B16/B32/B64/B128, 60s** budget sweep. Reference is B32;
   compare B32 against all three alternatives, never choose only the best budget.
3. Six VBench dimensions from the matching raw per-video files, if provenance
   and FPS are valid. Calculate uncertainty from per-video scores, not rounded
   aggregate means. Optional native N=10 and rFID results are distinct protocols,
   lower priority, and never pooled with this N=15 suite.

## Read And Audit Before Computing

Read `WORLDMEM_FINAL_METRICS.md`, `WORLDMEM_RESULTS_INVENTORY.md`, current
evaluation code, and actual source manifests/receipts. Local inventories list
aggregate numbers but are NOT proof that remote data or mappings are valid.
Starting source roots (verify, don't assume these exist on Newton):

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_budget_sweep_60s_n15
/data/ab575577/worldmem/outputs/memory_policy/metrics/vbench_budget_sweep_60s_n15
```

Relevant existing code:

```text
utils/evaluate_worldmem_fvd_prefix_curves.py
utils/evaluate_worldmem_lpips_prefix_curves.py
utils/worldmem_eval_common.py
utils/calculate_worldmem_vbench6.py
```

Important specifics to validate from the actual records:

- Batch IDs 0..14 must resolve to the SAME actual dataset sequence, start frame,
  control sequence and generation seed across policies. Batch number alone is
  not enough: retries/resampling can alter the underlying dataset identity.
  Export requested AND actual identities if available; unresolved mapping must
  be a provenance limitation/blocker, not a fabricated verified match.
- This is 600 GENERATED frames at simulation FPS 10, after 600 initial context
  frames. Stored MP4s have sometimes been encoded at 15 FPS. Use frame indices
  and the frozen dataset mapping; do not mistake 40s playback for a shorter
  rollout, resample it to 900 frames, or include the initial context as generated
  evaluation frames. Verify any VBench FPS correction separately.
- Long-horizon LPIPS uses exact-index raw dataset GT. Preserve its existing
  LPIPS backbone, value range, resize, frame set and aggregation. Inspect the
  actual metric class; do NOT assume MemCam's AlexNet/224/stride30 configuration.
- The recorded WorldMem FVD configuration is StyleGAN-V I3D, four 16-frame clips
  per trajectory, stride four, 224px. Confirm detector hash, preprocessing,
  clip indices and GT mapping from code and source receipts. Do not import the
  MemCam 180s eight-clip/stride-eight configuration.
- Verify all 15 full videos per configuration and common GT coverage. No silent
  intersection, short-video inclusion, duplicated IDs or all-frames-as-N.
- Hash videos, sampled GT, evaluator code, weights, manifests and cache outputs.
  Reuse cached I3D features/per-video LPIPS only if identities AND protocols
  match. Aggregate-only FVD cannot yield a trajectory-bootstrap interval.
- Independently recompute each full-cohort point estimate before bootstrapping.
  Export recorded vs recomputed values and investigate changes; never attach
  new intervals to mismatched old numbers. Expected historical anchors include
  Unbounded FVD 3077.599804 / LPIPS .652269 and KEEPSAKE B32 FVD 1116.924792 /
  LPIPS .533678, not numerical targets to force the analysis to reproduce.

For VBench, the draft says individual dimensions but tables contain a custom
average. Explicitly label any custom aggregate, preserve its exact normalization
and weights, and compute it per trajectory before resampling. Compare its mean
against the recorded table. The local WorldMem notes list MCE B32's custom
aggregate as 72.84 and K-center's as 72.67; the supplied manuscript reverses
those two values. Verify against raw files, not either text version.

## Statistics

Freeze all planned comparisons, metric definitions and resampling seed before
running. Preserve nonsignificant or unfavorable results alongside favorable ones.

1. Use 2,000 paired whole-trajectory percentile bootstrap draws, RNG seed 17.
   Sample the same fifteen indices with replacement for ALL policies and GT.
   Carry every clip/frame summary from a selected trajectory together.
2. LPIPS: average within each trajectory using the frozen full-horizon protocol,
   then equal-weight mean across trajectories. VBench: preserve the actual
   evaluator's aggregation; if it is not an equal-video mean, replicate its
   sufficient statistics and report the weighting explicitly, rather than
   silently change the estimator.
3. FVD: pool all 60 clip features for each full-cohort estimate; on EVERY draw
   pool all clips from the sampled trajectories and recompute FVD against the
   correspondingly sampled GT. Never average per-video FVDs or bootstrap four
   clips independently. Validate any low-rank acceleration against the original
   covariance implementation, including documented regularization/tolerance.
4. Export policy-wise point estimates and 95% CIs AND paired KEEPSAKE-minus-
   comparator differences with 95% CIs. Negative is better for LPIPS/FVD.
   Compute relative reduction per draw as 100*(comparator-KEEPSAKE)/comparator;
   guard zero denominators. Do not divide independent CI endpoints.
5. Run two-sided paired policy-label swaps at whole-trajectory level, 5,000 Monte
   Carlo draws, seed 18, plus-one p-value correction. Swap all clip features
   together and recompute pooled FVD. Scalar tests use corresponding swapped
   trajectory summaries. Enumerate all swaps only when feasible. State the
   within-trajectory label-exchangeability assumption; these are observational
   paired outputs, not a randomized policy-assignment study.
6. Holm across both LPIPS and FVD and all four main-table comparators (8 tests).
   Budget sweep: both metrics times three comparisons (6 tests). Also export
   Holm across all 14 WorldMem quality comparisons. RI supplementary and VBench
   have separate explicitly named families. Export raw p-values so a final
   manuscript-wide family can be corrected jointly with MemCam if claimed.
7. All percentile intervals above are UNADJUSTED, not simultaneous. Distinguish
   zero exclusion from adjusted significance; no significance stars based only
   on the unadjusted CI. Nonsignificance does not imply equivalence or unchanged
   quality. FVD's finite-sample bias is not corrected by this bootstrap.
8. Audit shared-world/environment identities. If trajectories share an
   environment, also run a labeled sensitivity analysis resampling/swapping
   complete environments using an explicit verified mapping. Do not guess the
   map from an arbitrary batch ID. Report the number of independent groups.

The uncertainty is across sampled trajectories conditional on existing outputs,
not across new generation seeds or held-out environments. N=15 and pairing do
not alone justify a universal detectable effect size; report actual interval
widths and the limitations of small samples.

## Runtime And Deliverables

- Start with CPU inspection. If validated per-trajectory metrics AND per-clip
  I3D features already exist, all resampling is CPU-only. Otherwise request ONE
  allocated GPU only for bounded-batch LPIPS/I3D extraction, not generation.
- On Newton use Slurm and inherited CUDA mask; highgpu type is
  `gpu:nvidia_h100_80gb_hbm3:1`. Use a working WorldMem environment and actual
  Newton data paths; do not assume CECSL's `/data` tree exists there. Do not
  overwrite the working MemCam/DFoT environments. No package changes in running
  environments. Fail clearly if CUDA is unavailable: the existing FVD/LPIPS
  classes can silently fall back to CPU, so the wrapper must explicitly check.
- CPU BLAS/OpenMP/NumExpr threads capped before NumPy import. Separate extraction
  from CPU inference if useful; bound memory and log progress. Do not allocate
  all 600 frames times all videos on GPU simultaneously or rerun unbounded
  VBench evaluation merely to obtain scalar uncertainty.
- One writer per output root; atomic per-video receipts/features; checksum-
  validated reuse. Interrupted scoring must preserve successful extraction.
  No silent changes to settings on resume. No duplicate live jobs.
- Deliver executable code, focused tests, one submission command, monitor and
  download commands. State exactly which tests ran locally versus on the GPU.
- Deliver `coverage.csv`, `per_video.csv`, `summary.csv`, `contrasts.csv`,
  `point_estimate_audit.csv`, `analysis.json`, frozen plan/provenance, saved
  bootstrap samples, and ready-to-input `table.tex`. Package a small
  `reports.zip`, no videos. Missing/failed groups must not look complete.

Do not bootstrap a trajectory-population timing CI from the single-trajectory
600-query latency pilot. Queries are dependent repeated measurements, not 600
independent trajectories. Likewise, B32 and the audited final count of 1,200
are not random estimates requiring invented CIs. Published-paper baseline
numbers without raw runs cannot be given retrospective paired intervals.

Keep this job separate from the running MemCam component array 849715 and its
report. No cancellation or modification of other experiments is requested.
