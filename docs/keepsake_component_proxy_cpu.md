# CPU Component Proxy Ablation

This is a separate, post-hoc mechanism diagnostic motivated by the inconclusive
closed-loop component ablation. It does not generate videos, change earlier
results, or replace the generated-quality ablation. No Newton run was performed
from the Mac while implementing this runner.

## Run In An Existing CPU Allocation

Transfer the new runner from the Mac checkout:

```bash
cd /Users/metalicana/projects_summer_2026/MemCam
scp paper/run_keepsake_component_proxy_cpu.py \
  ab575577@newton.ist.ucf.edu:MemCam/paper/
```

Inside the existing Slurm-allocated CPU shell on Newton, not a login shell:

```bash
cd "$HOME/MemCam"
"$HOME/.conda/envs/memcam/bin/python" -u paper/run_keepsake_component_proxy_cpu.py
```

Do not nest another allocation. The runner requires `SLURM_JOB_ID`, caps BLAS
threads at one before NumPy imports, hides GPUs, and uses the existing environment.
No Conda activation, package installation, model download or feature extraction is
needed. Keep an interactive shell connected until it finishes. Interruptions
preserve completed trajectory receipts; repeat the same command in an allocation
to resume. An unfinished trajectory replays from its beginning.

The earlier seven-setting CPU sensitivity job took 18m19s. This five-setting
job uses the same sources and a similar workload, so tens of minutes is a rough
estimate, not a guarantee. It prints validation/replay progress and the elapsed
time after each trajectory. Source hashing can take time on the shared filesystem.

For a separate batch job instead, transfer and submit the optional script:

```bash
# Mac, from the checkout:
scp slurm/newton_keepsake_component_proxy_cpu.sbatch \
  ab575577@newton.ist.ucf.edu:MemCam/slurm/
```

```bash
# Newton login shell, only if not using the existing allocation:
cd "$HOME/MemCam"
sbatch slurm/newton_keepsake_component_proxy_cpu.sbatch
```

The script requests two CPUs, 24G host memory, no GPU, and a two-hour time limit.
The limit is not an expected duration. Do not run both launch modes concurrently.

## Frozen Protocol

All fifteen original 60-second trajectories, in manifest order, use the same
cached unbounded-generated DINO feature stream and camera poses. Inputs are
`~/memcam_results/context_memory_60s/` and its `gap_feature_cache_fresh/` directory.
No cohort filtering or outcome-based parameter search is implemented.

| Setting | Pose / appearance weights | Retention priority |
| --- | --- | --- |
| Full KEEPSAKE | .65 / .35 | Both terms |
| Without appearance | 1 / 0 | Both terms |
| Without pose | 0 / 1 | Both terms |
| Without closest substitute | .65 / .35 | Degree saturation and degree floor |
| Without degree | .65 / .35 | Closest-substitute term only |

The runner calls production `compute_slam_covisibility_scores` and
`FrameMemoryBuffer`, including their component modes, one-shot scores and
oldest-first tie breaking. Threshold .65, tau 3, beta .5 and lambda .25 remain at
production values except for the removed terms. Removing degree removes both
its saturation and floor; it is not the same intervention as beta=0.

The 24 insertion batches each cover 77 frames with a one-frame overlap. B32
includes the protected initial frame and the current endpoint; previous endpoints
become evictable. Only retained/new frame features and currently available poses
are passed to the scorer. There is no GT or future-feature input to retention.

At four fixed target slots (0,19,38,57) in each retrieved section 1..23, evaluate:

`retention gap = best retained-bank DINO distance - best full-history DINO distance`

Distances compare generated historical descriptors with the target GT descriptor.
Both candidate sets use the same causal cutoff, `frame_id < section * 76 - 3`.
This gives 92 queries per trajectory per variant. Average queries within each
trajectory, then average the fifteen trajectory means equally. Lower gap is
better. The full-history oracle is an evaluation reference, not another generated
arm or a reader available to the policy.

Report all five means, four paired contrasts and mean bank Jaccard versus full.
Intervals use 5,000 paired whole-trajectory bootstrap draws, seed 17. The summary
uses variant-minus-full: positive means the removal lost more oracle evidence.
The separate paired CSV uses full-minus-variant and says so in its filename.
These are descriptive, unadjusted intervals; no confirmatory p-values or automatic
winner claim is produced. Trajectory-level resampling does not model dependence
between trajectories from a shared environment or variation across generation seeds.

The evaluation DINO metric is held fixed across variants, but is not independent
of the appearance descriptor used by the policy. Cached features are not asserted
byte-identical to online descriptors. This tests fixed-history retention coverage,
not actual selected views, view mismatch, memory corruption, or closed-loop quality.
Even favorable results cannot establish an FVD/LPIPS benefit or component necessity.
Zero-crossing intervals do not establish equivalence.

## Outputs And Resume

Default output is a new directory, separate from both earlier studies:

```bash
OUT="$HOME/memcam_results/keepsake_component_proxy_cpu_60s_n15"
cat "$OUT/status.json"
cat "$OUT/summary.csv"
cat "$OUT/paired_full_minus_variant.csv"
cat "$OUT/interpretation.txt"
```

Other exports: `component_proxy.tex`, `per_trajectory.csv`, `query_retention.csv`,
`updates.csv`, and each trajectory's bank snapshots in `cells/row_NNN/replay.json`.
Update timings include CPU scoring/buffer maintenance only, not encoding or
retrieval; they are not an end-to-end latency comparison.

Before replay, `plan.json` freezes configurations, cohort, sidecar hashes, pose
hashes, relevant source hashes, encoder identity and Python/NumPy/Torch versions.
Code copies are kept under `code/` for provenance; execution uses the checked
checkout. Do not edit those source files or the environment during a run/resume.
Each cell audits the unbounded trace, cached arrays, source video and original GT
frame hashes. Validated receipts can be reused; changed sources or outputs are
rejected. A filesystem lock prevents concurrent writers. No final table is
produced until all fifteen trajectories validate. Read `status.json` first.

## Completed Parameter Sweep

The earlier tau/beta/lambda sweep completed successfully on Newton (job 850637).
Its evidence remains separate from this component proxy:

- Tau 1,3,5 produced identical banks and gaps on this fixed-history cohort.
- Beta 0,1 and lambda 0,.5 changed banks, but all four paired gap-difference
  intervals included zero.
- Mean gaps ranged from .035453 to .039696; the largest absolute point-estimate
  difference from default was .002219 DINO cosine-distance units.

This is descriptive retention sensitivity, not proof the defaults are optimal,
that components are necessary, or that generated quality is robust.
