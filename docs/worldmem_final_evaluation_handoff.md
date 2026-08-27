# WorldMem Final Evaluation Handoff

Updated: 2026-08-27

This is the instruction set for the WorldMem Codex session. Policy development
is over for the current paper cycle. The final MemCam rarity ablation is now
complete; WorldMem should finish the locked metric matrix without adding more
generation policies.

## Decision from MemCam

The strongest completed MemCam method is **Geometric Coverage** (historical
code name `slam_covisibility`). More complicated additions did not improve the
complete metric suite.

The final interpretation is:

- Geometric Coverage is the primary method.
- RI is a complementary retention-focused ablation.
- FIFO is the required negative capacity-control baseline.
- K-center is the standard coreset baseline.
- MCE is the formal set-coverage baseline and a useful negative result.
- Unbounded is the original-system reference.

Do not invent or port another policy before the final metrics are complete.

## Completed rarity ablation

The completed `slam_ri_blend` combined Geometric Coverage with the full
RI product, `rarity * irreplaceability`. It therefore never answered whether
rarity itself helps or whether the RGB irreplaceability multiplier caused the
VBench degradation.

MemCam tested exactly these two policies using the existing clustering and
Geometric Coverage implementations:

1. `rarity_only`

   ```text
   R_i = log((N + 1) / |C_i|)
   ```

   `C_i` is frame `i`'s connected DINO cluster. Use the corrected `k=3`
   nearest-neighbor threshold calibration. Evict the lowest `R_i`.

2. `slam_rarity_blend`

   ```text
   U_i = 0.75 * minmax(Geo_i) + 0.25 * minmax(R_i)
   ```

   `Geo_i` must be the existing Geometric Coverage score. `R_i` must be the
   exact rarity-only score above. Do not include RGB irreplaceability,
   hysteresis, a quality gate, or a new retrieval rule.

MemCam run names:

```text
rarity_b32_k3
slamrarity_b32_s75_r25_k3
```

Matched MemCam results:

| Policy | 10s LPIPS/FVD | 20s LPIPS/FVD | 30s LPIPS/FVD | 60s LPIPS/FVD |
| --- | --- | --- | --- | --- |
| Geometric Coverage | .496427 / 719.279 | .542038 / 663.776 | .556215 / 656.271 | .584950 / 690.532 |
| Rarity Only | .487774 / 682.187 | .542236 / 651.255 | .563453 / 664.062 | .598634 / 732.294 |
| 75% Geo + 25% Rarity | .492719 / 713.056 | .543034 / 646.257 | .557632 / 656.149 | .587213 / 676.395 |

| VBench dimension | Geometric Coverage | Rarity Only | 75% Geo + 25% Rarity |
| --- | ---: | ---: | ---: |
| Subject consistency | .818691 | .800773 | .810406 |
| Background consistency | .907117 | .899559 | .900564 |
| Motion smoothness | .991426 | .992471 | .992187 |
| Dynamic degree | 1.000000 | 1.000000 | 1.000000 |
| Aesthetic quality | .479245 | .449500 | .455404 |
| Imaging quality | .529411 | .497612 | .502729 |

The blend improves FVD, including `690.532 -> 676.395` at 60 seconds, but
loses LPIPS at 20, 30, and 60 seconds and loses subject consistency,
background consistency, aesthetic quality, and imaging quality. Rarity-only
degrades more strongly at the long horizon. Therefore neither policy passes
the multi-metric promotion criterion: both already fail available LPIPS and
standard VBench checks. Do not port either to WorldMem for this paper cycle.
This also shows that RI's RGB irreplaceability multiplier was not the sole
cause of the earlier blend's perceptual-quality loss. VBench-Long remains
unavailable because its environment lacks `moviepy.editor`, and current CUT3R
camera scores remain invalid under the unresolved evaluator sanity check.

## MemCam policy outcomes that constrain WorldMem

### Keep in the final cross-system roster

1. **Unbounded**
2. **FIFO-32**
3. **RI-32**, with the corrected `k=3` clustering implementation
4. **Geometric Coverage-32**
5. **K-center-32**
6. **MCE-32**

The fixed B32 comparison is intentional. It matches the primary MemCam budget
and avoids selecting a different budget for every WorldMem policy after seeing
the test metrics. WorldMem's complete budget curves can still be reported as a
separate LPIPS/FVD sensitivity experiment.

### Do not add to the final WorldMem generation queue

- Coverage-Hysteresis: the offline older-view diagnostic was positive in both
  systems, but the completed MemCam runtime policy lost badly at 60-second FVD
  and on four meaningful VBench dimensions.
- 75/25 Geometric-RI blend: it improved seven of eight MemCam LPIPS/FVD prefix
  cells, but lost subject consistency, background consistency, aesthetic
  quality, and imaging quality to pure Geometric Coverage.
- 50/50 blend: already failed to dominate either constituent.
- Rarity Only: failed the completed MemCam long-horizon and VBench comparison.
- 75/25 Geometric Coverage-Rarity blend: improved FVD but lost LPIPS at three
  later prefixes and lost four meaningful VBench dimensions.
- Generic IQA gate: rejected by held-out calibration.
- Pose-calibrated causal-consistency gate: rejected by held-out calibration.
- `reliable_slam_ri` or `causal_consistency_coverage_ri`: superseded and not
  justified by the validation results.
- DBVC and Trajectory Coverage: completed negative MemCam results.
- Surprise Forcing: retain as a MemCam pilot baseline, but do not port it now
  if WorldMem does not already have generated outputs.

## WorldMem runs already generated

The WorldMem handoff records completed 60-second, first-15-video LPIPS results
for the following grids:

- FIFO: B16, B32, B64, B128
- Latent-RI: B16, B32, B64, B128
- Geometric Coverage: B16, B32, B64, B128
- K-center: B16, B32, B64, B128
- MCE: B16, B32, B64, B128
- Unbounded

The locked B32 run names are:

```text
worldmem_unbounded_60s_n30
worldmem_fifo_b32_60s_n30
worldmem_rarity_irreplaceability_b32_60s_n30
worldmem_slam_covisibility_b32_60s_n30
worldmem_kcenter_coreset_b32_60s_n15
worldmem_mce_b32_60s_n15
```

Paper-facing labels should be:

```text
Unbounded
FIFO-32
Latent-RI-32
Geometric Coverage-32
K-center-32
MCE-32
```

WorldMem's best observed LPIPS policy is Geometric Coverage B16, not B32.
Report the full budget curve separately and optionally identify B16 as the
WorldMem within-system best. Do not replace the fixed-B32 cross-system table
with independently selected test-optimal budgets.

## Existing WorldMem quality result

Matched first-15-video, 60-second LPIPS:

| Policy | B16 | B32 | B64 | B128 |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 0.717 | 0.689 | 0.688 | 0.647 |
| Latent-RI | 0.566 | 0.546 | 0.549 | 0.567 |
| Geometric Coverage | **0.525** | **0.534** | **0.545** | 0.577 |
| K-center | 0.545 | 0.559 | 0.575 | **0.559** |
| MCE | 0.576 | 0.575 | 0.596 | 0.604 |
| Unbounded | - | - | - | 0.652 |

Selected matched FVD@60s values already recorded:

| Policy | Budget | FVD |
| --- | ---: | ---: |
| Geometric Coverage | 16 | 1041.757 |
| Geometric Coverage | 32 | 1116.925 |
| Latent-RI | 32 | 1160.428 |
| FIFO | 128 | 2604.960 |
| Unbounded | - | 3077.600 |
| FIFO | 32 | 3554.909 |

Absolute WorldMem FVD is not directly comparable with MemCam or the original
short-horizon WorldMem paper. Use it only for matched within-WorldMem policy
comparisons.

## Required final metric suite

Every locked B32 run must have the same first 15 trajectories and the same
configuration for:

1. LPIPS prefix curve: 10, 20, 30, 60 seconds
2. FVD prefix curve: 10, 20, 30, 60 seconds
3. Standard VBench at 60 seconds
4. VBench-Long at 60 seconds
5. CUT3R camera consistency, only after GT sanity passes

The six VBench dimensions must match MemCam:

```text
subject_consistency
background_consistency
motion_smoothness
dynamic_degree
aesthetic_quality
imaging_quality
```

## First action: audit, do not regenerate blindly

Run the existing audit before launching anything:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
AUDIT_FILTER='worldmem_(unbounded|fifo_b32|rarity_irreplaceability_b32|slam_covisibility_b32|kcenter_coreset_b32|mce_b32)_60s_n(15|30)$' \
bash scripts/audit_worldmem_memory_policy_runs.sh
```

Expected generation count for the final table is at least 15 complete videos
per run. Existing `_n30` folders must be evaluated on the same first 15 videos
as `_n15` folders.

Then audit metric artifacts under:

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix
/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix
/data/ab575577/worldmem/outputs/memory_policy/metrics/vbench_results
/data/ab575577/worldmem/outputs/memory_policy/metrics/vbench_long_results
/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_pose_recon
/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_camera_metrics
```

Only generate missing videos. Metric wrappers should skip existing complete
outputs unless `FORCE=1` is explicitly requested.

## Critical VBench protocol fix

The current VBench wrappers point directly at each run's prediction directory.
That can evaluate 30 videos for an `_n30` run and 15 videos for an `_n15` run,
which is not a matched comparison.

Before final VBench or VBench-Long:

1. Add a `LIMIT=15` staging mechanism to both wrappers, or create one temporary
   per-run directory containing exactly the first 15 `video_batch*.mp4` files.
2. Match by batch index, not by arbitrary filesystem order.
3. Record the selected batch IDs in the output metadata.
4. Refuse to run when a locked policy is missing any required batch.
5. Do not mix stale results produced from 30 videos with new 15-video results.

Relevant scripts:

```text
scripts/run_worldmem_vbench.sh
scripts/run_worldmem_vbench_long.sh
utils/aggregate_vbench_results.py
```

Standard VBench should run first. Smoke-test VBench-Long on one B32 policy,
verify that all six dimensions and 15 source videos are represented, then run
the remaining locked policies.

## LPIPS and FVD protocol

Existing wrappers already support `LIMIT=15`:

```text
scripts/evaluate_worldmem_lpips.sh
scripts/evaluate_worldmem_fvd.sh
```

Use the locked comma-separated run list, `LIMIT=15`, and
`EVAL_DURATIONS=10,20,30,60`. Preserve the existing FVD configuration:

```text
clip length:       16
clips per video:   4
frame stride:      4
image size:        224
backend/detector:  the same cached I3D used by the existing WorldMem run
```

Do not overwrite a complete summary until its video IDs and configuration have
been checked.

## CUT3R is currently blocked by evaluator validity

WorldMem's CUT3R wrapper runs, but the evaluator produced huge camera errors and
zero WorldScore camera-control score on ground-truth Minecraft videos. That
means the current pose convention or alignment is invalid.

Do not fill the paper table with those numbers merely because the script
finishes.

Required order:

1. Run or inspect the GT sanity output from
   `scripts/run_worldmem_cut3r_gt_sanity.sh`.
2. Fix pose convention, alignment, or evaluator mapping until GT Minecraft
   produces sensible camera errors.
3. Freeze the corrected evaluator.
4. Run `scripts/run_worldmem_cut3r_metrics.sh` on the same locked first 15
   videos for all six policies.
5. If GT sanity cannot be repaired, mark WorldMem CUT3R as invalid/unavailable
   instead of reporting misleading values.

## Metrics completion deliverable

The WorldMem session should produce one machine-readable status table with one
row per locked policy and these columns:

```text
run_name
videos_matched
lpips_10
lpips_20
lpips_30
lpips_60
fvd_10
fvd_20
fvd_30
fvd_60
vbench_subject
vbench_background
vbench_motion
vbench_dynamic
vbench_aesthetic
vbench_imaging
vbench_long_subject
vbench_long_background
vbench_long_motion
vbench_long_dynamic
vbench_long_aesthetic
vbench_long_imaging
cut3r_valid
cut3r_rotation_error
cut3r_translation_error
cut3r_camera_control_score
```

Also save:

- the exact 15 batch IDs,
- metric configuration JSON,
- source result paths,
- git commit,
- GPU model for metric runs,
- a clear missing/invalid marker rather than silent `NaN` replacement.

## Stop conditions

- Do not port Rarity Only or the 75/25 Geometric Coverage-Rarity blend; both
  failed the MemCam multi-metric promotion gate.
- Do not generate Coverage-Hysteresis, another blend, or another gate.
- Do not expand to another budget merely to rescue one metric.
- Do not compare unmatched video counts.
- Do not report invalid CUT3R.
- Once the six-policy metric matrix is complete, stop policy experimentation
  and move to paper figures and analysis.
