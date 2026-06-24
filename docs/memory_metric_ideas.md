# Memory and Retrieval Metric Ideas

This note captures metric ideas for the Newton H100 60s memory-policy runs.
The goal is to move beyond average frame quality and measure whether a
policy actually preserves useful long-range scene memory.

## Current Coverage

The repo already computes a useful base layer:

- Frame/GT quality in `utils/evaluate_context_memory.py`: MAE, MSE, RMSE,
  PSNR, SSIM, temporal-delta MAE/RMSE, optional LPIPS, DINO, CLIP, and FVD.
- Access trace summaries in `utils/analyze_access_traces.py` and
  `utils/summarize_access_traces.py`: fallback rate, selected age, overlap,
  candidate count, stored memory size, reuse entropy, reuse gini, top selected
  frames, and eviction stats.
- Policy oracle analysis in `utils/analyze_memory_policies.py`: coverage,
  possible coverage, oracle recall, retained useful frames, available useful
  frames, and RI score rows.
- RI score alignment in `utils/summarize_ri_alignment.py`: Spearman/top-k
  agreement with future overlap usefulness and useful-mass-kept.
- Generated revisit consistency in `utils/analyze_revisit_consistency.py`:
  planned-pose revisit pairs with generated-vs-generated PSNR, SSIM, MAE,
  RMSE, and p90/p95/worst summaries.

The remaining missing piece is a stronger set of learned revisit, retrieval,
pose, occlusion, dynamic-event, and cost metrics.

## Source Papers

- MemCam: https://arxiv.org/abs/2603.26193
- Context-as-Memory: https://arxiv.org/abs/2506.03141
- VMem: https://arxiv.org/abs/2506.18903
- WorldMem: https://arxiv.org/abs/2504.12369
- SPMem / spatial memory occlusion: https://arxiv.org/abs/2606.10299
- OmniMem: https://arxiv.org/abs/2605.30519

## Recommended Next Metrics

| Metric | What it catches | Source inspiration | Implementation status |
| --- | --- | --- | --- |
| Revisit consistency | Whether a returned view matches the earlier generated view at the same or similar pose | MemCam, Context-as-Memory, VMem, WorldMem | Implemented for base image metrics in `utils/analyze_revisit_consistency.py`; learned metrics still needed |
| Cycle symmetry | Whether outbound frames match reversed inbound frames on round-trip trajectories | MemCam | Add paired segment comparison |
| History-context comparison | Whether newly generated revisit frames match prior generated history, not just GT | Context-as-Memory | Add generated history pair table |
| Tail revisit failure | Rare severe memory breaks hidden by means | All revisit papers | Add p90/p95/worst LPIPS, DINO distance, PSNR |
| Retrieval overlap capture | How much of oracle/highest-overlap memory the policy captures | Context-as-Memory, MemCam | Mostly available through trace usefulness analysis |
| Retrieval rank quality | Whether the policy ranks useful future memory high | RI/Belady policy work, Context-as-Memory | Extend traces to log all candidate scores |
| Redundancy collapse | Whether selected memory degenerates into adjacent/recent frames | Context-as-Memory, VMem, OmniMem | Partly available through entropy/gini/age bins |
| Long-range retrieval rate | Fraction of selections outside the recent window | VMem, OmniMem | Available from `memory_age`; add window-specific rates |
| Pose-control error | Whether generated views follow the intended camera trajectory | VMem | Needs DUSt3R/COLMAP-style pose extraction |
| Loop-closure pose error | Whether estimated pose at return matches estimated pose at start | VMem, WorldMem | Needs pose extraction |
| Occlusion false-visible rate | Whether FOV/co-visibility retrieval selects memory that should be hidden | SPMem, VMem | Needs depth/geometry or richer overlap labels |
| Dynamic/event recall | Whether changed scene state persists over time | WorldMem | Needs dynamic test cases or event labels |
| Quality-cost frontier | Memory quality per second/frame, VRAM, and memory budget | MemCam, VMem, Context-as-Memory, OmniMem | Add run-time/VRAM logging from Slurm or torch |

## Concrete Metric Definitions

### Revisit Consistency

For matched generated-frame pairs `(i, j)` where camera poses are the same or
close after a trajectory loop:

- `revisit_psnr`, `revisit_ssim`, `revisit_lpips`
- `revisit_dino_distance`, `revisit_clip_distance`
- `revisit_frame_gap = j - i`
- `revisit_pose_delta_t`, `revisit_pose_delta_rot_deg`

Aggregate by gap bucket and by trajectory phase:

- `mean`, `median`, `p90`, `p95`, `worst`
- `gap_76_151`, `gap_152_303`, `gap_304_plus`

This is more direct than GT-only quality because a generated world can be
internally consistent while differing from GT, or match GT locally while
forgetting earlier generated content.

### Cycle Symmetry

For a round-trip trajectory, compare the outbound segment to the reversed
inbound segment:

- `cycle_lpips_mean`
- `cycle_lpips_p95`
- `cycle_dino_distance_mean`
- `cycle_psnr_min`

MemCam uses 90 deg and 360 deg round-trip protocols. For the Newton 60s runs,
this should be reported separately for each policy and budget because memory
failures should widen as the return gap grows.

### History-Context Comparison

Context-as-Memory separates two views of memory capability:

- GT comparison: generated revisit frame vs ground truth frame.
- History comparison: generated revisit frame vs earlier generated frame.

We should keep both. If a policy improves history comparison but not GT, it is
remembering its own generated world but may be off-distribution. If it improves
GT but not history comparison, it may be locally plausible but unstable.

### Retrieval Overlap Capture

For each target query with selected memory frame `s` and an oracle candidate
`o` from unbounded or overlap labels:

- `overlap_capture_ratio = overlap(s, target) / max_overlap(target)`
- `overlap_gap = max_overlap(target) - overlap(s, target)`
- `exact_oracle_match`
- `near_oracle_match`, with a frame window such as 4 or 8 frames
- `gap_gt_0_05_rate`, `gap_gt_0_10_rate`

This is partly implemented in `utils/analyze_trace_usefulness.py`. The next
improvement is logging all candidate overlaps and policy scores at selection
time so we can compute top-k recall, MRR, and NDCG.

### Retrieval Rank Quality

When all candidate scores are logged:

- `useful_hit_at_1`, `useful_hit_at_5`
- `oracle_frame_rank`
- `mrr_oracle`
- `ndcg_overlap`
- `spearman_policy_score_vs_future_use`

This bridges current RI alignment with the retrieval-style metrics used in
memory papers.

### Redundancy and Long-Range Use

Context-as-Memory and VMem both show that simple recent-frame selection can
look okay short term but fail on revisits. Track:

- `recent_window_selected_frac`: selected age <= one segment
- `long_range_selected_frac`: selected age > two segments
- `non_adjacent_selected_frac`: selected frame not adjacent to another selected
  context frame in the same section
- `pose_spread_mean`: average pairwise pose distance among selected context
  frames
- `selected_entropy_norm` and `reuse_gini`, already partially available

### Pose-Control and Loop-Closure Error

VMem evaluates camera alignment via estimated generated poses:

- `rdist`: rotation distance between estimated generated pose and GT/planned
  pose
- `tdist`: translation distance after sequence-relative normalization
- `loop_rdist`: estimated return pose vs estimated start pose
- `loop_tdist`: estimated return pose vs estimated start pose

This needs a pose estimator such as DUSt3R/COLMAP and should be treated as a
heavier metric tier.

### Occlusion-Aware Retrieval

SPMem is not a video-generation paper, but its evaluation point is highly
relevant: recall and visibility are different. For MemCam-style FOV retrieval,
we should not reward a policy for retrieving a frame that is pose-near but
occluded from the target view.

Possible metrics:

- `false_visible_rate`: selected by FOV/covisibility but overlap/depth oracle
  says hidden
- `occlusion_precision`: selected memory has visible overlap with target
- `behind_wall_hit_rate`: for synthetic occlusion stress tests

This needs geometry, depth, or stronger overlap labels than the current traces.

### Quality-Cost Frontier

For each policy and budget:

- `seconds_per_frame`
- `peak_vram_gb`
- `stored_memory_size_mean`
- `fallback_rate`
- `revisit_lpips_p95`
- `fvd`
- `quality_per_gb` and `quality_per_second` style ratios for ranking

VMem, MemCam, Context-as-Memory, and OmniMem all report some form of
quality-speed-memory trade-off. We should make this first-class for the Newton
runs because a policy can be scientifically interesting but operationally poor.

## Implementation Plan

### Tier 0: No Rerun Needed

1. Run generated-vs-generated revisit pair metrics using
   `utils/analyze_revisit_consistency.py`.
2. Add tail summaries for all frame and revisit metrics: p90, p95, worst.
3. Add long-range retrieval fractions from existing `memory_age`.
4. Combine evaluator summaries with access summaries into one policy/budget
   report table.

### Tier 1: Light Instrumentation For Next Runs

1. Log all candidate memory frames, their overlap, and their policy score.
2. Log selected rank under each policy score and under overlap oracle.
3. Log per-section selected context frame lists.
4. Log wall-clock seconds/frame and `torch.cuda.max_memory_allocated()`.

### Tier 2: Heavy/Optional

1. Add DUSt3R/COLMAP pose extraction for `rdist`, `tdist`, and loop closure.
2. Add occlusion-aware overlap with depth/geometry.
3. Add object/region identity metrics with masks or tracks for revisit regions.

## Priority For The Current Newton 60s Sweep

The highest-value additions are:

1. Generated-vs-generated revisit consistency.
2. P95/worst revisit failure metrics.
3. Long-range retrieval rate and redundancy collapse.
4. Overlap capture ratio vs oracle/unbounded.
5. Quality-cost frontier per policy and budget.

These use the artifacts we already expect from the current jobs and should make
the policy comparison much more diagnostic than mean PSNR/SSIM/LPIPS/FVD alone.
