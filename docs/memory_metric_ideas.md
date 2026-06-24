# Memory Metric Plan For Newton 60s

This is the corrected metric plan. The earlier PSNR/SSIM/MAE/RMSE revisit
script was the wrong direction. Pixel similarity can be kept as a compatibility
number, but it is not the memory metric.

The core question for these 60s policy runs is:

> Does the memory policy retrieve the right old evidence, under long gaps,
> revisits, redundancy pressure, and occlusion, at a useful cost?

## Paper Read

### MemCam

Source: https://arxiv.org/abs/2603.26193

MemCam evaluates memory with explicit round-trip camera protocols:

- 90-degree round trip.
- 360-degree round trip.
- The generated video is split into two sequences; one side is reversed and
  compared with the other side when the camera returns.
- Reported metrics: PSNR, SSIM, LPIPS, FVD.
- Ablates retrieval strategies: Recent, Random, TopK, and co-visibility.
- Ablates context compression with both quality and seconds/frame.

What matters for us:

- Round-trip / cycle-return is the right evaluation shape.
- Retrieval strategy comparison is more important than raw frame quality.
- Seconds/frame belongs in the result table.

### VMem

Source: https://arxiv.org/abs/2506.18903

VMem says standard long-term NVS benchmarks are not enough because they rarely
revisit observed regions. It introduces cycle trajectories that go out and
return along the same path. It evaluates return trajectories and compares
retrieval strategies:

- Temporal retrieval.
- Camera-distance retrieval.
- Field-of-view retrieval.
- Surfel-indexed retrieval.

VMem metrics include:

- LPIPS, PSNR, SSIM for generated view quality.
- FID for generated-view distribution quality.
- Rdist and Tdist for camera-control accuracy, using DUSt3R-extracted poses.
- Context-view count K and speed/cost trade-off.

What matters for us:

- We need a cycle/revisit protocol over Newton 60s outputs.
- We need pose-control error, not just image metrics.
- We need retrieval-strategy ablations that match our policies.
- We need context budget versus speed and quality.

### SPMem / Spatial Memory Occlusion

Source: https://arxiv.org/abs/2606.10299

This is not a video-generation paper, but it directly attacks a retrieval
failure mode relevant to our policies: spatial recall and visibility are not
the same thing. A coordinate or FOV score can recall a location behind a wall,
but that does not mean it is visible for the current query.

Reported ideas include:

- Delta-Hit@5 for spatial recall.
- Separate visibility scoring.
- Behind-wall false-visible tests.
- Ray/voxel DDA visibility predicates.
- Exact tests such as McNemar for visibility improvements.

What matters for us:

- FOV overlap alone is not enough.
- A memory policy can retrieve a spatially nearby but visually wrong/occluded
  frame.
- We should measure false-visible or occlusion-incorrect retrieval when labels
  or geometry are available.

### WorldMem

Source: https://arxiv.org/abs/2504.12369

WorldMem evaluates long-term world simulation with memory frames plus states
such as poses and timestamps. The important parts are not just PSNR/LPIPS/rFID,
but how they slice the evaluation:

- Within-context-window versus beyond-context-window.
- Memory retrieval strategy ablations: random, confidence filter, similarity
  filter.
- Timestamp/time-condition ablation for dynamic world state.
- Memory context length ablation.
- Predicted-pose versus ground-truth-pose ablation.
- Retrieval latency, memory bank size, memory usage, and inference speed.

What matters for us:

- Report 60s policies by horizon/gap, not only one mean.
- Separate static revisit memory from dynamic/time-state memory.
- Add retrieval latency, memory footprint, and policy budget to summaries.
- Track when extra memory becomes noise instead of help.

### OmniMem

Source: https://arxiv.org/abs/2605.30519

OmniMem is KV-memory retrieval rather than frame-memory retrieval, but its
measurement framing is useful:

- Long-video quality.
- Temporal consistency.
- Dynamic degree / motion dynamics.
- Long-range memory access versus local-window bias.
- Union explosion / selected-memory buffer size.
- Latency and memory footprint.
- Head/query-specific retrieval specialization.

What matters for us:

- A good policy should not collapse to local recent frames.
- Measure long-range retrieval rate and local-bias.
- Measure selected-memory size/cost, not just output quality.
- Dynamic content should not be suppressed by over-conservative memory.

### I3DM

Source: https://arxiv.org/abs/2603.23413

I3DM is directly relevant because it criticizes naive camera-FOV retrieval:
FOV overlap ignores occlusion and can retrieve historical frames that are not
actually visible from the target view. It also evaluates revisit consistency,
generation fidelity, and camera-control precision.

What matters for us:

- Add an occlusion-aware retrieval audit if possible.
- Distinguish "pose/FOV relevant" from "visibly useful".
- Track camera control precision separately from visual consistency.

## Metrics We Should Actually Add

### 1. Retrieval Quality Metrics

These are the most important for comparing `baseline`, `fifo_*`, `ri_*`,
`slam_*`, and future policies.

- `oracle_overlap_capture`: selected overlap divided by the best available
  overlap for that target.
- `oracle_overlap_gap`: best available overlap minus selected overlap.
- `hit_at_1_oracle`: selected frame equals oracle best frame.
- `hit_at_k_oracle`: selected frame is within the oracle top-k candidates.
- `mrr_oracle`: reciprocal rank of the selected frame under oracle overlap.
- `ndcg_overlap`: ranking quality if we log all candidate overlap scores.
- `long_range_hit_rate`: selected frame age above one or two chunks.
- `local_bias_rate`: selected frame age inside the recent context window.
- `redundant_selection_rate`: multiple selected frames in a section are near
  duplicates by frame index or pose.
- `unique_view_coverage`: number of distinct pose/view clusters covered by
  selected memories.

Already close in repo:

- `utils/analyze_trace_usefulness.py` has upperbound alignment against
  unbounded trace selections.
- `utils/analyze_memory_policies.py` has overlap-label coverage/oracle recall.
- Need all-candidate logging for MRR/NDCG/top-k ranking.

### 2. Revisit / Cycle Metrics

This should follow MemCam and VMem, but with better slicing:

- Define return/cycle pairs from trajectory structure or overlap labels.
- Compare outward and return views at matched or near-matched camera poses.
- Report by gap bucket: one chunk, two chunks, four chunks, 60s tail.
- Report worst-k failures, not only mean.

Metrics:

- `cycle_lpips`
- `cycle_fid_or_fvd`
- `cycle_clip_region_score` only if used as a secondary semantic diagnostic.
- `cycle_identity_or_region_consistency` if object/region masks are available.
- `return_failure_rate`: percentage of matched return views above a failure
  threshold.

Pixel metrics can be included only to match MemCam/VMem tables; they are not
the central claim.

### 3. Camera-Control Metrics

From VMem/I3DM:

- `Rdist`: rotation distance between intended/GT pose and generated estimated
  pose.
- `Tdist`: translation distance after sequence-relative normalization.
- `loop_Rdist`: generated return pose versus generated starting pose.
- `loop_Tdist`: generated return pose versus generated starting pose.

Needed implementation:

- Run DUSt3R/COLMAP/CUT3R pose extraction on generated frames.
- Compare estimated generated poses with planned poses from the manifest.

### 4. Occlusion-Aware Retrieval Metrics

From SPMem, VMem, and I3DM:

- `false_visible_rate`: selected memory has high pose/FOV score but should be
  occluded or non-visible.
- `visible_precision_at_1`: selected memory is actually visible/useful.
- `behind_wall_hit_rate`: stress-test metric for occlusion scenes.
- `fov_vs_visible_gap`: FOV overlap score minus visibility-aware score.

Needed implementation:

- Use depth/pointmap/geometry if available.
- Or start with Context-as-Memory overlap labels as a proxy, then add an
  occlusion stress subset later.

### 5. Long-Range Memory Behavior

From VMem/OmniMem/WorldMem:

- `age_distribution`: selected memory age histogram.
- `long_range_selection_rate`: age > 152, age > 304, etc.
- `memory_reuse_gini`: whether one frame dominates retrieval.
- `selection_entropy`: whether policy uses diverse memories.
- `view_cluster_entropy`: diversity of selected camera views.
- `budget_saturation_rate`: how often budget is full.
- `fallback_rate`: how often no useful memory was selected.

Some of this already exists in `utils/analyze_access_traces.py`; the missing
piece is pose/view clustering and explicit local-bias reporting.

### 6. Dynamic / Time-State Metrics

From WorldMem:

- `event_state_recall`: if an object/state changes, does the later generation
  preserve the changed state?
- `stale_memory_error_rate`: policy retrieves an old state when a newer state
  should override it.
- `time_condition_ablation`: compare policies with/without recency or timestamp
  awareness.

This needs labeled dynamic scenes or synthetic probes.

### 7. Cost And Scaling Metrics

From MemCam, VMem, WorldMem, OmniMem:

- `seconds_per_frame`
- `retrieval_latency_ms`
- `peak_vram_gb`
- `stored_memory_size`
- `candidate_count`
- `selected_context_count`
- `quality_vs_budget`
- `quality_vs_latency`

For Newton 60s, every policy summary should include cost columns next to
retrieval and revisit columns.

## Immediate Plan For Current Newton 60s Outputs

No new generation needed:

1. Use `trace_usefulness_analysis` as the first serious metric source.
2. Add missing local-bias and long-range retrieval rates to
   `utils/analyze_access_traces.py`.
3. Add a policy summary joiner:
   - eval metrics from `eval/`
   - access trace metrics
   - trace usefulness/oracle metrics
   - runtime/status metrics
4. Report every metric by policy and budget:
   - `baseline`
   - `fifo_b32`, `fifo_b64`, `fifo_b128`
   - `ri_b32_dino_rgb`, `ri_b64_dino_rgb`
   - `slam_b16_covisibility`, `slam_b32_covisibility`,
     `slam_b64_covisibility`, `slam_b96_covisibility`

Light new analysis:

1. Add cycle-return pair construction from the manifest trajectory.
2. Use it to define *which frames should be compared*, but do not present raw
   PSNR/SSIM as the main memory result.
3. Add DUSt3R/COLMAP pose extraction if we want VMem-style Rdist/Tdist.

Next-run instrumentation:

1. Log all retrieval candidates per query:
   - frame index
   - age
   - overlap / covisibility
   - policy score
   - rank
   - selected flag
2. Log retrieval latency and memory size.
3. Log per-section selected context set.

## What Not To Do

- Do not claim PSNR/SSIM/MAE/RMSE are memory metrics.
- Do not add a random script just because it is easy to compute.
- Do not call DINO/CLIP/LPIPS "retrieval metrics"; they are output similarity
  diagnostics unless tied to a retrieval/cycle/visibility protocol.
- Do not average away the 60s tail; memory failures are tail failures.
