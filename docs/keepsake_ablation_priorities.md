# KEEPSAKE Ablation Priorities

Scope: assessment of the manuscript and critique supplied on September 25, 2026.
Compute constraint: approximately 15 elapsed hours on one H100.
The component pilot is implemented below; no new results or submitted job are
claimed by that implementation. The sensitivity and update-rule replays are now
implemented separately; see [Existing-Data Evidence Jobs](paper_evidence_overnight.md).

## What Exists

- The memory-budget sweep tests capacity, not the pose/appearance or priority terms.
- The original component job tests alpha=0, 0.65, 1 with the priority formula fixed.
  That is a valid affinity-component ablation, not a complete method ablation.
- Latest confirmed full-job status: 831412 failed during control VBench evaluation.
  Its serial control gate means neither endpoint started in that job. The planned
  `new_videos: 30` field was not a completed-video count.
- No completed endpoint set or final component table has been found locally.
  Separate earlier pilot outputs on Newton have not been exhaustively audited.
- The VBench wrapper now microbatches background CLIP encoding and checkpoints
  each dimension. Local tests do not certify Newton execution.

## Minimum Component Table

Use the same B32, scenes, initial images, poses, seed, denoising settings, descriptor
extraction, protection rules, and query mechanism for every row. Change only the
indicated component. In the formulas below, phi is the manuscript's degree score.

| Setting | Affinity | Retention priority | Question |
| --- | --- | --- | --- |
| Full KEEPSAKE | alpha=0.65 | phi(c) + 0.25(1-k_max) | Reference |
| Without appearance | alpha=1 | Unchanged | Is appearance useful? |
| Without pose | alpha=0 | Unchanged | Is pose useful? |
| Without closest-substitute term | Unchanged | phi(c) | Does the strongest alternative add value? |
| Without degree term | Unchanged | 0.25(1-k_max) | Does the number of alternatives add value? |

Keep T=0.65 fixed for the affinity removals. These test signal removal at the
default threshold; they do not establish superiority over separately tuned
single-signal methods. Log edge density and isolated-node frequency because
changing affinity also changes which edges survive the fixed threshold.

The generation CLI now supports both affinity removals and
`--keepsake_priority_mode full|degree_only|closest_only`. Default scores are
unchanged. Priority mode is recorded in the actual retrieval traces as well as
the runner metadata. The old full-job launcher still runs only affinity removal.

## One-H100 Pilot Design

Implemented scope: three prespecified distinct scenes, nominal 30-second
rollouts, all five settings, for 15 newly generated videos including a matched
full-method control. The selection is `random.Random(0).sample` of the fixed
15-distinct-scene 60-second manifest, followed by source-row sorting. Keep every
scene in the report. The pilot uses 913 frames (12 whole 76-frame chunks plus the
initial image) at 30 FPS, following the host's chunk-aligned convention; it does
not assert exactly 900 frames. Controls are regenerated to match configuration
and generation seed 42, with 50 denoising steps and unchanged B32.

At the previously observed roughly four hours per 180-second video, linear
duration scaling suggests about ten GPU-hours of generation. That is a planning
estimate, not a measured 30-second throughput or a 15-hour completion guarantee.
Model startup, retention, retrieval and evaluation must also fit. Time the first
complete video before claiming the whole workload fits; do not silently truncate
videos or change denoising settings to rescue a deadline.

Primary output: paired per-scene LPIPS, PSNR/SSIM, and all scene-level differences.
Three scenes are a small mechanism pilot, not the 15-video main evaluation and
not evidence of broad parameter robustness. Do not make three-video FVD the
primary conclusion. The pilot does not run FVD or VBench. This avoids presenting
a three-video FVD as strong distributional evidence, and leaves generation as the
priority within the allocation. Metric sampling is every 30th RGB frame, including
index zero, yielding 31 matched sampled frames per video; trajectories, not sampled
frames, receive equal weight in the summary. There are no inferred significance
claims or artificial frame-level confidence intervals.

After syncing the changes to Newton, submit just this job:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_keepsake_component_pilot.sbatch
```

The launcher requests one H100, eight CPUs, 96 GiB RAM, and a 15-hour time limit.
It runs in the existing `memcam` environment and does not modify the VBench checkout.
Optional CPU-only plan inspection before submission:

```bash
"$HOME/.conda/envs/memcam/bin/python" paper/run_keepsake_component_pilot.py \
  --output "$HOME/memcam_results/keepsake_component_pilot_30s_n3" --dry-run
```

Default output: `~/memcam_results/keepsake_component_pilot_30s_n3/`.
Final files are `ablation.csv`, `ablation.tex`, and `per_video.csv`, including paired
differences. `status.json` reports overall completion; `progress.json` records the
actual completed cell count. `partial_per_video.csv` is explicitly not a finished
ablation. Logs are `keep_components_JOBID.out/.err` and per-cell
`cells/SETTING/row_NNN/generation.log` and `quality.log`.

GT paths, input hashes, checkpoint availability, a real CUDA kernel and the LPIPS
model are checked before generation. Videos are validated against their frame
count, traces, actual affinity weight and priority mode. Each cell is evaluated
immediately. Completed videos and metrics have separate hash-checked receipts;
resubmitting unchanged code into the same output resumes rather than regenerates.
The final table requires all 15 cells. Synthetic tests exercise failed metric
recovery, cohort validation, unchanged default scores, and output generation;
Newton GPU execution remains unverified until the submitted run actually executes.

The existing two-GPU, 180-second launcher remains a different, much larger
experiment. Do not submit it with a 15-hour limit expecting this pilot.

## More Evidence Without Full Generation

Replay fixed common histories with cached descriptors and recorded poses on the
full matched cohort. Before running, specify one-at-a-time parameter perturbations
around alpha, T, tau, beta and lambda; keep the default and all outcomes.
Measure retained-set changes, edge density, retention/selection gaps, and update
time. Ground truth is for offline scoring only, never policy decisions.

Two update comparisons answer different questions:

- Recompute after every deletion versus one-shot ranking: isolate within-update
  recomputation with a fixed affinity matrix, recomputing only degrees/maxima.
- Reassess at each update versus frozen-at-admission priorities: test the stated
  dynamic reassessment principle. Define initialization of admission scores and
  keep later scores frozen in that variant.

These are counterfactual retention diagnostics on fixed pixels, not new generated
video quality measurements. They cannot fill LPIPS/FVD generation-ablation cells.

## Separate Manuscript Corrections

- The pasted 60-second main table uses the 180-second MemCam final count (5,397)
  and timing values. Separate or relabel the timing horizon and verify counts.
- The MemCam budget table is labeled 180 seconds but its VBench aggregates were
  supplied from the 60-second suite. Do not treat them as 180-second measurements.
- The metrics paragraph says no aggregate is constructed, while tables show one.
  Define the actual normalized weighted-six aggregate consistently.
- Define RI and present the scope of comparisons consistently. Removing a method
  because a metric favors it is not an ablation.
- An observed 35.1% FVD reduction is a valid descriptive point estimate; it is not
  automatically a statistically significant or seed-robust improvement. Add paired
  trajectory uncertainty from original features where available.
- The FOV-controlled null result limits a claim of intrinsic content-quality
  detection; it does not logically erase independently measured output improvements.

The external critique's acceptance scores are opinions, not reliable forecasts.
A full multi-parameter factorial generation sweep is not required to answer the
component questions above. A small pilot must still be labeled as such.
