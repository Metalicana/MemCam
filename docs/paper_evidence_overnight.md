# Existing-Data Evidence Jobs

These jobs supplement, not replace, the component-generation pilot (848505).
They do not regenerate videos or change the queued pilot's code. The new GPU
job waits for that pilot to end. Its time allowance is an additional two hours,
not part of the pilot's 15 hours. Each CPU job has an eight-hour cap. Queue waits
are separate, and these caps are not completion-time guarantees.

## Submit Once

After transferring the new files to Newton:

```bash
cd "$HOME/MemCam"
"$HOME/.conda/envs/memcam/bin/python" paper/submit_paper_evidence.py \
  --after-job 848505
```

Default output: `~/memcam_results/paper_evidence_20260925/`.
The command freezes Python analysis code into that directory and submits:

1. `cpu`: existing-data statistics, archive/profile audit, retrieval timing and
   fixed-history update replay. No GPU; independent of the generation pilot.
2. `gpu`: one `nvidia_h100_80gb_hbm3` on `highgpu`, after the pilot terminates.
   Extracts I3D features from existing videos only, then releases the GPU.
3. `fvd-score`: CPU bootstrap and paired randomization after the other two jobs
   terminate. This dependency permits reporting an incomplete upstream stage;
   it does not treat a failed extraction as valid input.

`jobs.json` records each accepted submission immediately. Re-running the same
command reuses active/completed job IDs; it does not submit duplicate active
jobs. If any submission fails, the already-accepted IDs remain recorded. Do
not change the original manifests, metric report, or frozen snapshot during a
run. Use a new output for a different experiment/configuration.

The submitter does not modify/cancel the component job or resubmit it. Keep only
one copy of that pilot queued/running. The highgpu hardware request is based on
the supplied Newton inventory, not the H100 PCIe name used by `normal`.

## Experiments And Outputs

### Main-Table Scalar Uncertainty

`scalar_statistics/summary.csv`, `contrasts.csv`, and `per_video.csv`.

Reads the exact raw LPIPS and VBench files referenced by
`budget_metrics_60s_820776/scores.csv`. Requires the complete original matched
15-scene, 60-second cohort for Unbounded, FIFO, MCE, K-center, RI and KEEPSAKE.
Rejects changed point estimates, missing/duplicate videos and incorrect identities.
No interval is manufactured from an aggregate CSV alone.

Reports 5,000 whole-scene paired bootstrap draws (seed 17). Computes exact
two-sided paired sign-randomization tests under within-scene label exchangeability,
and Holm correction across the five comparisons and eight scalar metrics.
These are equal-video averages, consistent with the existing metric-grid report.
VBench dimensions remain on their original 0..1 scale; only the custom aggregate
is multiplied by 100. The aggregate is the existing unclipped normalized weighted
six-score formula, not the official full VBench score:

```
100 / 5.5 * sum_d w[d] * (score[d] - lo[d]) / (hi[d] - lo[d])
subject:    lo=.1462 hi=1     w=1
background: lo=.2615 hi=1     w=1
motion:     lo=.706  hi=.9975 w=1
dynamic:    lo=0     hi=1     w=.5
aesthetic:  lo=0     hi=1     w=1
imaging:    lo=0     hi=1     w=1
```

### FVD Uncertainty

`fvd_statistics_60s/` and `fvd_statistics_180s/`, each with `summary.csv`,
`contrasts.csv`, and the joint bootstrap samples. Features are checkpointed per
video/policy in `fvd_features_60s/` and `fvd_features_180s/`.

- 60 seconds: all six B32 configurations, including RI and unbounded.
- 180 seconds: unbounded, FIFO-B32, KEEPSAKE-B32, all fifteen matched scenes.
- Both use the production StyleGAN-V I3D detector, four 16-frame clips per video,
  frame stride four, size 224. Same clip sampler as the current quality suite.
- Check source-video and sampled-GT hashes; require the cached detector at
  `~/hf_cache/memcam_fvd/i3d_torchscript.pt`. Do not silently change checkpoints
  or fall back to CPU if CUDA initialization fails.
- Resample complete scenes with all clips and methods paired. Recompute cohort
  FVD on every draw. Never average per-video FVDs.
- 2,000 bootstrap draws and 2,000 paired whole-scene label swaps per comparison,
  seed 17. Two-sided Monte Carlo p-values use the plus-one correction.
- Relative-difference intervals use paired bootstrap ratios, not a ratio of
  separately computed endpoints. Negative differences favor KEEPSAKE.
- `joint_tests/joint_multiplicity.csv` applies Holm across all 47 scalar/FVD
  contrasts when all corresponding source stages are complete.
- `fvd_statistics_60s/reported_score_check.csv` compares the freshly recomputed
  FVD points against the existing main-table report. Investigate disagreements;
  do not attach the new intervals to a different historical point estimate.

One recorded rollout per scene does **not** measure random-seed variance. These
are scene-sampling uncertainty estimates conditional on the recorded runs.
Finite-sample FVD bias remains; neither nonsignificance nor overlapping intervals
establishes equivalence. The 30-second component pilot is a separate cohort.

### Query Latency And Counts

`latency_60s/latency_summary.csv`: all six B32 methods plus KEEPSAKE B16/B64/B128,
measured on the same CPU allocation. One prespecified first manifest trajectory,
eight evenly sampled sections, one actual target query per section, three timing
repeats, one CPU thread. Reuses the existing production FOV retrieval replay.
This is query latency, excluding generation, feature encoding and memory updates.
The repeated measurements do not constitute fifteen independent videos.

`archive_audit/archive_summary.csv`: final bank sizes audited from complete access
traces across all fifteen trajectories, with explicit 60-second/frame-count labels.
This supplies a trace-verified answer to the 5,397-versus-60-second table issue.

`historical_profiles.csv` and `profile_coverage.csv`: records available historical
host RAM, GPU allocation, bank bytes and phase times. Missing or invalid profiles
stay visible. These runs are not certified hardware-matched, so their telemetry
must not become a matched end-to-end speedup or peak-VRAM ranking. A successful
archive audit is not a claim that every profile exists.

### Fixed-History Sensitivity And Update Rules

`update_replay/retention_summary.csv`, `paired_default_minus_variant.csv`,
`trajectory_summary.csv`, `query_retention.csv`, `updates.csv`, and bank snapshots.

Replays all fifteen complete 60-second unbounded histories, using the freshly
encoded GT/baseline DINO cache. Validates cache metadata, encoder agreement,
source/video/GT hashes, baseline traces and pose-index coverage. No new feature
extraction or video generation is performed.

Sixteen prespecified configurations:

- Default: alpha=.65, T=.65, tau=3, beta=.5, lambda=.25, B32.
- One-at-a-time alpha=0,.35,.5,.8,1; T=.5,.8; tau=1,5; beta=0,1;
  lambda=0,.5. All other values stay fixed. Keep all outcomes.
- Iterative deletion: hold each update's affinity matrix and pose normalization
  fixed, recomputing degrees and closest-alternative affinities after each eviction.
- Frozen-at-admission: compute finite scores when a frame first arrives; never
  update those scores later. Initial/current-endpoint protection still moves under
  the same rules and does not store infinite priorities.

The default formula is checked against the actual production scoring function.
All variants see one frozen generated stream; GT enters offline oracle scoring
only, never admission/eviction. Record edge density and isolated-node fraction
before eviction, retained IDs, retained-bank oracle mismatch, retention gap and
CPU update time. Four fixed targets per retrieved section, equal trajectory means.
Update timings exclude feature encoding and are not query timings.

This is a common-pixel retention diagnostic. Cached source DINO features need not
be byte-identical to descriptors computed online. We do not claim a reproduction
of each policy's original generated history. There are no actual counterfactual
reads or generations here, so this replay exports **no selection-gap, LPIPS or
FVD generation-ablation scores**. The existing generation pilot supplies the
separate component-removal quality experiment.

## Monitoring And Completion

```bash
cat "$HOME/memcam_results/paper_evidence_20260925/jobs.json"
squeue --me
cat "$HOME/memcam_results/paper_evidence_20260925/status_cpu.json"
cat "$HOME/memcam_results/paper_evidence_20260925/status_gpu.json"
cat "$HOME/memcam_results/paper_evidence_20260925/status_fvd-score.json"
```

Logs are inside the same directory: `cpu_JOB.out/.err`, `gpu_JOB.out/.err`,
`fvd-score_JOB.out/.err`. Each independent stage has a receipt or `error.txt`.
Only `status: complete` means that phase finished. A failure does not prevent
other independent stages from running, and the phase exits nonzero if incomplete.
Hash-checked stage/feature receipts allow reuse of valid completed work.

## Checklist Items That These Jobs Do Not Resolve

- New random seeds, new genuinely held-out scenes, or retrospective tuning history.
  Relabeling already-inspected scenes cannot make them a previously unseen test set.
- VMem, Memory Forcing, AnchorWeave/MosaicMem adaptations, or a SpMem host. These
  require verified code/checkpoints, a compatible reader contract and independent
  integration tests. No proxy is renamed to impersonate those systems.
- WorldMem evaluation/retraining/rollout extension. Its data and implementation
  live in another session; its FPS and latent/RGB mappings must stay explicit.
- The full fifteen-scene 180-second component-generation sweep. It remains distinct
  from the current three-scene 30-second pilot.
- Matched-hardware end-to-end generation resource measurements or CUT3R calibration.
- The author's development/test decisions, contributions, AI-use and ethics statements.
  Those must reflect what actually happened; software cannot supply missing facts.
- Manuscript references, citations, table formatting and causal wording in the final
  authoritative draft. The pasted critique is not itself ground truth, and its
  acceptance predictions are not guarantees.

Preserve the FOV-controlled null result. Do not search for a favorable controlled
subset and present it as the prespecified general effect. The new replay can support
statements about evidence availability, not prove content-quality detection or a
causal snowball mechanism.
