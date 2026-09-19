# Keepsake Affinity-Weight Ablation

Change only the affinity mixture, K = alpha * pose_similarity + (1-alpha) *
appearance_similarity. Test alpha = 0, 0.50, 0.65, 0.80, 1. Hold the separate
co-visibility threshold at 0.65, required support at three neighbors, all utility
coefficients, B32, protected-frame rules, DINO extraction, generator, retriever,
prompts, cameras, generation seed (42), and 50 denoising steps fixed.

Alpha zero removes pose from the affinity; alpha one removes appearance from the
affinity. These are signal ablations, not optimized endpoint implementations:
descriptor extraction is still performed. Keeping the threshold fixed tests this
mixture in the existing controller, including its effect on graph density.

The default is the matched 15-video, 180-second manifest. This requires new
closed-loop generation, not merely rescoring saved videos. The full sweep is 75
videos: 15 per setting. The 0.65 setting is a fresh control; existing outputs are
not silently reused. A 60-second study can be requested explicitly with
`--duration 60`; do not mix horizons in its results table.

## Launch

After syncing code to Newton, first test one control trajectory:

```bash
cd "$HOME/MemCam"
sbatch --array=0 slurm/newton_keepsake_weight_ablation.sbatch
```

Inspect that task's stdout, stderr and `tasks/000/status.json`. The job uses the
environment Python directly, no Conda activation and no short startup timeout.
After a successful real-node test, the full fresh sweep is:

```bash
sbatch slurm/newton_keepsake_weight_ablation.sbatch
```

This submits 75 array tasks, at most two simultaneous GPUs, one video per task.
Task blocks 0-14, 15-29, 30-44, 45-59 and 60-74 correspond to alpha 0.65, 0,
0.50, 0.80 and 1. The output defaults to
`~/memcam_results/keepsake_weights_JOBID/`. These are generation jobs only.
The launcher freezes the manifest/code/seed contract and records coefficients in
task status, run status, and memory traces. An existing task directory is rejected;
inspect a failed task before choosing a fresh output directory for a retry.

## Evaluation

Evaluate LPIPS, cohort-level FVD and the six standard VBench dimensions on the same
15 identities for each completed setting, with the same metric configuration and
horizon. Do not average per-video FVD. Reuse the established evaluator scripts;
the generation job does not automatically submit evaluation jobs.

Table columns: pose weight, appearance weight, FVD, LPIPS, and VBench dimensions
or the previously defined six-dimension aggregate. The key comparisons are the
current mixture against each single-signal endpoint; 0.50 and 0.80 test local
sensitivity. No direction of improvement is assumed.

Local tests cover affinity endpoints, unchanged default scores, pinned frames,
CLI rejection of invalid weights, parameter wiring, and the 75-cell task map.
They do not replace the real-node generation smoke test.
