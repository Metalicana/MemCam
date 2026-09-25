# CPU-Only Parameter Sensitivity

This is separate from the running component-generation array. It requests no GPU,
does not submit additional jobs, does not install packages, and only reads existing
source videos, traces, poses and DINO caches. It does not extract new features.

## Launch

After transferring `paper/run_keepsake_sensitivity_cpu.py` and
`slurm/newton_keepsake_sensitivity_cpu.sbatch` to the existing Newton checkout:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_keepsake_sensitivity_cpu.sbatch
```

The existing replay/statistics helpers must already be present in that checkout.
The script requests two CPUs, 24 GB RAM and an eight-hour limit on `normal`.
The limit is not a measured runtime estimate. Cache/source hashing may take time;
progress distinguishes source verification from replay. No Newton run has been
performed by the local implementation or its synthetic tests.

Default cache: `~/memcam_results/context_memory_60s/gap_feature_cache_fresh`.
Both `baseline/` and `gt/` must contain all fifteen feature arrays and their JSON
sidecars; directory existence alone does not certify completeness. GT must have
fresh-encoding provenance. Original GT frames, baseline videos, traces and pose
files must remain available for hash/identity validation.

## Frozen Design

Seven unique settings on all fifteen original 60-second trajectories, B32:

| Setting | tau | beta | lambda |
| --- | --- | --- | --- |
| Default | 3 | .5 | .25 |
| tau low | 1 | .5 | .25 |
| tau high | 5 | .5 | .25 |
| beta zero | 3 | 0 | .25 |
| beta high | 3 | 1 | .25 |
| lambda zero | 3 | .5 | 0 |
| lambda high | 3 | .5 | .5 |

All settings see the same fixed unbounded generated stream and poses. Each keeps
its own evolving bank. Alpha=.65, affinity threshold=.65, budget, protection,
insertion order and one-shot update behavior remain fixed. The zero-beta setting
removes only beta/(degree+1), not the entire degree contribution. GT features are
used only for offline scoring, never to select retained frames. The replay checks
its default priorities against the production scorer.

Four fixed target queries per retrieved section, 92 per trajectory. Primary
diagnostic: retention gap (best retained-bank DINO mismatch minus best eligible
full-history mismatch). Also report bank-oracle mismatch, bank Jaccard versus
default, graph edge density, isolated fraction and CPU update time. These are
not actual retrieved-frame scores or counterfactual generated-video metrics.

Each trajectory receives equal weight. Report 5,000 paired trajectory-bootstrap
draws, seed 17, with descriptive 95% percentile intervals and paired differences.
No significance or equivalence claims are generated. Scene IDs define resampling
units; potential dependence between trajectories sharing an environment is not
modeled. One recorded generated stream does not estimate generation-seed variance.
All outcomes are retained; the sweep does not tune parameters or establish
held-out generalization.

## Resume and Outputs

Default output: `~/memcam_results/keepsake_sensitivity_cpu_60s_n15`.
An exclusive writer lock prevents simultaneous runs in that root. Code hashes,
Python/NumPy versions, cohort, configuration and cache sidecars are frozen in
`plan.json`; source code is also copied to `code/` for provenance. The job executes
the checkout and checks its code hashes between trajectories, so do not edit
experiment code during the run. Changed code/configuration needs a new root.

Each trajectory gets atomic `replay.json` and `receipt.json` files. Completed
trajectories are reused only after validating their sources and artifact hashes.
An interrupted trajectory restarts; there is no mid-trajectory checkpoint.
Resubmit the same batch command after the earlier job terminates. The runner does
not cancel jobs or automatically queue retries. Mixed-host CPU times on resumption
are diagnostic measurements, not a controlled speed comparison.

Final tables require all fifteen completed trajectories. Always check `status.json`
before treating existing CSVs as current; a failed resume can leave earlier reports
on disk. `summary.csv` reports variant-minus-default differences; negative is
better. `paired_default_minus_variant.csv` uses the opposite sign in its filename.

```bash
OUT="$HOME/memcam_results/keepsake_sensitivity_cpu_60s_n15"
cat "$OUT/status.json"
cat "$OUT/summary.csv"
cat "$OUT/interpretation.txt"
```

Other exports: `sensitivity.tex` (booktabs tabular fragment), `per_trajectory.csv`,
`query_retention.csv`, `updates.csv`, and per-trajectory bank snapshots inside
`cells/row_NNN/replay.json`. Logs are `keep_sens_cpu_JOBID.out/.err` in the submission
directory. This is a retention sensitivity appendix, not LPIPS/FVD sensitivity.
