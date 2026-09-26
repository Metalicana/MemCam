# Headline Uncertainty Follow-Up

Metrics only, using existing videos. No generation, tuning, or changes to the
running component array, CPU sensitivity job, or successful CI job 850442.
This is a post-hoc analysis of previously inspected trajectories, not a new test
set or a preregistered experiment. All listed comparisons are retained.

## Coverage

| Manuscript result | New analysis |
| --- | --- |
| Main MemCam 60s table | Unbounded, FIFO B32, MCE B32, K-center B32, KEEPSAKE B32; LPIPS and FVD |
| 180s plot and appendix table | Unbounded, FIFO B32, KEEPSAKE B32; LPIPS and FVD |
| MemCam 180s budget table | KEEPSAKE B16/B32/B64/B128; LPIPS and FVD |
| Main 60s VBench entries | Attempt existing raw scalar audit; all six recorded policies including RI; optional, separately reported |
| WorldMem tables | Separate session: `docs/worldmem_headline_uncertainty_handoff.md` |
| 90/360 degree native results | NOT resolved here: published baseline raw data and FVD protocol equivalence are unverified |
| Stored items and fixed B | Audit counts, not a made-up sampling CI for a deterministic cap |
| Retrieval timing | Existing n=1 timing pilot cannot provide a trajectory-population CI |
| 180s budget VBench | NOT resolved here: verify actual duration/source, do not reuse 60s intervals |

The draft still has 734.2 -> 476.6 / 35.1% in the abstract, introduction,
long-horizon text/figure/table and conclusion. Successful matched job 850442
instead reported 534.446507 -> 476.666804 (10.8% reduction), with a paired
FVD difference CI [-142.926289, -6.230711]. Its LPIPS difference CI includes
zero. Correct those repetitions consistently; do not mix corrected intervals
with the older points. The suite exports further point-estimate audits.

The supplied draft also claims six individual VBench dimensions without an
aggregate in Metrics, but its main/budget tables contain an Average/custom
aggregate. Resolve this definition and source-provenance inconsistency before
attaching intervals. Optional `legacy_scalar_60s` uses the existing documented
normalized weighted six-score aggregate, NOT official full VBench.

## Protocol

- Original full fifteen-trajectory manifest, no silent intersection or dropping
  failed scenes. Policies must share the exact target/GT mapping.
- 60s: AlexNet LPIPS, size 224, frame stride 30 including index zero; I3D FVD,
  four 16-frame clips per trajectory, stride four, size 224.
- 180s: LPIPS stride 90 including index zero; eight 16-frame I3D clips per
  trajectory, stride eight, size 224. This matches the successful CI run and
  the historical long-horizon evaluator, not the older four-clip evidence job.
- 2,000 paired whole-trajectory bootstrap draws, seed 17. Keep all methods,
  GT and all clips from a sampled trajectory together. LPIPS averages complete
  per-trajectory means equally. FVD is recomputed on the pooled clip features
  every draw; NEVER average per-video FVDs or treat 60/120 clips as independent.
- Export each policy's point and unadjusted percentile 95% interval, PLUS paired
  difference intervals and paired percentage-reduction intervals. Negative
  difference favors B32; positive percentage reduction favors B32. Budget
  contrasts use B32 minus each alternative, not the retrospectively best budget.
- Two-sided policy-label swaps within complete trajectories, 5,000 Monte Carlo
  draws, plus-one p-value correction. If the resampling-unit count permits full
  enumeration within that limit, enumerate instead. Both metrics use the same
  swaps; FVD pools features again after swapping. Assumption: policy labels are
  exchangeable within resampling units under the null; this was not a randomized
  policy-assignment experiment.
- Holm across BOTH metrics and all comparisons within each table: 8 main,
  4 long, 6 budget tests. Also export Holm across all 18 MemCam quality tests
  after all three tables complete. Unadjusted intervals are not simultaneous
  intervals; their zero exclusion need not agree with adjusted tests. Do not
  turn a bootstrap fraction into a p-value or change test families after results.
- Optional `--cluster-map path.json` maps every exact scene ID to an explicitly
  verified environment ID. All trajectories from that environment travel/swap
  together. Use a separate output directory for this sensitivity analysis.
  Without it, repeated environments are not accounted for. No new-seed variance
  or held-out generalization is measured; small-sample FVD bias remains.

## Transfer And Submit

The two new runtime files are local until transferred. Existing paired extractor
and evidence helpers must already be present on Newton; they are not modified.

From the Mac:

```bash
cd /Users/metalicana/projects_summer_2026/MemCam
scp paper/run_headline_uncertainty_suite.py \
  ab575577@newton.ist.ucf.edu:MemCam/paper/
scp slurm/newton_headline_uncertainty_suite.sbatch \
  ab575577@newton.ist.ucf.edu:MemCam/slurm/
```

From Newton's login shell:

```bash
cd "$HOME/MemCam"
sbatch slurm/newton_headline_uncertainty_suite.sbatch
```

One H100 HBM3 on `highgpu`, 48G host RAM, no array. Four hours is the job limit,
NOT a runtime estimate. Job 850442 took 5m13s for one pair; this suite has more
comparisons and permutation tests. No remote job has been submitted by this
local implementation. Do not submit duplicate copies into the same output root.

Output: `~/memcam_results/headline_intervals_all`. Preserves Slurm GPU mask,
uses the working memcam environment and bounded extraction batches, and never
runs VBench on full videos. LPIPS/I3D extraction is in foreground child processes.
Code and sampling settings are frozen by hashes; don't edit dependencies while
the job runs. A changed protocol/code requires a new output root.

Read-only reuse of `headline_uncertainty_180s_matched15` verifies its plan, original
source hashes, per-pair receipts and feature arrays. A stale prior cache is an
error, not permission to attach its intervals to a different point estimate.
Other pairs checkpoint after each complete trajectory. Identical KEEPSAKE/GT
scores and features are required across repeated pair extractions in each table.
The paired extractor is intentionally unchanged to preserve old cache validity.

If a pair is unavailable, other pairs and tables still run. The affected table
is not exported as a full cohort. A failed optional legacy scalar audit cannot
invalidate completed fresh LPIPS/FVD tables; its coverage remains explicit.
Overall `status: complete` means the THREE quality groups completed, not that
all manuscript claims, VBench protocols, or external systems are validated.

## Monitor, Resume, Download

```bash
OUT="$HOME/memcam_results/headline_intervals_all"
cat "$OUT/status.json"
tail -n 60 keep_ci_all_JOB_ID.out keep_ci_all_JOB_ID.err
find "$OUT/pairs" -name extract.log -print -exec tail -n 3 {} \;
```

After the previous job terminates, resubmit the same command to reuse validated
completed pairs/reports. A lock prevents simultaneous writers. An interrupted
trajectory is rescored; previously completed trajectories remain. No video is
regenerated. Already exported tables may remain after a failed retry: check
status/coverage and receipt validity before using them.

If extraction finished and only reporting needs retry, use
`paper/run_headline_uncertainty_suite.py report` inside a CPU allocation with
BLAS/OpenMP threads set to one and the same environment/output. It reads cached
pair features and receipt hashes, without rereading all source videos. No new
GPU allocation is needed for that phase.

Outputs per quality group: `summary.csv`, `contrasts.csv`, `per_video.csv`,
`bootstrap.npz`, `fvd_numerical_check.csv`, `analysis.json`, `table.tex`, receipt.
Top-level `all_quality_contrasts.csv` exists only after all 18 contrasts finish.
`coverage.csv` identifies failures and optional missing scalar results.
`reports.zip` packages reports and new extraction receipts/features (no videos).

From the Mac after completion:

```bash
scp ab575577@newton.ist.ucf.edu:memcam_results/headline_intervals_all/reports.zip \
  "$HOME/Downloads/memcam_headline_intervals_all.zip"
```

Read `summary.csv`'s `manuscript_estimate` and `recomputed_minus_manuscript` for
every row. Investigate discrepancies; the corrected complete-cohort estimates
and intervals must be reported together. Never keep only favorable contrasts.

## Sample-Size Wording

Suggested wording, provided the grouping assumptions are disclosed:

> We compare policies on the same fifteen trajectories, using paired
> trajectory-level resampling to quantify uncertainty conditional on the
> recorded rollouts. Pairing reduces variability due to trajectory difficulty
> but does not establish representativeness or account for generation-seed
> variation. We report effect sizes and intervals rather than interpreting
> nonsignificance as equivalence; small effects may remain unresolved.

There is no defensible universal minimum detectable effect from N=15 alone.
It depends on paired variability and, for FVD, the nonlinear estimator. State
actual interval widths, not retrospective observed-power claims.

## Local Verification

Forty tests pass across `test_headline_uncertainty_suite`,
`test_paired_quality_uncertainty`, and `test_paper_evidence`. They include complete
synthetic suite export, CPU report-only reuse, checksum rejection, missing-pair
isolation, joint Holm output, clustered swaps, and pooled-FVD checks. Bash syntax
and CLI checks pass. These are local CPU tests, not real Newton GPU validation.

The separate CPU-sensitivity suite passes its six tests in a fresh process.
Combining it after these tests in one process exposes an existing source-file
discovery issue in `run_keepsake_sensitivity_cpu.code_files`: a Torch module's
synthetic `_classes.py` path is treated as a real repository file. That unrelated
runner is unchanged here, especially while its frozen job may be active. The
new headline runner uses an explicit source-file inventory instead.
