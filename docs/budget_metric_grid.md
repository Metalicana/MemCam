# Matched MemCam Metric Budget Grid

This launcher evaluates existing videos; it does not generate or retrain.
It covers **MemCam only**, not separate WorldMem or VMem repositories.

## Scope

- Complete retention once.
- FIFO, RI, GeoCov, K-center, and MCE at B16, B32, B64, B128.
- Exactly the same fifteen 60-second manifest entries in all 21 configurations.
- Four isolated task types per configuration: LPIPS/FVD, standard VBench,
  VBench-Long, and CUT3R. There are 84 possible GPU task slots.
- Six dimensions each for standard VBench and VBench-Long.
- FVD uses the entire 15-video distribution, never averages individual FVDs.
- CUT3R computes provisional camera results. Successful execution does not
  resolve the existing ground-truth calibration failure.

## Submit From Newton

After transferring the updated code to `$HOME/MemCam`, activate `memcam` and run:

```bash
cd "$HOME/MemCam"
python utils/run_budget_metric_grid.py prepare \
  --output "$HOME/memcam_results/budget_metrics_60s_v1" \
  --reuse-audit "$HOME/memcam_results/cohort_audit_20260914_135301/audit.json" \
  --parallel 2 --submit
```

Preparation checks all 315 required policy/video files for existence and
nonzero size before submitting anything. It freezes the manifest, source
paths/sizes/mtimes, and local evaluator Python code. GPU workers additionally
decode-count frames with ffprobe before starting evaluators.

This submits one H100 array (at most two GPUs concurrently) and an `afterany`
CPU collector. Each GPU task requests eight CPU cores, 128 GB RAM, and
24 hours. The existing Newton node exclusions are retained. The two evaluator
environments are `memcam` for LPIPS/FVD/CUT3R and `vbench` for VBench/Long.
CUDA device selection is left to Slurm. Change `--parallel` to control usage.

Use `--dataset-root /path/to/context_dataset` only when the manifest's existing
ground-truth paths need remapping. The VBench repository defaults to `$HOME/VBench`.

To inspect the plan without submitting, omit `--submit`. The frozen `plan.json`
contains every run, evaluator, reused source, and selected historical suite.

## Reuse and Recompute Rules

`--reuse-audit` is optional. It accepts legacy LPIPS/FVD files only after
re-reading their artifacts and checking:

- Exactly fifteen canonical identities, no duplicates or mixed policy rows.
- All rows completed; no short-video results.
- Full 60-second evaluation, with no shorter frame cap.
- LPIPS frame stride 30 and image size 224.
- FVD 16-frame clips, four clips/video, stride four, image size 224,
  StyleGAN-V I3D backend, epsilon 1e-6, sixty total clips, recorded detector path.

Missing or incompatible cells are recalculated. If only one of LPIPS/FVD is
missing, only that metric is requested. If both are reusable, the task is
already complete and is not submitted to a GPU. The source result artifacts
are hashed to detect changes after planning.

Legacy reuse establishes recorded identity and settings, **not** historical
video bytes, checkpoint hashes, or evaluator environment parity. Those old
records do not contain enough evidence to establish all of that retrospectively.
The plan and coverage report explicitly identify reused quality cells. Omit
`--reuse-audit` to recompute all LPIPS/FVD in the current environment.

Standard VBench is freshly run on an isolated fifteen-video symlink directory
for every configuration. Existing 30-video aggregates are never copied.
Fresh scores are equal-video means; per-video imaging quality is normalized
by 100, as in the existing matched diagnosis.

Long/CUT3R reuse the latest non-dry-run suite under `matched15_metrics_60s`
when one exists. The existing resume validator checks each completed row,
manifest, source path/size/mtime, and Long grouping adapter. Failed rows are
archived and rerun, not deleted. Recorded package freezes must match the
current environment; otherwise the task fails rather than mixing versions.
Legacy upstream/checkpoint parity is not retroactively certified.

Use `--fresh-suites` during preparation to avoid historical Long/CUT3R reuse.
A fully fresh grid uses neither `--reuse-audit` nor historical suites:

```bash
python utils/run_budget_metric_grid.py prepare \
  --output "$HOME/memcam_results/budget_metrics_60s_fresh_v1" \
  --fresh-suites --parallel 2 --submit
```

New task attempts have their own output directories. New tasks within a grid
must agree on their package/runtime fingerprints and VBench source/config
hashes. Do not update evaluator Python files, the environments, VBench code,
or the input videos while the grid is running. Model-weight cache hashes are
not fully captured; preserve those caches as well.

## Progress and Retry

```bash
python utils/run_budget_metric_grid.py report \
  --plan "$HOME/memcam_results/budget_metrics_60s_v1/plan.json"
```

The terminal shows 21 compact rows. Full scores, sources, errors, and log
locations are saved in `scores.csv`, `coverage.csv`, and `report.json` beside
the plan. CUT3R is always marked uncalibrated. Incomplete tasks have no final
15-video aggregate. The report exits nonzero until every task is complete.

After the previous array has stopped, retry only unfinished task slots:

```bash
python utils/run_budget_metric_grid.py submit \
  --plan "$HOME/memcam_results/budget_metrics_60s_v1/plan.json" --parallel 2
```

Do not resubmit while the previous array is still active. Task and suite locks
prevent simultaneous writes but do not prevent redundant Slurm allocations.
Long/CUT3R resume validated rows. An interrupted standard VBench or quality
task starts a fresh cohort attempt, preserving the previous attempt's output.
An incompatible source/environment requires a fresh plan, not bypassing checks.

The automatic collector runs even if some array tasks fail. Its failed exit
status means incomplete coverage, not that successful metrics were discarded.
No CUT3R calibration job is included: the convention/alignment problem needs
a separate validated fix before those scores can become paper evidence.
