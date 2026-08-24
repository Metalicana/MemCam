# MemCam Project Handoff for a New GPT Account

Last updated: 2026-08-22

This is the durable context for continuing Abdul's bounded-memory research
project. Read this before proposing methods, editing code, interpreting old
results, or giving cluster commands. The detailed scientific evidence lives in
[`memcam_diagnostic_experiments_summary.md`](memcam_diagnostic_experiments_summary.md),
and the claims that are currently allowed or forbidden live in
[`iclr_reviewer_risk_register.md`](iclr_reviewer_risk_register.md).

## Bootstrap prompt for a fresh session

The following block can be pasted into a new GPT session:

```text
We are working on Abdul's MemCam research fork. First read:
1. docs/project_handoff_context.md
2. docs/memcam_diagnostic_experiments_summary.md
3. docs/iclr_reviewer_risk_register.md
4. docs/cluster_runbook.md
5. paper/README.md

Work in the local Mac repository at
/Users/metalicana/projects_summer_2026/MemCam. Inspect the code before making
claims. Make and test edits locally. Never push to GitHub; Abdul pushes and then
runs git pull on Newton or CECSL. Do not submit jobs yourself. Prefer one
standalone sequential sbatch job over arrays or dependency chains unless Abdul
explicitly requests otherwise. Before suggesting GPU work, say exactly what it
will run, how many videos it will generate, and the expected rough cost.

Use normal, concise language. Define every symbol. Do not claim novelty,
causality, or victory before evidence exists. Do not resurrect failed MCE,
generic IQA, arbitrary-predecessor consistency, or direct MemCam softmax
dilution. In paper language call slam_covisibility "Geometric Coverage"; it is
our implementation, not a literal published SLAM formula.
```

## Collaboration contract

These rules matter as much as the code.

1. Work locally first. The normal flow is: inspect and edit on the Mac, run
   local tests, explain the diff, then Abdul handles Git commit/push as desired.
   Never push. Do not commit unless explicitly asked.
2. Abdul pulls the repository on the execution machine with `git pull`, then
   submits or runs the command. The assistant should provide commands but must
   not assume it can access Newton or CECSL directly.
3. Never submit an extra job while Abdul is inside an interactive allocation.
   Interactive analysis means run the requested analysis in that shell.
4. Default to one ordinary `.sbatch` file that performs its stages
   sequentially. Job arrays and dependency chains have repeatedly wasted time
   through partial failures and `DependencyNeverSatisfied` states.
5. Do not use `column`, `less`, or commands that trap the user in a pager.
   Prefer `cat`, `sed -n`, `tail`, `find`, and small Python printouts.
6. Existing generated videos are expensive and immutable. Generation tools are
   resume-safe and skip existing outputs. Never add `--overwrite` without an
   explicit reason and approval.
7. Before expensive work, report the number of trajectories, duration, policy
   points, and a rough runtime. One 180-second CECSL video has taken about
   11,987 seconds, so "three repeats" is not a small request.
8. Explain ideas a few sentences at a time. Use concrete terms such as "current
   target camera pose" and "stored frame," not undefined `q`, `m`, "oracle," or
   dense terminology.
9. Be openly uncertain. Do not declare a method principled, novel, ICLR-worthy,
   or superior before its matched experiment finishes.
10. Do not smooth noisy curves to conceal variation. Plot raw measurements;
    uncertainty bands or a clearly labeled secondary trend are acceptable.
11. In figures and paper prose call the `baseline` run **Unbounded**. Call
    `slam_covisibility` **Geometric Coverage**, not "SLAM-inspired," unless
    discussing implementation history.
12. Do not run `module purge` as a reflex. The established module workflow has
    worked before, and Abdul explicitly does not want destructive environment
    resets suggested without evidence.

## Repository and machine map

### Local Mac

```text
Repository: /Users/metalicana/projects_summer_2026/MemCam
Remote:     https://github.com/Metalicana/MemCam.git
Branch:     main
HEAD when this document was written: b8c1475db6dc44e497f8d06bc1f674cac3e9422f
```

The worktree was clean before this handoff document was added. Related local
repositories are siblings:

```text
/Users/metalicana/projects_summer_2026/WorldMem
/Users/metalicana/projects_summer_2026/vmem
/Users/metalicana/projects_summer_2026/spmem
/Users/metalicana/projects_summer_2026/VBench
```

### Newton

```text
Login:        ab575577@newton.ist.ucf.edu
Repository:   /home/ab575577/MemCam, normally written as ~/MemCam
Environment:  conda env memcam
Results root: /home/ab575577/memcam_results
HF cache:     /home/ab575577/hf_cache
Logs:         ~/MemCam/logs
```

Main Newton paths:

```text
60s videos:       ~/memcam_results/context_memory_60s
180s videos:      ~/memcam_results/context_180s
60s LPIPS/FVD:    ~/memcam_results/eval_prefix_duration_curves_60s_b32
VBench:           ~/memcam_results/vbench_results
VBench-Long:      ~/memcam_results/vbench_long_results
60s manifest:     ~/MemCam/testbeds/context_memory/manifest.jsonl
180s manifest:    ~/MemCam/testbeds/context_memory_180s/manifest.jsonl
CUT3R checkpoint: ~/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth
```

Known Newton nodes that have produced CUDA initialization failures are
`evc23`, `evc33`, `evc40`, `evc42`, `evc44`, and `evc46`. Job `770599` timed
out during the 90-second CUDA preflight on `evc46` on 2026-08-22. Critical H100
jobs should exclude these nodes. This list is historical rather than a
guarantee that every other node is healthy.

The normal job preamble is:

```bash
module load cuda
module load ffmpeg
module load anaconda
conda activate memcam
```

`ffmpeg` may reload CUDA from 13.x to 12.1. That message alone is not a
failure. Every GPU job should run a bounded CUDA preflight before model load:

```bash
timeout 90s python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

If a job disappears from `squeue`, inspect it without guessing:

```bash
sacct -j JOB_ID -X --format=JobID,State,ExitCode,Elapsed,NodeList,Reason
tail -n 120 ~/MemCam/logs/FILE_FOR_JOB.err
tail -n 120 ~/MemCam/logs/FILE_FOR_JOB.out
```

### CECSL workstation

```text
Login:              ab575577@CECSL4622128797
Repository:         /home/ab575577/MemCam, normally ~/MemCam
Conda environment:  memcam
Canonical data:     /data/ab575577/MemCam/outputs
60s outputs:        /data/ab575577/MemCam/outputs/context_memory_60s
180s outputs:       /data/ab575577/MemCam/outputs/context_180s
Dataset root:       /data/ab575577/Context-as-Memory-Dataset/Context-as-Memory-Dataset
```

CECSL is a direct workstation, not Slurm. Long commands must run in `tmux` and
should also log to a file because a vanished session otherwise hides the cause:

```bash
ssh ab575577@CECSL4622128797
tmux new -s memcam
cd ~/MemCam
conda activate memcam
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t memcam`.

CECSL holds the consolidated copy of many Newton outputs. It is the preferred
place for archiving, CPU analysis, plotting, and filling missing videos when
Newton is blocked. A long FIFO recovery process previously exited with code
137 near video completion, so log memory-intensive runs and check the output
count rather than assuming `tmux` disappearance means no progress.

### Anvil and Jetstream2

They were planned but not fully configured. Do not invent credentials or
claim they are active. The placeholders and setup procedure are in
[`cluster_runbook.md`](cluster_runbook.md).

Anvil is Slurm with project and scratch storage. Jetstream2 is an OpenStack VM
and should use a persistent attached volume plus `tmux`. Final FVD inputs must
be merged into one canonical archive before evaluation. Latency comparisons
must use the same GPU model and software stack.

## Git and execution workflow

The assistant works only in the local repository. After a change:

1. Run focused tests and `git diff --check` locally.
2. Show Abdul the changed files and verification result.
3. Do not push.
4. Abdul pushes from the Mac.
5. Abdul runs `git pull` inside `~/MemCam` on Newton or CECSL.
6. Abdul submits the provided job or executes the command.

Use the full Newton host in transfer commands:

```bash
scp ab575577@newton.ist.ucf.edu:REMOTE_PATH LOCAL_PATH
```

Do not shorten it to an unexplained `newton` hostname in commands intended for
copying files.

## Research problem

The project studies **bounded external memory for long-horizon autoregressive
video generation**.

At chunk `t`, the model has a full bank `M_(t-1)` and a set of newly generated
candidates `N_t`. It must causally select the next bank:

```text
M_t is a subset of M_(t-1) union N_t, with |M_t| <= B.
```

The future trajectory after the current generation chunk is unknown. The
current target camera trajectory used for retrieval is known. Do not propose a
policy that assumes all future poses are available.

The broad motivation is that external-memory generators often keep every
historical item. This produces linear storage growth and superlinear cumulative
exhaustive retrieval cost, while the generator still consumes a fixed-size
retrieved context. The surprising empirical result is that a carefully bounded
bank can also generate better video than the complete archive.

The desired long-term contribution is memory curation that can transfer across
representations:

| System | Stored representation | Current status |
| --- | --- | --- |
| MemCam | decoded RGB frames plus camera data | Main implemented testbed |
| WorldMem | learned/latent memory | Separate sibling repo; maximum tested horizon is 60s |
| VMem | surfel-style memory | Future transfer target |
| SpMem | point-cloud memory | Future transfer target |

The general budget may eventually need to be bytes or storage cost rather than
item count because 32 images, 32 latent entries, and 32 surfels are not
comparable. Current MemCam experiments use frame count, mainly `B=32`.

## MemCam mechanics

MemCam is a Wan2.1 1.3B camera-controlled generator. It generates video in
sections, stores generated RGB frames, and retrieves one historical context
frame per target frame using camera field-of-view overlap. The unbounded
archive grows, but the number of selected conditioning slots remains fixed.

This distinction has an important consequence: archive growth does **not**
directly enlarge the denoiser's attention softmax in MemCam. More candidates
can poison hard retrieval or expose damaged memories, but "softmax dilution
from archive token growth" is architecturally false for this system.

Important terminology:

- **Target/query:** a camera pose in the chunk currently being generated.
- **Memory item:** an older generated RGB frame with its corresponding pose.
- **Unbounded:** code/output folder `baseline`; retains all historical frames.
- **Frozen retriever:** MemCam's original FOV-overlap selector, unchanged across
  bank policies.
- **Policy:** the rule deciding which historical frames remain candidates.
- **Ground truth:** the Context-as-Memory dataset frame at an exact trajectory
  index. It can evaluate a generated frame, but it is unavailable to the online
  policy.
- **Hindsight-best:** an offline diagnostic proxy, not a true useful-frame
  oracle.

## Benchmark layout

The main benchmark uses 15 deterministic Context-as-Memory trajectories.

The 60-second manifest is scene-major and contains several durations. Its 15
60-second rows are:

```text
3,8,13,18,23,28,33,38,43,48,53,58,63,68,73
```

Do not use rows `0-14` for the 60-second subset. The dedicated 180-second
manifest has 15 lines, so its rows are `0-14`.

The scenes are:

```text
AncientTempleEnv_5
AnimeCitySuburbs_5
ChemicalPlantEnv_5
ClothingStore_1
ClothingStore_6
ContainerYard_3
ContainerYard_7
ContainerYard_9
DragonRise_1
DragonRise_9
FeudalJapan_0
FeudalJapan_2
FeudalJapan_3
IslandMap_2
Warehouse_0
```

Manifests contain absolute paths. Regenerate them per machine from the same
deterministic split and confirm that `output_prefix` values match before
merging results.

Generated videos use names ending in `custom.mp4`. Each run contains
`run_status.jsonl`, `access_traces/`, and often `profiles/`. The batch runner
skips a video whose expected output already exists.

## Current policies and exact meaning

The implementations are in
[`diffsynth/pipelines/memory_policies.py`](../diffsynth/pipelines/memory_policies.py)
and are wired through
[`wan_video_memcam.py`](../diffsynth/pipelines/wan_video_memcam.py) and
[`run_context_memory_batch.py`](../utils/run_context_memory_batch.py).

### FIFO

Keeps the most recent `B` frames. It is a capacity baseline, not a smart
curation method. It often improves FVD relative to Unbounded but badly worsens
LPIPS, showing that merely reducing capacity is insufficient.

### Rarity-Irreplaceability, RI

Code name: `rarity_irreplaceability`.

DINO features are clustered using connected components at the median
`k`-th-nearest-neighbor distance. For frame `i`:

```text
rarity_i = log((number_of_candidates + 1) / cluster_size_i)
irreplaceability_i = RGB distance to its nearest other candidate
RI_i = rarity_i * irreplaceability_i
```

Higher score means keep. The `rarity_neighbors` parameter was previously dead
because the threshold always used the first neighbor. That bug is fixed.
Historical `k=1` results preserve the old behavior; the current corrected
default is `k=3`.

The `k=3` result is mixed: it improved FVD over `k=1` at 10 and 20 seconds,
but was worse at 60 seconds. Do not state that `k=3` is globally better.

### Geometric Coverage

Code name and output naming: `slam_covisibility` and
`slam_b32_covisibility`. Paper name: **Geometric Coverage**.

This is our hand-designed score, not a literal SLAM paper formula. Pairwise
affinity is:

```text
K(i,j) = 0.65 * exp(-pose_distance(i,j))
       + 0.35 * max(DINO_cosine(i,j), 0)
```

Pose distance is median-normalized translation plus twice the normalized
relative rotation angle. A frame is less valuable when several other frames
have affinity at least 0.65 to it. The exact keep score combines low
redundancy, inverse observer count, and uniqueness; the buffer evicts the
lowest score.

Geometric Coverage is currently the strongest reliable policy. Do not weaken
the story by calling it a standard SLAM method, and do not pretend it already
solves a formal set objective.

### Geometric Coverage-RI blend

Code name: `slam_ri_blend`.

It calls the real RI and Geometric Coverage functions, min-max normalizes both
scores over the current candidate pool, and computes:

```text
blend_i = beta * normalized_Geo_i + (1 - beta) * normalized_RI_i
```

`beta=1` exactly reproduces Geometric Coverage's ranking; `beta=0` exactly
reproduces RI. A complete `beta=0.5` run did not beat both constituents and
lost to both on most VBench dimensions. A geometry-dominant `beta=0.75` is a
hypothesis, not a final method. It needs a sensitivity sweep and must beat pure
Geometric Coverage.

### Other implemented probes

| Policy | Status |
| --- | --- |
| `kcenter_coreset` | Useful baseline; strong but below Geometric Coverage in the main 180s results. |
| `trajectory_coverage` | Completed at B32/60s; worse than Geometric Coverage. |
| `density_balanced_view_coverage` | Beat Geometric Coverage only at 10s and lost clearly at 20/30/60s. |
| `slam_max_coverage` | Implements greedy top-1 facility coverage; completion/result not established in the shared record. |
| `facility_coreset` | Implemented exploratory coreset; not part of the current claim. |
| `future_view_coverage` | Pilot only; future poses beyond the current chunk may not be assumed. |
| `surprise_forcing` | Implemented baseline; mixed LPIPS/FVD result. |
| `mce` | Failed method direction. Parked. Do not resurrect it as the final policy. |
| `reliable_slam_ri` | Handcrafted predecessor/reference gate. Rejected as circular and unvalidated. |
| `h2o_heavy_hitter` | Baseline adapter, not a current method direction. |

## Canonical quality result

Use matched configurations only. Several old directories contain incomplete
runs or different FVD clip settings; those numbers are not directly
comparable. The current canonical 180-second, budget-32 table is:

| Policy | Stored frames | LPIPS lower is better | FVD lower is better |
| --- | ---: | ---: | ---: |
| Unbounded (`baseline`) | 5,397 | 0.5980 | 734.2 |
| FIFO-32 | 32 | 0.6514 | 677.3 |
| RI-32 | 32 | 0.5939 | 550.4 |
| Geometric Coverage-32 | 32 | 0.5876 | 476.6 |

The central empirical result is real: RI and especially Geometric Coverage
beat Unbounded while storing under one percent of its frames. FIFO proves the
gain is not caused by capacity reduction alone.

Do not compare the earlier partial baseline FVD around 797 against complete
15-video policy FVD values. That run used only 8/15 videos and a different FVD
sampling configuration.

## Mechanism evidence

### Retention and retrieval decomposition

At 180 seconds:

| Policy | Retention gap | Retrieval gap |
| --- | ---: | ---: |
| Unbounded | approximately 0 by construction | 0.2267 |
| FIFO | 0.2039 | unavailable in the saved summary |
| RI | 0.0474 | 0.1582 |
| Geometric Coverage | 0.0639 | 0.1338 |

RI mainly improves retention. Geometric Coverage more consistently improves
selection among the retained candidates. These gaps use a hindsight DINO proxy
and are not ground-truth utility.

### Pool-growth screen

On 4,200 unbounded queries from 15 trajectories:

- Spearman candidate count versus retrieval gap: `0.2029`.
- Positive within-trajectory trend on 13/15 trajectories, sign-test
  `p=0.00739`.
- Late-minus-early retrieval gap: `+0.07234`, 95% CI
  `[0.03960, 0.10882]`.
- Hindsight-best mismatch stayed approximately flat.

Pool size and elapsed autoregressive time grow together, so this is not causal
proof that candidate count alone creates the effect.

### View versus corruption split

As the archive grows:

| Diagnostic | Late minus early | Interpretation |
| --- | ---: | --- |
| Selected view mismatch | -0.03351 | Selected poses/views become slightly better aligned. |
| Selected memory corruption | +0.08735 | The generated content at selected indices becomes substantially worse. |
| Selected effective mismatch | +0.07316 | Net conditioning becomes worse. |

This is the clearest evidence for selection increasingly exposing the model to
corrupted historical content. It is observational.

### Selected image quality

Selected generated frames compared with exact-index dataset GT:

| Policy | Selected PSNR | Selected SSIM | Late selected PSNR | Late selected SSIM |
| --- | ---: | ---: | ---: | ---: |
| Unbounded | 11.703 | 0.3089 | 11.433 | 0.3146 |
| FIFO-32 | 10.619 | 0.2617 | 10.135 | 0.2546 |
| RI-32 | 13.396 | 0.3830 | 12.896 | 0.3723 |
| Geometric Coverage-32 | 16.522 | 0.4759 | 15.119 | 0.4386 |

### Common-source selection control

Every policy supplied only frame indices; all selected image content was read
from the same Unbounded rollout. Relative to Unbounded:

- RI selected indices were `+1.775 dB` PSNR and `+0.0672` SSIM cleaner.
- Geometric Coverage selected indices were `+4.629 dB` PSNR and `+0.1512`
  SSIM cleaner.
- Geometric Coverage won PSNR and SSIM on all 15 trajectories.

This shows that the selection rule itself favors cleaner indices. It does not
yet prove that replacing a corrupted conditioning image causes the next chunk
to improve.

### Causal experiments

The context-identity swap has only one valid matched case. It slightly improved
LPIPS by `-0.00177` and DINO distance by `-0.01691`; `n=1` is underpowered.

The decisive GT-content cleaning replay holds selected frame identities,
history, model, and noise fixed, then replaces only the selected generated
memory images with exact-index GT for one section. Four high-corruption cases
were planned. Some branches completed, others failed CUDA preflight. No final
multi-case report has been shared, so causality remains open.

The preferred implementation is one sequential job:

```text
slurm/newton_memory_cleaning_replay_180s_single.sbatch
```

Do not use the older array plus dependent evaluation unless explicitly asked.

## Failed and negative investigations

Do not silently rerun or rebuild these as if they were new ideas.

1. **Unbounded as oracle:** wrong. Unbounded's chosen frame is not a useful-frame
   ground-truth label.
2. **Occlusion poisoning:** real but too small to explain the main effect.
3. **Within-section context collapse:** exists, but becomes less severe as the
   pool grows.
4. **Aggregate memory drift:** average stored-frame corruption did not worsen
   monotonically. The query-conditioned selected subset did worsen; these are
   different questions.
5. **Zero-overlap fallback:** relevant in WorldMem, not MemCam. MemCam baseline
   had essentially 100% overlap hits and median overlap around 0.9335.
6. **Monte Carlo overlap noise:** winner flips are common, but become less
   frequent with larger pools and have negligible visual cost.
7. **Density-balanced coverage:** worse than Geometric Coverage after 10s.
8. **MCE:** failed and removed from the final direction.
9. **Generic no-reference IQA:** failed deployment calibration.
10. **Arbitrary previous-frame consistency:** circular because a corrupted
    predecessor can validate continued corruption.
11. **Direct softmax dilution in MemCam:** architecture does not support this
    mechanism because denoising receives fixed-size retrieved context.

Detailed numbers for items 2 through 6 are in
[`unbounded_degradation_mechanism_hunt.md`](unbounded_degradation_mechanism_hunt.md).

## Quality-gate work

### Generic IQA result

The calibration sampled 2,160 frames from Unbounded and Geometric Coverage,
with 10 train trajectories and 5 held-out trajectories. A bad frame was the
bottom 20% by within-run/trajectory PSNR+SSIM rank.

The best score, `unclipped_fraction`, achieved:

```text
held-out AUC:                 0.630
balanced accuracy:           0.550
deployable bad recall:       0.183
clean false-rejection rate:  0.125
```

At 20% prevalence, roughly 37 bad frames and 100 clean frames would be rejected
per 1,000 observations. MUSIQ, CLIP-IQA+, TOPIQ-NR, sharpness, contrast,
entropy, gradient energy, and Laplacian variance were near chance or
misaligned. Decision: do not inject generic IQA.

### Pose-calibrated conditioning consistency

This validation-only hypothesis was tested and rejected. It compared a newly
generated frame to the actual retrieved context that conditioned it, then
subtracted expected DINO similarity for that camera displacement. Expected
similarity was calibrated from GT frame pairs on training trajectories.

Code and job:

```text
utils/calibrate_causal_consistency_gate.py
slurm/newton_calibrate_causal_consistency_gate_180s.sbatch
```

The predeclared decision requires all of:

```text
AUC >= 0.70
bad precision >= 0.50
bad recall >= 0.20
held-out clean rejection <= 0.15
pose calibration AUC gain >= 0.02 over raw similarity
low-fidelity-parent AUC >= 0.60
```

The completed validator analyzed 2,100 pairs and returned `DO_NOT_INJECT`:

```text
context_pose_residual AUC:       0.511
bad precision:                   0.207
bad recall:                      0.119
clean false-rejection rate:      0.117
within-trajectory Spearman:     -0.020
raw context similarity AUC:      0.546
low-fidelity-parent AUC:         0.539
```

Only clean rejection passed. Pose calibration worsened raw similarity, and the
gate was near random at identifying corrupted frames. Do not inject, port, or
resurrect this mechanism without a materially different observable signal.

## Other completed probes

### Attention intervention

Ten 30-second videos produced 270 interventions and 90 probe groups.

- Total attention versus intervention effect: global Spearman `0.6787`, mean
  within-probe `0.9056`.
- Slot count performed similarly or slightly better: `0.7275` and `0.9072`.
- Attention per slot was weak: `-0.0321` global and `0.2222` within-probe.
- High-attention memory caused the larger immediate effect in 98.89% of pairs.

Interpretation: total attention mostly tracks how often a frame occupies a
retrieval slot. It demonstrates immediate denoiser sensitivity among already
selected memories, not future utility and not a validated eviction score.

### Surprise Forcing baseline

On the matched 30-second pilot:

| Prefix | Surprise LPIPS | Unbounded LPIPS | Surprise FVD | Unbounded FVD |
| --- | ---: | ---: | ---: | ---: |
| 10s | 0.44797 | 0.44468 | 592.55 | 570.97 |
| 20s | 0.50174 | 0.49027 | 602.97 | 677.80 |
| 30s | 0.53083 | 0.51563 | 755.46 | 880.58 |

It improves later FVD but consistently loses LPIPS. Treat it as a mixed
baseline, not a victory.

### RI k=3

Compared with historical RI k=1 FVD:

```text
10s: k3 better by 6.31
20s: k3 better by 36.88
60s: k3 worse by 45.78
```

### 50/50 Geometric Coverage-RI blend

All 15 videos, LPIPS/FVD, and standard VBench completed. It did not beat both
constituents; it lost to both on four of six VBench dimensions. VBench-Long
crashed and CUT3R did not run in the original chained job. A CUT3R-only resume
job exists, but completion was never confirmed.

## Metrics and comparison rules

- **LPIPS Alex:** lower is better; exact-index frame metric.
- **FVD:** lower is better; distribution-level I3D metric. Compare only runs
  with the same videos, clip count, detector, frame stride, and image size.
- **PSNR/SSIM:** higher is better; useful here because dataset GT supplies the
  exact camera trajectory and frame index. They still penalize plausible
  appearance differences.
- **DINO distance:** a diagnostic representation distance, not ground-truth
  utility.
- **VBench:** higher is better. The project uses subject consistency,
  background consistency, motion smoothness, dynamic degree, aesthetic
  quality, and imaging quality.
- **CUT3R:** estimates generated camera trajectory; use rotation, translation,
  and camera-control summaries. Setup is documented in
  [`cut3r_worldscore_metrics.md`](cut3r_worldscore_metrics.md).
- **VBench-Long:** currently broken in the blend pipeline and not required for
  interpreting the completed standard VBench result.
- **Revisit metrics:** parked. There is no unique useful-frame GT. Existing
  threshold/oracle directories are calibration experiments, not a solved
  metric.

Do not average or compare summaries from incomplete policy sets without
explicitly matching the evaluated trajectories. Never hide incomplete data in
scientific analysis, even if a slide temporarily omits completion annotations.

## Latency and memory profiling

MemCam normally stores the memory bank on CPU. A GPU-resident bank can trade
higher peak VRAM for lower transfer/retrieval latency, but the Pareto study was
deprioritized.

Last known CECSL profile root:

```text
/data/ab575577/MemCam/outputs/context_180s/latency_vram_pareto_60s
```

At the last check, 18/26 profiles were complete. The plotter is
`utils/plot_latency_vram_pareto.py`; `--require_full_grid` correctly fails when
points are missing. Verify the directory before claiming a final Pareto curve.

Timing curves must not mix Newton H100, CECSL GPU, Anvil A100, and Jetstream
virtual-GPU measurements. Storage scaling may be analytic, but label
extrapolation assumptions clearly.

## Current experiment state to verify after account migration

Cluster state is not visible from the local repo. A new session must ask Abdul
to run the following before assuming anything is pending or complete:

```bash
squeue --me
```

Then inspect output paths, not just scheduler state.

Known unresolved items:

1. GT-content cleaning replay: partial branches existed; final evaluation not
   confirmed.
2. Pose-calibrated consistency validator: completed; `DO_NOT_INJECT`.
3. `beta=0.75`, k=3 blend: completed 15/15; prefix LPIPS/FVD recorded in this
   document, but VBench and CUT3R remain to be confirmed.
4. `beta=0.25`, k=3 blend: completed 15/15; prefix LPIPS/FVD recorded.
5. `slam_max_coverage`: job was submitted historically; result not recorded.
6. Blend CUT3R-only resume: job exists; result not confirmed.
7. VBench-Long: crashed and was never diagnosed.
8. Latency/VRAM grid: last known 18/26 complete.

Do not rerun any of these before checking existing video, summary, and report
files.

## Paper state

Working title:

> The Archive Is Not the Context: Diagnosing and Curating Memory for
> Long-Horizon Video Generation

The manuscript is in [`paper/main.tex`](../paper/main.tex). It is an
aspirational end-state draft, not submission-ready evidence. It still contains
stale claims that a generic no-reference quality gate and fixed 75/25 blend
win. The warning in [`paper/README.md`](../paper/README.md) is authoritative.

The strongest defensible story today is:

1. Long-horizon external-memory systems often grow storage and retrieval cost
   with duration while consuming only a fixed retrieved context.
2. In MemCam, bounded smart curation can beat the complete archive.
3. The selected subset, rather than average stored history, becomes increasingly
   corrupted relative to exact-index GT.
4. RI and especially Geometric Coverage select much cleaner historical indices
   from a common rollout.
5. The causal propagation link and the final policy beyond Geometric Coverage
   remain open.

Unsupported paper claims include:

- pool size alone causally causes degradation;
- corrupted memories have already been causally proven to damage later chunks;
- generic IQA is a valid gate;
- 75/25 is principled or universally transferable;
- direct archive-induced softmax dilution occurs in MemCam;
- the method is representation agnostic before a second system succeeds;
- any experiment guarantees ICLR acceptance.

The reviewer-risk source of truth is
[`iclr_reviewer_risk_register.md`](iclr_reviewer_risk_register.md).

## Immediate decision tree

Do not expand the project until these decisions are resolved.

1. Complete the GT-content cleaning replay.
2. Run the consistency-gate validator and obey `INJECT` or `DO_NOT_INJECT`.
3. Verify the geometry-dominant blend and compare directly with pure Geometric
   Coverage.
4. If the gate and blend fail, stop method invention. Write the strongest
   honest diagnostic paper around unbounded-memory failure and bounded
   Geometric Coverage, then reassess venue.
5. If a final method wins, lock it before VBench, CUT3R, latency, memory, and
   cross-system transfer evaluation.
6. Audit WorldMem's actual memory-to-generator path before importing MemCam
   mechanisms or metrics. WorldMem can run only to 60 seconds in the current
   setup.

## Code map

| Purpose | Main files |
| --- | --- |
| Generation pipeline | `inference_memcam.py`, `diffsynth/pipelines/wan_video_memcam.py` |
| Memory policies | `diffsynth/pipelines/memory_policies.py` |
| Batch generation | `utils/run_context_memory_batch.py` |
| Prefix LPIPS/FVD | `utils/evaluate_context_memory_prefix_curves.py` |
| Policy tests | `utils/check_memory_policies.py`, `tests/test_slam_ri_blend.py`, related tests in `tests/` |
| Diagnostic decomposition | `utils/analyze_retrieval_quality_decomposition.py` |
| Pool-growth screen | `utils/analyze_pool_growth_scaling.py` |
| Selected-memory quality | `utils/analyze_selected_memory_image_quality.py` |
| Common-source selection | `utils/analyze_common_source_selection_quality.py` |
| Generic IQA calibration | `utils/calibrate_frame_quality_estimators.py` |
| Consistency-gate validator | `utils/calibrate_causal_consistency_gate.py` |
| Cleaning replay | `utils/build_memory_cleaning_replay_plan.py`, `utils/run_memory_cleaning_replay_case.py`, `utils/evaluate_memory_cleaning_replays.py` |
| Attention pilot | `slurm/newton_memcam_h100_30s_attention_audit.sbatch`, `utils/summarize_attention_utility_pilot.py` |
| Latency/VRAM | `utils/plot_latency_vram_pareto.py`, profiling support in generation pipeline |
| CUT3R | `utils/run_cut3r_context_memory.py`, `utils/evaluate_cut3r_camera_metrics.py` |
| Paper | `paper/main.tex`, `paper/make_figures.py`, `paper/README.md` |

Note: the actual decomposition filename in this repository is
`utils/analyze_retrieval_quality_decomposition.py`; do not confuse it with
older report names.

## First checks in a new session

Run locally:

```bash
cd /Users/metalicana/projects_summer_2026/MemCam
git status --short
git log -5 --oneline
python -m unittest \
  tests.test_slam_ri_blend \
  tests.test_causal_consistency_gate \
  tests.test_memory_cleaning_replay \
  -v
git diff --check
```

On Newton, after Abdul has pushed:

```bash
ssh ab575577@newton.ist.ucf.edu
cd ~/MemCam
git pull
git rev-parse HEAD
squeue --me
```

On CECSL:

```bash
ssh ab575577@CECSL4622128797
cd ~/MemCam
git pull
git rev-parse HEAD
find /data/ab575577/MemCam/outputs/context_180s -maxdepth 2 -type f \
  -name 'summary.json' -print | sort
```

The first response from a new GPT should summarize what it learned and ask for
the current scheduler/output status only if the next task depends on it. It
should not immediately propose a new policy.
