# MemCam Research Brief for Method Brainstorming

Updated: 2026-08-26

This document is a self-contained account of the project, the implemented
methods, the evidence, the failures, and the current decision point. It is
written so that another research collaborator can critique the work without
reconstructing months of experiments from chat logs.

## 1. Research problem

Long-horizon autoregressive video generators must preserve objects, scene
layout, and identity when the camera leaves a region and later returns. Several
systems address this by storing generated observations in an external memory
bank and retrieving a small context set for each new video chunk.

MemCam stores generated RGB frames with camera poses. For each target frame in
the next chunk, its native retriever searches the bank and selects one context
frame using estimated camera field-of-view overlap. The retrieved context size
is fixed; only the candidate archive grows.

With `N` generated frames:

- Unbounded archive storage is `O(N)`.
- One exhaustive retrieval round is `O(N)` for a fixed-size target chunk.
- Cumulative exhaustive retrieval over the complete rollout is `O(N^2)`.
- A fixed bank of `B` items has `O(B)` storage and `O(NB)` cumulative
  retrieval, which is linear in duration when `B` is constant.

The empirical surprise is more important than the complexity result: a
carefully curated 32-frame bank can produce better video than keeping all
5,397 historical frames at 180 seconds.

The project therefore asks:

> Which generated observations should an online long-video generator retain
> under a fixed memory budget when the future trajectory is unknown?

## 2. Architectural facts and boundaries

- A query `q` is a target camera pose in the chunk currently being generated.
- A memory item `m` is a previously generated RGB frame and its camera pose.
- The bank policy decides which historical frames remain candidates.
- The native MemCam retriever is frozen across policies.
- The complete archive does not directly enter the denoiser. It is first
  reduced to a fixed-size retrieved context.
- Therefore direct attention-softmax dilution from archive growth is not a
  valid MemCam mechanism. Candidate competition and exposure to corrupted
  selected memories remain valid possibilities.
- Dataset ground truth exists at every exact camera-trajectory index. It is
  available for offline evaluation only and cannot be used by an online
  policy.
- Future camera trajectory beyond the currently known control chunk is not
  assumed.

The broader ambition is to transfer memory curation across representations:

| System | Representation | Current status |
| --- | --- | --- |
| MemCam | RGB frames plus camera data | Main implemented testbed |
| WorldMem | Learned/latent memory | Separate repository; 60s maximum |
| VMem | Surfels | Future transfer target |
| SpMem | Point clouds | Future transfer target |

Representation-agnostic behavior is an ambition, not an established result.
Different representations may ultimately require a byte or compute budget
rather than a common item count.

## 3. Primary benchmark

- Dataset: Context-as-Memory trajectories.
- Main set: 15 deterministic scenes/trajectories.
- Primary budget: `B=32` stored frames.
- Main long-horizon result: 180 seconds.
- Policy-development runs: 60 seconds with prefix evaluation at 10, 20, 30,
  and 60 seconds.
- Main metrics: LPIPS, FVD, six VBench dimensions, CUT3R camera consistency,
  latency, and peak memory.

Important comparison rule: FVD values are comparable only when the evaluated
videos, clip count, detector, stride, and image size match. Several old runs
used incomplete video sets or different clip sampling.

## 4. Methods implemented and tested

All main policy implementations are in
`diffsynth/pipelines/memory_policies.py`. They are wired into generation by
`diffsynth/pipelines/wan_video_memcam.py` and exposed through
`utils/run_context_memory_batch.py`.

### 4.1 Unbounded memory

Code/output name: `baseline`.

Every generated frame remains in the archive. This is the original MemCam
behavior and the quality, storage, and retrieval-cost reference.

It has zero retention loss by construction, but this does not make its selected
frame an oracle. The frozen retriever can still choose a poor candidate from
the complete history.

### 4.2 FIFO

Code name: `fifo`.

FIFO retains the newest `B` frames and evicts the oldest. It is the capacity
control baseline.

FIFO demonstrates that limiting memory alone is insufficient. At 180 seconds,
FIFO-32 improves FVD over Unbounded but greatly worsens LPIPS because it forgets
old revisitable content.

### 4.3 Rarity-Irreplaceability (RI)

Code name: `rarity_irreplaceability`.

RI gives each candidate an individual keep score. DINO features are clustered
with connected components using a threshold based on the median distance to
the `k`-th nearest neighbor. For candidate `i`:

```math
R_i = \log\frac{|P|+1}{|C(i)|},
\qquad
I_i = \min_{j\ne i} d_{\mathrm{RGB}}(i,j),
\qquad
u_i^{\mathrm{RI}} = R_i I_i.
```

`P` is the current candidate pool and `C(i)` is the DINO cluster containing
`i`. Rarity rewards small appearance clusters. Irreplaceability rewards an
image whose nearest RGB substitute is still far away. The lowest scores are
evicted.

The historical implementation accidentally ignored `k` and behaved as `k=1`.
That bug is fixed. Current `k=3` is scientifically cleaner, but its result is
mixed rather than globally superior.

### 4.4 Geometric Coverage

Code name: `slam_covisibility`; historical folders use names such as
`slam_b32_covisibility`.

Paper-facing name: **Geometric Coverage**. This is our hand-designed
SLAM-motivated policy, not a copied SLAM keyframe formula.

For memories `i` and `j`, the pairwise affinity is

```math
K(i,j)=0.65\exp[-d_{\mathrm{pose}}(i,j)]
       +0.35\max(\cos(z_i,z_j),0),
```

where `z` is a DINO embedding. Pose distance combines normalized translation
and relative rotation. If DINO is unavailable, the implementation can use an
RGB similarity component instead.

For each candidate, the code counts how many other memories have
`K(i,j) >= 0.65`, then computes

```math
u_i^{\mathrm{Geo}} =
  (1-\rho_i)
  +\frac{0.5}{n_i+1}
  +0.25(1-\max_{j\ne i}K(i,j)),
```

where `n_i` is the number of covisible observers and
`rho_i=min(n_i/3,1)`. Frames already represented by several similar memories
receive low keep scores.

This remains the strongest reliable single policy.

### 4.5 K-center coreset

Code name: `kcenter_coreset`.

This baseline greedily chooses representatives that minimize the maximum
distance from historical archive items to their nearest retained center. Its
distance combines visual, pose, and optionally temporal terms. It is a standard
coverage/coreset baseline and performed strongly at 180 seconds, but below
Geometric Coverage.

### 4.6 Facility and set-coverage variants

Several policies attempted to replace per-frame utility with set utility.

`slam_max_coverage` greedily maximizes the facility objective

```math
F(M)=\frac{1}{|Q|}\sum_{q\in Q}\max_{m\in M}K(q,m),
```

using the same Geometric Coverage affinity. `facility_coreset` uses historical
archive descriptors as demand points. These implementations are useful
scientific baselines, but no shared complete result establishes either as the
winning policy.

### 4.7 Marginal Coverage Eviction (MCE)

Code name: `mce` / `compute_marginal_coverage_eviction_scores`.

MCE constructed historical DINO-cluster medoids as demand queries, optionally
included known control poses, used a geometric/visual kernel, and repeatedly
deleted the item with the smallest exact marginal loss until the bank met the
budget.

This was the most explicit attempt at a submodular set objective. It failed
empirically and is closed as the final direction. It must not be quietly
resurrected merely because the equation is elegant.

### 4.8 Trajectory Coverage

Code name: `trajectory_coverage`.

This policy attempted causal view coverage using camera trajectory geometry.
The complete B32/60s run was worse than Geometric Coverage. It showed that
geometry alone and a more formal coverage objective do not automatically
improve generation.

### 4.9 Density-Balanced View Coverage (DBVC)

Code name: `density_balanced_view_coverage`.

DBVC combined coverage with inverse-density weighting to protect sparse view
regions. At B32 it slightly beat Geometric Coverage at 10 seconds, then lost on
LPIPS and FVD at 20, 30, and 60 seconds. At 60 seconds:

```text
DBVC: LPIPS 0.594133, FVD 750.299
Geo:  LPIPS 0.584950, FVD 690.532
```

It is a completed negative result.

### 4.10 Surprise Forcing baseline

Code name: `surprise_forcing`.

This streaming baseline estimates descriptor surprise relative to the bank and
combines write surprise, usage, and age. In the matched 30-second pilot it
improved later FVD but consistently lost LPIPS:

| Prefix | Surprise LPIPS | Unbounded LPIPS | Surprise FVD | Unbounded FVD |
| --- | ---: | ---: | ---: | ---: |
| 10s | 0.44797 | 0.44468 | 592.55 | 570.97 |
| 20s | 0.50174 | 0.49027 | 602.97 | 677.80 |
| 30s | 0.53083 | 0.51563 | 755.46 | 880.58 |

It is useful as an external streaming-memory baseline, not the final method.

### 4.11 Attention-based utility pilot

This pilot measured the denoiser's sensitivity to memories that had already
been selected by the geometric retriever.

- 10 complete 30-second videos.
- 270 memory interventions and 90 probe groups.
- Total attention versus intervention effect: global Spearman `0.6787`, mean
  within-probe Spearman `0.9056`.
- Slot count was equally or slightly more predictive: `0.7275` global and
  `0.9072` within-probe.
- Attention per slot was weak: `-0.0321` global and `0.2222` within-probe.
- High-attention memory caused the larger immediate effect in 98.89% of pairs.

Interpretation: total attention mainly measures how often a frame occupies a
retrieval slot. It establishes immediate influence after retrieval, not future
retention utility for all bank items. It was not promoted to an eviction
policy.

### 4.12 Geometric Coverage-RI blend

Code name: `slam_ri_blend`.

The implementation calls the real Geometric Coverage and RI scoring functions,
min-max normalizes each over the current pool, and computes

```math
u_i(\beta)=\beta\,\widetilde u_i^{\mathrm{Geo}}
 +(1-\beta)\,\widetilde u_i^{\mathrm{RI}}.
```

`beta=1` exactly reproduces the Geometric Coverage ranking and `beta=0`
exactly reproduces RI. The buffer evicts minimum-score candidates.

The 50/50 run was competitive but did not beat both components and lost to
both on four of six VBench dimensions. The geometry-dominant `beta=0.75, k=3`
variant improved the matched LPIPS/FVD prefix table, but its completed VBench
result shows that it is not a broad improvement over Geometric Coverage.

Matched 60-second prefix results:

| Policy | 10s LPIPS/FVD | 20s LPIPS/FVD | 30s LPIPS/FVD | 60s LPIPS/FVD |
| --- | --- | --- | --- | --- |
| Geometric Coverage | .496427 / 719.279 | .542038 / 663.776 | .556215 / 656.271 | .584950 / 690.532 |
| Blend 25% Geo | .487822 / 681.202 | .541732 / 642.061 | .558461 / 656.165 | .590435 / 699.088 |
| Blend 50% Geo | .490108 / 691.568 | .542989 / 709.244 | .557879 / 659.388 | .587929 / 692.430 |
| Blend 75% Geo | .490634 / 711.058 | .538342 / 652.282 | .553307 / 653.595 | .582531 / 691.674 |

Against pure Geometric Coverage, the 75/25 blend wins seven of eight LPIPS/FVD
cells; its only loss is 60-second FVD by `1.142`, effectively a near tie. Its
standard VBench result is:

| Dimension | Geometric Coverage | Blend 75% Geo | Delta |
| --- | ---: | ---: | ---: |
| Subject consistency | 0.818691 | 0.811133 | -0.007557 |
| Background consistency | 0.907117 | 0.900029 | -0.007088 |
| Motion smoothness | 0.991426 | 0.992058 | +0.000632 |
| Dynamic degree | 1.000000 | 1.000000 | 0 |
| Aesthetic quality | 0.479245 | 0.458852 | -0.020393 |
| Imaging quality | 0.529411 | 0.501999 | -0.027411 |

The blend loses four meaningful dimensions, ties one saturated dimension, and
wins motion smoothness by only `0.000632`. It should remain an ablation showing
the tradeoff between RI and Geometric Coverage, not replace Geometric Coverage
as the primary method.

### 4.12a Rarity-only isolation

The final controlled ablation removed RI's RGB irreplaceability multiplier and
tested (1) DINO-cluster rarity alone and (2) 75% Geometric Coverage plus 25%
rarity. This asked whether irreplaceability caused the earlier blend's VBench
loss.

| Policy | 10s LPIPS/FVD | 20s LPIPS/FVD | 30s LPIPS/FVD | 60s LPIPS/FVD |
| --- | --- | --- | --- | --- |
| Geometric Coverage | .496427 / 719.279 | .542038 / 663.776 | .556215 / 656.271 | .584950 / 690.532 |
| Rarity Only | .487774 / 682.187 | .542236 / 651.255 | .563453 / 664.062 | .598634 / 732.294 |
| 75% Geo + 25% Rarity | .492719 / 713.056 | .543034 / 646.257 | .557632 / 656.149 | .587213 / 676.395 |

The Geo-Rarity blend improves 60-second FVD by `14.137`, but loses LPIPS at
20, 30, and 60 seconds. It also loses subject consistency (`-.008285`),
background consistency (`-.006553`), aesthetic quality (`-.023841`), and
imaging quality (`-.026682`) to pure Geometric Coverage. Rarity-only is worse
at 60 seconds and on the same four VBench dimensions. Therefore
irreplaceability was not the sole problem: adding cluster rarity itself trades
perceptual quality for distributional FVD improvement. Neither policy replaces
Geometric Coverage.

### 4.13 Generic no-reference quality gate

The project tested MUSIQ, CLIP-IQA+, TOPIQ-NR, PAQ2PIQ, unclipped fraction,
sharpness, contrast, entropy, gradient energy, and Laplacian variance on 2,160
sampled generated frames with held-out trajectories.

The best deployable candidate was `unclipped_fraction`:

```text
AUC:                         0.630
balanced accuracy:           0.550
bad-frame recall:            0.183
clean false rejection:       0.125
```

At 20% bad-frame prevalence, a 1,000-frame stream would reject roughly 37 bad
frames and 100 clean frames. Generic IQA detects conventional visual defects,
but the relevant failure can be a sharp, plausible, geometrically wrong image.
Decision: do not inject.

### 4.14 Pose-calibrated causal-consistency gate

This proposed gate compared a newly generated candidate with the actual
retrieved contexts that conditioned it, then subtracted expected DINO
similarity for the camera displacement. Expectations were calibrated on
training trajectories only.

On 2,100 pairs, the proposed score produced:

```text
AUC:                         0.511
bad precision:               0.207
bad recall:                  0.119
clean false rejection:       0.117
within-trajectory Spearman: -0.020
raw similarity AUC:          0.546
corrupted-parent AUC:        0.539
```

The predeclared decision was `DO_NOT_INJECT`. The gate is closed.

### 4.15 Reliable Geometric-RI

Code name: `reliable_slam_ri`.

This implementation gates a candidate when several older geometrically
matching references agree with one another more strongly than they agree with
the candidate, then scores survivors with the 75/25 blend. It was implemented
before validation was complete. The output run has 0/15 videos, and both
quality-gate studies rejected the assumptions needed to justify it. It should
not be run as the final method in its present form.

### 4.16 Coverage-Hysteresis Geometric-RI

Code name: `coverage_hysteresis`.

The motivating diagnostic compared older and newer generated representatives
of similar camera views. At view threshold 0.90, 823 matched pairs showed the
older representative was cleaner by `+0.860 dB` PSNR and `+0.0397` SSIM. The
effect remained positive from thresholds 0.80 through 0.95.

The policy converted that result into a hard admission rule. New frames were
processed causally; a candidate was admitted only if its maximum camera-view
similarity to the existing bank and already-admitted candidates was below
0.90. Admitted frames were then scored with 75% Geometric Coverage and 25% RI.

This policy completed 15/15 videos and all standard metrics, but failed as a
final method:

| Prefix | Hysteresis LPIPS | Geo LPIPS | Hysteresis FVD | Geo FVD |
| --- | ---: | ---: | ---: | ---: |
| 10s | 0.490931 | 0.496427 | 701.868 | 719.279 |
| 20s | 0.540792 | 0.542038 | 664.043 | 663.776 |
| 30s | 0.554736 | 0.556215 | 649.155 | 656.271 |
| 60s | 0.587964 | 0.584950 | 768.725 | 690.532 |

VBench versus Geometric Coverage:

| Dimension | Geo | Hysteresis | Delta |
| --- | ---: | ---: | ---: |
| Subject consistency | 0.818691 | 0.806657 | -0.012034 |
| Background consistency | 0.907117 | 0.900239 | -0.006878 |
| Motion smoothness | 0.991426 | 0.992312 | +0.000886 |
| Dynamic degree | 1.000000 | 1.000000 | 0 |
| Aesthetic quality | 0.479245 | 0.454018 | -0.025228 |
| Imaging quality | 0.529411 | 0.505961 | -0.023450 |

Conclusion: the observation that older same-view frames are cleaner was real,
but the hard rule over-preserved incumbents and blocked useful refresh. The
diagnostic should remain evidence; hysteresis should remain a negative
ablation.

## 5. Canonical 180-second quality result

The strongest matched 180-second, budget-32 comparison currently available is:

| Policy | Stored frames | LPIPS | FVD |
| --- | ---: | ---: | ---: |
| Unbounded | 5,397 | 0.5980 | 734.2 |
| FIFO-32 | 32 | 0.6514 | 677.3 |
| RI-32 | 32 | 0.5939 | 550.4 |
| Geometric Coverage-32 | 32 | 0.5876 | 476.6 |

Lower is better. RI and Geometric Coverage beat Unbounded while storing below
one percent of its frames. FIFO establishes that the gain is caused by which
frames are retained, not merely by reducing archive size.

## 6. What the diagnosis currently establishes

### 6.1 Retention versus retrieval

Using a hindsight DINO best-match diagnostic:

| Policy | Retention gap | Retrieval gap |
| --- | ---: | ---: |
| Unbounded | approximately 0 | 0.2267 |
| FIFO | 0.2039 | unavailable |
| RI | 0.0474 | 0.1582 |
| Geometric Coverage | 0.0639 | 0.1338 |

RI mainly protects rare historical modes. Geometric Coverage more consistently
reduces bad selection among retained candidates. The hindsight reference is a
diagnostic proxy, not true utility.

### 6.2 Pool-growth association

On 4,200 Unbounded queries across 15 trajectories:

- Spearman candidate count versus retrieval gap: `0.2029`.
- Positive trajectory-level trend: 13/15, exact sign-test `p=0.00739`.
- Late-minus-early retrieval gap: `+0.07234`, 95% CI
  `[0.03960, 0.10882]`.
- Hindsight-best mismatch stayed approximately flat.

Pool size and elapsed autoregressive depth co-vary, so this is an association,
not proof that candidate count alone causes degradation.

### 6.3 View alignment versus memory corruption

Late minus early for Unbounded:

| Diagnostic | Change |
| --- | ---: |
| Selected view mismatch | -0.03351 |
| Selected memory corruption | +0.08735 |
| Selected effective mismatch | +0.07316 |

Selected views become slightly better aligned while the generated content at
those selected indices becomes substantially worse. The net conditioning input
therefore becomes worse.

### 6.4 Selected-image quality

Retrieved generated frames compared against exact-index dataset GT:

| Policy | Selected PSNR | Selected SSIM | Late PSNR | Late SSIM |
| --- | ---: | ---: | ---: | ---: |
| Unbounded | 11.703 | 0.3089 | 11.433 | 0.3146 |
| FIFO-32 | 10.619 | 0.2617 | 10.135 | 0.2546 |
| RI-32 | 13.396 | 0.3830 | 12.896 | 0.3723 |
| Geometric Coverage-32 | 16.522 | 0.4759 | 15.119 | 0.4386 |

### 6.5 Common-source selection control

To separate policy choice from different generated histories, every policy
provided only frame indices while all image content was read from the same
Unbounded rollout.

Relative to Unbounded selection:

- RI selected indices were `+1.775 dB` PSNR and `+0.0672` SSIM cleaner.
- Geometric Coverage selected indices were `+4.629 dB` PSNR and `+0.1512`
  SSIM cleaner.
- Geometric Coverage won both metrics on all 15 trajectories.

This is strong evidence that curation changes which historical indices the
native retriever can choose and that those indices are cleaner. It is still
observational with respect to downstream generation.

### 6.6 Causal replay status

One matched context-identity swap case improved LPIPS by `-0.00177` and DINO
distance by `-0.01691`, but `n=1` is not evidence of prevalence.

The stronger planned intervention preserves history, selected frame identity,
model, and noise while replacing selected generated memory content with
exact-index GT for one section. The full multi-case report has not been
confirmed. Therefore the project should say that selected-memory corruption is
strongly associated with quality, not that propagation has been causally
proven.

## 7. Mechanisms tested and rejected or narrowed

1. **Unbounded as oracle:** invalid. Complete retention does not imply correct
   selection.
2. **Occlusion poisoning:** real but modest. Mean appearance penalty was
   `0.0455`, with only a coarse small-to-large-pool increase.
3. **Within-section retrieval collapse:** real in a minority of sections, but
   concentration decreases as the pool grows.
4. **Average archive corruption drift:** average memory corruption did not
   monotonically worsen. The selected subset worsened, which is a different
   result.
5. **WorldMem-style zero-overlap fallback:** does not transfer to MemCam.
   MemCam Unbounded had essentially 100% overlap hits and median overlap about
   0.9335.
6. **Monte Carlo IoU winner noise:** common, but decreases with larger pools
   and changes appearance by only about `0.0029` when the winner flips.
7. **Direct archive attention dilution:** architecturally false in MemCam
   because the generator sees a fixed-size retrieved context.
8. **Generic IQA gate:** not deployable.
9. **Pose-calibrated conditioning gate:** near random and rejected.
10. **Hard same-view hysteresis:** supported as an offline population
    observation, but harmful as an admission policy.

## 8. Current scientific interpretation

The most defensible statement is:

> In MemCam, keeping all history expands an exhaustive candidate set without
> expanding generator context. Over long autoregressive rollouts, the native
> selector increasingly exposes generation to corrupted historical content.
> Bounded curation changes the candidate distribution: RI preserves rare
> modes, while Geometric Coverage removes dense, replaceable view clusters.
> Both can outperform complete retention, and Geometric Coverage does so by a
> large margin at 180 seconds.

What is not yet defensible:

- Pool size alone causally causes the degradation.
- Every bounded-memory gain is caused by corrupted-memory propagation.
- A deployable online quality estimator has been found.
- The 75/25 blend is universally optimal.
- The policy is representation agnostic across WorldMem, VMem, and SpMem.
- The final method beats Geometric Coverage on all metric families.

## 9. Current decision point

1. Coverage-Hysteresis is closed as a negative method result.
2. The quality-gate branches are closed unless a materially new online signal
   is proposed.
3. `slamri_b32_beta0p75_k3` improves seven of eight LPIPS/FVD cells against
   Geometric Coverage, but loses four meaningful VBench dimensions. It is a
   metric tradeoff, not a final-policy victory.
4. Geometric Coverage is the strongest completed MemCam method across the
   complete evidence. It should be the primary method unless a future method
   clears a predeclared, multi-metric criterion.
5. Additional policy complexity should not be invented to force a win. The
   next high-value work is causal validation, efficiency measurement,
   visualization, and transfer to a second memory representation.

## 10. Better visualization program

The current RI visualizations are useful debugging artifacts, but they show one
policy, one section, and projected feature geometry. They do not yet explain
why an eviction is sensible or how it affects downstream retrieval. The next
visuals should connect **candidate structure -> policy decision -> future
conditioning quality**.

### Figure A: One eviction event, fully decomposed

Purpose: make the 75/25 policy understandable without equations alone.

Composition:

1. Left: current 32-frame bank plus the incoming section, ordered in time.
2. Middle top: camera trajectory with FOV arrows; incoming candidates in blue,
   incumbents in gray, retained memories in green, evictions in red.
3. Middle bottom: a reordered affinity matrix `K(i,j)` with cluster boundaries.
4. Right: the 12 candidates nearest the eviction boundary, each with three
   aligned bars: normalized Geometric score, normalized RI score, final 75/25
   score.
5. A callout pairs each evicted frame with its nearest retained substitute and
   prints pose distance, DINO cosine, RGB distance, and final score.

This should be the primary method-mechanics figure. It shows that low-scoring
frames are not simply old or unattractive; they are well-covered by retained
alternatives.

### Figure B: Candidate competition under Unbounded versus curated memory

Purpose: visualize the empirical failure mechanism.

For one late target query, show:

1. Target dataset frame and target camera pose.
2. Top 12 Unbounded candidates ranked by native FOV overlap.
3. Each thumbnail annotated with overlap, generated-to-GT PSNR/SSIM, age, and
   whether RI/Geo retained it.
4. Highlight the actual Unbounded winner and the actual Geometric Coverage
   winner.
5. Show the next generated chunk under each real rollout as context, clearly
   labeled observational unless a matched replay exists.

The key visual is that many candidates have nearly identical overlap, but the
complete archive offers low-fidelity historical winners that the curated bank
has removed.

### Figure C: Common-source selection gallery

Purpose: present the cleanest evidence that the policy, rather than different
rollout history, selects better indices.

For four representative trajectories:

- Use one shared Unbounded MP4 as the image source.
- For the same late target, show the index selected by Unbounded, FIFO, RI, and
  Geometric Coverage.
- Place exact-index GT beneath each selected image.
- Print PSNR, SSIM, pose overlap, and memory age.

This is stronger and easier to interpret than a scatter plot because all
policies are choosing from identical pixels.

### Figure D: Memory-bank evolution animation

Purpose: create the intuitive, visually compelling explanation requested for
talks and supplementary material.

At each generated section, animate:

- The camera path and current camera location.
- All retained memories as points along the path.
- Thumbnail tiles for the 32 active bank entries.
- New admissions flashing blue, retained entries green, and evictions red.
- Each eviction moving next to its retained substitute before disappearing.
- A small stacked bar showing the Geometric and RI contributions to the
  evicted candidate's score.

Render both Geometric Coverage and the 75/25 blend side by side. Export MP4 and
GIF. This should use actual trace events, not recomputed hypothetical choices.

### Figure E: Cluster survival Sankey or alluvial plot

Purpose: show whether rare semantic/view modes survive over time.

- Horizontal axis: section index.
- Bands: DINO appearance clusters, with width proportional to represented
  bank slots.
- Color: cluster identity fixed per trajectory.
- Thin persistent bands show protected rare modes; wide bands shrinking show
  duplicate removal.
- Compare FIFO, RI, Geometric Coverage, and the blend in aligned panels.

Because DINO clusters can change with the candidate pool, this plot must define
clusters once from the complete trajectory or use stable nearest-centroid
assignment. Re-clustering independently at every section would create false
cluster motion.

### Figure F: Score-space decision map

Purpose: explain the blend and its tradeoff.

- X axis: normalized RI score.
- Y axis: normalized Geometric Coverage score.
- Diagonal contours: final 75/25 keep score.
- Points: current candidate frames, with thumbnail callouts for boundary cases.
- Marker shape: incumbent versus incoming.
- Fill: retained versus evicted.
- Border color: DINO cluster.

Unlike a UMAP plot, this directly visualizes the coordinates the policy
actually uses.

### Figure G: Selected-memory quality over horizon

Purpose: connect archive growth to selected conditioning quality.

- X axis: elapsed seconds or section index.
- Y axes in separate panels: selected-memory PSNR, selected-memory SSIM, view
  mismatch, and effective mismatch.
- Lines: Unbounded, FIFO, RI, Geometric Coverage, and final candidate.
- Use trajectory-bootstrap confidence bands.
- Overlay Unbounded candidate count on a secondary light axis only in the
  Unbounded panel.

This replaces a single early/late bar with the full trajectory of the effect.
Do not smooth beyond a clearly declared section-bin mean.

### Figure H: Retention-retrieval frontier

Purpose: show why complete memory is not automatically optimal.

- X axis: retention gap, lower is better.
- Y axis: retrieval gap, lower is better.
- Bubble size: stored items or bytes.
- Bubble color: downstream 180-second FVD.
- Points: Unbounded, FIFO budgets, RI budgets, K-center budgets, Geometric
  Coverage budgets, and blend variants.

Unbounded should appear at zero retention gap but high retrieval gap. RI and
Geometric Coverage should show the useful trade: a small retention loss buys a
large retrieval improvement. Label the hindsight metric as diagnostic.

### Figure I: Policy outcome matrix

Purpose: prevent cherry-picking across many metrics.

- Rows: policies.
- Columns: LPIPS, FVD, VBench dimensions, CUT3R, latency, and memory.
- Cell values: percent change relative to Geometric Coverage or Unbounded,
  depending on the stated question.
- Use a diverging palette with direction normalized so positive always means
  better.
- Mark missing and unmatched cells explicitly.

A heatmap is more readable than a radar chart and makes the Hysteresis failure
or blend tradeoff immediately visible.

### Figure J: Representation-adapter diagram

Purpose: communicate the transferable interface without claiming transfer
before experiments exist.

Show RGB frames, latent entries, surfels, and point-cloud chunks entering
representation-specific descriptor adapters. Each adapter emits:

- an item cost,
- a pairwise similarity or coverage affinity,
- optional metadata such as pose and time.

The common online curation block then produces a bounded candidate bank, after
which each system retains its native retriever and generator. Label WorldMem,
VMem, and SpMem as transfer targets until measured.

## 11. Visualization priorities

### Immediate, highest scientific value

1. Figure A: decomposed Geometric Coverage eviction event, with the blend score
   shown only as a comparative ablation.
2. Figure C: common-source selected-frame gallery.
3. Figure G: selected-memory quality over time with confidence bands.
4. Figure I: matched metric outcome matrix after the blend VBench result.

### Best for talks and supplementary material

5. Figure D: memory-bank evolution animation.
6. Figure B: late-query candidate competition gallery.
7. Figure E: stable-cluster survival plot.

### Use with explicit caveats

8. Figure H: retention-retrieval frontier uses a hindsight diagnostic.
9. Figure J: representation adapter is a proposed interface until transfer is
   demonstrated.
10. UMAP/t-SNE plots should remain supplementary. They are attractive but do
    not show the actual score geometry and can visually invent cluster
    separation.

## 12. Questions for the next brainstorming session

1. Is Geometric Coverage, together with the mechanism study and efficiency
   result, sufficient as the paper's primary method contribution?
2. Why does adding 25% RI improve LPIPS/FVD prefixes while reducing subject,
   background, aesthetic, and imaging VBench scores?
3. Should the beta sweep remain an ablation rather than be tuned further now
   that no tested interior blend dominates Geometric Coverage?
4. What exact claim transfers across MemCam and WorldMem: bounded candidate
   curation, a shared score, or only a shared diagnostic framework?
5. Which matched intervention can establish that cleaner selected memory
   content improves the next chunk with more than one valid case?
6. Should the paper lead with the system result or with the counterintuitive
   mechanism finding that complete retention can degrade retrieval?
7. What is the minimum cross-system evidence needed before using the phrase
   representation agnostic?

## 13. Recommended honest paper spine today

1. Long-video memory systems optimize retrieval while allowing the candidate
   archive to grow.
2. Unbounded MemCam has linear storage and quadratic cumulative exhaustive
   retrieval cost.
3. Complete retention is not quality-optimal: 32-frame RI and Geometric
   Coverage banks beat 5,397-frame Unbounded memory at 180 seconds.
4. Diagnostics show that the selected subset increasingly exposes corrupted
   historical content, while curated policies select cleaner indices even on
   a common rollout.
5. Geometric Coverage is the strongest proven policy. The 75/25 RI blend is an
   informative ablation: it improves matched LPIPS/FVD prefixes but loses four
   meaningful VBench dimensions.
6. Failed gates and Hysteresis establish that diagnosing a population trend
   does not automatically yield a deployable admission rule.
7. Cross-representation claims remain conditional on WorldMem or another
   successful transfer experiment.
